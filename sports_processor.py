"""
SportsProcessor — Custom sports intelligence video processor.

This is what makes Vision Agents the HERO of PlayByt.

Raw YOLO output: "17 keypoints per person detected"
SportsProcessor output: "3 players in defensive third showing fatigue,
                          high pressing intensity on the right channel,
                          estimated 4-3-3 formation"

The processor:
1. Runs YOLO pose detection on each frame (reuses YOLOPoseProcessor internals)
2. Converts raw keypoint data into structured football intelligence
3. Draws a real-time HUD overlay on the video feed
4. Stores analysis for tool calling — Gemini accesses it via get_field_analysis()

Without this, Gemini is just watching TV like anyone else.
With this, Gemini has superhuman spatial data no human eye can compute in real time.
"""

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import av
import cv2
import numpy as np
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.ultralytics import YOLOPoseProcessor

ANALYSIS_FILE = Path(__file__).parent / ".analysis.json"
CONTROVERSIES_FILE = Path(__file__).parent / ".controversies.json"

logger = logging.getLogger(__name__)

# Participants to skip — SDK demo user and non-screenshare camera feeds
_SKIP_PARTICIPANT_PREFIXES = ("user-demo", "demo-agent", "user-demo-agent")

# Suppress noisy H264/VP8 decode errors from SDK demo browser feed
logging.getLogger("libav.h264").setLevel(logging.CRITICAL)
logging.getLogger("libav.libvpx").setLevel(logging.CRITICAL)
logging.getLogger("aiortc.codecs.h264").setLevel(logging.CRITICAL)
logging.getLogger("aiortc.codecs.vpx").setLevel(logging.CRITICAL)

# ── COCO Pose Keypoint Indices ──────────────────────────────────────────────
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


class SportsProcessor(VideoProcessorPublisher):
    """
    Custom video processor that wraps YOLO and adds structured sports intelligence.

    Per frame it computes:
    - Player count and spatial zone distribution
    - Fatigue indicators from spine posture angles
    - Pressing intensity from player clustering metrics
    - Formation estimate (e.g. 4-3-3)
    - Dominant side of play

    Draws a compact HUD overlay so Gemini can SEE the computed data.
    Stores latest analysis dict for tool calling access.
    """

    name = "sports_intelligence"

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        fps: int = 3,
    ):
        # Internal YOLO processor — we call its methods directly, not via the SDK pipeline
        self._yolo = YOLOPoseProcessor(
            model_path=model_path,
            conf_threshold=conf_threshold,
            fps=fps,
            enable_hand_tracking=False,
            enable_wrist_highlights=False,
        )
        self._video_track = QueuedVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None
        self._shutdown = False
        self.fps = fps
        self.conf_threshold = conf_threshold

        # Latest analysis — read by tool calling functions in main.py
        self.latest_analysis: Dict[str, Any] = {}
        self._analysis_history: List[Dict[str, Any]] = []

        # Controversy detection state
        self._controversies: List[Dict[str, Any]] = []
        self._prev_pressing: str = "none"
        self._prev_fatigue_count: int = 0
        self._prev_formation: str = "N/A"
        self._last_alert_time: Dict[str, float] = {}  # Cooldown per alert type
        self._ALERT_COOLDOWN: float = 30.0  # Min seconds between same alert type
        self._MIN_PLAYERS_FOR_ALERTS: int = 4  # Don't fire alerts with < 4 players
        self._start_time: float = time.time()
        self._frame_count: int = 0
        self._error_count: int = 0
        self._consecutive_errors: int = 0
        self._MAX_CONSECUTIVE_ERRORS: int = 20  # Stop logging after 20 in a row

        # Event queue — controversies are pushed here for the commentary loop
        # to fire IMMEDIATELY instead of waiting for the next timer tick.
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        # Timestamp of the most recent video frame received — used by the
        # commentary loop to detect when screen share has stopped.
        self.last_frame_time: float = 0.0

        # Clear stale files on boot
        try:
            ANALYSIS_FILE.write_text("{}")
            CONTROVERSIES_FILE.write_text("[]")
        except Exception:
            pass

        logger.info("⚽ Sports Intelligence Processor initialized")

    # ── Video Pipeline ──────────────────────────────────────────────────────

    async def process_video(
        self,
        incoming_track: Any,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Set up the video processing pipeline."""
        # Skip SDK demo user and non-screenshare camera feeds
        pid = participant_id or ""
        if any(pid.startswith(prefix) for prefix in _SKIP_PARTICIPANT_PREFIXES):
            logger.info("⚽ Skipping demo/non-screenshare participant: %s", pid)
            return

        # Only replace if we're getting a NEW track — don't tear down the same one
        if self._video_forwarder is not None:
            if shared_forwarder and shared_forwarder is self._video_forwarder:
                # Same forwarder — just re-add the handler (idempotent)
                logger.info("⚽ Re-attaching frame handler for participant: %s", pid)
            else:
                # Different track — tear down old and set up new
                logger.info("⚽ Replacing video forwarder for new participant: %s", pid)
                try:
                    await self._video_forwarder.remove_frame_handler(self._process_frame)
                except Exception:
                    pass

        logger.info(f"⚽ Starting Sports Intelligence processing at {self.fps} FPS")
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                incoming_track,
                max_buffer=self.fps,
                fps=self.fps,
                name="sports_intelligence_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._process_frame, fps=float(self.fps), name="sports_intelligence"
        )

    async def _process_frame(self, frame: av.VideoFrame) -> None:
        """Process a single frame: YOLO → analysis → HUD → publish."""
        if self._shutdown:
            return

        self._frame_count += 1
        self.last_frame_time = time.time()  # track when we last received a frame

        # Step 1: Decode frame
        try:
            frame_array = frame.to_ndarray(format="rgb24")
        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors <= 3:
                logger.warning("⚽ Frame decode failed (#%d): %s", self._frame_count, e)
            try:
                await self._video_track.add_frame(frame)
            except Exception:
                pass
            return

        # Step 2: YOLO pose detection (resize to max 480px wide to cut compute ~4x)
        try:
            h_orig, w_orig = frame_array.shape[:2]
            _MAX_W = 480
            if w_orig > _MAX_W:
                scale = _MAX_W / w_orig
                yolo_frame = cv2.resize(
                    frame_array,
                    (int(w_orig * scale), int(h_orig * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                yolo_frame = frame_array
            annotated_small, pose_data = await self._yolo.add_pose_to_ndarray(yolo_frame)
            # Scale annotated result back to original size for HUD display
            if w_orig > _MAX_W:
                annotated = cv2.resize(
                    annotated_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
                )
            else:
                annotated = annotated_small
        except Exception as e:
            self._error_count += 1
            self._consecutive_errors += 1
            if self._consecutive_errors <= 5:
                logger.warning("⚽ YOLO inference failed on frame #%d: %s", self._frame_count, e)
            try:
                raw_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                await self._video_track.add_frame(raw_frame)
            except Exception:
                try:
                    await self._video_track.add_frame(frame)
                except Exception:
                    pass
            return

        self._consecutive_errors = 0

        # Step 3: Compute analysis + store + detect controversies
        try:
            h, w = annotated.shape[:2]
            analysis = self._compute_analysis(pose_data, w, h)
            self.latest_analysis = analysis
            self._analysis_history.append(analysis)
            if len(self._analysis_history) > 30:
                self._analysis_history = self._analysis_history[-30:]
            # Only write to disk every 5th frame — avoids spawning 3 async tasks/sec
            if self._frame_count % 5 == 0:
                asyncio.ensure_future(self._persist_analysis(analysis))
            self._detect_controversies(analysis)
        except Exception as e:
            self._error_count += 1
            if self._error_count % 50 == 1:
                logger.warning("⚽ Analysis error on frame #%d (total: %d): %s", self._frame_count, self._error_count, e)

        # Step 4: Draw HUD
        try:
            self._draw_hud(annotated, self.latest_analysis or {})
        except Exception:
            pass  # HUD failure should never block video

        # Step 5: Publish processed frame (with fallback to raw)
        try:
            processed = av.VideoFrame.from_ndarray(annotated, format="rgb24")
            await self._video_track.add_frame(processed)
        except Exception:
            try:
                await self._video_track.add_frame(frame)
            except Exception:
                pass

    def publish_video_track(self) -> QueuedVideoTrack:
        """Return the output video track."""
        return self._video_track

    async def stop_processing(self) -> None:
        """Stop processing video."""
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._process_frame)
            self._video_forwarder = None
            logger.info("🛑 Sports Intelligence processing stopped")
        # Reset so commentary loop knows video is gone
        self.last_frame_time = 0.0
        self.latest_analysis = {}

    async def close(self) -> None:
        """Clean up resources."""
        self._shutdown = True
        await self.stop_processing()
        await self._yolo.close()
        logger.info("🛑 Sports Intelligence Processor closed")

    # ── Analysis Engine ─────────────────────────────────────────────────────

    def _compute_analysis(
        self, pose_data: Dict[str, Any], frame_w: int, frame_h: int
    ) -> Dict[str, Any]:
        """Convert raw YOLO keypoints into structured sports intelligence."""
        persons = pose_data.get("persons", [])

        empty_zones = {
            "left": 0, "center": 0, "right": 0,
            "def_third": 0, "mid_third": 0, "att_third": 0,
        }

        if not persons:
            return {
                "timestamp": time.time(),
                "player_count": 0,
                "zones": empty_zones,
                "fatigue_flags": [],
                "pressing_intensity": "none",
                "formation": "N/A",
                "dominant_side": "balanced",
                "positions": [],
            }

        # Extract player positions (hip midpoint = player center)
        positions: List[Dict[str, Any]] = []
        fatigue_flags: List[Dict[str, Any]] = []

        for person in persons:
            kpts = person.get("keypoints", [])
            if len(kpts) < 17:
                continue

            l_hip = kpts[L_HIP]
            r_hip = kpts[R_HIP]

            if l_hip[2] > self.conf_threshold and r_hip[2] > self.conf_threshold:
                cx = (l_hip[0] + r_hip[0]) / 2
                cy = (l_hip[1] + r_hip[1]) / 2
                positions.append({
                    "x": cx / frame_w,  # Normalized 0-1
                    "y": cy / frame_h,
                    "id": person["person_id"],
                })

            # Fatigue detection: forward lean from spine angle
            l_sh = kpts[L_SHOULDER]
            r_sh = kpts[R_SHOULDER]
            if all(k[2] > self.conf_threshold for k in [l_sh, r_sh, l_hip, r_hip]):
                sh_mid_x = (l_sh[0] + r_sh[0]) / 2
                sh_mid_y = (l_sh[1] + r_sh[1]) / 2
                hip_mid_x = (l_hip[0] + r_hip[0]) / 2
                hip_mid_y = (l_hip[1] + r_hip[1]) / 2

                # Spine angle relative to vertical
                dx = sh_mid_x - hip_mid_x
                dy = hip_mid_y - sh_mid_y  # Flip Y (screen coords inverted)

                if dy > 0:
                    spine_angle = math.degrees(math.atan2(abs(dx), dy))
                else:
                    spine_angle = 90.0

                if spine_angle > 25:  # More than 25° forward lean = fatigue signal
                    fatigue_flags.append({
                        "player_id": person["person_id"],
                        "spine_angle": round(spine_angle, 1),
                        "severity": "high" if spine_angle > 40 else "moderate",
                    })

        # Zone distribution
        zones = dict(empty_zones)
        for pos in positions:
            if pos["x"] < 0.33:
                zones["left"] += 1
            elif pos["x"] < 0.67:
                zones["center"] += 1
            else:
                zones["right"] += 1

            if pos["y"] < 0.33:
                zones["def_third"] += 1
            elif pos["y"] < 0.67:
                zones["mid_third"] += 1
            else:
                zones["att_third"] += 1

        # Dominant side (need at least 3 players for this to be meaningful)
        if len(positions) >= 3 and zones["left"] > zones["right"] + 1:
            dominant_side = "left"
        elif len(positions) >= 3 and zones["right"] > zones["left"] + 1:
            dominant_side = "right"
        else:
            dominant_side = "balanced"

        # Pressing intensity (average pairwise distance)
        pressing = "none"
        if len(positions) >= 2:
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = math.sqrt(
                        (positions[i]["x"] - positions[j]["x"]) ** 2
                        + (positions[i]["y"] - positions[j]["y"]) ** 2
                    )
                    distances.append(d)
            avg_dist = sum(distances) / len(distances)
            if avg_dist < 0.15:
                pressing = "high"
            elif avg_dist < 0.25:
                pressing = "medium"
            else:
                pressing = "low"

        # Formation estimate
        formation = self._estimate_formation(positions)

        return {
            "timestamp": time.time(),
            "player_count": len(positions),
            "zones": zones,
            "fatigue_flags": fatigue_flags,
            "pressing_intensity": pressing,
            "formation": formation,
            "dominant_side": dominant_side,
            "positions": positions,
        }

    def _estimate_formation(self, positions: List[Dict[str, Any]]) -> str:
        """Rough formation estimate from player vertical distribution."""
        if len(positions) < 4:
            return "N/A"

        sorted_pos = sorted(positions, key=lambda p: p["y"])
        n = len(sorted_pos)
        third = max(1, n // 3)

        defense = len(sorted_pos[:third])
        midfield = len(sorted_pos[third : third * 2])
        attack = len(sorted_pos[third * 2 :])

        return f"{defense}-{midfield}-{attack}"

    def get_trend(self) -> Dict[str, Any]:
        """Compute trends from analysis history (called by tool functions)."""
        if len(self._analysis_history) < 3:
            return {"trend": "insufficient_data"}

        recent = self._analysis_history[-10:]
        earlier = self._analysis_history[:-10] if len(self._analysis_history) > 10 else []

        avg_recent_players = sum(a["player_count"] for a in recent) / len(recent)
        fatigue_count = sum(len(a["fatigue_flags"]) for a in recent)

        pressing_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
        for a in recent:
            pressing_counts[a["pressing_intensity"]] += 1
        dominant_pressing = max(pressing_counts, key=pressing_counts.get)  # type: ignore

        trend: Dict[str, Any] = {
            "avg_players_visible": round(avg_recent_players, 1),
            "fatigue_events_last_10_frames": fatigue_count,
            "dominant_pressing": dominant_pressing,
            "frames_analyzed": len(self._analysis_history),
        }

        if earlier:
            avg_earlier = sum(a["player_count"] for a in earlier) / len(earlier)
            if avg_recent_players > avg_earlier + 1:
                trend["player_movement"] = "more players entering frame"
            elif avg_recent_players < avg_earlier - 1:
                trend["player_movement"] = "players leaving frame"
            else:
                trend["player_movement"] = "stable"

        return trend

    def _detect_controversies(self, analysis: Dict[str, Any]) -> None:
        """Detect threshold-based controversy events and persist them."""
        elapsed = round(time.time() - self._start_time)
        now = time.time()
        alerts: List[Dict[str, Any]] = []

        pressing = analysis.get("pressing_intensity", "none")
        formation = analysis.get("formation", "N/A")
        fatigue_flags = analysis.get("fatigue_flags", [])
        fatigue_count = len(fatigue_flags)
        player_count = analysis.get("player_count", 0)

        # Gate: don't fire any alerts with fewer than MIN_PLAYERS
        if player_count < self._MIN_PLAYERS_FOR_ALERTS:
            self._prev_pressing = pressing
            self._prev_formation = formation
            self._prev_fatigue_count = fatigue_count
            return

        def _can_fire(alert_type: str) -> bool:
            """Check cooldown for this alert type."""
            last = self._last_alert_time.get(alert_type, 0)
            return (now - last) >= self._ALERT_COOLDOWN

        # Alert: pressing spiked to HIGH from lower level
        if pressing == "high" and self._prev_pressing in ("none", "low") and _can_fire("pressing_spike"):
            alerts.append({
                "type": "pressing_spike",
                "title": "High Press Triggered",
                "description": f"Pressing intensity spiked to HIGH from {self._prev_pressing.upper()}",
                "elapsed": elapsed,
            })

        # Alert: pressing dropped from HIGH
        if self._prev_pressing == "high" and pressing in ("none", "low") and _can_fire("press_drop"):
            alerts.append({
                "type": "press_drop",
                "title": "Press Broken",
                "description": "High press dropped — counter-attack window open",
                "elapsed": elapsed,
            })

        # Alert: formation changed significantly
        if (
            formation not in ("N/A", self._prev_formation)
            and self._prev_formation != "N/A"
            and _can_fire("formation_change")
        ):
            alerts.append({
                "type": "formation_change",
                "title": "Formation Shift",
                "description": f"Formation changed: {self._prev_formation} -> {formation}",
                "elapsed": elapsed,
            })

        # Alert: fatigue spike (3+ new fatigue flags compared to previous)
        if fatigue_count >= 3 and self._prev_fatigue_count < 2 and _can_fire("fatigue_spike"):
            players = ", ".join(str(f["player_id"] + 1) for f in fatigue_flags[:3])
            alerts.append({
                "type": "fatigue_spike",
                "title": "Fatigue Alert",
                "description": f"{fatigue_count} players showing fatigue (players {players})",
                "elapsed": elapsed,
            })

        # Alert: dominant side overload (need 6+ players and 75%+ on one side)
        zones = analysis.get("zones", {})
        left, right = zones.get("left", 0), zones.get("right", 0)
        total = left + right
        if total >= 6 and _can_fire("overload"):
            if left / total > 0.75 or right / total > 0.75:
                side = "LEFT" if left > right else "RIGHT"
                alerts.append({
                    "type": "overload",
                    "title": f"{side} Overload",
                    "description": f"Strong {side.lower()}-side congestion — {max(left, right)}/{total} players",
                    "elapsed": elapsed,
                })

        if alerts:
            for alert in alerts:
                alert["id"] = len(self._controversies) + 1
                alert["timestamp"] = now
                self._controversies.append(alert)
                self._last_alert_time[alert["type"]] = now
                # Push to event queue for immediate commentary (non-blocking)
                try:
                    self._event_queue.put_nowait(alert)
                except asyncio.QueueFull:
                    pass  # Drop oldest — commentary will catch up
            asyncio.ensure_future(self._persist_controversies())

        # Update prev state
        self._prev_pressing = pressing
        self._prev_formation = formation
        self._prev_fatigue_count = fatigue_count

    def get_latest_controversies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent controversy events."""
        return self._controversies[-limit:]

    async def _persist_analysis(self, analysis: Dict[str, Any]) -> None:
        """Write analysis to disk with file locking for safe concurrent reads."""
        import fcntl
        try:
            data = json.dumps(analysis)
            def _write():
                with open(ANALYSIS_FILE, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(data)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            await asyncio.to_thread(_write)
        except Exception:
            pass

    async def _persist_controversies(self) -> None:
        """Write controversies to disk with file locking for safe concurrent reads."""
        import fcntl
        try:
            data = json.dumps(self._controversies[-50:])
            def _write():
                with open(CONTROVERSIES_FILE, "w") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(data)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            await asyncio.to_thread(_write)
        except Exception:
            pass

    # ── HUD Overlay ─────────────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, analysis: Dict[str, Any]) -> None:
        """Draw a compact intelligence HUD in the top-left of the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        hud_w, hud_h = 230, 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + hud_w, 8 + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Border
        cv2.rectangle(frame, (8, 8), (8 + hud_w, 8 + hud_h), (0, 255, 136), 1)

        y = 24
        font = cv2.FONT_HERSHEY_SIMPLEX
        sm = 0.32
        lg = 0.38

        # Title
        cv2.putText(frame, "PLAYBYT INTELLIGENCE", (14, y), font, sm, (0, 255, 136), 1, cv2.LINE_AA)
        y += 18

        # Player count
        pc = analysis.get("player_count", 0)
        cv2.putText(frame, f"Players Tracked: {pc}", (14, y), font, lg, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

        # Zones
        z = analysis.get("zones", {})
        cv2.putText(
            frame,
            f"L:{z.get('left',0)}  C:{z.get('center',0)}  R:{z.get('right',0)}",
            (14, y), font, sm, (200, 200, 200), 1, cv2.LINE_AA,
        )
        y += 14
        cv2.putText(
            frame,
            f"Def:{z.get('def_third',0)}  Mid:{z.get('mid_third',0)}  Att:{z.get('att_third',0)}",
            (14, y), font, sm, (200, 200, 200), 1, cv2.LINE_AA,
        )
        y += 18

        # Formation
        formation = analysis.get("formation", "N/A")
        cv2.putText(frame, f"Formation: {formation}", (14, y), font, lg, (68, 136, 255), 1, cv2.LINE_AA)
        y += 18

        # Pressing
        pressing = analysis.get("pressing_intensity", "none")
        press_colors = {"high": (0, 255, 136), "medium": (0, 200, 255), "low": (150, 150, 150), "none": (100, 100, 100)}
        cv2.putText(
            frame, f"Pressing: {pressing.upper()}", (14, y), font, lg,
            press_colors.get(pressing, (150, 150, 150)), 1, cv2.LINE_AA,
        )

        # Fatigue alerts (bottom-left)
        fatigue = analysis.get("fatigue_flags", [])
        if fatigue:
            fy = h - 20
            for f in fatigue[-3:]:  # Show up to 3 latest
                sev_color = (0, 0, 255) if f["severity"] == "high" else (0, 165, 255)
                text = f"! Player {f['player_id']+1} fatigue ({f['spine_angle']}deg)"
                # Background for readability
                (tw, th), _ = cv2.getTextSize(text, font, sm, 1)
                cv2.rectangle(frame, (12, fy - th - 2), (16 + tw, fy + 4), (0, 0, 0), -1)
                cv2.putText(frame, text, (14, fy), font, sm, sev_color, 1, cv2.LINE_AA)
                fy -= 20

        # Dominant side indicator (top-right)
        side = analysis.get("dominant_side", "balanced")
        if side != "balanced":
            side_text = f">> {side.upper()} OVERLOAD"
            (tw, _), _ = cv2.getTextSize(side_text, font, sm, 1)
            cv2.putText(frame, side_text, (w - tw - 14, 24), font, sm, (255, 170, 0), 1, cv2.LINE_AA)
