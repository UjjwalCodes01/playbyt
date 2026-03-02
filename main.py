"""
PlayByt — AI Sports Analyst That Catches What You Miss
Built with Vision Agents SDK by Stream

Architecture:
  Screen share → YOLO (player/pose detection) → Annotated frames → Gemini Realtime (analysis)
  Users join with a role (analyst/hype/stats/coach).
  Agent uses tool calling to log highlights and generate match reports.
  Google Search grounds stats and player info in real data.
"""

import asyncio
import concurrent.futures
import fcntl
import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from getstream.models import CallRequest
import httpx
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
)
from vision_agents.plugins import gemini, getstream
from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge

from sports_processor import SportsProcessor

# ── Torch thread limits ───────────────────────────────────────────────────────
# By default PyTorch uses all available CPU cores for inference threads.
# On a t2.small (1 vCPU) this causes massive context-switching overhead and
# drains CPU burst credits within minutes. Capping at 1 keeps inference
# sequential, halves per-inference memory (no thread stacks), and preserves
# burst credits so the instance stays responsive.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── SDK monkey-patch: Gemini reconnect on 1008 ────────────────────────────────
# The SDK's _should_reconnect only handles 1011-1014.  Code 1008 ("policy
# violation — operation not supported") leaves the processing loop spinning on
# a dead WebSocket.  Patch the function so the SDK auto-reconnects on 1008 too.
import vision_agents.plugins.gemini.gemini_realtime as _gemini_rt
import websockets as _ws

_RECONNECT_CODES = {1008, 1011, 1012, 1013, 1014}

def _patched_should_reconnect(exc: Exception) -> bool:
    if (
        isinstance(exc, _ws.ConnectionClosedError)
        and exc.rcvd
        and exc.rcvd.code in _RECONNECT_CODES
    ):
        return True
    return False

_gemini_rt._should_reconnect = _patched_should_reconnect
# ──────────────────────────────────────────────────────────────────────────────

# ── SDK monkey-patch: JPEG + resize for video frames ─────────────────────────
# The default _send_video_frame converts frames to PNG with NO resizing.
# A full-resolution screen share (e.g. 1920×1080) produces a multi-MB PNG that
# exceeds the Gemini Live API per-message limit → server closes with 1008.
# Patch to JPEG at 640×360 (≈15–40 KB) which stays well within the limit.
from functools import partial as _partial
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes as _frame_to_jpeg
from google.genai.types import Blob as _Blob

_VID_W, _VID_H, _VID_Q = 640, 360, 75  # target width, height, JPEG quality

async def _patched_send_video_frame(self, frame):
    loop = asyncio.get_running_loop()
    jpeg_bytes = await loop.run_in_executor(
        self._executor,
        _partial(_frame_to_jpeg, frame, _VID_W, _VID_H, _VID_Q),
    )
    blob = _Blob(data=jpeg_bytes, mime_type="image/jpeg")
    try:
        await self._session.send_realtime_input(media=blob)
    except Exception:
        logger.exception("Failed to send a video frame to Gemini Live API")

import types as _types
_gemini_rt.GeminiRealtime._send_video_frame = _patched_send_video_frame
# ──────────────────────────────────────────────────────────────────────────────

# ── SDK monkey-patch: _processing_loop with exponential backoff ───────────────
# The SDK's _processing_loop has no backoff in the generic `except Exception`
# handler.  When Gemini returns APIError 1011 (service unavailable), it is NOT
# a ConnectionClosedError so it bypasses _should_reconnect entirely — the loop
# just logs the error and spins instantly → millions of log lines per minute.
# Replace the loop with a version that: (a) sleeps before every retry, (b)
# re-calls self.connect() to get a fresh WebSocket, (c) resets delay on success.
from asyncio import CancelledError as _CancelledError

_LOOP_DELAYS = [1, 2, 5, 10, 30, 60]  # seconds between retries

async def _patched_processing_loop(self):
    logger.debug("Start processing events from Gemini Live API")
    _retry = 0
    try:
        while True:
            try:
                await self._process_events()
                _retry = 0  # clean exit — reset backoff
            except _CancelledError:
                raise
            except _ws.ConnectionClosedError as exc:
                if not _gemini_rt._should_reconnect(exc):
                    raise
                delay = _LOOP_DELAYS[min(_retry, len(_LOOP_DELAYS) - 1)]
                logger.warning(
                    "Gemini WS closed (%s), reconnecting in %ds (attempt %d)",
                    exc.rcvd.code if exc.rcvd else "?", delay, _retry + 1,
                )
                await asyncio.sleep(delay)
                _retry += 1
                await self.connect()
            except Exception as exc:
                delay = _LOOP_DELAYS[min(_retry, len(_LOOP_DELAYS) - 1)]
                logger.error(
                    "Gemini processing error (attempt %d), reconnecting in %ds: %s",
                    _retry + 1, delay, exc,
                )
                await asyncio.sleep(delay)
                _retry += 1
                try:
                    await self.connect()
                except Exception:
                    logger.exception("Reconnect after error failed — will retry")
    except _CancelledError:
        logger.debug("Processing loop has been cancelled")

_gemini_rt.GeminiRealtime._processing_loop = _patched_processing_loop
# ──────────────────────────────────────────────────────────────────────────────

# ── SDK monkey-patch: fix ParticipantLeftEvent not cleaning up video tracks ───
# When a participant leaves, _on_track_removed in StreamEdge relies on
# event.participant.published_tracks to find which tracks to remove.
# The Stream SFU proto often sends this as an empty list, so no TrackRemovedEvent
# is fired → Gemini keeps receiving video frames indefinitely, burning credits.
# Fix: on ParticipantLeftEvent, scan the full _track_map for the leaving
# participant and emit TrackRemovedEvent for every track that was published.
import vision_agents.plugins.getstream.stream_edge_transport as _stream_edge
from vision_agents.core.edge import events as _edge_events

_orig_stream_on_track_removed = _stream_edge.StreamEdge._on_track_removed

async def _patched_stream_on_track_removed(self, event):
    if not event.payload:
        return

    # Determine user_id + session_id from whichever field is populated
    if event.participant and event.participant.user_id:
        user_id  = event.participant.user_id
        session_id = event.participant.session_id
    else:
        user_id  = event.payload.user_id
        session_id = event.payload.session_id

    is_participant_left = not hasattr(event.payload, "type")

    if is_participant_left:
        # Scan _track_map for ALL published tracks belonging to this participant.
        # Don't rely on event.participant.published_tracks — it's often empty.
        for track_key, track_info in list(self._track_map.items()):
            tk_user, tk_session, tk_type_int = track_key
            if (tk_user == user_id and tk_session == session_id
                    and track_info.get("published")):
                track_type = _stream_edge._to_core_track_type(tk_type_int)
                self.events.send(
                    _edge_events.TrackRemovedEvent(
                        plugin_name="getstream",
                        track_id=track_info["track_id"],
                        track_type=track_type,
                        participant=_stream_edge._to_core_participant(event.participant),
                    )
                )
                self._track_map[track_key]["published"] = False
                logger.info(
                    "🛑 Cleaned up track %s for departed participant %s",
                    track_info["track_id"], user_id,
                )
        return  # handled — skip original

    # For TrackUnpublishedEvent use the original logic
    await _orig_stream_on_track_removed(self, event)

_stream_edge.StreamEdge._on_track_removed = _patched_stream_on_track_removed
# ──────────────────────────────────────────────────────────────────────────────

# ── SDK monkey-patch: create_call ──────────────────────────────────────────────
# The SDK's create_call passes data as a plain dict {"created_by_id": ...} which
# the getstream REST client doesn't serialize the same way as a CallRequest
# dataclass, causing a 400 from the Stream API. Patch it to use CallRequest.
async def _patched_create_call(self, call_id: str, **kwargs):
    call_type = kwargs.get("call_type", "default")
    # agents.py always passes agent_user_id as a kwarg; self.agent_user_id is None
    # on a fresh instance because create_user hasn't run yet on the real session.
    user_id = kwargs.get("agent_user_id") or self.agent_user_id or "playbyt-agent"
    call = self.client.video.call(call_type, call_id)

    # Retry with exponential backoff — Stream API can time out on first attempt
    # especially after a cold start or brief network blip.
    _retryable = (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError,
                  httpx.RemoteProtocolError, httpx.NetworkError)
    for attempt in range(1, 6):  # Up to 5 attempts
        try:
            await call.get_or_create(data=CallRequest(created_by_id=user_id))
            return call
        except _retryable as exc:
            wait = min(2 ** attempt, 30)  # 2, 4, 8, 16, 30 seconds
            logger.warning(
                "Stream API unreachable (attempt %d/5): %s — retrying in %ds…",
                attempt, type(exc).__name__, wait,
            )
            if attempt == 5:
                raise
            await asyncio.sleep(wait)
    return call  # unreachable, satisfies type checker

StreamEdge.create_call = _patched_create_call
# ───────────────────────────────────────────────────────────────────────────────

# Suppress noisy H264/VP8 decode errors from aiortc \u2014 these fire when the SDK
# demo browser or any H264 source joins; not actionable and flood the logs.
logging.getLogger("libav.h264").setLevel(logging.CRITICAL)
logging.getLogger("libav.libvpx").setLevel(logging.CRITICAL)
logging.getLogger("aiortc.codecs.h264").setLevel(logging.CRITICAL)
logging.getLogger("aiortc.codecs.vpx").setLevel(logging.CRITICAL)

load_dotenv()

CALL_ID_FILE = Path(__file__).parent / ".call_id"
HIGHLIGHTS_FILE = Path(__file__).parent / ".highlights.json"
REPORT_FILE = Path(__file__).parent / ".report.json"
TRANSCRIPT_FILE = Path(__file__).parent / ".transcript.json"
STATUS_FILE = Path(__file__).parent / ".status.json"
QUESTIONS_FILE = Path(__file__).parent / ".questions.json"
PRESENCE_FILE = Path(__file__).parent / ".presence.json"

# ── Presence Timeout ────────────────────────────────────────────────
# How long to keep running after the last frontend heartbeat.
# Frontend pings /api/presence every 20s.  3 missed pings (60s) → empty.
_PRESENCE_TIMEOUT = 60.0


def _room_has_users() -> bool:
    """Return True if at least one user is actively in the room.
    The frontend POSTs /api/presence every 20s while a user is connected.
    If no heartbeat for 60 seconds we treat the room as empty and skip
    sending to Gemini — avoiding burning API credits when nobody is watching.
    """
    data = _safe_read_json(PRESENCE_FILE, fallback=None)
    if not data:
        return False
    last_seen = data.get("last_seen", 0)
    return (time.time() - last_seen) < _PRESENCE_TIMEOUT

# ── Game State ──────────────────────────────────────────────────────────
game_state: dict = {
    "highlights": [],
    "participant_count": 0,
    "start_time": None,
}

# ── Restart backoff — prevents crash-reconnect storm ────────────────────
# Runner calls join_call again immediately on any exception.
# We track restart count and sleep with exponential backoff so a persistent
# Gemini 1011 / service-unavailable doesn't spin-log 23M lines.
_restart_count: int = 0
_restart_last: float = 0.0
_RESTART_DELAYS = [0, 5, 15, 30, 60, 120, 300]  # seconds — cap at 5 min

# ── Gemini Send Lock ──────────────────────────────────────────────────
# Prevents concurrent sends to the Gemini WebSocket which causes 1011 crashes.
# All simple_response() calls must acquire this lock before sending.
_gemini_send_lock = asyncio.Lock()
_backoff_until: float = 0.0  # timestamp — skip all sends until this time passes

# ── Bounded I/O Thread Pool ────────────────────────────────────────────
# 2-worker pool for all file writes. Avoids spawning a new OS thread on every
# highlight/transcript write, which was starving the Gemini WebSocket receive loop.
_io_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="playbyt-io"
)

# ── Transcript Ring Buffer ─────────────────────────────────────────────
# Stores agent speech transcripts for the frontend to poll
_transcript_lines: list[dict] = []
_transcript_counter = 0

# ── Transcript Chunk Buffer (sentence-level batching) ──────────────────
# The SDK fires RealtimeAgentSpeechTranscriptionEvent per streaming chunk
# (1-2 words each).  We accumulate chunks here and only write a transcript
# entry when a sentence boundary is detected or a silence gap elapses.
_chunk_buffer: list[str] = []
_chunk_flush_task: asyncio.Task | None = None
_CHUNK_FLUSH_DELAY = 2.5  # seconds of silence before flushing partial sentence


async def _flush_chunk_buffer() -> None:
    """Flush accumulated speech chunks into a single transcript entry."""
    global _chunk_flush_task
    if not _chunk_buffer:
        return
    text = " ".join(_chunk_buffer).strip()
    _chunk_buffer.clear()
    _chunk_flush_task = None
    if text:
        await _append_transcript(text, source="agent")


async def _buffer_chunk(text: str) -> None:
    """
    Buffer a streaming speech chunk.  Flushes to transcript when:
      - A sentence boundary is detected (. ! ?)
      - OR a 2.5-second gap occurs (agent stopped talking mid-sentence)
    """
    global _chunk_flush_task
    cleaned = text.strip()
    if not cleaned:
        return

    _chunk_buffer.append(cleaned)

    # Cancel any pending flush timer — we're still receiving chunks
    if _chunk_flush_task and not _chunk_flush_task.done():
        _chunk_flush_task.cancel()

    # Check if the accumulated text ends with a sentence boundary
    full_text = " ".join(_chunk_buffer)
    if full_text.rstrip().endswith((".", "!", "?", "…")):
        await _flush_chunk_buffer()
    else:
        # Schedule a delayed flush in case the agent stops mid-sentence
        _chunk_flush_task = asyncio.ensure_future(_delayed_flush())


async def _delayed_flush() -> None:
    """Wait for the silence gap, then flush whatever is buffered."""
    await asyncio.sleep(_CHUNK_FLUSH_DELAY)
    await _flush_chunk_buffer()


async def _append_transcript(text: str, source: str = "agent") -> None:
    """Add a transcript line and persist to disk for the API."""
    global _transcript_counter
    _transcript_counter += 1
    entry = {
        "id": _transcript_counter,
        "text": text,
        "source": source,
        "timestamp": time.time(),
        "elapsed": round(time.time() - game_state["start_time"]) if game_state["start_time"] else 0,
    }
    _transcript_lines.append(entry)
    # Keep last 100 lines
    if len(_transcript_lines) > 100:
        _transcript_lines[:] = _transcript_lines[-100:]
    # Write via bounded thread pool — reuses threads instead of spawning new ones
    asyncio.get_running_loop().run_in_executor(
        _io_executor, _safe_write_json, TRANSCRIPT_FILE, list(_transcript_lines[-50:])
    )


# ── Agent Status ───────────────────────────────────────────────────────
_agent_status: dict = {
    "gemini": "disconnected",
    "yolo": "standby",
    "commentary_loop": "off",
    "frames_processed": 0,
    "last_commentary": 0,
}


_last_status_write: float = 0.0


def _update_status(**kwargs: Any) -> None:
    """Update in-memory status; write to disk at most every 10 seconds."""
    global _last_status_write
    _agent_status.update(kwargs)
    now = time.time()
    if now - _last_status_write >= 10:
        _last_status_write = now
        try:
            _safe_write_json(STATUS_FILE, _agent_status)
        except Exception:
            pass


def _persist_call_id(call_type: str, call_id: str) -> None:
    """Write the active call ID to disk so the token server can serve it."""
    CALL_ID_FILE.write_text(json.dumps({"call_type": call_type, "call_id": call_id}))
    logger.info("Call ID persisted → %s", CALL_ID_FILE)


def _safe_read_json(path: Path, fallback: Any = None) -> Any:
    """Read a JSON file with shared file locking to avoid partial reads."""
    if not path.exists():
        return fallback
    try:
        with open(path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data
    except Exception:
        return fallback


def _safe_write_json(path: Path, data: Any) -> None:
    """Write JSON to disk with file locking to prevent race conditions with the server."""
    try:
        with open(path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(data, indent=2))
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.warning("Failed to write %s: %s", path, e)


def _save_highlights() -> None:
    """Persist highlights to disk for the API."""
    _safe_write_json(HIGHLIGHTS_FILE, game_state["highlights"])


# ── Agent Factory ───────────────────────────────────────────────────────
async def create_agent(**kwargs) -> Agent:
    """Create the PlayByt sports analyst agent with tools."""

    llm = gemini.Realtime(fps=5)

    # ── Tool: Log Highlight ─────────────────────────────────────────
    @llm.register_function(
        description=(
            "Log a key match highlight. Call this for goals, cards, big saves, "
            "controversial decisions, injuries, or any moment worth remembering. "
            "Provide a short vivid description of what happened."
        )
    )
    async def log_highlight(description: str, category: str = "moment") -> str:
        """Log a highlight moment during the match."""
        highlight = {
            "id": len(game_state["highlights"]) + 1,
            "description": description,
            "category": category,
            "timestamp": time.time(),
            "elapsed": (
                round(time.time() - game_state["start_time"])
                if game_state["start_time"]
                else 0
            ),
        }
        game_state["highlights"].append(highlight)
        asyncio.get_running_loop().run_in_executor(
            _io_executor, _safe_write_json, HIGHLIGHTS_FILE, list(game_state["highlights"])
        )
        logger.info("⚡ Highlight #%d: %s", highlight["id"], description)
        return f"Highlight #{highlight['id']} logged: {description}"

    # ── Tool: Get Match Summary ─────────────────────────────────────
    @llm.register_function(
        description=(
            "Generate a summary of the match so far. Call this when a user asks "
            "for a recap, summary, or 'what did I miss'. Returns all logged highlights."
        )
    )
    async def get_match_summary() -> str:
        """Return all logged highlights as a match summary."""
        if not game_state["highlights"]:
            return "No highlights logged yet. The match is still developing."

        elapsed_total = (
            round(time.time() - game_state["start_time"])
            if game_state["start_time"]
            else 0
        )
        mins = elapsed_total // 60

        lines = [f"Match summary ({mins} min watched, {len(game_state['highlights'])} key moments):"]
        for h in game_state["highlights"]:
            m = h["elapsed"] // 60
            s = h["elapsed"] % 60
            lines.append(f"  [{m:02d}:{s:02d}] {h['description']}")

        return "\n".join(lines)

    # ── Tool: Get Highlight Count ───────────────────────────────────
    @llm.register_function(
        description="Get the number of highlights logged so far in this session."
    )
    async def get_highlight_count() -> str:
        """Return highlight count."""
        count = len(game_state["highlights"])
        return f"{count} highlight{'s' if count != 1 else ''} logged so far."

    # ── Tool: Web Search (sports stats, player info, scores) ────────
    @llm.register_function(
        description=(
            "Search the web for sports stats, player info, team records, live scores, "
            "or any sports fact you cannot see on screen. Use this when a user asks "
            "about a specific player's stats, match history, or any verifiable fact. "
            "Returns text snippets from search results. Query should be specific."
        )
    )
    async def web_search(query: str) -> str:
        """Search the web for sports information."""
        logger.info("🔍 Web search: %s", query)
        # Try DuckDuckGo instant answers (no API key needed)
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                )
                data = r.json()
                results = []
                if data.get("AbstractText"):
                    results.append(data["AbstractText"])
                for topic in (data.get("RelatedTopics") or [])[:3]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(topic["Text"])
                if results:
                    return " | ".join(results[:3])
        except Exception as e:
            logger.debug("DuckDuckGo search failed: %s", e)

        return (
            "Search unavailable right now. Based on what I can see on screen, "
            "I'll give my best analysis from the visual data."
        )

    # ── Custom Sports Intelligence Processor ────────────────────────
    sports = SportsProcessor(
        model_path="yolo11n-pose.pt",
        conf_threshold=0.5,
        fps=1,
    )

    # ── Tool: Get Field Analysis ────────────────────────────────────
    @llm.register_function(
        description=(
            "Get the current real-time field analysis computed from YOLO pose data. "
            "Returns player count, zone distribution, formation estimate, pressing "
            "intensity, fatigue flags, and dominant side. Call this when you want "
            "to give a tactical breakdown or when users ask about player positioning, "
            "fatigue, formation, or pressing. This data comes from computer vision "
            "analysis that humans cannot compute in real time."
        )
    )
    async def get_field_analysis() -> str:
        """Return the latest sports intelligence analysis."""
        a = sports.latest_analysis
        if not a or a.get("player_count", 0) == 0:
            return "No players currently detected in frame."

        lines = [
            f"Players tracked: {a['player_count']}",
            f"Formation: {a['formation']}",
            f"Pressing intensity: {a['pressing_intensity']}",
            f"Dominant side: {a['dominant_side']}",
            f"Zones — L:{a['zones']['left']} C:{a['zones']['center']} R:{a['zones']['right']}",
            f"Thirds — Def:{a['zones']['def_third']} Mid:{a['zones']['mid_third']} Att:{a['zones']['att_third']}",
        ]

        if a["fatigue_flags"]:
            for f in a["fatigue_flags"]:
                lines.append(
                    f"⚠ Player {f['player_id']+1}: {f['severity']} fatigue "
                    f"(spine angle {f['spine_angle']}°)"
                )

        trend = sports.get_trend()
        if trend.get("trend") != "insufficient_data":
            lines.append(f"Trend: {trend.get('player_movement', 'stable')}")
            lines.append(f"Fatigue events (last 10 frames): {trend.get('fatigue_events_last_10_frames', 0)}")

        return "\n".join(lines)

    # ── Tool: Get Controversy Alerts ────────────────────────────────
    @llm.register_function(
        description=(
            "Get recent controversy or threshold-based alerts detected automatically "
            "by the Sports Intelligence Processor. Alerts include pressing spikes, "
            "formation changes, fatigue spikes, and side overloads. Call this when "
            "a user asks about controversies, formations changes, or suspicious events."
        )
    )
    async def get_controversy_alerts() -> str:
        """Return recent auto-detected controversy alerts."""
        alerts = sports.get_latest_controversies(limit=5)
        if not alerts:
            return "No controversy alerts detected yet."
        lines = ["Recent alerts:"]
        for a in alerts:
            m = a["elapsed"] // 60
            s = a["elapsed"] % 60
            lines.append(f"  [{m:02d}:{s:02d}] {a['title']}: {a['description']}")
        return "\n".join(lines)

    # ── Tool: Export Match Report ───────────────────────────────────
    @llm.register_function(
        description=(
            "Generate and export a comprehensive post-match report covering all "
            "highlights, controversy alerts, tactical observations, and player "
            "fatigue data from this session. Call this when a user asks for a "
            "match report, full analysis, or wants to save/export the session."
        )
    )
    async def export_match_report() -> str:
        """Generate a post-match report and persist it to disk."""
        elapsed_total = (
            round(time.time() - game_state["start_time"])
            if game_state["start_time"]
            else 0
        )
        trend = sports.get_trend()
        controversies = sports.get_latest_controversies(limit=50)
        a = sports.latest_analysis

        report = {
            "generated_at": time.time(),
            "duration_seconds": elapsed_total,
            "duration_formatted": f"{elapsed_total // 60}m {elapsed_total % 60}s",
            "highlights_count": len(game_state["highlights"]),
            "highlights": game_state["highlights"],
            "controversies_count": len(controversies),
            "controversies": controversies,
            "final_analysis": {
                "player_count": a.get("player_count", 0),
                "formation": a.get("formation", "N/A"),
                "pressing_intensity": a.get("pressing_intensity", "none"),
                "dominant_side": a.get("dominant_side", "balanced"),
                "fatigue_flags": a.get("fatigue_flags", []),
            } if a else {},
            "trend_summary": trend,
            "frames_analyzed": trend.get("frames_analyzed", 0),
        }

        REPORT_FILE.write_text(json.dumps(report, indent=2))
        logger.info("📄 Match report exported → %s", REPORT_FILE)

        return (
            f"Match report exported: {len(game_state['highlights'])} highlights, "
            f"{len(controversies)} alerts, {elapsed_total // 60} min watched, "
            f"{trend.get('frames_analyzed', 0)} frames analyzed. "
            f"Available at /api/report"
        )

    # ── Build Agent ─────────────────────────────────────────────────
    instructions_path = Path(__file__).parent / "instructions.md"
    if instructions_path.exists():
        instructions = "Read @instructions.md"
    else:
        logger.warning("instructions.md not found — using built-in fallback instructions")
        instructions = (
            "You are PlayByt, an AI sports analyst. You watch live sports streams "
            "via screen share, analyze player positions and tactics using YOLO pose data, "
            "and provide real-time commentary. Log highlights for key moments. "
            "Use get_field_analysis() for tactical data. Be concise and insightful."
        )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="PlayByt", id="playbyt-agent"),
        instructions=instructions,
        llm=llm,
        processors=[sports],
    )

    return agent


# ── Proactive Commentary Loop ──────────────────────────────────────────


async def _send_to_gemini(agent: Agent, prompt: str, label: str) -> bool:
    """
    Send a text prompt to Gemini through the shared send lock.
    Returns True if the prompt was delivered, False otherwise.
    Sets backoff on crash.
    """
    global _backoff_until
    # Hard gate: never burn Gemini credits when nobody is in the room.
    # This is the final safety net regardless of which loop calls us.
    if not _room_has_users():
        logger.debug("%s skipped — room empty, no Gemini call made", label)
        return False
    if time.time() < _backoff_until:
        return False
    try:
        async with _gemini_send_lock:
            await asyncio.wait_for(
                agent.llm.simple_response(text=prompt),
                timeout=12.0,
            )
        _update_status(last_commentary=time.time())
        logger.info("🎙️ %s delivered", label)
        return True
    except asyncio.TimeoutError:
        logger.debug("%s timed out", label)
        return False
    except Exception as e:
        err_str = str(e)
        if "1008" in err_str or "1011" in err_str or "ConnectionClosed" in type(e).__name__:
            _backoff_until = time.time() + 25
            logger.warning("🔴 Gemini crash (%s) — backing off 25s", type(e).__name__)
        else:
            logger.debug("%s failed: %s", label, e)
        return False


async def _event_watcher(agent: Agent, sports: SportsProcessor) -> None:
    """
    Watch the SportsProcessor event queue and fire commentary IMMEDIATELY
    when a controversy is detected. This is what makes PlayByt reactive —
    catching the exact moment something changes, faster than any human.
    """
    global _backoff_until
    await asyncio.sleep(20)  # Let agent fully stabilize first
    logger.info("⚡ Event watcher started — will react to controversies instantly")

    while True:
        try:
            # Block until a controversy event arrives (with timeout so we can check cancel)
            try:
                alert = await asyncio.wait_for(sports._event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue  # No event — loop back and wait again

            # Respect backoff
            if time.time() < _backoff_until:
                continue

            # Don't fire alerts into an empty room
            if not _room_has_users():
                continue

            # Don't fire alerts if screen share is gone
            if sports.last_frame_time == 0.0 or (time.time() - sports.last_frame_time) > 30:
                continue

            prompt = (
                f"[REAL-TIME ALERT] {alert['title']}: {alert['description']}. "
                f"This JUST happened on screen. React immediately — what does this mean "
                f"tactically? Keep it under 2 sentences. Be excited if it's a big change."
            )
            await _send_to_gemini(agent, prompt, f"Event: {alert['title']}")

        except asyncio.CancelledError:
            logger.info("⚡ Event watcher cancelled")
            return
        except Exception as e:
            logger.debug("Event watcher error: %s", e)


async def _commentary_loop(agent: Agent, sports: SportsProcessor) -> None:
    """
    Drive continuous proactive commentary every ~15 seconds.
    This is the HEARTBEAT — the agent always talks, never goes silent.

    PRIORITY: Questions are answered FIRST. If a user question is pending,
    it takes the tick instead of commentary.  This ensures questions never
    get starved by the constant commentary stream.

    Two commentary modes:
    - YOLO data available → include key data AND reference the HUD overlay
    - YOLO data unavailable → vision-only commentary from the raw video feed
    """
    global _backoff_until
    _update_status(commentary_loop="starting")
    await asyncio.sleep(15)  # Let Gemini session stabilize
    _update_status(commentary_loop="active")
    logger.info("🎙️ Commentary loop started — agent will speak every ~15s")

    tick = 0
    while True:
        try:
            tick += 1

            # Skip tick if recovering from a Gemini WebSocket crash
            if time.time() < _backoff_until:
                logger.debug("Commentary tick #%d skipped (backoff)", tick)
                await asyncio.sleep(15)
                continue

            # Skip tick if nobody is in the room — don't burn Gemini credits idle
            if not _room_has_users():
                logger.debug("Commentary tick #%d skipped (room empty)", tick)
                await asyncio.sleep(15)
                continue

            # Skip tick if no video frames have arrived in the last 30s —
            # screen share has stopped or hasn't started yet.
            # Avoids agent commenting on a frozen/last frame after share ends.
            video_age = time.time() - sports.last_frame_time
            if sports.last_frame_time == 0.0 or video_age > 30:
                logger.debug("Commentary tick #%d skipped (no video — last frame %.0fs ago)", tick, video_age)
                await asyncio.sleep(15)
                continue

            # ── PRIORITY: Answer pending questions FIRST ──────────────
            question_handled = False
            try:
                questions = _safe_read_json(QUESTIONS_FILE, fallback=[])
                pending = [q for q in questions if not q.get("answered")]
                if pending:
                    q = pending[0]
                    q["answered"] = True
                    user = q.get("user", "Fan")
                    question_text = q.get("question", "")
                    logger.info("❓ Commentary tick yielded to question from %s: %s", user, question_text)
                    await _append_transcript(question_text, source="user")

                    prompt = (
                        f"[USER QUESTION from {user}]: \"{question_text}\"\n"
                        f"Answer this question directly and helpfully. "
                        f"If it's about the game, use what you see on screen "
                        f"and the data from the HUD overlay. "
                        f"If it's about stats or players, give your best answer. "
                        f"Keep it under 3 sentences."
                    )
                    sent = await _send_to_gemini(agent, prompt, f"Question from {user}")
                    if sent:
                        question_handled = True
                    _safe_write_json(QUESTIONS_FILE, questions)
            except Exception as e:
                logger.debug("Question check in commentary loop error: %s", e)

            if question_handled:
                await asyncio.sleep(15)
                continue

            # ── Commentary ────────────────────────────────────────────
            analysis = sports.latest_analysis
            player_count = analysis.get("player_count", 0) if analysis else 0

            if player_count > 0:
                # ── Mode 1: YOLO data available ──
                # Include key data points in text AND reference the visual HUD.
                # Gemini performs best with BOTH text data + visual overlay.
                formation = analysis.get("formation", "unknown")
                pressing = analysis.get("pressing_intensity", "unknown")
                dominant = analysis.get("dominant_side", "balanced")
                zones = analysis.get("zones", {})
                zone_lr = f"L:{zones.get('left', 0)} C:{zones.get('center', 0)} R:{zones.get('right', 0)}"
                thirds = f"Def:{zones.get('def_third', 0)} Mid:{zones.get('mid_third', 0)} Att:{zones.get('att_third', 0)}"

                fatigue_info = ""
                fatigue_flags = analysis.get("fatigue_flags", [])
                if fatigue_flags:
                    fatigue_info = f" Fatigue alert: {len(fatigue_flags)} player(s) showing signs."

                prompts = [
                    (
                        f"[LIVE DATA] {player_count} players tracked | Formation: {formation} | "
                        f"Pressing: {pressing} | Zones: {zone_lr} | Thirds: {thirds} | "
                        f"Dominant side: {dominant}.{fatigue_info}\n"
                        f"The PLAYBYT HUD overlay on screen shows this data visually. "
                        f"Give one sharp tactical insight about what this means for the match. "
                        f"Be specific — mention formations, pressing, or positioning. One sentence."
                    ),
                    (
                        f"[LIVE DATA] {player_count} players | {formation} formation | "
                        f"Pressing {pressing} | {dominant} side dominant.{fatigue_info}\n"
                        f"Look at the HUD and the match action. "
                        f"What would a casual viewer miss right now? "
                        f"Point out something only an analyst would notice. One sentence."
                    ),
                    (
                        f"[LIVE DATA] Zones: {zone_lr} | Thirds: {thirds} | "
                        f"Formation: {formation} | Pressing: {pressing}.{fatigue_info}\n"
                        f"Combine this data with what you see on screen. "
                        f"What's the tactical story? Are they attacking, defending, transitioning? "
                        f"One energetic sentence."
                    ),
                ]
                prompt = prompts[tick % len(prompts)]
            else:
                # ── Mode 2: YOLO can't see players — vision-only fallback ──
                # Gemini still receives full video at 5 FPS. Use it.
                prompts = [
                    (
                        "Look at the screen right now and describe what's happening "
                        "in the match. One sentence of live commentary."
                    ),
                    (
                        "What's the current state of play? Describe what you see "
                        "on the broadcast. Keep it to one energetic sentence."
                    ),
                    (
                        "Give a quick observation about what's happening on screen "
                        "right now. What would the viewer want to know? One sentence."
                    ),
                ]
                prompt = prompts[tick % len(prompts)]

            await _send_to_gemini(agent, prompt, f"Commentary tick #{tick}")

        except asyncio.CancelledError:
            logger.info("🎙️ Commentary loop cancelled")
            _update_status(commentary_loop="stopped")
            return
        except Exception as e:
            err_str = str(e)
            if "1008" in err_str or "1011" in err_str or "ConnectionClosed" in type(e).__name__:
                _backoff_until = time.time() + 25
                logger.warning("🔴 Gemini crash (outer) — backing off 25s")
            else:
                logger.debug("Commentary loop error: %s", e)

        # Consistent 15s between ticks
        await asyncio.sleep(15)


# ── Backup Question Loop ────────────────────────────────────────────────
async def _question_loop(agent: Agent) -> None:
    """
    Backup loop that catches any questions the commentary loop didn't handle.
    Questions are primarily answered inside _commentary_loop (priority check),
    but this loop catches edge cases like backoff periods or timing gaps.
    Checks every 12 seconds — staggered from commentary's 15s cycle.
    """
    global _backoff_until
    await asyncio.sleep(18)  # Start after commentary loop is active
    logger.info("❓ Backup question loop started — checking every 12s")

    while True:
        try:
            if time.time() < _backoff_until:
                await asyncio.sleep(12)
                continue

            # Don't answer questions into an empty room — burns credits for nobody
            if not _room_has_users():
                await asyncio.sleep(12)
                continue

            questions = _safe_read_json(QUESTIONS_FILE, fallback=[])
            pending = [q for q in questions if not q.get("answered")]
            if pending:
                q = pending[0]  # Handle one at a time
                q["answered"] = True
                user = q.get("user", "Fan")
                question_text = q.get("question", "")
                logger.info("❓ Backup answering question from %s: %s", user, question_text)
                await _append_transcript(question_text, source="user")

                prompt = (
                    f"[USER QUESTION from {user}]: \"{question_text}\"\n"
                    f"Answer this question directly and helpfully. "
                    f"If it's about the game, use what you see on screen "
                    f"and the HUD data. "
                    f"Keep it under 3 sentences."
                )
                await _send_to_gemini(agent, prompt, f"Question from {user}")
                _safe_write_json(QUESTIONS_FILE, questions)
        except asyncio.CancelledError:
            logger.info("❓ Question loop cancelled")
            return
        except Exception as e:
            logger.debug("Question check error: %s", e)

        await asyncio.sleep(12)


# ── Transcript Capture via SDK Events ────────────────────────────────────
def _setup_transcript_capture(agent: Agent) -> None:
    """
    Subscribe to the SDK's proper event system to capture agent speech.
    Falls back to a Union-typed handler if the SDK rejects single-event
    subscription (SDK versions >= 0.4 require Union[EventType, ...]).
    User speech is NEVER captured — privacy first.
    """
    # Attempt 1: single-event subscription (works on older SDK versions)
    try:
        @agent.llm.events.subscribe
        async def _on_agent_speech(
            event: RealtimeAgentSpeechTranscriptionEvent,
        ) -> None:
            if event.text:
                await _buffer_chunk(event.text)

        logger.info("🎙️ Transcript capture hooked via SDK events (single-event)")
        return
    except Exception:
        pass  # SDK rejected single-event — try Union form below

    # Attempt 2: Union-typed handler (required by newer SDK versions)
    try:
        from typing import Union as _Union

        @agent.llm.events.subscribe
        async def _on_agent_speech_union(
            event: _Union[RealtimeAgentSpeechTranscriptionEvent, None],
        ) -> None:
            if isinstance(event, RealtimeAgentSpeechTranscriptionEvent) and event.text:
                await _buffer_chunk(event.text)

        logger.info("🎙️ Transcript capture hooked via SDK events (union-event)")
        return
    except Exception as e:
        logger.warning("Transcript capture setup failed (both methods): %s — transcripts may be missing", e)


# ── Call Lifecycle ──────────────────────────────────────────────────────
async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a Stream call and run until it ends."""
    global _restart_count, _restart_last

    # ── Exponential backoff on repeated restarts ──────────────────────────────
    # If the Runner is restarting us quickly (within 30s), back off so a
    # persistent Gemini 1011 / service-unavailable doesn't flood the log.
    now = time.time()
    if _restart_count > 0 and (now - _restart_last) < 30:
        delay = _RESTART_DELAYS[min(_restart_count, len(_RESTART_DELAYS) - 1)]
        logger.warning(
            "Gemini restart #%d — backing off %ds before reconnecting…",
            _restart_count, delay,
        )
        await asyncio.sleep(delay)
    _restart_last = time.time()
    _restart_count += 1
    # ─────────────────────────────────────────────────────────────────────────
    _persist_call_id(call_type, call_id)
    game_state["start_time"] = time.time()
    game_state["highlights"] = []
    _save_highlights()
    _safe_write_json(TRANSCRIPT_FILE, [])
    _safe_write_json(QUESTIONS_FILE, [])
    _update_status(gemini="connecting", yolo="starting")

    call = await agent.create_call(call_type, call_id)

    # Get the SportsProcessor instance from the agent's processors
    sports = None
    for proc in (agent.processors if hasattr(agent, 'processors') else []):
        if isinstance(proc, SportsProcessor):
            sports = proc
            break

    async with agent.join(call):
        _restart_count = 0  # successful connection — reset backoff
        _update_status(gemini="connected", yolo="active")
        _setup_transcript_capture(agent)

        # Let WebRTC + Gemini session fully handshake before the first send
        await asyncio.sleep(3)

        # Send greeting ONLY if a user is already in the room.
        # If nobody is present yet, skip — saves one Gemini call on every reconnect.
        if _room_has_users():
            try:
                async with _gemini_send_lock:
                    await asyncio.wait_for(
                        agent.llm.simple_response(
                            text="PlayByt online. Share your screen and I will catch what you miss."
                        ),
                        timeout=10.0,
                    )
                await _append_transcript(
                    "PlayByt online. Share your screen and I will catch what you miss.",
                    source="agent",
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.info("Startup greeting skipped: %s", e)
        else:
            logger.info("Startup greeting skipped — room empty, saving Gemini credits")

        logger.info("PlayByt is live — starting commentary loop...")

        async def _resilient(factory, name: str, restart_delay: float = 10.0):
            """Run factory() as a coroutine, restarting on unexpected crashes.
            This is a last-resort wrapper — each loop already handles its own
            inner exceptions.  This catches anything that slips through.
            """
            while True:
                try:
                    await factory()
                    return  # clean exit (e.g. CancelledError propagated)
                except asyncio.CancelledError:
                    raise  # always propagate cancellation
                except Exception as exc:
                    logger.error(
                        "🔴 Loop '%s' crashed unexpectedly — restarting in %.0fs: %s",
                        name, restart_delay, exc,
                    )
                    await asyncio.sleep(restart_delay)

        # Start commentary + event watcher + question loops
        commentary_task = None
        question_task = None
        event_task = None
        video_guard_task = None
        if sports:
            commentary_task = asyncio.ensure_future(
                _resilient(lambda: _commentary_loop(agent, sports), "commentary")
            )
            event_task = asyncio.ensure_future(
                _resilient(lambda: _event_watcher(agent, sports), "event_watcher")
            )
            question_task = asyncio.ensure_future(
                _resilient(lambda: _question_loop(agent), "question")
            )
            logger.info("🎙️ Commentary + event watcher + question loops scheduled")
        else:
            logger.warning("No SportsProcessor found — commentary loop disabled")

        # Safety-net: stop Gemini video when room is empty even if SDK track
        # removal event was missed (e.g. participant_left with empty published_tracks)
        async def _video_guard():
            """Kill video forwarding when room is empty >30s OR screen share stopped >30s.

            Two independent conditions are checked every 10s:
            1. Room empty >30s — user left without the SDK firing a proper leave event.
            2. No video frames received for >30s (sports.last_frame_time stale) while
               Gemini still holds an active video forwarder — this is the "stopped screen
               share but stayed in room" case.  Without this, Gemini keeps processing the
               frozen last frame indefinitely and the commentary loop keeps firing on it.
            """
            was_empty = False
            empty_since = 0.0
            while True:
                try:
                    await asyncio.sleep(10)

                    video_active = getattr(agent.llm, "_video_forwarder", None) is not None

                    # ── Condition 1: room empty ──────────────────────────────────
                    room_empty = not _room_has_users()
                    if room_empty and not was_empty:
                        was_empty = True
                        empty_since = time.time()
                    elif not room_empty:
                        was_empty = False
                        empty_since = 0.0

                    if was_empty and (time.time() - empty_since) > 30 and video_active:
                        try:
                            await agent.llm.stop_watching_video_track()
                            logger.info("🛑 Video guard: stopped Gemini video (room empty >30s)")
                        except Exception as exc:
                            logger.warning("Video guard stop (empty room) failed: %s", exc)
                        continue  # re-evaluate next tick

                    # ── Condition 2: screen share stopped (no frames for >30s) ───
                    # sports.last_frame_time > 0 means we received at least one frame
                    # this session, so stale = sharing genuinely stopped.
                    if (
                        sports is not None
                        and sports.last_frame_time > 0.0
                        and video_active
                    ):
                        frame_age = time.time() - sports.last_frame_time
                        if frame_age > 30:
                            try:
                                await agent.llm.stop_watching_video_track()
                            except Exception as exc:
                                logger.warning("Video guard stop_watching failed: %s", exc)
                            try:
                                await sports.stop_processing()  # resets last_frame_time → 0.0
                            except Exception as exc:
                                logger.warning("Video guard stop_processing failed: %s", exc)
                            logger.info(
                                "🛑 Video guard: stopped Gemini video (no frames for %.0fs"
                                " — screen share stopped)",
                                frame_age,
                            )

                except asyncio.CancelledError:
                    raise  # always propagate cancellation cleanly
                except Exception as exc:
                    logger.warning("Video guard tick error (will retry): %s", exc)

        video_guard_task = asyncio.ensure_future(_video_guard())

        # ── Memory watchdog ───────────────────────────────────────────────────
        # Runs periodic GC to return fragmented memory to OS, and kills the
        # process cleanly if RSS exceeds 1.7 GB so the OS can restart it fresh
        # rather than the OOM killer freezing the whole instance.
        async def _memory_watchdog():
            import gc
            import resource
            import sys as _sys
            while True:
                await asyncio.sleep(30)
                try:
                    gc.collect()
                    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
                    if rss_mb > 1700:
                        logger.error(
                            "💀 Memory watchdog: RSS %d MB > 1700 MB threshold — "
                            "exiting cleanly so supervisor can restart", rss_mb
                        )
                        _sys.exit(1)
                    elif rss_mb > 1400:
                        logger.warning("⚠️  Memory watchdog: RSS %d MB (approaching limit)", rss_mb)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Memory watchdog error: %s", exc)

        memory_watchdog_task = asyncio.ensure_future(_memory_watchdog())

        _shutdown = asyncio.Event()
        try:
            await _shutdown.wait()  # Block until cancelled — cleaner than Future()
        except asyncio.CancelledError:
            pass
        finally:
            for task in (commentary_task, event_task, question_task, video_guard_task, memory_watchdog_task):
                if task:
                    task.cancel()
            await asyncio.gather(
                *(t for t in (commentary_task, event_task, question_task, video_guard_task, memory_watchdog_task) if t),
                return_exceptions=True,
            )
            _update_status(gemini="disconnected", commentary_loop="stopped")
            logger.info("Agent session cancelled — shutting down.")


if __name__ == "__main__":
    import sys
    # Always suppress the SDK demo browser \u2014 it joins as user-demo-agent,
    # sends H264 video that aiortc can't decode, and crashes the WebRTC connection.
    if "run" in sys.argv and "--no-demo" not in sys.argv:
        sys.argv.append("--no-demo")
    Runner(AgentLauncher(
        create_agent=create_agent,
        join_call=join_call,
        agent_idle_timeout=0,  # Never exit while alone — wait for the user to join/share screen
    )).cli()
