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
from vision_agents.plugins import gemini, getstream
from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge

from sports_processor import SportsProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── SDK monkey-patch ───────────────────────────────────────────────────────────
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

# ── Game State ──────────────────────────────────────────────────────────
game_state: dict = {
    "highlights": [],
    "participant_count": 0,
    "start_time": None,
}

# ── Gemini Send Lock ──────────────────────────────────────────────────
# Prevents concurrent sends to the Gemini WebSocket which causes 1011 crashes.
# All simple_response() calls must acquire this lock before sending.
_gemini_send_lock = asyncio.Lock()
_backoff_until: float = 0.0  # timestamp — skip all sends until this time passes

# ── Transcript Ring Buffer ─────────────────────────────────────────────
# Stores agent speech transcripts for the frontend to poll
_transcript_lines: list[dict] = []
_transcript_counter = 0


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
    # Persist async
    asyncio.ensure_future(
        asyncio.to_thread(_safe_write_json, TRANSCRIPT_FILE, _transcript_lines[-50:])
    )


# ── Agent Status ───────────────────────────────────────────────────────
_agent_status: dict = {
    "gemini": "disconnected",
    "yolo": "standby",
    "commentary_loop": "off",
    "frames_processed": 0,
    "last_commentary": 0,
}


def _update_status(**kwargs: Any) -> None:
    """Update agent status and persist."""
    _agent_status.update(kwargs)
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

    llm = gemini.Realtime(fps=3)

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
        asyncio.ensure_future(asyncio.to_thread(
            _safe_write_json, HIGHLIGHTS_FILE, game_state["highlights"]
        ))
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
async def _commentary_loop(agent: Agent, sports: SportsProcessor) -> None:
    """
    Drive continuous proactive commentary every 15-20 seconds.
    This is what makes PlayByt a COMMENTATOR instead of a passive observer.
    
    Every tick:
    1. Read latest YOLO analysis
    2. Build a context prompt with real data
    3. Send it to Gemini as a nudge → Gemini speaks
    4. Log the transcript for the frontend
    """
    global _backoff_until
    _update_status(commentary_loop="starting")
    await asyncio.sleep(15)  # Let Gemini session stabilize before first commentary
    _update_status(commentary_loop="active")
    logger.info("🎙️ Commentary loop started — agent will speak every ~12s")

    tick = 0
    while True:
        try:
            tick += 1

            # Skip tick if recovering from a Gemini WebSocket crash (1011)
            if time.time() < _backoff_until:
                logger.debug("Commentary tick #%d skipped (backoff)", tick)
                await asyncio.sleep(12 + (tick % 4))
                continue

            analysis = sports.latest_analysis
            player_count = analysis.get("player_count", 0) if analysis else 0

            if player_count > 0:
                # Build a rich context nudge with real YOLO data
                zones = analysis.get("zones", {})
                fatigue = analysis.get("fatigue_flags", [])
                formation = analysis.get("formation", "N/A")
                pressing = analysis.get("pressing_intensity", "none")
                dominant = analysis.get("dominant_side", "balanced")

                # Vary the prompt to get different observations each tick
                prompts = [
                    (
                        f"[LIVE DATA] {player_count} players tracked. "
                        f"Formation {formation}. Pressing {pressing}. "
                        f"Zones L:{zones.get('left',0)} C:{zones.get('center',0)} R:{zones.get('right',0)}. "
                        f"Give a sharp 1-sentence tactical observation about what you see on screen right now."
                    ),
                    (
                        f"[LIVE DATA] Dominant side: {dominant}. "
                        f"Thirds — Def:{zones.get('def_third',0)} Mid:{zones.get('mid_third',0)} Att:{zones.get('att_third',0)}. "
                        f"Fatigue flags: {len(fatigue)}. "
                        f"What's the most interesting thing happening on screen right now? One sentence."
                    ),
                    (
                        f"[LIVE DATA] {player_count} visible. Pressing {pressing}. "
                        f"Look at the screen — describe the current play in 1-2 sentences. "
                        f"What did the broadcast miss? Use the HUD numbers."
                    ),
                ]

                # Also sprinkle in controversy/highlight nudges
                recent_controversies = sports.get_latest_controversies(limit=1)
                if recent_controversies and tick % 3 == 0:
                    c = recent_controversies[-1]
                    prompt = (
                        f"[ALERT] {c['title']}: {c['description']}. "
                        f"React to this. What does it mean tactically? Keep it under 2 sentences."
                    )
                else:
                    prompt = prompts[tick % len(prompts)]

                try:
                    async with _gemini_send_lock:
                        await asyncio.wait_for(
                            agent.llm.simple_response(text=prompt),
                            timeout=12.0,
                        )
                    _update_status(last_commentary=time.time())
                    logger.info("🎙️ Commentary tick #%d delivered", tick)
                except asyncio.TimeoutError:
                    logger.debug("Commentary tick #%d timed out", tick)
                except Exception as e:
                    err_str = str(e)
                    if "1011" in err_str or "ConnectionClosed" in type(e).__name__:
                        _backoff_until = time.time() + 20
                        logger.warning(
                            "🔴 Gemini 1011 crash — backing off 20s (tick #%d)", tick
                        )
                    else:
                        logger.debug("Commentary tick #%d failed: %s", tick, e)
            else:
                # No players visible — stay silent, don't describe empty/pixelated frames
                pass

        except asyncio.CancelledError:
            logger.info("🎙️ Commentary loop cancelled")
            _update_status(commentary_loop="stopped")
            return
        except Exception as e:
            err_str = str(e)
            if "1011" in err_str or "ConnectionClosed" in type(e).__name__:
                _backoff_until = time.time() + 20
                logger.warning("🔴 Gemini 1011 crash (outer) — backing off 20s")
            else:
                logger.debug("Commentary loop error: %s", e)

        # Wait 12-15 seconds between comments (slight jitter to feel natural)
        await asyncio.sleep(12 + (tick % 4))


# ── Independent Question Loop ──────────────────────────────────────────
async def _question_loop(agent: Agent) -> None:
    """
    Independent loop that checks for user text questions every 3 seconds.
    Runs separately from commentary so questions get answered faster.
    Uses the same send lock to prevent concurrent Gemini sends.
    """
    global _backoff_until
    await asyncio.sleep(10)  # Let agent settle before accepting questions
    logger.info("❓ Question loop started — checking every 3s")

    while True:
        try:
            # Respect backoff from Gemini crashes
            if time.time() < _backoff_until:
                await asyncio.sleep(3)
                continue

            questions = _safe_read_json(QUESTIONS_FILE, fallback=[])
            pending = [q for q in questions if not q.get("answered")]
            for q in pending:
                q["answered"] = True
                user = q.get("user", "Fan")
                question_text = q.get("question", "")
                logger.info("❓ Answering question from %s: %s", user, question_text)
                await _append_transcript(question_text, source="user")

                prompt = (
                    f"[USER QUESTION from {user}]: \"{question_text}\"\n"
                    f"Answer this question directly and helpfully. "
                    f"If it's about the game, use what you see on screen. "
                    f"If it's about stats or players, give your best answer. "
                    f"Keep it under 3 sentences."
                )
                try:
                    async with _gemini_send_lock:
                        await asyncio.wait_for(
                            agent.llm.simple_response(text=prompt),
                            timeout=12.0,
                        )
                    _update_status(last_commentary=time.time())
                    logger.info("✅ Answered question from %s", user)
                except asyncio.TimeoutError:
                    logger.debug("Question answer timed out for: %s", question_text)
                except Exception as e:
                    err_str = str(e)
                    if "1011" in err_str or "ConnectionClosed" in type(e).__name__:
                        _backoff_until = time.time() + 20
                        logger.warning(
                            "🔴 Gemini 1011 crash in question loop — backing off 20s"
                        )
                    else:
                        logger.debug("Question answer failed: %s", e)

            if pending:
                _safe_write_json(QUESTIONS_FILE, questions)
        except asyncio.CancelledError:
            logger.info("❓ Question loop cancelled")
            return
        except Exception as e:
            logger.debug("Question check error: %s", e)

        await asyncio.sleep(3)


# ── Transcript Capture Hook ────────────────────────────────────────────
def _setup_transcript_capture(agent: Agent) -> None:
    """Hook into agent transcript events to capture what the AI says."""
    original_on_agent_transcript = None

    # Try to hook the standard transcript callback
    try:
        # Vision Agents SDK emits transcript events we can listen to
        if hasattr(agent, 'on'):
            @agent.on('agent_transcript')
            async def _on_transcript(text: str, **kw: Any) -> None:
                await _append_transcript(text, source="agent")

            # NOTE: We intentionally do NOT hook user_transcript.
            # User voice/speech is never collected, stored, or logged — privacy first.
    except Exception:
        pass

    # Also hook the LLM transcript if available
    try:
        if hasattr(agent.llm, '_on_agent_transcript'):
            original_on_agent_transcript = agent.llm._on_agent_transcript

            async def _patched_transcript(text: str, **kw: Any) -> None:
                await _append_transcript(text, source="agent")
                if original_on_agent_transcript:
                    await original_on_agent_transcript(text, **kw)

            agent.llm._on_agent_transcript = _patched_transcript
    except Exception:
        pass


# ── Call Lifecycle ──────────────────────────────────────────────────────
async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a Stream call and run until it ends."""
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
        _update_status(gemini="connected", yolo="active")
        _setup_transcript_capture(agent)

        # Send greeting (uses send lock to avoid collisions)
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

        logger.info("PlayByt is live — starting commentary loop...")

        # Start the proactive commentary loop alongside the forever-block
        commentary_task = None
        question_task = None
        if sports:
            commentary_task = asyncio.ensure_future(_commentary_loop(agent, sports))
            question_task = asyncio.ensure_future(_question_loop(agent))
            logger.info("🎙️ Commentary + question loops scheduled")
        else:
            logger.warning("No SportsProcessor found — commentary loop disabled")

        try:
            await asyncio.Future()  # Block forever (until cancellation)
        except asyncio.CancelledError:
            if commentary_task:
                commentary_task.cancel()
            if question_task:
                question_task.cancel()
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
