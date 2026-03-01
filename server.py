"""
PlayByt — Token Server
FastAPI backend for Stream user-token generation and call-ID lookup.

Run with:  uvicorn server:app --port 8000
"""

import fcntl
import json
import os
from pathlib import Path

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

STREAM_API_KEY = os.getenv("STREAM_API_KEY", "")
STREAM_API_SECRET = os.getenv("STREAM_API_SECRET", "")
CALL_ID_FILE = Path(__file__).parent / ".call_id"
HIGHLIGHTS_FILE = Path(__file__).parent / ".highlights.json"
ANALYSIS_FILE = Path(__file__).parent / ".analysis.json"
CONTROVERSIES_FILE = Path(__file__).parent / ".controversies.json"
REPORT_FILE = Path(__file__).parent / ".report.json"
TRANSCRIPT_FILE = Path(__file__).parent / ".transcript.json"
STATUS_FILE = Path(__file__).parent / ".status.json"
QUESTIONS_FILE = Path(__file__).parent / ".questions.json"

app = FastAPI(title="PlayByt API")


def _safe_read_json(path: Path, fallback=None):
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────


class TokenRequest(BaseModel):
    user_id: str
    user_name: str


class TokenResponse(BaseModel):
    token: str
    api_key: str


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/api/token", response_model=TokenResponse)
def create_token(req: TokenRequest):
    """Generate a Stream Video user token signed with the API secret."""
    if not STREAM_API_SECRET:
        raise HTTPException(status_code=500, detail="STREAM_API_SECRET not configured")

    token = jwt.encode({"user_id": req.user_id}, STREAM_API_SECRET, algorithm="HS256")
    return TokenResponse(token=token, api_key=STREAM_API_KEY)


@app.get("/api/call-id")
def get_call_id():
    """Return the active Call ID written by the agent."""
    data = _safe_read_json(CALL_ID_FILE)
    if data:
        return data
    raise HTTPException(
        status_code=404,
        detail="No active call. Start the agent first: python main.py run",
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "api_key_set": bool(STREAM_API_KEY)}


@app.get("/api/highlights")
def get_highlights():
    """Return highlights logged by the agent via tool calling."""
    data = _safe_read_json(HIGHLIGHTS_FILE, fallback=[])
    return {"highlights": data}


@app.get("/api/analysis")
def get_analysis():
    """Return the latest real-time YOLO field analysis (for tactical map)."""
    fallback = {"player_count": 0, "positions": [], "zones": {}, "formation": "N/A",
                "pressing_intensity": "none", "dominant_side": "balanced", "fatigue_flags": []}
    data = _safe_read_json(ANALYSIS_FILE, fallback=fallback)
    return data


@app.get("/api/controversies")
def get_controversies():
    """Return auto-detected controversy/threshold alerts."""
    data = _safe_read_json(CONTROVERSIES_FILE, fallback=[])
    return {"controversies": data}


@app.get("/api/report")
def get_report():
    """Return the latest exported post-match report."""
    data = _safe_read_json(REPORT_FILE)
    if data:
        return data
    raise HTTPException(
        status_code=404,
        detail="No report generated yet. Ask PlayByt to export a match report.",
    )


@app.get("/api/transcript")
def get_transcript(since_id: int = 0):
    """Return agent transcript lines, optionally filtered since a given ID."""
    data = _safe_read_json(TRANSCRIPT_FILE, fallback=[])
    if since_id > 0:
        data = [line for line in data if line.get("id", 0) > since_id]
    return {"transcript": data}


@app.get("/api/status")
def get_status():
    """Return real-time agent status (Gemini, YOLO, commentary loop)."""
    fallback = {
        "gemini": "disconnected",
        "yolo": "standby",
        "commentary_loop": "off",
        "frames_processed": 0,
        "last_commentary": 0,
    }
    data = _safe_read_json(STATUS_FILE, fallback=fallback)
    return data


def _safe_write_json(path: Path, data):
    """Write JSON file with exclusive locking."""
    with open(path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(data, f)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class AskRequest(BaseModel):
    question: str
    user: str = "Fan"


@app.post("/api/ask")
def ask_question(req: AskRequest):
    """Queue a user question for the agent to answer."""
    questions = _safe_read_json(QUESTIONS_FILE, fallback=[])
    import time as _time
    questions.append({
        "question": req.question,
        "user": req.user,
        "timestamp": _time.time(),
        "answered": False,
    })
    _safe_write_json(QUESTIONS_FILE, questions)
    return {"status": "queued", "position": len(questions)}
