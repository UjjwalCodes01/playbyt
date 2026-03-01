# PlayByt — Hackathon Submission
### Vision Possible: Agent Protocol · March 1, 2026

---

## What Is PlayByt?

**PlayByt is an AI agent that joins your video room, watches the sports broadcast you share, and tells you things no human eye could notice — in real time, out loud, using live computer vision data.**

When you watch a football match, AI spots the 3v2 overload forming before the attack happens. It detects fatigue in a player's spine angle before the substitution board goes up. It measures pressing intensity with math your brain cannot run at 60fps.

Then it speaks. Not generic commentary — specific, data-backed observations using numbers grabbed live from YOLO pose detection on every frame.

---

## Live Demo — What Is Working Right Now

Every item below is **fully implemented, stable, and demonstrable**.

### Core Features

| Feature | Status | Notes |
|---------|--------|-------|
| Agent joins Stream video call | ✅ 100% | With exponential backoff retry on ConnectTimeout |
| Screen share received and processed | ✅ 100% | Via `VideoProcessorPublisher` pipeline |
| YOLO pose detection on every frame | ✅ 100% | YOLOv11n-pose, 5 FPS, skeletons drawn live |
| HUD overlay on agent video feed | ✅ 100% | Player count, zones, formation, pressing, fatigue |
| Gemini Realtime commentary (voice in/out) | ✅ 100% | Multimodal: sees video + hears user speech |
| Proactive commentary loop | ✅ 100% | Speaks every ~15s with YOLO data prompts |
| User questions answered (text) | ✅ 100% | Via chat box → `/api/ask` → agent replies by voice |
| User questions answered (voice) | ✅ 100% | Gemini Realtime hears microphone directly |
| Role-based response styles | ✅ 100% | Analyst, Hype Fan, Stats Nerd, Coach |
| Multi-user rooms | ✅ 100% | Multiple fans in same call, each with own role |

### AI Tools (Tool Calling)

| Tool | Status | When Used |
|------|--------|-----------|
| `log_highlight` | ✅ 100% | Agent calls automatically on goals, cards, big moments |
| `get_match_summary` | ✅ 100% | "What did I miss?" / "Recap the match" |
| `get_field_analysis` | ✅ 100% | Live YOLO data: zones, formation, fatigue, pressing |
| `get_highlight_count` | ✅ 100% | "How many highlights?" |
| `get_controversy_alerts` | ✅ 100% | Auto-detected threshold events |
| `export_match_report` | ✅ 100% | Full JSON report downloadable from UI |
| `web_search` | ✅ 100% | DuckDuckGo instant answers for player stats/history |

### Sports Intelligence (SportsProcessor)

| Computed Signal | Status | Data Source |
|----------------|--------|-------------|
| Player count | ✅ 100% | YOLO hip midpoint detection |
| Zone distribution (L/C/R, Def/Mid/Att) | ✅ 100% | Normalized hip positions |
| Formation estimate (e.g. 4-3-3) | ✅ 100% | Vertical thirds distribution |
| Pressing intensity (High/Medium/Low) | ✅ 100% | Average pairwise player distance |
| Dominant side overload | ✅ 100% | L vs R zone ratio |
| Fatigue flags per player | ✅ 100% | Spine angle from shoulder/hip keypoints |
| Controversy alerts (auto-detected) | ✅ 100% | Pressing spikes, formation changes, fatigue spikes |
| Analysis trend (last 10 frames) | ✅ 100% | Float average across history ring buffer |

### Frontend Dashboard

| UI Element | Status | Details |
|-----------|--------|---------|
| Split view (user cam / agent YOLO feed) | ✅ 100% | Shows on join, no screen share required |
| Screen share raw feed in split view | ✅ 100% | Left panel switches to raw feed when sharing |
| Agent Brain status panel | ✅ 100% | Real-time Gemini/YOLO/Commentary status from `/api/status` |
| Uptime timer | ✅ 100% | MM:SS from session start |
| Tactical pitch map (SVG) | ✅ 100% | Live player dots, zone colors, overload arrows |
| Controversy alerts feed | ✅ 100% | Auto-shows when alerts exist |
| Highlights timeline | ✅ 100% | Real-time from `/api/highlights`, with timestamp |
| Export match report button | ✅ 100% | Downloads full JSON from `/api/report` |
| Live commentary feed | ✅ 100% | Agent speech, user questions, room events |
| Toast notifications | ✅ 100% | Pops up 5s for new controversy/alert events |
| Text chat input | ✅ 100% | Questions go to backend → agent answers by voice |
| Speaking indicator + waveform | ✅ 100% | Animated bars driven by interval, not re-render |
| Role badge in header | ✅ 100% | Shows role color and icon |
| Participant count | ✅ 100% | Live from SDK |
| Mute / Camera / Screen Share controls | ✅ 100% | All wired to Stream SDK |
| Leave button | ✅ 100% | Clean disconnect |

### Backend API

| Endpoint | Status | Purpose |
|---------|--------|---------|
| `POST /api/token` | ✅ 100% | JWT user token for Stream |
| `GET /api/call-id` | ✅ 100% | Auto-serve active call ID to frontend |
| `GET /api/health` | ✅ 100% | Liveness check |
| `GET /api/highlights` | ✅ 100% | Agent-logged highlights |
| `GET /api/analysis` | ✅ 100% | Latest YOLO field data |
| `GET /api/controversies` | ✅ 100% | Auto-detected alerts |
| `GET /api/report` | ✅ 100% | Full match report JSON |
| `GET /api/transcript?since_id=N` | ✅ 100% | Agent speech transcript (ring buffer) |
| `GET /api/status` | ✅ 100% | Real-time agent status (Gemini, YOLO, loop) |
| `POST /api/ask` | ✅ 100% | Queue user text questions for agent |

### Infrastructure

| Concern | Status | How it's handled |
|---------|--------|-----------------|
| File write race conditions | ✅ Fixed | `fcntl` exclusive locking on all JSON writes |
| Agent crash on Stream timeout | ✅ Fixed | Exponential backoff retry (5 attempts, 2-30s) |
| YOLO errors killing video pipeline | ✅ Fixed | 5 isolated try/catch blocks per frame step |
| SDK `asyncio.Future()` block | ✅ Fixed | Replaced broken `agent.finish()` |
| Video forwarder teardown on re-attach | ✅ Fixed | Identity check before replacing forwarder |
| Frame count never incrementing | ✅ Fixed | `self._frame_count += 1` at start of `_process_frame` |

---

## Architecture

```
Browser (User)
    │  screen share (sports broadcast)
    │  microphone (voice questions)
    ▼
Stream Edge Network (WebRTC)
    │
    ▼
Vision Agents SDK (main.py)
    │
    ├── SportsProcessor (sports_processor.py)
    │     ├── VideoForwarder  ←── incoming screen share track
    │     ├── YOLOPoseProcessor ←── 17 keypoints per player
    │     ├── Analysis Engine  ←── zones, formation, pressing, fatigue
    │     ├── HUD Overlay (cv2) ←── drawn on annotated frame
    │     └── QueuedVideoTrack ──► published back to call (agent's video feed)
    │
    └── Gemini Realtime (gemini.Realtime, fps=5)
          ├── Sees annotated video with HUD
          ├── Hears all room audio
          ├── Speaks commentary back (voice)
          └── Tool calling:
                log_highlight / get_match_summary / get_field_analysis
                get_controversy_alerts / export_match_report / web_search

FastAPI (server.py)
    ├── /api/token  ←── signs Stream JWT (HS256)
    ├── /api/*      ←── serves JSON files written by agent
    └── /api/ask    ←── queues text questions for commentary loop

React Frontend (frontend/)
    ├── JoinRoom.tsx  ←── name + role picker, auto-fetches call ID
    └── PlayBytRoom.tsx
          ├── Split video view (user + agent)
          ├── Agent Brain status (real /api/status)
          ├── Tactical pitch map (SVG)
          ├── Highlights timeline
          ├── Live commentary feed
          └── Text chat input → /api/ask
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| AI Framework | [Vision Agents SDK](https://visionagents.ai/) by Stream |
| Vision Model | YOLOv11n Pose (Ultralytics) — 17 keypoints per person |
| Language Model | Gemini 2.5 Flash Realtime (multimodal: video + audio) |
| Video Transport | Stream Edge Network (WebRTC via getstream SDK) |
| Backend | FastAPI + PyJWT + fcntl file locking |
| Frontend | React 19 + TypeScript + Vite + @stream-io/video-react-sdk |
| Python Runtime | Python 3.12, managed by `uv` |

---

## What Makes This Different

**Normal watch party AI:** Shows a chatbot in the corner that answers questions.

**PlayByt:** The AI has eyes. It runs YOLO pose detection on every frame of your game and computes spatial statistics a human brain cannot produce in real time — then speaks about them before you even notice the pattern forming.

The gap between "an AI told me something" and "an AI showed me what it's seeing while it tells me something" is visible in the dual split view: left panel is the raw broadcast, right panel is the AI's annotated version with player skeletons and the intelligence HUD drawn live.

---

## Running It (30-Second Version)

```bash
# 1. Set API keys in .env
# GEMINI_API_KEY, STREAM_API_KEY, STREAM_API_SECRET

# 2. Terminal 1 — AI Agent
uv run python main.py run

# 3. Terminal 2 — Backend
uv run uvicorn server:app --port 8000

# 4. Terminal 3 — Frontend
cd frontend && npm run dev

# 5. Open http://localhost:5173
# Pick a role → Join Room → Share Screen → Ask anything
```

Full setup guide: [SETUP.md](SETUP.md)

---

## SDK Features Exercised

- `gemini.Realtime(fps=5)` — multimodal live video + audio processing
- `@llm.register_function()` — 7 registered tools with real side effects
- `VideoProcessorPublisher` — custom subclass running YOLO + analysis + HUD
- `ultralytics.YOLOPoseProcessor` — used internally by SportsProcessor
- `getstream.Edge()` — WebRTC multi-user room
- `Agent(instructions="Read @instructions.md")` — file-based system prompt
- `agent.llm.simple_response(text=...)` — programmatic agent nudges (commentary loop)
- `agent_idle_timeout=0` — keep agent alive indefinitely
- `--no-demo` CLI flag — suppress SDK demo browser

---

## File Map

```
PlayByt/
├── main.py              # Agent entry: tools, commentary loop, transcript capture
├── sports_processor.py  # VideoProcessorPublisher: YOLO → analysis → HUD
├── server.py            # FastAPI: 10 endpoints, fcntl locking, /api/ask
├── instructions.md      # Agent identity, role styles, tool usage rules
├── pyproject.toml       # Python deps (uv)
├── SETUP.md             # Full setup walkthrough
├── IDEA.md              # Concept explanation
├── ROADMAP.md           # Feature planning
└── frontend/
    ├── src/
    │   ├── App.tsx                  # Root, RoomConfig type, role routing
    │   └── components/
    │       ├── JoinRoom.tsx         # Landing: name, role, auto call-id
    │       └── PlayBytRoom.tsx      # Full dashboard (~1250 lines)
    ├── package.json
    └── vite.config.ts
```

---

*Submitted March 1, 2026 — Vision Possible: Agent Protocol Hackathon*
