import { useEffect, useState, useRef } from 'react'
import {
  StreamVideo,
  StreamCall,
  StreamVideoClient,
  ParticipantView,
  ParticipantsAudio,
  useCall,
  useCallStateHooks,
  hasScreenShare,
  CallingState,
} from '@stream-io/video-react-sdk'
import type { RoomConfig, FanRole } from '../App'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const ROLE_META: Record<FanRole, { icon: string; label: string; color: string }> = {
  analyst: { icon: '🧠', label: 'Analyst', color: '#4488ff' },
  hype: { icon: '🔥', label: 'Hype Fan', color: '#ff4444' },
  stats: { icon: '📊', label: 'Stats', color: '#ffaa00' },
  coach: { icon: '📋', label: 'Coach', color: '#00ff88' },
}

interface Highlight {
  id: number
  description: string
  category: string
  elapsed: number
}

interface PlayBytRoomProps {
  config: RoomConfig
  onLeave: () => void
}

interface CommentaryLine {
  id: number
  text: string
  time: string
  type: 'playbyt' | 'user' | 'event'
  user?: string
}

interface AnalysisData {
  player_count: number
  positions: Array<{ x: number; y: number; id: number }>
  zones: { left: number; center: number; right: number; def_third: number; mid_third: number; att_third: number }
  formation: string
  pressing_intensity: string
  dominant_side: string
  fatigue_flags: Array<{ player_id: number; spine_angle: number; severity: string }>
}

interface ControversyEvent {
  id: number
  type: string
  title: string
  description: string
  elapsed: number
  timestamp: number
}

interface ToastItem {
  id: number
  title: string
  description: string
  type: string
}

interface AgentStatus {
  gemini: string
  yolo: string
  commentary_loop: string
  frames_processed: number
  last_commentary: number
}

interface TranscriptLine {
  id: number
  text: string
  source: string
  timestamp: number
  elapsed: number
}

let commentaryIdCounter = 0

// ─── Tactical Pitch Map ─────────────────────────────────────────────────────

function TacticalMap({ analysis }: { analysis: AnalysisData | null }) {
  const W = 308
  const H = 160
  const pad = 10

  const pressColor: Record<string, string> = {
    high: 'rgba(0,255,136,0.18)',
    medium: 'rgba(255,170,0,0.12)',
    low: 'rgba(68,136,255,0.08)',
    none: 'rgba(30,30,40,0.6)',
  }
  const bg = analysis ? pressColor[analysis.pressing_intensity] ?? pressColor.none : pressColor.none

  const pitchW = W - pad * 2
  const pitchH = H - pad * 2

  return (
    <svg width={W} height={H} style={{ display: 'block', borderRadius: '8px', overflow: 'hidden' }}>
      {/* Pitch background */}
      <rect x={pad} y={pad} width={pitchW} height={pitchH} fill={bg} rx={4} />
      <rect x={pad} y={pad} width={pitchW} height={pitchH} fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth={1} rx={4} />

      {/* Center line */}
      <line x1={W / 2} y1={pad} x2={W / 2} y2={H - pad} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
      {/* Center circle */}
      <circle cx={W / 2} cy={H / 2} r={20} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />

      {/* Penalty boxes */}
      <rect x={pad} y={pad + pitchH * 0.25} width={pitchW * 0.14} height={pitchH * 0.5} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
      <rect x={W - pad - pitchW * 0.14} y={pad + pitchH * 0.25} width={pitchW * 0.14} height={pitchH * 0.5} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />

      {/* Third zone dividers */}
      <line x1={pad + pitchW / 3} y1={pad} x2={pad + pitchW / 3} y2={H - pad} stroke="rgba(255,255,255,0.05)" strokeWidth={1} strokeDasharray="3,3" />
      <line x1={pad + (pitchW * 2) / 3} y1={pad} x2={pad + (pitchW * 2) / 3} y2={H - pad} stroke="rgba(255,255,255,0.05)" strokeWidth={1} strokeDasharray="3,3" />

      {/* Player dots */}
      {analysis?.positions.map((pos, i) => {
        const cx = pad + pos.x * pitchW
        const cy = pad + pos.y * pitchH
        const isFatigued = analysis.fatigue_flags.some(f => f.player_id === pos.id)
        return (
          <g key={i}>
            <circle cx={cx} cy={cy} r={6} fill={isFatigued ? 'rgba(255,80,80,0.9)' : 'rgba(0,200,255,0.85)'} stroke="rgba(0,0,0,0.5)" strokeWidth={1} />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize={7} fill="white" fontWeight="bold">{pos.id + 1}</text>
          </g>
        )
      })}

      {/* No players message */}
      {(!analysis || analysis.player_count === 0) && (
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={10} fill="rgba(255,255,255,0.3)">
          No players detected
        </text>
      )}

      {/* Side overload arrow */}
      {analysis?.dominant_side === 'left' && (
        <text x={pad + 4} y={H - pad - 4} fontSize={9} fill="rgba(255,170,0,0.8)" fontWeight="bold">◀ OVERLOAD</text>
      )}
      {analysis?.dominant_side === 'right' && (
        <text x={W - pad - 4} y={H - pad - 4} textAnchor="end" fontSize={9} fill="rgba(255,170,0,0.8)" fontWeight="bold">OVERLOAD ▶</text>
      )}
    </svg>
  )
}

// ─── Main Room Component ─────────────────────────────────────────────────────

export function PlayBytRoom({ config, onLeave }: PlayBytRoomProps) {
  const [client, setClient] = useState<StreamVideoClient | null>(null)
  const [call, setCall] = useState<ReturnType<StreamVideoClient['call']> | null>(null)
  const [commentary, setCommentary] = useState<CommentaryLine[]>([
    {
      id: ++commentaryIdCounter,
      text: 'PlayByt is watching the game... Ask anything!',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      type: 'event',
    },
  ])

  const apiKey = config.apiKey || import.meta.env.VITE_STREAM_API_KEY

  useEffect(() => {
    const streamClient = new StreamVideoClient({
      apiKey,
      user: { id: config.userId, name: config.userName },
      token: config.userToken,
    })

    const streamCall = streamClient.call('default', config.callId)

    // Properly await the join — using create:true is safe (joins if exists, creates if not)
    streamCall.join({ create: true }).catch(err => {
      console.error('[PlayByt] Failed to join call:', err)
    })

    setClient(streamClient)
    setCall(streamCall)

    return () => {
      streamCall.leave().catch(() => {})
      streamClient.disconnectUser().catch(() => {})
    }
  }, [config, apiKey])

  function addCommentary(line: Omit<CommentaryLine, 'id' | 'time'>) {
    setCommentary(prev => [
      ...prev,
      {
        ...line,
        id: ++commentaryIdCounter,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ])
  }

  if (!client || !call) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-primary)',
        flexDirection: 'column',
        gap: '16px',
      }}>
        <div style={{
          width: '48px',
          height: '48px',
          border: '3px solid rgba(0,255,136,0.2)',
          borderTop: '3px solid #00ff88',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        <p style={{ color: 'var(--text-secondary)' }}>Connecting to game room...</p>
      </div>
    )
  }

  return (
    <StreamVideo client={client}>
      <StreamCall call={call}>
        <RoomLayout
          config={config}
          commentary={commentary}
          addCommentary={addCommentary}
          onLeave={onLeave}
        />
      </StreamCall>
    </StreamVideo>
  )
}

// ─── Inner layout (has access to call hooks) ─────────────────────────────────

interface RoomLayoutProps {
  config: RoomConfig
  commentary: CommentaryLine[]
  addCommentary: (line: Omit<CommentaryLine, 'id' | 'time'>) => void
  onLeave: () => void
}

function formatUptime(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0')
  const s = (seconds % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

function RoomLayout({ config, commentary, addCommentary, onLeave }: RoomLayoutProps) {
  const call = useCall()
  const { useParticipants, useCallCallingState } = useCallStateHooks()
  const participants = useParticipants()
  const callingState = useCallCallingState()
  const commentaryRef = useRef<HTMLDivElement>(null)
  const [isMuted, setIsMuted] = useState(false)
  const [isCamOff, setIsCamOff] = useState(false)
  const [isScreenSharing, setIsScreenSharing] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [agentSpeaking, setAgentSpeaking] = useState(false)
  const [highlights, setHighlights] = useState<Highlight[]>([])
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null)
  const [controversies, setControversies] = useState<ControversyEvent[]>([])
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null)
  const [chatInput, setChatInput] = useState('')
  const seenControversyCount = useRef(0)
  const toastIdCounter = useRef(0)
  const addRef = useRef(addCommentary)
  addRef.current = addCommentary
  const prevParticipantCount = useRef(participants.length)
  const lastKnownAgentParticipant = useRef<typeof participants[0] | null>(null)
  const lastTranscriptId = useRef(0)
  // Stable waveform heights — updated via interval, not in render
  const [waveHeights, setWaveHeights] = useState([8, 12, 16, 10, 14])

  // Uptime timer
  useEffect(() => {
    const t = setInterval(() => setElapsed(s => s + 1), 1000)
    return () => clearInterval(t)
  }, [])

  // Poll highlights from backend
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/highlights`)
        if (res.ok) {
          const data = await res.json()
          setHighlights(data.highlights || [])
        }
      } catch { /* ignore */ }
    }, 2000)
    return () => clearInterval(poll)
  }, [])

  // Poll field analysis for tactical map
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/analysis`)
        if (res.ok) {
          const data: AnalysisData = await res.json()
          if (data.player_count > 0) setAnalysis(data)
        }
      } catch { /* ignore */ }
    }, 2000)
    return () => clearInterval(poll)
  }, [])

  // Poll controversies and fire toasts for new ones
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/controversies`)
        if (res.ok) {
          const data = await res.json()
          const events: ControversyEvent[] = data.controversies || []
          setControversies(events)
          const newOnes = events.slice(seenControversyCount.current)
          if (newOnes.length > 0) {
            seenControversyCount.current = events.length
            newOnes.forEach(ev => {
              const toastId = ++toastIdCounter.current
              setToasts(prev => [...prev, { id: toastId, title: ev.title, description: ev.description, type: ev.type }])
              setTimeout(() => setToasts(prev => prev.filter(t => t.id !== toastId)), 5000)
            })
          }
        }
      } catch { /* ignore */ }
    }, 2000)
    return () => clearInterval(poll)
  }, [])

  // Poll agent transcript — feed AI speech into commentary
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/transcript?since_id=${lastTranscriptId.current}`)
        if (res.ok) {
          const data = await res.json()
          const lines: TranscriptLine[] = data.transcript || []
          lines.forEach(line => {
            if (line.id > lastTranscriptId.current) {
              lastTranscriptId.current = line.id
              addRef.current({
                type: line.source === 'agent' ? 'playbyt' : 'user',
                text: line.text,
                user: line.source === 'user' ? 'You' : undefined,
              })
            }
          })
        }
      } catch { /* ignore */ }
    }, 2000)
    return () => clearInterval(poll)
  }, [])

  // Poll real agent status (Gemini, YOLO, commentary loop)
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`)
        if (res.ok) {
          const data: AgentStatus = await res.json()
          setAgentStatus(data)
        }
      } catch { /* ignore */ }
    }, 3000)
    return () => clearInterval(poll)
  }, [])

  // Waveform animation — update heights in an interval, NOT in render
  useEffect(() => {
    if (!agentSpeaking) return
    const t = setInterval(() => {
      setWaveHeights([0, 1, 2, 3, 4].map(() => 6 + Math.random() * 14))
    }, 150)
    return () => clearInterval(t)
  }, [agentSpeaking])

  // Track PlayByt speaking state
  const agentParticipantEarly = participants.find(p =>
    p.userId === 'playbyt-agent' ||
    p.userId?.startsWith('playbyt') ||
    p.name?.toLowerCase() === 'playbyt'
  )
  const speaking = agentParticipantEarly?.isSpeaking ?? false
  const prevSpeaking = useRef(false)
  useEffect(() => {
    setAgentSpeaking(speaking)
    prevSpeaking.current = speaking
  }, [speaking])

  // Track participant count changes
  useEffect(() => {
    const prev = prevParticipantCount.current
    const curr = participants.length
    if (curr > prev && prev > 0) {
      addRef.current({ type: 'event', text: `👥 A new participant joined (${curr} in room)` })
    } else if (curr < prev && prev > 0) {
      addRef.current({ type: 'event', text: `👥 A participant left (${curr} in room)` })
    }
    prevParticipantCount.current = curr
  }, [participants.length])

  // Listen for call events → populate commentary feed
  useEffect(() => {
    if (!call) return
    const unsubs = [
      call.on('call.session_participant_joined', (evt: any) => {
        const name = evt?.participant?.user?.name || evt?.participant?.user_id || 'Someone'
        if (name !== 'PlayByt') {
          addRef.current({ type: 'event', text: `${name} joined the room` })
        } else {
          addRef.current({ type: 'event', text: '🤖 PlayByt AI connected!' })
        }
      }),
      call.on('call.session_participant_left', (evt: any) => {
        const name = evt?.participant?.user?.name || evt?.participant?.user_id || 'Someone'
        addRef.current({ type: 'event', text: `${name} left the room` })
      }),
    ]
    return () => unsubs.forEach(u => { if (typeof u === 'function') (u as () => void)(); else if (u && typeof (u as any).unsubscribe === 'function') (u as any).unsubscribe() })
  }, [call])

  // Auto-scroll commentary
  useEffect(() => {
    if (commentaryRef.current) {
      commentaryRef.current.scrollTop = commentaryRef.current.scrollHeight
    }
  }, [commentary])

  async function handleLeave() {
    await call?.leave()
    onLeave()
  }

  async function toggleMic() {
    if (isMuted) {
      await call?.microphone.enable()
    } else {
      await call?.microphone.disable()
    }
    setIsMuted(!isMuted)
  }

  async function toggleCam() {
    if (isCamOff) {
      await call?.camera.enable()
    } else {
      await call?.camera.disable()
    }
    setIsCamOff(!isCamOff)
  }

  async function toggleScreenShare() {
    if (isScreenSharing) {
      await call?.screenShare.disable()
    } else {
      await call?.screenShare.enable()
    }
    setIsScreenSharing(!isScreenSharing)
  }

  async function exportReport() {
    try {
      const res = await fetch(`${API_BASE}/api/report`)
      if (res.ok) {
        const data = await res.json()
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `playbyt-report-${new Date().toISOString().slice(0, 10)}.json`
        a.click()
        URL.revokeObjectURL(url)
      } else {
        alert('No report yet. Ask PlayByt: "Export a match report"')
      }
    } catch {
      alert('Report not available. Ask PlayByt to export one first.')
    }
  }

  // The PlayByt agent participant — match on id OR name (SDK may suffix session id)
  const agentParticipant = participants.find(p =>
    p.userId === 'playbyt-agent' ||
    p.userId?.startsWith('playbyt') ||
    p.name?.toLowerCase() === 'playbyt'
  )
  // Keep last known agent participant so split doesn't collapse on brief network blips
  if (agentParticipant) lastKnownAgentParticipant.current = agentParticipant
  const effectiveAgentParticipant = agentParticipant ?? lastKnownAgentParticipant.current
  // Screen-sharing participant (could be anyone)
  const screenShareParticipant = participants.find(p => hasScreenShare(p))
  // Other human participants (exclude agent, screen share, and local user since they're in the split view)
  const humanParticipants = participants.filter(p =>
    !p.userId?.startsWith('playbyt') &&
    p.name?.toLowerCase() !== 'playbyt' &&
    !hasScreenShare(p) &&
    !p.isLocalParticipant
  )
  // Local user participant (for split view when no screen share)
  const localParticipant = participants.find(p => p.isLocalParticipant)

  // Wait until fully joined — useParticipants() returns empty while JOINING
  if (callingState !== CallingState.JOINED && callingState !== CallingState.RECONNECTING) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-primary)', flexDirection: 'column', gap: '16px' }}>
        <div style={{ width: '48px', height: '48px', border: '3px solid rgba(0,255,136,0.2)', borderTop: '3px solid #00ff88', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        <p style={{ color: 'var(--text-secondary)' }}>Joining call...</p>
      </div>
    )
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--bg-primary)',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* SDK-managed audio — uses call.bindAudioElement() for correct playback */}
      <ParticipantsAudio participants={participants} />
      {/* Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '14px 24px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{
            fontSize: '22px',
            fontWeight: '900',
            background: 'linear-gradient(135deg, #00ff88, #4488ff)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>PlayByt</span>
          <span style={{
            background: 'rgba(255,0,0,0.15)',
            border: '1px solid rgba(255,0,0,0.3)',
            color: '#ff4444',
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: '700',
            letterSpacing: '0.1em',
            animation: 'pulse 2s infinite',
          }}>● LIVE</span>
          <style>{`@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }`}</style>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            background: 'rgba(0,255,136,0.08)',
            border: '1px solid rgba(0,255,136,0.15)',
            borderRadius: '20px',
            padding: '4px 12px',
            fontSize: '13px',
            color: '#00ff88',
          }}>
            👥 {participants.length} in room
          </div>
          <div style={{
            background: `${ROLE_META[config.role].color}15`,
            border: `1px solid ${ROLE_META[config.role].color}33`,
            borderRadius: '20px',
            padding: '4px 12px',
            fontSize: '13px',
            color: ROLE_META[config.role].color,
            fontWeight: '600',
          }}>
            {ROLE_META[config.role].icon} {ROLE_META[config.role].label}
          </div>
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            padding: '6px 12px',
            fontSize: '13px',
            color: 'var(--text-secondary)',
          }}>
            🎙️ {config.userName}
          </div>
        </div>
      </header>

      {/* Main content */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1fr 340px',
        gridTemplateRows: '1fr',
        gap: '16px',
        padding: '16px',
        overflow: 'hidden',
      }}>
        {/* LEFT: Video grid */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden', minHeight: 0 }}>

          {/* PlayByt agent video (main screen = YOLO annotated video) */}
          {effectiveAgentParticipant ? (
            /* ── SPLIT VIEW — always show when agent is in the call ── */
            <div style={{
              flex: 1,
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '8px',
              minHeight: 0,
            }}>
              {/* Left panel: Screen share (if active) or user's camera */}
              <div style={{
                background: 'var(--bg-card)',
                borderRadius: '12px',
                border: `1px solid ${screenShareParticipant ? 'rgba(68,136,255,0.3)' : 'rgba(153,102,255,0.25)'}`,
                overflow: 'hidden',
                position: 'relative',
              }}>
                {screenShareParticipant ? (
                  <>
                    <div style={{ width: '100%', height: '100%' }}>
                      <ParticipantView participant={screenShareParticipant} />
                    </div>
                    <div style={{
                      position: 'absolute', top: '8px', left: '8px',
                      background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)',
                      border: '1px solid rgba(68,136,255,0.4)', borderRadius: '6px',
                      padding: '4px 10px', fontSize: '11px', color: '#4488ff', fontWeight: '700',
                    }}>
                      📺 RAW FEED
                    </div>
                  </>
                ) : localParticipant ? (
                  <>
                    <div style={{ width: '100%', height: '100%' }}>
                      <ParticipantView participant={localParticipant} />
                    </div>
                    <div style={{
                      position: 'absolute', top: '8px', left: '8px',
                      background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)',
                      border: '1px solid rgba(153,102,255,0.4)', borderRadius: '6px',
                      padding: '4px 10px', fontSize: '11px', color: '#9966ff', fontWeight: '700',
                    }}>
                      📹 YOUR CAM
                    </div>
                  </>
                ) : (
                  <div style={{
                    height: '100%', display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center', gap: '8px',
                  }}>
                    <span style={{ fontSize: '28px' }}>📺</span>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '12px', textAlign: 'center', padding: '0 16px' }}>
                      Share your screen to show the game feed
                    </p>
                  </div>
                )}
              </div>
              {/* Right panel: Agent's YOLO annotated feed */}
              <div style={{
                background: 'var(--bg-card)',
                borderRadius: '12px',
                border: '1px solid rgba(0,255,136,0.3)',
                overflow: 'hidden',
                position: 'relative',
              }}>
                <div style={{ width: '100%', height: '100%' }}>
                  <ParticipantView participant={effectiveAgentParticipant!} />
                </div>
                <div style={{
                  position: 'absolute', top: '8px', left: '8px',
                  background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)',
                  border: '1px solid rgba(0,255,136,0.4)', borderRadius: '6px',
                  padding: '4px 10px', fontSize: '11px', color: '#00ff88', fontWeight: '700',
                  display: 'flex', alignItems: 'center', gap: '4px',
                }}>
                  <span style={{ animation: 'pulse 2s infinite', display: 'inline-block' }}>●</span>
                  PlayByt
                </div>
              </div>
            </div>
          ) : (
            /* ── SINGLE VIEW — waiting for agent to join ── */
            <div style={{
              flex: 1,
              background: 'var(--bg-card)',
              borderRadius: '16px',
              border: '1px solid rgba(0,255,136,0.15)',
              overflow: 'hidden',
              position: 'relative',
              display: 'flex',
              minHeight: 0,
            }}>
              {screenShareParticipant ? (
                <>
                  <div style={{ width: '100%', height: '100%' }}>
                    <ParticipantView participant={screenShareParticipant} />
                  </div>
                  <div style={{
                    position: 'absolute', top: '12px', left: '12px',
                    background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
                    border: '1px solid rgba(68,136,255,0.3)', borderRadius: '8px',
                    padding: '6px 12px', fontSize: '12px', color: '#4488ff', fontWeight: '700',
                  }}>
                    📺 Screen Share · Waiting for AI
                  </div>
                </>
              ) : (
                <div style={{
                  width: '100%', display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', gap: '12px',
                }}>
                  <div style={{
                    width: '64px', height: '64px',
                    background: 'rgba(0,255,136,0.08)', borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '28px',
                  }}>🤖</div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                    Waiting for PlayByt agent to join...
                  </p>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
                    Run: <code style={{ color: '#00ff88', background: 'rgba(0,255,136,0.1)', padding: '2px 6px', borderRadius: '4px' }}>
                      python main.py run
                    </code>
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Human participants grid */}
          {humanParticipants.length > 0 && (
            <div style={{
              flexShrink: 0,
              display: 'grid',
              gridTemplateColumns: `repeat(${Math.min(humanParticipants.length, 3)}, 1fr)`,
              gap: '8px',
              height: '120px',
            }}>
              {humanParticipants.map(p => (
                <div key={p.sessionId} style={{
                  background: 'var(--bg-card)',
                  borderRadius: '12px',
                  border: `1px solid ${p.isSpeaking ? '#00ff88' : 'var(--border)'}`,
                  boxShadow: p.isSpeaking ? '0 0 12px rgba(0,255,136,0.3)' : 'none',
                  overflow: 'hidden',
                  position: 'relative',
                  transition: 'border-color 0.3s, box-shadow 0.3s',
                }}>
                  <div style={{ width: '100%', height: '100%' }}><ParticipantView participant={p} /></div>
                  <div style={{
                    position: 'absolute',
                    bottom: '6px',
                    left: '6px',
                    background: 'rgba(0,0,0,0.7)',
                    borderRadius: '4px',
                    padding: '2px 6px',
                    fontSize: '11px',
                    color: p.isSpeaking ? '#00ff88' : 'var(--text-primary)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}>
                    {p.isSpeaking && <span style={{ fontSize: '8px', animation: 'pulse 1s infinite' }}>●</span>}
                    {p.name || p.userId.split('-')[1] || 'Fan'}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Controls */}
          <div style={{
            display: 'flex',
            gap: '8px',
            justifyContent: 'center',
            padding: '4px',
            flexShrink: 0,
            flexWrap: 'wrap',
          }}>
            {[
              { icon: isMuted ? '🔇' : '🎤', label: isMuted ? 'Unmute' : 'Mute', action: toggleMic, active: !isMuted, color: '#00ff88' },
              { icon: isCamOff ? '📵' : '📹', label: isCamOff ? 'Start Cam' : 'Stop Cam', action: toggleCam, active: !isCamOff, color: '#4488ff' },
              { icon: isScreenSharing ? '🛑' : '📺', label: isScreenSharing ? 'Stop Share' : 'Share Screen', action: toggleScreenShare, active: isScreenSharing, color: '#9966ff' },
            ].map(btn => (
              <button key={btn.label} onClick={btn.action} style={{
                background: btn.active ? `rgba(${btn.color === '#00ff88' ? '0,255,136' : btn.color === '#4488ff' ? '68,136,255' : '153,102,255'},0.1)` : 'var(--bg-card)',
                border: `1px solid ${btn.active ? btn.color + '33' : 'var(--border)'}`,
                borderRadius: '10px',
                padding: '10px 18px',
                color: btn.active ? btn.color : 'var(--text-secondary)',
                fontSize: '14px',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.2s',
              }}>
                {btn.icon} {btn.label}
              </button>
            ))}
            <button onClick={handleLeave} style={{
              background: 'rgba(255,68,68,0.1)',
              border: '1px solid rgba(255,68,68,0.3)',
              borderRadius: '10px',
              padding: '10px 18px',
              color: '#ff4444',
              fontSize: '14px',
              fontWeight: '600',
              marginLeft: 'auto',
            }}>
              🚪 Leave
            </button>
          </div>
        </div>

        {/* RIGHT: Agent Brain + Commentary */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' }}>

          {/* ── Agent Brain Stats ── */}
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '16px',
            padding: '14px 16px',
            flexShrink: 0,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <span style={{ fontSize: '14px' }}>🧠</span>
              <span style={{ fontWeight: '700', fontSize: '13px', color: 'var(--text-primary)' }}>Agent Brain</span>
              <span style={{
                marginLeft: 'auto',
                fontFamily: 'monospace',
                fontSize: '12px',
                color: 'var(--text-secondary)',
              }}>{formatUptime(elapsed)}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
              {[{
                  label: 'Gemini',
                  value: agentStatus?.gemini === 'connected' ? '● Connected' : agentParticipantEarly ? '● In Call' : '○ Waiting',
                  color: agentStatus?.gemini === 'connected' ? '#00ff88' : agentParticipantEarly ? '#ffaa00' : '#888',
                }, {
                  label: 'Commentary',
                  value: agentStatus?.commentary_loop === 'active' ? '● Live' : agentStatus?.commentary_loop === 'starting' ? '◐ Starting' : '○ Off',
                  color: agentStatus?.commentary_loop === 'active' ? '#00ff88' : agentStatus?.commentary_loop === 'starting' ? '#ffaa00' : '#888',
                }, {
                  label: 'YOLO + HUD',
                  value: agentStatus?.yolo === 'active' ? '● Tracking' : agentParticipantEarly ? '● Ready' : '○ Standby',
                  color: agentStatus?.yolo === 'active' ? '#00ff88' : agentParticipantEarly ? '#ffaa00' : '#888',
                }, {
                  label: 'Users',
                  value: `${participants.length} in call`,
                  color: '#9966ff',
                },
              ].map(s => (
                <div key={s.label} style={{
                  background: 'var(--bg-secondary)',
                  borderRadius: '8px',
                  padding: '8px 10px',
                }}>
                  <div style={{ fontSize: '10px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '2px' }}>{s.label}</div>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: s.color }}>{s.value}</div>
                </div>
              ))}
            </div>

            {/* Screen share status */}
            <div style={{
              marginTop: '8px',
              background: screenShareParticipant ? 'rgba(68,136,255,0.08)' : 'rgba(255,107,53,0.08)',
              border: `1px solid ${screenShareParticipant ? 'rgba(68,136,255,0.2)' : 'rgba(255,107,53,0.2)'}`,
              borderRadius: '8px',
              padding: '6px 10px',
              fontSize: '11px',
              color: screenShareParticipant ? '#4488ff' : '#ff6b35',
              fontWeight: '600',
              textAlign: 'center',
            }}>
              {screenShareParticipant ? '📺 Screen share active — AI is watching' : '⚠️ No screen share — click Share Screen to start'}
            </div>
          </div>

          {/* ── Speaking Indicator ── */}
          <div style={{
            background: agentSpeaking ? 'rgba(0,255,136,0.06)' : 'var(--bg-card)',
            border: `1px solid ${agentSpeaking ? 'rgba(0,255,136,0.3)' : 'var(--border)'}`,
            borderRadius: '12px',
            padding: '10px 14px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            flexShrink: 0,
            transition: 'all 0.3s',
          }}>
            {/* Audio waveform bars */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '2px', height: '20px' }}>
              {[0, 1, 2, 3, 4].map(i => (
                <div key={i} style={{
                  width: '3px',
                  borderRadius: '2px',
                  background: agentSpeaking ? '#00ff88' : '#333',
                  height: agentSpeaking ? `${waveHeights[i]}px` : '4px',
                  transition: 'height 0.15s, background 0.3s',
                }} />
              ))}
            </div>
            <div>
              <div style={{ fontSize: '12px', fontWeight: '700', color: agentSpeaking ? '#00ff88' : 'var(--text-secondary)' }}>
                {agentSpeaking ? 'PlayByt is commentating...' : 'PlayByt listening'}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                {agentSpeaking ? 'AI is analyzing the action live' : 'Speak or share screen to activate'}
              </div>
            </div>
          </div>

          {/* ── Tactical Map ── */}
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '12px',
            padding: '10px 14px',
            flexShrink: 0,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
              <span style={{ fontSize: '13px' }}>🗺</span>
              <span style={{ fontWeight: '700', fontSize: '12px', color: 'var(--text-primary)' }}>Tactical Map</span>
              {analysis && (
                <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#4488ff', fontWeight: '700' }}>
                  {analysis.formation} · {analysis.pressing_intensity.toUpperCase()}
                </span>
              )}
            </div>
            <TacticalMap analysis={analysis} />
          </div>

          {/* ── Controversy Alerts ── */}
          {controversies.length > 0 && (
            <div style={{
              background: 'var(--bg-card)',
              border: '1px solid rgba(255,170,0,0.2)',
              borderRadius: '12px',
              padding: '10px 14px',
              flexShrink: 0,
              maxHeight: '140px',
              display: 'flex',
              flexDirection: 'column',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                <span style={{ fontSize: '13px' }}>🚨</span>
                <span style={{ fontWeight: '700', fontSize: '12px', color: 'var(--text-primary)' }}>Alerts</span>
                <span style={{ marginLeft: 'auto', fontSize: '10px', color: '#ffaa00', fontWeight: '700' }}>
                  {controversies.length} detected
                </span>
              </div>
              <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {controversies.slice(-4).map(c => (
                  <div key={c.id} style={{
                    background: 'rgba(255,170,0,0.05)',
                    border: '1px solid rgba(255,170,0,0.12)',
                    borderRadius: '6px',
                    padding: '5px 8px',
                    display: 'flex',
                    gap: '8px',
                    alignItems: 'flex-start',
                  }}>
                    <span style={{ fontFamily: 'monospace', fontSize: '9px', color: '#ffaa00', fontWeight: '700', flexShrink: 0, marginTop: '2px' }}>
                      {Math.floor(c.elapsed / 60).toString().padStart(2, '0')}:{(c.elapsed % 60).toString().padStart(2, '0')}
                    </span>
                    <div>
                      <div style={{ fontSize: '10px', fontWeight: '700', color: 'var(--text-primary)' }}>{c.title}</div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)', lineHeight: '1.3' }}>{c.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Highlights Timeline ── */}
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '12px',
            padding: '10px 14px',
            flexShrink: 0,
            maxHeight: '180px',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
              <span style={{ fontSize: '13px' }}>⚡</span>
              <span style={{ fontWeight: '700', fontSize: '12px', color: 'var(--text-primary)' }}>Highlights</span>
              <span style={{
                marginLeft: 'auto',
                fontSize: '10px',
                color: '#ffaa00',
                fontWeight: '700',
              }}>{highlights.length} logged</span>
              <button
                onClick={exportReport}
                title="Export post-match report as JSON"
                style={{
                  background: 'rgba(0,255,136,0.08)',
                  border: '1px solid rgba(0,255,136,0.2)',
                  borderRadius: '6px',
                  padding: '2px 8px',
                  color: '#00ff88',
                  fontSize: '10px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  flexShrink: 0,
                }}>
                📄 Export
              </button>
            </div>
            {highlights.length === 0 ? (
              <p style={{ fontSize: '11px', color: 'var(--text-secondary)', textAlign: 'center', padding: '8px 0' }}>
                No highlights yet — AI logs key moments automatically
              </p>
            ) : (
              <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {highlights.slice(-6).map(h => (
                  <div key={h.id} style={{
                    background: 'rgba(255,170,0,0.05)',
                    border: '1px solid rgba(255,170,0,0.12)',
                    borderRadius: '6px',
                    padding: '6px 8px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '8px',
                  }}>
                    <span style={{
                      fontFamily: 'monospace',
                      fontSize: '10px',
                      color: '#ffaa00',
                      fontWeight: '700',
                      flexShrink: 0,
                      marginTop: '1px',
                    }}>
                      {Math.floor(h.elapsed / 60).toString().padStart(2, '0')}:{(h.elapsed % 60).toString().padStart(2, '0')}
                    </span>
                    <span style={{ fontSize: '11px', color: 'var(--text-primary)', lineHeight: '1.3' }}>
                      {h.description}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Live Commentary Feed ── */}
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '16px',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            flex: 1,
            minHeight: 0,
          }}>
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              flexShrink: 0,
            }}>
              <span style={{ fontSize: '14px' }}>📡</span>
              <span style={{ fontWeight: '700', fontSize: '13px' }}>Live Feed</span>
              <span style={{
                marginLeft: 'auto',
                background: 'rgba(255,0,0,0.1)',
                color: '#ff4444',
                borderRadius: '4px',
                padding: '2px 6px',
                fontSize: '10px',
                fontWeight: '700',
                animation: 'pulse 2s infinite',
              }}>● LIVE</span>
            </div>

            <div ref={commentaryRef} style={{
              flex: 1,
              overflowY: 'auto',
              padding: '10px',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
            }}>
              <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }`}</style>
              {commentary.map(line => (
                <div key={line.id} style={{
                  background: line.type === 'playbyt'
                    ? 'rgba(0,255,136,0.05)'
                    : line.type === 'user'
                    ? 'rgba(68,136,255,0.05)'
                    : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${
                    line.type === 'playbyt'
                      ? 'rgba(0,255,136,0.12)'
                      : line.type === 'user'
                      ? 'rgba(68,136,255,0.12)'
                      : 'rgba(255,255,255,0.04)'
                  }`,
                  borderRadius: '8px',
                  padding: '8px 10px',
                  animation: 'fadeIn 0.3s ease',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                    <span style={{
                      fontSize: '10px',
                      fontWeight: '700',
                      color: line.type === 'playbyt' ? '#00ff88' : line.type === 'user' ? '#4488ff' : 'var(--text-secondary)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}>
                      {line.type === 'playbyt' ? '🤖 PlayByt' : line.type === 'user' ? `👤 ${line.user}` : '📡 Event'}
                    </span>
                    <span style={{ fontSize: '9px', color: 'var(--text-secondary)' }}>{line.time}</span>
                  </div>
                  <p style={{ fontSize: '12px', color: 'var(--text-primary)', lineHeight: '1.4' }}>
                    {line.text}
                  </p>
                </div>
              ))}
            </div>

            <div style={{
              padding: '10px',
              borderTop: '1px solid var(--border)',
              flexShrink: 0,
            }}>
              <form
                onSubmit={async (e) => {
                  e.preventDefault()
                  const msg = chatInput.trim()
                  if (!msg) return
                  setChatInput('')
                  // Send question to backend — it will appear in the Live Feed
                  // via transcript polling (no local addCommentary to avoid duplicates)
                  try {
                    await fetch(`${API_BASE}/api/ask`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ question: msg, user: config.userName }),
                    })
                  } catch {
                    // If backend unreachable, show locally as fallback
                    addRef.current({ type: 'user', text: msg, user: config.userName })
                  }
                }}
                style={{ display: 'flex', gap: '6px' }}
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask PlayByt anything..."
                  style={{
                    flex: 1,
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '12px',
                    color: 'var(--text-primary)',
                    outline: 'none',
                  }}
                />
                <button
                  type="submit"
                  style={{
                    background: 'rgba(0,255,136,0.1)',
                    border: '1px solid rgba(0,255,136,0.3)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    color: '#00ff88',
                    fontSize: '12px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    flexShrink: 0,
                  }}
                >
                  Send
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* ── Toast Notifications ── */}
      <div style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        zIndex: 1000,
        pointerEvents: 'none',
      }}>
        {toasts.map(toast => (
          <div key={toast.id} style={{
            background: 'rgba(20,20,30,0.97)',
            border: `1px solid ${
              toast.type === 'pressing_spike' ? 'rgba(0,255,136,0.5)' :
              toast.type === 'fatigue_spike' ? 'rgba(255,100,100,0.5)' :
              toast.type === 'formation_change' ? 'rgba(68,136,255,0.5)' :
              'rgba(255,170,0,0.5)'
            }`,
            borderRadius: '12px',
            padding: '10px 16px',
            minWidth: '260px',
            maxWidth: '320px',
            boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
            animation: 'fadeIn 0.3s ease',
            pointerEvents: 'auto',
          }}>
            <div style={{ fontWeight: '700', fontSize: '13px', color: 'var(--text-primary)', marginBottom: '3px' }}>
              {toast.title}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
              {toast.description}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
