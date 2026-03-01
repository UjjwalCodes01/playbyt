import { useState, useEffect } from 'react'
import type { RoomConfig, FanRole } from '../App'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const ROLES: { id: FanRole; label: string; icon: string; desc: string; color: string }[] = [
  { id: 'analyst', label: 'Analyst', icon: '🧠', desc: 'Tactical breakdowns & formations', color: '#4488ff' },
  { id: 'hype', label: 'Hype Fan', icon: '🔥', desc: 'Pure energy & reactions', color: '#ff4444' },
  { id: 'stats', label: 'Stats Nerd', icon: '📊', desc: 'Patterns & probabilities', color: '#ffaa00' },
  { id: 'coach', label: 'Coach', icon: '📋', desc: 'Fitness & positioning analysis', color: '#00ff88' },
]

// Fetch a Stream Video token from our FastAPI backend
async function getToken(userId: string, userName: string): Promise<{ token: string; api_key: string }> {
  const res = await fetch(`${API_BASE}/api/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, user_name: userName }),
  })
  if (!res.ok) throw new Error('Token server error')
  return res.json()
}

// Try to auto-fetch the active call ID from the backend
async function fetchActiveCallId(): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/api/call-id`)
    if (!res.ok) return null
    const data = await res.json()
    return data.call_id || null
  } catch {
    return null
  }
}

interface JoinRoomProps {
  onJoin: (config: RoomConfig) => void
}

export function JoinRoom({ onJoin }: JoinRoomProps) {
  const [callId, setCallId] = useState('')
  const [userName, setUserName] = useState('')
  const [role, setRole] = useState<FanRole>('analyst')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [autoDetected, setAutoDetected] = useState(false)

  // Auto-detect active call on mount
  useEffect(() => {
    fetchActiveCallId().then(id => {
      if (id) {
        setCallId(id)
        setAutoDetected(true)
      }
    })
  }, [])

  async function handleJoin(e: React.FormEvent) {
    e.preventDefault()
    if (!callId.trim() || !userName.trim()) return

    setLoading(true)
    setError('')

    try {
      const userId = `user-${userName.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`
      const { token: userToken, api_key: apiKey } = await getToken(userId, userName.trim())

      onJoin({
        callId: callId.trim(),
        userName: userName.trim(),
        userId,
        userToken,
        apiKey,
        role,
      })
    } catch {
      setError('Could not connect. Make sure the backend is running (uvicorn server:app --port 8000).')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'var(--bg-primary)',
      padding: '24px',
    }}>
      {/* Logo */}
      <div style={{ textAlign: 'center', marginBottom: '48px' }}>
        <div style={{
          fontSize: '56px',
          fontWeight: '900',
          letterSpacing: '-2px',
          background: 'linear-gradient(135deg, #00ff88, #4488ff)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '8px',
        }}>
          PlayByt
        </div>
        <div style={{ color: 'var(--text-secondary)', fontSize: '16px', letterSpacing: '0.05em' }}>
          AI That Catches What You Miss
        </div>
        <div style={{
          marginTop: '12px',
          display: 'flex',
          gap: '8px',
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}>
          {['🧠 AI Analysis', '👁️ YOLO Tracking', '⚡ Tool Calling', '🔴 Live'].map(tag => (
            <span key={tag} style={{
              background: 'rgba(0,255,136,0.08)',
              border: '1px solid rgba(0,255,136,0.2)',
              color: '#00ff88',
              padding: '4px 10px',
              borderRadius: '20px',
              fontSize: '12px',
              fontWeight: '600',
            }}>{tag}</span>
          ))}
        </div>
      </div>

      {/* Join Card */}
      <form onSubmit={handleJoin} style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: '16px',
        padding: '40px',
        width: '100%',
        maxWidth: '440px',
        boxShadow: '0 8px 40px rgba(0,0,0,0.5)',
      }}>
        <h2 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '28px', color: 'var(--text-primary)' }}>
          Join the Game Room
        </h2>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '13px', fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Your Name
          </label>
          <input
            type="text"
            value={userName}
            onChange={e => setUserName(e.target.value)}
            placeholder="e.g. Jake"
            required
            style={{
              width: '100%',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: '10px',
              padding: '14px 16px',
              color: 'var(--text-primary)',
              fontSize: '15px',
              transition: 'border-color 0.2s',
              outline: 'none',
            }}
            onFocus={e => (e.target.style.borderColor = '#00ff88')}
            onBlur={e => (e.target.style.borderColor = 'var(--border)')}
          />
        </div>

        <div style={{ marginBottom: '28px' }}>
          <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '13px', fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Call ID
          </label>
          <input
            type="text"
            value={callId}
            onChange={e => { setCallId(e.target.value); setAutoDetected(false) }}
            placeholder="Paste the Call ID from your agent"
            required
            style={{
              width: '100%',
              background: 'var(--bg-secondary)',
              border: `1px solid ${autoDetected ? 'rgba(0,255,136,0.4)' : 'var(--border)'}`,
              borderRadius: '10px',
              padding: '14px 16px',
              color: 'var(--text-primary)',
              fontSize: '15px',
              outline: 'none',
            }}
            onFocus={e => (e.target.style.borderColor = '#4488ff')}
            onBlur={e => (e.target.style.borderColor = autoDetected ? 'rgba(0,255,136,0.4)' : 'var(--border)')}
          />
          {autoDetected ? (
            <p style={{ color: '#00ff88', fontSize: '12px', marginTop: '6px' }}>
              ✅ Auto-detected from running agent
            </p>
          ) : (
            <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginTop: '6px' }}>
              Run <code style={{ color: '#00ff88', background: 'rgba(0,255,136,0.1)', padding: '2px 6px', borderRadius: '4px' }}>python main.py run</code> and copy the Call ID from the logs
            </p>
          )}
        </div>

        {/* Role Picker */}
        <div style={{ marginBottom: '28px' }}>
          <label style={{ display: 'block', color: 'var(--text-secondary)', fontSize: '13px', fontWeight: '600', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Your Role
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {ROLES.map(r => (
              <button
                key={r.id}
                type="button"
                onClick={() => setRole(r.id)}
                style={{
                  background: role === r.id ? `${r.color}15` : 'var(--bg-secondary)',
                  border: `2px solid ${role === r.id ? r.color : 'var(--border)'}`,
                  borderRadius: '12px',
                  padding: '14px 12px',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.2s',
                }}
              >
                <div style={{ fontSize: '20px', marginBottom: '4px' }}>{r.icon}</div>
                <div style={{ color: role === r.id ? r.color : 'var(--text-primary)', fontSize: '14px', fontWeight: '700' }}>
                  {r.label}
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '11px', marginTop: '2px' }}>
                  {r.desc}
                </div>
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div style={{
            background: 'rgba(255,107,53,0.1)',
            border: '1px solid rgba(255,107,53,0.3)',
            borderRadius: '8px',
            padding: '12px',
            color: '#ff6b35',
            fontSize: '13px',
            marginBottom: '20px',
          }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !callId.trim() || !userName.trim()}
          style={{
            width: '100%',
            padding: '16px',
            background: loading ? 'var(--bg-secondary)' : 'linear-gradient(135deg, #00ff88, #4488ff)',
            borderRadius: '10px',
            color: loading ? 'var(--text-secondary)' : '#0a0a0f',
            fontWeight: '800',
            fontSize: '16px',
            letterSpacing: '0.02em',
            transition: 'opacity 0.2s, transform 0.1s',
            opacity: (!callId.trim() || !userName.trim()) ? 0.5 : 1,
          }}
        >
          {loading ? '🔄 Connecting...' : '⚽ Enter Game Room'}
        </button>
      </form>

      <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginTop: '24px', textAlign: 'center' }}>
        Built for Vision Possible: Agent Protocol hackathon · Powered by Stream + Gemini
      </p>
    </div>
  )
}
