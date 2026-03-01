import { useState } from 'react'
import { JoinRoom } from './components/JoinRoom'
import { PlayBytRoom } from './components/PlayBytRoom'
import '@stream-io/video-react-sdk/dist/css/styles.css'
import './App.css'

export type FanRole = 'analyst' | 'hype' | 'stats' | 'coach'

export interface RoomConfig {
  callId: string
  userName: string
  userId: string
  userToken: string
  apiKey: string
  role: FanRole
}

export default function App() {
  const [roomConfig, setRoomConfig] = useState<RoomConfig | null>(null)

  if (!roomConfig) {
    return <JoinRoom onJoin={setRoomConfig} />
  }

  return (
    <PlayBytRoom
      config={roomConfig}
      onLeave={() => setRoomConfig(null)}
    />
  )
}
