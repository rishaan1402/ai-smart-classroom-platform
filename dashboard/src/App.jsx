import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './AuthContext'
import LoginPage from './pages/LoginPage'
import EventsPage from './pages/EventsPage'
import CamerasPage from './pages/CamerasPage'
import HealthPage from './pages/HealthPage'
import PlaybackPage from './pages/PlaybackPage'
import Layout from './components/Layout'

function RequireAuth({ children }) {
  const { user, loading } = useAuth()
  if (loading) return (
    <div className="flex items-center justify-center h-screen text-slate-400">
      Loading…
    </div>
  )
  if (!user) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<RequireAuth><Layout /></RequireAuth>}>
            <Route index element={<Navigate to="/events" replace />} />
            <Route path="events" element={<EventsPage />} />
            <Route path="cameras" element={<CamerasPage />} />
            <Route path="playback" element={<PlaybackPage />} />
            <Route path="health" element={<HealthPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
