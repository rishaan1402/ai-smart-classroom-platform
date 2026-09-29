import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertTriangle, RefreshCw, Wifi, WifiOff } from 'lucide-react'
import { api } from '../api'

const SEVERITY_STYLES = {
  critical: 'bg-red-900/50 border-red-700 text-red-300',
  high: 'bg-orange-900/50 border-orange-700 text-orange-300',
  medium: 'bg-yellow-900/50 border-yellow-700 text-yellow-300',
  low: 'bg-slate-800 border-slate-700 text-slate-300',
}

const SEVERITY_DOT = {
  critical: 'bg-red-500',
  high: 'bg-orange-500',
  medium: 'bg-yellow-500',
  low: 'bg-slate-500',
}

const FILTERS = [
  { label: 'All', value: 'critical,high,medium,low' },
  { label: 'Critical', value: 'critical' },
  { label: 'High', value: 'high' },
  { label: 'Medium', value: 'medium' },
  { label: 'Low', value: 'low' },
]

function EventCard({ event }) {
  const style = SEVERITY_STYLES[event.severity] || SEVERITY_STYLES.low
  const dot = SEVERITY_DOT[event.severity] || SEVERITY_DOT.low
  const time = new Date(event.created_at).toLocaleTimeString()
  const date = new Date(event.created_at).toLocaleDateString()

  return (
    <div className={`rounded-lg border p-4 ${style}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 mt-1 ${dot}`} />
          <div>
            <div className="text-sm font-medium text-white">
              {event.event_type.replace(/_/g, ' ')}
            </div>
            <div className="text-xs opacity-70 mt-0.5">
              Person #{event.person_id} · Camera {event.camera_id?.slice(0, 8) ?? '—'}
            </div>
          </div>
        </div>
        <div className="text-right text-xs opacity-60 flex-shrink-0">
          <div>{time}</div>
          <div>{date}</div>
        </div>
      </div>
      {event.threat_level && (
        <div className="mt-2 text-xs opacity-80">
          Threat: <span className="font-medium">{event.threat_level}</span>
        </div>
      )}
      {event.face_display && event.face_display !== 'face detection off' && (
        <div className="mt-1 text-xs opacity-70">Face: {event.face_display}</div>
      )}
      {event.enrichment_skipped && (
        <div className="mt-1 text-xs text-yellow-400 opacity-80">⚠ Enrichment skipped</div>
      )}
    </div>
  )
}

export default function EventsPage() {
  const [events, setEvents] = useState([])
  const [filter, setFilter] = useState('critical,high')
  const [loading, setLoading] = useState(true)
  const [wsStatus, setWsStatus] = useState('disconnected') // connected | disconnected | error
  const wsRef = useRef(null)

  const fetchEvents = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.events({ severity: filter, limit: 100 })
      setEvents(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [filter])

  useEffect(() => {
    fetchEvents()
  }, [fetchEvents])

  // WebSocket for live updates
  useEffect(() => {
    let ws
    let authTimeout

    const connect = async () => {
      try {
        const { ws_token } = await api.wsToken()
        const proto = location.protocol === 'https:' ? 'wss' : 'ws'
        ws = new WebSocket(`${proto}://${location.host}/ws/events`)
        wsRef.current = ws

        ws.onopen = () => {
          // Wait for auth_required then send token
        }

        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data)
          if (msg.type === 'auth_required') {
            ws.send(JSON.stringify({ type: 'auth', token: ws_token, last_event_id: null }))
            authTimeout = setTimeout(() => ws.close(4401), 5000)
          } else if (msg.type === 'auth_ok') {
            clearTimeout(authTimeout)
            setWsStatus('connected')
          } else if (msg.type === 'event' || msg.type === 'replay') {
            const event = msg.event || msg
            setEvents(prev => [event, ...prev].slice(0, 200))
          }
        }

        ws.onerror = () => setWsStatus('error')
        ws.onclose = () => {
          setWsStatus('disconnected')
          // Reconnect after 5s
          setTimeout(connect, 5000)
        }
      } catch {
        setWsStatus('error')
        setTimeout(connect, 5000)
      }
    }

    connect()
    return () => {
      clearTimeout(authTimeout)
      wsRef.current?.close()
    }
  }, [])

  const wsIcon = wsStatus === 'connected'
    ? <Wifi size={14} className="text-green-400" />
    : <WifiOff size={14} className="text-slate-500" />

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-white">Events</h1>
          <div className="flex items-center gap-1.5 mt-1 text-xs text-slate-500">
            {wsIcon}
            <span>{wsStatus === 'connected' ? 'Live' : 'Polling'}</span>
          </div>
        </div>
        <button
          onClick={fetchEvents}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Severity filter */}
      <div className="flex gap-2 mb-5 flex-wrap">
        {FILTERS.map(f => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              filter === f.value
                ? 'bg-blue-600 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {loading && events.length === 0 ? (
        <div className="text-slate-500 text-sm text-center py-12">Loading events…</div>
      ) : events.length === 0 ? (
        <div className="text-slate-500 text-sm text-center py-12">
          <AlertTriangle size={32} className="mx-auto mb-3 opacity-30" />
          No events found
        </div>
      ) : (
        <div className="space-y-2">
          {events.map(e => <EventCard key={e.id} event={e} />)}
        </div>
      )}
    </div>
  )
}
