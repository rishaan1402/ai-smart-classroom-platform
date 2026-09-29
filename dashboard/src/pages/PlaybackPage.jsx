import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Film, Image, Play } from 'lucide-react'

const SEVERITY_COLOR = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#eab308',
  low: '#64748b',
}

function TimelineScrubber({ events, selected, onSelect }) {
  if (!events.length) return null

  const first = events[0].ts_ms
  const last = events[events.length - 1].ts_ms
  const span = last - first || 1

  return (
    <div className="relative h-12 bg-slate-800 rounded-lg overflow-hidden">
      {events.map(e => {
        const pct = ((e.ts_ms - first) / span) * 100
        const color = SEVERITY_COLOR[e.severity] || SEVERITY_COLOR.low
        return (
          <button
            key={e.id}
            onClick={() => onSelect(e)}
            title={`${e.event_type} — ${new Date(e.created_at).toLocaleTimeString()}`}
            className="absolute top-1 w-1.5 h-10 rounded-sm opacity-80 hover:opacity-100 transition-opacity"
            style={{ left: `${pct}%`, background: color, transform: 'translateX(-50%)' }}
          />
        )
      })}
      {selected && (
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white"
          style={{
            left: `${((selected.ts_ms - first) / span) * 100}%`,
          }}
        />
      )}
    </div>
  )
}

export default function PlaybackPage() {
  const [timeline, setTimeline] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)
  const [hoursBack, setHoursBack] = useState(24)

  useEffect(() => {
    const since = new Date(Date.now() - hoursBack * 3600 * 1000).toISOString()
    setLoading(true)
    fetch(`/api/playback/timeline?since=${since}&limit=500`, { credentials: 'include' })
      .then(r => r.json())
      .then(data => { setTimeline(Array.isArray(data) ? data : []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [hoursBack])

  const idx = selected ? timeline.findIndex(e => e.id === selected.id) : -1
  const prev = idx > 0 ? timeline[idx - 1] : null
  const next = idx >= 0 && idx < timeline.length - 1 ? timeline[idx + 1] : null

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold text-white">Playback</h1>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Show last</span>
          {[1, 6, 24, 72].map(h => (
            <button
              key={h}
              onClick={() => setHoursBack(h)}
              className={`text-xs px-2 py-1 rounded ${hoursBack === h ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
            >
              {h}h
            </button>
          ))}
        </div>
      </div>

      {/* Timeline scrubber */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-slate-600 mb-1">
          <span>{timeline.length > 0 ? new Date(timeline[0]?.created_at).toLocaleString() : '—'}</span>
          <span className="text-slate-500">{timeline.length} clips</span>
          <span>{timeline.length > 0 ? new Date(timeline[timeline.length - 1]?.created_at).toLocaleString() : '—'}</span>
        </div>
        {loading ? (
          <div className="h-12 bg-slate-800 rounded-lg animate-pulse" />
        ) : (
          <TimelineScrubber events={timeline} selected={selected} onSelect={setSelected} />
        )}
      </div>

      {/* Player */}
      {selected ? (
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
          {/* Clip or snapshot */}
          <div className="bg-black aspect-video flex items-center justify-center">
            {selected.has_clip ? (
              <video
                key={selected.id}
                controls
                autoPlay
                className="w-full h-full"
                src={`/api/playback/clip/${selected.id}`}
              />
            ) : selected.has_snapshot ? (
              <img
                key={selected.id}
                src={`/api/playback/snapshot/${selected.id}`}
                alt="Event snapshot"
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <div className="text-slate-600 text-sm">No media</div>
            )}
          </div>

          {/* Event details + nav */}
          <div className="p-4 flex items-center justify-between">
            <button
              onClick={() => setSelected(prev)}
              disabled={!prev}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 transition-colors"
            >
              <ChevronLeft size={16} />
            </button>

            <div className="text-center">
              <div className="text-sm font-medium text-white">
                {selected.event_type.replace(/_/g, ' ')}
              </div>
              <div className="text-xs text-slate-500 mt-0.5">
                {new Date(selected.created_at).toLocaleString()}
              </div>
              <div className="flex items-center justify-center gap-3 mt-2 text-xs text-slate-500">
                {selected.has_clip && (
                  <span className="flex items-center gap-1"><Film size={12} /> Clip</span>
                )}
                {selected.has_snapshot && (
                  <span className="flex items-center gap-1"><Image size={12} /> Snapshot</span>
                )}
              </div>
            </div>

            <button
              onClick={() => setSelected(next)}
              disabled={!next}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-slate-900 border border-slate-800 rounded-xl aspect-video flex flex-col items-center justify-center text-slate-600">
          <Play size={40} className="mb-3 opacity-30" />
          <p className="text-sm">Click an event on the timeline to play</p>
        </div>
      )}
    </div>
  )
}
