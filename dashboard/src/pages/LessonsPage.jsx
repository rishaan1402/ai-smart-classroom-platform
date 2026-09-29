import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { BookOpen, Mic, Video } from 'lucide-react'
import { api } from '../api'
import { minutes, pct } from '../lib/format'

export default function LessonsPage() {
  const [lessons, setLessons] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.lessons().then(setLessons).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="p-6 max-w-6xl">
      <h1 className="text-xl font-semibold text-white">Lessons</h1>
      <p className="text-sm text-slate-400 mt-1">
        Analysed lessons: class engagement from the classroom camera, teaching metrics from the teacher's mic.
      </p>

      {error && <div className="mt-4 text-sm text-red-300">{error}</div>}
      {!lessons && !error && <div className="mt-6 text-slate-400">Loading…</div>}
      {lessons?.length === 0 && (
        <div className="mt-6 text-slate-400 text-sm">
          No lessons yet. Run <code className="text-slate-300">classroom.pipeline</code> or{' '}
          <code className="text-slate-300">audio.pipeline</code> to produce one.
        </div>
      )}

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {lessons?.map((l) => (
          <Link key={l.id} to={`/lessons/${l.id}`}
            className="rounded-lg border border-slate-800 bg-slate-900 p-4 hover:border-slate-600 transition-colors">
            <div className="flex items-start gap-2">
              <BookOpen size={16} className="text-blue-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm font-medium text-white break-all">{l.id.replace(/_/g, ' ')}</div>
            </div>
            <div className="mt-3 flex gap-3 text-xs text-slate-400">
              {l.has_video && <span className="flex items-center gap-1"><Video size={12} /> classroom video</span>}
              {l.has_audio && <span className="flex items-center gap-1"><Mic size={12} /> teacher audio{l.language ? ` · ${l.language}` : ''}</span>}
            </div>
            <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <Stat label="Length" value={minutes(l.duration_s)} />
              <Stat label="Engaged" value={pct(l.mean_engaged_share)} />
              {l.has_audio
                ? <Stat label="Teacher talk" value={pct(l.talk_ratio)} />
                : <Stat label="Seats" value={l.seats ?? '—'} />}
            </dl>
            {l.topics > 0 && <div className="mt-2 text-xs text-slate-500">{l.topics} topics</div>}
          </Link>
        ))}
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-slate-100 font-medium">{value}</dd>
    </div>
  )
}
