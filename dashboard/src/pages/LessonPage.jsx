import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, ClipboardCheck, EyeOff, Info } from 'lucide-react'
import { api } from '../api'
import EngagementChart from '../components/charts/EngagementChart'
import StateMixBar from '../components/charts/StateMixBar'
import SeatMap from '../components/charts/SeatMap'
import { clock, minutes, pct } from '../lib/format'

export default function LessonPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [lesson, setLesson] = useState(null)
  const [seats, setSeats] = useState(null)
  const [error, setError] = useState(null)
  const [time, setTime] = useState(null)
  const videoRef = useRef(null)

  useEffect(() => {
    api.lesson(id).then(setLesson).catch((e) => setError(e.message))
    api.seats(id).then(setSeats).catch(() => setSeats(null))
  }, [id])

  const seek = (t) => {
    if (videoRef.current) {
      videoRef.current.currentTime = t
      videoRef.current.play().catch(() => {})
    }
    setTime(t)
  }

  if (error) return <div className="p-6 text-red-300 text-sm">{error}</div>
  if (!lesson) return <div className="p-6 text-slate-400">Loading…</div>

  const v = lesson.video
  const t = lesson.teacher
  const hasVideo = !!v
  const questionCount = lesson.questions.length

  return (
    <div className="p-6 max-w-7xl space-y-6">
      <div>
        <Link to="/lessons" className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
          <ArrowLeft size={12} /> Lessons
        </Link>
        <div className="mt-2 flex items-center justify-between gap-4">
          <h1 className="text-xl font-semibold text-white break-all">{lesson.id.replace(/_/g, ' ')}</h1>
          {lesson.has_exam && (
            <Link to={`/lessons/${lesson.id}/exam`}
              className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800 whitespace-nowrap">
              <ClipboardCheck size={14} /> Exam review
            </Link>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <Tile label="Length" value={minutes(t?.duration_s ?? v?.duration_analysed_s)} />
        <Tile label="Average engaged" value={pct(v?.mean_engaged_share)} muted={!hasVideo} />
        <Tile label="Seats" value={v?.seats ?? '—'} muted={!hasVideo} />
        <Tile label="Teacher talk" value={pct(t?.talk_ratio)} muted={!t} />
        <Tile label="Pace" value={t ? `${Math.round(t.words_per_minute)} wpm` : '—'} muted={!t} />
        <Tile label="Questions" value={t ? questionCount : '—'} muted={!t} />
      </div>

      {hasVideo && (
        <section className="grid gap-6 xl:grid-cols-5">
          <div className="xl:col-span-2">
            {lesson.has_preview ? (
              <>
                <video ref={videoRef} src={api.previewUrl(id)} controls preload="metadata"
                  onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
                  className="w-full rounded-lg border border-slate-800 bg-black" />
                <div className="mt-1 flex items-center gap-1 text-xs text-slate-500">
                  <EyeOff size={12} /> Faces of students without consent are blurred.
                </div>
              </>
            ) : (
              <div className="rounded-lg border border-slate-800 p-6 text-sm text-slate-400">No preview video for this lesson.</div>
            )}
          </div>
          <Panel className="xl:col-span-3" title="Class engagement" subtitle="Share of seated students attentive, writing or raising a hand">
            <EngagementChart points={lesson.engagement} topics={lesson.topics} current={time}
              onSeek={lesson.has_preview ? seek : undefined} />
            <div className="mt-4">
              <div className="text-xs text-slate-400 mb-2">What students were doing</div>
              <StateMixBar mix={lesson.state_mix} />
            </div>
          </Panel>
        </section>
      )}

      {hasVideo && seats && (
        <Panel title="Engagement by seat" subtitle={seatSubtitle(seats)}>
          <div className="max-w-3xl">
            <SeatMap seats={seats.seats} onSelectStudent={seats.can_see_students ? (s) => navigate(`/students/${s}`) : undefined} />
          </div>
        </Panel>
      )}

      {lesson.topics.length > 0 && (
        <Panel title="Topics" subtitle="Segmented from the teacher's speech">
          <TopicsTable lesson={lesson} onSeek={lesson.has_preview ? seek : undefined} />
        </Panel>
      )}

      {t && (
        <section className="grid gap-6 lg:grid-cols-2">
          <Panel title="Teacher report" subtitle="For the teacher's own coaching first">
            <TeacherReport lesson={lesson} />
          </Panel>
          <Panel title="Transcript" subtitle={`Teacher mic · ${t.language ?? ''} · ${t.model ?? ''}`}>
            <Transcript segments={lesson.transcript} time={time} onSeek={lesson.has_preview ? seek : undefined} />
          </Panel>
        </section>
      )}

      {!hasVideo && (
        <Note>This lesson has teacher audio only. Class engagement appears when a classroom video is analysed for the same lesson.</Note>
      )}
      {hasVideo && !t && (
        <Note>This lesson has classroom video only. Topics and teaching metrics appear when the teacher's mic audio is analysed.</Note>
      )}
    </div>
  )
}

function seatSubtitle({ can_see_students, seats }) {
  if (!can_see_students) return 'Seats are anonymous for your role. Positions are as seen from the classroom camera.'
  const named = seats.filter((s) => s.student_id).length
  if (!named) return 'No consented students identified in this lesson, so all seats are anonymous.'
  return `${named} seats are consented, identified students. Click one for their view.`
}

function TopicsTable({ lesson, onSeek }) {
  const rows = lesson.topic_table.length ? lesson.topic_table : lesson.topics.map((tp) => ({
    start_min: tp.start_s / 60, end_min: tp.end_s / 60, title: tp.title, method: tp.method, summary: tp.summary,
  }))
  const hasEngagement = rows.some((r) => r.engaged_share != null)
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-xs text-slate-400 text-left">
          <tr>
            <th className="py-2 pr-3 font-normal">Time</th>
            <th className="py-2 pr-3 font-normal">Topic</th>
            <th className="py-2 pr-3 font-normal">Method</th>
            {rows[0]?.words_per_min != null && <th className="py-2 pr-3 font-normal text-right">Pace</th>}
            {rows[0]?.questions != null && <th className="py-2 pr-3 font-normal text-right">Questions</th>}
            {hasEngagement && <th className="py-2 pr-3 font-normal">Engaged</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-slate-800 align-top">
              <td className="py-2 pr-3 text-slate-400 whitespace-nowrap">
                {onSeek
                  ? <button onClick={() => onSeek(r.start_min * 60)} className="hover:text-blue-300">{clock(r.start_min * 60)}</button>
                  : clock(r.start_min * 60)}
                <span className="text-slate-600"> – {clock(r.end_min * 60)}</span>
              </td>
              <td className="py-2 pr-3">
                <div className="text-slate-100">{r.title}</div>
                {r.summary && <div className="text-xs text-slate-500 mt-0.5">{r.summary}</div>}
              </td>
              <td className="py-2 pr-3 text-slate-300">{r.method}</td>
              {r.words_per_min != null && <td className="py-2 pr-3 text-right text-slate-300">{Math.round(r.words_per_min)}</td>}
              {r.questions != null && <td className="py-2 pr-3 text-right text-slate-300">{r.questions}</td>}
              {hasEngagement && (
                <td className="py-2 pr-3 w-40">
                  {r.engaged_share == null ? '—' : (
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 rounded bg-slate-800">
                        <div className="h-2 rounded" style={{ width: pct(r.engaged_share), background: 'var(--viz-series-1)' }} />
                      </div>
                      <span className="text-xs text-slate-300 w-9 text-right">{pct(r.engaged_share)}</span>
                    </div>
                  )}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TeacherReport({ lesson }) {
  const t = lesson.teacher
  const byType = useMemo(() => {
    const counts = {}
    lesson.questions.forEach((q) => { const k = q.effective_type || q.type; counts[k] = (counts[k] || 0) + 1 })
    return counts
  }, [lesson.questions])
  const waits = lesson.questions.map((q) => q.wait_time_s).filter((w) => w != null)
  const waited = waits.filter((w) => w >= 3).length

  return (
    <div className="space-y-4 text-sm">
      <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <Tile label="Talk ratio" value={pct(t.talk_ratio)} />
        <Tile label="Pace" value={`${Math.round(t.words_per_minute)} wpm`} />
        <Tile label="Longest monologue" value={minutes(t.longest_monologue_s)} />
        <Tile label="Questions / 10 min" value={lesson.questions.length && t.duration_s
          ? (lesson.questions.length / (t.duration_s / 600)).toFixed(1) : '0'} />
        <Tile label="Waited ≥ 3 s" value={waits.length ? `${waited} of ${waits.length}` : '—'} />
        <Tile label="ASR confidence" value={pct(t.mean_asr_confidence)} />
      </dl>

      {lesson.questions.length > 0 && (
        <div>
          <div className="text-xs text-slate-400 mb-1">
            Questions by type: {Object.entries(byType).map(([k, n]) => `${k.replace(/_/g, ' ')} ${n}`).join(' · ')}
          </div>
          <ul className="max-h-56 overflow-auto divide-y divide-slate-800">
            {lesson.questions.map((q, i) => (
              <li key={i} className="py-1.5 flex gap-3">
                <span className="text-slate-500 w-10 flex-shrink-0">{clock(q.time_s)}</span>
                <span className="text-slate-200 flex-1">{q.text_en || q.text}</span>
                <span className="text-xs text-slate-400 whitespace-nowrap">
                  {(q.effective_type || q.type).replace(/_/g, ' ')} · {q.wait_time_s?.toFixed(1)} s
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {lesson.methods.length > 0 && (
        <div>
          <div className="text-xs text-slate-400 mb-1">Time by teaching method</div>
          {lesson.methods.map((m) => (
            <div key={m.method} className="flex items-center gap-2 py-0.5">
              <span className="w-28 text-slate-300">{m.method}</span>
              <div className="h-2 flex-1 rounded bg-slate-800">
                <div className="h-2 rounded" style={{
                  width: `${(m.minutes / lesson.methods.reduce((a, x) => a + x.minutes, 0)) * 100}%`,
                  background: 'var(--viz-series-1)',
                }} />
              </div>
              <span className="w-16 text-right text-xs text-slate-400">{m.minutes.toFixed(1)} min</span>
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-slate-500 flex gap-1.5">
        <Info size={12} className="mt-0.5 flex-shrink-0" />
        Wait time is the pause before the teacher speaks again. Questions answered within 1.5 s count as rhetorical.
      </p>
    </div>
  )
}

function Transcript({ segments, time, onSeek }) {
  const activeRef = useRef(null)
  const active = time == null ? -1 : segments.findIndex((s) => time >= s.start && time < s.end)
  useEffect(() => { activeRef.current?.scrollIntoView({ block: 'nearest' }) }, [active])
  return (
    <div className="max-h-96 overflow-auto text-sm space-y-1 pr-2">
      {segments.map((s, i) => (
        <div key={i} ref={i === active ? activeRef : null}
          onClick={() => onSeek?.(s.start)}
          className={`flex gap-3 rounded px-2 py-1 ${i === active ? 'bg-slate-800' : ''} ${onSeek ? 'cursor-pointer hover:bg-slate-800/60' : ''}`}>
          <span className="text-slate-500 w-10 flex-shrink-0 text-xs mt-0.5">{clock(s.start)}</span>
          <span className="text-slate-200">{s.text}</span>
        </div>
      ))}
    </div>
  )
}

function Panel({ title, subtitle, children, className = '' }) {
  return (
    <section className={`rounded-lg border border-slate-800 bg-slate-900 p-4 ${className}`}>
      <h2 className="text-sm font-medium text-white">{title}</h2>
      {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
      <div className="mt-3">{children}</div>
    </section>
  )
}

function Tile({ label, value, muted }) {
  return (
    <div className={`rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 ${muted ? 'opacity-50' : ''}`}>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  )
}

function Note({ children }) {
  return (
    <div className="flex gap-2 rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3 text-sm text-slate-400">
      <Info size={14} className="mt-0.5 flex-shrink-0" /> {children}
    </div>
  )
}
