import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, ShieldCheck } from 'lucide-react'
import { api } from '../api'
import StateMixBar from '../components/charts/StateMixBar'
import { minutes, pct } from '../lib/format'

/**
 * One consented student across lessons. Trends and state mix only: no
 * ranking against classmates, no single "score" (see docs/step4_status.md).
 */
export default function StudentPage() {
  const { id } = useParams()
  const [student, setStudent] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.student(id).then(setStudent).catch((e) => setError(e.message))
  }, [id])

  if (error) return (
    <div className="p-6 text-sm text-slate-400">
      <Link to="/lessons" className="text-xs flex items-center gap-1 hover:text-slate-200"><ArrowLeft size={12} /> Lessons</Link>
      <p className="mt-4">{error === 'Student not found' ? 'No data for this student, or consent does not allow it.' : error}</p>
    </div>
  )
  if (!student) return <div className="p-6 text-slate-400">Loading…</div>

  return (
    <div className="p-6 max-w-4xl space-y-6">
      <div>
        <Link to="/lessons" className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
          <ArrowLeft size={12} /> Lessons
        </Link>
        <h1 className="mt-2 text-xl font-semibold text-white">{student.name}</h1>
        <div className="text-sm text-slate-400">{student.student_id} · Section {student.section}</div>
        <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-500">
          <ShieldCheck size={12} /> Shown under guardian consent for engagement analytics. This view is audit-logged.
        </div>
      </div>

      {student.lessons.length === 0 && <div className="text-sm text-slate-400">Not identified in any analysed lesson yet.</div>}

      {student.lessons.map((l) => (
        <section key={l.lesson_id} className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <div className="flex items-baseline justify-between gap-4">
            <Link to={`/lessons/${l.lesson_id}`} className="text-sm font-medium text-white hover:text-blue-300 break-all">
              {l.lesson_id.replace(/_/g, ' ')}
            </Link>
            <span className="text-xs text-slate-400 whitespace-nowrap">Seat {l.seat_id} · {minutes(l.seconds)} observed</span>
          </div>
          <div className="mt-3 flex items-center gap-3">
            <span className="text-2xl font-semibold text-slate-100">{pct(l.engaged_share)}</span>
            <span className="text-xs text-slate-400">of observed time attentive, writing or raising a hand</span>
          </div>
          <div className="mt-3"><StateMixBar mix={l.state_mix} height={12} /></div>
        </section>
      ))}

      <p className="text-xs text-slate-500">
        These are camera-based estimates, not grades. A student looking down may be taking notes; use this to start
        a conversation, never as a verdict.
      </p>
    </div>
  )
}
