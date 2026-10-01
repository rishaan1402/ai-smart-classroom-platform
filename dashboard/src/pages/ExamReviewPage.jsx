import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AlertTriangle, ArrowLeft, Bot, Check, EyeOff, HelpCircle, Info, X } from 'lucide-react'
import { api } from '../api'
import { clock } from '../lib/format'

const TYPE_LABELS = {
  gaze_neighbour: 'Looking towards a neighbour',
  talking: 'Facing a neighbour (talking?)',
  turned_around: 'Turned around',
  reaching: 'Hand moved towards a neighbour',
  device: 'Device in hand (phone or calculator)',
  out_of_seat: 'Out of seat',
}
const DECISIONS = [
  { value: 'confirmed', label: 'Confirm', icon: Check, key: 'c', style: 'border-red-500/60 hover:bg-red-950/40' },
  { value: 'dismissed', label: 'Dismiss', icon: X, key: 'd', style: 'border-slate-600 hover:bg-slate-800' },
  { value: 'unclear', label: 'Unclear', icon: HelpCircle, key: 'u', style: 'border-slate-600 hover:bg-slate-800' },
]

/**
 * Invigilator review queue. Flags are prompts, not findings: each needs a
 * person to confirm, dismiss, or mark unclear after watching the clip.
 */
export default function ExamReviewPage() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [filter, setFilter] = useState('all')
  const [showCleared, setShowCleared] = useState(false)
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.exam(id).then((d) => {
      setData(d)
      setSelected(d.flags.find((f) => !f.review && f.vlm?.verdict !== 'normal')?.id ?? d.flags[0]?.id ?? null)
    }).catch((e) => setError(e.message))
  }, [id])

  const cleared = (f) => f.vlm?.verdict === 'normal'
  const flags = useMemo(() => (data?.flags ?? []).filter((f) =>
    (filter === 'all' || f.type === filter) && (showCleared || !cleared(f))), [data, filter, showCleared])
  const flag = data?.flags.find((f) => f.id === selected)
  const reviewed = data?.flags.filter((f) => f.review).length ?? 0

  const decide = useCallback(async (decision) => {
    if (!flag || saving) return
    setSaving(true)
    try {
      const review = await api.reviewExamFlag(id, flag.id, decision, note)
      setData((d) => ({ ...d, flags: d.flags.map((f) => (f.id === flag.id ? { ...f, review } : f)) }))
      setNote('')
      const next = flags.find((f) => !f.review && f.id !== flag.id)
      if (next) setSelected(next.id)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }, [flag, flags, id, note, saving])

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'TEXTAREA') return
      const d = DECISIONS.find((x) => x.key === e.key)
      if (d) decide(d.value)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [decide])

  if (error) return <div className="p-6 text-sm text-red-300">{error}</div>
  if (!data) return <div className="p-6 text-slate-400">Loading…</div>
  const s = data.summary
  const types = Object.keys(s.by_type || {}).filter((t) => s.by_type[t] > 0)

  return (
    <div className="p-6 max-w-7xl space-y-4">
      <div>
        <Link to={`/lessons/${id}`} className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
          <ArrowLeft size={12} /> Lesson
        </Link>
        <h1 className="mt-2 text-xl font-semibold text-white">Exam review: {id.replace(/_/g, ' ')}</h1>
      </div>

      <div className="flex gap-2 rounded-lg border border-amber-700/50 bg-amber-950/30 px-4 py-3 text-sm text-amber-200">
        <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
        <span>
          Flags are automatic prompts from camera analysis, not evidence of cheating. Watch the clip and decide.
          On this camera most flags are expected to be false alarms.
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        <Tile label="Flags" value={s.flags} />
        <Tile label="Queue after VLM" value={s.queue_after_vlm ?? '—'} />
        <Tile label="High priority" value={s.high_priority ?? '—'} />
        <Tile label="Reviewed" value={`${reviewed} / ${data.flags.length}`} />
        <Tile label="Per student-hour" value={s.flags_per_student_hour ?? '—'} />
        <Tile label="Students assessable" value={s.assessable_share != null ? `${Math.round(s.assessable_share * 100)}%` : '—'} />
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        {['all', ...types].map((t) => (
          <button key={t} onClick={() => setFilter(t)}
            className={`rounded-full border px-3 py-1 ${filter === t ? 'border-blue-500 bg-blue-600 text-white' : 'border-slate-700 text-slate-300 hover:bg-slate-800'}`}>
            {t === 'all' ? `All (${data.flags.length})` : `${TYPE_LABELS[t] ?? t} (${s.by_type[t]})`}
          </button>
        ))}
        {s.vlm && (
          <label className="ml-auto flex items-center gap-2 text-slate-300">
            <input type="checkbox" checked={showCleared} onChange={(e) => setShowCleared(e.target.checked)} />
            Show {s.vlm.normal} flags the VLM judged normal
          </label>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <ul className="lg:col-span-2 max-h-[70vh] overflow-auto rounded-lg border border-slate-800 divide-y divide-slate-800">
          {flags.map((f) => (
            <li key={f.id}>
              <button onClick={() => setSelected(f.id)}
                className={`w-full text-left px-3 py-2 ${f.id === selected ? 'bg-slate-800' : 'hover:bg-slate-900'}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm text-slate-100">{TYPE_LABELS[f.type] ?? f.type}</span>
                  <ReviewBadge review={f.review} />
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  {f.student_id || `Seat ${f.seat_id}`} · {clock(f.start_s)}–{clock(f.end_s)} · score {f.score.toFixed(2)}
                  {f.vlm && <span className={`ml-1 ${VLM_STYLE[f.vlm.verdict]}`}>· VLM {f.vlm.verdict}</span>}
                </div>
              </button>
            </li>
          ))}
        </ul>

        {flag && (
          <div className="lg:col-span-3 space-y-3">
            {flag.clip ? (
              <video key={flag.id} src={api.examClipUrl(id, flag.id)} controls autoPlay muted loop
                className="w-full rounded-lg border border-slate-800 bg-black" />
            ) : (
              <div className="rounded-lg border border-slate-800 p-6 text-sm text-slate-400">No clip (beyond the clip limit for this run).</div>
            )}
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <EyeOff size={12} /> Yellow box: flagged student. Orange ring: neighbour's seat. Faces blurred unless identified under exam consent.
            </div>
            <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 text-sm space-y-1">
              <div className="text-white font-medium">{TYPE_LABELS[flag.type] ?? flag.type}</div>
              <div className="text-slate-300">
                {flag.student_id || `Seat ${flag.seat_id}`}{flag.neighbour_seat ? `, towards seat ${flag.neighbour_seat}` : ''}
              </div>
              <div className="text-slate-400">
                {clock(flag.start_s)}–{clock(flag.end_s)} · {flag.evidence.duration_s} s sustained · score {flag.score.toFixed(2)}
              </div>
              {flag.vlm && (
                <div className="flex gap-1.5 text-slate-300 pt-1">
                  <Bot size={14} className="mt-0.5 flex-shrink-0 text-slate-500" />
                  <span>
                    VLM ({flag.vlm.model}): <span className={VLM_STYLE[flag.vlm.verdict]}>{flag.vlm.verdict}</span>
                    {flag.vlm.description ? `: ${flag.vlm.description}` : ''}
                    {flag.vlm.disagree && (
                      <span className="block text-xs text-amber-300 mt-0.5">
                        The VLM's own label ({flag.vlm.vlm_activity?.replace(/_/g, ' ')}) disagreed with its description; verdict follows the description.
                      </span>
                    )}
                  </span>
                </div>
              )}
              {flag.review && (
                <div className="text-slate-400">Last review: {flag.review.decision}{flag.review.note ? ` ("${flag.review.note}")` : ''}</div>
              )}
            </div>
            <textarea value={note} onChange={(e) => setNote(e.target.value)} maxLength={500} rows={2}
              placeholder="Optional note (what you saw)"
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500" />
            <div className="flex gap-2">
              {DECISIONS.map((d) => {
                const Icon = d.icon
                return (
                  <button key={d.value} onClick={() => decide(d.value)} disabled={saving}
                    className={`flex items-center gap-2 rounded-lg border px-4 py-2 text-sm text-slate-100 disabled:opacity-50 ${d.style}`}>
                    <Icon size={14} /> {d.label}
                    <kbd className="ml-1 rounded bg-slate-800 border border-slate-600 px-1 text-xs text-slate-400">{d.key}</kbd>
                  </button>
                )
              })}
            </div>
            <p className="text-xs text-slate-500 flex gap-1.5">
              <Info size={12} className="mt-0.5 flex-shrink-0" /> Every review is recorded with your account in the audit log.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

const VLM_STYLE = { suspicious: 'text-red-300', normal: 'text-slate-400', unclear: 'text-amber-300' }

function ReviewBadge({ review }) {
  if (!review) return <span className="text-xs text-slate-500">to review</span>
  const style = review.decision === 'confirmed' ? 'text-red-300' : review.decision === 'dismissed' ? 'text-slate-400' : 'text-amber-300'
  return <span className={`text-xs ${style}`}>{review.decision}</span>
}

function Tile({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  )
}
