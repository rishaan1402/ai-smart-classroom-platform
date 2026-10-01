import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, CheckCircle2, Tag } from 'lucide-react'
import { api } from '../api'

/** Label definitions shown to labellers; keep in sync with docs/labeling_guide.md. */
const GUIDE = {
  attentive: 'Head up, facing the lesson (board / teacher / speaking classmate).',
  writing: 'Head down over the desk with a hand working: writing, drawing, reading the book.',
  hand_raised: 'A hand raised to ask or answer, at any height.',
  head_down: 'Head resting on the desk or arms, or slumped, not working.',
  turned_away: 'Facing clearly away from the lesson (window, behind, sideways) and not talking.',
  interacting: 'Turned towards a neighbour who is turned back: talking, passing something.',
  phone: 'Holding or looking at a phone.',
  out_of_seat: 'Standing or walking away from their seat.',
  cant_tell: 'Occluded, blurred or too small to judge.',
  not_a_student: 'The box is not on one student: teacher, empty seat, two people, or an object.',
}
const KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

export default function LabelPage() {
  const { lesson } = useParams()
  return lesson ? <Labeller lesson={lesson} /> : <TaskSets />
}

function TaskSets() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => { api.labelingSets().then(setData).catch((e) => setError(e.message)) }, [])

  return (
    <div className="p-6 max-w-3xl">
      <h1 className="text-xl font-semibold text-white">Label</h1>
      <p className="text-sm text-slate-400 mt-1">
        Ground truth for measuring the system. You see one student's region over 3 seconds with faces pixelated,
        and never the system's own guess.
      </p>
      {error && <div className="mt-4 text-sm text-red-300">{error}</div>}
      {data?.lessons.length === 0 && (
        <div className="mt-6 text-sm text-slate-400">No task sets. Create one with <code>python -m labeling.tasks</code>.</div>
      )}
      <div className="mt-6 space-y-3">
        {data?.lessons.map((l) => (
          <Link key={l.lesson} to={`/label/${l.lesson}`}
            className="block rounded-lg border border-slate-800 bg-slate-900 p-4 hover:border-slate-600">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-white">{l.lesson.replace(/_/g, ' ')}</span>
              <span className="text-xs text-slate-400">{l.mine} / {l.total} done by you · {l.labellers} labellers</span>
            </div>
            <div className="mt-2 h-1.5 rounded bg-slate-800">
              <div className="h-1.5 rounded" style={{ width: `${(l.mine / l.total) * 100}%`, background: 'var(--viz-series-1)' }} />
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

function Labeller({ lesson }) {
  const [task, setTask] = useState(null)
  const [labels, setLabels] = useState([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.labelingSets().then((d) => setLabels(d.labels)).catch((e) => setError(e.message))
    api.labelingNext(lesson).then(setTask).catch((e) => setError(e.message))
  }, [lesson])

  const submit = useCallback(async (label) => {
    if (!task?.id || saving) return
    setSaving(true)
    try {
      setTask(await api.submitLabel(lesson, task.id, label))
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }, [lesson, task, saving])

  useEffect(() => {
    const onKey = (e) => {
      const i = KEYS.indexOf(e.key)
      if (i >= 0 && labels[i]) submit(labels[i])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [labels, submit])

  return (
    <div className="p-6 max-w-5xl space-y-4">
      <Link to="/label" className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
        <ArrowLeft size={12} /> Task sets
      </Link>
      <div className="flex items-baseline justify-between">
        <h1 className="text-lg font-semibold text-white">{lesson.replace(/_/g, ' ')}</h1>
        {task && <span className="text-sm text-slate-400">{task.done} / {task.total} labelled</span>}
      </div>
      {error && <div className="text-sm text-red-300">{error}</div>}

      {task && !task.id && (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-slate-300 flex items-center gap-2">
          <CheckCircle2 size={18} className="text-green-400" /> All {task.total} tasks labelled. Thank you.
        </div>
      )}

      {task?.id && (
        <>
          <div className="rounded-lg border border-slate-800 bg-black p-2">
            <img src={api.labelingCropUrl(lesson, task.id)} alt="Student region over 3 seconds, faces pixelated"
              className="w-full max-h-[50vh] object-contain" />
          </div>
          <p className="text-xs text-slate-500">
            Judge the student in the cyan box, at the <strong>middle frame</strong> (left and right are ±1 s for context).
          </p>
          <div className="grid sm:grid-cols-2 gap-2">
            {labels.map((label, i) => (
              <button key={label} onClick={() => submit(label)} disabled={saving}
                className="text-left rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 hover:border-blue-500 hover:bg-slate-800 disabled:opacity-50">
                <div className="flex items-center gap-2 text-sm text-slate-100">
                  <kbd className="rounded bg-slate-800 border border-slate-600 px-1.5 text-xs text-slate-300">{KEYS[i]}</kbd>
                  <Tag size={12} className="text-slate-500" /> {label.replace(/_/g, ' ')}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">{GUIDE[label]}</div>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
