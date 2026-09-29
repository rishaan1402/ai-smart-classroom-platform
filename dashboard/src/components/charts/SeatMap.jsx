import { useState } from 'react'
import { useWidth } from '../../lib/useWidth'
import { pct, stateLabel } from '../../lib/format'

const STEPS = ['var(--viz-seq-0)', 'var(--viz-seq-1)', 'var(--viz-seq-2)', 'var(--viz-seq-3)', 'var(--viz-seq-4)']
const stepFor = (v) => STEPS[Math.min(STEPS.length - 1, Math.floor(v * STEPS.length))]

/**
 * Seats at their learned camera-view positions, filled by engaged share
 * (one-hue sequential ramp, 5 bins). Seats with no data are outlined only.
 * Identified students (consented, role-permitted) are labelled and clickable.
 */
export default function SeatMap({ seats, aspect = 16 / 9, onSelectStudent }) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState(null)
  const height = width / aspect
  // Each seat's learned radius is half the gap to its nearest neighbour, so
  // circles drawn inside it never overlap, even in tightly packed back rows
  const radiusOf = (s) => Math.max(5, Math.min(16, (s.radius ?? 0.02) * width * 0.8))

  return (
    <div ref={ref} className="relative">
      <svg width={width} height={height} role="img" aria-label="Engagement by seat"
        className="rounded-lg bg-slate-950/60 border border-slate-800">
        {seats.map((s) => {
          const cx = s.x * width, cy = s.y * height
          const r = radiusOf(s)
          const clickable = s.student_id && onSelectStudent
          return (
            <g key={s.seat_id}
              onMouseEnter={() => setHover(s)}
              onMouseLeave={() => setHover(null)}
              onClick={() => clickable && onSelectStudent(s.student_id)}
              className={clickable ? 'cursor-pointer' : ''}
            >
              <circle cx={cx} cy={cy} r={r + 6} fill="transparent" />
              <circle cx={cx} cy={cy} r={r}
                fill={s.engaged_share == null ? 'transparent' : stepFor(s.engaged_share)}
                stroke={s.engaged_share == null ? 'var(--viz-axis)' : 'var(--viz-surface)'}
                strokeWidth="2" strokeDasharray={s.engaged_share == null ? '3 3' : undefined} />
              {s.student_id && (
                <text x={cx} y={cy + r + 11} fontSize="10" textAnchor="middle" fill="var(--viz-text-primary)">
                  {s.student_id}
                </text>
              )}
            </g>
          )
        })}
      </svg>

      {hover && (
        <div className="absolute pointer-events-none z-10 rounded-md border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-lg"
          style={{ left: Math.min(hover.x * width + 16, width - 190), top: Math.max(0, hover.y * height - 30), width: 180 }}>
          <div className="text-slate-100 font-medium">{hover.student_id || `Seat ${hover.seat_id}`}</div>
          {hover.student_id && <div className="text-slate-400">Seat {hover.seat_id}</div>}
          <div className="text-slate-300">{hover.engaged_share == null ? 'No data' : `${pct(hover.engaged_share)} engaged`}</div>
          {hover.dominant_state && <div className="text-slate-400">Mostly {stateLabel(hover.dominant_state)}</div>}
          {hover.student_id && onSelectStudent && <div className="mt-1 text-slate-500">Click for student view</div>}
        </div>
      )}

      <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
        <span>Engaged share</span>
        {STEPS.map((c, i) => (
          <span key={i} className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: c }} />
            {i * 20}–{(i + 1) * 20}%
          </span>
        ))}
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-4 rounded-sm border border-dashed border-slate-500" />
          no data
        </span>
      </div>
    </div>
  )
}
