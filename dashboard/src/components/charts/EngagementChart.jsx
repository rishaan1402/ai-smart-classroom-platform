import { useMemo, useState } from 'react'
import { useWidth } from '../../lib/useWidth'
import { clock, pct } from '../../lib/format'

const M = { top: 22, right: 12, bottom: 24, left: 40 }

/**
 * Class engagement over the lesson (single series: share of students engaged).
 * Topic segments are neutral background bands; hover shows a crosshair +
 * tooltip; click seeks the lesson video. `current` draws the playhead.
 */
export default function EngagementChart({ points, topics = [], current = null, onSeek, height = 220 }) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState(null)

  const tMax = useMemo(() => {
    const last = points.length ? points[points.length - 1].t : 0
    const topicEnd = topics.length ? topics[topics.length - 1].end_s : 0
    return Math.max(1, last, topicEnd)
  }, [points, topics])

  const w = width - M.left - M.right
  const h = height - M.top - M.bottom
  const x = (t) => M.left + (t / tMax) * w
  const y = (v) => M.top + (1 - v) * h

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.engaged).toFixed(1)}`).join('')
  const area = points.length
    ? `${line}L${x(points[points.length - 1].t).toFixed(1)},${y(0)}L${x(points[0].t).toFixed(1)},${y(0)}Z`
    : ''

  const tickStep = niceStep(tMax / 60) * 60
  const ticks = []
  for (let t = 0; t <= tMax; t += tickStep) ticks.push(t)

  const nearest = (t) => {
    if (!points.length) return null
    let lo = 0, hi = points.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (points[mid].t < t) lo = mid + 1
      else hi = mid
    }
    const a = points[Math.max(0, lo - 1)], b = points[lo]
    return Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b
  }

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const t = ((e.clientX - rect.left - M.left) / w) * tMax
    if (t < 0 || t > tMax) return setHover(null)
    setHover({ p: nearest(t), t })
  }

  const topicAt = (t) => topics.find((tp) => t >= tp.start_s && t < tp.end_s)
  const hp = hover?.p
  const hoverTopic = hover ? topicAt(hp ? hp.t : hover.t) : null

  return (
    <div ref={ref} className="relative select-none">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label="Share of students engaged over the lesson"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={() => hover && onSeek?.(hp ? hp.t : hover.t)}
        className={onSeek ? 'cursor-pointer' : ''}
      >
        {topics.map((tp, i) => (
          <g key={i}>
            <rect x={x(tp.start_s)} y={M.top} width={Math.max(0, x(tp.end_s) - x(tp.start_s))} height={h}
              fill={i % 2 ? 'var(--viz-band-b)' : 'var(--viz-band-a)'} />
            {x(tp.end_s) - x(tp.start_s) > 70 && (
              <text x={x(tp.start_s) + 4} y={M.top - 7} fontSize="10" fill="var(--viz-text-secondary)">
                {truncate(tp.title, Math.floor((x(tp.end_s) - x(tp.start_s)) / 6))}
              </text>
            )}
          </g>
        ))}

        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={M.left} x2={M.left + w} y1={y(v)} y2={y(v)} stroke="var(--viz-grid)" />
            <text x={M.left - 6} y={y(v) + 3} fontSize="10" textAnchor="end" fill="var(--viz-text-secondary)">
              {pct(v)}
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={height - 6} fontSize="10" textAnchor="middle" fill="var(--viz-text-secondary)">
            {clock(t)}
          </text>
        ))}

        <path d={area} fill="var(--viz-series-1)" opacity="0.18" />
        <path d={line} fill="none" stroke="var(--viz-series-1)" strokeWidth="2" strokeLinejoin="round" />

        {current != null && (
          <line x1={x(current)} x2={x(current)} y1={M.top} y2={M.top + h} stroke="var(--viz-text-primary)"
            strokeWidth="1.5" opacity="0.8" />
        )}
        {hp && (
          <g pointerEvents="none">
            <line x1={x(hp.t)} x2={x(hp.t)} y1={M.top} y2={M.top + h} stroke="var(--viz-axis)" strokeDasharray="3 3" />
            <circle cx={x(hp.t)} cy={y(hp.engaged)} r="4.5" fill="var(--viz-series-1)"
              stroke="var(--viz-surface)" strokeWidth="2" />
          </g>
        )}
      </svg>

      {hover && (hp || hoverTopic) && (
        <div
          className="absolute pointer-events-none z-10 rounded-md border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-lg"
          style={{ left: Math.min(x(hp ? hp.t : hover.t) + 12, width - 190), top: 8, width: 180 }}
        >
          <div className="text-slate-400">{clock(hp ? hp.t : hover.t)}</div>
          {hp && (
            <>
              <div className="text-slate-100 font-medium">{pct(hp.engaged)} engaged</div>
              <div className="text-slate-400">{hp.students} students seen</div>
            </>
          )}
          {hoverTopic && <div className="mt-1 text-slate-300">{hoverTopic.title}</div>}
          {onSeek && <div className="mt-1 text-slate-500">Click to jump the video here</div>}
        </div>
      )}
    </div>
  )
}

function niceStep(minutes) {
  const target = minutes / 6
  for (const s of [0.25, 0.5, 1, 2, 5, 10, 15, 30]) if (s >= target) return s
  return 60
}

const truncate = (s, n) => (s.length > n ? s.slice(0, Math.max(1, n - 1)) + '…' : s)
