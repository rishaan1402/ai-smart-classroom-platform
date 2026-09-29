import { useState } from 'react'
import { foldStates } from '../../lib/states'
import { pct, stateLabel } from '../../lib/format'

/**
 * 100% stacked bar of student states with a legend that also carries the
 * values, so identity never relies on colour alone. 2px surface gaps
 * between segments; per-segment hover tooltip.
 */
export default function StateMixBar({ mix, height = 18 }) {
  const rows = foldStates(mix)
  const [hover, setHover] = useState(null)
  const total = rows.reduce((a, r) => a + r.share, 0) || 1

  return (
    <div className="relative">
      <div className="flex w-full gap-[2px] rounded" style={{ height }} role="img" aria-label="Student state mix">
        {rows.map((r, i) => (
          <div
            key={r.state}
            onMouseEnter={() => setHover(r)}
            onMouseLeave={() => setHover(null)}
            style={{
              width: `${(r.share / total) * 100}%`,
              background: r.color,
              borderRadius: `${i === 0 ? 4 : 0}px ${i === rows.length - 1 ? 4 : 0}px ${i === rows.length - 1 ? 4 : 0}px ${i === 0 ? 4 : 0}px`,
              minWidth: 2,
            }}
          />
        ))}
      </div>

      {hover && (
        <div className="absolute z-10 mt-1 rounded-md border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-lg">
          <div className="text-slate-100 font-medium">{hover.label}: {pct(hover.share / total, 1)}</div>
          {hover.parts && hover.parts.map(([s, v]) => (
            <div key={s} className="text-slate-400">{stateLabel(s)} {pct(v / total, 1)}</div>
          ))}
        </div>
      )}

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {rows.map((r) => (
          <span key={r.state} className="flex items-center gap-1.5 text-slate-300">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: r.color }} />
            {r.label}
            <span className="text-slate-400">{pct(r.share / total)}</span>
            {r.parts && (
              <span className="text-slate-500">
                ({r.parts.map(([s]) => stateLabel(s)).join(', ')})
              </span>
            )}
          </span>
        ))}
      </div>
    </div>
  )
}
