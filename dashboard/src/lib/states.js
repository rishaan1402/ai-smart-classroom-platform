/**
 * Student states → chart slots. Fixed order, never cycled: the same state
 * always has the same colour on every chart. Less frequent states fold into
 * "other" (their breakdown is shown in the tooltip and legend text).
 */
export const STATE_SLOTS = [
  { state: 'attentive', label: 'Attentive', color: 'var(--viz-series-1)', engaged: true },
  { state: 'writing', label: 'Writing', color: 'var(--viz-series-2)', engaged: true },
  { state: 'hand_raised', label: 'Hand raised', color: 'var(--viz-series-3)', engaged: true },
  { state: 'head_down', label: 'Head down', color: 'var(--viz-series-4)', engaged: false },
  { state: 'turned_away', label: 'Turned away', color: 'var(--viz-series-5)', engaged: false },
]
export const OTHER_SLOT = { state: 'other', label: 'Other', color: 'var(--viz-other)', engaged: false }

/** [{state, share}] or {state: share} → ordered slots with folded "other". */
export function foldStates(mix) {
  const entries = Array.isArray(mix) ? mix.map((m) => [m.state, m.share]) : Object.entries(mix || {})
  const byState = Object.fromEntries(entries)
  const rows = STATE_SLOTS.map((s) => ({ ...s, share: byState[s.state] || 0 }))
  const known = new Set(STATE_SLOTS.map((s) => s.state))
  const rest = entries.filter(([s]) => !known.has(s) && s !== 'unknown')
  const otherShare = rest.reduce((a, [, v]) => a + v, 0)
  if (otherShare > 0) rows.push({ ...OTHER_SLOT, share: otherShare, parts: rest })
  return rows.filter((r) => r.share > 0)
}
