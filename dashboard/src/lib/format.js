export const pct = (v, digits = 0) => (v == null ? '—' : `${(v * 100).toFixed(digits)}%`)

export function clock(seconds) {
  if (seconds == null) return '—'
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  return `${m}:${String(s % 60).padStart(2, '0')}`
}

export const minutes = (seconds) => (seconds == null ? '—' : `${Math.round(seconds / 60)} min`)

export const stateLabel = (s) => (s ? s.replace(/_/g, ' ') : '—')
