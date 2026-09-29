import { useEffect, useState } from 'react'
import { Activity, AlertTriangle, CheckCircle, RefreshCw, XCircle } from 'lucide-react'
import { api } from '../api'

function Stat({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className={`text-2xl font-semibold ${color}`}>{value ?? '—'}</div>
      {sub && <div className="text-xs text-slate-600 mt-1">{sub}</div>}
    </div>
  )
}

function StatusIcon({ value }) {
  if (value === 'ok' || value === true) return <CheckCircle size={16} className="text-green-400" />
  if (value === 'degraded' || value === false) return <AlertTriangle size={16} className="text-yellow-400" />
  return <XCircle size={16} className="text-red-400" />
}

const VRAM_COLOR = { low: 'text-green-400', medium: 'text-yellow-400', high: 'text-orange-400', critical: 'text-red-400' }

export default function HealthPage() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      setHealth(await api.health())
    } catch {
      setHealth({ status: 'unreachable' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  const vramColor = VRAM_COLOR[health?.gpu?.vram_pressure] || 'text-slate-400'

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-white">System Health</h1>
        <button
          onClick={load}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {!health ? (
        <div className="text-slate-500 text-sm text-center py-12">Loading…</div>
      ) : (
        <>
          {/* Overall status */}
          <div className={`rounded-xl border p-4 mb-6 flex items-center gap-3 ${
            health.status === 'healthy' ? 'bg-green-950 border-green-800' :
            health.status === 'degraded' ? 'bg-yellow-950 border-yellow-800' :
            'bg-red-950 border-red-800'
          }`}>
            <Activity size={20} className={
              health.status === 'healthy' ? 'text-green-400' :
              health.status === 'degraded' ? 'text-yellow-400' : 'text-red-400'
            } />
            <div>
              <div className="text-sm font-medium text-white capitalize">{health.status}</div>
              <div className="text-xs text-slate-400">Last checked just now</div>
            </div>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            <Stat
              label="Ollama"
              value={health.ollama_ready ? 'Ready' : 'Offline'}
              color={health.ollama_ready ? 'text-green-400' : 'text-red-400'}
            />
            <Stat
              label="VRAM Pressure"
              value={health.gpu?.vram_pressure ?? 'n/a'}
              sub={health.gpu?.used_mb ? `${health.gpu.used_mb} MB used` : undefined}
              color={vramColor}
            />
            <Stat
              label="Event Queue"
              value={health.redis?.xlen_events ?? '—'}
              sub="Redis stream depth"
              color={health.redis?.xlen_events > 10000 ? 'text-yellow-400' : 'text-white'}
            />
            <Stat
              label="Redis Memory"
              value={health.redis?.memory_used_mb ? `${health.redis.memory_used_mb} MB` : '—'}
              sub={health.redis?.memory_max_mb ? `of ${health.redis.memory_max_mb} MB` : undefined}
              color={
                health.redis?.memory_used_mb > (health.redis?.memory_max_mb * 0.9)
                  ? 'text-red-400' : 'text-white'
              }
            />
          </div>

          {/* Camera canary status */}
          {health.cameras && health.cameras.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <h2 className="text-sm font-medium text-slate-300 mb-3">Camera Quality</h2>
              <div className="space-y-2">
                {health.cameras.map(cam => (
                  <div key={cam.id} className="flex items-center justify-between text-sm">
                    <span className="text-slate-400 font-mono text-xs">{cam.id}</span>
                    <div className="flex items-center gap-2">
                      <StatusIcon value={cam.canary} />
                      <span className={`text-xs capitalize ${
                        cam.canary === 'ok' ? 'text-green-400' :
                        cam.canary === 'degraded' ? 'text-yellow-400' : 'text-slate-500'
                      }`}>
                        {cam.canary}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
