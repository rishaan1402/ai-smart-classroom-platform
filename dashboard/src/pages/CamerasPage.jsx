import { useEffect, useState } from 'react'
import { Camera, CheckCircle, Loader, Plus, Radio, Trash2, XCircle, ZoomIn } from 'lucide-react'
import { api } from '../api'
import { useAuth } from '../AuthContext'

function CameraCard({ camera, onDelete, onTest, isAdmin }) {
  const [thumb, setThumb] = useState(null)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null) // 'ok' | 'fail'

  const runTest = async () => {
    setTesting(true)
    setThumb(null)
    setTestResult(null)
    try {
      const url = await onTest(camera.id)
      if (url) {
        setThumb(url)
        setTestResult('ok')
      } else {
        setTestResult('fail')
      }
    } catch {
      setTestResult('fail')
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
            camera.enabled ? 'bg-blue-900/50' : 'bg-slate-800'
          }`}>
            <Camera size={16} className={camera.enabled ? 'text-blue-400' : 'text-slate-500'} />
          </div>
          <div>
            <div className="text-sm font-medium text-white">{camera.name || 'Unnamed'}</div>
            <div className="text-xs text-slate-500 font-mono mt-0.5">
              {camera.host}{camera.path}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            camera.enabled ? 'bg-green-900/50 text-green-400' : 'bg-slate-800 text-slate-500'
          }`}>
            {camera.enabled ? 'enabled' : 'disabled'}
          </span>
          {isAdmin && (
            <button
              onClick={() => onDelete(camera.id)}
              className="text-slate-600 hover:text-red-400 transition-colors"
            >
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Live test */}
      <div className="mt-3 flex items-center gap-2">
        <button
          onClick={runTest}
          disabled={testing}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50"
        >
          {testing ? <Loader size={12} className="animate-spin" /> : <ZoomIn size={12} />}
          Test feed
        </button>
        {testResult === 'ok' && <CheckCircle size={14} className="text-green-400" />}
        {testResult === 'fail' && <span className="text-xs text-red-400">Cannot connect</span>}
      </div>

      {thumb && (
        <div className="mt-3 rounded-lg overflow-hidden border border-slate-700">
          <img src={thumb} alt="Camera test frame" className="w-full" />
        </div>
      )}
    </div>
  )
}

function AddCameraModal({ onClose, onSave }) {
  const [form, setForm] = useState({ name: '', host: '', path: '/stream1', enabled: true })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const save = async (e) => {
    e.preventDefault()
    setSaving(true)
    setError('')
    try {
      await onSave(form)
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-md p-6">
        <h2 className="text-lg font-medium text-white mb-4">Add Camera</h2>
        <form onSubmit={save} className="space-y-3">
          {[
            { label: 'Name', key: 'name', placeholder: 'Front Door' },
            { label: 'Host:Port', key: 'host', placeholder: '192.168.1.100:554' },
            { label: 'Path', key: 'path', placeholder: '/stream1' },
          ].map(({ label, key, placeholder }) => (
            <div key={key}>
              <label className="block text-xs text-slate-400 mb-1">{label}</label>
              <input
                value={form[key]}
                onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                required={key !== 'name'}
                placeholder={placeholder}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          ))}
          {error && <div className="text-red-400 text-sm">{error}</div>}
          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2 rounded-lg bg-slate-800 text-slate-300 text-sm hover:bg-slate-700"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Add Camera'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function OnvifDiscoverModal({ onClose, onImport }) {
  const [discovering, setDiscovering] = useState(false)
  const [results, setResults] = useState([])
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const discover = async () => {
    setDiscovering(true)
    setResults([])
    try {
      const res = await fetch('/api/cameras/onvif/discover', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ timeout: 5, username, password }),
      })
      const data = await res.json()
      setResults(data.cameras || [])
    } catch (e) {
      setResults([])
    } finally {
      setDiscovering(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-lg p-6">
        <h2 className="text-lg font-medium text-white mb-4">ONVIF Discovery</h2>
        <div className="flex gap-3 mb-4">
          <input value={username} onChange={e => setUsername(e.target.value)} placeholder="Username"
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none" />
          <input value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" type="password"
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none" />
          <button onClick={discover} disabled={discovering}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-lg disabled:opacity-50">
            {discovering ? <Loader size={14} className="animate-spin" /> : 'Scan'}
          </button>
        </div>
        {discovering && <div className="text-sm text-slate-400 text-center py-4">Probing network (5s)…</div>}
        {results.length > 0 && (
          <div className="space-y-2 max-h-60 overflow-y-auto mb-4">
            {results.map((cam, i) => (
              <div key={i} className="flex items-center justify-between bg-slate-800 rounded-lg px-3 py-2">
                <div>
                  <div className="text-xs font-mono text-slate-300">{cam.host}</div>
                  <div className="text-xs text-slate-500">{cam.rtsp_url || 'No stream URI'}</div>
                </div>
                {cam.reachable && (
                  <button onClick={() => onImport(cam)} className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded">
                    Import
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
        {!discovering && results.length === 0 && (
          <div className="text-sm text-slate-500 text-center py-3">No devices found yet. Click Scan.</div>
        )}
        <button onClick={onClose} className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm rounded-lg">
          Close
        </button>
      </div>
    </div>
  )
}

export default function CamerasPage() {
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [showAdd, setShowAdd] = useState(false)
  const [showOnvif, setShowOnvif] = useState(false)
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'

  const load = async () => {
    setLoading(true)
    try {
      setCameras(await api.cameras())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleDelete = async (id) => {
    if (!confirm('Delete this camera?')) return
    await api.deleteCamera(id)
    load()
  }

  const handleSave = async (data) => {
    await api.createCamera(data)
    load()
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-white">Cameras</h1>
        {isAdmin && (
          <div className="flex gap-2">
            <button
              onClick={() => setShowOnvif(true)}
              className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm rounded-lg transition-colors"
            >
              <Radio size={14} />
              Discover
            </button>
            <button
              onClick={() => setShowAdd(true)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-lg transition-colors"
            >
              <Plus size={16} />
              Add Camera
            </button>
          </div>
        )}
      </div>

      {loading ? (
        <div className="text-slate-500 text-sm text-center py-12">Loading…</div>
      ) : cameras.length === 0 ? (
        <div className="text-slate-500 text-sm text-center py-12">
          <Camera size={32} className="mx-auto mb-3 opacity-30" />
          No cameras configured.{isAdmin ? ' Add one above.' : ''}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {cameras.map(cam => (
            <CameraCard
              key={cam.id}
              camera={cam}
              isAdmin={isAdmin}
              onDelete={handleDelete}
              onTest={api.testCamera}
            />
          ))}
        </div>
      )}

      {showAdd && <AddCameraModal onClose={() => setShowAdd(false)} onSave={handleSave} />}
      {showOnvif && (
        <OnvifDiscoverModal
          onClose={() => setShowOnvif(false)}
          onImport={async (cam) => {
            await api.createCamera({ name: cam.host, host: cam.host, path: cam.path || '/stream1' })
            setShowOnvif(false)
            load()
          }}
        />
      )}
    </div>
  )
}
