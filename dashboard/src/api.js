/**
 * Thin API client — all requests go through /api proxy → FastAPI.
 * Cookies are sent automatically (credentials: 'include').
 */

const BASE = '/api'

async function request(method, path, body) {
  const opts = {
    method,
    credentials: 'include',
    headers: {},
  }
  if (body) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  }
  const res = await fetch(BASE + path, opts)
  if (res.status === 401) {
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  // Auth
  login: (email, password) => request('POST', '/auth/login', { email, password }),
  register: (email, password) => request('POST', '/auth/register', { email, password }),
  logout: () => request('POST', '/auth/logout'),
  me: () => request('GET', '/auth/me'),
  wsToken: () => request('GET', '/auth/ws-token'),

  // Events
  events: (params = {}) => {
    const q = new URLSearchParams(params).toString()
    return request('GET', `/events${q ? '?' + q : ''}`)
  },
  event: (id) => request('GET', `/events/${id}`),

  // Cameras
  cameras: () => request('GET', '/cameras'),
  createCamera: (data) => request('POST', '/cameras', data),
  updateCamera: (id, data) => request('PUT', `/cameras/${id}`, data),
  deleteCamera: (id) => request('DELETE', `/cameras/${id}`),
  testCamera: async (id) => {
    const res = await fetch(`${BASE}/cameras/${id}/test`, { credentials: 'include' })
    if (!res.ok) return null
    const blob = await res.blob()
    return URL.createObjectURL(blob)
  },

  // Health
  health: () => request('GET', '/health'),

  // Classroom analytics
  lessons: () => request('GET', '/lessons'),
  lesson: (id) => request('GET', `/lessons/${encodeURIComponent(id)}`),
  seats: (id) => request('GET', `/lessons/${encodeURIComponent(id)}/seats`),
  previewUrl: (id) => `${BASE}/lessons/${encodeURIComponent(id)}/preview`,
  student: (id) => request('GET', `/students/${encodeURIComponent(id)}`),
}
