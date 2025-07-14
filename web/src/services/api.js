// web/src/services/api.js
import toast from 'react-hot-toast';     // add toast here for 401 handler

// ─────────── Base URL ─────────────────────────────────────────────
// VITE_API_URL should be *just* the API root (e.g. http://127.0.0.1:8000/api/v1)
const API_BASE_URL =
  typeof __API_URL__ !== 'undefined'
    ? __API_URL__.replace(/\/+$/, '')        // trim trailing slash
    : '/api/v1';

// ─────────── Helper to join paths safely ─────────────────────────
const join = (base, path) => {
  const normalBase = base.replace(/\/+$/, '');     // trim trailing /
  const normalPath = path.replace(/^\/+/, '');     // trim leading /
  return `${normalBase}/${normalPath}`;            // single slash in-between
};

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;            // now correct
    this.defaultHeaders = {};  // Remove default Content-Type to avoid CORS preflight for GETs
  }

  async request(endpoint, options = {}) {
    const url = join(this.baseURL, endpoint);
    const token = localStorage.getItem('jwt');

    const cfg = {
      method: options.method || 'GET',
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers
      }
    };

    console.debug(`🔍 [API] ${cfg.method} ${url}`);

    try {
      const res = await fetch(url, cfg);

      /* ── automatic session expiry handling ─────────────── */
      if (res.status === 401) {
        localStorage.removeItem('jwt');
        toast.error('Session expired – please log in again.');
        window.location.replace('/login');
        return;
      }

      if (!res.ok) {
        const text = await res.text();
        console.error(`❌ [API] ${res.status} ${url} – ${text}`);
        throw new Error(`${res.status}: ${text}`);
      }
      return res.status !== 204 ? res.json() : null;
    } catch (err) {
      console.error(`❌ [API] Failed request to ${url}:`, err);
      throw err;
    }
  }

  /* ── Health & Readiness helpers ─────────────────────────────── */
  getHealth() { return this.request('/health'); }
  getReady() { return this.request('/ready/full'); }
  getHello() { return this.request('/hello'); }

  /* ── Authentication ─────────────────────────────────────────── */
  login(credentials) {
    const body = new URLSearchParams({
      username: credentials.username,
      password: credentials.password
    });
    return this.request('/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body
    });
  }

  /* ── ML Prediction helpers ─────────────────────────────────── */
  predictIris(payload) {
    return this.request('/iris/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }

  predictCancer(payload) {
    return this.request('/cancer/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }

  /* ── Training helpers ─────────────────────────────────────── */
  trainIris() { return this.request('/iris/train', { method: 'POST' }); }
  trainCancer() { return this.request('/cancer/train', { method: 'POST' }); }

  /* ── Test helpers ─────────────────────────────────────────── */
  test401() { return this.request('/test/401'); }  // For testing 401 handling
}

export const apiService = new ApiService();
export default apiService;



