// web/src/services/api.js
import toast from 'react-hot-toast';     // add toast here for 401 handler

// ─────────── Base URL ─────────────────────────────────────────────
// VITE_API_URL should be *just* the API root (e.g. http://127.0.0.1:8000/api/v1)
const API_BASE_URL = (() => {
  // ① grab compile-time value if present
  const raw = (typeof __API_URL__ !== 'undefined' && __API_URL__) || '';

  // 🔍 DEBUG: Log the raw value
  console.warn('[ApiService] __API_URL__ raw value:', raw);

  // ② keep only valid absolute URLs, trim trailing slash
  const baseURL = (/^https?:\/\//.test(raw) ? raw : '').replace(/\/+$/, '');

  // 🔍 DEBUG: Log the processed baseURL
  console.warn('[ApiService] processed baseURL:', baseURL);

  // ③ HARD-FALLBACK for dev bundles that missed the define-plugin
  if (!baseURL) {
    console.warn('[ApiService] __API_URL__ missing – falling back to /api/v1');
    return '/api/v1';
  }
  return baseURL;
})();

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
    // If endpoint already contains /api/v1 we don't want it twice
    const cleanEndpoint = endpoint.replace(/^\/?api\/v1\//, '');
    const url = join(this.baseURL, cleanEndpoint);
    const token = localStorage.getItem('jwt');

    // 🔍 DEBUG: Log URL construction details
    console.warn('[ApiService] request debug:', {
      baseURL: this.baseURL,
      endpoint: endpoint,
      cleanEndpoint: cleanEndpoint,
      finalURL: url,
      method: options.method || 'GET'
    });

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

      /* ── Rate limiting header handling ───────────────────────── */
      const rateLimitRemaining = res.headers.get('X-RateLimit-Remaining');
      const rateLimitLimit = res.headers.get('X-RateLimit-Limit');
      const retryAfter = res.headers.get('Retry-After');

      if (rateLimitRemaining !== null) {
        const remaining = parseInt(rateLimitRemaining);
        const limit = parseInt(rateLimitLimit);

        // Show warning when rate limit is getting low
        if (remaining <= 3 && remaining > 0) {
          toast.warning(`Rate limit warning: ${remaining}/${limit} requests remaining`);
        }

        // Log rate limit info for debugging
        console.debug(`🔍 [API] Rate limit: ${remaining}/${limit} remaining`);
      }

      /* ── automatic session expiry handling ─────────────── */
      if (res.status === 401) {
        localStorage.removeItem('jwt');
        toast.error('Session expired – please log in again.');
        window.location.replace('/login');
        return;
      }

      /* ── Rate limit exceeded handling ─────────────────────── */
      if (res.status === 429) {
        const retrySeconds = retryAfter ? parseInt(retryAfter) : 60;
        toast.error(`Rate limit exceeded. Please wait ${retrySeconds} seconds before trying again.`);
        throw new Error(`Rate limit exceeded. Retry after ${retrySeconds} seconds.`);
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






