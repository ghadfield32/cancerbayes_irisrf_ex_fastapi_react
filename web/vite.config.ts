import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const raw = env.VITE_API_URL || 
              (mode === 'production'
                ? 'https://fastapi-production-1d13.up.railway.app/api/v1'
                : 'http://127.0.0.1:8000/api/v1')

  /** Prepend https:// if the scheme is missing */
  const API_URL = /^https?:\/\//.test(raw) ? raw.replace(/\/+$/, '') : `https://${raw}`

  // Final sanity-check – bail if still invalid
  if (!/^https?:\/\/[^/]+/.test(API_URL)) {
    throw new Error(`VITE_API_URL must be an absolute URL – got "${API_URL}"`);
  }

  console.log('🔍 Vite Config Debug:')
  console.log('Mode:', mode)
  console.log('Raw API_URL:', raw)
  console.log('Normalized API_URL:', API_URL)
  console.log('All VITE_ env vars:', Object.keys(env).filter(key => key.startsWith('VITE_')))

  return {
    plugins: [react()],
    /** Inject __API_URL__ constant */
    define: {
      __API_URL__: JSON.stringify(API_URL),
      'import.meta.env.VITE_API_URL': JSON.stringify(API_URL)
    },
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api/v1': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path, // Keep the /api/v1 prefix
          configure: (proxy, _options) => {
            proxy.on('error', (err, _req, _res) => {
              console.log('🔍 Proxy Error:', err)
            })
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              console.log('🔍 Proxy Request:', req.method, req.url, '-> ', proxyReq.path)
            })
            proxy.on('proxyRes', (proxyRes, req, _res) => {
              console.log('🔍 Proxy Response:', proxyRes.statusCode, req.url)
            })
          }
        }
      }
    },
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: false,
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom']
          }
        }
      }
    },
    esbuild: {
      logOverride: { 
        'this-is-undefined-in-esm': 'silent'
      },
      target: 'es2020',
      keepNames: true
    }
  }
})

