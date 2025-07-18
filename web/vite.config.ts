import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Centralizes build-time API URL logic; runtime JSON removed.
export default defineConfig(({ mode }) => {
  const envDir = __dirname
  const env = loadEnv(mode, envDir, '') // loads web/.env

  // Source of truth is now web/.env generated from config.yaml.
  const raw = env.VITE_API_URL || ''

  // Normalize: ensure single /api/v1 suffix
  const trimmed = raw.replace(/\/+$/, '')
  const API_URL = /\/api\/v1$/.test(trimmed) ? trimmed : `${trimmed}/api/v1`

  if (!API_URL) {
    if (mode === 'development') {
      // Dev-friendly fallback – loud warning (should rarely trigger)
      console.warn('[vite.config] VITE_API_URL missing – using http://127.0.0.1:8000/api/v1')
    } else {
      throw new Error('[vite.config] VITE_API_URL is required for non-dev builds')
    }
  }

  console.log('🔍 Vite Config:')
  console.log('  Mode              :', mode)
  console.log('  Loaded from       :', path.join(envDir, '.env'))
  console.log('  VITE_API_URL (raw):', raw)
  console.log('  API_URL (final)   :', API_URL)

  return {
    envDir,                // ← this is what tells Vite where to look
    plugins: [react()],
    define: {
      __BUILD_API_URL__: JSON.stringify(API_URL),      // one canonical build-time constant
    },
          server: {
        host: '0.0.0.0',
        port: 5173,
        proxy: {
          '/api/v1': {
            target: 'http://127.0.0.1:8000',
            changeOrigin: true,
            secure: false,
            rewrite: p => p
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






