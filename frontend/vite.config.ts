import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const api = process.env.AURA_API_URL ?? 'http://localhost:8000'

// The dev server proxies /api (REST + WebSocket) to the backend, so the session
// cookie is same-origin in development exactly as it is behind nginx in production.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { '/api': { target: api, changeOrigin: true, ws: true } },
  },
})
