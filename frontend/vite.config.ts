import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `pnpm dev` proxies the engine (websocket) and Pangram API to the Python server.
const backend = process.env.COWRITE_BACKEND ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ws': { target: backend.replace(/^http/, 'ws'), ws: true },
      '/api': backend,
    },
  },
  build: { outDir: '../steer/web/dist', emptyOutDir: true },
})
