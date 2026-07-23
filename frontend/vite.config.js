import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // Local dev mirrors production: the agent function runs on 8010 and static
  // findings are served straight out of public/data.
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8010', changeOrigin: true },
    },
  },
  plugins: [react()],
})
