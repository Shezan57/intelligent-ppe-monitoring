import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// ── Backend URL ───────────────────────────────────────────────────
// Local dev:  http://localhost:8000
// Colab:      your ngrok URL from Cell 10
const BACKEND_URL = 'https://chalkiest-shavonda-swimmingly.ngrok-free.dev'
// ─────────────────────────────────────────────────────────────────

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            '/api': {
                target: BACKEND_URL,
                changeOrigin: true,
                // Required: bypass ngrok's browser warning interstitial page
                headers: {
                    'ngrok-skip-browser-warning': 'true',
                },
            },
            '/uploads': {
                target: BACKEND_URL,
                changeOrigin: true,
                headers: {
                    'ngrok-skip-browser-warning': 'true',
                },
            },
        },
    },
})

