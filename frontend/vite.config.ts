import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// LOCAL_API_URL is a Node-only variable (no VITE_ prefix) — never sent to the browser.
// Set it in frontend/.env.local to the LocalStack API Gateway URL.
// Example: LOCAL_API_URL=http://localhost:4566/restapis/<api-id>/local/_user_request_
//
// When LOCAL_API_URL is set, the dev server proxies /chat → LocalStack.
// In production there is no proxy — the browser calls VITE_API_BASE_URL directly.

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const localApiUrl = env.LOCAL_API_URL

  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: localApiUrl
        ? {
            '/chat': {
              target: localApiUrl,
              changeOrigin: true,
            },
          }
        : undefined,
    },
  }
})
