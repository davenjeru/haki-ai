import { defineConfig, loadEnv } from 'vite'
import path from 'node:path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// LOCAL_API_URL is a Node-only variable (no VITE_ prefix) — never sent to the browser.
// Set it in frontend/.env.local to the LocalStack API Gateway URL.
// Example: LOCAL_API_URL=http://localhost:4566/restapis/<api-id>/local/_user_request_
//
// When LOCAL_API_URL is set, the dev server proxies /chat → LocalStack.
// In production there is no proxy — the browser calls VITE_API_BASE_URL directly.
//
// Env is loaded from two places and merged:
//   1. `../.env`        — repo-root secrets shared with the backend
//                         (e.g. VITE_CLERK_PUBLISHABLE_KEY lives here so the
//                         Python server and the Vite bundle read it from one
//                         canonical file).
//   2. `./.env.local`   — frontend-only local overrides (LOCAL_API_URL,
//                         VITE_API_BASE_URL). Wins on conflict.
// Non-VITE_* values remain server-side only per Vite's security model; VITE_*
// values are injected into the client bundle via `define`.

export default defineConfig(({ mode }) => {
  const rootEnv = loadEnv(mode, path.resolve(__dirname, '..'), '')
  const localEnv = loadEnv(mode, __dirname, '')
  const env: Record<string, string> = { ...rootEnv, ...localEnv }

  const localApiUrl = env.LOCAL_API_URL

  // Inject VITE_* keys from root into the browser bundle. Vite's default
  // bundle-injection only reads from `envDir`, so anything unique to the
  // root `.env` needs an explicit `define` entry to reach client code.
  const viteKeysFromRoot = Object.entries(rootEnv).filter(([k]) => k.startsWith('VITE_'))
  const defineRootVites = Object.fromEntries(
    viteKeysFromRoot.map(([k, v]) => [`import.meta.env.${k}`, JSON.stringify(v)]),
  )

  return {
    plugins: [react(), tailwindcss()],
    define: defineRootVites,
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
