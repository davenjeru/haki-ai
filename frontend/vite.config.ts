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
// Env resolution order (later wins):
//   1. `../.env`              — repo-root secrets shared with the backend
//                               (e.g. VITE_CLERK_PUBLISHABLE_KEY lives here
//                               in dev so the Python server and the Vite
//                               bundle read it from one canonical file).
//   2. `./.env[.local|.<mode>|.<mode>.local]` — frontend-only local
//                               overrides (LOCAL_API_URL,
//                               VITE_API_BASE_URL).
//   3. `process.env.VITE_*`   — CI/CD-supplied values (GitHub Actions sets
//                               the publishable key this way at `npm run
//                               build` time so it's inlined into the
//                               production bundle).
// Non-VITE_* values remain server-side only per Vite's security model; VITE_*
// values are injected into the client bundle via `define`.

export default defineConfig(({ mode }) => {
  const rootEnv = loadEnv(mode, path.resolve(__dirname, '..'), '')
  const localEnv = loadEnv(mode, __dirname, '')
  const env: Record<string, string> = { ...rootEnv, ...localEnv }

  const localApiUrl = env.LOCAL_API_URL

  // Collect VITE_* keys from every source and inject them via `define` so
  // the bundle picks them up no matter where they originate. `process.env`
  // wins over `../.env`, matching the conventional precedence (explicit
  // runtime override beats committed defaults).
  const viteKeys = new Set<string>([
    ...Object.keys(rootEnv).filter((k) => k.startsWith('VITE_')),
    ...Object.keys(process.env).filter((k) => k.startsWith('VITE_')),
  ])
  const viteDefines = Object.fromEntries(
    [...viteKeys].map((k) => {
      const value = process.env[k] ?? rootEnv[k] ?? ''
      return [`import.meta.env.${k}`, JSON.stringify(value)]
    }),
  )

  return {
    plugins: [react(), tailwindcss()],
    define: viteDefines,
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
