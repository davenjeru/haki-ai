import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * Playwright config for Haki AI end-to-end tests.
 *
 * Runs against the Vite dev server in mock mode (`VITE_USE_MOCK=true` →
 * `chatClient.ts` short-circuits to its built-in mock) so specs stay
 * hermetic — no AWS, no Bedrock, no DynamoDB. Signed-in specs additionally
 * stub `/chat/threads*` via `page.route()` so the thread sidebar exercises
 * the real component logic against canned responses.
 */
export default defineConfig({
  testDir: path.join(__dirname, 'specs'),
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? [['html', { outputFolder: 'playwright-report' }], ['github']] : [['list']],
  outputDir: path.join(__dirname, 'test-results'),
  timeout: 30_000,
  expect: { timeout: 5_000 },
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: {
    command: 'npm run dev -- --port 5173 --strictPort',
    cwd: path.join(__dirname, '..'),
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    env: {
      // Keep chat turns mocked (chatClient.ts short-circuits to mockSendMessage)
      // while still exercising the real threads API fetch paths against the
      // Playwright-installed page.route() handlers.
      VITE_USE_MOCK: 'true',
      VITE_API_BASE_URL: '',
    },
  },
  globalSetup: path.join(__dirname, 'global-setup.ts'),
})
