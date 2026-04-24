import { test as base, expect, type Page } from '@playwright/test'
import { installMockBackend, type MockBackendState, type MockThread } from './mockBackend'

/**
 * Custom test fixtures.
 *
 * `anonymousPage` — a fresh page with the threads API mocked and no Clerk
 * sign-in attempted. All "does it work logged-out?" specs use this.
 *
 * `signedInPage` — installs Clerk's testing bypass token and signs in via
 * `@clerk/testing/playwright`. Skips the test when Clerk test credentials
 * are not available so the suite still passes in environments without
 * secrets (e.g. fork PRs, first-time clones).
 */

interface Fixtures {
  anonymousPage: { page: Page; mock: MockBackendState }
  signedInPage: { page: Page; mock: MockBackendState }
  threadSeed: MockThread[]
}

export const CLERK_ENV_VARS = [
  'CLERK_PUBLISHABLE_KEY',
  'CLERK_SECRET_KEY',
  'E2E_CLERK_USER_USERNAME',
  'E2E_CLERK_USER_PASSWORD',
] as const

export function clerkCredentialsAvailable(): boolean {
  return CLERK_ENV_VARS.every((k) => Boolean(process.env[k] && process.env[k]!.length > 0))
}

export const test = base.extend<Fixtures>({
  threadSeed: [[], { option: true }],

  anonymousPage: async ({ page, threadSeed }, use) => {
    const mock = await installMockBackend(page, threadSeed)
    await use({ page, mock })
  },

  signedInPage: async ({ page, threadSeed }, use, testInfo) => {
    if (!clerkCredentialsAvailable()) {
      testInfo.skip(true, 'Clerk test credentials not available; skipping signed-in spec.')
      return
    }
    const mock = await installMockBackend(page, threadSeed)
    const { setupClerkTestingToken, clerk } = await import('@clerk/testing/playwright')
    await setupClerkTestingToken({ page })
    await page.goto('/')
    await clerk.signIn({
      page,
      signInParams: {
        strategy: 'password',
        identifier: process.env.E2E_CLERK_USER_USERNAME!,
        password: process.env.E2E_CLERK_USER_PASSWORD!,
      },
    })
    await use({ page, mock })
  },
})

export { expect }
