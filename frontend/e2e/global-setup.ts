import type { FullConfig } from '@playwright/test'

/**
 * Global setup hook.
 *
 * Wires `@clerk/testing`'s bypass tokens when the Clerk secret key is
 * available (signed-in specs rely on it). Skipped otherwise so the
 * anonymous-only specs keep running without any Clerk credentials — useful
 * for first-time local runs and for PRs from contributors without secrets.
 */
export default async function globalSetup(_config: FullConfig): Promise<void> {
  if (!process.env.CLERK_SECRET_KEY) {
    return
  }
  const { clerkSetup } = await import('@clerk/testing/playwright')
  await clerkSetup()
}
