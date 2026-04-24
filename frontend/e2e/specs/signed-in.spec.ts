import { expect, test } from '../fixtures/test'
import { ChatPage } from '../pages/ChatPage'
import { ThreadSidebarPage } from '../pages/ThreadSidebarPage'

const SEED = [
  { threadId: 'thread-alpha', title: 'Termination without notice rights', createdAt: 1700000000, updatedAt: 1700000500 },
  { threadId: 'thread-beta', title: 'Constitutional right to a fair trial', createdAt: 1700000100, updatedAt: 1700000400 },
]

/**
 * Signed-in specs rely on a real Clerk test instance + test user. They are
 * skipped automatically when the `CLERK_SECRET_KEY` / `E2E_CLERK_USER_*`
 * env vars are not present — so `npm run test:e2e` stays green locally
 * without secrets and fail-closed on CI once the Actions secrets are set.
 */

test.use({ threadSeed: SEED })

test.describe('Signed-in user', () => {
  test('sees their thread list in the sidebar', async ({ signedInPage }) => {
    const sidebar = new ThreadSidebarPage(signedInPage.page)
    await signedInPage.page.goto('/')
    await sidebar.expectVisible()
    for (const seed of SEED) {
      await expect(sidebar.row(seed.threadId)).toBeVisible()
      await expect(sidebar.row(seed.threadId)).toContainText(seed.title)
    }
    // GET /chat/threads was called with a Bearer token.
    expect(signedInPage.mock.listCalls.length).toBeGreaterThan(0)
    expect(signedInPage.mock.listCalls[0].authorization).toMatch(/^Bearer /)
  })

  test('claims the current anonymous session on sign-in', async ({ signedInPage }) => {
    // The claim effect fires once per sign-in transition.
    await signedInPage.page.goto('/')
    await expect(async () => {
      expect(signedInPage.mock.claimCalls.length).toBeGreaterThan(0)
    }).toPass({ timeout: 5000 })
    expect(signedInPage.mock.claimCalls[0].authorization).toMatch(/^Bearer /)
  })

  test('renames a thread inline and persists via PATCH', async ({ signedInPage }) => {
    const sidebar = new ThreadSidebarPage(signedInPage.page)
    await signedInPage.page.goto('/')
    await sidebar.expectVisible()

    const target = SEED[0]
    await sidebar.startRename(target.threadId)
    await sidebar.fillTitle('Unfair dismissal basics')

    await expect(sidebar.row(target.threadId)).toContainText('Unfair dismissal basics')
    await expect(async () => {
      const call = signedInPage.mock.renameCalls.at(-1)
      expect(call?.threadId).toBe(target.threadId)
      expect(call?.title).toBe('Unfair dismissal basics')
      expect(call?.authorization).toMatch(/^Bearer /)
    }).toPass()
  })

  test('clicking a thread switches the active session', async ({ signedInPage }) => {
    const sidebar = new ThreadSidebarPage(signedInPage.page)
    await signedInPage.page.goto('/')
    await sidebar.expectVisible()

    await sidebar.row('thread-beta').click()
    // The selected row should carry the active styling.
    await expect(sidebar.row('thread-beta')).toHaveClass(/border-border/)
  })

  test('sign-in CTA disappears and UserButton appears when signed in', async ({ signedInPage }) => {
    const chat = new ChatPage(signedInPage.page)
    await chat.goto()
    await expect(chat.signInButton).toHaveCount(0)
    // Clerk's UserButton renders an avatar trigger — its a11y name is "Open user button"
    await expect(signedInPage.page.getByRole('button', { name: /open user button|manage account/i })).toBeVisible()
  })
})
