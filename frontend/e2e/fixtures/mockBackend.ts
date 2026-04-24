import type { Page, Route } from '@playwright/test'

export interface MockThread {
  threadId: string
  title: string
  createdAt: number
  updatedAt: number
}

export interface MockBackendState {
  threads: MockThread[]
  renameCalls: { threadId: string; title: string; authorization: string | null }[]
  claimCalls: { threadId: string; authorization: string | null }[]
  listCalls: { authorization: string | null }[]
}

/**
 * Installs `page.route()` handlers for the threads API so the React sidebar
 * can be exercised end-to-end without a running backend. The returned state
 * object is mutated in-place by the handlers so specs can inspect which
 * calls fired and rewrite the canned thread list mid-test.
 *
 * The chat turn itself never hits the network in these tests — Vite is
 * started with `VITE_USE_MOCK=true`, which routes `sendChatMessage` through
 * the built-in `mockSendMessage` in `chatClient.ts`.
 */
export async function installMockBackend(
  page: Page,
  initialThreads: MockThread[] = [],
): Promise<MockBackendState> {
  const state: MockBackendState = {
    threads: [...initialThreads],
    renameCalls: [],
    claimCalls: [],
    listCalls: [],
  }

  async function handleThreads(route: Route) {
    const req = route.request()
    const method = req.method()
    const authorization = req.headers()['authorization'] ?? null

    if (method === 'GET') {
      state.listCalls.push({ authorization })
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ threads: state.threads }),
      })
    }

    if (method === 'PATCH') {
      const body = (req.postDataJSON() ?? {}) as { threadId?: string; title?: string }
      const threadId = (body.threadId ?? '').trim()
      const title = (body.title ?? '').trim()
      state.renameCalls.push({ threadId, title, authorization })
      if (!threadId || !title) {
        return route.fulfill({ status: 400, body: '{}' })
      }
      const row = state.threads.find((t) => t.threadId === threadId)
      if (!row) {
        return route.fulfill({ status: 404, body: '{}' })
      }
      row.title = title
      row.updatedAt = Math.floor(Date.now() / 1000)
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ thread: row }),
      })
    }

    return route.fallback()
  }

  async function handleClaim(route: Route) {
    const req = route.request()
    const authorization = req.headers()['authorization'] ?? null
    const body = (req.postDataJSON() ?? {}) as { threadId?: string }
    const threadId = (body.threadId ?? '').trim()
    state.claimCalls.push({ threadId, authorization })
    if (!threadId) {
      return route.fulfill({ status: 400, body: '{}' })
    }
    const now = Math.floor(Date.now() / 1000)
    const existing = state.threads.find((t) => t.threadId === threadId)
    const row = existing
      ? { ...existing, updatedAt: now }
      : { threadId, title: 'New chat', createdAt: now, updatedAt: now }
    if (!existing) {
      state.threads.unshift(row)
    } else {
      existing.updatedAt = now
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ thread: row }),
    })
  }

  await page.route('**/chat/threads', handleThreads)
  await page.route('**/chat/threads/claim', handleClaim)

  return state
}
