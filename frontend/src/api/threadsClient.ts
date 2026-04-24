/**
 * Thin client for the signed-in-only thread index API.
 *
 *  GET    /chat/threads             — list user's threads
 *  PATCH  /chat/threads             — rename title { threadId, title }
 *  POST   /chat/threads/claim       — claim anonymous session { threadId }
 *
 * All calls attach a Clerk Bearer token via `authedFetch`. If the user is
 * signed out (no token provider registered or provider returns `null`), we
 * skip the network entirely and return empty — the UI paths that use these
 * helpers never call them in that state anyway.
 */

import { authedFetch, getAuthToken } from '../lib/authedFetch'

export interface ChatThread {
  threadId: string
  title: string
  updatedAt: number
  createdAt: number
}

function apiBase(): string | undefined {
  const raw = import.meta.env.VITE_API_BASE_URL
  if (raw === undefined) return undefined
  return raw.trim().replace(/\/$/, '')
}

/**
 * The threads API is independent of the chat mock switch (VITE_USE_MOCK):
 * a developer can keep chat turns mocked while still exercising the real
 * thread-index endpoints, and e2e tests intercept these calls directly
 * via `page.route()`. We only short-circuit when there's no API base URL
 * configured at all.
 */
function hasApi(): boolean {
  return apiBase() !== undefined
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null
}

function parseThread(v: unknown): ChatThread | null {
  if (!isRecord(v)) return null
  const threadId = typeof v.threadId === 'string' ? v.threadId : ''
  if (!threadId) return null
  const title = typeof v.title === 'string' && v.title ? v.title : 'Untitled chat'
  const updatedAt = typeof v.updatedAt === 'number' ? v.updatedAt : 0
  const createdAt = typeof v.createdAt === 'number' ? v.createdAt : updatedAt
  return { threadId, title, updatedAt, createdAt }
}

export async function listThreads(): Promise<ChatThread[]> {
  if (!hasApi()) return []
  const token = await getAuthToken()
  if (!token) return []
  const url = `${apiBase()}/chat/threads`
  const res = await authedFetch(url, { method: 'GET' })
  if (!res.ok) return []
  const data: unknown = await res.json().catch(() => null)
  if (!isRecord(data)) return []
  const raw = data.threads
  if (!Array.isArray(raw)) return []
  return raw
    .map(parseThread)
    .filter((t): t is ChatThread => t !== null)
    .sort((a, b) => b.updatedAt - a.updatedAt)
}

export async function renameThread(threadId: string, title: string): Promise<boolean> {
  if (!hasApi()) return true
  const token = await getAuthToken()
  if (!token) return false
  const url = `${apiBase()}/chat/threads`
  const res = await authedFetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threadId, title }),
  })
  return res.ok
}

export async function claimThread(threadId: string): Promise<boolean> {
  if (!hasApi()) return true
  const token = await getAuthToken()
  if (!token) return false
  const url = `${apiBase()}/chat/threads/claim`
  const res = await authedFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threadId }),
  })
  return res.ok
}
