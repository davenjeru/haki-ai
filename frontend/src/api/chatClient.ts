import type { ChatResponse, Citation, DetectedLanguage } from '../types/chat'

const BLOCKED_COPY =
  'Mimi ni msaidizi wa kisheria wa Kenya tu.\n\nI can only help with Kenyan legal matters.'

// ── Session management ───────────────────────────────────────────────────────
// The backend uses this id as the LangGraph thread_id for conversation memory.
// Persisted in localStorage so a hard refresh keeps the same chat thread;
// falls back to in-memory when localStorage is unavailable (private mode, SSR).

const SESSION_STORAGE_KEY = 'haki.sessionId'

function generateSessionId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function readStoredSessionId(): string | null {
  try {
    return typeof localStorage !== 'undefined'
      ? localStorage.getItem(SESSION_STORAGE_KEY)
      : null
  } catch {
    return null
  }
}

function writeStoredSessionId(id: string): void {
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(SESSION_STORAGE_KEY, id)
    }
  } catch {
    // localStorage unavailable — we still keep the value in module memory.
  }
}

let currentSessionId: string = readStoredSessionId() ?? (() => {
  const fresh = generateSessionId()
  writeStoredSessionId(fresh)
  return fresh
})()

/** Returns the current chat session id (thread_id). */
export function getSessionId(): string {
  return currentSessionId
}

/**
 * Mints a fresh session id, persisting it to localStorage. Call this from a
 * future "New chat" button to start a new memory thread on the server.
 */
export function resetChatSession(): string {
  currentSessionId = generateSessionId()
  writeStoredSessionId(currentSessionId)
  return currentSessionId
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null
}

function parseLanguage(v: unknown): DetectedLanguage {
  if (v === 'english' || v === 'swahili' || v === 'mixed') return v
  return 'english'
}

function parseCitation(v: unknown): Citation | null {
  if (!isRecord(v)) return null
  const source = v.source
  if (typeof source !== 'string' || !source.trim()) return null
  const chapter = v.chapter
  const section = v.section
  const title = v.title
  return {
    source: source.trim(),
    ...(typeof chapter === 'string' ? { chapter } : {}),
    ...(typeof section === 'string' ? { section } : {}),
    ...(typeof title === 'string' ? { title } : {}),
    ...(typeof v.pageImageUrl === 'string' ? { pageImageUrl: v.pageImageUrl } : {}),
  }
}

export function parseChatResponse(json: unknown): ChatResponse {
  if (!isRecord(json)) {
    throw new Error('Invalid response: expected JSON object')
  }
  const response = json.response
  const blocked = json.blocked === true
  if (typeof response !== 'string') {
    throw new Error('Invalid response: missing string "response"')
  }
  // Defense in depth: if the server returns a different sessionId (e.g.
  // because we sent an empty one), sync our local copy so subsequent
  // requests stay on the same thread_id.
  if (typeof json.sessionId === 'string' && json.sessionId.trim()) {
    const serverSid = json.sessionId.trim()
    if (serverSid !== currentSessionId) {
      currentSessionId = serverSid
      writeStoredSessionId(serverSid)
    }
  }
  const rawCitations = json.citations
  const citations: Citation[] = Array.isArray(rawCitations)
    ? rawCitations.map(parseCitation).filter((c): c is Citation => c !== null)
    : []
  if (blocked) {
    const text = response.trim() || BLOCKED_COPY
    return {
      response: text,
      citations: [],
      language: parseLanguage(json.language),
      blocked: true,
    }
  }
  return {
    response,
    citations,
    language: parseLanguage(json.language),
    blocked: false,
  }
}

async function mockSendMessage(message: string): Promise<ChatResponse> {
  await new Promise((r) => setTimeout(r, 650))
  const lower = message.toLowerCase()
  if (lower.includes('tax') || lower.includes('usa') || lower.includes('contract california')) {
    return {
      response: BLOCKED_COPY,
      citations: [],
      language: 'english',
      blocked: true,
    }
  }
  const looksSwahili = /\b(na|ya|wa|habari|sheria|katika|mimi)\b/i.test(message)
  return {
    response:
      'Under the **Employment Act 2007**, an employer must give notice or payment in lieu when terminating a contract (subject to exceptions). Always verify against the latest gazetted text and your specific facts.',
    citations: [
      {
        source: 'Employment Act 2007',
        chapter: 'IV',
        section: '40',
        title: 'Termination of employment',
        pageImageUrl: 'https://placehold.co/600x800/0c0f0e/8a9a94?text=Employment+Act+p.40',
      },
    ],
    language: looksSwahili ? 'swahili' : 'english',
    blocked: false,
  }
}

// undefined  → VITE_API_BASE_URL not set at all → mock mode
// ""         → VITE_API_BASE_URL set to empty   → real API via Vite dev proxy (/chat)
// "https://…" → absolute URL                    → real API called directly (production)
function apiBase(): string | undefined {
  const raw = import.meta.env.VITE_API_BASE_URL
  if (raw === undefined) return undefined
  return raw.trim().replace(/\/$/, '')
}

function isMockMode(): boolean {
  return import.meta.env.VITE_USE_MOCK === 'true' || apiBase() === undefined
}

/**
 * POST { message } to `${VITE_API_BASE_URL}/chat` unless mock mode.
 * Adjust path in one place when backend finalizes the route.
 */
export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const trimmed = message.trim()
  if (!trimmed) {
    throw new Error('Message is empty')
  }

  if (isMockMode()) {
    return mockSendMessage(trimmed)
  }

  const base = apiBase()
  const path = import.meta.env.VITE_CHAT_PATH ?? '/chat'
  const url = `${base}${path.startsWith('/') ? path : `/${path}`}`

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: trimmed, sessionId: currentSessionId }),
  })

  const text = await res.text()
  let data: unknown
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    throw new Error(`Could not parse JSON (${res.status})`)
  }

  if (!res.ok) {
    const msg =
      isRecord(data) && typeof data.message === 'string'
        ? data.message
        : `Request failed (${res.status})`
    throw new Error(msg)
  }

  return parseChatResponse(data)
}
