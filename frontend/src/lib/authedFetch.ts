/**
 * Keeps `api/chatClient.ts` framework-free by letting React code register a
 * token provider here. When the user is signed in via Clerk the provider
 * returns a fresh session JWT; otherwise it returns `null` and calls go out
 * anonymously — exactly the shape the backend expects.
 */

export type TokenProvider = () => Promise<string | null>

let tokenProvider: TokenProvider | null = null

export function setTokenProvider(provider: TokenProvider | null): void {
  tokenProvider = provider
}

export async function getAuthToken(): Promise<string | null> {
  if (!tokenProvider) return null
  try {
    return await tokenProvider()
  } catch {
    return null
  }
}

export async function authedFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const token = await getAuthToken()
  const headers = new Headers(init.headers ?? {})
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  return fetch(input, { ...init, headers })
}
