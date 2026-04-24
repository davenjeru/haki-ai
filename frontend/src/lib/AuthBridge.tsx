import { useEffect } from 'react'
import { useAuth } from '@clerk/clerk-react'
import { setTokenProvider } from './authedFetch'

/**
 * Registers Clerk's `getToken` with `authedFetch` so downstream API clients
 * (chatClient, threadsClient) can attach a Bearer header without importing
 * any Clerk code. Rendered once near the root of the tree.
 */
export function AuthBridge() {
  const { getToken, isLoaded, isSignedIn } = useAuth()

  useEffect(() => {
    if (!isLoaded) return
    if (!isSignedIn) {
      setTokenProvider(null)
      return
    }
    setTokenProvider(async () => {
      try {
        return await getToken()
      } catch {
        return null
      }
    })
    return () => setTokenProvider(null)
  }, [getToken, isLoaded, isSignedIn])

  return null
}
