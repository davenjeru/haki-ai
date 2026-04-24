import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { SignedIn, SignedOut, SignInButton, UserButton, useAuth } from '@clerk/clerk-react'
import {
  fetchChatHistory,
  getSessionId,
  resetChatSession,
  sendChatMessage,
  setSessionId,
} from '../api/chatClient'
import { claimThread } from '../api/threadsClient'
import { useI18n } from '../lib/I18nContext'
import { placeholderForLanguage } from '../lib/placeholders'
import type { ChatMessage, Citation, DetectedLanguage } from '../types/chat'
import { Composer } from './Composer'
import { LanguageToggle } from './LanguageToggle'
import { MessageThread } from './MessageThread'
import { SourcePanel } from './SourcePanel'
import { ThreadSidebar } from './ThreadSidebar'

function newId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

interface ActiveSource {
  messageId: string
  index: number
}

export function ChatApp() {
  const { t, locale, onDetectedLanguage } = useI18n()
  const { isLoaded: authLoaded, isSignedIn } = useAuth()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [hydrating, setHydrating] = useState(true)
  const [hintLang, setHintLang] = useState<DetectedLanguage | undefined>(undefined)
  const [activeSource, setActiveSource] = useState<ActiveSource | null>(null)
  const [activeThreadId, setActiveThreadId] = useState<string>(() => getSessionId())
  const [threadsRefreshKey, setThreadsRefreshKey] = useState<number>(0)
  const claimedSessionRef = useRef<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const placeholder = useMemo(() => placeholderForLanguage(locale, hintLang), [locale, hintLang])

  const activeCitations = useMemo<Citation[]>(() => {
    if (!activeSource) return []
    const msg = messages.find((m) => m.id === activeSource.messageId)
    return msg?.citations ?? []
  }, [messages, activeSource])

  const hydrateFromSession = useCallback(
    async (sid: string) => {
      setHydrating(true)
      try {
        const restored = await fetchChatHistory(sid)
        setMessages(restored)
        const lastAssistant = [...restored].reverse().find((m) => m.role === 'assistant')
        if (lastAssistant?.language) {
          setHintLang(lastAssistant.language)
          onDetectedLanguage(lastAssistant.language)
        } else {
          setHintLang(undefined)
        }
        if (lastAssistant?.citations && lastAssistant.citations.length > 0) {
          const firstWithPage = lastAssistant.citations.findIndex((c) => Boolean(c.pageImageUrl))
          setActiveSource({
            messageId: lastAssistant.id,
            index: firstWithPage === -1 ? 0 : firstWithPage,
          })
        } else {
          setActiveSource(null)
        }
      } finally {
        // Pick up any self-healed id (chatClient resets on 403) so the
        // sidebar's active-thread styling stays in sync with reality.
        setActiveThreadId(getSessionId())
        setHydrating(false)
      }
    },
    [onDetectedLanguage],
  )

  // Restore the persisted conversation on mount. The sessionId lives in
  // localStorage (chatClient.ts), so a refresh picks up exactly where the
  // user left off — including freshly re-presigned citation URLs.
  useEffect(() => {
    let cancelled = false
    const sid = getSessionId()
    hydrateFromSession(sid).catch(() => {
      if (!cancelled) setHydrating(false)
    })
    return () => {
      cancelled = true
    }
  }, [hydrateFromSession])

  // React to auth transitions:
  //   - sign-in  → claim the current (likely anonymous) sessionId so the
  //                in-progress conversation becomes the user's first thread.
  //   - sign-out → mint a fresh anonymous sessionId and clear local UI
  //                state. Without this, the next anonymous turn would POST
  //                /chat with the signed-in-user-owned thread_id and the
  //                backend correctly rejects it with 403.
  const wasSignedInRef = useRef<boolean>(false)
  useEffect(() => {
    if (!authLoaded) return
    const wasSignedIn = wasSignedInRef.current
    wasSignedInRef.current = Boolean(isSignedIn)

    if (isSignedIn && !wasSignedIn) {
      const sid = getSessionId()
      if (sid && claimedSessionRef.current !== sid) {
        claimedSessionRef.current = sid
        claimThread(sid).finally(() => {
          setThreadsRefreshKey((k) => k + 1)
        })
      }
    } else if (!isSignedIn && wasSignedIn) {
      const nextId = resetChatSession()
      setActiveThreadId(nextId)
      claimedSessionRef.current = null
      setMessages([])
      setDraft('')
      setActiveSource(null)
      setHintLang(undefined)
      setThreadsRefreshKey((k) => k + 1)
    }
  }, [authLoaded, isSignedIn])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = useCallback(async () => {
    const text = draft.trim()
    if (!text || loading) return

    const userMsg: ChatMessage = { id: newId(), role: 'user', content: text }
    setMessages((m) => [...m, userMsg])
    setDraft('')
    setLoading(true)

    try {
      const res = await sendChatMessage(text)
      setHintLang(res.language)
      onDetectedLanguage(res.language)
      const assistantId = newId()
      const assistant: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: res.response,
        citations: res.citations,
        language: res.language,
        blocked: res.blocked,
      }
      setMessages((m) => [...m, assistant])
      // Auto-focus the first citation with a page preview so the sidebar
      // updates as soon as the new answer arrives.
      if (res.citations.length > 0) {
        const firstWithPage = res.citations.findIndex((c) => Boolean(c.pageImageUrl))
        setActiveSource({
          messageId: assistantId,
          index: firstWithPage === -1 ? 0 : firstWithPage,
        })
      }
      // The backend creates/updates the thread index row on signed-in turns;
      // nudge the sidebar to re-fetch so a fresh thread shows up immediately.
      if (isSignedIn) {
        setThreadsRefreshKey((k) => k + 1)
      }
      // Pick up any self-healed id (chatClient resets on 403 and retries).
      setActiveThreadId(getSessionId())
    } catch (e) {
      const err = e instanceof Error ? e.message : t('errors.generic')
      setMessages((m) => [
        ...m,
        {
          id: newId(),
          role: 'assistant',
          content: '',
          error: err,
        },
      ])
    } finally {
      setLoading(false)
    }
  }, [draft, loading, t, onDetectedLanguage, isSignedIn])

  const handleSelectCitation = useCallback((messageId: string, index: number) => {
    setActiveSource({ messageId, index })
  }, [])

  const handleCarouselIndexChange = useCallback((index: number) => {
    setActiveSource((prev) => (prev ? { ...prev, index } : prev))
  }, [])

  // "Clear chat" mints a fresh sessionId (= new LangGraph thread_id) so the
  // next turn starts with no server-side memory, and wipes all local state
  // that was tied to the old conversation.
  const handleClearChat = useCallback(() => {
    if (loading) return
    const nextId = resetChatSession()
    setActiveThreadId(nextId)
    claimedSessionRef.current = null
    setMessages([])
    setDraft('')
    setActiveSource(null)
    setHintLang(undefined)
  }, [loading])

  const handleSelectThread = useCallback(
    (threadId: string) => {
      if (loading || threadId === activeThreadId) return
      setSessionId(threadId)
      setActiveThreadId(threadId)
      hydrateFromSession(threadId)
    },
    [loading, activeThreadId, hydrateFromSession],
  )

  return (
    <div className="h-dvh flex">
      <ThreadSidebar
        activeThreadId={activeThreadId}
        onSelectThread={handleSelectThread}
        onNewChat={handleClearChat}
        refreshKey={threadsRefreshKey}
        busy={loading}
      />

      <div className="flex-1 flex flex-col min-w-0 px-4 py-5 max-w-[1220px] w-full mx-auto">
        <header className="mb-4 pb-4 border-b border-border flex-shrink-0">
          <div className="mb-2 flex items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-[1.75rem] font-semibold tracking-[-0.02em] m-0 mb-1 bg-[linear-gradient(120deg,#d8f3dc_0%,#95d5b2_45%,#52b788_100%)] bg-clip-text text-transparent">
                {t('header.title')}
              </h1>
              <p className="m-0 text-muted text-[0.95rem]">
                {t('header.subtitle')}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <LanguageToggle />
              <SignedOut>
                <SignInButton mode="modal">
                  <button
                    type="button"
                    className="inline-flex items-center px-3 py-[0.45rem] rounded-full bg-accent text-strong text-[0.8rem] font-semibold hover:bg-accent-bright transition-colors cursor-pointer"
                  >
                    {t('auth.signIn')}
                  </button>
                </SignInButton>
              </SignedOut>
              <SignedIn>
                <UserButton afterSignOutUrl="/" />
              </SignedIn>
            </div>
          </div>
          <p className="mt-3 text-[0.8rem] text-muted tracking-[0.02em]">
            {t('header.corpus')}
          </p>
        </header>

        <div className="flex-1 flex gap-6 min-h-0">
          <section className="flex-1 flex flex-col min-w-0 max-w-[760px] mx-auto lg:mx-0">
            <main className="flex-1 overflow-y-auto pb-2">
              {messages.length === 0 && !loading && !hydrating && (
                <p className="mb-4 p-[1rem_1.1rem] bg-elevated border border-border rounded-[12px] text-muted text-[0.95rem]">
                  {t('chat.empty')}
                </p>
              )}
              <MessageThread
                messages={messages}
                activeMessageId={activeSource?.messageId}
                activeCitationIndex={activeSource?.index}
                onSelectCitation={handleSelectCitation}
              />
              {loading && (
                <div className="flex items-center gap-[0.35rem] py-2 text-muted text-[0.88rem]" aria-busy="true">
                  <span className="w-[6px] h-[6px] rounded-full bg-accent-bright animate-thinking" />
                  <span className="w-[6px] h-[6px] rounded-full bg-accent-bright animate-thinking [animation-delay:0.15s]" />
                  <span className="w-[6px] h-[6px] rounded-full bg-accent-bright animate-thinking [animation-delay:0.3s]" />
                  <span className="ml-[0.35rem]">{t('chat.loading')}</span>
                </div>
              )}
              <div ref={bottomRef} />
            </main>

            <footer className="pt-3 border-t border-border flex-shrink-0">
              <Composer
                value={draft}
                onChange={setDraft}
                onSubmit={send}
                disabled={loading}
                placeholder={placeholder}
              />
              <p className="mt-[0.65rem] text-[0.75rem] text-muted">
                {t('chat.disclaimer')}
              </p>
            </footer>
          </section>

          <div className="hidden lg:block w-[420px] flex-shrink-0 h-full min-h-0">
            <SourcePanel
              citations={activeCitations}
              activeIndex={activeSource?.index ?? 0}
              onIndexChange={handleCarouselIndexChange}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
