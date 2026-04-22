import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { sendChatMessage } from '../api/chatClient'
import { placeholderForLanguage } from '../lib/placeholders'
import type { ChatMessage, Citation, DetectedLanguage } from '../types/chat'
import { Composer } from './Composer'
import { MessageThread } from './MessageThread'
import { SourcePanel } from './SourcePanel'

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
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [hintLang, setHintLang] = useState<DetectedLanguage | undefined>(undefined)
  const [activeSource, setActiveSource] = useState<ActiveSource | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const placeholder = useMemo(() => placeholderForLanguage(hintLang), [hintLang])

  const activeCitations = useMemo<Citation[]>(() => {
    if (!activeSource) return []
    const msg = messages.find((m) => m.id === activeSource.messageId)
    return msg?.citations ?? []
  }, [messages, activeSource])

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
    } catch (e) {
      const err = e instanceof Error ? e.message : 'Something went wrong'
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
  }, [draft, loading])

  const handleSelectCitation = useCallback((messageId: string, index: number) => {
    setActiveSource({ messageId, index })
  }, [])

  const handleCarouselIndexChange = useCallback((index: number) => {
    setActiveSource((prev) => (prev ? { ...prev, index } : prev))
  }, [])

  return (
    <div className="h-dvh flex flex-col max-w-[1400px] mx-auto px-4 py-5">
      <header className="mb-4 pb-4 border-b border-border flex-shrink-0">
        <div className="mb-2">
          <h1 className="font-display text-[1.75rem] font-semibold tracking-[-0.02em] m-0 mb-1 bg-[linear-gradient(120deg,#d8f3dc_0%,#95d5b2_45%,#52b788_100%)] bg-clip-text text-transparent">
            Haki AI
          </h1>
          <p className="m-0 text-muted text-[0.95rem]">
            Kenyan legal aid — answers with Act, Chapter, and Section citations
          </p>
        </div>
        <p className="mt-3 text-[0.8rem] text-muted tracking-[0.02em]">
          Constitution 2010 · Employment Act 2007 · Land Act 2012
        </p>
      </header>

      <div className="flex-1 flex gap-6 min-h-0">
        <section className="flex-1 flex flex-col min-w-0 max-w-[760px] mx-auto lg:mx-0">
          <main className="flex-1 overflow-y-auto pb-2">
            {messages.length === 0 && !loading && (
              <p className="mb-4 p-[1rem_1.1rem] bg-elevated border border-border rounded-[12px] text-muted text-[0.95rem]">
                Ask a question about Kenyan law. The assistant replies in English or Swahili and cites the relevant provisions.
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
                <span className="ml-[0.35rem]">Searching statutes…</span>
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
              Not legal advice. Verify citations against official sources.
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
  )
}
