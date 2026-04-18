import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { sendChatMessage } from '../api/chatClient'
import { placeholderForLanguage } from '../lib/placeholders'
import type { ChatMessage, DetectedLanguage } from '../types/chat'
import { Composer } from './Composer'
import { MessageThread } from './MessageThread'

function newId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function ChatApp() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [hintLang, setHintLang] = useState<DetectedLanguage | undefined>(undefined)
  const bottomRef = useRef<HTMLDivElement>(null)

  const placeholder = useMemo(() => placeholderForLanguage(hintLang), [hintLang])

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
      const assistant: ChatMessage = {
        id: newId(),
        role: 'assistant',
        content: res.response,
        citations: res.citations,
        language: res.language,
        blocked: res.blocked,
      }
      setMessages((m) => [...m, assistant])
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

  return (
    <div className="min-h-dvh flex flex-col max-w-[720px] mx-auto px-4 py-5 pb-6">
      <header className="mb-4 pb-4 border-b border-border">
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

      <main className="flex-1 overflow-y-auto pb-2">
        {messages.length === 0 && !loading && (
          <p className="mb-4 p-[1rem_1.1rem] bg-elevated border border-border rounded-[12px] text-muted text-[0.95rem]">
            Ask a question about Kenyan law. The assistant replies in English or Swahili and cites the relevant provisions.
          </p>
        )}
        <MessageThread messages={messages} />
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

      <footer className="pt-3 border-t border-border">
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
    </div>
  )
}
