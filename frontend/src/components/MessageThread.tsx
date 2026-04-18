import type { ChatMessage } from '../types/chat'
import { renderAssistantText } from '../lib/renderAssistantText'
import { CitationBlock } from './CitationBlock'
import { LanguageBadge } from './LanguageBadge'

interface Props {
  messages: ChatMessage[]
}

function bubbleClasses(role: 'user' | 'assistant', blocked?: boolean, error?: boolean): string {
  const base = 'rounded-[12px] py-[0.85rem] px-4 border'
  if (error) {
    return `${base} border-[rgba(201,76,76,0.5)] bg-[#1f1414] self-stretch`
  }
  if (blocked) {
    return `${base} border-[rgba(212,160,23,0.45)] bg-[linear-gradient(145deg,#2a2618_0%,#1a1810_100%)] self-stretch`
  }
  if (role === 'user') {
    return `${base} border-user-border bg-[linear-gradient(145deg,#1b3328_0%,#14261f_100%)] self-end max-w-[92%]`
  }
  return `${base} border-border bg-elevated self-stretch`
}

export function MessageThread({ messages }: Props) {
  return (
    <div className="flex flex-col gap-4" role="log" aria-live="polite" aria-relevant="additions">
      {messages.map((m) => (
        <article key={m.id} className={bubbleClasses(m.role, m.blocked, !!m.error)}>
          <div className="min-h-[1.5rem] mb-[0.35rem]">
            {m.role === 'assistant' && m.language && !m.error && (
              <LanguageBadge language={m.language} />
            )}
          </div>
          <div className="text-[0.98rem]">
            {m.error ? (
              <p className="m-0 text-error-text text-[0.95rem]">{m.error}</p>
            ) : (
              <>
                <div className="whitespace-pre-wrap break-words">
                  {m.role === 'assistant'
                    ? renderAssistantText(m.content)
                    : m.content}
                </div>
                {m.role === 'assistant' && m.citations && m.citations.length > 0 && (
                  <CitationBlock citations={m.citations} />
                )}
              </>
            )}
          </div>
        </article>
      ))}
    </div>
  )
}
