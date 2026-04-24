import { useCallback, useEffect, useRef, useState } from 'react'
import { SignedIn, SignedOut, SignInButton } from '@clerk/clerk-react'
import { useI18n } from '../lib/I18nContext'
import type { ChatThread } from '../api/threadsClient'
import { listThreads, renameThread } from '../api/threadsClient'

const COLLAPSED_KEY = 'haki.sidebar.collapsed'

interface ThreadSidebarProps {
  /** Currently active thread/session id. */
  activeThreadId: string
  /** Switch to an existing thread. Parent is responsible for history hydration. */
  onSelectThread: (threadId: string) => void
  /** Mint a fresh session id (new chat). */
  onNewChat: () => void
  /** Bumped whenever the parent knows the list may have changed (e.g. after a turn). */
  refreshKey?: number
  /** Disables the new-chat + thread-switch controls while a chat turn is in flight. */
  busy?: boolean
}

function readCollapsed(): boolean {
  if (typeof localStorage === 'undefined') return false
  return localStorage.getItem(COLLAPSED_KEY) === '1'
}

function writeCollapsed(value: boolean): void {
  try {
    localStorage.setItem(COLLAPSED_KEY, value ? '1' : '0')
  } catch {
    // localStorage unavailable — fine.
  }
}

export function ThreadSidebar({
  activeThreadId,
  onSelectThread,
  onNewChat,
  refreshKey = 0,
  busy = false,
}: ThreadSidebarProps) {
  const { t } = useI18n()
  const [collapsed, setCollapsedState] = useState<boolean>(() => readCollapsed())
  const [threads, setThreads] = useState<ChatThread[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [draftTitle, setDraftTitle] = useState<string>('')
  const renameInputRef = useRef<HTMLInputElement | null>(null)

  const setCollapsed = useCallback((next: boolean) => {
    setCollapsedState(next)
    writeCollapsed(next)
  }, [])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const list = await listThreads()
      setThreads(list)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh, refreshKey])

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus()
      renameInputRef.current.select()
    }
  }, [renamingId])

  const startRename = useCallback((thread: ChatThread) => {
    setRenamingId(thread.threadId)
    setDraftTitle(thread.title)
  }, [])

  const cancelRename = useCallback(() => {
    setRenamingId(null)
    setDraftTitle('')
  }, [])

  const commitRename = useCallback(async () => {
    if (!renamingId) return
    const nextTitle = draftTitle.trim()
    if (!nextTitle) {
      cancelRename()
      return
    }
    setThreads((prev) =>
      prev.map((t) => (t.threadId === renamingId ? { ...t, title: nextTitle } : t)),
    )
    const ok = await renameThread(renamingId, nextTitle)
    if (!ok) {
      refresh()
    }
    cancelRename()
  }, [renamingId, draftTitle, cancelRename, refresh])

  if (collapsed) {
    return (
      <aside
        aria-label={t('sidebar.label')}
        className="hidden lg:flex flex-col w-12 flex-shrink-0 border-r border-border bg-elevated/40"
      >
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          title={t('sidebar.expand')}
          aria-label={t('sidebar.expand')}
          className="p-3 text-muted hover:text-strong transition-colors cursor-pointer"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
      </aside>
    )
  }

  return (
    <aside
      aria-label={t('sidebar.label')}
      className="hidden lg:flex flex-col w-[260px] flex-shrink-0 border-r border-border bg-elevated/40"
    >
      <div className="flex items-center justify-between px-3 py-3 border-b border-border">
        <span className="text-[0.72rem] font-semibold tracking-[0.08em] uppercase text-muted">
          {t('sidebar.heading')}
        </span>
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          title={t('sidebar.collapse')}
          aria-label={t('sidebar.collapse')}
          className="text-muted hover:text-strong transition-colors cursor-pointer"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
      </div>

      <button
        type="button"
        onClick={onNewChat}
        disabled={busy}
        className="mx-3 mt-3 inline-flex items-center gap-2 justify-center px-3 py-[0.55rem] rounded-full border border-border bg-bg text-strong text-[0.82rem] font-semibold hover:border-accent-bright transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:border-border"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
        {t('sidebar.newChat')}
      </button>

      <SignedOut>
        <div className="flex-1 flex flex-col items-center justify-center gap-3 px-6 py-10 text-center">
          <p className="text-muted text-[0.85rem]">{t('sidebar.signedOut.prompt')}</p>
          <SignInButton mode="modal">
            <button
              type="button"
              className="inline-flex items-center px-3 py-[0.5rem] rounded-full bg-accent text-strong text-[0.8rem] font-semibold hover:bg-accent-bright transition-colors cursor-pointer"
            >
              {t('auth.signIn')}
            </button>
          </SignInButton>
        </div>
      </SignedOut>

      <SignedIn>
        <div
          className="flex-1 overflow-y-auto px-2 py-3"
          role="list"
          aria-label={t('sidebar.list.label')}
        >
          {loading && threads.length === 0 && (
            <p className="px-3 py-2 text-[0.8rem] text-muted">{t('sidebar.loading')}</p>
          )}
          {!loading && threads.length === 0 && (
            <p className="px-3 py-2 text-[0.8rem] text-muted">{t('sidebar.empty')}</p>
          )}
          {threads.map((thread) => {
            const isActive = thread.threadId === activeThreadId
            const isRenaming = thread.threadId === renamingId
            return (
              <div
                key={thread.threadId}
                role="listitem"
                data-testid={`thread-row-${thread.threadId}`}
                className={`group rounded-[10px] px-3 py-[0.55rem] mb-1 text-[0.85rem] cursor-pointer flex items-center gap-2 transition-colors ${
                  isActive
                    ? 'bg-bg text-strong border border-border'
                    : 'text-muted hover:text-strong hover:bg-bg/60 border border-transparent'
                }`}
                onClick={() => {
                  if (!isRenaming) onSelectThread(thread.threadId)
                }}
              >
                {isRenaming ? (
                  <input
                    ref={renameInputRef}
                    value={draftTitle}
                    onChange={(e) => setDraftTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        commitRename()
                      } else if (e.key === 'Escape') {
                        e.preventDefault()
                        cancelRename()
                      }
                    }}
                    onBlur={commitRename}
                    aria-label={t('sidebar.renameAria')}
                    className="flex-1 min-w-0 bg-transparent text-strong text-[0.85rem] outline-none border-b border-accent-bright"
                  />
                ) : (
                  <span className="flex-1 min-w-0 truncate">{thread.title}</span>
                )}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    startRename(thread)
                  }}
                  className="opacity-0 group-hover:opacity-100 text-muted hover:text-accent-bright cursor-pointer"
                  title={t('sidebar.renameTooltip')}
                  aria-label={`${t('sidebar.renameTooltip')}: ${thread.title}`}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M12 20h9" />
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
                  </svg>
                </button>
              </div>
            )
          })}
        </div>
      </SignedIn>
    </aside>
  )
}
