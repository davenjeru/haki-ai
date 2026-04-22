import { useEffect, useState } from 'react'
import { useI18n } from '../lib/I18nContext'
import type { Citation } from '../types/chat'

interface Props {
  citations: Citation[]
  /** Index of the citation currently displayed in the source panel. */
  activeIndex?: number
  /** Called when the user changes the current citation (pagination or click). */
  onSelect: (index: number) => void
}

export function CitationBlock({ citations, activeIndex, onSelect }: Props) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)
  // Local pagination state. Kept in sync with activeIndex whenever the parent
  // changes it (e.g. when a new answer auto-selects the first citation).
  const [page, setPage] = useState(() => activeIndex ?? 0)

  useEffect(() => {
    if (typeof activeIndex === 'number' && activeIndex !== page) {
      setPage(activeIndex)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeIndex])

  if (citations.length === 0) return null

  const total = citations.length
  const safePage = Math.min(Math.max(page, 0), total - 1)
  const current = citations[safePage]
  const isActive = safePage === activeIndex
  const hasPage = Boolean(current.pageImageUrl)

  const go = (next: number) => {
    const clamped = Math.min(Math.max(next, 0), total - 1)
    setPage(clamped)
    onSelect(clamped)
  }

  return (
    <div
      className="mt-4 pt-[0.85rem] border-t border-border"
      role="region"
      aria-label="Legal citations"
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-controls="citation-panel"
        className="w-full flex items-center justify-between gap-2 bg-transparent border-0 p-0 cursor-pointer text-muted hover:text-white transition-colors"
      >
        <span className="text-[0.72rem] font-bold uppercase tracking-[0.08em]">
          {t('citations.heading')} ({total})
        </span>
        <span
          aria-hidden="true"
          className={[
            'text-[0.9rem] leading-none transition-transform duration-150',
            expanded ? 'rotate-180' : 'rotate-0',
          ].join(' ')}
        >
          ▾
        </span>
      </button>

      {expanded && (
        <div id="citation-panel" className="mt-[0.6rem]">
          <button
            type="button"
            onClick={() => onSelect(safePage)}
            aria-pressed={isActive}
            aria-label={
              hasPage
                ? `View source page for ${current.source}${current.section ? `, ${current.section}` : ''}`
                : `${current.source}${current.section ? `, ${current.section}` : ''} (no source page available)`
            }
            className={[
              'w-full text-left block py-[0.55rem] px-[0.65rem] rounded-[8px]',
              'border transition-colors cursor-pointer',
              'bg-citation-bg',
              isActive
                ? 'border-accent-bright shadow-[0_0_0_1px_var(--color-accent-bright)]'
                : 'border-citation-border hover:border-accent',
              !hasPage ? 'opacity-75' : '',
            ].join(' ')}
          >
            <div className="flex items-center gap-2">
              <span
                className={[
                  'inline-block w-[6px] h-[6px] rounded-full flex-shrink-0',
                  isActive ? 'bg-accent-bright' : 'bg-transparent border border-citation-border',
                ].join(' ')}
                aria-hidden="true"
              />
              <span className="font-semibold text-[0.9rem] text-citation truncate">
                {current.source}
              </span>
            </div>
            {(current.chapter || current.section) && (
              <div className="text-[0.82rem] text-muted mt-[0.2rem] pl-[14px]">
                {current.chapter && <span>Ch. {current.chapter}</span>}
                {current.chapter && current.section && (
                  <span className="mx-[0.35rem] opacity-70">·</span>
                )}
                {current.section && <span>Sec. {current.section}</span>}
              </div>
            )}
            {current.title && (
              <div className="text-[0.8rem] text-muted mt-[0.35rem] italic pl-[14px]">
                {current.title}
              </div>
            )}
          </button>

          {total > 1 && (
            <div className="flex items-center justify-center gap-3 mt-[0.5rem]">
              <button
                type="button"
                className="bg-transparent border border-citation-border rounded-[6px] text-muted text-[1.1rem] leading-none px-[0.55rem] py-[0.1rem] cursor-pointer hover:text-white hover:border-accent-bright disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                onClick={() => go(safePage - 1)}
                disabled={safePage === 0}
                aria-label="Previous citation"
              >
                ‹
              </button>
              <span className="text-[0.75rem] text-muted min-w-[3rem] text-center">
                {safePage + 1} / {total}
              </span>
              <button
                type="button"
                className="bg-transparent border border-citation-border rounded-[6px] text-muted text-[1.1rem] leading-none px-[0.55rem] py-[0.1rem] cursor-pointer hover:text-white hover:border-accent-bright disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                onClick={() => go(safePage + 1)}
                disabled={safePage === total - 1}
                aria-label="Next citation"
              >
                ›
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
