import type { Citation } from '../types/chat'
import { PageCarousel } from './PageCarousel'

interface Props {
  /** Citations from the currently-focused assistant message. */
  citations: Citation[]
  /** Which citation is currently highlighted. */
  activeIndex: number
  /** Called when the user flips pages via the carousel arrows. */
  onIndexChange: (index: number) => void
}

export function SourcePanel({ citations, activeIndex, onIndexChange }: Props) {
  const imageUrls = citations
    .map((c) => c.pageImageUrl)
    .filter((url): url is string => typeof url === 'string')

  return (
    <aside
      className="flex flex-col gap-3 h-full min-h-0 rounded-[12px] bg-elevated border border-border p-4"
      role="complementary"
      aria-label="Source page viewer"
    >
      <header className="flex items-baseline justify-between gap-2">
        <h2 className="font-display text-[1.05rem] font-semibold tracking-[-0.01em] m-0 text-strong">
          Source page
        </h2>
        {imageUrls.length > 0 && (
          <span className="text-[0.72rem] text-muted tracking-[0.04em] uppercase">
            {activeIndex + 1} of {imageUrls.length}
          </span>
        )}
      </header>

      {imageUrls.length === 0 ? (
        <EmptyState hasCitations={citations.length > 0} />
      ) : (
        <>
          <ActiveCitationLabel citation={citations[activeIndex]} />
          <div className="flex-1 min-h-0 overflow-hidden">
            <PageCarousel
              imageUrls={imageUrls}
              index={activeIndex}
              onIndexChange={onIndexChange}
            />
          </div>
        </>
      )}
    </aside>
  )
}

function ActiveCitationLabel({ citation }: { citation: Citation | undefined }) {
  if (!citation) return null
  return (
    <div className="rounded-[8px] border border-citation-border bg-citation-bg px-3 py-2">
      <div className="text-[0.82rem] font-semibold text-citation truncate">{citation.source}</div>
      {(citation.chapter || citation.section) && (
        <div className="text-[0.74rem] text-muted mt-[0.15rem] truncate">
          {citation.chapter && <span>{citation.chapter}</span>}
          {citation.chapter && citation.section && (
            <span className="mx-[0.35rem] opacity-70">·</span>
          )}
          {citation.section && <span>{citation.section}</span>}
        </div>
      )}
      {citation.title && (
        <div className="text-[0.72rem] text-muted mt-[0.2rem] italic truncate">
          {citation.title}
        </div>
      )}
    </div>
  )
}

function EmptyState({ hasCitations }: { hasCitations: boolean }) {
  return (
    <div className="flex-1 rounded-[10px] border border-dashed border-border flex items-center justify-center px-4 py-10 text-center">
      <p className="m-0 text-[0.82rem] text-muted leading-[1.5]">
        {hasCitations
          ? 'No source page available for this citation.'
          : 'Ask a question to see the source pages from Kenyan statutes here.'}
      </p>
    </div>
  )
}
