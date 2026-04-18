import type { Citation } from '../types/chat'
import { PageCarousel } from './PageCarousel'

interface Props {
  citations: Citation[]
}

export function CitationBlock({ citations }: Props) {
  if (citations.length === 0) return null

  const imageUrls = citations
    .map((c) => c.pageImageUrl)
    .filter((url): url is string => typeof url === 'string')

  return (
    <div className="mt-4 pt-[0.85rem] border-t border-border" role="region" aria-label="Legal citations">
      <div className="text-[0.72rem] font-bold uppercase tracking-[0.08em] text-muted mb-[0.6rem]">
        Citations
      </div>

      <PageCarousel imageUrls={imageUrls} />

      <ul className="list-none m-0 p-0 flex flex-col gap-2">
        {citations.map((c, i) => (
          <li key={`${c.source}-${c.section ?? i}`} className="m-0 py-[0.55rem] px-[0.65rem] rounded-[8px] bg-citation-bg border border-citation-border">
            <div className="font-semibold text-[0.9rem] text-citation">{c.source}</div>
            {(c.chapter || c.section) && (
              <div className="text-[0.82rem] text-muted mt-[0.2rem]">
                {c.chapter && <span>Ch. {c.chapter}</span>}
                {c.chapter && c.section && <span className="mx-[0.35rem] opacity-70">·</span>}
                {c.section && <span>Sec. {c.section}</span>}
              </div>
            )}
            {c.title && <div className="text-[0.8rem] text-muted mt-[0.35rem] italic">{c.title}</div>}
          </li>
        ))}
      </ul>
    </div>
  )
}
