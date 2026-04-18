import { useState } from 'react'

interface Props {
  imageUrls: string[]
}

export function PageCarousel({ imageUrls }: Props) {
  const [index, setIndex] = useState(0)

  if (imageUrls.length === 0) return null

  const isFirst = index === 0
  const isLast = index === imageUrls.length - 1

  return (
    <div className="mb-[0.85rem] rounded-[8px] overflow-hidden border border-citation-border bg-bg" role="region" aria-label="Source page images">
      {/* Page image */}
      <div className="w-full aspect-[4/5] overflow-hidden flex items-center justify-center">
        <img
          key={imageUrls[index]}
          src={imageUrls[index]}
          alt={`Source page ${index + 1} of ${imageUrls.length}`}
          className="w-full h-full object-contain block"
          loading="lazy"
        />
      </div>

      {/* Controls — only shown when there are multiple pages */}
      {imageUrls.length > 1 && (
        <div className="flex items-center justify-center gap-3 px-[0.65rem] py-[0.45rem] border-t border-citation-border">
          <button
            className="bg-transparent border border-citation-border rounded-[6px] text-muted text-[1.2rem] leading-none px-[0.55rem] py-[0.15rem] cursor-pointer hover:text-white hover:border-accent-bright disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            onClick={() => setIndex((i) => i - 1)}
            disabled={isFirst}
            aria-label="Previous page"
          >
            ‹
          </button>
          <span className="text-[0.75rem] text-muted min-w-[3rem] text-center">
            {index + 1} / {imageUrls.length}
          </span>
          <button
            className="bg-transparent border border-citation-border rounded-[6px] text-muted text-[1.2rem] leading-none px-[0.55rem] py-[0.15rem] cursor-pointer hover:text-white hover:border-accent-bright disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            onClick={() => setIndex((i) => i + 1)}
            disabled={isLast}
            aria-label="Next page"
          >
            ›
          </button>
        </div>
      )}
    </div>
  )
}
