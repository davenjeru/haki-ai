import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import type { PDFDocumentLoadingTask, RenderTask } from 'pdfjs-dist'
import PdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

// Configure the worker exactly once. Vite's ?url import resolves to a hashed
// static asset URL so the worker is served from the same origin and cached.
pdfjsLib.GlobalWorkerOptions.workerSrc = PdfWorkerUrl

function isCancellation(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false
  const name = (err as { name?: unknown }).name
  const message = (err as { message?: unknown }).message
  if (typeof name === 'string' && name === 'RenderingCancelledException') return true
  if (typeof message === 'string' && /worker was destroyed/i.test(message)) return true
  return false
}

interface Props {
  /** Presigned S3 GET URLs pointing at single-page PDFs (one per citation). */
  imageUrls: string[]
  /** Which URL is currently rendered. Controlled by the parent. */
  index: number
  /** Called when the user clicks the prev/next arrows. */
  onIndexChange: (index: number) => void
}

export function PageCarousel({ imageUrls, index, onIndexChange }: Props) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  const url = imageUrls[index]

  useEffect(() => {
    if (!url) return
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    let cancelled = false
    let loadingTask: PDFDocumentLoadingTask | null = null
    let renderTask: RenderTask | null = null

    setStatus('loading')

    const render = async () => {
      try {
        loadingTask = pdfjsLib.getDocument({ url, withCredentials: false })
        const pdf = await loadingTask.promise
        if (cancelled) return
        const page = await pdf.getPage(1)
        if (cancelled) return

        const baseViewport = page.getViewport({ scale: 1 })
        const targetWidth = container.clientWidth || 600
        // Multiply by devicePixelRatio for crisp rendering on HiDPI screens.
        const dpr = Math.min(window.devicePixelRatio || 1, 2)
        const scale = (targetWidth / baseViewport.width) * dpr
        const viewport = page.getViewport({ scale })

        const ctx = canvas.getContext('2d')
        if (!ctx) throw new Error('Canvas 2D context unavailable')

        canvas.width = Math.floor(viewport.width)
        canvas.height = Math.floor(viewport.height)
        canvas.style.width = `${Math.floor(viewport.width / dpr)}px`
        canvas.style.height = `${Math.floor(viewport.height / dpr)}px`

        renderTask = page.render({ canvasContext: ctx, viewport, canvas })
        await renderTask.promise
        if (!cancelled) setStatus('ready')
      } catch (err) {
        // RenderingCancelledException + "Worker was destroyed" are normal
        // when the user flips pages quickly. Don't surface them as errors.
        if (cancelled || isCancellation(err)) return
        console.error('PageCarousel: PDF render failed', err)
        setStatus('error')
      }
    }

    void render()

    return () => {
      cancelled = true
      try {
        renderTask?.cancel()
      } catch {
        // noop — cancel throws when the task has already settled
      }
      void loadingTask?.destroy().catch(() => {})
    }
  }, [url])

  if (imageUrls.length === 0) return null

  const isFirst = index === 0
  const isLast = index === imageUrls.length - 1

  return (
    <div
      className="mb-[0.85rem] rounded-[8px] overflow-hidden border border-citation-border bg-bg"
      role="region"
      aria-label="Source page images"
    >
      <div
        ref={containerRef}
        className="w-full aspect-[4/5] overflow-hidden flex items-center justify-center bg-elevated"
      >
        {status === 'loading' && (
          <div className="text-[0.8rem] text-muted" aria-live="polite">
            Loading source page…
          </div>
        )}
        {status === 'error' && (
          <div className="text-[0.8rem] text-muted px-4 text-center" role="alert">
            Could not load source page.
          </div>
        )}
        <canvas
          ref={canvasRef}
          className={`block ${status === 'ready' ? '' : 'hidden'}`}
          aria-label={`Source page ${index + 1} of ${imageUrls.length}`}
        />
      </div>

      {imageUrls.length > 1 && (
        <div className="flex items-center justify-center gap-3 px-[0.65rem] py-[0.45rem] border-t border-citation-border">
          <button
            className="bg-transparent border border-citation-border rounded-[6px] text-muted text-[1.2rem] leading-none px-[0.55rem] py-[0.15rem] cursor-pointer hover:text-white hover:border-accent-bright disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            onClick={() => onIndexChange(index - 1)}
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
            onClick={() => onIndexChange(index + 1)}
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
