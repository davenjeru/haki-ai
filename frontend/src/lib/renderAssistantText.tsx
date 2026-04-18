import { Fragment, type ReactNode } from 'react'

/** Minimal inline formatting: **bold** and newlines. Safe for assistant-only text. */
export function renderAssistantText(text: string): ReactNode {
  const lines = text.split('\n')
  return lines.map((line, lineIndex) => (
    <Fragment key={lineIndex}>
      {lineIndex > 0 ? <br /> : null}
      {renderBoldSegments(line)}
    </Fragment>
  ))
}

function renderBoldSegments(line: string): ReactNode {
  const parts = line.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    const m = part.match(/^\*\*(.+)\*\*$/)
    if (m) return <strong key={i} className="text-strong font-semibold">{m[1]}</strong>
    return <span key={i}>{part}</span>
  })
}
