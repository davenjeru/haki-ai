import type { ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

/**
 * Render assistant-authored markdown safely.
 *
 * react-markdown outputs React elements (no dangerouslySetInnerHTML) and
 * drops raw HTML by default, so this is XSS-safe for model output. Each
 * element gets a class that matches the app's design tokens (muted,
 * strong, citation, accent) rather than relying on @tailwindcss/typography
 * so the look stays consistent with the rest of the UI.
 */

const components: Components = {
  h1: ({ children }) => (
    <h1 className="font-display text-[1.35rem] font-semibold tracking-[-0.01em] mt-4 mb-2 first:mt-0 text-strong">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="font-display text-[1.15rem] font-semibold tracking-[-0.01em] mt-4 mb-2 first:mt-0 text-strong">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-[1rem] font-semibold mt-3 mb-1.5 first:mt-0 text-strong">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="text-[0.95rem] font-semibold mt-3 mb-1 first:mt-0 text-strong">
      {children}
    </h4>
  ),
  p: ({ children }) => (
    <p className="m-0 mb-2 last:mb-0 leading-[1.55]">{children}</p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-strong">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,
  ul: ({ children }) => (
    <ul className="list-disc pl-5 my-2 marker:text-muted [&>li]:mb-1 last:[&>li]:mb-0">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal pl-5 my-2 marker:text-muted [&>li]:mb-1 last:[&>li]:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-[1.55]">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-accent-bright pl-3 my-2 text-muted italic">
      {children}
    </blockquote>
  ),
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent-bright underline underline-offset-2 hover:text-strong"
    >
      {children}
    </a>
  ),
  code: ({ className, children }) => {
    const isBlock = /language-/.test(className ?? '')
    if (isBlock) {
      return (
        <code className="block bg-bg border border-border rounded-[6px] px-3 py-2 my-2 text-[0.85rem] font-mono overflow-x-auto">
          {children}
        </code>
      )
    }
    return (
      <code className="bg-bg border border-border rounded-[4px] px-1.5 py-[1px] text-[0.88em] font-mono">
        {children}
      </code>
    )
  },
  pre: ({ children }) => <pre className="m-0">{children}</pre>,
  hr: () => <hr className="my-3 border-0 border-t border-border" />,
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="text-[0.88rem] border-collapse border border-border">
        {children}
      </table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border px-2 py-1 text-left font-semibold bg-bg">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-border px-2 py-1 align-top">{children}</td>
  ),
}

export function renderAssistantText(text: string): ReactNode {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {text}
    </ReactMarkdown>
  )
}
