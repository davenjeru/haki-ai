import type { DetectedLanguage } from '../types/chat'

const LABELS: Record<DetectedLanguage, string> = {
  english: 'English',
  swahili: 'Kiswahili',
  mixed: 'Mixed',
}

interface Props {
  language: DetectedLanguage
}

export function LanguageBadge({ language }: Props) {
  return (
    <span
      className="inline-block text-[0.7rem] font-semibold uppercase tracking-[0.06em] px-[0.45rem] py-[0.2rem] rounded-[6px] bg-lang-bg text-lang-text border border-lang-border"
      title="Detected response language"
    >
      {LABELS[language]}
    </span>
  )
}
