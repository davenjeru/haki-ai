import { useI18n } from '../lib/I18nContext'
import type { I18nKey } from '../lib/i18n'
import type { DetectedLanguage } from '../types/chat'

const LABEL_KEYS: Record<DetectedLanguage, I18nKey> = {
  english: 'lang.en.label',
  swahili: 'lang.sw.label',
  mixed: 'lang.mixed.label',
}

interface Props {
  language: DetectedLanguage
}

export function LanguageBadge({ language }: Props) {
  const { t } = useI18n()
  return (
    <span
      className="inline-block text-[0.7rem] font-semibold uppercase tracking-[0.06em] px-[0.45rem] py-[0.2rem] rounded-[6px] bg-lang-bg text-lang-text border border-lang-border"
      title={t('lang.badge.tooltip')}
    >
      {t(LABEL_KEYS[language])}
    </span>
  )
}
