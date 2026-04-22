import type { DetectedLanguage } from '../types/chat'
import { translate, type Locale } from './i18n'

/**
 * Placeholder text for the composer input.
 *
 * Uses the UI locale as the primary source so the composer speaks the
 * same language as the rest of the chrome. We still surface a
 * bilingual fallback whenever either the backend or the frontend
 * detects mixed input so code-switching users see both languages.
 */
export function placeholderForLanguage(locale: Locale, detected: DetectedLanguage | undefined): string {
  if (detected === 'mixed') return translate(locale, 'chat.placeholder.mixed')
  if (detected === 'swahili') return translate(locale, 'chat.placeholder.sw')
  if (detected === 'english') return translate(locale, 'chat.placeholder.en')
  return translate(locale, locale === 'sw' ? 'chat.placeholder.sw' : 'chat.placeholder.en')
}
