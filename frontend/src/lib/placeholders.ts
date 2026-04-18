import type { DetectedLanguage } from '../types/chat'

const PLACEHOLDERS: Record<DetectedLanguage, string> = {
  english: 'Ask about Kenyan law in English…',
  swahili: 'Uliza kuhusu sheria za Kenya kwa Kiswahili…',
  mixed: 'Ask in English or Swahili — Uliza kwa Kiingereza au Kiswahili…',
}

export function placeholderForLanguage(lang: DetectedLanguage | undefined): string {
  return PLACEHOLDERS[lang ?? 'english']
}
