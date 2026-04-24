/**
 * UI dictionaries for the Haki AI frontend.
 *
 * Keeps every user-facing English/Swahili string in one place so we can
 * flip the whole UI between languages with a single toggle. Keys are
 * dotted paths to mirror the component they live in, which keeps diffs
 * narrow when new copy is added.
 *
 * When a new string is added: update BOTH `en` and `sw`. TypeScript will
 * not let `sw` drift out of sync because its type is derived from `en`.
 */

export type Locale = 'en' | 'sw'

const en = {
  'header.title': 'Haki AI',
  'header.subtitle': 'Kenyan legal aid — answers with Act, Chapter, and Section citations',
  'header.corpus': 'Constitution 2010 · Employment Act 2007 · Land Act 2012',

  'chat.empty':
    'Ask a question about Kenyan law. The assistant replies in English or Swahili and cites the relevant provisions.',
  'chat.loading': 'Searching statutes…',
  'chat.placeholder.en': 'Ask about Kenyan law in English…',
  'chat.placeholder.sw': 'Uliza kuhusu sheria za Kenya kwa Kiswahili…',
  'chat.placeholder.mixed': 'Ask in English or Swahili — Uliza kwa Kiingereza au Kiswahili…',
  'chat.disclaimer': 'Not legal advice. Verify citations against official sources.',

  'composer.label': 'Your question',
  'composer.send': 'Send',

  'lang.toggle.tooltip': 'Switch interface language',

  'source.title': 'Source page',
  'source.counter': '{current} of {total}',
  'source.empty.noCitations': 'Ask a question to see the source pages from Kenyan statutes here.',
  'source.empty.unavailable': 'No source page available for this citation.',

  'citations.heading': 'Citations',

  'sidebar.label': 'Chat history',
  'sidebar.heading': 'Your chats',
  'sidebar.list.label': 'Previous chats',
  'sidebar.newChat': 'New chat',
  'sidebar.collapse': 'Collapse sidebar',
  'sidebar.expand': 'Expand sidebar',
  'sidebar.loading': 'Loading chats…',
  'sidebar.empty': 'No chats yet. Ask something to start one.',
  'sidebar.signedOut.prompt': 'Sign in to keep a history of your chats.',
  'sidebar.renameTooltip': 'Rename chat',
  'sidebar.renameAria': 'Chat title',

  'auth.signIn': 'Sign in',
  'auth.signUp': 'Create account',

  'errors.generic': 'Something went wrong',
} as const

export type I18nKey = keyof typeof en

const sw: Record<I18nKey, string> = {
  'header.title': 'Haki AI',
  'header.subtitle': 'Msaada wa kisheria Kenya — majibu yenye kunukuu Sheria, Sura, na Kifungu',
  'header.corpus': 'Katiba 2010 · Sheria ya Ajira 2007 · Sheria ya Ardhi 2012',

  'chat.empty':
    'Uliza swali kuhusu sheria za Kenya. Msaidizi atajibu kwa Kiingereza au Kiswahili na atanukuu vifungu husika.',
  'chat.loading': 'Natafuta katika sheria…',
  'chat.placeholder.en': 'Ask about Kenyan law in English…',
  'chat.placeholder.sw': 'Uliza kuhusu sheria za Kenya kwa Kiswahili…',
  'chat.placeholder.mixed': 'Ask in English or Swahili — Uliza kwa Kiingereza au Kiswahili…',
  'chat.disclaimer': 'Si ushauri wa kisheria. Hakikisha manukuu dhidi ya vyanzo rasmi.',

  'composer.label': 'Swali lako',
  'composer.send': 'Tuma',

  'lang.toggle.tooltip': 'Badilisha lugha ya kiolesura',

  'source.title': 'Ukurasa wa chanzo',
  'source.counter': '{current} kati ya {total}',
  'source.empty.noCitations': 'Uliza swali kuona kurasa za vyanzo vya sheria za Kenya hapa.',
  'source.empty.unavailable': 'Hakuna ukurasa wa chanzo kwa manukuu haya.',

  'citations.heading': 'Manukuu',

  'sidebar.label': 'Historia ya mazungumzo',
  'sidebar.heading': 'Mazungumzo yako',
  'sidebar.list.label': 'Mazungumzo ya awali',
  'sidebar.newChat': 'Mazungumzo mapya',
  'sidebar.collapse': 'Funga kando',
  'sidebar.expand': 'Fungua kando',
  'sidebar.loading': 'Inapakia mazungumzo…',
  'sidebar.empty': 'Hakuna mazungumzo bado. Uliza swali kuanza.',
  'sidebar.signedOut.prompt': 'Ingia ili kuhifadhi historia ya mazungumzo yako.',
  'sidebar.renameTooltip': 'Badilisha jina',
  'sidebar.renameAria': 'Jina la mazungumzo',

  'auth.signIn': 'Ingia',
  'auth.signUp': 'Fungua akaunti',

  'errors.generic': 'Hitilafu imetokea',
}

export const DICTIONARIES: Record<Locale, Record<I18nKey, string>> = { en, sw }

export function translate(locale: Locale, key: I18nKey, vars?: Record<string, string | number>): string {
  const dict = DICTIONARIES[locale] ?? DICTIONARIES.en
  let out = dict[key] ?? en[key] ?? key
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      out = out.replace(`{${k}}`, String(v))
    }
  }
  return out
}
