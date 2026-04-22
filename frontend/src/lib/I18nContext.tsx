import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { DetectedLanguage } from '../types/chat'
import { DICTIONARIES, translate, type I18nKey, type Locale } from './i18n'

/**
 * Locale manager for the Haki AI UI.
 *
 * Two knobs:
 *  - explicit user toggle via `LanguageToggle` (persisted in localStorage,
 *    sets `manualLocaleOverride` so the backend's language detection can
 *    never fight the user)
 *  - automatic switch when the `/chat` response comes back with
 *    `language: "swahili"` and the user has NOT manually overridden. We
 *    only flip forward to Swahili automatically — switching back to
 *    English silently would be jarring so we require an explicit toggle.
 */

const LOCALE_KEY = 'haki.locale'
const OVERRIDE_KEY = 'haki.localeOverride'

function readInitialLocale(): Locale {
  if (typeof localStorage === 'undefined') return 'en'
  const stored = localStorage.getItem(LOCALE_KEY)
  return stored === 'sw' || stored === 'en' ? stored : 'en'
}

function readOverride(): boolean {
  if (typeof localStorage === 'undefined') return false
  return localStorage.getItem(OVERRIDE_KEY) === '1'
}

interface I18nContextValue {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: I18nKey, vars?: Record<string, string | number>) => string
  onDetectedLanguage: (language: DetectedLanguage | undefined) => void
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => readInitialLocale())
  const overrideRef = useRef<boolean>(readOverride())

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = locale === 'sw' ? 'sw' : 'en'
    }
  }, [locale])

  const setLocale = useCallback((next: Locale) => {
    if (!(next in DICTIONARIES)) return
    overrideRef.current = true
    try {
      localStorage.setItem(LOCALE_KEY, next)
      localStorage.setItem(OVERRIDE_KEY, '1')
    } catch {
      // localStorage can throw in private mode — the UI still works in-memory.
    }
    setLocaleState(next)
  }, [])

  const onDetectedLanguage = useCallback((language: DetectedLanguage | undefined) => {
    if (!language) return
    if (overrideRef.current) return
    const next: Locale | null = language === 'swahili' ? 'sw' : null
    if (next && next !== locale) {
      try {
        localStorage.setItem(LOCALE_KEY, next)
      } catch {
        // ignore
      }
      setLocaleState(next)
    }
  }, [locale])

  const t = useCallback(
    (key: I18nKey, vars?: Record<string, string | number>) => translate(locale, key, vars),
    [locale],
  )

  const value = useMemo<I18nContextValue>(
    () => ({ locale, setLocale, t, onDetectedLanguage }),
    [locale, setLocale, t, onDetectedLanguage],
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext)
  if (!ctx) {
    throw new Error('useI18n must be used inside an <I18nProvider>')
  }
  return ctx
}
