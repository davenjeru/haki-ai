import { useI18n } from '../lib/I18nContext'

/**
 * EN | SW toggle shown in the header.
 *
 * Tapping a button flips the UI locale AND sets `manualLocaleOverride`
 * so future `/chat` responses don't silently switch the interface back.
 */
export function LanguageToggle() {
  const { locale, setLocale, t } = useI18n()
  return (
    <div
      className="inline-flex rounded-full border border-border bg-elevated p-[2px]"
      role="group"
      aria-label={t('lang.toggle.tooltip')}
    >
      <ToggleButton current={locale} value="en" onClick={() => setLocale('en')}>
        EN
      </ToggleButton>
      <ToggleButton current={locale} value="sw" onClick={() => setLocale('sw')}>
        SW
      </ToggleButton>
    </div>
  )
}

interface ToggleButtonProps {
  current: 'en' | 'sw'
  value: 'en' | 'sw'
  children: React.ReactNode
  onClick: () => void
}

function ToggleButton({ current, value, children, onClick }: ToggleButtonProps) {
  const active = current === value
  const className = [
    'text-[0.72rem] font-semibold tracking-[0.08em] px-3 py-[0.3rem] rounded-full cursor-pointer transition-colors',
    active
      ? 'bg-[linear-gradient(145deg,#40916c_0%,#2d6a4f_100%)] text-white'
      : 'text-muted hover:text-strong',
  ].join(' ')
  return (
    <button
      type="button"
      aria-pressed={active}
      className={className}
      onClick={onClick}
    >
      {children}
    </button>
  )
}
