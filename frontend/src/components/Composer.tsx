import { useCallback, useRef } from 'react'
import { useI18n } from '../lib/I18nContext'

interface Props {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  disabled: boolean
  placeholder: string
}

export function Composer({ value, onChange, onSubmit, disabled, placeholder }: Props) {
  const { t } = useI18n()
  const ta = useRef<HTMLTextAreaElement>(null)

  const submit = useCallback(() => {
    if (disabled || !value.trim()) return
    onSubmit()
  }, [disabled, onSubmit, value])

  return (
    <div className="flex gap-[0.6rem] items-end">
      <label className="sr-only" htmlFor="haki-message">
        {t('composer.label')}
      </label>
      <textarea
        id="haki-message"
        ref={ta}
        className="flex-1 resize-none py-3 px-[0.9rem] rounded-[12px] border border-border bg-composer-bg text-[inherit] min-h-[3.25rem] focus:outline-2 focus:outline-[rgba(64,145,108,0.45)] focus:outline-offset-1 focus:border-accent-bright placeholder:text-placeholder"
        rows={2}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
          }
        }}
      />
      <button
        type="button"
        className="flex-shrink-0 min-w-[4.5rem] py-[0.65rem] px-4 border-0 rounded-[12px] bg-[linear-gradient(180deg,#40916c_0%,#2d6a4f_100%)] text-white font-semibold text-[0.95rem] cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed"
        disabled={disabled || !value.trim()}
        onClick={submit}
      >
        {disabled ? '…' : t('composer.send')}
      </button>
    </div>
  )
}
