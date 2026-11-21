import { useState } from 'react'
import { useT } from '@/i18n'
import { exact, parseNumber, sig } from '@/model/format'
import { cn } from '@/lib/utils'

/** A number field that commits on Enter and blur; Escape reverts. */
export function NumberInput({
  value,
  onCommit,
  min,
  max,
  integer,
  nullable,
  className,
  ariaLabel,
  placeholder,
}: {
  value: number | null
  onCommit: (v: number | null) => void
  min?: number
  max?: number
  integer?: boolean
  nullable?: boolean
  className?: string
  ariaLabel?: string
  placeholder?: string
}) {
  const t = useT()
  const [text, setText] = useState<string | null>(null)
  const shown = text ?? (value == null ? '' : sig(value, 8).replace('−', '-'))
  const commit = () => {
    if (text == null) return
    const tt = text.trim()
    setText(null)
    if (tt === '' || tt.toLowerCase() === 'auto' || tt.toLowerCase() === t('lens.settings.auto').toLowerCase()) {
      if (nullable) onCommit(null)
      return
    }
    let v = parseNumber(tt)
    if (v == null) return
    if (integer) v = Math.round(v)
    if (min != null && v < min) v = min
    if (max != null && v > max) v = max
    if (v !== value) onCommit(v)
  }
  return (
    <input
      aria-label={ariaLabel}
      inputMode="decimal"
      className={cn(
        'num h-6 w-full min-w-0 rounded-md border border-control-border bg-control px-1.5 text-right text-[13px] outline-none focus:border-accent',
        className,
      )}
      value={shown}
      placeholder={placeholder ?? (nullable ? t('lens.settings.auto') : '')}
      onFocus={(e) => {
        setText(value == null ? '' : exact(value))
        requestAnimationFrame(() => e.target.select())
      }}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        if (e.key === 'Escape') {
          setText(null)
          requestAnimationFrame(() => (e.target as HTMLInputElement).blur())
        }
      }}
    />
  )
}
