import { useState } from 'react'
import type { ParamPath } from '@/model/params'
import { cn } from '@/lib/utils'
import { useValueFocus } from '@/state/commands'
import { useView } from '@/state/view'
import { NumberInput } from '../SettingsForm'
import { SolveBadge, VariableToggle } from './VariableToggle'

/** Inspector form building blocks: grouped sections with small controls. */
export function Section({
  title,
  children,
  footer,
  accessory,
  testId,
}: {
  title?: string
  children: React.ReactNode
  footer?: React.ReactNode
  accessory?: React.ReactNode
  testId?: string
}) {
  return (
    <section className="mb-3" data-testid={testId}>
      {(title || accessory) && (
        <div className="mb-1 flex min-h-6 items-center justify-between px-1">
          {title && <h3 className="text-[12px] font-semibold text-secondary">{title}</h3>}
          {accessory}
        </div>
      )}
      <div className="rounded-lg bg-content px-2.5 py-1.5 shadow-[0_0_0_0.5px_var(--separator)]">{children}</div>
      {footer && <p className="mt-1 px-1 text-[11px] text-secondary">{footer}</p>}
    </section>
  )
}

export function FormRow({
  label,
  help,
  unit,
  children,
  className,
  accessory,
}: {
  label: React.ReactNode
  help?: string
  unit?: string
  children: React.ReactNode
  className?: string
  /** Shown after the control (a variable toggle, a solve badge). */
  accessory?: React.ReactNode
}) {
  return (
    <div className={cn('grid grid-cols-[minmax(0,0.8fr)_minmax(0,1.6fr)_2rem] items-center gap-x-2 py-[3px]', className)} title={help}>
      <span className="truncate text-secondary">{label}</span>
      <span className="flex min-w-0 items-center justify-end gap-1">
        {children}
        {accessory}
      </span>
      <span className="truncate text-[11px] text-secondary">{unit}</span>
    </div>
  )
}

export function NumberRow({
  label,
  detail,
  value,
  onCommit,
  unit,
  help,
  nullable,
  integer,
  min,
  param,
  validate,
}: {
  label: string
  /** A short note after the label (`melt`, the term's meaning). */
  detail?: string
  value: number | null
  onCommit: (v: number | null) => void
  unit?: string
  help?: string
  nullable?: boolean
  integer?: boolean
  min?: number
  /** The model parameter: adds its variable toggle and solve marker, and
   *  makes it the target of Toggle Variable while focused. */
  param?: ParamPath
  /** Why a typed value is refused (shown in a toast; the value reverts). */
  validate?: (v: number) => string | null
}) {
  const setFocus = useValueFocus((s) => s.setParam)
  const commit = (v: number | null) => {
    const why = v != null && validate ? validate(v) : null
    if (why) return useView.getState().toast(why, 'warning')
    onCommit(v)
  }
  return (
    <FormRow
      label={
        detail ? (
          <>
            {label} <span className="text-[11px] text-tertiary">{detail}</span>
          </>
        ) : (
          label
        )
      }
      help={help}
      unit={unit}
      accessory={
        param ? (
          <>
            <SolveBadge path={param} />
            <VariableToggle path={param} />
          </>
        ) : undefined
      }
    >
      <span className="contents" onFocus={() => param && setFocus(param)}>
        <NumberInput ariaLabel={label} value={value} onCommit={commit} nullable={nullable} integer={integer} min={min} />
      </span>
    </FormRow>
  )
}

/** A text field that commits on Enter and when the focus leaves; Escape reverts.
 *  With `suggestions`, typing offers matching entries (↑/↓ to choose). */
export function TextInput({
  value,
  onCommit,
  placeholder,
  ariaLabel,
  suggestions,
}: {
  value: string
  onCommit: (v: string) => void
  placeholder?: string
  ariaLabel?: string
  suggestions?: (text: string) => string[]
}) {
  const [text, setText] = useState<string | null>(null)
  const [hi, setHi] = useState(0)
  const list = text != null && suggestions ? suggestions(text) : []
  const commit = (t: string | null) => {
    if (t != null && t !== value) onCommit(t)
    setText(null)
  }
  return (
    <span className="relative w-full min-w-0">
      <input
        aria-label={ariaLabel}
        aria-autocomplete={suggestions ? 'list' : undefined}
        className="h-6 w-full min-w-0 rounded-md border border-control-border bg-control px-1.5 text-[13px] outline-none focus:border-accent"
        value={text ?? value}
        placeholder={placeholder}
        onFocus={() => setText(value)}
        onChange={(e) => {
          setText(e.target.value)
          setHi(0)
        }}
        onBlur={() => commit(text)}
        onKeyDown={(e) => {
          if (list.length && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
            e.preventDefault()
            setHi((h) => (h + (e.key === 'ArrowDown' ? 1 : list.length - 1)) % list.length)
            return
          }
          if (e.key === 'Enter') {
            const input = e.target as HTMLInputElement
            if (list.length) {
              e.preventDefault()
              commit(list[hi])
              requestAnimationFrame(() => input.blur())
            } else input.blur()
          }
          if (e.key === 'Escape') {
            setText(value)
            requestAnimationFrame(() => (e.target as HTMLInputElement).blur())
          }
        }}
      />
      {list.length > 0 && (
        <ul role="listbox" className="glass-popover absolute left-0 right-0 top-full z-50 mt-1 max-h-48 overflow-y-auto rounded-md p-1 text-[12px]">
          {list.map((g, k) => (
            <li
              key={g}
              role="option"
              aria-selected={k === hi}
              className={cn('cursor-default rounded px-1.5 py-[2px]', k === hi && 'bg-accent text-on-accent')}
              onMouseDown={(e) => {
                e.preventDefault()
                commit(g)
                ;(document.activeElement as HTMLElement | null)?.blur()
              }}
            >
              {g}
            </li>
          ))}
        </ul>
      )}
    </span>
  )
}
