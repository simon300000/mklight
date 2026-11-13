import { useState } from 'react'
import { cn } from '@/lib/utils'
import { NumberInput } from '../SettingsForm'

/** Inspector form building blocks: grouped sections with small controls. */
export function Section({ title, children, footer, accessory }: { title?: string; children: React.ReactNode; footer?: React.ReactNode; accessory?: React.ReactNode }) {
  return (
    <section className="mb-3">
      {(title || accessory) && (
        <div className="mb-1 flex items-center justify-between px-1">
          {title && <h3 className="text-[12px] font-semibold text-secondary">{title}</h3>}
          {accessory}
        </div>
      )}
      <div className="rounded-lg bg-content px-2.5 py-1.5 shadow-[0_0_0_0.5px_var(--separator)]">{children}</div>
      {footer && <p className="mt-1 px-1 text-[11px] text-secondary">{footer}</p>}
    </section>
  )
}

export function FormRow({ label, help, unit, children, className }: { label: React.ReactNode; help?: string; unit?: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn('grid grid-cols-[minmax(0,0.8fr)_minmax(0,1.6fr)_2rem] items-center gap-x-2 py-[3px]', className)} title={help}>
      <span className="truncate text-secondary">{label}</span>
      <span className="flex min-w-0 justify-end">{children}</span>
      <span className="truncate text-[11px] text-secondary">{unit}</span>
    </div>
  )
}

export function NumberRow({
  label,
  value,
  onCommit,
  unit,
  help,
  nullable,
  integer,
  min,
}: {
  label: string
  value: number | null
  onCommit: (v: number | null) => void
  unit?: string
  help?: string
  nullable?: boolean
  integer?: boolean
  min?: number
}) {
  return (
    <FormRow label={label} help={help} unit={unit}>
      <NumberInput ariaLabel={label} value={value} onCommit={onCommit} nullable={nullable} integer={integer} min={min} />
    </FormRow>
  )
}

/** A text field that commits on Enter and when the focus leaves; Escape reverts. */
export function TextInput({ value, onCommit, placeholder, ariaLabel }: { value: string; onCommit: (v: string) => void; placeholder?: string; ariaLabel?: string }) {
  const [text, setText] = useState<string | null>(null)
  return (
    <input
      aria-label={ariaLabel}
      className="h-6 w-full min-w-0 rounded-md border border-control-border bg-control px-1.5 text-[13px] outline-none focus:border-accent"
      value={text ?? value}
      placeholder={placeholder}
      onFocus={() => setText(value)}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        if (text != null && text !== value) onCommit(text)
        setText(null)
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        if (e.key === 'Escape') {
          setText(value)
          requestAnimationFrame(() => (e.target as HTMLInputElement).blur())
        }
      }}
    />
  )
}
