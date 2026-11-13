// Small controls of the Tolerance panel: a list field (levels, frequencies)
// that commits on Enter or blur and reverts text it cannot read, a checklist
// menu of indices (fields, wavelengths), and bars (share of the RSS, a
// perturbation within its limits, a compensator's used range).

import { ChevronsUpDown } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import { short } from '@/model/tolerance'

/** Numbers typed as a list: "50, 90, 97.7". Text that is not such a list reverts with a note. */
export function ListInput({
  values,
  parse,
  onCommit,
  placeholder,
  ariaLabel,
  invalidNote,
  className,
}: {
  values: number[]
  parse: (text: string) => number[] | null
  onCommit: (v: number[]) => void
  placeholder?: string
  ariaLabel: string
  invalidNote: string
  className?: string
}) {
  const [text, setText] = useState<string | null>(null)
  const [invalid, setInvalid] = useState(false)
  const shown = values.map(short).join(', ')
  const commit = () => {
    if (text == null) return
    const t = text.trim()
    setText(null)
    if (t === shown) return
    const v = parse(t)
    if (!v) {
      setInvalid(t !== '')
      return
    }
    setInvalid(false)
    if (JSON.stringify(v) !== JSON.stringify(values)) onCommit(v)
  }
  return (
    <span className={cn('relative inline-flex flex-col items-end', className)}>
      <input
        aria-label={ariaLabel}
        aria-invalid={invalid || undefined}
        className={cn(
          'num h-6 w-full min-w-0 rounded-md border bg-control px-1.5 text-right text-[13px] outline-none focus:border-accent',
          invalid ? 'border-danger' : 'border-control-border',
        )}
        value={text ?? shown}
        placeholder={placeholder}
        onFocus={(e) => {
          setText(shown.replace(/−/g, '-'))
          requestAnimationFrame(() => e.target.select())
        }}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          if (e.key === 'Escape') {
            setText(null)
            setInvalid(false)
            requestAnimationFrame(() => (e.target as HTMLInputElement).blur())
          }
        }}
      />
      {invalid && <span className="mt-0.5 text-[10px] text-danger">{invalidNote}</span>}
    </span>
  )
}

/** A menu of checkable items (fields, wavelengths); `null` is every item. */
export function IndexChecklist({
  title,
  items,
  selected,
  onChange,
}: {
  title: string
  items: string[]
  selected: number[] | null
  onChange: (v: number[] | null) => void
}) {
  const on = (k: number) => selected == null || selected.includes(k)
  const label = selected == null ? `${title}: all` : `${title}: ${selected.length ? selected.map((k) => k + 1).join(', ') : 'none'}`
  const toggle = (k: number) => {
    const cur = selected ?? items.map((_, i) => i)
    const next = cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k].sort((a, b) => a - b)
    if (!next.length) return
    onChange(next.length === items.length ? null : next)
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="sm" className="max-w-full justify-between" aria-label={title}>
          <span className="truncate">{label}</span>
          <ChevronsUpDown className="size-3 text-secondary" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuItem disabled={selected == null} onSelect={() => onChange(null)}>
          All
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        {items.map((t, k) => (
          <DropdownMenuCheckboxItem
            key={k}
            checked={on(k)}
            onSelect={(e) => {
              e.preventDefault()
              toggle(k)
            }}
          >
            {t}
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** A share of the RSS as a bar with its percentage. */
export function ShareBar({ fraction, highlighted }: { fraction: number; highlighted?: boolean }) {
  const f = Math.min(Math.max(fraction, 0), 1)
  return (
    <span className="flex items-center gap-1.5" aria-hidden>
      <span className="relative h-[5px] w-[90px] overflow-hidden rounded-full bg-selected">
        <span
          className={cn('absolute inset-y-0 left-0 rounded-full', highlighted ? 'bg-warning' : 'bg-accent')}
          style={{ width: f > 0 ? `max(2px, ${f * 100}%)` : 0 }}
        />
      </span>
      <span className="num w-[34px] text-right text-[10px] text-secondary">{percentShort(fraction)}</span>
    </span>
  )
}

export function percentShort(f: number): string {
  if (!Number.isFinite(f)) return '—'
  const p = f * 100
  if (p > 0 && p < 0.5) return '<1 %'
  return `${Math.round(p)} %`
}

/** A perturbation within its limits: a bar from the centre, right for +, left for −. */
export function LimitBar({ fraction }: { fraction: number }) {
  const f = Math.min(Math.max(fraction, -1), 1)
  const w = Math.max(Math.abs(f) * 50, 1)
  return (
    <span className="relative inline-block h-[6px] w-[44px] shrink-0 rounded-full bg-selected" aria-hidden>
      <span className="absolute inset-y-0 bg-accent" style={{ left: f >= 0 ? '50%' : `${50 - w}%`, width: `${w}%` }} />
      <span className="absolute inset-y-0 left-1/2 w-px bg-label/50" />
    </span>
  )
}

/** The used part of a compensator's ±range, centred on nominal. */
export function RangeUsageBar({ lo, hi, range }: { lo: number; hi: number; range: number }) {
  const x0 = 50 + Math.max(lo / range, -1) * 50
  const x1 = 50 + Math.min(hi / range, 1) * 50
  return (
    <span className="relative block h-[10px] w-full" aria-hidden>
      <span className="absolute inset-x-0 top-1/2 h-[5px] -translate-y-1/2 rounded-full bg-selected" />
      <span className="absolute top-1/2 h-[5px] -translate-y-1/2 rounded-full bg-accent" style={{ left: `${x0}%`, width: `max(2px, ${x1 - x0}%)` }} />
      <span className="absolute inset-y-0 left-1/2 w-px bg-label/60" />
    </span>
  )
}

export function Hint({ children, className }: { children: React.ReactNode; className?: string }) {
  return <p className={cn('text-[11px] leading-snug text-secondary', className)}>{children}</p>
}

/** A key–value list. */
export function KeyValues({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[12px]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-secondary">{k}</dt>
          <dd className="num truncate text-right">{v}</dd>
        </div>
      ))}
    </dl>
  )
}
