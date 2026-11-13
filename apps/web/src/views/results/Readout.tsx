// The hover readout of the charts and maps: a small glass label beside the
// pointer with the value(s) under it (the macOS charts show values on hover).
// It never takes the pointer, and it flips to the other side of the pointer
// near the right and top edges so it stays inside its plot.
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

export interface ReadoutRow {
  /** Series or quantity name; may be left out when the colour says it. */
  label?: ReactNode
  value: string
  color?: string
  dash?: boolean
}

export function Readout(props: { x: number; y: number; width: number; title?: ReactNode; rows: ReadoutRow[]; className?: string }) {
  const { x, y, width, title, rows } = props
  if (!rows.length && !title) return null
  const left = x > width / 2
  const below = y < 56
  return (
    <div
      className={cn('glass-popover pointer-events-none absolute z-10 max-w-[240px] rounded-md px-1.5 py-1 text-[11px] leading-[15px] whitespace-nowrap text-label', props.className)}
      style={{
        left: x,
        top: y,
        transform: `translate(${left ? 'calc(-100% - 10px)' : '10px'}, ${below ? '12px' : 'calc(-100% - 8px)'})`,
      }}
      data-testid="hover-readout"
      aria-hidden
    >
      {title != null && <div className="num text-secondary">{title}</div>}
      {rows.map((r, k) => (
        <div key={k} className="flex items-center gap-1.5">
          {r.color && (
            <span
              className="inline-block h-0 w-2.5 shrink-0"
              style={{ borderTop: `${r.dash ? 1.5 : 2}px ${r.dash ? 'dashed' : 'solid'} ${r.color}` }}
            />
          )}
          {r.label != null && <span className="min-w-0 truncate text-secondary">{r.label}</span>}
          <span className="num ml-auto pl-1 font-medium">{r.value}</span>
        </div>
      ))}
    </div>
  )
}

/** A thin vertical guide at the pointer's x inside a plot box. */
export function Guide(props: { x: number; top: number; height: number }) {
  return <div className="pointer-events-none absolute w-px bg-secondary/50" style={{ left: Math.round(props.x), top: props.top, height: props.height }} aria-hidden />
}
