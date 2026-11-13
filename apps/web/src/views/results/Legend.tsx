// A quiet legend row under (or above) a plot.
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

export function Legend(props: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 px-3 py-1.5 text-[11px] text-secondary', props.className)}>
      {props.children}
    </div>
  )
}

export function Swatch(props: { color: string; label: ReactNode; kind?: 'dot' | 'line' | 'dash' | 'thin' }) {
  const kind = props.kind ?? 'line'
  return (
    <span className="inline-flex items-center gap-1.5">
      {kind === 'dot' ? (
        <span className="inline-block size-2 rounded-full" style={{ background: props.color }} />
      ) : (
        <svg width="18" height="6" aria-hidden>
          <line
            x1="0"
            y1="3"
            x2="18"
            y2="3"
            stroke={props.color}
            strokeWidth={kind === 'thin' ? 1 : 1.5}
            strokeDasharray={kind === 'dash' ? '4 3' : undefined}
          />
        </svg>
      )}
      <span>{props.label}</span>
    </span>
  )
}
