// A thin uPlot wrapper that fills its parent and follows its size, with a
// hover readout: a vertical guide at the pointer and a small label with x and
// each series' y there (a tap shows it on touch screens).
import { useEffect, useRef, useState } from 'react'
import uPlot from 'uplot'
import { cssVar } from '@/lib/theme'
import { sig } from '@/model/format'
import { cn } from '@/lib/utils'
import { FONT, useSize } from './common'
import { Readout, type ReadoutRow } from './Readout'

let canvasOK = false
/** uPlot needs a 2-D canvas (absent in jsdom). */
function canvasAvailable(): boolean {
  if (!canvasOK) {
    try {
      canvasOK = !!document.createElement('canvas').getContext('2d')
    } catch {
      canvasOK = false
    }
  }
  return canvasOK
}

export type PlotOptions = Omit<uPlot.Options, 'width' | 'height'>

interface Hover {
  /** Pointer, relative to the wrapper. */
  x: number
  y: number
  title: string
  rows: ReadoutRow[]
}

const strokeOf = (s: uPlot.Series): string | undefined => {
  if (typeof s.stroke === 'string' && (s.width ?? 1) > 0) return s.stroke
  const f = s.points?.fill
  return typeof f === 'string' ? f : typeof s.stroke === 'string' ? s.stroke : undefined
}

/** The readout at data index `idx`: x with its axis label, then each shown
 *  series' y (repeated series, such as a curve and its sample dots, once). */
export function readoutAt(u: Pick<uPlot, 'data' | 'series' | 'axes'>, idx: number): { title: string; rows: ReadoutRow[] } | null {
  const x = u.data[0]?.[idx]
  if (x == null || !Number.isFinite(x)) return null
  const xLabel = typeof u.axes?.[0]?.label === 'string' ? u.axes[0].label : 'x'
  const rows: ReadoutRow[] = []
  const seen = new Set<string>()
  for (let i = 1; i < u.series.length; i++) {
    const s = u.series[i]
    if (s.show === false) continue
    const v = u.data[i]?.[idx]
    if (v == null || !Number.isFinite(v)) continue
    const color = strokeOf(s)
    const value = sig(v, 4)
    const key = `${color}|${value}`
    if (seen.has(key)) continue
    seen.add(key)
    rows.push({ label: typeof s.label === 'string' && s.label ? s.label : undefined, value, color, dash: !!s.dash?.length })
  }
  return { title: `${xLabel.replace(/\s*\(.*\)$/, '')} ${sig(x, 4)}`, rows }
}

export function UPlot(props: { options: PlotOptions; data: uPlot.AlignedData; label: string; className?: string }) {
  const [ref, size] = useSize<HTMLDivElement>()
  const plot = useRef<uPlot | null>(null)
  const [hover, setHover] = useState<Hover | null>(null)
  const { options, data } = props

  useEffect(() => {
    const el = ref.current
    if (!el || size.w <= 0 || size.h <= 0 || !canvasAvailable()) return
    const onCursor = (u: uPlot) => {
      const { left, top, idx } = u.cursor
      if (idx == null || left == null || top == null || left < 0 || top < 0) return setHover(null)
      const r = readoutAt(u, idx)
      if (!r) return setHover(null)
      const o = u.over.getBoundingClientRect(), b = el.getBoundingClientRect()
      setHover({ x: o.left - b.left + left, y: o.top - b.top + top, ...r })
    }
    const u = new uPlot(
      {
        ...options,
        width: size.w,
        height: size.h,
        // Hover: a guide and the readout; no drag zoom, no point markers.
        cursor: { show: true, x: true, y: false, points: { show: false }, drag: { x: false, y: false, setScale: false }, ...options.cursor },
        hooks: { ...options.hooks, setCursor: [...(options.hooks?.setCursor ?? []), onCursor] },
      },
      data,
      el,
    )
    const guide = u.root.querySelector<HTMLElement>('.u-cursor-x')
    if (guide) guide.style.borderRight = `1px solid ${cssVar('--secondary') || '#888'}`
    if (guide) guide.style.opacity = '0.5'
    u.root.querySelector('canvas')?.setAttribute('role', 'img')
    u.root.querySelector('canvas')?.setAttribute('aria-label', props.label)
    plot.current = u
    return () => {
      u.destroy()
      plot.current = null
      setHover(null)
    }
    // Size changes are applied by setSize below, not by rebuilding.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options, data, size.w > 0 && size.h > 0])

  useEffect(() => {
    if (plot.current && size.w > 0 && size.h > 0) plot.current.setSize({ width: size.w, height: size.h })
  }, [size.w, size.h])

  // A tap on a touch screen places the cursor (there is no hover).
  const onPointerDown = (e: React.PointerEvent) => {
    const u = plot.current
    if (!u || e.pointerType !== 'touch') return
    const o = u.over.getBoundingClientRect()
    u.setCursor({ left: e.clientX - o.left, top: e.clientY - o.top })
  }

  return (
    <div className={cn('relative h-full w-full min-w-0 overflow-hidden', props.className)} onPointerDown={onPointerDown}>
      <div ref={ref} className="h-full w-full" />
      {hover && <Readout x={hover.x} y={hover.y} width={size.w} title={hover.title} rows={hover.rows} />}
    </div>
  )
}

/** Calm axis styling: secondary labels, light grid, short light ticks, few ticks. */
export function axis(label: string, extra: Partial<uPlot.Axis> = {}): uPlot.Axis {
  const grid = cssVar('--grid') || 'rgba(128,128,128,0.2)'
  return {
    label,
    stroke: cssVar('--secondary') || '#888',
    font: FONT,
    labelFont: FONT,
    labelSize: 16,
    labelGap: 2,
    size: 36,
    gap: 3,
    space: 44,
    grid: { stroke: grid, width: 1 },
    ticks: { stroke: grid, width: 1, size: 3 },
    ...extra,
  }
}

export const LINE = 1.25
export const DASH = [5, 4]

/** Non-finite values become null so lines break there. */
export const gaps = (vs: ArrayLike<number>): (number | null)[] => Array.from(vs, (v) => (Number.isFinite(v) ? v : null))

/** The charts' cursor: the wrapper adds the guide and the readout. */
export const calmCursor: uPlot.Cursor = {}
