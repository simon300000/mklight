// A thin uPlot wrapper that fills its parent and follows its size.
import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import { cssVar } from '@/lib/theme'
import { cn } from '@/lib/utils'
import { FONT, useSize } from './common'

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

export function UPlot(props: { options: PlotOptions; data: uPlot.AlignedData; label: string; className?: string }) {
  const [ref, size] = useSize<HTMLDivElement>()
  const plot = useRef<uPlot | null>(null)
  const { options, data } = props

  useEffect(() => {
    const el = ref.current
    if (!el || size.w <= 0 || size.h <= 0 || !canvasAvailable()) return
    const u = new uPlot({ ...options, width: size.w, height: size.h }, data, el)
    u.root.querySelector('canvas')?.setAttribute('role', 'img')
    u.root.querySelector('canvas')?.setAttribute('aria-label', props.label)
    plot.current = u
    return () => {
      u.destroy()
      plot.current = null
    }
    // Size changes are applied by setSize below, not by rebuilding.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options, data, size.w > 0 && size.h > 0])

  useEffect(() => {
    if (plot.current && size.w > 0 && size.h > 0) plot.current.setSize({ width: size.w, height: size.h })
  }, [size.w, size.h])

  return <div ref={ref} className={cn('h-full w-full min-w-0 overflow-hidden', props.className)} />
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

export const calmCursor: uPlot.Cursor = { show: false }
