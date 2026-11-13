// The Monte-Carlo distribution (TolerancePanel.swift `MonteCarloChart`): a
// histogram of the trials' shown values, the cumulative share of builds on the
// right axis, the nominal design and the engine's levels marked; a click on a
// bar selects it (its trials are listed), and the previewed trial is marked
// under the axis. Drawn on a canvas in the theme's colours.

import { useEffect, useMemo, useRef } from 'react'
import { cssVar, seriesColor, useDark } from '@/lib/theme'
import { DISPLAY, histogramBins, labelledLevels, short, type Bin, type ToleranceResult } from '@/model/tolerance'
import { FONT, niceTicks, setupCanvas, useSize } from '../../results/common'

const PAD = { left: 38, right: 36, top: 18, bottom: 30 }

export function MonteCarloChart({
  result,
  selection,
  onSelect,
  marked,
}: {
  result: ToleranceResult
  selection: Bin | null
  onSelect: (b: Bin | null) => void
  /** The shown value of the previewed trial. */
  marked: number | null
}) {
  const [box, size] = useSize<HTMLDivElement>()
  const canvas = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  const d = result.display
  const bins = useMemo(() => histogramBins(result), [result])
  const shown = useMemo(() => result.criteria.map((c) => DISPLAY.value(d, c)).filter(Number.isFinite), [result, d])
  const nominal = Number.isFinite(result.nominal) ? DISPLAY.value(d, result.nominal) : null
  const domain = useMemo((): [number, number] => {
    const all = [...shown, ...(nominal != null && Number.isFinite(nominal) ? [nominal] : []), ...bins.flatMap((b) => [b.lo, b.hi])]
    if (!all.length) return [0, 1]
    const lo = Math.min(...all), hi = Math.max(...all)
    const pad = Math.max((hi - lo) * 0.06, Math.abs(hi) * 1e-6, 1e-12)
    return [lo - pad, hi + pad]
  }, [shown, nominal, bins])
  const peak = Math.max(1, ...bins.map((b) => b.count))

  useEffect(() => {
    const ctx = setupCanvas(canvas.current, size.w, size.h)
    if (!ctx) return
    const W = size.w, H = size.h
    const pw = W - PAD.left - PAD.right, ph = H - PAD.top - PAD.bottom
    if (pw <= 10 || ph <= 10) return
    const X = (v: number) => PAD.left + ((v - domain[0]) / (domain[1] - domain[0])) * pw
    const Y = (n: number) => PAD.top + ph - (n / peak) * ph
    const label = cssVar('--label') || (dark ? '#fff' : '#000')
    const secondary = cssVar('--secondary') || '#888'
    const separator = cssVar('--separator') || '#ccc'
    const accent = cssVar('--accent') || '#0a74ff'
    ctx.font = FONT
    ctx.lineWidth = 1
    // Grid and left axis (trials).
    ctx.fillStyle = secondary
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (const t of niceTicks(0, peak, 4).filter((t) => Number.isInteger(t))) {
      const y = Math.round(Y(t)) + 0.5
      ctx.strokeStyle = separator
      ctx.beginPath()
      ctx.moveTo(PAD.left, y)
      ctx.lineTo(PAD.left + pw, y)
      ctx.stroke()
      ctx.fillText(String(t), PAD.left - 5, y)
    }
    // Right axis: cumulative share.
    ctx.textAlign = 'left'
    for (const f of [0, 0.5, 1]) ctx.fillText(`${Math.round(f * 100)}%`, PAD.left + pw + 5, Y(f * peak))
    // Bottom axis.
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (const t of niceTicks(domain[0], domain[1], Math.max(2, Math.floor(pw / 70)))) ctx.fillText(DISPLAY.format(d, t), X(t), PAD.top + ph + 4)
    const unit = DISPLAY.unit(d)
    ctx.fillText(DISPLAY.title(d) + (unit ? ` (${unit})` : ''), PAD.left + pw / 2, PAD.top + ph + 16)
    ctx.save()
    ctx.translate(10, PAD.top + ph / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textBaseline = 'middle'
    ctx.fillText('Trials', 0, 0)
    ctx.restore()
    // Bars.
    const base = seriesColor(0, dark)
    for (const b of bins) {
      const isSel = selection != null && Math.abs(selection.lo - b.lo) < (b.hi - b.lo) * 1e-6
      const w = (b.hi - b.lo) * 0.92
      const x0 = X(b.center - w / 2), x1 = X(b.center + w / 2)
      ctx.globalAlpha = isSel ? 1 : selection ? 0.4 : 0.75
      ctx.fillStyle = isSel ? accent : base
      const y = Y(b.count)
      ctx.fillRect(x0, y, Math.max(x1 - x0, 1), PAD.top + ph - y)
    }
    ctx.globalAlpha = 1
    // Cumulative share, rising from the best to the worst.
    const sorted = DISPLAY.higherIsBetter(d) ? [...shown].sort((a, b) => b - a) : [...shown].sort((a, b) => a - b)
    if (sorted.length) {
      ctx.strokeStyle = label
      ctx.globalAlpha = 0.75
      ctx.lineWidth = 1.4
      ctx.beginPath()
      let prev = 0
      sorted.forEach((v, k) => {
        const f = (k + 1) / sorted.length
        const x = X(v)
        if (k === 0) ctx.moveTo(x, Y(0))
        ctx.lineTo(x, Y(prev * peak))
        ctx.lineTo(x, Y(f * peak))
        prev = f
      })
      ctx.stroke()
      ctx.globalAlpha = 1
    }
    // Levels.
    const levels = labelledLevels(result.percentiles.map((p) => ({ p: p.percentile, v: DISPLAY.value(d, p.value) })), domain[1] - domain[0])
    ctx.lineWidth = 0.8
    ctx.setLineDash([3, 3])
    ctx.strokeStyle = secondary
    ctx.fillStyle = secondary
    ctx.textBaseline = 'bottom'
    for (const l of levels) {
      if (!Number.isFinite(l.v)) continue
      const x = Math.round(X(l.v)) + 0.5
      ctx.beginPath()
      ctx.moveTo(x, PAD.top)
      ctx.lineTo(x, PAD.top + ph)
      ctx.stroke()
      if (l.label) ctx.fillText(l.label, x, PAD.top - 3)
    }
    ctx.setLineDash([])
    // Nominal.
    if (nominal != null && Number.isFinite(nominal)) {
      ctx.strokeStyle = accent
      ctx.lineWidth = 1.6
      const x = X(nominal)
      ctx.beginPath()
      ctx.moveTo(x, PAD.top)
      ctx.lineTo(x, PAD.top + ph)
      ctx.stroke()
    }
    // The previewed trial.
    if (marked != null && Number.isFinite(marked)) {
      const x = X(marked), y = PAD.top + ph
      ctx.fillStyle = accent
      ctx.beginPath()
      ctx.moveTo(x, y - 7)
      ctx.lineTo(x - 5, y + 1)
      ctx.lineTo(x + 5, y + 1)
      ctx.closePath()
      ctx.fill()
    }
  }, [size.w, size.h, bins, shown, domain, peak, selection, marked, nominal, result, d, dark])

  const levelsText = result.percentiles
    .map((p) => `${short(p.percentile)} percent of builds ${DISPLAY.higherIsBetter(d) ? 'at least' : 'at most'} ${DISPLAY.formatWithUnit(d, DISPLAY.value(d, p.value))}`)
    .join(', ')

  return (
    <div ref={box} className="relative h-[190px] w-full" title="Click a bar to list its trials">
      <canvas
        ref={canvas}
        role="img"
        aria-label={`Distribution of ${shown.length} Monte Carlo trials of the ${DISPLAY.title(d)}. ${levelsText}`}
        data-testid="tolerance-histogram"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const pw = size.w - PAD.left - PAD.right
          const v = domain[0] + ((e.clientX - r.left - PAD.left) / pw) * (domain[1] - domain[0])
          const hit = bins.find((b) => b.count > 0 && v >= b.lo && v <= b.hi) ?? null
          onSelect(hit && selection && Math.abs(hit.lo - selection.lo) < 1e-12 ? null : hit)
        }}
      />
    </div>
  )
}
