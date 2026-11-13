// Shared pieces of the curve and table renderers: number formats as the
// macOS panes use them (`Fmt`), subscripted labels (`Subscripted`), JSON or
// buffer series, and two small canvas charts (lines over x/y, grouped bars)
// that follow the pane's size and the light or dark appearance.
import { Fragment, useEffect, useRef, type ReactNode } from 'react'
import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { nf, sig } from '@/model/format'
import type { System } from '@/model/system'
import { cssVar, useDark } from '@/lib/theme'
import { FONT, arr, buffer, isObj, niceTicks, nums, setupCanvas, systemFieldUnit, useSize } from '../../results/common'

// MARK: Formats

const MINUS = '−'
const signFix = (s: string) => (s.startsWith('-') ? MINUS + s.slice(1) : s)

/** Up to 6 significant digits, scientific below 1e-3 and from 1e6 (`Fmt.short`). */
export function short(v: number): string {
  if (Number.isNaN(v)) return '—'
  if (!Number.isFinite(v)) return v > 0 ? '∞' : MINUS + '∞'
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a < 1e-3 || a >= 1e6) return signFix(v.toExponential(3))
  return sig(v, 6)
}

/** A length in mm, shown in µm below 0.1 mm (`Fmt.length`); never "−0.00". */
export function length(mm: number): string {
  if (!Number.isFinite(mm)) return short(mm)
  let s = Math.abs(mm) < 0.1 ? `${(mm * 1000).toFixed(2)} µm` : `${mm.toFixed(4)} mm`
  if (/^-[0. ]*(µm|mm)$/.test(s)) s = s.slice(1)
  return signFix(s)
}

export const percent = (f: number, digits = 1) => (Number.isFinite(f) ? `${(f * 100).toFixed(digits)} %` : '—')

/** A signed value with fixed decimals; a value that rounds to zero has no sign. */
export function signed(v: number, decimals = 4): string {
  if (!Number.isFinite(v)) return short(v)
  const s = Math.abs(v).toFixed(decimals)
  if (Number(s) === 0) return ' ' + s
  return (v < 0 ? MINUS : '+') + s
}

export const valueUnit = (v: number, unit: string) => (unit && Number.isFinite(v) ? (unit === '°' ? short(v) + '°' : `${short(v)} ${unit}`) : short(v))

// MARK: Subscripts

/** `S_III spherical` with "III" as a subscript: a subscript runs from `_` to the end of its word. */
export function Sub({ text }: { text: string }) {
  return (
    <>
      {text.split(' ').map((w, k) => {
        const u = w.indexOf('_')
        return (
          <Fragment key={k}>
            {k > 0 && ' '}
            {u > 0 ? (
              <>
                {w.slice(0, u)}
                <sub className="text-[0.75em]">{w.slice(u + 1)}</sub>
              </>
            ) : (
              w
            )}
          </Fragment>
        )
      })}
    </>
  )
}

/** The spoken form of a subscripted label ("S I spherical"). */
export const spoken = (text: string) => text.replace(/_/g, ' ')

// MARK: Data access

/** A numeric series given inline as a JSON array or by buffer name. */
export function series(r: AnalysisResult, v: Json | undefined): number[] {
  if (Array.isArray(v)) return nums(v)
  const b = buffer(r, v)
  return b ? Array.from(b, Number) : []
}

export const objects = (v: Json | undefined): JsonObject[] => arr(v).filter(isObj)

/** "Field 2 · 14°" from an index and a value [x, y] (the y value unless x is set). */
export function fieldTitle(system: System, field: number, value: Json | undefined): string {
  const [x, y] = nums(value)
  const unit = systemFieldUnit(system)
  const u = unit === '°' ? '°' : ` ${unit}`
  let v = ''
  if (x !== undefined && y === undefined) v = sig(x, 4) + u
  else if (x !== undefined && y !== undefined) v = Number.isFinite(x) && x !== 0 ? `(${sig(x, 4)}, ${sig(y, 4)})${u}` : sig(y, 4) + u
  return v ? `Field ${field + 1} · ${v}` : `Field ${field + 1}`
}

export const finite = (vs: Iterable<number>) => [...vs].filter(Number.isFinite)
export const maxOf = (vs: Iterable<number>) => {
  const f = finite(vs)
  return f.length ? Math.max(...f) : NaN
}
export const minOf = (vs: Iterable<number>) => {
  const f = finite(vs)
  return f.length ? Math.min(...f) : NaN
}

export const intOf = (v: Json | undefined, fallback: number) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback)
export { nf }

// MARK: Layout pieces

export function Empty({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 p-4 text-center">
      <span className="text-[13px] font-medium">{title}</span>
      {detail && <span className="max-w-sm text-[12px] text-secondary">{detail}</span>}
    </div>
  )
}

/** A pane body: a wrapping legend row and a plot that takes the rest. */
export function PlotPane({ legend, children, top }: { legend?: ReactNode; children: ReactNode; top?: ReactNode }) {
  return (
    <div className="absolute inset-0 flex min-w-0 flex-col">
      {top}
      {legend && (
        <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-0.5 px-3 pt-1.5 text-[11px] text-secondary @min-[420px]:gap-x-4">
          {legend}
        </div>
      )}
      <div className="min-h-0 flex-1 px-2 pb-2">{children}</div>
    </div>
  )
}

export function Key({ color, label, kind = 'line' }: { color: string; label: ReactNode; kind?: 'dot' | 'line' | 'dash' | 'thin' | 'box' }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5">
      {kind === 'dot' || kind === 'box' ? (
        <span className={kind === 'dot' ? 'inline-block size-2 shrink-0 rounded-full' : 'inline-block size-2 shrink-0 rounded-[2px]'} style={{ background: color }} />
      ) : (
        <svg width="16" height="6" aria-hidden className="shrink-0">
          <line x1="0" y1="3" x2="16" y2="3" stroke={color} strokeWidth={kind === 'thin' ? 1 : 1.5} strokeDasharray={kind === 'dash' ? '4 3' : undefined} />
        </svg>
      )}
      <span className="truncate">{label}</span>
    </span>
  )
}

/** Label/value rows (`KeyValueGrid`). */
export function KeyValues({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <div className="grid grid-cols-[auto_auto] justify-start gap-x-6 gap-y-1 text-[13px] select-text">
      {rows.map(([k, v], i) => (
        <Fragment key={i}>
          <span className="text-secondary">{k}</span>
          <span className="num text-right">{v}</span>
        </Fragment>
      ))}
    </div>
  )
}

// MARK: Canvas charts

export interface XYSeries {
  x: ArrayLike<number>
  y: ArrayLike<number>
  color: string
  dash?: boolean
  width?: number
  /** Dots instead of a line. */
  dots?: boolean
}

export interface XYPlotProps {
  series: XYSeries[]
  xLabel: string
  yLabel: string
  xDomain?: [number, number]
  yDomain?: [number, number]
  /** Vertical rules with a small label at the top. */
  markers?: { x: number; label?: string; dash?: boolean }[]
  /** Horizontal rules. */
  hRules?: { y: number; dash?: boolean }[]
  zeroLine?: boolean
  label: string
}

function domainOf(vs: number[], pad = 0.04): [number, number] {
  const f = finite(vs)
  if (!f.length) return [0, 1]
  let lo = Math.min(...f), hi = Math.max(...f)
  if (hi - lo < 1e-12 * Math.max(1, Math.abs(hi))) {
    const d = Math.abs(hi) * 0.05 || 1
    lo -= d
    hi += d
  } else {
    const d = (hi - lo) * pad
    lo -= d
    hi += d
  }
  return [lo, hi]
}

/** Draws axes into a plot box and returns the data → pixel maps. */
function drawAxes(ctx: CanvasRenderingContext2D, W: number, H: number, xd: [number, number], yd: [number, number], xLabel: string, yLabel: string, xTicks = true) {
  const secondary = cssVar('--secondary') || '#888'
  const grid = cssVar('--grid') || 'rgba(128,128,128,0.2)'
  ctx.font = FONT
  const yTicks = niceTicks(yd[0], yd[1], Math.max(2, Math.floor((H - 44) / 44)))
  const tickText = yTicks.map((t) => sig(t, 4))
  const tw = Math.max(18, ...tickText.map((t) => ctx.measureText(t).width))
  const ml = Math.ceil(tw) + 22, mb = 32, mt = 10, mr = 12
  const pw = Math.max(1, W - ml - mr), ph = Math.max(1, H - mt - mb)
  const px = (v: number) => ml + ((v - xd[0]) / (xd[1] - xd[0])) * pw
  const py = (v: number) => mt + ph - ((v - yd[0]) / (yd[1] - yd[0])) * ph
  ctx.strokeStyle = grid
  ctx.lineWidth = 1
  ctx.fillStyle = secondary
  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'
  yTicks.forEach((t, k) => {
    const Y = Math.round(py(t)) + 0.5
    ctx.beginPath()
    ctx.moveTo(ml, Y)
    ctx.lineTo(ml + pw, Y)
    ctx.stroke()
    ctx.fillText(tickText[k], ml - 4, Math.min(Math.max(Y, mt + 5), mt + ph - 3))
  })
  if (xTicks) {
    ctx.textBaseline = 'top'
    const xt = niceTicks(xd[0], xd[1], Math.max(2, Math.floor(pw / 70)))
    xt.forEach((t, k) => {
      const X = Math.round(px(t)) + 0.5
      ctx.beginPath()
      ctx.moveTo(X, mt)
      ctx.lineTo(X, mt + ph)
      ctx.stroke()
      // End labels stay inside the plot box.
      ctx.textAlign = k === 0 && X - ml < 12 ? 'left' : k === xt.length - 1 && ml + pw - X < 12 ? 'right' : 'center'
      ctx.fillText(sig(t, 4), X, mt + ph + 4)
    })
  }
  ctx.strokeRect(ml + 0.5, mt + 0.5, pw - 1, ph - 1)
  ctx.textAlign = 'center'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText(xLabel, ml + pw / 2, H - 3)
  ctx.save()
  ctx.translate(11, mt + ph / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textBaseline = 'middle'
  ctx.fillText(yLabel, 0, 0)
  ctx.restore()
  return { px, py, ml, mt, pw, ph, secondary }
}

/** A canvas filling its parent, redrawn on size and appearance changes. */
export function CanvasBox({ label, draw, deps, className }: { label: string; draw: (ctx: CanvasRenderingContext2D, w: number, h: number, dark: boolean) => void; deps: unknown[]; className?: string }) {
  const [box, size] = useSize<HTMLDivElement>()
  const ref = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  useEffect(() => {
    const ctx = setupCanvas(ref.current, size.w, size.h)
    if (!ctx || size.w < 40 || size.h < 40) return
    draw(ctx, size.w, size.h, dark)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size.w, size.h, dark, ...deps])
  return (
    <div ref={box} className={className ?? 'relative h-full w-full min-w-0 overflow-hidden'}>
      <canvas ref={ref} role="img" aria-label={label} className="absolute inset-0" />
    </div>
  )
}

/** Lines (or dots) over x and y with calm axes; series need not share x. */
export function XYPlot(p: XYPlotProps) {
  return (
    <CanvasBox
      label={p.label}
      deps={[p]}
      draw={(ctx, W, H) => {
        const allX: number[] = [], allY: number[] = []
        for (const s of p.series) {
          const n = Math.min(s.x.length, s.y.length)
          for (let i = 0; i < n; i++) if (Number.isFinite(s.x[i]) && Number.isFinite(s.y[i])) (allX.push(s.x[i]), allY.push(s.y[i]))
        }
        for (const m of p.markers ?? []) allX.push(m.x)
        const xd = p.xDomain ?? domainOf(allX, 0)
        const yd = p.yDomain ?? domainOf(p.zeroLine ? [...allY, 0] : allY)
        if (!(xd[1] > xd[0]) || !(yd[1] > yd[0])) return
        const { px, py, ml, mt, pw, ph, secondary } = drawAxes(ctx, W, H, xd, yd, p.xLabel, p.yLabel)
        ctx.save()
        ctx.beginPath()
        ctx.rect(ml, mt, pw, ph)
        ctx.clip()
        if (p.zeroLine && yd[0] < 0 && yd[1] > 0) {
          ctx.strokeStyle = secondary
          ctx.globalAlpha = 0.5
          ctx.beginPath()
          ctx.moveTo(ml, Math.round(py(0)) + 0.5)
          ctx.lineTo(ml + pw, Math.round(py(0)) + 0.5)
          ctx.stroke()
          ctx.globalAlpha = 1
        }
        for (const r of p.hRules ?? []) {
          ctx.strokeStyle = secondary
          ctx.setLineDash(r.dash ? [3, 3] : [])
          ctx.beginPath()
          ctx.moveTo(ml, Math.round(py(r.y)) + 0.5)
          ctx.lineTo(ml + pw, Math.round(py(r.y)) + 0.5)
          ctx.stroke()
        }
        for (const s of p.series) {
          const n = Math.min(s.x.length, s.y.length)
          ctx.strokeStyle = s.color
          ctx.fillStyle = s.color
          ctx.lineWidth = s.width ?? 1.4
          ctx.setLineDash(s.dash ? [5, 4] : [])
          ctx.lineJoin = 'round'
          if (s.dots) {
            for (let i = 0; i < n; i++) {
              if (!Number.isFinite(s.x[i]) || !Number.isFinite(s.y[i])) continue
              ctx.beginPath()
              ctx.arc(px(s.x[i]), py(s.y[i]), 2, 0, Math.PI * 2)
              ctx.fill()
            }
            continue
          }
          ctx.beginPath()
          let on = false
          for (let i = 0; i < n; i++) {
            if (!Number.isFinite(s.x[i]) || !Number.isFinite(s.y[i])) {
              on = false
              continue
            }
            const X = px(s.x[i]), Y = py(s.y[i])
            if (on) ctx.lineTo(X, Y)
            else ctx.moveTo(X, Y)
            on = true
          }
          ctx.stroke()
        }
        ctx.setLineDash([])
        ctx.restore()
        ctx.font = FONT
        for (const m of p.markers ?? []) {
          if (!Number.isFinite(m.x) || m.x < xd[0] || m.x > xd[1]) continue
          const X = Math.round(px(m.x)) + 0.5
          ctx.strokeStyle = secondary
          ctx.setLineDash(m.dash === false ? [] : [3, 3])
          ctx.beginPath()
          ctx.moveTo(X, mt)
          ctx.lineTo(X, mt + ph)
          ctx.stroke()
          ctx.setLineDash([])
          if (m.label) {
            ctx.fillStyle = secondary
            ctx.textBaseline = 'top'
            ctx.textAlign = X - ml > pw - 40 ? 'right' : 'left'
            ctx.fillText(m.label, X + (ctx.textAlign === 'right' ? -3 : 3), mt + 3)
          }
        }
      }}
    />
  )
}

export interface BarGroup {
  label: string
  color: string
  values: number[]
}

/** Grouped bars per category with a zero line. */
export function BarPlot(p: { categories: string[]; groups: BarGroup[]; xLabel: string; yLabel: string; yDomain?: [number, number]; label: string }) {
  return (
    <CanvasBox
      label={p.label}
      deps={[p]}
      draw={(ctx, W, H) => {
        const vals = p.groups.flatMap((g) => g.values)
        const yd = p.yDomain ?? domainOf([...vals, 0], 0.08)
        if (!(yd[1] > yd[0]) || !p.categories.length) return
        const { py, ml, mt, pw, ph, secondary } = drawAxes(ctx, W, H, [0, 1], yd, p.xLabel, p.yLabel, false)
        const n = p.categories.length, g = Math.max(1, p.groups.length)
        const cw = pw / n
        const bw = Math.max(1, Math.min(18, (cw * 0.8) / g))
        const y0 = py(Math.min(Math.max(0, yd[0]), yd[1]))
        ctx.strokeStyle = secondary
        ctx.beginPath()
        ctx.moveTo(ml, Math.round(y0) + 0.5)
        ctx.lineTo(ml + pw, Math.round(y0) + 0.5)
        ctx.stroke()
        p.groups.forEach((grp, gi) => {
          ctx.fillStyle = grp.color
          grp.values.forEach((v, ci) => {
            if (!Number.isFinite(v) || ci >= n) return
            const x = ml + ci * cw + (cw - bw * g) / 2 + gi * bw
            const y = py(v)
            ctx.fillRect(x + 0.5, Math.min(y, y0), Math.max(1, bw - 1), Math.max(1, Math.abs(y - y0)))
          })
        })
        ctx.fillStyle = secondary
        ctx.textAlign = 'center'
        ctx.textBaseline = 'top'
        const every = Math.max(1, Math.ceil(28 / cw))
        p.categories.forEach((c, ci) => {
          if (ci % every !== 0 && ci !== n - 1) return
          ctx.fillText(c, ml + (ci + 0.5) * cw, mt + ph + 4)
        })
      }}
    />
  )
}
