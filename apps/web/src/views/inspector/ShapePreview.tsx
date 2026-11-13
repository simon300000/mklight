import { useEffect, useRef } from 'react'
import { fmt, sig } from '@/model/format'
import {
  departureMap,
  minimumPeriod,
  phase as phaseAt,
  phaseOf,
  phasePower,
  previewKind,
  previewRadius,
  profile,
  type Phase,
} from '@/model/shapeSag'
import type { Shape } from '@/model/system'
import { cssVar, seriesColor, useDark } from '@/lib/theme'

interface Series {
  label: string
  color: string
  dashed?: boolean
  points: [number, number][]
}

const FONT = '10px -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif'

/** A compact line chart on a canvas: zero lines, the extremes as ticks. */
function LineChart({ series, xDomain, xLabel, yLabel, height = 116, ariaLabel }: { series: Series[]; xDomain: [number, number]; xLabel: string; yLabel: string; height?: number; ariaLabel: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const draw = () => {
      const w = c.clientWidth
      const h = height
      if (!w) return
      const dpr = devicePixelRatio || 1
      c.width = w * dpr
      c.height = h * dpr
      const g = c.getContext('2d')
      if (!g) return
      g.setTransform(dpr, 0, 0, dpr, 0, 0)
      g.clearRect(0, 0, w, h)
      const ys = series.flatMap((s) => s.points.map((p) => p[1]))
      let ymin = Math.min(0, ...ys)
      let ymax = Math.max(0, ...ys)
      if (ymax - ymin < 1e-12) {
        ymin -= 1e-3
        ymax += 1e-3
      }
      const pad = { l: 46, r: 6, t: 6, b: 18 }
      const pw = w - pad.l - pad.r
      const ph = h - pad.t - pad.b
      const X = (x: number) => pad.l + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * pw
      const Y = (y: number) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * ph
      g.font = FONT
      g.lineWidth = 1
      g.strokeStyle = cssVar('--grid')
      g.strokeRect(pad.l + 0.5, pad.t + 0.5, pw - 1, ph - 1)
      g.strokeStyle = cssVar('--separator')
      g.beginPath()
      g.moveTo(pad.l, Y(0))
      g.lineTo(pad.l + pw, Y(0))
      if (xDomain[0] < 0 && xDomain[1] > 0) {
        g.moveTo(X(0), pad.t)
        g.lineTo(X(0), pad.t + ph)
      }
      g.stroke()
      g.fillStyle = cssVar('--secondary')
      g.textAlign = 'right'
      g.textBaseline = 'top'
      g.fillText(sig(ymax, 3), pad.l - 4, pad.t)
      g.textBaseline = 'bottom'
      g.fillText(sig(ymin, 3), pad.l - 4, pad.t + ph)
      g.textBaseline = 'middle'
      g.save()
      g.translate(9, pad.t + ph / 2)
      g.rotate(-Math.PI / 2)
      g.textAlign = 'center'
      g.fillText(yLabel, 0, 0)
      g.restore()
      g.textBaseline = 'top'
      g.textAlign = 'left'
      g.fillText(sig(xDomain[0], 3), pad.l, pad.t + ph + 4)
      g.textAlign = 'right'
      g.fillText(sig(xDomain[1], 3), pad.l + pw, pad.t + ph + 4)
      g.textAlign = 'center'
      g.fillText(xLabel, pad.l + pw / 2, pad.t + ph + 4)
      for (const s of series) {
        g.strokeStyle = s.color
        g.lineWidth = 1.5
        g.setLineDash(s.dashed ? [4, 3] : [])
        g.beginPath()
        s.points.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))))
        g.stroke()
      }
      g.setLineDash([])
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(c)
    return () => ro.disconnect()
  }, [series, xDomain, xLabel, yLabel, height, dark])
  return <canvas ref={ref} className="w-full" style={{ height }} role="img" aria-label={ariaLabel} />
}

function Legend({ series }: { series: Series[] }) {
  return (
    <span className="flex gap-2.5 text-[11px] text-secondary">
      {series.map((s) => (
        <span key={s.label} className="flex items-center gap-1">
          <svg width="14" height="4" aria-hidden>
            <line x1="0" y1="2" x2="14" y2="2" stroke={s.color} strokeWidth="1.5" strokeDasharray={s.dashed ? '4 3' : undefined} />
          </svg>
          {s.label}
        </span>
      ))}
    </span>
  )
}

function Profiles({ shape, h, axes, title = 'Sag' }: { shape: Shape; h: number; axes: ('x' | 'y')[]; title?: string }) {
  const dark = useDark()
  const series: Series[] = axes.flatMap((axis) => {
    const pts = profile(shape, h, axis, 96)
    return pts ? [{ label: axis === 'x' ? 'x section' : 'y section', color: seriesColor(axis === 'x' ? 1 : 0, dark), dashed: axis === 'x', points: pts }] : []
  })
  const zs = series.flatMap((s) => s.points.map((p) => p[1]))
  const aria = series.length
    ? series.map((s) => `${s.label}: sag ${sig(Math.min(...s.points.map((p) => p[1])), 4)} to ${sig(Math.max(...s.points.map((p) => p[1])), 4)} mm over ±${sig(h, 4)} mm`).join('; ')
    : 'No sag'
  return (
    <div className="py-1" data-testid="sag-profile">
      {(series.length > 1 || title !== 'Sag') && (
        <div className="mb-0.5 flex items-center justify-between">
          <span className="text-[11px] font-semibold">{title}</span>
          {series.length > 1 && <Legend series={series} />}
        </div>
      )}
      {series.length === 0 ? (
        <p className="text-[11px] text-secondary">The shape has no real sag over ±{sig(h, 4)} mm.</p>
      ) : (
        <>
          <LineChart series={series} xDomain={[-h, h]} xLabel="Position (mm)" yLabel="Sag (mm)" ariaLabel={aria} />
          <p className="num text-[11px] text-secondary">
            sag {sig(Math.min(...zs), 4)} … {sig(Math.max(...zs), 4)} mm over ±{sig(h, 4)} mm
          </p>
        </>
      )}
    </div>
  )
}

/** Diverging blue (negative) – neutral – red (positive). */
function diverging(t: number, dark: boolean): [number, number, number] {
  const mid: [number, number, number] = dark ? [58, 58, 60] : [247, 247, 247]
  const neg: [number, number, number] = [33, 102, 172]
  const pos: [number, number, number] = [178, 24, 43]
  const a = Math.min(1, Math.abs(t))
  const end = t < 0 ? neg : pos
  return [0, 1, 2].map((k) => Math.round(mid[k] + (end[k] - mid[k]) * a)) as [number, number, number]
}

/** The departure from the base conic over the clear aperture, with its PV. */
function DepartureMap({ shape, h }: { shape: Shape; h: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const bar = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  const n = 64
  const cells = departureMap(shape, h, n)
  const finite = cells.filter((v): v is number => v != null && Number.isFinite(v))
  const lo = finite.length ? Math.min(...finite) : 0
  const hi = finite.length ? Math.max(...finite) : 0
  const peak = Math.max(Math.abs(lo), Math.abs(hi), 1e-15)
  const pv = hi - lo
  const micro = peak < 0.1
  useEffect(() => {
    const c = ref.current
    if (!c) return
    c.width = n
    c.height = n
    const g = c.getContext('2d')
    if (!g) return
    const img = g.createImageData(n, n)
    cells.forEach((v, k) => {
      if (v == null || !Number.isFinite(v)) return
      const [r, gg, b] = diverging(v / peak, dark)
      img.data.set([r, gg, b, 255], k * 4)
    })
    g.putImageData(img, 0, 0)
    const cb = bar.current
    const bg = cb?.getContext('2d')
    if (cb && bg) {
      cb.width = 1
      cb.height = 64
      const im = bg.createImageData(1, 64)
      for (let k = 0; k < 64; k++) im.data.set([...diverging(1 - (2 * k) / 63, dark), 255], k * 4)
      bg.putImageData(im, 0, 0)
    }
  }, [cells, peak, dark])
  const scale = (v: number) => sig(v * (micro ? 1000 : 1), 3)
  return (
    <div className="py-1" data-testid="departure-map">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[11px] font-semibold">Departure from base</span>
        <span className="num text-[11px] text-secondary">PV {micro ? `${sig(pv * 1000, 4)} µm` : `${sig(pv, 4)} mm`}</span>
      </div>
      <div className="flex items-center gap-3">
        <div className="relative size-[120px] shrink-0">
          <canvas
            ref={ref}
            className="size-full rounded-full shadow-[0_0_0_0.5px_var(--separator)]"
            style={{ imageRendering: 'pixelated' }}
            role="img"
            aria-label={`Departure map from the base, peak to valley ${sig(pv, 4)} mm over a radius of ${sig(h, 4)} mm`}
          />
          <span className="absolute -left-0.5 top-0 text-[10px] text-secondary">y ↑</span>
          <span className="absolute -right-1 bottom-0 text-[10px] text-secondary">x →</span>
        </div>
        <div className="flex h-[112px] items-stretch gap-1.5">
          <canvas ref={bar} className="h-full w-2.5 rounded-sm" style={{ imageRendering: 'pixelated' }} aria-hidden />
          <div className="num flex flex-col justify-between text-[10px] text-secondary">
            <span>{scale(peak)}</span>
            <span>0</span>
            <span>{scale(-peak)}</span>
          </div>
          <span className="self-end text-[10px] text-secondary">{micro ? 'µm' : 'mm'}</span>
        </div>
      </div>
    </div>
  )
}

function PhaseProfile({ p, h }: { p: Phase; h: number }) {
  const dark = useDark()
  const pts: [number, number][] = Array.from({ length: 97 }, (_, k) => {
    const r = (h * k) / 96
    return [r, phaseAt(p, r) / (2 * Math.PI)]
  })
  return (
    <div className="py-1" data-testid="phase-profile">
      <span className="text-[11px] font-semibold">Phase (order +1)</span>
      <LineChart
        series={[{ label: 'Φ / 2π', color: seriesColor(6, dark), points: pts }]}
        xDomain={[0, h]}
        xLabel="Radius (mm)"
        yLabel="Φ (waves)"
        ariaLabel={`Phase profile from 0 to ${sig(pts.at(-1)?.[1] ?? 0, 4)} waves at a radius of ${sig(h, 4)} mm`}
      />
    </div>
  )
}

function Figure({ title, detail, value, extra }: { title: string; detail?: string; value: string; extra?: string | null }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-[2px]">
      <span className="min-w-0">
        <span className="block text-[12px]">{title}</span>
        {detail && <span className="block text-[11px] text-secondary">{detail}</span>}
      </span>
      <span className="num shrink-0 text-right">
        <span className="block text-[12px] font-medium">{value}</span>
        {extra && <span className="block text-[11px] text-secondary">{extra}</span>}
      </span>
    </div>
  )
}

/** The DOE's paraxial power (from A₁, SURFACES §3), focal length, smallest
 *  zone period and zone count over the clear aperture. */
function DOEFigures({ p, h, primaryUm }: { p: Phase; h: number; primaryUm: number }) {
  const trace = phasePower(p, p.traceOrder, primaryUm)
  const design = phasePower(p, p.designOrder, p.designWavelengthUm)
  const period = minimumPeriod(p, h)
  const zones = Math.abs(phaseAt(p, h)) / (2 * Math.PI)
  const same = p.traceOrder === p.designOrder && Math.abs(primaryUm - p.designWavelengthUm) < 1e-9
  const power = (v: number) => (v === 0 ? '0' : `${sig(v, 4)} mm⁻¹`)
  return (
    <div className="select-text py-1" title="Paraxial power φ = −m λ A₁ / (π r_norm²), added to the base's refractive power. A DOE's Abbe number is about −3.45." data-testid="doe-figures">
      <Figure title="Paraxial power" detail={`order ${p.traceOrder}, ${sig(primaryUm, 5)} µm`} value={power(trace)} extra={trace === 0 ? null : `f ${sig(1 / trace, 5)} mm`} />
      {!same && <Figure title="At design" detail={`order ${p.designOrder}, ${sig(p.designWavelengthUm, 5)} µm`} value={power(design)} extra={design === 0 ? null : `f ${sig(1 / design, 5)} mm`} />}
      <Figure
        title="Smallest zone period"
        detail={period ? `at r = ${sig(period.radius, 4)} mm` : undefined}
        value={period ? (period.period < 0.1 ? `${sig(period.period * 1000, 4)} µm` : `${sig(period.period, 4)} mm`) : '–'}
      />
      <Figure title="Zones" detail={`over ±${sig(h, 4)} mm`} value={fmt(zones, 1)} />
    </div>
  )
}

/** The inspector's picture of a surface shape: sag profiles (x and y for
 *  anamorphic and freeform shapes), a false-colour departure map for
 *  freeforms, and for a DOE its phase profile and paraxial figures.
 *  Display geometry from `shapeSag`; the engine traces. */
export function ShapePreview({ shape, semiDiameter, primaryUm }: { shape: Shape; semiDiameter: number | null; primaryUm: number }) {
  const h = previewRadius(shape, semiDiameter)
  switch (previewKind(shape.type)) {
    case 'radial':
      return <Profiles shape={shape} h={h} axes={['y']} />
    case 'anamorphic':
      return <Profiles shape={shape} h={h} axes={['x', 'y']} />
    case 'freeform':
      return (
        <>
          <Profiles shape={shape} h={h} axes={['x', 'y']} />
          <DepartureMap shape={shape} h={h} />
        </>
      )
    case 'diffractive': {
      const p = phaseOf(shape)
      if (!p) return null
      const base = shape.base as Shape | undefined
      return (
        <>
          <PhaseProfile p={p} h={h} />
          <DOEFigures p={p} h={h} primaryUm={primaryUm} />
          {base && base.type !== 'plane' && <Profiles shape={base} h={h} axes={['y']} title="Base sag" />}
        </>
      )
    }
    default:
      return null
  }
}
