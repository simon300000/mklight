// Geometric curves and point plots: Seidel, chromatic focal shift,
// longitudinal aberration, lateral colour, relative illumination, footprint,
// grid distortion, ray grid, best focus, encircled energy and geometric MTF.
import { useMemo, useState } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import { Segmented } from '@/components/ui/segmented'
import { cssVar, seriesColor, useDark, wavelengthColor } from '@/lib/theme'
import { sig } from '@/model/format'
import { FONT, buffer, nums, obj, wavelengthsUm, wlColor } from '../../results/common'
import type { KeyNumber, Renderer, RendererProps } from '../registry'
import {
  BarPlot,
  CanvasBox,
  Empty,
  Key,
  KeyValues,
  PlotPane,
  Sub,
  XYPlot,
  fieldTitle,
  intOf,
  length,
  maxOf,
  minOf,
  nf,
  objects,
  percent,
  series,
  short,
  signed,
  spoken,
  type XYSeries,
} from './kit'

const um1000 = (v: number[]) => v.map((x) => x * 1000)
const secondaryColor = () => cssVar('--secondary') || '#888'
const CLIPPED = '#e34a47'

// MARK: Seidel

const TERMS: [string, string][] = [
  ['s1', 'S_I spherical'],
  ['s2', 'S_II coma'],
  ['s3', 'S_III astigmatism'],
  ['s4', 'S_IV Petzval'],
  ['s5', 'S_V distortion'],
]
const COLOUR: [string, string][] = [
  ['cl', 'C_L axial'],
  ['ct', 'C_T lateral'],
]

function SeidelView({ result }: RendererProps) {
  const dark = useDark()
  const [chromatic, setChromatic] = useState(false)
  const j = result.json
  if (j.status === 'not_applicable')
    return <Empty title="Seidel Coefficients Do Not Apply" detail={typeof j.reason === 'string' ? j.reason : 'Seidel coefficients need a rotationally symmetric system.'} />
  const list = objects(j.surfaces)
  const sums = obj(j.sums) ?? {}
  const terms = chromatic ? COLOUR : TERMS
  const offset = chromatic ? 5 : 0
  const categories = [...list.map((s, k) => String(intOf(s.surface, k))), 'Σ']
  const groups = terms.map(([key, title], k) => ({
    label: title,
    color: seriesColor(k + offset, dark),
    values: [...list.map((s) => nf(s[key])), nf(sums[key])],
  }))
  const pr = nf(j.petzval_radius)
  const wf = obj(j.wavefront_waves)
  const wavefront = wf
    ? Object.keys(wf)
        .sort()
        .map((k) => [k.toUpperCase(), nf(wf[k])] as const)
        .filter(([, v]) => Number.isFinite(v))
    : []
  return (
    <div className="absolute inset-0 flex min-w-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 px-3 pt-2">
        <Segmented
          label="Coefficients"
          value={chromatic ? 'c' : 's'}
          onChange={(v) => setChromatic(v === 'c')}
          options={[
            { value: 's', title: 'Seidel' },
            { value: 'c', title: 'Chromatic' },
          ]}
        />
        {Number.isFinite(pr) && <span className="num ml-auto text-[11px] text-secondary">Petzval radius {short(pr)} mm</span>}
      </div>
      <div className="flex shrink-0 flex-wrap gap-x-3 gap-y-0.5 px-3 pt-1.5 text-[11px] text-secondary">
        {groups.map((g) => (
          <span key={g.label} aria-label={spoken(g.label)}>
            <Key kind="dot" color={g.color} label={<Sub text={g.label} />} />
          </span>
        ))}
      </div>
      <div className="min-h-0 flex-1 px-2 pb-1">
        <BarPlot categories={categories} groups={groups} xLabel="Surface" yLabel="mm" label="Seidel coefficients per surface and their sum" />
      </div>
      {wavefront.length > 0 && (
        <div className="flex shrink-0 flex-wrap gap-x-3 gap-y-0.5 px-3 pb-2 text-[11px]">
          <span className="font-semibold text-secondary">Wavefront (waves)</span>
          {wavefront.map(([k, v]) => (
            <span key={k} className="inline-flex gap-1">
              <span className="text-secondary">{k}</span>
              <span className="num">{signed(v)}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function seidelKeys(r: AnalysisResult): KeyNumber[] {
  if (r.json.status === 'not_applicable') return []
  const sums = obj(r.json.sums)
  if (!sums) return []
  return (
    [
      ['Σ S_I', 's1', 'Spherical'],
      ['Σ S_II', 's2', 'Coma'],
      ['Σ S_III', 's3', 'Astigmatism'],
    ] as const
  )
    .filter(([, k]) => Number.isFinite(nf(sums[k])))
    .map(([label, k, help]) => ({ label, value: short(nf(sums[k])), help: `${help} sum (mm)` }))
}

// MARK: Chromatic focal shift

function ChromaticFocalShift({ result }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const w = nums(j.wavelengths_um)
  const s = nums(j.focal_shift)
  const sx = nums(j.focal_shift_x)
  const primary = nf(j.primary_wavelength_um)
  const anamorphic = sx.length === w.length && sx.some((v, k) => Number.isFinite(v) && Math.abs(v - s[k]) > 1e-9)
  const plot: XYSeries[] = [{ x: um1000(s), y: w, color: seriesColor(0, dark) }]
  if (anamorphic) plot.push({ x: um1000(sx), y: w, color: seriesColor(1, dark), dash: true })
  return (
    <PlotPane
      legend={
        anamorphic || Number.isFinite(primary) ? (
          <>
            {anamorphic && (
              <>
                <Key color={seriesColor(0, dark)} label="y focus" />
                <Key color={seriesColor(1, dark)} kind="dash" label="x focus" />
              </>
            )}
            {Number.isFinite(primary) && <Key color={secondaryColor()} kind="thin" label={`Primary ${sig(primary, 5)} µm`} />}
          </>
        ) : undefined
      }
    >
      <XYPlot
        series={plot}
        xLabel="Focal shift (µm)"
        yLabel="Wavelength (µm)"
        hRules={Number.isFinite(primary) ? [{ y: primary, dash: true }] : []}
        label="Chromatic focal shift versus wavelength"
      />
    </PlotPane>
  )
}

// MARK: Longitudinal aberration

function Longitudinal({ result, system }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const pupil = nums(j.pupil)
  const wls = wavelengthsUm(j, system)
  const curves = objects(j.curves)
  const axes = new Set(curves.map((c) => c.axis ?? 'y'))
  const umOf = (c: JsonObject) => (Number.isFinite(nf(c.wavelength_um)) ? nf(c.wavelength_um) : wls[intOf(c.wavelength, 0)])
  const seen = new Map<number, number>()
  for (const c of curves) seen.set(intOf(c.wavelength, 0), umOf(c))
  return (
    <PlotPane
      legend={
        <>
          {[...seen].map(([w, um]) => (
            <Key key={w} kind="dot" color={Number.isFinite(um) ? wavelengthColor(um, dark) : '#888'} label={`${sig(um, 4)} µm`} />
          ))}
          {axes.size > 1 && (
            <>
              <Key color={secondaryColor()} label="y fan" />
              <Key color={secondaryColor()} kind="dash" label="x fan" />
            </>
          )}
        </>
      }
    >
      <XYPlot
        series={curves.map((c) => ({
          x: nums(c.aberration),
          y: pupil,
          color: Number.isFinite(umOf(c)) ? wavelengthColor(umOf(c), dark) : '#888',
          dash: axes.size > 1 && c.axis === 'x',
        }))}
        xLabel="Longitudinal aberration (mm)"
        yLabel="Pupil zone"
        yDomain={[0, 1]}
        markers={[{ x: 0, dash: false }]}
        label="Longitudinal aberration versus pupil zone per wavelength"
      />
    </PlotPane>
  )
}

// MARK: Lateral colour

function LateralColour({ result, system }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const field = nums(j.field_values)
  const unit = typeof j.field_unit === 'string' && j.field_unit ? j.field_unit : system.fields?.kind === 'object_angle' ? '°' : 'mm'
  const curves = objects(j.curves)
  const wls = wavelengthsUm(j, system)
  const airy = nf(j.airy_radius)
  const umOf = (c: JsonObject) => (Number.isFinite(nf(c.wavelength_um)) ? nf(c.wavelength_um) : wls[intOf(c.wavelength, 0)])
  return (
    <PlotPane
      legend={
        <>
          {curves.map((c, k) => (
            <Key key={k} kind="dot" color={Number.isFinite(umOf(c)) ? wavelengthColor(umOf(c), dark) : '#888'} label={`${sig(umOf(c), 4)} µm`} />
          ))}
          {Number.isFinite(airy) && <Key color={secondaryColor()} kind="dash" label="± Airy radius" />}
        </>
      }
    >
      <XYPlot
        series={curves.map((c) => ({ x: um1000(nums(c.offset)), y: field, color: Number.isFinite(umOf(c)) ? wavelengthColor(umOf(c), dark) : '#888' }))}
        markers={Number.isFinite(airy) ? [{ x: airy * 1000 }, { x: -airy * 1000 }] : []}
        xLabel="Lateral color (µm)"
        yLabel={`Field (${unit})`}
        label="Lateral color versus field"
      />
    </PlotPane>
  )
}

// MARK: Relative illumination

function RelativeIllumination({ result, system }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const field = nums(j.field_values)
  const unit = typeof j.field_unit === 'string' && j.field_unit ? j.field_unit : system.fields?.kind === 'object_angle' ? '°' : 'mm'
  const ri = nums(j.relative_illumination)
  const vig = nums(j.vignetting_fraction)
  const cos4 = nums(j.cos4_object)
  const plot: XYSeries[] = [{ x: field, y: ri, color: seriesColor(0, dark), label: 'RI' }]
  if (vig.length) plot.push({ x: field, y: vig, color: seriesColor(1, dark), dash: true, label: 'Vignetted' })
  if (cos4.some(Number.isFinite)) plot.push({ x: field, y: cos4, color: secondaryColor(), width: 1, label: 'cos⁴' })
  return (
    <PlotPane
      legend={
        <>
          <Key color={seriesColor(0, dark)} label="Relative illumination" />
          {vig.length > 0 && <Key color={seriesColor(1, dark)} kind="dash" label="Vignetted fraction" />}
          {cos4.some(Number.isFinite) && <Key color={secondaryColor()} kind="thin" label="cos⁴ (object)" />}
        </>
      }
    >
      <XYPlot series={plot} xLabel={`Field (${unit})`} yLabel="Fraction" yDomain={[0, 1.05]} label="Relative illumination and vignetting versus field" />
    </PlotPane>
  )
}

// MARK: Square point plots

/** Draws onto a square box centred in the canvas, ±m data units. */
function squareFrame(_ctx: CanvasRenderingContext2D, W: number, H: number, m: number) {
  const side = Math.min(W, H) - 8
  const s = side / 2 / m
  const c = { x: W / 2, y: H / 2 }
  return { s, pt: (x: number, y: number) => [c.x + x * s, c.y - y * s] as const, c }
}

function Footprint({ result, system }: RendererProps) {
  const j = result.json
  const pts = buffer(result, 'points')
  const clip = buffer(result, 'clipped')
  const sets = objects(j.sets)
  const wls = wavelengthsUm(j, system)
  const sd = nf(j.semi_diameter)
  const surface = intOf(j.surface, 0)
  const nClipped = clip ? Math.floor(clip.length / 2) : 0
  const nPass = sets.reduce((a, s) => a + intOf(nums(s.points)[1], 0), 0)
  return (
    <PlotPane
      legend={
        <>
          <span className="font-semibold text-label">Surface {surface}</span>
          {Number.isFinite(sd) && <Key color={secondaryColor()} kind="dash" label={`Clear aperture ${short(sd)} mm`} />}
          {nClipped > 0 && <Key color={CLIPPED} kind="dot" label={`${nClipped} clipped`} />}
        </>
      }
    >
      <CanvasBox
        label={`Footprint on surface ${surface}, ${nPass} rays pass, ${nClipped} clipped`}
        deps={[result]}
        draw={(ctx, W, H, dk) => {
          let m = Number.isFinite(sd) ? sd : 0
          for (const b of [pts, clip]) if (b) for (let i = 0; i < b.length; i++) if (Number.isFinite(b[i])) m = Math.max(m, Math.abs(b[i]))
          m = Math.max(m, 1e-9) * 1.05
          const { s, pt, c } = squareFrame(ctx, W, H, m)
          if (Number.isFinite(sd)) {
            ctx.strokeStyle = secondaryColor()
            ctx.setLineDash([4, 3])
            ctx.beginPath()
            ctx.arc(c.x, c.y, sd * s, 0, Math.PI * 2)
            ctx.stroke()
            ctx.setLineDash([])
          }
          if (pts) {
            ctx.globalAlpha = 0.8
            for (const set of sets) {
              const [off, cnt] = nums(set.points)
              if (!(cnt > 0)) continue
              ctx.fillStyle = wlColor(wls, intOf(set.wavelength, 0), dk)
              for (let i = Math.max(0, off); i < Math.min(off + cnt, pts.length / 2); i++) {
                const [x, y] = pt(pts[2 * i], pts[2 * i + 1])
                if (Number.isFinite(x) && Number.isFinite(y)) ctx.fillRect(x - 1.3, y - 1.3, 2.6, 2.6)
              }
            }
            ctx.globalAlpha = 1
          }
          if (clip) {
            ctx.fillStyle = CLIPPED
            for (let i = 0; i < clip.length / 2; i++) {
              const [x, y] = pt(clip[2 * i], clip[2 * i + 1])
              if (Number.isFinite(x) && Number.isFinite(y)) ctx.fillRect(x - 1.3, y - 1.3, 2.6, 2.6)
            }
          }
        }}
      />
    </PlotPane>
  )
}

function GridDistortion({ result }: RendererProps) {
  const j = result.json
  const n = intOf(j.grid_size, 0)
  const par = buffer(result, 'paraxial')
  const real = buffer(result, 'real')
  const max = nf(j.max_abs_distortion_percent)
  if (n < 2 || !par || !real || par.length < n * n * 2 || real.length < n * n * 2) return <Empty title="No distortion grid in this result" />
  return (
    <PlotPane
      legend={
        <>
          <Key color={secondaryColor()} kind="dash" label="Paraxial grid" />
          <Key color={cssVar('--label') || '#222'} label="Real chief rays" />
          {Number.isFinite(max) && <span className="num">Max {max.toFixed(3)} %</span>}
        </>
      }
    >
      <CanvasBox
        label={`Grid distortion, maximum ${Number.isFinite(max) ? max.toFixed(3) : 'unknown'} percent`}
        deps={[result]}
        draw={(ctx, W, H) => {
          let m = 1e-9
          for (const b of [par, real]) for (let i = 0; i < n * n * 2; i++) if (Number.isFinite(b[i])) m = Math.max(m, Math.abs(b[i]))
          const { pt } = squareFrame(ctx, W, H, m * 1.06)
          const grid = (b: ArrayLike<number>) => {
            ctx.beginPath()
            for (let line = 0; line < n; line++)
              for (let dir = 0; dir < 2; dir++) {
                let on = false
                for (let k = 0; k < n; k++) {
                  const i = dir === 0 ? line * n + k : k * n + line
                  const x = b[2 * i], y = b[2 * i + 1]
                  if (!Number.isFinite(x) || !Number.isFinite(y)) {
                    on = false
                    continue
                  }
                  const [X, Y] = pt(x, y)
                  if (on) ctx.lineTo(X, Y)
                  else ctx.moveTo(X, Y)
                  on = true
                }
              }
            ctx.stroke()
          }
          ctx.strokeStyle = secondaryColor()
          ctx.globalAlpha = 0.6
          ctx.lineWidth = 1
          ctx.setLineDash([3, 3])
          grid(par)
          ctx.globalAlpha = 1
          ctx.setLineDash([])
          ctx.strokeStyle = cssVar('--label') || '#222'
          ctx.lineWidth = 1.2
          grid(real)
        }}
      />
    </PlotPane>
  )
}

function RayGrid({ result, system }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const F = nums(j.fields).length || (Array.isArray(j.field_values) ? j.field_values.length : 0)
  const W = nums(j.wavelengths).length || nums(j.wavelengths_um).length
  const P = intOf(j.pupil_count, 0)
  const pts = buffer(result, 'points')
  const chief = buffer(result, 'chief')
  const umList = nums(j.wavelengths_um)
  const wls = umList.length ? umList : wavelengthsUm(j, system)
  const fieldIdx = nums(j.fields)
  const fv = Array.isArray(j.field_values) ? j.field_values : []
  if (!pts || F < 1 || W < 1 || P < 1 || pts.length < F * W * P * 2) return <Empty title="No ray grid in this result" />
  // The primary-wavelength chief ray (else the first) is each cell's centre.
  const sysPrimary = system.wavelengths?.primary ?? 0
  const wIdx = nums(j.wavelengths)
  const centreW = Math.max(0, wIdx.indexOf(sysPrimary))
  return (
    <div className="absolute inset-0 flex min-w-0 flex-col">
      <div className="flex shrink-0 flex-wrap gap-x-3 gap-y-0.5 px-3 pt-1.5 text-[11px] text-secondary">
        {wls.slice(0, W).map((um, k) => (
          <Key key={k} kind="dot" color={wavelengthColor(um, dark)} label={`${sig(um, 4)} µm`} />
        ))}
        <span className="num ml-auto">
          {intOf(j.arrived, 0)} / {intOf(j.launched, F * W * P)} arrived
        </span>
      </div>
      <div className="grid min-h-0 flex-1 auto-rows-[minmax(140px,1fr)] grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2 overflow-auto p-2">
        {Array.from({ length: F }, (_, f) => {
          const title = fieldTitle(system, fieldIdx.length ? fieldIdx[f] : f, fv[f])
          return (
            <div key={f} className="flex min-h-0 flex-col">
              <div className="truncate text-center text-[11px] text-secondary">{title}</div>
              <div className="min-h-0 flex-1">
                <CanvasBox
                  label={`Ray grid hits, ${title}`}
                  deps={[result, f]}
                  draw={(ctx, Wd, Hd, dk) => {
                    const cx = chief ? chief[(f * W + centreW) * 2] : NaN
                    const cy = chief ? chief[(f * W + centreW) * 2 + 1] : NaN
                    let sx = 0, sy = 0, cnt = 0
                    if (!Number.isFinite(cx) || !Number.isFinite(cy)) {
                      for (let i = f * W * P; i < (f + 1) * W * P; i++) if (Number.isFinite(pts[2 * i])) (sx += pts[2 * i], sy += pts[2 * i + 1], cnt++)
                    }
                    const ox = Number.isFinite(cx) ? cx : cnt ? sx / cnt : 0
                    const oy = Number.isFinite(cy) ? cy : cnt ? sy / cnt : 0
                    let m = 1e-9
                    for (let i = f * W * P; i < (f + 1) * W * P; i++) {
                      const x = pts[2 * i] - ox, y = pts[2 * i + 1] - oy
                      if (Number.isFinite(x) && Number.isFinite(y)) m = Math.max(m, Math.abs(x), Math.abs(y))
                    }
                    const { pt } = squareFrame(ctx, Wd, Hd, m * 1.08)
                    ctx.strokeStyle = cssVar('--grid') || 'rgba(128,128,128,0.2)'
                    ctx.strokeRect(4.5, 4.5, Wd - 9, Hd - 9)
                    ctx.globalAlpha = 0.75
                    for (let w = 0; w < W; w++) {
                      ctx.fillStyle = wlColor(wls, w, dk)
                      for (let p = 0; p < P; p++) {
                        const i = (f * W + w) * P + p
                        const [X, Y] = pt(pts[2 * i] - ox, pts[2 * i + 1] - oy)
                        if (Number.isFinite(X) && Number.isFinite(Y)) ctx.fillRect(X - 1, Y - 1, 2, 2)
                      }
                    }
                    ctx.globalAlpha = 1
                    ctx.fillStyle = secondaryColor()
                    ctx.font = FONT
                    ctx.textAlign = 'right'
                    ctx.textBaseline = 'bottom'
                    ctx.fillText(`±${length(m)}`, Wd - 7, Hd - 6)
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// MARK: Best focus

function BestFocus({ result }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const d = nf(j.defocus)
  const x = nums(j.through_focus_defocus)
  const y = nums(j.through_focus_merit)
  const rows: [string, string][] = [
    ['Best defocus', length(d)],
    ['RMS at best focus', length(nf(j.merit))],
    ['RMS at the image surface', length(nf(j.merit_at_zero))],
  ]
  const msd = nf(j.mean_square_defocus)
  if (Number.isFinite(msd)) rows.push(['Mean-square best defocus', length(msd)])
  if (typeof j.converged === 'boolean') rows.push(['Search', `${j.converged ? 'Converged' : 'Not converged'} · ${intOf(j.evaluations, 0)} evaluations`])
  return (
    <div className="absolute inset-0 flex min-w-0 flex-col gap-2 p-3">
      <KeyValues rows={rows} />
      {x.length > 2 && y.length === x.length ? (
        <div className="min-h-0 flex-1">
          <XYPlot
            series={[{ x, y: um1000(y), color: seriesColor(0, dark) }]}
            xLabel="Defocus (mm)"
            yLabel="RMS radius (µm)"
            markers={[{ x: d, label: 'Best' }]}
            label="Field-weighted RMS radius versus defocus"
          />
        </div>
      ) : (
        <p className="text-[12px] text-secondary">Set Through Focus Samples to plot the merit against defocus.</p>
      )}
    </div>
  )
}

// MARK: Encircled energy (geometric and diffraction)

interface EECurve {
  field: number
  label: string
  r80: number
  encircled: number[]
  ensquared: number[]
  limit: number[]
}

function readEncircled(result: AnalysisResult, system: RendererProps['system']) {
  const j = result.json
  const radius = Array.isArray(j.radius) ? nums(j.radius) : series(result, j.radius_mm)
  const curves: EECurve[] = objects(j.fields)
    .map((f, k) => {
      const field = intOf(f.field, k)
      return {
        field,
        label: fieldTitle(system, field, f.field_value),
        r80: nf(f.r80 ?? f.r80_mm),
        encircled: series(result, f.encircled),
        ensquared: series(result, f.ensquared),
        limit: series(result, f.diffraction_limit),
      }
    })
    .filter((c) => c.encircled.length > 0 && c.encircled.length === radius.length)
  return { radius, curves }
}

function EncircledEnergy({ result, system }: RendererProps) {
  const dark = useDark()
  const [square, setSquare] = useState(false)
  const { radius, curves } = useMemo(() => readEncircled(result, system), [result, system])
  if (!curves.length) return <Empty title="No encircled-energy curves in this result" />
  const x = um1000(radius)
  const limit = curves.find((c) => c.limit.length === radius.length)?.limit
  const hasSquare = curves.some((c) => c.ensquared.length === radius.length)
  const plot: XYSeries[] = curves.map((c, k) => ({ x, y: square ? c.ensquared : c.encircled, color: seriesColor(k, dark) }))
  if (limit && !square) plot.push({ x, y: limit, color: secondaryColor(), dash: true, width: 1 })
  return (
    <PlotPane
      top={
        hasSquare ? (
          <div className="flex shrink-0 px-3 pt-2">
            <Segmented
              label="Energy"
              value={square ? 'sq' : 'enc'}
              onChange={(v) => setSquare(v === 'sq')}
              options={[
                { value: 'enc', title: 'Encircled' },
                { value: 'sq', title: 'Ensquared' },
              ]}
            />
          </div>
        ) : undefined
      }
      legend={
        <>
          {curves.map((c, k) => (
            <Key key={c.field} kind="dot" color={seriesColor(k, dark)} label={`${c.label} · r80 ${length(c.r80)}`} />
          ))}
          {limit && !square && <Key color={secondaryColor()} kind="dash" label="Diffraction limit" />}
        </>
      }
    >
      <XYPlot
        series={plot}
        xLabel={square ? 'Half-width (µm)' : 'Radius (µm)'}
        yLabel={square ? 'Ensquared energy' : 'Encircled energy'}
        yDomain={[0, 1.02]}
        hRules={[{ y: 0.8, dash: true }]}
        label={`${square ? 'Ensquared' : 'Encircled'} energy versus ${square ? 'half-width' : 'radius'} per field`}
      />
    </PlotPane>
  )
}

function encircledKeys(r: AnalysisResult): KeyNumber[] {
  const r80 = maxOf(objects(r.json.fields).map((f) => nf(f.r80 ?? f.r80_mm)))
  return Number.isFinite(r80) ? [{ label: 'Max r80', value: length(r80), help: 'Largest radius enclosing 80 % of the energy over the fields' }] : []
}

// MARK: Geometric MTF

function GeometricMtf({ result, system }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const freq = Array.isArray(j.frequencies) ? nums(j.frequencies) : series(result, j.frequencies)
  const byField = new Map<number, { t: number[]; s: number[] }>()
  for (const c of objects(j.curves)) {
    const f = intOf(c.field, 0)
    const cur = byField.get(f) ?? { t: [], s: [] }
    if (c.orientation === 'sagittal') cur.s = nums(c.modulation)
    else cur.t = nums(c.modulation)
    byField.set(f, cur)
  }
  const fields = [...byField.keys()].sort((a, b) => a - b)
  const values = Array.isArray(j.field_values) ? j.field_values : []
  if (!freq.length || !fields.length) return <Empty title="No MTF curves in this result" />
  const plot: XYSeries[] = fields.flatMap((f, k) => [
    { x: freq, y: byField.get(f)!.t, color: seriesColor(k, dark) },
    { x: freq, y: byField.get(f)!.s, color: seriesColor(k, dark), dash: true },
  ])
  const last = maxOf(freq)
  return (
    <PlotPane
      legend={
        <>
          {fields.map((f, k) => (
            <Key key={f} kind="dot" color={seriesColor(k, dark)} label={fieldTitle(system, f, values[k])} />
          ))}
          <Key color={secondaryColor()} label="T" />
          <Key color={secondaryColor()} kind="dash" label="S" />
        </>
      }
    >
      <XYPlot
        series={plot}
        xLabel="Spatial frequency (cycles/mm)"
        yLabel="Modulation"
        xDomain={[0, Number.isFinite(last) && last > 0 ? last : 1]}
        yDomain={[0, 1.02]}
        label="Geometric MTF versus spatial frequency, tangential and sagittal per field"
      />
    </PlotPane>
  )
}

export const GEOMETRIC_RENDERERS: Record<string, Renderer> = {
  seidel: {
    accepts: (j) => Array.isArray(j.surfaces) || j.status === 'not_applicable',
    View: SeidelView,
    keyNumbers: seidelKeys,
  },
  chromatic_focal_shift: {
    accepts: (j) => Array.isArray(j.wavelengths_um) && Array.isArray(j.focal_shift),
    View: ChromaticFocalShift,
    keyNumbers: (r) => (Number.isFinite(nf(r.json.range)) ? [{ label: 'Range', value: length(nf(r.json.range)), help: 'Largest minus smallest focal shift over the sweep' }] : []),
  },
  longitudinal_aberration: {
    accepts: (j) => Array.isArray(j.pupil) && Array.isArray(j.curves),
    View: Longitudinal,
    keyNumbers: (r) => {
      const m = maxOf(objects(r.json.curves).flatMap((c) => nums(c.aberration).map(Math.abs)))
      return Number.isFinite(m) ? [{ label: 'Max |LA|', value: length(m), help: 'Largest longitudinal aberration over the zones and wavelengths' }] : []
    },
  },
  lateral_colour: {
    accepts: (j) => Array.isArray(j.field_values) && Array.isArray(j.curves),
    View: LateralColour,
    keyNumbers: (r) => {
      const m = maxOf(nums(r.json.lateral_colour).map(Math.abs))
      return Number.isFinite(m) ? [{ label: 'Max lateral color', value: length(m), help: 'Largest short − long chief-ray height difference' }] : []
    },
  },
  relative_illumination: {
    accepts: (j) => Array.isArray(j.field_values) && Array.isArray(j.relative_illumination),
    View: RelativeIllumination,
    keyNumbers: (r) => {
      const m = minOf(nums(r.json.relative_illumination))
      return Number.isFinite(m) ? [{ label: 'Min RI', value: percent(m), help: 'Lowest relative illumination over the field' }] : []
    },
  },
  footprint: {
    accepts: (j) => Array.isArray(j.sets),
    View: Footprint,
    keyNumbers: (r) => {
      const out: KeyNumber[] = []
      const sd = nf(r.json.semi_diameter)
      if (Number.isFinite(sd)) out.push({ label: 'Semi-diameter', value: `${short(sd)} mm` })
      const mr = nf(r.json.max_radius)
      if (Number.isFinite(mr)) out.push({ label: 'Max radius', value: `${short(mr)} mm`, help: 'Largest hit radius on the surface' })
      return out
    },
  },
  grid_distortion: {
    accepts: (j) => typeof j.grid_size === 'number',
    View: GridDistortion,
    keyNumbers: (r) => (Number.isFinite(nf(r.json.max_abs_distortion_percent)) ? [{ label: 'Max distortion', value: `${nf(r.json.max_abs_distortion_percent).toFixed(3)} %` }] : []),
  },
  ray_grid: {
    accepts: (j) => typeof j.pupil_count === 'number',
    View: RayGrid,
    keyNumbers: (r) => (typeof r.json.arrived === 'number' ? [{ label: 'Arrived', value: `${r.json.arrived} / ${intOf(r.json.launched, 0)}`, help: 'Rays that reached the image surface' }] : []),
  },
  best_focus: {
    accepts: (j) => j.defocus !== undefined,
    View: BestFocus,
    keyNumbers: (r) => (Number.isFinite(nf(r.json.defocus)) ? [{ label: 'Best defocus', value: length(nf(r.json.defocus)) }] : []),
  },
  encircled_energy: {
    accepts: (j) => Array.isArray(j.fields),
    View: EncircledEnergy,
    keyNumbers: encircledKeys,
  },
  diffraction_encircled_energy: {
    accepts: (j) => Array.isArray(j.fields),
    View: EncircledEnergy,
    keyNumbers: encircledKeys,
  },
  geometric_mtf: {
    accepts: (j) => Array.isArray(j.curves),
    View: GeometricMtf,
    keyNumbers: (r) => {
      const c = nf(r.json.cutoff_frequency)
      return Number.isFinite(c) ? [{ label: 'Diffraction cutoff', value: `${sig(c, 4)} cycles/mm`, help: '1/(λN) for reference; the geometric MTF has no diffraction' }] : []
    },
  },
}

