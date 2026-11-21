// Ghosts, stray light and polarisation: the ghost-path ranking, sensor maps
// (one-hue magnitude ramp, linear or logarithmic) with their summaries, and
// the polarisation pupil map.
import { AlertTriangle } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { Select } from '@/components/ui/select'
import { engineText, t, tMaybe, useT } from '@/i18n'
import { Switch } from '@/components/ui/switch'
import { useDark } from '@/lib/theme'
import { cn } from '@/lib/utils'
import { useView } from '@/state/view'
import type { Key } from '@/i18n'
import { nf, sig } from '@/model/format'
import { arr, buffer, isObj, nums, obj } from '@/views/results/common'
import type { KeyNumber, RendererProps } from '../registry'
import { KeyValues, SegmentedOrMenu } from './Controls'
import { finiteRange, heatColor, heatRange, type Grid } from './heatScale'
import { MapCanvas } from './MapCanvas'

const Empty = ({ text }: { text: string }) => <div className="p-4 text-secondary">{text}</div>

/** A number that may be undefined (null, non-finite): "—". */
export const defined = (v: Json | undefined): number | null => {
  const x = nf(v)
  return Number.isFinite(x) ? x : null
}
/** A ghost or stray-light class (engine name) as a title. */
const strayClass = (c: string) => tMaybe(`maps.stray.classes.${c}`) ?? c.charAt(0).toUpperCase() + c.slice(1)
const sci = (v: number | null) => (v === null ? '—' : v.toExponential(1).replace('-', '−'))
const short = (v: number | null, unit = '') => (v === null ? '—' : `${sig(v, 4)}${unit ? ' ' + unit : ''}`)
const fixed2 = (v: number | null) => (v === null ? '—' : v.toFixed(2).replace(/^-/, '−'))

// MARK: Ghost paths

/** Selects a surface and shows it in the inspector (a ghost row's first
 *  reflecting surface, as the macOS rows do on a click). */
function selectSurface(i: number) {
  const v = useView.getState()
  v.select(i)
  v.setPage('surface')
}

export function GhostPathsView(p: RendererProps) {
  const t = useT()
  const j = p.result.json
  const ghosts = arr(j.ghosts).filter(isObj)
  if (!ghosts.length)
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-secondary">
        <span className="font-medium text-label">{t('maps.ghosts.none')}</span>
        <span className="text-[12px]">{t('maps.ghosts.noneDetail')}</span>
      </div>
    )
  return (
    <div className="absolute inset-0 flex flex-col">
      <div className="min-h-0 flex-1 overflow-auto p-2.5">
        <table className="num text-[12px] select-text">
          <thead>
            <tr className="text-[11px] text-secondary">
              <th className="pr-4 pb-1 text-left font-medium">{t('maps.ghosts.colGhost')}</th>
              <th className="px-2 pb-1 text-right font-medium">{t('maps.ghosts.colIrradiance')}</th>
              <th className="px-2 pb-1 text-right font-medium">{t('maps.ghosts.colFocus')}</th>
              <th className="px-2 pb-1 text-right font-medium">{t('maps.ghosts.colBlur')}</th>
              <th className="px-2 pb-1 text-right font-medium">ΠR</th>
              <th className="px-2 pb-1 text-right font-medium">{t('maps.ghosts.colHeight')}</th>
            </tr>
          </thead>
          <tbody>
            {ghosts.map((g, k) => {
              const first = nums(g.path)[0]
              const target = Number.isInteger(first) ? first : null
              const open = () => target !== null && selectSurface(target)
              return (
                <tr
                  key={k}
                  className={cn('border-t border-separator/60', target !== null && 'hover:bg-hover focus-visible:bg-hover')}
                  title={target !== null ? t('maps.ghosts.selectSurface', { n: target }) : undefined}
                  tabIndex={target !== null ? 0 : undefined}
                  onClick={open}
                  onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), open())}
                >
                  <td className="whitespace-nowrap py-[2px] pr-4">
                    <span className="inline-flex items-center gap-1">
                      {String(g.label ?? '') || nums(g.path).join('·')}
                      {g.near_focus === true && <AlertTriangle className="size-3 text-warning" aria-label={t('maps.ghosts.nearFocusLabel')} />}
                    </span>
                  </td>
                  <td className="px-2 text-right">{sci(defined(g.relative_irradiance))}</td>
                  <td className="px-2 text-right">{fixed2(defined(g.focus_from_sensor))}</td>
                  <td className="px-2 text-right">{fixed2(defined(g.spot_diameter))}</td>
                  <td className="px-2 text-right">{sci(defined(g.reflectance_product))}</td>
                  <td className="px-2 text-right">{fixed2(defined(g.image_height))}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="shrink-0 px-2.5 pb-2 text-[11px] text-secondary">
        {t('maps.ghosts.summary', { n: ghosts.length })} · <AlertTriangle className="inline size-3 text-warning" /> {t('maps.ghosts.nearSensor')}
        {j.truncated === true && ` · ${t('maps.ghosts.truncated')}`}
      </div>
    </div>
  )
}

export function ghostPathsKeyNumbers(r: AnalysisResult): KeyNumber[] {
  const g = arr(r.json.ghosts).filter(isObj)
  const out: KeyNumber[] = [{ label: t('maps.ghosts.keyPaths'), value: String(g.length) }]
  const top = g[0] ? defined(g[0].relative_irradiance) : null
  if (top !== null) out.push({ label: t('maps.ghosts.keyStrongest'), value: sci(top), help: t('maps.ghosts.keyStrongestHelp', { label: String(g[0].label ?? t('maps.ghosts.theStrongest')) }) })
  const near = g.filter((x) => x.near_focus === true).length
  if (near) out.push({ label: t('maps.ghosts.keyNearFocus'), value: String(near) })
  return out
}

// MARK: Sensor maps

interface SensorMap {
  title: string
  grid: Grid
}

function sensorMap(r: AnalysisResult, name: string, half: [number, number], slice = 0, of = 1): Grid | null {
  const b = buffer(r, name)
  if (!b) return null
  const shape = r.shapes?.[name]
  let ny = 0, nx = 0
  if (shape && shape.length >= 2) [ny, nx] = shape.slice(-2)
  if (!(nx > 0 && ny > 0)) {
    const n = Math.round(Math.sqrt(b.length / of))
    nx = ny = n
  }
  const len = nx * ny
  if (b.length < (slice + 1) * len) return null
  const values = Array.from({ length: len }, (_, k) => b[slice * len + k])
  return { nx, ny, values, x: [-half[0], half[0]], y: [-half[1], half[1]] }
}

function SensorMapView(props: { maps: SensorMap[]; unit: string; rows: [string, string][] }) {
  const t = useT()
  const dark = useDark()
  const [sel, setSel] = useState(0)
  const [log, setLog] = useState(true)
  const m = props.maps[Math.min(sel, props.maps.length - 1)]
  const range = useMemo(() => {
    const f = finiteRange(m.grid.values)
    return f && f.max > 0 ? { min: 0, max: f.max } : { min: 0, max: 1 }
  }, [m])
  const kind = log ? 'magnitude-log' : 'magnitude'
  const color = heatColor(kind, range, dark)
  const w = m.grid.x[1] - m.grid.x[0], h = m.grid.y[1] - m.grid.y[0]
  return (
    <div className="absolute inset-0 flex flex-col gap-1.5 p-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
        <Select aria-label={t('maps.sensor.map')} value={String(sel)} onChange={(e) => setSel(Number(e.target.value))}>
          {props.maps.map((x, k) => (
            <option key={k} value={k}>
              {x.title}
            </option>
          ))}
        </Select>
        <label className="ml-auto flex items-center gap-1.5 text-secondary">
          <Switch aria-label={t('maps.sensor.logScale')} checked={log} onCheckedChange={setLog} />
          {t('maps.sensor.logScale')}
        </label>
      </div>
      <div className="flex min-h-0 flex-1 gap-3 @max-[480px]:flex-col">
        <div className="min-h-0 min-w-0 flex-1">
          <MapCanvas grid={m.grid} cells color={color} bar={{ range, color, unit: props.unit, log }} xLabel={t('maps.sensor.xAxis')} yLabel={t('maps.sensor.yAxis')} dark={dark} label={t('maps.sensor.mapOn', { title: m.title })} />
        </div>
        {props.rows.length > 0 && <KeyValues rows={props.rows} className="shrink-0 self-start @max-[480px]:self-stretch" />}
      </div>
      <div className="text-[11px] text-secondary">
        {t('maps.sensor.size', { w: w.toFixed(2), h: h.toFixed(2) })}
      </div>
    </div>
  )
}

export function GhostIrradianceView(p: RendererProps) {
  const t = useT()
  const r = p.result, j = r.json
  const maps = useMemo(() => {
    const half = nums(obj(j.map)?.half_size)
    const hs: [number, number] = half.length === 2 && half.every((v) => v >= 0) ? [half[0], half[1]] : [1, 1]
    const out: SensorMap[] = []
    const tot = sensorMap(r, 'total_ghost_map', hs)
    if (tot) out.push({ title: t('maps.ghosts.allGhosts'), grid: tot })
    const pm = sensorMap(r, 'primary_map', hs)
    if (pm) out.push({ title: t('maps.ghosts.primaryImage'), grid: pm })
    const ghosts = arr(j.ghosts).filter(isObj)
    const G = ghosts.length
    ghosts.forEach((g, k) => {
      const mi = typeof g.map_index === 'number' ? g.map_index : null
      const grid = mi !== null && G ? sensorMap(r, 'path_maps', hs, mi, G) : null
      if (grid) out.push({ title: String(g.label ?? t('maps.ghosts.ghostN', { n: k + 1 })), grid })
    })
    return out
  }, [r, j, t])
  const rows: [string, string][] = [
    [t('maps.ghosts.fluxRatio'), short(defined(j.total_flux_ratio))],
    [t('maps.ghosts.peakRatio'), short(defined(j.total_peak_ratio))],
    ...arr(j.ghosts)
      .filter(isObj)
      .slice(0, 6)
      .map((g): [string, string] => [String(g.label ?? ''), short(defined(g.peak_ratio))]),
  ]
  if (!maps.length) return <Empty text={t('maps.ghosts.noMaps')} />
  return <SensorMapView maps={maps} unit="mm⁻²" rows={rows} />
}

export function StrayLightView(p: RendererProps) {
  const t = useT()
  const r = p.result, j = r.json
  const maps = useMemo(() => {
    const d = obj(j.detector)
    const hx = nf(d?.half_x), hy = nf(d?.half_y)
    const hs: [number, number] = [Number.isFinite(hx) && hx > 0 ? hx : 1, Number.isFinite(hy) && hy > 0 ? hy : 1]
    const out: SensorMap[] = []
    const tot = sensorMap(r, 'total_irradiance', hs)
    if (tot) out.push({ title: t('maps.stray.total'), grid: tot })
    const classes = arr(j.map_classes).filter((c): c is string => typeof c === 'string')
    classes.forEach((c, k) => {
      const g = sensorMap(r, 'class_irradiance', hs, k, classes.length)
      if (g) out.push({ title: strayClass(c), grid: g })
    })
    return out
  }, [r, j, t])
  const rows: [string, string][] = [[t('maps.stray.flux'), short(defined(j.stray_flux), 'W')], [t('maps.stray.stdError'), short(defined(j.stray_std_error), 'W')]]
  const frac = defined(j.stray_fraction)
  if (frac !== null) rows.push([t('maps.stray.fraction'), short(frac)])
  for (const tot of arr(j.totals).filter(isObj)) {
    const c = strayClass(String(tot.class ?? ''))
    rows.push([tot.included === false ? t('maps.stray.notInTotal', { name: c }) : c, short(defined(tot.flux), 'W')])
  }
  const warnings = arr(j.warnings).filter((w): w is string => typeof w === 'string')
  if (!maps.length) return <Empty text={t('maps.stray.noMaps')} />
  return (
    <>
      <SensorMapView maps={maps} unit="W/mm²" rows={rows} />
      {warnings.length > 0 && (
        <ul className="absolute right-2.5 bottom-6 max-w-[60%] text-right text-[11px] text-warning">
          {warnings.map((w, k) => (
            <li key={k}>{engineText(w)}</li>
          ))}
        </ul>
      )}
    </>
  )
}

export const ghostIrradianceKeyNumbers = (r: AnalysisResult): KeyNumber[] => {
  const v = defined(r.json.total_flux_ratio)
  const pk = defined(r.json.total_peak_ratio)
  const out: KeyNumber[] = []
  if (v !== null) out.push({ label: t('maps.ghosts.keyFlux'), value: sig(v, 4), help: t('maps.ghosts.keyFluxHelp') })
  if (pk !== null) out.push({ label: t('maps.ghosts.keyPeak'), value: sig(pk, 4), help: t('maps.ghosts.keyPeakHelp') })
  return out
}

export const strayLightKeyNumbers = (r: AnalysisResult): KeyNumber[] => {
  const out: KeyNumber[] = []
  const f = defined(r.json.stray_fraction)
  if (f !== null) out.push({ label: t('maps.stray.fraction'), value: sig(f, 4), help: t('maps.stray.fractionHelp') })
  const s = defined(r.json.stray_flux)
  if (s !== null) out.push({ label: t('maps.stray.flux'), value: `${sig(s, 4)} W` })
  return out
}

// MARK: Polarisation pupil map

type Quantity = 'intensity' | 'diattenuation' | 'retardance'

export function PolarizationView(p: RendererProps) {
  const t = useT()
  const dark = useDark()
  const r = p.result, j = r.json
  const [q, setQ] = useState<Quantity>('intensity')
  const data = useMemo(() => {
    const pupil = buffer(r, 'pupil')
    if (!pupil || pupil.length < 2) return null
    const n = Math.floor(pupil.length / 2)
    const points: [number, number][] = Array.from({ length: n }, (_, k) => [pupil[2 * k], pupil[2 * k + 1]])
    const reached = buffer(r, 'reached')
    const ok = (k: number) => !reached || reached.length !== n || reached[k] !== 0
    const pick = (name: string) => {
      const b = buffer(r, name)
      return Array.from({ length: n }, (_, k) => (b && ok(k) ? b[k] : NaN))
    }
    return { points, n, values: { intensity: pick('intensity'), diattenuation: pick('diattenuation'), retardance: pick('retardance') } }
  }, [r])
  if (!data) return <Empty text={t('maps.pol.noSamples')} />
  const v = data.values[q]
  const fr = finiteRange(v) ?? { min: 0, max: 1 }
  const range = heatRange(fr, false)
  const color = heatColor('perceptual', range, dark)
  const grid = nf(j.grid)
  const size = 2 / Math.max(Number.isFinite(grid) && grid > 1 ? grid - 1 : Math.sqrt((data.n * 4) / Math.PI), 1)
  const f = obj(j.field)
  const wl = nf(j.wavelength_um)
  const stats: [string, string][] = [
    [t('maps.pol.meanIntensity'), short(defined(j.mean_intensity))],
    [t('maps.pol.maxDiattenuation'), short(defined(j.max_diattenuation))],
    [t('maps.pol.maxRetardance'), short(defined(j.max_retardance), 'rad')],
    [t('maps.pol.rmsRetardance'), short(defined(j.rms_retardance), 'rad')],
    [t('maps.pol.samples'), String(nf(j.sample_count) || data.n)],
  ]
  const undef = nf(j.undefined_retarder)
  if (undef > 0) stats.push([t('maps.pol.undefinedRetarder'), String(undef)])
  if (f) stats.push([t('maps.pol.field'), `${sig(nf(f.hx), 3)}, ${sig(nf(f.hy), 3)}`])
  if (Number.isFinite(wl)) stats.push([t('maps.pol.wavelength'), `${sig(wl, 5)} µm`])
  return (
    <div className="absolute inset-0 flex flex-col gap-1.5 p-2.5">
      <div>
        <SegmentedOrMenu
          label={t('maps.quantity')}
          value={q}
          onChange={setQ}
          options={[
            { value: 'intensity', title: t('maps.pol.intensity'), help: t('maps.pol.intensityHelp') },
            { value: 'diattenuation', title: t('maps.pol.diattenuation'), help: '(Tmax − Tmin)/(Tmax + Tmin)' },
            { value: 'retardance', title: t('maps.pol.retardance'), help: t('maps.pol.retardanceHelp') },
          ]}
        />
      </div>
      <div className="flex min-h-0 flex-1 gap-3 @max-[480px]:flex-col">
        <div className="min-h-0 min-w-0 flex-1">
          <MapCanvas
            extent={{ x: [-1.05, 1.05], y: [-1.05, 1.05] }}
            scatter={{ points: data.points, values: v, size, disk: true }}
            color={color}
            bar={{ range, color, unit: q === 'retardance' ? 'rad' : '' }}
            xLabel="Px"
            yLabel="Py"
            dark={dark}
            label={t('maps.pol.overPupil', { quantity: t(`maps.pol.${q}` as Key) })}
          />
        </div>
        <KeyValues rows={stats} className="shrink-0 self-start @max-[480px]:self-stretch" />
      </div>
    </div>
  )
}

export const polarizationKeyNumbers = (r: AnalysisResult): KeyNumber[] => {
  const j: JsonObject = r.json
  const out: KeyNumber[] = []
  const m = defined(j.mean_intensity)
  if (m !== null) out.push({ label: t('maps.pol.meanIntensity'), value: sig(m, 4) })
  const d = defined(j.max_diattenuation)
  if (d !== null) out.push({ label: t('maps.pol.maxDiattenuation'), value: sig(d, 3) })
  const rr = defined(j.rms_retardance)
  if (rr !== null) out.push({ label: t('maps.pol.rmsRetardance'), value: `${sig(rr, 3)} rad` })
  return out
}
