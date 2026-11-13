// Ghosts, stray light and polarisation: the ghost-path ranking, sensor maps
// (one-hue magnitude ramp, linear or logarithmic) with their summaries, and
// the polarisation pupil map.
import { AlertTriangle } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useDark } from '@/lib/theme'
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
const sci = (v: number | null) => (v === null ? '—' : v.toExponential(1).replace('-', '−'))
const short = (v: number | null, unit = '') => (v === null ? '—' : `${sig(v, 4)}${unit ? ' ' + unit : ''}`)
const fixed2 = (v: number | null) => (v === null ? '—' : v.toFixed(2).replace(/^-/, '−'))

// MARK: Ghost paths

export function GhostPathsView(p: RendererProps) {
  const j = p.result.json
  const ghosts = arr(j.ghosts).filter(isObj)
  if (!ghosts.length)
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-secondary">
        <span className="font-medium text-label">No Ghost Paths</span>
        <span className="text-[12px]">The enumeration found no ghost reaching the sensor.</span>
      </div>
    )
  return (
    <div className="absolute inset-0 flex flex-col">
      <div className="min-h-0 flex-1 overflow-auto p-2.5">
        <table className="num text-[12px] select-text">
          <thead>
            <tr className="text-[11px] text-secondary">
              <th className="pr-4 pb-1 text-left font-medium">Ghost</th>
              <th className="px-2 pb-1 text-right font-medium">Rel. irradiance</th>
              <th className="px-2 pb-1 text-right font-medium">Focus (mm)</th>
              <th className="px-2 pb-1 text-right font-medium">Blur ⌀ (mm)</th>
              <th className="px-2 pb-1 text-right font-medium">ΠR</th>
              <th className="px-2 pb-1 text-right font-medium">Height (mm)</th>
            </tr>
          </thead>
          <tbody>
            {ghosts.map((g, k) => (
              <tr key={k} className="border-t border-separator/60">
                <td className="whitespace-nowrap py-[2px] pr-4">
                  <span className="inline-flex items-center gap-1">
                    {String(g.label ?? '') || nums(g.path).join('·')}
                    {g.near_focus === true && <AlertTriangle className="size-3 text-warning" aria-label="near focus" />}
                  </span>
                </td>
                <td className="px-2 text-right">{sci(defined(g.relative_irradiance))}</td>
                <td className="px-2 text-right">{fixed2(defined(g.focus_from_sensor))}</td>
                <td className="px-2 text-right">{fixed2(defined(g.spot_diameter))}</td>
                <td className="px-2 text-right">{sci(defined(g.reflectance_product))}</td>
                <td className="px-2 text-right">{fixed2(defined(g.image_height))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="shrink-0 px-2.5 pb-2 text-[11px] text-secondary">
        {ghosts.length} paths, strongest first · <AlertTriangle className="inline size-3 text-warning" /> focused near the sensor
        {j.truncated === true && ' · higher-order paths were dropped by the path budget'}
      </div>
    </div>
  )
}

export function ghostPathsKeyNumbers(r: AnalysisResult): KeyNumber[] {
  const g = arr(r.json.ghosts).filter(isObj)
  const out: KeyNumber[] = [{ label: 'Paths', value: String(g.length) }]
  const top = g[0] ? defined(g[0].relative_irradiance) : null
  if (top !== null) out.push({ label: 'Strongest', value: sci(top), help: `Relative irradiance of ${String(g[0].label ?? 'the strongest ghost')}` })
  const near = g.filter((x) => x.near_focus === true).length
  if (near) out.push({ label: 'Near focus', value: String(near) })
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
        <Select aria-label="Map" value={String(sel)} onChange={(e) => setSel(Number(e.target.value))}>
          {props.maps.map((x, k) => (
            <option key={k} value={k}>
              {x.title}
            </option>
          ))}
        </Select>
        <label className="ml-auto flex items-center gap-1.5 text-secondary">
          <Switch aria-label="Log scale" checked={log} onCheckedChange={setLog} />
          Log scale
        </label>
      </div>
      <div className="flex min-h-0 flex-1 gap-3 @max-[480px]:flex-col">
        <div className="min-h-0 min-w-0 flex-1">
          <MapCanvas grid={m.grid} cells color={color} bar={{ range, color, unit: props.unit, log }} xLabel="x (mm)" yLabel="y (mm)" dark={dark} label={`${m.title} map on the sensor`} />
        </div>
        {props.rows.length > 0 && <KeyValues rows={props.rows} className="shrink-0 self-start @max-[480px]:self-stretch" />}
      </div>
      <div className="text-[11px] text-secondary">
        Sensor {w.toFixed(2)} × {h.toFixed(2)} mm
      </div>
    </div>
  )
}

export function GhostIrradianceView(p: RendererProps) {
  const r = p.result, j = r.json
  const maps = useMemo(() => {
    const half = nums(obj(j.map)?.half_size)
    const hs: [number, number] = half.length === 2 && half.every((v) => v >= 0) ? [half[0], half[1]] : [1, 1]
    const out: SensorMap[] = []
    const t = sensorMap(r, 'total_ghost_map', hs)
    if (t) out.push({ title: 'All ghosts', grid: t })
    const pm = sensorMap(r, 'primary_map', hs)
    if (pm) out.push({ title: 'Primary image', grid: pm })
    const ghosts = arr(j.ghosts).filter(isObj)
    const G = ghosts.length
    ghosts.forEach((g, k) => {
      const mi = typeof g.map_index === 'number' ? g.map_index : null
      const grid = mi !== null && G ? sensorMap(r, 'path_maps', hs, mi, G) : null
      if (grid) out.push({ title: String(g.label ?? `Ghost ${k + 1}`), grid })
    })
    return out
  }, [r, j])
  const rows: [string, string][] = [
    ['Ghost / primary flux', short(defined(j.total_flux_ratio))],
    ['Ghost / primary peak', short(defined(j.total_peak_ratio))],
    ...arr(j.ghosts)
      .filter(isObj)
      .slice(0, 6)
      .map((g): [string, string] => [String(g.label ?? ''), short(defined(g.peak_ratio))]),
  ]
  if (!maps.length) return <Empty text="No ghost irradiance maps in this result." />
  return <SensorMapView maps={maps} unit="mm⁻²" rows={rows} />
}

export function StrayLightView(p: RendererProps) {
  const r = p.result, j = r.json
  const maps = useMemo(() => {
    const d = obj(j.detector)
    const hx = nf(d?.half_x), hy = nf(d?.half_y)
    const hs: [number, number] = [Number.isFinite(hx) && hx > 0 ? hx : 1, Number.isFinite(hy) && hy > 0 ? hy : 1]
    const out: SensorMap[] = []
    const t = sensorMap(r, 'total_irradiance', hs)
    if (t) out.push({ title: 'Stray total', grid: t })
    const classes = arr(j.map_classes).filter((c): c is string => typeof c === 'string')
    classes.forEach((c, k) => {
      const g = sensorMap(r, 'class_irradiance', hs, k, classes.length)
      if (g) out.push({ title: c.charAt(0).toUpperCase() + c.slice(1), grid: g })
    })
    return out
  }, [r, j])
  const rows: [string, string][] = [['Stray flux', short(defined(j.stray_flux), 'W')], ['± standard error', short(defined(j.stray_std_error), 'W')]]
  const frac = defined(j.stray_fraction)
  if (frac !== null) rows.push(['Stray fraction', short(frac)])
  for (const t of arr(j.totals).filter(isObj)) {
    const c = String(t.class ?? '')
    rows.push([c.charAt(0).toUpperCase() + c.slice(1) + (t.included === false ? ' (not in total)' : ''), short(defined(t.flux), 'W')])
  }
  const warnings = arr(j.warnings).filter((w): w is string => typeof w === 'string')
  if (!maps.length) return <Empty text="No stray light maps in this result." />
  return (
    <>
      <SensorMapView maps={maps} unit="W/mm²" rows={rows} />
      {warnings.length > 0 && (
        <ul className="absolute right-2.5 bottom-6 max-w-[60%] text-right text-[11px] text-warning">
          {warnings.map((w, k) => (
            <li key={k}>{w}</li>
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
  if (v !== null) out.push({ label: 'Ghost/primary flux', value: sig(v, 4), help: 'Total ghost flux on the map over the primary flux' })
  if (pk !== null) out.push({ label: 'Peak', value: sig(pk, 4), help: 'Total ghost peak over the primary peak' })
  return out
}

export const strayLightKeyNumbers = (r: AnalysisResult): KeyNumber[] => {
  const out: KeyNumber[] = []
  const f = defined(r.json.stray_fraction)
  if (f !== null) out.push({ label: 'Stray fraction', value: sig(f, 4), help: 'Stray flux over the reference flux' })
  const s = defined(r.json.stray_flux)
  if (s !== null) out.push({ label: 'Stray flux', value: `${sig(s, 4)} W` })
  return out
}

// MARK: Polarisation pupil map

type Quantity = 'intensity' | 'diattenuation' | 'retardance'

export function PolarizationView(p: RendererProps) {
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
  if (!data) return <Empty text="No pupil samples in this result." />
  const v = data.values[q]
  const fr = finiteRange(v) ?? { min: 0, max: 1 }
  const range = heatRange(fr, false)
  const color = heatColor('perceptual', range, dark)
  const grid = nf(j.grid)
  const size = 2 / Math.max(Number.isFinite(grid) && grid > 1 ? grid - 1 : Math.sqrt((data.n * 4) / Math.PI), 1)
  const f = obj(j.field)
  const wl = nf(j.wavelength_um)
  const stats: [string, string][] = [
    ['Mean intensity', short(defined(j.mean_intensity))],
    ['Max diattenuation', short(defined(j.max_diattenuation))],
    ['Max retardance', short(defined(j.max_retardance), 'rad')],
    ['RMS retardance', short(defined(j.rms_retardance), 'rad')],
    ['Samples', String(nf(j.sample_count) || data.n)],
  ]
  const undef = nf(j.undefined_retarder)
  if (undef > 0) stats.push(['Undefined retarder', String(undef)])
  if (f) stats.push(['Field (Hx, Hy)', `${sig(nf(f.hx), 3)}, ${sig(nf(f.hy), 3)}`])
  if (Number.isFinite(wl)) stats.push(['Wavelength', `${sig(wl, 5)} µm`])
  return (
    <div className="absolute inset-0 flex flex-col gap-1.5 p-2.5">
      <div>
        <SegmentedOrMenu
          label="Quantity"
          value={q}
          onChange={setQ}
          options={[
            { value: 'intensity', title: 'Intensity', help: 'Transmitted power fraction for the input state' },
            { value: 'diattenuation', title: 'Diattenuation', help: '(Tmax − Tmin)/(Tmax + Tmin)' },
            { value: 'retardance', title: 'Retardance', help: 'Proper retardance (rad)' },
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
            label={`${q} over the pupil`}
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
  if (m !== null) out.push({ label: 'Mean intensity', value: sig(m, 4) })
  const d = defined(j.max_diattenuation)
  if (d !== null) out.push({ label: 'Max diattenuation', value: sig(d, 3) })
  const rr = defined(j.rms_retardance)
  if (rr !== null) out.push({ label: 'RMS retardance', value: `${sig(rr, 3)} rad` })
  return out
}
