// The shared reader of the signed Cartesian field maps (`pupil_aberration`,
// `relative_illumination_map`, `chief_ray_angle`; Swift `FieldMapData`):
// `field_coordinates [ny, nx, 2]`, `image_mm [ny, nx, 2]` and
// `values [ny, nx, columns]`, row 0 at the lowest Hy, x fastest.

import type { AnalysisResult, Json } from '@/engine/types'
import { nf } from '@/model/format'
import { arr, buffer, isObj, nums } from '@/views/results/common'
import { finiteRange, outsideDisk, type Grid, type Range } from './heatScale'

export interface FieldColumn {
  name: string
  unit: string
  definition: string
}

export const columnTitle = (c: FieldColumn) => {
  const t = c.name.replace(/_/g, ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}
export const columnUnit = (c: FieldColumn) => (c.unit === '' || c.unit === '1' ? '' : c.unit === 'deg' ? '°' : c.unit)
export const columnTitleWithUnit = (c: FieldColumn) => (columnUnit(c) ? `${columnTitle(c)} (${columnUnit(c)})` : columnTitle(c))

export type FieldLayout = 'map' | 'sweepX' | 'sweepY'

export interface FieldMapData {
  nx: number
  ny: number
  columns: FieldColumn[]
  /** Normalised [Hx, Hy] per sample `j * nx + i`. */
  field: [number, number][]
  /** Chief-ray image point (mm), NaN for unusable fields. */
  image: [number, number][]
  /** `values[(j * nx + i) * columns + c]`. */
  values: ArrayLike<number>
}

/** Reads a field map; null when the result does not have the shape. */
export function readFieldMap(r: AnalysisResult): FieldMapData | null {
  const j = r.json
  const columns: FieldColumn[] = arr(j.columns)
    .filter(isObj)
    .map((c) => ({ name: String(c.name ?? ''), unit: String(c.unit ?? ''), definition: String(c.definition ?? '') }))
  if (!columns.length) return null
  const f = buffer(r, j.field_coordinates ?? 'field_coordinates')
  const v = buffer(r, j.values ?? 'values')
  if (!f || !v) return null
  // Shape: the JSON's [nx, ny], else the buffer's [ny, nx, …].
  let [nx, ny] = nums(j.shape)
  if (!(nx > 0 && ny > 0)) {
    const s = r.shapes?.[typeof j.field_coordinates === 'string' ? j.field_coordinates : 'field_coordinates']
    if (s && s.length === 3) [ny, nx] = s
  }
  if (!(nx > 0 && ny > 0)) return null
  const n = nx * ny
  if (f.length !== 2 * n || v.length !== n * columns.length) return null
  const img = buffer(r, j.image_mm ?? 'image_mm')
  const field: [number, number][] = []
  const image: [number, number][] = []
  for (let k = 0; k < n; k++) {
    field.push([f[2 * k], f[2 * k + 1]])
    image.push(img && img.length === 2 * n ? [img[2 * k], img[2 * k + 1]] : [NaN, NaN])
  }
  return { nx, ny, columns, field, image, values: v }
}

export const acceptsFieldMap = (j: { [k: string]: Json }) => Array.isArray(j.columns) && j.columns.length > 0

export const layoutOf = (d: FieldMapData): FieldLayout => (d.nx > 1 && d.ny > 1 ? 'map' : d.ny === 1 && d.nx > 1 ? 'sweepX' : 'sweepY')
export const countOf = (d: FieldMapData) => d.nx * d.ny
export const valueAt = (d: FieldMapData, k: number, c: number) => d.values[k * d.columns.length + c]
export const columnIndex = (d: FieldMapData, name: string) => {
  const k = d.columns.findIndex((c) => c.name === name)
  return k < 0 ? null : k
}
export function columnValues(d: FieldMapData, c: number): number[] {
  const out: number[] = []
  for (let k = 0; k < countOf(d); k++) out.push(valueAt(d, k, c))
  return out
}
export const columnRange = (d: FieldMapData, c: number): Range | null => finiteRange(columnValues(d, c))

/** Range of a coordinate list; a degenerate one widens by ±1. */
export function spanOf(vs: number[]): [number, number] {
  const r = finiteRange(vs)
  if (!r) return [-1, 1]
  return r.min < r.max ? [r.min, r.max] : [r.min - 1, r.max + 1]
}

/** Samples outside the normalised field are no data (all null), and there are some. */
export function outsideFieldIsNoData(d: FieldMapData): boolean {
  const outside = d.field.map((p, k) => (outsideDisk(p[0], p[1]) ? k : -1)).filter((k) => k >= 0)
  if (!outside.length) return false
  return outside.every((k) => d.columns.every((_, c) => !Number.isFinite(valueAt(d, k, c))))
}

export function gridOf(d: FieldMapData, c: number): Grid {
  return {
    nx: d.nx,
    ny: d.ny,
    values: columnValues(d, c),
    x: spanOf(d.field.map((p) => p[0])),
    y: spanOf(d.field.map((p) => p[1])),
    unitDisk: outsideFieldIsNoData(d),
  }
}

/** `(H, value)` along the swept field axis. */
export function sweepOf(d: FieldMapData, c: number): [number, number][] {
  const lay = layoutOf(d)
  return d.field.map((p, k) => [lay === 'sweepX' ? p[0] : p[1], valueAt(d, k, c)])
}

export const imageRadius = (d: FieldMapData) => d.image.map((p) => Math.hypot(p[0], p[1]))

/** `(image radius, value)` at every usable sample, by radius. */
export function radialOf(d: FieldMapData, c: number): [number, number][] {
  const r = imageRadius(d)
  return r
    .map((x, k): [number, number] => [x, valueAt(d, k, c)])
    .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))
    .sort((a, b) => a[0] - b[0])
}

/** A column best shown on a signed scale: both signs, or a difference zero on axis. */
export function isSigned(d: FieldMapData, c: number): boolean {
  const r = columnRange(d, c)
  if (!r) return false
  if (r.min < 0 && r.max > 0) return true
  const name = d.columns[c].name
  return r.min < 0 || name.includes('error') || name.includes('variation') || name.startsWith('angle_') || name === 'margin'
}

/** Whether a column depends on the field radius only (rotational symmetry). */
export function isRadial(d: FieldMapData, c: number, tolerance = 1e-3): boolean {
  const r = columnRange(d, c)
  if (!r) return true
  const scale = Math.max(r.max - r.min, Math.abs(r.max), 1e-12)
  const xs = spanOf(d.field.map((p) => p[0])), ys = spanOf(d.field.map((p) => p[1]))
  const rx = Math.max(Math.abs(xs[0]), Math.abs(xs[1]), 1e-12), ry = Math.max(Math.abs(ys[0]), Math.abs(ys[1]), 1e-12)
  if (Math.abs(rx - ry) >= 1e-9) return false
  const byRadius = new Map<number, number>()
  for (let k = 0; k < countOf(d); k++) {
    const v = valueAt(d, k, c)
    if (!Number.isFinite(v)) continue
    const key = Math.trunc(Math.hypot(d.field[k][0], d.field[k][1]) * 1e6)
    const w = byRadius.get(key)
    if (w !== undefined && Math.abs(w - v) > tolerance * scale) return false
    byRadius.set(key, v)
  }
  return true
}

export function argExtreme(d: FieldMapData, c: number, sign: 1 | -1): number | null {
  let best: number | null = null
  for (let k = 0; k < countOf(d); k++) {
    const v = valueAt(d, k, c)
    if (!Number.isFinite(v)) continue
    if (best === null || sign * v > sign * valueAt(d, best, c)) best = k
  }
  return best
}

// MARK: Sensor limit (chief ray angle)

/** `[[radius mm, max deg], …]` from `sensor_limit`. */
export function sensorLimit(v: Json | undefined): [number, number][] {
  return arr(v)
    .map((row) => nums(row))
    .filter((r) => r.length === 2 && Number.isFinite(r[0]) && Number.isFinite(r[1]))
    .map((r) => [r[0], r[1]] as [number, number])
}

/** The limit at a radius: linear, held at the end points (as the engine). */
export function limitAt(table: [number, number][], r: number): number {
  if (!table.length) return NaN
  if (r <= table[0][0]) return table[0][1]
  const last = table[table.length - 1]
  if (r >= last[0]) return last[1]
  for (let k = 1; k < table.length; k++) {
    if (r <= table[k][0]) {
      const [r0, a0] = table[k - 1], [r1, a1] = table[k]
      return r1 > r0 ? a0 + ((a1 - a0) * (r - r0)) / (r1 - r0) : a1
    }
  }
  return last[1]
}

export interface Band {
  violated: boolean
  points: { x: number; lo: number; hi: number }[]
}

/** Areas between the CRA curve and the limit, split where they cross. */
export function craBands(curve: [number, number][], limit: [number, number][]): Band[] {
  if (curve.length < 2 || !limit.length) return []
  const lo = curve[0][0], hi = curve[curve.length - 1][0]
  const xs = new Set(curve.map((p) => p[0]))
  for (const [r] of limit) for (const x of [r, -r]) if (x > lo && x < hi) xs.add(x)
  const sorted = [...xs].sort((a, b) => a - b)
  const cra = (x: number) => {
    for (let k = 1; k < curve.length; k++)
      if (x <= curve[k][0]) {
        const [x0, y0] = curve[k - 1], [x1, y1] = curve[k]
        return x1 > x0 ? y0 + ((y1 - y0) * (x - x0)) / (x1 - x0) : y1
      }
    return curve[curve.length - 1][1]
  }
  const pts = sorted.map((x) => {
    const c = cra(x), l = limitAt(limit, Math.abs(x))
    return { x, m: l - c, c, l }
  })
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1], b = pts[k]
    if (a.m < 0 !== b.m < 0 && a.m !== 0 && b.m !== 0) {
      const t = a.m / (a.m - b.m)
      const x = a.x + (b.x - a.x) * t
      const c = cra(x)
      pts.splice(k, 0, { x, m: 0, c, l: c })
      k++
    }
  }
  const bands: Band[] = []
  for (let k = 1; k < pts.length; k++) {
    const a = pts[k - 1], b = pts[k]
    const violated = (a.m + b.m) / 2 < 0
    const ib = { x: b.x, lo: Math.min(b.c, b.l), hi: Math.max(b.c, b.l) }
    const last = bands[bands.length - 1]
    if (last && last.violated === violated) last.points.push(ib)
    else bands.push({ violated, points: [{ x: a.x, lo: Math.min(a.c, a.l), hi: Math.max(a.c, a.l) }, ib] })
  }
  return bands.filter((b) => b.points.length >= 2)
}

/** Value format of a column: degrees, lengths, plain numbers. */
export function columnFormat(c: FieldColumn): (v: number) => string {
  return (v: number) => {
    if (!Number.isFinite(v)) return '—'
    switch (c.unit) {
      case 'deg':
        return `${v.toFixed(2).replace('-', '−')}°`
      case 'mm':
        return lengthText(v)
      case '':
      case '1':
        return sig4(v)
      default:
        return `${sig4(v)} ${c.unit}`
    }
  }
}

const sig4 = (v: number) => {
  if (v === 0) return '0'
  const a = Math.abs(v)
  const s = a < 1e-3 || a >= 1e6 ? v.toExponential(2) : String(Number(v.toPrecision(4)))
  return s.replace('-', '−')
}

/** A length in mm, as µm below 0.1 mm. */
export function lengthText(mm: number): string {
  if (!Number.isFinite(mm)) return '—'
  if (mm !== 0 && Math.abs(mm) < 0.1) return `${sig4(mm * 1000)} µm`
  return `${sig4(mm)} mm`
}

export { nf }
