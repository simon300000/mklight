// Heat-map scales, iso-lines and grid helpers of the map views (docs/APP.md
// "Field maps …", Swift `HeatMapScale`, `Contours`, `Grid2D.upsampled`):
// one rule for every map of a quantity. Signed data keeps the diverging
// blue–red scale centred on 0 and symmetric about it; other data takes the
// perceptual sequential ramp (viridis from its 0.1 step) over the data's own
// range; fractions may use 0…1; a flat map gets a narrow band around its value.
// PSF and irradiance images keep the one-hue magnitude ramp.

import { divergingColor, magnitudeColor } from '@/views/results/common'

export type RGB = [number, number, number]

/** Viridis from its 0.1 step: lightness rises monotonically, readable in
 *  light and dark and with colour-vision deficiencies. */
const VIRIDIS = [0x482475, 0x414487, 0x355f8d, 0x2a788e, 0x21918c, 0x22a884, 0x44bf70, 0x7ad151, 0xbddf26, 0xfde725].map(
  (c): RGB => [(c >> 16) & 255, (c >> 8) & 255, c & 255],
)

function ramp(stops: RGB[], t: number): RGB {
  const x = (Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(x))
  const f = x - i
  const a = stops[i], b = stops[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** A quantity t in 0…1 on the perceptual ramp (same in both appearances). */
export const perceptualColor = (t: number): RGB => ramp(VIRIDIS, t)

export type HeatScaleKind = 'diverging' | 'perceptual' | 'magnitude' | 'magnitude-log'
export type HeatRangeMode = 'data' | 'unit'

export interface Range {
  min: number
  max: number
}

/** Finite range of values, or null when none is finite. */
export function finiteRange(vs: ArrayLike<number>): Range | null {
  let lo = Infinity, hi = -Infinity
  for (let k = 0; k < vs.length; k++) {
    const v = vs[k]
    if (!Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return lo <= hi ? { min: lo, max: hi } : null
}

export const scaleFor = (signed: boolean): HeatScaleKind => (signed ? 'diverging' : 'perceptual')

/** The colour range of a map (`HeatMapScale.range`). */
export function heatRange(data: Range, signed: boolean, mode: HeatRangeMode = 'data', isFraction = false): Range {
  if (signed) {
    const m = Math.max(Math.abs(data.min), Math.abs(data.max), 1e-300)
    return { min: -m, max: m }
  }
  if (mode === 'unit' && isFraction) return { min: 0, max: Math.max(1, data.max) }
  const lo = data.min, hi = data.max
  const tiny = Math.max(Math.abs(lo), Math.abs(hi), 1e-300) * 1e-9
  if (hi - lo <= tiny) {
    const d = Math.max(Math.abs(lo) * 1e-3, 1e-12)
    return { min: lo - d, max: hi + d }
  }
  return { min: lo, max: hi }
}

/** The smallest value a logarithmic scale tells apart: five decades below the largest. */
export const logFloor = (r: Range) => Math.max(r.max * 1e-5, 1e-300)

/** Position of a value on the scale: −1…1 for diverging, else 0…1. */
export function normalise(v: number, kind: HeatScaleKind, r: Range): number {
  switch (kind) {
    case 'diverging':
      return v / Math.max(Math.abs(r.min), Math.abs(r.max), 1e-300)
    case 'magnitude-log': {
      const lo = logFloor(r)
      return (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / Math.max(Math.log10(r.max) - Math.log10(lo), 1e-12)
    }
    default:
      return (v - r.min) / Math.max(r.max - r.min, 1e-300)
  }
}

/** The colour function of a scale over a range. */
export function heatColor(kind: HeatScaleKind, r: Range, dark: boolean): (v: number) => RGB {
  switch (kind) {
    case 'diverging':
      return (v) => divergingColor(Math.max(-1, Math.min(1, normalise(v, kind, r))), dark)
    case 'perceptual':
      return (v) => perceptualColor(normalise(v, kind, r))
    default:
      return (v) => magnitudeColor(normalise(v, kind, r), dark)
  }
}

// MARK: Grids

/** Samples at the nodes, row-major [ny][nx], row 0 at the low y; the nodes span x × y. */
export interface Grid {
  nx: number
  ny: number
  values: ArrayLike<number>
  x: [number, number]
  y: [number, number]
  /** Outside the unit disk is no data by definition (relative illumination map). */
  unitDisk?: boolean
}

export const outsideDisk = (x: number, y: number) => Math.hypot(x, y) > 1 + 1e-9

export function nodeAt(g: Grid, k: number): [number, number] {
  const i = k % g.nx, j = Math.floor(k / g.nx)
  const dx = g.nx > 1 ? (g.x[1] - g.x[0]) / (g.nx - 1) : 0
  const dy = g.ny > 1 ? (g.y[1] - g.y[0]) / (g.ny - 1) : 0
  return [g.x[0] + i * dx, g.y[0] + j * dy]
}

/** Bilinear upsampling for display, so the map and its linear iso-lines agree;
 *  with `unitDisk`, cells outside the field are NaN and the rest is
 *  interpolated from the finite corners only. */
export function upsample(g: Grid, n: number): Grid {
  if (g.nx < 2 || g.ny < 2 || n <= Math.max(g.nx, g.ny)) return g
  const w = n
  const h = Math.max(2, Math.round((n * (g.ny - 1)) / Math.max(g.nx - 1, 1)))
  const out = new Float64Array(w * h).fill(NaN)
  const v = g.values
  for (let row = 0; row < h; row++) {
    const yy = (row / (h - 1)) * (g.ny - 1)
    const j = Math.min(Math.floor(yy), g.ny - 2), b = yy - j
    for (let col = 0; col < w; col++) {
      const xx = (col / (w - 1)) * (g.nx - 1)
      const i = Math.min(Math.floor(xx), g.nx - 2), a = xx - i
      const c = [v[j * g.nx + i], v[j * g.nx + i + 1], v[(j + 1) * g.nx + i], v[(j + 1) * g.nx + i + 1]]
      const wt = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
      if (g.unitDisk) {
        const px = g.x[0] + (col / (w - 1)) * (g.x[1] - g.x[0])
        const py = g.y[0] + (row / (h - 1)) * (g.y[1] - g.y[0])
        if (outsideDisk(px, py)) continue
        let s = 0, ws = 0
        for (let k = 0; k < 4; k++) if (Number.isFinite(c[k])) ((s += wt[k] * c[k]), (ws += wt[k]))
        if (ws > 1e-12) out[row * w + col] = s / ws
        continue
      }
      out[row * w + col] = wt[0] * c[0] + wt[1] * c[1] + wt[2] * c[2] + wt[3] * c[3]
    }
  }
  return { nx: w, ny: h, values: out, x: g.x, y: g.y, unitDisk: g.unitDisk }
}

// MARK: Iso-lines

export type Segment = [[number, number], [number, number]]

/** Marching squares with linear interpolation along the cell edges and the
 *  cell mean to resolve saddles; cells with a non-finite corner are skipped.
 *  Segments are in the grid's x / y units. */
export function contourSegments(g: Grid, level: number): Segment[] {
  const { nx, ny } = g
  if (nx < 2 || ny < 2 || !Number.isFinite(level)) return []
  const dx = (g.x[1] - g.x[0]) / (nx - 1), dy = (g.y[1] - g.y[0]) / (ny - 1)
  const p = (i: number, j: number): [number, number] => [g.x[0] + i * dx, g.y[0] + j * dy]
  const t = (a: number, b: number) => (a === b ? 0.5 : Math.min(Math.max((level - a) / (b - a), 0), 1))
  const out: Segment[] = []
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const v0 = g.values[j * nx + i], v1 = g.values[j * nx + i + 1]
      const v2 = g.values[(j + 1) * nx + i + 1], v3 = g.values[(j + 1) * nx + i]
      if (!Number.isFinite(v0) || !Number.isFinite(v1) || !Number.isFinite(v2) || !Number.isFinite(v3)) continue
      const code = (v0 >= level ? 1 : 0) | (v1 >= level ? 2 : 0) | (v2 >= level ? 4 : 0) | (v3 >= level ? 8 : 0)
      if (code === 0 || code === 15) continue
      const bottom = p(i + t(v0, v1), j)
      const right = p(i + 1, j + t(v1, v2))
      const top = p(i + t(v3, v2), j + 1)
      const left = p(i, j + t(v0, v3))
      switch (code) {
        case 1: case 14: out.push([left, bottom]); break
        case 2: case 13: out.push([bottom, right]); break
        case 3: case 12: out.push([left, right]); break
        case 4: case 11: out.push([right, top]); break
        case 6: case 9: out.push([bottom, top]); break
        case 7: case 8: out.push([left, top]); break
        case 5: case 10: {
          const centreHigh = (v0 + v1 + v2 + v3) / 4 >= level
          if ((code === 5) === centreHigh) out.push([left, top], [bottom, right])
          else out.push([left, bottom], [right, top])
          break
        }
      }
    }
  }
  return out
}

/** Iso-lines as drawn: a map with no data outside the field is traced on its
 *  finely interpolated grid, so the lines reach the field's edge. */
export function mapContours(g: Grid, level: number): Segment[] {
  if (!g.unitDisk) return contourSegments(g, level)
  return contourSegments(upsample(g, Math.max(4 * Math.max(g.nx, g.ny), 48)), level)
}

/** 1, 2, 2.5, 5 × 10^k nearest above `v` (Swift `Nice.length`). */
export function niceStep(v: number): number {
  if (!(v > 0) || !Number.isFinite(v)) return 1
  const e = Math.pow(10, Math.floor(Math.log10(v)))
  for (const m of [1, 2, 2.5, 5, 10]) if (m * e >= v * (1 - 1e-9)) return m * e
  return 10 * e
}

/** Round iso-illumination levels strictly inside the range, about five
 *  (0.88, 0.90, … for a map from 0.87 to 1). */
export function isoLevels(r: Range, count = 5): number[] {
  const span = r.max - r.min
  if (!(span > 1e-9) || !Number.isFinite(span)) return []
  const step = niceStep(span / count)
  const out: number[] = []
  for (let k = Math.floor(r.min / step) + 1; k * step < r.max - step * 1e-6; k++) {
    const v = Math.round(k * step * 1e9) / 1e9
    if (v > r.min + step * 1e-6) out.push(v)
  }
  return out
}
