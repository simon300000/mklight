// The `image_simulation` result, its source and the view's settings helpers
// (Swift `ImageSimulationData`, `LinearImage`, `ImageSimulationSettings`,
// `ImageSourceImport`). Pure functions, tested on their own.

import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { nf } from '@/model/format'
import type { System } from '@/model/system'
import { arr, buffer, isObj, nums, obj } from '@/views/results/common'

// MARK: sRGB

export function srgbEncode(linear: number): number {
  if (!Number.isFinite(linear)) return 0
  if (linear >= 1) return 1
  const x = Math.max(linear, 0)
  return x <= 0.0031308 ? 12.92 * x : Math.min(1.055 * Math.pow(x, 1 / 2.4) - 0.055, 1)
}

export function srgbDecode(code: number): number {
  const x = Math.min(Math.max(code, 0), 1)
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4)
}

/** 8-bit sRGB code → linear, as a table. */
const DECODE8 = Array.from({ length: 256 }, (_, k) => srgbDecode(k / 255))

// MARK: Linear images

/** A linear-light raster, row 0 at the low-y edge (the engine's convention), 1 or 3 interleaved channels. */
export interface LinearImage {
  width: number
  height: number
  channels: number
  data: ArrayLike<number>
}

export const isValidImage = (i: LinearImage) => i.width > 0 && i.height > 0 && (i.channels === 1 || i.channels === 3) && i.data.length === i.width * i.height * i.channels

/** RGBA8 sRGB bytes with row 0 at the top (flipped), after scaling by `gain`. */
export function toRgba(img: LinearImage, gain = 1): Uint8ClampedArray {
  const { width: w, height: h, channels: c, data } = img
  const px = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    const row = h - 1 - y
    for (let x = 0; x < w; x++) {
      const s = (y * w + x) * c, o = (row * w + x) * 4
      const r = data[s] * gain
      px[o] = Math.round(srgbEncode(r) * 255)
      px[o + 1] = Math.round(srgbEncode(c === 3 ? data[s + 1] * gain : r) * 255)
      px[o + 2] = Math.round(srgbEncode(c === 3 ? data[s + 2] * gain : r) * 255)
      px[o + 3] = 255
    }
  }
  return px
}

/** Fraction of samples above 1 after `gain` (clipped highlights). */
export function clippedFraction(img: LinearImage, gain = 1): number {
  if (!img.data.length) return 0
  let n = 0
  for (let k = 0; k < img.data.length; k++) if (img.data[k] * gain > 1) n++
  return n / img.data.length
}

export const PATTERNS: { id: string; title: string }[] = [
  { id: 'grid', title: 'Grid' },
  { id: 'siemens_star', title: 'Siemens star' },
  { id: 'slanted_edge', title: 'Slanted edge' },
  { id: 'text_chart', title: 'Text chart' },
]

const FONT = [
  [17, 27, 21, 17, 17, 17, 17],
  [17, 18, 20, 24, 20, 18, 17],
  [16, 16, 16, 16, 16, 16, 31],
]

/** A built-in pattern at w × h with the engine's definition (`pattern_value`
 *  in mklight-wave's image_simulation.rs); with `ss` > 1 each pixel is the
 *  mean of ss × ss samples (the pattern's area average, as the engine). */
export function patternImage(name: string, w: number, h: number, ss = 1): LinearImage | null {
  if (!(w > 0 && h > 0 && ss >= 1) || !PATTERNS.some((p) => p.id === name)) return null
  const value = (px: number, py: number): number => {
    const u = px / w, v = py / h
    switch (name) {
      case 'grid': {
        const fu = (u * 8) % 1, fv = (v * 8) % 1
        return fu < 0.1 || fv < 0.1 ? 0 : 1
      }
      case 'siemens_star':
        return Math.cos(Math.atan2(v - 0.5, u - 0.5) * 24) > 0 ? 1 : 0
      case 'slanted_edge':
        return u > 0.5 + 0.1 * (v - 0.5) ? 1 : 0
      default: {
        const x = Math.min(Math.floor(px), w - 1), y = h - 1 - Math.min(Math.floor(py), h - 1)
        const band = Math.min(Math.floor((y * 3) / h), 2)
        const scale = [4, 2, 1][band]
        const gx = Math.floor(x / scale), gy = Math.floor(y / scale)
        const glyph = Math.floor(gx / 6) % 3
        return gy % 9 < 7 && gx % 6 < 5 && (FONT[glyph][gy % 9] & (1 << (4 - (gx % 6)))) !== 0 ? 0 : 1
      }
    }
  }
  const data = new Float64Array(w * h)
  const inv = 1 / ss, norm = 1 / (ss * ss)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0
      for (let j = 0; j < ss; j++) for (let i = 0; i < ss; i++) s += value(x + (i + 0.5) * inv, y + (j + 0.5) * inv)
      data[y * w + x] = s * norm
    }
  return { width: w, height: h, channels: 1, data }
}

/** sRGB RGBA bytes (row 0 at the top, as ImageData) cropped about the centre
 *  to `aspect` (width / height; default the output's) and area-averaged in
 *  linear light to `w × h`, row 0 at the bottom; one channel when grey. */
export function resampleSrgb(src: { width: number; height: number; data: ArrayLike<number> }, w: number, h: number, aspect?: number | null): LinearImage {
  const iw = src.width, ih = src.height
  const a = aspect && Number.isFinite(aspect) && aspect > 0 ? aspect : w / h
  const cw = Math.min(iw, ih * a), ch = cw / a
  const cx0 = (iw - cw) / 2, cy0 = (ih - ch) / 2
  let grey = true
  for (let k = 0; k < iw * ih && grey; k += 7) {
    const o = 4 * k
    if (src.data[o] !== src.data[o + 1] || src.data[o] !== src.data[o + 2]) grey = false
  }
  const c = grey ? 1 : 3
  const out = new Float64Array(w * h * c)
  const sx = cw / w, sy = ch / h
  for (let oy = 0; oy < h; oy++) {
    // Output row oy (bottom-up) covers source rows from the top: flip.
    const top = cy0 + (h - 1 - oy) * sy
    for (let ox = 0; ox < w; ox++) {
      const left = cx0 + ox * sx
      const acc = [0, 0, 0]
      let wsum = 0
      const y0 = Math.floor(top), y1 = Math.min(Math.ceil(top + sy), ih)
      const x0 = Math.floor(left), x1 = Math.min(Math.ceil(left + sx), iw)
      for (let yy = Math.max(y0, 0); yy < y1; yy++) {
        const wy = Math.min(yy + 1, top + sy) - Math.max(yy, top)
        if (wy <= 0) continue
        for (let xx = Math.max(x0, 0); xx < x1; xx++) {
          const wx = Math.min(xx + 1, left + sx) - Math.max(xx, left)
          if (wx <= 0) continue
          const wt = wx * wy, o = 4 * (yy * iw + xx)
          acc[0] += wt * DECODE8[src.data[o] | 0]
          acc[1] += wt * DECODE8[src.data[o + 1] | 0]
          acc[2] += wt * DECODE8[src.data[o + 2] | 0]
          wsum += wt
        }
      }
      const o = (oy * w + ox) * c
      for (let k = 0; k < c; k++) out[o + k] = wsum > 0 ? acc[k] / wsum : 0
    }
  }
  return { width: w, height: h, channels: c, data: out }
}

/** A linear image area-averaged to w × h (a size change of an imported source). */
export function resampleLinear(img: LinearImage, w: number, h: number): LinearImage {
  const c = img.channels
  const out = new Float64Array(w * h * c)
  const sx = img.width / w, sy = img.height / h
  for (let oy = 0; oy < h; oy++)
    for (let ox = 0; ox < w; ox++) {
      const bx = ox * sx, by = oy * sy
      const acc = [0, 0, 0]
      let ws = 0
      for (let yy = Math.floor(by); yy < Math.min(Math.ceil(by + sy), img.height); yy++) {
        const wy = Math.min(yy + 1, by + sy) - Math.max(yy, by)
        for (let xx = Math.floor(bx); xx < Math.min(Math.ceil(bx + sx), img.width); xx++) {
          const wx = Math.min(xx + 1, bx + sx) - Math.max(xx, bx)
          if (wx <= 0 || wy <= 0) continue
          for (let k = 0; k < c; k++) acc[k] += wx * wy * img.data[(yy * img.width + xx) * c + k]
          ws += wx * wy
        }
      }
      for (let k = 0; k < c; k++) out[(oy * w + ox) * c + k] = ws > 0 ? acc[k] / ws : 0
    }
  return { width: w, height: h, channels: c, data: out }
}

/** The `{kind: buffer}` source the engine's settings expect. */
export function sourceJSON(img: LinearImage): JsonObject {
  return { kind: 'buffer', width: img.width, height: img.height, channels: img.channels, data: Array.from(img.data, (v) => Math.round(v * 1e6) / 1e6) }
}

export function imageFromSource(s: Json | undefined): LinearImage | null {
  const o = obj(s)
  if (!o || o.kind !== 'buffer') return null
  const img = { width: nf(o.width), height: nf(o.height), channels: nf(o.channels), data: nums(o.data) }
  return isValidImage(img) ? img : null
}

/** Imported sources are at most this many pixels on a side. */
export const MAX_SOURCE_SIDE = 512

/** The size the source is resampled to: the run's `size`, at most 512 on its longer side. */
export function targetSize(size: Json | undefined): [number, number] {
  const a = nums(size)
  const w = a.length === 2 && a[0] >= 2 ? Math.floor(a[0]) : 64
  const h = a.length === 2 && a[1] >= 2 ? Math.floor(a[1]) : 64
  const s = Math.min(1, MAX_SOURCE_SIDE / Math.max(w, h))
  return [Math.max(Math.round(w * s), 2), Math.max(Math.round(h * s), 2)]
}

// MARK: The result

export interface SimNode {
  field: [number, number]
  ideal: [number, number]
  real: [number, number]
  ri: number
}

export function distortionPercent(n: SimNode): number {
  const ri = Math.hypot(n.ideal[0], n.ideal[1])
  if (!(ri > 1e-12)) return NaN
  return ((Math.hypot(n.real[0], n.real[1]) - ri) / ri) * 100
}

export function readNodes(j: JsonObject): SimNode[] {
  return arr(j.field_mapping)
    .filter(isObj)
    .flatMap((n) => {
      const f = nums(n.field), i = nums(n.ideal_mm), r = nums(n.real_mm)
      if (f.length !== 2 || i.length !== 2 || r.length !== 2) return []
      return [{ field: [f[0], f[1]] as [number, number], ideal: [i[0], i[1]] as [number, number], real: [r[0], r[1]] as [number, number], ri: nf(n.relative_illumination) }]
    })
}

export function maxDistortion(nodes: SimNode[]): number | null {
  const d = nodes.map(distortionPercent).filter(Number.isFinite)
  return d.length ? d.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a)) : null
}

export function minRelativeIllumination(nodes: SimNode[]): number | null {
  const r = nodes.map((n) => n.ri).filter(Number.isFinite)
  return r.length ? Math.min(...r) : null
}

export type SimSource = { kind: 'pattern'; pattern: string } | { kind: 'buffer'; image: LinearImage } | { kind: 'path'; path: string }

export function readSource(s: Json | undefined): SimSource {
  const o = obj(s)
  if (o?.kind === 'buffer') {
    const img = imageFromSource(o)
    return img ? { kind: 'buffer', image: img } : { kind: 'pattern', pattern: 'grid' }
  }
  if (o?.kind === 'path') return { kind: 'path', path: String(o.path ?? '') }
  return { kind: 'pattern', pattern: typeof o?.pattern === 'string' ? o.pattern : 'grid' }
}

export interface ImageSimData {
  image: LinearImage
  pixelSize: [number, number]
  origin: [number, number]
  nodes: SimNode[]
  fieldGrid: number
  psfSize: number | null
  psfSpacing: number | null
}

export function readImageSim(r: AnalysisResult): ImageSimData | null {
  const j = r.json
  const b = buffer(r, j.image ?? 'image')
  if (!b) return null
  let [w, h] = nums(j.size)
  let c = nf(j.channels)
  const shape = r.shapes?.[typeof j.image === 'string' ? j.image : 'image']
  if (shape?.length === 3) {
    ;[h, w, c] = shape
  }
  if (!(c === 1 || c === 3)) c = w > 0 && h > 0 ? Math.round(b.length / (w * h)) : 1
  const image = { width: w, height: h, channels: c, data: b }
  if (!isValidImage(image)) return null
  const px = nums(j.pixel_size_mm), o = nums(j.origin_mm)
  const ps = nf(j.psf_size), sp = nf(j.psf_spacing_mm)
  return {
    image,
    pixelSize: px.length === 2 ? [px[0], px[1]] : [NaN, NaN],
    origin: o.length === 2 ? [o[0], o[1]] : [NaN, NaN],
    nodes: readNodes(j),
    fieldGrid: nf(j.field_grid) || 0,
    psfSize: Number.isFinite(ps) ? ps : null,
    psfSpacing: Number.isFinite(sp) ? sp : null,
  }
}

// MARK: Field of view, centre, resolution

/** Field extents offered (half-width of the square in normalised field units); 1/√2 inscribes the square in the field disk. */
export const FULL_EXTENT = Math.SQRT1_2
export const EXTENTS = [FULL_EXTENT, 0.5, 0.25, 0.1, 0.05, 0.02]
export const SIZES = [128, 256, 512, 1024]

export function extentTitle(e: number): string {
  if (e >= 1) return 'full square'
  if (Math.abs(e - FULL_EXTENT) < 1e-6) return 'full field'
  if (Math.abs(e - 0.5) < 1e-9) return '½'
  if (Math.abs(e - 0.25) < 1e-9) return '¼'
  return `${Number((e * 100).toPrecision(6))} %`
}

/** The run's field extent: half the span of its field mapping nodes. */
export function extentFromRun(d: ImageSimData): number | null {
  const xs = d.nodes.map((n) => n.field[0])
  if (xs.length < 2) return null
  const lo = Math.min(...xs), hi = Math.max(...xs)
  return hi > lo ? (hi - lo) / 2 : null
}

/** The run's patch centre: the middle of its field mapping nodes. */
export function centerFromRun(d: ImageSimData): [number, number] | null {
  if (!d.nodes.length) return null
  const xs = d.nodes.map((n) => n.field[0]), ys = d.nodes.map((n) => n.field[1])
  return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2]
}

export interface FieldPointChoice {
  title: string
  field: [number, number]
}

const signed2 = (v: number) => {
  const s = (Math.abs(v) < 0.005 ? 0 : v).toFixed(2)
  return v >= 0.005 ? `+${s}` : s.replace('-', '−')
}

export const coordinates = (c: [number, number]) => `(${signed2(c[0])}, ${signed2(c[1])})`

/** The axis and the design's field points, normalised by the largest radial field. */
export function fieldPoints(system: System | null | undefined): FieldPointChoice[] {
  const out: FieldPointChoice[] = [{ title: 'On axis', field: [0, 0] }]
  const f = system?.fields?.fields ?? []
  const r = Math.max(0, ...f.map((p) => Math.hypot(nf(p.x), nf(p.y))).filter(Number.isFinite))
  if (!(r > 0)) return out
  f.forEach((p, k) => {
    const h: [number, number] = [nf(p.x) / r, nf(p.y) / r]
    if (!h.every(Number.isFinite) || out.some((q) => Math.hypot(q.field[0] - h[0], q.field[1] - h[1]) < 1e-9)) return
    out.push({ title: `Field ${k + 1} ${coordinates(h)}`, field: h })
  })
  return out
}

/** "on axis", "Field 3 (0.00, +1.00)" or "(+0.30, −0.45)". */
export function centerTitle(c: [number, number], points: FieldPointChoice[]): string {
  if (c[0] === 0 && c[1] === 0) return 'on axis'
  const p = points.find((q) => Math.hypot(q.field[0] - c[0], q.field[1] - c[1]) < 1e-4)
  return p ? p.title : coordinates(c)
}

/** The nearest of `targets` (and the axis) within `d`, else `p`. */
export function snap(p: [number, number], targets: [number, number][], d: number): [number, number] {
  let best: [number, number] | null = null, bd = Infinity
  for (const t of [[0, 0] as [number, number], ...targets]) {
    const x = Math.hypot(t[0] - p[0], t[1] - p[1])
    if (x < bd) ((bd = x), (best = t))
  }
  return best && bd < d ? best : p
}

/** Whether a corner of the patch lies outside the unit field disk. */
export const outsideField = (c: [number, number], e: number) => Math.hypot(Math.abs(c[0]) + e, Math.abs(c[1]) + e) > 1 + 1e-9

/** `[hx, hy]` rounded to 1e-4 so picked points read cleanly. */
export const centerJSON = (c: [number, number]): number[] => c.map((v) => {
  const x = Math.round(v * 1e4) / 1e4
  return x === 0 ? 0 : x
})
