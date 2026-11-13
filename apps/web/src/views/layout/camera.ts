// View transforms of the layout (as ViewportMath.swift): orthographic only,
// optical layouts are read with true proportions. World → view: world +z (the
// optical axis) to the right and world +y up, so the 2-D view is the
// meridional Y–Z section; the X–Z section puts world x up; the 3-D view
// orbits the Y–Z basis by `yaw` (about view up) and `pitch` (about view right).

export type LayoutMode = 'yz' | 'xz' | '3d'
export type Vec3 = [number, number, number]
/** Row-major 3×3 rotation, world → view. */
export type Mat3 = number[]

export interface Camera {
  /** World point (mm) at the centre of the view. */
  center: Vec3
  /** Zoom in CSS pixels per millimetre. */
  ppmm: number
  yaw: number
  pitch: number
}

export const defaultCamera = (): Camera => ({ center: [0, 0, 0], ppmm: 4, yaw: -0.55, pitch: 0.35 })

/** The vertical stretches offered for the section views. */
export const STRETCHES = [1, 2, 5, 10, 20, 50]

export const verticalScale = (mode: LayoutMode, stretch: number) => (mode === '3d' ? 1 : Math.max(stretch, 1e-3))

function mul(a: Mat3, b: Mat3): Mat3 {
  const o = new Array(9).fill(0)
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) o[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c]
  return o
}

/** Rotation from world to view axes. */
export function rotation(cam: Pick<Camera, 'yaw' | 'pitch'>, mode: LayoutMode): Mat3 {
  // view x = world z, view y = world y, view z = −world x.
  const basis = [0, 0, 1, 0, 1, 0, -1, 0, 0]
  if (mode === 'xz') return [0, 0, 1, 1, 0, 0, 0, 1, 0]
  if (mode === 'yz') return basis
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw)
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch)
  const yawM = [cy, 0, sy, 0, 1, 0, -sy, 0, cy]
  const pitchM = [1, 0, 0, 0, cp, -sp, 0, sp, cp]
  return mul(pitchM, mul(yawM, basis))
}

export const apply = (r: Mat3, p: ArrayLike<number>, o = 0): Vec3 => [
  r[0] * p[o] + r[1] * p[o + 1] + r[2] * p[o + 2],
  r[3] * p[o] + r[4] * p[o + 1] + r[5] * p[o + 2],
  r[6] * p[o] + r[7] * p[o + 1] + r[8] * p[o + 2],
]
const applyT = (r: Mat3, v: Vec3): Vec3 => [
  r[0] * v[0] + r[3] * v[1] + r[6] * v[2],
  r[1] * v[0] + r[4] * v[1] + r[7] * v[2],
  r[2] * v[0] + r[5] * v[1] + r[8] * v[2],
]

export interface Size {
  w: number
  h: number
}

/** A projector from world points to screen pixels (y down) and depth (towards the viewer). */
export function projector(cam: Camera, mode: LayoutMode, size: Size, stretch: number) {
  const r = rotation(cam, mode)
  const c = apply(r, cam.center)
  const vs = verticalScale(mode, stretch)
  const k = cam.ppmm
  return (p: ArrayLike<number>, o = 0): [number, number, number] => {
    const x = r[0] * p[o] + r[1] * p[o + 1] + r[2] * p[o + 2]
    const y = r[3] * p[o] + r[4] * p[o + 1] + r[5] * p[o + 2]
    const z = r[6] * p[o] + r[7] * p[o + 1] + r[8] * p[o + 2]
    return [size.w / 2 + (x - c[0]) * k, size.h / 2 - (y - c[1]) * k * vs, z - c[2]]
  }
}

/** Width and height (mm, true proportions) of `points` (flat xyz) in the view. */
export function extent(cam: Camera, mode: LayoutMode, points: ArrayLike<number>): { w: number; h: number; lo: Vec3; hi: Vec3 } {
  const r = rotation(cam, mode)
  const lo: Vec3 = [Infinity, Infinity, Infinity]
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i + 2 < points.length; i += 3) {
    const v = apply(r, points, i)
    if (!v.every(Number.isFinite)) continue
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], v[k])
      hi[k] = Math.max(hi[k], v[k])
    }
  }
  if (!Number.isFinite(lo[0])) return { w: 0, h: 0, lo: [0, 0, 0], hi: [0, 0, 0] }
  return { w: hi[0] - lo[0], h: hi[1] - lo[1], lo, hi }
}

/** A camera that frames `points` with a margin, keeping the orbit angles. */
export function fit(cam: Camera, mode: LayoutMode, points: ArrayLike<number>, size: Size, stretch: number): Camera {
  if (points.length < 3 || size.w <= 1 || size.h <= 1) return cam
  const e = extent(cam, mode, points)
  const r = rotation(cam, mode)
  const mid: Vec3 = [(e.lo[0] + e.hi[0]) / 2, (e.lo[1] + e.hi[1]) / 2, (e.lo[2] + e.hi[2]) / 2]
  const w = Math.max(e.w, 1e-6), h = Math.max(e.h, 1e-6) * verticalScale(mode, stretch)
  return { ...cam, center: applyT(r, mid), ppmm: Math.min(size.w / (w * 1.12), size.h / (h * 1.12)) }
}

/** Pans by a screen delta in pixels (y down). */
export function pan(cam: Camera, mode: LayoutMode, dx: number, dy: number, stretch: number): Camera {
  const r = rotation(cam, mode)
  const d = applyT(r, [dx / cam.ppmm, -dy / (cam.ppmm * verticalScale(mode, stretch)), 0])
  return { ...cam, center: [cam.center[0] - d[0], cam.center[1] - d[1], cam.center[2] - d[2]] }
}

/** Zooms by `factor`, keeping the point under the cursor (pixels from the view centre, y down) fixed. */
export function zoom(cam: Camera, mode: LayoutMode, factor: number, cx: number, cy: number, stretch: number): Camera {
  if (!Number.isFinite(factor) || factor <= 0) return cam
  const next = Math.min(Math.max(cam.ppmm * factor, 1e-4), 1e6)
  const o: Vec3 = [cx, -cy / verticalScale(mode, stretch), 0]
  const shift: Vec3 = [o[0] / cam.ppmm - o[0] / next, o[1] / cam.ppmm - o[1] / next, 0]
  const d = applyT(rotation(cam, mode), shift)
  return { ...cam, ppmm: next, center: [cam.center[0] + d[0], cam.center[1] + d[1], cam.center[2] + d[2]] }
}

export function orbit(cam: Camera, dx: number, dy: number): Camera {
  return { ...cam, yaw: cam.yaw + dx * 0.01, pitch: Math.min(Math.max(cam.pitch - dy * 0.01, -1.55), 1.55) }
}

/** The stretch that makes a section `height` pixels tall (at true
 *  proportions) readable in a view `viewHeight` tall: 1 unless it fills less
 *  than an eighth of the view, else the smallest offered stretch that brings
 *  it to a third (at most ×50). */
export function automaticStretch(height: number, viewHeight: number): number {
  if (!(height > 0) || !(viewHeight > 0) || height >= viewHeight / 8) return 1
  const want = viewHeight / 3 / height
  return STRETCHES.find((s) => s >= want) ?? STRETCHES[STRETCHES.length - 1]
}

/** A round length (1-2-5 series) close to `target` mm. */
export function niceLength(target: number): number {
  if (!(target > 0) || !Number.isFinite(target)) return 1
  const e = 10 ** Math.floor(Math.log10(target))
  const m = target / e
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * e
}

export function lengthLabel(mm: number): string {
  if (mm >= 1000) return `${+(mm / 1000).toPrecision(3)} m`
  if (mm < 1) return `${+(mm * 1000).toPrecision(3)} µm`
  return `${+mm.toPrecision(3)} mm`
}
