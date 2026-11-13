// Sag z(x, y) of every surface shape, for the inspector's profile previews and
// departure maps only (docs/SURFACES.md §1–6; apps/macos/.../ShapeSag.swift).
// Display geometry: no ray tracing, no first order. The engine's surface math
// is the definition; these follow the same formulas.

import type { Json, JsonObject } from '@/engine/types'
import type { Shape } from './system'
import { xyTerm, zernikeTerm } from './surfaceTypes'

const num = (v: Json | undefined, d = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : d)
const arr = (v: Json | undefined): number[] => (Array.isArray(v) ? v.map((x) => num(x)) : [])

/** Rationalised conic sag with s = r², or null outside its real domain. */
export function conicSag(c: number, k: number, s: number): number | null {
  const q = 1 - (1 + k) * c * c * s
  if (q < 0) return null
  return (c * s) / (1 + Math.sqrt(q))
}

export function horner(a: number[], s: number): number {
  let v = 0
  for (let i = a.length - 1; i >= 0; i--) v = v * s + a[i]
  return v
}

/** t = (r / r_norm)², or null beyond the normalisation radius. */
function normalised(s: number, rn: number): number | null {
  if (!(rn > 0)) return null
  const t = s / (rn * rn)
  return t <= 1 + 1e-12 ? t : null
}

/** Σ a_m Q_m^con(t) with Q_m^con(t) = P_m^(0,4)(2t − 1). */
export function qCon(a: number[], t: number): number {
  const x = 2 * t - 1
  const alpha = 0
  const beta = 4
  let sum = 0
  let p0 = 1
  let p1 = alpha + 1 + ((alpha + beta + 2) * (x - 1)) / 2
  a.forEach((am, m) => {
    if (m === 0) sum += am * p0
    else if (m === 1) sum += am * p1
    else {
      const n = m
      const ab = alpha + beta
      const c1 = 2 * n * (n + ab) * (2 * n + ab - 2)
      const c2 = (2 * n + ab - 1) * ((2 * n + ab) * (2 * n + ab - 2) * x + alpha * alpha - beta * beta)
      const c3 = 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + ab)
      const p2 = (c2 * p1 - c3 * p0) / c1
      p0 = p1
      p1 = p2
      sum += am * p2
    }
  })
  return sum
}

/** Σ a_m Q_m^bfs(t) through Forbes' auxiliary basis (SURFACES §2). */
export function qBfs(a: number[], t: number): number {
  const n = a.length
  if (n === 0) return 0
  const f = new Array<number>(n).fill(0)
  const g = new Array<number>(n).fill(0)
  const h = new Array<number>(n).fill(0)
  f[0] = 2
  if (n > 1) f[1] = Math.sqrt(19) / 2
  g[0] = -0.5
  for (let m = 2; m < n; m++) {
    h[m - 2] = -(m * (m - 1)) / (2 * f[m - 2])
    g[m - 1] = -(1 + g[m - 2] * h[m - 2]) / f[m - 1]
    f[m] = Math.sqrt(m * (m + 1) + 3 - g[m - 1] * g[m - 1] - h[m - 2] * h[m - 2])
  }
  const b = new Array<number>(n + 2).fill(0)
  for (let m = n - 1; m >= 0; m--) {
    b[m] = (a[m] - (m + 1 < n ? g[m] * b[m + 1] : 0) - (m + 2 < n ? h[m] * b[m + 2] : 0)) / f[m]
  }
  const al = new Array<number>(n + 2).fill(0)
  for (let m = n - 1; m >= 0; m--) al[m] = b[m] + (2 - 4 * t) * al[m + 1] - al[m + 2]
  return 2 * (al[0] + (n > 1 ? al[1] : 0))
}

/** Σ C_j X^m Y^n in the engine's term order (SURFACES §5, at most 65 terms). */
export function xyDeparture(c: number[], x: number, y: number): number {
  let z = 0
  for (let j = 0; j < Math.min(c.length, 65); j++) {
    if (c[j] === 0) continue
    const { m, n } = xyTerm(j)
    z += c[j] * x ** m * y ** n
  }
  return z
}

const factorial = (n: number) => {
  let f = 1
  for (let i = 2; i <= n; i++) f *= i
  return f
}

/** One Zernike term: Fringe unnormalised, Standard (Noll) normalised. */
export function zernike(t: { n: number; m: number }, ordering: string, rho: number, theta: number): number {
  const m = Math.abs(t.m)
  let r = 0
  for (let k = 0; k <= (t.n - m) / 2; k++) {
    const den = factorial(k) * factorial((t.n + m) / 2 - k) * factorial((t.n - m) / 2 - k)
    r += ((k % 2 === 0 ? 1 : -1) * factorial(t.n - k)) / den * rho ** (t.n - 2 * k)
  }
  const angular = t.m > 0 ? Math.cos(m * theta) : t.m < 0 ? Math.sin(m * theta) : 1
  const norm = ordering === 'standard' ? (t.m === 0 ? Math.sqrt(t.n + 1) : Math.sqrt(2 * (t.n + 1))) : 1
  return norm * r * angular
}

/** Σ c_j Z_j(ρ, θ), θ from +x toward +y (SURFACES §6). */
export function zernikeDeparture(c: number[], ordering: string, x: number, y: number): number {
  const rho = Math.hypot(x, y)
  const theta = Math.atan2(y, x)
  const limit = ordering === 'standard' ? 45 : 37
  let z = 0
  for (let k = 0; k < Math.min(c.length, limit); k++) {
    if (c[k] === 0) continue
    z += c[k] * zernike(zernikeTerm(k + 1, ordering), ordering, rho, theta)
  }
  return z
}

/** The sag at local (x, y) in mm, or null outside the shape's domain
 *  (normalised polynomials beyond their radius included). */
export function sag(shape: Shape, x: number, y: number): number | null {
  const s = x * x + y * y
  const c = num(shape.curvature)
  const k = num(shape.conic)
  switch (shape.type) {
    case 'plane':
    case 'paraxial':
    case 'coordinate_break':
      return 0
    case 'sphere':
      return conicSag(c, 0, s)
    case 'conic':
      return conicSag(c, k, s)
    case 'even_asphere': {
      const b = conicSag(c, k, s)
      return b == null ? null : b + horner(arr(shape.coefficients), s) * s * s
    }
    case 'odd_asphere': {
      const b = conicSag(c, k, s)
      if (b == null) return null
      const r = Math.sqrt(s)
      return arr(shape.coefficients).reduce((z, a, i) => z + a * r ** (i + 1), b)
    }
    case 'q_con': {
      const b = conicSag(c, k, s)
      const t = normalised(s, num(shape.norm_radius))
      return b == null || t == null ? null : b + t * t * qCon(arr(shape.coefficients), t)
    }
    case 'q_bfs': {
      const b = conicSag(c, 0, s)
      const t = normalised(s, num(shape.norm_radius))
      if (b == null || t == null) return null
      const a = arr(shape.coefficients)
      if (a.every((v) => v === 0)) return b
      const w = 1 - c * c * s
      return w > 0 ? b + (t * (1 - t) * qBfs(a, t)) / Math.sqrt(w) : null
    }
    case 'diffractive':
      return shape.base ? sag(shape.base as Shape, x, y) : 0
    case 'toroid': {
      const cx = num(shape.curvature_x)
      const yy = y * y
      const zb = conicSag(num(shape.curvature_y), num(shape.conic_y), yy)
      if (zb == null) return null
      const zy = zb + horner(arr(shape.y_asphere), yy) * yy * yy
      if (cx === 0) return zy
      const a = 1 - cx * zy
      const d = a * a - cx * cx * x * x
      return a > 0 && d >= 0 ? zy + (cx * x * x) / (a + Math.sqrt(d)) : null
    }
    case 'biconic': {
      const cx = num(shape.curvature_x)
      const cy = num(shape.curvature_y)
      const xx = x * x
      const yy = y * y
      const w = 1 - (1 + num(shape.conic_x)) * cx * cx * xx - (1 + num(shape.conic_y)) * cy * cy * yy
      if (w < 0) return null
      return (cx * xx + cy * yy) / (1 + Math.sqrt(w)) + horner(arr(shape.x_asphere), xx) * xx * xx + horner(arr(shape.y_asphere), yy) * yy * yy
    }
    case 'xy_polynomial': {
      const rn = num(shape.norm_radius)
      const b = conicSag(c, k, s)
      if (b == null || normalised(s, rn) == null) return null
      return b + xyDeparture(arr(shape.coefficients), x / rn, y / rn)
    }
    case 'zernike_sag': {
      const rn = num(shape.norm_radius)
      const b = conicSag(c, k, s)
      if (b == null || normalised(s, rn) == null) return null
      return b + zernikeDeparture(arr(shape.coefficients), String(shape.ordering ?? 'fringe'), x / rn, y / rn)
    }
    default:
      return null
  }
}

/** The sag minus its base conic: what the polynomial terms add. */
export function departure(shape: Shape, x: number, y: number): number | null {
  const z = sag(shape, x, y)
  if (z == null) return null
  const s = x * x + y * y
  switch (shape.type) {
    case 'xy_polynomial':
    case 'zernike_sag':
    case 'q_con':
    case 'odd_asphere':
    case 'even_asphere': {
      const b = conicSag(num(shape.curvature), num(shape.conic), s)
      return b == null ? null : z - b
    }
    case 'q_bfs': {
      const b = conicSag(num(shape.curvature), 0, s)
      return b == null ? null : z - b
    }
    default:
      return z
  }
}

// MARK: Previews

export type PreviewKind = 'radial' | 'anamorphic' | 'freeform' | 'diffractive' | 'none'

export function previewKind(type: string): PreviewKind {
  switch (type) {
    case 'toroid':
    case 'biconic':
      return 'anamorphic'
    case 'xy_polynomial':
    case 'zernike_sag':
      return 'freeform'
    case 'diffractive':
      return 'diffractive'
    case 'paraxial':
    case 'coordinate_break':
      return 'none'
    default:
      return 'radial'
  }
}

/** The half-width a preview covers: the clear semi-diameter, else the
 *  normalisation radius, else 10 mm. */
export function previewRadius(shape: Shape, semiDiameter: number | null | undefined): number {
  if (semiDiameter != null && semiDiameter > 0 && Number.isFinite(semiDiameter)) return semiDiameter
  const rn = num(shape.norm_radius)
  if (rn > 0) return rn
  const p = shape.type === 'diffractive' ? num((shape.phase as JsonObject | undefined)?.norm_radius) : 0
  return p > 0 ? p : 10
}

/** Sag profile points [t, z] from −h to h along x or y, or null. */
export function profile(shape: Shape, h: number, axis: 'x' | 'y' = 'y', samples = 64): [number, number][] | null {
  if (!(h > 0) || !Number.isFinite(h)) return null
  const out: [number, number][] = []
  for (let k = 0; k <= samples; k++) {
    const t = -h + (2 * h * k) / samples
    const z = axis === 'y' ? sag(shape, 0, t) : sag(shape, t, 0)
    if (z != null && Number.isFinite(z)) out.push([t, z])
  }
  return out.length > 2 ? out : null
}

/** An n × n grid of the departure from the base over the disk of radius h,
 *  row-major from the top row (y = +h); null outside the disk or the domain. */
export function departureMap(shape: Shape, h: number, n = 40): (number | null)[] {
  const out: (number | null)[] = []
  for (let k = 0; k < n * n; k++) {
    const i = k % n
    const j = Math.floor(k / n)
    const x = -h + (2 * h * (i + 0.5)) / n
    const y = h - (2 * h * (j + 0.5)) / n
    out.push(x * x + y * y <= h * h ? departure(shape, x, y) : null)
  }
  return out
}

// MARK: Diffractive phase

export interface Phase {
  normRadius: number
  coefficients: number[]
  designOrder: number
  designWavelengthUm: number
  traceOrder: number
}

export function phaseOf(shape: Shape): Phase | null {
  if (shape.type !== 'diffractive') return null
  const p = (shape.phase as JsonObject | undefined) ?? {}
  return {
    normRadius: num(p.norm_radius),
    coefficients: arr(p.coefficients),
    designOrder: num(p.design_order, 1),
    designWavelengthUm: num(p.design_wavelength_um, 0.5875618),
    traceOrder: num(p.trace_order, 1),
  }
}

/** Φ(r) = Σ A_i (r/r_norm)^(2i) in radians (order +1). */
export function phase(p: Phase, r: number): number {
  if (!(p.normRadius > 0)) return 0
  const rho2 = (r / p.normRadius) ** 2
  return p.coefficients.reduce((acc, a, i) => acc + a * rho2 ** (i + 1), 0)
}

/** dΦ/dr in rad/mm. */
export function phaseSlope(p: Phase, r: number): number {
  if (!(p.normRadius > 0)) return 0
  const rn2 = p.normRadius * p.normRadius
  const rho2 = (r * r) / rn2
  return p.coefficients.reduce((acc, a, k) => {
    const i = k + 1
    return acc + a * i * rho2 ** (i - 1) * ((2 * r) / rn2)
  }, 0)
}

/** Paraxial power of the phase in order m at λ (µm): φ = −m λ A₁ / (π r_norm²), 1/mm. */
export function phasePower(p: Phase, order: number, wavelengthUm: number): number {
  const a1 = p.coefficients[0]
  if (a1 == null || !(p.normRadius > 0)) return 0
  return (-order * wavelengthUm * 1e-3 * a1) / (Math.PI * p.normRadius * p.normRadius)
}

/** The smallest zone period 2π / |dΦ/dr| (mm) over 0 < r ≤ h, and where. */
export function minimumPeriod(p: Phase, h: number, samples = 400): { period: number; radius: number } | null {
  if (!(h > 0) || !Number.isFinite(h)) return null
  let best: { period: number; radius: number } | null = null
  for (let k = 1; k <= samples; k++) {
    const r = (h * k) / samples
    const slope = Math.abs(phaseSlope(p, r))
    if (!(slope > 0) || !Number.isFinite(slope)) continue
    const period = (2 * Math.PI) / slope
    if (!best || period < best.period) best = { period, radius: r }
  }
  return best
}

// MARK: Diffraction efficiency

export type EfficiencyModel = { type: 'ideal' } | { type: 'kinoform' } | { type: 'multilevel'; levels: number }

export function efficiencyModel(j: Json | undefined): EfficiencyModel {
  const o = (j && typeof j === 'object' && !Array.isArray(j) ? j : {}) as JsonObject
  if (o.type === 'kinoform') return { type: 'kinoform' }
  if (o.type === 'multilevel') return { type: 'multilevel', levels: Math.max(1, Math.round(num(o.levels, 8))) }
  return { type: 'ideal' }
}

/** The efficiency of `order` for a relief blazed at `blaze` (α·m₀), `selected`
 *  being the trace order; mirrors the engine's `DiffractionEfficiency`. */
export function efficiency(model: EfficiencyModel, order: number, selected: number, blaze: number): number {
  const sinc = (x: number) => {
    if (x === 0) return 1
    const r = x - Math.round(x)
    return Math.sin(Math.PI * r) / (Math.PI * x)
  }
  const m = order
  switch (model.type) {
    case 'ideal':
      return order === selected ? 1 : 0
    case 'kinoform':
      return sinc(m - blaze) ** 2
    case 'multilevel': {
      const n = model.levels
      let r = (blaze - m) / n
      r -= Math.round(r)
      return ((sinc(m / n) * sinc(n * r)) / sinc(r)) ** 2
    }
  }
}

/** The trace order's efficiency at λ (µm), blazed for the design order at λ₀
 *  (non-dispersive estimate away from λ₀). */
export function traceEfficiency(model: EfficiencyModel, p: Phase, um: number): number {
  if (!(um > 0)) return NaN
  const blaze = (p.designOrder * p.designWavelengthUm) / um
  return efficiency(model, p.traceOrder, p.traceOrder, blaze)
}
