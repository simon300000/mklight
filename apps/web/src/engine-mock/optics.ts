// MOCK OPTICS — illustrative only, not the engine. Port of the Swift
// `MockOptics`: nominal Cauchy indices fitted to n_d / V_d, a rotationally
// symmetric real-ray trace that ignores placements, ray aiming and
// configurations.
import { CATALOG, apertureBoundingRadius, type Mat, type Shape, type Surf, type Sys } from './model'

export type V3 = [number, number, number]
export interface Ray {
  p: V3
  d: V3
}
export interface Trace {
  /** Global points: start, then every surface hit. */
  points: V3[]
  failedAt: number | null
  atImage: Ray | null
  /** Radial height at each surface index. */
  heights: Map<number, number>
  ok: boolean
}
export type FieldPoint = { kind: 'angle'; tx: number; ty: number } | { kind: 'height'; x: number; y: number }

const LINE_F = 0.4861327
const LINE_C = 0.6562725
const LINE_D = 0.5875618

export function cauchy(nd: number, vd: number, um: number): number {
  const b = (nd - 1) / vd / (1 / (LINE_F * LINE_F) - 1 / (LINE_C * LINE_C))
  return nd + b * (1 / (um * um) - 1 / (LINE_D * LINE_D))
}

export function nominalIndex(m: Mat, um: number, warn: Set<string>): number {
  switch (m.kind) {
    case 'air':
    case 'vacuum':
      return 1
    case 'constant':
      return m.n
    case 'model':
      return cauchy(m.nd, Math.max(m.vd, 1), um)
    case 'catalog': {
      const g = CATALOG[m.name.toUpperCase()]
      if (g) return cauchy(g.nd + m.dnd, g.vd + m.dvd, um)
      warn.add(`unknown glass '${m.name}': mock uses n_d = 1.5, V_d = 50`)
      return cauchy(1.5, 50, um)
    }
  }
}

/** (nd, vd) of a material at the d line, for the summary tables. */
export function materialNdVd(m: Mat): { nd: number; vd: number | null } {
  switch (m.kind) {
    case 'air':
    case 'vacuum':
      return { nd: 1, vd: null }
    case 'constant':
      return { nd: m.n, vd: null }
    case 'model':
      return { nd: m.nd, vd: m.vd }
    case 'catalog': {
      const g = CATALOG[m.name.toUpperCase()]
      return g ? { nd: g.nd + m.dnd, vd: g.vd + m.dvd } : { nd: 1.5, vd: 50 }
    }
  }
}

export function shapeCurvature(s: Shape): number {
  switch (s.type) {
    case 'conic':
    case 'even_asphere':
    case 'odd_asphere':
    case 'q_con':
    case 'q_bfs':
    case 'xy_polynomial':
    case 'zernike_sag':
      return Number(s.curvature) || 0
    case 'toroid':
    case 'biconic':
      return Number(s.curvature_y) || 0
    case 'diffractive':
    case 'sag_overlay':
      return s.base ? shapeCurvature(s.base as Shape) : 0
    default:
      return 0
  }
}

export function shapeConic(s: Shape): number {
  switch (s.type) {
    case 'conic':
    case 'even_asphere':
    case 'odd_asphere':
    case 'q_con':
    case 'xy_polynomial':
    case 'zernike_sag':
      return Number(s.conic) || 0
    case 'toroid':
    case 'biconic':
      return Number(s.conic_y) || 0
    case 'diffractive':
    case 'sag_overlay':
      return s.base ? shapeConic(s.base as Shape) : 0
    default:
      return 0
  }
}

// --- Vector helpers -------------------------------------------------------

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
function norm(v: V3): V3 {
  const l = Math.hypot(v[0], v[1], v[2])
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : v
}

export function linspace(a: number, b: number, n: number): number[] {
  if (n <= 1) return [(a + b) / 2]
  return Array.from({ length: n }, (_, k) => a + (k / (n - 1)) * (b - a))
}

// --- Surface sag, intersection, interaction ------------------------------

type Sag = { z: number; dz: number }

function conicSag(c: number, k: number, rho: number): Sag | null {
  const arg = 1 - (1 + k) * c * c * rho
  if (arg < 0) return null
  const s = Math.sqrt(arg)
  return { z: (c * rho) / (1 + s), dz: s > 0 ? c / (2 * s) : Infinity }
}

/** Sag z(ρ) and dz/dρ with ρ = r². */
export function sag(shape: Shape, rho: number): Sag | null {
  switch (shape.type) {
    case 'plane':
    case 'paraxial':
    case 'coordinate_break':
      return { z: 0, dz: 0 }
    case 'conic':
      return conicSag(Number(shape.curvature) || 0, Number(shape.conic) || 0, rho)
    case 'even_asphere':
      return evenSag(Number(shape.curvature) || 0, Number(shape.conic) || 0, shape.coefficients ?? [], rho)
    case 'odd_asphere': {
      const c0 = conicSag(Number(shape.curvature) || 0, Number(shape.conic) || 0, rho)
      if (!c0) return null
      const r = Math.sqrt(rho)
      let z = c0.z
      let dz = c0.dz
      ;((shape.coefficients ?? []) as number[]).forEach((b, i) => {
        const p = i + 1
        z += b * Math.pow(r, p)
        if (r > 0) dz += ((b * p) / 2) * Math.pow(r, p - 2)
      })
      return { z, dz }
    }
    case 'diffractive':
    case 'sag_overlay':
      return sag(shape.base as Shape, rho)
    case 'toroid':
    case 'biconic':
      return evenSag(Number(shape.curvature_y) || 0, Number(shape.conic_y) || 0, shape.y_asphere ?? [], rho)
    case 'q_con':
    case 'xy_polynomial':
    case 'zernike_sag':
      return conicSag(Number(shape.curvature) || 0, Number(shape.conic) || 0, rho)
    case 'q_bfs':
      return conicSag(Number(shape.curvature) || 0, 0, rho)
    default:
      return { z: 0, dz: 0 }
  }
}

function evenSag(c: number, k: number, a: number[], rho: number): Sag | null {
  const c0 = conicSag(c, k, rho)
  if (!c0) return null
  let poly = 0
  let dpoly = 0
  for (let i = a.length - 1; i >= 0; i--) {
    poly = poly * rho + a[i]
    dpoly = dpoly * rho + (i + 2) * a[i]
  }
  return { z: c0.z + poly * rho * rho, dz: c0.dz + dpoly * rho }
}

/** Ray–surface intersection distance in the surface's local frame. */
export function intersect(shape: Shape, q: V3, d: V3): number | null {
  const planeT = () => (Math.abs(d[2]) > 1e-15 ? -q[2] / d[2] : null)
  const conicT = (c: number, k: number): number | null => {
    const a = c * (d[0] * d[0] + d[1] * d[1] + (1 + k) * d[2] * d[2])
    const b = c * (q[0] * d[0] + q[1] * d[1] + (1 + k) * q[2] * d[2]) - d[2]
    const cc = c * (q[0] * q[0] + q[1] * q[1] + (1 + k) * q[2] * q[2]) - 2 * q[2]
    const disc = b * b - a * cc
    if (disc < 0) return null
    const root = Math.sqrt(disc)
    const denom = -b + (b <= 0 ? root : -root)
    if (!(Math.abs(denom) > 1e-300)) return null
    const t = cc / denom
    const px = q[0] + t * d[0]
    const py = q[1] + t * d[1]
    if (conicSag(c, k, px * px + py * py) === null) return null
    return t
  }
  switch (shape.type) {
    case 'plane':
    case 'paraxial':
    case 'coordinate_break':
      return planeT()
    case 'conic':
      return conicT(Number(shape.curvature) || 0, Number(shape.conic) || 0)
    case 'diffractive':
      return intersect(shape.base as Shape, q, d)
    default: {
      let t = conicT(shapeCurvature(shape), shapeConic(shape)) ?? planeT()
      if (t === null) return null
      for (let it = 0; it < 30; it++) {
        const px = q[0] + t * d[0]
        const py = q[1] + t * d[1]
        const pz = q[2] + t * d[2]
        const s = sag(shape, px * px + py * py)
        if (!s) return null
        const g = pz - s.z
        const gp = d[2] - s.dz * 2 * (px * d[0] + py * d[1])
        if (!(Math.abs(gp) > 1e-14) || !Number.isFinite(gp)) return null
        const step = g / gp
        t -= step
        if (Math.abs(step) <= 1e-13 * (1 + Math.abs(t))) return t
      }
      return null
    }
  }
}

/** New direction after the surface interaction (null on TIR or failure). */
function interact(s: Surf, hit: V3, d: V3, n1: number, n2: number): V3 | null {
  if (s.interaction === 'dummy') return d
  if (s.shape.type === 'paraxial') {
    const f = Number(s.shape.focal_length) || 0
    if (!(Math.abs(d[2]) > 1e-12) || f === 0) return null
    const sz = d[2] > 0 ? 1 : -1
    const target: V3 = [(f * d[0]) / Math.abs(d[2]), (f * d[1]) / Math.abs(d[2]), sz * f]
    const sgn = f > 0 ? 1 : -1
    return norm([sgn * (target[0] - hit[0]), sgn * (target[1] - hit[1]), sgn * target[2]])
  }
  const sg = sag(s.shape, hit[0] * hit[0] + hit[1] * hit[1])
  if (!sg || !Number.isFinite(sg.dz)) return null
  let nrm = norm([-2 * hit[0] * sg.dz, -2 * hit[1] * sg.dz, 1])
  let cosI = dot(d, nrm)
  if (cosI < 0) {
    nrm = [-nrm[0], -nrm[1], -nrm[2]]
    cosI = -cosI
  }
  if (s.interaction === 'reflect')
    return norm([d[0] - 2 * cosI * nrm[0], d[1] - 2 * cosI * nrm[1], d[2] - 2 * cosI * nrm[2]])
  const mu = n1 / n2
  const rad = 1 - mu * mu * ((1 - cosI) * (1 + cosI))
  if (rad < 0) return null
  const k = Math.sqrt(rad) - mu * cosI
  return norm([mu * d[0] + k * nrm[0], mu * d[1] + k * nrm[1], mu * d[2] + k * nrm[2]])
}

function passes(s: Surf, x: number, y: number): boolean {
  const r2 = x * x + y * y
  const ap = s.aperture
  if (ap) {
    switch (ap.type) {
      case 'circular':
        return r2 <= ap.radius * ap.radius
      case 'annular':
        return r2 >= ap.inner * ap.inner && r2 <= ap.outer * ap.outer
      case 'rectangular':
        return Math.abs(x) <= ap.hx && Math.abs(y) <= ap.hy
      case 'elliptical': {
        const u = x / ap.sx
        const v = y / ap.sy
        return u * u + v * v <= 1
      }
    }
  }
  if (s.sdFixed !== null) return r2 <= s.sdFixed * s.sdFixed
  return true
}

export interface Paraxial {
  y: number[]
  /** Reduced slope n·u after each surface (signed index). */
  nu: number[]
}

export class MockOptics {
  readonly sys: Sys
  readonly wavelengthsUm: number[]
  readonly primary: number
  warnings: string[] = []
  /** Global z of each surface vertex (surface 1 at z = 0). */
  readonly vertexZ: number[]
  /** Positive index of the medium after surface i at wavelength w. */
  readonly nAfter: number[][]
  /** Propagation sign (±1 along z) after surface i. */
  readonly travelSign: number[]
  readonly objectDistance: number | null
  epZ = 0
  epRadius = 1
  chiefImagePerUnit = 0

  constructor(sys: Sys) {
    this.sys = sys
    const wls = sys.wavelengths.map((w) => w.um)
    this.wavelengthsUm = wls
    this.primary = sys.primary
    const surfaces = sys.surfaces
    const n = surfaces.length
    const warn = new Set<string>()

    const z = new Array<number>(n).fill(0)
    for (let i = 1; i < Math.max(n - 1, 1); i++) z[i + 1] = z[i] + surfaces[i].thickness
    this.objectDistance = sys.objectAtInfinity ? null : surfaces[0].thickness
    z[0] = -(this.objectDistance ?? 0)
    this.vertexZ = z

    const media: number[][] = new Array(n)
    const signs = new Array<number>(n).fill(1)
    let before = surfaces[0].material
    let sign = 1
    for (let i = 0; i < n; i++) {
      const s = surfaces[i]
      const inherit = i > 0 && (s.interaction !== 'refract' || s.isCB)
      const after = inherit ? before : s.material
      media[i] = wls.map((w) => nominalIndex(after, w, warn))
      if (i > 0 && s.interaction === 'reflect' && !s.isCB) sign = -sign
      signs[i] = sign
      before = after
    }
    this.nAfter = media
    this.travelSign = signs
    if (surfaces.some((s) => s.hasPlacement)) warn.add('mock engine ignores tilts and decenters')
    if (sys.rayAimingReal) warn.add('mock engine ignores ray aiming')
    this.warnings = [...warn].sort()
    this.computePupils()
  }

  nBefore(i: number, w: number): number {
    return this.nAfter[Math.max(i - 1, 0)][w]
  }

  paraxial(y1: number, u0: number, w: number): Paraxial {
    const surfaces = this.sys.surfaces
    const n = surfaces.length
    const ys = new Array<number>(n).fill(0)
    const nus = new Array<number>(n).fill(0)
    let bigN = this.nAfter[0][w]
    let nu = bigN * u0
    let y = y1
    nus[0] = nu
    for (let i = 1; i < n; i++) {
      ys[i] = y
      const s = surfaces[i]
      const nPrime = this.nAfter[i][w] * this.travelSign[i]
      let phi = 0
      if (!s.isCB && s.interaction !== 'dummy') {
        const sh = s.shape
        switch (sh.type) {
          case 'conic':
          case 'even_asphere':
            phi = (Number(sh.curvature) || 0) * (nPrime - bigN)
            break
          case 'paraxial': {
            const f = Number(sh.focal_length) || 0
            phi = f !== 0 ? 1 / f : 0
            break
          }
          case 'plane':
          case 'coordinate_break':
            phi = 0
            break
          case 'diffractive': {
            const ph = (sh.phase ?? {}) as { coefficients?: number[]; trace_order?: number; norm_radius?: number }
            const a1 = ph.coefficients?.[0] ?? 0
            const lambdaMM = this.wavelengthsUm[w] * 1e-3
            const nr = ph.norm_radius || 1
            phi =
              shapeCurvature(sh) * (nPrime - bigN) -
              ((ph.trace_order ?? 1) * lambdaMM * a1) / (Math.PI * nr * nr)
            break
          }
          default:
            phi = shapeCurvature(sh) * (nPrime - bigN)
        }
      }
      nu -= y * phi
      nus[i] = nu
      if (i < n - 1) y += (s.thickness / nPrime) * nu
      bigN = nPrime
    }
    return { y: ys, nu: nus }
  }

  imageN(w: number): number {
    const k = this.sys.surfaces.length - 2
    return this.nAfter[k][w] * this.travelSign[k]
  }

  private computePupils(): void {
    const sys = this.sys
    const n = sys.surfaces.length
    const stop = Math.min(Math.max(sys.stop, 1), n - 2)
    const w = this.primary
    const a = this.paraxial(1, 0, w).y[stop]
    const b = this.paraxial(0, 1, w).y[stop]
    this.epZ = Math.abs(a) > 1e-12 ? b / a : 0
    const d = this.objectDistance
    const st = sys.surfaces[stop]
    const stopSD = st.sdFixed ?? (st.aperture ? apertureBoundingRadius(st.aperture) : null)
    const val = sys.aperture.value ?? 0
    switch (sys.aperture.type) {
      case 'entrance_pupil_diameter':
        this.epRadius = val / 2
        break
      case 'image_space_f_number':
        this.epRadius = Math.abs(this.efl(w) ?? 100) / (2 * Math.max(val, 1e-6))
        if (d !== null) this.warnings.push('image-space F/# assumes an object at infinity')
        break
      case 'object_space_na': {
        const u = val / this.nAfter[0][w]
        this.epRadius = Math.abs(u * (d !== null ? this.objectToPupil(d) : this.epZ))
        break
      }
      case 'paraxial_working_f_number': {
        const uTarget = 1 / (2 * Math.max(val, 1e-6) * Math.abs(this.imageN(w)))
        if (d !== null) {
          const pr = this.paraxial(d, 1, w)
          const u = Math.abs(pr.nu[pr.nu.length - 2] / this.imageN(w))
          this.epRadius = u > 0 ? (uTarget / u) * Math.abs(this.objectToPupil(d)) : 1
        } else {
          const pr = this.paraxial(1, 0, w)
          const u = Math.abs(pr.nu[pr.nu.length - 2] / this.imageN(w))
          this.epRadius = u > 0 ? uTarget / u : 1
        }
        break
      }
      default: {
        // float_by_stop_size
        const rs = stopSD ?? 1
        if (stopSD === null) this.warnings.push('float-by-stop without a fixed stop size: mock uses 1 mm')
        if (d !== null) {
          const u = Math.abs(a * d + b) > 1e-12 ? rs / (a * d + b) : 0
          this.epRadius = Math.abs(u * this.objectToPupil(d))
        } else {
          this.epRadius = Math.abs(a) > 1e-12 ? Math.abs(rs / a) : 1
        }
      }
    }
    if (d !== null && Math.abs(d + this.epZ) <= 1e-9 * Math.max(1, Math.abs(d), Math.abs(this.epZ)))
      this.warnings.push('object lies on the entrance pupil: paraxial pupil slopes are degenerate')
    if (!(Number.isFinite(this.epRadius) && this.epRadius > 0)) {
      this.warnings.push('entrance pupil undefined for this aperture specification: mock uses a 1 mm radius')
      this.epRadius = 1
    }
    const unit = this.chiefStart()
    const pr = this.paraxial(unit.y1, unit.u0, w)
    this.chiefImagePerUnit = pr.y[pr.y.length - 1]
  }

  private chiefStart(): { y1: number; u0: number } {
    const d = this.objectDistance
    if (d !== null) {
      const u0 = -1 / this.objectToPupil(d)
      return { y1: 1 + u0 * d, u0 }
    }
    return { y1: -this.epZ, u0: 1 }
  }

  objectToPupil(d: number): number {
    const l = d + this.epZ
    const floor = 1e-9 * Math.max(1, Math.abs(d), Math.abs(this.epZ))
    return Math.abs(l) > floor ? l : l < 0 ? -floor : floor
  }

  marginalStart(): { y1: number; u0: number } {
    const d = this.objectDistance
    if (d === null) return { y1: this.epRadius, u0: 0 }
    const u0 = this.epRadius / this.objectToPupil(d)
    return { y1: u0 * d, u0 }
  }

  efl(w: number): number | null {
    const pr = this.paraxial(1, 0, w)
    const nu = pr.nu[pr.nu.length - 2]
    return Math.abs(nu) > 1e-15 ? -1 / nu : null
  }

  workingFNumber(w: number): number {
    const m = this.marginalStart()
    const pr = this.paraxial(m.y1, m.u0, w)
    const nu = Math.abs(pr.nu[pr.nu.length - 2])
    return nu > 1e-12 ? 1 / (2 * nu) : Infinity
  }

  // --- Fields ---

  get maxField(): number {
    const m = Math.max(0, ...this.sys.fields.map((f) => Math.hypot(f.x, f.y)))
    return m > 0 ? m : 1
  }

  get fieldUnit(): string {
    return this.sys.fieldKind === 'object_angle' ? '°' : 'mm'
  }

  fieldPointXY(x: number, y: number): FieldPoint {
    const d = this.objectDistance
    switch (this.sys.fieldKind) {
      case 'object_angle': {
        const tx = Math.tan((x * Math.PI) / 180)
        const ty = Math.tan((y * Math.PI) / 180)
        if (d !== null) return { kind: 'height', x: -(this.epZ + d) * tx, y: -(this.epZ + d) * ty }
        return { kind: 'angle', tx, ty }
      }
      case 'object_height':
        return { kind: 'height', x, y }
      default: {
        const k = Math.abs(this.chiefImagePerUnit) > 1e-12 ? 1 / this.chiefImagePerUnit : 0
        return d === null ? { kind: 'angle', tx: x * k, ty: y * k } : { kind: 'height', x: x * k, y: y * k }
      }
    }
  }

  fieldPoint(i: number): FieldPoint {
    const f = this.sys.fields[i]
    return this.fieldPointXY(f.x, f.y)
  }

  fieldParameterY(fp: FieldPoint): number {
    return fp.kind === 'angle' ? fp.ty : fp.y
  }

  get trackLength(): number {
    const z = this.vertexZ
    return Math.abs(z[z.length - 1] - z[Math.min(1, z.length - 1)])
  }

  // --- Real rays ---

  makeRay(fp: FieldPoint, px: number, py: number): Ray {
    const pupil: V3 = [px * this.epRadius, py * this.epRadius, this.epZ]
    if (fp.kind === 'angle') {
      const d = norm([fp.tx, fp.ty, 1])
      const zStart = Math.min(0, this.epZ) - Math.max(0.15 * this.trackLength, 5)
      const t = (zStart - this.epZ) / d[2]
      return { p: [pupil[0] + d[0] * t, pupil[1] + d[1] * t, pupil[2] + d[2] * t], d }
    }
    const o: V3 = [fp.x, fp.y, this.vertexZ[0]]
    return { p: o, d: norm([pupil[0] - o[0], pupil[1] - o[1], pupil[2] - o[2]]) }
  }

  trace(start: Ray, w: number): Trace {
    const out: Trace = { points: [start.p], failedAt: null, atImage: null, heights: new Map(), ok: true }
    const fail = (i: number): Trace => {
      out.failedAt = i
      out.ok = false
      return out
    }
    let ray = start
    const surfaces = this.sys.surfaces
    const n = surfaces.length
    for (let i = 1; i < n; i++) {
      const s = surfaces[i]
      if (s.isCB) continue
      const vz = this.vertexZ[i]
      const q: V3 = [ray.p[0], ray.p[1], ray.p[2] - vz]
      const t = intersect(s.shape, q, ray.d)
      if (t === null) return fail(i)
      const hit: V3 = [q[0] + t * ray.d[0], q[1] + t * ray.d[1], q[2] + t * ray.d[2]]
      const global: V3 = [hit[0], hit[1], hit[2] + vz]
      out.points.push(global)
      out.heights.set(i, Math.hypot(hit[0], hit[1]))
      if (!passes(s, hit[0], hit[1])) return fail(i)
      if (i === n - 1) {
        out.atImage = { p: global, d: ray.d }
        break
      }
      const d = interact(s, hit, ray.d, this.nBefore(i, w), this.nAfter[i][w])
      if (!d) return fail(i)
      ray = { p: global, d }
    }
    return out
  }
}
