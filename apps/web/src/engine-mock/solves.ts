// MOCK ENGINE — `solves` (paraxial evaluation of system.solves) and
// `convert_spec` (field kind and aperture type conversion on the paraxial model).
import type { Json, JsonObject } from '../engine/types'
import { MockOptics, sag } from './optics'
import { parseSystem, type Obj } from './model'
import { newOut, nf, semiDiameters, type Out } from './analyses'
import { asPath, clone, getParamValue, setParamValue, surfaceIndex, surfaceOf, type Path } from './params'

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d)
const optics = (system: Obj) => new MockOptics(parseSystem(system as JsonObject))

/** Chief-ray scale: the paraxial chief ray of the maximum field. */
function chiefScale(o: MockOptics): number {
  if (o.sys.fieldKind === 'object_angle') return Math.tan((o.maxField * Math.PI) / 180)
  if (o.sys.fieldKind === 'object_height') return o.maxField
  return Math.abs(o.chiefImagePerUnit) > 1e-12 ? o.maxField / o.chiefImagePerUnit : 1
}

function chiefStart(o: MockOptics): { y1: number; u0: number } {
  const d = o.objectDistance
  const k = chiefScale(o)
  if (d !== null) {
    const u0 = -1 / o.objectToPupil(d)
    return { y1: k * (1 + u0 * d), u0: k * u0 }
  }
  return { y1: -o.epZ * k, u0: k }
}

interface RayState {
  /** Height at each surface, reduced slope n·u after each surface (signed). */
  y: number[]
  nu: number[]
}
const marginal = (o: MockOptics): RayState => {
  const m = o.marginalStart()
  return o.paraxial(m.y1, m.u0, o.primary)
}
const chief = (o: MockOptics): RayState => {
  const c = chiefStart(o)
  return o.paraxial(c.y1, c.u0, o.primary)
}
const nPrime = (o: MockOptics, i: number) => o.nAfter[i][o.primary] * o.travelSign[i]
const nBefore = (o: MockOptics, i: number) => (i <= 0 ? o.nAfter[0][o.primary] : o.nAfter[i - 1][o.primary] * o.travelSign[i - 1])

export interface SolveDef {
  target: Path
  kind: Obj
}

/** The value the solve gives its target in the current system (closed form), or null. */
function closedForm(system: Obj, sv: SolveDef): { value: number | null; note?: string } {
  const o = optics(system)
  const idx = typeof sv.target.surface === 'number' ? surfaceIndex(system, sv.target.surface) : -1
  const k = sv.kind
  const cur = getParamValue(system, sv.target)
  switch (k.type) {
    case 'pickup': {
      const src = asPath(k.source)
      const v = src ? getParamValue(system, src) : null
      return { value: v === null ? null : num(k.scale, 1) * v + num(k.offset, 0) }
    }
    case 'marginal_ray_height':
    case 'chief_ray_height': {
      if (sv.target.param !== 'thickness' || idx < 1) return { value: null, note: 'needs a thickness target' }
      const r = k.type === 'marginal_ray_height' ? marginal(o) : chief(o)
      const u = r.nu[idx] / nPrime(o, idx)
      return Math.abs(u) > 1e-14 ? { value: (num(k.height, 0) - r.y[idx]) / u } : { value: null, note: 'ray is parallel to the axis' }
    }
    case 'marginal_ray_angle':
    case 'chief_ray_angle': {
      if (idx < 1) return { value: null }
      const r = k.type === 'marginal_ray_angle' ? marginal(o) : chief(o)
      const nu = num(k.slope, 0) * nPrime(o, idx)
      const y = r.y[idx]
      const n0 = nBefore(o, idx)
      if (Math.abs(y) < 1e-12) return { value: null, note: 'ray height is zero on the surface' }
      if (sv.target.param === 'paraxial_focal_length') {
        const dnu = r.nu[idx - 1] - nu
        return Math.abs(dnu) > 1e-14 ? { value: y / dnu } : { value: null }
      }
      if (sv.target.param !== 'curvature') return { value: null, note: 'needs a curvature or focal length' }
      const dn = nPrime(o, idx) - n0
      return Math.abs(dn) > 1e-12 ? { value: (r.nu[idx - 1] - nu) / (y * dn) } : { value: null }
    }
    case 'position': {
      const from = surfaceIndex(system, num(k.from, -1))
      const to = surfaceIndex(system, num(k.to, -1))
      if (sv.target.param !== 'thickness' || from < 0 || to < 0 || idx < Math.min(from, to) || idx >= Math.max(from, to)) return { value: null, note: 'target must lie between the surfaces' }
      let others = 0
      for (let i = Math.min(from, to); i < Math.max(from, to); i++) if (i !== idx) others += parseSysThickness(system, i)
      return { value: num(k.distance, 0) - others }
    }
    case 'center_of_curvature': {
      const sIdx = surfaceIndex(system, num(k.surface, -1))
      if (sv.target.param !== 'curvature' || sIdx < 0 || idx < 0) return { value: null }
      const dz = o.vertexZ[sIdx] - o.vertexZ[idx]
      return Math.abs(dz) > 1e-12 ? { value: 1 / dz } : { value: null }
    }
    case 'edge_thickness': {
      if (sv.target.param !== 'thickness' || idx < 1 || idx >= o.sys.surfaces.length - 1) return { value: null }
      const sd = semiDiameters(o)
      const r = typeof k.radius === 'number' ? k.radius : Math.min(sd[idx], sd[idx + 1])
      const a = sag(o.sys.surfaces[idx].shape, r * r)
      const b = sag(o.sys.surfaces[idx + 1].shape, r * r)
      if (!a || !b) return { value: null }
      return { value: num(k.value, 0) - b.z + a.z }
    }
    case 'f_number':
      return { value: cur }
    default:
      return { value: null, note: `unknown solve kind ${String(k.type)}` }
  }
}

function parseSysThickness(system: Obj, i: number): number {
  const t = (system.surfaces as Obj[])[i]?.thickness
  return typeof t === 'number' ? t : 0
}

/** measured − goal of the solve's condition in the current system. */
function residualOf(system: Obj, sv: SolveDef): number {
  const o = optics(system)
  const idx = typeof sv.target.surface === 'number' ? surfaceIndex(system, sv.target.surface) : -1
  const k = sv.kind
  switch (k.type) {
    case 'pickup': {
      const src = asPath(k.source)
      const v = src ? getParamValue(system, src) : null
      const t = getParamValue(system, sv.target)
      return v === null || t === null ? NaN : t - (num(k.scale, 1) * v + num(k.offset, 0))
    }
    case 'marginal_ray_height':
      return marginal(o).y[Math.min(idx + 1, o.sys.surfaces.length - 1)] - num(k.height, 0)
    case 'chief_ray_height':
      return chief(o).y[Math.min(idx + 1, o.sys.surfaces.length - 1)] - num(k.height, 0)
    case 'marginal_ray_angle':
      return marginal(o).nu[idx] / nPrime(o, idx) - num(k.slope, 0)
    case 'chief_ray_angle':
      return chief(o).nu[idx] / nPrime(o, idx) - num(k.slope, 0)
    case 'f_number': {
      const efl = o.efl(o.primary)
      return efl === null ? NaN : Math.abs(efl) / (2 * o.epRadius) - num(k.value, 0)
    }
    case 'position': {
      const from = surfaceIndex(system, num(k.from, -1))
      const to = surfaceIndex(system, num(k.to, -1))
      return Math.abs(o.vertexZ[to] - o.vertexZ[from]) - num(k.distance, 0)
    }
    case 'center_of_curvature': {
      const sIdx = surfaceIndex(system, num(k.surface, -1))
      const c = getParamValue(system, sv.target) ?? 0
      return c * (o.vertexZ[sIdx] - o.vertexZ[idx]) - 1
    }
    case 'edge_thickness': {
      const sd = semiDiameters(o)
      const r = typeof k.radius === 'number' ? k.radius : Math.min(sd[idx], sd[idx + 1])
      const a = sag(o.sys.surfaces[idx].shape, r * r)
      const b = sag(o.sys.surfaces[idx + 1].shape, r * r)
      return a && b ? o.sys.surfaces[idx].thickness + b.z - a.z - num(k.value, 0) : NaN
    }
    default:
      return NaN
  }
}

export function solvesOf(system: Obj): SolveDef[] {
  const list: Obj[] = Array.isArray(system.solves) ? system.solves : []
  const out: SolveDef[] = []
  for (const s of list) {
    const target = asPath(s.target)
    if (target && s.kind && typeof s.kind === 'object') out.push({ target, kind: s.kind as Obj })
  }
  return out
}

/** Evaluates the solves without changing the input; returns the report and the solved copy. */
export function evaluateSolves(system: Obj, s: Obj): { report: JsonObject; solved: Obj } {
  const work = clone(system)
  const solves = solvesOf(work)
  const maxSweeps = Math.max(1, Math.floor(num(s.max_sweeps, 50)))
  const tol = num(s.tolerance, 1e-12)
  const previous = solves.map((sv) => getParamValue(work, sv.target))
  const methods = solves.map(() => 'unchanged')
  const evals = solves.map(() => 0)
  const warnings: string[] = []
  const unsolved = new Set<number>()
  let sweeps = 0
  let maxChange = 0
  for (; sweeps < maxSweeps; ) {
    sweeps++
    maxChange = 0
    solves.forEach((sv, i) => {
      if (unsolved.has(i)) return
      const before = getParamValue(work, sv.target)
      let value: number | null
      if (sv.kind.type === 'f_number') {
        value = solveFNumber(work, sv)
        methods[i] = 'root_finding'
      } else {
        const r = closedForm(work, sv)
        value = r.value
        if (value === null) {
          unsolved.add(i)
          warnings.push(`solve ${i + 1} (${String(sv.kind.type)}): ${r.note ?? 'cannot be evaluated by the mock engine'}`)
          return
        }
        methods[i] = 'closed_form'
      }
      evals[i]++
      if (value === null || !Number.isFinite(value)) return
      setParamValue(work, sv.target, value)
      if (before !== null) maxChange = Math.max(maxChange, Math.abs(value - before) / Math.max(Math.abs(before), 1e-9))
    })
    if (maxChange <= tol || solves.length === 0) break
  }
  const outcomes = solves.map((sv, i) => {
    const value = getParamValue(work, sv.target)
    return {
      index: i,
      kind: String(sv.kind.type),
      target: sv.target as JsonObject,
      previous: nf(previous[i]),
      value: nf(value),
      residual: nf(unsolved.has(i) ? NaN : residualOf(work, sv)),
      method: previous[i] === value ? 'unchanged' : methods[i],
      evaluations: evals[i],
    }
  })
  return {
    report: {
      outcomes,
      order: solves.map((_, i) => i),
      sweeps,
      max_change: nf(maxChange),
      warnings,
      solved_params: solves.map((sv) => sv.target as JsonObject),
      configuration: (s.configuration ?? null) as Json,
    },
    solved: work,
  }
}

/** F/# solve: secant search on the target (curvature or focal length). */
function solveFNumber(work: Obj, sv: SolveDef): number | null {
  const goal = num(sv.kind.value, 0)
  const x0 = getParamValue(work, sv.target)
  if (x0 === null) return null
  const f = (x: number): number => {
    const w = clone(work)
    setParamValue(w, sv.target, x)
    const o = optics(w)
    const efl = o.efl(o.primary)
    return efl === null ? NaN : Math.abs(efl) / (2 * o.epRadius) - goal
  }
  let a = x0
  let b = x0 * 1.01 + (x0 === 0 ? 1e-4 : 0)
  let fa = f(a)
  let fb = f(b)
  for (let it = 0; it < 40 && Number.isFinite(fa) && Number.isFinite(fb) && Math.abs(fb) > 1e-12; it++) {
    if (fb === fa) break
    const c = b - (fb * (b - a)) / (fb - fa)
    a = b
    fa = fb
    b = c
    fb = f(b)
  }
  return Number.isFinite(fb) ? b : null
}

export function solves(system: Obj, s: Obj): Out {
  return newOut(evaluateSolves(system, s).report)
}

// --- convert_spec ------------------------------------------------------------------

const APERTURES = ['entrance_pupil_diameter', 'image_space_f_number', 'object_space_na', 'paraxial_working_f_number', 'float_by_stop_size']
const FIELD_KINDS = ['object_angle', 'object_height', 'paraxial_image_height', 'real_image_height']

export function convertSpec(system: Obj, s: Obj): Out {
  const o = optics(system)
  const w = o.primary
  const m = marginal(o)
  const stop = Math.min(Math.max(o.sys.stop, 1), o.sys.surfaces.length - 2)
  const out: JsonObject = { aperture: null, aperture_error: null, fields: null, fields_error: null, stop_semi_diameter: Math.abs(m.y[stop]) }

  const ap = typeof s.aperture === 'string' ? s.aperture : null
  if (ap !== null) {
    const d = o.objectDistance
    const efl = o.efl(w)
    if (!APERTURES.includes(ap)) out.aperture_error = `unknown aperture type '${ap}'`
    else if (ap === 'entrance_pupil_diameter') out.aperture = { type: ap, value: 2 * o.epRadius }
    else if (ap === 'image_space_f_number') {
      if (d !== null) out.aperture_error = 'image-space F/# needs an object at infinity'
      else if (efl === null) out.aperture_error = 'the system is afocal'
      else out.aperture = { type: ap, value: Math.abs(efl) / (2 * o.epRadius) }
    } else if (ap === 'object_space_na') {
      if (d === null) out.aperture_error = 'object-space NA needs a finite object distance'
      else {
        const u = o.epRadius / Math.abs(o.objectToPupil(d))
        out.aperture = { type: ap, value: o.nAfter[0][w] * Math.sin(Math.atan(u)) }
      }
    } else if (ap === 'paraxial_working_f_number') out.aperture = { type: ap, value: o.workingFNumber(w) }
    else out.aperture = { type: ap, value: null }
  }

  const kind = typeof s.field_kind === 'string' ? s.field_kind : null
  if (kind !== null) {
    if (!FIELD_KINDS.includes(kind)) out.fields_error = `unknown field kind '${kind}'`
    else if (kind === 'object_height' && o.objectDistance === null) out.fields_error = 'object heights need a finite object distance'
    else {
      const values: Json[] = []
      let err: string | null = null
      o.sys.fields.forEach((f, i) => {
        if (err) return
        const fp = o.fieldPointXY(f.x, f.y)
        const d = o.objectDistance
        const q = -(o.epZ + (d ?? 0))
        if (kind === 'object_angle') {
          const [tx, ty] = fp.kind === 'angle' ? [fp.tx, fp.ty] : Math.abs(q) > 1e-12 ? [fp.x / q, fp.y / q] : [0, 0]
          values.push([(Math.atan(tx) * 180) / Math.PI, (Math.atan(ty) * 180) / Math.PI])
        } else if (kind === 'object_height') {
          if (fp.kind === 'height') values.push([fp.x, fp.y])
          else values.push([0, 0])
        } else if (kind === 'paraxial_image_height') {
          const [px, py] = fp.kind === 'angle' ? [fp.tx, fp.ty] : [fp.x, fp.y]
          values.push([o.chiefImagePerUnit * px, o.chiefImagePerUnit * py])
        } else {
          const hit = o.trace(o.makeRay(fp, 0, 0), w).atImage
          if (!hit) err = `field ${i + 1}: the chief ray cannot be traced`
          else values.push([hit.p[0], hit.p[1]])
        }
      })
      if (err) out.fields_error = err
      else out.fields = { kind, values }
    }
  }
  return newOut(out)
}

export { surfaceOf }
