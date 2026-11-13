// The tolerance set of a design (docs/APP.md "Design jobs" → Tolerance,
// apps/macos/Sources/MKLightUI/ViewModels/ToleranceModel.swift, ToleranceReport.swift
// and the result half of ToleranceSession.swift): rows in the units a lens
// designer thinks in (fringes of power and irregularity, ΔR, arc minutes, Δn,
// % of V_d), expanded to the engine's `ToleranceOperand`s on parameter paths;
// compensators; the criterion; the run's mode, trials, seed, levels and
// recorded trials. Saved in the project's `tolerances` section as the engine's
// `ToleranceSpec` plus the app's rows under `app`. Pure model code: no optics.

import type { Json, JsonObject } from '@/engine/types'
import { nf, sig } from './format'
import { getParam, paramKey, paramTitle, paramUnit, surfaceIndexOfId, type ParamPath } from './params'
import { imageIndex, type Surface, type System } from './system'

// MARK: Kinds and units

export type ToleranceKind =
  | 'radius'
  | 'irregularity_astigmatism'
  | 'irregularity_spherical'
  | 'thickness'
  | 'decenter_x'
  | 'decenter_y'
  | 'tilt_x'
  | 'tilt_y'
  | 'element_decenter_x'
  | 'element_decenter_y'
  | 'element_tilt_x'
  | 'element_tilt_y'
  | 'index'
  | 'abbe'
  | 'other'

export type ToleranceUnit = 'fringes' | 'radius_mm' | 'percent' | 'mm' | 'arcmin' | 'degrees' | 'index' | 'abbe' | 'native'

export type PlacementComponent = 'decenter_x' | 'decenter_y' | 'tilt_x' | 'tilt_y'

interface KindInfo {
  title: string
  short: string
  units: ToleranceUnit[]
  component?: PlacementComponent
}

export const KINDS: Record<ToleranceKind, KindInfo> = {
  radius: { title: 'Radius', short: 'Radius', units: ['fringes', 'radius_mm', 'percent'] },
  irregularity_astigmatism: { title: 'Astig. irregularity', short: 'Irreg. astig.', units: ['fringes'] },
  irregularity_spherical: { title: 'Sph. irregularity', short: 'Irreg. sph.', units: ['fringes'] },
  thickness: { title: 'Thickness', short: 'Thickness', units: ['mm'] },
  decenter_x: { title: 'Surface decenter x', short: 'Decenter x', units: ['mm'] },
  decenter_y: { title: 'Surface decenter y', short: 'Decenter y', units: ['mm'] },
  tilt_x: { title: 'Surface tilt x', short: 'Tilt x', units: ['arcmin', 'degrees'] },
  tilt_y: { title: 'Surface tilt y', short: 'Tilt y', units: ['arcmin', 'degrees'] },
  element_decenter_x: { title: 'Element decenter x', short: 'Decenter x', units: ['mm'], component: 'decenter_x' },
  element_decenter_y: { title: 'Element decenter y', short: 'Decenter y', units: ['mm'], component: 'decenter_y' },
  element_tilt_x: { title: 'Element tilt x', short: 'Tilt x', units: ['arcmin', 'degrees'], component: 'tilt_x' },
  element_tilt_y: { title: 'Element tilt y', short: 'Tilt y', units: ['arcmin', 'degrees'], component: 'tilt_y' },
  index: { title: 'Index', short: 'Index', units: ['index'] },
  abbe: { title: 'Abbe', short: 'Abbe', units: ['percent', 'abbe'] },
  other: { title: 'Parameter', short: 'Parameter', units: ['native'] },
}

const ALL_KINDS = Object.keys(KINDS) as ToleranceKind[]

export const isElementKind = (k: ToleranceKind) => KINDS[k].component != null
const isTiltKind = (k: ToleranceKind) => k === 'tilt_x' || k === 'tilt_y' || k === 'element_tilt_x' || k === 'element_tilt_y'

/** Kinds offered when adding a row on a surface. */
export const ADDABLE_SURFACE_KINDS: ToleranceKind[] = [
  'radius', 'irregularity_astigmatism', 'irregularity_spherical', 'thickness', 'decenter_x', 'decenter_y', 'tilt_x', 'tilt_y', 'index', 'abbe',
]
/** Kinds offered when adding a row on an element. */
export const ADDABLE_ELEMENT_KINDS: ToleranceKind[] = ['element_decenter_x', 'element_decenter_y', 'element_tilt_x', 'element_tilt_y']

export const UNITS: Record<ToleranceUnit, { label: string; title: string }> = {
  fringes: { label: 'fr', title: 'Test fringes' },
  radius_mm: { label: 'mm', title: 'ΔR (mm)' },
  percent: { label: '%', title: 'Percent' },
  mm: { label: 'mm', title: 'mm' },
  arcmin: { label: '′', title: 'Arc minutes' },
  degrees: { label: '°', title: 'Degrees' },
  index: { label: 'Δn', title: 'Δn' },
  abbe: { label: 'ΔV', title: 'ΔV_d' },
  native: { label: '', title: 'Parameter units' },
}

// MARK: Distributions

export type Distribution = { type: 'uniform' } | { type: 'truncated_normal'; sigmas: number } | { type: 'end_point' }

export const DISTRIBUTIONS: Distribution[] = [
  { type: 'uniform' },
  { type: 'truncated_normal', sigmas: 2 },
  { type: 'truncated_normal', sigmas: 3 },
  { type: 'end_point' },
]

export function readDistribution(j: Json | undefined): Distribution {
  const o = obj(j)
  if (o?.type === 'truncated_normal') {
    const k = nf(o.sigmas)
    return { type: 'truncated_normal', sigmas: Number.isFinite(k) ? k : 2 }
  }
  if (o?.type === 'end_point') return { type: 'end_point' }
  return { type: 'uniform' }
}

export function distributionJson(d: Distribution): JsonObject {
  return d.type === 'truncated_normal' ? { type: d.type, sigmas: d.sigmas } : { type: d.type }
}

export function distributionTitle(d: Distribution): string {
  if (d.type === 'truncated_normal') return `Normal (limits at ±${short(d.sigmas)} σ)`
  return d.type === 'end_point' ? 'End points' : 'Uniform'
}

export const sameDistribution = (a: Distribution, b: Distribution) => JSON.stringify(distributionJson(a)) === JSON.stringify(distributionJson(b))

// MARK: Small helpers

function obj(j: Json | undefined): JsonObject | null {
  return j && typeof j === 'object' && !Array.isArray(j) ? (j as JsonObject) : null
}
const arr = (j: Json | undefined): Json[] => (Array.isArray(j) ? j : [])
const nums = (j: Json | undefined): number[] => arr(j).map(nf)
const int = (j: Json | undefined): number | null => (typeof j === 'number' && Number.isFinite(j) ? Math.round(j) : null)

/** Short number text: 6 significant digits, a real minus sign. */
export const short = (v: number) => sig(v, 6)

/** Rounds off conversion noise (9 significant digits). */
export function tidy(v: number, digits = 9): number {
  if (v === 0 || !Number.isFinite(v)) return v
  return Number(v.toPrecision(digits))
}

let seq = 0
export const newId = (prefix = 't') => `${prefix}${++seq}`

const surfaceAt = (s: System, id: number): Surface | null => {
  const i = surfaceIndexOfId(s, id)
  return i >= 0 ? s.surfaces[i] : null
}
const isAir = (m: JsonObject) => m.type === 'air'
export const isSolid = (s: System, i: number) => {
  const m = s.surfaces[i].material
  return !isAir(m) && m.type !== 'vacuum'
}
export const hasGlass = (m: JsonObject) => m.type === 'model' || m.type === 'catalog'
const isCatalog = (m: JsonObject) => m.type === 'catalog'
const isCoordinateBreak = (x: Surface) => x.shape.type === 'coordinate_break'
export const materialName = (m: JsonObject): string => {
  switch (m.type) {
    case 'catalog': return String(m.name ?? '')
    case 'model': return `${short(nf(m.nd))}/${short(nf(m.vd))}`
    case 'constant': return `n=${short(nf(m.n))}`
    default: return String(m.type)
  }
}
const curvature = (s: System, id: number) => getParam(s, { param: 'curvature', surface: id })

/** The engine's bounds of a catalogue glass's melt offset. */
export const MAX_DN_D = 0.05
export const MAX_DV_D = 10

// MARK: Context

/** What a row needs to become an engine operand: the system, the clear
 *  semi-diameters by surface id (fringes of power are counted over them), the
 *  test wavelength and the V_d of catalogue glasses by surface id. */
export interface ToleranceContext {
  system: System
  semiDiameters: Record<number, number>
  testWavelengthUm: number
  abbeNumbers: Record<number, number>
}

export function makeContext(system: System, opts: Partial<Omit<ToleranceContext, 'system'>> = {}): ToleranceContext {
  return { system, semiDiameters: opts.semiDiameters ?? {}, testWavelengthUm: opts.testWavelengthUm ?? DEFAULT_TEST_WAVELENGTH_UM, abbeNumbers: opts.abbeNumbers ?? {} }
}

export function semiDiameterOf(c: ToleranceContext, id: number): number | null {
  const sf = surfaceAt(c.system, id)
  if (sf?.semi_diameter.mode === 'fixed' && typeof sf.semi_diameter.value === 'number') return sf.semi_diameter.value
  const v = c.semiDiameters[id]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

export function abbeNumberOf(c: ToleranceContext, id: number): number | null {
  const sf = surfaceAt(c.system, id)
  if (!sf) return null
  if (sf.material.type === 'model') return nf(sf.material.vd)
  const v = c.abbeNumbers[id]
  return typeof v === 'number' ? v : null
}

// MARK: Rows

/** One tolerance: a kind on a surface (or an element's surface range, or any
 *  parameter), its − and + limits in the row's unit, and the distribution. */
export interface ToleranceRow {
  /** Session identity (not saved). */
  id: string
  kind: ToleranceKind
  /** The surface id, or an element's first surface id. */
  surface: number
  /** An element's last surface id (inclusive; null: `surface` alone). */
  last: number | null
  /** The vertex an element tilts about (null: its first surface). */
  pivot: number | null
  /** The parameter of an `other` row. */
  param: ParamPath | null
  minus: number
  plus: number
  unit: ToleranceUnit
  distribution: Distribution
}

export function makeRow(
  kind: ToleranceKind,
  surface: number,
  minus: number,
  opts: { plus?: number; unit?: ToleranceUnit; param?: ParamPath | null; last?: number | null; pivot?: number | null; distribution?: Distribution } = {},
): ToleranceRow {
  const el = isElementKind(kind)
  return {
    id: newId(),
    kind,
    surface,
    last: el ? (opts.last ?? null) : null,
    pivot: el ? (opts.pivot ?? null) : null,
    param: opts.param ?? null,
    minus,
    plus: opts.plus ?? minus,
    unit: opts.unit ?? KINDS[kind].units[0],
    distribution: opts.distribution ?? { type: 'uniform' },
  }
}

/** The engine parameter a row perturbs. Index and Abbe address a model
 *  glass's n_d and V_d, a catalogue glass's melt offset. */
export function rowPath(r: ToleranceRow, s: System): ParamPath {
  const surface = r.surface
  const catalogue = (() => {
    const sf = surfaceAt(s, surface)
    return sf ? isCatalog(sf.material) : false
  })()
  switch (r.kind) {
    case 'radius': return { param: 'curvature', surface }
    case 'thickness': return { param: 'thickness', surface }
    case 'decenter_x': case 'decenter_y': case 'tilt_x': case 'tilt_y': return { param: r.kind, surface }
    case 'element_decenter_x': case 'element_decenter_y': case 'element_tilt_x': case 'element_tilt_y': {
      const p: ParamPath = { param: 'element_placement', surface, last: r.last ?? surface, component: KINDS[r.kind].component! }
      if (r.pivot != null) p.pivot = r.pivot
      return p
    }
    case 'irregularity_astigmatism': return { param: 'irregularity', surface, mode: 'astigmatism' }
    case 'irregularity_spherical': return { param: 'irregularity', surface, mode: 'spherical' }
    case 'index': return { param: catalogue ? 'material_dn_d' : 'model_nd', surface }
    case 'abbe': return { param: catalogue ? 'material_dv_d' : 'model_vd', surface }
    case 'other': return r.param ?? { param: 'curvature', surface }
  }
}

/** Surface indices that move together (an element's range, inclusive). */
export function movingSurfaces(r: ToleranceRow, s: System): number[] {
  const i = surfaceIndexOfId(s, r.surface)
  if (i < 0) return []
  const l = r.last != null ? surfaceIndexOfId(s, r.last) : -1
  if (!isElementKind(r.kind) || l < i) return [i]
  return Array.from({ length: l - i + 1 }, (_, k) => i + k)
}

export function pivotIndex(r: ToleranceRow, s: System): number | null {
  const i = surfaceIndexOfId(s, r.pivot ?? r.surface)
  return i >= 0 ? i : null
}

/** `S3`, `S3–4` (a thickness between two surfaces, or an element). */
export function surfaceLabel(r: ToleranceRow, s: System): string {
  const i = surfaceIndexOfId(s, r.surface)
  if (i < 0) return 'S?'
  if (r.kind === 'thickness') return `S${i}–${i + 1}`
  if (isElementKind(r.kind)) {
    const l = r.last != null ? surfaceIndexOfId(s, r.last) : i
    return l === i || l < 0 ? `S${i}` : `S${i}–${l}`
  }
  return `S${i}`
}

/** The element (`L2`) whose surfaces this row moves, if it is one. */
export function elementTitle(r: ToleranceRow, s: System): string | null {
  if (!isElementKind(r.kind)) return null
  const m = movingSurfaces(r, s)
  const f = m[0], l = m[m.length - 1]
  if (f == null) return null
  return lensGroups(s).find((g) => g.surfaces[0] === f && g.surfaces[g.surfaces.length - 1] === l)?.title ?? null
}

/** `Radius S3`, `Thickness S3–4`, `Element tilt x L2`, `Index S5`. */
export function rowLabel(r: ToleranceRow, s: System): string {
  if (r.kind === 'other') return paramTitle(s, rowPath(r, s))
  if (isElementKind(r.kind)) {
    const e = elementTitle(r, s)
    if (e) return `${KINDS[r.kind].title} ${e}`
  }
  return `${KINDS[r.kind].title} ${surfaceLabel(r, s)}`
}

const unitSuffix = (u: ToleranceUnit) => (UNITS[u].label === '' ? '' : u === 'arcmin' ? UNITS[u].label : ' ' + UNITS[u].label)

/** `±3 fr`, `−0.02 / +0.05 mm`. */
export function limitsText(r: ToleranceRow): string {
  if (r.minus === r.plus) return '±' + short(r.minus) + unitSuffix(r.unit)
  return '−' + short(r.minus) + ' / +' + short(r.plus) + unitSuffix(r.unit)
}

/** `+0.8′`, `−0.02 mm`. */
export function limitText(r: ToleranceRow, plus: boolean): string {
  return (plus ? '+' : '−') + short(plus ? r.plus : r.minus) + unitSuffix(r.unit)
}

/** The unit text a row shows (a parameter row: the parameter's unit). */
export function unitLabel(r: ToleranceRow, s: System): string {
  return UNITS[r.unit].label || paramUnit(rowPath(r, s))
}

export type Limits = { ok: true; minus: number; plus: number } | { ok: false; issue: string }

const fail = (issue: string): Limits => ({ ok: false, issue })

/** Why an element row's range or pivot cannot be moved, if it cannot. */
function elementProblem(r: ToleranceRow, s: System, i: number): string | null {
  const l = surfaceIndexOfId(s, r.last ?? r.surface)
  if (l < 0) return "The element's last surface no longer exists."
  if (!(i > 0 && l >= i && l < imageIndex(s))) return 'An element moves surfaces between the object and the image, first to last.'
  if (r.pivot != null && surfaceIndexOfId(s, r.pivot) < 0) return 'The pivot surface no longer exists.'
  const e = obj(s.surfaces[i].element_placement)
  if (e) {
    const eLast = int(e.last)
    const ePivot = int(e.pivot)
    if (eLast !== (r.last ?? r.surface) || ePivot !== r.pivot) {
      const el = eLast != null && surfaceIndexOfId(s, eLast) >= 0 ? String(surfaceIndexOfId(s, eLast)) : '?'
      const pi = surfaceIndexOfId(s, ePivot ?? r.surface)
      return `The design already moves S${i}–${el} as an element about S${pi >= 0 ? pi : '?'}: use that range and pivot.`
    }
  }
  return null
}

/** The engine's limits (minus, plus) in parameter units, or why the row
 *  cannot be expanded. */
export function engineLimits(r: ToleranceRow, c: ToleranceContext): Limits {
  const { minus, plus } = r
  if (!(Number.isFinite(minus) && Number.isFinite(plus) && minus >= 0 && plus >= 0)) return fail('Limits must be zero or positive numbers.')
  if (!(minus > 0 || plus > 0)) return fail('Both limits are zero.')
  const s = c.system
  const i = surfaceIndexOfId(s, r.surface)
  if (i < 0) return fail('The surface no longer exists.')
  const sf = s.surfaces[i]
  switch (r.kind) {
    case 'radius': {
      const curv = curvature(s, r.surface)
      if (curv == null) return fail(`Surface ${i} has no radius.`)
      if (r.unit === 'fringes') {
        const h = semiDiameterOf(c, r.surface)
        if (!(h != null && h > 0)) return fail(`Fringes need the clear aperture of surface ${i}; run the layout or set a semi-diameter.`)
        // N fringes of power: sag change N·λ/2 at the rim, Δc = N·λ/h².
        const k = (c.testWavelengthUm * 1e-3) / (h * h)
        return { ok: true, minus: minus * k, plus: plus * k }
      }
      if (r.unit === 'radius_mm' || r.unit === 'percent') {
        if (curv === 0) return fail(`Surface ${i} is flat: give its power in fringes.`)
        const R = 1 / curv
        const dm = r.unit === 'percent' ? (Math.abs(R) * minus) / 100 : minus
        const dp = r.unit === 'percent' ? (Math.abs(R) * plus) / 100 : plus
        // R in [R − ΔR−, R + ΔR+]: c in [1/(R + ΔR+), 1/(R − ΔR−)].
        const lo = R + dp, hi = R - dm
        if (lo === 0 || hi === 0 || lo > 0 !== R > 0 || hi > 0 !== R > 0) return fail(`ΔR must stay below |R| = ${short(Math.abs(R))} mm.`)
        return { ok: true, minus: curv - 1 / lo, plus: 1 / hi - curv }
      }
      return fail('Radius tolerances are in fringes, mm or %.')
    }
    case 'thickness':
      if (i >= imageIndex(s)) return fail('The image surface has no thickness.')
      return { ok: true, minus, plus }
    case 'decenter_x':
    case 'decenter_y':
      if (!(i > 0 && i < imageIndex(s))) return fail('Decenter a surface between the object and the image.')
      return { ok: true, minus, plus }
    case 'tilt_x':
    case 'tilt_y':
    case 'element_tilt_x':
    case 'element_tilt_y':
    case 'element_decenter_x':
    case 'element_decenter_y': {
      if (isElementKind(r.kind)) {
        const why = elementProblem(r, s, i)
        if (why) return fail(why)
      } else if (i === 0 || i >= imageIndex(s)) return fail('Tilt a surface between the object and the image.')
      const k = isTiltKind(r.kind) && r.unit === 'arcmin' ? 1 / 60 : 1
      return { ok: true, minus: minus * k, plus: plus * k }
    }
    case 'irregularity_astigmatism':
    case 'irregularity_spherical':
      if (!(i > 0 && i < imageIndex(s)) || isCoordinateBreak(sf) || sf.shape.type === 'paraxial') return fail(`Surface ${i} has no form to make irregular.`)
      return { ok: true, minus, plus }
    case 'index':
    case 'abbe': {
      if (!hasGlass(sf.material)) {
        const name = isAir(sf.material) ? 'air' : materialName(sf.material)
        return fail(`Surface ${i} is followed by ${name}, not a glass with n_d and V_d.`)
      }
      let lim: [number, number] = [minus, plus]
      if (r.kind === 'abbe' && r.unit === 'percent') {
        const vd = abbeNumberOf(c, r.surface)
        if (!(vd != null && vd > 0)) {
          return fail(`The Abbe number of ${materialName(sf.material)} is not known yet: give the limit as ΔV_d, or wait for the glass data.`)
        }
        lim = [(vd * minus) / 100, (vd * plus) / 100]
      }
      if (isCatalog(sf.material)) {
        // A tolerance adds to the saved melt offset, which stays within the engine's bounds.
        const o = obj(sf.material.offset) ?? {}
        const [base, cap, what] = r.kind === 'index' ? [nf(o.dn_d ?? 0), MAX_DN_D, 'Δn_d'] : [nf(o.dv_d ?? 0), MAX_DV_D, 'ΔV_d']
        if (Math.abs(base - lim[0]) > cap + 1e-12 || Math.abs(base + lim[1]) > cap + 1e-12) {
          return fail(`With the melt offset (${short(base)}), ${what} would pass ±${short(cap)}, the engine's limit.`)
        }
      }
      return { ok: true, minus: lim[0], plus: lim[1] }
    }
    case 'other': {
      if (!r.param || getParam(s, r.param) == null) return fail('The parameter no longer exists.')
      return { ok: true, minus, plus }
    }
  }
}

/** The engine's `ToleranceOperand`, or why the row cannot run. */
export function rowOperand(r: ToleranceRow, c: ToleranceContext): { ok: true; operand: JsonObject } | { ok: false; issue: string } {
  const l = engineLimits(r, c)
  if (!l.ok) return l
  return {
    ok: true,
    operand: { param: rowPath(r, c.system), minus: l.minus, plus: l.plus, distribution: distributionJson(r.distribution), label: rowLabel(r, c.system) },
  }
}

/** The row with its limits written in `unit` (the same engine limits):
 *  fringes ↔ ΔR ↔ % of R for radii, arc minutes ↔ degrees, % of V_d ↔ ΔV_d.
 *  null when the conversion needs what the design lacks. */
export function convertRow(r: ToleranceRow, unit: ToleranceUnit, c: ToleranceContext): ToleranceRow | null {
  if (unit === r.unit || !KINDS[r.kind].units.includes(unit)) return r
  const out: ToleranceRow = { ...r, unit }
  switch (r.kind) {
    case 'tilt_x':
    case 'tilt_y':
    case 'element_tilt_x':
    case 'element_tilt_y': {
      const k = unit === 'arcmin' ? 60 : 1 / 60
      return { ...out, minus: tidy(r.minus * k), plus: tidy(r.plus * k) }
    }
    case 'abbe': {
      const vd = abbeNumberOf(c, r.surface)
      if (!(vd != null && vd > 0)) return null
      const k = unit === 'percent' ? 100 / vd : vd / 100
      return { ...out, minus: tidy(r.minus * k), plus: tidy(r.plus * k) }
    }
    case 'radius': {
      const l = engineLimits(r, c)
      const curv = curvature(c.system, r.surface)
      if (!l.ok || curv == null) return null
      const { minus: m, plus: p } = l
      if (unit === 'fringes') {
        const h = semiDiameterOf(c, r.surface)
        if (!(h != null && h > 0)) return null
        const k = (h * h) / (c.testWavelengthUm * 1e-3)
        return { ...out, minus: tidy(m * k), plus: tidy(p * k) }
      }
      if (unit === 'radius_mm' || unit === 'percent') {
        if (curv === 0 || curv + p === 0 || curv - m === 0) return null
        const R = 1 / curv
        // c in [c − m, c + p]: R in [1/(c + p), 1/(c − m)] (same sign as R).
        const r1 = 1 / (curv + p), r2 = 1 / (curv - m)
        if (r1 > 0 !== R > 0 || r2 > 0 !== R > 0) return null
        const k = unit === 'percent' ? 100 / Math.abs(R) : 1
        return { ...out, minus: tidy(Math.max(R - r1, 0) * k), plus: tidy(Math.max(r2 - R, 0) * k) }
      }
      return null
    }
    default:
      return out
  }
}

/** The saved form of a row (no session id). */
export function rowJson(r: ToleranceRow): JsonObject {
  const o: JsonObject = { kind: r.kind, surface: r.surface, minus: r.minus, plus: r.plus, unit: r.unit, distribution: distributionJson(r.distribution) }
  if (r.param) o.param = r.param
  if (r.last != null) o.last = r.last
  if (r.pivot != null) o.pivot = r.pivot
  return o
}

export function readRow(j: Json): ToleranceRow | null {
  const o = obj(j)
  if (!o) return null
  const kind = o.kind as ToleranceKind
  const surface = int(o.surface)
  if (!ALL_KINDS.includes(kind) || surface == null || surface < 0) return null
  const units = KINDS[kind].units
  const unit = units.includes(o.unit as ToleranceUnit) ? (o.unit as ToleranceUnit) : units[0]
  const param = obj(o.param) && typeof obj(o.param)!.param === 'string' ? (o.param as ParamPath) : null
  if (kind === 'other' && !param) return null
  const el = isElementKind(kind)
  const last = el ? int(o.last) : null
  const pivot = el ? int(o.pivot) : null
  return {
    id: newId(),
    kind,
    surface,
    last: last != null && last >= 0 ? last : null,
    pivot: pivot != null && pivot >= 0 ? pivot : null,
    param,
    minus: Number.isFinite(nf(o.minus)) ? nf(o.minus) : 0,
    plus: Number.isFinite(nf(o.plus)) ? nf(o.plus) : 0,
    unit,
    distribution: readDistribution(o.distribution),
  }
}

/** A row from an engine operand written elsewhere (Python, CLI): kinds the app
 *  knows in their natural unit (curvature as ΔR), others as parameters. */
export function rowFromOperand(j: Json, s: System): ToleranceRow | null {
  const o = obj(j)
  const p = obj(o?.param) as ParamPath | null
  if (!o || !p || typeof p.param !== 'string' || typeof p.surface !== 'number') return null
  const id = p.surface
  const m = nf(o.minus ?? 0), pl = nf(o.plus ?? 0)
  const distribution = readDistribution(o.distribution)
  const row = (k: ToleranceKind, minus: number, plus: number, extra: Parameters<typeof makeRow>[3] = {}) => makeRow(k, id, minus, { plus, distribution, ...extra })
  switch (p.param) {
    case 'curvature': {
      const c = getParam(s, p)
      if (c == null || c === 0) return row('other', m, pl, { param: p })
      const R = 1 / c
      return row('radius', Math.abs(R - 1 / (c + pl)), Math.abs(1 / (c - m) - R), { unit: 'radius_mm' })
    }
    case 'thickness': case 'decenter_x': case 'decenter_y':
      return row(p.param as ToleranceKind, m, pl)
    case 'tilt_x': case 'tilt_y':
      return row(p.param as ToleranceKind, m * 60, pl * 60)
    case 'model_nd': case 'material_dn_d':
      return row('index', m, pl)
    case 'model_vd': case 'material_dv_d':
      return row('abbe', m, pl, { unit: 'abbe' })
    case 'irregularity':
      return row(p.mode === 'astigmatism' ? 'irregularity_astigmatism' : 'irregularity_spherical', m, pl)
    case 'element_placement': {
      const k = ADDABLE_ELEMENT_KINDS.find((x) => KINDS[x].component === p.component)
      if (!k) return row('other', m, pl, { param: p })
      const t = isTiltKind(k)
      return row(k, t ? m * 60 : m, t ? pl * 60 : pl, { last: int(p.last), pivot: int(p.pivot) })
    }
    default:
      return row('other', m, pl, { param: p })
  }
}

export const rowKey = (r: ToleranceRow) => JSON.stringify(rowJson(r))

// MARK: Elements

/** A lens element: a run of surfaces joined by glass (a cemented group is one). */
export interface LensGroup {
  title: string
  surfaces: number[]
  glasses: string[]
}

/** Surface indices that bound or lie inside a lens element. */
export function lensSurfaces(s: System): number[] {
  const img = imageIndex(s)
  if (img <= 1) return []
  const out: number[] = []
  for (let i = 1; i < img; i++) if (!isCoordinateBreak(s.surfaces[i]) && (isSolid(s, i) || isSolid(s, i - 1))) out.push(i)
  return out
}

export function lensGroups(s: System): LensGroup[] {
  const out: LensGroup[] = []
  let current: number[] = []
  for (const i of lensSurfaces(s)) {
    current.push(i)
    if (!isSolid(s, i)) {
      out.push({ title: `L${out.length + 1}`, surfaces: current, glasses: current.slice(0, -1).map((k) => materialName(s.surfaces[k].material)) })
      current = []
    }
  }
  if (current.length) out.push({ title: `L${out.length + 1}`, surfaces: current, glasses: current.map((k) => materialName(s.surfaces[k].material)) })
  return out
}

/** The group a row belongs to: its element (an air space goes with the element
 *  before it), else the surface itself (`Stop S6`). */
export function groupTitleOf(r: ToleranceRow, s: System, groups: LensGroup[]): string {
  const i = surfaceIndexOfId(s, r.surface)
  if (i < 0) return 'Other'
  const g = groups.find((x) => x.surfaces.includes(i))
  if (g) {
    if (r.kind === 'thickness' && g.surfaces[g.surfaces.length - 1] === i) return `Air after ${g.title}`
    return g.title
  }
  return i === s.stop ? `Stop S${i}` : `S${i}`
}

export interface RowGroup {
  key: string
  title: string
  rows: ToleranceRow[]
}

/** Rows by element (`L1 · N-SK16`, `Air after L1`, `Stop S6`) in surface order. */
export function groupRows(rows: ToleranceRow[], s: System): RowGroup[] {
  const lens = lensGroups(s)
  const order: string[] = []
  const by = new Map<string, ToleranceRow[]>()
  const key = new Map<string, [number, number]>()
  for (const r of rows) {
    const t = groupTitleOf(r, s, lens)
    if (!by.has(t)) {
      order.push(t)
      by.set(t, [])
    }
    by.get(t)!.push(r)
    const i = surfaceIndexOfId(s, r.surface)
    const k: [number, number] = [i < 0 ? Number.MAX_SAFE_INTEGER : i, t.startsWith('Air') ? 1 : 0]
    const old = key.get(t)
    if (!old || k[0] < old[0] || (k[0] === old[0] && k[1] < old[1])) key.set(t, k)
  }
  order.sort((a, b) => {
    const x = key.get(a)!, y = key.get(b)!
    return x[0] - y[0] || x[1] - y[1]
  })
  return order.map((t) => {
    const g = lens.find((x) => x.title === t)
    const glasses = g && g.glasses.length ? ' · ' + g.glasses.join(', ') : ''
    return { key: t, title: t + glasses, rows: by.get(t)! }
  })
}

/** `L2 · S3–5 · N-SK16, F2`. */
export function elementMenuTitle(g: LensGroup): string {
  const f = g.surfaces[0], l = g.surfaces[g.surfaces.length - 1]
  return [g.title, g.surfaces.length > 1 ? `S${f}–${l}` : `S${f}`, ...(g.glasses.length ? [g.glasses.join(', ')] : [])].join(' · ')
}

/** Surface indices a kind can be added on. */
export function surfacesFor(k: ToleranceKind, s: System): number[] {
  const all: number[] = []
  for (let i = 1; i < imageIndex(s); i++) if (!isCoordinateBreak(s.surfaces[i])) all.push(i)
  switch (k) {
    case 'index':
    case 'abbe':
      return all.filter((i) => hasGlass(s.surfaces[i].material))
    case 'radius':
    case 'irregularity_astigmatism':
    case 'irregularity_spherical':
      return all.filter((i) => curvature(s, s.surfaces[i].id) != null)
    default:
      return all
  }
}

// MARK: Default tolerances

export type Preset = 'commercial' | 'precision' | 'high_precision'
export const PRESETS: { value: Preset; title: string }[] = [
  { value: 'commercial', title: 'Commercial' },
  { value: 'precision', title: 'Precision' },
  { value: 'high_precision', title: 'High precision' },
]

/** The "Default tolerances" generator: a class of manufacture and its values
 *  per kind. A value of 0 leaves its kind out. */
export interface Generator {
  preset: Preset | null
  radiusFringes: number
  thicknessMM: number
  decentreMM: number
  tiltArcmin: number
  elementDecentreMM: number
  elementTiltArcmin: number
  index: number
  abbePercent: number
  irregularityFringes: number
  distribution: Distribution
}

export type GeneratorValue = Exclude<keyof Generator, 'preset' | 'distribution'>

const PRESET_VALUES: Record<Preset, Omit<Generator, 'preset' | 'distribution'>> = {
  commercial: { radiusFringes: 5, thicknessMM: 0.2, decentreMM: 0.1, tiltArcmin: 3, index: 0.001, abbePercent: 1, irregularityFringes: 2, elementDecentreMM: 0.1, elementTiltArcmin: 3 },
  precision: { radiusFringes: 3, thicknessMM: 0.05, decentreMM: 0.025, tiltArcmin: 1, index: 0.0005, abbePercent: 0.5, irregularityFringes: 0.5, elementDecentreMM: 0.03, elementTiltArcmin: 1 },
  high_precision: { radiusFringes: 1, thicknessMM: 0.025, decentreMM: 0.01, tiltArcmin: 0.5, index: 0.0002, abbePercent: 0.2, irregularityFringes: 0.25, elementDecentreMM: 0.015, elementTiltArcmin: 0.5 },
}

export function presetGenerator(p: Preset, distribution: Distribution = { type: 'uniform' }): Generator {
  return { preset: p, ...PRESET_VALUES[p], distribution }
}

export const defaultGenerator = () => presetGenerator('precision')

/** The value a newly added row of a kind starts with: the generator's, or the
 *  precision class's where the generator leaves the kind out. */
export function generatorValue(g: Generator, k: ToleranceKind): number {
  const p = PRESET_VALUES.precision
  const or = (key: GeneratorValue) => (g[key] > 0 ? g[key] : p[key])
  switch (k) {
    case 'radius': return or('radiusFringes')
    case 'irregularity_astigmatism': case 'irregularity_spherical': return or('irregularityFringes')
    case 'thickness': return or('thicknessMM')
    case 'decenter_x': case 'decenter_y': return or('decentreMM')
    case 'tilt_x': case 'tilt_y': return or('tiltArcmin')
    case 'element_decenter_x': case 'element_decenter_y': return or('elementDecentreMM')
    case 'element_tilt_x': case 'element_tilt_y': return or('elementTiltArcmin')
    case 'index': return or('index')
    case 'abbe': return or('abbePercent')
    case 'other': return 0
  }
}

/** The tolerances of `s`: power and irregularity of every lens surface, every
 *  gap inside the lens (not the back focus), decentre and tilt of every lens
 *  surface and every element, index and Abbe number of every glass. */
export function generateRows(g: Generator, s: System): ToleranceRow[] {
  const out: ToleranceRow[] = []
  const lens = lensSurfaces(s)
  if (!lens.length) return []
  const first = lens[0], last = lens[lens.length - 1]
  const elements = lensGroups(s)
  const add = (k: ToleranceKind, id: number, v: number, lastId: number | null = null) => {
    if (v > 0) out.push(makeRow(k, id, v, { last: lastId, distribution: g.distribution }))
  }
  for (let i = first; i <= last; i++) {
    const sf = s.surfaces[i]
    if (isCoordinateBreak(sf)) continue
    const id = sf.id
    if (lens.includes(i) && curvature(s, id) != null) {
      add('radius', id, g.radiusFringes)
      if (sf.interaction !== 'dummy') {
        add('irregularity_astigmatism', id, g.irregularityFringes)
        add('irregularity_spherical', id, g.irregularityFringes)
      }
    }
    // Every gap inside the lens, glass or air (the back focus is the compensator).
    if (i < last) add('thickness', id, g.thicknessMM)
    if (lens.includes(i)) {
      add('decenter_x', id, g.decentreMM)
      add('decenter_y', id, g.decentreMM)
      add('tilt_x', id, g.tiltArcmin)
      add('tilt_y', id, g.tiltArcmin)
    }
    if (i < last && hasGlass(sf.material)) {
      add('index', id, g.index)
      add('abbe', id, g.abbePercent)
    }
    // The element's own motion after its last surface's rows.
    const e = elements.find((x) => x.surfaces[x.surfaces.length - 1] === i)
    if (e && e.surfaces[0] < i) {
      const start = s.surfaces[e.surfaces[0]].id
      add('element_decenter_x', start, g.elementDecentreMM, id)
      add('element_decenter_y', start, g.elementDecentreMM, id)
      add('element_tilt_x', start, g.elementTiltArcmin, id)
      add('element_tilt_y', start, g.elementTiltArcmin, id)
    }
  }
  return out
}

export function generatorJson(g: Generator): JsonObject {
  return {
    preset: g.preset,
    radius_fringes: g.radiusFringes,
    thickness_mm: g.thicknessMM,
    decentre_mm: g.decentreMM,
    tilt_arcmin: g.tiltArcmin,
    element_decentre_mm: g.elementDecentreMM,
    element_tilt_arcmin: g.elementTiltArcmin,
    index: g.index,
    abbe_percent: g.abbePercent,
    irregularity_fringes: g.irregularityFringes,
    distribution: distributionJson(g.distribution),
  }
}

export function readGenerator(j: Json | undefined): Generator {
  const g = defaultGenerator()
  const o = obj(j)
  if (!o) return g
  const v = (k: string, d: number) => (Number.isFinite(nf(o[k])) ? nf(o[k]) : d)
  return {
    preset: PRESETS.some((p) => p.value === o.preset) ? (o.preset as Preset) : null,
    radiusFringes: v('radius_fringes', g.radiusFringes),
    thicknessMM: v('thickness_mm', g.thicknessMM),
    decentreMM: v('decentre_mm', g.decentreMM),
    tiltArcmin: v('tilt_arcmin', g.tiltArcmin),
    elementDecentreMM: v('element_decentre_mm', g.elementDecentreMM),
    elementTiltArcmin: v('element_tilt_arcmin', g.elementTiltArcmin),
    index: v('index', g.index),
    abbePercent: v('abbe_percent', g.abbePercent),
    irregularityFringes: v('irregularity_fringes', g.irregularityFringes),
    distribution: readDistribution(o.distribution),
  }
}

// MARK: Compensators

/** A parameter re-optimised for every trial, within ±range of nominal (null: free). */
export interface Compensator {
  id: string
  param: ParamPath
  range: number | null
}

export const makeCompensator = (param: ParamPath, range: number | null = null): Compensator => ({ id: newId('c'), param, range })

export function backFocus(s: System, range = 0.5): Compensator | null {
  const img = imageIndex(s)
  return img >= 1 ? makeCompensator({ param: 'thickness', surface: s.surfaces[img - 1].id }, range) : null
}

export function isBackFocus(p: ParamPath, s: System): boolean {
  const img = imageIndex(s)
  return p.param === 'thickness' && img >= 1 && s.surfaces[img - 1].id === p.surface
}

export const compensatorTitle = (p: ParamPath, s: System) => (isBackFocus(p, s) ? 'Back focus' : paramTitle(s, p))

export const compensatorJson = (c: Compensator): JsonObject => ({ param: c.param, range: c.range })
export const compensatorKey = (c: Compensator) => JSON.stringify(compensatorJson(c))

export function readCompensator(j: Json): Compensator | null {
  const o = obj(j)
  const p = obj(o?.param)
  if (!o || !p || typeof p.param !== 'string') return null
  const r = nf(o.range)
  return makeCompensator(p as ParamPath, Number.isFinite(r) ? r : null)
}

/** The default ± range of a new compensator. */
export function defaultCompensatorRange(p: ParamPath): number | null {
  switch (p.param) {
    case 'thickness': return 0.5
    case 'decenter_x': case 'decenter_y': return 0.1
    case 'tilt_x': case 'tilt_y': return 0.1
    default: return null
  }
}

/** Parameters a compensator can take: the back focus, every air space, the
 *  design's variables and the surfaces' decentres, tilts, thicknesses and
 *  curvatures, each once. */
export function compensatorCandidates(s: System, variables: ParamPath[]): { group: string; params: ParamPath[] }[] {
  const seen = new Set<string>()
  const fresh = (ps: ParamPath[]) => ps.filter((p) => !seen.has(paramKey(p)) && (seen.add(paramKey(p)), true))
  const img = imageIndex(s)
  const bf = backFocus(s)
  const air: ParamPath[] = []
  for (let i = 1; i < Math.max(img - 1, 1); i++) if (isAir(s.surfaces[i].material)) air.push({ param: 'thickness', surface: s.surfaces[i].id })
  const out = [
    { group: 'Back focus', params: fresh(bf ? [bf.param] : []) },
    { group: 'Air spaces', params: fresh(air) },
    { group: 'Design variables', params: fresh(variables) },
  ]
  const other: ParamPath[] = []
  for (let i = 1; i < Math.max(img, 1); i++) {
    const id = s.surfaces[i].id
    for (const t of ['decenter_x', 'decenter_y', 'tilt_x', 'tilt_y', 'thickness']) other.push({ param: t, surface: id })
    const sh = s.surfaces[i].shape.type
    if (sh !== 'plane' && sh !== 'coordinate_break' && sh !== 'paraxial') other.push({ param: 'curvature', surface: id })
  }
  out.push({ group: 'Surfaces', params: fresh(other.filter((p) => getParam(s, p) != null)) })
  return out.filter((g) => g.params.length > 0)
}

// MARK: Criterion

export type CriterionKind = 'rms_spot' | 'wavefront' | 'mtf' | 'design_merit'
export const CRITERION_KINDS: { value: CriterionKind; title: string }[] = [
  { value: 'rms_spot', title: 'RMS spot radius' },
  { value: 'wavefront', title: 'RMS wavefront' },
  { value: 'mtf', title: 'MTF' },
  { value: 'design_merit', title: 'Optimizer’s merit' },
]

/** The merit editor's generator settings the criterion uses (`DefaultMeritSpec`). */
export interface CriterionDefaults {
  reference: string
  scope: string
  mtfFrequencies: number[]
  mtfTarget: number
  mtfMethod: string
  worstField: boolean
  rings: number
  arms: number
  /** null: every field (or wavelength) with positive weight. */
  fields: number[] | null
  wavelengths: number[] | null
}

export interface Criterion {
  kind: CriterionKind
  defaults: CriterionDefaults
  /** Pass limit for the yield in the display unit (µm, waves, MTF or merit). */
  threshold: number | null
}

export const defaultCriterion = (): Criterion => ({
  kind: 'rms_spot',
  defaults: { reference: 'centroid', scope: 'monochromatic', mtfFrequencies: [20], mtfTarget: 0.5, mtfMethod: 'diffraction', worstField: false, rings: 3, arms: 6, fields: null, wavelengths: null },
  threshold: null,
})

const ints = (j: Json | undefined): number[] | null => (Array.isArray(j) ? j.map((x) => int(x)).filter((x): x is number => x != null) : null)

function readCriterionDefaults(j: Json | undefined, base: CriterionDefaults): CriterionDefaults {
  const o = obj(j)
  if (!o) return base
  const c = obj(o.criterion) ?? {}
  const smp = obj(o.sampling) ?? {}
  return {
    reference: typeof c.reference === 'string' ? c.reference : base.reference,
    scope: typeof c.scope === 'string' ? c.scope : base.scope,
    mtfFrequencies: Array.isArray(c.frequencies) ? nums(c.frequencies) : base.mtfFrequencies,
    mtfTarget: Number.isFinite(nf(c.target)) ? nf(c.target) : base.mtfTarget,
    mtfMethod: typeof c.method === 'string' ? c.method : base.mtfMethod,
    worstField: c.worst_field === true,
    rings: int(smp.rings) ?? base.rings,
    arms: int(smp.arms) ?? base.arms,
    fields: ints(o.fields),
    wavelengths: ints(o.wavelengths),
  }
}

/** The `default_merit` of a criterion kind (no thickness boundaries). */
export function criterionDefaultMerit(d: CriterionDefaults, type: 'rms_spot' | 'wavefront' | 'mtf'): JsonObject {
  const crit: JsonObject = { type }
  if (type === 'rms_spot') Object.assign(crit, { reference: d.reference, scope: d.scope })
  else if (type === 'wavefront') crit.reference = d.reference
  else Object.assign(crit, { frequencies: d.mtfFrequencies, target: d.mtfTarget, method: d.mtfMethod, worst_field: d.worstField })
  return {
    criterion: crit,
    sampling: { rings: d.rings, arms: d.arms },
    fields: d.fields,
    wavelengths: d.wavelengths,
    thickness: null,
  }
}

export function criterionJson(c: Criterion): JsonObject {
  const type = c.kind === 'design_merit' ? 'rms_spot' : c.kind
  return { kind: c.kind, default_merit: criterionDefaultMerit(c.defaults, type), threshold: c.threshold }
}

export function readCriterion(j: Json | undefined): Criterion {
  const c = defaultCriterion()
  const o = obj(j)
  if (!o) return c
  const kind = CRITERION_KINDS.some((k) => k.value === o.kind) ? (o.kind as CriterionKind) : 'rms_spot'
  const t = nf(o.threshold)
  return { kind, defaults: o.default_merit != null ? readCriterionDefaults(o.default_merit, c.defaults) : c.defaults, threshold: Number.isFinite(t) ? t : null }
}

/** The design's merit (the project's `merit` section), as far as tolerancing reads it. */
export interface MeritInfo {
  mode: 'default' | 'custom'
  defaultMerit: JsonObject
  spec: JsonObject
  operandCount: number
}

export function readMeritInfo(section: Json | undefined): MeritInfo {
  const o = obj(section) ?? {}
  const spec = obj(o.spec) ?? { name: '', schema_version: 1, operands: [] }
  return {
    mode: o.mode === 'custom' ? 'custom' : 'default',
    defaultMerit: obj(o.default_merit) ?? {
      criterion: { type: 'rms_spot', reference: 'centroid', scope: 'monochromatic' },
      sampling: { rings: 3, arms: 6 },
      fields: null,
      wavelengths: null,
    },
    spec,
    operandCount: arr(spec.operands).length,
  }
}

/** The `criterion` / `default_merit` settings of the job. */
export function criterionJobSettings(c: Criterion, merit: MeritInfo): JsonObject {
  if (c.kind === 'design_merit') {
    if (merit.mode === 'custom') return { criterion: merit.spec }
    return { default_merit: { ...merit.defaultMerit, thickness: null } }
  }
  const d = { ...c.defaults }
  // The MTF shortfall from 1, so the criterion never saturates at 0.
  if (c.kind === 'mtf') d.mtfTarget = 1
  return { default_merit: criterionDefaultMerit(d, c.kind) }
}

/** A one-line description of the design's merit. */
export function meritInfoSummary(m: MeritInfo): string {
  if (m.mode === 'custom') return `The Optimize panel's custom merit (${m.operandCount} operands).`
  const d = readCriterionDefaults(m.defaultMerit, defaultCriterion().defaults)
  const type = obj(m.defaultMerit.criterion)?.type
  let s = type === 'wavefront' ? 'RMS wavefront' : type === 'mtf' ? `MTF ≥ ${short(d.mtfTarget)} at ${d.mtfFrequencies.map(short).join(', ')} cyc/mm` : 'RMS spot radius'
  s += ` · ${d.rings} rings × ${d.arms} arms`
  return `The Optimize panel's merit: ${s}.`
}

/** How criterion values (merits) read: the engine's criterion is a weighted
 *  mean of squares, so √ is the RMS spot (mm) or wavefront (waves); for the
 *  MTF, 1 − √ is the mean MTF. Lower merits are better in every case. */
export type CriterionDisplay = 'spot' | 'wavefront' | 'mtf' | 'merit'

export function criterionDisplay(c: Criterion, merit: MeritInfo): CriterionDisplay {
  switch (c.kind) {
    case 'rms_spot': return 'spot'
    case 'wavefront': return 'wavefront'
    case 'mtf': return 'mtf'
    case 'design_merit': {
      if (merit.mode !== 'default') return 'merit'
      const crit = obj(merit.defaultMerit.criterion) ?? {}
      if (crit.type === 'wavefront') return 'wavefront'
      if (crit.type === 'mtf') return nf(crit.target ?? 0.5) >= 1 ? 'mtf' : 'merit'
      return 'spot'
    }
  }
}

export const DISPLAY = {
  title: (d: CriterionDisplay) => ({ spot: 'RMS spot', wavefront: 'RMS wavefront', mtf: 'MTF', merit: 'Merit' })[d],
  unit: (d: CriterionDisplay) => ({ spot: 'µm', wavefront: 'waves', mtf: '', merit: '' })[d],
  higherIsBetter: (d: CriterionDisplay) => d === 'mtf',
  value(d: CriterionDisplay, c: number): number {
    if (!Number.isFinite(c)) return NaN
    switch (d) {
      case 'spot': return Math.sqrt(c) * 1000
      case 'wavefront': return Math.sqrt(c)
      case 'mtf': return 1 - Math.sqrt(c)
      case 'merit': return c
    }
  },
  /** The criterion (merit) of a shown value: the yield threshold. */
  criterion(d: CriterionDisplay, v: number): number {
    switch (d) {
      case 'spot': return (v / 1000) ** 2
      case 'wavefront': return v * v
      case 'mtf': return (1 - v) ** 2
      case 'merit': return v
    }
  },
  format(d: CriterionDisplay, v: number): string {
    if (!Number.isFinite(v)) return '—'
    switch (d) {
      case 'spot': return v.toFixed(v < 10 ? 2 : 1)
      case 'wavefront': case 'mtf': return v.toFixed(3)
      case 'merit': return short(v)
    }
  },
  formatWithUnit(d: CriterionDisplay, v: number): string {
    const u = DISPLAY.unit(d)
    return u ? `${DISPLAY.format(d, v)} ${u}` : DISPLAY.format(d, v)
  },
  /** A change in the shown value, signed. */
  formatChange(d: CriterionDisplay, x: number): string {
    if (!Number.isFinite(x)) return '—'
    const s = DISPLAY.format(d, Math.abs(x))
    if (/^[0.]*$/.test(s)) return ' ' + s
    return (x < 0 ? '−' : '+') + s
  },
}

export function criterionCaption(d: CriterionDisplay): string {
  switch (d) {
    case 'spot': return 'RMS spot radius over the chosen fields and wavelengths (root of the weighted mean square), after the compensators.'
    case 'wavefront': return 'RMS wavefront error in waves over the chosen fields and wavelengths, after the compensators.'
    case 'mtf': return 'Polychromatic tangential and sagittal MTF, the root-mean-square shortfall from 1 shown as an MTF. Each trial costs as much as an MTF analysis per compensator step: start with a few dozen trials.'
    case 'merit': return 'The merit value itself (lower is better).'
  }
}

// MARK: The saved set

export type Mode = 'sensitivity' | 'monte_carlo' | 'both'
export const MODES: { value: Mode; title: string }[] = [
  { value: 'sensitivity', title: 'Sensitivity' },
  { value: 'monte_carlo', title: 'Monte Carlo' },
  { value: 'both', title: 'Both' },
]

/** He–Ne, the usual interferometer. */
export const DEFAULT_TEST_WAVELENGTH_UM = 0.6328
export const DEFAULT_PERCENTILES = [50, 90, 97.7]
export const MAX_RECORD_TRIALS = 10_000
export const MAX_TRIALS = 100_000

export interface ToleranceSettings {
  rows: ToleranceRow[]
  compensators: Compensator[]
  generator: Generator
  criterion: Criterion
  mode: Mode
  trials: number
  seed: number
  /** The interferometer wavelength (µm) fringes are counted at. */
  testWavelengthUm: number
  percentiles: number[]
  recordTrials: number
}

export function emptySettings(): ToleranceSettings {
  return {
    rows: [],
    compensators: [],
    generator: defaultGenerator(),
    criterion: defaultCriterion(),
    mode: 'both',
    trials: 200,
    seed: 0,
    testWavelengthUm: DEFAULT_TEST_WAVELENGTH_UM,
    percentiles: [...DEFAULT_PERCENTILES],
    recordTrials: 200,
  }
}

/** The default set of a design: the precision class and the back focus. */
export function defaultSettings(s: System): ToleranceSettings {
  const t = emptySettings()
  t.rows = generateRows(t.generator, s)
  const bf = backFocus(s)
  t.compensators = bf ? [bf] : []
  return t
}

/** Percent levels in [0, 100], sorted, without repeats; null when none is left. */
export function validPercentiles(v: number[]): number[] | null {
  const ok = [...new Set(v.filter((x) => Number.isFinite(x) && x >= 0 && x <= 100))].sort((a, b) => a - b)
  return ok.length ? ok : null
}

/** Parses "50, 90, 97.7" (commas, semicolons or spaces; % allowed). */
export function parsePercentiles(text: string): number[] | null {
  const parts = text.split(/[,;\s]+/).map((p) => p.replace('%', '').trim()).filter(Boolean)
  if (!parts.length) return null
  const v = parts.map((p) => Number(p.replace(/−/g, '-')))
  if (v.some((x) => !Number.isFinite(x) || x < 0 || x > 100)) return null
  return validPercentiles(v)
}

/** Parses a frequency list "10, 30" (cyc/mm). */
export function parseNumberList(text: string): number[] | null {
  const parts = text.split(/[,;\s]+/).filter(Boolean)
  if (!parts.length) return null
  const v = parts.map((p) => Number(p.replace(/−/g, '-')))
  return v.some((x) => !Number.isFinite(x) || x <= 0) ? null : v
}

/** Reads the project's `tolerances` section; null (a new design) gives the default set. */
export function readSettings(section: Json | undefined, s: System): ToleranceSettings {
  const o = obj(section)
  if (!o) return defaultSettings(s)
  const t = emptySettings()
  const app = obj(o.app)
  if (app && Array.isArray(app.rows)) t.rows = app.rows.map(readRow).filter((r): r is ToleranceRow => r != null)
  else t.rows = arr(o.operands).map((x) => rowFromOperand(x, s)).filter((r): r is ToleranceRow => r != null)
  t.compensators = arr(o.compensators).map(readCompensator).filter((c): c is Compensator => c != null)
  t.generator = readGenerator(app?.generator)
  t.criterion = readCriterion(app?.criterion)
  t.mode = MODES.some((m) => m.value === app?.mode) ? (app!.mode as Mode) : 'both'
  const mc = obj(o.monte_carlo)
  t.trials = int(mc?.trials) ?? t.trials
  t.seed = int(mc?.seed) ?? t.seed
  const p = Array.isArray(mc?.percentiles) ? validPercentiles(nums(mc!.percentiles)) : null
  if (p) t.percentiles = p
  const rt = int(mc?.record_trials)
  if (rt != null) t.recordTrials = Math.min(Math.max(rt, 0), MAX_RECORD_TRIALS)
  const w = nf(app?.test_wavelength_um)
  if (w > 0 && Number.isFinite(w)) t.testWavelengthUm = w
  return t
}

/** `c` with the set's test wavelength. */
export const withWavelength = (t: ToleranceSettings, c: ToleranceContext): ToleranceContext => ({ ...c, testWavelengthUm: t.testWavelengthUm })

/** Records the engine keeps: at most every trial, at most its cap. */
export const effectiveRecordTrials = (t: ToleranceSettings) => Math.min(Math.max(t.recordTrials, 0), t.trials, MAX_RECORD_TRIALS)

/** The section: the engine's spec as expanded in `c` (no operands without a
 *  context), and the app's rows. */
export function settingsSection(t: ToleranceSettings, c: ToleranceContext | null, merit: MeritInfo | null = null): JsonObject {
  const ctx = c ? withWavelength(t, c) : null
  const ops: Json[] = []
  if (ctx) for (const r of t.rows) {
    const o = rowOperand(r, ctx)
    if (o.ok) ops.push(o.operand)
  }
  const m = merit ?? readMeritInfo(null)
  const display = criterionDisplay(t.criterion, m)
  return {
    schema_version: 1,
    operands: ops,
    compensators: t.compensators.map(compensatorJson),
    criterion: criterionJobSettings(t.criterion, m).criterion ?? null,
    monte_carlo: {
      trials: t.trials,
      seed: t.seed,
      batch: 16,
      threshold: t.criterion.threshold != null ? DISPLAY.criterion(display, t.criterion.threshold) : null,
      percentiles: t.percentiles,
      record_trials: effectiveRecordTrials(t),
    },
    app: {
      rows: t.rows.map(rowJson),
      generator: generatorJson(t.generator),
      criterion: criterionJson(t.criterion),
      mode: t.mode,
      test_wavelength_um: t.testWavelengthUm,
    },
  }
}

/** Identity of the saved content (session ids aside). */
export const settingsKey = (t: ToleranceSettings) => JSON.stringify(settingsSection(t, null))

/** Rows keep their session ids where they match by content (longest common
 *  subsequence, so duplicates keep their own), then a row of the same kind and
 *  surface; compensators by content. */
export function keepingIds(next: ToleranceSettings, old: ToleranceSettings): ToleranceSettings {
  const o = old.rows.map(rowKey), n = next.rows.map(rowKey)
  const rows = next.rows.map((r) => ({ ...r }))
  const lcs: number[][] = Array.from({ length: o.length + 1 }, () => new Array<number>(n.length + 1).fill(0))
  for (let i = o.length - 1; i >= 0; i--) for (let j = n.length - 1; j >= 0; j--) lcs[i][j] = o[i] === n[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1])
  const used = new Set<number>(), matched = new Set<number>()
  let i = 0, j = 0
  while (i < o.length && j < n.length) {
    if (o[i] === n[j] && lcs[i][j] === lcs[i + 1][j + 1] + 1) {
      rows[j].id = old.rows[i].id
      used.add(i)
      matched.add(j)
      i++
      j++
    } else if (lcs[i][j + 1] >= lcs[i + 1][j]) j++
    else i++
  }
  rows.forEach((r, k) => {
    if (matched.has(k)) return
    const m = old.rows.findIndex((x, q) => !used.has(q) && x.kind === r.kind && x.surface === r.surface)
    if (m >= 0) {
      r.id = old.rows[m].id
      used.add(m)
    }
  })
  const pool = [...old.compensators]
  const compensators = next.compensators.map((c) => {
    const q = pool.findIndex((x) => compensatorKey(x) === compensatorKey(c))
    if (q < 0) return c
    const id = pool[q].id
    pool.splice(q, 1)
    return { ...c, id }
  })
  return { ...next, rows, compensators }
}

// MARK: Editing (each returns a new set)

export function cloneSettings(t: ToleranceSettings): ToleranceSettings {
  return structuredClone(t)
}

/** Adds rows after `after` (or at the end). An element row joins the range and
 *  pivot of the rows already moving from its first surface. */
export function addRows(t: ToleranceSettings, add: ToleranceRow[], after: string | null = null): void {
  const k = after ? t.rows.findIndex((r) => r.id === after) : -1
  t.rows.splice(k >= 0 ? k + 1 : t.rows.length, 0, ...add)
  const ids = new Set(add.map((r) => r.id))
  for (const r of add) {
    if (!isElementKind(r.kind)) continue
    const other = t.rows.find((x) => !ids.has(x.id) && isElementKind(x.kind) && x.surface === r.surface)
    if (other) {
      r.last = other.last
      r.pivot = other.pivot
    }
  }
}

export function deleteRows(t: ToleranceSettings, ids: Set<string>): void {
  t.rows = t.rows.filter((r) => !ids.has(r.id))
}

/** Copies of the rows, each after its original; returns their ids. */
export function duplicateRows(t: ToleranceSettings, ids: Set<string>): string[] {
  const out: string[] = []
  for (let i = t.rows.length - 1; i >= 0; i--) {
    if (!ids.has(t.rows[i].id)) continue
    const copy = { ...structuredClone(t.rows[i]), id: newId() }
    t.rows.splice(i + 1, 0, copy)
    out.push(copy.id)
  }
  return out
}

/** Moves element row `id` to `first…last` about `pivot` (null: the first
 *  vertex); every element row starting on `first` follows (the engine moves
 *  them as one rigid group). */
export function setElement(t: ToleranceSettings, id: string, first: number, last: number, pivot: number | null): void {
  const row = t.rows.find((r) => r.id === id)
  if (!row || !isElementKind(row.kind)) return
  row.surface = first
  row.last = last
  row.pivot = pivot === first ? null : pivot
  for (const r of t.rows) {
    if (isElementKind(r.kind) && r.surface === first) {
      r.last = row.last
      r.pivot = row.pivot
    }
  }
}

/** The rows that cannot run, with why. */
export function settingsIssues(t: ToleranceSettings, c: ToleranceContext): Map<string, string> {
  const ctx = withWavelength(t, c)
  const out = new Map<string, string>()
  for (const r of t.rows) {
    const l = engineLimits(r, ctx)
    if (!l.ok) out.set(r.id, l.issue)
  }
  // Element rows that start on one surface share their range and pivot.
  const groups = new Map<number, ToleranceRow[]>()
  for (const r of t.rows) if (isElementKind(r.kind) && !out.has(r.id)) groups.set(r.surface, [...(groups.get(r.surface) ?? []), r])
  for (const [first, group] of groups) {
    const ranges = new Set(group.map((r) => `${r.last ?? first}/${r.pivot ?? first}`))
    if (ranges.size <= 1) continue
    const i = surfaceIndexOfId(ctx.system, first)
    const at = i >= 0 ? `S${i}` : 'one surface'
    for (const r of group) out.set(r.id, `Element tolerances starting at ${at} move one group: give them the same last surface and pivot.`)
  }
  return out
}

/** Why the run cannot start, if it cannot. */
export function runProblem(t: ToleranceSettings, c: ToleranceContext): string | null {
  const n = settingsIssues(t, c).size
  if (n > 0) return n === 1 ? 'Fix the tolerance marked in red first.' : `Fix the ${n} tolerances marked in red first.`
  if (!t.rows.length) return 'Add a tolerance first.'
  if (t.mode !== 'sensitivity' && t.trials < 2) return 'Run at least 2 Monte Carlo trials.'
  return null
}

/** The job settings, with the rows sent in the engine's operand order. */
export function jobSettings(t: ToleranceSettings, c: ToleranceContext, merit: MeritInfo): { settings: JsonObject; rows: ToleranceRow[] } {
  const ctx = withWavelength(t, c)
  const blocked = settingsIssues(t, ctx)
  const ops: Json[] = []
  const rows: ToleranceRow[] = []
  for (const r of t.rows) {
    if (blocked.has(r.id)) continue
    const o = rowOperand(r, ctx)
    if (o.ok) {
      ops.push(o.operand)
      rows.push(r)
    }
  }
  const s: JsonObject = {
    tolerances: ops,
    compensators: t.compensators.map(compensatorJson),
    mode: t.mode,
    trials: t.trials,
    seed: t.seed,
    percentiles: t.percentiles,
    record_trials: t.mode === 'sensitivity' ? 0 : effectiveRecordTrials(t),
    irregularity_wavelength_um: t.testWavelengthUm,
    ...criterionJobSettings(t.criterion, merit),
  }
  if (t.criterion.threshold != null) s.threshold = DISPLAY.criterion(criterionDisplay(t.criterion, merit), t.criterion.threshold)
  return { settings: s, rows }
}

// MARK: Results

export interface Sensitivity {
  /** The engine's operand index (= the row's position in the run). */
  id: number
  label: string
  param: ParamPath | null
  minus: number
  plus: number
  criterionMinus: number
  criterionPlus: number
  changeMinus: number
  changePlus: number
  worst: number
  ok: boolean
  compensatorsMinus: number[]
  compensatorsPlus: number[]
}

/** The limit whose change is the larger. */
export const worsePlus = (s: Sensitivity) => !(s.changeMinus > s.changePlus)

export interface TrialRecord {
  trial: number
  criterion: number
  ok: boolean
  compensators: number[]
  perturbations: number[]
}

export interface ToleranceResult {
  status: string
  stopped: boolean
  display: CriterionDisplay
  compensators: { param: ParamPath; range: number | null }[]
  nominal: number
  nominalCompensators: number[]
  hasSensitivity: boolean
  /** Worst first. */
  sensitivities: Sensitivity[]
  rssChange: number
  predicted: number
  sensitivityCancelled: boolean
  trials: number
  failures: number
  seed: number
  mean: number
  stdDev: number
  min: number
  max: number
  percentiles: { percentile: number; value: number }[]
  /** Criterion by trial (not finite: failed). */
  trialCriteria: number[]
  /** Finite criteria. */
  criteria: number[]
  records: TrialRecord[]
  secondsPerTrial: number | null
  yieldFraction: number | null
  threshold: number | null
  compensatorMean: number[]
  compensatorStd: number[]
}

const finiteOrNull = (v: Json | undefined) => {
  const x = nf(v)
  return Number.isFinite(x) ? x : null
}

export function readResult(json: JsonObject, s: System | null, display: CriterionDisplay, criteriaBuffer?: ArrayLike<number> | null): ToleranceResult {
  const sens = obj(json.sensitivity)
  const mc = obj(json.monte_carlo)
  const nominal = obj(sens?.nominal) ?? obj(mc?.nominal)
  const ops: Sensitivity[] = arr(sens?.operands).map((x, k) => {
    const o = obj(x) ?? {}
    const param = obj(o.param) && typeof obj(o.param)!.param === 'string' ? (o.param as ParamPath) : null
    let label = typeof o.label === 'string' ? o.label : ''
    if ((!label || label.startsWith('{')) && param && s) label = paramTitle(s, param)
    return {
      id: int(o.index) ?? k,
      label,
      param,
      minus: nf(o.minus),
      plus: nf(o.plus),
      criterionMinus: nf(o.criterion_minus),
      criterionPlus: nf(o.criterion_plus),
      changeMinus: nf(o.change_minus),
      changePlus: nf(o.change_plus),
      worst: nf(o.worst),
      ok: o.ok !== false,
      compensatorsMinus: nums(o.compensators_minus),
      compensatorsPlus: nums(o.compensators_plus),
    }
  })
  const ranking = arr(sens?.ranking).map(int)
  const ranked = ranking.map((r) => ops.find((o) => o.id === r)).filter((o): o is Sensitivity => o != null)
  const w = (x: Sensitivity) => (Number.isFinite(x.worst) ? x.worst : -Infinity)
  const sensitivities = ranked.length === ops.length ? ranked : [...ops].sort((a, b) => w(b) - w(a))
  const trialCriteria = mc && Array.isArray(mc.criteria) ? nums(mc.criteria) : criteriaBuffer ? Array.from(criteriaBuffer) : []
  const spt = nf(mc?.seconds_per_trial)
  return {
    status: typeof json.status === 'string' ? json.status : '',
    stopped: typeof json.stopped === 'boolean' ? json.stopped : json.status === 'stopped',
    display,
    compensators: arr(json.compensators)
      .map((c) => {
        const o = obj(c)
        const p = obj(o?.param)
        return p && typeof p.param === 'string' ? { param: p as ParamPath, range: finiteOrNull(o!.range) } : null
      })
      .filter((c): c is { param: ParamPath; range: number | null } => c != null),
    nominal: nf(nominal?.criterion),
    nominalCompensators: nums(nominal?.compensators),
    hasSensitivity: sens != null,
    sensitivities,
    rssChange: nf(sens?.rss_change),
    predicted: nf(sens?.predicted_criterion),
    sensitivityCancelled: sens?.cancelled === true,
    trials: int(mc?.trials) ?? 0,
    failures: int(mc?.failures) ?? 0,
    seed: int(mc?.seed) ?? 0,
    mean: nf(mc?.mean),
    stdDev: nf(mc?.std_dev),
    min: nf(mc?.min),
    max: nf(mc?.max),
    percentiles: arr(mc?.percentiles)
      .map((p) => {
        const o = obj(p)
        return o && Number.isFinite(nf(o.percentile)) ? { percentile: nf(o.percentile), value: nf(o.value) } : null
      })
      .filter((p): p is { percentile: number; value: number } => p != null),
    trialCriteria,
    criteria: trialCriteria.filter(Number.isFinite),
    records: arr(mc?.records)
      .map((r) => {
        const o = obj(r)
        const t = int(o?.trial)
        if (!o || t == null) return null
        return { trial: t, criterion: nf(o.criterion), ok: o.ok !== false, compensators: nums(o.compensators), perturbations: nums(o.perturbations) }
      })
      .filter((r): r is TrialRecord => r != null),
    secondsPerTrial: Number.isFinite(spt) && spt >= 0 ? spt : null,
    yieldFraction: finiteOrNull(mc?.yield_fraction),
    threshold: finiteOrNull(mc?.threshold),
    compensatorMean: nums(mc?.compensator_mean),
    compensatorStd: nums(mc?.compensator_std),
  }
}

export const hasMonteCarlo = (r: ToleranceResult) => r.trials > 0
export const hasNominal = (r: ToleranceResult) => Number.isFinite(r.nominal)
export const recordOf = (r: ToleranceResult, k: number) => r.records.find((x) => x.trial === k) ?? null

/** An operand's share of the RSS prediction: worst² / Σ worst². */
export function share(r: ToleranceResult, s: Sensitivity): number {
  const sq = (x: number) => (Number.isFinite(x) ? Math.max(x, 0) ** 2 : 0)
  const total = r.sensitivities.reduce((a, x) => a + sq(x.worst), 0)
  if (!(total > 0) || !Number.isFinite(s.worst)) return 0
  return sq(s.worst) / total
}

/** The worst offenders: the largest shares that make up half of the RSS (at
 *  most five, each above 5 %). */
export function worstOffenders(r: ToleranceResult): Set<number> {
  const out = new Set<number>()
  let sum = 0
  for (const s of r.sensitivities) {
    const f = share(r, s)
    if (!(f > 0.05 && sum < 0.5 && out.size < 5)) break
    out.add(s.id)
    sum += f
  }
  return out
}

/** Trials whose shown value lies in [lo, hi], worst first. */
export function trialsShownIn(r: ToleranceResult, lo: number, hi: number): number[] {
  const ks: number[] = []
  r.trialCriteria.forEach((c, k) => {
    const v = DISPLAY.value(r.display, c)
    if (Number.isFinite(v) && v >= lo && v <= hi) ks.push(k)
  })
  return ks.sort((a, b) => r.trialCriteria[b] - r.trialCriteria[a])
}

/** Every completed trial, worst first. */
export function trialsWorstFirst(r: ToleranceResult): number[] {
  const ks: number[] = []
  r.trialCriteria.forEach((c, k) => Number.isFinite(c) && ks.push(k))
  return ks.sort((a, b) => r.trialCriteria[b] - r.trialCriteria[a])
}

/** A compensator's range over the sensitivity evaluations, relative to its
 *  nominal value: [lowest, highest]. */
export function compensatorRange(r: ToleranceResult, j: number): [number, number] | null {
  if (!r.hasSensitivity || j >= r.nominalCompensators.length) return null
  const n = r.nominalCompensators[j]
  const v = r.sensitivities.flatMap((s) => [s.compensatorsMinus[j], s.compensatorsPlus[j]]).filter((x) => x != null && Number.isFinite(x))
  if (!v.length) return null
  return [Math.min(Math.min(...v) - n, 0), Math.max(Math.max(...v) - n, 0)]
}

// MARK: Histogram

export interface Bin {
  center: number
  count: number
  lo: number
  hi: number
}

/** The histogram of the shown values (10, 16 or 24 bars by trial count). */
export function histogramBins(r: ToleranceResult): Bin[] {
  const v = r.criteria.map((c) => DISPLAY.value(r.display, c)).filter(Number.isFinite)
  if (!v.length) return []
  const n = v.length >= 200 ? 24 : v.length >= 60 ? 16 : 10
  const lo = Math.min(...v), hi = Math.max(...v)
  if (!(hi > lo)) {
    const w = Math.max(Math.abs(lo) * 0.1, 1e-9)
    return [{ center: lo, count: v.length, lo: lo - w / 2, hi: lo + w / 2 }]
  }
  const w = (hi - lo) / n
  const c = new Array<number>(n).fill(0)
  for (const x of v) c[Math.min(Math.floor((x - lo) / w), n - 1)]++
  return c.map((count, k) => ({ center: lo + (k + 0.5) * w, count, lo: lo + k * w, hi: lo + (k + 1) * w }))
}

/** Levels with labels; one too close to the one before it (under 8 % of the
 *  axis) shares that label ("90, 97.7"). */
export function labelledLevels(levels: { p: number; v: number }[], span: number): { p: number; v: number; label: string }[] {
  const out: { p: number; v: number; label: string }[] = []
  for (const l of levels) {
    const last = out[out.length - 1]
    if (last && Number.isFinite(last.v) && Number.isFinite(l.v) && Math.abs(l.v - last.v) < 0.08 * span && last.label) {
      last.label += ', ' + short(l.p)
      out.push({ ...l, label: '' })
    } else out.push({ ...l, label: short(l.p) })
  }
  return out
}

// MARK: Previews

/** A toleranced build the engine rebuilds (`preview_case`, zero-based). */
export type PreviewCase = { type: 'trial'; trial: number } | { type: 'sensitivity'; operand: number; positive: boolean }

export const previewCaseJson = (c: PreviewCase): JsonObject => ({ ...c })
export const samePreviewCase = (a: PreviewCase | null, b: PreviewCase | null) => JSON.stringify(a) === JSON.stringify(b)

export interface PreviewBuild {
  system: System
  criterion: number
  compensators: number[]
  perturbations: number[]
}

export function readPreview(j: Json | undefined): PreviewBuild | null {
  const o = obj(j)
  const s = obj(o?.system)
  if (!o || !s) return null
  return { system: s as System, criterion: nf(o.criterion), compensators: nums(o.compensators), perturbations: nums(o.perturbations) }
}

/** "Tolerance: trial 17, compensated", "Tolerance: Surface decenter y S3 at +0.02 mm". */
export function previewTitle(c: PreviewCase, rows: ToleranceRow[], system: System | null, result: ToleranceResult | null): string {
  const compensated = result && result.compensators.length ? ', compensated' : ''
  if (c.type === 'trial') return `Tolerance: trial ${c.trial + 1}${compensated}`
  const row = rows[c.operand]
  const what = row && system ? `${rowLabel(row, system)} at ${limitText(row, c.positive)}` : (result?.sensitivities.find((x) => x.id === c.operand)?.label ?? `operand ${c.operand + 1}`)
  return `Tolerance: ${what}${compensated}`
}

export interface BuildChange {
  id: number
  label: string
  /** The perturbation over the limit on its side (−1 … 1). */
  fraction: number
  text: string
  limits: string
}

/** A build's perturbations (parameter units, run order) as the rows read them,
 *  largest share of its limit first; zero ones left out. */
export function buildChanges(perturbations: number[], rows: ToleranceRow[], runSettings: JsonObject | null, system: System | null): BuildChange[] {
  const limits = arr(runSettings?.tolerances).map((t) => ({ minus: nf(obj(t)?.minus), plus: nf(obj(t)?.plus) }))
  const out: BuildChange[] = []
  perturbations.forEach((d, k) => {
    if (d === 0 || !Number.isFinite(d) || !limits[k] || !rows[k]) return
    const row = rows[k]
    const lim = d > 0 ? limits[k].plus : limits[k].minus
    if (!(lim > 0)) return
    const f = d / lim
    const shown = (d > 0 ? row.plus : row.minus) * Math.abs(f)
    out.push({
      id: k,
      label: system ? rowLabel(row, system) : KINDS[row.kind].title,
      fraction: f,
      text: (d > 0 ? '+' : '−') + short(tidy(shown, 3)) + unitSuffix(row.unit),
      limits: limitsText(row),
    })
  })
  return out.sort((a, b) => Math.abs(b.fraction) - Math.abs(a.fraction))
}

// MARK: Timing

/** The share of the job's progress the sensitivity takes. */
const sensitivityShare = (m: Mode) => (m === 'sensitivity' ? 1 : m === 'monte_carlo' ? 0 : 0.2)

export interface Timing {
  trialsDone: number | null
  secondsPerTrial: number | null
  secondsLeft: number | null
}

export function timing(a: {
  elapsed: number
  monteCarloElapsed: number | null
  fraction: number
  mode: Mode
  trials: number
  operands: number
  lastSecondsPerTrial: number | null
}): Timing {
  const f = Math.min(Math.max(a.fraction, 0), 1)
  const split = sensitivityShare(a.mode)
  const t: Timing = { trialsDone: null, secondsPerTrial: null, secondsLeft: null }
  if (a.mode !== 'sensitivity' && (f > split || a.monteCarloElapsed != null)) {
    const mcf = split < 1 ? Math.min(Math.max((f - split) / (1 - split), 0), 1) : 0
    const done = Math.round(mcf * a.trials)
    t.trialsDone = done
    t.secondsPerTrial = a.monteCarloElapsed != null && done >= 2 && a.monteCarloElapsed > 0 ? a.monteCarloElapsed / done : a.lastSecondsPerTrial
    if (t.secondsPerTrial != null) t.secondsLeft = Math.max(a.trials - done, 0) * t.secondsPerTrial
    return t
  }
  const sf = split > 0 ? f / split : 0
  const evaluations = 2 * Math.max(a.operands, 0) + 1
  if (sf < 0.05) {
    if (a.mode !== 'sensitivity' && a.lastSecondsPerTrial != null) {
      t.secondsPerTrial = a.lastSecondsPerTrial
      t.secondsLeft = evaluations * a.lastSecondsPerTrial + a.trials * a.lastSecondsPerTrial
    }
    return t
  }
  const sensLeft = (a.elapsed * (1 - sf)) / sf
  const perEvaluation = a.elapsed / (sf * evaluations)
  if (a.mode === 'sensitivity') t.secondsLeft = sensLeft
  else {
    t.secondsPerTrial = a.lastSecondsPerTrial ?? perEvaluation
    t.secondsLeft = sensLeft + a.trials * t.secondsPerTrial
  }
  return t
}

/** "Monte Carlo 40 / 200", "Sensitivity". */
export function stageText(stage: string | null | undefined, trialsDone: number | null, trials: number | null): string {
  const s = (stage ?? '').toLowerCase()
  if (s.includes('monte carlo') || s.includes('trial')) return trialsDone != null && trials != null ? `Monte Carlo ${trialsDone} / ${trials}` : 'Monte Carlo'
  if (s.includes('sensitiv')) return 'Sensitivity'
  if (!s || s === 'queued') return 'Starting'
  if (s === 'stopping') return 'Stopping'
  return stage!.charAt(0).toUpperCase() + stage!.slice(1)
}

/** "0.021 s", "12 ms". */
export function secondsText(s: number): string {
  if (!Number.isFinite(s)) return '—'
  if (s < 0.1) return `${short(Math.round(s * 10000) / 10)} ms`
  return `${short(Math.round(s * 100) / 100)} s`
}

/** "4 s", "2 min", "1 h 5 min". */
export function durationText(s: number): string {
  if (!Number.isFinite(s)) return '—'
  if (s < 1) return 'under a second'
  if (s < 90) return `${Math.round(s)} s`
  const m = Math.round(s / 60)
  return m < 90 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`
}

/** "0:42 elapsed · 0.021 s per trial · about 2 min left". */
export function timingText(elapsed: number, t: Timing): string {
  const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s) % 60).padStart(2, '0')}`
  const parts = [clock(elapsed) + ' elapsed']
  if (t.secondsPerTrial != null) parts.push(secondsText(t.secondsPerTrial) + ' per trial')
  if (t.secondsLeft != null) parts.push(t.secondsLeft < 1 ? 'almost done' : `about ${durationText(t.secondsLeft)} left`)
  return parts.join(' · ')
}

/** "Last run: 21 ms per trial, so 200 trials take about 4 s." */
export const estimateText = (spt: number, trials: number) => `Last run: ${secondsText(spt)} per trial, so ${trials} trials take about ${durationText(spt * trials)}.`

// MARK: Export

const pct = (f: number, digits = 1) => (Number.isFinite(f) ? `${(f * 100).toFixed(digits)} %` : '—')
const num = (v: number) => (Number.isFinite(v) ? String(Number(v.toPrecision(10))) : '')
const csv = (s: string) => (/[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s)
const csvName = (d: CriterionDisplay) => ({ spot: 'rms_spot_um', wavefront: 'rms_wavefront_waves', mtf: 'mtf', merit: 'merit' })[d]
const sensLabel = (s: Sensitivity, rows: ToleranceRow[], system: System) => (rows[s.id] ? rowLabel(rows[s.id], system) : s.label)
export const sensLimits = (s: Sensitivity, rows: ToleranceRow[]) => (rows[s.id] ? limitsText(rows[s.id]) : '±' + short(Math.max(s.minus, s.plus)))

/** A plain-text summary: criterion, nominal, prediction, levels, yield, the
 *  largest contributors and the compensators. */
export function summaryText(r: ToleranceResult, rows: ToleranceRow[], system: System, title: string): string {
  const d = r.display
  const unit = DISPLAY.unit(d)
  const lines = [`Tolerance analysis: ${title}`, `Criterion: ${DISPLAY.title(d)}${unit ? ` (${unit})` : ''}`]
  if (r.stopped) lines.push('Stopped early: statistics of the completed evaluations.')
  if (r.hasSensitivity) {
    lines.push(`Nominal: ${DISPLAY.format(d, DISPLAY.value(d, r.nominal))}`)
    lines.push(`Predicted (nominal + RSS): ${DISPLAY.format(d, DISPLAY.value(d, r.predicted))}`)
  }
  if (hasMonteCarlo(r)) {
    lines.push(`Monte Carlo: ${r.trials} trials, seed ${r.seed}${r.failures > 0 ? `, ${r.failures} failed` : ''}`)
    if (r.secondsPerTrial != null) lines.push(`  ${short(r.secondsPerTrial)} s per trial`)
    for (const p of r.percentiles) lines.push(`  ${short(p.percentile)} % of builds ${DISPLAY.higherIsBetter(d) ? 'at least' : 'at most'}: ${DISPLAY.format(d, DISPLAY.value(d, p.value))}`)
    if (r.yieldFraction != null) lines.push(`  Yield: ${pct(r.yieldFraction)}`)
  }
  if (r.hasSensitivity) {
    lines.push('Largest contributors:')
    for (const s of r.sensitivities.slice(0, 5)) lines.push(`  ${sensLabel(s, rows, system)} ${sensLimits(s, rows)}: ${pct(share(r, s), 0)} of the RSS`)
  }
  r.compensators.forEach((c, j) => {
    let t = `Compensator ${compensatorTitle(c.param, system)}`
    const used = compensatorRange(r, j)
    if (used) t += `: moved ${short(used[0])} to +${short(used[1])} ${paramUnit(c.param)}`
    if (c.range != null) t += ` of ±${short(c.range)}`
    if (j < r.compensatorStd.length) t += `; Monte Carlo σ ${short(r.compensatorStd[j])}`
    lines.push(t)
  })
  return lines.join('\n')
}

/** One line per sensitivity operand, worst first. */
export function sensitivityCsv(r: ToleranceResult, rows: ToleranceRow[], system: System): string {
  const d = r.display
  const n = csvName(d)
  const nominal = DISPLAY.value(d, r.nominal)
  const out = [`rank,tolerance,minus,plus,unit,parameter,criterion_minus,criterion_plus,${n}_minus,${n}_plus,change_minus,change_plus,share`]
  r.sensitivities.forEach((s, k) => {
    const row = rows[s.id]
    const vm = DISPLAY.value(d, s.criterionMinus), vp = DISPLAY.value(d, s.criterionPlus)
    out.push(
      [
        String(k + 1), csv(sensLabel(s, rows, system)), num(row ? row.minus : s.minus), num(row ? row.plus : s.plus), csv(row ? UNITS[row.unit].label : ''),
        csv(s.param ? paramTitle(system, s.param) : s.label), num(s.criterionMinus), num(s.criterionPlus), num(vm), num(vp), num(vm - nominal), num(vp - nominal), num(share(r, s)),
      ].join(','),
    )
  })
  return out.join('\n') + '\n'
}

/** One line per Monte-Carlo trial (a failed trial's values empty), with the
 *  recorded compensators and perturbations (parameter units). */
export function trialsCsv(r: ToleranceResult, rows: ToleranceRow[], system: System | null): string {
  const d = r.display
  const recorded = r.records.length > 0
  const comps = recorded ? r.compensators.map((_, j) => `compensator_${j + 1}`) : []
  const perts = recorded ? rows.map((row) => csv(system ? rowLabel(row, system) : KINDS[row.kind].title)) : []
  const out = [['trial', 'criterion', csvName(d), ...comps, ...perts].join(',')]
  const byTrial = new Map(r.records.map((x) => [x.trial, x]))
  r.trialCriteria.forEach((c, k) => {
    const f = [String(k + 1), num(c), num(DISPLAY.value(d, c))]
    if (recorded) {
      const rec = byTrial.get(k)
      f.push(...comps.map((_, j) => (rec && j < rec.compensators.length ? num(rec.compensators[j]) : '')))
      f.push(...perts.map((_, j) => (rec && j < rec.perturbations.length ? num(rec.perturbations[j]) : '')))
    }
    out.push(f.join(','))
  })
  return out.join('\n') + '\n'
}

export const percentText = pct
