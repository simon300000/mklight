// Solves and pickups (docs/SOLVES.md, apps/macos/Sources/MKLightModel/Solves.swift):
// a solve makes one parameter (`target`) follow from a rule (`kind`, tagged by
// `type`). The engine's solver is the authority; this is the editing model.
// Angle solves take the paraxial slope u = dy/dz, dimensionless, never degrees.

import type { Json, JsonObject } from '@/engine/types'
import { canonicalParam, paramKey, paramTitle, sameParam, solveFor, solvesOf as solvesOfParams, surfaceIndexOfId, type ParamPath } from './params'
import { imageIndex, type System } from './system'

export interface Solve {
  target: ParamPath
  kind: JsonObject
}

export type SolveKindId =
  | 'marginal_ray_height'
  | 'marginal_ray_angle'
  | 'chief_ray_height'
  | 'chief_ray_angle'
  | 'f_number'
  | 'pickup'
  | 'position'
  | 'center_of_curvature'
  | 'edge_thickness'

export interface SolveKindInfo {
  id: SolveKindId
  title: string
  /** One-letter marker shown next to a solved value in the lens table. */
  letter: string
  /** The kind's main number: its JSON key, label and unit. */
  value: { key: string; label: string; unit: string } | null
  /** What the kind can set: a thickness, a power (curvature or ideal-lens
   *  focal length), or any parameter (pickups). */
  targets: 'thickness' | 'power' | 'any'
  help: string
}

export const SOLVE_KINDS: SolveKindInfo[] = [
  { id: 'marginal_ray_height', title: 'Marginal ray height', letter: 'M', value: { key: 'height', label: 'Height', unit: 'mm' }, targets: 'thickness', help: 'The marginal ray’s height on the next vertex; 0 on the last thickness puts the image at the paraxial focus.' },
  { id: 'chief_ray_height', title: 'Chief ray height', letter: 'C', value: { key: 'height', label: 'Height', unit: 'mm' }, targets: 'thickness', help: 'The chief ray’s height on the next vertex; 0 puts the next surface on a pupil image.' },
  { id: 'marginal_ray_angle', title: 'Marginal ray angle', letter: 'M', value: { key: 'slope', label: 'Paraxial angle u', unit: '' }, targets: 'power', help: 'The marginal ray’s paraxial slope u = dy/dz after the surface (dimensionless, not degrees).' },
  { id: 'chief_ray_angle', title: 'Chief ray angle', letter: 'C', value: { key: 'slope', label: 'Paraxial angle u', unit: '' }, targets: 'power', help: 'The chief ray’s paraxial slope u = dy/dz after the surface (dimensionless, not degrees).' },
  { id: 'f_number', title: 'F-number', letter: 'F', value: { key: 'value', label: 'F/#', unit: '' }, targets: 'power', help: 'The image-space F/# f′/EPD. Not with an F/# aperture, which already fixes it.' },
  { id: 'pickup', title: 'Pickup', letter: 'P', value: null, targets: 'any', help: 'target = scale · source + offset, from any other parameter.' },
  { id: 'position', title: 'Position', letter: 'D', value: { key: 'distance', label: 'Distance', unit: 'mm' }, targets: 'thickness', help: 'The axial distance from one vertex to another (surface 1 to the image keeps the total track).' },
  { id: 'center_of_curvature', title: 'Centre of curvature', letter: 'O', value: null, targets: 'power', help: 'Puts the centre of curvature on the vertex of another surface.' },
  { id: 'edge_thickness', title: 'Edge thickness', letter: 'E', value: { key: 'value', label: 'Edge thickness', unit: 'mm' }, targets: 'thickness', help: 'The axial distance between this surface and the next at a radius (empty: from the semi-diameters).' },
]

export const solveKindInfo = (id: string): SolveKindInfo | undefined => SOLVE_KINDS.find((k) => k.id === id)
export const solveKindTitle = (kind: JsonObject): string => solveKindInfo(String(kind.type))?.title ?? String(kind.type)
export const solveLetter = (kind: JsonObject): string => solveKindInfo(String(kind.type))?.letter ?? 'S'

const solvesOf = (s: System): Solve[] => solvesOfParams(s)
export { solveFor, solvesOf }

/** The kind's main number, if it has one. */
export function solveValue(kind: JsonObject): number | null {
  const info = solveKindInfo(String(kind.type))
  if (!info?.value) return null
  const v = kind[info.value.key]
  return typeof v === 'number' ? v : null
}

/** Whether a kind can set `target` (docs/SOLVES.md §3). */
export function kindFits(id: SolveKindId, target: ParamPath): boolean {
  const info = solveKindInfo(id)
  if (!info) return false
  if (info.targets === 'any') return true
  if (info.targets === 'thickness') return target.param === 'thickness'
  if (id === 'center_of_curvature') return target.param === 'curvature'
  return target.param === 'curvature' || target.param === 'paraxial_focal_length'
}

/** A new solve kind `id` on `target`, with sensible numbers. */
export function newSolveKind(id: SolveKindId, s: System, target: ParamPath): JsonObject {
  const i = typeof target.surface === 'number' ? surfaceIndexOfId(s, target.surface) : -1
  const idAt = (k: number) => s.surfaces[Math.min(Math.max(k, 0), imageIndex(s))]?.id ?? 0
  switch (id) {
    case 'marginal_ray_height':
    case 'chief_ray_height':
      return { type: id, height: 0 }
    case 'marginal_ray_angle':
    case 'chief_ray_angle':
      return { type: id, slope: 0 }
    case 'f_number':
      return { type: id, value: 4 }
    case 'pickup': {
      // The same parameter on the surface before, else the one after.
      const k = i > 1 ? i - 1 : i + 1
      const source: ParamPath = typeof target.surface === 'number' ? { ...target, surface: idAt(k) } : { ...target }
      return { type: id, source, scale: 1, offset: 0 }
    }
    case 'position': {
      const t = Number(s.surfaces[i]?.thickness)
      return { type: id, from: idAt(i), to: idAt(i + 1), distance: Number.isFinite(t) ? t : 0 }
    }
    case 'center_of_curvature':
      return { type: id, surface: idAt(i > 0 ? i - 1 : i + 1) }
    case 'edge_thickness':
      return { type: id, value: 1, radius: null }
  }
}

/** Adds the solve of `target`, replacing one that sets the same parameter. */
export function setSolve(s: System, target: ParamPath, kind: JsonObject): void {
  const list = solvesOf(s).filter((x) => !sameParam(s, x.target, target))
  list.push({ target: canonicalParam(s, target), kind })
  s.solves = list as unknown as Json
}

export function removeSolve(s: System, target: ParamPath): void {
  const list = solvesOf(s).filter((x) => !sameParam(s, x.target, target))
  if (list.length) s.solves = list as unknown as Json
  else delete s.solves
}

/** The parameters of a surface a solve can set, with the kinds that fit. */
export function solveTargets(s: System, index: number): { path: ParamPath; title: string; kinds: SolveKindId[] }[] {
  const surf = s.surfaces[index]
  if (!surf || index <= 0 || index >= imageIndex(s)) return []
  const id = surf.id
  const type = surf.shape.type
  const out: { path: ParamPath; title: string; kinds: SolveKindId[] }[] = []
  const add = (path: ParamPath, title: string) => out.push({ path, title, kinds: SOLVE_KINDS.filter((k) => kindFits(k.id, path)).map((k) => k.id) })
  add({ param: 'thickness', surface: id }, 'Thickness')
  if (type === 'paraxial') add({ param: 'paraxial_focal_length', surface: id }, 'Focal length')
  else if (type !== 'coordinate_break') add({ param: 'curvature', surface: id }, type === 'toroid' || type === 'biconic' ? 'Radius y' : 'Radius')
  if (['conic', 'sphere', 'even_asphere', 'odd_asphere', 'q_con', 'xy_polynomial', 'zernike_sag'].includes(type)) add({ param: 'conic', surface: id }, 'Conic')
  if (type === 'toroid' || type === 'biconic') add({ param: 'shape_param', surface: id, parameter: 'curvature_x' }, 'Radius x')
  return out
}

/** One line describing a solve's rule: `Marginal ray height 0 mm`,
 *  `Pickup of R3 × −1 + 0`, `Position S1 → S7 = 120 mm`. */
export function solveSummary(s: System, kind: JsonObject): string {
  const info = solveKindInfo(String(kind.type))
  const title = info?.title ?? String(kind.type)
  const sIdx = (v: Json | undefined) => (typeof v === 'number' ? surfaceIndexOfId(s, v) : -1)
  switch (kind.type) {
    case 'pickup':
      return `Pickup of ${paramTitle(s, kind.source as ParamPath)} × ${Number(kind.scale ?? 1)} + ${Number(kind.offset ?? 0)}`
    case 'position':
      return `Position S${sIdx(kind.from)} → S${sIdx(kind.to)} = ${Number(kind.distance)} mm`
    case 'center_of_curvature':
      return `Centre of curvature on S${sIdx(kind.surface)}`
    default: {
      const v = solveValue(kind)
      return v == null ? title : `${title} ${v}${info?.value?.unit ? ' ' + info.value.unit : ''}`
    }
  }
}

/** Structural problems of a solve that need no optics (the engine's
 *  validation is authoritative). */
export function solveProblem(solve: Solve): string | null {
  const nums = Object.values(solve.kind).filter((v) => typeof v === 'number') as number[]
  if (nums.some((v) => !Number.isFinite(v))) return 'Values must be finite.'
  if (solve.kind.type === 'f_number' && !(Number(solve.kind.value) > 0)) return 'The F/# must be positive.'
  if (solve.kind.type === 'edge_thickness' && solve.kind.radius != null && !(Number(solve.kind.radius) > 0)) return 'The edge radius must be positive.'
  if (solve.kind.type === 'pickup' && paramKey(solve.kind.source) === paramKey(solve.target)) return 'A pickup cannot read its own target.'
  return null
}
