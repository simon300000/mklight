// Parameter paths (the engine's `ParamPath`, apps/macos/Sources/MKLightModel/ParamPath.swift):
// how variables, solves, tolerances and pickups address one number of the
// system. Surfaces are addressed by their stable id, never by index.

import type { Json, JsonObject } from '@/engine/types'
import { nf } from './format'
import { curvatureOf, type Shape, type System } from './system'
import { sub } from './surfaceTypes'

export interface ParamPath extends JsonObject {
  param: string
}

/** A design variable (the project's `variables` section, the engine's `VariableSpec`). */
export interface VariableSpec extends JsonObject {
  param: ParamPath
  lower: number | null
  upper: number | null
}

export const SURFACE_TAGS = [
  'curvature', 'thickness', 'conic', 'paraxial_focal_length', 'model_nd', 'model_vd', 'model_dpgf',
  'material_dn_d', 'material_dv_d', 'decenter_x', 'decenter_y', 'tilt_x', 'tilt_y', 'tilt_z', 'semi_diameter',
] as const
export const INDEX_TAGS = ['field_x', 'field_y', 'field_weight', 'wavelength_um', 'wavelength_weight'] as const

/** A stable string identity of a path (sorted keys). */
export function paramKey(p: Json): string {
  if (!p || typeof p !== 'object' || Array.isArray(p)) return String(p)
  const o = p as JsonObject
  return Object.keys(o)
    .sort()
    .map((k) => `${k}=${JSON.stringify(o[k])}`)
    .join('&')
}

/** `curvature` and `shape_param curvature_y` name one parameter of a toroid or biconic. */
export function canonicalParam(system: System, p: ParamPath): ParamPath {
  if (p.param === 'shape_param' && p.parameter === 'curvature_y') return { param: 'curvature', surface: p.surface as number }
  void system
  return p
}

export const sameParam = (s: System, a: ParamPath, b: ParamPath) => paramKey(canonicalParam(s, a)) === paramKey(canonicalParam(s, b))

export const surfaceIndexOfId = (s: System, id: number) => s.surfaces.findIndex((x) => x.id === id)

const PLACEMENT = new Set(['decenter_x', 'decenter_y', 'tilt_x', 'tilt_y', 'tilt_z'])

function coefficientArray(shape: Shape, set: string): number[] | null {
  const base = shape.type === 'diffractive' ? (shape.base as Shape) : shape
  switch (set) {
    case 'odd':
    case 'q':
    case 'xy':
    case 'zernike':
      return (shape.coefficients as number[]) ?? []
    case 'phase':
      return ((shape.phase as JsonObject | undefined)?.coefficients as number[]) ?? []
    case 'x_asphere':
    case 'y_asphere':
      return (shape[set] as number[]) ?? []
    default:
      return (base.coefficients as number[]) ?? null
  }
}

/** The current value of a parameter, or null when it does not resolve. */
export function getParam(s: System, p: ParamPath): number | null {
  const idx = typeof p.surface === 'number' ? surfaceIndexOfId(s, p.surface) : -1
  const surf = idx >= 0 ? s.surfaces[idx] : null
  const i = typeof p.index === 'number' ? p.index : -1
  switch (p.param) {
    case 'curvature':
      return surf ? curvatureOf(surf.shape) : null
    case 'thickness':
      return surf ? nf(surf.thickness) : null
    case 'conic': {
      if (!surf) return null
      const b = surf.shape.type === 'diffractive' ? (surf.shape.base as Shape) : surf.shape
      return typeof b.conic === 'number' ? b.conic : null
    }
    case 'asphere': {
      const a = surf ? coefficientArray(surf.shape, 'asphere') : null
      return a && i >= 0 ? (a[i] ?? 0) : null
    }
    case 'shape_coefficient': {
      const a = surf ? coefficientArray(surf.shape, String(p.set)) : null
      return a && i >= 0 ? (a[i] ?? 0) : null
    }
    case 'shape_param': {
      if (!surf) return null
      const k = String(p.parameter)
      if (k === 'norm_radius' && surf.shape.type === 'diffractive') return Number((surf.shape.phase as JsonObject).norm_radius)
      const v = surf.shape[k]
      return typeof v === 'number' ? v : null
    }
    case 'paraxial_focal_length':
      return surf && typeof surf.shape.focal_length === 'number' ? surf.shape.focal_length : null
    case 'model_nd':
    case 'model_vd':
    case 'model_dpgf': {
      const m = surf?.material
      if (!m || m.type !== 'model') return null
      return Number(m[p.param.slice(6)] ?? 0)
    }
    case 'material_dn_d':
    case 'material_dv_d': {
      const m = surf?.material
      if (!m || m.type !== 'catalog') return null
      return Number(((m.offset as JsonObject | undefined) ?? {})[p.param.slice(9)] ?? 0)
    }
    case 'semi_diameter':
      return surf?.semi_diameter.mode === 'fixed' ? Number(surf.semi_diameter.value) : null
    case 'aperture_value':
      return s.aperture.value ?? null
    case 'temperature_c':
    case 'pressure_atm':
      return Number((s.environment as JsonObject)?.[p.param] ?? NaN)
    case 'field_x':
    case 'field_y':
      return s.fields.fields[i]?.[p.param.slice(6) as 'x' | 'y'] ?? null
    case 'field_weight':
      return s.fields.fields[i]?.weight ?? null
    case 'wavelength_um':
      return s.wavelengths.wavelengths[i]?.um ?? null
    case 'wavelength_weight':
      return s.wavelengths.wavelengths[i]?.weight ?? null
    default:
      if (PLACEMENT.has(p.param) && surf) return Number((surf.placement as JsonObject | null)?.[p.param] ?? 0)
      return null
  }
}

/** A short title: `R3`, `T5`, `k2`, `A₄ S3`, `Δn_d S2`, `EPD`, `Field 2 y`. */
export function paramTitle(s: System, p: ParamPath): string {
  const idx = typeof p.surface === 'number' ? surfaceIndexOfId(s, p.surface) : -1
  const S = idx >= 0 ? String(idx) : '?'
  const i = typeof p.index === 'number' ? p.index : 0
  switch (p.param) {
    case 'curvature': return `R${S}`
    case 'thickness': return `T${S}`
    case 'conic': return `k${S}`
    case 'asphere': return `A${sub(2 * i + 4)} S${S}`
    case 'shape_coefficient': return `${String(p.set)}[${i}] S${S}`
    case 'shape_param': return `${String(p.parameter).replace(/_/g, ' ')} S${S}`
    case 'paraxial_focal_length': return `f S${S}`
    case 'model_nd': return `n_d S${S}`
    case 'model_vd': return `V_d S${S}`
    case 'model_dpgf': return `ΔP_g,F S${S}`
    case 'material_dn_d': return `Δn_d S${S}`
    case 'material_dv_d': return `ΔV_d S${S}`
    case 'semi_diameter': return `SD${S}`
    case 'aperture_value': return 'Aperture'
    case 'temperature_c': return 'Temperature'
    case 'pressure_atm': return 'Pressure'
    case 'field_x': return `Field ${i + 1} x`
    case 'field_y': return `Field ${i + 1} y`
    case 'field_weight': return `Field ${i + 1} weight`
    case 'wavelength_um': return `λ${i + 1}`
    case 'wavelength_weight': return `λ${i + 1} weight`
    default:
      return PLACEMENT.has(p.param) ? `${p.param.replace('_', ' ')} S${S}` : p.param
  }
}

/** The unit of a parameter's model value (curvatures in 1/mm). */
export function paramUnit(p: ParamPath): string {
  switch (p.param) {
    case 'curvature': return '1/mm'
    case 'thickness': case 'semi_diameter': case 'paraxial_focal_length': case 'decenter_x': case 'decenter_y': return 'mm'
    case 'tilt_x': case 'tilt_y': case 'tilt_z': return '°'
    case 'wavelength_um': return 'µm'
    case 'asphere': return `mm${'⁻³'}`
    default: return '–'
  }
}

export function variablesOf(project: { variables: Json }): VariableSpec[] {
  return Array.isArray(project.variables) ? (project.variables as VariableSpec[]) : []
}

/** Solves of the system (`system.solves`: [{target: ParamPath, kind: {...}}]). */
export function solvesOf(s: System): { target: ParamPath; kind: JsonObject }[] {
  return Array.isArray(s.solves) ? (s.solves as { target: ParamPath; kind: JsonObject }[]) : []
}

export function solveFor(s: System, p: ParamPath) {
  return solvesOf(s).find((x) => sameParam(s, x.target, p)) ?? null
}
