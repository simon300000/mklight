// MOCK ENGINE — read and write one number of the engine's OpticalSystem JSON
// by `ParamPath` (surfaces by stable id). Independent of the UI's model layer.
import type { JsonObject } from '../engine/types'
import { parseNum, type Obj } from './model'

export type Path = { param: string; surface?: number; index?: number; set?: string; parameter?: string; [k: string]: unknown }

const sub = (n: number) => String(n).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)])

export function surfaceOf(system: Obj, p: Path): Obj | null {
  const list: Obj[] = Array.isArray(system.surfaces) ? system.surfaces : []
  return typeof p.surface === 'number' ? (list.find((s) => s.id === p.surface) ?? null) : null
}
export const surfaceIndex = (system: Obj, id: number) => (system.surfaces as Obj[]).findIndex((s) => s.id === id)

const baseShape = (shape: Obj): Obj => (shape.type === 'diffractive' || shape.type === 'sag_overlay' ? ((shape.base as Obj) ?? shape) : shape)

const PLACEMENTS = ['decenter_x', 'decenter_y', 'tilt_x', 'tilt_y', 'tilt_z']

function curvatureKey(shape: Obj): string | null {
  const b = baseShape(shape)
  if (b.type === 'toroid' || b.type === 'biconic') return 'curvature_y'
  return 'curvature' in b ? 'curvature' : null
}

function coefArray(shape: Obj, p: Path, create: boolean): number[] | null {
  const b = baseShape(shape)
  const set = p.param === 'asphere' ? 'asphere' : String(p.set ?? '')
  let holder: Obj = b
  let key = 'coefficients'
  switch (set) {
    case 'phase':
      holder = (shape.phase ?? {}) as Obj
      break
    case 'x_asphere':
    case 'y_asphere':
      key = set
      break
    default:
      if (shape.type === 'diffractive' && !('coefficients' in b)) return null
  }
  if (!Array.isArray(holder[key])) {
    if (!create) return null
    holder[key] = []
  }
  return holder[key] as number[]
}

export function getParamValue(system: Obj, p: Path): number | null {
  const surf = surfaceOf(system, p)
  const i = typeof p.index === 'number' ? p.index : -1
  switch (p.param) {
    case 'curvature': {
      if (!surf) return null
      const k = curvatureKey(surf.shape ?? {})
      if (!k) return baseShape(surf.shape ?? {}).type === 'plane' ? 0 : null
      return Number(baseShape(surf.shape)[k]) || 0
    }
    case 'irregularity':
      return surf && (p.mode === 'astigmatism' || p.mode === 'spherical') ? Number(surf.irregularity?.[p.mode] ?? 0) : null
    case 'element_placement':
      return surf && PLACEMENTS.includes(String(p.component)) ? Number(surf.element_placement?.[String(p.component)] ?? 0) : null
    case 'thickness':
      return surf ? (parseNum(surf.thickness) ?? null) : null
    case 'conic': {
      if (!surf) return null
      const b = baseShape(surf.shape ?? {})
      const k = b.type === 'toroid' || b.type === 'biconic' ? 'conic_y' : 'conic'
      return typeof b[k] === 'number' ? b[k] : null
    }
    case 'asphere':
    case 'shape_coefficient': {
      const a = surf ? coefArray(surf.shape ?? {}, p, false) : null
      return a && i >= 0 ? (a[i] ?? 0) : surf && i >= 0 ? 0 : null
    }
    case 'shape_param': {
      const v = surf?.shape?.[String(p.parameter)]
      return typeof v === 'number' ? v : null
    }
    case 'model_nd':
    case 'model_vd':
    case 'model_dpgf': {
      const m = surf?.material
      return m?.type === 'model' ? Number(m[p.param.slice(6)] ?? 0) : null
    }
    case 'material_dn_d':
    case 'material_dv_d': {
      const m = surf?.material
      return m?.type === 'catalog' ? Number(m.offset?.[p.param.slice(9)] ?? 0) : null
    }
    case 'semi_diameter':
      return surf?.semi_diameter?.mode === 'fixed' ? Number(surf.semi_diameter.value) : null
    case 'aperture_value':
      return typeof system.aperture?.value === 'number' ? system.aperture.value : null
    case 'field_x':
    case 'field_y':
      return system.fields?.fields?.[i]?.[p.param.slice(6)] ?? null
    case 'wavelength_um':
      return system.wavelengths?.wavelengths?.[i]?.um ?? null
    default:
      if (surf && /^(decenter|tilt)_/.test(p.param)) return Number(surf.placement?.[p.param] ?? 0)
      return null
  }
}

/** Sets a parameter in place; false when the system has no such parameter. */
export function setParamValue(system: Obj, p: Path, v: number): boolean {
  const surf = surfaceOf(system, p)
  const i = typeof p.index === 'number' ? p.index : -1
  switch (p.param) {
    case 'curvature': {
      if (!surf) return false
      const k = curvatureKey(surf.shape ?? {})
      if (!k) {
        if (baseShape(surf.shape ?? {}).type !== 'plane') return false
        // A plane given curvature becomes a conic.
        if (v !== 0) surf.shape = { type: 'conic', curvature: v, conic: 0 }
        return true
      }
      baseShape(surf.shape)[k] = v
      return true
    }
    case 'irregularity':
      if (!surf || (p.mode !== 'astigmatism' && p.mode !== 'spherical')) return false
      surf.irregularity = { ...(surf.irregularity ?? {}), [p.mode]: v }
      return true
    case 'element_placement':
      if (!surf || !PLACEMENTS.includes(String(p.component))) return false
      surf.element_placement = { last: p.last ?? surf.id, pivot: p.pivot ?? null, ...(surf.element_placement ?? {}), [String(p.component)]: v }
      return true
    case 'thickness':
      if (!surf) return false
      surf.thickness = v
      return true
    case 'conic': {
      if (!surf) return false
      const b = baseShape(surf.shape ?? {})
      const k = b.type === 'toroid' || b.type === 'biconic' ? 'conic_y' : 'conic'
      if (!(k in b)) return false
      b[k] = v
      return true
    }
    case 'asphere':
    case 'shape_coefficient': {
      if (!surf || i < 0) return false
      const a = coefArray(surf.shape ?? {}, p, true)
      if (!a) return false
      while (a.length <= i) a.push(0)
      a[i] = v
      return true
    }
    case 'shape_param':
      if (!surf || typeof surf.shape?.[String(p.parameter)] !== 'number') return false
      surf.shape[String(p.parameter)] = v
      return true
    case 'model_nd':
    case 'model_vd':
    case 'model_dpgf':
      if (surf?.material?.type !== 'model') return false
      surf.material[p.param.slice(6)] = v
      return true
    case 'material_dn_d':
    case 'material_dv_d':
      if (surf?.material?.type !== 'catalog') return false
      surf.material.offset = { ...(surf.material.offset ?? {}), [p.param.slice(9)]: v }
      return true
    case 'semi_diameter':
      if (!surf) return false
      surf.semi_diameter = { mode: 'fixed', value: v }
      return true
    case 'aperture_value':
      if (!system.aperture) return false
      system.aperture.value = v
      return true
    default:
      if (surf && /^(decenter|tilt)_/.test(p.param)) {
        surf.placement = { ...(surf.placement ?? {}), [p.param]: v }
        return true
      }
      return false
  }
}

export function paramLabel(system: Obj, p: Path): string {
  const idx = typeof p.surface === 'number' ? surfaceIndex(system, p.surface) : -1
  const S = idx >= 0 ? String(idx) : '?'
  const i = typeof p.index === 'number' ? p.index : 0
  switch (p.param) {
    case 'curvature': return `R${S}`
    case 'thickness': return `T${S}`
    case 'conic': return `k${S}`
    case 'asphere': return `A${sub(2 * i + 4)} S${S}`
    case 'shape_coefficient': return `${String(p.set)}[${i}] S${S}`
    case 'model_nd': return `n_d S${S}`
    case 'model_vd': return `V_d S${S}`
    case 'irregularity': return `Irreg. ${String(p.mode)} S${S}`
    case 'element_placement': return `${String(p.component).replace('_', ' ')} S${S}–${typeof p.last === 'number' ? surfaceIndex(system, p.last) : '?'}`
    case 'material_dn_d': return `Δn_d S${S}`
    case 'material_dv_d': return `ΔV_d S${S}`
    case 'semi_diameter': return `SD${S}`
    default: return p.param
  }
}

export const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x))

export const asPath = (v: unknown): Path | null =>
  v && typeof v === 'object' && !Array.isArray(v) && typeof (v as Obj).param === 'string' ? (v as Path) : null

/** Curvatures of the curved optical surfaces plus the back-focus thickness (the default variables). */
export function defaultVariables(system: JsonObject): Path[] {
  const sys = system as Obj
  const surfaces: Obj[] = sys.surfaces ?? []
  const out: Path[] = []
  surfaces.forEach((s, i) => {
    if (i === 0 || i === surfaces.length - 1 || s.interaction === 'dummy') return
    const c = getParamValue(sys, { param: 'curvature', surface: s.id })
    if (c !== null && c !== 0) out.push({ param: 'curvature', surface: s.id })
  })
  if (surfaces.length >= 3) out.push({ param: 'thickness', surface: surfaces[surfaces.length - 2].id })
  return out
}
