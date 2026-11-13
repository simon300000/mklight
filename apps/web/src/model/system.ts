// The optical system as the engine serialises it (`OpticalSystem` of
// mklight-core). The UI edits it as plain JSON values and never computes optics;
// these helpers are pure model manipulation, like the Swift `MKLightModel`.

import type { Json, JsonObject } from '@/engine/types'
import { nf } from './format'
import { applyMaterial, offsetCompactText, offsetOf, parseMaterial as parseMaterialText, type MaterialEdit as ParsedMaterial } from './materials'

export interface Shape extends JsonObject {
  type: string
}
export interface Material extends JsonObject {
  type: string
}
export interface Surface extends JsonObject {
  id: number
  label: string
  shape: Shape
  thickness: number | string
  material: Material
  interaction: string
  semi_diameter: { mode: string; value?: number }
}
export interface FieldPoint extends JsonObject {
  x: number
  y: number
  weight: number
}
export interface System extends JsonObject {
  name: string
  notes: string
  aperture: { type: string; value?: number }
  fields: { kind: string; fields: FieldPoint[] }
  wavelengths: { wavelengths: { um: number; weight: number }[]; primary: number }
  surfaces: Surface[]
  stop: number
}

export const clone = <T>(v: T): T => structuredClone(v)

// MARK: Aperture and fields

export const APERTURE_TYPES: { type: string; title: string; unit: string }[] = [
  { type: 'entrance_pupil_diameter', title: 'Entrance pupil diameter', unit: 'mm' },
  { type: 'image_space_f_number', title: 'Image-space F/#', unit: '' },
  { type: 'object_space_na', title: 'Object-space NA', unit: '' },
  { type: 'paraxial_working_f_number', title: 'Paraxial working F/#', unit: '' },
  { type: 'float_by_stop_size', title: 'Float by stop size', unit: '' },
]

export const FIELD_KINDS: { kind: string; title: string; unit: string }[] = [
  { kind: 'object_angle', title: 'Object angle', unit: '°' },
  { kind: 'object_height', title: 'Object height', unit: 'mm' },
  { kind: 'paraxial_image_height', title: 'Paraxial image height', unit: 'mm' },
  { kind: 'real_image_height', title: 'Real image height', unit: 'mm' },
]

export const fieldUnit = (kind: string) => FIELD_KINDS.find((k) => k.kind === kind)?.unit ?? ''

export const WAVELENGTH_PRESETS: { name: string; list: [number, number][]; primary: number }[] = [
  { name: 'F, d, C (visible)', list: [[0.4861327, 1], [0.5875618, 1], [0.6562725, 1]], primary: 1 },
  { name: 'F′, e, C′ (visible)', list: [[0.4799914, 1], [0.546074, 1], [0.6438469, 1]], primary: 1 },
  { name: 'g, F, e, d, C', list: [[0.4358343, 1], [0.4861327, 1], [0.546074, 1], [0.5875618, 1], [0.6562725, 1]], primary: 3 },
  { name: 'd only', list: [[0.5875618, 1]], primary: 0 },
  { name: 'e only', list: [[0.546074, 1]], primary: 0 },
  { name: 'Photopic (5)', list: [[0.47, 0.091], [0.51, 0.503], [0.555, 1], [0.61, 0.503], [0.65, 0.107]], primary: 2 },
  { name: 'Near infrared', list: [[0.75, 1], [0.85, 1], [0.95, 1]], primary: 1 },
]

// MARK: Surfaces

export const imageIndex = (s: System) => s.surfaces.length - 1
export const objectAtInfinity = (s: System) => !Number.isFinite(nf(s.surfaces[0]?.thickness))

/** Shapes whose first section is described by `curvature` (or `curvature_y`). */
export function curvatureOf(shape: Shape): number | null {
  switch (shape.type) {
    case 'plane':
    case 'coordinate_break':
    case 'paraxial':
      return shape.type === 'plane' ? 0 : null
    case 'toroid':
    case 'biconic':
      return Number(shape.curvature_y ?? 0)
    case 'diffractive':
      return curvatureOf(shape.base as Shape)
    default:
      return typeof shape.curvature === 'number' ? shape.curvature : null
  }
}

export function conicOf(shape: Shape): number | null {
  switch (shape.type) {
    case 'toroid':
    case 'biconic':
      return Number(shape.conic_y ?? 0)
    case 'diffractive':
      return conicOf(shape.base as Shape)
    case 'q_bfs':
    case 'plane':
    case 'coordinate_break':
    case 'paraxial':
      return null
    default:
      return typeof shape.conic === 'number' ? shape.conic : null
  }
}

export const radiusFromCurvature = (c: number) => (c === 0 ? Infinity : 1 / c)
export const curvatureFromRadius = (r: number) => (!Number.isFinite(r) || r === 0 ? 0 : 1 / r)

/** Sets the (y-section) curvature; a plane becomes a conic when curved. */
export function setCurvature(shape: Shape, c: number): Shape {
  switch (shape.type) {
    case 'plane':
      return c === 0 ? shape : { type: 'conic', curvature: c, conic: 0 }
    case 'toroid':
    case 'biconic':
      return { ...shape, curvature_y: c }
    case 'diffractive':
      return { ...shape, base: setCurvature(shape.base as Shape, c) }
    case 'coordinate_break':
    case 'paraxial':
      return shape
    default:
      return { ...shape, curvature: c }
  }
}

/** The x-section curvature of anamorphic shapes. */
export function curvatureXOf(shape: Shape): number | null {
  return shape.type === 'toroid' || shape.type === 'biconic' ? Number(shape.curvature_x ?? 0) : null
}

export function setConic(shape: Shape, k: number): Shape {
  switch (shape.type) {
    case 'plane':
      return k === 0 ? shape : { type: 'conic', curvature: 0, conic: k }
    case 'toroid':
    case 'biconic':
      return { ...shape, conic_y: k }
    case 'diffractive':
      return { ...shape, base: setConic(shape.base as Shape, k) }
    case 'q_bfs':
    case 'coordinate_break':
    case 'paraxial':
      return shape
    default:
      return { ...shape, conic: k }
  }
}

export function nextSurfaceId(s: System): number {
  return s.surfaces.reduce((m, x) => Math.max(m, x.id), -1) + 1
}

export function newSurface(id: number, thickness = 0): Surface {
  return {
    id,
    label: '',
    shape: { type: 'plane' },
    thickness,
    material: { type: 'air' },
    interaction: 'refract',
    semi_diameter: { mode: 'auto' },
    aperture: null,
    placement: null,
    coating: null,
    scatter: null,
    diffraction_efficiency: { type: 'ideal' },
    edge: { mechanical_semi_diameter: null, edge_scatter: null },
  }
}

const surfaceIdsOf = (j: Json): number[] => {
  if (j && typeof j === 'object' && !Array.isArray(j)) {
    const o = j as JsonObject
    const out: number[] = []
    for (const k of ['surface', 'surface_id']) if (typeof o[k] === 'number') out.push(o[k] as number)
    for (const k of ['surfaces']) if (Array.isArray(o[k])) for (const x of o[k] as Json[]) if (typeof x === 'number') out.push(x)
    return out
  }
  return []
}

export function insertSurface(s: System, index: number): number {
  const i = Math.min(Math.max(index, 1), imageIndex(s))
  const id = nextSurfaceId(s)
  s.surfaces.splice(i, 0, newSurface(id))
  if (i <= s.stop) s.stop += 1
  return i
}

export function deleteSurface(s: System, index: number): boolean {
  if (index <= 0 || index >= imageIndex(s) || s.surfaces.length <= 3) return false
  const [removed] = s.surfaces.splice(index, 1)
  if (index < s.stop) s.stop -= 1
  s.stop = Math.min(Math.max(s.stop, 1), imageIndex(s) - 1)
  for (const key of ['mechanics', 'solves'] as const) {
    if (Array.isArray(s[key])) s[key] = (s[key] as Json[]).filter((m) => !surfaceIdsOf(m).includes(removed.id))
  }
  return true
}

export function canDelete(s: System, index: number): boolean {
  return index > 0 && index < imageIndex(s) && s.surfaces.length > 3
}

// MARK: Materials

export function materialText(surface: Surface): string {
  if (surface.interaction === 'reflect') return 'MIRROR'
  if (surface.interaction === 'dummy') return 'DUMMY'
  if (surface.shape?.type === 'coordinate_break') return ''
  const m = surface.material
  switch (m.type) {
    case 'air':
      return ''
    case 'vacuum':
      return 'VACUUM'
    case 'catalog':
      return String(m.name ?? '')
    case 'model':
      return `${Number(m.nd).toFixed(5)}/${Number(m.vd).toFixed(2)}`
    case 'constant':
      return `n=${sigShort(Number(m.n))}`
    default:
      return m.type
  }
}

function sigShort(v: number) {
  return Number.isFinite(v) ? String(Number(v.toPrecision(6))) : '?'
}

export type MaterialEdit = ParsedMaterial

/** Parses the material column (see `materials.ts`). */
export const parseMaterial = parseMaterialText

/** Applies a parsed material; retyping the same glass keeps its melt offset,
 *  another glass starts without. */
export const setMaterial = applyMaterial

/** Melt offset suffix of a catalogue glass (`+0.0012/−0.8`), or ''. */
export function meltOffsetText(m: Material): string {
  return offsetCompactText(offsetOf(m))
}

// MARK: New design

export function newSystem(): System {
  const s: System = {
    name: 'Untitled',
    notes: '',
    aperture: { type: 'entrance_pupil_diameter', value: 10 },
    fields: {
      kind: 'object_angle',
      fields: [0, 5, 7].map((y) => ({ x: 0, y, weight: 1, vignetting: { vdx: 0, vdy: 0, vcx: 0, vcy: 0 } })),
    },
    wavelengths: {
      wavelengths: [
        { um: 0.4861327, weight: 1 },
        { um: 0.5875618, weight: 1 },
        { um: 0.6562725, weight: 1 },
      ],
      primary: 1,
    },
    surfaces: [],
    stop: 1,
    ray_aiming: { mode: 'off' },
    environment: { temperature_c: 20, pressure_atm: 1 },
    coatings: {},
    mechanics: [],
    sensor: null,
    configurations: [],
  }
  const obj = newSurface(0, 'inf' as unknown as number)
  obj.thickness = 'inf'
  obj.label = 'OBJ'
  const a = newSurface(1, 4)
  a.shape = { type: 'conic', curvature: 1 / 50, conic: 0 }
  a.material = { type: 'catalog', catalog: null, name: 'N-BK7' }
  const b = newSurface(2, 95)
  b.shape = { type: 'conic', curvature: -1 / 50, conic: 0 }
  const img = newSurface(3, 0)
  img.label = 'IMA'
  s.surfaces = [obj, a, b, img]
  return s
}

export function surfaceTitle(s: System, i: number): string {
  if (i === 0) return 'Object'
  if (i === imageIndex(s)) return 'Image'
  return `Surface ${i}`
}
