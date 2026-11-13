// Surface types described as data (docs/APP.md "Progressive disclosure",
// apps/macos/Sources/MKLightUI/ViewModels/SurfaceTypes.swift): sections of
// parameters addressed by their key path in the shape JSON, and coefficient
// lists with term labels. The inspector draws any type from its descriptor.

import type { Json, JsonObject } from '@/engine/types'
import type { Shape, Surface, System } from './system'
import { conicOf, curvatureOf } from './system'

export type Group = 'Spherical' | 'Aspheric' | 'Freeform' | 'Anamorphic' | 'Diffractive' | 'Special'
export const GROUPS: Group[] = ['Spherical', 'Aspheric', 'Freeform', 'Anamorphic', 'Diffractive', 'Special']

export interface ShapeParam {
  key: string
  title: string
  unit: string
  /** Shown as a radius of curvature (1/value), edited as a radius. */
  radius?: boolean
  integer?: boolean
  help?: string
  options?: { value: string; title: string }[]
}

export interface CoefficientList {
  key: string
  title: string
  max: (shape: Shape) => number
  term: (k: number, shape: Shape) => string
  meaning?: (k: number, shape: Shape) => string
  unit?: (k: number) => string
}

export interface ParamSection {
  title: string
  params: ShapeParam[]
  coefficients?: CoefficientList
  footer?: string
}

export interface SurfaceTypeDescriptor {
  id: string
  title: string
  group: Group
  summary: string
  sections: ParamSection[]
}

const SUB = '₀₁₂₃₄₅₆₇₈₉'
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹'
export const sub = (n: number) => (n < 0 ? '₋' : '') + [...String(Math.abs(n))].map((d) => SUB[+d]).join('')
export const sup = (n: number) => (n < 0 ? '⁻' : '') + [...String(Math.abs(n))].map((d) => SUP[+d]).join('')
const mmPower = (p: number) => (p === 0 ? '' : p === 1 ? 'mm' : 'mm' + sup(p))

const radius = (key = 'curvature', title = 'Radius'): ShapeParam => ({ key, title, unit: 'mm', radius: true })
const conic = (key = 'conic', title = 'Conic'): ShapeParam => ({
  key,
  title,
  unit: '',
  help: 'Conic constant k: 0 sphere, −1 paraboloid, < −1 hyperboloid.',
})
const normRadius = (key = 'norm_radius'): ShapeParam => ({
  key,
  title: 'Norm radius',
  unit: 'mm',
  help: 'Normalisation radius of the polynomial.',
})

const axisList = (axis: string, key: string, letter: string): CoefficientList => ({
  key,
  title: `${axis.toUpperCase()} Polynomial`,
  max: () => 10,
  term: (k) => `${letter}${sub(2 * k + 4)}`,
  meaning: (k) => axis + sup(2 * k + 4),
  unit: (k) => mmPower(-(2 * k + 3)),
})

/** XY term `X^m Y^n` at zero-based index (SURFACES §5). */
export function xyTerm(index: number): { m: number; n: number } {
  let d = 1
  while (((d + 1) * (d + 2)) / 2 - 1 <= index) d += 1
  const n = index - ((d * (d + 1)) / 2 - 1)
  return { m: d - n, n }
}
const monomial = ({ m, n }: { m: number; n: number }) => {
  const p = (v: string, k: number) => (k === 0 ? '' : k === 1 ? v : v + sup(k))
  return p('x', m) + p('y', n)
}

/** Zernike term (n, m) of 1-based j (SURFACES §6). */
export function zernikeTerm(j: number, ordering: string): { n: number; m: number } {
  if (ordering === 'fringe') {
    if (j === 37) return { n: 12, m: 0 }
    const k = Math.floor(Math.sqrt(j - 1))
    const o = j - 1 - k * k
    const am = k - Math.floor(o / 2)
    return { n: 2 * k - am, m: o % 2 === 0 ? am : -am }
  }
  let nn = 0
  while (((nn + 1) * (nn + 2)) / 2 < j) nn += 1
  const q = j - (nn * (nn + 1)) / 2 - 1
  const am = nn % 2 === 0 ? 2 * Math.floor((q + 1) / 2) : 2 * Math.floor(q / 2) + 1
  return { n: nn, m: am === 0 ? 0 : j % 2 === 0 ? am : -am }
}

export function zernikeName(t: { n: number; m: number }): string {
  const m = Math.abs(t.m)
  const axis = t.m > 0 ? ' x' : t.m < 0 ? ' y' : ''
  const names: Record<string, string> = {
    '0,0': 'piston', '1,1': 'tilt', '2,0': 'defocus', '2,2': 'astigmatism', '3,1': 'coma', '3,3': 'trefoil',
    '4,0': 'spherical', '4,2': '2nd astigmatism', '4,4': 'tetrafoil', '5,1': '2nd coma', '5,3': '2nd trefoil',
    '5,5': 'pentafoil', '6,0': '2nd spherical', '8,0': '3rd spherical',
  }
  const name = names[`${t.n},${m}`]
  if (name) return name + (m === 0 ? '' : t.n === 1 || t.n === 3 || t.n === 5 ? axis : t.m > 0 ? ' 0°' : ' 45°')
  return `n${t.n} m${t.m}`
}

export const SURFACE_TYPES: SurfaceTypeDescriptor[] = [
  { id: 'plane', title: 'Plane', group: 'Spherical', summary: 'A flat surface.', sections: [] },
  {
    id: 'conic',
    title: 'Sphere / Conic',
    group: 'Spherical',
    summary: 'Sphere or conic of revolution.',
    sections: [{ title: 'Shape', params: [radius(), conic()] }],
  },
  {
    id: 'even_asphere',
    title: 'Even Asphere',
    group: 'Aspheric',
    summary: 'Conic plus even polynomial terms A₄r⁴ + A₆r⁶ + …',
    sections: [
      { title: 'Base Conic', params: [radius(), conic()] },
      {
        title: 'Polynomial',
        params: [],
        coefficients: {
          key: 'coefficients',
          title: 'Coefficients',
          max: () => 10,
          term: (k) => `A${sub(2 * k + 4)}`,
          meaning: (k) => 'r' + sup(2 * k + 4),
          unit: (k) => mmPower(-(2 * k + 3)),
        },
      },
    ],
  },
  {
    id: 'odd_asphere',
    title: 'Odd Asphere',
    group: 'Aspheric',
    summary: 'Conic plus all powers of r: β₁r + β₂r² + β₃r³ + … A nonzero β₁ is a cone at the vertex.',
    sections: [
      { title: 'Base Conic', params: [radius(), conic()] },
      {
        title: 'Polynomial',
        params: [],
        coefficients: {
          key: 'coefficients',
          title: 'Coefficients',
          max: () => 20,
          term: (k) => `β${sub(k + 1)}`,
          meaning: (k) => 'r' + (k === 0 ? '' : sup(k + 1)),
          unit: (k) => mmPower(-k),
        },
      },
    ],
  },
  {
    id: 'q_con',
    title: 'Q-Type (Conic)',
    group: 'Aspheric',
    summary: 'Forbes conic-based asphere: conic plus u⁴ Σ aₘ Qₘᶜᵒⁿ(u²), u = r/r_norm. Keeps the vertex curvature.',
    sections: [
      { title: 'Base Conic', params: [radius(), conic(), normRadius()] },
      {
        title: 'Q Polynomial',
        params: [],
        coefficients: {
          key: 'coefficients',
          title: 'Coefficients',
          max: () => 20,
          term: (k) => `a${sub(k)}`,
          meaning: (k) => `u⁴Q${sub(k)}ᶜᵒⁿ`,
          unit: () => 'mm',
        },
      },
    ],
  },
  {
    id: 'q_bfs',
    title: 'Q-Type (Best-Fit Sphere)',
    group: 'Aspheric',
    summary: 'Forbes slope-orthogonal departure from a best-fit sphere: u²(1−u²) Σ aₘ Qₘᵇᶠˢ(u²) / √(1−c²r²).',
    sections: [
      { title: 'Best-Fit Sphere', params: [radius(), normRadius()] },
      {
        title: 'Q Polynomial',
        params: [],
        coefficients: {
          key: 'coefficients',
          title: 'Coefficients',
          max: () => 20,
          term: (k) => `a${sub(k)}`,
          meaning: (k) => `Q${sub(k)}ᵇᶠˢ`,
          unit: () => 'mm',
        },
      },
    ],
  },
  {
    id: 'xy_polynomial',
    title: 'XY Polynomial',
    group: 'Freeform',
    summary: 'Freeform: conic plus Σ C(xᵐyⁿ) Xᵐ Yⁿ with X = x/r_norm, Y = y/r_norm; degrees 1 to 10.',
    sections: [
      { title: 'Base Conic', params: [radius(), conic(), normRadius()] },
      {
        title: 'XY Terms',
        params: [],
        coefficients: {
          key: 'coefficients',
          title: 'Coefficients',
          max: () => 65,
          term: (k) => `C(${monomial(xyTerm(k))})`,
          unit: () => 'mm',
        },
      },
    ],
  },
  {
    id: 'zernike_sag',
    title: 'Zernike Sag',
    group: 'Freeform',
    summary: 'Freeform: conic plus Σ cⱼ Zⱼ(ρ, θ), ρ = r/r_norm, θ from +x. Sag coefficients in mm, piston and tilt kept.',
    sections: [
      {
        title: 'Base Conic',
        params: [
          radius(),
          conic(),
          normRadius(),
          {
            key: 'ordering',
            title: 'Ordering',
            unit: '',
            options: [
              { value: 'fringe', title: 'Fringe (37)' },
              { value: 'standard', title: 'Standard (45)' },
            ],
          },
        ],
      },
      {
        title: 'Zernike Terms',
        params: [],
        coefficients: {
          key: 'coefficients',
          title: 'Coefficients',
          max: (s) => (s.ordering === 'standard' ? 45 : 37),
          term: (k) => `Z${sub(k + 1)}`,
          meaning: (k, s) => zernikeName(zernikeTerm(k + 1, String(s.ordering ?? 'fringe'))),
          unit: () => 'mm',
        },
      },
    ],
  },
  {
    id: 'toroid',
    title: 'Toroid',
    group: 'Anamorphic',
    summary: 'The y profile (conic plus even terms) rotated about an axis parallel to y at the distance R_x.',
    sections: [
      {
        title: 'Y Profile',
        params: [radius('curvature_y', 'Radius y'), conic('conic_y', 'Conic y')],
        coefficients: axisList('y', 'y_asphere', 'B'),
      },
      {
        title: 'Rotation',
        params: [radius('curvature_x', 'Radius x')],
        footer: 'Radius x is the distance from the vertex to the axis of rotation; ∞ makes a y cylinder.',
      },
    ],
  },
  {
    id: 'biconic',
    title: 'Biconic',
    group: 'Anamorphic',
    summary: 'Independent conics in x and y with even axis polynomials. A cylinder is a biconic with one plane axis.',
    sections: [
      {
        title: 'X Section',
        params: [radius('curvature_x', 'Radius x'), conic('conic_x', 'Conic x')],
        coefficients: axisList('x', 'x_asphere', 'A'),
      },
      {
        title: 'Y Section',
        params: [radius('curvature_y', 'Radius y'), conic('conic_y', 'Conic y')],
        coefficients: axisList('y', 'y_asphere', 'B'),
      },
    ],
  },
  {
    id: 'diffractive',
    title: 'Diffractive',
    group: 'Diffractive',
    summary: 'A fixed-period phase Φ = Σ Aᵢ ρ²ⁱ (order +1, radians) on a plane, conic or even-asphere base.',
    sections: [
      { title: 'Base Shape', params: [radius('base.curvature'), conic('base.conic')] },
      {
        title: 'Base Polynomial',
        params: [],
        coefficients: {
          key: 'base.coefficients',
          title: 'Coefficients',
          max: () => 10,
          term: (k) => `A${sub(2 * k + 4)}`,
          meaning: (k) => 'r' + sup(2 * k + 4),
        },
      },
      {
        title: 'Phase',
        params: [normRadius('phase.norm_radius')],
        coefficients: {
          key: 'phase.coefficients',
          title: 'Coefficients',
          max: () => 10,
          term: (k) => `A${sub(k + 1)}`,
          meaning: (k) => 'ρ' + sup(2 * (k + 1)),
          unit: () => 'rad',
        },
      },
      {
        title: 'Orders',
        params: [
          { key: 'phase.design_order', title: 'Design order', unit: '', integer: true },
          { key: 'phase.design_wavelength_um', title: 'Design λ', unit: 'µm' },
          { key: 'phase.trace_order', title: 'Trace order', unit: '', integer: true },
        ],
      },
    ],
  },
  {
    id: 'paraxial',
    title: 'Ideal Lens',
    group: 'Special',
    summary: 'A thin, aberration-free lens of the given focal length.',
    sections: [
      {
        title: 'Ideal Lens',
        params: [{ key: 'focal_length', title: 'Focal length', unit: 'mm', help: 'Focal length of the ideal lens.' }],
      },
    ],
  },
  {
    id: 'coordinate_break',
    title: 'Coordinate Break',
    group: 'Special',
    summary: 'Changes the coordinate frame only: decenters, then tilts (or the reverse).',
    sections: [],
  },
]

export const descriptorFor = (type: string) => SURFACE_TYPES.find((d) => d.id === type)

/** `sphere` is the engine's legacy alias of a conic with k = 0. */
export const typeTitle = (type: string) => descriptorFor(type === 'sphere' ? 'conic' : type)?.title ?? type

/** Short type label for the table's type column. */
export const TYPE_SHORT: Record<string, string> = {
  plane: 'Plane', conic: 'Conic', sphere: 'Sphere', even_asphere: 'Even Asph', odd_asphere: 'Odd Asph',
  q_con: 'Q-con', q_bfs: 'Q-bfs', xy_polynomial: 'XY Poly', zernike_sag: 'Zernike', toroid: 'Toroid',
  biconic: 'Biconic', diffractive: 'DOE', paraxial: 'Ideal', coordinate_break: 'Coord Brk',
}

// MARK: Key paths

export function getPath(obj: Json, path: string): Json | undefined {
  let cur: Json | undefined = obj
  for (const k of path.split('.')) {
    if (cur && typeof cur === 'object' && !Array.isArray(cur)) cur = (cur as JsonObject)[k]
    else return undefined
  }
  return cur
}

/** Returns a copy of `obj` with `path` set to `value`. */
export function setPath<T extends Json>(obj: T, path: string, value: Json): T {
  const [head, ...rest] = path.split('.')
  const o = { ...(obj as JsonObject) }
  o[head] = rest.length ? setPath((o[head] ?? {}) as JsonObject, rest.join('.'), value) : value
  return o as T
}

// MARK: Changing the type (keeps what carries over)

export function changeShapeType(system: System, index: number, type: string) {
  const surface: Surface = system.surfaces[index]
  const old = surface.shape
  if (old.type === type && type !== 'coordinate_break') return
  const source = old.type === 'diffractive' ? (old.base as Shape) : old
  const c = curvatureOf(source) ?? 0
  const k = conicOf(source) ?? 0
  const sd = surface.semi_diameter.mode === 'fixed' ? surface.semi_diameter.value : undefined
  const r = [old.norm_radius, (old.phase as JsonObject | undefined)?.norm_radius, sd].find(
    (v) => typeof v === 'number' && v > 0,
  ) as number | undefined ?? 10
  const asph = source.type === 'even_asphere' ? (source.coefficients as number[]) : []
  const primaryUm = system.wavelengths.wavelengths[system.wavelengths.primary]?.um ?? 0.5875618
  let shape: Shape
  switch (type) {
    case 'plane': shape = { type }; break
    case 'conic': shape = { type, curvature: c, conic: k }; break
    case 'even_asphere': shape = { type, curvature: c, conic: k, coefficients: asph }; break
    case 'odd_asphere': shape = { type, curvature: c, conic: k, coefficients: [] }; break
    case 'q_con': shape = { type, curvature: c, conic: k, norm_radius: r, coefficients: [] }; break
    case 'q_bfs': shape = { type, curvature: c, norm_radius: r, coefficients: [] }; break
    case 'xy_polynomial': shape = { type, curvature: c, conic: k, norm_radius: r, coefficients: [] }; break
    case 'zernike_sag': shape = { type, curvature: c, conic: k, norm_radius: r, ordering: 'fringe', coefficients: [] }; break
    case 'diffractive': {
      const base = ['plane', 'conic', 'even_asphere'].includes(source.type) ? source : { type: 'conic', curvature: c, conic: k }
      shape = {
        type,
        base,
        phase: { norm_radius: r, coefficients: [], design_order: 1, design_wavelength_um: primaryUm, trace_order: 1 },
      }
      break
    }
    case 'toroid':
      shape = source.type === 'biconic'
        ? { type, curvature_x: source.curvature_x, curvature_y: source.curvature_y, conic_y: source.conic_y, y_asphere: source.y_asphere }
        : { type, curvature_x: c, curvature_y: c, conic_y: k, y_asphere: [] }
      break
    case 'biconic':
      shape = source.type === 'toroid'
        ? { type, curvature_x: source.curvature_x, curvature_y: source.curvature_y, conic_x: 0, conic_y: source.conic_y, x_asphere: [], y_asphere: source.y_asphere }
        : { type, curvature_x: c, curvature_y: c, conic_x: k, conic_y: k, x_asphere: [], y_asphere: [] }
      break
    case 'paraxial':
      shape = { type, focal_length: old.type === 'paraxial' ? old.focal_length : 100 }
      break
    case 'coordinate_break':
      shape = { type }
      if (!surface.placement) {
        surface.placement = { decenter_x: 0, decenter_y: 0, tilt_x: 0, tilt_y: 0, tilt_z: 0, order: 'decenter_then_tilt', restore_after: false }
      }
      break
    default:
      return
  }
  surface.shape = shape
}

/** A cylinder: a biconic with one plane axis ("Make Cylinder"). */
export function makeCylinder(system: System, index: number, axis: 'x' | 'y') {
  const c = curvatureOf(system.surfaces[index].shape) ?? 0
  system.surfaces[index].shape = {
    type: 'biconic',
    curvature_x: axis === 'x' ? c : 0,
    curvature_y: axis === 'y' ? c : 0,
    conic_x: 0,
    conic_y: 0,
    x_asphere: [],
    y_asphere: [],
  }
}
