// Surface types described as data (docs/APP.md "Progressive disclosure",
// apps/macos/Sources/MKLightUI/ViewModels/SurfaceTypes.swift): sections of
// parameters addressed by their key path in the shape JSON, and coefficient
// lists with term labels. The inspector draws any type from its descriptor.

import type { Json, JsonObject } from '@/engine/types'
import { t, tMaybe, type Key } from '@/i18n'
import type { ParamPath } from './params'
import type { Shape, Surface, System } from './system'
import { conicOf, curvatureOf } from './system'

export type Group = 'Spherical' | 'Aspheric' | 'Freeform' | 'Anamorphic' | 'Diffractive' | 'Special'
export const GROUPS: Group[] = ['Spherical', 'Aspheric', 'Freeform', 'Anamorphic', 'Diffractive', 'Special']

/** A group's name in the UI language (`GROUPS` holds the ids). */
export const groupTitle = (g: Group): string => t(`inspector.groups.${g}` as Key)

export interface ShapeParam {
  key: string
  title: string
  unit: string
  /** Shown as a radius of curvature (1/value), edited as a radius. */
  radius?: boolean
  integer?: boolean
  help?: string
  options?: { value: string; title: string }[]
  /** The parameter path that makes it a variable or solve target. */
  path?: (surface: number) => ParamPath
}

export interface CoefficientList {
  key: string
  readonly title: string
  max: (shape: Shape) => number
  term: (k: number, shape: Shape) => string
  meaning?: (k: number, shape: Shape) => string
  unit?: (k: number) => string
  /** Path of term k, for variables. */
  path?: (surface: number, k: number) => ParamPath
  /** Long lists (XY, Zernike): only nonzero terms show; any term from a menu. */
  sparse?: boolean
}

export interface ParamSection {
  id: string
  title: string
  params: ShapeParam[]
  coefficients?: CoefficientList
  footer?: string
  /** Whether the section applies to the shape (e.g. a DOE's base). */
  applies?: (shape: Shape) => boolean
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

const curvaturePath = (surface: number): ParamPath => ({ param: 'curvature', surface })
const conicPath = (surface: number): ParamPath => ({ param: 'conic', surface })
const shapeParamPath = (parameter: string) => (surface: number): ParamPath => ({ param: 'shape_param', surface, parameter })
const coefficientPath = (set: string) => (surface: number, index: number): ParamPath => ({ param: 'shape_coefficient', surface, set, index })
const asphere = (surface: number, index: number): ParamPath => ({ param: 'asphere', surface, index })

const text = (key: string): string => t(`inspector.${key}` as Key)

/** A parameter whose title and help are read through `t()` when used. */
const param = (key: string, title: string, rest: Omit<ShapeParam, 'key' | 'title' | 'help'> & { help?: string }): ShapeParam => ({
  key,
  ...rest,
  get title() {
    return text(`params.${title}`)
  },
  get help() {
    return rest.help ? text(`params.${rest.help}`) : undefined
  },
})

const radius = (key = 'curvature', title = 'radius', path: ShapeParam['path'] = curvaturePath): ShapeParam =>
  param(key, title, { unit: 'mm', radius: true, help: 'radiusHelp', path })
const conic = (key = 'conic', title = 'conic', path: ShapeParam['path'] = conicPath): ShapeParam => param(key, title, { unit: '', help: 'conicHelp', path })
const normRadius = (key = 'norm_radius'): ShapeParam =>
  param(key, 'normRadius', { unit: 'mm', help: 'normRadiusHelp', path: shapeParamPath('norm_radius') })

/** A section whose title and footer are read through `t()` when used. */
const section = (id: string, rest: Omit<ParamSection, 'id' | 'title' | 'footer'>, footer = false): ParamSection => ({
  id,
  ...rest,
  get title() {
    return text(`sections.${id}`)
  },
  get footer() {
    return footer ? text(`footers.${id}`) : undefined
  },
})

const axisList = (axis: string, key: string, letter: string): CoefficientList => ({
  key,
  get title() {
    return t('inspector.sections.axisPolynomial', { axis: axis.toUpperCase() })
  },
  max: () => 10,
  term: (k) => `${letter}${sub(2 * k + 4)}`,
  meaning: (k) => axis + sup(2 * k + 4),
  unit: (k) => mmPower(-(2 * k + 3)),
  path: coefficientPath(key),
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

const ZERNIKE_NAMES = new Set(['0,0', '1,1', '2,0', '2,2', '3,1', '3,3', '4,0', '4,2', '4,4', '5,1', '5,3', '5,5', '6,0', '8,0'])

export function zernikeName(z: { n: number; m: number }): string {
  const m = Math.abs(z.m)
  const axis = z.m > 0 ? ' x' : z.m < 0 ? ' y' : ''
  const id = `${z.n},${m}`
  if (ZERNIKE_NAMES.has(id)) {
    const name = text(`zernike.n${z.n}m${m}`)
    return name + (m === 0 ? '' : z.n === 1 || z.n === 3 || z.n === 5 ? axis : z.m > 0 ? ' 0°' : ' 45°')
  }
  return `n${z.n} m${z.m}`
}

/** A surface type whose title and summary are read through `t()` when used. */
const surfaceType = (id: string, group: Group, sections: ParamSection[]): SurfaceTypeDescriptor => ({
  id,
  group,
  sections,
  get title() {
    return text(`types.${id}.title`)
  },
  get summary() {
    return text(`types.${id}.summary`)
  },
})

/** The coefficient list's title: "Coefficients". */
const coefficientsTitle = {
  get title() {
    return text('sections.coefficients')
  },
}

export const SURFACE_TYPES: SurfaceTypeDescriptor[] = [
  surfaceType('plane', 'Spherical', []),
  surfaceType('conic', 'Spherical', [section('shape', { params: [radius(), conic()] })]),
  surfaceType('even_asphere', 'Aspheric', [
    section('baseConic', { params: [radius(), conic()] }),
    section('polynomial', {
      params: [],
      coefficients: {
        key: 'coefficients',
        ...coefficientsTitle,
        max: () => 10,
        term: (k) => `A${sub(2 * k + 4)}`,
        meaning: (k) => 'r' + sup(2 * k + 4),
        unit: (k) => mmPower(-(2 * k + 3)),
        path: asphere,
      },
    }),
  ]),
  surfaceType('odd_asphere', 'Aspheric', [
    section('baseConic', { params: [radius(), conic()] }),
    section('polynomial', {
      params: [],
      coefficients: {
        key: 'coefficients',
        ...coefficientsTitle,
        max: () => 20,
        term: (k) => `β${sub(k + 1)}`,
        meaning: (k) => 'r' + (k === 0 ? '' : sup(k + 1)),
        unit: (k) => mmPower(-k),
        path: coefficientPath('odd'),
      },
    }),
  ]),
  surfaceType('q_con', 'Aspheric', [
    section('baseConic', { params: [radius(), conic(), normRadius()] }),
    section('qPolynomial', {
      params: [],
      coefficients: {
        key: 'coefficients',
        ...coefficientsTitle,
        max: () => 20,
        term: (k) => `a${sub(k)}`,
        meaning: (k) => `u⁴Q${sub(k)}ᶜᵒⁿ`,
        unit: () => 'mm',
        path: coefficientPath('q'),
      },
    }),
  ]),
  surfaceType('q_bfs', 'Aspheric', [
    section('bestFitSphere', { params: [radius(), normRadius()] }),
    section('qPolynomial', {
      params: [],
      coefficients: {
        key: 'coefficients',
        ...coefficientsTitle,
        max: () => 20,
        term: (k) => `a${sub(k)}`,
        meaning: (k) => `Q${sub(k)}ᵇᶠˢ`,
        unit: () => 'mm',
        path: coefficientPath('q'),
      },
    }),
  ]),
  surfaceType('xy_polynomial', 'Freeform', [
    section('baseConic', { params: [radius(), conic(), normRadius()] }),
    section('xyTerms', {
      params: [],
      coefficients: {
        key: 'coefficients',
        ...coefficientsTitle,
        max: () => 65,
        term: (k) => `C(${monomial(xyTerm(k))})`,
        unit: () => 'mm',
        path: coefficientPath('xy'),
        sparse: true,
      },
    }),
  ]),
  surfaceType('zernike_sag', 'Freeform', [
    section('baseConic', {
      params: [
        radius(),
        conic(),
        normRadius(),
        {
          key: 'ordering',
          get title() {
            return text('params.ordering')
          },
          unit: '',
          options: [
            { value: 'fringe', title: 'Fringe (37)' },
            { value: 'standard', title: 'Standard (45)' },
          ],
        },
      ],
    }),
    section('zernikeTerms', {
      params: [],
      coefficients: {
        key: 'coefficients',
        ...coefficientsTitle,
        max: (s) => (s.ordering === 'standard' ? 45 : 37),
        term: (k) => `Z${sub(k + 1)}`,
        meaning: (k, s) => zernikeName(zernikeTerm(k + 1, String(s.ordering ?? 'fringe'))),
        unit: () => 'mm',
        path: coefficientPath('zernike'),
        sparse: true,
      },
    }),
  ]),
  surfaceType('toroid', 'Anamorphic', [
    section('yProfile', {
      params: [radius('curvature_y', 'radiusY'), conic('conic_y', 'conicY', shapeParamPath('conic_y'))],
      coefficients: axisList('y', 'y_asphere', 'B'),
    }),
    section('rotation', { params: [radius('curvature_x', 'radiusX', shapeParamPath('curvature_x'))] }, true),
  ]),
  surfaceType('biconic', 'Anamorphic', [
    section('xSection', {
      params: [radius('curvature_x', 'radiusX', shapeParamPath('curvature_x')), conic('conic_x', 'conicX', shapeParamPath('conic_x'))],
      coefficients: axisList('x', 'x_asphere', 'A'),
    }),
    section('ySection', {
      params: [radius('curvature_y', 'radiusY'), conic('conic_y', 'conicY', shapeParamPath('conic_y'))],
      coefficients: axisList('y', 'y_asphere', 'B'),
    }),
  ]),
  surfaceType('diffractive', 'Diffractive', [
    section('baseShape', {
      params: [radius('base.curvature'), conic('base.conic')],
      applies: (s) => (s.base as Shape | undefined)?.type !== 'plane',
    }),
    section('basePolynomial', {
      params: [],
      coefficients: {
        key: 'base.coefficients',
        ...coefficientsTitle,
        max: () => 10,
        term: (k) => `A${sub(2 * k + 4)}`,
        meaning: (k) => 'r' + sup(2 * k + 4),
        unit: (k) => mmPower(-(2 * k + 3)),
        path: asphere,
      },
      applies: (s) => (s.base as Shape | undefined)?.type === 'even_asphere',
    }),
    section(
      'phase',
      {
        params: [normRadius('phase.norm_radius')],
        coefficients: {
          key: 'phase.coefficients',
          ...coefficientsTitle,
          max: () => 10,
          term: (k) => `A${sub(k + 1)}`,
          meaning: (k) => 'ρ' + sup(2 * (k + 1)),
          unit: () => 'rad',
          path: coefficientPath('phase'),
        },
      },
      true,
    ),
    section(
      'orders',
      {
        params: [
          param('phase.design_order', 'designOrder', { unit: '', integer: true, help: 'designOrderHelp' }),
          param('phase.design_wavelength_um', 'designWavelength', { unit: 'µm', help: 'designWavelengthHelp' }),
          param('phase.trace_order', 'traceOrder', { unit: '', integer: true, help: 'traceOrderHelp' }),
        ],
      },
      true,
    ),
  ]),
  surfaceType('paraxial', 'Special', [
    section('idealLens', {
      params: [param('focal_length', 'focalLength', { unit: 'mm', help: 'focalLengthHelp', path: (surface) => ({ param: 'paraxial_focal_length', surface }) })],
    }),
  ]),
  surfaceType('coordinate_break', 'Special', []),
]

export const descriptorFor = (type: string) => SURFACE_TYPES.find((d) => d.id === type)

/** `sphere` is the engine's legacy alias of a conic with k = 0. */
export const typeTitle = (type: string) => descriptorFor(type === 'sphere' ? 'conic' : type)?.title ?? type

/** The power axis when the shape is a cylinder: a biconic with one curved
 *  axis and the other flat (zero curvature and polynomial). */
export function cylinderAxis(shape: Shape): 'x' | 'y' | null {
  if (shape.type !== 'biconic') return null
  const zero = (v: Json | undefined) => !Array.isArray(v) || v.every((x) => x === 0)
  const cx = Number(shape.curvature_x ?? 0)
  const cy = Number(shape.curvature_y ?? 0)
  const flatX = cx === 0 && zero(shape.x_asphere)
  const flatY = cy === 0 && zero(shape.y_asphere)
  if (flatY && cx !== 0) return 'x'
  if (flatX && cy !== 0) return 'y'
  return null
}

/** The type menu's entry a shape is: a cylinder shows as its preset. */
export function typeChoice(shape: Shape): string {
  const a = cylinderAxis(shape)
  if (a) return `cylinder_${a}`
  return shape.type === 'sphere' ? 'conic' : shape.type
}

export const CYLINDER_CHOICES = [
  {
    id: 'cylinder_x',
    get title() {
      return text('types.cylinder_x.title')
    },
  },
  {
    id: 'cylinder_y',
    get title() {
      return text('types.cylinder_y.title')
    },
  },
]

/** Short type name for the table: a conic of k = 0 is a sphere. */
export function tableTypeTitle(shape: Shape): string {
  switch (shape.type) {
    case 'conic':
      return Number(shape.conic ?? 0) === 0 ? text('table.sphere') : text('table.conic')
    case 'sphere':
      return text('table.sphere')
    case 'q_con':
      return text('table.qCon')
    case 'q_bfs':
      return text('table.qBfs')
    case 'biconic':
      return cylinderAxis(shape) ? text('table.cylinder') : text('table.biconic')
    default:
      return typeTitle(shape.type)
  }
}

/** Short type label for the table's type column (the type id when it has none). */
export const typeShort = (id: string): string => tMaybe(`inspector.short.${id}`) ?? id

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

/** A DOE's base as another type (plane, conic, even asphere), keeping its
 *  curvature, conic and polynomial where they carry over. */
export function setDiffractiveBase(shape: Shape, kind: string): Shape {
  if (shape.type !== 'diffractive') return shape
  const base = (shape.base as Shape | undefined) ?? { type: 'plane' }
  const c = curvatureOf(base) ?? 0
  const k = conicOf(base) ?? 0
  const next: Shape =
    kind === 'conic'
      ? { type: 'conic', curvature: c, conic: k }
      : kind === 'even_asphere'
        ? { type: 'even_asphere', curvature: c, conic: k, coefficients: (base.coefficients as number[] | undefined) ?? [] }
        : { type: 'plane' }
  return { ...shape, base: next }
}
