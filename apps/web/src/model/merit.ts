// The merit function (apps/macos/Sources/MKLightUI/ViewModels/MeritModel.swift,
// docs/OPTIMIZATION.md): custom operands with their catalogue, the engine's
// default-merit generator settings, the project's saved `merit` section
// (`{mode, default_merit, spec, hold_efl}`) and the engine's
// `merit_evaluation` report.

import type { Json, JsonObject } from '@/engine/types'
import { nf, sig } from './format'

// MARK: Operands

export type Comparison = 'equal' | 'less_than' | 'greater_than'
export const COMPARISONS: { value: Comparison; symbol: string; help: string }[] = [
  { value: 'equal', symbol: '=', help: 'Equal to the target' },
  { value: 'less_than', symbol: '≤', help: 'An upper bound' },
  { value: 'greater_than', symbol: '≥', help: 'A lower bound' },
]
export const comparisonSymbol = (c: Comparison) => COMPARISONS.find((x) => x.value === c)?.symbol ?? '='

/** One line of a custom merit function: the engine's `Operand`, with its kind
 *  kept as JSON so parameters the app does not edit survive a round trip. */
export interface Operand {
  kind: JsonObject & { type: string }
  target: number
  weight: number
  comparison: Comparison
  /** An explicit constraint rather than a weighted residual. */
  constraint: boolean
  field: JsonObject
  wavelength: JsonObject
  config: number
  enabled: boolean
  label: string
}

export type ParamKind =
  | { type: 'number'; unit: string }
  | { type: 'integer'; min: number }
  | { type: 'surface' }
  | { type: 'wavelength' }
  | { type: 'choice'; options: string[] }
  | { type: 'flag' }
  | { type: 'sampling' }
  | { type: 'operandList' }

export interface OperandParam {
  key: string
  title: string
  kind: ParamKind
  required?: boolean
  defaultValue?: Json
}

export interface OperandDescriptor {
  type: string
  title: string
  category: string
  unit: string
  summary: string
  params: OperandParam[]
  /** Defined at one field point. */
  singleField?: boolean
}

export const OPERAND_CATEGORIES = ['First order', 'Geometry', 'Real rays', 'Image quality', 'Ghosts and stray light', 'Manufacturing', 'Composition']

const n = (key: string, title: string, unit = '', def = 0, required = true): OperandParam => ({
  key,
  title,
  kind: { type: 'number', unit },
  required,
  defaultValue: required ? def : undefined,
})
const surface = (key = 'surface', title = 'Surface', required = false): OperandParam => ({ key, title, kind: { type: 'surface' }, required })
const choice = (key: string, title: string, options: string[], def?: string): OperandParam => ({ key, title, kind: { type: 'choice', options }, defaultValue: def })
const wl = (key: string, title: string, def: number): OperandParam => ({ key, title, kind: { type: 'wavelength' }, required: true, defaultValue: def })
const pupil = [n('px', 'Pupil x', '', 0), n('py', 'Pupil y', '', 1)]
const sampling: OperandParam = { key: 'sampling', title: 'Pupil sampling', kind: { type: 'sampling' }, defaultValue: { rings: 3, arms: 6 } }

function d(type: string, title: string, category: string, unit: string, summary: string, params: OperandParam[] = [], singleField = false): OperandDescriptor {
  return { type, title, category, unit, summary, params, singleField }
}

const FO = 'First order', GE = 'Geometry', RR = 'Real rays', IQ = 'Image quality', GS = 'Ghosts and stray light', MA = 'Manufacturing', CO = 'Composition'

/** The engine's operands (docs/OPTIMIZATION.md, operand reference). */
export const OPERANDS: OperandDescriptor[] = [
  d('efl', 'EFL', FO, 'mm', 'Effective focal length (of the y section)'),
  d('efl_x', 'EFL x', FO, 'mm', 'Focal length of the x section'),
  d('efl_y', 'EFL y', FO, 'mm', 'Focal length of the y section'),
  d('bfl', 'BFL', FO, 'mm', 'Back focal length: last surface to paraxial focus'),
  d('bfl_x', 'BFL x', FO, 'mm', 'Back focus of the x section'),
  d('bfl_y', 'BFL y', FO, 'mm', 'Back focus of the y section'),
  d('anamorphic_squeeze', 'Anamorphic squeeze', FO, '', 'Signed EFL_y / EFL_x'),
  d('f_number', 'F-number', FO, '', 'Image-space, working or real working F/#', [
    choice('kind', 'Kind', ['paraxial_image_space', 'paraxial_working', 'real_working'], 'paraxial_image_space'),
  ]),
  d('magnification', 'Magnification', FO, '', 'Paraxial lateral magnification'),
  d('entrance_pupil_diameter', 'Entrance pupil diameter', FO, 'mm', ''),
  d('entrance_pupil_position', 'Entrance pupil position', FO, 'mm', 'From the first surface'),
  d('exit_pupil_position', 'Exit pupil position', FO, 'mm', 'From the image surface'),
  d('paraxial_ray', 'Paraxial ray', FO, 'mm / – / °', 'Marginal or chief ray height, slope or angle after a surface', [
    choice('ray', 'Ray', ['marginal', 'chief'], 'marginal'),
    choice('quantity', 'Quantity', ['height', 'slope', 'angle'], 'height'),
    surface(),
  ]),
  d('petzval_sum', 'Petzval sum', FO, '1/mm', 'Σ c (N′ − N)/(N N′)'),
  d('axial_colour', 'Axial color', FO, 'mm', 'Paraxial focus at short minus long λ', [wl('short', 'Short λ', 0), wl('long', 'Long λ', 2)]),
  d('total_track', 'Total track', GE, 'mm', 'Sum of thicknesses', [surface('from', 'From'), surface('to', 'To')]),
  d('centre_thickness', 'Center thickness', GE, 'mm', 'Axial gap after a surface', [surface('surface', 'Surface', true), surface('to', 'To')]),
  d('edge_thickness', 'Edge thickness', GE, 'mm', 'Gap at the rim (or a height)', [surface('surface', 'Surface', true), surface('to', 'To'), n('height', 'Height', 'mm', 0, false)]),
  d('min_thickness', 'Minimum thickness', GE, 'mm', 'Smallest gap of a medium', [
    choice('measure', 'Measure', ['centre', 'edge'], 'centre'),
    choice('medium', 'Medium', ['glass', 'air', 'any'], 'glass'),
    surface('from', 'From'),
    surface('to', 'To'),
  ]),
  d('max_thickness', 'Maximum thickness', GE, 'mm', 'Largest gap of a medium', [
    choice('measure', 'Measure', ['centre', 'edge'], 'centre'),
    choice('medium', 'Medium', ['glass', 'air', 'any'], 'glass'),
    surface('from', 'From'),
    surface('to', 'To'),
  ]),
  d(
    'real_ray',
    'Real ray',
    RR,
    'mm / – / °',
    'Position, direction or angle of one ray',
    [
      choice('quantity', 'Quantity', ['x', 'y', 'z', 'l', 'm', 'n', 'angle_x', 'angle_y', 'incidence_angle', 'optical_path'], 'y'),
      surface(),
      ...pupil,
      { key: 'global', title: 'Global frame', kind: { type: 'flag' } },
    ],
    true,
  ),
  d('ray_aberration', 'Ray aberration', RR, 'mm', 'Transverse aberration of one ray', [choice('axis', 'Axis', ['x', 'y'], 'y'), ...pupil, choice('reference', 'Reference', ['chief', 'centroid'], 'chief')], true),
  d('wavefront_aberration', 'Wavefront aberration', RR, 'waves', 'OPD of one ray', [...pupil, choice('reference', 'Reference', ['chief', 'centroid'], 'chief')], true),
  d('chief_ray_angle', 'Chief ray angle', RR, '°', 'Incidence of the chief ray (0 telecentric)', [surface()], true),
  d('rms_spot', 'RMS spot radius', IQ, 'mm', 'About the centroid or the chief ray', [choice('reference', 'Reference', ['centroid', 'chief'], 'centroid'), sampling]),
  d('spot_centroid', 'Spot centroid', IQ, 'mm', 'Centroid minus the chief ray', [choice('axis', 'Axis', ['x', 'y'], 'y')], true),
  d('wavefront_rms', 'RMS wavefront', IQ, 'waves', 'Piston (and tilt) removed', [choice('reference', 'Reference', ['centroid', 'chief'], 'centroid'), sampling]),
  d('mtf', 'MTF', IQ, '', 'Modulation at one frequency; use ≥ for a lower bound', [
    n('frequency', 'Frequency', 'cyc/mm', 30),
    choice('orientation', 'Orientation', ['tangential', 'sagittal', 'average'], 'average'),
    choice('method', 'Method', ['diffraction', 'geometric'], 'diffraction'),
    n('defocus_mm', 'Defocus', 'mm', 0, false),
  ]),
  d('distortion', 'Distortion', IQ, '%', 'Real chief ray against the mapping', [choice('mapping', 'Mapping', ['f_tan_theta', 'f_theta'], 'f_tan_theta')], true),
  d('lateral_colour', 'Lateral color', IQ, 'mm', 'Chief height at short minus long λ', [wl('short', 'Short λ', 0), wl('long', 'Long λ', 2)], true),
  d('relative_illumination', 'Relative illumination', IQ, '', 'Against on axis', [], true),
  d('ghost_focus', 'Ghost focus distance', GS, 'mm', 'Distance of the strongest ghosts’ focus from the image (larger is better)', [
    { key: 'worst', title: 'Worst ghosts', kind: { type: 'integer', min: 1 }, defaultValue: 1 },
    choice('metric', 'Metric', ['distance', 'peak'], 'distance'),
  ]),
  d('ghost_irradiance', 'Ghost irradiance', GS, 'rel.', 'Ghost against primary', [
    choice('metric', 'Metric', ['peak', 'mean', 'power'], 'peak'),
    { key: 'max_reflections', title: 'Max reflections', kind: { type: 'integer', min: 2 }, defaultValue: 2 },
  ]),
  d('stray_light_power', 'Stray light power', GS, 'rel.', 'Monte Carlo fraction; not for least squares', [
    choice('class', 'Class', ['ghost', 'scatter', 'mechanical', 'sneak', 'all_non_primary'], 'all_non_primary'),
    { key: 'rays', title: 'Rays', kind: { type: 'integer', min: 1000 }, defaultValue: 100000 },
  ]),
  d('doe_min_zone_period', 'DOE minimum zone period', MA, 'mm', '', [surface('surface', 'Surface', true)]),
  d('doe_max_phase_slope', 'DOE maximum phase slope', MA, 'rad/mm', '', [surface('surface', 'Surface', true)]),
  d('doe_total_phase', 'DOE total phase', MA, 'rad', '', [surface('surface', 'Surface', true)]),
  d('q_max_departure', 'Q-type maximum departure', MA, 'mm', 'From the best-fit sphere', [surface('surface', 'Surface', true)]),
  d('q_max_departure_slope', 'Q-type departure slope', MA, 'mm/mm', '', [surface('surface', 'Surface', true)]),
  d('freeform_max_departure', 'Freeform maximum departure', MA, 'mm', 'XY or Zernike sag from its base conic', [surface('surface', 'Surface', true)]),
  d('combine', 'Combine', CO, '', 'Arithmetic on earlier operands', [
    choice('op', 'Operation', ['sum', 'difference', 'product', 'quotient', 'abs', 'min', 'max'], 'sum'),
    { key: 'args', title: 'Operands', kind: { type: 'operandList' }, required: true, defaultValue: [] },
  ]),
  d('param_value', 'Parameter value', CO, '', 'Any model parameter'),
  d('tolerance_sensitivity', 'Tolerance sensitivity', CO, '', 'Of a criterion to a parameter'),
]

const BY_TYPE = new Map(OPERANDS.map((o) => [o.type, o]))
export const operandDescriptor = (type: string) => BY_TYPE.get(type)

/** Operands a user adds from the picker (those that need a model path or a
 *  nested criterion are read and kept, not added). */
export const ADDABLE_OPERANDS = OPERANDS.filter((o) => o.type !== 'param_value' && o.type !== 'tolerance_sensitivity')

export function operandTitle(type: string): string {
  return BY_TYPE.get(type)?.title ?? type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Descriptors matching a search, by category. */
export function searchOperands(text: string): { category: string; items: OperandDescriptor[] }[] {
  const q = text.trim().toLowerCase()
  const hits = ADDABLE_OPERANDS.filter(
    (o) => !q || o.title.toLowerCase().includes(q) || o.type.includes(q) || o.summary.toLowerCase().includes(q) || o.category.toLowerCase().includes(q),
  )
  return OPERAND_CATEGORIES.map((category) => ({ category, items: hits.filter((o) => o.category === category) })).filter((g) => g.items.length)
}

export function newOperand(desc: OperandDescriptor): Operand {
  const kind: JsonObject & { type: string } = { type: desc.type }
  for (const p of desc.params) if (p.defaultValue !== undefined) kind[p.key] = structuredClone(p.defaultValue)
  return {
    kind,
    target: 0,
    weight: 1,
    comparison: 'equal',
    constraint: false,
    field: desc.singleField ? { field: 'index', index: 0 } : { field: 'all' },
    wavelength: { wavelength: 'primary' },
    config: 0,
    enabled: true,
    label: '',
  }
}

const obj = (v: Json | undefined): JsonObject | null => (v && typeof v === 'object' && !Array.isArray(v) ? (v as JsonObject) : null)
const num = (v: Json | undefined, def: number) => {
  const x = nf(v)
  return Number.isFinite(x) ? x : def
}

/** Reads an engine operand (a spec's or a result's operand table). */
export function readOperand(j: Json): Operand | null {
  const o = obj(j)
  const k = obj(o?.kind)
  if (!o || !k || typeof k.type !== 'string') return null
  const c = o.comparison
  return {
    kind: structuredClone(k) as Operand['kind'],
    target: num(o.target, 0),
    weight: num(o.weight, 1),
    comparison: c === 'less_than' || c === 'greater_than' ? c : 'equal',
    constraint: o.role === 'constraint',
    field: obj(o.field) ?? { field: 'all' },
    wavelength: obj(o.wavelength) ?? { wavelength: 'primary' },
    config: typeof o.config === 'number' ? o.config : 0,
    enabled: o.enabled !== false,
    label: typeof o.label === 'string' ? o.label : '',
  }
}

/** The engine's JSON form. */
export function operandJson(op: Operand): JsonObject {
  const o: JsonObject = {
    kind: op.kind,
    target: op.target,
    weight: op.weight,
    comparison: op.comparison,
    role: op.constraint ? 'constraint' : 'residual',
    field: op.field,
    wavelength: op.wavelength,
    config: op.config,
    enabled: op.enabled,
  }
  if (op.label) o.label = op.label
  return o
}

/** Problems the app can see before the engine does. */
export function localIssues(op: Operand, fieldCount: number, wavelengthCount: number): string[] {
  const out: string[] = []
  const desc = operandDescriptor(op.kind.type)
  if (!desc) return out
  if (desc.singleField && op.field.field === 'all') out.push('Needs one field, not all fields.')
  if (op.field.field === 'index' && typeof op.field.index === 'number' && op.field.index >= fieldCount) out.push(`Field ${op.field.index + 1} does not exist.`)
  if (op.wavelength.wavelength === 'index' && typeof op.wavelength.index === 'number' && op.wavelength.index >= wavelengthCount)
    out.push(`Wavelength ${op.wavelength.index + 1} does not exist.`)
  for (const p of desc.params) if (p.required && (op.kind[p.key] === undefined || op.kind[p.key] === null)) out.push(`${p.title} is required.`)
  if (!Number.isFinite(op.weight) || op.weight < 0) out.push('The weight must be zero or positive.')
  return out
}

export const choiceTitle = (s: string) => {
  const t = s.replace(/_/g, ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}

/** A short text of an operand's selectors and parameters, field and wavelength first. */
export function operandParameters(op: Operand): string {
  const parts: string[] = []
  const f = op.field
  if (f.field === 'index') parts.push(`F${num(f.index, 0) + 1}`)
  else if (f.field === 'normalized') parts.push(`H(${sig(num(f.hx, 0), 4)}, ${sig(num(f.hy, 0), 4)})`)
  const w = op.wavelength
  if (w.wavelength === 'all') parts.push('poly')
  else if (w.wavelength === 'index') parts.push(`λ${num(w.index, 0) + 1}`)
  for (const p of operandDescriptor(op.kind.type)?.params ?? []) {
    const v = op.kind[p.key]
    if (v === undefined || v === null) continue
    switch (p.kind.type) {
      case 'surface':
        parts.push(`${p.title.toLowerCase()} S${typeof v === 'number' ? v : '?'}`)
        break
      case 'wavelength':
        parts.push(`${p.title} λ${num(v, 0) + 1}`)
        break
      case 'choice':
        parts.push(String(v).replace(/_/g, ' '))
        break
      case 'flag':
        if (v === true) parts.push(p.title.toLowerCase())
        break
      case 'sampling': {
        const s = obj(v)
        parts.push(`${num(s?.rings, 3)}×${num(s?.arms, 6)}`)
        break
      }
      case 'operandList':
        parts.push('of ' + (Array.isArray(v) ? v.map((x) => `#${num(x, 0) + 1}`).join(', ') : ''))
        break
      default:
        parts.push(`${p.title.toLowerCase()} ${typeof v === 'number' ? sig(v, 4) : String(v)}`)
    }
  }
  return parts.join(' ')
}

/** A share of the merit as a short percentage. */
export function percentShare(f: number): string {
  if (!Number.isFinite(f)) return ''
  if (f >= 0.995) return '100%'
  if (f >= 0.1) return `${(f * 100).toFixed(0)}%`
  if (f >= 0.001) return `${(f * 100).toFixed(1)}%`
  return f > 0 ? '<0.1%' : '0%'
}

// MARK: Default merit

export type Criterion = 'rms_spot' | 'wavefront' | 'mtf'
export const CRITERIA: { value: Criterion; title: string }[] = [
  { value: 'rms_spot', title: 'RMS spot radius' },
  { value: 'wavefront', title: 'RMS wavefront' },
  { value: 'mtf', title: 'MTF' },
]

/** The engine's default merit generator (`DefaultMeritSpec`). */
export interface DefaultMerit {
  criterion: Criterion
  /** Spot or wavefront reference: centroid or chief. */
  reference: string
  /** Spot centroid per wavelength (`monochromatic`) or of the white-light spot. */
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
  thickness: boolean
  glassMinCentre: number | null
  glassMaxCentre: number | null
  glassMinEdge: number | null
  airMinCentre: number | null
  airMaxCentre: number | null
  airMinEdge: number | null
  squeezeTarget: number | null
}

export const defaultMerit = (): DefaultMerit => ({
  criterion: 'rms_spot',
  reference: 'centroid',
  scope: 'monochromatic',
  mtfFrequencies: [10, 30],
  mtfTarget: 0.5,
  mtfMethod: 'diffraction',
  worstField: false,
  rings: 3,
  arms: 6,
  fields: null,
  wavelengths: null,
  thickness: true,
  glassMinCentre: 1.0,
  glassMaxCentre: 30.0,
  glassMinEdge: 1.0,
  airMinCentre: 0.1,
  airMaxCentre: null,
  airMinEdge: 0.1,
  squeezeTarget: null,
})

const optNumber = (v: Json | undefined): number | null => {
  if (v === undefined || v === null) return null
  const x = nf(v)
  return Number.isFinite(x) ? x : null
}
const intArray = (v: Json | undefined): number[] | null => (Array.isArray(v) ? v.filter((x): x is number => typeof x === 'number') : null)

export function readDefaultMerit(j: Json | undefined): DefaultMerit {
  const m = defaultMerit()
  const o = obj(j)
  if (!o) return m
  const c = obj(o.criterion)
  if (c) {
    if (c.type === 'rms_spot' || c.type === 'wavefront' || c.type === 'mtf') m.criterion = c.type
    if (typeof c.reference === 'string') m.reference = c.reference
    if (typeof c.scope === 'string') m.scope = c.scope
    const f = intArray(c.frequencies)
    if (f) m.mtfFrequencies = f
    m.mtfTarget = optNumber(c.target) ?? m.mtfTarget
    if (typeof c.method === 'string') m.mtfMethod = c.method
    m.worstField = c.worst_field === true
  }
  const s = obj(o.sampling)
  if (typeof s?.rings === 'number') m.rings = s.rings
  if (typeof s?.arms === 'number') m.arms = s.arms
  m.fields = intArray(o.fields)
  m.wavelengths = intArray(o.wavelengths)
  if ('thickness' in o) {
    const t = obj(o.thickness)
    m.thickness = t != null
    if (t) {
      m.glassMinCentre = optNumber(t.glass_min_centre)
      m.glassMaxCentre = optNumber(t.glass_max_centre)
      m.glassMinEdge = optNumber(t.glass_min_edge)
      m.airMinCentre = optNumber(t.air_min_centre)
      m.airMaxCentre = optNumber(t.air_max_centre)
      m.airMinEdge = optNumber(t.air_min_edge)
    }
  }
  m.squeezeTarget = optNumber(o.squeeze_target)
  return m
}

/** The `default_merit` settings object (laid over the engine's defaults). */
export function defaultMeritJson(m: DefaultMerit): JsonObject {
  const crit: JsonObject = { type: m.criterion }
  if (m.criterion === 'rms_spot') {
    crit.reference = m.reference
    crit.scope = m.scope
  } else if (m.criterion === 'wavefront') {
    crit.reference = m.reference
  } else {
    crit.frequencies = [...m.mtfFrequencies]
    crit.target = m.mtfTarget
    crit.method = m.mtfMethod
    crit.worst_field = m.worstField
  }
  const o: JsonObject = {
    criterion: crit,
    sampling: { rings: m.rings, arms: m.arms },
    fields: m.fields ? [...m.fields] : null,
    wavelengths: m.wavelengths ? [...m.wavelengths] : null,
    thickness: m.thickness
      ? {
          glass_min_centre: m.glassMinCentre,
          glass_max_centre: m.glassMaxCentre,
          glass_min_edge: m.glassMinEdge,
          air_min_centre: m.airMinCentre,
          air_max_centre: m.airMaxCentre,
          air_min_edge: m.airMinEdge,
        }
      : null,
  }
  if (m.squeezeTarget != null) o.squeeze_target = m.squeezeTarget
  return o
}

export function defaultMeritSummary(m: DefaultMerit): string {
  let s = m.criterion === 'rms_spot' ? 'RMS spot radius' : m.criterion === 'wavefront' ? 'RMS wavefront' : `MTF ≥ ${sig(m.mtfTarget, 4)} at ${m.mtfFrequencies.map((f) => sig(f, 4)).join(', ')} cyc/mm`
  if (m.criterion === 'mtf' && m.worstField) s += ', weakest field'
  s += ` · ${m.rings} rings × ${m.arms} arms`
  if (m.thickness) s += ' · thickness limits'
  if (m.squeezeTarget != null) s += ` · squeeze ${sig(m.squeezeTarget, 4)}`
  return s
}

export function defaultMeritCaption(c: Criterion): string {
  switch (c) {
    case 'rms_spot':
      return 'The mean square RMS spot radius over the chosen fields and wavelengths, by their weights.'
    case 'wavefront':
      return 'The mean square RMS wavefront error over the chosen fields and wavelengths, by their weights.'
    case 'mtf':
      return 'Polychromatic tangential and sagittal MTF lower bounds at each frequency.'
  }
}

// MARK: Saved merit

/** The merit function of a design, saved in the project's `merit` section. */
export interface MeritSettings {
  mode: 'default' | 'custom'
  defaults: DefaultMerit
  operands: Operand[]
  holdEfl: boolean
  /** The custom spec's other keys (`name`, `on_failure`, …), kept as the
   *  engine wrote them so a converted default merit evaluates the same. */
  specOptions: JsonObject
}

export function readMerit(section: Json | undefined): MeritSettings {
  const s = obj(section)
  const spec = obj(s?.spec)
  const specOptions: JsonObject = {}
  for (const [k, v] of Object.entries(spec ?? {})) if (k !== 'operands' && k !== 'schema_version') specOptions[k] = v
  return {
    mode: s?.mode === 'custom' ? 'custom' : 'default',
    defaults: readDefaultMerit(s?.default_merit),
    operands: (Array.isArray(spec?.operands) ? spec.operands : []).map(readOperand).filter((x): x is Operand => x != null),
    holdEfl: s?.hold_efl !== false,
    specOptions,
  }
}

/** The custom operands as a `MeritFunctionSpec`. */
export function meritSpec(m: MeritSettings): JsonObject {
  return { name: '', ...m.specOptions, schema_version: 1, operands: m.operands.map(operandJson) }
}

export function meritSection(m: MeritSettings): JsonObject {
  return { mode: m.mode, default_merit: defaultMeritJson(m.defaults), hold_efl: m.holdEfl, spec: meritSpec(m) }
}

/** The merit part of `optimize` / `merit_evaluation` / `glass_substitution` settings. */
export function meritJobSettings(m: MeritSettings): JsonObject {
  const s: JsonObject = { hold_efl: m.holdEfl }
  if (m.mode === 'default') s.default_merit = defaultMeritJson(m.defaults)
  else s.merit = meritSpec(m)
  return s
}

// MARK: Editing (pure; `combine` references follow their operands)

function remapCombine(ops: Operand[], f: (old: number) => number | null): Operand[] {
  return ops.map((op) => {
    if (op.kind.type !== 'combine') return op
    const args = Array.isArray(op.kind.args) ? (op.kind.args as Json[]).filter((x): x is number => typeof x === 'number') : []
    return { ...op, kind: { ...op.kind, args: args.map(f).filter((x): x is number => x != null) } }
  })
}

/** Inserts `op` after `index` (at the end when null); returns the list and its index. */
export function addOperand(ops: Operand[], op: Operand, after: number | null): { operands: Operand[]; index: number } {
  const at = after == null ? ops.length : Math.min(after + 1, ops.length)
  const next = remapCombine(ops, (k) => (k >= at ? k + 1 : k))
  next.splice(at, 0, op)
  return { operands: next, index: at }
}

export function deleteOperands(ops: Operand[], indices: number[]): Operand[] {
  const removed = new Set(indices)
  const kept = ops.filter((_, k) => !removed.has(k))
  return remapCombine(kept, (old) => (removed.has(old) ? null : old - indices.filter((r) => r < old).length))
}

export function duplicateOperand(ops: Operand[], index: number): { operands: Operand[]; index: number } {
  return addOperand(ops, structuredClone(ops[index]), index)
}

/** Moves operand `from` to position `to` (both in the current list). */
export function moveOperand(ops: Operand[], from: number, to: number): Operand[] {
  if (from === to || to < 0 || to >= ops.length) return ops
  const order = ops.map((_, k) => k)
  order.splice(to, 0, order.splice(from, 1)[0])
  const newIndex = new Map(order.map((old, k) => [old, k]))
  return remapCombine(
    order.map((k) => ops[k]),
    (old) => newIndex.get(old) ?? null,
  )
}

/** A field for menus as specified (FieldLabel): `1: 14°`, `2: (3, 4) mm`. */
export function fieldMenuLabel(k: number, f: { x: number; y: number }, unit: string): string {
  const u = unit === '°' ? '°' : unit ? ' ' + unit : ''
  const v = f.x === 0 ? sig(f.y, 6) : `(${sig(f.x, 6)}, ${sig(f.y, 6)})`
  return `${k + 1}: ${v}${u}`
}

/** Duplicates the operands at `indices`, each right after itself; returns the
 *  list and the copies' indices (MeritSettings.duplicate). */
export function duplicateOperands(ops: Operand[], indices: number[]): { operands: Operand[]; indices: number[] } {
  let next = ops
  const sorted = [...new Set(indices)].filter((k) => k >= 0 && k < ops.length).sort((a, b) => a - b)
  // From the end, so the earlier indices stay valid.
  for (const k of [...sorted].reverse()) next = duplicateOperand(next, k).operands
  // Copy j sits after its original, shifted by the j copies inserted before it.
  return { operands: next, indices: sorted.map((k, j) => k + j + 1) }
}

/** Moves the operands at `from` to the insertion offset `to` (a gap in the
 *  current list, 0…length; List `onMove` semantics), keeping `combine`
 *  references; returns the list and the moved operands' new indices. */
export function moveOperandsTo(ops: Operand[], from: number[], to: number): { operands: Operand[]; indices: number[] } {
  const moving = [...new Set(from)].filter((k) => k >= 0 && k < ops.length).sort((a, b) => a - b)
  if (!moving.length) return { operands: ops, indices: [] }
  const set = new Set(moving)
  const rest = ops.map((_, k) => k).filter((k) => !set.has(k))
  const at = Math.max(0, Math.min(to, ops.length)) - moving.filter((k) => k < to).length
  const order = [...rest.slice(0, at), ...moving, ...rest.slice(at)]
  if (order.every((old, k) => old === k)) return { operands: ops, indices: moving }
  const newIndex = new Map(order.map((old, k) => [old, k]))
  return {
    operands: remapCombine(
      order.map((k) => ops[k]),
      (old) => newIndex.get(old) ?? null,
    ),
    indices: moving.map((k) => newIndex.get(k)!),
  }
}

/** The EFL hold the engine adds (`hold EFL`, or `hold EFL (penalty)`). */
export const isEflHold = (op: Operand) => op.kind.type === 'efl' && op.label.startsWith('hold EFL')

// MARK: Evaluation

export interface EvaluationRow {
  index: number
  kind: string
  label: string
  constraint: boolean
  enabled: boolean
  value: number
  target: number
  weight: number
  residual: number
  contribution: number
  fraction: number
  error: string | null
}

export interface MeritEvaluation {
  merit: number
  violation: number
  status: { status: 'ok' } | { status: 'degraded'; failures: number } | { status: 'failed' }
  reportedOn: string
  rows: EvaluationRow[]
  /** The expanded spec (generated operands and the EFL hold included). */
  spec: JsonObject | null
}

export function readEvaluation(j: Json | undefined): MeritEvaluation | null {
  const o = obj(j)
  if (!o || !('merit' in o)) return null
  const st = obj(o.status)
  const backend = obj(o.backend)
  return {
    merit: nf(o.merit),
    violation: nf(o.violation),
    status:
      st?.status === 'degraded'
        ? { status: 'degraded', failures: typeof st.failures === 'number' ? st.failures : 0 }
        : st?.status === 'failed'
          ? { status: 'failed' }
          : { status: 'ok' },
    reportedOn: String(backend?.reported_on ?? backend?.evaluated_on ?? ''),
    spec: obj(o.merit_spec),
    rows: (Array.isArray(o.operands) ? o.operands : []).map((r, k) => {
      const x = obj(r) ?? {}
      return {
        index: typeof x.index === 'number' ? x.index : k,
        kind: typeof x.kind === 'string' ? x.kind : '',
        label: typeof x.label === 'string' ? x.label : '',
        constraint: x.role === 'constraint',
        enabled: x.enabled !== false,
        value: nf(x.value),
        target: nf(x.target),
        weight: nf(x.weight),
        residual: nf(x.residual),
        contribution: nf(x.contribution),
        fraction: nf(x.contribution_fraction),
        error: typeof x.error === 'string' ? x.error : null,
      }
    }),
  }
}

/** The row of operand `index`, if it is an operand of type `kind`. */
export function evaluationRow(e: MeritEvaluation | null, index: number, kind?: string): EvaluationRow | null {
  return e?.rows.find((r) => r.index === index && (kind == null || r.kind === kind)) ?? null
}

/** The operands of the expanded spec and its other keys. */
export function specOperands(e: MeritEvaluation): { operands: Operand[]; options: JsonObject } {
  const spec = e.spec ?? {}
  const options: JsonObject = {}
  for (const [k, v] of Object.entries(spec)) if (k !== 'operands' && k !== 'schema_version') options[k] = v
  const operands = (Array.isArray(spec.operands) ? spec.operands : []).map(readOperand).filter((x): x is Operand => x != null)
  return { operands, options }
}

export function evaluationHelp(e: MeritEvaluation | null): string {
  if (!e) return 'The merit at the current design, before optimizing.'
  const on = !e.reportedOn || e.reportedOn.startsWith('cpu') ? 'CPU f64 reference' : e.reportedOn
  let s = `Merit ${sig(e.merit, 4)} at the current design (${on}).`
  if (Number.isFinite(e.violation) && e.violation > 1e-9) s += ` Constraint violation ${sig(e.violation, 4)}.`
  if (e.status.status === 'degraded') s += ` ${e.status.failures} residual${e.status.failures === 1 ? '' : 's'} carry penalty values (failed rays or operands).`
  if (e.status.status === 'failed') s += ' The evaluation failed: no meaningful merit.'
  return s
}
