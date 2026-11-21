// The merit function (apps/macos/Sources/MKLightUI/ViewModels/MeritModel.swift,
// docs/OPTIMIZATION.md): custom operands with their catalogue, the engine's
// default-merit generator settings, the project's saved `merit` section
// (`{mode, default_merit, spec, hold_efl}`) and the engine's
// `merit_evaluation` report.

import { currentLanguage, t, tMaybe } from '@/i18n'
import type { Json, JsonObject } from '@/engine/types'
import { nf, sig } from './format'

// MARK: Operands

export type Comparison = 'equal' | 'less_than' | 'greater_than'
export const COMPARISONS: { value: Comparison; symbol: string; help: string }[] = [
  { value: 'equal', symbol: '=', get help() { return t('optimize.comparisons.equal') } },
  { value: 'less_than', symbol: '≤', get help() { return t('optimize.comparisons.lessThan') } },
  { value: 'greater_than', symbol: '≥', get help() { return t('optimize.comparisons.greaterThan') } },
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
  /** The parameter's name in the current language. */
  readonly title: string
  kind: ParamKind
  required?: boolean
  defaultValue?: Json
}

export interface OperandDescriptor {
  type: string
  /** The operand's name and summary in the current language. */
  readonly title: string
  /** The category's stable name (group key); `categoryTitle` shows it. */
  category: string
  unit: string
  readonly summary: string
  params: OperandParam[]
  /** Defined at one field point. */
  singleField?: boolean
}

export const OPERAND_CATEGORIES = ['First order', 'Geometry', 'Real rays', 'Image quality', 'Ghosts and stray light', 'Manufacturing', 'Composition']
const CATEGORY_KEYS: Record<string, string> = {
  'First order': 'firstOrder',
  Geometry: 'geometry',
  'Real rays': 'realRays',
  'Image quality': 'imageQuality',
  'Ghosts and stray light': 'ghostsAndStrayLight',
  Manufacturing: 'manufacturing',
  Composition: 'composition',
}

/** A category's name in the current language. */
export const categoryTitle = (category: string): string => tMaybe(`optimize.categories.${CATEGORY_KEYS[category] ?? category}`) ?? category

/** A unit as shown (`rel.` is a word; the rest are symbols). */
export const operandUnitText = (unit: string): string => (unit === 'rel.' ? t('optimize.relativeUnit') : unit)

const paramText = (key: string) => tMaybe(`optimize.params.${key}`) ?? key
const param = (key: string, kind: ParamKind, rest: Partial<OperandParam> = {}): OperandParam => ({
  key,
  get title() {
    return paramText(key)
  },
  kind,
  ...rest,
})
const n = (key: string, unit = '', def = 0, required = true): OperandParam =>
  param(key, { type: 'number', unit }, { required, defaultValue: required ? def : undefined })
const surface = (key = 'surface', required = false): OperandParam => param(key, { type: 'surface' }, { required })
const choice = (key: string, options: string[], def?: string): OperandParam => param(key, { type: 'choice', options }, { defaultValue: def })
const wl = (key: string, def: number): OperandParam => param(key, { type: 'wavelength' }, { required: true, defaultValue: def })
const pupil = [n('px', '', 0), n('py', '', 1)]
const sampling: OperandParam = param('sampling', { type: 'sampling' }, { defaultValue: { rings: 3, arms: 6 } })

function d(type: string, category: string, unit: string, params: OperandParam[] = [], singleField = false): OperandDescriptor {
  return {
    type,
    get title() {
      return tMaybe(`optimize.operands.${type}.title`) ?? type
    },
    category,
    unit,
    get summary() {
      return tMaybe(`optimize.operands.${type}.summary`) ?? ''
    },
    params,
    singleField,
  }
}

const FO = 'First order', GE = 'Geometry', RR = 'Real rays', IQ = 'Image quality', GS = 'Ghosts and stray light', MA = 'Manufacturing', CO = 'Composition'

/** The engine's operands (docs/OPTIMIZATION.md, operand reference); their names
 *  and summaries are in the UI catalog (`optimize.operands`). */
export const OPERANDS: OperandDescriptor[] = [
  d('efl', FO, 'mm'),
  d('efl_x', FO, 'mm'),
  d('efl_y', FO, 'mm'),
  d('bfl', FO, 'mm'),
  d('bfl_x', FO, 'mm'),
  d('bfl_y', FO, 'mm'),
  d('anamorphic_squeeze', FO, ''),
  d('f_number', FO, '', [choice('kind', ['paraxial_image_space', 'paraxial_working', 'real_working'], 'paraxial_image_space')]),
  d('magnification', FO, ''),
  d('entrance_pupil_diameter', FO, 'mm'),
  d('entrance_pupil_position', FO, 'mm'),
  d('exit_pupil_position', FO, 'mm'),
  d('paraxial_ray', FO, 'mm / – / °', [choice('ray', ['marginal', 'chief'], 'marginal'), choice('quantity', ['height', 'slope', 'angle'], 'height'), surface()]),
  d('petzval_sum', FO, '1/mm'),
  d('axial_colour', FO, 'mm', [wl('short', 0), wl('long', 2)]),
  d('total_track', GE, 'mm', [surface('from'), surface('to')]),
  d('centre_thickness', GE, 'mm', [surface('surface', true), surface('to')]),
  d('edge_thickness', GE, 'mm', [surface('surface', true), surface('to'), n('height', 'mm', 0, false)]),
  d('min_thickness', GE, 'mm', [choice('measure', ['centre', 'edge'], 'centre'), choice('medium', ['glass', 'air', 'any'], 'glass'), surface('from'), surface('to')]),
  d('max_thickness', GE, 'mm', [choice('measure', ['centre', 'edge'], 'centre'), choice('medium', ['glass', 'air', 'any'], 'glass'), surface('from'), surface('to')]),
  d(
    'real_ray',
    RR,
    'mm / – / °',
    [
      choice('quantity', ['x', 'y', 'z', 'l', 'm', 'n', 'angle_x', 'angle_y', 'incidence_angle', 'optical_path'], 'y'),
      surface(),
      ...pupil,
      param('global', { type: 'flag' }),
    ],
    true,
  ),
  d('ray_aberration', RR, 'mm', [choice('axis', ['x', 'y'], 'y'), ...pupil, choice('reference', ['chief', 'centroid'], 'chief')], true),
  d('wavefront_aberration', RR, 'waves', [...pupil, choice('reference', ['chief', 'centroid'], 'chief')], true),
  d('chief_ray_angle', RR, '°', [surface()], true),
  d('rms_spot', IQ, 'mm', [choice('reference', ['centroid', 'chief'], 'centroid'), sampling]),
  d('spot_centroid', IQ, 'mm', [choice('axis', ['x', 'y'], 'y')], true),
  d('wavefront_rms', IQ, 'waves', [choice('reference', ['centroid', 'chief'], 'centroid'), sampling]),
  d('mtf', IQ, '', [
    n('frequency', 'cyc/mm', 30),
    choice('orientation', ['tangential', 'sagittal', 'average'], 'average'),
    choice('method', ['diffraction', 'geometric'], 'diffraction'),
    n('defocus_mm', 'mm', 0, false),
  ]),
  d('distortion', IQ, '%', [choice('mapping', ['f_tan_theta', 'f_theta'], 'f_tan_theta')], true),
  d('lateral_colour', IQ, 'mm', [wl('short', 0), wl('long', 2)], true),
  d('relative_illumination', IQ, '', [], true),
  d('ghost_focus', GS, 'mm', [
    param('worst', { type: 'integer', min: 1 }, { defaultValue: 1 }),
    choice('metric', ['distance', 'peak'], 'distance'),
  ]),
  d('ghost_irradiance', GS, 'rel.', [
    choice('metric', ['peak', 'mean', 'power'], 'peak'),
    param('max_reflections', { type: 'integer', min: 2 }, { defaultValue: 2 }),
  ]),
  d('stray_light_power', GS, 'rel.', [
    choice('class', ['ghost', 'scatter', 'mechanical', 'sneak', 'all_non_primary'], 'all_non_primary'),
    param('rays', { type: 'integer', min: 1000 }, { defaultValue: 100000 }),
  ]),
  d('doe_min_zone_period', MA, 'mm', [surface('surface', true)]),
  d('doe_max_phase_slope', MA, 'rad/mm', [surface('surface', true)]),
  d('doe_total_phase', MA, 'rad', [surface('surface', true)]),
  d('q_max_departure', MA, 'mm', [surface('surface', true)]),
  d('q_max_departure_slope', MA, 'mm/mm', [surface('surface', true)]),
  d('freeform_max_departure', MA, 'mm', [surface('surface', true)]),
  d('combine', CO, '', [
    choice('op', ['sum', 'difference', 'product', 'quotient', 'abs', 'min', 'max'], 'sum'),
    param('args', { type: 'operandList' }, { required: true, defaultValue: [] }),
  ]),
  d('param_value', CO, ''),
  d('tolerance_sensitivity', CO, ''),
]

const BY_TYPE = new Map(OPERANDS.map((o) => [o.type, o]))
export const operandDescriptor = (type: string) => BY_TYPE.get(type)

/** Operands a user adds from the picker (those that need a model path or a
 *  nested criterion are read and kept, not added). */
export const ADDABLE_OPERANDS = OPERANDS.filter((o) => o.type !== 'param_value' && o.type !== 'tolerance_sensitivity')

export function operandTitle(type: string): string {
  return BY_TYPE.get(type)?.title ?? type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Descriptors matching a search (the shown names and summaries, the operand
 *  type id, the category), by category. */
export function searchOperands(text: string): { category: string; items: OperandDescriptor[] }[] {
  const q = text.trim().toLowerCase()
  const hits = ADDABLE_OPERANDS.filter(
    (o) => !q || o.title.toLowerCase().includes(q) || o.type.includes(q) || o.summary.toLowerCase().includes(q) || categoryTitle(o.category).toLowerCase().includes(q),
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
  if (desc.singleField && op.field.field === 'all') out.push(t('optimize.issues.needsOneField'))
  if (op.field.field === 'index' && typeof op.field.index === 'number' && op.field.index >= fieldCount) out.push(t('optimize.issues.fieldMissing', { n: op.field.index + 1 }))
  if (op.wavelength.wavelength === 'index' && typeof op.wavelength.index === 'number' && op.wavelength.index >= wavelengthCount)
    out.push(t('optimize.issues.wavelengthMissing', { n: op.wavelength.index + 1 }))
  for (const p of desc.params) if (p.required && (op.kind[p.key] === undefined || op.kind[p.key] === null)) out.push(t('optimize.issues.required', { title: p.title }))
  if (!Number.isFinite(op.weight) || op.weight < 0) out.push(t('optimize.issues.weight'))
  return out
}

export const choiceTitle = (s: string) => {
  const shown = tMaybe(`optimize.choices.${s}`)
  if (shown != null) return shown
  const text = s.replace(/_/g, ' ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/** A short text of an operand's selectors and parameters, field and wavelength first. */
export function operandParameters(op: Operand): string {
  const parts: string[] = []
  const f = op.field
  if (f.field === 'index') parts.push(`F${num(f.index, 0) + 1}`)
  else if (f.field === 'normalized') parts.push(`H(${sig(num(f.hx, 0), 4)}, ${sig(num(f.hy, 0), 4)})`)
  const w = op.wavelength
  if (w.wavelength === 'all') parts.push(t('optimize.parameters.poly'))
  else if (w.wavelength === 'index') parts.push(`λ${num(w.index, 0) + 1}`)
  for (const p of operandDescriptor(op.kind.type)?.params ?? []) {
    const v = op.kind[p.key]
    if (v === undefined || v === null) continue
    switch (p.kind.type) {
      case 'surface':
        parts.push(t('optimize.parameters.surface', { title: p.title.toLowerCase(), n: typeof v === 'number' ? v : '?' }))
        break
      case 'wavelength':
        parts.push(t('optimize.parameters.wavelength', { title: p.title, n: num(v, 0) + 1 }))
        break
      case 'choice':
        parts.push(choiceTitle(String(v)).toLowerCase())
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
        parts.push(t('optimize.parameters.of', { list: Array.isArray(v) ? v.map((x) => `#${num(x, 0) + 1}`).join(', ') : '' }))
        break
      default:
        parts.push(t('optimize.parameters.value', { title: p.title.toLowerCase(), value: typeof v === 'number' ? sig(v, 4) : String(v) }))
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
  { value: 'rms_spot', get title() { return t('optimize.criteria.rmsSpot') } },
  { value: 'wavefront', get title() { return t('optimize.criteria.wavefront') } },
  { value: 'mtf', get title() { return t('optimize.criteria.mtf') } },
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
  let s: string
  if (m.criterion === 'rms_spot') s = t('optimize.criteria.rmsSpot')
  else if (m.criterion === 'wavefront') s = t('optimize.criteria.wavefront')
  else
    s = t(m.worstField ? 'optimize.summary.mtfWeakest' : 'optimize.summary.mtf', {
      target: sig(m.mtfTarget, 4),
      frequencies: m.mtfFrequencies.map((f) => sig(f, 4)).join(', '),
    })
  s += ` · ${t('optimize.summary.sampling', { rings: m.rings, arms: m.arms })}`
  if (m.thickness) s += ` · ${t('optimize.summary.thickness')}`
  if (m.squeezeTarget != null) s += ` · ${t('optimize.summary.squeeze', { value: sig(m.squeezeTarget, 4) })}`
  return s
}

export function defaultMeritCaption(c: Criterion): string {
  switch (c) {
    case 'rms_spot':
      return t('optimize.criteria.rmsSpotCaption')
    case 'wavefront':
      return t('optimize.criteria.wavefrontCaption')
    case 'mtf':
      return t('optimize.criteria.mtfCaption')
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
  if (!e) return t('optimize.evaluation.none')
  const on = !e.reportedOn || e.reportedOn.startsWith('cpu') ? t('optimize.evaluation.cpuReference') : e.reportedOn
  const parts = [t('optimize.evaluation.merit', { merit: sig(e.merit, 4), on })]
  if (Number.isFinite(e.violation) && e.violation > 1e-9) parts.push(t('optimize.evaluation.violation', { value: sig(e.violation, 4) }))
  if (e.status.status === 'degraded') parts.push(t('optimize.evaluation.degraded', { count: e.status.failures }))
  if (e.status.status === 'failed') parts.push(t('optimize.evaluation.failed'))
  return parts.join(currentLanguage() === 'zh-Hans' ? '' : ' ')
}
