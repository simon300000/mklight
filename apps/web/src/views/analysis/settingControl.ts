// The control an analysis setting gets, chosen from its schema alone
// (`SettingControls` of the macOS app), so every analysis the engine publishes
// is editable without typed JSON.

import type { Json, JsonObject } from '@/engine/types'
import { autoSentinel } from './autoSettings'

export type Schema = JsonObject
export type IndexDomain = 'fields' | 'wavelengths' | 'surfaces'

export interface Column {
  key: string
  schema: Schema
}

export interface Variant {
  /** The tag's key (`type`, `mode`, `kind`) and value. */
  tagKey: string
  tag: string
  title: string
  properties: Record<string, Schema>
}

export type SettingControl =
  | { kind: 'hidden' }
  | { kind: 'toggle' }
  | { kind: 'number'; integer: boolean }
  | { kind: 'text' }
  | { kind: 'picker' }
  | { kind: 'fieldSelection' }
  | { kind: 'wavelengthSelection' }
  | { kind: 'index'; domain: IndexDomain }
  | { kind: 'pair'; integer: boolean; nullable: boolean }
  | { kind: 'indexList'; domain: IndexDomain }
  | { kind: 'numberList' }
  | { kind: 'table'; columns: Column[]; rowsAreArrays: boolean }
  | { kind: 'group'; nullable: boolean; properties: Record<string, Schema> }
  | { kind: 'variants'; nullable: boolean; variants: Variant[] }
  | { kind: 'spectrum' }
  | { kind: 'imageSource' }
  | { kind: 'sensorLimit' }

export const types = (s: Schema): string[] => (Array.isArray(s.type) ? (s.type as string[]) : typeof s.type === 'string' ? [s.type] : [])
export const allowsNull = (s: Schema) => types(s).includes('null')
const baseTypes = (s: Schema) => types(s).filter((t) => t !== 'null')
const obj = (v: Json | undefined): JsonObject | null => (v !== null && typeof v === 'object' && !Array.isArray(v) ? v : null)

/** `oneOf` alternatives whose tag is a `const` (`schema::variants`). */
export function variantsOf(s: Schema): Variant[] {
  if (!Array.isArray(s.oneOf)) return []
  const out: Variant[] = []
  for (const alt of s.oneOf) {
    const a = obj(alt)
    const props = obj(a?.properties) as Record<string, Schema> | null
    if (!a || !props) return []
    const tagEntry = Object.entries(props).find(([, p]) => obj(p) && 'const' in p)
    if (!tagEntry) return []
    const [tagKey, tagSchema] = tagEntry
    const tag = String(tagSchema.const)
    const rest = Object.fromEntries(Object.entries(props).filter(([k]) => k !== tagKey))
    out.push({ tagKey, tag, title: typeof a.title === 'string' ? a.title : tag.replace(/_/g, ' '), properties: rest })
  }
  return out
}

/** The properties of an object setting: the schema's, else its default's keys. */
export function propertiesOf(s: Schema): Record<string, Schema> {
  const p = obj(s.properties)
  if (p && Object.keys(p).length) return p as Record<string, Schema>
  const d = obj(s.default)
  if (!d) return {}
  return Object.fromEntries(
    Object.entries(d).map(([k, v]) => {
      const t = v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v === 'number' ? (Number.isInteger(v) ? 'integer' : 'number') : typeof v
      return [k, { type: t, default: v } as Schema]
    }),
  )
}

function indexDomain(name: string): IndexDomain | null {
  if (name === 'fields') return 'fields'
  if (name === 'wavelengths') return 'wavelengths'
  if (name === 'surfaces' || name === 'exclude_surfaces') return 'surfaces'
  return null
}

/** The row keys a description states as `[{"hx", "hy"}]`. */
export function rowKeys(description: string): string[] | null {
  const m = /\[\{("[a-z_0-9]+"(,\s*)?)+\}\]/.exec(description)
  if (!m) return null
  const keys = m[0].split(/[^A-Za-z0-9_]+/).filter(Boolean)
  return keys.length ? keys : null
}

const scalar = (s: Schema) => baseTypes(s).every((t) => ['number', 'integer', 'boolean', 'string'].includes(t)) && baseTypes(s).length > 0

function arrayControl(name: string, s: Schema): SettingControl {
  const d = Array.isArray(s.default) ? s.default : null
  const items = obj(s.items)
  const itemType = typeof items?.type === 'string' ? items.type : null
  if (d && d.length === 2 && d.every((x) => typeof x === 'number')) {
    // JSON does not keep 1.0 apart from 1, so whole defaults need the name too.
    const integer = itemType === 'integer' || (itemType !== 'number' && d.every((x) => Number.isInteger(x)) && /grid|bins|pixels|size/.test(name))
    return { kind: 'pair', integer, nullable: allowsNull(s) }
  }
  if ((!d || d.length === 0) && allowsNull(s) && (itemType === 'number' || !items)) return { kind: 'pair', integer: false, nullable: true }
  if (itemType === 'integer') {
    const dom = indexDomain(name)
    return dom ? { kind: 'indexList', domain: dom } : { kind: 'numberList' }
  }
  if (itemType === 'number') return { kind: 'numberList' }
  const itemProps = items ? (obj(items.properties) as Record<string, Schema> | null) : null
  if (itemProps && Object.keys(itemProps).length && Object.values(itemProps).every(scalar))
    return { kind: 'table', columns: Object.entries(itemProps).map(([key, schema]) => ({ key, schema })), rowsAreArrays: false }
  const first = d?.[0]
  if (first !== undefined) {
    const fo = obj(first)
    if (fo && Object.values(fo).every((v) => ['number', 'boolean', 'string'].includes(typeof v)))
      return {
        kind: 'table',
        columns: Object.keys(fo).sort().map((key) => ({ key, schema: { type: typeof fo[key] === 'number' ? 'number' : typeof fo[key], default: fo[key] } as Schema })),
        rowsAreArrays: false,
      }
    if (Array.isArray(first) && first.length && first.every((x) => typeof x === 'number'))
      return { kind: 'table', columns: first.map((x, k) => ({ key: String(k), schema: { type: 'number', default: x } })), rowsAreArrays: true }
    return { kind: 'hidden' }
  }
  const keys = rowKeys(typeof s.description === 'string' ? s.description : '')
  if (keys) return { kind: 'table', columns: keys.map((key) => ({ key, schema: { type: 'number', default: 0 } })), rowsAreArrays: false }
  return { kind: 'hidden' }
}

/** Keys never shown (they keep their value). */
const ALWAYS_HIDDEN = new Set(['delay_ms', 'chunk', 'availability', 'route', 'paths', 'metadata', 'include_points'])

export function controlFor(
  name: string,
  s: Schema,
  opts: { analysis: string; nested?: boolean; hasConfigurations?: boolean; pattern?: string | null } = { analysis: '' },
): SettingControl {
  const t = baseTypes(s)
  const def = obj(s.default)
  const variants = variantsOf(s)
  const isSelection = t.includes('object') && typeof def?.type === 'string' && variants.length === 0
  const integer = t.includes('integer')
  if (!opts.nested) {
    if (ALWAYS_HIDDEN.has(name)) return { kind: 'hidden' }
    if (name === 'fields' && isSelection) return { kind: 'fieldSelection' }
    if ((name === 'wavelengths' || name === 'wavelength') && isSelection) return { kind: 'wavelengthSelection' }
    if (name === 'field' && integer) return { kind: 'index', domain: 'fields' }
    if (['wavelength', 'long_wavelength', 'short_wavelength'].includes(name) && integer) return { kind: 'index', domain: 'wavelengths' }
    if (name === 'surface' && integer) return { kind: 'index', domain: 'surfaces' }
    if (name === 'spectrum' && t.includes('object')) return { kind: 'spectrum' }
    if (name === 'sensor_limit' && t.length === 1 && t[0] === 'array') return { kind: 'sensorLimit' }
    if (name === 'source' && t.length === 1 && t[0] === 'object' && def?.kind === 'pattern') return { kind: 'imageSource' }
    if (name === 'configuration') return opts.hasConfigurations ? { kind: 'number', integer: true } : { kind: 'hidden' }
    if (name === 'seed' && opts.pattern && !['random', 'dithered'].includes(opts.pattern)) return { kind: 'hidden' }
  }
  if (Array.isArray(s.enum)) return { kind: 'picker' }
  if (variants.length) return { kind: 'variants', nullable: allowsNull(s), variants }
  if (t.length === 1 && t[0] === 'boolean') return { kind: 'toggle' }
  if (t.length && t.every((x) => x === 'number' || x === 'integer')) return { kind: 'number', integer: integer && !t.includes('number') }
  if (t.length === 1 && t[0] === 'string') return { kind: 'text' }
  if (t.length === 1 && t[0] === 'array') return arrayControl(name, s)
  if (t.length === 1 && t[0] === 'object') {
    const props = propertiesOf(s)
    return Object.keys(props).length ? { kind: 'group', nullable: allowsNull(s), properties: props } : { kind: 'hidden' }
  }
  return { kind: 'hidden' }
}

/** Whether a number setting reads 0 as automatic (descriptions say so). */
export const zeroIsAuto = (s: Schema) => /0 selects|0 = auto|0 chooses|0 picks|or 0 to choose/.test(String(s.description ?? ''))

/** The automatic sentinel of a setting at `path`, from the known list or the
 *  description. */
export function sentinelAt(path: string, analysis: string, s: Schema): ReturnType<typeof autoSentinel> {
  const known = autoSentinel(path, analysis)
  if (known) return known
  if (zeroIsAuto(s)) return 'zero'
  return null
}

/** The value a nullable setting starts from when switched on. */
export function enabledValue(s: Schema): Json {
  if (s.default !== undefined && s.default !== null) return s.default
  const v = variantsOf(s)[0]
  if (v) return variantDefault(v)
  const props = propertiesOf(s)
  if (Object.keys(props).length) return Object.fromEntries(Object.entries(props).map(([k, p]) => [k, p.default ?? null]))
  const t = baseTypes(s)
  if (t.includes('array')) return [0, 0]
  if (t.includes('boolean')) return true
  if (t.includes('string')) return ''
  return 0
}

export function variantDefault(v: Variant): JsonObject {
  return { [v.tagKey]: v.tag, ...Object.fromEntries(Object.entries(v.properties).map(([k, p]) => [k, p.default ?? null])) }
}

/** Labels of the two numbers of a pair. */
export function pairLabels(name: string, s: Schema): [string, string, string] {
  const d = String(s.description ?? '')
  if (/center|point/.test(name) || /\[hx,\s*hy\]/i.test(d)) return ['Hx', 'Hy', ' ']
  if (/range/.test(name) || /\[min,\s*max\]/i.test(d)) return ['', '', 'to']
  if (/grid|bins|pixels|size/.test(name) || /\[nx,\s*ny\]|\[width,\s*height\]/i.test(d)) return ['', '', '×']
  return ['x', 'y', ' ']
}
