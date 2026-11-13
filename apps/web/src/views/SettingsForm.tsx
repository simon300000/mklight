import { useEffect, useState } from 'react'
import type { Json, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { exact, parseNumber, sig } from '@/model/format'
import type { System } from '@/model/system'
import { cn } from '@/lib/utils'

// Settings forms chosen from the engine's settings schema alone
// (`SettingControls`): scalars are number fields with units and bounds,
// toggles, pop-ups and text fields; field and wavelength selections are menus
// of the system's fields and wavelengths; objects with known properties are
// groups, with an on/off switch when null is allowed; number lists are
// comma-separated fields. Settings with no sensible control are left out and
// keep their value.

type Schema = JsonObject

const types = (s: Schema): string[] => (Array.isArray(s.type) ? (s.type as string[]) : s.type ? [String(s.type)] : [])
const allowsNull = (s: Schema) => types(s).includes('null')

/** Keys left out of the form (they keep their value). */
const HIDDEN = new Set(['spectrum', 'configuration', 'include_points', 'route', 'availability'])

function labelOf(key: string, s: Schema): string {
  if (typeof s.title === 'string') return s.title
  const t = key.replace(/_mm$|_um$|_deg$/, '').replace(/_/g, ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}

function isSelection(key: string, s: Schema): 'fields' | 'wavelengths' | null {
  const def = s.default as JsonObject | undefined
  if (!def || typeof def !== 'object' || Array.isArray(def) || typeof def.type !== 'string') return null
  if (/^fields?$/.test(key)) return 'fields'
  if (/^wavelengths?$/.test(key)) return 'wavelengths'
  return null
}

/** A number field that commits on Enter and blur; Escape reverts. */
export function NumberInput({
  value,
  onCommit,
  min,
  max,
  integer,
  nullable,
  className,
  ariaLabel,
  placeholder,
}: {
  value: number | null
  onCommit: (v: number | null) => void
  min?: number
  max?: number
  integer?: boolean
  nullable?: boolean
  className?: string
  ariaLabel?: string
  placeholder?: string
}) {
  const [text, setText] = useState<string | null>(null)
  const shown = text ?? (value == null ? '' : sig(value, 8).replace('−', '-'))
  const commit = () => {
    if (text == null) return
    const t = text.trim()
    setText(null)
    if (t === '' || t.toLowerCase() === 'auto') {
      if (nullable) onCommit(null)
      return
    }
    let v = parseNumber(t)
    if (v == null) return
    if (integer) v = Math.round(v)
    if (min != null && v < min) v = min
    if (max != null && v > max) v = max
    if (v !== value) onCommit(v)
  }
  return (
    <input
      aria-label={ariaLabel}
      inputMode="decimal"
      className={cn(
        'num h-6 w-full min-w-0 rounded-md border border-control-border bg-control px-1.5 text-right text-[13px] outline-none focus:border-accent',
        className,
      )}
      value={shown}
      placeholder={placeholder ?? (nullable ? 'Auto' : '')}
      onFocus={(e) => {
        setText(value == null ? '' : exact(value))
        requestAnimationFrame(() => e.target.select())
      }}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        if (e.key === 'Escape') {
          setText(null)
          requestAnimationFrame(() => (e.target as HTMLInputElement).blur())
        }
      }}
    />
  )
}

function SelectionControl({
  kind,
  value,
  system,
  onChange,
  label,
}: {
  kind: 'fields' | 'wavelengths'
  value: JsonObject
  system: System
  onChange: (v: Json) => void
  label: string
}) {
  const items =
    kind === 'fields'
      ? system.fields.fields.map((f, k) => `Field ${k + 1} (${sig(f.x, 4)}, ${sig(f.y, 4)})`)
      : system.wavelengths.wavelengths.map((w, k) => `${sig(w.um, 5)} µm${k === system.wavelengths.primary ? ' (primary)' : ''}`)
  const cur =
    value.type === 'index' ? `i${value.index}` : value.type === 'list' && Array.isArray(value.indices) && value.indices.length === 1 ? `i${value.indices[0]}` : String(value.type)
  return (
    <Select
      aria-label={label}
      value={cur}
      onChange={(e) => {
        const v = e.target.value
        onChange(v.startsWith('i') ? { type: 'index', index: Number(v.slice(1)) } : { type: v })
      }}
    >
      <option value="all">All</option>
      {kind === 'wavelengths' && <option value="primary">Primary</option>}
      {value.type === 'list' && (value.indices as number[]).length !== 1 && <option value="list">Selected ({(value.indices as number[]).length})</option>}
      {items.map((t, k) => (
        <option key={k} value={`i${k}`}>
          {t}
        </option>
      ))}
    </Select>
  )
}

function Row({ label, unit, help, children }: { label: string; unit?: string; help?: string; children: React.ReactNode }) {
  return (
    <label className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_2.5rem] items-center gap-x-2 py-[3px]" title={help}>
      <span className="truncate text-secondary">{label}</span>
      <span className="flex min-w-0 justify-end">{children}</span>
      <span className="truncate text-[11px] text-secondary">{unit}</span>
    </label>
  )
}

function Control({
  k,
  schema,
  value,
  system,
  onChange,
}: {
  k: string
  schema: Schema
  value: Json | undefined
  system: System
  onChange: (v: Json) => void
}) {
  const t = types(schema)
  const label = labelOf(k, schema)
  const help = typeof schema.description === 'string' ? schema.description : undefined
  const unit = typeof schema.unit === 'string' ? schema.unit : ''
  const sel = isSelection(k, schema)
  if (sel) {
    return (
      <Row label={label} help={help}>
        <SelectionControl kind={sel} label={label} value={(value as JsonObject) ?? (schema.default as JsonObject)} system={system} onChange={onChange} />
      </Row>
    )
  }
  if (Array.isArray(schema.enum)) {
    return (
      <Row label={label} help={help} unit={unit}>
        <Select aria-label={label} value={String(value ?? schema.default)} onChange={(e) => onChange(e.target.value)} className="w-full">
          {(schema.enum as string[]).map((o) => (
            <option key={o} value={o}>
              {o.replace(/_/g, ' ')}
            </option>
          ))}
        </Select>
      </Row>
    )
  }
  if (t.includes('boolean')) {
    return (
      <Row label={label} help={help}>
        <Switch aria-label={label} checked={value === true} onCheckedChange={(c) => onChange(c)} />
      </Row>
    )
  }
  if (t.includes('number') || t.includes('integer')) {
    const min = typeof schema.minimum === 'number' ? schema.minimum : undefined
    const max = typeof schema.maximum === 'number' ? schema.maximum : undefined
    const zeroAuto = /0 selects|0 = auto|0 chooses|0 picks/.test(help ?? '')
    return (
      <Row label={label} help={help} unit={unit}>
        <NumberInput
          ariaLabel={label}
          value={typeof value === 'number' ? (zeroAuto && value === 0 ? null : value) : null}
          integer={t.includes('integer') && !t.includes('number')}
          min={min}
          max={max}
          nullable={allowsNull(schema) || zeroAuto}
          onCommit={(v) => onChange(v == null ? (zeroAuto ? 0 : null) : v)}
        />
      </Row>
    )
  }
  if (t.includes('object') && schema.properties) {
    const props = schema.properties as Record<string, Schema>
    const on = value != null
    const obj = (value as JsonObject) ?? {}
    const defaults: JsonObject = Object.fromEntries(Object.entries(props).map(([pk, ps]) => [pk, ps.default ?? null]))
    return (
      <fieldset className="my-1 rounded-lg border border-separator px-2 py-1" title={help}>
        <legend className="flex items-center gap-2 px-1 text-[12px] font-medium">
          {label}
          {allowsNull(schema) && (
            <Switch aria-label={label} checked={on} onCheckedChange={(c) => onChange(c ? defaults : null)} />
          )}
        </legend>
        {on &&
          Object.entries(props).map(([pk, ps]) => (
            <Control key={pk} k={pk} schema={ps} value={obj[pk]} system={system} onChange={(v) => onChange({ ...obj, [pk]: v })} />
          ))}
      </fieldset>
    )
  }
  if (t.includes('array')) {
    const arr = Array.isArray(value) ? value : Array.isArray(schema.default) ? (schema.default as Json[]) : []
    if (arr.every((x) => typeof x === 'number')) {
      return (
        <Row label={label} help={help} unit={unit}>
          <ListInput value={arr as number[]} nullable={allowsNull(schema)} isNull={value === null} onCommit={onChange} label={label} />
        </Row>
      )
    }
    return null
  }
  if (t.includes('string')) {
    return (
      <Row label={label} help={help}>
        <input
          aria-label={label}
          className="h-6 w-full rounded-md border border-control-border bg-control px-1.5 text-[13px] outline-none focus:border-accent"
          defaultValue={String(value ?? '')}
          onBlur={(e) => e.target.value !== value && onChange(e.target.value)}
        />
      </Row>
    )
  }
  return null
}

function ListInput({ value, onCommit, nullable, isNull, label }: { value: number[]; onCommit: (v: Json) => void; nullable: boolean; isNull: boolean; label: string }) {
  const [text, setText] = useState<string | null>(null)
  const shown = text ?? (isNull ? '' : value.map((v) => sig(v, 6).replace('−', '-')).join(', '))
  return (
    <input
      aria-label={label}
      className="num h-6 w-full rounded-md border border-control-border bg-control px-1.5 text-right text-[13px] outline-none focus:border-accent"
      value={shown}
      placeholder={nullable ? 'Auto' : ''}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        if (text == null) return
        setText(null)
        if (text.trim() === '' && nullable) return onCommit(null)
        const nums = text.split(/[;,\s]+/).filter(Boolean).map(parseNumber)
        if (nums.every((n) => n != null && Number.isFinite(n))) onCommit(nums as number[])
      }}
      onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
    />
  )
}

/** The values that differ from a new pane's, for the header's summary line. */
export function settingsSummary(defaults: JsonObject | null, own: JsonObject): string {
  const parts: string[] = []
  for (const [k, v] of Object.entries(own)) {
    if (HIDDEN.has(k)) continue
    if (JSON.stringify(v) === JSON.stringify(defaults?.[k])) continue
    if (v && typeof v === 'object' && !Array.isArray(v) && 'type' in v) {
      const o = v as JsonObject
      parts.push(o.type === 'index' ? `${k.replace(/s$/, '')} ${Number(o.index) + 1}` : `${k} ${o.type}`)
    } else if (typeof v === 'number') parts.push(`${k.replace(/_/g, ' ')} ${sig(v, 4)}`)
    else if (typeof v === 'string' || typeof v === 'boolean') parts.push(`${k.replace(/_/g, ' ')} ${String(v).replace(/_/g, ' ')}`)
    else if (v === null) parts.push(`${k.replace(/_/g, ' ')} off`)
    else parts.push(k.replace(/_/g, ' '))
  }
  return parts.join(' · ')
}

export function SettingsForm({
  schema,
  defaults,
  value,
  system,
  onChange,
  onDone,
}: {
  schema: Schema
  defaults: JsonObject | null
  value: JsonObject
  system: System
  onChange: (v: JsonObject) => void
  onDone: () => void
}) {
  const props = (schema.properties as Record<string, Schema>) ?? {}
  const keys = Object.keys(props)
    .filter((k) => !HIDDEN.has(k))
    .sort((a, b) => {
      // Selections first, then scalars, then groups.
      const rank = (k: string) => (isSelection(k, props[k]) ? 0 : props[k].properties ? 2 : 1)
      return rank(a) - rank(b) || a.localeCompare(b)
    })
  const eff = { ...(defaults ?? {}), ...value }
  const [, force] = useState(0)
  useEffect(() => force((n) => n + 1), [system])
  return (
    <div className="max-h-[45%] shrink-0 overflow-y-auto border-b border-separator bg-window/60 px-3 py-2" data-testid="settings-form">
      <div className="rounded-lg bg-content px-2.5 py-1.5 shadow-[0_0_0_0.5px_var(--separator)]">
        {keys.length === 0 && <p className="py-1 text-secondary">This analysis has no settings.</p>}
        {keys.map((k) => (
          <Control key={k} k={k} schema={props[k]} value={eff[k]} system={system} onChange={(v) => onChange({ ...value, [k]: v })} />
        ))}
      </div>
      <div className="mt-2 flex justify-end gap-2">
        <Button size="sm" onClick={() => onChange({})}>
          Defaults
        </Button>
        <Button size="sm" variant="prominent" onClick={onDone}>
          Done
        </Button>
      </div>
    </div>
  )
}
