import { ChevronsUpDown, Minus, Plus } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { parseNumber, sig } from '@/model/format'
import { parseSpectrum, spectrumSummary } from '@/model/spectrum'
import type { System } from '@/model/system'
import { autoDisplay, autoUsed, isAuto } from './analysis/autoSettings'
import { NumberInput } from './analysis/NumberInput'
import {
  allowsNull,
  controlFor,
  enabledValue,
  pairLabels,
  sentinelAt,
  types,
  variantDefault,
  type Column,
  type IndexDomain,
  type Schema,
  type SettingControl,
} from './analysis/settingControl'
import { SpectrumButton } from './analysis/SpectrumControl'

export { NumberInput }

// Settings forms chosen from the engine's settings schema alone
// (`SettingControls`, `SettingsBar`): scalars are number fields with units and
// bounds, toggles, pop-ups and text fields; field and wavelength selections and
// index settings are menus of the system's fields, wavelengths and surfaces;
// two-number arrays are pairs (with an automatic state when null is allowed);
// integer lists are check-mark menus (empty: all); number lists are
// comma-separated; arrays of same-shaped rows are small tables in a popover;
// objects are groups (an on/off switch when null is allowed); tagged unions are
// a picker of the variants and the chosen variant's rows; the spectrum, the
// image source and the sensor CRA limit have dedicated controls. Automatic
// settings read "Auto (n)" with the value the last run used. Settings with no
// sensible control are left out and keep their value.

interface Ctx {
  analysis: string
  system: System
  result: AnalysisResult | null | undefined
  pattern: string | null
  hasConfigurations: boolean
}

function labelOf(key: string, s: Schema): string {
  if (typeof s.title === 'string') return s.title
  const t = key.replace(/_mm$|_um$|_deg$|_k$/, '').replace(/_/g, ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}

const unitOf = (s: Schema) => (typeof s.unit === 'string' ? s.unit : '')

function fieldItems(system: System): string[] {
  return (system.fields?.fields ?? []).map((f, k) => `Field ${k + 1} (${sig(f.x, 4)}, ${sig(f.y, 4)})`)
}
function wavelengthItems(system: System): string[] {
  return (system.wavelengths?.wavelengths ?? []).map((w, k) => `${sig(w.um, 5)} µm${k === system.wavelengths.primary ? ' (primary)' : ''}`)
}
function surfaceItems(system: System): string[] {
  return (system.surfaces ?? []).map((s, k) => `Surface ${k}${s.label ? ` · ${s.label}` : ''}`)
}
const itemsOf = (d: IndexDomain, system: System) => (d === 'fields' ? fieldItems(system) : d === 'wavelengths' ? wavelengthItems(system) : surfaceItems(system))
const shortItem = (d: IndexDomain, k: number) => (d === 'fields' ? `Field ${k + 1}` : d === 'wavelengths' ? `λ${k + 1}` : `Surface ${k}`)

function SelectionControl({ kind, value, system, onChange, label }: { kind: 'fields' | 'wavelengths'; value: JsonObject; system: System; onChange: (v: Json) => void; label: string }) {
  const items = kind === 'fields' ? fieldItems(system) : wavelengthItems(system)
  const indices = Array.isArray(value.indices) ? (value.indices as number[]) : []
  const cur = value.type === 'index' ? `i${value.index}` : value.type === 'list' && indices.length === 1 ? `i${indices[0]}` : String(value.type)
  return (
    <Select
      aria-label={label}
      className="w-full"
      value={cur}
      onChange={(e) => {
        const v = e.target.value
        onChange(v.startsWith('i') ? { type: 'index', index: Number(v.slice(1)) } : { type: v })
      }}
    >
      <option value="all">All</option>
      {kind === 'wavelengths' && <option value="primary">Primary</option>}
      {value.type === 'list' && indices.length !== 1 && <option value="list">{indices.map((k) => shortItem(kind, k)).join(', ') || 'None'}</option>}
      {items.map((t, k) => (
        <option key={k} value={`i${k}`}>
          {t}
        </option>
      ))}
    </Select>
  )
}

function Row({ label, unit, help, children, block }: { label: ReactNode; unit?: string; help?: string; children: ReactNode; block?: boolean }) {
  const Tag = block ? 'div' : 'label'
  return (
    <Tag className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_2.5rem] items-center gap-x-2 py-[3px] @max-[300px]:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_0]" title={help}>
      <span className="truncate text-secondary">{label}</span>
      <span className="flex min-w-0 justify-end">{children}</span>
      <span className="truncate text-[11px] text-secondary">{unit}</span>
    </Tag>
  )
}

const menuButton =
  'flex h-6 w-full min-w-0 cursor-default items-center justify-end gap-1 rounded-md border border-control-border bg-control px-2 text-[13px] hover:bg-hover'

function IndexListMenu({ domain, value, system, onChange, label }: { domain: IndexDomain; value: number[]; system: System; onChange: (v: Json) => void; label: string }) {
  const items = itemsOf(domain, system)
  const text = value.length === 0 ? 'All' : value.map((k) => shortItem(domain, k)).join(', ')
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button type="button" className={menuButton} aria-label={`${label}: ${text}`}>
          <span className="truncate">{text}</span>
          <ChevronsUpDown className="size-3 shrink-0 text-secondary" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuCheckboxItem checked={value.length === 0} onSelect={(e) => (e.preventDefault(), onChange([]))}>
          All
        </DropdownMenuCheckboxItem>
        <DropdownMenuSeparator />
        {items.map((t, k) => (
          <DropdownMenuCheckboxItem
            key={k}
            checked={value.includes(k)}
            onSelect={(e) => {
              e.preventDefault()
              onChange(value.includes(k) ? value.filter((x) => x !== k) : [...value, k].sort((a, b) => a - b))
            }}
          >
            {t}
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ListInput({ value, onCommit, nullable, isNull, label, placeholder }: { value: number[]; onCommit: (v: Json) => void; nullable: boolean; isNull: boolean; label: string; placeholder?: string }) {
  const [text, setText] = useState<string | null>(null)
  const shown = text ?? (isNull ? '' : value.map((v) => sig(v, 6).replace('−', '-')).join(', '))
  return (
    <input
      aria-label={label}
      className="num h-6 w-full min-w-0 rounded-md border border-control-border bg-control px-1.5 text-right text-[13px] outline-none focus:border-accent"
      value={shown}
      placeholder={placeholder ?? (nullable ? 'Auto' : '')}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        if (text == null) return
        setText(null)
        const t = text.trim()
        if (t === '' || t.toLowerCase() === 'auto') return onCommit(nullable ? null : [])
        const nums = t.split(/[;,\s]+/).filter(Boolean).map(parseNumber)
        if (nums.every((n) => n != null && Number.isFinite(n))) onCommit(nums as number[])
      }}
      onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
    />
  )
}

function PairInput({ name, schema, value, integer, nullable, label, onChange }: { name: string; schema: Schema; value: Json | undefined; integer: boolean; nullable: boolean; label: string; onChange: (v: Json) => void }) {
  const [a, b, sep] = pairLabels(name, schema)
  const cur = Array.isArray(value) && value.length === 2 ? (value as number[]) : null
  const commit = (k: 0 | 1, v: number | null) => {
    if (v == null) return nullable ? onChange(null) : undefined
    const next: [number, number] = cur ? [cur[0], cur[1]] : [v, v]
    next[k] = v
    onChange(next)
  }
  const min = name === 'size' || /grid|bins|pixels/.test(name) ? 1 : undefined
  return (
    <span className="flex w-full min-w-0 items-center gap-1 text-[11px] text-secondary">
      {a && <span>{a}</span>}
      <NumberInput ariaLabel={`${label} ${a || 'first'}`} value={cur?.[0] ?? null} integer={integer} min={min} nullable={nullable} onCommit={(v) => commit(0, v)} />
      {sep.trim() && <span>{sep}</span>}
      {b && <span>{b}</span>}
      <NumberInput ariaLabel={`${label} ${b || 'second'}`} value={cur?.[1] ?? null} integer={integer} min={min} nullable={nullable} onCommit={(v) => commit(1, v)} />
    </span>
  )
}

function TableControl({ columns, rowsAreArrays, value, label, onChange, titles }: { columns: Column[]; rowsAreArrays: boolean; value: Json | undefined; label: string; onChange: (v: Json) => void; titles?: string[] }) {
  const rows = Array.isArray(value) ? value : []
  const cell = (r: Json, c: Column): Json => (rowsAreArrays ? (Array.isArray(r) ? r[Number(c.key)] : null) : r && typeof r === 'object' && !Array.isArray(r) ? r[c.key] : null)
  const setCell = (i: number, c: Column, v: Json) =>
    onChange(
      rows.map((r, k) => {
        if (k !== i) return r
        if (rowsAreArrays) {
          const a = Array.isArray(r) ? [...r] : columns.map((x) => x.schema.default ?? 0)
          a[Number(c.key)] = v
          return a
        }
        return { ...(r && typeof r === 'object' && !Array.isArray(r) ? r : {}), [c.key]: v }
      }),
    )
  const blank = (): Json => (rowsAreArrays ? columns.map((c) => c.schema.default ?? 0) : Object.fromEntries(columns.map((c) => [c.key, c.schema.default ?? 0])))
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button type="button" className={menuButton} aria-label={`${label}: ${rows.length} rows`}>
          <span className="truncate">{rows.length ? `${rows.length} row${rows.length === 1 ? '' : 's'}` : 'None'}</span>
          <ChevronsUpDown className="size-3 shrink-0 text-secondary" />
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-auto min-w-[240px] max-w-[420px]" align="end">
        <h3 className="mb-1.5 text-[13px] font-semibold">{label}</h3>
        <div className="max-h-64 overflow-y-auto">
          <table className="text-[12px]">
            <thead className="text-secondary">
              <tr>
                {columns.map((c, k) => (
                  <th key={c.key} className="px-1 py-0.5 text-left font-normal">
                    {titles?.[k] ?? labelOf(c.key, c.schema)}
                  </th>
                ))}
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {columns.map((c) => {
                    const v = cell(r, c)
                    const t = types(c.schema)
                    return (
                      <td key={c.key} className="w-24 px-1 py-[2px]">
                        {t.includes('boolean') ? (
                          <Switch aria-label={`${c.key} ${i + 1}`} checked={v === true} onCheckedChange={(x) => setCell(i, c, x)} />
                        ) : t.includes('string') ? (
                          <input
                            aria-label={`${c.key} ${i + 1}`}
                            className="h-6 w-full rounded-md border border-control-border bg-control px-1.5 text-[13px]"
                            defaultValue={String(v ?? '')}
                            onBlur={(e) => setCell(i, c, e.target.value)}
                          />
                        ) : (
                          <NumberInput ariaLabel={`${c.key} ${i + 1}`} value={typeof v === 'number' ? v : null} integer={t.includes('integer') && !t.includes('number')} onCommit={(x) => x != null && setCell(i, c, x)} />
                        )}
                      </td>
                    )
                  })}
                  <td>
                    <Button variant="ghost" size="iconSm" aria-label={`Remove row ${i + 1}`} onClick={() => onChange(rows.filter((_, k) => k !== i))}>
                      <Minus />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Button size="sm" variant="ghost" className="mt-1 -ml-1" onClick={() => onChange([...rows, blank()])}>
          <Plus /> Add Row
        </Button>
      </PopoverContent>
    </Popover>
  )
}

/** The engine's built-in image patterns, from the source setting's description. */
export function imagePatterns(s: Schema): string[] {
  const m = /pattern:([a-z_|]+)/.exec(String(s.description ?? ''))
  return m ? m[1].split('|') : ['grid', 'siemens_star', 'slanted_edge', 'text_chart']
}

function ImageSourceControl({ schema, value, label, onChange }: { schema: Schema; value: Json | undefined; label: string; onChange: (v: Json) => void }) {
  const v = value && typeof value === 'object' && !Array.isArray(value) ? value : { kind: 'pattern', pattern: 'grid' }
  const cur = v.kind === 'pattern' ? `p:${v.pattern}` : String(v.kind)
  return (
    <Select aria-label={label} className="w-full" value={cur} onChange={(e) => e.target.value.startsWith('p:') && onChange({ kind: 'pattern', pattern: e.target.value.slice(2) })}>
      {imagePatterns(schema).map((p) => (
        <option key={p} value={`p:${p}`}>
          {p.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())}
        </option>
      ))}
      {v.kind === 'buffer' && <option value="buffer">Image ({String(v.width)} × {String(v.height)})</option>}
      {v.kind === 'path' && <option value="path">{String(v.path ?? 'Image file')}</option>}
    </Select>
  )
}

/** One setting at `path` (`pupil.grid`), with its control from the schema. */
function Control({ k, path, schema, value, ctx, onChange, nested }: { k: string; path: string; schema: Schema; value: Json | undefined; ctx: Ctx; onChange: (v: Json) => void; nested?: boolean }) {
  const c: SettingControl = controlFor(k, schema, { analysis: ctx.analysis, nested, hasConfigurations: ctx.hasConfigurations, pattern: ctx.pattern })
  const label = labelOf(k, schema)
  const help = typeof schema.description === 'string' ? schema.description : undefined
  const unit = unitOf(schema)
  const sentinel = sentinelAt(path, ctx.analysis, schema)
  const usedKey = path.startsWith('through_focus.') ? 'through_focus' : path
  const autoText = sentinel ? autoDisplay(autoUsed(usedKey, ctx.analysis, ctx.result)) : undefined
  switch (c.kind) {
    case 'hidden':
      return null
    case 'fieldSelection':
    case 'wavelengthSelection': {
      const v = value && typeof value === 'object' && !Array.isArray(value) ? value : ((schema.default as JsonObject) ?? { type: 'all' })
      return (
        <Row label={label} help={help}>
          <SelectionControl kind={c.kind === 'fieldSelection' ? 'fields' : 'wavelengths'} label={label} value={v} system={ctx.system} onChange={onChange} />
        </Row>
      )
    }
    case 'index': {
      const items = itemsOf(c.domain, ctx.system)
      const nullable = allowsNull(schema)
      return (
        <Row label={label} help={help}>
          <Select aria-label={label} className="w-full" value={typeof value === 'number' ? String(value) : ''} onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}>
            {(nullable || typeof value !== 'number') && <option value="">{c.domain === 'wavelengths' ? 'Primary' : 'Auto'}</option>}
            {items.map((t, i) => (
              <option key={i} value={i}>
                {t}
              </option>
            ))}
          </Select>
        </Row>
      )
    }
    case 'spectrum':
      return (
        <Row label="Spectrum" help={help} block>
          <SpectrumButton value={value} system={ctx.system} onChange={(v) => onChange(v)} />
        </Row>
      )
    case 'imageSource':
      return (
        <Row label={label} help={help}>
          <ImageSourceControl schema={schema} value={value} label={label} onChange={onChange} />
        </Row>
      )
    case 'sensorLimit':
      return (
        <Row label="Sensor CRA limit" help={help} block>
          <TableControl
            columns={[
              { key: '0', schema: { type: 'number', default: 0 } },
              { key: '1', schema: { type: 'number', default: 30 } },
            ]}
            titles={['Image radius (mm)', 'Max angle (°)']}
            rowsAreArrays
            value={value}
            label="Sensor CRA limit"
            onChange={onChange}
          />
        </Row>
      )
    case 'picker':
      return (
        <Row label={label} help={help} unit={unit}>
          <Select aria-label={label} value={String(value ?? schema.default)} onChange={(e) => onChange(e.target.value)} className="w-full">
            {(schema.enum as string[]).map((o) => (
              <option key={o} value={o}>
                {String(o).replace(/_/g, ' ')}
              </option>
            ))}
          </Select>
        </Row>
      )
    case 'toggle':
      return (
        <Row label={label} help={help}>
          <Switch aria-label={label} checked={value === true} onCheckedChange={(x) => onChange(x)} />
        </Row>
      )
    case 'number': {
      const min = typeof schema.minimum === 'number' ? schema.minimum : undefined
      const max = typeof schema.maximum === 'number' ? schema.maximum : undefined
      const auto = sentinel != null && isAuto(value, sentinel)
      return (
        <Row label={label} help={help} unit={unit}>
          <NumberInput
            ariaLabel={label}
            value={typeof value === 'number' && !auto ? value : null}
            integer={c.integer}
            min={sentinel === 'zero' ? undefined : min}
            max={max}
            nullable={allowsNull(schema) || sentinel != null}
            placeholder={autoText ?? (allowsNull(schema) ? 'Auto' : '')}
            onCommit={(v) => onChange(v == null || (sentinel === 'zero' && v === 0) ? (sentinel === 'zero' ? 0 : null) : v)}
          />
        </Row>
      )
    }
    case 'text':
      return (
        <Row label={label} help={help}>
          <input
            key={String(value ?? '')}
            aria-label={label}
            className="h-6 w-full min-w-0 rounded-md border border-control-border bg-control px-1.5 text-[13px] outline-none focus:border-accent"
            defaultValue={String(value ?? '')}
            onBlur={(e) => e.target.value !== (value ?? '') && onChange(e.target.value)}
          />
        </Row>
      )
    case 'pair':
      return (
        <Row label={label} help={help} unit={unit}>
          <PairInput name={k} schema={schema} value={value} integer={c.integer} nullable={c.nullable} label={label} onChange={onChange} />
        </Row>
      )
    case 'indexList':
      return (
        <Row label={label} help={help}>
          <IndexListMenu domain={c.domain} value={Array.isArray(value) ? (value as number[]) : []} system={ctx.system} onChange={onChange} label={label} />
        </Row>
      )
    case 'numberList': {
      const list = Array.isArray(value) ? (value as number[]) : []
      return (
        <Row label={label} help={help} unit={unit}>
          <ListInput value={list} nullable={allowsNull(schema)} isNull={value === null || (sentinel === 'empty' && list.length === 0)} onCommit={onChange} label={label} placeholder={autoText} />
        </Row>
      )
    }
    case 'table':
      return (
        <Row label={label} help={help} block>
          <TableControl columns={c.columns} rowsAreArrays={c.rowsAreArrays} value={value} label={label} onChange={onChange} />
        </Row>
      )
    case 'group':
      return <GroupControl k={k} path={path} schema={schema} properties={c.properties} nullable={c.nullable} value={value} ctx={ctx} onChange={onChange} label={label} help={help} />
    case 'variants': {
      const on = value != null
      const o = value && typeof value === 'object' && !Array.isArray(value) ? value : null
      const tagKey = c.variants[0].tagKey
      const cur = c.variants.find((v) => o && o[tagKey] === v.tag) ?? c.variants[0]
      return (
        <fieldset className="my-1 rounded-lg border border-separator px-2 py-1" title={help}>
          <legend className="flex items-center gap-2 px-1 text-[12px] font-medium">
            {label}
            {c.nullable && <Switch aria-label={label} checked={on} onCheckedChange={(x) => onChange(x ? enabledValue(schema) : null)} />}
          </legend>
          {on && (
            <>
              <Row label="Kind">
                <Select aria-label={`${label} kind`} className="w-full" value={cur.tag} onChange={(e) => onChange(variantDefault(c.variants.find((v) => v.tag === e.target.value)!))}>
                  {c.variants.map((v) => (
                    <option key={v.tag} value={v.tag}>
                      {v.title}
                    </option>
                  ))}
                </Select>
              </Row>
              {Object.entries(cur.properties).map(([pk, ps]) => (
                <Control key={pk} k={pk} path={`${path}.${pk}`} schema={ps} value={o?.[pk] ?? ps.default} ctx={ctx} nested onChange={(v) => onChange({ ...(o ?? variantDefault(cur)), [pk]: v })} />
              ))}
            </>
          )}
        </fieldset>
      )
    }
  }
}

/** Keys of a group shown first; the rest fold under "More" (the compact pupil control). */
const PRIMARY_KEYS: Record<string, string[]> = { pupil: ['grid', 'rays'], optics: ['field_extent', 'field_center', 'psf_size', 'psf_spacing_mm', 'pupil'] }

function GroupControl({ k, path, schema, properties, nullable, value, ctx, onChange, label, help }: {
  k: string
  path: string
  schema: Schema
  properties: Record<string, Schema>
  nullable: boolean
  value: Json | undefined
  ctx: Ctx
  onChange: (v: Json) => void
  label: string
  help?: string
}) {
  const on = value != null
  const o = value && typeof value === 'object' && !Array.isArray(value) ? value : {}
  const primary = PRIMARY_KEYS[k]
  const entries = Object.entries(properties)
  const first = primary ? entries.filter(([pk]) => primary.includes(pk)) : entries
  const rest = primary ? entries.filter(([pk]) => !primary.includes(pk)) : []
  const row = ([pk, ps]: [string, Schema]) => (
    <Control key={pk} k={pk} path={`${path}.${pk}`} schema={ps} value={o[pk] ?? (on ? ps.default : undefined)} ctx={ctx} nested onChange={(v) => onChange({ ...o, [pk]: v })} />
  )
  return (
    <fieldset className="my-1 min-w-0 rounded-lg border border-separator px-2 py-1" title={help}>
      <legend className="flex items-center gap-2 px-1 text-[12px] font-medium">
        {label}
        {nullable && <Switch aria-label={label} checked={on} onCheckedChange={(x) => onChange(x ? enabledValue(schema) : null)} />}
      </legend>
      {on && first.map(row)}
      {on && rest.length > 0 && (
        <details className="group">
          <summary className="cursor-default py-[3px] text-[12px] text-secondary">More</summary>
          {rest.map(row)}
        </details>
      )}
    </fieldset>
  )
}

// MARK: Summary

const human = (k: string) => k.replace(/_(mm|um|deg)$/, '').replace(/_/g, ' ')

function valueSummary(k: string, v: Json, d: Json | undefined): string | null {
  if (k === 'spectrum') {
    const s = parseSpectrum(v)
    return s && s.kind !== 'design' ? `Spectrum ${spectrumSummary(s)}` : null
  }
  if (v === null) return `${human(k)} ${d === null || d === undefined ? 'auto' : 'off'}`
  if (k === 'through_focus' && typeof v === 'object' && !Array.isArray(v)) {
    const steps = typeof v.steps === 'number' ? v.steps : 3
    const lo = typeof v.min === 'number' ? v.min : null, hi = typeof v.max === 'number' ? v.max : null
    const range = lo != null && hi != null ? (Math.abs(lo + hi) < 1e-12 ? `±${sig(hi, 3)} mm` : `${sig(lo, 3)} to ${sig(hi, 3)} mm`) : 'auto range'
    return `${range} · ${steps} plane${steps === 1 ? '' : 's'}`
  }
  if (typeof v === 'object' && !Array.isArray(v)) {
    if (typeof v.type === 'string') {
      if (v.type === 'index') return `${human(k).replace(/s$/, '')} ${Number(v.index) + 1}`
      if (v.type === 'list' && Array.isArray(v.indices)) return `${human(k)} ${(v.indices as number[]).map((x) => x + 1).join(', ')}`
      return `${human(k)} ${String(v.type).replace(/_/g, ' ')}`
    }
    if (v.kind === 'pattern') return String(v.pattern).replace(/_/g, ' ')
    if (v.kind === 'buffer') return 'image'
    // Groups: the changed members.
    const dd = d && typeof d === 'object' && !Array.isArray(d) ? d : {}
    const parts = Object.entries(v)
      .filter(([kk, vv]) => JSON.stringify(vv) !== JSON.stringify(dd[kk]))
      .map(([kk, vv]) => valueSummary(kk, vv, dd[kk]))
      .filter(Boolean)
    return parts.length ? parts.join(' · ') : human(k)
  }
  if (Array.isArray(v)) {
    if (!v.length) return `${human(k)} all`
    if (v.every((x) => typeof x === 'number')) {
      if (/^(fields|wavelengths)$/.test(k)) return `${human(k)} ${v.map((x) => (x as number) + 1).join(', ')}`
      if (v.length === 2 && /grid|bins|pixels|size/.test(k)) return `${human(k)} ${v[0]} × ${v[1]}`
      return `${human(k)} ${v.map((x) => sig(x as number, 4)).join(', ')}`
    }
    return `${human(k)} ${v.length}`
  }
  if (typeof v === 'number') return `${human(k)} ${v === 0 && d === 0 ? 'auto' : sig(v, 4)}`
  if (typeof v === 'boolean') return v ? human(k) : `no ${human(k)}`
  return `${human(k)} ${String(v).replace(/_/g, ' ')}`
}

/** The values that differ from a new pane's, for the header's summary line. */
export function settingsSummary(defaults: JsonObject | null, own: JsonObject): string {
  const parts: string[] = []
  for (const [k, v] of Object.entries(own)) {
    if (['configuration', 'include_points', 'route', 'availability', 'paths', 'metadata'].includes(k)) continue
    if (JSON.stringify(v) === JSON.stringify(defaults?.[k])) continue
    const s = valueSummary(k, v, defaults?.[k])
    if (s) parts.push(s)
  }
  return parts.join(' · ')
}

// MARK: Form

export function SettingsForm({
  analysis,
  schema,
  defaults,
  value,
  system,
  result,
  onChange,
  onDone,
}: {
  analysis?: string
  schema: Schema
  defaults: JsonObject | null
  value: JsonObject
  system: System
  /** The pane's last result, for "Auto (n)". */
  result?: AnalysisResult | null
  onChange: (v: JsonObject) => void
  onDone: () => void
}) {
  const props = (schema.properties as Record<string, Schema>) ?? {}
  const eff: JsonObject = { ...(defaults ?? {}), ...value }
  const name = analysis ?? ''
  const pupil = eff.pupil && typeof eff.pupil === 'object' && !Array.isArray(eff.pupil) ? eff.pupil : null
  const ctx: Ctx = {
    analysis: name,
    system,
    result,
    pattern: typeof eff.pattern === 'string' ? eff.pattern : typeof pupil?.pattern === 'string' ? pupil.pattern : null,
    hasConfigurations: Array.isArray((system as JsonObject).configurations) && ((system as JsonObject).configurations as Json[]).length > 0,
  }
  const rank = (k: string) => {
    if (k === 'spectrum') return 0
    const c = controlFor(k, props[k], { analysis: name, hasConfigurations: ctx.hasConfigurations, pattern: ctx.pattern })
    if (c.kind === 'hidden') return 9
    if (c.kind === 'fieldSelection' || c.kind === 'wavelengthSelection' || c.kind === 'index' || c.kind === 'imageSource') return 1
    if (c.kind === 'group' || c.kind === 'variants') return 3
    return 2
  }
  const keys = Object.keys(props)
    .filter((k) => rank(k) < 9)
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
  return (
    <div className="@container max-h-[45%] shrink-0 overflow-y-auto border-b border-separator bg-window/60 px-3 py-2" data-testid="settings-form">
      <div className="rounded-lg bg-content px-2.5 py-1.5 shadow-[0_0_0_0.5px_var(--separator)]">
        {keys.length === 0 && <p className="py-1 text-secondary">This analysis has no settings.</p>}
        {keys.map((k) => (
          <Control key={k} k={k} path={k} schema={props[k]} value={eff[k]} ctx={ctx} onChange={(v) => onChange(withSetting(value, k, v, defaults))} />
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

/** The pane's own settings with `k` set; a removed spectrum is left out. */
function withSetting(own: JsonObject, k: string, v: Json, defaults: JsonObject | null): JsonObject {
  const next = { ...own }
  if (k === 'spectrum' && v === null && (defaults?.spectrum === undefined || defaults.spectrum === null)) delete next.spectrum
  else next[k] = v
  return next
}
