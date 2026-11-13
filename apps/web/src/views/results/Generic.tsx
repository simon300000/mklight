// Any result as data: key/value lists, tables of same-shaped rows, collapsible
// sections for nested objects, and the buffers by name, dtype and shape.
import type { ReactNode } from 'react'
import type { AnalysisResult, Json, JsonObject, TypedArray } from '@/engine/types'
import { nf, sig } from '@/model/format'
import { cn } from '@/lib/utils'
import { arr, isObj } from './common'

const MAX_ROWS = 400
const INLINE = 12

const isScalar = (v: Json | undefined) => v === null || typeof v !== 'object'
const isScalarArray = (v: Json | undefined): v is Json[] => Array.isArray(v) && v.every(isScalar)

/** "rms_radius" → "Rms radius"; keeps short codes (s1, w040) as they are. */
export function humanize(key: string): string {
  if (/^[a-z]\d+[a-z]?$/i.test(key)) return key
  const s = key.replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function scalarText(v: Json | undefined): string {
  if (v === undefined || v === null) return '–'
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  if (typeof v === 'number') return Number.isInteger(v) && Math.abs(v) < 1e9 ? String(v).replace('-', '−') : sig(v, 6)
  if (typeof v === 'string') return ['inf', '-inf', 'nan', 'Infinity', '-Infinity'].includes(v) ? sig(nf(v)) : v
  return ''
}

function inlineText(v: Json | undefined): string {
  if (isScalarArray(v)) {
    const shown = v.length > INLINE ? v.slice(0, 8) : v
    const s = shown.map(scalarText).join(', ')
    return v.length > INLINE ? `${s}, … (${v.length} values)` : `[${s}]`
  }
  if (Array.isArray(v)) return `${v.length} items`
  return scalarText(v)
}

function KeyValues(props: { entries: [string, Json][] }) {
  if (!props.entries.length) return null
  return (
    <table className="mb-2 text-[12px]">
      <tbody>
        {props.entries.map(([k, v]) => (
          <tr key={k}>
            <td className="py-0.5 pr-4 align-top text-secondary">{humanize(k)}</td>
            <td className={cn('py-0.5 break-all text-label', typeof v !== 'string' && 'num')}>{inlineText(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Section(props: { title: string; open: boolean; children: ReactNode; note?: string }) {
  return (
    <details open={props.open} className="mb-1">
      <summary className="cursor-default py-0.5 text-[12px] font-medium text-label select-none">
        {props.title}
        {props.note && <span className="ml-2 font-normal text-secondary">{props.note}</span>}
      </summary>
      <div className="border-l border-separator pl-3">{props.children}</div>
    </details>
  )
}

/** Columns of a row table: keys holding scalars or scalar arrays in some row. */
function columnsOf(rows: JsonObject[]): string[] {
  const cols: string[] = []
  for (const r of rows.slice(0, 50))
    for (const [k, v] of Object.entries(r)) if ((isScalar(v) || isScalarArray(v)) && !cols.includes(k)) cols.push(k)
  return cols
}

function RowTable(props: { rows: JsonObject[] }) {
  const cols = columnsOf(props.rows)
  const shown = props.rows.slice(0, MAX_ROWS)
  if (!cols.length) return <div className="text-[12px] text-secondary">{props.rows.length} rows</div>
  return (
    <div className="mb-2 overflow-x-auto">
      <table className="text-[12px]">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c} className="border-b border-separator px-2 py-0.5 text-left font-normal whitespace-nowrap text-secondary">
                {humanize(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c} className={cn('px-2 py-0.5 whitespace-nowrap text-label', typeof r[c] === 'number' && 'num text-right')}>
                  {inlineText(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {props.rows.length > MAX_ROWS && <div className="text-[11px] text-secondary">… {props.rows.length - MAX_ROWS} more rows</div>}
    </div>
  )
}

function Matrix(props: { rows: Json[][] }) {
  const shown = props.rows.slice(0, MAX_ROWS)
  return (
    <div className="mb-2 overflow-x-auto">
      <table className="num text-[12px]">
        <tbody>
          {shown.map((r, i) => (
            <tr key={i}>
              {r.map((v, j) => (
                <td key={j} className="px-2 py-0.5 text-right whitespace-nowrap text-label">
                  {isScalar(v) ? scalarText(v) : inlineText(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ArrayBlock(props: { name: string; value: Json[]; depth: number }) {
  const v = props.value
  const open = props.depth < 1 && v.length <= 200
  const note = `${v.length} ${v.length === 1 ? 'item' : 'items'}`
  if (v.length && v.every(isObj)) {
    return (
      <Section title={humanize(props.name)} open={open} note={note}>
        <RowTable rows={v} />
        {/* Nested parts of rows (shapes, frames, …) one row at a time. */}
        {v.some((r) => Object.values(r).some((x) => !isScalar(x) && !isScalarArray(x))) &&
          v.slice(0, 50).map((r, i) => {
            const nested = Object.entries(r).filter(([, x]) => !isScalar(x) && !isScalarArray(x))
            if (!nested.length) return null
            return (
              <Section key={i} title={`${humanize(props.name)} ${i}`} open={false}>
                <JsonBlock value={Object.fromEntries(nested)} depth={props.depth + 2} />
              </Section>
            )
          })}
      </Section>
    )
  }
  if (v.length && v.every(isScalarArray))
    return (
      <Section title={humanize(props.name)} open={open} note={note}>
        <Matrix rows={v as Json[][]} />
      </Section>
    )
  return (
    <Section title={humanize(props.name)} open={open} note={note}>
      {v.slice(0, MAX_ROWS).map((x, i) =>
        isScalar(x) || isScalarArray(x) ? (
          <div key={i} className="num text-[12px] text-label">
            {inlineText(x)}
          </div>
        ) : (
          <JsonBlock key={i} name={`${i}`} value={x} depth={props.depth + 1} />
        ),
      )}
    </Section>
  )
}

export function JsonBlock(props: { value: Json; depth?: number; name?: string }): ReactNode {
  const depth = props.depth ?? 0
  const v = props.value
  if (!isObj(v)) {
    if (Array.isArray(v) && !isScalarArray(v)) return <ArrayBlock name={props.name ?? 'Items'} value={v} depth={depth} />
    const content = <div className="num text-[12px] text-label">{inlineText(v)}</div>
    return props.name ? <KeyValues entries={[[props.name, v]]} /> : content
  }
  const entries = Object.entries(v)
  const scalars = entries.filter(([, x]) => isScalar(x) || isScalarArray(x))
  const rest = entries.filter(([, x]) => !(isScalar(x) || isScalarArray(x)))
  const body = (
    <>
      <KeyValues entries={scalars} />
      {rest.map(([k, x]) =>
        Array.isArray(x) ? (
          <ArrayBlock key={k} name={k} value={x} depth={depth} />
        ) : (
          <Section key={k} title={humanize(k)} open={depth < 1}>
            <JsonBlock value={x} depth={depth + 1} />
          </Section>
        ),
      )}
    </>
  )
  return props.name !== undefined ? (
    <Section title={humanize(props.name)} open={depth < 1}>
      {body}
    </Section>
  ) : (
    body
  )
}

const DTYPES: [new (...a: never[]) => TypedArray, string][] = [
  [Float64Array, 'f64'],
  [Float32Array, 'f32'],
  [Int32Array, 'i32'],
  [Uint32Array, 'u32'],
  [Uint8Array, 'u8'],
  [BigInt64Array, 'i64'],
  [BigUint64Array, 'u64'],
]
export const dtypeOf = (b: TypedArray) => DTYPES.find(([T]) => b instanceof T)?.[1] ?? '?'

function Buffers(props: { result: AnalysisResult }) {
  const names = Object.keys(props.result.buffers ?? {})
  if (!names.length) return null
  return (
    <Section title="Buffers" open={false} note={`${names.length}`}>
      <table className="text-[12px]">
        <tbody>
          {names.map((n) => {
            const b = props.result.buffers[n]
            const shape = props.result.shapes?.[n] ?? [b.length]
            return (
              <tr key={n}>
                <td className="py-0.5 pr-4 font-mono text-label">{n}</td>
                <td className="py-0.5 pr-4 text-secondary">{dtypeOf(b)}</td>
                <td className="num py-0.5 text-secondary">[{shape.join(' × ')}]</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </Section>
  )
}

const SEVERITY: Record<string, string> = { error: 'text-danger', warning: 'text-warning', info: 'text-secondary' }

function Issues(props: { json: JsonObject }) {
  const issues = arr(props.json.issues).filter(isObj)
  const ok = props.json.ok === true
  return (
    <div className="mb-3">
      <div className={cn('mb-1 text-[13px]', ok ? 'text-label' : 'text-danger')}>
        {ok ? 'The system is valid.' : 'The system has errors.'}
        <span className="num ml-2 text-[12px] text-secondary">
          {scalarText(props.json.errors ?? 0)} errors · {scalarText(props.json.warnings ?? 0)} warnings
        </span>
      </div>
      {issues.length > 0 && (
        <ul className="text-[12px]">
          {issues.map((i, k) => {
            const sev = String(i.severity ?? 'info')
            return (
              <li key={k} className="flex gap-2 py-0.5">
                <span className={cn('w-14 shrink-0 capitalize', SEVERITY[sev] ?? 'text-secondary')}>{sev}</span>
                <span className="text-label">
                  {String(i.message ?? i.code ?? '')}
                  {typeof i.surface === 'number' && <span className="text-secondary"> · surface {i.surface}</span>}
                  {typeof i.code === 'string' && <span className="ml-1 font-mono text-[11px] text-tertiary">{i.code}</span>}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export function GenericView(props: { analysis: string; result: AnalysisResult }) {
  const json = isObj(props.result.json) ? props.result.json : ({ value: props.result.json ?? null } as JsonObject)
  const validate = props.analysis === 'validate' && Array.isArray(json.issues)
  const rest: JsonObject = validate
    ? Object.fromEntries(Object.entries(json).filter(([k]) => !['issues', 'ok', 'errors', 'warnings'].includes(k)))
    : json
  return (
    <div className="absolute inset-0 overflow-auto p-3 select-text">
      {validate && <Issues json={json} />}
      <JsonBlock value={rest} />
      <Buffers result={props.result} />
    </div>
  )
}
