// Report tables: first-order data (with the anamorphic x/y summary), the
// system summary and validation.
import { AlertTriangle, CheckCircle2, Info, XCircle } from 'lucide-react'
import type { Json, JsonObject } from '@/engine/types'
import { sig } from '@/model/format'
import type { System } from '@/model/system'
import { cn } from '@/lib/utils'
import { nums, obj } from '../../results/common'
import type { KeyNumber, Renderer, RendererProps } from '../registry'
import { intOf, length, nf, objects, short, valueUnit } from './kit'

// MARK: First order

interface XYRow {
  wavelength: number
  um: number
  eflX: number
  eflY: number
  bflX: number
  bflY: number
  squeeze: number
  doe: { surface: number; power: number }[]
}

const NON_ROTATIONAL = new Set(['toroid', 'biconic', 'xy_polynomial', 'zernike_sag'])
const SYMMETRY: Record<string, string> = { rotational: 'Rotational', meridional: 'Meridional (y–z plane)', none: 'None' }

export function paraxialXY(j: JsonObject) {
  const p = obj(j.paraxial_xy)
  if (!p) return null
  const rows: XYRow[] = objects(p.wavelengths).map((w, k) => ({
    wavelength: intOf(w.wavelength, k),
    um: nf(w.wavelength_um),
    eflX: nf(w.efl_x),
    eflY: nf(w.efl_y),
    bflX: nf(w.bfl_x),
    bflY: nf(w.bfl_y),
    squeeze: nf(w.squeeze),
    doe: objects(w.doe_power).map((d) => ({ surface: intOf(d.surface, 0), power: nf(d.power) })),
  }))
  const pw = typeof p.primary_wavelength === 'number' ? p.primary_wavelength : null
  const primary = rows.find((r) => r.wavelength === pw) ?? rows[0] ?? null
  return {
    rows,
    primary,
    others: primary ? rows.filter((r) => r !== primary) : [],
    symmetry: typeof p.symmetry === 'string' ? p.symmetry : null,
    squeezeDefinition: typeof p.squeeze_definition === 'string' ? p.squeeze_definition : null,
    warnings: (Array.isArray(p.warnings) ? p.warnings : []).filter((w): w is string => typeof w === 'string'),
  }
}

/** The engine's symmetry class when it reports one; else the surface kinds decide. */
export function isAnamorphic(j: JsonObject, system?: System): boolean {
  const s = paraxialXY(j)?.symmetry ?? (typeof j.symmetry === 'string' ? j.symmetry : null)
  if (s) return s !== 'rotational'
  return (system?.surfaces ?? []).some((x) => NON_ROTATIONAL.has(String(x.shape?.type)))
}

const squeezeText = (v: number) => (Number.isFinite(v) ? `${v.toFixed(4)}×` : short(v))

function DoePower({ row }: { row: XYRow }) {
  return (
    <div className="grid grid-cols-[auto_auto] justify-start gap-x-6 gap-y-0.5 text-[12px]" title="Phase power of the trace order (base refractive power excluded)">
      {row.doe.map((d) => (
        <div key={d.surface} className="contents">
          <span className="text-secondary">DOE power, surface {d.surface}</span>
          <span className="num text-right">{valueUnit(d.power, '1/mm')}</span>
        </div>
      ))}
    </div>
  )
}

function AnamorphicCard({ j }: { j: JsonObject }) {
  const xy = paraxialXY(j)
  const p = xy?.primary
  return (
    <section className="mx-3 mt-3 max-w-xl rounded-lg bg-selected/60 p-2.5 text-[12px]" aria-label="Anamorphic first-order data">
      <h3 className="mb-1.5 text-[12px] font-semibold">Anamorphic</h3>
      {!xy || !p ? (
        <p className="flex gap-1.5 text-secondary">
          <Info className="mt-[1px] size-3.5 shrink-0" />
          This design is not rotationally symmetric. The engine's first-order data covers the y section.
        </p>
      ) : (
        <div className="space-y-2">
          <table className="text-[12px]">
            <thead>
              <tr className="text-[11px] text-secondary">
                <th />
                <th className="pl-4 text-right font-normal">x (mm)</th>
                <th className="pl-4 text-right font-normal">y (mm)</th>
              </tr>
            </thead>
            <tbody className="num">
              <tr title="Effective focal length">
                <td className="text-secondary">EFL</td>
                <td className="pl-4 text-right">{short(p.eflX)}</td>
                <td className="pl-4 text-right">{short(p.eflY)}</td>
              </tr>
              <tr title="Back focal length">
                <td className="text-secondary">BFL</td>
                <td className="pl-4 text-right">{short(p.bflX)}</td>
                <td className="pl-4 text-right">{short(p.bflY)}</td>
              </tr>
              <tr title={'Squeeze: ' + (xy.squeezeDefinition ?? 'signed EFL y / EFL x')}>
                <td className="text-secondary">Squeeze</td>
                <td colSpan={2} className="pl-4 text-right">
                  {squeezeText(p.squeeze)}
                </td>
              </tr>
              {xy.symmetry && (
                <tr>
                  <td className="text-secondary">Symmetry</td>
                  <td colSpan={2} className="pl-4 text-right font-sans">
                    {SYMMETRY[xy.symmetry] ?? xy.symmetry}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          {Number.isFinite(p.um) && <div className="num text-[11px] text-secondary">At λ = {p.um.toFixed(4)} µm (primary)</div>}
          {p.doe.length > 0 && <DoePower row={p} />}
          {xy.warnings.map((w, k) => (
            <p key={k} className="flex gap-1.5 text-[11px] text-warning">
              <AlertTriangle className="mt-[1px] size-3 shrink-0" />
              <span>{w}</span>
            </p>
          ))}
          {xy.others.length > 0 && (
            <details>
              <summary className="cursor-default text-[12px]">Other wavelengths ({xy.others.length})</summary>
              <div className="overflow-x-auto">
                <table className="num mt-1 text-[11px]">
                  <thead className="text-secondary">
                    <tr>
                      {['λ µm', 'EFL x', 'EFL y', 'BFL x', 'BFL y', 'Squeeze'].map((h) => (
                        <th key={h} className="pl-3 text-right font-normal first:pl-0">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {xy.others.map((r) => (
                      <tr key={r.wavelength}>
                        {[sig(r.um, 5), short(r.eflX), short(r.eflY), short(r.bflX), short(r.bflY), squeezeText(r.squeeze)].map((v, k) => (
                          <td key={k} className="pl-3 text-right first:pl-0">
                            {v}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      )}
    </section>
  )
}

function FirstOrder({ result, system }: RendererProps) {
  const j = result.json
  const entries = objects(j.entries)
  const anamorphic = isAnamorphic(j, system)
  const p = paraxialXY(j)?.primary
  const wl = nf(j.wavelength_um)
  return (
    <div className="absolute inset-0 overflow-auto select-text">
      {anamorphic ? (
        <AnamorphicCard j={j} />
      ) : (
        p &&
        p.doe.length > 0 && (
          <section className="mx-3 mt-3 max-w-xl rounded-lg bg-selected/60 p-2.5">
            <h3 className="mb-1.5 text-[12px] font-semibold">Diffractive</h3>
            <DoePower row={p} />
          </section>
        )
      )}
      <table className="m-3 max-w-xl text-[13px]">
        <tbody>
          {entries.map((e, k) => (
            <tr key={k} className="border-b border-separator/60 last:border-0">
              <td className="py-1 pr-4 text-secondary">{String(e.label ?? e.key ?? '')}</td>
              <td className="num py-1 text-right">{typeof e.value === 'string' && Number.isNaN(nf(e.value)) && e.value !== 'nan' ? e.value : short(nf(e.value))}</td>
              <td className="py-1 pl-1.5 text-[12px] text-secondary">{typeof e.unit === 'string' ? e.unit : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {Number.isFinite(wl) && <div className="num mx-3 mb-3 text-[11px] text-secondary">At λ = {wl.toFixed(4)} µm</div>}
    </div>
  )
}

function firstOrderKeys(r: { json: JsonObject }): KeyNumber[] {
  const es = objects(r.json.entries)
  const find = (...keys: string[]) => {
    for (const k of keys) {
      const e = es.find((x) => x.key === k)
      if (e) return e
    }
    return undefined
  }
  const out: KeyNumber[] = []
  const add = (label: string, e: JsonObject | undefined) => {
    if (!e) return
    const v = nf(e.value)
    if (Number.isNaN(v)) return
    out.push({ label, value: valueUnit(v, typeof e.unit === 'string' ? e.unit : ''), help: typeof e.label === 'string' ? e.label : undefined })
  }
  add('EFL', find('efl'))
  add('F/#', find('working_f_number', 'image_f_number', 'f_number'))
  add('BFL', find('bfl'))
  return out
}

// MARK: System summary

const surfaceTypeTitle = (t: string) => t.replace(/_/g, ' ').replace(/^q /, 'Q-').replace(/^./, (c) => c.toUpperCase())

function semiDiameterText(v: Json | undefined): string {
  const o = obj(v)
  if (!o) return typeof v === 'number' ? short(v) : ''
  const value = nf(o.value)
  return Number.isFinite(value) ? short(value) + (o.mode === 'fixed' ? '' : ' ' + String(o.mode ?? '').charAt(0).toUpperCase()) : String(o.mode ?? '')
}

function SystemSummary({ result }: RendererProps) {
  const j = result.json
  const surfaces = objects(j.surfaces)
  const um = nums(j.wavelengths_um)
  const pw = intOf(j.primary_wavelength, 0)
  const ap = obj(j.aperture)
  const fs = obj(j.fields)
  const env = obj(j.environment)
  const xy = paraxialXY(j)
  const head: [string, string][] = []
  if (typeof j.name === 'string' && j.name) head.push(['Name', j.name])
  if (ap) head.push(['Aperture', `${String(ap.type ?? '').replace(/_/g, ' ')}${ap.value != null ? ' ' + short(nf(ap.value)) : ''}`])
  if (fs) head.push(['Fields', `${String(fs.kind ?? '').replace(/_/g, ' ')} · ${Array.isArray(fs.fields) ? fs.fields.length : 0}`])
  if (um.length) head.push(['Wavelengths', um.map((w, k) => (k === pw ? `${sig(w, 5)}*` : sig(w, 5))).join(', ') + ' µm'])
  if (typeof j.stop === 'number') head.push(['Stop', `Surface ${j.stop}`])
  if (j.object_at_infinity === true) head.push(['Object', 'At infinity'])
  const L = nf(j.first_to_image_distance)
  if (Number.isFinite(L)) head.push(['Surface 1 to image', length(L)])
  if (env) head.push(['Environment', `${short(nf(env.temperature_c ?? env.temperature))} °C · ${short(nf(env.pressure_atm ?? env.pressure))} atm`])
  if (typeof j.symmetry === 'string') head.push(['Symmetry', SYMMETRY[j.symmetry] ?? j.symmetry])
  if (xy?.primary && isAnamorphic(j)) head.push(['EFL x / y', `${short(xy.primary.eflX)} / ${short(xy.primary.eflY)} mm`])
  const warnings = (Array.isArray(j.warnings) ? j.warnings : []).filter((w): w is string => typeof w === 'string')
  const cols = ['#', 'Label', 'Type', 'Radius', 'Thickness', 'Material', 'n_d', 'V_d', 'Semi-dia.']
  return (
    <div className="absolute inset-0 overflow-auto p-3 select-text">
      <div className="mb-3 grid max-w-2xl grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-[12px]">
        {head.map(([k, v]) => (
          <div key={k} className="contents">
            <span className="text-secondary">{k}</span>
            <span className="num min-w-0 break-words">{v}</span>
          </div>
        ))}
      </div>
      <table className="text-[12px]">
        <thead>
          <tr className="border-b border-separator text-[11px] text-secondary">
            {cols.map((c, k) => (
              <th key={c} className={cn('px-1.5 py-0.5 font-normal whitespace-nowrap', k < 3 || k === 5 ? 'text-left' : 'text-right')}>
                {c === 'n_d' ? (
                  <>
                    n<sub>d</sub>
                  </>
                ) : c === 'V_d' ? (
                  <>
                    V<sub>d</sub>
                  </>
                ) : (
                  c
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="num">
          {surfaces.map((s, k) => {
            const index = intOf(s.index, k)
            const stop = s.is_stop === true || index === j.stop
            return (
              <tr key={k} className="border-b border-separator/50 last:border-0">
                <td className="px-1.5 py-0.5 text-secondary">{stop ? `${index} (stop)` : index}</td>
                <td className="max-w-[10rem] truncate px-1.5 py-0.5 font-sans">{String(s.label ?? '')}</td>
                <td className="px-1.5 py-0.5 font-sans whitespace-nowrap">{surfaceTypeTitle(String(s.type ?? obj(s.shape)?.type ?? ''))}</td>
                <td className="px-1.5 py-0.5 text-right">{short(nf(s.radius))}</td>
                <td className="px-1.5 py-0.5 text-right">{short(nf(s.thickness))}</td>
                <td className="px-1.5 py-0.5 font-sans">{s.interaction === 'reflect' ? 'Mirror' : String(s.material ?? '')}</td>
                <td className="px-1.5 py-0.5 text-right">{s.nd_after == null ? '' : nf(s.nd_after).toFixed(6)}</td>
                <td className="px-1.5 py-0.5 text-right">{s.vd_after == null || !Number.isFinite(nf(s.vd_after)) ? '' : nf(s.vd_after).toFixed(2)}</td>
                <td className="px-1.5 py-0.5 text-right whitespace-nowrap">{semiDiameterText(s.semi_diameter)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {um.length > 1 && <p className="mt-1 text-[11px] text-secondary">* primary wavelength</p>}
      {warnings.map((w, k) => (
        <p key={k} className="mt-1.5 flex gap-1.5 text-[11px] text-warning">
          <AlertTriangle className="mt-[1px] size-3 shrink-0" />
          {w}
        </p>
      ))}
    </div>
  )
}

// MARK: Validate

function ValidateView({ result }: RendererProps) {
  const issues = objects(result.json.issues)
  if (!issues.length) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 p-4 text-center">
        <CheckCircle2 className="size-6 text-secondary" />
        <span className="text-[13px] font-medium">No Issues</span>
        <span className="text-[12px] text-secondary">The design passed every check.</span>
      </div>
    )
  }
  const rank: Record<string, number> = { error: 0, warning: 1, info: 2 }
  const sorted = [...issues].sort((a, b) => (rank[String(a.severity)] ?? 3) - (rank[String(b.severity)] ?? 3))
  return (
    <ul className="absolute inset-0 overflow-auto py-1 select-text" aria-label="Validation issues">
      {sorted.map((i, k) => {
        const sev = String(i.severity ?? 'warning')
        const Icon = sev === 'error' ? XCircle : sev === 'info' ? Info : AlertTriangle
        return (
          <li key={k} className="flex gap-2 border-b border-separator/50 px-3 py-1.5 text-[12px] last:border-0">
            <Icon className={cn('mt-[1px] size-3.5 shrink-0', sev === 'error' ? 'text-danger' : sev === 'info' ? 'text-secondary' : 'text-warning')} aria-label={sev} />
            <span className="min-w-0 flex-1">
              {typeof i.surface === 'number' && <span className="mr-1.5 text-secondary">Surface {i.surface}</span>}
              {String(i.message ?? '')}
              {typeof i.code === 'string' && <span className="ml-1.5 text-[11px] text-secondary">{i.code}</span>}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

export const TABLE_RENDERERS: Record<string, Renderer> = {
  first_order: {
    accepts: (j) => Array.isArray(j.entries),
    View: FirstOrder,
    keyNumbers: firstOrderKeys,
  },
  system_summary: {
    accepts: (j) => Array.isArray(j.surfaces),
    View: SystemSummary,
    keyNumbers: (r) => {
      const out: KeyNumber[] = []
      const n = objects(r.json.surfaces).length
      if (n) out.push({ label: 'Surfaces', value: String(n) })
      const L = nf(r.json.first_to_image_distance)
      if (Number.isFinite(L)) out.push({ label: 'Length', value: length(L), help: 'Surface 1 to the image vertex' })
      return out
    },
  },
  validate: {
    accepts: (j) => Array.isArray(j.issues) || typeof j.errors === 'number',
    View: ValidateView,
    keyNumbers: (r) => [
      { label: 'Errors', value: String(intOf(r.json.errors, 0)) },
      { label: 'Warnings', value: String(intOf(r.json.warnings, 0)) },
    ],
  },
}

