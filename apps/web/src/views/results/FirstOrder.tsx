// First-order properties: a two-column report table.
import type { AnalysisResult } from '@/engine/types'
import { nf, sig } from '@/model/format'
import { arr, isObj, obj } from './common'

export function valueText(v: unknown, unit: unknown): string {
  const n = nf(v)
  const u = typeof unit === 'string' && unit ? (unit === '°' ? '°' : ` ${unit}`) : ''
  return typeof v === 'string' && Number.isNaN(n) && v !== 'nan' ? v : sig(n, 6) + u
}

export function FirstOrderView(props: { result: AnalysisResult }) {
  const j = props.result.json
  const entries = arr(j.entries).filter(isObj)
  const warnings = arr(obj(j.paraxial_xy)?.warnings).filter((w): w is string => typeof w === 'string')
  const wl = nf(j.wavelength_um)
  return (
    <div className="absolute inset-0 overflow-auto p-3 select-text">
      {Number.isFinite(wl) && <div className="mb-2 text-[11px] text-secondary">At {sig(wl, 6)} µm</div>}
      <table className="w-full max-w-xl text-[13px]">
        <tbody>
          {entries.map((e, k) => (
            <tr key={k} className="border-b border-separator/60 last:border-0">
              <td className="py-1 pr-4 text-secondary">{String(e.label ?? e.key ?? '')}</td>
              <td className="num py-1 text-right text-label">{valueText(e.value, e.unit)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {warnings.length > 0 && (
        <ul className="mt-3 max-w-xl list-disc pl-5 text-[11px] text-secondary">
          {warnings.map((w, k) => (
            <li key={k}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
