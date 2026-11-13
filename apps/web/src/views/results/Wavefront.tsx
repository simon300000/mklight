// Wavefront map: OPD over the pupil on the diverging scale, with RMS/PV and Zernikes.
import { useMemo, useState } from 'react'
import type { AnalysisResult } from '@/engine/types'
import type { System } from '@/model/system'
import { useDark } from '@/lib/theme'
import { fmt, nf } from '@/model/format'
import { arr, buffer, divergingColor, isObj, maxAbs, obj, wavelengthsUm } from './common'
import { HeatMap } from './HeatMap'
import { Picker } from './Picker'
import { psfTitle } from './Psf'

const ZERNIKES_SHOWN = 9

export function WavefrontView(props: { result: AnalysisResult; system: System }) {
  const dark = useDark()
  const r = props.result
  const maps = useMemo(() => arr(r.json.maps).filter(isObj), [r])
  const wls = useMemo(() => wavelengthsUm(r.json, props.system), [r, props.system])
  const [sel, setSel] = useState(0)
  const k = Math.min(sel, Math.max(maps.length - 1, 0))
  const m = maps[k]
  const opd = m ? buffer(r, m.opd_waves) : null
  const shape = m && typeof m.opd_waves === 'string' ? r.shapes?.[m.opd_waves] : undefined
  const grid = m ? Math.round(nf(m.grid)) : 0
  const ny = shape?.length === 2 ? shape[0] : grid
  const nx = shape?.length === 2 ? shape[1] : grid
  const scale = useMemo(() => (opd ? maxAbs(Array.from(opd)) : 0), [opd])
  const coeffs = m ? buffer(r, obj(m.zernike)?.coefficients) : null

  if (!m || !opd || !(nx > 0 && ny > 0) || opd.length < nx * ny)
    return <div className="p-4 text-secondary">No wavefront map in this result.</div>
  const s = scale || 1
  const title = psfTitle(m, props.system, wls)
  const ordering = String(obj(m.zernike)?.ordering ?? 'fringe')
  const zs = coeffs ? Array.from(coeffs).slice(0, ZERNIKES_SHOWN) : []

  return (
    <div className="absolute inset-0 flex flex-col">
      {maps.length > 1 && <Picker value={k} onChange={setSel} options={maps.map((q) => psfTitle(q, props.system, wls))} label="Map" />}
      <div className="min-h-0 flex-1 p-2">
        <HeatMap
          data={opd}
          nx={nx}
          ny={ny}
          color={(v) => divergingColor(v / s, dark)}
          x={[-1, 1]}
          y={[-1, 1]}
          xLabel="Px"
          yLabel="Py"
          colorBar={{ min: -s, max: s, color: (v) => divergingColor(v / s, dark), label: 'OPD (waves)' }}
          label={`Wavefront map, ${title}`}
          dark={dark}
          smooth={false}
          frame={false}
        />
      </div>
      <div className="num shrink-0 px-3 text-[11px] text-secondary">
        {title} · RMS {fmt(nf(m.rms_waves), 4)} λ · PV {fmt(nf(m.pv_waves), 4)} λ
        {Number.isFinite(nf(m.strehl_marechal)) && <> · Strehl (Maréchal) {fmt(nf(m.strehl_marechal), 3)}</>}
      </div>
      {zs.length > 0 && (
        <div className="shrink-0 px-3 pt-1 pb-2 text-[11px]">
          <div className="text-secondary">Zernike coefficients ({ordering === 'standard' ? 'Standard' : 'Fringe'}, waves)</div>
          <div className="num grid gap-x-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(96px, 1fr))' }}>
            {zs.map((z, i) => (
              <div key={i} className="flex justify-between gap-2 select-text">
                <span className="text-secondary">Z{i + 1}</span>
                <span className="text-label">{fmt(z, 4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
