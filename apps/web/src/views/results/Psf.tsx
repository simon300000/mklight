// PSF: one image at a time as a one-hue magnitude heat map, axes in µm.
import { useMemo, useState } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import type { System } from '@/model/system'
import { useDark } from '@/lib/theme'
import { fmt, nf, sig } from '@/model/format'
import { arr, buffer, fieldLabel, isObj, magnitudeColor, nums, obj, wavelengthsUm } from './common'
import { HeatMap } from './HeatMap'
import { Picker } from './Picker'

export function psfTitle(p: JsonObject, system: System, wls: number[]): string {
  const f = fieldLabel(system, p.field, p.field_value)
  const w = typeof p.wavelength === 'number' ? wls[p.wavelength] : nf(obj(p.psf)?.wavelength_um)
  return Number.isFinite(w) ? `${f} · ${sig(w, 4)} µm` : f
}

export function PsfView(props: { result: AnalysisResult; system: System }) {
  const dark = useDark()
  const r = props.result
  const psfs = useMemo(() => arr(r.json.psfs).filter(isObj), [r])
  const wls = useMemo(() => wavelengthsUm(r.json, props.system), [r, props.system])
  const [sel, setSel] = useState(0)
  const k = Math.min(sel, Math.max(psfs.length - 1, 0))
  const p = psfs[k]
  const img = obj(p?.psf)
  const data = img ? buffer(r, img.data) : null
  const n = img ? Math.round(nf(img.size)) : 0
  const max = useMemo(() => {
    let m = 0
    if (data) for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i]) && data[i] > m) m = data[i]
    return m
  }, [data])

  if (!p || !img || !data || !(n > 0) || data.length < n * n)
    return <div className="p-4 text-secondary">No PSF image in this result.</div>
  const sp = nf(img.spacing_mm) * 1000
  const [cx, cy] = nums(img.center_mm).map((v) => (Number.isFinite(v) ? v * 1000 : 0))
  const h = Math.floor(n / 2)
  const ext = (c: number): [number, number] => [c + (-h - 0.5) * sp, c + (n - 1 - h + 0.5) * sp]
  const title = psfTitle(p, props.system, wls)
  const strehl = nf(p.strehl)
  const unitEnergy = img.normalization === 'unit_energy'

  return (
    <div className="absolute inset-0 flex flex-col">
      {psfs.length > 1 && (
        <Picker value={k} onChange={setSel} options={psfs.map((q) => psfTitle(q, props.system, wls))} label="PSF" />
      )}
      <div className="min-h-0 flex-1 p-2">
        <HeatMap
          data={data}
          nx={n}
          ny={n}
          color={(v) => magnitudeColor(max > 0 ? v / max : 0, dark)}
          x={ext(cx ?? 0)}
          y={ext(cy ?? 0)}
          xLabel="x (µm)"
          yLabel="y (µm)"
          colorBar={{ min: 0, max: max || 1, color: (v) => magnitudeColor(max > 0 ? v / max : 0, dark), label: unitEnergy ? 'Intensity (mm⁻²)' : 'Intensity (Strehl)' }}
          label={`Point spread function, ${title}`}
          dark={dark}
        />
      </div>
      <div className="num shrink-0 px-3 pb-2 text-[11px] text-secondary">
        {title} · Strehl {Number.isFinite(strehl) ? fmt(strehl, 3) : '–'} · {n} × {n} at {sig(sp, 3)} µm
      </div>
    </div>
  )
}
