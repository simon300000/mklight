// Ray and OPD fans: per field, a tangential and a sagittal plot on one y scale.
import { useMemo } from 'react'
import uPlot from 'uplot'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import type { System } from '@/model/system'
import { useDark } from '@/lib/theme'
import { sig } from '@/model/format'
import { arr, fieldLabel, isObj, maxAbs, nums, useSize, wavelengthsUm, wlColor } from './common'
import { Legend, Swatch } from './Legend'
import { LINE, UPlot, axis, calmCursor, gaps, type PlotOptions } from './UPlot'

interface Fan {
  wavelength: number
  pupil: number[]
  values: number[]
}

const curves = (v: JsonObject, key: 'tangential' | 'sagittal'): Fan[] =>
  arr(v[key])
    .filter(isObj)
    .map((c) => ({ wavelength: typeof c.wavelength === 'number' ? c.wavelength : 0, pupil: nums(c.pupil), values: nums(c.values) }))

/** Largest |value| over every fan of a fans result (in result units). */
export function fanMax(json: JsonObject): number {
  let m = 0
  for (const f of arr(json.fields).filter(isObj))
    for (const k of ['tangential', 'sagittal'] as const) for (const c of curves(f, k)) m = Math.max(m, maxAbs(c.values))
  return m
}

function FanPlot(props: { fans: Fan[]; xLabel: string; yLabel: string; range: number; scale: number; wls: number[]; dark: boolean; label: string }) {
  const { fans, range, scale, wls, dark } = props
  const data = useMemo<uPlot.AlignedData>(() => {
    const tables = fans.filter((f) => f.pupil.length).map((f) => [f.pupil, gaps(f.values.map((v) => v * scale))] as uPlot.AlignedData)
    if (!tables.length) return [[]]
    return tables.length === 1 ? tables[0] : uPlot.join(tables)
  }, [fans, scale])
  const options = useMemo<PlotOptions>(
    () => ({
      legend: { show: false },
      cursor: calmCursor,
      scales: { x: { time: false, range: [-1, 1] }, y: { range: [-range, range] } },
      axes: [axis(props.xLabel, { size: 30 }), axis(props.yLabel, { size: 46 })],
      series: [{}, ...fans.filter((f) => f.pupil.length).map((f) => ({ stroke: wlColor(wls, f.wavelength, dark), width: LINE, points: { show: false } }))],
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [fans, range, wls, dark, props.xLabel, props.yLabel],
  )
  return <UPlot options={options} data={data} label={props.label} />
}

export function FansView(props: { result: AnalysisResult; system: System; opd: boolean }) {
  const dark = useDark()
  const j = props.result.json
  const [ref, size] = useSize<HTMLDivElement>()
  const wls = useMemo(() => wavelengthsUm(j, props.system), [j, props.system])
  const fields = useMemo(() => arr(j.fields).filter(isObj), [j])
  const fans = useMemo(() => fields.map((f) => ({ t: curves(f, 'tangential'), s: curves(f, 'sagittal') })), [fields])
  const m = fanMax(j)
  const um = !props.opd && m < 0.1
  const scale = um ? 1000 : 1
  const unit = props.opd ? 'waves' : um ? 'µm' : 'mm'
  const range = (m * scale || 1) * 1.08
  const shown = new Set<number>()
  for (const f of fans) for (const c of [...f.t, ...f.s]) shown.add(c.wavelength)

  const rowH = Math.max(150, Math.min(260, (size.h - 30) / Math.max(fields.length, 1) - 20))
  const narrow = size.w > 0 && size.w < 420
  const what = props.opd ? 'OPD' : 'Ray aberration'

  if (!fields.length) return <div className="p-4 text-secondary">No fields in this result.</div>
  return (
    <div ref={ref} className="absolute inset-0 flex flex-col">
      <Legend>
        {[...shown].sort((a, b) => a - b).map((w) => (
          <Swatch key={w} color={wlColor(wls, w, dark)} label={`${sig(wls[w], 4)} µm`} />
        ))}
        <span className="ml-auto">{props.opd ? 'OPD (waves)' : `εy, εx (${unit})`}</span>
      </Legend>
      <div className="min-h-0 flex-1 overflow-auto px-2 pb-2">
        {fields.map((f, k) => {
          const label = fieldLabel(props.system, f.field, f.field_value)
          return (
            <section key={k} className="mb-2">
              <div className="px-1 pt-1 text-[12px] text-label">{label}</div>
              <div className={narrow ? 'flex flex-col' : 'grid grid-cols-2 gap-2'}>
                {(['t', 's'] as const).map((side) => (
                  <div key={side} className="flex min-w-0 flex-col">
                    <div className="px-1 text-[11px] text-secondary">{side === 't' ? 'Tangential' : 'Sagittal'}</div>
                    <div style={{ height: rowH }}>
                      <FanPlot
                        fans={fans[k][side]}
                        xLabel={side === 't' ? 'Py' : 'Px'}
                        yLabel={props.opd ? 'W (waves)' : `${side === 't' ? 'εy' : 'εx'} (${unit})`}
                        range={range}
                        scale={scale}
                        wls={wls}
                        dark={dark}
                        label={`${what}, ${side === 't' ? 'tangential' : 'sagittal'} fan, ${label}`}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )
        })}
      </div>
    </div>
  )
}
