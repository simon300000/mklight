// Field curvature (T solid, S dashed) and distortion versus field, field upward.
import { useMemo } from 'react'
import type uPlot from 'uplot'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import type { System } from '@/model/system'
import { useDark } from '@/lib/theme'
import { sig } from '@/model/format'
import { arr, isObj, maxAbs, nums, systemFieldUnit, useSize, wavelengthsUm, wlColor } from './common'
import { Legend, Swatch } from './Legend'
import { DASH, LINE, UPlot, axis, calmCursor, gaps, type PlotOptions } from './UPlot'

/** Largest |distortion| (%) over every curve. */
export function maxDistortion(json: JsonObject): number {
  return Math.max(0, ...arr(json.curves).filter(isObj).map((c) => maxAbs(nums(c.distortion_percent))))
}

function rotated(xLabel: string, yLabel: string, series: uPlot.Series[], symmetric: number | null): PlotOptions {
  return {
    legend: { show: false },
    cursor: calmCursor,
    scales: {
      x: { time: false, ori: 1, dir: 1 },
      y: { ori: 0, dir: 1, ...(symmetric ? { range: [-symmetric, symmetric] as [number, number] } : {}) },
    },
    axes: [axis(yLabel, { side: 3, size: 44 }), axis(xLabel, { side: 2, size: 30 })],
    series,
  }
}

export function FieldCurvatureView(props: { result: AnalysisResult; system: System }) {
  const dark = useDark()
  const j = props.result.json
  const [ref, size] = useSize<HTMLDivElement>()
  const wls = useMemo(() => wavelengthsUm(j, props.system), [j, props.system])
  const unit = typeof j.field_unit === 'string' ? j.field_unit : systemFieldUnit(props.system)
  const curves = useMemo(() => arr(j.curves).filter(isObj), [j])

  const built = useMemo(() => {
    const fv = nums(j.field_values)
    const order = fv.map((_, i) => i).filter((i) => Number.isFinite(fv[i])).sort((a, b) => fv[a] - fv[b])
    const pick = (vs: number[]) => gaps(order.map((i) => vs[i] ?? NaN))
    const x = order.map((i) => fv[i])
    const fcData: (number | null)[][] = [x]
    const fcSeries: uPlot.Series[] = [{}]
    const dData: (number | null)[][] = [x]
    const dSeries: uPlot.Series[] = [{}]
    let fcMax = 0
    for (const c of curves) {
      const color = wlColor(wls, typeof c.wavelength === 'number' ? c.wavelength : 0, dark)
      const t = nums(c.tangential), s = nums(c.sagittal), d = nums(c.distortion_percent)
      fcMax = Math.max(fcMax, maxAbs(t), maxAbs(s))
      fcData.push(pick(t), pick(s))
      fcSeries.push({ stroke: color, width: LINE, points: { show: false } }, { stroke: color, width: LINE, dash: DASH, points: { show: false } })
      dData.push(pick(d))
      dSeries.push({ stroke: color, width: LINE, points: { show: false } })
    }
    const dMax = maxDistortion(j)
    return {
      fc: { data: fcData as uPlot.AlignedData, options: rotated('Focus shift (mm)', `Field (${unit})`, fcSeries, (fcMax || 0.01) * 1.1) },
      dist: { data: dData as uPlot.AlignedData, options: rotated('Distortion (%)', `Field (${unit})`, dSeries, (dMax || 0.1) * 1.1) },
    }
  }, [j, curves, wls, dark, unit])

  if (!curves.length) return <div className="p-4 text-secondary">No curves in this result.</div>
  const stacked = size.w > 0 && size.w < 480
  return (
    <div ref={ref} className="absolute inset-0 flex flex-col">
      <Legend>
        {curves.map((c, k) => (
          <Swatch key={k} color={wlColor(wls, typeof c.wavelength === 'number' ? c.wavelength : 0, dark)} kind="dot" label={`${sig(wls[Number(c.wavelength)], 4)} µm`} />
        ))}
        <span className="ml-auto inline-flex gap-3">
          <Swatch color="currentColor" label="T" />
          <Swatch color="currentColor" kind="dash" label="S" />
        </span>
      </Legend>
      <div className={stacked ? 'flex min-h-0 flex-1 flex-col gap-2 px-2 pb-2' : 'grid min-h-0 flex-1 grid-cols-2 gap-3 px-2 pb-2'}>
        {(
          [
            ['Field curvature', built.fc],
            ['Distortion', built.dist],
          ] as const
        ).map(([title, p]) => (
          <div key={title} className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div className="px-1 text-[11px] text-secondary">{title}</div>
            <div className="min-h-0 flex-1">
              <UPlot options={p.options} data={p.data} label={`${title} versus field`} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
