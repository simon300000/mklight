// MTF: tangential solid and sagittal dashed per field, with the diffraction limit.
import { useMemo } from 'react'
import type uPlot from 'uplot'
import type { AnalysisResult } from '@/engine/types'
import type { System } from '@/model/system'
import { useT } from '@/i18n'
import { cssVar, seriesColor, useDark } from '@/lib/theme'
import { arr, buffer, fieldLabel, isObj } from './common'
import { Legend, Swatch } from './Legend'
import { DASH, LINE, UPlot, axis, calmCursor, gaps, type PlotOptions } from './UPlot'

export function MtfView(props: { result: AnalysisResult; system: System }) {
  const t = useT()
  const dark = useDark()
  const r = props.result
  const fields = useMemo(() => arr(r.json.fields).filter(isObj), [r])
  const freq = buffer(r, r.json.frequencies) ?? buffer(r, 'frequencies')

  const { data, series } = useMemo(() => {
    const x = freq ? Array.from(freq) : []
    const data: (number | null)[][] = [x]
    const series: uPlot.Series[] = [{}]
    const limit = fields.length ? buffer(r, fields[0].diffraction_limit) : null
    if (limit) {
      data.push(gaps(limit))
      series.push({ label: t('analysis.curves.diffractionLimit'), stroke: cssVar('--secondary') || '#888', width: 1, points: { show: false } })
    }
    fields.forEach((f, k) => {
      const color = seriesColor(k, dark)
      const tan = buffer(r, f.tangential)
      const s = buffer(r, f.sagittal)
      if (tan) {
        data.push(gaps(tan))
        series.push({ label: t('analysis.results.fieldT', { n: (typeof f.field === 'number' ? f.field : k) + 1 }), stroke: color, width: LINE, points: { show: false } })
      }
      if (s) {
        data.push(gaps(s))
        series.push({ label: t('analysis.results.fieldS', { n: (typeof f.field === 'number' ? f.field : k) + 1 }), stroke: color, width: LINE, dash: DASH, points: { show: false } })
      }
    })
    return { data: data as uPlot.AlignedData, series }
  }, [r, fields, freq, dark, t])

  const options = useMemo<PlotOptions>(
    () => ({
      legend: { show: false },
      cursor: calmCursor,
      scales: { x: { time: false }, y: { range: [0, 1] } },
      axes: [axis(t('analysis.geometricMtf.frequency')), axis(t('analysis.results.modulus'), { size: 40 })],
      series,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, dark, t],
  )

  if (!freq || !fields.length) return <div className="p-4 text-secondary">{t('analysis.results.noMtf')}</div>
  const secondary = cssVar('--secondary') || '#888'
  return (
    <div className="absolute inset-0 flex flex-col">
      <Legend>
        {fields.map((f, k) => (
          <Swatch key={k} color={seriesColor(k, dark)} kind="dot" label={fieldLabel(props.system, f.field, f.field_value)} />
        ))}
        <span className="ml-auto inline-flex gap-3">
          <Swatch color={secondary} label="T" />
          <Swatch color={secondary} kind="dash" label="S" />
          <Swatch color={secondary} kind="thin" label={t('analysis.curves.diffractionLimit')} />
        </span>
      </Legend>
      <div className="min-h-0 flex-1 px-2 pb-2">
        <UPlot options={options} data={data} label={t('analysis.results.mtfLabel')} />
      </div>
    </div>
  )
}
