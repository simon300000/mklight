// Wave-optics curves (MTF vs field, through-focus MTF, line spread) and the
// system transmission.
import { useState } from 'react'
import type { AnalysisResult } from '@/engine/types'
import { t, useT } from '@/i18n'
import { cssVar, seriesColor, useDark } from '@/lib/theme'
import { sig } from '@/model/format'
import { nums } from '../../results/common'
import { Picker } from '../../results/Picker'
import type { KeyNumber, Renderer, RendererProps } from '../registry'
import { BarPlot, Empty, Key, PlotPane, XYPlot, fieldTitle, intOf, minOf, nf, objects, percent, series, short, type XYSeries } from './kit'

const secondaryColor = () => cssVar('--secondary') || '#888'

function TS() {
  return (
    <>
      <Key color={secondaryColor()} label="T" />
      <Key color={secondaryColor()} kind="dash" label="S" />
    </>
  )
}

// MARK: MTF vs field

function MtfVsField({ result }: RendererProps) {
  const t = useT()
  const dark = useDark()
  const j = result.json
  const field = series(result, j.normalized_field)
  const curves = objects(j.curves).map((c) => ({ f: nf(c.frequency), t: series(result, c.tangential), s: series(result, c.sagittal) }))
  if (!field.length || !curves.length) return <Empty title={t('analysis.mtfVsField.empty')} />
  const plot: XYSeries[] = curves.flatMap((c, k) => [
    { x: field, y: c.t, color: seriesColor(k, dark) },
    { x: field, y: c.s, color: seriesColor(k, dark), dash: true },
  ])
  return (
    <PlotPane
      legend={
        <>
          {curves.map((c, k) => (
            <Key key={k} kind="dot" color={seriesColor(k, dark)} label={`${short(c.f)} cycles/mm`} />
          ))}
          <TS />
        </>
      }
    >
      <XYPlot series={plot} xLabel={t('analysis.mtfVsField.normalizedField')} yLabel={t('analysis.curves.modulation')} xDomain={[0, 1]} yDomain={[0, 1.02]} label={t('analysis.mtfVsField.label')} />
    </PlotPane>
  )
}

function mtfVsFieldKeys(r: AnalysisResult): KeyNumber[] {
  const c = objects(r.json.curves)[0]
  if (!c) return []
  const m = minOf([...series(r, c.tangential), ...series(r, c.sagittal)])
  return Number.isFinite(m) ? [{ label: t('analysis.mtfVsField.min', { frequency: short(nf(c.frequency)) }), value: m.toFixed(3), help: t('analysis.mtfVsField.minHelp', { frequency: short(nf(c.frequency)) }) }] : []
}

// MARK: Through-focus MTF

function ThroughFocus({ result, system }: RendererProps) {
  const t = useT()
  const dark = useDark()
  const [sel, setSel] = useState(0)
  const j = result.json
  const focus = series(result, j.focus_mm)
  const fields = objects(j.fields).map((f, k) => ({
    title: fieldTitle(system, intOf(f.field, k), f.field_value),
    curves: objects(f.curves).map((c) => ({ f: nf(c.frequency), t: series(result, c.tangential), s: series(result, c.sagittal) })),
  }))
  if (!focus.length || !fields.length) return <Empty title={t('analysis.throughFocus.empty')} />
  const f = fields[Math.min(sel, fields.length - 1)]
  const fin = focus.filter(Number.isFinite)
  const lo = Math.min(...fin), hi = Math.max(...fin)
  const plot: XYSeries[] = f.curves.flatMap((c, k) => [
    { x: focus, y: c.t, color: seriesColor(k, dark) },
    { x: focus, y: c.s, color: seriesColor(k, dark), dash: true },
  ])
  return (
    <PlotPane
      top={fields.length > 1 ? <Picker label={t('common.field')} value={Math.min(sel, fields.length - 1)} onChange={setSel} options={fields.map((x) => x.title)} /> : undefined}
      legend={
        <>
          {fields.length === 1 && <span className="text-label">{f.title}</span>}
          {f.curves.map((c, k) => (
            <Key key={k} kind="dot" color={seriesColor(k, dark)} label={`${short(c.f)} cycles/mm`} />
          ))}
          <TS />
        </>
      }
    >
      <XYPlot
        series={plot}
        xLabel={t('analysis.throughFocus.xLabel')}
        yLabel={t('analysis.curves.modulation')}
        xDomain={hi > lo ? [lo, hi] : undefined}
        yDomain={[0, 1.02]}
        markers={[{ x: 0, label: t('analysis.throughFocus.image') }]}
        label={t('analysis.throughFocus.label', { field: f.title })}
      />
    </PlotPane>
  )
}

function throughFocusKeys(r: AnalysisResult): KeyNumber[] {
  // The focus shift of the highest tangential modulation at the first frequency, field 1.
  const focus = series(r, r.json.focus_mm)
  const c = objects(objects(r.json.fields)[0]?.curves)[0]
  if (!c || !focus.length) return []
  const tan = series(r, c.tangential)
  let best = -1
  for (let i = 0; i < Math.min(tan.length, focus.length); i++) if (Number.isFinite(tan[i]) && (best < 0 || tan[i] > tan[best])) best = i
  if (best < 0) return []
  return [{ label: t('analysis.throughFocus.peak'), value: `${focus[best] >= 0 ? '+' : '−'}${Math.abs(focus[best]).toFixed(4)} mm`, help: t('analysis.throughFocus.peakHelp', { frequency: short(nf(c.frequency)) }) }]
}

// MARK: Line spread

function LineSpread({ result, system }: RendererProps) {
  const t = useT()
  const dark = useDark()
  const items = objects(result.json.fields)
    .map((f, k) => ({
      label: fieldTitle(system, intOf(f.field, k), f.field_value),
      x: series(result, f.positions_mm).map((v) => v * 1000),
      lx: series(result, f.x),
      ly: series(result, f.y),
    }))
    .filter((i) => i.x.length > 0)
  if (!items.length) return <Empty title={t('analysis.lineSpread.empty')} />
  const plot: XYSeries[] = items.flatMap((i, k) => [
    { x: i.x, y: i.lx, color: seriesColor(k, dark) },
    { x: i.x, y: i.ly, color: seriesColor(k, dark), dash: true },
  ])
  return (
    <PlotPane
      legend={
        <>
          {items.map((i, k) => (
            <Key key={k} kind="dot" color={seriesColor(k, dark)} label={i.label} />
          ))}
          <Key color={secondaryColor()} label="x" />
          <Key color={secondaryColor()} kind="dash" label="y" />
        </>
      }
    >
      <XYPlot series={plot} xLabel={t('analysis.lineSpread.xLabel')} yLabel={t('analysis.lineSpread.yLabel')} label={t('analysis.lineSpread.label')} />
    </PlotPane>
  )
}

// MARK: System transmission

function Transmission({ result, system }: RendererProps) {
  const t = useT()
  const dark = useDark()
  const j = result.json
  const points = objects(j.points).map((p) => ({ field: intOf(p.field, 0), um: nf(p.wavelength_um), t: nf(p.transmission) }))
  const poly = nums(j.polychromatic)
  const fieldDefs = objects(j.fields)
  if (!points.length) return <Empty title={t('analysis.transmission.empty')} />
  const fields = [...new Set(points.map((p) => p.field))].sort((a, b) => a - b)
  const labelOf = (f: number) => {
    const d = fieldDefs[f]
    if (d && (d.hx !== undefined || d.hy !== undefined)) return `H (${nf(d.hx ?? 0).toFixed(2)}, ${nf(d.hy ?? 0).toFixed(2)})`
    if (d && d.field !== undefined) return fieldTitle(system, intOf(d.field, f), d.field_value)
    return t('common.fieldN', { n: f + 1 })
  }
  const spectral = new Set(points.map((p) => p.um)).size > 1
  return (
    <PlotPane
      legend={fields.map((f, k) => (
        <Key key={f} kind="dot" color={seriesColor(k, dark)} label={labelOf(f) + (Number.isFinite(poly[f]) ? ' · ' + percent(poly[f]) : '')} />
      ))}
    >
      {spectral ? (
        <XYPlot
          series={fields.map((f, k) => {
            const ps = points.filter((p) => p.field === f).sort((a, b) => a.um - b.um)
            return { x: ps.map((p) => p.um), y: ps.map((p) => p.t), color: seriesColor(k, dark) }
          })}
          xLabel={t('analysis.curves.wavelengthUm')}
          yLabel={t('analysis.curves.transmission')}
          yDomain={[0, 1.02]}
          label={t('analysis.transmission.labelSpectral')}
        />
      ) : (
        <BarPlot
          categories={fields.map((f) => String(f + 1))}
          groups={[{ label: t('analysis.curves.transmission'), color: seriesColor(0, dark), values: fields.map((f) => points.find((p) => p.field === f)?.t ?? NaN) }]}
          xLabel={Number.isFinite(points[0].um) ? t('analysis.transmission.fieldAt', { wavelength: sig(points[0].um, 4) }) : t('common.field')}
          yLabel={t('analysis.curves.transmission')}
          yDomain={[0, 1]}
          label={t('analysis.transmission.labelField')}
        />
      )}
    </PlotPane>
  )
}

function transmissionKeys(r: AnalysisResult): KeyNumber[] {
  const poly = nums(r.json.polychromatic)
  const m = minOf(poly)
  const out: KeyNumber[] = []
  if (Number.isFinite(m)) out.push({ label: t('analysis.transmission.min'), value: percent(m), help: t('analysis.transmission.minHelp') })
  const ax = poly[0]
  if (Number.isFinite(ax) && poly.length > 1) out.push({ label: t('common.fieldN', { n: 1 }), value: percent(ax) })
  return out
}

export const WAVE_RENDERERS: Record<string, Renderer> = {
  mtf_vs_field: {
    accepts: (j) => Array.isArray(j.curves),
    View: MtfVsField,
    keyNumbers: mtfVsFieldKeys,
  },
  through_focus_mtf: {
    accepts: (j) => Array.isArray(j.fields),
    View: ThroughFocus,
    keyNumbers: throughFocusKeys,
  },
  line_spread: {
    accepts: (j) => Array.isArray(j.fields),
    View: LineSpread,
  },
  system_transmission: {
    accepts: (j) => Array.isArray(j.points),
    View: Transmission,
    keyNumbers: transmissionKeys,
  },
}

