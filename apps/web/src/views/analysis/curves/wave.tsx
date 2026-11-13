// Wave-optics curves (MTF vs field, through-focus MTF, line spread) and the
// system transmission.
import { useState } from 'react'
import type { AnalysisResult } from '@/engine/types'
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
  const dark = useDark()
  const j = result.json
  const field = series(result, j.normalized_field)
  const curves = objects(j.curves).map((c) => ({ f: nf(c.frequency), t: series(result, c.tangential), s: series(result, c.sagittal) }))
  if (!field.length || !curves.length) return <Empty title="No MTF curves in this result" />
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
      <XYPlot series={plot} xLabel="Normalized field" yLabel="Modulation" xDomain={[0, 1]} yDomain={[0, 1.02]} label="MTF versus normalized field at fixed frequencies" />
    </PlotPane>
  )
}

function mtfVsFieldKeys(r: AnalysisResult): KeyNumber[] {
  const c = objects(r.json.curves)[0]
  if (!c) return []
  const m = minOf([...series(r, c.tangential), ...series(r, c.sagittal)])
  return Number.isFinite(m) ? [{ label: `Min @ ${short(nf(c.frequency))}`, value: m.toFixed(3), help: `Lowest modulation over the field at ${short(nf(c.frequency))} cycles/mm` }] : []
}

// MARK: Through-focus MTF

function ThroughFocus({ result, system }: RendererProps) {
  const dark = useDark()
  const [sel, setSel] = useState(0)
  const j = result.json
  const focus = series(result, j.focus_mm)
  const fields = objects(j.fields).map((f, k) => ({
    title: fieldTitle(system, intOf(f.field, k), f.field_value),
    curves: objects(f.curves).map((c) => ({ f: nf(c.frequency), t: series(result, c.tangential), s: series(result, c.sagittal) })),
  }))
  if (!focus.length || !fields.length) return <Empty title="No through-focus curves in this result" />
  const f = fields[Math.min(sel, fields.length - 1)]
  const fin = focus.filter(Number.isFinite)
  const lo = Math.min(...fin), hi = Math.max(...fin)
  const plot: XYSeries[] = f.curves.flatMap((c, k) => [
    { x: focus, y: c.t, color: seriesColor(k, dark) },
    { x: focus, y: c.s, color: seriesColor(k, dark), dash: true },
  ])
  return (
    <PlotPane
      top={fields.length > 1 ? <Picker label="Field" value={Math.min(sel, fields.length - 1)} onChange={setSel} options={fields.map((x) => x.title)} /> : undefined}
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
        xLabel="Focus shift (mm)"
        yLabel="Modulation"
        xDomain={hi > lo ? [lo, hi] : undefined}
        yDomain={[0, 1.02]}
        markers={[{ x: 0, label: 'Image' }]}
        label={`Through-focus MTF, ${f.title}`}
      />
    </PlotPane>
  )
}

function throughFocusKeys(r: AnalysisResult): KeyNumber[] {
  // The focus shift of the highest tangential modulation at the first frequency, field 1.
  const focus = series(r, r.json.focus_mm)
  const c = objects(objects(r.json.fields)[0]?.curves)[0]
  if (!c || !focus.length) return []
  const t = series(r, c.tangential)
  let best = -1
  for (let i = 0; i < Math.min(t.length, focus.length); i++) if (Number.isFinite(t[i]) && (best < 0 || t[i] > t[best])) best = i
  if (best < 0) return []
  return [{ label: 'Peak (T)', value: `${focus[best] >= 0 ? '+' : '−'}${Math.abs(focus[best]).toFixed(4)} mm`, help: `Focus shift of the highest tangential MTF at ${short(nf(c.frequency))} cycles/mm, first field` }]
}

// MARK: Line spread

function LineSpread({ result, system }: RendererProps) {
  const dark = useDark()
  const items = objects(result.json.fields)
    .map((f, k) => ({
      label: fieldTitle(system, intOf(f.field, k), f.field_value),
      x: series(result, f.positions_mm).map((v) => v * 1000),
      lx: series(result, f.x),
      ly: series(result, f.y),
    }))
    .filter((i) => i.x.length > 0)
  if (!items.length) return <Empty title="No line-spread functions in this result" />
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
      <XYPlot series={plot} xLabel="Position (µm)" yLabel="Line spread (mm⁻¹)" label="Line-spread functions along x and y per field" />
    </PlotPane>
  )
}

// MARK: System transmission

function Transmission({ result, system }: RendererProps) {
  const dark = useDark()
  const j = result.json
  const points = objects(j.points).map((p) => ({ field: intOf(p.field, 0), um: nf(p.wavelength_um), t: nf(p.transmission) }))
  const poly = nums(j.polychromatic)
  const fieldDefs = objects(j.fields)
  if (!points.length) return <Empty title="No transmission in this result" />
  const fields = [...new Set(points.map((p) => p.field))].sort((a, b) => a - b)
  const labelOf = (f: number) => {
    const d = fieldDefs[f]
    if (d && (d.hx !== undefined || d.hy !== undefined)) return `H (${nf(d.hx ?? 0).toFixed(2)}, ${nf(d.hy ?? 0).toFixed(2)})`
    if (d && d.field !== undefined) return fieldTitle(system, intOf(d.field, f), d.field_value)
    return `Field ${f + 1}`
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
          xLabel="Wavelength (µm)"
          yLabel="Transmission"
          yDomain={[0, 1.02]}
          label="System transmission versus wavelength per field"
        />
      ) : (
        <BarPlot
          categories={fields.map((f) => String(f + 1))}
          groups={[{ label: 'Transmission', color: seriesColor(0, dark), values: fields.map((f) => points.find((p) => p.field === f)?.t ?? NaN) }]}
          xLabel={`Field${Number.isFinite(points[0].um) ? ` · ${sig(points[0].um, 4)} µm` : ''}`}
          yLabel="Transmission"
          yDomain={[0, 1]}
          label="System transmission per field"
        />
      )}
    </PlotPane>
  )
}

function transmissionKeys(r: AnalysisResult): KeyNumber[] {
  const poly = nums(r.json.polychromatic)
  const m = minOf(poly)
  const out: KeyNumber[] = []
  if (Number.isFinite(m)) out.push({ label: 'Min transmission', value: percent(m), help: 'Lowest polychromatic transmission over the fields' })
  const ax = poly[0]
  if (Number.isFinite(ax) && poly.length > 1) out.push({ label: 'Field 1', value: percent(ax) })
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

