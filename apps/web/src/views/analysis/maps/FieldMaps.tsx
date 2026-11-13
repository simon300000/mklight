// Field maps (`pupil_aberration`, `relative_illumination_map`,
// `chief_ray_angle`): a grid run is a heat map over the normalised field with
// the sample dots, an x / y sweep is a curve.
import { useMemo, useState } from 'react'
import type uPlot from 'uplot'
import { Segmented } from '@/components/ui/segmented'
import { cssVar, seriesColor, useDark } from '@/lib/theme'
import { Legend, Swatch } from '@/views/results/Legend'
import { DASH, LINE, UPlot, axis, calmCursor, gaps, type PlotOptions } from '@/views/results/UPlot'
import type { KeyNumber, RendererProps } from '../registry'
import { ColumnPicker } from './Controls'
import {
  argExtreme,
  columnFormat,
  columnIndex,
  columnRange,
  columnTitle,
  columnTitleWithUnit,
  columnUnit,
  countOf,
  craBands,
  gridOf,
  imageRadius,
  isRadial,
  isSigned,
  layoutOf,
  lengthText,
  limitAt,
  radialOf,
  readFieldMap,
  sensorLimit,
  spanOf,
  sweepOf,
  valueAt,
  type FieldMapData,
} from './fieldMap'
import { finiteRange, heatColor, heatRange, isoLevels, mapContours, scaleFor, type HeatRangeMode } from './heatScale'
import { MapCanvas } from './MapCanvas'

const Empty = ({ text }: { text: string }) => <div className="p-4 text-secondary">{text}</div>

/** A column: the map (grid runs) or the curve (sweeps). */
function FieldMapContent(props: { data: FieldMapData; column: number; contours?: number[]; rangeMode?: HeatRangeMode; isFraction?: boolean }) {
  const dark = useDark()
  const { data, column } = props
  const col = data.columns[column]
  const format = columnFormat(col)
  const range = columnRange(data, column)
  const signed = isSigned(data, column)
  const grid = useMemo(() => gridOf(data, column), [data, column])
  const contours = useMemo(() => (props.contours ?? []).map((level) => ({ level, segments: mapContours(grid, level) })), [grid, props.contours])
  if (!range) return <Empty text={`${columnTitle(col)} is undefined at every sample${col.definition ? `: ${col.definition}` : ''}.`} />
  if (layoutOf(data) !== 'map') return <SweepPlot data={data} column={column} />
  const r = heatRange(range, signed, props.rangeMode, props.isFraction)
  const kind = scaleFor(signed)
  const color = heatColor(kind, r, dark)
  return (
    <MapCanvas
      grid={grid}
      color={color}
      bar={{ range: r, color, unit: columnUnit(col) }}
      contours={contours}
      nodes
      hatch={grid.unitDisk || Array.from(grid.values).some((v) => !Number.isFinite(v))}
      xLabel="Hx"
      yLabel="Hy"
      format={format}
      dark={dark}
      label={`${columnTitle(col)} map over the normalized field, ${data.nx} by ${data.ny} samples`}
    />
  )
}

function SweepPlot({ data, column }: { data: FieldMapData; column: number }) {
  const dark = useDark()
  const col = data.columns[column]
  const pts = useMemo(() => sweepOf(data, column).sort((a, b) => a[0] - b[0]), [data, column])
  const plotData = useMemo(() => [pts.map((p) => p[0]), gaps(pts.map((p) => p[1]))] as uPlot.AlignedData, [pts])
  const options = useMemo<PlotOptions>(
    () => ({
      legend: { show: false },
      cursor: calmCursor,
      scales: { x: { time: false } },
      axes: [axis(layoutOf(data) === 'sweepX' ? 'Hx' : 'Hy'), axis(columnTitleWithUnit(col), { size: 48 })],
      series: [{}, { label: columnTitle(col), stroke: seriesColor(0, dark), width: LINE * 1.3, points: { show: pts.length <= 41, size: 4 } }],
    }),
    [data, col, dark, pts.length],
  )
  return <UPlot options={options} data={plotData} label={`${columnTitle(col)} along the field sweep`} />
}

function firstVarying(d: FieldMapData): number {
  const k = d.columns.findIndex((_, c) => {
    const r = columnRange(d, c)
    return r !== null && r.max - r.min > 1e-12
  })
  return k < 0 ? 0 : k
}

// MARK: Pupil aberration

export function PupilAberrationView(p: RendererProps) {
  const data = useMemo(() => readFieldMap(p.result), [p.result])
  const [column, setColumn] = useState<number | null>(null)
  if (!data) return <Empty text="No field map in this result." />
  const c = Math.min(column ?? firstVarying(data), data.columns.length - 1)
  return (
    <div className="absolute inset-0 flex flex-col gap-1.5 p-2.5">
      <ColumnPicker columns={data.columns} value={c} onChange={setColumn} />
      <div className="min-h-0 flex-1">
        <FieldMapContent data={data} column={c} />
      </div>
    </div>
  )
}

// MARK: Relative illumination map

const FRACTIONS = ['relative_illumination', 'vignetting_fraction']

export function RelativeIlluminationMapView(p: RendererProps) {
  const dark = useDark()
  const data = useMemo(() => readFieldMap(p.result), [p.result])
  const [column, setColumn] = useState<number | null>(null)
  const [rangeMode, setRangeMode] = useState<HeatRangeMode>('data')
  const c = data ? Math.min(column ?? columnIndex(data, 'relative_illumination') ?? 0, data.columns.length - 1) : 0
  const col = data?.columns[c]
  const isFraction = !!col && FRACTIONS.includes(col.name)
  const isRI = col?.name === 'relative_illumination'
  const levels = useMemo(() => {
    if (!data || !isRI) return []
    return isoLevels(columnRange(data, c) ?? { min: 0, max: 1 })
  }, [data, c, isRI])
  const radial = useMemo(() => (data ? radialOf(data, c) : []), [data, c])
  const line = data ? isRadial(data, c, 1e-2) : false
  const radialData = useMemo(() => [radial.map((q) => q[0]), radial.map((q) => q[1])] as uPlot.AlignedData, [radial])
  const radialOptions = useMemo<PlotOptions>(() => {
    const ys = radial.map((q) => q[1])
    const yr = isFraction && rangeMode === 'unit' ? ([0, 1.02] as [number, number]) : spanOf(ys)
    const pad = isFraction && rangeMode === 'unit' ? 0 : (yr[1] - yr[0]) * 0.05
    return {
      legend: { show: false },
      cursor: calmCursor,
      scales: { x: { time: false }, y: { range: [yr[0] - pad, yr[1] + pad] } },
      axes: [axis('Image radius (mm)'), axis(col ? columnTitle(col) : '', { size: 44 })],
      series: [{}, { label: col ? columnTitle(col) : undefined, stroke: seriesColor(0, dark), width: line ? 1.6 : 0, points: { show: true, size: line ? 4 : 5, fill: seriesColor(0, dark) } }],
    }
  }, [radial, isFraction, rangeMode, col, dark, line])
  if (!data || !col) return <Empty text="No field map in this result." />
  const isMap = layoutOf(data) === 'map'
  return (
    <div className="absolute inset-0 flex flex-col gap-1.5 p-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <ColumnPicker columns={data.columns} value={c} onChange={setColumn} />
        {isMap && isFraction && (
          <Segmented
            label="Color scale"
            value={rangeMode}
            onChange={setRangeMode}
            options={[
              { value: 'data', title: 'Data range', help: "Colors over the map's own range: small differences show" },
              { value: 'unit', title: '0 to 1', help: 'Colors over 0 to 1, to compare designs' },
            ]}
          />
        )}
      </div>
      {isMap ? (
        <div className="flex min-h-0 flex-1 flex-row gap-3 @max-[560px]:flex-col">
          <div className="min-h-0 min-w-0 flex-[3]">
            <FieldMapContent data={data} column={c} contours={levels} rangeMode={rangeMode} isFraction={isFraction} />
          </div>
          <div className="flex min-h-[140px] min-w-0 flex-[2] flex-col">
            <div className="px-1 text-[11px] font-semibold">{columnTitle(col)} vs image radius</div>
            <div className="min-h-0 flex-1">
              <UPlot options={radialOptions} data={radialData} label={`${columnTitle(col)} versus image radius, ${radial.length} samples`} />
            </div>
          </div>
        </div>
      ) : (
        <div className="min-h-0 flex-1">
          <FieldMapContent data={data} column={c} />
        </div>
      )}
    </div>
  )
}

// MARK: Chief ray angle

export function ChiefRayAngleView(p: RendererProps) {
  const dark = useDark()
  const data = useMemo(() => readFieldMap(p.result), [p.result])
  const inc = data ? columnIndex(data, 'incidence') : null
  const limit = useMemo(() => sensorLimit(p.result.runInfo.settings?.sensor_limit ?? p.settings.sensor_limit), [p.result, p.settings])
  const [display, setDisplay] = useState<'curve' | 'map' | null>(null)
  const [column, setColumn] = useState<number | null>(null)
  if (!data || inc === null) return <Empty text="No chief ray angles in this result." />
  const isMap = layoutOf(data) === 'map'
  const shown = display ?? (isMap && !isRadial(data, inc, 1e-3) ? 'map' : 'curve')
  const mapColumns = data.columns.map((_, k) => k).filter((k) => data.columns[k].name !== 'limit' && columnRange(data, k) !== null)
  const col = column !== null && mapColumns.includes(column) ? column : inc
  return (
    <div className="absolute inset-0 flex flex-col gap-1.5 p-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {isMap && (
          <Segmented
            label="Display"
            value={shown}
            onChange={setDisplay}
            options={[
              { value: 'curve', title: 'Curve' },
              { value: 'map', title: 'Map' },
            ]}
          />
        )}
        {shown === 'map' && isMap ? (
          <ColumnPicker columns={data.columns} indices={mapColumns} value={col} onChange={setColumn} />
        ) : (
          <Legend className="px-0 py-0">
            <Swatch color={seriesColor(0, dark)} label="Chief ray incidence" />
            {limit.length > 0 && (
              <>
                <Swatch color={cssVar('--secondary') || '#888'} kind="dash" label="Sensor limit" />
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2 w-3 rounded-[2px]" style={{ background: seriesColor(5, dark), opacity: 0.3 }} />
                  Margin
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2 w-3 rounded-[2px]" style={{ background: dark ? '#ff6961' : '#d70015', opacity: 0.5 }} />
                  Violation
                </span>
              </>
            )}
          </Legend>
        )}
      </div>
      <div className="min-h-0 flex-1">{shown === 'map' && isMap ? <FieldMapContent data={data} column={col} /> : <CraCurve data={data} inc={inc} limit={limit} dark={dark} />}</div>
    </div>
  )
}

function CraCurve({ data, inc, limit, dark }: { data: FieldMapData; inc: number; limit: [number, number][]; dark: boolean }) {
  const lay = layoutOf(data)
  const { plotData, bands, xLabel } = useMemo(() => {
    const heights = lay === 'sweepX' ? data.image.map((q) => q[0]) : lay === 'sweepY' ? data.image.map((q) => q[1]) : imageRadius(data)
    const samples = heights
      .map((h, k) => [h, valueAt(data, k, inc)] as [number, number])
      .filter((q) => Number.isFinite(q[0]) && Number.isFinite(q[1]))
      .sort((a, b) => a[0] - b[0])
    let curve: [number, number][] | null = null
    if (lay !== 'map') curve = samples
    else if (isRadial(data, inc)) {
      curve = []
      for (const q of samples) if (!curve.length || Math.abs(curve[curve.length - 1][0] - q[0]) > 1e-9) curve.push(q)
    }
    const bands = curve ? craBands(curve, limit) : []
    const xr = spanOf([...samples.map((q) => q[0]), ...(lay === 'map' ? [0] : [])])
    const xs = new Set<number>(samples.map((q) => q[0]))
    if (limit.length) {
      xs.add(xr[0])
      xs.add(xr[1])
      for (const [r] of limit) for (const x of [r, -r]) if (x > xr[0] && x < xr[1]) xs.add(x)
    }
    const x = [...xs].sort((a, b) => a - b)
    const at = (pts: [number, number][]) => {
      const m = new Map(pts.map((q) => [q[0], q[1]]))
      return x.map((v) => (m.has(v) ? m.get(v)! : null))
    }
    const series: (number | null)[][] = [x, curve ? at(curve) : x.map(() => null), at(samples)]
    if (limit.length) series.push(x.map((v) => limitAt(limit, Math.abs(v))))
    const xLabel = lay === 'map' ? 'Image radius (mm)' : lay === 'sweepX' ? 'Image height x (mm)' : 'Image height y (mm)'
    return { plotData: series as uPlot.AlignedData, bands, xLabel }
  }, [data, inc, limit, lay])

  const options = useMemo<PlotOptions>(() => {
    const secondary = cssVar('--secondary') || '#888'
    const ok = seriesColor(5, dark)
    const bad = dark ? 'rgba(255,105,97,0.5)' : 'rgba(215,0,21,0.42)'
    const ys = [...(plotData[2] as (number | null)[]), ...((plotData[3] as (number | null)[] | undefined) ?? [])].filter((v): v is number => v !== null && Number.isFinite(v))
    const ymax = Math.max((finiteRange(ys)?.max ?? 1) * 1.08, 1e-3)
    const series: uPlot.Series[] = [
      {},
      { label: 'Incidence', stroke: seriesColor(0, dark), width: 1.8, spanGaps: true, points: { show: false } },
      { label: 'Incidence', stroke: seriesColor(0, dark), width: 0, points: { show: true, size: 4, fill: seriesColor(0, dark) } },
    ]
    if (plotData.length > 3) series.push({ label: 'Sensor limit', stroke: secondary, width: 1.4, dash: DASH, points: { show: false } })
    return {
      legend: { show: false },
      cursor: calmCursor,
      scales: { x: { time: false }, y: { range: [0, ymax] } },
      axes: [axis(xLabel), axis('Incidence (°)', { size: 40 })],
      series,
      hooks: {
        drawClear: [
          (u: uPlot) => {
            const ctx = u.ctx
            ctx.save()
            for (const b of bands) {
              ctx.beginPath()
              b.points.forEach((q, k) => {
                const X = u.valToPos(q.x, 'x', true), Y = u.valToPos(q.hi, 'y', true)
                if (k === 0) ctx.moveTo(X, Y)
                else ctx.lineTo(X, Y)
              })
              for (let k = b.points.length - 1; k >= 0; k--) ctx.lineTo(u.valToPos(b.points[k].x, 'x', true), u.valToPos(b.points[k].lo, 'y', true))
              ctx.closePath()
              ctx.globalAlpha = b.violated ? 1 : 0.16
              ctx.fillStyle = b.violated ? bad : ok
              ctx.fill()
            }
            ctx.restore()
          },
        ],
      },
    }
  }, [plotData, bands, xLabel, dark])

  return <UPlot options={options} data={plotData} label="Chief ray incidence versus image height, with the sensor limit" />
}

// MARK: Key numbers

const pct = (v: number) => `${(v * 100).toFixed(2).replace('-', '−')} %`

export function fieldMapKeyNumbers(analysis: string) {
  return (r: RendererProps['result']): KeyNumber[] => {
    const d = readFieldMap(r)
    if (!d) return []
    const out: KeyNumber[] = []
    if (analysis === 'pupil_aberration') {
      const x = columnIndex(d, 'stop_error_x'), y = columnIndex(d, 'stop_error_y')
      if (x !== null && y !== null) {
        const e = Array.from({ length: countOf(d) }, (_, k) => Math.hypot(valueAt(d, k, x), valueAt(d, k, y))).filter(Number.isFinite)
        if (e.length) out.push({ label: 'Max stop error', value: lengthText(Math.max(...e)), help: 'Largest chief-ray miss of the stop centre' })
      }
      const v = ['variation_x', 'variation_y']
        .map((n) => columnIndex(d, n))
        .filter((c): c is number => c !== null)
        .flatMap((c) => Array.from({ length: countOf(d) }, (_, k) => valueAt(d, k, c)))
        .filter(Number.isFinite)
      if (v.length) {
        const m = v.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a))
        out.push({ label: 'Max variation', value: `${m >= 0 ? '+' : '−'}${Math.abs(m * 100).toFixed(2)} %`, help: 'Largest pupil size variation' })
      }
    } else if (analysis === 'relative_illumination_map') {
      const c = columnIndex(d, 'relative_illumination')
      const r = c !== null ? columnRange(d, c) : null
      if (r) out.push({ label: 'Min RI', value: r.min.toFixed(3), help: 'Lowest relative illumination over the map' })
      const v = columnIndex(d, 'vignetting_fraction')
      const rv = v !== null ? columnRange(d, v) : null
      if (rv && rv.max > 0) out.push({ label: 'Max vignetting', value: pct(rv.max) })
    } else if (analysis === 'chief_ray_angle') {
      const c = columnIndex(d, 'incidence')
      const k = c !== null ? argExtreme(d, c, 1) : null
      if (c !== null && k !== null) {
        out.push({ label: 'Max CRA', value: `${valueAt(d, k, c).toFixed(2)}°`, help: 'Largest chief ray incidence on the sensor' })
        const h = imageRadius(d)[k]
        if (Number.isFinite(h)) out.push({ label: 'at', value: `${Number(h.toPrecision(3))} mm`, help: 'Image radius of the largest incidence' })
      }
      const m = columnIndex(d, 'margin')
      const rm = m !== null ? columnRange(d, m) : null
      if (rm) out.push({ label: 'Min margin', value: `${rm.min >= 0 ? '+' : '−'}${Math.abs(rm.min).toFixed(2)}°`, help: 'Smallest margin below the sensor limit' })
    }
    return out
  }
}
