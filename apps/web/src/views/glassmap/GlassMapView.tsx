// The `glass_map` and `glass_info` analysis panes (apps/macos/Sources/MKLightUI/Views/
// Analysis/GlassMapView.swift and GlassDataView.swift): the map with a plane
// switch, a catalogue filter, the status legend, the selected glass and the
// design's glasses; and one glass's data with its dispersion curve.

import { useEffect, useMemo, useRef, useState } from 'react'
import type { AnalysisResult } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { fmt, nf, sig } from '@/model/format'
import { displayName, inDesign, mapCatalogs, mapStatuses, readGlassInfo, readGlassMap, surfaceLabel, type GlassInfo } from '@/model/glass'
import type { System } from '@/model/system'
import { useView } from '@/state/view'
import { cssVar, useDark, wavelengthColor } from '@/lib/theme'
import { FONT, niceTicks, setupCanvas, useSize } from '../results/common'
import { GlassCard, GlassMapChart, GlassStatusLegend, Sub, type GlassMapPlane } from './GlassMapChart'

export function GlassMapResultView({ result, system }: { result: AnalysisResult; system: System }) {
  const glasses = useMemo(() => readGlassMap(result), [result])
  const [plane, setPlane] = useState<GlassMapPlane>('nd')
  const [selection, setSelection] = useState<string | null>(null)
  const [catalog, setCatalog] = useState<string | null>(null)
  const [box, size] = useSize<HTMLDivElement>()
  const catalogs = useMemo(() => mapCatalogs(glasses), [glasses])
  const design = glasses.filter(inDesign)
  const selected = selection ? glasses.find((g) => g.id === selection) : null
  const wide = size.w >= 600
  if (!glasses.length) return <p className="p-4 text-[12px] text-secondary">No glasses are loaded.</p>

  const selectedCard = selected ? (
    <div className="rounded-lg bg-hover p-2.5" data-testid="glass-selected">
      <GlassCard glass={selected} />
      {selected.surfaces.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1 text-[11px] text-secondary">
          Select
          {selected.surfaces.map((s) => (
            <Button key={s} size="sm" title={`Select surface ${s} in the lens table`} onClick={() => useView.getState().select(s)}>
              S{s}
            </Button>
          ))}
        </div>
      )}
    </div>
  ) : (
    <p className="text-[11px] text-secondary">Click a glass to inspect it.</p>
  )

  return (
    <div ref={box} className="flex h-full min-h-0 flex-col gap-2 overflow-auto p-3" data-testid="glass-map-view">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <Segmented
          label="Plane"
          value={plane}
          onChange={setPlane}
          options={[
            { value: 'nd', title: <Sub base="n" sub="d" />, help: 'Refractive index n_d against Abbe number V_d' },
            { value: 'pgf', title: <Sub base="P" sub="g,F" />, help: 'Partial dispersion P_g,F against Abbe number V_d' },
          ]}
        />
        {catalogs.length > 1 && (
          <Select aria-label="Catalog" value={catalog ?? ''} onChange={(e) => setCatalog(e.target.value || null)}>
            <option value="">All catalogs</option>
            {catalogs.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </Select>
        )}
        <GlassStatusLegend statuses={mapStatuses(glasses)} showsOffset={glasses.some((g) => g.kind === 'offset')} />
      </div>
      <div className={wide ? 'flex min-h-0 flex-1 gap-3' : 'flex flex-col gap-3'}>
        <div className="min-w-0 flex-1">
          <GlassMapChart
            map={result}
            system={system}
            plane={plane}
            selection={selection}
            onSelect={setSelection}
            catalog={catalog}
            height={wide ? Math.max(260, size.h - 60) : 280}
          />
        </div>
        <div className={wide ? 'flex w-[210px] shrink-0 flex-col gap-3' : 'flex flex-col gap-3'}>
          {selectedCard}
          {wide && design.length > 0 && (
            <div>
              <div className="mb-1 text-[11px] font-semibold">In the design</div>
              {design.map((g) => (
                <button
                  key={g.id}
                  type="button"
                  className="flex w-full cursor-default items-center justify-between gap-2 rounded px-1 py-0.5 text-left text-[12px] hover:bg-hover"
                  onClick={() => setSelection(g.id)}
                  aria-label={`${g.name}, surfaces ${surfaceLabel(g)}`}
                >
                  <span className="truncate">{displayName(g)}</span>
                  <span className="shrink-0 text-secondary">{surfaceLabel(g)}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// MARK: Glass data

export function GlassInfoResultView({ result, system }: { result: AnalysisResult; system: System }) {
  const d = useMemo(() => readGlassInfo(result), [result])
  if (!d) return <p className="p-4 text-[12px] text-secondary">No glass data.</p>
  const settingsWl = result.runInfo.settings?.spectrum
  const primaryUm =
    settingsWl == null && system.wavelengths?.wavelengths?.[system.wavelengths.primary] ? nf(system.wavelengths.wavelengths[system.wavelengths.primary].um) : null
  const hasOffset = d.dnd !== 0 || d.dvd !== 0
  const env = [d.temperatureC != null ? `${fmt(d.temperatureC, 1)} °C` : '', d.pressureAtm != null ? `${fmt(d.pressureAtm, 3)} atm` : '']
    .filter(Boolean)
    .join(', ')
  const range = Number.isFinite(d.minUm) && Number.isFinite(d.maxUm) ? `${fmt(d.minUm, 3)}–${fmt(d.maxUm, 3)} µm` : ''
  const props: [string, string][] = []
  if (d.formula) props.push(['Dispersion formula', d.formula])
  if (range) props.push(['Valid range', range])
  props.push(['Thermo-optic data (dn/dT)', d.hasThermalData ? 'Yes' : 'None'])
  props.push(['Thermal expansion', d.cte != null ? `${fmt(d.cte * 1e6, 2)} × 10⁻⁶/K` : '–'])
  if (d.density != null) props.push(['Density', `${fmt(d.density, 3)} g/cm³`])
  if (d.relativeCost != null) props.push(['Relative cost', fmt(d.relativeCost, 2)])

  return (
    <div className="flex h-full flex-col gap-4 overflow-auto p-3 text-[12px]" data-testid="glass-info-view">
      <div className="rounded-lg bg-hover p-3">
        <div className="flex items-baseline gap-2">
          <span className="select-text text-[16px] font-semibold">{d.name}</span>
          {d.catalog && <span className="text-secondary">{d.catalog}</span>}
          {hasOffset && (
            <span
              className="rounded-full bg-accent-soft px-2 text-[11px] font-medium"
              title={`Δn_d ${sig(d.dnd, 5)}, ΔV_d ${sig(d.dvd, 4)} from the catalog values`}
            >
              Offset
            </span>
          )}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-6 gap-y-2">
          <Stat label={<Sub base="n" sub="d" />} value={fmt(d.nd, 6)} help="Refractive index at the d line (587.56 nm)" />
          <Stat label={<Sub base="V" sub="d" />} value={fmt(d.vd, 2)} help="Abbe number (n_d − 1) / (n_F − n_C)" />
          <Stat label={<Sub base="P" sub="g,F" />} value={fmt(d.pgf, 4)} help="Relative partial dispersion (n_g − n_F) / (n_F − n_C)" />
          <Stat label={<Sub base="n" sub="e" />} value={fmt(d.ne, 6)} help="Refractive index at the e line (546.07 nm)" />
        </div>
        {hasOffset && (
          <p className="mt-1 text-[11px] text-secondary">
            Catalog <Sub base="n" sub="d" /> {fmt(d.catalogNd, 6)}, <Sub base="V" sub="d" /> {fmt(d.catalogVd, 2)}
          </p>
        )}
      </div>
      <section>
        <h4 className="mb-1 font-semibold">
          Dispersion {range && <span className="font-normal text-secondary">valid {range}</span>}
        </h4>
        {d.curve.length > 1 ? (
          <DispersionChart info={d} />
        ) : (
          <p className="text-secondary">Set Curve Samples above zero to plot the index over the glass’s range.</p>
        )}
      </section>
      {d.indices.length > 0 && (
        <section>
          <h4 className="mb-1 font-semibold">
            Indices {env && <span className="font-normal text-secondary">{env}</span>}
          </h4>
          <table className="num select-text">
            <thead className="text-[11px] text-secondary">
              <tr>
                <th className="pr-4 text-left font-medium">λ (µm)</th>
                <th className="pr-4 text-right font-medium" title="Index relative to air at the analysis temperature and pressure">
                  n
                </th>
                <th className="pr-4 text-right font-medium" title="Index at the catalog's reference conditions">
                  n cat.
                </th>
                <th className="text-right font-medium" title="Internal absorption coefficient">
                  α (1/mm)
                </th>
              </tr>
            </thead>
            <tbody>
              {d.indices.map((r, k) => {
                const primary = primaryUm != null && Math.abs(primaryUm - r.um) < 1e-9
                return (
                  <tr key={k}>
                    <td className={primary ? 'pr-4 font-semibold' : 'pr-4'}>
                      <span className="mr-1.5 inline-block size-2 rounded-full" style={{ background: wavelengthColor(r.um) }} />
                      {fmt(r.um, 4)}
                      {!r.inRange && (
                        <span className="ml-1 text-warning" title="Outside the glass's data range: the index is extrapolated">
                          ⚠
                        </span>
                      )}
                    </td>
                    <td className="pr-4 text-right">{fmt(r.n, 6)}</td>
                    <td className="pr-4 text-right text-secondary">{fmt(r.nCatalog, 6)}</td>
                    <td className="text-right text-secondary">{Number.isFinite(r.absorption) ? sig(r.absorption, 4) : '–'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      )}
      <section>
        <h4 className="mb-1 font-semibold">Properties</h4>
        <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
          {props.map(([k, v]) => (
            <div key={k} className="contents">
              <span className="text-secondary">{k}</span>
              <span className="num select-text">{v}</span>
            </div>
          ))}
        </div>
        {d.comment && <p className="mt-1 select-text text-[11px] text-secondary">{d.comment}</p>}
      </section>
    </div>
  )
}

function Stat({ label, value, help }: { label: React.ReactNode; value: string; help: string }) {
  return (
    <div title={help}>
      <div className="text-[11px] text-secondary">{label}</div>
      <div className="num select-text text-[16px]">{value}</div>
    </div>
  )
}

/** n against λ over the glass's range, the system wavelengths marked on the curve
 *  (triangles when outside the data range). */
function DispersionChart({ info }: { info: GlassInfo }) {
  const [box, size] = useSize<HTMLDivElement>()
  const dark = useDark()
  const ref = useRef<HTMLCanvasElement>(null)
  const H = 220
  useEffect(() => {
    const g = setupCanvas(ref.current, size.w, H)
    if (!g) return
    const pts = info.curve.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    const marks = info.indices.filter((r) => Number.isFinite(r.um) && Number.isFinite(r.n))
    const xs = [...pts.map((p) => p[0]), ...marks.map((m) => m.um)]
    const ys = [...pts.map((p) => p[1]), ...marks.map((m) => m.n)]
    if (!xs.length) return
    const x0 = Math.min(...xs), x1 = Math.max(...xs) > x0 ? Math.max(...xs) : x0 + 1
    let y0 = Math.min(...ys), y1 = Math.max(...ys)
    if (y0 === y1) {
      y0 -= 0.01
      y1 += 0.01
    }
    const pad = (y1 - y0) * 0.05
    y0 -= pad
    y1 += pad
    const L = 58, R = 10, T = 8, B = 30
    const pw = size.w - L - R, ph = H - T - B
    const X = (v: number) => L + ((v - x0) / (x1 - x0)) * pw
    const Y = (v: number) => T + ph - ((v - y0) / (y1 - y0)) * ph
    g.font = FONT
    g.strokeStyle = cssVar('--grid') || 'rgba(0,0,0,0.07)'
    g.fillStyle = cssVar('--secondary') || '#888'
    g.textAlign = 'center'
    g.textBaseline = 'top'
    for (const t of niceTicks(x0, x1, 6)) {
      g.beginPath()
      g.moveTo(X(t), T)
      g.lineTo(X(t), T + ph)
      g.stroke()
      g.fillText(sig(t, 3), X(t), T + ph + 3)
    }
    g.textAlign = 'right'
    g.textBaseline = 'middle'
    for (const t of niceTicks(y0, y1, 4)) {
      g.beginPath()
      g.moveTo(L, Y(t))
      g.lineTo(L + pw, Y(t))
      g.stroke()
      g.fillText(fmt(t, 4), L - 4, Y(t))
    }
    g.textAlign = 'center'
    g.textBaseline = 'alphabetic'
    g.fillText('Wavelength (µm)', L + pw / 2, H - 3)
    g.strokeStyle = cssVar('--label') || '#333'
    g.globalAlpha = 0.75
    g.lineWidth = 1.6
    g.beginPath()
    pts.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))))
    g.stroke()
    g.globalAlpha = 1
    for (const m of marks) {
      const c = wavelengthColor(m.um, dark)
      g.strokeStyle = c
      g.globalAlpha = 0.35
      g.lineWidth = 1
      g.setLineDash([2, 3])
      g.beginPath()
      g.moveTo(X(m.um), T)
      g.lineTo(X(m.um), T + ph)
      g.stroke()
      g.setLineDash([])
      g.globalAlpha = 1
      g.fillStyle = c
      g.beginPath()
      if (m.inRange) g.arc(X(m.um), Y(m.n), 4, 0, 2 * Math.PI)
      else {
        g.moveTo(X(m.um), Y(m.n) - 5)
        g.lineTo(X(m.um) + 4.5, Y(m.n) + 3.5)
        g.lineTo(X(m.um) - 4.5, Y(m.n) + 3.5)
        g.closePath()
      }
      g.fill()
    }
  }, [info, size.w, dark])
  return (
    <div ref={box} className="w-full" style={{ height: H }}>
      <canvas ref={ref} role="img" aria-label="Refractive index versus wavelength" />
    </div>
  )
}
