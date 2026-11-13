// Spot diagram: a field × focus grid of spots on one common box scale.
import { useEffect, useMemo, useRef } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import type { System } from '@/model/system'
import { cssVar, useDark } from '@/lib/theme'
import { fmt, nf, sig } from '@/model/format'
import { arr, buffer, fieldLabel, isObj, nums, niceCeil, obj, setupCanvas, useSize, wavelengthsUm, wlColor } from './common'
import { Legend, Swatch } from './Legend'

interface Cell {
  field: number
  plane: number
  rms: number
}

interface SpotData {
  fields: JsonObject[]
  planes: number[]
  hasPlanes: boolean
  base: number
  idx: number[][]
  pts: ArrayLike<number> | null
  slope: ArrayLike<number> | null
  pw: ArrayLike<number> | null
  box: number
  airy: number
  wls: number[]
  selectedWls: number[]
}

function prepare(r: AnalysisResult, system: System): SpotData {
  const j = r.json
  const fields = arr(j.fields).filter(isObj)
  const base = Number.isFinite(nf(j.defocus)) ? nf(j.defocus) : 0
  const tf = fields.length ? arr(fields[0].through_focus).filter(isObj) : []
  const hasPlanes = tf.length > 0
  const planes = hasPlanes ? tf.map((p) => nf(p.defocus)) : [base]
  const pts = buffer(r, 'points')
  const pf = buffer(r, 'point_field')
  const pw = buffer(r, 'point_wavelength')
  const slope = buffer(r, 'point_slope')
  const n = pts ? Math.floor(pts.length / 2) : 0
  const idx: number[][] = fields.map(() => [])
  for (let i = 0; i < n; i++) {
    const f = pf ? pf[i] : 0
    if (f >= 0 && f < idx.length) idx[f].push(i)
  }
  // Common box: the largest extent over every cell, and the Airy disk.
  let ext = 0
  for (let f = 0; f < fields.length; f++)
    for (const d of planes) {
      const dz = d - base
      for (const i of idx[f]) {
        let x = pts![2 * i], y = pts![2 * i + 1]
        if (slope && hasPlanes) {
          x += dz * slope[2 * i]
          y += dz * slope[2 * i + 1]
        }
        if (Number.isFinite(x) && Number.isFinite(y)) ext = Math.max(ext, Math.abs(x), Math.abs(y))
      }
    }
  const airy = nf(j.airy_radius)
  if (Number.isFinite(airy)) ext = Math.max(ext, airy)
  if (ext === 0) for (const f of fields) ext = Math.max(ext, Number.isFinite(nf(f.geo_radius)) ? nf(f.geo_radius) : 0)
  const box = niceCeil(2 * ext * 1.08 || 0.01)
  return { fields, planes, hasPlanes, base, idx, pts, slope, pw, box, airy, wls: wavelengthsUm(j, system), selectedWls: nums(j.wavelengths) }
}

function cellRms(d: SpotData, f: number, p: number, reference: string): number {
  const fj = d.fields[f]
  if (!d.hasPlanes) return nf(fj.rms_radius)
  const tf = obj(arr(fj.through_focus)[p])
  if (!tf) return NaN
  return nf(reference === 'centroid' ? tf.rms_radius_centroid : tf.rms_radius_chief)
}

function SpotCell(props: { d: SpotData; cell: Cell; size: number; dark: boolean; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const { d, cell, size, dark } = props
  useEffect(() => {
    const ctx = setupCanvas(ref.current, size, size)
    if (!ctx) return
    const c = size / 2
    const s = size / d.box
    ctx.strokeStyle = cssVar('--separator') || '#ccc'
    ctx.lineWidth = 1
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1)
    ctx.strokeStyle = cssVar('--grid') || '#eee'
    ctx.beginPath()
    ctx.moveTo(c, 1)
    ctx.lineTo(c, size - 1)
    ctx.moveTo(1, c)
    ctx.lineTo(size - 1, c)
    ctx.stroke()
    if (Number.isFinite(d.airy) && d.airy > 0) {
      ctx.strokeStyle = cssVar('--secondary') || '#888'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.arc(c, c, d.airy * s, 0, 2 * Math.PI)
      ctx.stroke()
      ctx.setLineDash([])
    }
    const ids = d.idx[cell.field] ?? []
    const pts = d.pts
    if (!pts) return
    const dz = d.planes[cell.plane] - d.base
    ctx.globalAlpha = ids.length > 1500 ? 0.35 : ids.length > 400 ? 0.6 : 0.9
    let colour = ''
    for (const i of ids) {
      let x = pts[2 * i], y = pts[2 * i + 1]
      if (d.slope && d.hasPlanes) {
        x += dz * d.slope[2 * i]
        y += dz * d.slope[2 * i + 1]
      }
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue
      const col = wlColor(d.wls, d.pw ? d.pw[i] : 0, dark)
      if (col !== colour) ctx.fillStyle = colour = col
      ctx.fillRect(c + x * s - 0.75, c - y * s - 0.75, 1.5, 1.5)
    }
    ctx.globalAlpha = 1
  }, [d, cell, size, dark])
  return (
    <div className="flex flex-col items-center gap-0.5">
      <canvas ref={ref} role="img" aria-label={props.label} />
      <div className="num text-[11px] text-secondary">RMS {Number.isFinite(cell.rms) ? fmt(cell.rms * 1000, 2) : '–'} µm</div>
    </div>
  )
}

export function SpotGrid(props: { result: AnalysisResult; system: System }) {
  const dark = useDark()
  const d = useMemo(() => prepare(props.result, props.system), [props.result, props.system])
  const [ref, size] = useSize<HTMLDivElement>()
  const reference = String(props.result.runInfo?.settings?.reference ?? 'chief')
  const cols = d.planes.length
  const rows = d.fields.length
  const headW = 88
  const colHead = d.hasPlanes ? 18 : 0
  const footer = 40
  const caption = 18
  const gap = 8
  const byW = (size.w - headW - 16 - gap * cols) / Math.max(cols, 1)
  const byH = (size.h - colHead - footer - 16) / Math.max(rows, 1) - caption - gap
  const cell = Math.floor(Math.max(56, Math.min(260, byW, Math.max(byH, 96))))

  if (!rows) return <div className="p-4 text-secondary">No fields in this result.</div>
  const boxText = d.box >= 1 ? `${sig(d.box, 3)} mm` : `${sig(d.box, 3)} mm (${sig(d.box * 1000, 3)} µm)`

  return (
    <div ref={ref} className="absolute inset-0 flex flex-col">
      <div className="min-h-0 flex-1 overflow-auto p-2">
        <div className="grid w-max gap-x-2 gap-y-2" style={{ gridTemplateColumns: `${headW}px repeat(${cols}, ${cell}px)` }}>
          {d.hasPlanes && (
            <>
              <div />
              {d.planes.map((p, k) => (
                <div key={k} className="num text-center text-[11px] text-secondary">
                  Δ {fmt(p, 3)}
                </div>
              ))}
            </>
          )}
          {d.fields.map((f, fi) => {
            const label = fieldLabel(props.system, f.field, f.field_value)
            return [
              <div key={`h${fi}`} className="self-center pr-1 text-right text-[12px] text-label" style={{ marginBottom: caption }}>
                {label.split(' · ').map((t, i) => (
                  <div key={i} className={i ? 'num text-secondary' : ''}>
                    {t}
                  </div>
                ))}
              </div>,
              ...d.planes.map((_, pi) => (
                <SpotCell
                  key={`${fi}-${pi}`}
                  d={d}
                  cell={{ field: fi, plane: pi, rms: cellRms(d, fi, pi, reference) }}
                  size={cell}
                  dark={dark}
                  label={`Spot diagram, ${label}${d.hasPlanes ? `, defocus ${fmt(d.planes[pi], 3)} mm` : ''}`}
                />
              )),
            ]
          })}
        </div>
      </div>
      <Legend>
        <span className="num">Box {boxText}</span>
        {Number.isFinite(d.airy) && <span className="num">Airy radius {fmt(d.airy * 1000, 2)} µm (dashed)</span>}
        {(d.selectedWls.length ? d.selectedWls : d.wls.map((_, k) => k)).map((w) => (
          <Swatch key={w} color={wlColor(d.wls, w, dark)} kind="dot" label={`${sig(d.wls[w], 4)} µm`} />
        ))}
      </Legend>
    </div>
  )
}
