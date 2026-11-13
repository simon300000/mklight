// A heat map on a canvas with true proportions, axes, a colour bar, optional
// iso-lines (white on a dark halo, labels kept apart), sample dots, a fine
// no-data hatch, pupil scatter cells and the nearest sample on hover.
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { cssVar } from '@/lib/theme'
import { sig } from '@/model/format'
import { FONT, niceTicks, rgbCss, setupCanvas, useSize } from '@/views/results/common'
import { logFloor, nodeAt, outsideDisk, upsample, type Grid, type Range, type RGB, type Segment } from './heatScale'

export interface ColorBarSpec {
  range: Range
  color: (v: number) => RGB
  unit: string
  log?: boolean
  format?: (v: number) => string
}

export interface MapCanvasProps {
  /** Node grid (samples at the nodes, interpolated) or, with `cells`, pixel cells. */
  grid?: Grid
  cells?: boolean
  /** Plot extent when there is no grid (scatter). */
  extent?: { x: [number, number]; y: [number, number] }
  color: (v: number) => RGB | null
  bar?: ColorBarSpec
  contours?: { level: number; segments: Segment[] }[]
  /** Draw the sample nodes as dots. */
  nodes?: boolean
  /** Fine hatch behind the map where it has no data. */
  hatch?: boolean
  /** Scattered samples (pupil maps): square cells of `size` plot units. */
  scatter?: { points: [number, number][]; values: ArrayLike<number>; size: number; disk?: boolean }
  xLabel: string
  yLabel: string
  label: string
  format?: (v: number) => string
  dark: boolean
  /** Extra content under the canvas (legends). */
  children?: ReactNode
}

const DISPLAY = 192

export function MapCanvas(p: MapCanvasProps) {
  const [box, size] = useSize<HTMLDivElement>()
  const ref = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ k: number; X: number; Y: number } | null>(null)
  const display = useMemo(() => (p.grid && !p.cells ? upsample(p.grid, DISPLAY) : p.grid), [p.grid, p.cells])
  const ext = p.grid ? { x: p.grid.x, y: p.grid.y } : (p.extent ?? { x: [-1, 1] as [number, number], y: [-1, 1] as [number, number] })
  const fmt = p.format ?? ((v: number) => sig(v, 4))

  // Plot rectangle in CSS pixels (shared by drawing and hover).
  const rect = useMemo(() => {
    const ml = 46, mb = 34, mt = 8, mr = p.bar ? 78 : 12
    const aw = size.w - ml - mr, ah = size.h - mt - mb
    const aspect = (ext.x[1] - ext.x[0]) / Math.max(ext.y[1] - ext.y[0], 1e-12)
    let w = Math.min(aw, ah * aspect)
    let h = w / Math.max(aspect, 1e-6)
    if (h > ah) ((h = ah), (w = h * aspect))
    return { x0: ml + Math.max(0, Math.floor((aw - w) / 2)), y0: mt, w: Math.floor(w), h: Math.floor(h) }
  }, [size.w, size.h, ext.x[0], ext.x[1], ext.y[0], ext.y[1], p.bar]) // eslint-disable-line react-hooks/exhaustive-deps

  const px = (v: number) => rect.x0 + ((v - ext.x[0]) / (ext.x[1] - ext.x[0])) * rect.w
  const py = (v: number) => rect.y0 + rect.h - ((v - ext.y[0]) / (ext.y[1] - ext.y[0])) * rect.h

  useEffect(() => {
    const ctx = setupCanvas(ref.current, size.w, size.h)
    if (!ctx || rect.w < 16 || rect.h < 16) return
    const { x0, y0, w, h } = rect
    const secondary = cssVar('--secondary') || '#888'
    const grid = cssVar('--grid') || 'rgba(128,128,128,0.25)'
    ctx.font = FONT

    // No-data hatch, under the image.
    if (p.hatch) {
      ctx.save()
      ctx.beginPath()
      ctx.rect(x0, y0, w, h)
      ctx.clip()
      ctx.strokeStyle = p.dark ? 'rgba(161,161,166,0.35)' : 'rgba(110,110,115,0.35)'
      ctx.lineWidth = 0.6
      ctx.beginPath()
      for (let x = -h; x < w; x += 6) {
        ctx.moveTo(x0 + x, y0 + h)
        ctx.lineTo(x0 + x + h, y0)
      }
      ctx.stroke()
      ctx.restore()
    }

    // Image.
    const g = display
    if (g && g.nx > 0 && g.ny > 0) {
      const off = document.createElement('canvas')
      off.width = g.nx
      off.height = g.ny
      let octx: CanvasRenderingContext2D | null = null
      try {
        octx = off.getContext('2d')
      } catch {
        octx = null
      }
      if (octx) {
        const img = octx.createImageData(g.nx, g.ny)
        for (let j = 0; j < g.ny; j++) {
          const row = g.ny - 1 - j
          for (let i = 0; i < g.nx; i++) {
            const v = g.values[j * g.nx + i]
            const c = Number.isFinite(v) ? p.color(v) : null
            if (!c) continue
            const o = 4 * (row * g.nx + i)
            img.data[o] = c[0]
            img.data[o + 1] = c[1]
            img.data[o + 2] = c[2]
            img.data[o + 3] = 255
          }
        }
        octx.putImageData(img, 0, 0)
        ctx.save()
        ctx.beginPath()
        ctx.rect(x0, y0, w, h)
        ctx.clip()
        if (p.cells) {
          ctx.imageSmoothingEnabled = g.nx > 96
          ctx.drawImage(off, x0, y0, w, h)
        } else {
          // Pixels centred on the interpolated nodes: extend by half a pixel.
          const sx = w / Math.max(g.nx - 1, 1), sy = h / Math.max(g.ny - 1, 1)
          ctx.imageSmoothingEnabled = true
          ctx.drawImage(off, x0 - sx / 2, y0 - sy / 2, w + sx, h + sy)
        }
        ctx.restore()
      }
    }

    // Scatter cells.
    if (p.scatter) {
      const s = p.scatter
      const cw = Math.max((s.size / (ext.x[1] - ext.x[0])) * w, 1.5)
      if (s.disk) {
        ctx.strokeStyle = secondary
        ctx.globalAlpha = 0.5
        ctx.beginPath()
        ctx.ellipse(px(0), py(0), w / (ext.x[1] - ext.x[0]), h / (ext.y[1] - ext.y[0]), 0, 0, 2 * Math.PI)
        ctx.stroke()
        ctx.globalAlpha = 1
      }
      s.points.forEach((q, k) => {
        const v = s.values[k]
        const c = Number.isFinite(v) ? p.color(v) : null
        if (!c) return
        ctx.fillStyle = rgbCss(c)
        ctx.fillRect(px(q[0]) - cw / 2, py(q[1]) - cw / 2, cw, cw)
      })
    }

    // Sample nodes.
    if (p.nodes && p.grid && p.grid.nx * p.grid.ny <= 441) {
      const n = p.grid.nx * p.grid.ny
      const r = n > 121 ? 1 : 1.5
      for (let k = 0; k < n; k++) {
        const [x, y] = nodeAt(p.grid, k)
        ctx.fillStyle = p.dark ? '#ffffff' : '#000000'
        ctx.globalAlpha = Number.isFinite(p.grid.values[k]) ? 0.35 : 0.15
        ctx.beginPath()
        ctx.arc(px(x), py(y), r, 0, 2 * Math.PI)
        ctx.fill()
      }
      ctx.globalAlpha = 1
    }

    // Iso-lines and their labels.
    if (p.contours?.length) {
      ctx.save()
      ctx.beginPath()
      ctx.rect(x0, y0, w, h)
      ctx.clip()
      ctx.lineCap = 'round'
      for (const [lw, style] of [
        [2.2, 'rgba(0,0,0,0.45)'],
        [0.9, 'rgba(255,255,255,0.9)'],
      ] as const) {
        ctx.lineWidth = lw
        ctx.strokeStyle = style
        ctx.beginPath()
        for (const c of p.contours)
          for (const [a, b] of c.segments) {
            ctx.moveTo(px(a[0]), py(a[1]))
            ctx.lineTo(px(b[0]), py(b[1]))
          }
        ctx.stroke()
      }
      ctx.restore()
      const placed: { x: number; y: number; w: number; h: number }[] = []
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      for (const c of p.contours) {
        if (!c.segments.length) continue
        const text = fmt(c.level)
        const tw = (ctx.measureText?.(text)?.width ?? text.length * 6) + 4, th = 13
        // Prefer the upper right; never on top of another label.
        const cands = [...c.segments].sort((s, t) => t[0][0] + t[0][1] + t[1][0] + t[1][1] - (s[0][0] + s[0][1] + s[1][0] + s[1][1]))
        for (const s of cands) {
          const mx = px((s[0][0] + s[1][0]) / 2), my = py((s[0][1] + s[1][1]) / 2)
          const b = { x: mx - tw / 2, y: my - th / 2, w: tw, h: th }
          if (b.x < x0 - 2 || b.y < y0 - 2 || b.x + b.w > x0 + w + 2 || b.y + b.h > y0 + h + 2) continue
          if (placed.some((q) => b.x < q.x + q.w + 2 && q.x < b.x + b.w + 2 && b.y < q.y + q.h + 1 && q.y < b.y + b.h + 1)) continue
          placed.push(b)
          ctx.fillStyle = p.dark ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.75)'
          ctx.fillRect(b.x, b.y, b.w, b.h)
          ctx.fillStyle = p.dark ? '#f5f5f7' : '#1d1d1f'
          ctx.fillText(text, mx, my)
          break
        }
      }
    }

    // Axes.
    ctx.fillStyle = secondary
    ctx.strokeStyle = grid
    ctx.lineWidth = 1
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (const t of niceTicks(ext.x[0], ext.x[1], Math.max(2, Math.floor(w / 70)))) {
      const X = Math.round(px(t)) + 0.5
      ctx.beginPath()
      ctx.moveTo(X, y0 + h)
      ctx.lineTo(X, y0 + h + 3)
      ctx.stroke()
      ctx.fillText(sig(t, 3), X, y0 + h + 5)
    }
    ctx.fillText(p.xLabel, x0 + w / 2, y0 + h + 19)
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (const t of niceTicks(ext.y[0], ext.y[1], Math.max(2, Math.floor(h / 60)))) {
      const Y = Math.round(py(t)) + 0.5
      ctx.beginPath()
      ctx.moveTo(x0 - 3, Y)
      ctx.lineTo(x0, Y)
      ctx.stroke()
      ctx.fillText(sig(t, 3), x0 - 5, Y)
    }
    ctx.save()
    ctx.translate(11, y0 + h / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText(p.yLabel, 0, 0)
    ctx.restore()

    // Colour bar.
    const cb = p.bar
    if (cb && cb.range.max > cb.range.min) {
      const bx = x0 + w + 14, bw = 10, bh = Math.min(h, 240), by = y0 + (h - bh) / 2
      const lo = cb.log ? logFloor(cb.range) : cb.range.min
      const pos = (v: number) => (cb.log ? (Math.log10(v) - Math.log10(lo)) / Math.max(Math.log10(cb.range.max) - Math.log10(lo), 1e-12) : (v - lo) / (cb.range.max - lo))
      const steps = Math.max(2, Math.floor(bh))
      for (let i = 0; i < steps; i++) {
        const t = 1 - (i + 0.5) / steps
        const v = cb.log ? Math.pow(10, Math.log10(lo) + t * (Math.log10(cb.range.max) - Math.log10(lo))) : lo + t * (cb.range.max - lo)
        ctx.fillStyle = rgbCss(cb.color(v))
        ctx.fillRect(bx, by + (i * bh) / steps, bw, bh / steps + 1)
      }
      ctx.strokeStyle = grid
      ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1)
      ctx.fillStyle = secondary
      ctx.textAlign = 'left'
      const f = cb.format ?? ((v: number) => sig(v, 3))
      const ticks = cb.log
        ? Array.from({ length: Math.floor(Math.log10(cb.range.max)) - Math.ceil(Math.log10(lo)) + 1 }, (_, k) => Math.pow(10, Math.ceil(Math.log10(lo)) + k)).filter(
            (_, k, a) => a.length <= 6 || k % Math.ceil(a.length / 5) === 0,
          )
        : niceTicks(lo, cb.range.max, 4)
      for (const t of ticks) ctx.fillText(f(t), bx + bw + 4, by + bh - pos(t) * bh)
      if (cb.unit) {
        ctx.save()
        ctx.translate(bx + bw + 58, by + bh / 2)
        ctx.rotate(-Math.PI / 2)
        ctx.textAlign = 'center'
        ctx.fillText(cb.unit, 0, 0)
        ctx.restore()
      }
    }
  }, [p, display, rect, size.w, size.h]) // eslint-disable-line react-hooks/exhaustive-deps

  const onMove = (e: React.PointerEvent) => {
    const g = p.grid
    if (!g || p.cells) return
    const b = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const X = e.clientX - b.left, Y = e.clientY - b.top
    if (X < rect.x0 - 6 || X > rect.x0 + rect.w + 6 || Y < rect.y0 - 6 || Y > rect.y0 + rect.h + 6) return setHover(null)
    let best = -1, bd = Infinity
    for (let k = 0; k < g.nx * g.ny; k++) {
      const [x, y] = nodeAt(g, k)
      const d = Math.hypot(px(x) - X, py(y) - Y)
      if (d < bd) ((bd = d), (best = k))
    }
    setHover(best >= 0 ? { k: best, X, Y } : null)
  }

  let tip: ReactNode = null
  if (hover && p.grid) {
    const [x, y] = nodeAt(p.grid, hover.k)
    const v = p.grid.values[hover.k]
    tip = (
      <div
        className="pointer-events-none absolute z-10 rounded-[5px] border border-separator bg-content px-1.5 py-0.5 text-[11px] whitespace-nowrap shadow-sm"
        style={{ left: Math.min(px(x) + 8, rect.x0 + rect.w - 90), top: Math.max(py(y) - 34, rect.y0) }}
      >
        <div className="num font-semibold">{Number.isFinite(v) ? fmt(v) : p.grid.unitDisk && outsideDisk(x, y) ? 'Outside the field' : 'No value'}</div>
        <div className="num text-secondary">
          Hx {sig(x, 3)} Hy {sig(y, 3)}
        </div>
      </div>
    )
  }

  return (
    <div ref={box} className="relative h-full min-h-0 w-full min-w-0 overflow-hidden" onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
      <canvas ref={ref} role="img" aria-label={p.label} className="absolute inset-0" />
      {tip}
      {p.children}
    </div>
  )
}
