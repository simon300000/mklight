// A square heat-map canvas with axes and an optional colour bar.
import { useEffect, useRef } from 'react'
import { cssVar } from '@/lib/theme'
import { sig } from '@/model/format'
import { FONT, niceTicks, rgbCss, setupCanvas, useSize } from './common'

type RGB = [number, number, number]

export interface HeatMapProps {
  /** Row-major [ny, nx], x fastest, row 0 at low y. */
  data: ArrayLike<number>
  nx: number
  ny: number
  /** Colour of a value; null leaves the cell transparent. */
  color: (v: number) => RGB | null
  /** Extent of the image edges (not cell centres). */
  x: [number, number]
  y: [number, number]
  xLabel: string
  yLabel: string
  colorBar?: { min: number; max: number; color: (v: number) => RGB; label: string }
  label: string
  /** Redraw key (theme). */
  dark: boolean
  smooth?: boolean
  /** Outline of the plot area (off for the pupil). */
  frame?: boolean
}

export function HeatMap(p: HeatMapProps) {
  const [box, size] = useSize<HTMLDivElement>()
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const W = size.w, H = size.h
    const ctx = setupCanvas(ref.current, W, H)
    if (!ctx) return
    const ml = 48, mb = 34, mt = 8, mr = p.colorBar ? 72 : 12
    const side = Math.floor(Math.min(W - ml - mr, H - mt - mb))
    if (side < 20 || p.nx < 1 || p.ny < 1) return
    const x0 = ml + Math.max(0, Math.floor((W - ml - mr - side) / 2))
    const y0 = mt
    const secondary = cssVar('--secondary') || '#888'
    const grid = cssVar('--grid') || 'rgba(128,128,128,0.25)'

    // Image
    const off = document.createElement('canvas')
    off.width = p.nx
    off.height = p.ny
    const octx = off.getContext('2d')
    if (octx) {
      const img = octx.createImageData(p.nx, p.ny)
      for (let iy = 0; iy < p.ny; iy++) {
        const row = p.ny - 1 - iy
        for (let ix = 0; ix < p.nx; ix++) {
          const v = p.data[iy * p.nx + ix]
          const c = Number.isFinite(v) ? p.color(v) : null
          const o = 4 * (row * p.nx + ix)
          if (!c) continue
          img.data[o] = c[0]
          img.data[o + 1] = c[1]
          img.data[o + 2] = c[2]
          img.data[o + 3] = 255
        }
      }
      octx.putImageData(img, 0, 0)
      ctx.imageSmoothingEnabled = p.smooth ?? true
      ctx.drawImage(off, x0, y0, side, side)
    }

    // Axes
    ctx.font = FONT
    ctx.fillStyle = secondary
    ctx.strokeStyle = grid
    ctx.lineWidth = 1
    if (p.frame !== false) ctx.strokeRect(x0 + 0.5, y0 + 0.5, side - 1, side - 1)
    const px = (v: number) => x0 + ((v - p.x[0]) / (p.x[1] - p.x[0])) * side
    const py = (v: number) => y0 + side - ((v - p.y[0]) / (p.y[1] - p.y[0])) * side
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (const t of niceTicks(p.x[0], p.x[1], Math.max(2, Math.floor(side / 70)))) {
      const X = Math.round(px(t)) + 0.5
      ctx.beginPath()
      ctx.moveTo(X, y0 + side)
      ctx.lineTo(X, y0 + side + 3)
      ctx.stroke()
      ctx.fillText(sig(t, 3), X, y0 + side + 5)
    }
    ctx.fillText(p.xLabel, x0 + side / 2, y0 + side + 19)
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (const t of niceTicks(p.y[0], p.y[1], Math.max(2, Math.floor(side / 60)))) {
      const Y = Math.round(py(t)) + 0.5
      ctx.beginPath()
      ctx.moveTo(x0 - 3, Y)
      ctx.lineTo(x0, Y)
      ctx.stroke()
      ctx.fillText(sig(t, 3), x0 - 5, Y)
    }
    ctx.save()
    ctx.translate(11, y0 + side / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText(p.yLabel, 0, 0)
    ctx.restore()

    // Colour bar
    const cb = p.colorBar
    if (cb && cb.max > cb.min) {
      const bx = x0 + side + 14, bw = 10
      const steps = Math.max(2, Math.floor(side))
      for (let i = 0; i < steps; i++) {
        const v = cb.max - ((i + 0.5) / steps) * (cb.max - cb.min)
        ctx.fillStyle = rgbCss(cb.color(v))
        ctx.fillRect(bx, y0 + (i * side) / steps, bw, side / steps + 1)
      }
      ctx.strokeRect(bx + 0.5, y0 + 0.5, bw - 1, side - 1)
      ctx.fillStyle = secondary
      ctx.textAlign = 'left'
      for (const t of niceTicks(cb.min, cb.max, 4)) {
        const Y = y0 + side - ((t - cb.min) / (cb.max - cb.min)) * side
        ctx.fillText(sig(t, 3), bx + bw + 4, Y)
      }
      ctx.save()
      ctx.translate(bx + bw + 52, y0 + side / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillText(cb.label, 0, 0)
      ctx.restore()
    }
  }, [p, size.w, size.h])
  return (
    <div ref={box} className="relative h-full w-full min-w-0 overflow-hidden">
      <canvas ref={ref} role="img" aria-label={p.label} className="absolute inset-0" />
    </div>
  )
}
