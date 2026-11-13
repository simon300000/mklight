// Drawing the layout's section views on a 2-D canvas, and the colours shared
// with the 3-D view.

import { seriesColor, wavelengthColor } from '@/lib/theme'
import { projector, type Camera, type LayoutMode, type Size } from './camera'
import { KIND, drawn, type Scene } from './scene'

export interface DrawOptions {
  cam: Camera
  mode: LayoutMode
  size: Size
  stretch: number
  dark: boolean
  selection: number
  coloring: 'field' | 'wavelength'
  colors: { label: string; secondary: string; accent: string }
}

/** Violet of diffractive (DOE) surfaces. */
export const doeColor = (dark: boolean) => (dark ? '#b197fc' : '#7048e8')

/** A ray's colour: by field, or by its wavelength's colour of light. */
export function rayColor(s: Scene, p: number, coloring: 'field' | 'wavelength', dark: boolean): string {
  if (coloring === 'wavelength') {
    const um = s.wavelengthsUm[s.wavelengths[p]]
    if (um) return wavelengthColor(um, dark)
  }
  return seriesColor(Math.max(s.fields[p], 0), dark)
}

/** Draws the section (or a wireframe of any view) of `s`. */
export function drawSection(g: CanvasRenderingContext2D, s: Scene, o: DrawOptions) {
  const proj = projector(o.cam, o.mode, o.size, o.stretch)
  const { xyz, offsets, kinds, flags, surfaces } = s
  const pts = (p: number): [number, number][][] => {
    // Runs of finite points (a polyline may break where a ray is lost).
    const runs: [number, number][][] = []
    let run: [number, number][] = []
    for (let i = offsets[p]; i < offsets[p + 1]; i++) {
      if (!Number.isFinite(xyz[3 * i]) || !Number.isFinite(xyz[3 * i + 1]) || !Number.isFinite(xyz[3 * i + 2])) {
        if (run.length) runs.push(run)
        run = []
        continue
      }
      const [x, y] = proj(xyz, 3 * i)
      run.push([x, y])
    }
    if (run.length) runs.push(run)
    return runs
  }
  const path = (runs: [number, number][][]) => {
    g.beginPath()
    for (const r of runs) {
      g.moveTo(r[0][0], r[0][1])
      for (let k = 1; k < r.length; k++) g.lineTo(r[k][0], r[k][1])
    }
  }
  const outlineFill = o.dark ? 'rgba(120,170,255,0.13)' : 'rgba(80,140,230,0.10)'
  // Element bodies, then rays, then profiles and marks on top.
  const order = [KIND.outline, KIND.axis, KIND.ray, KIND.rim, KIND.edge, KIND.profile, KIND.image, KIND.stop]
  for (const k of order) {
    for (let p = 0; p < kinds.length; p++) {
      if (kinds[p] !== k || !drawn(s, p, o.mode)) continue
      if (k === KIND.outline && o.mode === '3d') continue
      const runs = pts(p)
      if (!runs.length) continue
      path(runs)
      g.setLineDash([])
      g.globalAlpha = 1
      switch (k) {
        case KIND.outline:
          g.fillStyle = outlineFill
          g.fill()
          break
        case KIND.axis:
          g.strokeStyle = o.colors.secondary
          g.globalAlpha = 0.5
          g.setLineDash([6, 4])
          g.lineWidth = 0.75
          g.stroke()
          break
        case KIND.ray:
          g.strokeStyle = rayColor(s, p, o.coloring, o.dark)
          g.globalAlpha = flags[p] & 2 ? 0.25 : 0.85
          g.lineWidth = 0.75
          g.stroke()
          break
        case KIND.stop:
          g.strokeStyle = o.colors.accent
          g.lineWidth = 2
          g.stroke()
          break
        default: {
          const sel = surfaces[p] === o.selection && surfaces[p] >= 0
          const doe = (flags[p] & 4) !== 0
          g.strokeStyle = sel ? o.colors.accent : doe ? doeColor(o.dark) : o.colors.label
          g.lineWidth = sel ? 2 : k === KIND.image ? 1.25 : k === KIND.rim ? 0.6 : 1
          if (k === KIND.rim) g.globalAlpha = 0.6
          g.stroke()
          if (doe && k === KIND.profile) hatch(g, runs, sel ? o.colors.accent : doeColor(o.dark))
        }
      }
    }
  }
  g.globalAlpha = 1
  g.setLineDash([])
}

/** A fine hatch along a DOE profile: short ticks on one side every few pixels. */
function hatch(g: CanvasRenderingContext2D, runs: [number, number][][], color: string) {
  const step = 4, len = 5
  g.save()
  g.strokeStyle = color
  g.lineWidth = 0.6
  g.globalAlpha = 0.8
  g.beginPath()
  for (const r of runs) {
    let carry = 0
    for (let k = 1; k < r.length; k++) {
      const [x0, y0] = r[k - 1], [x1, y1] = r[k]
      const dx = x1 - x0, dy = y1 - y0
      const d = Math.hypot(dx, dy)
      if (d < 1e-9) continue
      const ux = dx / d, uy = dy / d
      // 45° to the profile, towards its right-hand side.
      const tx = (ux - uy) * Math.SQRT1_2 * len, ty = (uy + ux) * Math.SQRT1_2 * len
      for (let t = step - carry; t <= d; t += step) {
        const x = x0 + ux * t, y = y0 + uy * t
        g.moveTo(x, y)
        g.lineTo(x + tx, y + ty)
      }
      carry = (carry + d) % step
    }
  }
  g.stroke()
  g.restore()
}

/** The nearest surface to a screen point (profiles and rims), or −1. */
export function pickSurface(s: Scene, o: Pick<DrawOptions, 'cam' | 'mode' | 'size' | 'stretch'>, x: number, y: number, radius = 8): number {
  const proj = projector(o.cam, o.mode, o.size, o.stretch)
  let best = -1, bestD = radius
  for (let p = 0; p < s.kinds.length; p++) {
    const k = s.kinds[p]
    if ((k !== KIND.profile && k !== KIND.rim && k !== KIND.stop) || s.surfaces[p] < 0 || !drawn(s, p, o.mode)) continue
    for (let i = s.offsets[p]; i < s.offsets[p + 1]; i++) {
      const [sx, sy] = proj(s.xyz, 3 * i)
      const d = Math.hypot(sx - x, sy - y)
      if (d < bestD) {
        bestD = d
        best = s.surfaces[p]
      }
    }
  }
  return best
}

const rgbaCache = new Map<string, [number, number, number, number]>()

/** A CSS colour as RGBA in 0…1, for WebGL. */
export function rgba(css: string): [number, number, number, number] {
  const hit = rgbaCache.get(css)
  if (hit) return hit
  let out: [number, number, number, number] = [0.5, 0.5, 0.5, 1]
  try {
    const c = document.createElement('canvas')
    c.width = c.height = 1
    const g = c.getContext('2d', { willReadFrequently: true })
    if (g) {
      g.clearRect(0, 0, 1, 1)
      g.fillStyle = css
      g.fillRect(0, 0, 1, 1)
      const d = g.getImageData(0, 0, 1, 1).data
      out = [d[0] / 255, d[1] / 255, d[2] / 255, d[3] / 255]
    }
  } catch {
    // Keep grey.
  }
  rgbaCache.set(css, out)
  return out
}
