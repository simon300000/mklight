import { Maximize2, RefreshCw, ScanLine } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import { Tip } from '@/components/ui/tooltip'
import { nf } from '@/model/format'
import { useView } from '@/state/view'
import { isStale, LAYOUT, useWorkspace } from '@/state/workspace'
import { cssVar, seriesColor, useDark } from '@/lib/theme'

/** Polylines of the engine's `layout` scene (docs/RESULTS.md "layout"). */
interface Scene {
  xyz: Float64Array
  offsets: Uint32Array
  kinds: Uint8Array
  surfaces: Int32Array
  fields: Int32Array
  flags: Uint8Array
  bounds: { min: number[]; max: number[] }
  surfaceInfo: JsonObject[]
}

const KIND = { profile: 0, rim: 1, edge: 2, stop: 3, image: 4, ray: 5, axis: 6, outline: 7 }

export function readScene(r: AnalysisResult): Scene | null {
  const b = r.buffers
  const offsets = b.offsets as Uint32Array | undefined
  const kinds = b.kinds as Uint8Array | undefined
  if (!offsets || !kinds) return null
  let xyz = b.vertices_f64 as Float64Array | undefined
  if (!xyz && b.vertices) {
    const o = (r.json.origin as number[]) ?? [0, 0, 0]
    const v = b.vertices as Float32Array
    xyz = new Float64Array(v.length)
    for (let i = 0; i < v.length; i++) xyz[i] = v[i] + o[i % 3]
  }
  if (!xyz) return null
  const n = kinds.length
  const bounds = r.json.bounds as JsonObject | undefined
  return {
    xyz,
    offsets,
    kinds,
    surfaces: (b.surfaces as Int32Array) ?? new Int32Array(n).fill(-1),
    fields: (b.fields as Int32Array) ?? new Int32Array(n).fill(-1),
    flags: (b.flags as Uint8Array) ?? new Uint8Array(n).fill(1),
    bounds: {
      min: ((bounds?.min as unknown[]) ?? [0, 0, 0]).map(nf),
      max: ((bounds?.max as unknown[]) ?? [0, 0, 0]).map(nf),
    },
    surfaceInfo: (r.json.surfaces as JsonObject[]) ?? [],
  }
}

interface ViewBox {
  cx: number
  cy: number
  scale: number
}

/** A round length for the scale bar. */
function niceLength(target: number) {
  const p = 10 ** Math.floor(Math.log10(target))
  for (const m of [1, 2, 5, 10]) if (m * p >= target) return m * p
  return 10 * p
}

/** The layout pane: the engine's `layout` scene as a 2-D section, with floating
 *  controls (title chip, section selector, fit), a scale bar and a "Design
 *  changed" button when stale. Pan by dragging, zoom with the wheel. */
export function LayoutView() {
  const run = useWorkspace((s) => s.runs[LAYOUT])
  const snapshot = useWorkspace((s) => s.snapshot)
  const settings = useWorkspace((s) => s.layoutSettings)
  const setLayoutSettings = useWorkspace((s) => s.setLayoutSettings)
  const selection = useView((s) => s.selection)
  const select = useView((s) => s.select)
  const dark = useDark()
  const canvas = useRef<HTMLCanvasElement>(null)
  const wrap = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 400, h: 300 })
  const [view, setView] = useState<ViewBox | null>(null)
  const drag = useRef<{ x: number; y: number; v: ViewBox; moved: boolean } | null>(null)

  const result = run?.result
  const scene = useMemo(() => (result ? readScene(result) : null), [result])
  const section = String(settings.section ?? 'yz')
  const axis = section === 'xz' ? 0 : 1
  // The layout always updates by itself; it reads as stale only when that failed or was cancelled.
  const stale = isStale({ snapshot }, run, 'layout', settings) && run?.status !== 'running'

  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const fit = useCallback((): ViewBox | null => {
    if (!scene) return null
    // Frame the elements: rays before the first surface can be long.
    let zmin = Infinity, zmax = -Infinity, hmax = 0
    const { xyz, offsets, kinds, flags } = scene
    for (let p = 0; p < kinds.length; p++) {
      if (kinds[p] === KIND.ray || kinds[p] === KIND.axis || !(flags[p] & 1)) continue
      for (let i = offsets[p]; i < offsets[p + 1]; i++) {
        const z = xyz[3 * i + 2], h = Math.abs(xyz[3 * i + axis])
        if (Number.isFinite(z)) { zmin = Math.min(zmin, z); zmax = Math.max(zmax, z) }
        if (Number.isFinite(h)) hmax = Math.max(hmax, h)
      }
    }
    if (!Number.isFinite(zmin)) {
      zmin = scene.bounds.min[2]; zmax = scene.bounds.max[2]
      hmax = Math.max(Math.abs(scene.bounds.min[axis]), Math.abs(scene.bounds.max[axis]))
    }
    if (size.w < 10 || size.h < 10) return null
    const w = Math.max(zmax - zmin, 1e-6), h = Math.max(2 * hmax, 1e-6)
    const scale = Math.min((size.w * 0.86) / w, (size.h * 0.78) / h)
    return { cx: (zmin + zmax) / 2, cy: 0, scale }
  }, [scene, size, axis])

  // The view frames the lens by itself until it is panned or zoomed; a section
  // change frames it again.
  useEffect(() => setView(null), [section])

  const toScreen = useCallback(
    (v: ViewBox, z: number, h: number): [number, number] => [size.w / 2 + (z - v.cx) * v.scale, size.h / 2 - (h - v.cy) * v.scale],
    [size],
  )

  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const dpr = window.devicePixelRatio || 1
    c.width = Math.round(size.w * dpr)
    c.height = Math.round(size.h * dpr)
    const g = c.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, size.w, size.h)
    const v = view ?? fit()
    if (!scene || !v) return
    const label = cssVar('--label')
    const secondary = cssVar('--secondary')
    const accent = cssVar('--accent')
    const { xyz, offsets, kinds, flags, surfaces, fields } = scene
    const path = (p: number) => {
      g.beginPath()
      let started = false
      for (let i = offsets[p]; i < offsets[p + 1]; i++) {
        const z = xyz[3 * i + 2], h = xyz[3 * i + axis]
        if (!Number.isFinite(z) || !Number.isFinite(h)) { started = false; continue }
        const [x, y] = toScreen(v, z, h)
        if (started) g.lineTo(x, y)
        else { g.moveTo(x, y); started = true }
      }
    }
    // Element bodies, then rays, then profiles and marks on top.
    const order = [KIND.outline, KIND.axis, KIND.ray, KIND.edge, KIND.profile, KIND.image, KIND.stop]
    for (const k of order) {
      for (let p = 0; p < kinds.length; p++) {
        if (kinds[p] !== k || !(flags[p] & 1)) continue
        path(p)
        g.setLineDash([])
        switch (k) {
          case KIND.outline:
            g.fillStyle = dark ? 'rgba(120,170,255,0.13)' : 'rgba(80,140,230,0.10)'
            g.fill()
            break
          case KIND.axis:
            g.strokeStyle = secondary
            g.globalAlpha = 0.5
            g.setLineDash([6, 4])
            g.lineWidth = 0.75
            g.stroke()
            g.globalAlpha = 1
            break
          case KIND.ray:
            g.strokeStyle = seriesColor(Math.max(fields[p], 0), dark)
            g.globalAlpha = flags[p] & 2 ? 0.25 : 0.85
            g.lineWidth = 0.75
            g.stroke()
            g.globalAlpha = 1
            break
          case KIND.stop:
            g.strokeStyle = accent
            g.lineWidth = 2
            g.stroke()
            break
          default: {
            const sel = surfaces[p] === selection
            g.strokeStyle = sel ? accent : label
            g.lineWidth = sel ? 2 : k === KIND.image ? 1.25 : 1
            if (flags[p] & 4) g.setLineDash([2, 2])
            g.stroke()
          }
        }
      }
    }
    // Scale bar.
    const len = niceLength(90 / v.scale)
    const px = len * v.scale
    g.strokeStyle = secondary
    g.lineWidth = 1
    g.beginPath()
    g.moveTo(14, size.h - 16); g.lineTo(14 + px, size.h - 16)
    g.moveTo(14, size.h - 20); g.lineTo(14, size.h - 12)
    g.moveTo(14 + px, size.h - 20); g.lineTo(14 + px, size.h - 12)
    g.stroke()
    g.fillStyle = secondary
    g.font = '11px -apple-system, system-ui, sans-serif'
    g.fillText(`${len} mm`, 14, size.h - 24)
  }, [scene, view, size, dark, selection, axis, fit, toScreen])

  /** A click selects the nearest surface profile. */
  const pick = (x: number, y: number) => {
    const v = view ?? fit()
    if (!scene || !v) return
    let best = -1, bestD = 8
    const { xyz, offsets, kinds, surfaces, flags } = scene
    for (let p = 0; p < kinds.length; p++) {
      if (kinds[p] !== KIND.profile || surfaces[p] < 0 || !(flags[p] & 1)) continue
      for (let i = offsets[p]; i < offsets[p + 1]; i++) {
        const [sx, sy] = toScreen(v, xyz[3 * i + 2], xyz[3 * i + axis])
        const d = Math.hypot(sx - x, sy - y)
        if (d < bestD) { bestD = d; best = surfaces[p] }
      }
    }
    if (best >= 0) select(best)
  }

  return (
    <section className="relative h-full min-h-0 overflow-hidden bg-content" aria-label="Layout" data-testid="layout">
      <div
        ref={wrap}
        className="absolute inset-0"
        onWheel={(e) => {
          const v = view ?? fit()
          if (!v) return
          const r = wrap.current!.getBoundingClientRect()
          const mx = e.clientX - r.left, my = e.clientY - r.top
          const f = Math.exp(-e.deltaY * 0.0015)
          const z = v.cx + (mx - size.w / 2) / v.scale
          const h = v.cy - (my - size.h / 2) / v.scale
          const scale = v.scale * f
          setView({ scale, cx: z - (mx - size.w / 2) / scale, cy: h + (my - size.h / 2) / scale })
        }}
        onPointerDown={(e) => {
          const v = view ?? fit()
          if (!v) return
          ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
          drag.current = { x: e.clientX, y: e.clientY, v, moved: false }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (!d) return
          const dx = e.clientX - d.x, dy = e.clientY - d.y
          if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true
          if (d.moved) setView({ ...d.v, cx: d.v.cx - dx / d.v.scale, cy: d.v.cy + dy / d.v.scale })
        }}
        onPointerUp={(e) => {
          const d = drag.current
          drag.current = null
          if (d && !d.moved) {
            const r = wrap.current!.getBoundingClientRect()
            pick(e.clientX - r.left, e.clientY - r.top)
          }
        }}
        onDoubleClick={() => setView(null)}
      >
        <canvas
          ref={canvas}
          style={{ width: size.w, height: size.h, filter: stale ? 'saturate(0.4) blur(1px)' : undefined, opacity: stale ? 0.55 : 1 }}
          role="img"
          aria-label={`Layout, ${section === 'xz' ? 'X–Z' : 'Y–Z'} section`}
        />
      </div>
      {/* Glass controls float over the scene. */}
      <div className="pointer-events-none absolute inset-x-2 top-2 flex items-start justify-between gap-2">
        <div className="glass-popover pointer-events-auto flex h-7 items-center gap-1.5 rounded-full px-3 text-[12px] font-semibold">
          <ScanLine className="size-3.5 text-secondary" /> Layout
        </div>
        <div className="glass-popover pointer-events-auto flex h-7 items-center gap-1 rounded-full px-1">
          <Segmented
            label="Section"
            value={section as 'yz' | 'xz'}
            options={[
              { value: 'yz', title: 'Y–Z' },
              { value: 'xz', title: 'X–Z' },
            ]}
            onChange={(s) => setLayoutSettings({ section: s })}
            className="bg-transparent"
          />
          <Tip label="Fit Lens">
            <Button variant="ghost" size="iconSm" aria-label="Fit Lens" onClick={() => setView(null)}>
              <Maximize2 />
            </Button>
          </Tip>
        </div>
      </div>
      {stale && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Button variant="prominent" className="rounded-full" onClick={() => useWorkspace.getState().run(LAYOUT)}>
            <RefreshCw /> Design changed · Update
          </Button>
        </div>
      )}
      {run?.status === 'failed' && (
        <div className="absolute inset-x-6 bottom-10 rounded-lg bg-content/90 p-2 text-center text-[12px] text-danger">{run.error}</div>
      )}
      {!scene && run?.status !== 'failed' && (
        <div className="absolute inset-0 flex items-center justify-center text-secondary">{run?.status === 'running' ? 'Drawing…' : 'No layout yet'}</div>
      )}
    </section>
  )
}
