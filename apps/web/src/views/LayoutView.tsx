import { ArrowUpDown, CircleDashed, Ellipsis, Eye, Loader2, Maximize2, RefreshCw, ScanLine } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import { LAYOUT_PANEL } from '@/state/arrangement'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { isStale, LAYOUT, SUMMARY, useWorkspace } from '@/state/workspace'
import { cssVar, useDark } from '@/lib/theme'
import { cn, shortcutLabel } from '@/lib/utils'
import { automaticStretch, extent, fit, lengthLabel, niceLength, orbit, pan, STRETCHES, zoom, type LayoutMode } from './layout/camera'
import { useCamera } from './layout/cameraStore'
import { drawSection, pickSurface } from './layout/draw'
import { useFootprints } from './layout/footprints'
import { Gl3D } from './layout/gl3d'
import { fitPoints, isNonRotational, isSlim, KIND, readScene, type Scene } from './layout/scene'
import { setLayoutMode, syncLayoutSettings } from './layout/shortcuts'
import { usePanelWindow } from './panels/context'
import {
  Menu,
  MenuCheckboxItem,
  MenuContent,
  MenuItem,
  MenuLabel,
  MenuRadioGroup,
  MenuRadioItem,
  MenuSeparator,
  MenuSub,
  MenuSubContent,
  MenuSubTrigger,
  MenuTrigger,
} from './panels/Menu'
import { PanelMenuItems, usePanelHandle } from './panels/PanelArea'
import { usePanelTabbed } from './panels/context'

export { readScene, type Scene }

const MODE_TITLE: Record<LayoutMode, string> = { yz: 'meridional section', xz: 'x–z section', '3d': '3D view' }

function useSize(ref: React.RefObject<HTMLElement | null>) {
  const [size, setSize] = useState({ w: 400, h: 300 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    // The observer of the element's own window: the layout may be in a separate window.
    const RO = (el.ownerDocument.defaultView as (Window & typeof globalThis) | null)?.ResizeObserver ?? ResizeObserver
    const ro = new RO(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}

/** A scale bar of a round length; with a vertical stretch, an upright bar of
 *  a round length as drawn and "vertical ×N". */
function ScaleBar({ ppmm, stretch }: { ppmm: number; stretch: number }) {
  const mm = niceLength(90 / Math.max(ppmm, 1e-9))
  const len = Math.max(mm * ppmm, 1)
  const up = niceLength(60 / Math.max(ppmm * stretch, 1e-9))
  return (
    <div
      className="pointer-events-none absolute bottom-3 left-3 flex items-end gap-2 text-[11px] text-secondary num"
      aria-label={`Scale bar ${lengthLabel(mm)}${stretch !== 1 ? `, vertical stretch ${stretch} times, upright bar ${lengthLabel(up)}` : ''}`}
      data-testid="scale-bar"
    >
      {stretch !== 1 && (
        <div className="flex items-center gap-1 text-accent">
          <div className="w-[1.5px] bg-current" style={{ height: Math.max(up * ppmm * stretch, 1) }} />
          <span>{lengthLabel(up)}</span>
        </div>
      )}
      <div className="flex flex-col gap-0.5">
        <div className="h-[1.5px] bg-current" style={{ width: len }} />
        <div className="flex gap-2">
          <span>{lengthLabel(mm)}</span>
          {stretch !== 1 && <span className="font-semibold text-accent">vertical ×{stretch}</span>}
        </div>
      </div>
    </div>
  )
}

function StretchItems() {
  const stretch = useView((s) => s.stretch)
  return (
    <MenuRadioGroup value={String(stretch)} onValueChange={(v) => chooseStretch(Number(v))}>
      {STRETCHES.map((s) => (
        <MenuRadioItem key={s} value={String(s)}>
          {s === 1 ? 'Off (true proportions)' : `×${s}`}
        </MenuRadioItem>
      ))}
    </MenuRadioGroup>
  )
}

/** The vertical stretch of the section views; a change refits the view. */
function chooseStretch(s: number) {
  useView.getState().set({ stretch: s, stretchChosen: true })
  useView.getState().fit(false)
}

/** The Display menu: ray colours and sampling, fitting, stretch, mesh and footprints. */
function DisplayMenu() {
  const v = useView()
  const settings = useWorkspace((s) => s.layoutSettings)
  const set = useWorkspace((s) => s.setLayoutSettings)
  const rays = Number(settings.rays_per_field ?? 7)
  const wl = String((settings.wavelengths as JsonObject | undefined)?.type ?? 'primary')
  const mesh = Number(settings.mesh_rings ?? 0) > 0
  return (
    <Menu>
      <MenuTrigger asChild>
        <Button variant="ghost" size="iconSm" aria-label="Display" title="Ray colors and sampling">
          <Eye />
        </Button>
      </MenuTrigger>
      <MenuContent align="end">
        <MenuRadioGroup value={v.rayColoring} onValueChange={(c) => v.set({ rayColoring: c as 'field' | 'wavelength' })}>
          <MenuRadioItem value="field">Color by Field</MenuRadioItem>
          <MenuRadioItem value="wavelength">Color by Wavelength</MenuRadioItem>
        </MenuRadioGroup>
        <MenuSeparator />
        <MenuSub>
          <MenuSubTrigger>Rays per Field</MenuSubTrigger>
          <MenuSubContent>
            <MenuRadioGroup value={String(rays)} onValueChange={(n) => set({ rays_per_field: Number(n) })}>
              {[3, 5, 7, 11, 15].map((n) => (
                <MenuRadioItem key={n} value={String(n)}>
                  {n}
                </MenuRadioItem>
              ))}
            </MenuRadioGroup>
          </MenuSubContent>
        </MenuSub>
        <MenuSub>
          <MenuSubTrigger>Wavelengths</MenuSubTrigger>
          <MenuSubContent>
            <MenuRadioGroup value={wl} onValueChange={(t) => set({ wavelengths: { type: t } })}>
              <MenuRadioItem value="primary">Primary</MenuRadioItem>
              <MenuRadioItem value="all">All</MenuRadioItem>
            </MenuRadioGroup>
          </MenuSubContent>
        </MenuSub>
        <MenuSeparator />
        <MenuItem onSelect={() => v.fit(false)}>Fit All</MenuItem>
        <MenuItem onSelect={() => v.fit(true)}>Fit Lens</MenuItem>
        <MenuSeparator />
        <MenuSub>
          <MenuSubTrigger disabled={v.layoutMode === '3d'}>Vertical Stretch</MenuSubTrigger>
          <MenuSubContent>
            <StretchItems />
          </MenuSubContent>
        </MenuSub>
        <MenuSeparator />
        <MenuCheckboxItem checked={mesh} onCheckedChange={(on) => set({ mesh_rings: on ? 16 : 0 })}>
          Surface Mesh in 3D
        </MenuCheckboxItem>
        <MenuCheckboxItem checked={v.showFootprints} onCheckedChange={(on) => v.set({ showFootprints: on === true })}>
          Beam Footprints in 3D
        </MenuCheckboxItem>
      </MenuContent>
    </Menu>
  )
}

/** The layout's floating controls: the title chip (the panel's handle), the
 *  view selector, display options, stretch, footprints, fit and More. */
function LayoutControls({ running, xz, footprints }: { running: boolean; xz: boolean; footprints: { running: boolean; error: string | null } }) {
  const mode = useView((s) => s.layoutMode)
  const stretch = useView((s) => s.stretch)
  const showFootprints = useView((s) => s.showFootprints)
  const tabbed = usePanelTabbed()
  const handle = usePanelHandle(LAYOUT_PANEL)
  const modes: { value: LayoutMode; title: string }[] = xz || mode === 'xz'
    ? [
        { value: 'yz', title: 'Y–Z' },
        { value: 'xz', title: 'X–Z' },
        { value: '3d', title: '3D' },
      ]
    : [
        { value: 'yz', title: 'Section' },
        { value: '3d', title: '3D' },
      ]
  return (
    <div className="pointer-events-none absolute inset-x-2 top-2 flex items-start justify-between gap-2">
      {!tabbed ? (
        <div
          {...handle.props}
          className="glass-popover pointer-events-auto flex h-7 min-w-0 cursor-grab items-center gap-1.5 rounded-full px-3 text-[12px] font-semibold @max-[440px]:hidden"
          data-testid="layout-title"
        >
          <ScanLine className="size-3.5 shrink-0 text-secondary" /> Layout
          {running && <Loader2 className="size-3 animate-spin text-secondary" aria-label="Updating" />}
          {handle.menu}
        </div>
      ) : (
        <span />
      )}
      <div className="flex items-center gap-1.5">
        <div className="glass-popover pointer-events-auto flex h-7 items-center rounded-full px-1">
          <Segmented
            label="View"
            value={mode}
            options={modes.map((m) => ({
              value: m.value,
              title: m.title,
              help: xz
                ? `Y–Z section, X–Z section or 3D view (${shortcutLabel('alt+mod+y')}, ${shortcutLabel('alt+mod+x')}, ${shortcutLabel('alt+mod+3')})`
                : `Meridional section or 3D view (${shortcutLabel('alt+mod+y')}, ${shortcutLabel('alt+mod+3')})`,
            }))}
            onChange={setLayoutMode}
            className="bg-transparent"
          />
        </div>
        {mode !== '3d' && stretch !== 1 && (
          <Menu>
            <MenuTrigger asChild>
              <button
                type="button"
                className="glass-popover pointer-events-auto flex h-7 cursor-default items-center gap-1 whitespace-nowrap rounded-full px-2.5 text-[11px] font-semibold text-accent"
                title={`The section is drawn ${stretch} times taller than true proportions. Click to change or turn off.`}
                aria-label="Vertical stretch"
                data-testid="stretch-menu"
              >
                <ArrowUpDown className="size-3" /> Stretch ×{stretch}
              </button>
            </MenuTrigger>
            <MenuContent align="end">
              <MenuLabel>Vertical Stretch</MenuLabel>
              <StretchItems />
            </MenuContent>
          </Menu>
        )}
        <div className="glass-popover pointer-events-auto flex h-7 items-center gap-0.5 rounded-full px-0.5">
          <DisplayMenu />
          {mode === '3d' && (
            <Button
              variant="ghost"
              size="iconSm"
              aria-label="Beam footprints"
              aria-pressed={showFootprints}
              title={footprints.error ?? (showFootprints ? 'Hide the beam footprints on the surfaces' : 'Show the beam footprint of every field on every surface')}
              className={cn(showFootprints && 'text-accent')}
              onClick={() => useView.getState().set({ showFootprints: !showFootprints })}
            >
              {showFootprints && footprints.running ? <Loader2 className="animate-spin" /> : <CircleDashed />}
            </Button>
          )}
          <Button variant="ghost" size="iconSm" aria-label="Fit Lens" title={`Fit the layout (${shortcutLabel('alt+mod+0')}, or double-click)`} onClick={() => useView.getState().fit(false)}>
            <Maximize2 />
          </Button>
          <Menu>
            <MenuTrigger asChild>
              <Button variant="ghost" size="iconSm" aria-label="More">
                <Ellipsis />
              </Button>
            </MenuTrigger>
            <MenuContent align="end">
              <MenuLabel>Layout</MenuLabel>
              <PanelMenuItems panel={LAYOUT_PANEL} />
            </MenuContent>
          </Menu>
        </div>
      </div>
    </div>
  )
}

/** The layout pane: the engine's `layout` scene edge to edge, as a Y–Z or
 *  X–Z section (canvas) or in 3-D (WebGL2), with glass controls floating over
 *  it, a scale bar, and a "Design changed" button when stale. Drag to pan (to
 *  orbit in 3-D; ⇧-drag pans), scroll to zoom, double-click to fit, click a
 *  surface to select it. */
export function LayoutView() {
  const run = useWorkspace((s) => s.runs[LAYOUT])
  const snapshot = useWorkspace((s) => s.snapshot)
  const preview = useWorkspace((s) => s.preview)
  const settings = useWorkspace((s) => s.layoutSettings)
  const summary = useWorkspace((s) => s.runs[SUMMARY]?.result?.json as JsonObject | undefined)
  const surfacesModel = useDocument((s) => s.project.system?.surfaces)
  const v = useView()
  const dark = useDark()
  const { win } = usePanelWindow()
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const glCanvas = useRef<HTMLCanvasElement>(null)
  const gl = useRef<Gl3D | null>(null)
  const [glError, setGlError] = useState<string | null>(null)
  const size = useSize(wrap)
  const cam = useCamera((s) => s.cam)
  const drag = useRef<{ x: number; y: number; moved: boolean; pan: boolean } | null>(null)

  const mode = v.layoutMode
  const stretch = mode === '3d' ? 1 : v.stretch
  const xz = isNonRotational(summary, surfacesModel)
  const result = run?.result
  const scene = useMemo(() => (result ? readScene(result) : null), [result])
  const footprints = useFootprints(v.showFootprints && mode === '3d')
  // The layout always updates by itself; it reads as stale only when that failed or was cancelled.
  const stale = isStale({ snapshot, preview }, run, 'layout', settings) && run?.status !== 'running'
  // The scene shows the view's section (a mode change re-runs the layout).
  const sceneMatches = !!scene && (mode === '3d' || scene.section === mode)

  // The engine draws the section the view shows; a design that lost its X–Z section goes back to Y–Z.
  useEffect(() => syncLayoutSettings(), [mode])
  useEffect(() => {
    if (mode === 'xz' && !xz && summary) setLayoutMode('yz')
  }, [mode, xz, summary])

  /** Frames the scene (`lensOnly`: the lens). The first fit after a document
   *  opens frames a slim, long design on its lens and may turn on a stretch. */
  const doFit = useCallback(
    (lensOnly: boolean) => {
      if (!scene || size.w < 10 || size.h < 10) return
      const view = useView.getState()
      const store = useCamera.getState()
      const initial = store.epoch !== view.layoutEpoch
      const lens = lensOnly || (initial && isSlim(scene, mode))
      const pts = fitPoints(scene, mode, lens)
      let s = mode === '3d' ? 1 : view.stretch
      let c = fit(store.cam, mode, pts, size, s)
      if (initial && mode !== '3d' && !view.stretchChosen && view.stretch === 1) {
        const auto = automaticStretch(extent(c, mode, pts).h * c.ppmm, size.h)
        if (auto !== 1) {
          s = auto
          view.set({ stretch: auto })
          c = fit(c, mode, pts, size, s)
        }
      }
      useCamera.setState({ cam: c, fitted: true, lens, epoch: view.layoutEpoch })
    },
    [scene, size, mode],
  )

  // Fit on request, on a mode change, and while fitted, when the scene, the size or the stretch changes.
  const last = useRef({ request: v.fitRequest, mode })
  useEffect(() => {
    if (!sceneMatches) return
    const l = last.current
    if (l.request !== v.fitRequest) doFit(v.fitLens)
    else if (l.mode !== mode || useCamera.getState().fitted) doFit(useCamera.getState().lens)
    last.current = { request: v.fitRequest, mode }
  }, [doFit, v.fitRequest, v.fitLens, mode, v.stretch, sceneMatches])

  const colors = useMemo(
    () => ({ label: cssVar('--label'), secondary: cssVar('--secondary'), accent: cssVar('--accent') }),
    // Theme colours change with the appearance.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dark],
  )

  // Sections on a 2-D canvas.
  useEffect(() => {
    const c = canvas.current
    if (!c || mode === '3d') return
    const dpr = win.devicePixelRatio || 1
    c.width = Math.round(size.w * dpr)
    c.height = Math.round(size.h * dpr)
    const g = c.getContext('2d')
    if (!g) return
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, size.w, size.h)
    if (!scene) return
    drawSection(g, scene, { cam, mode, size, stretch, dark, selection: v.selection, coloring: v.rayColoring, colors })
  }, [scene, cam, size, mode, stretch, dark, v.selection, v.rayColoring, colors, win])

  // 3-D in WebGL2: buffers when the scene or its colours change, a draw when the camera moves.
  useEffect(() => {
    if (mode !== '3d' || !glCanvas.current) return
    if (!gl.current) {
      try {
        gl.current = new Gl3D(glCanvas.current)
        setGlError(null)
      } catch (e) {
        setGlError((e as Error).message)
        return
      }
    }
  }, [mode])
  useEffect(() => () => {
    gl.current?.dispose()
    gl.current = null
  }, [])
  useEffect(() => {
    if (mode !== '3d' || !gl.current || !scene) return
    gl.current.setScene(scene, { dark, selection: v.selection, coloring: v.rayColoring, colors, footprints: footprints.footprints })
  }, [mode, scene, dark, v.selection, v.rayColoring, colors, footprints.footprints, glError])
  useEffect(() => {
    if (mode !== '3d' || !gl.current) return
    gl.current.draw(cam, size, win.devicePixelRatio || 1)
  }, [mode, scene, cam, size, dark, v.selection, v.rayColoring, footprints.footprints, win, glError])

  const local = (e: { clientX: number; clientY: number }) => {
    const r = wrap.current!.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  const rays = scene ? scene.kinds.reduce((n, k) => n + (k === KIND.ray ? 1 : 0), 0) : 0
  const length = scene ? scene.bounds.max[2] - scene.bounds.min[2] : 0
  const label = scene ? `Layout, ${MODE_TITLE[mode]}, ${Math.max((surfacesModel?.length ?? 2) - 2, 0)} surfaces, ${rays} rays, ${lengthLabel(length)} long` : 'Layout, not available'

  return (
    <section className="@container relative h-full min-h-0 overflow-hidden bg-content" aria-label="Layout" data-testid="layout">
      <div
        ref={wrap}
        className="absolute inset-0 touch-none"
        onWheel={(e) => {
          if (!scene) return
          const p = local(e)
          const f = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015))
          useCamera.setState({ cam: zoom(useCamera.getState().cam, mode, f, p.x - size.w / 2, p.y - size.h / 2, stretch), fitted: false })
        }}
        onPointerDown={(e) => {
          if (e.button !== 0 && e.button !== 1 && e.button !== 2) return
          ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
          drag.current = { x: e.clientX, y: e.clientY, moved: false, pan: mode !== '3d' || e.shiftKey || e.altKey || e.button !== 0 }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (!d) return
          const dx = e.clientX - d.x, dy = e.clientY - d.y
          if (!d.moved && Math.abs(dx) + Math.abs(dy) <= 3) return
          d.moved = true
          d.x = e.clientX
          d.y = e.clientY
          const c = useCamera.getState().cam
          if (d.pan) useCamera.setState({ cam: pan(c, mode, dx, dy, stretch), fitted: false })
          else useCamera.setState({ cam: orbit(c, dx, dy) })
        }}
        onPointerUp={(e) => {
          const d = drag.current
          drag.current = null
          if (d && !d.moved && e.button === 0 && scene) {
            const p = local(e)
            const s = pickSurface(scene, { cam, mode, size, stretch }, p.x, p.y)
            if (s >= 0) v.select(s)
          }
        }}
        onContextMenu={(e) => e.preventDefault()}
        onDoubleClick={() => v.fit(false)}
      >
        <canvas
          ref={canvas}
          hidden={mode === '3d'}
          style={{ width: size.w, height: size.h, filter: stale ? 'saturate(0.4) blur(1px)' : undefined, opacity: stale ? 0.55 : 1 }}
          role="img"
          aria-label={label}
        />
        <canvas
          ref={glCanvas}
          hidden={mode !== '3d'}
          style={{ width: size.w, height: size.h, filter: stale ? 'saturate(0.4) blur(1px)' : undefined, opacity: stale ? 0.55 : 1 }}
          role="img"
          aria-label={label}
          data-testid="layout-3d"
        />
      </div>
      <LayoutControls running={run?.status === 'running'} xz={xz} footprints={footprints} />
      {preview && (
        <div className="pointer-events-none absolute inset-x-0 top-11 flex justify-center">
          <div className="glass-popover flex items-center gap-1.5 rounded-full bg-accent-soft px-3 py-1 text-[12px] font-medium" data-testid="preview-label">
            <Eye className="size-3.5 text-accent" /> Previewing {preview.label} · <span className="text-secondary">not applied</span>
          </div>
        </div>
      )}
      {scene && <ScaleBar ppmm={cam.ppmm} stretch={stretch} />}
      {stale && scene && (
        <div className="absolute bottom-3 right-3">
          <Button className="glass-popover rounded-full" size="sm" title="The layout is out of date. Click to update it." onClick={() => useWorkspace.getState().run(LAYOUT)}>
            <RefreshCw /> Design changed
          </Button>
        </div>
      )}
      {stale && !scene && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Button variant="prominent" className="rounded-full" onClick={() => useWorkspace.getState().run(LAYOUT)}>
            <RefreshCw /> Design changed · Update
          </Button>
        </div>
      )}
      {mode === '3d' && glError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-content text-center text-secondary">
          <span className="font-semibold text-label">3D View Unavailable</span>
          <span className="text-[12px]">{glError}</span>
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
