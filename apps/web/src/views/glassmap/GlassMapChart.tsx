// The glass map (`glass_map`, docs/RESULTS.md; apps/macos/Sources/MKLightUI/Views/
// Analysis/GlassMapView.swift): catalogue glasses in the n_d – V_d or P_g,F – V_d
// plane, V_d decreasing to the right as glass maps are drawn. Status by colour
// (obsolete hollow); the marked glasses (the design's, or a candidate's) are
// ringed and labelled with their surfaces; a melt offset glass is dotted from its
// catalogue point. Hover shows a glass's data, a click selects it. Reused by the
// Glass Substitution panel.

import { useEffect, useMemo, useRef, useState } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import { fmt } from '@/model/format'
import {
  deltaPgF,
  designHighlights,
  displayName,
  domain,
  inDesign,
  nominalId,
  normalLine,
  offsetText,
  readGlassMap,
  statusColor,
  statusHollow,
  statusTitle,
  surfaceLabel,
  type GlassHighlight,
  type GlassMapPoint,
  type GlassStatus,
} from '@/model/glass'
import type { System } from '@/model/system'
import { cssVar, useDark } from '@/lib/theme'
import { cn } from '@/lib/utils'
import { FONT, niceTicks, setupCanvas, useSize } from '../results/common'

export type GlassMapPlane = 'nd' | 'pgf'

const M = { left: 48, right: 12, top: 10, bottom: 34 }

/** Text with a subscript ("V" + "d"), drawn at (x, y) with the given alignment. */
function subText(g: CanvasRenderingContext2D, base: string, sub: string, x: number, y: number, align: 'left' | 'center' | 'right' = 'left') {
  g.font = FONT
  const wb = g.measureText(base).width
  g.font = FONT.replace('11px', '8.5px')
  const ws = g.measureText(sub).width
  const w = wb + ws
  const x0 = align === 'left' ? x : align === 'center' ? x - w / 2 : x - w
  g.textAlign = 'left'
  g.font = FONT
  g.fillText(base, x0, y)
  g.font = FONT.replace('11px', '8.5px')
  g.fillText(sub, x0 + wb, y + 3)
  g.font = FONT
}

export function Sub({ base, sub }: { base: string; sub: string }) {
  return (
    <span>
      {base}
      <sub className="text-[0.75em]">{sub}</sub>
    </span>
  )
}

export function GlassMapChart({
  map,
  system,
  highlighted,
  plane = 'nd',
  selection,
  onSelect,
  catalog,
  height = 260,
  className,
}: {
  /** The `glass_map` result (or its JSON). */
  map: AnalysisResult | JsonObject
  system?: System
  /** Glasses to ring with their labels; default: the design's glasses. */
  highlighted?: GlassHighlight[]
  plane?: GlassMapPlane
  selection?: string | null
  onSelect?: (id: string | null) => void
  /** Show one catalogue only (and the design's glasses). */
  catalog?: string | null
  height?: number
  className?: string
}) {
  const all = useMemo(() => readGlassMap(map), [map])
  const [ownSelection, setOwnSelection] = useState<string | null>(null)
  const sel = selection !== undefined ? selection : ownSelection
  const select = onSelect ?? setOwnSelection
  const dark = useDark()
  const [box, size] = useSize<HTMLDivElement>()
  const canvas = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<string | null>(null)

  const yOf = (g: GlassMapPoint) => (plane === 'nd' ? g.nd : g.pgf)
  const shown = useMemo(() => (catalog ? all.filter((g) => g.catalog === catalog || inDesign(g)) : all), [all, catalog])
  const plotted = useMemo(
    () => shown.filter((g) => Number.isFinite(g.vd) && Number.isFinite(plane === 'nd' ? g.nd : g.pgf)),
    [shown, plane],
  )
  const labels = useMemo(() => {
    const m = new Map<string, string>()
    for (const h of highlighted ?? designHighlights(all)) if (!m.has(h.id)) m.set(h.id, h.label)
    return m
  }, [highlighted, all])

  const xd = useMemo(() => domain(plotted.map((g) => g.vd), 2, [20, 90]), [plotted])
  const yd = useMemo(
    () => domain(plotted.map((g) => (plane === 'nd' ? g.nd : g.pgf)), plane === 'nd' ? 0.02 : 0.005, plane === 'nd' ? [1.4, 2] : [0.5, 0.65]),
    [plotted, plane],
  )
  const w = size.w, h = height
  const pw = Math.max(w - M.left - M.right, 1), ph = Math.max(h - M.top - M.bottom, 1)
  // V_d decreases to the right.
  const X = (vd: number) => M.left + ((xd[1] - vd) / (xd[1] - xd[0])) * pw
  const Y = (y: number) => M.top + ph - ((y - yd[0]) / (yd[1] - yd[0])) * ph
  const at = (g: GlassMapPoint) => [X(g.vd), Y(yOf(g))] as const

  useEffect(() => {
    const g = setupCanvas(canvas.current, w, h)
    if (!g) return
    const label = cssVar('--label') || (dark ? '#f5f5f7' : '#1d1d1f')
    const secondary = cssVar('--secondary') || '#888'
    const grid = cssVar('--grid') || 'rgba(0,0,0,0.07)'
    const accent = cssVar('--accent') || '#0a74ff'
    g.font = FONT
    // Grid and ticks.
    g.strokeStyle = grid
    g.fillStyle = secondary
    g.lineWidth = 1
    g.textBaseline = 'top'
    g.textAlign = 'center'
    for (const t of niceTicks(xd[0], xd[1], Math.max(3, Math.round(pw / 70)))) {
      const x = X(t)
      g.beginPath()
      g.moveTo(x, M.top)
      g.lineTo(x, M.top + ph)
      g.stroke()
      g.fillText(String(Math.round(t * 10) / 10), x, M.top + ph + 3)
    }
    g.textAlign = 'right'
    g.textBaseline = 'middle'
    for (const t of niceTicks(yd[0], yd[1], Math.max(3, Math.round(ph / 50)))) {
      const y = Y(t)
      g.beginPath()
      g.moveTo(M.left, y)
      g.lineTo(M.left + pw, y)
      g.stroke()
      g.fillText(fmt(t, plane === 'nd' ? 2 : 3), M.left - 4, y)
    }
    g.strokeStyle = secondary
    g.strokeRect(M.left + 0.5, M.top + 0.5, pw - 1, ph - 1)
    g.textBaseline = 'alphabetic'
    subText(g, 'Abbe number V', 'd', M.left + pw / 2, h - 4, 'center')
    g.save()
    g.translate(11, M.top + ph / 2)
    g.rotate(-Math.PI / 2)
    g.textBaseline = 'middle'
    if (plane === 'nd') subText(g, 'Refractive index n', 'd', 0, 0, 'center')
    else subText(g, 'Partial dispersion P', 'g,F', 0, 0, 'center')
    g.restore()

    g.save()
    g.beginPath()
    g.rect(M.left - 4, M.top - 4, pw + 8, ph + 8)
    g.clip()
    if (plane === 'pgf') {
      g.strokeStyle = secondary
      g.globalAlpha = 0.7
      g.setLineDash([4, 3])
      g.beginPath()
      g.moveTo(X(xd[0]), Y(normalLine(xd[0])))
      g.lineTo(X(xd[1]), Y(normalLine(xd[1])))
      g.stroke()
      g.setLineDash([])
      g.globalAlpha = 1
    }
    // Catalogue glasses first, the marked ones on top.
    for (const p of plotted) {
      if (labels.has(p.id)) continue
      const [x, y] = at(p)
      const c = statusColor(p.status, dark)
      g.beginPath()
      g.arc(x, y, 2.6, 0, 2 * Math.PI)
      if (statusHollow(p.status)) {
        g.strokeStyle = c
        g.lineWidth = 1
        g.stroke()
      } else {
        g.fillStyle = c
        g.globalAlpha = 0.85
        g.fill()
        g.globalAlpha = 1
      }
    }
    const marked = plotted.filter((p) => labels.has(p.id))
    // A melt offset glass: dotted from its catalogue position.
    for (const p of marked) {
      if (p.kind !== 'offset') continue
      const n = all.find((q) => q.id === nominalId(p))
      if (!n || !Number.isFinite(yOf(n))) continue
      const [ax, ay] = at(n), [bx, by] = at(p)
      g.strokeStyle = accent
      g.globalAlpha = 0.8
      g.lineWidth = 1.2
      g.setLineDash([2, 2])
      g.beginPath()
      g.moveTo(ax, ay)
      g.lineTo(bx, by)
      g.stroke()
      g.setLineDash([])
      g.lineWidth = 1
      g.beginPath()
      g.arc(ax, ay, 3.5, 0, 2 * Math.PI)
      g.stroke()
      g.globalAlpha = 1
    }
    const placed: [number, number, number, number][] = []
    for (const p of marked) {
      const [x, y] = at(p)
      g.fillStyle = statusColor(p.status, dark)
      g.beginPath()
      g.arc(x, y, 5, 0, 2 * Math.PI)
      g.fill()
      g.strokeStyle = accent
      g.lineWidth = 2
      g.beginPath()
      g.arc(x, y, 7, 0, 2 * Math.PI)
      g.stroke()
      placed.push([x - 7, y - 7, 14, 14])
    }
    g.restore()
    // Labels, kept apart and inside the plot.
    g.font = '600 ' + FONT.replace('11px', '10px')
    g.textBaseline = 'middle'
    g.textAlign = 'left'
    const inside = (b: number[]) => b[0] >= M.left && b[1] >= M.top && b[0] + b[2] <= M.left + pw && b[1] + b[3] <= M.top + ph
    const hits = (b: number[]) => placed.some((q) => b[0] < q[0] + q[2] && q[0] < b[0] + b[2] && b[1] < q[1] + q[3] && q[1] < b[1] + b[3])
    for (const p of marked) {
      const [x, y] = at(p)
      const l = labels.get(p.id) ?? ''
      const text = l ? `${p.name}  ${l}` : p.name
      const tw = g.measureText(text).width + 6, th = 14
      const cands: [number, number, number, number][] = [
        [x + 9, y - th / 2, tw, th],
        [x - 9 - tw, y - th / 2, tw, th],
        [x - tw / 2, y - 9 - th, tw, th],
        [x - tw / 2, y + 9, tw, th],
      ]
      const b = cands.find((c) => inside(c) && !hits(c)) ?? cands.find(inside) ?? cands[0]
      placed.push(b)
      g.fillStyle = dark ? 'rgba(0,0,0,0.55)' : 'rgba(255,255,255,0.82)'
      g.beginPath()
      g.roundRect(b[0], b[1], b[2], b[3], 3)
      g.fill()
      g.fillStyle = label
      g.fillText(text, b[0] + 3, b[1] + th / 2 + 0.5)
    }
    const s = sel ? plotted.find((p) => p.id === sel) : null
    if (s) {
      const [x, y] = at(s)
      g.strokeStyle = label
      g.lineWidth = 1.5
      g.setLineDash([3, 2])
      g.beginPath()
      g.arc(x, y, labels.has(s.id) ? 10 : 7, 0, 2 * Math.PI)
      g.stroke()
      g.setLineDash([])
    }
  }, [plotted, labels, sel, dark, w, h, plane, xd, yd, all])

  const nearest = (e: React.MouseEvent): string | null => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const px = e.clientX - r.left, py = e.clientY - r.top
    if (px < M.left - 8 || px > M.left + pw + 8 || py < M.top - 8 || py > M.top + ph + 8) return null
    let best: string | null = null
    let bd = Infinity
    for (const p of plotted) {
      const [x, y] = at(p)
      // Design glasses win ties: they are drawn larger.
      const d = (x - px) ** 2 + (y - py) ** 2 - (labels.has(p.id) ? 9 : 0)
      if (d < bd) {
        bd = d
        best = p.id
      }
    }
    return bd <= 100 ? best : null
  }

  const hovered = hover ? plotted.find((p) => p.id === hover) : null
  const marked = [...labels.entries()].map(([id, l]) => {
    const g = all.find((x) => x.id === id)
    return g ? `${g.name} ${l}` : null
  })
  return (
    <div
      ref={box}
      className={cn('relative w-full select-none', className)}
      style={{ height }}
      role="img"
      aria-label={`Glass map, ${plane === 'nd' ? 'n_d' : 'P_g,F'} against Abbe number, ${plotted.length} glasses${system?.name ? ' for ' + system.name : ''}. ${
        marked.filter(Boolean).length ? 'Marked: ' + marked.filter(Boolean).join('; ') : ''
      }`}
      data-testid="glass-map"
      onMouseMove={(e) => setHover(nearest(e))}
      onMouseLeave={() => setHover(null)}
      onClick={(e) => select(nearest(e))}
    >
      <canvas ref={canvas} className="absolute inset-0" />
      {hovered && <Tooltip glass={hovered} x={at(hovered)[0]} y={at(hovered)[1]} w={w} h={h} />}
    </div>
  )
}

function Tooltip({ glass, x, y, w, h }: { glass: GlassMapPoint; x: number; y: number; w: number; h: number }) {
  const tw = 176, th = 92
  const left = x + 12 + tw > w ? x - 12 - tw : x + 12
  const top = Math.min(Math.max(y - th / 2, 0), Math.max(h - th, 0))
  return (
    <div
      className="glass-popover pointer-events-none absolute z-10 rounded-md px-2 py-1"
      style={{ left: Math.max(left, 0), top, width: tw }}
      data-testid="glass-map-tooltip"
    >
      <GlassCard glass={glass} compact />
    </div>
  )
}

/** A glass's data: name, catalogue, status, n_d, V_d, P_g,F (and, full, ΔP_g,F,
 *  its surfaces and whether substitution skips it). */
export function GlassCard({ glass, compact }: { glass: GlassMapPoint; compact?: boolean }) {
  const meta = [
    glass.catalog ?? (glass.kind === 'model' ? 'Model glass' : ''),
    glass.kind === 'offset' ? 'melt offset' : '',
    glass.kind !== 'model' && glass.status !== 'unknown' ? glass.statusLabel : '',
  ].filter(Boolean)
  const rows: [React.ReactNode, string][] = [
    [<Sub base="n" sub="d" />, Number.isFinite(glass.nd) ? glass.nd.toFixed(5) : '–'],
    [<Sub base="V" sub="d" />, Number.isFinite(glass.vd) ? glass.vd.toFixed(2) : '–'],
  ]
  if (glass.offset && offsetText(glass.offset)) rows.push(['Offset', offsetText(glass.offset)])
  rows.push([<Sub base="P" sub="g,F" />, Number.isFinite(glass.pgf) ? glass.pgf.toFixed(4) : '–'])
  if (!compact) {
    const d = deltaPgF(glass)
    rows.push([<Sub base="ΔP" sub="g,F" />, Number.isFinite(d) ? (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(4) : '–'])
    if (glass.status !== 'unknown') rows.push(['Status', glass.statusLabel])
    if (glass.relativeCost != null) rows.push(['Relative cost', String(glass.relativeCost)])
    if (glass.climate != null) rows.push(['Climate class', String(glass.climate)])
    if (glass.stain != null) rows.push(['Stain class', String(glass.stain)])
    if (glass.surfaces.length) rows.push(['Surfaces', surfaceLabel(glass)])
    if (glass.excludeSubstitution) rows.push(['Substitution', 'Excluded'])
  }
  return (
    <div className={cn('min-w-0', compact ? 'text-[11px]' : 'text-[12px]')} data-testid="glass-card">
      <div className={cn('truncate font-semibold', !compact && 'text-[14px]')}>{displayName(glass)}</div>
      {meta.length > 0 && <div className="truncate text-[10px] text-secondary">{meta.join(' · ')}</div>}
      <div className="mt-0.5 grid grid-cols-[auto_1fr] gap-x-3">
        {rows.map(([k, v], i) => (
          <div key={i} className="contents">
            <span className="text-secondary">{k}</span>
            <span className="num text-right">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/** The status legend of a glass map, for the statuses present. */
export function GlassStatusLegend({ statuses, showsOffset }: { statuses: GlassStatus[]; showsOffset?: boolean }) {
  const dark = useDark()
  const list: GlassStatus[] = statuses.length ? statuses : ['unknown']
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-secondary" data-testid="glass-legend">
      {list.map((s) => (
        <span key={s} className="inline-flex items-center gap-1">
          <span
            className="inline-block size-[7px] rounded-full"
            style={statusHollow(s) ? { border: `1px solid ${statusColor(s, dark)}` } : { background: statusColor(s, dark) }}
          />
          {list.length === 1 && s === 'unknown' ? 'Catalog glass' : statusTitle(s)}
        </span>
      ))}
      <span className="inline-flex items-center gap-1">
        <span className="inline-flex size-3 items-center justify-center rounded-full border-2 border-accent">
          <span className="size-[5px] rounded-full bg-secondary" />
        </span>
        In the design
      </span>
      {showsOffset && (
        <span className="inline-flex items-center gap-1" title="A design glass with a melt offset, dotted from its catalog position">
          <span className="inline-block w-3.5 border-t-[1.5px] border-dotted border-accent" />
          Melt offset
        </span>
      )}
    </div>
  )
}
