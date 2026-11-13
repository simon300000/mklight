// A window's panels: its tree of splits and tab groups (`PanelViews` in the
// macOS app). While a panel is dragged, the place it would land is outlined
// in the accent colour.

import { LayoutPanelLeft, PanelsTopLeft, ScanLine, Table2, X } from 'lucide-react'
import { memo, useRef, useState } from 'react'
import * as A from '@/state/arrangement'
import { LAYOUT_PANEL, LENS_DATA, MAIN, useArrangement, type GroupNode, type PanelKey, type PanelNode, type SplitNode } from '@/state/arrangement'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { AnalysisIcon } from '@/views/AnalysisIcon'
import { AnalysisPane } from '@/views/AnalysisPane'
import { LayoutView } from '@/views/LayoutView'
import { LensData } from '@/views/LensData'
import { closePanel, dockAll, focusPanel, isAloneInWindow, isDetached, moveToMainWindow, moveToNewWindow, panelTitle, showPanel } from './actions'
import { PanelTabbedContext, usePanelWindow } from './context'
import { pressHandle, useDrag } from './drag'
import { clientOrigin } from './windows'
import { MenuItem, MenuSeparator, PointMenu } from './Menu'

export function PanelIcon({ panel, className }: { panel: PanelKey; className?: string }) {
  const analysis = useWorkspace((s) => s.panes.find((p) => p.key === panel)?.analysis)
  if (panel === LENS_DATA) return <Table2 className={className} aria-hidden />
  if (panel === LAYOUT_PANEL) return <ScanLine className={className} aria-hidden />
  return <AnalysisIcon name={analysis ?? ''} className={className} />
}

/** Moving and closing a panel, for its context menu and More menu. */
export function PanelMenuItems({ panel, Item = MenuItem, Separator = MenuSeparator }: { panel: PanelKey; Item?: typeof MenuItem; Separator?: typeof MenuSeparator }) {
  const detached = isDetached(panel)
  const alone = isAloneInWindow(panel)
  const title = panelTitle(panel)
  return (
    <>
      {detached && <Item onSelect={() => moveToMainWindow(panel)}>Move to Main Window</Item>}
      {!alone && <Item onSelect={() => moveToNewWindow(panel)}>Move to New Window</Item>}
      {(detached || !alone) && <Separator />}
      {A.isPane(panel) ? <Item onSelect={() => closePanel(panel)}>Close Pane</Item> : <Item onSelect={() => closePanel(panel)}>Hide {title}</Item>}
    </>
  )
}

/** The props that make an element a panel's handle: drag it to move the panel
 *  (to another place, another window, or out into a window of its own); its
 *  context menu moves or closes it. */
export function usePanelHandle(panel: PanelKey) {
  const { id } = usePanelWindow()
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null)
  return {
    props: {
      onPointerDown: (e: React.PointerEvent) => {
        focusPanel(panel)
        pressHandle(panel, e, id)
      },
      onContextMenu: (e: React.MouseEvent) => {
        e.preventDefault()
        setMenu({ x: e.clientX, y: e.clientY })
      },
      title: `${panelTitle(panel)}. Drag to move it, or out of the window to open it in a window of its own.`,
      // A touch drags the panel instead of scrolling.
      style: { touchAction: 'none' } as React.CSSProperties,
    },
    menu: (
      <PointMenu at={menu} onClose={() => setMenu(null)}>
        <PanelMenuItems panel={panel} />
      </PointMenu>
    ),
  }
}

/** The panel's view. Kept apart so a divider drag does not redraw it. */
const PanelBody = memo(function PanelBody({ panel }: { panel: PanelKey }) {
  const pane = useWorkspace((s) => s.panes.find((p) => p.key === panel))
  if (panel === LENS_DATA) return <LensData />
  if (panel === LAYOUT_PANEL) return <LayoutView />
  return pane ? <AnalysisPane pane={pane} /> : null
})

const INTERACTIVE = 'button, input, select, textarea, a, [role="button"], [role="menuitem"], [role="combobox"], [contenteditable="true"]'

/** A panel with its header as the handle: the lens data's and the analysis
 *  panes' header (`PanelHeader`, the first child of the panel's section) away
 *  from its buttons; the layout's title chip is a handle of its own. */
function PanelFrame({ panel }: { panel: PanelKey }) {
  const { id } = usePanelWindow()
  const ref = useRef<HTMLDivElement>(null)
  const dragging = useDrag((s) => s.dragging === panel)
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null)
  const inHeader = (t: EventTarget | null) => {
    if (panel === LAYOUT_PANEL) return false
    const header = ref.current?.firstElementChild?.firstElementChild
    const el = t as Element | null
    return !!header && !!el && header.contains(el) && !el.closest(INTERACTIVE)
  }
  return (
    <div
      ref={ref}
      data-panel={panel}
      className={cn(
        'flex min-h-0 min-w-0 flex-1 flex-col [&>section]:min-h-0 [&>section]:flex-1',
        panel !== LAYOUT_PANEL && '[&>section>div:first-child]:cursor-grab [&>section>div:first-child]:touch-none',
        dragging && 'opacity-45',
      )}
      onPointerDownCapture={() => focusPanel(panel)}
      onPointerDown={(e) => inHeader(e.target) && pressHandle(panel, e, id)}
      onContextMenu={(e) => {
        if (!inHeader(e.target)) return
        e.preventDefault()
        setMenu({ x: e.clientX, y: e.clientY })
      }}
    >
      <PanelBody panel={panel} />
      <PointMenu at={menu} onClose={() => setMenu(null)}>
        <PanelMenuItems panel={panel} />
      </PointMenu>
    </div>
  )
}

function Tab({ panel, group }: { panel: PanelKey; group: GroupNode }) {
  const selected = A.current(group) === panel
  const handle = usePanelHandle(panel)
  const dragging = useDrag((s) => s.dragging === panel)
  const title = panelTitle(panel)
  const select = () => {
    useArrangement.getState().update((a) => A.select(a, panel))
    focusPanel(panel)
  }
  return (
    <div
      data-panel-tab={panel}
      role="tab"
      aria-selected={selected}
      tabIndex={0}
      {...handle.props}
      onClick={select}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && select()}
      className={cn(
        'group/tab flex h-6 min-w-0 max-w-[220px] shrink cursor-default items-center gap-1.5 rounded-full pl-2.5 pr-1 text-[12px]',
        selected ? 'glass-popover font-semibold' : 'text-secondary hover:bg-hover',
        dragging && 'opacity-40',
      )}
    >
      <PanelIcon panel={panel} className={cn('size-3.5 shrink-0', selected ? 'text-accent' : '')} />
      <span className="min-w-0 truncate">{title}</span>
      <button
        type="button"
        aria-label={A.isPane(panel) ? `Close ${title}` : `Hide ${title}`}
        title={A.isPane(panel) ? `Close ${title}` : `Hide ${title}`}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation()
          closePanel(panel)
        }}
        className={cn(
          'flex size-4 shrink-0 cursor-default items-center justify-center rounded-full hover:bg-selected',
          selected ? 'opacity-100' : 'pointer-events-none opacity-0 group-hover/tab:pointer-events-auto group-hover/tab:opacity-100',
        )}
      >
        <X className="size-2.5" strokeWidth={3} />
      </button>
      {handle.menu}
    </div>
  )
}

/** The tabs of a group of several panels: click to switch, drag to move. */
function TabStrip({ group }: { group: GroupNode }) {
  return (
    <div data-tab-strip role="tablist" aria-label="Tabs" className="flex h-[34px] shrink-0 items-center gap-1 overflow-x-auto border-b border-separator bg-content px-2 [scrollbar-width:none]">
      {group.panels.map((p) => (
        <Tab key={p} panel={p} group={group} />
      ))}
    </div>
  )
}

/** Where a dragged panel would land in this group: the whole group (a tab) or the half on one side (a split). */
function GroupHighlight({ gid }: { gid: string }) {
  const target = useDrag((s) => s.target)
  if (!target || (target.kind !== 'tab' && target.kind !== 'split') || target.group !== gid) return null
  let style: React.CSSProperties = { inset: 4 }
  let label = 'Add as Tab'
  if (target.kind === 'split') {
    label = 'Split'
    style =
      target.edge === 'leading'
        ? { left: 4, top: 4, bottom: 4, width: 'calc(50% - 8px)' }
        : target.edge === 'trailing'
          ? { right: 4, top: 4, bottom: 4, width: 'calc(50% - 8px)' }
          : target.edge === 'top'
            ? { left: 4, right: 4, top: 4, height: 'calc(50% - 8px)' }
            : { left: 4, right: 4, bottom: 4, height: 'calc(50% - 8px)' }
  }
  return <DropZone style={style} label={label} />
}

function DropZone({ style, label }: { style: React.CSSProperties; label: string }) {
  return (
    <div
      data-testid="drop-zone"
      className="pointer-events-none absolute z-40 flex items-center justify-center rounded-xl border-2 border-accent/75 bg-accent/15 transition-all duration-150"
      style={style}
    >
      <span className="glass-popover rounded-full px-3 py-1 text-[13px] font-medium">{label}</span>
    </div>
  )
}

/** Along a window's edge: a band a fifth of the window deep. */
function WindowEdgeHighlight({ windowId }: { windowId: string }) {
  const target = useDrag((s) => s.target)
  if (!target || target.kind !== 'windowEdge' || target.window !== windowId) return null
  const e = target.edge
  const style: React.CSSProperties =
    e === 'leading'
      ? { left: 4, top: 4, bottom: 4, width: 'calc(22% - 8px)' }
      : e === 'trailing'
        ? { right: 4, top: 4, bottom: 4, width: 'calc(22% - 8px)' }
        : e === 'top'
          ? { left: 4, right: 4, top: 4, height: 'calc(22% - 8px)' }
          : { left: 4, right: 4, bottom: 4, height: 'calc(22% - 8px)' }
  return <DropZone style={style} label={A.edgeAxis(e) === 'h' ? 'Full-Height Column' : 'Full-Width Row'} />
}

/** What follows the pointer during a drag, in the window the drag started in. */
function DragGhost({ windowId }: { windowId: string }) {
  const d = useDrag()
  if (!d.dragging || !d.ghost || d.ghost.window !== windowId) return null
  const outside = d.target?.kind === 'newWindow'
  return (
    <div
      className={cn(
        'glass-popover pointer-events-none fixed z-[80] flex items-center gap-2 rounded-full px-3 py-1.5 text-[13px] font-medium shadow-lg',
        !d.target && 'opacity-60',
      )}
      style={{ left: d.ghost.x + 14, top: d.ghost.y + 10 }}
    >
      <PanelIcon panel={d.dragging} className="size-4 text-accent" />
      {panelTitle(d.dragging)}
      {outside && <span className="text-[12px] font-normal text-secondary">· New Window</span>}
    </div>
  )
}

/** A narrow main window (under 900 px, a tablet): its panels stack in one
 *  scrolling column, each at least a usable height. Separate windows keep their tree. */
function useStacked(): boolean {
  const { id } = usePanelWindow()
  const narrow = useView((s) => s.narrow)
  return narrow && id === MAIN
}

/** Outside the design's windows: the window a drop would open, outlined where
 *  it would appear (each window draws the part that overlaps it). */
function NewWindowOutline({ windowId }: { windowId: string }) {
  const outline = useDrag((s) => s.outline)
  const dragging = useDrag((s) => s.dragging)
  if (!outline || !dragging) return null
  const o = clientOrigin(windowId)
  if (!o) return null
  return (
    <div
      data-testid="new-window-outline"
      className="pointer-events-none fixed z-[79] rounded-2xl border-2 border-accent/80 bg-accent/10"
      style={{ left: outline[0] - o.x, top: outline[1] - o.y, width: outline[2], height: outline[3] }}
    >
      <span className="glass-popover absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full px-3 py-1 text-[13px] font-medium text-secondary">
        New Window
      </span>
    </div>
  )
}

function GroupView({ group }: { group: GroupNode }) {
  const panel = A.current(group)
  const stacked = useStacked()
  return (
    <div data-panel-group={group.id} className={cn('relative flex min-w-0 flex-1 flex-col overflow-hidden bg-content', stacked ? 'min-h-[320px]' : 'min-h-0')}>
      {group.panels.length > 1 && <TabStrip group={group} />}
      {panel && (
        <PanelTabbedContext.Provider value={group.panels.length > 1}>
          <PanelFrame key={panel} panel={panel} />
        </PanelTabbedContext.Provider>
      )}
      <GroupHighlight gid={group.id} />
    </div>
  )
}

/** A divider between two children: drag to resize, double-click for equal shares. */
function Divider({ split, k, box }: { split: SplitNode; k: number; box: React.RefObject<HTMLDivElement | null> }) {
  const start = useRef<{ p: number; f: number[] } | null>(null)
  const h = split.axis === 'h'
  const min = h ? 200 : 120
  return (
    <div
      role="separator"
      aria-orientation={h ? 'vertical' : 'horizontal'}
      aria-valuenow={Math.round((split.fractions[k] ?? 0) * 100)}
      tabIndex={0}
      className={cn('relative z-10 shrink-0 bg-separator outline-none focus-visible:bg-accent', h ? 'w-px cursor-col-resize' : 'h-px cursor-row-resize')}
      onDoubleClick={() => useArrangement.getState().update((a) => {
        const s = A.findSplit(a, split.id)
        if (s) A.equalize(s)
      })}
      onKeyDown={(e) => {
        const step = e.key === (h ? 'ArrowRight' : 'ArrowDown') ? 0.05 : e.key === (h ? 'ArrowLeft' : 'ArrowUp') ? -0.05 : 0
        if (!step) return
        e.preventDefault()
        useArrangement.getState().update((a) => {
          const s = A.findSplit(a, split.id)
          if (!s) return
          const pair = s.fractions[k] + s.fractions[k + 1]
          const x = Math.min(Math.max(s.fractions[k] + step, 0.05), pair - 0.05)
          A.resizeSplit(s, k, [x, pair - x])
        })
      }}
      onPointerDown={(e) => {
        ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
        start.current = { p: h ? e.clientX : e.clientY, f: [...split.fractions] }
      }}
      onPointerMove={(e) => {
        const st = start.current
        const el = box.current
        if (!st || !el) return
        const total = Math.max((h ? el.clientWidth : el.clientHeight) - (split.children.length - 1), 1)
        const delta = ((h ? e.clientX : e.clientY) - st.p) / total
        const pair = st.f[k] + st.f[k + 1]
        const minF = Math.min(min / total, pair / 2)
        const x = Math.min(Math.max(st.f[k] + delta, minF), pair - minF)
        useArrangement.getState().update((a) => {
          const s = A.findSplit(a, split.id)
          if (s) A.resizeSplit(s, k, [x, pair - x])
        })
      }}
      onPointerUp={() => (start.current = null)}
    >
      <div className={cn('absolute', h ? '-left-[3px] -right-[3px] inset-y-0' : '-top-[3px] -bottom-[3px] inset-x-0')} />
    </div>
  )
}

function SplitView({ split }: { split: SplitNode }) {
  const box = useRef<HTMLDivElement>(null)
  const stacked = useStacked()
  const h = split.axis === 'h' && !stacked
  const axisSplit = stacked ? { ...split, axis: 'v' as const } : split
  return (
    <div ref={box} className={cn('flex min-w-0 flex-1', h ? 'flex-row' : 'flex-col', stacked ? 'min-h-fit' : 'min-h-0')}>
      {split.children.map((c, k) => (
        <div key={c.id} className="contents">
          {k > 0 && <Divider split={axisSplit} k={k - 1} box={box} />}
          <div
            className={cn('flex min-w-0', stacked ? 'min-h-fit' : h ? 'min-h-0 min-w-[120px]' : 'min-h-[80px]')}
            style={{ flex: `${split.fractions[k] ?? 0} 1 0px` }}
          >
            <NodeView node={c} />
          </div>
        </div>
      ))}
    </div>
  )
}

function NodeView({ node }: { node: PanelNode }) {
  return node.kind === 'group' ? <GroupView group={node} /> : <SplitView split={node} />
}

/** The main window with every panel closed or moved away. */
function EmptyPanels() {
  const windows = useArrangement((s) => s.arr.windows.length)
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-content text-center text-secondary">
      <PanelsTopLeft className="size-8 opacity-60" />
      <div className="text-[15px] font-semibold text-label">No Panels</div>
      <div className="max-w-[360px] text-[12px]">Show the lens data or the layout, or add an analysis. Panels moved to other windows can come back here.</div>
      <div className="flex gap-2">
        <Button onClick={() => showPanel(LENS_DATA)}>Show Lens Data</Button>
        <Button onClick={() => showPanel(LAYOUT_PANEL)}>
          <LayoutPanelLeft /> Show Layout
        </Button>
      </div>
      {windows > 0 && (
        <Button variant="prominent" onClick={dockAll}>
          Move All Panels Here
        </Button>
      )}
    </div>
  )
}

/** A window's content: its panel tree, the window-edge drop zones and the drag chip. */
export function PanelWindowView({ windowId }: { windowId: string }) {
  const root = useArrangement((s) => A.windowById(s.arr, windowId)?.root ?? null)
  const stacked = useView((s) => s.narrow) && windowId === MAIN
  const tree = root ? <NodeView node={root} /> : windowId === MAIN ? <EmptyPanels /> : null
  return (
    <div data-panel-window={windowId} className={cn('relative flex min-h-0 min-w-0 flex-1', stacked && 'overflow-y-auto overscroll-contain')}>
      {stacked ? <div className="flex min-h-full w-full flex-col">{tree}</div> : tree}
      <WindowEdgeHighlight windowId={windowId} />
      <NewWindowOutline windowId={windowId} />
      <DragGhost windowId={windowId} />
    </div>
  )
}
