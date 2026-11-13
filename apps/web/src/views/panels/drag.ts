// Moving panels by dragging their title or tab, within a window, between the
// design's windows, and out into a window of its own (`PanelDrag` in the
// macOS app). The drag follows the pointer in screen coordinates, so it works
// across the main window and its popups: each window's groups, tabs and
// content area are found by their data attributes, and their rectangles are
// read when the pointer moves. Over a group, the drop zone is a tab (the
// header, or the middle) or a split (its outer 30 % towards an edge); along a
// window's edge, a whole row or column; outside the design's windows, a new
// window, previewed by an outline of that window where it would open (drawn
// in each of the design's windows, so it shows where it overlaps them; the
// browser cannot draw outside its windows). Escape cancels. Windows are
// judged by their real frames, front (last focused) to back.

import { create } from 'zustand'
import * as A from '@/state/arrangement'
import { useArrangement, type DropTarget, type Edge, type Frame, type PanelKey } from '@/state/arrangement'
import { movePanel } from './actions'
import { clientOrigin, inClientArea, onScreen, refreshFrames, windowIds, windowOf, windowsAt } from './windows'

interface DragState {
  dragging: PanelKey | null
  target: DropTarget | null
  /** The window the pointer is over (null: outside the design's windows). */
  overWindow: string | null
  /** The chip that follows the pointer, in the source window's client coordinates. */
  ghost: { window: string; x: number; y: number } | null
  /** Outside the design's windows: the frame of the window a drop would open (screen coordinates). */
  outline: Frame | null
}

export const useDrag = create<DragState>()(() => ({ dragging: null, target: null, overWindow: null, ghost: null, outline: null }))

const sameTarget = (a: DropTarget | null, b: DropTarget | null) => JSON.stringify(a) === JSON.stringify(b)

const HEADER = 34
const EDGE_BAND = 16

/** Where a drop at a screen point lands. */
export function hitTest(sx: number, sy: number, panel: PanelKey): { target: DropTarget | null; window: string | null } {
  const { arr } = useArrangement.getState()
  const [wid] = windowsAt(sx, sy)
  if (!wid) return { target: A.canMove(arr, panel, { kind: 'newWindow' }) || isOnlyPanelOfSeparateWindow(panel) ? { kind: 'newWindow' } : null, window: null }
  // Over a window's title bar or border: nothing (the front window hides those behind it).
  if (!inClientArea(wid, sx, sy)) return { target: null, window: wid }
  const w = windowOf(wid)!
  const o = clientOrigin(wid)!
  const x = sx - o.x, y = sy - o.y
  const content = w.document.querySelector(`[data-panel-window="${wid}"]`)
  const rect = content?.getBoundingClientRect()
  if (!rect || x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) return { target: null, window: wid }
  // Along the window's edges, when it has more than one group: a whole row or column.
  const edge: Edge | null =
    x - rect.left < EDGE_BAND ? 'leading' : rect.right - x < EDGE_BAND ? 'trailing' : y - rect.top < EDGE_BAND ? 'top' : rect.bottom - y < EDGE_BAND ? 'bottom' : null
  const win = A.windowById(arr, wid)
  if (edge && win && A.groupsOf(win.root).length > 1) {
    const t: DropTarget = { kind: 'windowEdge', window: wid, edge }
    if (A.canMove(arr, panel, t)) return { target: t, window: wid }
  }
  for (const el of w.document.querySelectorAll<HTMLElement>('[data-panel-group]')) {
    const r = el.getBoundingClientRect()
    if (x < r.left || x > r.right || y < r.top || y > r.bottom) continue
    const gid = el.dataset.panelGroup!
    const t = zone(x, y, r, gid, el)
    return { target: A.canMove(arr, panel, t) ? t : null, window: wid }
  }
  return { target: null, window: wid }
}

/** The only panel of a separate window: dropping it outside moves that window. */
function isOnlyPanelOfSeparateWindow(panel: PanelKey): boolean {
  const loc = A.locate(useArrangement.getState().arr, panel)
  return !!loc && loc.window.id !== A.MAIN && A.panelsOf(loc.window.root).length === 1
}

/** The header and the middle add a tab; the outer 30 % towards an edge splits that side. */
function zone(x: number, y: number, r: DOMRect, gid: string, el: HTMLElement): DropTarget {
  if (y - r.top < HEADER) return { kind: 'tab', group: gid, index: tabIndex(x, el) }
  const nx = (x - r.left) / Math.max(r.width, 1), ny = (y - r.top) / Math.max(r.height, 1)
  const edges: [Edge, number][] = [['leading', nx], ['trailing', 1 - nx], ['top', ny], ['bottom', 1 - ny]]
  const nearest = edges.reduce((a, b) => (b[1] < a[1] ? b : a))
  return nearest[1] < 0.3 ? { kind: 'split', group: gid, edge: nearest[0] } : { kind: 'tab', group: gid, index: null }
}

/** Over a tab strip: the position between the tabs under the pointer. */
function tabIndex(x: number, el: HTMLElement): number | null {
  const tabs = [...el.querySelectorAll<HTMLElement>(':scope > [data-tab-strip] [data-panel-tab]')]
  if (tabs.length < 2) return null
  let index = 0
  tabs.forEach((t, k) => {
    const r = t.getBoundingClientRect()
    if (x > r.left + r.width / 2) index = k + 1
  })
  return index
}

/** A new window under the pointer, the size of the group the panel left, its title bar at the pointer. */
export function frameAt(sx: number, sy: number, source: { w: number; h: number }): Frame {
  const w = Math.min(Math.max(source.w, 420), 1100), h = Math.min(Math.max(source.h + 28, 320), 860)
  return onScreen([sx - Math.min(160, w / 2), sy - 14, w, h])
}

/** Starts watching a pointer press on a panel's handle: past a few pixels it
 *  becomes a drag; a press that does not move is left alone (a click). */
export function pressHandle(panel: PanelKey, e: PointerEvent | React.PointerEvent, sourceWindow: string) {
  if (e.button !== 0) return
  const win = windowOf(sourceWindow) ?? window
  const el = e.currentTarget as HTMLElement | null
  const startX = e.screenX, startY = e.screenY
  const pointerId = e.pointerId
  let started = false
  let sourceSize = { w: 560, h: 420 }

  const update = (ev: PointerEvent) => {
    refreshFrames()
    const { target, window: over } = hitTest(ev.screenX, ev.screenY, panel)
    const s = useDrag.getState()
    if (!sameTarget(s.target, target) || s.overWindow !== over) useDrag.setState({ target, overWindow: over })
    const outline = target?.kind === 'newWindow' ? frameAt(ev.screenX, ev.screenY, sourceSize) : null
    useDrag.setState({ ghost: { window: sourceWindow, x: ev.clientX, y: ev.clientY }, outline })
  }
  // Windows move and resize while the pointer is still: keep their frames fresh.
  let poll = 0
  // Escape cancels in whichever of the design's windows has the keyboard.
  let keyWindows: Window[] = []
  const move = (ev: PointerEvent) => {
    if (ev.pointerId !== pointerId) return
    if (!started) {
      if (Math.hypot(ev.screenX - startX, ev.screenY - startY) < 6) return
      started = true
      try {
        el?.setPointerCapture(pointerId)
      } catch {
        // The handle may have gone; window listeners still follow the pointer.
      }
      const loc = A.locate(useArrangement.getState().arr, panel)
      const g = loc ? windowOf(loc.window.id)?.document.querySelector(`[data-panel-group="${loc.group.id}"]`) : null
      const r = g?.getBoundingClientRect()
      if (r) sourceSize = { w: r.width, h: r.height }
      useDrag.setState({ dragging: panel })
      win.document.body.style.cursor = 'grabbing'
      poll = window.setInterval(refreshFrames, 250)
      keyWindows = windowIds().map(windowOf).filter((x): x is Window => !!x && x !== win)
      for (const k of keyWindows) k.addEventListener('keydown', key, true)
    }
    ev.preventDefault()
    update(ev)
  }
  const finish = (drop: boolean, ev?: PointerEvent) => {
    win.removeEventListener('pointermove', move, true)
    win.removeEventListener('pointerup', up, true)
    win.removeEventListener('pointercancel', cancel, true)
    win.removeEventListener('keydown', key, true)
    if (!started) return
    window.clearInterval(poll)
    for (const k of keyWindows) k.removeEventListener('keydown', key, true)
    win.document.body.style.cursor = ''
    if (drop && ev) refreshFrames()
    const target = drop && ev ? hitTest(ev.screenX, ev.screenY, panel).target : null
    useDrag.setState({ dragging: null, target: null, overWindow: null, ghost: null, outline: null })
    if (target) movePanel(panel, target, target.kind === 'newWindow' && ev ? frameAt(ev.screenX, ev.screenY, sourceSize) : null)
  }
  const up = (ev: PointerEvent) => {
    if (ev.pointerId !== pointerId) return
    if (started) {
      ev.preventDefault()
      ev.stopPropagation()
    }
    finish(true, ev)
  }
  const cancel = () => finish(false)
  const key = (ev: KeyboardEvent) => {
    if (ev.key === 'Escape' && started) {
      ev.preventDefault()
      ev.stopPropagation()
      finish(false)
    }
  }
  win.addEventListener('pointermove', move, true)
  win.addEventListener('pointerup', up, true)
  win.addEventListener('pointercancel', cancel, true)
  win.addEventListener('keydown', key, true)
}
