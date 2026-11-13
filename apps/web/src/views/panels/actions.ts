// What the panel commands do in any of the design's windows (the macOS app's
// `DesignSession`): show, hide, close and move panels, open and dock separate
// windows, reset the arrangement; and keeping the arrangement in step with
// the workspace's panes and the view's Lens Data / Layout flags.

import * as A from '@/state/arrangement'
import { LAYOUT_PANEL, LENS_DATA, MAIN, useArrangement, type DropTarget, type Frame, type PanelKey } from '@/state/arrangement'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { descriptorFor, useWorkspace } from '@/state/workspace'
import { analysisTitle } from '@/model/catalog'
import { closePopup, focusWindow, openPopup, setPopupHandlers, usePopups, windowOf } from './windows'

const arrangement = () => useArrangement.getState()
const ws = () => useWorkspace.getState()

export function panelTitle(panel: PanelKey): string {
  if (panel === LENS_DATA) return 'Lens Data'
  if (panel === LAYOUT_PANEL) return 'Layout'
  const p = ws().panes.find((x) => x.key === panel)
  return p ? analysisTitle(p.analysis, descriptorFor(ws().descriptors, p.analysis)) : 'Analysis'
}

export function windowTitle(windowId: string): string {
  const w = A.windowById(arrangement().arr, windowId)
  const name = useDocument.getState().project.system?.name || 'Untitled'
  const panels = A.panelsOf(w?.root ?? null).map(panelTitle)
  return `${panels.join(', ') || 'Panels'} — ${name}`
}

export const isDetached = (panel: PanelKey) => {
  const loc = A.locate(arrangement().arr, panel)
  return !!loc && loc.window.id !== MAIN
}

/** Alone in a separate window: moving it to a new window would change nothing. */
export const isAloneInWindow = (panel: PanelKey) => !A.canMove(arrangement().arr, panel, { kind: 'newWindow' })

export function focusPanel(panel: PanelKey) {
  arrangement().setFocused(panel)
  if (A.isPane(panel) && useView.getState().focusedPane !== panel) useView.getState().set({ focusedPane: panel })
}

/** Opens the popup of each separate window that has none (after a user
 *  action, or when a project opens). Windows the browser blocked: their
 *  panels go back into the main window, and a toast offers to try again. */
export function syncPopups() {
  const { arr } = arrangement()
  const open = usePopups.getState().open
  for (const id of Object.keys(open)) if (!arr.windows.some((w) => w.id === id)) closePopup(id)
  const blocked: A.PanelWindow[] = []
  for (const w of arr.windows) {
    if (open[w.id]) continue
    if (!openPopup(w.id, w.frame, windowTitle(w.id))) blocked.push(w)
  }
  if (!blocked.length) return
  const saved = blocked.map((w) => ({ panels: A.panelsOf(w.root), root: structuredClone(w.root), frame: w.frame }))
  arrangement().update((a) => {
    for (const w of blocked) A.dock(a, w.id)
  })
  const n = blocked.length
  useView.getState().toast(
    `The browser blocked ${n === 1 ? 'a separate window' : `${n} separate windows`}; ${n === 1 ? 'its panels are' : 'their panels are'} in this window.`,
    'warning',
    {
      label: n === 1 ? 'Open Window' : 'Open Windows',
      run: () => {
        arrangement().update((a) => {
          for (const s of saved) {
            if (!s.panels.every((p) => A.contains(a, p))) continue
            for (const p of s.panels) A.remove(a, p)
            a.windows.push({ id: A.newId('w'), root: s.root, frame: s.frame })
          }
        })
        syncPopups()
      },
    },
  )
}

/** A frame for Move to New Window: beside the panel's place now, a little offset. */
function defaultFrame(panel: PanelKey): Frame {
  const loc = A.locate(arrangement().arr, panel)
  const w = loc ? windowOf(loc.window.id) : window
  const el = loc ? w?.document.querySelector(`[data-panel-group="${loc.group.id}"]`) : null
  const r = el?.getBoundingClientRect()
  const base = w ?? window
  if (r) {
    const fw = Math.min(Math.max(r.width, 420), 1100), fh = Math.min(Math.max(r.height + 28, 320), 860)
    return [base.screenX + r.left + 40, base.screenY + (base.outerHeight - base.innerHeight) + r.top + 40, fw, fh]
  }
  return [window.screenX + window.outerWidth / 2 - 300, window.screenY + window.outerHeight / 2 - 220, 600, 440]
}

/** Moves a panel (a drop, or a menu command). A new window opens at `frame`;
 *  the only panel of a separate window moves that window instead. */
export function movePanel(panel: PanelKey, target: DropTarget, frame: Frame | null = null) {
  const { arr } = arrangement()
  const loc = A.locate(arr, panel)
  if (target.kind === 'newWindow' && loc && loc.window.id !== MAIN && A.panelsOf(loc.window.root).length === 1) {
    const w = windowOf(loc.window.id)
    if (w && frame) {
      try {
        w.moveTo(frame[0], frame[1])
      } catch {
        // Ignored: the window stays where it is.
      }
    }
    return
  }
  if (!A.canMove(arr, panel, target)) return
  if (target.kind === 'newWindow') {
    // Open the popup first (still inside the user's action); a blocked one keeps the panel where it is.
    const next = A.cloneArrangement(arr)
    const id = A.move(next, panel, target, frame ?? defaultFrame(panel))
    if (!id) return
    const w = next.windows.find((x) => x.id === id)!
    if (!openPopup(id, w.frame, `${panelTitle(panel)} — ${useDocument.getState().project.system?.name || 'Untitled'}`)) {
      useView.getState().toast('The browser blocked the new window. Allow pop-ups for MKLight to open panels in windows of their own.', 'warning')
      return
    }
    A.select(next, panel)
    arrangement().replace(next)
  } else {
    arrangement().update((a) => {
      A.move(a, panel, target, frame)
      A.select(a, panel)
    })
  }
  focusPanel(panel)
  syncPopups()
  const now = A.locate(arrangement().arr, panel)
  if (now) focusWindow(now.window.id)
}

export const moveToNewWindow = (panel: PanelKey) => movePanel(panel, { kind: 'newWindow' })

export function moveToMainWindow(panel: PanelKey) {
  if (!isDetached(panel)) return
  arrangement().update((a) => A.returnToMain(a, panel))
  focusPanel(panel)
  syncPopups()
  window.focus()
}

/** Moves a separate window's panels back into the main window. */
export function dockWindow(id: string) {
  arrangement().update((a) => A.dock(a, id))
  syncPopups()
  window.focus()
}

export function dockAll() {
  arrangement().update((a) => {
    for (const w of [...a.windows]) A.dock(a, w.id)
  })
  syncPopups()
}

/** Closes a panel: an analysis pane is removed, the lens data and the layout
 *  are hidden until shown again. */
export function closePanel(panel: PanelKey) {
  if (A.isPane(panel) && ws().panes.some((p) => p.key === panel)) ws().removePane(panel)
  arrangement().update((a) => A.remove(a, panel))
  if (arrangement().focused === panel) arrangement().setFocused(null)
  syncPopups()
}

/** Shows the lens data or the layout (opening it where it belongs if closed),
 *  brings its tab to the front and its window forward, and focuses it. */
export function showPanel(panel: PanelKey) {
  arrangement().update((a) => {
    if (!A.contains(a, panel)) A.show(a, panel)
    A.select(a, panel)
  })
  focusPanel(panel)
  const loc = A.locate(arrangement().arr, panel)
  if (loc && loc.window.id !== MAIN) focusWindow(loc.window.id)
}

export function togglePanel(panel: PanelKey) {
  if (A.contains(arrangement().arr, panel)) closePanel(panel)
  else showPanel(panel)
}

/** Every panel back in the main window, in the standard arrangement. */
export function resetPanelLayout() {
  arrangement().replace(A.standard(ws().panes.map((p) => p.key)))
  syncPopups()
  window.focus()
}

/** The panel the panel commands act on: the focused one, else the front one of the main window. */
export function commandPanel(): PanelKey | null {
  const { arr, focused } = arrangement()
  if (focused && A.contains(arr, focused)) return focused
  const g = A.groupsOf(arr.main.root)[0]
  return g ? A.current(g) : null
}

/** New panes to place in a given separate window, by pane key. */
const pendingWindow = new Map<string, string>()

/** Adds an analysis pane to a given window (a separate window's Add Analysis). */
export function addPaneInWindow(analysis: string, windowId: string) {
  const key = `pane-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
  pendingWindow.set(key, windowId)
  ws().addPane(analysis, { key })
  focusPanel(key)
}

let installed = false

/** Keeps the arrangement, the panes and the view's flags in step. */
export function installPanelSync() {
  if (installed) return
  installed = true
  useWorkspace.subscribe((s, prev) => {
    if (s.panes === prev.panes) return
    const keys = s.panes.map((p) => p.key)
    const { arr } = arrangement()
    const inTree = A.allPanels(arr).filter(A.isPane)
    const same = inTree.length === keys.length && keys.every((k) => inTree.includes(k))
    if (same) return
    arrangement().update((a) => A.reconcile(a, keys, pendingWindow))
    syncPopups()
  })
  const flags = () => {
    const { arr } = arrangement()
    const lens = A.contains(arr, LENS_DATA), layout = A.contains(arr, LAYOUT_PANEL)
    const v = useView.getState()
    if (v.lensDataVisible !== lens || v.layoutVisible !== layout) v.set({ lensDataVisible: lens, layoutVisible: layout })
  }
  useArrangement.subscribe((s, prev) => {
    if (s.arr !== prev.arr) flags()
  })
  // Setting a flag (a menu command, the sidebar) shows or hides the panel.
  useView.subscribe((s, prev) => {
    const { arr } = arrangement()
    if (s.lensDataVisible !== prev.lensDataVisible && s.lensDataVisible !== A.contains(arr, LENS_DATA)) {
      if (s.lensDataVisible) showPanel(LENS_DATA)
      else closePanel(LENS_DATA)
    }
    if (s.layoutVisible !== prev.layoutVisible && s.layoutVisible !== A.contains(arrangement().arr, LAYOUT_PANEL)) {
      if (s.layoutVisible) showPanel(LAYOUT_PANEL)
      else closePanel(LAYOUT_PANEL)
    }
    if (s.focusedPane !== prev.focusedPane && s.focusedPane) arrangement().setFocused(s.focusedPane)
  })
  setPopupHandlers({
    // The user closed a separate window: its panels close with it.
    closed(id) {
      const { arr } = arrangement()
      const w = arr.windows.find((x) => x.id === id)
      if (!w) return
      const panels = A.panelsOf(w.root)
      arrangement().update((a) => A.discard(a, id))
      for (const p of panels) if (A.isPane(p) && ws().panes.some((x) => x.key === p)) ws().removePane(p)
    },
    frame(id, f) {
      // Frames are not drawn: change them in place, without a redraw; autosave picks them up.
      const w = arrangement().arr.windows.find((x) => x.id === id)
      if (w) w.frame = f
      frameListeners.forEach((l) => l())
    },
  })
  flags()
}

const frameListeners = new Set<() => void>()
/** Called when a separate window moves or resizes (for autosave). */
export function onWindowFrames(l: () => void) {
  frameListeners.add(l)
  return () => frameListeners.delete(l)
}

