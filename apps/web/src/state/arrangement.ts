// Where a design's panels are (docs/APP.md "Panels and windows",
// `PanelArrangement` in the macOS app): a tree of splits and tab groups for
// the main window and for each separate window. A panel is in one place at a
// time; closing one takes it out of the tree (an analysis pane is then removed
// from the workspace, the lens data and the layout can be shown again).
//
// The tree is plain data. The functions below edit it in place; the store
// (`useArrangement`) clones it before each edit, so React sees new values.

import { create } from 'zustand'
import type { Json, JsonObject } from '@/engine/types'

/** `lens_data`, `layout`, or an analysis pane's key. */
export type PanelKey = string
export const LENS_DATA = 'lens_data'
export const LAYOUT_PANEL = 'layout'
export const isPane = (k: PanelKey) => k !== LENS_DATA && k !== LAYOUT_PANEL

export type Axis = 'h' | 'v'
export type Edge = 'leading' | 'trailing' | 'top' | 'bottom'
export const edgeAxis = (e: Edge): Axis => (e === 'leading' || e === 'trailing' ? 'h' : 'v')
/** The new panel comes before the existing content (leading or top). */
const leadingSide = (e: Edge) => e === 'leading' || e === 'top'

export interface GroupNode {
  kind: 'group'
  id: string
  panels: PanelKey[]
  selected: PanelKey | null
}
export interface SplitNode {
  kind: 'split'
  id: string
  axis: Axis
  children: PanelNode[]
  /** The children's shares of the space (they sum to 1). */
  fractions: number[]
}
export type PanelNode = GroupNode | SplitNode

/** Frame of a separate window on screen: x, y (top left), width, height in CSS pixels. */
export type Frame = [number, number, number, number]

export interface PanelWindow {
  id: string
  root: PanelNode | null
  frame: Frame | null
}

export interface Arrangement {
  main: PanelWindow
  windows: PanelWindow[]
}

export type DropTarget =
  | { kind: 'tab'; group: string; index: number | null }
  | { kind: 'split'; group: string; edge: Edge }
  | { kind: 'windowEdge'; window: string; edge: Edge }
  | { kind: 'newWindow' }

export const MAIN = 'main'

let seq = 0
export const newId = (p: string) => `${p}${(++seq).toString(36)}${Math.random().toString(36).slice(2, 6)}`

export const group = (panels: PanelKey[], selected: PanelKey | null = null): GroupNode => ({ kind: 'group', id: newId('g'), panels, selected })

export function split(axis: Axis, children: PanelNode[], fractions?: number[]): SplitNode {
  const f = fractions && fractions.length === children.length ? fractions : children.map(() => 1)
  return { kind: 'split', id: newId('s'), axis, children, fractions: normalizedShares(f) }
}

export function normalizedShares(f: number[]): number[] {
  const clean = f.map((x) => (Number.isFinite(x) && x > 0 ? x : 0))
  const sum = clean.reduce((a, b) => a + b, 0)
  if (!(sum > 0)) return f.map(() => 1 / Math.max(f.length, 1))
  return clean.map((x) => x / sum)
}

/** The panel on screen in a group. */
export const current = (g: GroupNode): PanelKey | null => (g.selected && g.panels.includes(g.selected) ? g.selected : (g.panels[0] ?? null))

export function groupsOf(node: PanelNode | null): GroupNode[] {
  if (!node) return []
  return node.kind === 'group' ? [node] : node.children.flatMap(groupsOf)
}
export const panelsOf = (node: PanelNode | null): PanelKey[] => groupsOf(node).flatMap((g) => g.panels)
export const allWindows = (a: Arrangement): PanelWindow[] => [a.main, ...a.windows]
export const allPanels = (a: Arrangement): PanelKey[] => allWindows(a).flatMap((w) => panelsOf(w.root))
export const windowById = (a: Arrangement, id: string) => allWindows(a).find((w) => w.id === id) ?? null

export function locate(a: Arrangement, panel: PanelKey): { window: PanelWindow; group: GroupNode } | null {
  for (const w of allWindows(a)) {
    const g = groupsOf(w.root).find((x) => x.panels.includes(panel))
    if (g) return { window: w, group: g }
  }
  return null
}
export const contains = (a: Arrangement, panel: PanelKey) => locate(a, panel) !== null

export function findGroup(a: Arrangement, id: string): { window: PanelWindow; group: GroupNode } | null {
  for (const w of allWindows(a)) {
    const g = groupsOf(w.root).find((x) => x.id === id)
    if (g) return { window: w, group: g }
  }
  return null
}

export function findSplit(a: Arrangement, id: string): SplitNode | null {
  const walk = (n: PanelNode | null): SplitNode | null => {
    if (!n || n.kind === 'group') return null
    if (n.id === id) return n
    for (const c of n.children) {
      const f = walk(c)
      if (f) return f
    }
    return null
  }
  for (const w of allWindows(a)) {
    const s = walk(w.root)
    if (s) return s
  }
  return null
}

// MARK: Defaults

const EDITOR_FRACTION = 0.36
const VIEWPORT_FRACTION = 0.42

/** The default window: the lens data across the top (36 %); under it the
 *  layout (42 %) and the analysis panes side by side. */
export function standard(panes: PanelKey[], opts: { lensData?: boolean; layout?: boolean } = {}): Arrangement {
  const showLens = opts.lensData ?? true
  const showLayout = opts.layout ?? true
  const lower: PanelNode[] = []
  const f: number[] = []
  if (showLayout) {
    lower.push(group([LAYOUT_PANEL]))
    f.push(panes.length ? VIEWPORT_FRACTION : 1)
  }
  for (const k of panes) {
    lower.push(group([k]))
    f.push((showLayout ? 1 - VIEWPORT_FRACTION : 1) / panes.length)
  }
  const below = lower.length === 0 ? null : lower.length === 1 ? lower[0] : split('h', lower, f)
  const top = showLens ? group([LENS_DATA]) : null
  const root = top && below ? split('v', [top, below], [EDITOR_FRACTION, 1 - EDITOR_FRACTION]) : (top ?? below)
  return { main: { id: MAIN, root, frame: null }, windows: [] }
}

// MARK: Tree editing (in place)

function take(panel: PanelKey, g: GroupNode) {
  const i = g.panels.indexOf(panel)
  if (i < 0) return
  g.panels.splice(i, 1)
  if (g.selected === panel) g.selected = g.panels.length ? g.panels[Math.min(i, g.panels.length - 1)] : null
}

function parentOf(id: string, node: PanelNode | null): [SplitNode, number] | null {
  if (!node || node.kind !== 'split') return null
  const i = node.children.findIndex((c) => c.id === id)
  if (i >= 0) return [node, i]
  for (const c of node.children) {
    const f = parentOf(id, c)
    if (f) return f
  }
  return null
}

function replaceNode(w: PanelWindow, id: string, node: PanelNode) {
  if (w.root?.id === id) {
    w.root = node
    return
  }
  const p = parentOf(id, w.root)
  if (p) p[0].children[p[1]] = node
}

/** Puts `node` beside `anchor`, taking `share` of the anchor's space. */
function insertBeside(w: PanelWindow, node: PanelNode, anchor: PanelNode, edge: Edge, share: number) {
  const axis = edgeAxis(edge)
  const p = parentOf(anchor.id, w.root)
  if (p && p[0].axis === axis) {
    const [parent, i] = p
    const taken = parent.fractions[i] * share
    parent.fractions[i] -= taken
    const at = leadingSide(edge) ? i : i + 1
    parent.children.splice(at, 0, node)
    parent.fractions.splice(at, 0, taken)
    parent.fractions = normalizedShares(parent.fractions)
    return
  }
  if (anchor.kind === 'split' && anchor.axis === axis) {
    const keep = anchor.fractions.map((x) => x * (1 - share))
    if (leadingSide(edge)) {
      anchor.children.unshift(node)
      anchor.fractions = normalizedShares([share, ...keep])
    } else {
      anchor.children.push(node)
      anchor.fractions = normalizedShares([...keep, share])
    }
    return
  }
  const s = split(axis, leadingSide(edge) ? [node, anchor] : [anchor, node], leadingSide(edge) ? [share, 1 - share] : [1 - share, share])
  replaceNode(w, anchor.id, s)
}

/** Puts `node` along a whole edge of a window's content. */
function attach(w: PanelWindow, node: PanelNode, edge: Edge, share: number) {
  if (!w.root) {
    w.root = node
    return
  }
  insertBeside(w, node, w.root, edge, share)
}

/** Adds `node` as a column after `anchor`: in a row, an equal share of it;
 *  otherwise splitting the anchor's space. */
function addColumn(w: PanelWindow, node: PanelNode, anchor: PanelNode) {
  const p = parentOf(anchor.id, w.root)
  if (p && p[0].axis === 'h') {
    const [parent, i] = p
    const n = parent.children.length
    parent.fractions = parent.fractions.map((x) => (x * n) / (n + 1))
    parent.children.splice(i + 1, 0, node)
    parent.fractions.splice(i + 1, 0, 1 / (n + 1))
    return
  }
  if (anchor.kind === 'split' && anchor.axis === 'h') {
    const n = anchor.children.length
    anchor.fractions = [...anchor.fractions.map((x) => (x * n) / (n + 1)), 1 / (n + 1)]
    anchor.children.push(node)
    return
  }
  insertBeside(w, node, anchor, 'trailing', 0.5)
}

/** The area under the lens data in the main window (the last child of a
 *  vertical root with the lens data on top), or the whole content when there
 *  is no lens data; null when the main window shows only the lens data. */
function lowerArea(a: Arrangement): PanelNode | null {
  const root = a.main.root
  if (!root) return null
  if (root.kind === 'split' && root.axis === 'v' && root.children.length > 1 && panelsOf(root.children[0]).includes(LENS_DATA)) {
    return root.children[root.children.length - 1]
  }
  if (panelsOf(root).includes(LENS_DATA)) return null
  return root
}

const onlyLens = (a: Arrangement) => {
  const p = panelsOf(a.main.root)
  return p.length === 1 && p[0] === LENS_DATA
}

/** Drops empty groups, collapses splits of one child, merges a split into a
 *  parent of the same axis, and removes empty separate windows. */
export function normalize(a: Arrangement) {
  const norm = (node: PanelNode): PanelNode | null => {
    if (node.kind === 'group') {
      if (node.selected && !node.panels.includes(node.selected)) node.selected = node.panels[0] ?? null
      return node.panels.length ? node : null
    }
    const kids: PanelNode[] = []
    const f: number[] = []
    node.children.forEach((c, k) => {
      const n = norm(c)
      if (!n) return
      const share = node.fractions[k] ?? 0
      if (n.kind === 'split' && n.axis === node.axis) {
        kids.push(...n.children)
        f.push(...n.fractions.map((x) => x * share))
      } else {
        kids.push(n)
        f.push(share)
      }
    })
    if (!kids.length) return null
    if (kids.length === 1) return kids[0]
    node.children = kids
    node.fractions = normalizedShares(f)
    return node
  }
  for (const w of allWindows(a)) w.root = w.root ? norm(w.root) : null
  a.windows = a.windows.filter((w) => w.root)
}

// MARK: Operations

export function select(a: Arrangement, panel: PanelKey) {
  const loc = locate(a, panel)
  if (loc) loc.group.selected = panel
}

/** Takes a panel out of the tree; a window left empty goes away. */
export function remove(a: Arrangement, panel: PanelKey) {
  const loc = locate(a, panel)
  if (!loc) return
  take(panel, loc.group)
  normalize(a)
}

/** Places a new analysis pane: as another column beside `hint`, else beside
 *  the layout, else in the area under the lens data, else along the main
 *  window's trailing edge. */
export function place(a: Arrangement, panel: PanelKey, hint: PanelKey | null) {
  if (contains(a, panel)) return
  const node = group([panel])
  const near = hint ? locate(a, hint) : null
  const lay = locate(a, LAYOUT_PANEL)
  if (near) addColumn(near.window, node, near.group)
  else if (lay && lay.window === a.main) addColumn(a.main, node, lay.group)
  else {
    const area = lowerArea(a)
    if (area) addColumn(a.main, node, area)
    else if (onlyLens(a)) attach(a.main, node, 'bottom', 0.64)
    else attach(a.main, node, 'trailing', 0.4)
  }
  normalize(a)
}

/** Places a new analysis pane in a given window, along its trailing edge. */
export function placeInWindow(a: Arrangement, panel: PanelKey, windowId: string) {
  const w = windowById(a, windowId)
  if (contains(a, panel) || !w) return
  attach(w, group([panel]), 'trailing', 1 / (groupsOf(w.root).length + 1))
  normalize(a)
}

/** Shows the lens data or the layout where they belong by default: the lens
 *  data across the top of the main window, the layout at the left of the
 *  area under it. */
export function show(a: Arrangement, panel: PanelKey) {
  if (contains(a, panel)) return
  const node = group([panel])
  if (panel === LENS_DATA) attach(a.main, node, 'top', EDITOR_FRACTION)
  else if (panel === LAYOUT_PANEL) {
    const area = lowerArea(a)
    if (area) insertBeside(a.main, node, area, 'leading', VIEWPORT_FRACTION)
    else attach(a.main, node, 'bottom', 0.64)
  } else place(a, panel, null)
  normalize(a)
}

/** Whether dropping `panel` on `target` changes anything. */
export function canMove(a: Arrangement, panel: PanelKey, target: DropTarget): boolean {
  const source = locate(a, panel)
  switch (target.kind) {
    case 'tab':
    case 'split': {
      const g = findGroup(a, target.group)
      if (!g) return false
      return g.group !== source?.group || g.group.panels.length > 1
    }
    case 'windowEdge': {
      const w = windowById(a, target.window)
      if (!w) return false
      const p = panelsOf(w.root)
      return !(p.length === 1 && p[0] === panel)
    }
    case 'newWindow': {
      if (!source) return true
      const p = panelsOf(source.window.root)
      return source.window === a.main || !(p.length === 1 && p[0] === panel)
    }
  }
}

/** Moves a panel. A `newWindow` target opens a separate window with `frame`;
 *  its id is returned. Windows left empty go away. */
export function move(a: Arrangement, panel: PanelKey, target: DropTarget, frame: Frame | null = null): string | null {
  if (!canMove(a, panel, target)) return null
  let opened: string | null = null
  switch (target.kind) {
    case 'tab': {
      const g = findGroup(a, target.group)?.group
      if (!g) return null
      const i = g.panels.indexOf(panel)
      if (i >= 0) {
        // Reorder within the group.
        const p = [...g.panels]
        p.splice(i, 1)
        const idx = target.index ?? p.length
        const j = Math.min(Math.max(idx - (idx > i ? 1 : 0), 0), p.length)
        p.splice(j, 0, panel)
        g.panels = p
      } else {
        const from = locate(a, panel)
        if (from) take(panel, from.group)
        g.panels.splice(Math.min(Math.max(target.index ?? g.panels.length, 0), g.panels.length), 0, panel)
      }
      g.selected = panel
      break
    }
    case 'split': {
      const from = locate(a, panel)
      if (from) take(panel, from.group)
      normalize(a)
      const g = findGroup(a, target.group)
      if (!g) return null
      insertBeside(g.window, group([panel]), g.group, target.edge, 0.5)
      break
    }
    case 'windowEdge': {
      const from = locate(a, panel)
      if (from) take(panel, from.group)
      normalize(a)
      const w = windowById(a, target.window)
      if (!w) return null
      attach(w, group([panel]), target.edge, 0.3)
      break
    }
    case 'newWindow': {
      const from = locate(a, panel)
      if (from) take(panel, from.group)
      const w: PanelWindow = { id: newId('w'), root: group([panel]), frame }
      a.windows.push(w)
      opened = w.id
      break
    }
  }
  normalize(a)
  return opened
}

/** Moves a panel from a separate window back to where its kind belongs in
 *  the main window: an analysis beside the other analyses, the lens data
 *  across the top, the layout at the left. */
export function returnToMain(a: Arrangement, panel: PanelKey) {
  const loc = locate(a, panel)
  if (!loc || loc.window === a.main) return
  take(panel, loc.group)
  normalize(a)
  if (isPane(panel)) {
    const neighbour = [...panelsOf(a.main.root)].reverse().find(isPane) ?? null
    place(a, panel, neighbour)
  } else show(a, panel)
  select(a, panel)
}

/** Moves a separate window's panels into the main window, keeping their
 *  arrangement (a lone panel goes where its kind belongs). */
export function dock(a: Arrangement, windowId: string) {
  const w = a.windows.find((x) => x.id === windowId)
  if (!w) return
  const panels = panelsOf(w.root)
  if (!w.root || panels.length === 0) {
    a.windows = a.windows.filter((x) => x !== w)
    return
  }
  if (panels.length === 1) {
    returnToMain(a, panels[0])
    return
  }
  const root = w.root
  a.windows = a.windows.filter((x) => x !== w)
  const area = lowerArea(a)
  if (area) addColumn(a.main, root, area)
  else attach(a.main, root, onlyLens(a) ? 'bottom' : 'trailing', 0.6)
  normalize(a)
}

/** Removes a separate window, returning the panels it showed. */
export function discard(a: Arrangement, windowId: string): PanelKey[] {
  const w = a.windows.find((x) => x.id === windowId)
  if (!w) return []
  a.windows = a.windows.filter((x) => x !== w)
  return panelsOf(w.root)
}

/** Moves the divider after child `k`; each side keeps at least `minShare`. */
export function resizeSplit(s: SplitNode, k: number, pair: [number, number]) {
  if (k + 1 >= s.fractions.length) return
  s.fractions[k] = pair[0]
  s.fractions[k + 1] = pair[1]
}

export function equalize(s: SplitNode) {
  s.fractions = s.children.map(() => 1 / Math.max(s.children.length, 1))
}

/** Brings the arrangement in line with the panes: closed panes leave it, new
 *  ones go beside the pane before them (or into a window they were added in). */
export function reconcile(a: Arrangement, paneKeys: PanelKey[], pendingWindow: Map<string, string> = new Map()) {
  const keys = new Set(paneKeys)
  for (const p of allPanels(a)) if (isPane(p) && !keys.has(p)) remove(a, p)
  paneKeys.forEach((k, i) => {
    if (contains(a, k)) return
    const w = pendingWindow.get(k)
    pendingWindow.delete(k)
    if (w && windowById(a, w)) placeInWindow(a, k, w)
    else place(a, k, i > 0 ? paneKeys[i - 1] : null)
  })
}

// MARK: Saving

const round = (x: number, k: number) => Math.round(x * k) / k

function encodeNode(n: PanelNode): JsonObject {
  if (n.kind === 'group') {
    const o: JsonObject = { tabs: [...n.panels] }
    const c = current(n)
    if (c && n.panels.length > 1) o.selected = c
    return o
  }
  return { axis: n.axis, children: n.children.map(encodeNode), fractions: n.fractions.map((x) => round(x, 10000)) }
}

/** `{"main": node, "windows": [{"frame": [x, y, w, h], "root": node}]}`. */
export function encode(a: Arrangement): JsonObject {
  const out: JsonObject = {}
  if (a.main.root) out.main = encodeNode(a.main.root)
  out.windows = a.windows
    .filter((w) => w.root)
    .map((w) => {
      const o: JsonObject = { root: encodeNode(w.root!) }
      if (w.frame) o.frame = w.frame.map((v) => round(v, 10))
      return o
    })
  return out
}

function decodeNode(j: Json, known: Set<PanelKey>, seen: Set<PanelKey>): PanelNode | null {
  if (!j || typeof j !== 'object' || Array.isArray(j)) return null
  if (Array.isArray(j.tabs)) {
    const panels: PanelKey[] = []
    for (const t of j.tabs) {
      if (typeof t !== 'string' || !known.has(t) || seen.has(t)) continue
      seen.add(t)
      panels.push(t)
    }
    if (!panels.length) return null
    return group(panels, typeof j.selected === 'string' ? j.selected : null)
  }
  if ((j.axis !== 'h' && j.axis !== 'v') || !Array.isArray(j.children)) return null
  const shares = Array.isArray(j.fractions) ? j.fractions : []
  const kids: PanelNode[] = []
  const f: number[] = []
  j.children.forEach((c, i) => {
    const n = decodeNode(c, known, seen)
    if (!n) return
    kids.push(n)
    f.push(typeof shares[i] === 'number' ? (shares[i] as number) : 1)
  })
  if (!kids.length) return null
  return kids.length === 1 ? kids[0] : split(j.axis, kids, f)
}

/** Reads a saved arrangement, keeping only the panels in `known` (each once);
 *  null when `j` is not one. */
export function decode(j: Json | undefined, known: Set<PanelKey>): Arrangement | null {
  if (!j || typeof j !== 'object' || Array.isArray(j)) return null
  const seen = new Set<PanelKey>()
  const main = j.main !== undefined ? decodeNode(j.main, known, seen) : null
  const windows: PanelWindow[] = []
  for (const w of Array.isArray(j.windows) ? j.windows : []) {
    if (!w || typeof w !== 'object' || Array.isArray(w)) continue
    const root = decodeNode(w.root, known, seen)
    if (!root) continue
    const f = Array.isArray(w.frame) ? w.frame.filter((x): x is number => typeof x === 'number' && Number.isFinite(x)) : []
    const frame: Frame | null = f.length === 4 && f[2] > 100 && f[3] > 100 ? [f[0], f[1], f[2], f[3]] : null
    windows.push({ id: newId('w'), root, frame })
  }
  const a: Arrangement = { main: { id: MAIN, root: main, frame: null }, windows }
  normalize(a)
  return a
}

// MARK: Store

interface ArrangementState {
  arr: Arrangement
  /** The panel the panel commands act on (Move to New Window, Close, …). */
  focused: PanelKey | null
  /** Edits a copy of the arrangement and normalises it. */
  update(fn: (a: Arrangement) => void): void
  /** Replaces the arrangement. */
  replace(a: Arrangement): void
  setFocused(k: PanelKey | null): void
}

export const cloneArrangement = (a: Arrangement): Arrangement => structuredClone(a)

export const useArrangement = create<ArrangementState>()((set, get) => ({
  arr: standard([]),
  focused: null,
  update(fn) {
    const a = cloneArrangement(get().arr)
    fn(a)
    normalize(a)
    set({ arr: a })
  },
  replace(a) {
    normalize(a)
    set({ arr: a })
  },
  setFocused(k) {
    if (get().focused !== k) set({ focused: k })
  },
}))
