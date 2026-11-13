// The panel tree: moving panels into tabs, splits and separate windows,
// showing and hiding, and saving it with the project (as PanelArrangementTests.swift).

import { describe, expect, it } from 'vitest'
import {
  canMove,
  decode,
  discard,
  dock,
  encode,
  LAYOUT_PANEL,
  LENS_DATA,
  locate,
  MAIN,
  move,
  place,
  placeInWindow,
  reconcile,
  remove,
  returnToMain,
  show,
  standard,
  current,
  type Arrangement,
  type PanelNode,
} from './arrangement'

const short = (k: string) => (k === LENS_DATA ? 'lens' : k)
function describeNode(n: PanelNode | null): string {
  if (!n) return '-'
  if (n.kind === 'group') return '[' + n.panels.map((p) => (n.panels.length > 1 && current(n) === p ? '*' : '') + short(p)).join(',') + ']'
  return n.axis + '(' + n.children.map(describeNode).join('|') + ')'
}
const tree = (a: Arrangement) => describeNode(a.main.root)
const gid = (a: Arrangement, p: string) => locate(a, p)!.group.id

describe('panel arrangement', () => {
  it('has a standard arrangement', () => {
    const a = standard(['p1', 'p2'])
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1]|[p2]))')
    const root = a.main.root!
    if (root.kind !== 'split') throw new Error('not split')
    const lower = root.children[1]
    if (lower.kind !== 'split') throw new Error('not split')
    expect(root.fractions[0]).toBeCloseTo(0.36, 9)
    expect(lower.fractions[0]).toBeCloseTo(0.42, 9)
    expect(lower.fractions[1]).toBeCloseTo(0.29, 9)
    expect(tree(standard([], { layout: false }))).toBe('[lens]')
    expect(tree(standard(['p1'], { lensData: false }))).toBe('h([layout]|[p1])')
  })

  it('makes tabs when dropped on a group', () => {
    const a = standard(['p1', 'p2'])
    const target = gid(a, 'p1')
    move(a, 'p2', { kind: 'tab', group: target, index: null })
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1,*p2]))')
    move(a, 'p2', { kind: 'tab', group: target, index: 0 })
    expect(tree(a)).toBe('v([lens]|h([layout]|[*p2,p1]))')
    move(a, 'p1', { kind: 'split', group: gid(a, LAYOUT_PANEL), edge: 'bottom' })
    expect(tree(a)).toBe('v([lens]|h(v([layout]|[p1])|[p2]))')
  })

  it('merges splits into a row of the same axis', () => {
    const a = standard(['p1', 'p2'])
    move(a, 'p2', { kind: 'split', group: gid(a, LAYOUT_PANEL), edge: 'leading' })
    expect(tree(a)).toBe('v([lens]|h([p2]|[layout]|[p1]))')
    move(a, LENS_DATA, { kind: 'windowEdge', window: MAIN, edge: 'trailing' })
    expect(tree(a)).toBe('h([p2]|[layout]|[p1]|[lens])')
    const root = a.main.root!
    if (root.kind !== 'split') throw new Error('not split')
    expect(root.fractions.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 9)
    expect(root.fractions.at(-1)).toBeCloseTo(0.3, 9)
  })

  it('does nothing where nothing would change', () => {
    const a = standard(['p1'])
    const g = gid(a, 'p1')
    expect(canMove(a, 'p1', { kind: 'tab', group: g, index: null })).toBe(false)
    expect(canMove(a, 'p1', { kind: 'split', group: g, edge: 'leading' })).toBe(false)
    expect(canMove(a, 'p1', { kind: 'newWindow' })).toBe(true)
    const w = move(a, 'p1', { kind: 'newWindow' }, [10, 20, 500, 400])!
    expect(canMove(a, 'p1', { kind: 'newWindow' })).toBe(false)
    expect(canMove(a, 'p1', { kind: 'windowEdge', window: w, edge: 'leading' })).toBe(false)
    expect(canMove(a, 'p1', { kind: 'windowEdge', window: MAIN, edge: 'trailing' })).toBe(true)
  })

  it('opens and closes separate windows', () => {
    const a = standard(['p1', 'p2'])
    const w = move(a, 'p1', { kind: 'newWindow' }, [10, 20, 500, 400])!
    expect(a.windows.length).toBe(1)
    expect(a.windows[0].frame?.[2]).toBe(500)
    expect(tree(a)).toBe('v([lens]|h([layout]|[p2]))')
    move(a, 'p2', { kind: 'split', group: gid(a, 'p1'), edge: 'trailing' })
    expect(describeNode(a.windows[0].root)).toBe('h([p1]|[p2])')
    dock(a, w)
    expect(a.windows).toEqual([])
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1]|[p2]))')
    move(a, 'p1', { kind: 'newWindow' })
    move(a, LENS_DATA, { kind: 'newWindow' })
    returnToMain(a, 'p1')
    expect(tree(a)).toBe('h([layout]|[p2]|[p1])')
    dock(a, a.windows[0].id)
    expect(tree(a)).toBe('v([lens]|h([layout]|[p2]|[p1]))')
    expect(a.windows).toEqual([])
    const w2 = move(a, LENS_DATA, { kind: 'newWindow' })!
    expect(a.windows.map((x) => x.id)).toEqual([w2])
    move(a, LENS_DATA, { kind: 'windowEdge', window: MAIN, edge: 'top' })
    expect(a.windows).toEqual([])
    expect(discard(a, MAIN)).toEqual([])
  })

  it('shows and hides the lens data and the layout', () => {
    const a = standard(['p1'])
    remove(a, LAYOUT_PANEL)
    expect(tree(a)).toBe('v([lens]|[p1])')
    show(a, LAYOUT_PANEL)
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1]))')
    remove(a, LENS_DATA)
    expect(tree(a)).toBe('h([layout]|[p1])')
    show(a, LENS_DATA)
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1]))')
    remove(a, LAYOUT_PANEL)
    remove(a, 'p1')
    remove(a, LENS_DATA)
    expect(a.main.root).toBeNull()
    show(a, LAYOUT_PANEL)
    show(a, LENS_DATA)
    expect(tree(a)).toBe('v([lens]|[layout])')
  })

  it('puts new panes beside their neighbour', () => {
    const a = standard(['p1', 'p2'])
    place(a, 'p3', 'p1')
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1]|[p3]|[p2]))')
    const a2 = standard([])
    place(a2, 'p1', null)
    expect(tree(a2)).toBe('v([lens]|h([layout]|[p1]))')
    const a3 = standard([], { layout: false })
    place(a3, 'p1', null)
    expect(tree(a3)).toBe('v([lens]|[p1])')
    const w = move(a, 'p2', { kind: 'newWindow' })!
    placeInWindow(a, 'p4', w)
    expect(describeNode(a.windows[0].root)).toBe('h([p2]|[p4])')
  })

  it('reconciles with the panes', () => {
    const a = standard(['p1', 'p2'])
    reconcile(a, ['p1', 'p2', 'p3'])
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1]|[p2]|[p3]))')
    reconcile(a, ['p1', 'p3'])
    expect(tree(a)).toBe('v([lens]|h([layout]|[p1]|[p3]))')
    const w = move(a, 'p3', { kind: 'newWindow' })!
    reconcile(a, ['p1', 'p3', 'p4'], new Map([['p4', w]]))
    expect(describeNode(a.windows[0].root)).toBe('h([p3]|[p4])')
    reconcile(a, ['p1'])
    expect(a.windows).toEqual([])
  })

  it('is saved with the project', () => {
    const a = standard(['p1', 'p2', 'p3'])
    move(a, 'p2', { kind: 'tab', group: gid(a, 'p1'), index: null })
    move(a, 'p3', { kind: 'newWindow' }, [100, 120, 640, 480])
    const known = new Set([LENS_DATA, LAYOUT_PANEL, 'p1', 'p2', 'p3'])
    const json = JSON.parse(JSON.stringify(encode(a)))
    const back = decode(json, known)!
    expect(tree(back)).toBe(tree(a))
    expect(back.windows.length).toBe(1)
    expect(back.windows[0].frame).toEqual([100, 120, 640, 480])
    expect(describeNode(back.windows[0].root)).toBe('[p3]')
    const fewer = decode(json, new Set([LENS_DATA, 'p1']))!
    expect(tree(fewer)).toBe('v([lens]|[p1])')
    expect(fewer.windows).toEqual([])
    expect(decode([], known)).toBeNull()
  })
})
