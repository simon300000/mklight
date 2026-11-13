// The layout's view commands (docs/APP.md "Keyboard"): ⌘2 Y–Z section,
// ⌥⌘2 X–Z section, ⌘3 3-D, ⌘0 fit.

import { eventShortcut } from '@/state/commands'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { SUMMARY, useWorkspace } from '@/state/workspace'
import type { JsonObject } from '@/engine/types'
import type { LayoutMode } from './camera'
import { isNonRotational } from './scene'

/** Whether the design offers the X–Z section (no rotational symmetry). */
export function offersXZ(): boolean {
  const summary = useWorkspace.getState().runs[SUMMARY]?.result?.json as JsonObject | undefined
  return isNonRotational(summary, useDocument.getState().project.system?.surfaces)
}

/** Switches the layout's view; the engine draws the section the view shows. */
export function setLayoutMode(mode: LayoutMode) {
  const v = useView.getState()
  if (mode === 'xz' && !offersXZ()) return
  if (v.layoutMode !== mode) v.set({ layoutMode: mode })
  syncLayoutSettings()
}

/** The layout's `section` and `include_3d` follow the view. */
export function syncLayoutSettings() {
  const mode = useView.getState().layoutMode
  const ws = useWorkspace.getState()
  const section = mode === 'xz' ? 'xz' : 'yz'
  const include3d = mode === '3d'
  if (ws.layoutSettings.section !== section || ws.layoutSettings.include_3d !== include3d) {
    ws.setLayoutSettings({ section, include_3d: include3d })
  }
}

export const fitLayout = () => useView.getState().fit(false)

export const LAYOUT_SHORTCUTS: { shortcut: string; title: string; run: () => void }[] = [
  { shortcut: 'mod+2', title: 'Y–Z Section', run: () => setLayoutMode('yz') },
  { shortcut: 'alt+mod+2', title: 'X–Z Section', run: () => setLayoutMode('xz') },
  { shortcut: 'mod+3', title: '3D View', run: () => setLayoutMode('3d') },
  { shortcut: 'mod+0', title: 'Fit Layout', run: fitLayout },
]

/** Handles ⌘2 / ⌥⌘2 / ⌘3 / ⌘0; true when the key was one of them. */
export function handleLayoutShortcut(e: KeyboardEvent): boolean {
  const sc = eventShortcut(e)
  const c = sc ? LAYOUT_SHORTCUTS.find((x) => x.shortcut === sc) : null
  if (!c) return false
  e.preventDefault()
  c.run()
  return true
}
