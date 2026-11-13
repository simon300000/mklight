// Menu commands and their keyboard shortcuts (docs/APP.md "Keyboard"): ⌘ on
// Mac, Ctrl elsewhere (`mod`). Every toolbar item is also a menu command.

import { canDelete, deleteSurface, imageIndex, insertSurface } from '@/model/system'
import { isMac } from '@/lib/utils'
import { useDocument } from './document'
import { newDocument, openFromPicker, save, saveAs } from './files'
import { useView } from './view'
import { useWorkspace } from './workspace'

export interface Command {
  id: string
  title: string
  /** `mod+shift+i`: `mod` is ⌘ on Mac and Ctrl elsewhere. */
  shortcut?: string
  run(): void
  enabled?(): boolean
}

const doc = () => useDocument.getState()
const view = () => useView.getState()
const ws = () => useWorkspace.getState()

export function insertSurfaceAt(index: number) {
  let at = index
  doc().edit('Insert Surface', (s) => {
    at = insertSurface(s, index)
  })
  view().select(at)
}

export function deleteSelectedSurface() {
  const i = view().selection
  const s = doc().project.system
  if (!canDelete(s, i)) {
    view().toast(i === 0 ? 'The object surface cannot be deleted.' : i === imageIndex(s) ? 'The image surface cannot be deleted.' : 'A design needs at least one surface between object and image.', 'warning')
    return
  }
  doc().edit('Delete Surface', (sys) => deleteSurface(sys, i))
  view().select(Math.min(i, doc().project.system.surfaces.length - 2))
}

export function makeStop(i = view().selection) {
  const s = doc().project.system
  if (i <= 0 || i >= imageIndex(s)) {
    view().toast('The object and the image cannot be the stop.', 'warning')
    return
  }
  doc().edit('Set Stop', (sys) => {
    sys.stop = i
  })
}

export const COMMANDS: Command[] = [
  { id: 'new', title: 'New Design', shortcut: 'alt+mod+n', run: newDocument },
  { id: 'samples', title: 'Samples…', shortcut: 'shift+mod+n', run: () => view().set({ welcomeOpen: true }) },
  { id: 'open', title: 'Open…', shortcut: 'mod+o', run: () => void openFromPicker() },
  { id: 'save', title: 'Save', shortcut: 'mod+s', run: () => void save() },
  { id: 'saveAs', title: 'Save As…', shortcut: 'shift+mod+s', run: () => void saveAs() },
  { id: 'undo', title: 'Undo', shortcut: 'mod+z', run: () => doc().undo(), enabled: () => doc().past.length > 0 },
  { id: 'redo', title: 'Redo', shortcut: 'shift+mod+z', run: () => doc().redo(), enabled: () => doc().future.length > 0 },
  { id: 'insertBefore', title: 'Insert Surface Before', shortcut: 'shift+mod+i', run: () => insertSurfaceAt(Math.max(view().selection, 1)) },
  { id: 'insertAfter', title: 'Insert Surface After', shortcut: 'alt+mod+i', run: () => insertSurfaceAt(view().selection + 1) },
  { id: 'delete', title: 'Delete Surface', shortcut: 'mod+Backspace', run: deleteSelectedSurface },
  { id: 'stop', title: 'Make Aperture Stop', shortcut: 'shift+mod+t', run: () => makeStop() },
  { id: 'runAll', title: 'Run Analyses', shortcut: 'mod+r', run: () => ws().runAll({ fresh: true }) },
  { id: 'cancel', title: 'Cancel', shortcut: 'mod+Period', run: () => ws().cancelAll() },
  { id: 'issues', title: 'Show Issues', shortcut: 'ctrl+mod+e', run: () => view().set({ issuesOpen: !view().issuesOpen }) },
  { id: 'lensData', title: 'Lens Data', shortcut: 'alt+mod+e', run: () => view().set({ lensDataVisible: true }) },
  { id: 'layout', title: 'Layout', shortcut: 'alt+mod+l', run: () => view().set({ layoutVisible: true }) },
  { id: 'inspector', title: 'Show Inspector', shortcut: 'ctrl+mod+i', run: () => view().set({ inspectorVisible: !view().inspectorVisible }) },
  { id: 'sidebar', title: 'Show Sidebar', shortcut: 'ctrl+mod+s', run: () => view().set({ sidebarVisible: !view().sidebarVisible }) },
  { id: 'surfacePage', title: 'Surface', shortcut: 'mod+1', run: () => view().setPage('surface') },
  { id: 'systemPage', title: 'System', shortcut: 'alt+mod+1', run: () => view().setPage('system') },
]

export const command = (id: string) => COMMANDS.find((c) => c.id === id)!

const KEY_NAMES: Record<string, string> = { '.': 'Period', Backspace: 'Backspace', Delete: 'Backspace' }

/** The shortcut string of a key event, in the commands' notation. */
export function eventShortcut(e: KeyboardEvent): string | null {
  const mod = isMac ? e.metaKey : e.ctrlKey
  if (!mod) return null
  const parts: string[] = []
  if (isMac && e.ctrlKey) parts.push('ctrl')
  if (e.altKey) parts.push('alt')
  if (e.shiftKey) parts.push('shift')
  parts.push('mod')
  let key = KEY_NAMES[e.key] ?? e.key
  // With ⌥ or ⇧ held, e.key is the composed character; use the physical key.
  if (/^Key[A-Z]$/.test(e.code)) key = e.code.slice(3).toLowerCase()
  else if (/^Digit\d$/.test(e.code)) key = e.code.slice(5)
  else if (e.code === 'Period') key = 'Period'
  parts.push(key)
  return parts.join('+')
}

/** True when typing in a text field should keep the key (undo of text, …). */
function inTextField(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null
  if (!t) return false
  return t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable
}

export function handleShortcut(e: KeyboardEvent): boolean {
  const sc = eventShortcut(e)
  if (!sc) return false
  const cmd = COMMANDS.find((c) => c.shortcut === sc)
  if (!cmd) return false
  // Text fields keep their own undo, redo and delete.
  if (inTextField(e) && ['undo', 'redo', 'delete'].includes(cmd.id)) return false
  e.preventDefault()
  if (cmd.enabled && !cmd.enabled()) return true
  cmd.run()
  return true
}
