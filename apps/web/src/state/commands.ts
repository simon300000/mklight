// Menu commands and their keyboard shortcuts (docs/APP.md "Keyboard",
// apps/macos/Sources/MKLightUI/Views/App/Commands.swift): ⌘ on Mac, Ctrl
// elsewhere (`mod`). Every toolbar item is also a menu command.

import { rememberCatalog } from './catalogs'
import { usePalette } from './palette'
import { LAYOUT_PANEL, LENS_DATA } from './arrangement'
import { resetPanelLayout, togglePanel } from '@/views/panels/actions'
import { create } from 'zustand'
import { canDelete, deleteSurface, imageIndex, insertSurface, type System } from '@/model/system'
import { isVariable, paramResolves, paramTitle, setVariable, variableRefusal, type ParamPath } from '@/model/params'
import { addCatalogReference, catalogReference, catalogReferences } from '@/model/materials'
import { isMac, platformShortcut } from '@/lib/utils'
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

// MARK: The lens table's focused value

/** The parameter of the lens table cell (or inspector field) last focused,
 *  for Toggle Variable (⌃⌘V). */
interface FocusState {
  param: ParamPath | null
  setParam(p: ParamPath | null): void
}

export const useValueFocus = create<FocusState>()((set) => ({
  param: null,
  setParam: (param) => set({ param }),
}))

// MARK: Surfaces

export function insertSurfaceAt(index: number) {
  let at = index
  doc().edit('Insert Surface', (s) => {
    at = insertSurface(s, index)
  })
  view().select(at)
}

export function deleteSurfaceAt(i: number) {
  const s = doc().project.system
  if (!canDelete(s, i)) {
    view().toast(i === 0 ? 'The object surface cannot be deleted.' : i === imageIndex(s) ? 'The image surface cannot be deleted.' : 'A design needs at least one surface between object and image.', 'warning')
    return
  }
  doc().edit('Delete Surface', (sys) => deleteSurface(sys, i))
  view().select(Math.min(i, doc().project.system.surfaces.length - 2))
}

export const deleteSelectedSurface = () => deleteSurfaceAt(view().selection)

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

const hasSurfaceSelection = () => {
  const s = doc().project.system
  const i = view().selection
  return !!s?.surfaces && i > 0 && i < imageIndex(s)
}

// MARK: Variables

/** Makes `p` a variable or a fixed parameter as one undoable step ("Make
 *  Variable" / "Remove Variable"). A solved parameter is refused with why. */
export function setParamVariable(p: ParamPath, on: boolean): boolean {
  const project = doc().project
  if (on) {
    const why = variableRefusal(project.system, p)
    if (why) {
      view().toast(why, 'warning')
      return false
    }
  }
  return doc().editProject(on ? 'Make Variable' : 'Remove Variable', (proj) => setVariable(proj, p, on))
}

export function toggleParamVariable(p: ParamPath): boolean {
  return setParamVariable(p, !isVariable(doc().project, p))
}

/** Toggle Variable (⌃⌘V): the focused value of the lens table or inspector. */
export function toggleFocusedVariable() {
  const p = useValueFocus.getState().param
  const s = doc().project.system
  if (!p || !paramResolves(s, p)) {
    view().toast('Select a value in the lens data to make it a variable.', 'info')
    return
  }
  const on = !isVariable(doc().project, p)
  if (setParamVariable(p, on)) view().toast(`${paramTitle(s, p)} is ${on ? 'now a variable' : 'no longer a variable'}.`)
}

// MARK: Glass catalogs

/** Asks for an AGF (or GlassCatalog JSON) file. */
export function pickCatalogFile(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.agf,.AGF,.json,text/plain,application/json'
    input.style.display = 'none'
    input.addEventListener('change', () => {
      resolve(input.files?.[0] ?? null)
      input.remove()
    })
    input.addEventListener('cancel', () => {
      resolve(null)
      input.remove()
    })
    document.body.appendChild(input)
    input.click()
  })
}

/** Load Glass Catalog…: reads the file into the engine's material library,
 *  records it in the project ("Add Glass Catalog", undoable) and rebuilds the
 *  snapshot so glasses resolve. Returns the catalog's name, or null. */
export async function loadGlassCatalog(file?: File | null): Promise<string | null> {
  const f = file ?? (await pickCatalogFile())
  if (!f) return null
  const engine = ws().engine
  if (!engine) {
    view().toast('The engine is not ready yet.', 'warning')
    return null
  }
  try {
    const text = await f.text()
    const info = await engine.loadCatalog(f.name, text)
    await rememberCatalog(info.name, f.name, text)
    const materials = await engine.materialNames()
    useWorkspace.setState({ materials })
    doc().editProject('Add Glass Catalog', (p) => {
      p.catalogs = addCatalogReference(p.catalogs, catalogReference(f.name, info.name))
    })
    ws().designChanged(doc().project.system)
    useCatalogs.getState().refresh()
    view().toast(`Loaded ${info.name}: ${info.glass_count} glasses.`)
    return info.name
  } catch (e) {
    view().toast(`${f.name} could not be loaded: ${e instanceof Error ? e.message : String(e)}`, 'error')
    return null
  }
}

/** Removes a catalog reference from the project ("Remove Glass Catalog"). The
 *  session keeps its glasses until the page reloads. */
export function removeGlassCatalog(name: string) {
  doc().editProject('Remove Glass Catalog', (p) => {
    p.catalogs = catalogReferences(p.catalogs).filter((c) => c.name !== name)
  })
}

/** The engine's loaded catalogs (built-in first). */
interface CatalogsState {
  list: { name: string; source: string; glass_count: number }[]
  refresh(): Promise<void>
}

export const useCatalogs = create<CatalogsState>()((set) => ({
  list: [],
  async refresh() {
    const engine = useWorkspace.getState().engine
    if (!engine) return
    try {
      set({ list: await engine.catalogs() })
    } catch {
      set({ list: [] })
    }
  },
}))

// MARK: Commands

export const COMMANDS: Command[] = [
  { id: 'palette', title: 'Command Palette…', shortcut: 'mod+k', run: () => usePalette.getState().set(!usePalette.getState().open) },
  { id: 'new', title: 'New Design', shortcut: 'alt+mod+n', run: newDocument },
  { id: 'samples', title: 'Samples…', shortcut: 'alt+mod+o', run: () => view().set({ welcomeOpen: true }) },
  { id: 'welcome', title: 'Show Welcome Window', run: () => view().set({ welcomeOpen: true }) },
  { id: 'open', title: 'Open…', shortcut: 'mod+o', run: () => void openFromPicker() },
  { id: 'save', title: 'Save', shortcut: 'mod+s', run: () => void save() },
  { id: 'saveAs', title: 'Save As…', shortcut: 'shift+mod+s', run: () => void saveAs() },
  { id: 'undo', title: 'Undo', shortcut: 'mod+z', run: () => doc().undo(), enabled: () => doc().past.length > 0 },
  { id: 'redo', title: 'Redo', shortcut: 'shift+mod+z', run: () => doc().redo(), enabled: () => doc().future.length > 0 },
  { id: 'insertBefore', title: 'Insert Surface Before', shortcut: 'shift+mod+i', run: () => insertSurfaceAt(Math.max(view().selection, 1)) },
  { id: 'insertAfter', title: 'Insert Surface After', shortcut: 'alt+mod+Enter', run: () => insertSurfaceAt(view().selection + 1) },
  { id: 'delete', title: 'Delete Surface', shortcut: 'mod+Backspace', run: deleteSelectedSurface, enabled: () => canDelete(doc().project.system, view().selection) },
  { id: 'stop', title: 'Make Aperture Stop', shortcut: 'alt+mod+p', run: () => makeStop(), enabled: hasSurfaceSelection },
  {
    id: 'toggleVariable',
    title: 'Toggle Variable',
    shortcut: 'ctrl+mod+v',
    run: toggleFocusedVariable,
    enabled: () => {
      const p = useValueFocus.getState().param
      return !!p && paramResolves(doc().project.system, p)
    },
  },
  { id: 'loadCatalog', title: 'Load Glass Catalog…', run: () => void loadGlassCatalog(), enabled: () => !!ws().engine },
  { id: 'runAll', title: 'Run Analyses', shortcut: 'mod+Enter', run: () => ws().runAll({ fresh: true }) },
  { id: 'cancel', title: 'Cancel Running Analyses', shortcut: 'mod+Period', run: () => ws().cancelAll() },
  { id: 'issues', title: 'Show Issues', shortcut: 'ctrl+mod+e', run: () => view().set({ issuesOpen: !view().issuesOpen }) },
  { id: 'lensData', title: 'Lens Data', shortcut: 'alt+mod+e', run: () => togglePanel(LENS_DATA) },
  { id: 'layout', title: 'Layout', shortcut: 'alt+mod+l', run: () => togglePanel(LAYOUT_PANEL) },
  { id: 'resetPanels', title: 'Reset Panel Layout', run: () => resetPanelLayout() },
  { id: 'inspector', title: 'Show Inspector', shortcut: 'ctrl+mod+i', run: () => view().set({ inspectorVisible: !view().inspectorVisible }) },
  { id: 'sidebar', title: 'Show Sidebar', shortcut: 'ctrl+mod+s', run: () => view().set({ sidebarVisible: !view().sidebarVisible }) },
  { id: 'surfacePage', title: 'Surface', shortcut: 'alt+mod+1', run: () => view().setPage('surface') },
  { id: 'systemPage', title: 'System', shortcut: 'alt+mod+2', run: () => view().setPage('system') },
  { id: 'optimize', title: 'Optimize…', shortcut: 'ctrl+mod+o', run: () => view().setPage('optimize') },
  { id: 'glass', title: 'Glass Substitution…', shortcut: 'ctrl+mod+g', run: () => view().setPage('glass') },
  { id: 'tolerance', title: 'Tolerance…', shortcut: 'ctrl+mod+t', run: () => view().setPage('tolerance') },
  { id: 'solves', title: 'Apply Solves…', shortcut: 'ctrl+mod+l', run: () => view().setPage('solves') },
]

export const command = (id: string) => COMMANDS.find((c) => c.id === id)!

const KEY_NAMES: Record<string, string> = {
  Enter: 'Enter', '.': 'Period', Backspace: 'Backspace', Delete: 'Backspace' }

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
  const cmd = COMMANDS.find((c) => c.shortcut && platformShortcut(c.shortcut) === sc)
  if (!cmd) return false
  // Text fields keep their own undo, redo and delete.
  if (inTextField(e) && ['undo', 'redo', 'delete'].includes(cmd.id)) return false
  e.preventDefault()
  if (cmd.enabled && !cmd.enabled()) {
    if (cmd.id === 'toggleVariable') toggleFocusedVariable()
    return true
  }
  cmd.run()
  return true
}

/** Whether the system has a value at `p` (re-exported for menus). */
export const canToggle = (s: System, p: ParamPath | null) => !!p && paramResolves(s, p)
