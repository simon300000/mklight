// Menu commands and their keyboard shortcuts (docs/APP.md "Keyboard",
// apps/macos/Sources/MKLightUI/Views/App/Commands.swift): ⌘ on Mac, Ctrl
// elsewhere (`mod`). Every toolbar item is also a menu command.

import { catalogText, rememberCatalog } from './catalogs'
import { usePalette } from './palette'
import { LAYOUT_PANEL, LENS_DATA } from './arrangement'
import { resetPanelLayout, togglePanel } from '@/views/panels/actions'
import { create } from 'zustand'
import { canDelete, deleteSurface, imageIndex, insertSurface, type System } from '@/model/system'
import { isVariable, paramResolves, paramTitle, setVariable, variableRefusal, type ParamPath } from '@/model/params'
import { addCatalogReference, catalogReference, catalogReferences } from '@/model/materials'
import { isMac, platformShortcut } from '@/lib/utils'
import { engineText, t, type Key } from '@/i18n'
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
  doc().edit(t('app.undoNames.insertSurface'), (s) => {
    at = insertSurface(s, index)
  })
  view().select(at)
}

export function deleteSurfaceAt(i: number) {
  const s = doc().project.system
  if (!canDelete(s, i)) {
    view().toast(i === 0 ? t('app.toasts.cannotDeleteObject') : i === imageIndex(s) ? t('app.toasts.cannotDeleteImage') : t('app.toasts.needOneSurface'), 'warning')
    return
  }
  doc().edit(t('app.undoNames.deleteSurface'), (sys) => deleteSurface(sys, i))
  view().select(Math.min(i, doc().project.system.surfaces.length - 2))
}

export const deleteSelectedSurface = () => deleteSurfaceAt(view().selection)

export function makeStop(i = view().selection) {
  const s = doc().project.system
  if (i <= 0 || i >= imageIndex(s)) {
    view().toast(t('app.toasts.stopObjectImage'), 'warning')
    return
  }
  doc().edit(t('app.undoNames.setStop'), (sys) => {
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
  return doc().editProject(on ? t('app.undoNames.makeVariable') : t('app.undoNames.removeVariable'), (proj) => setVariable(proj, p, on))
}

export function toggleParamVariable(p: ParamPath): boolean {
  return setParamVariable(p, !isVariable(doc().project, p))
}

/** Toggle Variable (⌃⌘V): the focused value of the lens table or inspector. */
export function toggleFocusedVariable() {
  const p = useValueFocus.getState().param
  const s = doc().project.system
  if (!p || !paramResolves(s, p)) {
    view().toast(t('app.toasts.selectValue'), 'info')
    return
  }
  const on = !isVariable(doc().project, p)
  if (setParamVariable(p, on)) view().toast(t(on ? 'app.toasts.nowVariable' : 'app.toasts.noLongerVariable', { name: paramTitle(s, p) }))
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
    view().toast(t('app.toasts.engineNotReady'), 'warning')
    return null
  }
  try {
    const text = catalogText(await f.arrayBuffer())
    const info = await engine.loadCatalog(f.name, text)
    await rememberCatalog(info.name, f.name, text)
    const materials = await engine.materialNames()
    useWorkspace.setState({ materials })
    doc().editProject(t('app.undoNames.addCatalog'), (p) => {
      p.catalogs = addCatalogReference(p.catalogs, catalogReference(f.name, info.name))
    })
    ws().designChanged(doc().project.system)
    useCatalogs.getState().refresh()
    const warnings = info.warnings ?? []
    if (warnings.length === 0) view().toast(t('app.toasts.catalogLoaded', { name: info.name, n: info.glass_count }))
    else {
      const w = warnings[0]
      const message = engineText(w.message)
      const first =
        w.line != null && w.glass ? t('app.toasts.warnLineGlass', { line: w.line, glass: w.glass, message })
        : w.line != null ? t('app.toasts.warnLine', { line: w.line, message })
        : w.glass ? t('app.toasts.warnGlass', { glass: w.glass, message })
        : message
      view().toast(t('app.toasts.catalogLoadedWarnings', { name: info.name, n: info.glass_count, warnings: t('common.warnings', { count: warnings.length }), first }), 'warning')
    }
    return info.name
  } catch (e) {
    view().toast(t('app.toasts.catalogLoadFailed', { name: f.name, error: engineText(e instanceof Error ? e.message : String(e)) }), 'error')
    return null
  }
}

/** Removes a catalog reference from the project ("Remove Glass Catalog"). The
 *  session keeps its glasses until the page reloads. */
export function removeGlassCatalog(name: string) {
  doc().editProject(t('app.undoNames.removeCatalog'), (p) => {
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

const DEFINITIONS: Omit<Command, 'title'>[] = [
  { id: 'palette', shortcut: 'mod+k', run: () => usePalette.getState().set(!usePalette.getState().open) },
  { id: 'new', shortcut: 'alt+mod+n', run: newDocument },
  { id: 'samples', shortcut: 'alt+mod+o', run: () => view().set({ welcomeOpen: true }) },
  { id: 'welcome', run: () => view().set({ welcomeOpen: true }) },
  { id: 'open', shortcut: 'mod+o', run: () => void openFromPicker() },
  { id: 'save', shortcut: 'mod+s', run: () => void save() },
  { id: 'saveAs', shortcut: 'shift+mod+s', run: () => void saveAs() },
  { id: 'undo', shortcut: 'mod+z', run: () => doc().undo(), enabled: () => doc().past.length > 0 },
  { id: 'redo', shortcut: 'shift+mod+z', run: () => doc().redo(), enabled: () => doc().future.length > 0 },
  { id: 'insertBefore', shortcut: 'shift+mod+i', run: () => insertSurfaceAt(Math.max(view().selection, 1)) },
  { id: 'insertAfter', shortcut: 'alt+mod+Enter', run: () => insertSurfaceAt(view().selection + 1) },
  { id: 'delete', shortcut: 'mod+Backspace', run: deleteSelectedSurface, enabled: () => canDelete(doc().project.system, view().selection) },
  { id: 'stop', shortcut: 'alt+mod+p', run: () => makeStop(), enabled: hasSurfaceSelection },
  {
    id: 'toggleVariable',
    shortcut: 'ctrl+mod+v',
    run: toggleFocusedVariable,
    enabled: () => {
      const p = useValueFocus.getState().param
      return !!p && paramResolves(doc().project.system, p)
    },
  },
  { id: 'loadCatalog', run: () => void loadGlassCatalog(), enabled: () => !!ws().engine },
  { id: 'runAll', shortcut: 'mod+Enter', run: () => ws().runAll({ fresh: true }) },
  { id: 'cancel', shortcut: 'mod+Period', run: () => ws().cancelAll() },
  { id: 'issues', shortcut: 'ctrl+mod+e', run: () => view().set({ issuesOpen: !view().issuesOpen }) },
  { id: 'lensData', shortcut: 'alt+mod+e', run: () => togglePanel(LENS_DATA) },
  { id: 'layout', shortcut: 'alt+mod+l', run: () => togglePanel(LAYOUT_PANEL) },
  { id: 'resetPanels', run: () => resetPanelLayout() },
  { id: 'inspector', shortcut: 'ctrl+mod+i', run: () => view().set({ inspectorVisible: !view().inspectorVisible }) },
  { id: 'sidebar', shortcut: 'ctrl+mod+s', run: () => view().set({ sidebarVisible: !view().sidebarVisible }) },
  { id: 'surfacePage', shortcut: 'alt+mod+1', run: () => view().setPage('surface') },
  { id: 'systemPage', shortcut: 'alt+mod+2', run: () => view().setPage('system') },
  { id: 'optimize', shortcut: 'ctrl+mod+o', run: () => view().setPage('optimize') },
  { id: 'glass', shortcut: 'ctrl+mod+g', run: () => view().setPage('glass') },
  { id: 'tolerance', shortcut: 'ctrl+mod+t', run: () => view().setPage('tolerance') },
  { id: 'solves', shortcut: 'ctrl+mod+l', run: () => view().setPage('solves') },
]

/** The commands. A title is read in the current language when shown. */
export const COMMANDS: Command[] = DEFINITIONS.map((c) =>
  Object.defineProperty({ ...c } as Command, 'title', {
    enumerable: true,
    get: () => t(`app.commands.${c.id}` as Key),
  }),
)

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
