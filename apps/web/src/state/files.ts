// Documents: New, Samples, Open / Save `.mklight`, Import `.zmx` / `.seq`, and
// autosave to IndexedDB. Uses the File System Access API where the browser has
// it (Chrome) and falls back to a file input and a download (Safari).

import { del as idbDel, get as idbGet, set as idbSet } from 'idb-keyval'
import type { Json, JsonObject } from '@/engine/types'
import { readProject, writeProject, projectFromSystem, type Project, SAMPLES, ProjectError } from '@/model/project'
import { newSystem, clone, type System } from '@/model/system'
import { useDocument, type DocumentFile } from './document'
import { useView } from './view'
import { useWorkspace, LAYOUT, type Pane } from './workspace'
import { useImportReport } from './importReport'
import { useJobs } from './jobs'
import { endTolerancePreview } from './tolerance'
import { restoreCatalogs } from './catalogs'
import * as A from './arrangement'
import { useArrangement } from './arrangement'
import type { LayoutMode } from '@/views/layout/camera'
import { fromSavedFrame, toSavedFrame } from '@/views/panels/windows'

const AUTOSAVE_KEY = 'mklight-autosave-v1'
/** An autosaved document with unsaved changes, kept aside until the user recovers or discards it. */
const RECOVERY_KEY = 'mklight-recovery-v1'

type PickerWindow = Window & {
  showOpenFilePicker?: (o?: object) => Promise<FileSystemFileHandle[]>
  showSaveFilePicker?: (o?: object) => Promise<FileSystemFileHandle>
}
const w = () => window as PickerWindow
export const hasFileSystemAccess = () => typeof window !== 'undefined' && typeof w().showOpenFilePicker === 'function'

const PROJECT_TYPES = [{ description: 'MKLight project', accept: { 'application/json': ['.mklight'] } }]
const OPEN_TYPES = [
  { description: 'MKLight project or lens prescription', accept: { 'application/octet-stream': ['.mklight', '.zmx', '.seq'] } },
]

// MARK: Views ↔ panes

/** Writes the panes and the layout's settings and view (mode, ray colours,
 *  stretch, the panel arrangement of every window) into the project's `views`. */
export function viewsFromWorkspace(): Json[] {
  const ws = useWorkspace.getState()
  const v = useView.getState()
  const views: Json[] = ws.panes.map((p) => ({
    key: p.key,
    analysis: p.analysis,
    settings: p.settings,
    ui: { ...p.ui, auto: p.auto, settings_visible: p.settingsOpen },
  }))
  const arr = A.cloneArrangement(useArrangement.getState().arr)
  for (const w of arr.windows) if (w.frame) w.frame = toSavedFrame(w.frame)
  const ui: JsonObject = { mode: v.layoutMode, ray_coloring: v.rayColoring, arrangement: A.encode(arr) }
  if (v.stretchChosen || v.stretch !== 1) ui.stretch = v.stretch
  views.unshift({ key: LAYOUT, analysis: 'layout', settings: ws.layoutSettings, ui })
  return views
}

const MODES = ['yz', 'xz', '3d']

export function applyViews(views: Json[]) {
  const panes: Pane[] = []
  let layoutSettings: JsonObject | null = null
  let layoutUI: JsonObject = {}
  for (const raw of views) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const v = raw as JsonObject
    const ui = (v.ui as JsonObject) ?? {}
    if (v.key === LAYOUT || v.analysis === 'layout') {
      layoutSettings = (v.settings as JsonObject) ?? null
      layoutUI = ui
      continue
    }
    if (typeof v.analysis !== 'string') continue
    panes.push({
      key: String(v.key ?? `pane-${panes.length}`),
      analysis: v.analysis,
      settings: (v.settings as JsonObject) ?? {},
      auto: ui.auto !== false && ui.auto_update !== false,
      settingsOpen: ui.settings_visible === true,
      ui: Object.fromEntries(Object.entries(ui).filter(([k]) => !['auto', 'auto_update', 'settings_visible'].includes(k))),
    })
  }
  // The layout's view: a saved stretch (even ×1) is kept instead of the automatic choice.
  const mode = typeof layoutUI.mode === 'string' && MODES.includes(layoutUI.mode) ? (layoutUI.mode as LayoutMode)
    : layoutSettings?.section === 'xz' ? 'xz' : 'yz'
  const stretch = typeof layoutUI.stretch === 'number' && Number.isFinite(layoutUI.stretch) && layoutUI.stretch >= 1 ? layoutUI.stretch : null
  useView.getState().set({
    layoutMode: mode,
    rayColoring: layoutUI.ray_coloring === 'wavelength' ? 'wavelength' : 'field',
    stretch: stretch ?? 1,
    stretchChosen: stretch !== null,
  })
  // The arrangement first, so the panes find their places.
  const known = new Set<string>([A.LENS_DATA, A.LAYOUT_PANEL, ...panes.map((p) => p.key)])
  const arr = A.decode(layoutUI.arrangement, known) ?? A.standard(panes.map((p) => p.key))
  for (const w of arr.windows) if (w.frame) w.frame = fromSavedFrame(w.frame)
  useArrangement.getState().replace(arr)
  const ws = useWorkspace.getState()
  const base = { ...ws.layoutSettings, ...(layoutSettings ?? {}) }
  useWorkspace.setState({ layoutSettings: { ...base, section: mode === 'xz' ? 'xz' : 'yz', include_3d: mode === '3d' } })
  ws.setPanes(panes)
}

/** Default panes of a new window: spot diagram and ray fan. */
const DEFAULT_PANES: Pane[] = [
  { key: 'pane-spot', analysis: 'spot_diagram', settings: { through_focus: { min: null, max: null, steps: 3 } }, auto: true, settingsOpen: false },
  { key: 'pane-fan', analysis: 'ray_fan', settings: {}, auto: true, settingsOpen: false },
]

function openProject(project: Project, file: DocumentFile | null, opts?: { dirty?: boolean; keepPanes?: boolean }) {
  // A design job's run or result belongs to the design it was started on.
  endTolerancePreview()
  useJobs.getState().reset()
  useDocument.getState().open(project, file, opts)
  const v = useView.getState()
  v.set({ selection: Math.min(1, project.system.surfaces.length - 1), welcomeOpen: false, layoutEpoch: v.layoutEpoch + 1 })
  if (project.views.length) applyViews(project.views)
  else if (!opts?.keepPanes) {
    // Views without an arrangement open in the standard one.
    v.set({ layoutMode: 'yz', stretch: 1, stretchChosen: false })
    const panes = DEFAULT_PANES.map((p) => ({ ...p }))
    useArrangement.getState().replace(A.standard(panes.map((p) => p.key)))
    useWorkspace.getState().setPanes(panes)
    useWorkspace.getState().setLayoutSettings({ section: 'yz', include_3d: false })
  }
}

// MARK: New, samples

export function newDocument() {
  openProject(projectFromSystem(newSystem()), null)
}

export function openSample(id: string) {
  const s = SAMPLES.find((x) => x.id === id)
  if (!s) return
  openProject(projectFromSystem(clone(s.system)), null)
}

/** Replace with Sample: one undoable step in the current document. */
export function replaceWithSample(id: string) {
  const s = SAMPLES.find((x) => x.id === id)
  if (!s) return
  const doc = useDocument.getState()
  doc.replace({ ...doc.project, system: clone(s.system) }, 'Replace with Sample')
}

// MARK: Open

export async function openFromPicker() {
  if (hasFileSystemAccess()) {
    try {
      const [handle] = await w().showOpenFilePicker!({ types: OPEN_TYPES, multiple: false })
      const file = await handle.getFile()
      await openFile(file, handle)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') useView.getState().toast((e as Error).message, 'error')
    }
    return
  }
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.mklight,.zmx,.seq'
  input.onchange = () => {
    const f = input.files?.[0]
    if (f) void openFile(f, null)
  }
  input.click()
}

export function prescriptionFormat(name: string): 'zmx' | 'seq' | null {
  const ext = name.toLowerCase().split('.').pop()
  return ext === 'zmx' ? 'zmx' : ext === 'seq' ? 'seq' : null
}

/** Opens a dropped or picked file: a project, or a lens prescription to import. */
export async function openFile(file: File, handle: FileSystemFileHandle | null) {
  const toast = useView.getState().toast
  const format = prescriptionFormat(file.name)
  try {
    if (format) {
      const engine = useWorkspace.getState().engine
      if (!engine) throw new Error('The engine is not ready yet.')
      const data = await file.arrayBuffer()
      const result = await engine.importPrescription(data.slice(0), format)
      const system = result.system as System
      if (!system.name) system.name = file.name.replace(/\.(zmx|seq)$/i, '')
      openProject(projectFromSystem(system), { name: file.name, handle: null, imported: true }, { dirty: true })
      // A clean import shows a short note instead of the report.
      if (result.warnings.length) useImportReport.getState().show(file.name, result.warnings, { data, format })
      else toast(`Imported ${file.name}: ${system.surfaces.length - 2} surfaces.`)
      return
    }
    const project = readProject(await file.text())
    openProject(project, { name: file.name, handle, imported: false })
    const missing = await restoreCatalogs(project)
    if (missing.length) toast(`Glass catalog${missing.length > 1 ? 's' : ''} not loaded: ${missing.join(', ')}. Load ${missing.length > 1 ? 'them' : 'it'} in the System inspector.`, 'warning')
  } catch (e) {
    const msg = e instanceof ProjectError ? e.message : (e as Error).message
    toast(`${file.name} could not be opened. ${msg}`, 'error')
  }
}

// MARK: Save

function currentProjectText(): string {
  const doc = useDocument.getState()
  const ws = useWorkspace.getState()
  const project = { ...doc.project, views: viewsFromWorkspace() }
  return writeProject(project, ws.engine?.info.version ?? '')
}

function suggestedName(): string {
  const doc = useDocument.getState()
  const base = doc.file?.name.replace(/\.(mklight|zmx|seq)$/i, '') || doc.project.system.name || 'Untitled'
  return base.replace(/[\\/:*?"<>|]/g, '-') + '.mklight'
}

export async function save() {
  const doc = useDocument.getState()
  if (doc.file?.handle && !doc.file.imported) {
    try {
      await writeHandle(doc.file.handle, currentProjectText())
      doc.markSaved(doc.file)
    } catch (e) {
      useView.getState().toast(`Could not save: ${(e as Error).message}`, 'error')
    }
    return
  }
  await saveAs()
}

export async function saveAs() {
  const text = currentProjectText()
  const name = suggestedName()
  if (typeof w().showSaveFilePicker === 'function') {
    try {
      const handle = await w().showSaveFilePicker!({ suggestedName: name, types: PROJECT_TYPES })
      await writeHandle(handle, text)
      useDocument.getState().markSaved({ name: handle.name, handle, imported: false })
    } catch (e) {
      if ((e as Error).name !== 'AbortError') useView.getState().toast(`Could not save: ${(e as Error).message}`, 'error')
    }
    return
  }
  download(name, text)
  useDocument.getState().markSaved({ name, handle: null, imported: false })
}

async function writeHandle(handle: FileSystemFileHandle, text: string) {
  const writable = await handle.createWritable()
  await writable.write(text)
  await writable.close()
}

export function download(name: string, text: string, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// MARK: Autosave

let autosaveTimer: ReturnType<typeof setTimeout> | null = null

export function scheduleAutosave() {
  if (autosaveTimer) clearTimeout(autosaveTimer)
  autosaveTimer = setTimeout(() => void autosaveNow(), 800)
}

export async function autosaveNow() {
  const doc = useDocument.getState()
  if (!doc.project.system?.surfaces) return
  try {
    await idbSet(AUTOSAVE_KEY, {
      text: currentProjectText(),
      file: doc.file ? { name: doc.file.name, imported: doc.file.imported } : null,
      handle: doc.file?.handle ?? null,
      dirty: doc.revision !== doc.savedRevision,
      time: Date.now(),
    })
  } catch {
    // Private windows may refuse IndexedDB; autosave is a convenience.
  }
}

interface Autosave {
  text: string
  file: { name: string; imported: boolean } | null
  handle: FileSystemFileHandle | null
  dirty: boolean
  time?: number
}

/** Restores the autosaved document when it had no unsaved changes; false when
 *  there is none. One with unsaved changes is not opened silently: it is kept
 *  aside and offered in the welcome window ("Recovered: …"). */
export async function restoreAutosave(): Promise<boolean> {
  try {
    const saved = (await idbGet(AUTOSAVE_KEY)) as Autosave | undefined
    if (!saved?.text) {
      await offerRecovery()
      return false
    }
    if (saved.dirty) {
      await idbSet(RECOVERY_KEY, saved)
      await offerRecovery()
      return false
    }
    openAutosave(saved)
    await offerRecovery()
    return true
  } catch {
    return false
  }
}

function openAutosave(saved: Autosave) {
  const project = readProject(saved.text)
  const file = saved.file ? { ...saved.file, handle: saved.handle ?? null } : null
  openProject(project, file, { dirty: saved.dirty })
}

function recoveryName(saved: Autosave): string {
  if (saved.file?.name) return saved.file.name
  try {
    return readProject(saved.text).system.name || 'Untitled'
  } catch {
    return 'Untitled'
  }
}

/** Shows the kept-aside document, if any, in the welcome window. */
async function offerRecovery() {
  const saved = (await idbGet(RECOVERY_KEY).catch(() => undefined)) as Autosave | undefined
  useView.getState().set({ recovery: saved?.text ? { name: recoveryName(saved), time: saved.time ?? Date.now() } : null })
}

/** Opens the recovered document, with its unsaved changes. */
export async function recover() {
  try {
    const saved = (await idbGet(RECOVERY_KEY)) as Autosave | undefined
    if (!saved?.text) return
    openAutosave(saved)
    await idbDel(RECOVERY_KEY)
    useView.getState().set({ recovery: null })
  } catch (e) {
    useView.getState().toast(`The recovered design could not be opened. ${(e as Error).message}`, 'error')
  }
}

/** Discards the recovered document. */
export async function discardRecovery() {
  await idbDel(RECOVERY_KEY).catch(() => undefined)
  useView.getState().set({ recovery: null })
}
