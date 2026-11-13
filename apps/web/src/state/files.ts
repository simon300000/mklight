// Documents: New, Samples, Open / Save `.mklight`, Import `.zmx` / `.seq`, and
// autosave to IndexedDB. Uses the File System Access API where the browser has
// it (Chrome) and falls back to a file input and a download (Safari).

import { get as idbGet, set as idbSet } from 'idb-keyval'
import type { Json, JsonObject } from '@/engine/types'
import { readProject, writeProject, projectFromSystem, type Project, SAMPLES, ProjectError } from '@/model/project'
import { newSystem, clone, type System } from '@/model/system'
import { useDocument, type DocumentFile } from './document'
import { useView } from './view'
import { useWorkspace, LAYOUT, type Pane } from './workspace'
import { useImportReport } from './importReport'

const AUTOSAVE_KEY = 'mklight-autosave-v1'

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

/** Writes the panes and the layout's settings into the project's `views`. */
export function viewsFromWorkspace(): Json[] {
  const ws = useWorkspace.getState()
  const v = useView.getState()
  const views: Json[] = ws.panes.map((p) => ({
    key: p.key,
    analysis: p.analysis,
    settings: p.settings,
    ui: { auto: p.auto, settings_visible: p.settingsOpen },
  }))
  views.unshift({
    key: LAYOUT,
    analysis: 'layout',
    settings: ws.layoutSettings,
    ui: { web: { lens_share: v.lensShare, layout_share: v.layoutShare } },
  })
  return views
}

export function applyViews(views: Json[]) {
  const panes: Pane[] = []
  let layoutSettings: JsonObject | null = null
  for (const raw of views) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const v = raw as JsonObject
    const ui = (v.ui as JsonObject) ?? {}
    if (v.key === LAYOUT || v.analysis === 'layout') {
      layoutSettings = (v.settings as JsonObject) ?? null
      const web = ui.web as JsonObject | undefined
      if (web) useView.getState().set({ lensShare: Number(web.lens_share) || 0.36, layoutShare: Number(web.layout_share) || 0.5 })
      continue
    }
    if (typeof v.analysis !== 'string') continue
    panes.push({
      key: String(v.key ?? `pane-${panes.length}`),
      analysis: v.analysis,
      settings: (v.settings as JsonObject) ?? {},
      auto: ui.auto !== false,
      settingsOpen: ui.settings_visible === true,
    })
  }
  const ws = useWorkspace.getState()
  if (layoutSettings) useWorkspace.setState({ layoutSettings: { ...ws.layoutSettings, ...layoutSettings } })
  ws.setPanes(panes)
}

/** Default panes of a new window: spot diagram and ray fan. */
const DEFAULT_PANES: Pane[] = [
  { key: 'pane-spot', analysis: 'spot_diagram', settings: {}, auto: true, settingsOpen: false },
  { key: 'pane-fan', analysis: 'ray_fan', settings: {}, auto: true, settingsOpen: false },
]

function openProject(project: Project, file: DocumentFile | null, opts?: { dirty?: boolean; keepPanes?: boolean }) {
  useDocument.getState().open(project, file, opts)
  useView.getState().set({ selection: Math.min(1, project.system.surfaces.length - 1), welcomeOpen: false })
  if (project.views.length) applyViews(project.views)
  else if (!opts?.keepPanes) useWorkspace.getState().setPanes(DEFAULT_PANES.map((p) => ({ ...p })))
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
      const result = await engine.importPrescription(await file.arrayBuffer(), format)
      const system = result.system as System
      openProject(projectFromSystem(system), { name: file.name, handle: null, imported: true }, { dirty: true })
      useImportReport.getState().show(file.name, result.warnings)
      return
    }
    const project = readProject(await file.text())
    openProject(project, { name: file.name, handle, imported: false })
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

/** Restores the autosaved document; false when there is none. */
export async function restoreAutosave(): Promise<boolean> {
  try {
    const saved = (await idbGet(AUTOSAVE_KEY)) as
      | { text: string; file: { name: string; imported: boolean } | null; handle: FileSystemFileHandle | null; dirty: boolean }
      | undefined
    if (!saved?.text) return false
    const project = readProject(saved.text)
    const file = saved.file ? { ...saved.file, handle: saved.handle ?? null } : null
    openProject(project, file, { dirty: saved.dirty })
    return true
  } catch {
    return false
  }
}
