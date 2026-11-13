// Simulation state (docs/APP.md "State model"): the project value and its
// undo history. Every edit replaces the system with a new value and records the
// previous project under the action's name, so undo and redo restore whole
// design states. View state lives elsewhere (state/view.ts).

import { create } from 'zustand'
import type { Json } from '@/engine/types'
import { clone, type System } from '@/model/system'
import { projectFromSystem, type Project } from '@/model/project'

export interface UndoEntry {
  name: string
  project: Project
}

/** Where the document came from, for Save. */
export interface DocumentFile {
  /** File name shown in the title (`Double Gauss.mklight`). */
  name: string
  /** File System Access handle (Chrome); null when saved by download. */
  handle: FileSystemFileHandle | null
  /** Imported `.zmx` / `.seq` files are never written: Save asks for a new file. */
  imported: boolean
}

interface DocumentState {
  project: Project
  file: DocumentFile | null
  /** Bumped on every change to the design, for the dirty state and autosave. */
  revision: number
  savedRevision: number
  past: UndoEntry[]
  future: UndoEntry[]

  /** Applies a named, undoable edit to a copy of the system. A change that leaves
   *  the system equal is not recorded. Returns false when nothing changed. */
  edit(actionName: string, change: (s: System) => void | boolean): boolean
  /** Replaces the project's other sections (views are not undoable). */
  setViews(views: Json[]): void
  /** Replaces the whole project as one undoable step (Replace with Sample, Apply…). */
  replace(project: Project, actionName: string): void
  /** Opens a project: a new document with no undo history. */
  open(project: Project, file: DocumentFile | null, opts?: { dirty?: boolean }): void
  markSaved(file: DocumentFile): void
  undo(): void
  redo(): void
}

const MAX_UNDO = 200

export const useDocument = create<DocumentState>()((set, get) => ({
  project: projectFromSystem({} as System),
  file: null,
  revision: 0,
  savedRevision: 0,
  past: [],
  future: [],

  edit(actionName, change) {
    const before = get().project
    const system = clone(before.system)
    const result = change(system)
    if (result === false) return false
    if (JSON.stringify(system) === JSON.stringify(before.system)) return false
    set((s) => ({
      project: { ...before, system },
      past: [...s.past, { name: actionName, project: before }].slice(-MAX_UNDO),
      future: [],
      revision: s.revision + 1,
    }))
    return true
  },

  setViews(views) {
    set((s) => ({ project: { ...s.project, views } }))
  },

  replace(project, actionName) {
    const before = get().project
    set((s) => ({
      project,
      past: [...s.past, { name: actionName, project: before }].slice(-MAX_UNDO),
      future: [],
      revision: s.revision + 1,
    }))
  },

  open(project, file, opts) {
    set((s) => {
      const revision = s.revision + 1
      return { project, file, past: [], future: [], revision, savedRevision: opts?.dirty ? -1 : revision }
    })
  },

  markSaved(file) {
    set((s) => ({ file, savedRevision: s.revision }))
  },

  undo() {
    const { past, project } = get()
    const last = past.at(-1)
    if (!last) return
    set((s) => ({
      // Views (pane arrangement) are not part of the undo history.
      project: { ...last.project, views: project.views },
      past: s.past.slice(0, -1),
      future: [{ name: last.name, project }, ...s.future],
      revision: s.revision + 1,
    }))
  },

  redo() {
    const { future, project } = get()
    const next = future[0]
    if (!next) return
    set((s) => ({
      project: { ...next.project, views: project.views },
      future: s.future.slice(1),
      past: [...s.past, { name: next.name, project }],
      revision: s.revision + 1,
    }))
  },
}))

export const undoName = (s: DocumentState) => s.past.at(-1)?.name ?? null
export const redoName = (s: DocumentState) => s.future[0]?.name ?? null
export const isDirty = (s: DocumentState) => s.revision !== s.savedRevision

/** The document's display name: the file name without extension, else the system name. */
export function documentTitle(s: DocumentState): string {
  if (s.file) return s.file.name.replace(/\.(mklight|zmx|seq)$/i, '')
  return s.project.system?.name || 'Untitled'
}
