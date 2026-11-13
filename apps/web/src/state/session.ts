// Starts the engine and wires the document to the workspace: every design
// value goes to the engine as a snapshot, and every change is autosaved.

import { loadEngine } from '@/engine'
import { curvatureOf } from '@/model/system'
import { useDocument } from './document'
import { newDocument, openSample, restoreAutosave, scheduleAutosave } from './files'
import { useView } from './view'
import { useWorkspace } from './workspace'

let started = false

/** A small inspection hook for tests and debugging (`window.mklight`). */
function exposeHooks() {
  ;(window as unknown as { mklight: object }).mklight = {
    document: useDocument,
    workspace: useWorkspace,
    view: useView,
    radius(i: number) {
      const c = curvatureOf(useDocument.getState().project.system.surfaces[i].shape) ?? 0
      return c === 0 ? Infinity : 1 / c
    },
  }
}

export async function startSession() {
  if (started) return
  started = true
  exposeHooks()

  // The design goes to the engine whenever it changes.
  useDocument.subscribe((s, prev) => {
    if (s.project.system !== prev.project.system && s.project.system?.surfaces) {
      useWorkspace.getState().designChanged(s.project.system)
      const v = useView.getState()
      const n = s.project.system.surfaces.length
      if (v.selection >= n) v.select(n - 1)
    }
    if (s.revision !== prev.revision || s.savedRevision !== prev.savedRevision) scheduleAutosave()
  })
  // The arrangement of panes is saved with the document (not undoable).
  useWorkspace.subscribe((s, prev) => {
    if (s.panes !== prev.panes || s.layoutSettings !== prev.layoutSettings) scheduleAutosave()
  })

  const params = new URLSearchParams(location.search)
  const sample = params.get('sample')
  const fresh = params.has('fresh')

  try {
    const engine = await loadEngine()
    await useWorkspace.getState().setEngine(engine)
  } catch (e) {
    useWorkspace.getState().setEngineError((e as Error).message)
  }

  if (sample) openSample(sample)
  else if (fresh || !(await restoreAutosave())) {
    newDocument()
    if (!fresh) useView.getState().set({ welcomeOpen: true })
  }
  const sys = useDocument.getState().project.system
  if (sys?.surfaces) useWorkspace.getState().designChanged(sys)
}
