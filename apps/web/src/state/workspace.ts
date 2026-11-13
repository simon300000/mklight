// Engine snapshots and analysis runs (docs/APP.md "State model", `Workspace`):
// each design value is debounced (150 ms) and frozen into an engine snapshot.
// Analyses run against the snapshot they were submitted with; a newer request
// cancels the older job of the same pane. Results are stale as soon as the
// snapshot or the request differs; panes with automatic update re-run, the
// others wait for Update.

import { create } from 'zustand'
import type {
  AnalysisDescriptor,
  AnalysisJob,
  AnalysisResult,
  EngineClient,
  Json,
  JsonObject,
  ModelSnapshot,
} from '@/engine/types'
import { EngineError } from '@/engine/types'
import type { System } from '@/model/system'
import { backendRequest, usePreferences } from './preferences'

export type RunStatus = 'idle' | 'running' | 'done' | 'failed' | 'cancelled'

export interface PaneRun {
  status: RunStatus
  progress: number
  stage?: string | null
  result?: AnalysisResult
  error?: string
  /** Request identity (analysis + settings) of `result`. */
  resultRequest?: string
}

export interface Pane {
  key: string
  analysis: string
  settings: JsonObject
  auto: boolean
  settingsOpen: boolean
  /** View-only state saved with the pane in `views[].ui` and never sent to
   *  the engine (the image simulation's imported image, `source_image`). */
  ui?: JsonObject
}

export interface Issue {
  message: string
  severity: 'error' | 'warning' | 'info'
  surface: number | null
  /** The engine's stable code (docs/RESULTS.md §6), when it gives one. */
  code?: string
  /** The field the issue concerns, as a system index (0-based; shown from 1). */
  field?: number | null
}

/** `field 2` / `field index 2` in an engine message → 2, the system index as
 *  the engine writes it (its messages count fields from 0). */
export function fieldIndexIn(message: string): number | null {
  const words = message.toLowerCase().split(/[^\p{L}\p{N}]+/u)
  for (let k = 0; k < words.length; k++) {
    if (words[k] !== 'field') continue
    for (const next of words.slice(k + 1, k + 3)) if (/^\d+$/.test(next)) return Number(next)
  }
  return null
}

/** Internal runs that are not panes. */
export const LAYOUT = 'layout'
export const SUMMARY = '__summary'
export const VALIDATE = '__validate'

/** Long analyses start manual (docs/APP.md). */
const MANUAL_ANALYSES = new Set(['stray_light', 'ghost_irradiance', 'through_focus_mtf', 'mtf_vs_field', 'image_simulation', 'optimize', 'tolerance', 'glass_substitution'])

interface WorkspaceState {
  engine: EngineClient | null
  engineError: string | null
  descriptors: AnalysisDescriptor[]
  materials: string[]
  snapshot: ModelSnapshot | null
  /** True while a newer design value waits for its snapshot. */
  snapshotPending: boolean
  /** Issues from the last snapshot attempt (structural errors) and validation. */
  modelIssues: Issue[]
  validationIssues: Issue[]
  panes: Pane[]
  runs: Record<string, PaneRun>
  layoutSettings: JsonObject
  /** A candidate design shown in the layout and every pane instead of the
   *  document's (`Workspace.preview`); any design change ends it. Design jobs
   *  always run on the design's own `snapshot`. */
  preview: { snapshot: ModelSnapshot; label: string } | null

  /** Shows a candidate system (an optimisation or substitution result, a
   *  tolerance build) everywhere, labelled as not applied. */
  setPreview(system: System | null, label?: string): Promise<void>

  setEngine(engine: EngineClient): Promise<void>
  setEngineError(message: string): void
  designChanged(system: System): void
  addPane(analysis: string, opts?: { settings?: JsonObject; key?: string; at?: number }): string
  removePane(key: string): void
  setPanes(panes: Pane[]): void
  updatePane(key: string, patch: Partial<Pane>): void
  setPaneAnalysis(key: string, analysis: string): void
  setLayoutSettings(patch: JsonObject): void
  run(key: string, opts?: { fresh?: boolean }): void
  cancel(key: string): void
  cancelAll(): void
  runAll(opts?: { fresh?: boolean }): void
  refresh(): void
}

const jobs = new Map<string, AnalysisJob>()
/** What each running job computes: its snapshot's hash and its request. */
const jobRequests = new WeakMap<AnalysisJob, string>()
let snapshotTimer: ReturnType<typeof setTimeout> | null = null
let snapshotSeq = 0
/** The latest preview request: an older one that finishes later is dropped. */
let previewSeq = 0
let paneSeq = 0

/** A new pane's settings (`Workspace.startingSettings`): a spot diagram shows
 *  three planes through focus; image simulation starts on a visible patch (10 %
 *  of the field on axis, 256 px, a Siemens star); glass data samples its
 *  dispersion curve for the design's first catalogue glass. */
export function startingSettings(analysis: string, ds: AnalysisDescriptor[], system?: System | null): JsonObject {
  switch (analysis) {
    case 'spot_diagram':
      return { through_focus: { min: null, max: null, steps: 3 } }
    case 'image_simulation': {
      const optics = { ...(((descriptorFor(ds, analysis)?.default_settings ?? {}).optics as JsonObject) ?? {}), field_extent: 0.1 }
      return { size: [256, 256], optics, source: { kind: 'pattern', pattern: 'siemens_star' } }
    }
    case 'glass_info': {
      const glass = system?.surfaces.find((x) => x.material.type === 'catalog')?.material.name
      return { curve_samples: 200, ...(typeof glass === 'string' ? { name: glass } : {}) }
    }
    default:
      return {}
  }
}

export const descriptorFor = (ds: AnalysisDescriptor[], name: string) => ds.find((d) => d.name === name)

const requestKey = (analysis: string, settings: JsonObject) => analysis + ' ' + stableJson(settings)

export function stableJson(v: Json | undefined): string {
  if (v === undefined) return 'null'
  if (Array.isArray(v)) return '[' + v.map(stableJson).join(',') + ']'
  if (v && typeof v === 'object') {
    return '{' + Object.keys(v).sort().map((k) => JSON.stringify(k) + ':' + stableJson((v as JsonObject)[k])).join(',') + '}'
  }
  return JSON.stringify(v)
}

/** Settings of a request: defaults merged with the pane's own. */
export function effectiveSettings(d: AnalysisDescriptor | undefined, own: JsonObject): JsonObject {
  return { ...(d?.default_settings ?? {}), ...own }
}

/** The snapshot analyses show: the preview's, else the design's. */
export function activeSnapshot(ws: { snapshot: ModelSnapshot | null; preview?: { snapshot: ModelSnapshot } | null }): ModelSnapshot | null {
  return ws.preview?.snapshot ?? ws.snapshot
}

export function isStale(
  ws: { snapshot: ModelSnapshot | null; preview?: { snapshot: ModelSnapshot } | null },
  run: PaneRun | undefined,
  analysis: string,
  settings: JsonObject,
): boolean {
  if (!run?.result) return false
  const snap = activeSnapshot(ws)
  if (!snap) return true
  if (run.result.runInfo.snapshot_hash !== snap.hash) return true
  return run.resultRequest !== requestKey(analysis, settings)
}

function errorText(e: unknown): string {
  if (e instanceof EngineError) return e.issues.length ? e.issues.join('\n') : e.message
  if (e instanceof Error) return e.message
  return String(e)
}

function surfaceOf(message: string): number | null {
  const m = /surface\s+(\d+)/i.exec(message)
  return m ? Number(m[1]) : null
}

export const useWorkspace = create<WorkspaceState>()((set, get) => {
  /** Settings sent for a run key. */
  const requestFor = (key: string): { analysis: string; settings: JsonObject } | null => {
    if (key === LAYOUT) return { analysis: 'layout', settings: get().layoutSettings }
    if (key === SUMMARY) return { analysis: 'first_order', settings: {} }
    if (key === VALIDATE) return { analysis: 'validate', settings: {} }
    const p = get().panes.find((x) => x.key === key)
    return p ? { analysis: p.analysis, settings: p.settings } : null
  }

  const patchRun = (key: string, patch: Partial<PaneRun>) =>
    set((s) => ({ runs: { ...s.runs, [key]: { ...(s.runs[key] ?? { status: 'idle', progress: 0 }), ...patch } } }))

  return {
    engine: null,
    engineError: null,
    descriptors: [],
    materials: [],
    snapshot: null,
    snapshotPending: false,
    modelIssues: [],
    validationIssues: [],
    panes: [],
    runs: {},
    layoutSettings: { section: 'yz', rays_per_field: 7, include_3d: false },
    preview: null,

    async setPreview(system, label = '') {
      const seq = ++previewSeq
      const engine = get().engine
      if (!system || !engine) {
        if (get().preview) {
          set({ preview: null })
          get().refresh()
        }
        return
      }
      try {
        const snap = await engine.openSnapshot(system)
        if (seq !== previewSeq) return
        set({ preview: { snapshot: snap, label } })
        get().refresh()
      } catch (e) {
        if (seq !== previewSeq) return
        set({ preview: null })
        throw e
      }
    },

    async setEngine(engine) {
      set({ engine, engineError: null })
      const [descriptors, materials] = await Promise.all([engine.analysisDescriptors(), engine.materialNames()])
      set({ descriptors, materials })
    },

    setEngineError(message) {
      set({ engineError: message })
    },

    designChanged(system) {
      set({ snapshotPending: true })
      if (snapshotTimer) clearTimeout(snapshotTimer)
      const seq = ++snapshotSeq
      snapshotTimer = setTimeout(async () => {
        const engine = get().engine
        if (!engine) return
        try {
          const snap = await engine.openSnapshot(system)
          if (seq !== snapshotSeq) return
          // A design change ends the preview, and one still opening.
          previewSeq++
          set({ snapshot: snap, snapshotPending: false, modelIssues: [], preview: null })
          get().refresh()
        } catch (e) {
          if (seq !== snapshotSeq) return
          const issues = e instanceof EngineError && e.issues.length ? e.issues : [errorText(e)]
          set({
            snapshotPending: false,
            modelIssues: issues.map((m) => ({ message: m, severity: 'error' as const, surface: surfaceOf(m) })),
          })
        }
      }, 150)
    },

    addPane(analysis, opts) {
      const key = opts?.key ?? `pane-${Date.now().toString(36)}-${++paneSeq}`
      const settings = opts?.settings ?? startingSettings(analysis, get().descriptors, get().snapshot?.system as System | undefined)
      const pane: Pane = { key, analysis, settings, auto: !MANUAL_ANALYSES.has(analysis), settingsOpen: false }
      set((s) => {
        const panes = [...s.panes]
        panes.splice(opts?.at ?? panes.length, 0, pane)
        return { panes }
      })
      if (pane.auto) get().run(key)
      return key
    },

    removePane(key) {
      get().cancel(key)
      set((s) => {
        const runs = { ...s.runs }
        delete runs[key]
        return { panes: s.panes.filter((p) => p.key !== key), runs }
      })
    },

    setPanes(panes) {
      for (const k of jobs.keys()) if (k !== LAYOUT && k !== SUMMARY && k !== VALIDATE) get().cancel(k)
      set({ panes, runs: Object.fromEntries(Object.entries(get().runs).filter(([k]) => k.startsWith('__') || k === LAYOUT)) })
      get().refresh()
    },

    updatePane(key, patch) {
      set((s) => ({ panes: s.panes.map((p) => (p.key === key ? { ...p, ...patch } : p)) }))
      if (patch.settings || patch.auto) {
        const p = get().panes.find((x) => x.key === key)
        if (p?.auto) get().run(key)
      }
    },

    setPaneAnalysis(key, analysis) {
      get().cancel(key)
      set((s) => ({
        panes: s.panes.map((p) =>
          p.key === key ? { ...p, analysis, settings: startingSettings(analysis, s.descriptors, s.snapshot?.system as System | undefined), auto: !MANUAL_ANALYSES.has(analysis) } : p,
        ),
        runs: { ...s.runs, [key]: { status: 'idle', progress: 0 } },
      }))
      const p = get().panes.find((x) => x.key === key)
      if (p?.auto) get().run(key)
    },

    setLayoutSettings(patch) {
      set((s) => ({ layoutSettings: { ...s.layoutSettings, ...patch } }))
      get().run(LAYOUT)
    },

    run(key, opts) {
      const { engine } = get()
      const snapshot = activeSnapshot(get())
      const req = requestFor(key)
      if (!engine || !snapshot || !req) return
      jobs.get(key)?.cancel()
      const rk = requestKey(req.analysis, req.settings)
      let job: AnalysisJob
      try {
        const backend = backendRequest(usePreferences.getState().backend, engine.info)
        job = engine.submit(
          { analysis: req.analysis, settings: req.settings, fresh: opts?.fresh || undefined, ...(backend.kind === 'auto' ? {} : { backend }) },
          snapshot,
        )
      } catch (e) {
        patchRun(key, { status: 'failed', error: errorText(e) })
        return
      }
      jobs.set(key, job)
      jobRequests.set(job, snapshot.hash + ' ' + rk)
      patchRun(key, { status: 'running', progress: 0, stage: null, error: undefined })
      job.onProgress((p) => {
        if (jobs.get(key) === job) patchRun(key, { progress: p.progress, stage: p.stage ?? null })
      })
      job.result.then(
        (result) => {
          if (jobs.get(key) !== job) return
          jobs.delete(key)
          patchRun(key, { status: 'done', progress: 1, result, resultRequest: rk, error: undefined })
          if (key === VALIDATE) {
            const issues = (result.json.issues as JsonObject[] | undefined) ?? []
            set({
              validationIssues: issues.map((i) => {
                const message = String(i.message ?? '')
                return {
                  message,
                  severity: (i.severity as Issue['severity']) ?? 'warning',
                  surface: typeof i.surface === 'number' ? i.surface : null,
                  code: typeof i.code === 'string' ? i.code : '',
                  field: typeof i.field === 'number' ? i.field : fieldIndexIn(message),
                }
              }),
            })
          }
        },
        (e) => {
          if (jobs.get(key) !== job) return
          jobs.delete(key)
          const cancelled = e instanceof EngineError && e.kind === 'cancelled'
          patchRun(key, cancelled ? { status: 'cancelled' } : { status: 'failed', error: errorText(e) })
        },
      )
    },

    cancel(key) {
      const job = jobs.get(key)
      if (job) {
        jobs.delete(key)
        job.cancel()
        patchRun(key, { status: 'cancelled' })
      }
    },

    cancelAll() {
      for (const k of [...jobs.keys()]) get().cancel(k)
    },

    runAll(opts) {
      for (const p of get().panes) get().run(p.key, opts)
      get().run(LAYOUT, opts)
    },

    /** Re-runs what the current snapshot made stale: the layout, the summary,
     *  validation and every pane with automatic update. */
    refresh() {
      const s = get()
      if (!activeSnapshot(s)) return
      const keys = [LAYOUT, SUMMARY, VALIDATE, ...s.panes.filter((p) => p.auto).map((p) => p.key)]
      for (const k of keys) {
        const req = requestFor(k)
        if (!req) continue
        const run = s.runs[k]
        const current = run?.result && run.status !== 'failed' && !isStale(s, run, req.analysis, req.settings)
        // A job already computing this snapshot and request keeps running.
        const job = jobs.get(k)
        const computing = job && jobRequests.get(job) === activeSnapshot(s)!.hash + ' ' + requestKey(req.analysis, req.settings)
        if (!current && !computing) s.run(k)
      }
    },
  }
})

/** Issues of the design: structural errors first, then validation. */
export function allIssues(s: Pick<WorkspaceState, 'modelIssues' | 'validationIssues'>): Issue[] {
  return [...s.modelIssues, ...s.validationIssues.filter((i) => i.severity !== 'info')]
}
