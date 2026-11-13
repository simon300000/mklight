// The Tolerance panel's session (apps/macos/Sources/MKLightUI/ViewModels/ToleranceSession.swift):
// the design's tolerance set being edited (a working copy of the project's
// `tolerances` section whose rows keep their identity across undo), the run
// (its rows, system and settings, for labels and previews), previews of
// toleranced builds rebuilt by the engine (`preview_case`), and the context
// rows expand in: the clear apertures the layout resolved and the Abbe
// numbers of catalogue glasses (`glass_info`). The run's life cycle itself is
// the shared `useJobs` store (kind 'tolerance').

import { useEffect, useMemo } from 'react'
import { create } from 'zustand'
import type { AnalysisJob, JsonObject, ModelSnapshot } from '@/engine/types'
import { EngineError } from '@/engine/types'
import { nf } from '@/model/format'
import type { System } from '@/model/system'
import {
  cloneSettings,
  jobSettings,
  keepingIds,
  makeContext,
  previewTitle,
  readMeritInfo,
  readPreview,
  readSettings,
  samePreviewCase,
  settingsKey,
  settingsSection,
  type CriterionDisplay,
  type MeritInfo,
  type PreviewBuild,
  type PreviewCase,
  type ToleranceContext,
  type ToleranceResult,
  type ToleranceRow,
  type ToleranceSettings,
  criterionDisplay,
} from '@/model/tolerance'
import { useDocument } from './document'
import { useJobs } from './jobs'
import { LAYOUT, useWorkspace } from './workspace'

interface ToleranceStore {
  settings: ToleranceSettings | null
  selection: string[]
  collapsed: string[]
  /** Clear semi-diameters by surface id from the design's last layout. */
  semiDiameters: Record<number, number>
  /** V_d by glass key (`catalog/name/dn/dv`); null while asked. */
  abbe: Record<string, number | null>
  // The run
  runRows: ToleranceRow[]
  runSystem: System | null
  runSettings: JsonObject | null
  runSnapshot: ModelSnapshot | null
  runDisplay: CriterionDisplay
  monteCarloStartedAt: number | null
  lastSecondsPerTrial: number | null
  // Previews
  previewed: PreviewCase | null
  preview: PreviewBuild | null
  previewError: string | null
}

export const useTolerance = create<ToleranceStore>()(() => ({
  settings: null,
  selection: [],
  collapsed: [],
  semiDiameters: {},
  abbe: {},
  runRows: [],
  runSystem: null,
  runSettings: null,
  runSnapshot: null,
  runDisplay: 'spot',
  monteCarloStartedAt: null,
  lastSecondsPerTrial: null,
  previewed: null,
  preview: null,
  previewError: null,
}))

const set = (p: Partial<ToleranceStore>) => useTolerance.setState(p)

// MARK: The set

/** The set the document holds, with the session's row identities. */
export function currentSettings(): ToleranceSettings {
  const { project } = useDocument.getState()
  const doc = readSettings(project.tolerances, project.system)
  const cur = useTolerance.getState().settings
  if (cur && settingsKey(cur) === settingsKey(doc)) return cur
  const next = cur ? keepingIds(doc, cur) : doc
  set({ settings: next })
  return next
}

/** The set, following the document (open, undo, redo, design changes). */
export function useToleranceSettings(): ToleranceSettings {
  const tolerances = useDocument((s) => s.project.tolerances)
  const system = useDocument((s) => s.project.system)
  const cur = useTolerance((s) => s.settings)
  const doc = useMemo(() => readSettings(tolerances, system), [tolerances, system])
  const aligned = useMemo(() => {
    if (cur && settingsKey(cur) === settingsKey(doc)) return cur
    return cur ? keepingIds(doc, cur) : doc
  }, [doc, cur])
  useEffect(() => {
    if (aligned !== cur) set({ settings: aligned })
  }, [aligned, cur])
  return aligned
}

export const meritInfo = (): MeritInfo => readMeritInfo(useDocument.getState().project.merit)
export const useMeritInfo = (): MeritInfo => {
  const merit = useDocument((s) => s.project.merit)
  return useMemo(() => readMeritInfo(merit), [merit])
}

/** Applies an edit as one undoable step, saved in the document. */
export function editTolerances(actionName: string, change: (t: ToleranceSettings) => void): boolean {
  const cur = currentSettings()
  const next = cloneSettings(cur)
  change(next)
  if (settingsKey(next) === settingsKey(cur)) return false
  const section = settingsSection(next, currentContext(next), meritInfo())
  set({ settings: next })
  return useDocument.getState().editProject(actionName, (p) => {
    p.tolerances = section
  })
}

// MARK: Context

const glassKey = (m: JsonObject) => {
  const o = (m.offset as JsonObject | undefined) ?? {}
  return `${String(m.catalog ?? '')}/${String(m.name ?? '')}/${nf(o.dn_d ?? 0)}/${nf(o.dv_d ?? 0)}`
}

function contextOf(system: System, semi: Record<number, number>, abbe: Record<string, number | null>, t: ToleranceSettings | null): ToleranceContext {
  const abbeNumbers: Record<number, number> = {}
  for (const s of system.surfaces ?? []) {
    if (s.material?.type !== 'catalog') continue
    const v = abbe[glassKey(s.material)]
    if (typeof v === 'number') abbeNumbers[s.id] = v
  }
  return makeContext(system, { semiDiameters: semi, abbeNumbers, testWavelengthUm: t?.testWavelengthUm })
}

export function currentContext(t: ToleranceSettings | null = useTolerance.getState().settings): ToleranceContext {
  const { semiDiameters, abbe } = useTolerance.getState()
  return contextOf(useDocument.getState().project.system, semiDiameters, abbe, t)
}

/** The context rows expand in, following the design, its layout and glass data. */
export function useToleranceContext(t: ToleranceSettings): ToleranceContext {
  const system = useDocument((s) => s.project.system)
  const semi = useTolerance((s) => s.semiDiameters)
  const abbe = useTolerance((s) => s.abbe)
  return useMemo(() => contextOf(system, semi, abbe, t), [system, semi, abbe, t])
}

/** Keeps the clear apertures of the design's layout (not of a preview) and
 *  asks the engine for the V_d of every catalogue glass once. */
export function useToleranceContextSources() {
  const layout = useWorkspace((s) => s.runs[LAYOUT]?.result)
  const snapHash = useWorkspace((s) => s.snapshot?.hash)
  const engine = useWorkspace((s) => s.engine)
  const snapshot = useWorkspace((s) => s.snapshot)
  const system = useDocument((s) => s.project.system)

  useEffect(() => {
    if (!layout || layout.runInfo.snapshot_hash !== snapHash) return
    const out: Record<number, number> = {}
    for (const s of (layout.json.surfaces as JsonObject[] | undefined) ?? []) {
      const v = nf(s.semi_diameter)
      if (typeof s.id === 'number' && Number.isFinite(v) && v > 0) out[s.id] = v
    }
    const cur = useTolerance.getState().semiDiameters
    if (JSON.stringify(cur) !== JSON.stringify(out)) set({ semiDiameters: out })
  }, [layout, snapHash])

  useEffect(() => {
    if (!engine || !snapshot) return
    const abbe = useTolerance.getState().abbe
    const todo = new Map<string, JsonObject>()
    for (const s of system.surfaces ?? []) {
      if (s.material?.type !== 'catalog') continue
      const k = glassKey(s.material)
      if (!(k in abbe)) todo.set(k, s.material)
    }
    if (!todo.size) return
    set({ abbe: { ...abbe, ...Object.fromEntries([...todo.keys()].map((k) => [k, null])) } })
    for (const [k, m] of todo) {
      const o = (m.offset as JsonObject | undefined) ?? {}
      let job: AnalysisJob
      try {
        job = engine.submit(
          { analysis: 'glass_info', settings: { name: String(m.name ?? ''), catalog: (m.catalog as string | null) ?? null, offset: { dn_d: nf(o.dn_d ?? 0), dv_d: nf(o.dv_d ?? 0) } } },
          snapshot,
        )
      } catch {
        continue
      }
      job.result.then(
        (r) => {
          const vd = nf(r.json.vd)
          if (Number.isFinite(vd)) set({ abbe: { ...useTolerance.getState().abbe, [k]: vd } })
        },
        () => {
          // Unknown glass: the row says the Abbe number is not known.
        },
      )
    }
  }, [engine, snapshot, system])
}

// MARK: The run

/** Starts the run on the design (`fresh`: recompute instead of reusing). */
export function startTolerance(fresh = false) {
  const t = currentSettings()
  const ctx = currentContext(t)
  const merit = meritInfo()
  const { settings, rows } = jobSettings(t, ctx, merit)
  endTolerancePreview()
  set({
    runRows: rows,
    runSystem: useDocument.getState().project.system,
    runSettings: settings,
    runSnapshot: useWorkspace.getState().snapshot,
    runDisplay: criterionDisplay(t.criterion, merit),
    monteCarloStartedAt: null,
  })
  useJobs.getState().start('tolerance', settings, { fresh })
}

/** Try Again / Run again: the same settings, recomputed. */
export function retryTolerance() {
  endTolerancePreview()
  set({ monteCarloStartedAt: null, runSnapshot: useWorkspace.getState().snapshot, runSystem: useDocument.getState().project.system })
  useJobs.getState().retry('tolerance')
}

export function noteMonteCarloStarted() {
  if (useTolerance.getState().monteCarloStartedAt == null) set({ monteCarloStartedAt: Date.now() })
}

export function noteSecondsPerTrial(s: number | null) {
  if (s != null && s > 0 && s !== useTolerance.getState().lastSecondsPerTrial) set({ lastSecondsPerTrial: s })
}

// MARK: Previews

let previewJob: AnalysisJob | null = null

/** Rebuilds a build of the last run in the engine (`preview_case`, with the
 *  run's own settings and system) and previews it everywhere, read-only. */
export async function previewTolerance(c: PreviewCase | null, result: ToleranceResult | null) {
  previewJob?.cancel()
  previewJob = null
  const { runSettings, runSnapshot, runSystem, runRows } = useTolerance.getState()
  const ws = useWorkspace.getState()
  if (!c || !runSettings || !ws.engine) {
    endTolerancePreview()
    return
  }
  set({ previewed: c, preview: null, previewError: null })
  try {
    const snap = runSnapshot ?? (runSystem ? await ws.engine.openSnapshot(runSystem) : ws.snapshot)
    if (!snap || !samePreviewCase(useTolerance.getState().previewed, c)) return
    const job = ws.engine.submit({ analysis: 'tolerance', settings: { ...runSettings, preview_case: { ...c }, record_trials: 0 } }, snap)
    previewJob = job
    const out = await job.result
    if (!samePreviewCase(useTolerance.getState().previewed, c)) return
    const p = readPreview(out.json.preview)
    if (!p) {
      set({ previewError: 'The engine could not rebuild this build: no ray reached the image.' })
      void useWorkspace.getState().setPreview(null)
      return
    }
    set({ preview: p })
    await useWorkspace.getState().setPreview(p.system, previewTitle(c, runRows, runSystem, result))
  } catch (e) {
    if (e instanceof EngineError && e.kind === 'cancelled') return
    if (!samePreviewCase(useTolerance.getState().previewed, c)) return
    set({ previewError: e instanceof Error ? e.message : String(e) })
  }
}

/** Ends any preview of a toleranced build. */
export function endTolerancePreview() {
  previewJob?.cancel()
  previewJob = null
  const was = useTolerance.getState().previewed != null
  set({ previewed: null, preview: null, previewError: null })
  const label = useWorkspace.getState().preview?.label ?? ''
  if (was || label.startsWith('Tolerance')) void useWorkspace.getState().setPreview(null)
}
