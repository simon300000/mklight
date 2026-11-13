// Design jobs (docs/APP.md "Design jobs", apps/macos/Sources/MKLightUI/ViewModels/DesignJobs.swift):
// Optimize, Glass Substitution, Tolerance and Apply Solves submit the engine's
// `optimize`, `glass_substitution`, `tolerance` and `solves` analyses on the
// design's own snapshot, follow their progress and live detail, and apply a
// result to the document as one undoable step. A job outlives its panel and
// keeps running when the inspector shows another page; several jobs can run at
// once. Each panel keeps its own settings and views; this store is the shared
// run life cycle every panel builds its action bar from (`JobActions`).

import { create } from 'zustand'
import type { AnalysisJob, AnalysisResult, Json, JsonObject } from '@/engine/types'
import { EngineError } from '@/engine/types'
import { useWorkspace } from './workspace'

export type JobKind = 'optimize' | 'glass' | 'tolerance' | 'solves'

export const JOB_TITLES: Record<JobKind, string> = {
  optimize: 'Optimize',
  glass: 'Glass Substitution',
  tolerance: 'Tolerance',
  solves: 'Apply Solves',
}

/** The running verb for the status capsule ("Optimizing · 479"). */
export const JOB_VERBS: Record<JobKind, string> = {
  optimize: 'Optimizing',
  glass: 'Substituting Glasses',
  tolerance: 'Tolerancing',
  solves: 'Solving',
}

export const JOB_ANALYSES: Record<JobKind, string> = {
  optimize: 'optimize',
  glass: 'glass_substitution',
  tolerance: 'tolerance',
  solves: 'solves',
}

/** setup → running → review (finished, not yet applied) → done (applied); or failed. */
export type JobPhase = 'setup' | 'running' | 'review' | 'done' | 'failed'

export interface JobState {
  phase: JobPhase
  progress: number
  stage: string | null
  /** The engine's live progress detail (iteration, merit, trials, …). */
  detail: Json | null
  /** Every progress detail so far, for sparklines. */
  history: Json[]
  result: AnalysisResult | null
  error: string | null
  /** The settings the last run was submitted with (Try Again resubmits them). */
  settings: JsonObject | null
  /** Snapshot hash the run was submitted on. */
  snapshotHash: string | null
  startedAt: number | null
  /** True after Stop: the engine keeps the best result so far. */
  stopped: boolean
}

const initial = (): JobState => ({
  phase: 'setup',
  progress: 0,
  stage: null,
  detail: null,
  history: [],
  result: null,
  error: null,
  settings: null,
  snapshotHash: null,
  startedAt: null,
  stopped: false,
})

interface JobsStore {
  jobs: Record<JobKind, JobState>
  /** Submits the job's analysis with `settings` on the design's snapshot. */
  start(kind: JobKind, settings: JsonObject, opts?: { fresh?: boolean }): void
  /** Try Again: the same settings, recomputed (`fresh: true`). */
  retry(kind: JobKind): void
  /** Cancel discards the run. */
  cancel(kind: JobKind): void
  /** Stop keeps the best result so far (stoppable analyses). */
  stop(kind: JobKind): void
  /** Discard a finished result; the design stays as it is. */
  discard(kind: JobKind): void
  /** Back from a failure to the settings. */
  back(kind: JobKind): void
  /** Marks the result applied (the panel applies it to the document itself). */
  applied(kind: JobKind): void
  /** Done: back to the job's starting page. */
  done(kind: JobKind): void
  patch(kind: JobKind, p: Partial<JobState>): void
}

const handles = new Map<JobKind, AnalysisJob>()

function errorText(e: unknown): string {
  if (e instanceof EngineError) return e.issues.length ? e.issues.join('\n') : e.message
  return e instanceof Error ? e.message : String(e)
}

export const useJobs = create<JobsStore>()((set, get) => {
  const patch = (kind: JobKind, p: Partial<JobState>) =>
    set((s) => ({ jobs: { ...s.jobs, [kind]: { ...s.jobs[kind], ...p } } }))

  return {
    jobs: { optimize: initial(), glass: initial(), tolerance: initial(), solves: initial() },
    patch,

    start(kind, settings, opts) {
      const ws = useWorkspace.getState()
      const engine = ws.engine
      const snap = ws.snapshot
      if (!engine || !snap) {
        patch(kind, { phase: 'failed', error: ws.modelIssues.length ? 'The design has errors: ' + ws.modelIssues.map((i) => i.message).join('; ') : 'The engine is not ready.' })
        return
      }
      handles.get(kind)?.cancel()
      let job: AnalysisJob
      try {
        job = engine.submit({ analysis: JOB_ANALYSES[kind], settings, fresh: opts?.fresh || undefined }, snap)
      } catch (e) {
        patch(kind, { phase: 'failed', error: errorText(e), settings })
        return
      }
      handles.set(kind, job)
      patch(kind, {
        ...initial(),
        phase: 'running',
        settings,
        snapshotHash: snap.hash,
        startedAt: Date.now(),
      })
      job.onProgress((p) => {
        if (handles.get(kind) !== job) return
        const cur = get().jobs[kind]
        patch(kind, {
          progress: p.progress,
          stage: p.stage ?? cur.stage,
          detail: p.detail ?? cur.detail,
          history: p.detail != null ? [...cur.history, p.detail] : cur.history,
        })
      })
      job.result.then(
        (result) => {
          if (handles.get(kind) !== job) return
          handles.delete(kind)
          patch(kind, { phase: 'review', progress: 1, result, error: null })
        },
        (e) => {
          if (handles.get(kind) !== job) return
          handles.delete(kind)
          if (e instanceof EngineError && e.kind === 'cancelled') patch(kind, { phase: 'setup', progress: 0 })
          else patch(kind, { phase: 'failed', error: errorText(e) })
        },
      )
    },

    retry(kind) {
      const s = get().jobs[kind].settings
      if (s) get().start(kind, s, { fresh: true })
    },

    cancel(kind) {
      const job = handles.get(kind)
      handles.delete(kind)
      job?.cancel()
      patch(kind, { phase: 'setup', progress: 0, stage: null, detail: null })
      void useWorkspace.getState().setPreview(null)
    },

    stop(kind) {
      handles.get(kind)?.stop()
      patch(kind, { stopped: true, stage: 'Stopping' })
    },

    discard(kind) {
      patch(kind, { phase: 'setup', result: null })
      void useWorkspace.getState().setPreview(null)
    },

    back(kind) {
      patch(kind, { phase: 'setup', error: null })
    },

    applied(kind) {
      patch(kind, { phase: 'done' })
    },

    done(kind) {
      patch(kind, { ...initial(), settings: get().jobs[kind].settings })
    },
  }
})

export const isRunning = (j: JobState) => j.phase === 'running'
