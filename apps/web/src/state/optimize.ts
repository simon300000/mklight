// The Optimize panel's session (apps/macos/Sources/MKLightUI/ViewModels/DesignJobs.swift,
// `OptimizeSession`): algorithm, iterations and seed, which variables to vary,
// the live merit evaluation (`merit_evaluation` with exactly `optimize`'s
// settings, 200 ms after the last edit; a newer edit cancels the pending
// request, an edit undone before its answer drops it) and Convert to Custom.
// The run itself goes through `useJobs` (state/jobs.ts); the merit and the
// variables are project sections edited through `useDocument.editProject`.

import { create } from 'zustand'
import type { AnalysisJob, JsonObject, ModelSnapshot } from '@/engine/types'
import { EngineError } from '@/engine/types'
import { isPinned, boundsOf, type CurvatureBoundsUnit } from '@/model/bounds'
import {
  isEflHold,
  meritJobSettings,
  meritSection,
  readEvaluation,
  readMerit,
  specOperands,
  type MeritEvaluation,
  type MeritSettings,
} from '@/model/merit'
import { solveFor, variablesOf, type VariableSpec } from '@/model/params'
import type { Project } from '@/model/project'
import type { System } from '@/model/system'
import { useDocument } from './document'
import { stableJson, useWorkspace } from './workspace'

export const ALGORITHMS: { id: string; title: string }[] = [
  { id: 'lm', title: 'Damped least squares (LM)' },
  { id: 'dogleg', title: 'Trust region (dogleg)' },
  { id: 'de', title: 'Differential evolution' },
  { id: 'cma_es', title: 'CMA-ES' },
  { id: 'nelder_mead', title: 'Nelder–Mead' },
  { id: 'hammer', title: 'Basin hopping' },
]
export const STOCHASTIC = new Set(['de', 'cma_es', 'hammer'])
/** The engine's default iterations per algorithm (RESULTS.md `max_iterations`). */
export const DEFAULT_ITERATIONS: Record<string, number> = { lm: 200, dogleg: 200, de: 2000, cma_es: 100, nelder_mead: 200, hammer: 20 }

export const AUTOMATIC_VARIABLES = 'Automatic: every curved surface and the back focus'

const UNIT_KEY = 'mklight.curvatureBoundsUnit'
function storedUnit(): CurvatureBoundsUnit {
  try {
    return localStorage.getItem(UNIT_KEY) === 'curvature' ? 'curvature' : 'radius'
  } catch {
    return 'radius'
  }
}

interface OptimizeState {
  algorithm: string
  /** null: the algorithm's default. */
  maxIterations: number | null
  seed: number
  /** Vary the design's variables (else the engine's automatic set). */
  useDesignVariables: boolean
  curvatureUnit: CurvatureBoundsUnit
  /** The merit and every operand at the current design. */
  evaluation: MeritEvaluation | null
  /** True from an edit until its evaluation answers (the readout dims). */
  evaluating: boolean
  evaluationError: string | null
  converting: boolean

  set(p: Partial<Pick<OptimizeState, 'algorithm' | 'maxIterations' | 'seed' | 'useDesignVariables'>>): void
  setCurvatureUnit(u: CurvatureBoundsUnit): void
  /** Re-evaluates the merit after a short pause, unless it was evaluated already. */
  scheduleEvaluation(delay?: number): void
  cancelEvaluation(): void
  /** Replaces the default merit by the operands the engine generates for it. */
  convertToCustom(): Promise<void>
}

/** The project's merit function. */
export const meritOf = (p: Project): MeritSettings => readMerit(p.merit)

/** Applies an edit to the merit function as one undoable step. */
export function editMerit(actionName: string, change: (m: MeritSettings) => MeritSettings | void): boolean {
  return useDocument.getState().editProject(actionName, (p) => {
    const m = readMerit(p.merit)
    const next = change(m) ?? m
    p.merit = meritSection(next)
  })
}

/** Replaces the design's variables as one undoable step. */
export function editVariables(actionName: string, change: (v: VariableSpec[]) => VariableSpec[]): boolean {
  return useDocument.getState().editProject(actionName, (p) => {
    p.variables = change(variablesOf(p))
  })
}

export const isSolved = (s: System, v: VariableSpec) => solveFor(s, v.param) != null

/** Whether the design's variables are used (there must be some). */
export function usesDesignVariables(project: Project, useDesign: boolean): boolean {
  return useDesign && variablesOf(project).length > 0
}

/** The `variables` setting: the design's variables without the pinned ones and
 *  those a solve controls, or null for the engine's automatic set. */
export function variableSettings(project: Project, useDesign: boolean): JsonObject[] | null {
  if (!usesDesignVariables(project, useDesign)) return null
  const free = variablesOf(project).filter((v) => !isPinned(boundsOf(v)) && !isSolved(project.system, v))
  return free.length ? free.map((v) => ({ ...v })) : null
}

/** Why the run cannot start with these variables, if it cannot. */
export function variableProblem(project: Project, useDesign: boolean): string | null {
  if (!usesDesignVariables(project, useDesign)) return null
  const vars = variablesOf(project)
  const solved = (v: VariableSpec) => isSolved(project.system, v)
  if (vars.every((v) => isPinned(boundsOf(v)) || solved(v))) {
    if (!vars.some(solved)) return 'Every variable is pinned. Unpin one, or vary the automatic set.'
    return 'Every variable is pinned or controlled by a solve. Add a free variable, or vary the automatic set.'
  }
  return null
}

/** `merit_evaluation`'s settings: `optimize`'s merit, variables and algorithm. */
export function evaluationSettings(project: Project, s: Pick<OptimizeState, 'algorithm' | 'useDesignVariables'>, merit?: MeritSettings): JsonObject {
  const out: JsonObject = { ...meritJobSettings(merit ?? meritOf(project)), algorithm: s.algorithm }
  const v = variableSettings(project, s.useDesignVariables)
  if (v) out.variables = v
  return out
}

/** `optimize`'s settings. */
export function optimizeSettings(project: Project, s: Pick<OptimizeState, 'algorithm' | 'useDesignVariables' | 'maxIterations' | 'seed'>): JsonObject {
  const out = evaluationSettings(project, s)
  out.seed = s.seed
  if (s.maxIterations != null) out.max_iterations = s.maxIterations
  return out
}

function errorText(e: unknown): string {
  if (e instanceof EngineError) return e.issues.length ? e.issues.join('\n') : e.message
  return e instanceof Error ? e.message : String(e)
}

let timer: ReturnType<typeof setTimeout> | null = null
let job: AnalysisJob | null = null
let pendingKey: string | null = null
let evaluatedKey: string | null = null

function current(): { snap: ModelSnapshot; project: Project } | null {
  const ws = useWorkspace.getState()
  if (!ws.engine || !ws.snapshot) return null
  return { snap: ws.snapshot, project: useDocument.getState().project }
}

export const useOptimize = create<OptimizeState>()((set, get) => ({
  algorithm: 'lm',
  maxIterations: null,
  seed: 0,
  useDesignVariables: true,
  curvatureUnit: storedUnit(),
  evaluation: null,
  evaluating: false,
  evaluationError: null,
  converting: false,

  set(p) {
    set(p)
  },

  setCurvatureUnit(u) {
    try {
      localStorage.setItem(UNIT_KEY, u)
    } catch {
      // Private mode: the choice lasts for the session.
    }
    set({ curvatureUnit: u })
  },

  scheduleEvaluation(delay = 200) {
    const c = current()
    if (!c) return
    const settings = evaluationSettings(c.project, get())
    const key = c.snap.hash + '|' + stableJson(settings)
    if (key === pendingKey) return
    if (timer) clearTimeout(timer)
    timer = null
    if (key === evaluatedKey) {
      // Back to what is shown (an edit undone): drop the newer request.
      job?.cancel()
      job = null
      pendingKey = null
      set({ evaluating: false })
      return
    }
    pendingKey = key
    set({ evaluating: true })
    const snap = c.snap
    timer = setTimeout(() => {
      timer = null
      const engine = useWorkspace.getState().engine
      if (!engine) return
      job?.cancel()
      let j: AnalysisJob
      try {
        j = engine.submit({ analysis: 'merit_evaluation', settings }, snap)
      } catch (e) {
        evaluatedKey = key
        pendingKey = null
        set({ evaluating: false, evaluationError: errorText(e) })
        return
      }
      job = j
      j.result.then(
        (r) => {
          if (job !== j) return
          job = null
          const e = readEvaluation(r.json)
          evaluatedKey = key
          pendingKey = null
          set({ evaluation: e, evaluating: false, evaluationError: e ? null : 'The engine returned no merit.' })
        },
        (e) => {
          if (job !== j) return
          job = null
          if (e instanceof EngineError && e.kind === 'cancelled') return
          evaluatedKey = key
          pendingKey = null
          set({ evaluating: false, evaluationError: errorText(e) })
        },
      )
    }, delay)
  },

  cancelEvaluation() {
    if (timer) clearTimeout(timer)
    timer = null
    job?.cancel()
    job = null
    pendingKey = null
    set({ evaluating: false })
  },

  async convertToCustom() {
    const c = current()
    if (!c) return
    const merit = { ...meritOf(c.project), mode: 'default' as const }
    const settings = evaluationSettings(c.project, get(), merit)
    const key = c.snap.hash + '|' + stableJson(settings)
    let e = key === evaluatedKey ? get().evaluation : null
    if (!e) {
      set({ converting: true })
      try {
        const engine = useWorkspace.getState().engine!
        const r = await engine.submit({ analysis: 'merit_evaluation', settings }, c.snap).result
        e = readEvaluation(r.json)
      } catch (err) {
        set({ converting: false, evaluationError: errorText(err) })
        return
      }
      set({ converting: false })
    }
    if (!e) {
      set({ evaluationError: 'The engine returned no merit.' })
      return
    }
    const spec = specOperands(e)
    // The EFL hold stays with the "Hold the focal length" switch, which adds it again.
    const operands = spec.operands.filter((op) => !isEflHold(op) || !merit.holdEfl)
    if (!operands.length) {
      set({ evaluationError: 'The engine generated no operands for this design.' })
      return
    }
    editMerit('Convert Merit to Custom', (m) => ({ ...m, operands, specOptions: spec.options, mode: 'custom' }))
  },
}))
