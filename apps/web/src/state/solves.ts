// Apply Solves (SolvesSession in apps/macos/Sources/MKLightUI/ViewModels/DesignJobs.swift):
// a dry run (the engine's `solves` analysis, run as the `useJobs` kind 'solves')
// shown as now → solved with the residual of each solve, then one undoable
// commit of the solved design through `engine.applySolves`.

import { create } from 'zustand'
import type { Json, JsonObject } from '@/engine/types'
import { nf } from '@/model/format'
import { paramTitle, type ParamPath } from '@/model/params'
import { radiusFromCurvature, type System } from '@/model/system'
import { useDocument } from './document'
import { useJobs } from './jobs'
import { useWorkspace } from './workspace'

export const SOLVE_KIND_TITLES: Record<string, string> = {
  marginal_ray_height: 'Marginal ray height',
  marginal_ray_angle: 'Marginal ray angle',
  chief_ray_height: 'Chief ray height',
  chief_ray_angle: 'Chief ray angle',
  f_number: 'F-number',
  pickup: 'Pickup',
  position: 'Position',
  center_of_curvature: 'Centre of curvature',
  edge_thickness: 'Edge thickness',
}

export interface SolveRow {
  id: number
  /** `T3`, `R5`, … */
  target: string
  kind: string
  /** Now (radii for curvatures); null when the engine did not report it. */
  previous: number | null
  /** Solved. */
  value: number
  residual: number
  method: string
  changed: boolean
}

const isCurvature = (p: ParamPath | null) =>
  !!p && (p.param === 'curvature' || (p.param === 'shape_param' && (p.parameter === 'curvature_x' || p.parameter === 'curvature_y')))

/** The rows of a `solves` report (dry run or applied), curvatures shown as radii. */
export function solveRows(report: JsonObject | null | undefined, system: System | null): SolveRow[] {
  const list = Array.isArray(report?.outcomes) ? report.outcomes : []
  return list.map((x: Json, k): SolveRow => {
    const o = x && typeof x === 'object' && !Array.isArray(x) ? x : {}
    const t = o.target && typeof o.target === 'object' && !Array.isArray(o.target) && typeof o.target.param === 'string' ? (o.target as ParamPath) : null
    const kind = typeof o.kind === 'string' ? o.kind : ''
    let previous: number | null = o.previous == null ? null : nf(o.previous)
    let value = nf(o.value)
    if (isCurvature(t)) {
      previous = previous == null ? null : radiusFromCurvature(previous)
      value = radiusFromCurvature(value)
    }
    const changed = previous == null ? true : !(previous === value || (Number.isNaN(previous) && Number.isNaN(value)))
    return {
      id: typeof o.index === 'number' ? o.index : k,
      target: t ? (system ? paramTitle(system, t) : t.param) : '?',
      kind: SOLVE_KIND_TITLES[kind] ?? kind.replace(/_/g, ' '),
      previous,
      value,
      residual: nf(o.residual),
      method: typeof o.method === 'string' ? o.method : '',
      changed,
    }
  })
}

export function solveWarnings(report: JsonObject | null | undefined): string[] {
  return Array.isArray(report?.warnings) ? report.warnings.filter((w): w is string => typeof w === 'string') : []
}

/** Method titles, as the report shows them. */
export const METHOD_TITLES: Record<string, string> = {
  unchanged: 'unchanged',
  closed_form: 'closed form',
  fixed_point: 'fixed point',
  root_finding: 'root finding',
}

interface SolvesState {
  /** The report of the applied commit (shown after Apply). */
  appliedReport: JsonObject | null
  committing: boolean
  /** Runs the dry run on the design's snapshot. */
  dryRun(fresh?: boolean): void
  /** Solves in the engine and replaces the design as one undoable step. */
  commit(): Promise<void>
  /** Done: back to a new dry run. */
  done(): void
}

export const useSolves = create<SolvesState>()((set, get) => ({
  appliedReport: null,
  committing: false,

  dryRun(fresh) {
    set({ appliedReport: null })
    useJobs.getState().start('solves', {}, { fresh })
  },

  async commit() {
    const ws = useWorkspace.getState()
    const snap = ws.snapshot
    if (!ws.engine || !snap || get().committing) return
    set({ committing: true })
    try {
      const applied = await ws.engine.applySolves({}, snap)
      const doc = useDocument.getState()
      doc.replace({ ...doc.project, system: applied.snapshot.system as System }, 'Apply Solves')
      set({ appliedReport: applied.report, committing: false })
      useJobs.getState().applied('solves')
    } catch (e) {
      set({ committing: false })
      useJobs.getState().patch('solves', { phase: 'failed', error: e instanceof Error ? e.message : String(e) })
    }
  },

  done() {
    useJobs.getState().done('solves')
    get().dryRun()
  },
}))
