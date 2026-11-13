// The Glass Substitution panel's own state (GlassSubstitutionSession in
// apps/macos/Sources/MKLightUI/ViewModels/GlassSubstitution.swift): which glasses
// may change and from which candidates, the search settings, the live trace of
// the best merit, the previewed combination, and the design's glass map. The run
// life cycle is the shared `useJobs` kind 'glass'.

import { create } from 'zustand'
import type { AnalysisResult } from '@/engine/types'
import {
  defaultCandidates,
  glassSlots,
  glassSubstitutionSettings,
  parseGlassProgress,
  readSubstitution,
  startGlasses,
  type GlassAlgorithm,
  type GlassCandidates,
  type GlassCombination,
  type GlassProgress,
  type GlassSearchSettings,
} from '@/model/glass'
import { variablesOf } from '@/model/params'
import type { System } from '@/model/system'
import { useDocument } from './document'
import { useJobs } from './jobs'
import { useWorkspace } from './workspace'

interface GlassSubstitutionState extends GlassSearchSettings {
  /** The best merit after each combination of the running search. */
  trace: number[]
  progress: GlassProgress | null
  /** The previewed combination (its rank). */
  selectedRank: number | null
  /** Why the previewed combination could not be shown, if it could not. */
  previewError: string | null
  /** The design's glasses when the run started, for the comparison. */
  startGlasses: Record<number, string>
  /** Glass gaps the selection was configured for. */
  configuredFor: number[]
  /** The design's glass map (by snapshot hash), for the result's map. */
  map: { hash: string; result: AnalysisResult } | null
  mapPlane: 'nd' | 'pgf'

  set(p: Partial<GlassSearchSettings & { mapPlane: 'nd' | 'pgf' }>): void
  setCandidates(c: GlassCandidates): void
  setOverride(id: number, c: GlassCandidates | null): void
  toggle(id: number, on: boolean): void
  /** Ticks every glass of a design the first time (or when its glass gaps changed). */
  configure(system: System): void
  start(fresh?: boolean): void
  select(rank: number | null): void
  apply(): void
  /** Back to the settings; ends any preview. */
  reset(): void
  loadMap(): void
}

export function combinationOf(rank: number | null): GlassCombination | null {
  const r = readSubstitution(useJobs.getState().jobs.glass.result?.json)
  return r?.ranked.find((c) => c.rank === rank) ?? null
}

let mapLoading: string | null = null

export const useGlassSubstitution = create<GlassSubstitutionState>()((set, get) => ({
  selected: [],
  candidates: defaultCandidates(),
  overrides: {},
  algorithm: 'tabu' as GlassAlgorithm,
  maxCombinations: 100,
  maxSeconds: null,
  localIterations: 100,
  topN: 10,
  useDesignVariables: true,
  trace: [],
  progress: null,
  selectedRank: null,
  previewError: null,
  startGlasses: {},
  configuredFor: [],
  map: null,
  mapPlane: 'nd',

  set(p) {
    set(p)
  },

  setCandidates(c) {
    set({ candidates: c })
  },

  setOverride(id, c) {
    const o = { ...get().overrides }
    if (c) o[id] = c
    else delete o[id]
    set({ overrides: o })
  },

  toggle(id, on) {
    const s = get().selected.filter((x) => x !== id)
    set({ selected: on ? [...s, id] : s })
  },

  configure(system) {
    const slots = glassSlots(system).map((s) => s.id)
    const was = get().configuredFor
    if (slots.length === was.length && slots.every((x, i) => x === was[i])) return
    const cur = get().selected
    const keep = cur.length === 0 && was.length === 0 ? slots : slots.filter((id) => cur.includes(id) || !was.includes(id))
    const overrides = Object.fromEntries(Object.entries(get().overrides).filter(([k]) => slots.includes(Number(k))))
    set({ configuredFor: slots, selected: keep, overrides })
  },

  start(fresh) {
    const { project } = useDocument.getState()
    const system = project.system
    const settings = glassSubstitutionSettings(system, project.merit, variablesOf(project), get())
    set({ trace: [], progress: null, selectedRank: null, startGlasses: startGlasses(system) })
    void useWorkspace.getState().setPreview(null)
    if (fresh) {
      // Try Again recomputes the same run (its settings were stored on start).
      useJobs.getState().start('glass', useJobs.getState().jobs.glass.settings ?? settings, { fresh: true })
    } else useJobs.getState().start('glass', settings)
  },

  select(rank) {
    set({ selectedRank: rank, previewError: null })
    const job = useJobs.getState().jobs.glass
    const c = combinationOf(rank)
    if (!c?.system || job.phase !== 'review') {
      void useWorkspace.getState().setPreview(null)
      return
    }
    useWorkspace
      .getState()
      .setPreview(c.system, `Candidate ${c.rank}`)
      .catch((e: unknown) => {
        if (get().selectedRank === rank) set({ previewError: e instanceof Error ? e.message : String(e) })
      })
  },

  apply() {
    const c = combinationOf(get().selectedRank)
    const job = useJobs.getState().jobs.glass
    if (!c?.system || job.phase !== 'review') return
    const doc = useDocument.getState()
    doc.replace({ ...doc.project, system: c.system }, 'Apply Glass Substitution')
    useJobs.getState().applied('glass')
    void useWorkspace.getState().setPreview(null)
  },

  reset() {
    void useWorkspace.getState().setPreview(null)
    set({ trace: [], progress: null, selectedRank: null })
  },

  loadMap() {
    const ws = useWorkspace.getState()
    const snap = ws.snapshot
    if (!ws.engine || !snap) return
    if (get().map?.hash === snap.hash || mapLoading === snap.hash) return
    mapLoading = snap.hash
    let job
    try {
      job = ws.engine.submit({ analysis: 'glass_map', settings: {} }, snap)
    } catch {
      mapLoading = null
      return
    }
    job.result.then(
      (result) => {
        if (mapLoading === snap.hash) mapLoading = null
        set({ map: { hash: snap.hash, result } })
      },
      () => {
        if (mapLoading === snap.hash) mapLoading = null
      },
    )
  },
}))

// Live progress and the sparkline follow the job even while the panel is closed;
// a finished run previews its best combination.
useJobs.subscribe((s, prev) => {
  const job = s.jobs.glass
  const was = prev.jobs.glass
  if (job === was) return
  const g = useGlassSubstitution.getState()
  if (job.phase === 'running' && was.phase !== 'running') useGlassSubstitution.setState({ trace: [], progress: null, selectedRank: null })
  if (job.phase === 'running' && (job.stage !== was.stage || job.detail !== was.detail)) {
    const p = parseGlassProgress(job.stage, job.detail)
    const cur = useGlassSubstitution.getState()
    if (p) {
      const trace = Number.isFinite(p.bestMerit) && p.bestMerit !== cur.trace.at(-1) ? [...cur.trace, p.bestMerit] : cur.trace
      useGlassSubstitution.setState({ progress: p, trace })
    }
  }
  if (job.phase === 'review' && was.phase === 'running') {
    const r = readSubstitution(job.result?.json)
    if (r?.history.length) useGlassSubstitution.setState({ trace: r.history })
    const best = r?.ranked[0]
    if (best) g.select(best.rank)
  }
  if (job.phase === 'setup' && was.phase !== 'setup') useGlassSubstitution.setState({ selectedRank: null })
})
