// Glass substitution as a design job (apps/macos/Sources/MKLightUI/Views/Jobs/
// GlassSubstitutionPanel.swift; docs/APP.md "Design jobs"): choose the lenses whose
// glass may change and the candidates, run a quick or thorough search that
// re-optimises the shape for every glass combination, then compare the ranked
// combinations (each previewed in the window's layout and analyses, and on the
// glass map) and apply one as an undoable step.

import { ArrowRight, Droplet } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { sig } from '@/model/format'
import {
  ALGORITHMS,
  candidatesSummary,
  changeSummary,
  combinationHighlights,
  freeVariables,
  glassSlots,
  movedChanges,
  meritSummary,
  offsetNote,
  readSubstitution,
  statusText,
  type GlassAlgorithm,
  type GlassCombination,
  type GlassSlot,
} from '@/model/glass'
import { paramTitle, variablesOf, type ParamPath } from '@/model/params'
import { radiusFromCurvature, type System } from '@/model/system'
import { useDocument } from '@/state/document'
import { combinationOf, useGlassSubstitution } from '@/state/glassSubstitution'
import { useJobs } from '@/state/jobs'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import { cn } from '@/lib/utils'
import { NumberInput } from '../SettingsForm'
import { FormRow, Section } from '../inspector/Form'
import { GlassMapChart, Sub } from '../glassmap/GlassMapChart'
import { CandidateForm, OverrideButton, useCatalogs } from './glass/CandidateForm'
import { LayoutCompare } from './glass/LayoutCompare'
import { AppliedNotice, BigNumber, JobActions, JobFailure, JobPanelFrame, Sparkline } from './JobPanel'

const short = (v: number) => sig(v, 4)
/** Rows with long labels and short numbers. */
const WIDE = 'grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_2rem]'

export function GlassSubstitutionPanel() {
  const system = useDocument((s) => s.project.system)
  const job = useJobs((s) => s.jobs.glass)
  const selected = useGlassSubstitution((s) => s.selected)
  const configure = useGlassSubstitution((s) => s.configure)
  const progress = useGlassSubstitution((s) => s.progress)
  const snapshot = useWorkspace((s) => s.snapshot)
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const slots = useMemo(() => glassSlots(system), [system])

  useEffect(() => configure(system), [system, configure])
  // Closing the panel shows the design again; the ranking is kept.
  useEffect(
    () => () => {
      const p = useWorkspace.getState().preview
      if (useJobs.getState().jobs.glass.phase !== 'running' && p && /^Candidate \d+$/.test(p.label)) void useWorkspace.getState().setPreview(null)
    },
    [],
  )

  const n = selected.length
  const subtitle = n === 0 ? 'Snap model glasses to real glasses' : `${n} glass${n === 1 ? '' : 'es'} may change`
  const cannotRun = !snapshot || modelIssues.length > 0 || (n === 0 && !slots.some((s) => s.isModel))
  const g = useGlassSubstitution.getState()

  let body: React.ReactNode
  let actions: React.ReactNode
  switch (job.phase) {
    case 'setup':
      body = <Setup system={system} slots={slots} />
      actions = JobActions.start('Find Glasses', () => g.start(), {
        disabled: cannotRun,
        help: cannotRun ? (slots.length ? 'Tick a glass, or use model glasses to snap' : 'The design has no glass to replace') : undefined,
      })
      break
    case 'running':
      body = <Running slots={slots} />
      actions = JobActions.running('glass', job, {
        detail: progress ? `${progress.combinations} combinations` : 'Preparing',
        canStop: true,
        stopHelp: 'End now and keep the combinations ranked so far',
      })
      break
    case 'review':
    case 'done':
      body = <Results system={system} applied={job.phase === 'done'} />
      actions =
        job.phase === 'done'
          ? JobActions.done('glass', 'Back to the glasses and the search settings')
          : JobActions.review('glass', () => g.apply(), { applyDisabled: !combinationOf(g.selectedRank)?.system })
      break
    case 'failed':
      body = <JobFailure title="Glass Substitution Failed" reason={job.error ?? 'The search failed.'} />
      // Back, and Try Again with the same settings recomputed.
      actions = JobActions.failed('glass', { retryDisabled: cannotRun })
      break
  }
  return (
    <JobPanelFrame kind="glass" subtitle={subtitle} actions={actions}>
      {body}
    </JobPanelFrame>
  )
}

// MARK: Setup

function Setup({ system, slots }: { system: System; slots: GlassSlot[] }) {
  const s = useGlassSubstitution()
  const merit = useDocument((d) => d.project.merit)
  const rawVariables = useDocument((d) => d.project.variables)
  const variables = useMemo(() => variablesOf({ variables: rawVariables }), [rawVariables])
  const modelIssues = useWorkspace((w) => w.modelIssues)
  const catalogs = useCatalogs()
  const note = offsetNote(system, s.selected)
  const free = freeVariables(system, variables)
  const useDesign = s.useDesignVariables
  useEffect(() => {
    if (!variables.length && useDesign) useGlassSubstitution.getState().set({ useDesignVariables: false })
  }, [variables.length, useDesign])

  return (
    <div className="text-[13px]">
      <Section
        title="Glasses That May Change"
        footer={s.selected.length === 0 && slots.length > 0 ? 'With no glass ticked, every model glass (n_d, V_d) snaps to its nearest real glasses.' : undefined}
      >
        {slots.length === 0 && <p className="py-1 text-[12px] text-secondary">The design has no glass to replace.</p>}
        {slots.map((slot) => (
          <div key={slot.id} className="flex items-center gap-2 py-[3px]" data-testid={`glass-slot-${slot.index}`}>
            <label className="flex min-w-0 flex-1 items-center gap-2">
              <input type="checkbox" className="accent-[var(--accent)]" checked={s.selected.includes(slot.id)} onChange={(e) => s.toggle(slot.id, e.target.checked)} />
              <span className="flex min-w-0 flex-col">
                <span className="flex min-w-0 items-center gap-1.5">
                  <span className="num text-secondary">S{slot.index}</span>
                  <span className="truncate font-medium">{slot.material}</span>
                  {slot.isModel && <span className="rounded-full bg-selected px-1.5 text-[10px] text-secondary">model</span>}
                  {slot.offset && (
                    <span className="num text-[10px] text-secondary" title="Melt offset Δn_d / ΔV_d: a replacement glass is installed without it">
                      {slot.offset}
                    </span>
                  )}
                </span>
                {s.overrides[slot.id] && <span className="truncate text-[11px] text-accent">{candidatesSummary(s.overrides[slot.id])}</span>}
              </span>
            </label>
            <OverrideButton slot={slot} catalogs={catalogs} />
          </div>
        ))}
        {slots.length > 0 && (
          <div className="flex gap-1 pt-1">
            <Button size="sm" variant="ghost" onClick={() => s.set({ selected: slots.map((x) => x.id) })}>
              All
            </Button>
            <Button size="sm" variant="ghost" onClick={() => s.set({ selected: [] })}>
              None
            </Button>
          </div>
        )}
        {note && (
          <p className="flex gap-1.5 pb-1 pt-1 text-[11px] text-secondary" data-testid="offset-note">
            <Droplet className="mt-[1px] size-3.5 shrink-0" />
            {note}
          </p>
        )}
      </Section>
      <Section title="Candidates for Every Lens">
        <CandidateForm candidates={s.candidates} onChange={s.setCandidates} catalogs={catalogs} />
      </Section>
      <Section title="Search" footer={ALGORITHMS.find((a) => a.value === s.algorithm)?.detail}>
        <FormRow label="Search">
          <Select aria-label="Search" value={s.algorithm} onChange={(e) => s.set({ algorithm: e.target.value as GlassAlgorithm })}>
            {ALGORITHMS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.title}
              </option>
            ))}
          </Select>
        </FormRow>
        <FormRow className={WIDE} label="Combinations to try">
          <NumberInput ariaLabel="Combinations to try" value={s.maxCombinations} integer min={1} onCommit={(v) => v != null && s.set({ maxCombinations: v })} />
        </FormRow>
        <FormRow className={WIDE} label="Time limit" unit="s">
          <NumberInput ariaLabel="Time limit" value={s.maxSeconds} nullable placeholder="none" onCommit={(v) => s.set({ maxSeconds: v != null && v > 0 ? v : null })} />
        </FormRow>
        <FormRow className={WIDE} label="Shape iterations each" help="Damped least-squares iterations that re-optimize the shape for each glass combination">
          <NumberInput ariaLabel="Shape iterations each" value={s.localIterations} integer min={1} onCommit={(v) => v != null && s.set({ localIterations: v })} />
        </FormRow>
        <FormRow className={WIDE} label="Keep the best">
          <NumberInput ariaLabel="Keep the best" value={s.topN} integer min={1} onCommit={(v) => v != null && s.set({ topN: v })} />
        </FormRow>
      </Section>
      <Section title="Merit">
        <p className="py-1 text-[12px]">{meritSummary(merit)}</p>
        {variables.length === 0 ? (
          <p className="pb-1 text-[12px] text-secondary">Each combination re-optimizes every curvature and the back focus.</p>
        ) : (
          <div className="flex flex-col gap-1 pb-1 text-[12px]" role="radiogroup" aria-label="Shape">
            <label className="flex items-center gap-2">
              <input type="radio" checked={s.useDesignVariables} onChange={() => s.set({ useDesignVariables: true })} />
              Re-optimize the design variables ({free.length})
            </label>
            <label className="flex items-center gap-2">
              <input type="radio" checked={!s.useDesignVariables} onChange={() => s.set({ useDesignVariables: false })} />
              Re-optimize every curvature and the back focus
            </label>
          </div>
        )}
        <button type="button" className="cursor-default pb-1 text-[12px] text-accent hover:underline" onClick={() => useView.getState().setPage('optimize')}>
          Edit Merit and Variables…
        </button>
      </Section>
      {modelIssues.length > 0 && <p className="text-[12px] text-danger">Fix the design’s errors first.</p>}
    </div>
  )
}

// MARK: Running

function Running({ slots }: { slots: GlassSlot[] }) {
  const progress = useGlassSubstitution((s) => s.progress)
  const trace = useGlassSubstitution((s) => s.trace)
  const selected = useGlassSubstitution((s) => s.selected)
  const max = useJobs((s) => (s.jobs.glass.settings?.max_combinations as number | undefined) ?? null)
  const ticked = slots.filter((s) => selected.includes(s.id))
  const order = ticked.length ? ticked : slots.filter((s) => s.isModel)
  return (
    <div className="flex flex-col gap-3" data-testid="glass-running">
      <div className="flex gap-6">
        <BigNumber label="Combinations" value={progress ? `${progress.combinations}${max ? ` / ${max}` : ''}` : '—'} testId="glass-combinations" />
        <BigNumber label="Best merit" value={progress && Number.isFinite(progress.bestMerit) ? short(progress.bestMerit) : '—'} testId="glass-best-merit" />
      </div>
      {progress && progress.glasses.length > 0 && (
        <div>
          <div className="mb-1 text-[11px] text-secondary">Best glasses so far</div>
          <div className="flex flex-wrap gap-1">
            {progress.glasses.map((name, k) => (
              <span key={k} className="num rounded-full bg-accent-soft px-2 py-[1px] text-[11px]">
                {order[k] ? `S${order[k].index} ` : ''}
                {name}
              </span>
            ))}
          </div>
        </div>
      )}
      <Sparkline values={trace} log={trace.every((v) => v > 0)} height={110} />
      <p className="text-[11px] text-secondary">
        Stop keeps the combinations ranked so far. Cancel discards the run. You can close this panel; the search continues.
      </p>
    </div>
  )
}

// MARK: Results

function Results({ system, applied }: { system: System; applied: boolean }) {
  const result = useJobs((s) => s.jobs.glass.result)
  const r = useMemo(() => readSubstitution(result?.json), [result])
  const rank = useGlassSubstitution((s) => s.selectedRank)
  const select = useGlassSubstitution((s) => s.select)
  if (!r) return <JobFailure title="No Combinations" reason="The engine returned no glass combinations." />
  const chosen = r.ranked.find((c) => c.rank === rank) ?? null
  return (
    <div className="flex flex-col gap-3 text-[13px]" data-testid="glass-results">
      {applied && <AppliedNotice />}
      <p className="text-[12px] text-secondary">{statusText(r, applied)}</p>
      {r.ranked.length === 0 ? (
        <p className="py-6 text-center text-[12px] text-secondary">No combination could be evaluated.</p>
      ) : (
        <>
          <div
            role="listbox"
            aria-label="Ranked glass combinations"
            className={cn('max-h-[280px] overflow-y-auto rounded-lg bg-content shadow-[0_0_0_0.5px_var(--separator)]', applied && 'pointer-events-none opacity-70')}
            onKeyDown={(e) => {
              if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
              e.preventDefault()
              const k = r.ranked.findIndex((c) => c.rank === rank)
              const next = r.ranked[Math.min(Math.max((k < 0 ? -1 : k) + (e.key === 'ArrowDown' ? 1 : -1), 0), r.ranked.length - 1)]
              if (next) select(next.rank)
            }}
            tabIndex={0}
          >
            {r.ranked.map((c, i) => (
              <RankRow key={c.rank} c={c} start={r.startingMerit} selected={c.rank === rank} odd={i % 2 === 1} onSelect={() => select(c.rank)} />
            ))}
          </div>
          {chosen && <Comparison c={chosen} start={r.startingMerit} system={system} applied={applied} />}
          <CandidateMap c={chosen} system={system} />
        </>
      )}
    </div>
  )
}

function RankRow({ c, start, selected, odd, onSelect }: { c: GlassCombination; start: number; selected: boolean; odd: boolean; onSelect: () => void }) {
  const gain = start > 0 && Number.isFinite(c.merit) ? 1 - c.merit / start : NaN
  return (
    <div
      role="option"
      aria-selected={selected}
      aria-label={`Rank ${c.rank}, merit ${short(c.merit)}, glasses ${c.glasses.map((g) => g.name).join(', ')}`}
      data-testid={`glass-rank-${c.rank}`}
      className={cn('flex cursor-default items-baseline gap-2 px-2 py-1.5', selected ? 'bg-accent text-on-accent' : odd && 'bg-hover')}
      onClick={onSelect}
    >
      <span className="num w-5 shrink-0 text-right font-semibold">{c.rank}</span>
      <span className="min-w-0 flex-1">
        <span className="block truncate">{c.glasses.map((g) => g.name).join(' · ')}</span>
        <span className={cn('block truncate text-[11px]', selected ? 'opacity-80' : 'text-secondary')}>{changeSummary(c)}</span>
      </span>
      <span className="shrink-0 text-right">
        <span className="num block">{short(c.merit)}</span>
        {!c.feasible ? (
          <span className={cn('block text-[10px]', !selected && 'text-warning')}>constraints not met</span>
        ) : (
          Number.isFinite(gain) && (
            <span className={cn('num block text-[10px]', !selected && (c.merit < start ? 'text-success' : 'text-secondary'))}>
              {gain >= 0 ? '' : '−'}
              {Math.round(Math.abs(gain) * 100)}%
            </span>
          )
        )}
      </span>
    </div>
  )
}

function display(v: number, p: ParamPath | null): number {
  if (p && (p.param === 'curvature' || (p.param === 'shape_param' && /^curvature_/.test(String(p.parameter))))) return radiusFromCurvature(v)
  return v
}

/** Before / after: glasses per surface, merit, shape changes and the two layouts. */
function Comparison({ c, start, system, applied }: { c: GlassCombination; start: number; system: System; applied: boolean }) {
  const before = useGlassSubstitution((s) => s.startGlasses)
  const previewError = useGlassSubstitution((s) => s.previewError)
  const appliedLayouts = useGlassSubstitution((s) => s.appliedLayouts)
  const design = useWorkspace((s) => s.snapshot)
  const moved = movedChanges(c)
  return (
    <div className="select-text rounded-lg bg-hover p-3" data-testid="glass-comparison">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-[14px] font-semibold">Candidate {c.rank}</h3>
        {!applied && !previewError && <span className="text-[11px] text-accent">Previewed in the window</span>}
      </div>
      {previewError && (
        <p className="mb-2 text-[11px] text-warning" role="alert">
          Cannot preview this candidate: {previewError}
        </p>
      )}
      <div className="mb-2 flex gap-6">
        <BigNumber label="Merit now" value={short(start)} />
        <BigNumber label="Candidate" value={short(c.merit)} />
      </div>
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto_minmax(0,1fr)_auto] items-baseline gap-x-2 gap-y-0.5 text-[12px]">
        {c.glasses.map((g) => {
          const i = system.surfaces.findIndex((s) => s.id === g.surface)
          return (
            <div key={g.surface} className="contents">
              <span className="num text-secondary">S{i >= 0 ? i : '?'}</span>
              <span className="truncate">{before[g.surface] ?? '—'}</span>
              <ArrowRight className="size-3 text-secondary" aria-label="to" />
              <span className="truncate font-semibold">{g.name}</span>
              <span className="num text-[11px] text-secondary">
                {Number.isFinite(g.nd) ? g.nd.toFixed(4) : '–'} / {Number.isFinite(g.vd) ? g.vd.toFixed(1) : '–'}
              </span>
            </div>
          )
        })}
      </div>
      {moved.length > 0 && (
        <details className="mt-2 text-[12px]">
          <summary className="cursor-default">Shape changes ({moved.length})</summary>
          <div className="mt-1 flex flex-col gap-0.5 text-[11px]">
            {moved.map((ch, k) => (
              <div key={k} className="flex justify-between gap-2">
                <span className="truncate">{ch.param ? paramTitle(system, ch.param) : '?'}</span>
                <span className="num">
                  {short(display(ch.start, ch.param))} → {short(display(ch.value, ch.param))}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
      <div className="mt-2">
        {/* While previewed, the candidate's layout; once applied, the design's. */}
        {applied ? <LayoutCompare before={appliedLayouts?.before ?? null} after={appliedLayouts?.after ?? design} /> : <LayoutCompare />}
      </div>
    </div>
  )
}

/** The design's glass map with the candidate's glasses ringed. */
function CandidateMap({ c, system }: { c: GlassCombination | null; system: System }) {
  const map = useGlassSubstitution((s) => s.map)
  const plane = useGlassSubstitution((s) => s.mapPlane)
  const set = useGlassSubstitution((s) => s.set)
  const loadMap = useGlassSubstitution((s) => s.loadMap)
  const snapHash = useWorkspace((s) => s.snapshot?.hash)
  const libraryHash = useWorkspace((s) => s.snapshot?.libraryHash)
  // Reloads when the design or the glass catalogs change.
  const materials = useWorkspace((s) => s.materials)
  useEffect(() => loadMap(), [snapHash, libraryHash, materials, loadMap])
  const highlighted = useMemo(() => (c ? combinationHighlights(c, system) : []), [c, system])
  if (!map) return null
  return (
    <div data-testid="glass-candidate-map">
      <div className="mb-1 flex items-center justify-between">
        <h3 className="text-[14px] font-semibold">Glass Map</h3>
        <Segmented
          label="Plane"
          value={plane}
          onChange={(v) => set({ mapPlane: v })}
          options={[
            { value: 'nd', title: <Sub base="n" sub="d" /> },
            { value: 'pgf', title: <Sub base="P" sub="g,F" /> },
          ]}
        />
      </div>
      <GlassMapChart map={map.result} system={system} highlighted={highlighted} plane={plane} height={260} />
      <p className="mt-1 text-[11px] text-secondary">Ringed: the candidate’s glasses, labeled with their surfaces.</p>
    </div>
  )
}
