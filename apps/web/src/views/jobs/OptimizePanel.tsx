// The Optimize design job (apps/macos/Sources/MKLightUI/Views/Jobs/JobPanel.swift,
// `OptimizePanel`; docs/APP.md "Design jobs" → Optimize): the merit function
// with its live evaluation, the variables and their bounds, the algorithm; a
// running view with the live iteration, merit, improvement and a log-scale
// sparkline (Stop keeps the best, Cancel discards); the result previewed in the
// window with every variable before → after, applied as one undoable step.

import { ArrowRight, Pin } from 'lucide-react'
import { useEffect } from 'react'
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import type { Json, JsonObject } from '@/engine/types'
import { isCurvature } from '@/model/bounds'
import { nf, sig } from '@/model/format'
import { paramTitle, paramUnit, variablesOf, type ParamPath } from '@/model/params'
import { radiusFromCurvature, type System } from '@/model/system'
import { useDocument } from '@/state/document'
import { useJobs, type JobState } from '@/state/jobs'
import {
  ALGORITHMS,
  AUTOMATIC_VARIABLES,
  DEFAULT_ITERATIONS,
  STOCHASTIC,
  optimizeSettings,
  usesDesignVariables,
  variableProblem,
  useOptimize,
} from '@/state/optimize'
import { useWorkspace } from '@/state/workspace'
import { cn } from '@/lib/utils'
import { NumberInput } from '../SettingsForm'
import { FormRow, Section } from '../inspector/Form'
import { AppliedNotice, BigNumber, JobActions, JobFailure, JobPanelFrame, Sparkline } from './JobPanel'
import { MeritFunctionSection } from './MeritEditor'
import { VariablesBoundsList } from './VariablesBounds'

const PREVIEW_LABEL = 'Optimization'

/** A live progress record: `{iteration, evaluations, merit, best_merit}`. */
interface MeritPoint {
  iteration: number
  merit: number
  best: number
}

function point(d: Json | null | undefined, stage?: string | null): MeritPoint | null {
  if (d && typeof d === 'object' && !Array.isArray(d)) {
    const o = d as JsonObject
    const merit = nf(o.merit)
    const best = nf(o.best_merit)
    if (typeof o.iteration === 'number' && (Number.isFinite(merit) || Number.isFinite(best)))
      return { iteration: o.iteration, merit, best: Number.isFinite(best) ? best : merit }
  }
  // The engine's stage text, e.g. `lm 12: merit 3.4e-4`.
  const m = stage?.match(/(\d+):\s*merit\s+([-+.\deE]+)/)
  if (m) return { iteration: Number(m[1]), merit: Number(m[2]), best: Number(m[2]) }
  return null
}

const percent = (f: number) => (Number.isFinite(f) ? `${(f * 100).toFixed(Math.abs(f) < 0.1 ? 2 : 1)}%` : '–')

export function OptimizePanel() {
  const project = useDocument((s) => s.project)
  const job = useJobs((s) => s.jobs.optimize)
  const opt = useOptimize()
  const snapshot = useWorkspace((s) => s.snapshot)
  const modelErrors = useWorkspace((s) => s.modelIssues.filter((i) => i.severity === 'error').length)
  const phase = job.phase

  // Live merit: re-evaluate when the design, its merit, the variables, the
  // snapshot or the algorithm change (debounced in the store).
  const snapHash = snapshot?.hash
  useEffect(() => {
    if (phase === 'setup') useOptimize.getState().scheduleEvaluation()
    else useOptimize.getState().cancelEvaluation()
  }, [phase, project.merit, project.variables, project.system, snapHash, opt.algorithm, opt.useDesignVariables])

  // A finished, unapplied result is previewed in the window.
  const resultSystem = (job.result?.json.system ?? null) as System | null
  useEffect(() => {
    if (phase === 'review' && resultSystem) void useWorkspace.getState().setPreview(resultSystem, PREVIEW_LABEL).catch(() => {})
  }, [phase, resultSystem])
  useEffect(
    () => () => {
      const ws = useWorkspace.getState()
      if (useJobs.getState().jobs.optimize.phase !== 'running' && ws.preview?.label === PREVIEW_LABEL) void ws.setPreview(null)
    },
    [],
  )

  const problem = variableProblem(project, opt.useDesignVariables)
  const blocked = !snapshot || modelErrors > 0
  const start = () => {
    useOptimize.getState().cancelEvaluation()
    useJobs.getState().start('optimize', optimizeSettings(project, useOptimize.getState()))
  }
  const apply = () => {
    const sys = job.result?.json.system as System | undefined
    if (!sys) return
    const doc = useDocument.getState()
    doc.replace({ ...doc.project, system: sys }, 'Apply Optimization')
    useJobs.getState().applied('optimize')
    void useWorkspace.getState().setPreview(null)
  }

  const last = point(job.detail, job.stage)
  let actions: React.ReactNode
  switch (phase) {
    case 'setup':
      actions = (
        <span className="contents" data-testid="optimize-start">
          {JobActions.start('Optimize', start, {
            disabled: blocked || problem != null,
            help: problem ?? (blocked ? 'Fix the design’s errors first' : 'Optimize the variables against the merit function'),
          })}
        </span>
      )
      break
    case 'running':
      actions = JobActions.running('optimize', job, {
        detail: last ? `Iteration ${last.iteration} · merit ${sig(last.best, 3)}` : '',
        canStop: true,
        stopHelp: 'End now and keep the best design so far',
      })
      break
    case 'review':
      actions = JobActions.review('optimize', apply, { applyDisabled: !job.result?.json.system })
      break
    case 'done':
      actions = JobActions.done('optimize', 'Back to the merit function and variables')
      break
    case 'failed':
      actions = JobActions.failed('optimize', { retryDisabled: blocked || problem != null })
      break
  }

  return (
    <JobPanelFrame kind="optimize" subtitle={project.system?.name ?? ''} actions={actions}>
      {phase === 'setup' && <Setup problem={problem} blocked={blocked} modelErrors={modelErrors} />}
      {phase === 'running' && <Running job={job} />}
      {(phase === 'review' || phase === 'done') && <Review job={job} system={project.system} applied={phase === 'done'} />}
      {phase === 'failed' && <JobFailure title="Optimization Failed" reason={job.error ?? 'The engine gave no reason.'} />}
    </JobPanelFrame>
  )
}

function Setup({ problem, blocked, modelErrors }: { problem: string | null; blocked: boolean; modelErrors: number }) {
  const project = useDocument((s) => s.project)
  const opt = useOptimize()
  const n = variablesOf(project).length
  const design = usesDesignVariables(project, opt.useDesignVariables)
  return (
    <>
      <MeritFunctionSection />
      <Section title="Variables">
        <div className="py-1">
          <Segmented<'design' | 'auto'>
            label="Vary"
            value={design ? 'design' : 'auto'}
            onChange={(v) => n > 0 && opt.set({ useDesignVariables: v === 'design' })}
            options={[
              { value: 'design', title: `Design variables (${n})`, help: n ? 'The variables marked with V, with their bounds' : 'No variables marked yet' },
              { value: 'auto', title: 'Automatic', help: 'Every curvature and the back focus' },
            ]}
          />
        </div>
        {design ? (
          <VariablesBoundsList />
        ) : (
          <div data-testid="variables-list" className="py-1">
            <p className="text-[13px]">{AUTOMATIC_VARIABLES}</p>
            <p className="pt-0.5 text-[11px] text-secondary">
              {n === 0
                ? 'Mark values with V in the lens data or the inspector to choose them; each gets sensible bounds you can edit here.'
                : 'Solve-controlled parameters are left out.'}
            </p>
          </div>
        )}
        {problem && (
          <p className="flex items-center gap-1 py-1 text-[11px] text-warning" role="alert">
            <Pin className="size-3 shrink-0" />
            {problem}
          </p>
        )}
      </Section>
      <Section title="Algorithm">
        <FormRow label="Method">
          <Select aria-label="Method" className="w-full" value={opt.algorithm} onChange={(e) => opt.set({ algorithm: e.target.value })}>
            {ALGORITHMS.map((a) => (
              <option key={a.id} value={a.id}>
                {a.title}
              </option>
            ))}
          </Select>
        </FormRow>
        <FormRow label="Iterations" help="Iterations, generations or rounds; empty is the algorithm’s default">
          <NumberInput
            ariaLabel="Iterations"
            value={opt.maxIterations}
            nullable
            integer
            min={1}
            placeholder={`default (${DEFAULT_ITERATIONS[opt.algorithm] ?? ''})`}
            onCommit={(v) => opt.set({ maxIterations: v })}
          />
        </FormRow>
        {STOCHASTIC.has(opt.algorithm) && (
          <FormRow label="Seed" help="The random seed: the same seed repeats the run">
            <NumberInput ariaLabel="Seed" value={opt.seed} integer min={0} onCommit={(v) => opt.set({ seed: v ?? 0 })} />
          </FormRow>
        )}
      </Section>
      {blocked && (
        <p className="px-1 text-[12px] text-danger" role="alert">
          {modelErrors ? 'Fix the design’s errors first.' : 'Waiting for the engine…'}
        </p>
      )}
    </>
  )
}

function Running({ job }: { job: JobState }) {
  const points = job.history.map((d) => point(d)).filter((p): p is MeritPoint => p != null)
  const last = points.at(-1) ?? point(job.detail, job.stage)
  const first = points[0]
  const algorithm = useOptimize((s) => s.algorithm)
  const improvement = first && last && first.merit > 0 ? 1 - last.best / first.merit : NaN
  return (
    <div className="flex flex-col gap-3" data-testid="optimize-running">
      <div className="grid grid-cols-3 gap-3">
        <BigNumber label="Iteration" value={last ? String(last.iteration) : '–'} testId="optimize-iteration" />
        <div
          title={
            ['de', 'cma_es', 'nelder_mead'].includes(algorithm)
              ? 'The search’s merit, including the penalty that holds the focal length; the result reports the merit itself.'
              : 'The best merit so far, as the search evaluates it.'
          }
        >
          <BigNumber label="Search merit" value={last ? sig(last.best, 4) : '–'} />
        </div>
        <BigNumber label="Improvement" value={percent(improvement)} />
      </div>
      <div className="rounded-lg bg-content px-2 py-2 shadow-[0_0_0_0.5px_var(--separator)]">
        <Sparkline values={points.map((p) => p.best)} log height={110} />
      </div>
      <p className="text-[11px] text-secondary">Stop keeps the best design so far. Cancel discards the run. You can close this panel; the job continues.</p>
    </div>
  )
}

interface ResultVariable {
  label: string
  param: ParamPath | null
  start: number
  value: number
  atBound: boolean
}

function Review({ job, system, applied }: { job: JobState; system: System; applied: boolean }) {
  const r = job.result?.json ?? {}
  const preview = useWorkspace((s) => s.preview)
  const start = nf((r.start as JsonObject | undefined)?.merit)
  const final = r.final as JsonObject | undefined
  const finalMerit = nf(final?.merit)
  const improvement = r.improvement == null ? (start > 0 ? 1 - finalMerit / start : NaN) : nf(r.improvement)
  const improved = r.improved !== false
  const stopped = r.stopped === true || r.status === 'stopped'
  const resultSystem = (r.system ?? null) as System | null
  const history = (Array.isArray(r.history) ? r.history : []).map((h) => point(h)).filter((p): p is MeritPoint => p != null)
  const vars: ResultVariable[] = (Array.isArray(r.variables) ? r.variables : []).map((v) => {
    const o = (v ?? {}) as JsonObject
    const param = o.param && typeof o.param === 'object' ? (o.param as ParamPath) : null
    return { label: String(o.label ?? ''), param, start: nf(o.start), value: nf(o.value), atBound: o.at_bound === true }
  })
  // The variables are shown in the design they belong to (the result's, for surfaces).
  const titleSystem = resultSystem ?? system

  let status = `${nf(r.iterations) || 0} iterations, ${nf(r.evaluations) || 0} evaluations. `
  if (stopped) status += 'Stopped early: this is the best design found so far. '
  if (!improved) status += 'No better design was found; the result equals the start. '
  if (final && final.feasible === false) status += 'Some constraints are not met. '
  const evaluatedOn = String((r.backend as JsonObject | undefined)?.evaluated_on ?? '')

  const shown = (v: number, p: ParamPath | null) => (p && isCurvature(p) ? radiusFromCurvature(v) : v)
  const unitOf = (p: ParamPath | null) => (p ? (isCurvature(p) ? 'mm' : paramUnit(p)) : '')

  return (
    <div className="flex flex-col gap-3" data-testid="optimize-result">
      {applied && <AppliedNotice />}
      <div className="grid grid-cols-3 gap-3">
        <BigNumber label="Merit before" value={sig(start, 4)} />
        <BigNumber label="After" value={sig(finalMerit, 4)} testId="optimize-final-merit" />
        <BigNumber label="Improvement" value={percent(improvement)} />
      </div>
      {!applied && resultSystem && (
        <label className="flex items-center justify-between gap-2" title="Show the optimized design in the layout and analyses before applying it">
          <span>Preview in the window</span>
          <Switch
            aria-label="Preview in the window"
            checked={preview?.label === PREVIEW_LABEL}
            onCheckedChange={(on) => void useWorkspace.getState().setPreview(on ? resultSystem : null, PREVIEW_LABEL)}
          />
        </label>
      )}
      <p className="text-[12px] text-secondary">{status}</p>
      {history.length > 1 && (
        <div className="rounded-lg bg-content px-2 py-2 shadow-[0_0_0_0.5px_var(--separator)]">
          <Sparkline values={history.map((p) => p.best)} log height={56} />
        </div>
      )}
      <Section title="Variables">
        {vars.length === 0 && <p className="py-1 text-secondary">The engine reported no variables.</p>}
        <div data-testid="optimize-result-variables">
          {vars.map((v, k) => {
            const before = shown(v.start, v.param)
            const after = shown(v.value, v.param)
            const unit = unitOf(v.param)
            const d = after - before
            return (
              <div key={k} className={cn('py-1', k > 0 && 'border-t border-separator')}>
                <div className="flex items-center gap-1.5 text-[13px]">
                  <span className="truncate">{v.param ? paramTitle(titleSystem, v.param) : v.label}</span>
                  {v.atBound && <span className="text-[10px] text-warning">at bound</span>}
                </div>
                <div className="num flex items-center gap-1.5 text-[12px]">
                  <span>{sig(before, 6)}</span>
                  <ArrowRight className="size-3 text-secondary" aria-label="to" />
                  <span className="font-semibold">{sig(after, 6)}</span>
                  <span className="ml-auto text-secondary">
                    {Number.isFinite(d) ? (d >= 0 ? '+' : '') + sig(d, 4) : ''}
                    {unit && unit !== '–' ? ' ' + unit : ''}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </Section>
      {evaluatedOn && (
        <p className="text-[11px] text-secondary">
          {evaluatedOn.startsWith('cpu') ? 'Evaluated on the CPU f64 reference.' : `Searched on ${evaluatedOn}; merits reported on the CPU f64 reference.`}
        </p>
      )}
    </div>
  )
}
