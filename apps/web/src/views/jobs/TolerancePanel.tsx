// The Tolerance panel (docs/APP.md "Design jobs" → Tolerance,
// apps/macos/Sources/MKLightUI/Views/Jobs/TolerancePanel.swift): the tolerance
// set (default tolerances, the table grouped by element, the test
// wavelength), compensators, criterion and analysis; the run with live stage
// and timing, Stop and Cancel; and the result with previews of toleranced
// builds. Tolerancing never changes the design, so there is no Apply: a run
// ends on its results, with Back to settings and Run again.

import { BarChart3 } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { compensatorTitle, readResult, runProblem, settingsIssues, stageText } from '@/model/tolerance'
import { useDocument } from '@/state/document'
import { useJobs } from '@/state/jobs'
import {
  endTolerancePreview,
  noteSecondsPerTrial,
  retryTolerance,
  startTolerance,
  useMeritInfo,
  useTolerance,
  useToleranceContext,
  useToleranceContextSources,
  useToleranceSettings,
} from '@/state/tolerance'
import { useWorkspace } from '@/state/workspace'
import { JobActions, JobFailure, JobPanelFrame } from './JobPanel'
import { ToleranceProgress, useNow, useToleranceTiming } from './tolerance/Progress'
import { ToleranceResults } from './tolerance/Results'
import { AnalysisSection, CompensatorSection, CriterionSection } from './tolerance/Sections'
import { ToleranceTable } from './tolerance/ToleranceTable'

export function TolerancePanel() {
  const settings = useToleranceSettings()
  useToleranceContextSources()
  const context = useToleranceContext(settings)
  const merit = useMeritInfo()
  const system = useDocument((s) => s.project.system)
  const job = useJobs((s) => s.jobs.tolerance)
  const snapshot = useWorkspace((s) => s.snapshot)
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const wsPreview = useWorkspace((s) => s.preview)
  const stoppable = useWorkspace((s) => s.descriptors.find((d) => d.name === 'tolerance')?.stoppable ?? true)
  const runSystem = useTolerance((s) => s.runSystem)
  const runDisplay = useTolerance((s) => s.runDisplay)
  const previewLoaded = useTolerance((s) => s.preview != null)

  const issues = useMemo(() => settingsIssues(settings, context), [settings, context])
  const result = useMemo(
    () => (job.result ? readResult(job.result.json, runSystem ?? system, runDisplay, job.result.buffers?.monte_carlo_criteria as ArrayLike<number> | undefined) : null),
    [job.result, runSystem, system, runDisplay],
  )

  useEffect(() => noteSecondsPerTrial(result?.secondsPerTrial ?? null), [result])
  // Previews belong to the results page; the design changing ends them too.
  useEffect(() => {
    if (job.phase !== 'review' && useTolerance.getState().previewed) endTolerancePreview()
  }, [job.phase])
  useEffect(() => {
    if (!wsPreview && previewLoaded) useTolerance.setState({ previewed: null, preview: null, previewError: null })
  }, [wsPreview, previewLoaded])

  const problem = !snapshot
    ? 'The design has not been checked by the engine yet.'
    : modelIssues.length
      ? "Fix the design's errors first."
      : runProblem(settings, context)

  const n = settings.rows.length, c = settings.compensators.length
  const comp = c === 0 ? 'no compensator' : c === 1 ? compensatorTitle(settings.compensators[0].param, system).toLowerCase() : `${c} compensators`
  const subtitle = `${system.name || 'Untitled'} · ${n} tolerance${n === 1 ? '' : 's'} · ${comp}`

  return (
    <JobPanelFrame kind="tolerance" subtitle={subtitle} actions={<Actions problem={problem} stoppable={stoppable} hasResult={result != null} />}>
      {job.phase === 'running' ? (
        <ToleranceProgress job={job} settings={settings} />
      ) : job.phase === 'failed' ? (
        <JobFailure title="Tolerancing Failed" reason={job.error ?? 'The engine could not run the tolerance analysis.'} />
      ) : (job.phase === 'review' || job.phase === 'done') && result ? (
        <ToleranceResults result={result} system={system} />
      ) : (
        <div className="flex flex-col">
          {result && (
            <button
              type="button"
              className="mb-2 flex cursor-default items-center gap-1.5 self-start text-[12px] text-accent hover:underline"
              onClick={() => useJobs.getState().patch('tolerance', { phase: 'review' })}
            >
              <BarChart3 className="size-3.5" />
              Show the last results
            </button>
          )}
          <ToleranceTable settings={settings} context={context} issues={issues} />
          <CompensatorSection settings={settings} system={system} />
          <CriterionSection settings={settings} system={system} merit={merit} />
          <AnalysisSection settings={settings} />
        </div>
      )}
    </JobPanelFrame>
  )
}

function Actions({ problem, stoppable, hasResult }: { problem: string | null; stoppable: boolean; hasResult: boolean }) {
  const job = useJobs((s) => s.jobs.tolerance)
  const settings = useToleranceSettings()
  const now = useNow()
  const t = useToleranceTiming(job, settings, now)
  switch (job.phase) {
    case 'running':
      return JobActions.running('tolerance', job, {
        detail: job.stopped ? 'Stopping' : stageText(job.stage, t.trialsDone, settings.trials),
        canStop: stoppable,
        stopHelp: 'End now and keep the sensitivities and trials completed so far',
      })
    case 'failed':
      return (
        <>
          <BarButton title="Back" help="Back to the tolerances, compensators and criterion" onClick={() => useJobs.getState().back('tolerance')} />
          {JobActions.start('Try Again', retryTolerance, { disabled: problem != null, help: problem ?? 'Run the tolerance analysis again, recomputed' })}
        </>
      )
    case 'review':
    case 'done':
      return (
        <>
          <BarButton
            title="Back to settings"
            help="Back to the tolerances, compensators and criterion; Show Results returns here"
            onClick={() => useJobs.getState().back('tolerance')}
          />
          {JobActions.start('Run again', retryTolerance, { disabled: problem != null, help: problem ?? 'Run the same analysis again, recomputed' })}
        </>
      )
    default:
      return (
        <>
          {hasResult && <BarButton title="Show Results" help="Show the last run's results" onClick={() => useJobs.getState().patch('tolerance', { phase: 'review' })} />}
          {JobActions.start('Run', () => startTolerance(false), { disabled: problem != null, help: problem ?? 'Run the sensitivity analysis and the Monte Carlo trials' })}
        </>
      )
  }
}

/** A secondary action of the glass bar. */
function BarButton({ title, help, onClick }: { title: string; help: string; onClick: () => void }) {
  return (
    <Button className="h-8 rounded-full px-4" title={help} onClick={onClick}>
      {title}
    </Button>
  )
}
