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
import { engineText, useT } from '@/i18n'
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
  const t = useT()
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

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const issues = useMemo(() => settingsIssues(settings, context), [settings, context, t])
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
    ? t('tolerance.panel.notChecked')
    : modelIssues.length
      ? t('tolerance.panel.fixDesign')
      : runProblem(settings, context)

  const n = settings.rows.length, c = settings.compensators.length
  const comp = c === 0 ? t('tolerance.panel.noCompensator') : c === 1 ? compensatorTitle(settings.compensators[0].param, system).toLowerCase() : t('tolerance.panel.compensatorsCount', { count: c })
  const subtitle = t('tolerance.panel.subtitle', { name: system.name || t('tolerance.panel.untitled'), tolerances: t('tolerance.panel.tolerances', { count: n }), compensators: comp })

  return (
    <JobPanelFrame kind="tolerance" subtitle={subtitle} actions={<Actions problem={problem} stoppable={stoppable} hasResult={result != null} />}>
      {job.phase === 'running' ? (
        <ToleranceProgress job={job} settings={settings} />
      ) : job.phase === 'failed' ? (
        <JobFailure title={t('tolerance.panel.failedTitle')} reason={job.error ? engineText(job.error) : t('tolerance.panel.failedReason')} />
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
              {t('tolerance.panel.showLast')}
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
  const t = useT()
  const job = useJobs((s) => s.jobs.tolerance)
  const settings = useToleranceSettings()
  const now = useNow()
  const timing = useToleranceTiming(job, settings, now)
  switch (job.phase) {
    case 'running':
      return JobActions.running('tolerance', job, {
        detail: job.stopped ? t('tolerance.panel.stopping') : stageText(job.stage, timing.trialsDone, settings.trials),
        canStop: stoppable,
        stopHelp: t('tolerance.panel.stopHelp'),
      })
    case 'failed':
      return (
        <>
          <BarButton title={t('tolerance.panel.back')} help={t('tolerance.panel.backHelp')} onClick={() => useJobs.getState().back('tolerance')} />
          {JobActions.start(t('tolerance.panel.tryAgain'), retryTolerance, { disabled: problem != null, help: problem ?? t('tolerance.panel.tryAgainHelp') })}
        </>
      )
    case 'review':
    case 'done':
      return (
        <>
          <BarButton
            title={t('tolerance.panel.backToSettings')}
            help={t('tolerance.panel.backToSettingsHelp')}
            onClick={() => useJobs.getState().back('tolerance')}
          />
          {JobActions.start(t('tolerance.panel.runAgain'), retryTolerance, { disabled: problem != null, help: problem ?? t('tolerance.panel.runAgainHelp') })}
        </>
      )
    default:
      return (
        <>
          {hasResult && <BarButton title={t('tolerance.panel.showResults')} help={t('tolerance.panel.showResultsHelp')} onClick={() => useJobs.getState().patch('tolerance', { phase: 'review' })} />}
          {JobActions.start(t('tolerance.panel.run'), () => startTolerance(false), { disabled: problem != null, help: problem ?? t('tolerance.panel.runHelp') })}
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
