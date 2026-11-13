// The running Tolerance job (TolerancePanel.swift `ToleranceProgress`):
// progress, the stage ("Monte Carlo 8048 / 20000"), elapsed time, time per
// trial and time left, refreshed every second.

import { useEffect, useState } from 'react'
import type { Json, JsonObject } from '@/engine/types'
import { stageText, timing, timingText, type Timing, type ToleranceSettings } from '@/model/tolerance'
import type { JobState } from '@/state/jobs'
import { useTolerance } from '@/state/tolerance'
import { BigNumber } from '../JobPanel'
import { Hint } from './controls'

/** Trials done as the engine's live detail reports them, if it does. */
function detailTrials(detail: Json | null): number | null {
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return null
  const o = detail as JsonObject
  for (const k of ['trials_done', 'completed_trials', 'trials_completed', 'trial']) if (typeof o[k] === 'number') return o[k] as number
  return null
}

export function useToleranceTiming(job: JobState, settings: ToleranceSettings, now: number): Timing {
  const mc = useTolerance((s) => s.monteCarloStartedAt)
  const last = useTolerance((s) => s.lastSecondsPerTrial)
  const operands = useTolerance((s) => s.runRows.length)
  const t = timing({
    elapsed: job.startedAt ? (now - job.startedAt) / 1000 : 0,
    monteCarloElapsed: mc != null ? (now - mc) / 1000 : null,
    fraction: job.progress,
    mode: settings.mode,
    trials: settings.trials,
    operands,
    lastSecondsPerTrial: last,
  })
  const d = detailTrials(job.detail)
  return d != null ? { ...t, trialsDone: d } : t
}

export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(id)
  }, [ms])
  return now
}

export function ToleranceProgress({ job, settings }: { job: JobState; settings: ToleranceSettings }) {
  const now = useNow()
  const t = useToleranceTiming(job, settings, now)
  const n = settings.rows.length, c = settings.compensators.length
  return (
    <div className="flex flex-col gap-4" data-testid="tolerance-progress">
      <div className="grid grid-cols-2 gap-4">
        <BigNumber label="Progress" value={`${Math.round(Math.min(Math.max(job.progress, 0), 1) * 100)} %`} />
        <BigNumber label="Stage" value={stageText(job.stage, t.trialsDone, settings.trials)} />
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-selected" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress * 100)}>
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${Math.min(Math.max(job.progress, 0), 1) * 100}%` }} />
      </div>
      {job.startedAt != null && <p className="num text-[12px] text-secondary">{timingText((now - job.startedAt) / 1000, t)}</p>}
      <p className="text-[12px] text-secondary">
        {n} tolerance{n === 1 ? '' : 's'} · {c} compensator{c === 1 ? '' : 's'}
        {settings.mode === 'sensitivity' ? '' : ` · ${settings.trials} trials, seed ${settings.seed}`}
      </p>
      <Hint>Stop keeps the sensitivities and trials completed so far. Cancel discards the run. You can close this panel; the job continues.</Hint>
    </div>
  )
}
