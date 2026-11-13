// Apply Solves (the SolvesPanel of apps/macos/Sources/MKLightUI/Views/Jobs/JobPanel.swift):
// a dry run of the design's solves (now → solved, residual), re-run whenever the
// design changes, then Apply to Design as one undoable step.

import { ArrowRight, Sigma } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { sig } from '@/model/format'
import { solvesOf } from '@/model/params'
import { useDocument } from '@/state/document'
import { useJobs } from '@/state/jobs'
import { METHOD_TITLES, solveRows, solveWarnings, useSolves } from '@/state/solves'
import { useWorkspace } from '@/state/workspace'
import { cn } from '@/lib/utils'
import { AppliedNotice, JobActions, JobFailure, JobPanelFrame } from './JobPanel'

const short = (v: number | null) => (v == null ? '—' : sig(v, 6))

export function SolvesPanel() {
  const system = useDocument((s) => s.project.system)
  const job = useJobs((s) => s.jobs.solves)
  const snapshot = useWorkspace((s) => s.snapshot)
  const pending = useWorkspace((s) => s.snapshotPending)
  const appliedReport = useSolves((s) => s.appliedReport)
  const committing = useSolves((s) => s.committing)
  const count = solvesOf(system).length
  const committed = job.phase === 'done'

  // The dry run follows the design until it is applied (and starts again after Done).
  useEffect(() => {
    if (!count || !snapshot) return
    const j = useJobs.getState().jobs.solves
    if (j.phase === 'running' || j.phase === 'done') return
    if (j.snapshotHash === snapshot.hash && j.phase !== 'setup') return
    useSolves.getState().dryRun()
  }, [count, snapshot, job.phase])

  // A panel reopened after a commit starts over with a new dry run.
  useEffect(() => {
    if (useJobs.getState().jobs.solves.phase === 'done' && solvesOf(useDocument.getState().project.system).length) useSolves.getState().done()
  }, [])

  const report = committed ? appliedReport : job.result?.json
  const rows = useMemo(() => solveRows(report, system), [report, system])
  const warnings = solveWarnings(report)
  const changes = rows.filter((r) => r.changed).length

  let body: React.ReactNode
  if (!count) {
    body = (
      <div className="flex flex-col items-center gap-2 px-4 py-10 text-center" data-testid="no-solves">
        <Sigma className="size-8 text-secondary" />
        <h3 className="text-[15px] font-semibold">No Solves</h3>
        <p className="text-[12px] text-secondary">
          Add solves in the surface inspector: marginal or chief ray height on a thickness, angle or F-number on a curvature, pickups.
        </p>
      </div>
    )
  } else if (job.phase === 'failed') {
    body = <JobFailure title="Solving Failed" reason={job.error ?? 'The solves could not be evaluated.'} />
  } else if (job.phase === 'review' || job.phase === 'done') {
    body = (
      <div className="flex flex-col gap-2 text-[13px]" data-testid="solves-report">
        {committed && <AppliedNotice />}
        <p className="text-[12px] text-secondary">
          {committed ? 'The design now holds these solved values.' : 'Dry run: the values the solves would set. Nothing has changed yet.'}
        </p>
        <div className="select-text rounded-lg bg-content shadow-[0_0_0_0.5px_var(--separator)]">
          {rows.map((r, i) => (
            <div key={r.id} className={cn('px-2.5 py-1.5', i > 0 && 'border-t border-separator')} data-testid={`solve-row-${r.id}`}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-medium">{r.target}</span>
                <span className="truncate text-[11px] text-secondary">{r.kind}</span>
              </div>
              <div className="num flex items-center gap-1.5 text-[12px]">
                <span>{short(r.previous)}</span>
                <ArrowRight className="size-3 text-secondary" aria-label="to" />
                <span className={cn(r.changed && 'font-semibold')}>{short(r.value)}</span>
                <span className="ml-auto text-[11px] text-secondary" title={r.method ? `Method: ${METHOD_TITLES[r.method] ?? r.method}` : undefined}>
                  residual {sig(r.residual, 3)}
                </span>
              </div>
            </div>
          ))}
        </div>
        {warnings.map((w, k) => (
          <p key={k} className="text-[12px] text-warning" role="note">
            {w}
          </p>
        ))}
      </div>
    )
  } else {
    body = (
      <p className="py-10 text-center text-[12px] text-secondary" role="status">
        Solving…
      </p>
    )
  }

  let actions: React.ReactNode = null
  if (count) {
    if (job.phase === 'failed') actions = JobActions.failed('solves', { back: false })
    else if (committed) actions = JobActions.done('solves', 'Back to the design’s solves')
    else
      actions = JobActions.review('solves', () => void useSolves.getState().commit(), {
        discard: false,
        applyDisabled: job.phase !== 'review' || changes === 0 || committing || pending || job.snapshotHash !== snapshot?.hash,
      })
  }
  return (
    <JobPanelFrame kind="solves" subtitle={`${count} solve${count === 1 ? '' : 's'} in ${system.name || 'the design'}`} actions={actions}>
      {body}
    </JobPanelFrame>
  )
}
