import { AlertTriangle, CheckCircle2, Info, XOctagon } from 'lucide-react'
import { useView } from '@/state/view'
import { allIssues, useWorkspace, type Issue } from '@/state/workspace'

const RANK = { error: 0, warning: 1, info: 2 } as const

/** Field index of an issue that names one (`field 2: …`), 0-based. */
function fieldOf(i: Issue): number | null {
  const f = (i as Issue & { field?: number | null }).field
  if (typeof f === 'number') return f
  const m = /\bfield\s+(\d+)/i.exec(i.message)
  // Messages count fields from 1.
  return m ? Number(m[1]) - 1 : null
}

/** The design's validation issues, errors first; choosing one selects the
 *  surface it concerns (or shows the fields) and closes the list. */
export function IssuesList() {
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const validationIssues = useWorkspace((s) => s.validationIssues)
  const select = useView((s) => s.select)
  const setPage = useView((s) => s.setPage)
  const setView = useView((s) => s.set)
  const issues = allIssues({ modelIssues, validationIssues })
    .map((i, k) => ({ i, k }))
    .sort((a, b) => RANK[a.i.severity] - RANK[b.i.severity] || a.k - b.k)
    .map((x) => x.i)
  const errors = issues.filter((i) => i.severity === 'error').length
  const warnings = issues.filter((i) => i.severity === 'warning').length
  const summary = [errors ? `${errors} error${errors === 1 ? '' : 's'}` : '', warnings ? `${warnings} warning${warnings === 1 ? '' : 's'}` : ''].filter(Boolean).join(', ')
  const choose = (i: Issue) => {
    const f = fieldOf(i)
    if (i.surface != null) {
      select(i.surface)
      setPage('surface')
    } else if (f != null) setPage('system')
    else return
    setView({ issuesOpen: false })
  }
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-[13px] font-semibold">Issues</h3>
        {summary && <span className="text-[11px] text-secondary">{summary}</span>}
      </div>
      {issues.length === 0 ? (
        <div className="flex flex-col items-center gap-1 py-4 text-center text-secondary">
          <CheckCircle2 className="size-6" />
          <span className="font-medium text-label">No Issues</span>
          <span className="text-[12px]">The engine checked the design and found nothing to report.</span>
        </div>
      ) : (
        <ul className="-mx-1 max-h-[50vh] space-y-0.5 overflow-y-auto" data-testid="issues-list">
          {issues.map((i, k) => {
            const f = fieldOf(i)
            const target = i.surface != null ? `Select surface ${i.surface}` : f != null ? `Show the fields` : undefined
            return (
              <li key={k}>
                <button
                  type="button"
                  disabled={!target}
                  title={target}
                  onClick={() => choose(i)}
                  className="flex w-full cursor-default gap-2 rounded-md px-1 py-1 text-left text-[12px] enabled:hover:bg-hover"
                >
                  {i.severity === 'error' ? (
                    <XOctagon className="mt-[1px] size-3.5 shrink-0 text-danger" aria-label="error" />
                  ) : i.severity === 'warning' ? (
                    <AlertTriangle className="mt-[1px] size-3.5 shrink-0 text-warning" aria-label="warning" />
                  ) : (
                    <Info className="mt-[1px] size-3.5 shrink-0 text-secondary" aria-label="info" />
                  )}
                  <span className="min-w-0">
                    <span className="block select-text text-label">{i.message}</span>
                    {(i.surface != null || f != null) && (
                      <span className="block text-[11px] text-secondary">{i.surface != null ? `Surface ${i.surface}` : `Field ${f! + 1}`}</span>
                    )}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
