import { AlertTriangle, CheckCircle2, Info, XOctagon } from 'lucide-react'
import { useView } from '@/state/view'
import { allIssues, fieldIndexIn, useWorkspace, type Issue } from '@/state/workspace'

const RANK = { error: 0, warning: 1, info: 2 } as const

/** Field index of an issue that names one (`field 2: …`), 0-based as the
 *  engine writes it. */
export function fieldOf(i: Issue): number | null {
  if (typeof i.field === 'number') return i.field
  return fieldIndexIn(i.message)
}

/** The message with its field numbered from 1, as everywhere in the app
 *  (the engine's `field 0: …` reads `Field 1: …`). */
export function issueText(i: Issue): string {
  const f = fieldOf(i)
  if (f == null) return i.message
  return i.message.replace(/\b(field)(\s+(?:index\s+)?)(\d+)/i, (_m, w: string, sp: string, n: string) =>
    Number(n) === f ? `${w[0].toUpperCase()}${w.slice(1)}${sp.replace(/index\s+/i, '')}${f + 1}` : _m,
  )
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
    setView({ issuesOpen: false, inspectorVisible: true })
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
            const target = i.surface != null ? `Select surface ${i.surface}` : f != null ? `Show field ${f + 1}` : undefined
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
                    <span className="block select-text text-label">{issueText(i)}</span>
                    {(i.code || i.surface != null || f != null) && (
                      <span className="flex gap-2 text-[11px] text-secondary">
                        {i.code && <span className="font-mono">{i.code}</span>}
                        {i.surface != null && <span>Surface {i.surface}</span>}
                        {f != null && <span>Field {f + 1}</span>}
                      </span>
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
