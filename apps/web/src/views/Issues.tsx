import { AlertTriangle, CheckCircle2, Info, XOctagon } from 'lucide-react'
import { useView } from '@/state/view'
import { allIssues, useWorkspace } from '@/state/workspace'

export function IssuesList() {
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const validationIssues = useWorkspace((s) => s.validationIssues)
  const select = useView((s) => s.select)
  const issues = allIssues({ modelIssues, validationIssues })
  return (
    <div>
      <h3 className="mb-2 text-[13px] font-semibold">Issues</h3>
      {issues.length === 0 ? (
        <div className="flex flex-col items-center gap-1 py-4 text-center text-secondary">
          <CheckCircle2 className="size-6" />
          <span className="font-medium text-label">No Issues</span>
          <span className="text-[12px]">The engine checked the design and found nothing to report.</span>
        </div>
      ) : (
        <ul className="max-h-[50vh] space-y-1.5 overflow-y-auto" data-testid="issues-list">
          {issues.map((i, k) => (
            <li key={k} className="flex gap-2 text-[12px]">
              {i.severity === 'error' ? (
                <XOctagon className="mt-[1px] size-3.5 shrink-0 text-danger" />
              ) : i.severity === 'warning' ? (
                <AlertTriangle className="mt-[1px] size-3.5 shrink-0 text-warning" />
              ) : (
                <Info className="mt-[1px] size-3.5 shrink-0 text-secondary" />
              )}
              <div className="min-w-0">
                <p className="select-text">{i.message}</p>
                {i.surface != null && (
                  <button type="button" className="text-accent hover:underline" onClick={() => select(i.surface!)}>
                    Surface {i.surface}
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
