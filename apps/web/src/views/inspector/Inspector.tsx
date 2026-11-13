import { Circle, SlidersHorizontal } from 'lucide-react'
import { Tip } from '@/components/ui/tooltip'
import { JOB_TITLES, useJobs, type JobKind } from '@/state/jobs'
import { useView, type InspectorPage } from '@/state/view'
import { cn, shortcutLabel } from '@/lib/utils'
import { GlassSubstitutionPanel } from '../jobs/GlassSubstitutionPanel'
import { JOB_ICONS } from '../jobs/JobPanel'
import { OptimizePanel } from '../jobs/OptimizePanel'
import { SolvesPanel } from '../jobs/SolvesPanel'
import { TolerancePanel } from '../jobs/TolerancePanel'
import { SurfacePage } from './SurfacePage'
import { SystemPage } from './SystemPage'

const PAGES: { id: InspectorPage; title: string; icon: React.ReactNode; shortcut: string; job?: JobKind }[] = [
  { id: 'surface', title: 'Surface', icon: <Circle />, shortcut: 'alt+mod+1' },
  { id: 'system', title: 'System', icon: <SlidersHorizontal />, shortcut: 'alt+mod+2' },
  ...(['optimize', 'glass', 'tolerance', 'solves'] as JobKind[]).map((k) => ({
    id: k as InspectorPage,
    title: JOB_TITLES[k],
    icon: JOB_ICONS[k],
    shortcut: { optimize: 'ctrl+mod+o', glass: 'ctrl+mod+g', tolerance: 'ctrl+mod+t', solves: 'ctrl+mod+l' }[k],
    job: k,
  })),
]

/** The inspector: a full-height column whose part of the toolbar holds its
 *  page tabs (`InspectorTabBar`), as in Xcode. */
export function Inspector() {
  const page = useView((s) => s.inspectorPage)
  const setPage = useView((s) => s.setPage)
  const jobs = useJobs((s) => s.jobs)
  return (
    <aside className="flex h-full w-[340px] shrink-0 flex-col border-l border-separator bg-window" aria-label="Inspector" data-testid="inspector">
      <div className="glass flex h-[52px] shrink-0 items-center justify-center gap-1 border-b border-separator" role="tablist" aria-label="Inspector pages">
        {PAGES.map((p) => (
          <Tip key={p.id} label={`${p.title} (${shortcutLabel(p.shortcut)})`}>
            <button
              type="button"
              role="tab"
              aria-selected={page === p.id}
              aria-label={p.title}
              onClick={() => setPage(p.id)}
              className={cn(
                'relative flex size-8 cursor-default items-center justify-center rounded-full text-[12px] [&_svg]:size-4',
                page === p.id ? 'bg-selected text-accent' : 'text-secondary hover:bg-hover',
              )}
            >
              {p.icon}
              {p.job && jobs[p.job].phase === 'running' && (
                <span className="absolute right-1 top-1 size-[6px] rounded-full bg-accent" aria-label="running" data-testid={`job-dot-${p.job}`} />
              )}
            </button>
          </Tip>
        ))}
      </div>
      {page === 'surface' || page === 'system' ? (
        <div className="min-h-0 flex-1 overflow-y-auto p-3">{page === 'system' ? <SystemPage /> : <SurfacePage />}</div>
      ) : (
        <div className="min-h-0 flex-1">
          {page === 'optimize' && <OptimizePanel />}
          {page === 'glass' && <GlassSubstitutionPanel />}
          {page === 'tolerance' && <TolerancePanel />}
          {page === 'solves' && <SolvesPanel />}
        </div>
      )}
    </aside>
  )
}
