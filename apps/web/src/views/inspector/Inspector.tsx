import { Circle, SlidersHorizontal } from 'lucide-react'
import { Tip } from '@/components/ui/tooltip'
import { useView, type InspectorPage } from '@/state/view'
import { cn, shortcutLabel } from '@/lib/utils'
import { SurfacePage } from './SurfacePage'
import { SystemPage } from './SystemPage'

const PAGES: { id: InspectorPage; title: string; icon: React.ReactNode; shortcut: string }[] = [
  { id: 'surface', title: 'Surface', icon: <Circle />, shortcut: 'mod+1' },
  { id: 'system', title: 'System', icon: <SlidersHorizontal />, shortcut: 'alt+mod+1' },
]

/** The inspector: a full-height column whose part of the toolbar holds its
 *  page tabs (`InspectorTabBar`), as in Xcode. */
export function Inspector() {
  const page = useView((s) => s.inspectorPage)
  const setPage = useView((s) => s.setPage)
  return (
    <aside className="flex h-full w-[320px] shrink-0 flex-col border-l border-separator bg-window" aria-label="Inspector" data-testid="inspector">
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
                'flex h-7 cursor-default items-center gap-1.5 rounded-full px-3 text-[12px] [&_svg]:size-3.5',
                page === p.id ? 'bg-selected font-medium text-accent' : 'text-secondary hover:bg-hover',
              )}
            >
              {p.icon}
              {p.title}
            </button>
          </Tip>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">{page === 'system' ? <SystemPage /> : <SurfacePage />}</div>
    </aside>
  )
}
