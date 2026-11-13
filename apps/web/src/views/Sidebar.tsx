import { AlertTriangle, CheckCircle2, PanelLeft, Search, Table2, XOctagon, ScanLine } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Tip } from '@/components/ui/tooltip'
import { analysisTitle, categoryTitle, groupedAnalyses } from '@/model/catalog'
import { useView } from '@/state/view'
import { allIssues, useWorkspace } from '@/state/workspace'
import { cn, shortcutLabel } from '@/lib/utils'
import { AnalysisIcon } from './AnalysisIcon'

function Row({
  icon,
  title,
  open,
  dimmed,
  help,
  onClick,
  children,
  testId,
}: {
  icon: React.ReactNode
  title: string
  open?: boolean
  dimmed?: boolean
  help?: string
  onClick?: (e: React.MouseEvent) => void
  children?: React.ReactNode
  testId?: string
}) {
  return (
    <button
      type="button"
      title={help}
      data-testid={testId}
      onClick={onClick}
      className={cn(
        'flex w-full cursor-default items-center gap-2 rounded-md px-2 py-[3px] text-left text-[13px] hover:bg-hover',
        dimmed && 'opacity-45',
      )}
    >
      <span className="flex w-4 shrink-0 justify-center text-accent [&_svg]:size-[15px]">{icon}</span>
      <span className="min-w-0 flex-1 truncate">{title}</span>
      {children}
      {open && <span className="size-[5px] shrink-0 rounded-full bg-secondary" aria-label="open" />}
    </button>
  )
}

/** The sidebar (`Sidebar`): the design's lens data, layout and issues, then
 *  every engine analysis by category, searchable. A click shows the analysis
 *  in the focused pane; ⌥-click or the context menu opens a new pane. */
export function Sidebar() {
  const v = useView()
  const descriptors = useWorkspace((s) => s.descriptors)
  const panes = useWorkspace((s) => s.panes)
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const validationIssues = useWorkspace((s) => s.validationIssues)
  const issues = allIssues({ modelIssues, validationIssues })
  const errors = issues.filter((i) => i.severity === 'error').length
  const q = v.sidebarQuery.trim().toLowerCase()

  const show = (name: string, newPane: boolean) => {
    const ws = useWorkspace.getState()
    const focused = ws.panes.find((p) => p.key === v.focusedPane) ?? ws.panes.at(-1)
    if (newPane || !focused) {
      const key = ws.addPane(name)
      v.set({ focusedPane: key })
    } else if (focused.analysis !== name) {
      ws.setPaneAnalysis(focused.key, name)
    }
  }

  return (
    <aside className="flex h-full w-[220px] shrink-0 flex-col border-r border-separator bg-sidebar backdrop-blur-xl" aria-label="Sidebar">
      <div className="flex h-[52px] shrink-0 items-center px-2">
        <Tip label={`Hide Sidebar (${shortcutLabel('ctrl+mod+s')})`}>
          <Button variant="ghost" size="icon" aria-label="Hide Sidebar" onClick={() => v.set({ sidebarVisible: false })}>
            <PanelLeft />
          </Button>
        </Tip>
        <span className="ml-1 text-[13px] font-semibold tracking-tight">MKLight</span>
      </div>
      <div className="px-2 pb-2">
        <label className="flex h-6 items-center gap-1.5 rounded-md bg-selected px-2 text-secondary">
          <Search className="size-3.5" />
          <input
            type="search"
            placeholder="Search"
            aria-label="Search analyses"
            value={v.sidebarQuery}
            onChange={(e) => v.set({ sidebarQuery: e.target.value })}
            className="min-w-0 flex-1 bg-transparent text-[13px] text-label outline-none placeholder:text-secondary"
          />
        </label>
      </div>
      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {!q && (
          <>
            <h4 className="px-2 pb-0.5 pt-1 text-[11px] font-semibold text-secondary">Design</h4>
            <Row icon={<Table2 />} title="Lens Data" open={v.lensDataVisible} onClick={() => v.set({ lensDataVisible: true })} testId="sidebar-lens-data" />
            <Row icon={<ScanLine />} title="Layout" open={v.layoutVisible} onClick={() => v.set({ layoutVisible: true })} testId="sidebar-layout" />
            <Row
              icon={issues.length === 0 ? <CheckCircle2 /> : errors ? <XOctagon className="text-danger" /> : <AlertTriangle className="text-warning" />}
              title="Issues"
              onClick={() => v.set({ issuesOpen: true })}
            >
              {issues.length > 0 && <span className="num text-[12px] text-secondary">{issues.length}</span>}
            </Row>
          </>
        )}
        {groupedAnalyses(descriptors).map((g) => {
          const items = g.items.filter((d) => !q || analysisTitle(d.name, d).toLowerCase().includes(q) || d.name.includes(q))
          if (!items.length) return null
          return (
            <section key={g.category}>
              <h4 className="px-2 pb-0.5 pt-3 text-[11px] font-semibold text-secondary">{categoryTitle(g.category)}</h4>
              {items.map((d) => {
                const open = panes.find((p) => p.analysis === d.name)
                return (
                  <ContextMenu key={d.name}>
                    <ContextMenuTrigger asChild>
                      <div>
                        <Row
                          icon={<AnalysisIcon name={d.name} />}
                          title={analysisTitle(d.name, d)}
                          open={!!open}
                          dimmed={!d.available}
                          help={d.available ? d.description : (d.unavailable_reason ?? undefined)}
                          testId={`sidebar-analysis-${d.name}`}
                          onClick={(e) => d.available && show(d.name, e.altKey)}
                        />
                      </div>
                    </ContextMenuTrigger>
                    <ContextMenuContent>
                      <ContextMenuItem disabled={!d.available} onSelect={() => show(d.name, true)}>
                        Open in New Pane
                      </ContextMenuItem>
                      {open && <ContextMenuItem onSelect={() => useWorkspace.getState().removePane(open.key)}>Close Pane</ContextMenuItem>}
                    </ContextMenuContent>
                  </ContextMenu>
                )
              })}
            </section>
          )
        })}
      </nav>
    </aside>
  )
}
