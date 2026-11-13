import { Menu, PanelLeft, PanelRight, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { analysisTitle, groupedAnalyses, categoryTitle } from '@/model/catalog'
import { SAMPLES } from '@/model/project'
import { command } from '@/state/commands'
import { documentTitle, isDirty, redoName, undoName, useDocument } from '@/state/document'
import { replaceWithSample } from '@/state/files'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import { cn, shortcutLabel } from '@/lib/utils'
import { StatusCapsule } from './StatusCapsule'

function CommandItem({ id, title }: { id: string; title?: string }) {
  const c = command(id)
  const enabled = c.enabled ? c.enabled() : true
  return (
    <DropdownMenuItem disabled={!enabled} shortcut={c.shortcut} onSelect={() => c.run()}>
      {title ?? c.title}
    </DropdownMenuItem>
  )
}

/** The app menu: File, Edit, Lens and View, as the macOS menu bar has them. */
function AppMenu() {
  const undo = useDocument(undoName)
  const redo = useDocument(redoName)
  const v = useView()
  return (
    <DropdownMenu>
      <Tip label="Menu">
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" aria-label="Menu">
            <Menu />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="start" className="min-w-[15rem]">
        <DropdownMenuLabel>File</DropdownMenuLabel>
        <CommandItem id="new" />
        <CommandItem id="samples" />
        <CommandItem id="open" />
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Replace with Sample</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            {SAMPLES.map((s) => (
              <DropdownMenuItem key={s.id} onSelect={() => replaceWithSample(s.id)}>
                {s.system.name}
              </DropdownMenuItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <CommandItem id="save" />
        <CommandItem id="saveAs" />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Edit</DropdownMenuLabel>
        <CommandItem id="undo" title={undo ? `Undo ${undo}` : 'Undo'} />
        <CommandItem id="redo" title={redo ? `Redo ${redo}` : 'Redo'} />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Lens</DropdownMenuLabel>
        <CommandItem id="insertBefore" />
        <CommandItem id="insertAfter" />
        <CommandItem id="delete" />
        <CommandItem id="stop" />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>View</DropdownMenuLabel>
        <CommandItem id="lensData" title={v.lensDataVisible ? 'Hide Lens Data' : 'Show Lens Data'} />
        <CommandItem id="layout" title={v.layoutVisible ? 'Hide Layout' : 'Show Layout'} />
        <CommandItem id="sidebar" title={v.sidebarVisible ? 'Hide Sidebar' : 'Show Sidebar'} />
        <CommandItem id="inspector" title={v.inspectorVisible ? 'Hide Inspector' : 'Show Inspector'} />
        <CommandItem id="issues" />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Design</DropdownMenuLabel>
        <CommandItem id="optimize" />
        <CommandItem id="glass" />
        <CommandItem id="tolerance" />
        <CommandItem id="solves" />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Analysis</DropdownMenuLabel>
        <CommandItem id="runAll" />
        <CommandItem id="cancel" />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function AddAnalysisMenu() {
  const descriptors = useWorkspace((s) => s.descriptors)
  const addPane = useWorkspace((s) => s.addPane)
  return (
    <DropdownMenu>
      <Tip label="Add Analysis">
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" aria-label="Add Analysis">
            <Plus />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end">
        {groupedAnalyses(descriptors).map((g) => (
          <DropdownMenuSub key={g.category}>
            <DropdownMenuSubTrigger>{categoryTitle(g.category)}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {g.items.map((d) => (
                <DropdownMenuItem
                  key={d.name}
                  disabled={!d.available}
                  title={d.unavailable_reason ?? d.description}
                  onSelect={() => {
                    const key = addPane(d.name)
                    useView.getState().set({ focusedPane: key })
                  }}
                >
                  {analysisTitle(d.name, d)}
                </DropdownMenuItem>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function Toolbar() {
  const title = useDocument(documentTitle)
  const dirty = useDocument(isDirty)
  const imported = useDocument((s) => s.file?.imported ?? false)
  const sidebarVisible = useView((s) => s.sidebarVisible)
  const inspectorVisible = useView((s) => s.inspectorVisible)
  const set = useView((s) => s.set)
  return (
    <header className="glass relative z-20 flex h-[52px] shrink-0 items-center gap-1 border-b border-separator px-2">
      {!sidebarVisible && (
        <Tip label={`Show Sidebar (${shortcutLabel('ctrl+mod+s')})`}>
          <Button variant="ghost" size="icon" aria-label="Show Sidebar" onClick={() => set({ sidebarVisible: true })}>
            <PanelLeft />
          </Button>
        </Tip>
      )}
      <AppMenu />
      <div className="ml-1 flex min-w-0 flex-col leading-tight">
        <span className="truncate text-[13px] font-semibold" data-testid="document-title">
          {title}
        </span>
        <span className="truncate text-[11px] text-secondary">
          {imported ? 'Imported' : dirty ? 'Edited' : 'Saved'}
        </span>
      </div>
      <div className="flex flex-1 justify-center px-2">
        <StatusCapsule />
      </div>
      <AddAnalysisMenu />
      <Tip label={`${inspectorVisible ? 'Hide' : 'Show'} Inspector (${shortcutLabel('ctrl+mod+i')})`}>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Inspector"
          aria-pressed={inspectorVisible}
          className={cn(inspectorVisible && 'text-accent')}
          onClick={() => set({ inspectorVisible: !inspectorVisible })}
        >
          <PanelRight />
        </Button>
      </Tip>
    </header>
  )
}
