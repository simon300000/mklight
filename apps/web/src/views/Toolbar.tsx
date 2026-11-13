import { Menu, PanelLeft, PanelRight, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
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
import { useView, type RayColoring } from '@/state/view'
import { BACKEND_CHOICES, usePreferences, type BackendPreference } from '@/state/preferences'
import { LAYOUT_SHORTCUTS, offersXZ } from './layout/shortcuts'
import { useWorkspace } from '@/state/workspace'
import { cn, shortcutLabel } from '@/lib/utils'
import { StatusCapsule } from './StatusCapsule'
import { closePanel, commandPanel, isDetached, moveToMainWindow, moveToNewWindow, panelTitle, resetPanelLayout } from './panels/actions'
import { LAYOUT_PANEL, LENS_DATA } from '@/state/arrangement'

/** View ▸ Panel: for the focused panel (the selected tab of the window in front). */
function PanelMenu() {
  const panel = commandPanel()
  const name = panel ? panelTitle(panel) : null
  const hide = panel === LENS_DATA ? 'Hide Lens Data' : panel === LAYOUT_PANEL ? 'Hide Layout' : 'Close Pane'
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>Panel{name ? ` · ${name}` : ''}</DropdownMenuSubTrigger>
      <DropdownMenuSubContent>
        <DropdownMenuItem disabled={!panel} onSelect={() => panel && moveToNewWindow(panel)}>
          Move to New Window
        </DropdownMenuItem>
        <DropdownMenuItem disabled={!panel || !isDetached(panel)} onSelect={() => panel && moveToMainWindow(panel)}>
          Move to Main Window
        </DropdownMenuItem>
        <DropdownMenuItem disabled={!panel} onSelect={() => panel && closePanel(panel)}>
          {hide}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => resetPanelLayout()}>Reset Panel Layout</DropdownMenuItem>
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  )
}

function CommandItem({ id, title }: { id: string; title?: string }) {
  const c = command(id)
  const enabled = c.enabled ? c.enabled() : true
  return (
    <DropdownMenuItem disabled={!enabled} shortcut={c.shortcut} onSelect={() => c.run()}>
      {title ?? c.title}
    </DropdownMenuItem>
  )
}

function MenuGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>{title}</DropdownMenuSubTrigger>
      <DropdownMenuSubContent className="min-w-[15rem]">{children}</DropdownMenuSubContent>
    </DropdownMenuSub>
  )
}

/** The app menu: File, Edit, Lens, Design, Analysis and View, as the macOS
 *  menu bar has them (Commands.swift). */
function AppMenu() {
  const undo = useDocument(undoName)
  const redo = useDocument(redoName)
  const v = useView()
  const descriptors = useWorkspace((s) => s.descriptors)
  const addPane = useWorkspace((s) => s.addPane)
  const backend = usePreferences((s) => s.backend)
  const setBackend = usePreferences((s) => s.setBackend)
  const replaceSamples = (
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
  )
  return (
    <DropdownMenu>
      <Tip label="Menu">
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" aria-label="Menu">
            <Menu />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="start" className="min-w-[15rem]" data-testid="app-menu">
        <DropdownMenuLabel>File</DropdownMenuLabel>
        <CommandItem id="new" />
        <CommandItem id="samples" />
        <CommandItem id="open" />
        <CommandItem id="save" />
        <CommandItem id="saveAs" />
        <CommandItem id="welcome" />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Edit</DropdownMenuLabel>
        <CommandItem id="undo" title={undo ? `Undo ${undo}` : 'Undo'} />
        <CommandItem id="redo" title={redo ? `Redo ${redo}` : 'Redo'} />
        <DropdownMenuSeparator />
        <MenuGroup title="Lens">
          <CommandItem id="insertBefore" />
          <CommandItem id="insertAfter" />
          <CommandItem id="delete" />
          <DropdownMenuSeparator />
          <CommandItem id="stop" />
          <CommandItem id="toggleVariable" />
          <DropdownMenuSeparator />
          {replaceSamples}
          <CommandItem id="loadCatalog" />
        </MenuGroup>
        <MenuGroup title="Design">
          <CommandItem id="optimize" />
          <CommandItem id="glass" />
          <CommandItem id="tolerance" />
          <CommandItem id="solves" />
          <DropdownMenuSeparator />
          <CommandItem id="issues" />
        </MenuGroup>
        <MenuGroup title="Analysis">
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>New Analysis Pane</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {groupedAnalyses(descriptors).map((g) => (
                <div key={g.category}>
                  <DropdownMenuLabel>{categoryTitle(g.category)}</DropdownMenuLabel>
                  {g.items.map((d) => (
                    <DropdownMenuItem
                      key={d.name}
                      inset
                      disabled={!d.available}
                      title={d.unavailable_reason ?? d.description}
                      onSelect={() => useView.getState().set({ focusedPane: addPane(d.name) })}
                    >
                      {analysisTitle(d.name, d)}
                    </DropdownMenuItem>
                  ))}
                </div>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuSeparator />
          <CommandItem id="runAll" />
          <CommandItem id="cancel" />
          <DropdownMenuSeparator />
          <DropdownMenuLabel>Compute On</DropdownMenuLabel>
          <DropdownMenuRadioGroup value={backend} onValueChange={(b) => setBackend(b as BackendPreference)}>
            {BACKEND_CHOICES.map((c) => (
              <DropdownMenuRadioItem key={c.value} value={c.value} title={c.help}>
                {c.title}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </MenuGroup>
        <MenuGroup title="View">
          <CommandItem id="lensData" title={v.lensDataVisible ? 'Hide Lens Data' : 'Show Lens Data'} />
          <CommandItem id="layout" title={v.layoutVisible ? 'Hide Layout' : 'Show Layout'} />
          <PanelMenu />
          <DropdownMenuSeparator />
          {LAYOUT_SHORTCUTS.map((c) => (
            <DropdownMenuItem key={c.shortcut} shortcut={c.shortcut} disabled={c.shortcut === 'alt+mod+2' && !offersXZ()} onSelect={() => c.run()}>
              {c.title}
            </DropdownMenuItem>
          ))}
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>Ray Colors</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuRadioGroup value={v.rayColoring} onValueChange={(c) => v.set({ rayColoring: c as RayColoring })}>
                <DropdownMenuRadioItem value="field">Color by Field</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="wavelength">Color by Wavelength</DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuSeparator />
          <CommandItem id="sidebar" title={v.sidebarVisible ? 'Hide Sidebar' : 'Show Sidebar'} />
          <CommandItem id="inspector" title={v.inspectorVisible ? 'Hide Inspector' : 'Show Inspector'} />
          <CommandItem id="surfacePage" title="Surface Inspector" />
          <CommandItem id="systemPage" title="System Inspector" />
        </MenuGroup>
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
