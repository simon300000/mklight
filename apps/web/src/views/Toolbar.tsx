import { Languages, Menu, PanelLeft, PanelRight, Plus } from 'lucide-react'
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
import { CommandPalette } from './CommandPalette'
import { closePanel, commandPanel, isDetached, moveToMainWindow, moveToNewWindow, panelTitle, resetPanelLayout } from './panels/actions'
import { LAYOUT_PANEL, LENS_DATA } from '@/state/arrangement'
import { LANGUAGES, engineText, useI18n, useT, type LanguageChoice } from '@/i18n'

/** Language: the system's (default) or a fixed one; switches live. Each
 *  language is named in itself. */
function LanguageMenu() {
  const t = useT()
  const choice = useI18n((s) => s.choice)
  const setChoice = useI18n((s) => s.setChoice)
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger data-testid="language-menu">
        <Languages />
        {t('app.language.menu')}
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent>
        <DropdownMenuRadioGroup value={choice} onValueChange={(c) => void setChoice(c as LanguageChoice)}>
          <DropdownMenuRadioItem value="system">{t('app.language.system')}</DropdownMenuRadioItem>
          {LANGUAGES.map((l) => (
            <DropdownMenuRadioItem key={l.value} value={l.value} lang={l.value}>
              {l.name}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  )
}

/** View ▸ Panel: for the focused panel (the selected tab of the window in front). */
function PanelMenu() {
  const t = useT()
  const panel = commandPanel()
  const name = panel ? panelTitle(panel) : null
  const hide = panel === LENS_DATA ? t('app.menu.hideLensData') : panel === LAYOUT_PANEL ? t('app.menu.hideLayout') : t('app.menu.closePane')
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>{name ? t('app.menu.panelNamed', { name }) : t('app.menu.panel')}</DropdownMenuSubTrigger>
      <DropdownMenuSubContent>
        <DropdownMenuItem disabled={!panel} onSelect={() => panel && moveToNewWindow(panel)}>
          {t('app.menu.moveToNewWindow')}
        </DropdownMenuItem>
        <DropdownMenuItem disabled={!panel || !isDetached(panel)} onSelect={() => panel && moveToMainWindow(panel)}>
          {t('app.menu.moveToMainWindow')}
        </DropdownMenuItem>
        <DropdownMenuItem disabled={!panel} onSelect={() => panel && closePanel(panel)}>
          {hide}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => resetPanelLayout()}>{t('app.menu.resetPanelLayout')}</DropdownMenuItem>
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  )
}

function CommandItem({ id, title }: { id: string; title?: string }) {
  useT()
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
  const t = useT()
  const undo = useDocument(undoName)
  const redo = useDocument(redoName)
  const v = useView()
  const descriptors = useWorkspace((s) => s.descriptors)
  const addPane = useWorkspace((s) => s.addPane)
  const backend = usePreferences((s) => s.backend)
  const setBackend = usePreferences((s) => s.setBackend)
  const replaceSamples = (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>{t('app.menu.replaceWithSample')}</DropdownMenuSubTrigger>
      <DropdownMenuSubContent>
        {SAMPLES.map((s) => (
          <DropdownMenuItem key={s.id} onSelect={() => replaceWithSample(s.id)}>
            {engineText(s.system.name)}
          </DropdownMenuItem>
        ))}
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  )
  return (
    <DropdownMenu>
      <Tip label={t('app.menu.menu')}>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" aria-label={t('app.menu.menu')}>
            <Menu />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="start" className="min-w-[15rem]" data-testid="app-menu">
        <DropdownMenuLabel>{t('app.menu.file')}</DropdownMenuLabel>
        <CommandItem id="new" />
        <CommandItem id="samples" />
        <CommandItem id="open" />
        <CommandItem id="save" />
        <CommandItem id="saveAs" />
        <CommandItem id="welcome" />
        <DropdownMenuSeparator />
        <DropdownMenuLabel>{t('app.menu.edit')}</DropdownMenuLabel>
        <CommandItem id="undo" title={undo ? t('app.menu.undoNamed', { name: undo }) : t('app.commands.undo')} />
        <CommandItem id="redo" title={redo ? t('app.menu.redoNamed', { name: redo }) : t('app.commands.redo')} />
        <DropdownMenuSeparator />
        <MenuGroup title={t('app.menu.lens')}>
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
        <MenuGroup title={t('app.menu.design')}>
          <CommandItem id="optimize" />
          <CommandItem id="glass" />
          <CommandItem id="tolerance" />
          <CommandItem id="solves" />
          <DropdownMenuSeparator />
          <CommandItem id="issues" />
        </MenuGroup>
        <MenuGroup title={t('app.menu.analysis')}>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{t('app.menu.newAnalysisPane')}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {groupedAnalyses(descriptors).map((g) => (
                <div key={g.category}>
                  <DropdownMenuLabel>{categoryTitle(g.category)}</DropdownMenuLabel>
                  {g.items.map((d) => (
                    <DropdownMenuItem
                      key={d.name}
                      inset
                      disabled={!d.available}
                      title={engineText(d.unavailable_reason ?? d.description)}
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
          <DropdownMenuLabel>{t('app.menu.computeOn')}</DropdownMenuLabel>
          <DropdownMenuRadioGroup value={backend} onValueChange={(b) => setBackend(b as BackendPreference)}>
            {BACKEND_CHOICES.map((c) => (
              <DropdownMenuRadioItem key={c.value} value={c.value} title={c.help}>
                {c.title}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </MenuGroup>
        <MenuGroup title={t('app.menu.view')}>
          <CommandItem id="lensData" title={v.lensDataVisible ? t('app.menu.hideLensData') : t('app.menu.showLensData')} />
          <CommandItem id="layout" title={v.layoutVisible ? t('app.menu.hideLayout') : t('app.menu.showLayout')} />
          <PanelMenu />
          <DropdownMenuSeparator />
          {LAYOUT_SHORTCUTS.map((c) => (
            <DropdownMenuItem key={c.shortcut} shortcut={c.shortcut} disabled={c.shortcut === 'alt+mod+x' && !offersXZ()} onSelect={() => c.run()}>
              {c.title}
            </DropdownMenuItem>
          ))}
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{t('app.menu.rayColors')}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuRadioGroup value={v.rayColoring} onValueChange={(c) => v.set({ rayColoring: c as RayColoring })}>
                <DropdownMenuRadioItem value="field">{t('app.menu.colorByField')}</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="wavelength">{t('app.menu.colorByWavelength')}</DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuSeparator />
          <CommandItem id="sidebar" title={v.sidebarVisible ? t('app.menu.hideSidebar') : t('app.menu.showSidebar')} />
          <CommandItem id="inspector" title={v.inspectorVisible ? t('app.menu.hideInspector') : t('app.menu.showInspector')} />
          <CommandItem id="surfacePage" title={t('app.menu.surfaceInspector')} />
          <CommandItem id="systemPage" title={t('app.menu.systemInspector')} />
        </MenuGroup>
        <DropdownMenuSeparator />
        <LanguageMenu />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function AddAnalysisMenu() {
  const t = useT()
  const descriptors = useWorkspace((s) => s.descriptors)
  const addPane = useWorkspace((s) => s.addPane)
  return (
    <DropdownMenu>
      <Tip label={t('app.menu.addAnalysis')}>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" aria-label={t('app.menu.addAnalysis')}>
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
                  title={engineText(d.unavailable_reason ?? d.description)}
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
  const t = useT()
  const title = useDocument(documentTitle)
  const dirty = useDocument(isDirty)
  const imported = useDocument((s) => s.file?.imported ?? false)
  const hasFile = useDocument((s) => s.file != null)
  const sidebarVisible = useView((s) => s.sidebarVisible)
  const inspectorVisible = useView((s) => s.inspectorVisible)
  const set = useView((s) => s.set)
  return (
    <header className="glass relative z-20 flex h-[52px] shrink-0 items-center gap-1 border-b border-separator px-2">
      {!sidebarVisible && (
        <Tip label={`${t('app.menu.showSidebar')} (${shortcutLabel('ctrl+mod+s')})`}>
          <Button variant="ghost" size="icon" aria-label={t('app.menu.showSidebar')} onClick={() => set({ sidebarVisible: true })}>
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
          {/* A new design or a sample was never saved: no "Saved" for it. */}
          {imported ? t('app.toolbar.imported') : dirty ? t('app.toolbar.edited') : hasFile ? t('app.toolbar.saved') : ''}
        </span>
      </div>
      <div className="flex flex-1 justify-center px-2">
        <StatusCapsule />
      </div>
      <AddAnalysisMenu />
      <Tip label={`${inspectorVisible ? t('app.menu.hideInspector') : t('app.menu.showInspector')} (${shortcutLabel('ctrl+mod+i')})`}>
        <Button
          variant="ghost"
          size="icon"
          aria-label={t('app.menu.inspector')}
          aria-pressed={inspectorVisible}
          className={cn(inspectorVisible && 'text-accent')}
          onClick={() => set({ inspectorVisible: !inspectorVisible })}
        >
          <PanelRight />
        </Button>
      </Tip>
          <CommandPalette />
    </header>
  )
}
