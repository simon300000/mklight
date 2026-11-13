// The separate windows' content, rendered from the main window's React tree
// into each popup through a portal (see windows.ts for why that is the
// browser's equivalent of the macOS app's separate windows).

import { Plus, PanelTopClose } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { createPortal } from 'react-dom'
import { Button } from '@/components/ui/button'
import { analysisTitle, categoryTitle, groupedAnalyses } from '@/model/catalog'
import { handleShortcut } from '@/state/commands'
import { useArrangement } from '@/state/arrangement'
import { useDocument } from '@/state/document'
import { useWorkspace } from '@/state/workspace'
import { handleLayoutShortcut } from '@/views/layout/shortcuts'
import { addPaneInWindow, dockWindow, syncPopups, windowTitle } from './actions'
import { WindowContext } from './context'
import { Menu, MenuContent, MenuItem, MenuSub, MenuSubContent, MenuSubTrigger, MenuTrigger } from './Menu'
import { PanelWindowView } from './PanelArea'
import { usePopups, type Popup } from './windows'

/** A separate window's toolbar: the design's name, Add Analysis and Move to Main Window. */
function PopupToolbar({ id }: { id: string }) {
  const name = useDocument((s) => s.project.system?.name || 'Untitled')
  const descriptors = useWorkspace((s) => s.descriptors)
  return (
    <div className="flex h-10 shrink-0 items-center gap-2 border-b border-separator bg-window px-3">
      <span className="min-w-0 flex-1 truncate text-[12px] text-secondary">{name}</span>
      <Menu>
        <MenuTrigger asChild>
          <Button variant="ghost" size="sm" aria-label="Add Analysis">
            <Plus /> Add Analysis
          </Button>
        </MenuTrigger>
        <MenuContent align="end">
          {groupedAnalyses(descriptors).map((g) => (
            <MenuSub key={g.category}>
              <MenuSubTrigger>{categoryTitle(g.category)}</MenuSubTrigger>
              <MenuSubContent>
                {g.items.map((d) => (
                  <MenuItem key={d.name} disabled={!d.available} onSelect={() => addPaneInWindow(d.name, id)}>
                    {analysisTitle(d.name, d)}
                  </MenuItem>
                ))}
              </MenuSubContent>
            </MenuSub>
          ))}
        </MenuContent>
      </Menu>
      <Button variant="ghost" size="sm" onClick={() => dockWindow(id)} title="Move this window's panels back into the main window">
        <PanelTopClose /> Move to Main Window
      </Button>
    </div>
  )
}

function PopupContent({ popup }: { popup: Popup }) {
  const title = useArrangement(() => windowTitle(popup.id))
  const name = useDocument((s) => s.project.system?.name)
  useEffect(() => {
    popup.win.document.title = title
  }, [popup, title, name])
  // The app's keyboard shortcuts work in every window of the design.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!handleShortcut(e)) handleLayoutShortcut(e)
    }
    popup.win.addEventListener('keydown', onKey, true)
    return () => popup.win.removeEventListener('keydown', onKey, true)
  }, [popup])
  const ctx = useMemo(() => ({ id: popup.id, win: popup.win, portal: popup.win.document.body }), [popup])
  return (
    <WindowContext.Provider value={ctx}>
      <div className="flex h-full w-full flex-col overflow-hidden bg-window text-label">
        <PopupToolbar id={popup.id} />
        <PanelWindowView windowId={popup.id} />
      </div>
    </WindowContext.Provider>
  )
}

/** Opens a popup for each separate window and renders its panels there. */
export function DetachedWindows() {
  const popups = usePopups((s) => s.open)
  const windows = useArrangement((s) => s.arr.windows.map((w) => w.id).join(' '))
  // Windows that appear without a user action (a project opening) open here;
  // moves open theirs at once, inside the click or drop.
  useEffect(() => {
    syncPopups()
  }, [windows])
  return (
    <>
      {Object.values(popups).map((p) => createPortal(<PopupContent popup={p} />, p.mount, p.id))}
    </>
  )
}
