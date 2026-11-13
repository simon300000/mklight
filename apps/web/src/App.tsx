import { useEffect, useState } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { MAIN } from '@/state/arrangement'
import { handleShortcut } from '@/state/commands'
import { openFile } from '@/state/files'
import { useView } from '@/state/view'
import { ImportReport } from '@/views/ImportReport'
import { Inspector } from '@/views/inspector/Inspector'
import { handleLayoutShortcut } from '@/views/layout/shortcuts'
import { DetachedWindows } from '@/views/panels/DetachedWindows'
import { PanelWindowView } from '@/views/panels/PanelArea'
import { Sidebar } from '@/views/Sidebar'
import { Toasts } from '@/views/Toasts'
import { Toolbar } from '@/views/Toolbar'
import { Welcome } from '@/views/Welcome'

/** Narrow windows and tablets (docs/WEB.md): drawer, sheet and stacked panels. */
const NARROW = '(max-width: 899px)'

export function App() {
  const sidebar = useView((s) => s.sidebarVisible)
  const inspector = useView((s) => s.inspectorVisible)
  const narrow = useView((s) => s.narrow)
  const [dropping, setDropping] = useState(false)

  // Below 900 px the sidebar and the inspector float over the content, closed
  // at first; back at full width they come back as they were.
  useEffect(() => {
    const mq = window.matchMedia(NARROW)
    let wide = { sidebarVisible: true, inspectorVisible: true }
    const apply = () => {
      const v = useView.getState()
      if (v.narrow === mq.matches) return
      if (mq.matches) {
        wide = { sidebarVisible: v.sidebarVisible, inspectorVisible: v.inspectorVisible }
        v.set({ narrow: true, sidebarVisible: false, inspectorVisible: false })
      } else v.set({ narrow: false, ...wide })
    }
    apply()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [])

  // Escape closes the drawer or the sheet.
  useEffect(() => {
    if (!narrow || (!sidebar && !inspector)) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
      useView.getState().set({ sidebarVisible: false, inspectorVisible: false })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [narrow, sidebar, inspector])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!handleShortcut(e)) handleLayoutShortcut(e)
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [])

  return (
    <TooltipProvider delayDuration={700}>
      <div
        className="flex h-full w-full overflow-hidden bg-window"
        onDragOver={(e) => {
          if ([...e.dataTransfer.items].some((i) => i.kind === 'file')) {
            e.preventDefault()
            setDropping(true)
          }
        }}
        onDragLeave={(e) => e.currentTarget === e.target && setDropping(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDropping(false)
          const f = e.dataTransfer.files[0]
          if (f) void openFile(f, null)
        }}
      >
        {sidebar && !narrow && <Sidebar />}
        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <Toolbar />
          {/* The content: a tree of splits and tab groups (docs/APP.md "Panels and windows"). */}
          <PanelWindowView windowId={MAIN} />
        </main>
        {inspector && !narrow && <Inspector />}
        {narrow && (sidebar || inspector) && (
          <div
            className="fixed inset-0 z-40 bg-black/25 dark:bg-black/45"
            data-testid="overlay-scrim"
            onClick={() => useView.getState().set({ sidebarVisible: false, inspectorVisible: false })}
          />
        )}
        {narrow && sidebar && (
          <div className="fixed inset-y-0 left-0 z-50 flex max-w-[85vw] shadow-2xl" data-testid="sidebar-drawer">
            <Sidebar />
          </div>
        )}
        {narrow && inspector && (
          <div className="fixed inset-y-0 right-0 z-50 flex max-w-full shadow-2xl [&>aside]:max-w-full" data-testid="inspector-sheet">
            <Inspector />
          </div>
        )}
        {dropping && (
          <div className="pointer-events-none fixed inset-2 z-50 flex items-center justify-center rounded-2xl border-2 border-dashed border-accent bg-accent-soft text-[15px] font-medium text-accent">
            Drop a .mklight, .zmx or .seq file to open it
          </div>
        )}
      </div>
      <Welcome />
      <ImportReport />
      <Toasts />
      <DetachedWindows />
    </TooltipProvider>
  )
}
