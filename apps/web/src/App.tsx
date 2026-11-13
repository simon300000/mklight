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

export function App() {
  const sidebar = useView((s) => s.sidebarVisible)
  const inspector = useView((s) => s.inspectorVisible)
  const [dropping, setDropping] = useState(false)

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
        {sidebar && <Sidebar />}
        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <Toolbar />
          {/* The content: a tree of splits and tab groups (docs/APP.md "Panels and windows"). */}
          <PanelWindowView windowId={MAIN} />
        </main>
        {inspector && <Inspector />}
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
