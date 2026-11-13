import { useEffect, useRef, useState } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { handleShortcut } from '@/state/commands'
import { openFile } from '@/state/files'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import { cn } from '@/lib/utils'
import { AnalysisPane, EmptyPanes } from '@/views/AnalysisPane'
import { ImportReport } from '@/views/ImportReport'
import { Inspector } from '@/views/inspector/Inspector'
import { LayoutView } from '@/views/LayoutView'
import { LensData } from '@/views/LensData'
import { Sidebar } from '@/views/Sidebar'
import { Toasts } from '@/views/Toasts'
import { Toolbar } from '@/views/Toolbar'
import { Welcome } from '@/views/Welcome'

/** A divider between two panels; drag to resize, double-click for equal shares. */
function Divider({ axis, onDrag, onReset }: { axis: 'h' | 'v'; onDrag: (delta: number) => void; onReset: () => void }) {
  const last = useRef<number | null>(null)
  return (
    <div
      role="separator"
      aria-orientation={axis === 'h' ? 'vertical' : 'horizontal'}
      className={cn('relative z-10 shrink-0 bg-separator', axis === 'h' ? 'w-px cursor-col-resize' : 'h-px cursor-row-resize')}
      onDoubleClick={onReset}
      onPointerDown={(e) => {
        ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
        last.current = axis === 'h' ? e.clientX : e.clientY
      }}
      onPointerMove={(e) => {
        if (last.current == null) return
        const p = axis === 'h' ? e.clientX : e.clientY
        onDrag(p - last.current)
        last.current = p
      }}
      onPointerUp={() => (last.current = null)}
    >
      <div className={cn('absolute', axis === 'h' ? '-left-1 -right-1 inset-y-0' : '-top-1 -bottom-1 inset-x-0')} />
    </div>
  )
}

/** The content area: the lens data across the top, the layout and the analysis
 *  panes side by side under it (the macOS default arrangement). */
function Content() {
  const v = useView()
  const panes = useWorkspace((s) => s.panes)
  const box = useRef<HTMLDivElement>(null)
  const row = useRef<HTMLDivElement>(null)
  const [paneShares, setPaneShares] = useState<number[]>([])
  const shares = panes.length === paneShares.length ? paneShares : panes.map(() => 1 / Math.max(panes.length, 1))
  const lower = v.layoutVisible || panes.length > 0

  return (
    <div ref={box} className="flex min-h-0 min-w-0 flex-1 flex-col">
      {v.lensDataVisible && (
        <div style={{ height: lower ? `${v.lensShare * 100}%` : '100%' }} className="min-h-[90px] shrink-0">
          <LensData />
        </div>
      )}
      {v.lensDataVisible && lower && (
        <Divider
          axis="v"
          onDrag={(d) => v.set({ lensShare: Math.min(0.8, Math.max(0.12, v.lensShare + d / (box.current?.clientHeight || 1))) })}
          onReset={() => v.set({ lensShare: 0.36 })}
        />
      )}
      {lower && (
        <div ref={row} className="flex min-h-0 flex-1">
          {v.layoutVisible && (
            <div style={{ width: panes.length ? `${v.layoutShare * 100}%` : '100%' }} className="min-w-[160px] shrink-0">
              <LayoutView />
            </div>
          )}
          {v.layoutVisible && panes.length > 0 && (
            <Divider
              axis="h"
              onDrag={(d) => v.set({ layoutShare: Math.min(0.85, Math.max(0.15, v.layoutShare + d / (row.current?.clientWidth || 1))) })}
              onReset={() => v.set({ layoutShare: 0.5 })}
            />
          )}
          <div className="flex min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
            {panes.map((p, k) => (
              <div key={p.key} className="flex min-w-[250px]" style={{ flex: `${shares[k]} 1 0` }}>
                {k > 0 && (
                  <Divider
                    axis="h"
                    onDrag={(d) => {
                      const w = row.current?.clientWidth || 1
                      const next = [...shares]
                      const delta = d / w
                      if (next[k - 1] + delta < 0.08 || next[k] - delta < 0.08) return
                      next[k - 1] += delta
                      next[k] -= delta
                      setPaneShares(next)
                    }}
                    onReset={() => setPaneShares(panes.map(() => 1 / panes.length))}
                  />
                )}
                <div className="min-w-0 flex-1">
                  <AnalysisPane pane={p} />
                </div>
              </div>
            ))}
            {!v.layoutVisible && panes.length === 0 && <EmptyPanes />}
          </div>
        </div>
      )}
      {!v.lensDataVisible && !lower && (
        <div className="flex flex-1 items-center justify-center gap-2 bg-content text-secondary">
          <button type="button" className="text-accent" onClick={() => v.set({ lensDataVisible: true })}>
            Show Lens Data
          </button>
          ·
          <button type="button" className="text-accent" onClick={() => v.set({ layoutVisible: true })}>
            Show Layout
          </button>
        </div>
      )}
    </div>
  )
}

export function App() {
  const sidebar = useView((s) => s.sidebarVisible)
  const inspector = useView((s) => s.inspectorVisible)
  const [dropping, setDropping] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => handleShortcut(e)
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
          <Content />
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
    </TooltipProvider>
  )
}
