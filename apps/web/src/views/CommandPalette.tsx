// The command palette (⌘K / Ctrl+K): every menu command, adding an analysis
// pane, opening a sample and selecting a surface, by typing. The web's usual
// way to reach everything from the keyboard; the toolbar's menu stays for
// pointing.

import { Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { analysisTitle, groupedAnalyses } from '@/model/catalog'
import { SAMPLES } from '@/model/project'
import { surfaceTitle } from '@/model/system'
import { COMMANDS } from '@/state/commands'
import { useDocument } from '@/state/document'
import { openSample } from '@/state/files'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import { usePalette } from '@/state/palette'
import { cn, shortcutLabel } from '@/lib/utils'


interface Item {
  id: string
  title: string
  group: string
  shortcut?: string
  run(): void
}

function items(): Item[] {
  const out: Item[] = []
  for (const c of COMMANDS) {
    if (c.enabled && !c.enabled()) continue
    out.push({ id: `cmd:${c.id}`, title: c.title.replace(/…$/, ''), group: 'Commands', shortcut: c.shortcut, run: c.run })
  }
  const ws = useWorkspace.getState()
  for (const g of groupedAnalyses(ws.descriptors)) {
    for (const d of g.items) {
      if (!d.available) continue
      out.push({
        id: `pane:${d.name}`,
        title: `Add ${analysisTitle(d.name, d)}`,
        group: 'Analyses',
        run: () => {
          const key = useWorkspace.getState().addPane(d.name)
          useView.getState().set({ focusedPane: key })
        },
      })
    }
  }
  const sys = useDocument.getState().project.system
  sys?.surfaces?.forEach((s, i) => {
    out.push({
      id: `surf:${i}`,
      title: `Select ${surfaceTitle(sys, i)}${s.label ? ` · ${s.label}` : ''}`,
      group: 'Surfaces',
      run: () => {
        useView.getState().select(i)
        useView.getState().setPage('surface')
      },
    })
  })
  for (const s of SAMPLES) out.push({ id: `sample:${s.id}`, title: `Open Sample ${s.system.name}`, group: 'Samples', run: () => openSample(s.id) })
  return out
}

/** Every word of the query appears in the title (case-insensitive), in any order. */
function matches(title: string, q: string) {
  const t = title.toLowerCase()
  return q.toLowerCase().split(/\s+/).filter(Boolean).every((w) => t.includes(w))
}

export function CommandPalette() {
  const open = usePalette((s) => s.open)
  const setOpen = usePalette((s) => s.set)
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const list = useRef<HTMLUListElement>(null)
  const all = useMemo(() => (open ? items() : []), [open])
  const shown = useMemo(() => all.filter((i) => matches(i.title, q)).slice(0, 60), [all, q])

  useEffect(() => {
    if (open) {
      setQ('')
      setActive(0)
    }
  }, [open])
  useEffect(() => setActive(0), [q])
  useEffect(() => {
    list.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const run = (i: Item | undefined) => {
    if (!i) return
    setOpen(false)
    // Let the dialog close before the command opens anything of its own.
    requestAnimationFrame(() => i.run())
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="top-[18%] w-[min(560px,92vw)] translate-y-0 p-0" aria-describedby={undefined}>
        <DialogTitle className="sr-only">Command Palette</DialogTitle>
        <div className="flex items-center gap-2 border-b border-separator px-3">
          <Search className="size-4 text-secondary" aria-hidden />
          <input
            autoFocus
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={shown[active] ? `palette-${shown[active].id}` : undefined}
            aria-label="Search commands, analyses, surfaces and samples"
            placeholder="Search commands, analyses, surfaces, samples"
            className="h-11 min-w-0 flex-1 bg-transparent text-[14px] outline-none placeholder:text-secondary"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setActive((a) => Math.min(a + 1, shown.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setActive((a) => Math.max(a - 1, 0))
              } else if (e.key === 'Enter') {
                e.preventDefault()
                run(shown[active])
              }
            }}
          />
        </div>
        <ul ref={list} id="palette-list" role="listbox" aria-label="Results" className="max-h-[50vh] overflow-y-auto p-1.5" data-testid="command-palette">
          {shown.length === 0 && <li className="px-3 py-6 text-center text-secondary">No matches</li>}
          {shown.map((i, k) => (
            <li
              key={i.id}
              id={`palette-${i.id}`}
              role="option"
              aria-selected={k === active}
              data-index={k}
              onMouseMove={() => setActive(k)}
              onClick={() => run(i)}
              className={cn(
                'flex cursor-default items-center gap-3 rounded-md px-2.5 py-1.5 text-[13px]',
                k === active ? 'bg-accent text-on-accent' : '',
              )}
            >
              <span className="min-w-0 flex-1 truncate">{i.title}</span>
              <span className={cn('shrink-0 text-[11px]', k === active ? 'opacity-80' : 'text-secondary')}>
                {i.shortcut ? shortcutLabel(i.shortcut) : i.group}
              </span>
            </li>
          ))}
        </ul>
      </DialogContent>
    </Dialog>
  )
}
