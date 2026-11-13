import { AlertTriangle, ChevronDown, Ellipsis, Play, RefreshCw, Settings2, Square, X, Zap } from 'lucide-react'
import { Component, useMemo, type ReactNode } from 'react'
import type { AnalysisResult } from '@/engine/types'
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
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { analysisTitle, categoryTitle, groupedAnalyses } from '@/model/catalog'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { descriptorFor, effectiveSettings, isStale, useWorkspace, type Pane } from '@/state/workspace'
import { cn, shortcutLabel } from '@/lib/utils'
import { AnalysisIcon } from './AnalysisIcon'
import { PanelHeader } from './PanelHeader'
import { SettingsForm, settingsSummary } from './SettingsForm'
import { keyNumbers, ResultView } from './results'

/** Unknown or malformed results are shown as data, never as an error page. */
class RenderGuard extends Component<{ children: ReactNode; resetKey: unknown }, { error: string | null; key: unknown }> {
  state = { error: null as string | null, key: undefined as unknown }
  static getDerivedStateFromError(e: Error) {
    return { error: e.message }
  }
  static getDerivedStateFromProps(p: { resetKey: unknown }, s: { key: unknown }) {
    return p.resetKey !== s.key ? { error: null, key: p.resetKey } : null
  }
  render() {
    if (this.state.error) return <div className="p-3 text-[12px] text-secondary">This result could not be drawn: {this.state.error}</div>
    return this.props.children
  }
}

/** An analysis pane (`AnalysisPaneView`): header with title, analysis menu and
 *  a state or settings summary, and a cluster of Run (Cancel while running),
 *  automatic update, Settings and More; the plot; a footer with key numbers
 *  and a backend badge that opens the run details. */
export function AnalysisPane({ pane }: { pane: Pane }) {
  const ws = useWorkspace()
  const system = useDocument((s) => s.project.system)
  const focused = useView((s) => s.focusedPane === pane.key)
  const setView = useView((s) => s.set)
  const d = descriptorFor(ws.descriptors, pane.analysis)
  const run = ws.runs[pane.key]
  const settings = effectiveSettings(d, pane.settings)
  const stale = isStale(ws, run, pane.analysis, pane.settings)
  const running = run?.status === 'running'
  const result = run?.result
  const title = analysisTitle(pane.analysis, d)
  const summary = settingsSummary(d?.default_settings ?? null, pane.settings)
  const caption = running
    ? run.stage
      ? `${run.stage}…`
      : 'Updating…'
    : run?.status === 'failed'
      ? 'Failed'
      : !result
        ? pane.auto
          ? 'Waiting'
          : 'Not run'
        : stale
          ? 'Design changed'
          : summary || 'Current'

  const keys = useMemo(() => {
    if (!result) return []
    try {
      return keyNumbers(pane.analysis, result)
    } catch {
      return []
    }
  }, [pane.analysis, result])

  return (
    <section
      className={cn('@container flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-content', focused && 'shadow-[inset_0_2px_0_var(--accent)]')}
      aria-label={title}
      data-testid={`pane-${pane.analysis}`}
      onPointerDown={() => !focused && setView({ focusedPane: pane.key })}
    >
      <PanelHeader
        icon={<AnalysisIcon name={pane.analysis} />}
        focused={focused}
        title={title}
        caption={caption}
        titleSlot={
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button type="button" className="flex min-w-0 cursor-default items-center gap-0.5 rounded px-0.5 hover:bg-hover" aria-label={`${title}, choose analysis`}>
                <h2 className="truncate text-[13px] font-semibold">{title}</h2>
                <ChevronDown className="size-3 shrink-0 text-secondary" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              {groupedAnalyses(ws.descriptors).map((g) => (
                <DropdownMenuSub key={g.category}>
                  <DropdownMenuSubTrigger>{categoryTitle(g.category)}</DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {g.items.map((x) => (
                      <DropdownMenuItem key={x.name} disabled={!x.available} onSelect={() => ws.setPaneAnalysis(pane.key, x.name)}>
                        {analysisTitle(x.name, x)}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        }
      >
        {running ? (
          <Tip label={`Cancel (${shortcutLabel('mod+Period')})`}>
            <Button variant="ghost" size="iconSm" aria-label="Cancel" onClick={() => ws.cancel(pane.key)}>
              <Square />
            </Button>
          </Tip>
        ) : (
          <Tip label={`Run (${shortcutLabel('mod+r')})`}>
            <Button variant="ghost" size="iconSm" aria-label="Run" onClick={() => ws.run(pane.key, { fresh: true })}>
              <Play />
            </Button>
          </Tip>
        )}
        <Tip label={pane.auto ? 'Automatic update is on' : 'Automatic update is off'}>
          <Button
            variant="ghost"
            size="iconSm"
            aria-label="Automatic update"
            aria-pressed={pane.auto}
            className={cn('@max-[340px]:hidden', pane.auto && 'text-accent')}
            onClick={() => ws.updatePane(pane.key, { auto: !pane.auto })}
          >
            <Zap />
          </Button>
        </Tip>
        <Tip label="Settings">
          <Button
            variant="ghost"
            size="iconSm"
            aria-label="Settings"
            aria-pressed={pane.settingsOpen}
            className={cn('@max-[340px]:hidden', pane.settingsOpen && 'bg-selected')}
            onClick={() => ws.updatePane(pane.key, { settingsOpen: !pane.settingsOpen })}
          >
            <Settings2 />
          </Button>
        </Tip>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="iconSm" aria-label="More">
              <Ellipsis />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>{title}</DropdownMenuLabel>
            <DropdownMenuItem onSelect={() => ws.run(pane.key, { fresh: true })}>Run</DropdownMenuItem>
            <DropdownMenuItem onSelect={() => ws.updatePane(pane.key, { auto: !pane.auto })}>
              {pane.auto ? 'Turn Off Automatic Update' : 'Turn On Automatic Update'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => ws.updatePane(pane.key, { settingsOpen: !pane.settingsOpen })}>
              {pane.settingsOpen ? 'Hide Settings' : 'Show Settings'}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => ws.updatePane(pane.key, { settings: {} })}>Reset Settings</DropdownMenuItem>
            <DropdownMenuItem onSelect={() => ws.addPane(pane.analysis, { settings: pane.settings })}>Duplicate Pane</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => ws.removePane(pane.key)}>Close Pane</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </PanelHeader>
      {running && (
        <div className="relative h-[2px] shrink-0 overflow-hidden bg-selected" role="progressbar" aria-valuenow={Math.round(run.progress * 100)}>
          <div className="h-full bg-accent transition-[width]" style={{ width: `${Math.max(run.progress, 0.04) * 100}%` }} />
        </div>
      )}
      {pane.settingsOpen && d && (
        <SettingsForm
          schema={d.settings_schema}
          defaults={d.default_settings}
          value={pane.settings}
          system={system}
          onChange={(v) => ws.updatePane(pane.key, { settings: v })}
          onDone={() => ws.updatePane(pane.key, { settingsOpen: false })}
        />
      )}
      <div className="relative min-h-0 flex-1">
        {result && (
          <div className={cn('absolute inset-0', stale && !running && 'pointer-events-none opacity-50 blur-[1.5px] saturate-50', running && stale && 'updating')}>
            <RenderGuard resetKey={result}>
              <ResultView analysis={pane.analysis} result={result} system={system} />
            </RenderGuard>
          </div>
        )}
        {stale && !running && (
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <div className="glass-popover flex flex-col items-center gap-2 rounded-2xl px-5 py-3 text-center" data-testid="stale-card">
              <span className="font-medium">Design changed</span>
              <Button variant="prominent" className="rounded-full" onClick={() => ws.run(pane.key, { fresh: true })}>
                <RefreshCw /> Update
              </Button>
            </div>
          </div>
        )}
        {run?.status === 'failed' && (
          <div className="absolute inset-x-0 bottom-0 m-3 rounded-lg border border-separator bg-content p-3 text-[12px]" role="alert">
            <p className="mb-1 font-medium text-danger">{title} failed</p>
            <p className="select-text whitespace-pre-wrap text-secondary">{run.error}</p>
            <Button size="sm" className="mt-2" onClick={() => ws.run(pane.key, { fresh: true })}>
              Try Again
            </Button>
          </div>
        )}
        {!result && run?.status !== 'failed' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-secondary">
            {running ? (
              <span>Computing {Math.round((run?.progress ?? 0) * 100)}%</span>
            ) : (
              <>
                <span>{pane.auto ? 'Waiting for the design' : 'Runs on request'}</span>
                {!pane.auto && (
                  <Button variant="prominent" className="rounded-full" onClick={() => ws.run(pane.key, { fresh: true })}>
                    <Play /> Run
                  </Button>
                )}
              </>
            )}
          </div>
        )}
      </div>
      <footer className="flex h-7 shrink-0 items-center gap-3 border-t border-separator px-2.5 text-[11px]">
        <div className="flex min-w-0 flex-1 gap-3 overflow-hidden" data-testid="key-numbers">
          {keys.map((k) => (
            <span key={k.label} className="num shrink-0 truncate" title={k.help}>
              <span className="text-secondary">{k.label}</span> {k.value}
            </span>
          ))}
        </div>
        {result && <BackendBadge result={result} settingsShown={settings} />}
      </footer>
    </section>
  )
}

/** CPU · f64 neutral, GPU · f32 in the accent, an orange mark for warnings;
 *  opens the run details. */
function BackendBadge({ result }: { result: AnalysisResult; settingsShown: unknown }) {
  const ri = result.runInfo
  const backend = String(ri.backend ?? '')
  const gpu = backend.startsWith('gpu')
  const precision = String(ri.precision ?? (gpu ? 'f32' : 'f64')).replace(/_.*/, '')
  const label = backend === 'mock' ? 'Mock' : `${gpu ? 'GPU' : 'CPU'} · ${precision}`
  const warnings = (ri.warnings ?? []).filter((w) => !/^MOCK ENGINE/.test(w))
  const env = result.envelope ?? {}
  const be = (env.backend ?? {}) as Record<string, unknown>
  const rows: [string, string][] = [
    ['Backend', backend],
    ['Requested', String(be.requested ?? '')],
    ['Reason', String(be.fallback_reason ?? '')],
    ['Precision', String(ri.precision ?? '')],
    ['Seed', ri.seed == null ? '' : String(ri.seed)],
    ['Tier', ri.tier && ri.tier !== 'design' ? String(ri.tier) : ''],
    ['Snapshot', String(ri.snapshot_hash ?? '').slice(0, 12)],
    ['Engine', String(ri.engine_version ?? '')],
    ['Time', ri.duration_ms != null ? `${Number(ri.duration_ms).toFixed(1)} ms` : ''],
  ]
  return (
    <Popover>
      <Tip label={warnings.length ? warnings.join('\n') : 'Run details'}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className={cn(
              'flex shrink-0 cursor-default items-center gap-1 rounded-full border px-2 py-[1px] text-[10px] font-medium',
              gpu ? 'border-accent/40 text-accent' : 'border-control-border text-secondary',
            )}
            aria-label={`Run details: ${label}${warnings.length ? `, ${warnings.length} warnings` : ''}`}
          >
            {warnings.length > 0 && <AlertTriangle className="size-3 text-warning" />}
            {label}
          </button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent className="w-[320px]" align="end">
        <h3 className="mb-1.5 text-[13px] font-semibold">Run Details</h3>
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-[1px] text-[12px]">
          {rows
            .filter(([, v]) => v)
            .map(([k, v]) => (
              <div key={k} className="contents">
                <span className="text-secondary">{k}</span>
                <span className="num select-text break-all">{v}</span>
              </div>
            ))}
        </div>
        {(ri.warnings ?? []).length > 0 && (
          <>
            <h4 className="mb-1 mt-2.5 text-[12px] font-semibold">Warnings</h4>
            <ul className="space-y-1 text-[12px]">
              {(ri.warnings ?? []).map((w, k) => (
                <li key={k} className="flex gap-1.5">
                  <AlertTriangle className="mt-[1px] size-3 shrink-0 text-warning" />
                  <span className="select-text">{w}</span>
                </li>
              ))}
            </ul>
          </>
        )}
        <Button size="sm" variant="ghost" className="mt-2 -ml-2" onClick={() => void navigator.clipboard?.writeText(JSON.stringify(ri, null, 2))}>
          Copy Run Info
        </Button>
      </PopoverContent>
    </Popover>
  )
}

export function EmptyPanes() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 bg-content text-secondary">
      <X className="size-5 opacity-0" />
      <span>No analysis panes</span>
      <span className="text-[12px]">Choose an analysis in the sidebar or with Add Analysis.</span>
    </div>
  )
}
