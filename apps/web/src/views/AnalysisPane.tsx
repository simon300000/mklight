import { AlertTriangle, ChevronDown, Cpu, Ellipsis, Play, RefreshCw, Settings2, Square, X, Zap } from 'lucide-react'
import { Component, useMemo, useState, type ReactNode } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
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
import { analysisShortTitle, analysisTitle, categoryTitle, groupedAnalyses } from '@/model/catalog'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { descriptorFor, effectiveSettings, isStale, useWorkspace, type Pane } from '@/state/workspace'
import { cn, shortcutLabel } from '@/lib/utils'
import { AnalysisIcon } from './AnalysisIcon'
import { PanelHeader } from './PanelHeader'
import { SettingsForm, settingsSummary } from './SettingsForm'
import { keyNumbers, ResultView } from './results'
import { badgeHelp, durationText, remedyFor, runFacts, warningLine } from './analysis/runInfo'
import { Subscripted } from './analysis/Subscripted'

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
  const [detailsOpen, setDetailsOpen] = useState(false)
  const setView = useView((s) => s.set)
  const d = descriptorFor(ws.descriptors, pane.analysis)
  const run = ws.runs[pane.key]
  const settings = effectiveSettings(d, pane.settings)
  const stale = isStale(ws, run, pane.analysis, pane.settings)
  const running = run?.status === 'running'
  const result = run?.result
  const title = analysisTitle(pane.analysis, d)
  const shortTitle = analysisShortTitle(pane.analysis)
  const setOwn = (patch: JsonObject) => ws.updatePane(pane.key, { settings: { ...pane.settings, ...patch } })
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
      return keyNumbers(pane.analysis, result, system)
    } catch {
      return []
    }
  }, [pane.analysis, result, system])

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
              <button
                type="button"
                className="flex min-w-0 cursor-default items-center gap-0.5 rounded px-0.5 hover:bg-hover"
                aria-label={`${title}, choose analysis`}
                title={title}
              >
                <h2 className="truncate text-[13px] font-semibold @max-[360px]:text-[12px]">
                  <span className="@max-[300px]:hidden">{title}</span>
                  <span className="hidden @max-[300px]:inline" aria-hidden>
                    {shortTitle}
                  </span>
                </h2>
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
            {result && <DropdownMenuItem onSelect={() => setDetailsOpen(true)}>Run Details…</DropdownMenuItem>}
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
          analysis={pane.analysis}
          result={result}
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
              <ResultView analysis={pane.analysis} result={result} system={system} settings={settings} onSettingsChange={setOwn} />
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
            <span key={k.label} className="num shrink-0 truncate" title={k.help} aria-label={`${k.label.replace(/_/g, ' ')} ${k.value}`}>
              <Subscripted text={k.label} className="text-secondary" /> <span className="font-medium">{k.value}</span>
            </span>
          ))}
        </div>
        {result && (
          <BackendBadge
            result={result}
            settings={settings}
            open={detailsOpen}
            onOpenChange={setDetailsOpen}
            onRemedy={(key, value) => setOwn({ [key]: value })}
          />
        )}
      </footer>
    </section>
  )
}

/** CPU · f64 neutral, GPU · f32 in the accent, an orange mark for warnings
 *  (with their count when there are several kinds); opens the run details. */
function BackendBadge({
  result,
  settings,
  open,
  onOpenChange,
  onRemedy,
}: {
  result: AnalysisResult
  settings: JsonObject
  open: boolean
  onOpenChange: (o: boolean) => void
  onRemedy: (key: string, value: JsonObject[string]) => void
}) {
  const ri = result.runInfo
  const backend = String(ri.backend ?? '')
  const gpu = backend.startsWith('gpu')
  const precision = String(ri.precision ?? (gpu ? 'f32' : 'f64')).replace(/_.*/, '')
  const label = backend === 'mock' ? 'Mock' : `${gpu ? 'GPU' : 'CPU'} · ${precision}`
  const facts = useMemo(() => runFacts(result), [result])
  const marks = facts.groups.length + (facts.fallback ? 1 : 0)
  const spoken = [
    `Computed on ${facts.rows[0]?.[1] ?? backend}`,
    facts.fallback ? `Ran on the CPU: ${facts.fallback}` : '',
    facts.groups.length ? `${facts.groups.length} warning${facts.groups.length === 1 ? '' : 's'}: ${facts.groups.map(warningLine).join('. ')}` : '',
  ]
    .filter(Boolean)
    .join('. ')
  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <Tip label={<span className="whitespace-pre-line">{badgeHelp(facts)}</span>}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className={cn(
              'flex shrink-0 cursor-default items-center gap-1 rounded-full px-2 py-[1px] text-[10px] font-medium',
              gpu ? 'bg-accent/15 text-accent' : 'bg-selected text-secondary',
            )}
            aria-label={`Run details: ${spoken}`}
            data-testid="backend-badge"
          >
            {marks > 0 && (
              <span className="flex items-center gap-[2px] text-warning">
                <AlertTriangle className="size-3" />
                {marks > 1 && marks}
              </span>
            )}
            <Cpu className="hidden size-3 @max-[300px]:block" aria-hidden />
            <span className="@max-[300px]:hidden">{label}</span>
            {ri.duration_ms != null && <span className="opacity-70 @max-[420px]:hidden">{durationText(Number(ri.duration_ms))}</span>}
          </button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent className="max-h-[70vh] w-[380px] overflow-y-auto" align="end">
        <RunDetails result={result} settings={settings} onRemedy={(k, v) => (onRemedy(k, v), onOpenChange(false))} />
      </PopoverContent>
    </Popover>
  )
}

/** Run information: backend, route, fallback, precision, tier, seed, snapshot,
 *  engine and time; the attempts that did not compute the result; warnings
 *  grouped with their fields and a one-click remedy; the raw messages. */
export function RunDetails({ result, settings, onRemedy }: { result: AnalysisResult; settings: JsonObject; onRemedy?: (key: string, value: JsonObject[string]) => void }) {
  const f = useMemo(() => runFacts(result), [result])
  const [all, setAll] = useState(false)
  return (
    <div className="select-text text-[12px]" data-testid="run-details">
      <h3 className="mb-1.5 text-[13px] font-semibold">Run Information</h3>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-[2px]">
        {f.rows.map(([k, v]) => (
          <div key={k} className="contents">
            <span className="text-secondary">{k}</span>
            <span className="num break-words">{v}</span>
          </div>
        ))}
      </div>
      {f.attempts.length > 0 && (
        <section className="mt-2.5 border-t border-separator pt-2">
          <h4 className="mb-1 font-semibold">{f.attempts.length === 1 ? 'Attempt Not Used' : `${f.attempts.length} Attempts Not Used`}</h4>
          <ul className="space-y-1.5">
            {f.attempts.map((a, k) => (
              <li key={k}>
                <span className="flex gap-1.5">
                  <AlertTriangle className="mt-[1px] size-3 shrink-0 text-warning" />
                  <span>{a.title}</span>
                </span>
                {a.reason && <span className="block pl-[18px] text-[11px] text-secondary">{a.reason}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}
      {f.groups.length > 0 && (
        <section className="mt-2.5 border-t border-separator pt-2">
          <h4 className="mb-1 font-semibold">{f.groups.length === 1 ? 'Warning' : `${f.groups.length} Warnings`}</h4>
          <ul className="space-y-1.5">
            {f.groups.map((g, k) => {
              const fix = remedyFor(g.remedy, settings)
              return (
                <li key={k}>
                  <span className="flex gap-1.5">
                    <AlertTriangle className="mt-[1px] size-3 shrink-0 text-warning" />
                    <span>{g.message}</span>
                  </span>
                  {g.scopes.length > 0 && <span className="block pl-[18px] text-[11px] text-secondary">{g.scopes.join(', ')}</span>}
                  {fix && onRemedy && (
                    <Button size="sm" className="mt-1 ml-[18px]" title={fix.help} onClick={() => onRemedy(fix.key, fix.value)}>
                      {fix.title}
                    </Button>
                  )}
                </li>
              )
            })}
          </ul>
          {f.raw.length > f.groups.length && (
            <details className="mt-1.5" open={all} onToggle={(e) => setAll((e.target as HTMLDetailsElement).open)}>
              <summary className="cursor-default text-[11px] text-secondary">All {f.raw.length} messages from the engine</summary>
              <ul className="mt-1 space-y-0.5 font-mono text-[11px] text-secondary">
                {f.raw.map((w, k) => (
                  <li key={k}>{w}</li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}
      <Button size="sm" variant="ghost" className="mt-2 -ml-2" onClick={() => void navigator.clipboard?.writeText(JSON.stringify(result.runInfo, null, 2))}>
        Copy Run Info
      </Button>
    </div>
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
