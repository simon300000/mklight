// The Tolerance result (TolerancePanel.swift `ToleranceResultView`,
// `TolerancePreviewCard`, `SensitivityRow`, `TrialList`, `CompensatorStats`):
// nominal, RSS prediction, the engine's levels and the yield; Export; the card
// of a previewed build; the sensitivities by share of the RSS with the worst
// offenders tinted; the Monte-Carlo histogram with its trials; the
// compensators' statistics.

import { AlertTriangle, BadgeCheck, Eye, EyeOff, Info, Loader2, Share } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import { sig } from '@/model/format'
import { paramUnit, surfaceIndexOfId } from '@/model/params'
import type { System } from '@/model/system'
import {
  DISPLAY,
  buildChanges,
  compensatorRange,
  compensatorTitle,
  hasMonteCarlo,
  hasNominal,
  histogramBins,
  limitsText,
  percentText,
  previewTitle,
  recordOf,
  rowLabel,
  samePreviewCase,
  secondsText,
  sensLimits,
  sensitivityCsv,
  share,
  short,
  summaryText,
  trialsCsv,
  trialsShownIn,
  trialsWorstFirst,
  worsePlus,
  worstOffenders,
  type Bin,
  type ToleranceResult,
  type ToleranceRow,
} from '@/model/tolerance'
import { download } from '@/state/files'
import { endTolerancePreview, previewTolerance, useTolerance } from '@/state/tolerance'
import { useView } from '@/state/view'
import { BigNumber } from '../JobPanel'
import { Hint, KeyValues, LimitBar, RangeUsageBar, ShareBar } from './controls'
import { MonteCarloChart } from './MonteCarloChart'

const signed = (v: number, digits = 4) => (Number.isFinite(v) ? (v < 0 ? '−' : '+') + Math.abs(v).toFixed(digits) : '—')

export function ToleranceResults({ result, system }: { result: ToleranceResult; system: System }) {
  const rows = useTolerance((s) => s.runRows)
  const runSystem = useTolerance((s) => s.runSystem) ?? system
  const previewed = useTolerance((s) => s.previewed)
  const [bin, setBin] = useState<Bin | null>(null)
  const d = result.display

  // A previewed trial shows its bar's trials.
  useEffect(() => {
    if (previewed?.type !== 'trial') return
    const v = DISPLAY.value(d, result.trialCriteria[previewed.trial])
    if (!Number.isFinite(v) || (bin && v >= bin.lo && v <= bin.hi)) return
    setBin(histogramBins(result).find((b) => v >= b.lo && v <= b.hi) ?? null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewed, result])

  return (
    <div className="flex flex-col gap-5" data-testid="tolerance-results">
      <Summary result={result} rows={rows} system={runSystem} />
      {previewed && <PreviewCard result={result} rows={rows} system={runSystem} />}
      {result.hasSensitivity && <SensitivityList result={result} rows={rows} system={runSystem} />}
      {hasMonteCarlo(result) && (
        <section className="flex flex-col gap-2">
          <h3 className="text-[13px] font-semibold">Monte Carlo</h3>
          <MonteCarloChart
            result={result}
            selection={bin}
            onSelect={setBin}
            marked={previewed?.type === 'trial' ? DISPLAY.value(d, result.trialCriteria[previewed.trial] ?? NaN) : null}
          />
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-secondary">
            {hasNominal(result) && (
              <span className="flex items-center gap-1">
                <span className="inline-block h-3 w-[2px] bg-accent" />
                Nominal {DISPLAY.format(d, DISPLAY.value(d, result.nominal))}
              </span>
            )}
            {result.percentiles.length > 0 && (
              <span className="flex items-center gap-1">
                <span className="inline-block h-3 w-0 border-l border-dashed border-secondary" />
                {result.percentiles.map((p) => short(p.percentile)).join(', ')} % of builds
              </span>
            )}
            <span className="flex items-center gap-1">
              <span className="inline-block h-[2px] w-3 bg-label/75" />
              Cumulative share (right axis)
            </span>
          </div>
          <TrialList result={result} bin={bin} onClearBin={() => setBin(null)} />
          <KeyValues
            rows={[
              ['Trials', `${result.trials}, seed ${result.seed}`],
              ['Mean', DISPLAY.formatWithUnit(d, DISPLAY.value(d, result.mean))],
              ['Best', DISPLAY.formatWithUnit(d, DISPLAY.value(d, result.min))],
              ['Worst', DISPLAY.formatWithUnit(d, DISPLAY.value(d, result.max))],
              ...(result.secondsPerTrial != null ? [['Time per trial', secondsText(result.secondsPerTrial)] as [string, string]] : []),
              ...(result.failures > 0 ? [['Failed trials', String(result.failures)] as [string, string]] : []),
            ]}
          />
        </section>
      )}
      {result.compensators.length > 0 && (
        <section className="flex flex-col gap-2">
          <h3 className="text-[13px] font-semibold">Compensators</h3>
          {result.compensators.map((_, j) => (
            <CompensatorStats key={j} j={j} result={result} system={runSystem} />
          ))}
        </section>
      )}
    </div>
  )
}

// MARK: Summary

function Summary({ result, rows, system }: { result: ToleranceResult; rows: ToleranceRow[]; system: System }) {
  const d = result.display
  const unit = DISPLAY.unit(d)
  const [copied, setCopied] = useState(false)
  const notes: string[] = []
  if (result.stopped) notes.push(`Stopped early: ${hasMonteCarlo(result) ? `statistics of the ${result.trials} completed trials` : 'the sensitivities completed so far'}.`)
  if (result.failures > 0) notes.push(`${result.failures} trials failed (no image) and are left out.`)
  if (hasMonteCarlo(result) && !result.hasSensitivity) notes.push('Run the sensitivity too to see the largest contributors.')
  const name = system.name || 'Tolerance'
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-[13px] font-semibold">
          {DISPLAY.title(d)}
          {unit ? ` (${unit})` : ''}
        </h3>
        <DropdownMenu onOpenChange={(o) => o && setCopied(false)}>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" title="Copy a summary as text, or save the tables as CSV">
              <Share className="size-3.5" />
              Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onSelect={() => {
                void navigator.clipboard?.writeText(summaryText(result, rows, system, name))
                setCopied(true)
                useView.getState().toast('Copied the tolerance summary.')
              }}
            >
              {copied ? 'Copied' : 'Copy Summary'}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {result.hasSensitivity && (
              <DropdownMenuItem onSelect={() => download(`${name} sensitivity.csv`, sensitivityCsv(result, rows, system), 'text/csv')}>Save Sensitivity Table as CSV…</DropdownMenuItem>
            )}
            {hasMonteCarlo(result) && (
              <DropdownMenuItem onSelect={() => download(`${name} trials.csv`, trialsCsv(result, rows, system), 'text/csv')}>Save Monte Carlo Trials as CSV…</DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="grid grid-cols-3 gap-x-4 gap-y-3">
        {hasNominal(result) && (
          <div title={result.compensators.length ? 'The design as it is, with the compensators re-optimized' : 'The design as it is'}>
            <BigNumber label="Nominal" value={DISPLAY.format(d, DISPLAY.value(d, result.nominal))} testId="tolerance-nominal" />
          </div>
        )}
        {result.hasSensitivity && (
          <div className="col-span-2" title="Nominal plus the root-sum-square of every tolerance's worst change">
            <BigNumber label="Predicted (RSS)" value={DISPLAY.format(d, DISPLAY.value(d, result.predicted))} testId="tolerance-rss" />
          </div>
        )}
        {hasMonteCarlo(result) &&
          result.percentiles.map((p) => (
            <div key={p.percentile} title={`${short(p.percentile)} % of the Monte Carlo trials reach ${DISPLAY.higherIsBetter(d) ? 'at least' : 'at most'} this value`}>
              <BigNumber label={`${short(p.percentile)} % of builds`} value={DISPLAY.format(d, DISPLAY.value(d, p.value))} />
            </div>
          ))}
      </div>
      {result.yieldFraction != null && (
        <p className="flex items-center gap-1.5 text-[12px]" data-testid="tolerance-yield">
          <BadgeCheck className="size-4 text-success" />
          Yield {percentText(result.yieldFraction)} at {DISPLAY.higherIsBetter(d) ? '≥' : '≤'} {DISPLAY.format(d, DISPLAY.value(d, result.threshold ?? NaN))}
        </p>
      )}
      {notes.length > 0 && (
        <Hint className="flex items-start gap-1">
          <Info className="mt-px size-3 shrink-0" />
          {notes.join(' ')}
        </Hint>
      )}
    </section>
  )
}

// MARK: Preview

function PreviewCard({ result, rows, system }: { result: ToleranceResult; rows: ToleranceRow[]; system: System }) {
  const previewed = useTolerance((s) => s.previewed)
  const preview = useTolerance((s) => s.preview)
  const error = useTolerance((s) => s.previewError)
  const runSettings = useTolerance((s) => s.runSettings)
  const [all, setAll] = useState(false)
  useEffect(() => setAll(false), [previewed])
  if (!previewed) return null
  const d = result.display
  const title = previewTitle(previewed, rows, system, result).replace(/^Tolerance: /, '')
  const changes = preview ? buildChanges(preview.perturbations, rows, runSettings, system) : []
  return (
    <section className="rounded-[10px] border border-accent/25 bg-accent/[0.08] p-3" aria-label="Previewed build" data-testid="tolerance-preview-card">
      <div className="flex items-start gap-2">
        <Eye className="mt-0.5 size-4 shrink-0 text-accent" />
        <h4 className="min-w-0 flex-1 text-[12px] font-semibold">{title.charAt(0).toUpperCase() + title.slice(1)}</h4>
        <Button size="sm" onClick={endTolerancePreview}>
          End Preview
        </Button>
      </div>
      {error ? (
        <p className="mt-1.5 flex items-start gap-1 text-[12px] text-warning">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          {error}
        </p>
      ) : preview ? (
        <div className="mt-1.5 flex flex-col gap-1.5">
          <Hint>
            {DISPLAY.title(d)} {DISPLAY.formatWithUnit(d, DISPLAY.value(d, preview.criterion))} · shown in the layout and the analyses, not applied
          </Hint>
          {preview.compensators.length > 0 && (
            <div className="flex flex-col gap-0.5 text-[11px]">
              {result.compensators.map((c, j) => {
                const v = preview.compensators[j]
                const n = result.nominalCompensators[j]
                const unit = paramUnit(c.param)
                return (
                  <div key={j} className="flex items-baseline gap-2" title={`Re-optimized for this build${n != null ? `; nominal ${sig(n, 6)} ${unit}` : ''}`}>
                    <span className="min-w-0 flex-1 truncate">{compensatorTitle(c.param, system)}</span>
                    <span className="num">
                      {sig(v, 6)} {unit}
                    </span>
                    {n != null && <span className="num text-secondary">({signed(v - n)})</span>}
                  </div>
                )
              })}
            </div>
          )}
          {!changes.length ? (
            <Hint>No perturbation.</Hint>
          ) : (
            <div className="flex flex-col gap-0.5 text-[11px]">
              <span className="font-medium text-secondary">{changes.length === 1 ? 'Perturbation' : 'Perturbations, largest share of their limit first'}</span>
              {(all ? changes : changes.slice(0, 6)).map((c) => (
                <div key={c.id} className="flex items-center gap-2" title={`${c.label}: ${c.text} of ${c.limits}`}>
                  <span className="min-w-0 flex-1 truncate">{c.label}</span>
                  <span className="num">{c.text}</span>
                  <LimitBar fraction={c.fraction} />
                </div>
              ))}
              {changes.length > 6 && (
                <button type="button" className="cursor-default self-start text-accent hover:underline" onClick={() => setAll(!all)}>
                  {all ? 'Show the 6 largest' : `Show all ${changes.length}`}
                </button>
              )}
            </div>
          )}
        </div>
      ) : (
        <p className="mt-1.5 flex items-center gap-1.5 text-[11px] text-secondary">
          <Loader2 className="size-3.5 animate-spin" />
          Rebuilding the build in the engine…
        </p>
      )}
    </section>
  )
}

// MARK: Sensitivity

function SensitivityList({ result, rows, system }: { result: ToleranceResult; rows: ToleranceRow[]; system: System }) {
  const [all, setAll] = useState(false)
  const previewed = useTolerance((s) => s.previewed)
  const d = result.display
  const worst = worstOffenders(result)
  const list = all ? result.sensitivities : result.sensitivities.slice(0, 12)
  const nominal = DISPLAY.value(d, result.nominal)
  const unit = DISPLAY.unit(d)
  return (
    <section className="flex flex-col gap-1" data-testid="tolerance-sensitivity">
      <div className="flex items-baseline justify-between">
        <h3 className="text-[13px] font-semibold">Sensitivity</h3>
        <span className="text-[11px] text-secondary">Δ {unit || DISPLAY.title(d)} at − / +</span>
      </div>
      <Hint className="pb-1">Each tolerance alone at its limits, largest share of the RSS first. Click a row to select its surface; the eye previews that build.</Hint>
      {list.map((s, k) => {
        const row = rows[s.id]
        const label = row ? rowLabel(row, system) : s.label
        const f = share(result, s)
        const isWorst = worst.has(s.id)
        const selected = previewed?.type === 'sensitivity' && previewed.operand === s.id
        const c = { type: 'sensitivity' as const, operand: s.id, positive: worsePlus(s) }
        return (
          <div
            key={s.id}
            role="button"
            tabIndex={0}
            aria-label={`${k + 1}. ${label}${isWorst ? ', a worst offender' : ''}, ${Math.round(f * 100)} % of the RSS`}
            title={`${label}: ${Math.round(f * 100)} % of the RSS. Click to select the surface.`}
            className={cn('group cursor-default rounded-md px-1 py-[3px] outline-none focus-visible:ring-2 focus-visible:ring-accent', selected ? 'bg-accent-soft' : isWorst ? 'bg-warning/12' : 'hover:bg-hover')}
            onClick={() => {
              const id = s.param && typeof s.param.surface === 'number' ? s.param.surface : row?.surface
              const i = id != null ? surfaceIndexOfId(useTolerance.getState().runSystem ?? system, id) : -1
              if (i >= 0) useView.getState().select(i)
            }}
            onKeyDown={(e) => e.key === 'Enter' && (e.currentTarget as HTMLElement).click()}
          >
            <div className="flex items-center gap-2">
              <span className="num w-[18px] shrink-0 text-right text-[11px] text-secondary">{k + 1}</span>
              <span className={cn('min-w-0 truncate', isWorst && 'font-semibold')}>{label}</span>
              <span className="truncate text-[11px] text-secondary">{row ? limitsText(row) : sensLimits(s, rows)}</span>
              <span className="flex-1" />
              <Button
                variant="ghost"
                size="iconSm"
                className={cn('shrink-0', selected ? 'text-accent' : 'opacity-40 group-hover:opacity-100')}
                aria-label={selected ? 'End preview' : 'Preview'}
                title={selected ? 'End the preview' : `Preview this build in the window: the tolerance at its worse limit${result.compensators.length ? ', compensated' : ''}; not applied`}
                onClick={(e) => {
                  e.stopPropagation()
                  if (selected) endTolerancePreview()
                  else void previewTolerance(c, result)
                }}
              >
                {selected ? <EyeOff /> : <Eye />}
              </Button>
            </div>
            <div className="flex items-center gap-2 pl-[26px]">
              <ShareBar fraction={f} highlighted={isWorst} />
              <span className="flex-1" />
              <span className="num truncate text-[11px] text-secondary">
                {DISPLAY.formatChange(d, DISPLAY.value(d, s.criterionMinus) - nominal)} / {DISPLAY.formatChange(d, DISPLAY.value(d, s.criterionPlus) - nominal)}
              </span>
            </div>
          </div>
        )
      })}
      {result.sensitivities.length > 12 && (
        <button type="button" className="cursor-default self-start text-[12px] text-accent hover:underline" onClick={() => setAll(!all)}>
          {all ? 'Show the 12 largest' : `Show all ${result.sensitivities.length}`}
        </button>
      )}
      {result.sensitivityCancelled && <p className="text-[11px] text-warning">Stopped: tolerances not evaluated are missing.</p>}
    </section>
  )
}

// MARK: Trials

function TrialList({ result, bin, onClearBin }: { result: ToleranceResult; bin: Bin | null; onClearBin: () => void }) {
  const [limit, setLimit] = useState(6)
  const previewed = useTolerance((s) => s.previewed)
  const runSystem = useTolerance((s) => s.runSystem)
  useEffect(() => setLimit(6), [bin])
  const d = result.display
  const unit = DISPLAY.unit(d)
  const trials = bin ? trialsShownIn(result, bin.lo, bin.hi) : trialsWorstFirst(result)
  const title = bin ? `${trials.length} trial${trials.length === 1 ? '' : 's'} at ${DISPLAY.format(d, bin.lo)}–${DISPLAY.format(d, bin.hi)}${unit ? ' ' + unit : ''}` : 'Worst trials'
  return (
    <div className="flex flex-col gap-0.5" data-testid="tolerance-trials">
      <div className="flex items-baseline justify-between">
        <span className="text-[12px] font-medium">{title}</span>
        {bin && (
          <button type="button" className="cursor-default text-[11px] text-accent hover:underline" title="List the worst trials instead of the selected bar's" onClick={onClearBin}>
            Worst Trials
          </button>
        )}
      </div>
      {!trials.length && <Hint>No trial in this bar.</Hint>}
      {trials.slice(0, limit).map((k) => {
        const selected = samePreviewCase(previewed, { type: 'trial', trial: k })
        const rec = recordOf(result, k)
        const c0 = result.compensators[0]
        const comp =
          rec && c0 && rec.compensators.length && result.nominalCompensators.length && runSystem
            ? `${compensatorTitle(c0.param, runSystem)} ${signed(rec.compensators[0] - result.nominalCompensators[0], 3)}`
            : null
        const value = DISPLAY.formatWithUnit(d, DISPLAY.value(d, result.trialCriteria[k]))
        return (
          <button
            key={k}
            type="button"
            className={cn('flex cursor-default items-center gap-2 rounded-[5px] px-1 py-[2px] text-left text-[12px]', selected ? 'bg-accent-soft' : 'hover:bg-hover')}
            aria-label={`Trial ${k + 1}, ${value}`}
            title={selected ? 'End the preview' : `Preview trial ${k + 1} in the window: its perturbations${result.compensators.length ? ' and re-optimised compensators' : ''}; not applied`}
            onClick={() => (selected ? endTolerancePreview() : void previewTolerance({ type: 'trial', trial: k }, result))}
          >
            <span className="num w-[72px] shrink-0">Trial {k + 1}</span>
            <span className="num">{value}</span>
            {comp && <span className="num truncate text-[11px] text-secondary">{comp}</span>}
            <span className="flex-1" />
            {selected ? <EyeOff className="size-3.5 text-accent" /> : <Eye className="size-3.5 text-secondary" />}
          </button>
        )
      })}
      {trials.length > limit && (
        <button type="button" className="cursor-default self-start text-[11px] text-accent hover:underline" onClick={() => setLimit(limit + 20)}>
          Show {Math.min(trials.length - limit, 20)} more
        </button>
      )}
      <Hint>{bin ? 'The eye rebuilds a trial in the window, read-only.' : 'Click a bar of the histogram for its trials; the eye rebuilds a trial in the window.'}</Hint>
    </div>
  )
}

// MARK: Compensators

function CompensatorStats({ j, result, system }: { j: number; result: ToleranceResult; system: System }) {
  const c = result.compensators[j]
  const unit = paramUnit(c.param)
  const used = compensatorRange(result, j)
  const nominal = j < result.nominalCompensators.length ? result.nominalCompensators[j] : null
  const rows: [string, string][] = []
  if (nominal != null) rows.push(['Nominal (re-optimized)', `${sig(nominal, 6)} ${unit}`])
  if (used) rows.push(['Sensitivity range used', `${signed(used[0])} to ${signed(used[1])} ${unit}`])
  if (j < result.compensatorMean.length) {
    const mean = result.compensatorMean[j]
    const sd = result.compensatorStd[j] ?? NaN
    rows.push(['Monte Carlo mean', `${sig(mean, 6)} ${unit}${nominal != null ? ` (${signed(mean - nominal)})` : ''}`])
    rows.push(['Monte Carlo σ (±3σ)', Number.isFinite(sd) ? `${sig(sd, 4)} (±${sig(3 * sd, 4)}) ${unit}` : '—'])
  }
  const atLimit = used && c.range != null && Math.max(-used[0], used[1]) >= c.range * 0.99
  return (
    <div className="flex flex-col gap-1 rounded-lg bg-selected/50 p-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-medium">{compensatorTitle(c.param, system)}</span>
        {c.range != null && (
          <span className="text-[11px] text-secondary">
            allowed ±{short(c.range)} {unit}
          </span>
        )}
      </div>
      {used && c.range != null && c.range > 0 && (
        <span title={`The sensitivity evaluations moved it from ${short(used[0])} to +${short(used[1])} ${unit} of ±${short(c.range)} allowed`}>
          <RangeUsageBar lo={used[0]} hi={used[1]} range={c.range} />
        </span>
      )}
      <KeyValues rows={rows} />
      {atLimit && (
        <p className="flex items-start gap-1 text-[11px] text-warning">
          <AlertTriangle className="mt-px size-3 shrink-0" />
          It reached its limit: some builds need more adjustment than allowed.
        </p>
      )}
    </div>
  )
}
