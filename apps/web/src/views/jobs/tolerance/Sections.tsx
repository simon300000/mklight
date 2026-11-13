// The Tolerance panel's other settings (TolerancePanel.swift
// `CompensatorSection`, `CriterionSection`, `AnalysisSection`): compensators
// with a ± range, the criterion with its pass limit, and the analysis (mode,
// trials, seed, levels, recorded trials, the last run's estimate).

import { Clock, MinusCircle, Plus } from 'lucide-react'
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
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { sig } from '@/model/format'
import { getParam, paramKey, paramUnit, surfaceIndexOfId, variablesOf, type ParamPath } from '@/model/params'
import { fieldUnit, type System } from '@/model/system'
import { fieldMenuLabel } from '@/model/merit'
import {
  CRITERION_KINDS,
  DISPLAY,
  MAX_RECORD_TRIALS,
  MAX_TRIALS,
  MODES,
  compensatorCandidates,
  compensatorTitle,
  criterionCaption,
  criterionDisplay,
  defaultCompensatorRange,
  estimateText,
  makeCompensator,
  meritInfoSummary,
  parseNumberList,
  parsePercentiles,
  type Compensator,
  type Criterion,
  type CriterionDefaults,
  type CriterionKind,
  type MeritInfo,
  type Mode,
  type ToleranceSettings,
} from '@/model/tolerance'
import { useDocument } from '@/state/document'
import { editTolerances, useTolerance } from '@/state/tolerance'
import { NumberInput } from '../../SettingsForm'
import { FormRow, Section } from '../../inspector/Form'
import { Hint, IndexChecklist, ListInput } from './controls'

// MARK: Compensators

export function CompensatorSection({ settings, system }: { settings: ToleranceSettings; system: System }) {
  const variablesSection = useDocument((s) => s.project.variables)
  const variables = variablesOf({ variables: variablesSection })
  const groups = compensatorCandidates(system, variables.map((v) => v.param))
  const used = new Set(settings.compensators.map((c) => paramKey(c.param)))
  const setC = (id: string, name: string, change: (c: Compensator) => void) =>
    editTolerances(name, (t) => {
      const c = t.compensators.find((x) => x.id === id)
      if (c) change(c)
    })
  const add = (p: ParamPath) => editTolerances('Add Compensator', (t) => void t.compensators.push(makeCompensator(p, defaultCompensatorRange(p))))
  const item = (p: ParamPath) => (
    <DropdownMenuItem key={paramKey(p)} disabled={used.has(paramKey(p))} onSelect={() => add(p)}>
      {compensatorTitle(p, system)}
    </DropdownMenuItem>
  )
  const bySurface = (ps: ParamPath[]) => {
    const out = new Map<string, ParamPath[]>()
    for (const p of ps) {
      const i = typeof p.surface === 'number' ? surfaceIndexOfId(system, p.surface) : -1
      const t = i >= 0 ? `Surface ${i}` : 'Other'
      out.set(t, [...(out.get(t) ?? []), p])
    }
    return [...out]
  }
  return (
    <Section title="Compensators" footer="Re-optimized for every trial within ± the range around its nominal value; empty is free.">
      {!settings.compensators.length && <p className="py-1 text-[12px] text-secondary">None: every trial is judged as built, without refocusing.</p>}
      {settings.compensators.map((c) => {
        const v = getParam(system, c.param)
        const title = compensatorTitle(c.param, system)
        return (
          <div key={c.id} className="flex items-center gap-1.5 py-[3px]">
            <span className="min-w-0 truncate">{title}</span>
            {v != null && <span className="num truncate text-[12px] text-secondary">{sig(v, 6)}</span>}
            <span className="flex-1" />
            <span className="text-secondary">±</span>
            <NumberInput className="w-[4.5rem]" ariaLabel={`${title} range`} value={c.range} nullable placeholder="free" onCommit={(r) => setC(c.id, 'Change Compensator Range', (x) => void (x.range = r == null ? null : Math.abs(r)))} />
            <span className="w-6 truncate text-[11px] text-secondary">{paramUnit(c.param)}</span>
            <Button
              variant="ghost"
              size="iconSm"
              aria-label={`Remove ${title}`}
              title="Remove this compensator"
              onClick={() => editTolerances('Remove Compensator', (t) => void (t.compensators = t.compensators.filter((x) => x.id !== c.id)))}
            >
              <MinusCircle />
            </Button>
          </div>
        )
      })}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="my-1">
            <Plus className="size-3.5" />
            Add Compensator
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {groups.map((g, k) => (
            <div key={g.group}>
              {k > 0 && <DropdownMenuSeparator />}
              <DropdownMenuLabel>{g.group}</DropdownMenuLabel>
              {g.group === 'Surfaces'
                ? bySurface(g.params).map(([t, ps]) => (
                    <DropdownMenuSub key={t}>
                      <DropdownMenuSubTrigger>{t}</DropdownMenuSubTrigger>
                      <DropdownMenuSubContent>{ps.map(item)}</DropdownMenuSubContent>
                    </DropdownMenuSub>
                  ))
                : g.params.slice(0, 40).map(item)}
            </div>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </Section>
  )
}

// MARK: Criterion

export function CriterionSection({ settings, system, merit }: { settings: ToleranceSettings; system: System; merit: MeritInfo }) {
  const c = settings.criterion
  const d = c.defaults
  const display = criterionDisplay(c, merit)
  const setC = (name: string, change: (c: Criterion) => void) => editTolerances(name, (t) => change(t.criterion))
  const setD = <K extends keyof CriterionDefaults>(name: string, key: K, v: CriterionDefaults[K]) => setC(name, (x) => void (x.defaults[key] = v))
  const fUnit = fieldUnit(system.fields.kind) || (system.fields.kind === 'object_angle' ? '°' : 'mm')
  const unit = DISPLAY.unit(display)
  return (
    <Section title="Criterion" footer={criterionCaption(display)}>
      <FormRow label="Criterion">
        <Select
          className="w-full"
          aria-label="Criterion"
          value={c.kind}
          onChange={(e) =>
            setC('Change Criterion', (x) => {
              x.kind = e.target.value as CriterionKind
              x.threshold = null
            })
          }
        >
          {CRITERION_KINDS.map((k) => (
            <option key={k.value} value={k.value}>
              {k.title}
            </option>
          ))}
        </Select>
      </FormRow>
      {(c.kind === 'rms_spot' || c.kind === 'wavefront') && (
        <FormRow label="Reference">
          <Select className="w-full" aria-label="Reference" value={d.reference} onChange={(e) => setD('Change Reference', 'reference', e.target.value)}>
            <option value="centroid">{c.kind === 'wavefront' ? 'Centroid (piston and tilt removed)' : 'Centroid'}</option>
            <option value="chief">{c.kind === 'wavefront' ? 'Chief ray (piston removed)' : 'Chief ray'}</option>
          </Select>
        </FormRow>
      )}
      {c.kind === 'mtf' && (
        <>
          <FormRow label="Frequencies" unit="cyc/mm">
            <ListInput
              className="w-full"
              ariaLabel="MTF frequencies"
              values={d.mtfFrequencies}
              parse={parseNumberList}
              invalidNote="positive numbers"
              placeholder="20"
              onCommit={(v) => setD('Change MTF Frequencies', 'mtfFrequencies', v)}
            />
          </FormRow>
          <FormRow label="Method">
            <Select className="w-full" aria-label="MTF method" value={d.mtfMethod} onChange={(e) => setD('Change MTF Method', 'mtfMethod', e.target.value)}>
              <option value="diffraction">Diffraction</option>
              <option value="geometric">Geometric</option>
            </Select>
          </FormRow>
          <FormRow label="Weakest field" help="Judge by the weakest field and direction at each frequency instead of the weighted mean.">
            <Switch aria-label="Weakest field" checked={d.worstField} onCheckedChange={(v) => setD('Change MTF Field Rule', 'worstField', v)} />
          </FormRow>
        </>
      )}
      {c.kind === 'design_merit' && <Hint className="py-1 text-[12px]">{meritInfoSummary(merit)}</Hint>}
      {c.kind !== 'design_merit' && (
        <>
          <FormRow label="Pupil">
            <span className="flex items-center gap-1 text-[12px] text-secondary">
              <NumberInput className="w-10" ariaLabel="Rings" value={d.rings} integer min={1} max={12} onCommit={(v) => v != null && setD('Change Pupil Rings', 'rings', v)} />
              rings ×
              <NumberInput className="w-10" ariaLabel="Arms" value={d.arms} integer min={3} max={24} onCommit={(v) => v != null && setD('Change Pupil Arms', 'arms', v)} />
              arms
            </span>
          </FormRow>
          <div className="flex flex-wrap gap-1.5 py-1">
            <IndexChecklist
              title="Fields"
              items={system.fields.fields.map((f, k) => fieldMenuLabel(k, f, fUnit))}
              selected={d.fields}
              onChange={(v) => setD('Change Criterion Fields', 'fields', v)}
            />
            <IndexChecklist
              title="Wavelengths"
              items={system.wavelengths.wavelengths.map((w) => `${w.um.toFixed(4)} µm`)}
              selected={d.wavelengths}
              onChange={(v) => setD('Change Criterion Wavelengths', 'wavelengths', v)}
            />
          </div>
        </>
      )}
      <FormRow label={DISPLAY.higherIsBetter(display) ? 'Pass at or above' : 'Pass at or below'} unit={unit} help="The yield is the share of Monte Carlo trials that pass.">
        <NumberInput ariaLabel="Pass limit" value={c.threshold} nullable placeholder="none" onCommit={(v) => setC('Change Pass Limit', (x) => void (x.threshold = v))} />
      </FormRow>
    </Section>
  )
}

// MARK: Analysis

export function AnalysisSection({ settings }: { settings: ToleranceSettings }) {
  const t = settings
  const last = useTolerance((s) => s.lastSecondsPerTrial)
  return (
    <Section title="Analysis">
      <Segmented<Mode>
        label="Run"
        className="my-1 flex w-full [&>button]:flex-1"
        value={t.mode}
        options={MODES}
        onChange={(m) => editTolerances('Change Tolerance Analysis', (x) => void (x.mode = m))}
      />
      {t.mode !== 'monte_carlo' && (
        <Hint className="py-1">
          Sensitivity: each tolerance at its − and + limit alone ({2 * t.rows.length + 1} evaluations), ranked, with the root-sum-square prediction.
        </Hint>
      )}
      {t.mode !== 'sensitivity' && (
        <>
          <FormRow label="Trials">
            <NumberInput ariaLabel="Trials" value={t.trials} integer min={1} max={MAX_TRIALS} placeholder="200" onCommit={(v) => v != null && v >= 1 && editTolerances('Change Trials', (x) => void (x.trials = v))} />
          </FormRow>
          <FormRow label="Seed">
            <NumberInput ariaLabel="Seed" value={t.seed} integer min={0} placeholder="0" onCommit={(v) => v != null && v >= 0 && editTolerances('Change Seed', (x) => void (x.seed = v))} />
          </FormRow>
          <FormRow label="Levels" unit="%" help="The percentiles of the builds the results report, e.g. 50, 90, 97.7">
            <ListInput
              className="w-full"
              ariaLabel="Levels"
              values={t.percentiles}
              parse={parsePercentiles}
              invalidNote="0 to 100"
              placeholder="50, 90, 97.7"
              onCommit={(v) => editTolerances('Change Percentiles', (x) => void (x.percentiles = v))}
            />
          </FormRow>
          <FormRow
            label="Keep"
            help="The first trials keep their perturbations and compensator values for the trial list and the CSV export (at most 10 000). Any trial can be previewed."
          >
            <span className="flex items-center gap-1.5">
              <NumberInput
                className="w-[4.5rem]"
                ariaLabel="Recorded trials"
                value={t.recordTrials}
                integer
                min={0}
                max={MAX_RECORD_TRIALS}
                onCommit={(v) => v != null && v >= 0 && editTolerances('Change Recorded Trials', (x) => void (x.recordTrials = v))}
              />
              <span className="whitespace-nowrap text-[11px] text-secondary">trials’ perturbations</span>
            </span>
          </FormRow>
          <Hint className="py-1">Monte Carlo: every tolerance drawn at random per trial; the same seed gives the same trials.</Hint>
          {last != null && (
            <Hint className="flex items-center gap-1 pb-1">
              <Clock className="size-3" />
              {estimateText(last, t.trials)}
            </Hint>
          )}
        </>
      )}
    </Section>
  )
}
