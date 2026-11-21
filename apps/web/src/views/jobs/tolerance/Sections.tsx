// The Tolerance panel's other settings (TolerancePanel.swift
// `CompensatorSection`, `CriterionSection`, `AnalysisSection`): compensators
// with a ± range, the criterion with its pass limit, and the analysis (mode,
// trials, seed, levels, recorded trials, the last run's estimate).

import { Clock, MinusCircle, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useT } from '@/i18n'
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
  const t = useT()
  const variablesSection = useDocument((s) => s.project.variables)
  const variables = variablesOf({ variables: variablesSection })
  const groups = compensatorCandidates(system, variables.map((v) => v.param))
  const used = new Set(settings.compensators.map((c) => paramKey(c.param)))
  const setC = (id: string, name: string, change: (c: Compensator) => void) =>
    editTolerances(name, (t) => {
      const c = t.compensators.find((x) => x.id === id)
      if (c) change(c)
    })
  const add = (p: ParamPath) => editTolerances(t('tolerance.undo.addCompensator'), (x) => void x.compensators.push(makeCompensator(p, defaultCompensatorRange(p))))
  const item = (p: ParamPath) => (
    <DropdownMenuItem key={paramKey(p)} disabled={used.has(paramKey(p))} onSelect={() => add(p)}>
      {compensatorTitle(p, system)}
    </DropdownMenuItem>
  )
  const bySurface = (ps: ParamPath[]) => {
    const out = new Map<string, ParamPath[]>()
    for (const p of ps) {
      const i = typeof p.surface === 'number' ? surfaceIndexOfId(system, p.surface) : -1
      const label = i >= 0 ? t('tolerance.compensators.surfaceN', { n: i }) : t('tolerance.compensators.other')
      out.set(label, [...(out.get(label) ?? []), p])
    }
    return [...out]
  }
  return (
    <Section title={t('tolerance.compensatorSection.title')} footer={t('tolerance.compensatorSection.footer')}>
      {!settings.compensators.length && <p className="py-1 text-[12px] text-secondary">{t('tolerance.compensatorSection.none')}</p>}
      {settings.compensators.map((c) => {
        const v = getParam(system, c.param)
        const title = compensatorTitle(c.param, system)
        return (
          <div key={c.id} className="flex items-center gap-1.5 py-[3px]">
            <span className="min-w-0 truncate">{title}</span>
            {v != null && <span className="num truncate text-[12px] text-secondary">{sig(v, 6)}</span>}
            <span className="flex-1" />
            <span className="text-secondary">±</span>
            <NumberInput className="w-[4.5rem]" ariaLabel={t('tolerance.compensatorSection.range', { title })} value={c.range} nullable placeholder={t('tolerance.compensatorSection.free')} onCommit={(r) => setC(c.id, t('tolerance.undo.changeCompensatorRange'), (x) => void (x.range = r == null ? null : Math.abs(r)))} />
            <span className="w-6 truncate text-[11px] text-secondary">{paramUnit(c.param)}</span>
            <Button
              variant="ghost"
              size="iconSm"
              aria-label={t('tolerance.compensatorSection.remove', { title })}
              title={t('tolerance.compensatorSection.removeHelp')}
              onClick={() => editTolerances(t('tolerance.undo.removeCompensator'), (x) => void (x.compensators = x.compensators.filter((y) => y.id !== c.id)))}
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
            {t('tolerance.compensatorSection.add')}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {groups.map((g, k) => (
            <div key={g.group}>
              {k > 0 && <DropdownMenuSeparator />}
              <DropdownMenuLabel>{g.title}</DropdownMenuLabel>
              {g.group === 'Surfaces'
                ? bySurface(g.params).map(([label, ps]) => (
                    <DropdownMenuSub key={label}>
                      <DropdownMenuSubTrigger>{label}</DropdownMenuSubTrigger>
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
  const t = useT()
  const c = settings.criterion
  const d = c.defaults
  const display = criterionDisplay(c, merit)
  const setC = (name: string, change: (c: Criterion) => void) => editTolerances(name, (x) => change(x.criterion))
  const setD = <K extends keyof CriterionDefaults>(name: string, key: K, v: CriterionDefaults[K]) => setC(name, (x) => void (x.defaults[key] = v))
  const fUnit = fieldUnit(system.fields.kind) || (system.fields.kind === 'object_angle' ? '°' : 'mm')
  const unit = DISPLAY.unit(display)
  return (
    <Section title={t('tolerance.criterionSection.title')} footer={criterionCaption(display)}>
      <FormRow label={t('tolerance.criterionSection.criterion')}>
        <Select
          className="w-full"
          aria-label={t('tolerance.criterionSection.criterion')}
          value={c.kind}
          onChange={(e) =>
            setC(t('tolerance.undo.changeCriterion'), (x) => {
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
        <FormRow label={t('tolerance.criterionSection.reference')}>
          <Select className="w-full" aria-label={t('tolerance.criterionSection.reference')} value={d.reference} onChange={(e) => setD(t('tolerance.undo.changeReference'), 'reference', e.target.value)}>
            <option value="centroid">{c.kind === 'wavefront' ? t('tolerance.criterionSection.centroidWavefront') : t('tolerance.criterionSection.centroid')}</option>
            <option value="chief">{c.kind === 'wavefront' ? t('tolerance.criterionSection.chiefWavefront') : t('tolerance.criterionSection.chief')}</option>
          </Select>
        </FormRow>
      )}
      {c.kind === 'mtf' && (
        <>
          <FormRow label={t('tolerance.criterionSection.frequencies')} unit="cyc/mm">
            <ListInput
              className="w-full"
              ariaLabel={t('tolerance.criterionSection.mtfFrequencies')}
              values={d.mtfFrequencies}
              parse={parseNumberList}
              invalidNote={t('tolerance.criterionSection.invalidList')}
              placeholder="20"
              onCommit={(v) => setD(t('tolerance.undo.changeMtfFrequencies'), 'mtfFrequencies', v)}
            />
          </FormRow>
          <FormRow label={t('tolerance.criterionSection.method')}>
            <Select className="w-full" aria-label={t('tolerance.criterionSection.mtfMethod')} value={d.mtfMethod} onChange={(e) => setD(t('tolerance.undo.changeMtfMethod'), 'mtfMethod', e.target.value)}>
              <option value="diffraction">{t('tolerance.criterionSection.diffraction')}</option>
              <option value="geometric">{t('tolerance.criterionSection.geometric')}</option>
            </Select>
          </FormRow>
          <FormRow label={t('tolerance.criterionSection.weakestField')} help={t('tolerance.criterionSection.weakestFieldHelp')}>
            <Switch aria-label={t('tolerance.criterionSection.weakestField')} checked={d.worstField} onCheckedChange={(v) => setD(t('tolerance.undo.changeMtfFieldRule'), 'worstField', v)} />
          </FormRow>
        </>
      )}
      {c.kind === 'design_merit' && <Hint className="py-1 text-[12px]">{meritInfoSummary(merit)}</Hint>}
      {c.kind !== 'design_merit' && (
        <>
          <FormRow label={t('tolerance.criterionSection.pupil')}>
            <span className="flex items-center gap-1 text-[12px] text-secondary">
              <NumberInput className="w-10" ariaLabel={t('tolerance.criterionSection.rings')} value={d.rings} integer min={1} max={12} onCommit={(v) => v != null && setD(t('tolerance.undo.changePupilRings'), 'rings', v)} />
              {t('tolerance.criterionSection.ringsUnit')}
              <NumberInput className="w-10" ariaLabel={t('tolerance.criterionSection.arms')} value={d.arms} integer min={3} max={24} onCommit={(v) => v != null && setD(t('tolerance.undo.changePupilArms'), 'arms', v)} />
              {t('tolerance.criterionSection.armsUnit')}
            </span>
          </FormRow>
          <div className="flex flex-wrap gap-1.5 py-1">
            <IndexChecklist
              title={t('tolerance.criterionSection.fields')}
              items={system.fields.fields.map((f, k) => fieldMenuLabel(k, f, fUnit))}
              selected={d.fields}
              onChange={(v) => setD(t('tolerance.undo.changeCriterionFields'), 'fields', v)}
            />
            <IndexChecklist
              title={t('tolerance.criterionSection.wavelengths')}
              items={system.wavelengths.wavelengths.map((w) => `${w.um.toFixed(4)} µm`)}
              selected={d.wavelengths}
              onChange={(v) => setD(t('tolerance.undo.changeCriterionWavelengths'), 'wavelengths', v)}
            />
          </div>
        </>
      )}
      <FormRow label={DISPLAY.higherIsBetter(display) ? t('tolerance.criterionSection.passAbove') : t('tolerance.criterionSection.passBelow')} unit={unit} help={t('tolerance.criterionSection.passHelp')}>
        <NumberInput ariaLabel={t('tolerance.criterionSection.passLimit')} value={c.threshold} nullable placeholder={t('tolerance.criterionSection.none')} onCommit={(v) => setC(t('tolerance.undo.changePassLimit'), (x) => void (x.threshold = v))} />
      </FormRow>
    </Section>
  )
}

// MARK: Analysis

export function AnalysisSection({ settings }: { settings: ToleranceSettings }) {
  const t = useT()
  const ts = settings
  const last = useTolerance((s) => s.lastSecondsPerTrial)
  return (
    <Section title={t('tolerance.analysisSection.title')}>
      <Segmented<Mode>
        label={t('tolerance.analysisSection.run')}
        className="my-1 flex w-full [&>button]:flex-1"
        value={ts.mode}
        options={MODES}
        onChange={(m) => editTolerances(t('tolerance.undo.changeToleranceAnalysis'), (x) => void (x.mode = m))}
      />
      {ts.mode !== 'monte_carlo' && <Hint className="py-1">{t('tolerance.analysisSection.sensitivityHint', { count: 2 * ts.rows.length + 1 })}</Hint>}
      {ts.mode !== 'sensitivity' && (
        <>
          <FormRow label={t('tolerance.analysisSection.trials')}>
            <NumberInput ariaLabel={t('tolerance.analysisSection.trials')} value={ts.trials} integer min={1} max={MAX_TRIALS} placeholder="200" onCommit={(v) => v != null && v >= 1 && editTolerances(t('tolerance.undo.changeTrials'), (x) => void (x.trials = v))} />
          </FormRow>
          <FormRow label={t('tolerance.analysisSection.seed')}>
            <NumberInput ariaLabel={t('tolerance.analysisSection.seed')} value={ts.seed} integer min={0} placeholder="0" onCommit={(v) => v != null && v >= 0 && editTolerances(t('tolerance.undo.changeSeed'), (x) => void (x.seed = v))} />
          </FormRow>
          <FormRow label={t('tolerance.analysisSection.levels')} unit="%" help={t('tolerance.analysisSection.levelsHelp')}>
            <ListInput
              className="w-full"
              ariaLabel={t('tolerance.analysisSection.levels')}
              values={ts.percentiles}
              parse={parsePercentiles}
              invalidNote={t('tolerance.analysisSection.levelsInvalid')}
              placeholder="50, 90, 97.7"
              onCommit={(v) => editTolerances(t('tolerance.undo.changePercentiles'), (x) => void (x.percentiles = v))}
            />
          </FormRow>
          <FormRow label={t('tolerance.analysisSection.keep')} help={t('tolerance.analysisSection.keepHelp')}>
            <span className="flex items-center gap-1.5">
              <NumberInput
                className="w-[4.5rem]"
                ariaLabel={t('tolerance.analysisSection.recordedTrials')}
                value={ts.recordTrials}
                integer
                min={0}
                max={MAX_RECORD_TRIALS}
                onCommit={(v) => v != null && v >= 0 && editTolerances(t('tolerance.undo.changeRecordedTrials'), (x) => void (x.recordTrials = v))}
              />
              <span className="whitespace-nowrap text-[11px] text-secondary">{t('tolerance.analysisSection.trialsPerturbations')}</span>
            </span>
          </FormRow>
          <Hint className="py-1">{t('tolerance.analysisSection.monteCarloHint')}</Hint>
          {last != null && (
            <Hint className="flex items-center gap-1 pb-1">
              <Clock className="size-3" />
              {estimateText(last, ts.trials)}
            </Hint>
          )}
        </>
      )}
    </Section>
  )
}
