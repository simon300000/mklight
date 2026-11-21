// The merit function editor (apps/macos/Sources/MKLightUI/Views/Jobs/MeritEditor.swift):
// Default (the engine's generator) or Custom (an operand list with a
// searchable picker, values and shares from the live evaluation, the selected
// operand's editor), the live merit readout and the EFL hold. Every change is
// one undoable step saved in the project's `merit` section.

import { AlertTriangle, ArrowDown, ArrowUp, Copy, ListPlus, Loader2, Minus, Plus } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import type { Json, JsonObject } from '@/engine/types'
import { engineText, useT } from '@/i18n'
import { parseNumber, sig } from '@/model/format'
import {
  COMPARISONS,
  CRITERIA,
  addOperand,
  categoryTitle,
  choiceTitle,
  comparisonSymbol,
  defaultMeritCaption,
  deleteOperands,
  fieldMenuLabel,
  duplicateOperands,
  evaluationHelp,
  evaluationRow,
  localIssues,
  moveOperandsTo,
  newOperand,
  operandDescriptor,
  operandParameters,
  operandTitle,
  operandUnitText,
  percentShare,
  searchOperands,
  type Comparison,
  type Criterion,
  type DefaultMerit,
  type EvaluationRow,
  type MeritSettings,
  type Operand,
  type OperandDescriptor,
  type OperandParam,
} from '@/model/merit'
import { fieldUnit, imageIndex, type System } from '@/model/system'
import { useDocument } from '@/state/document'
import { editMerit, meritOf, useOptimize } from '@/state/optimize'
import { SUMMARY, useWorkspace } from '@/state/workspace'
import { anamorphicSummary } from '../StatusCapsule'
import { cn, isMac } from '@/lib/utils'
import { NumberInput } from '../SettingsForm'
import { FormRow, Section, TextInput } from '../inspector/Form'

export function MeritFunctionSection() {
  const t = useT()
  const project = useDocument((s) => s.project)
  const merit = meritOf(project)
  const evaluating = useOptimize((s) => s.evaluating)
  const converting = useOptimize((s) => s.converting)
  const error = useOptimize((s) => s.evaluationError)
  const convert = useOptimize((s) => s.convertToCustom)
  const hasSnapshot = useWorkspace((s) => s.snapshot != null)

  return (
    <Section
      title={t('optimize.merit.title')}
      accessory={
        <span className="text-[11px] text-secondary" title={t('optimize.merit.modeHelp')}>
          {merit.mode === 'default' ? t('optimize.merit.generated') : t('optimize.merit.operandCount', { count: merit.operands.length })}
        </span>
      }
    >
      <div className="flex items-center gap-2 py-1">
        <Segmented<'default' | 'custom'>
          label={t('optimize.merit.mode')}
          value={merit.mode}
          onChange={(m) => editMerit(m === 'custom' ? t('optimize.undo.useCustomMerit') : t('optimize.undo.useDefaultMerit'), (x) => ({ ...x, mode: m }))}
          options={[
            { value: 'default', title: t('optimize.merit.default'), help: t('optimize.merit.defaultHelp') },
            { value: 'custom', title: t('optimize.merit.custom'), help: t('optimize.merit.customHelp') },
          ]}
        />
        <span className="flex-1" />
        <MeritReadout />
      </div>
      {merit.mode === 'default' ? (
        <>
          <DefaultMeritForm system={project.system} merit={merit} />
          <div className="py-1">
            <Button
              size="sm"
              disabled={converting || !hasSnapshot}
              title={t('optimize.merit.convertHelp')}
              onClick={() => void convert()}
            >
              {converting ? <Loader2 className="animate-spin" /> : <ListPlus />}
              {t('optimize.merit.convert')}
            </Button>
          </div>
        </>
      ) : (
        <OperandTable system={project.system} merit={merit} />
      )}
      {error && !evaluating && (
        <p className="flex items-start gap-1 py-1 text-[11px] text-warning" role="alert">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          <span className="select-text whitespace-pre-wrap">{t('optimize.merit.evaluationFailed', { error })}</span>
        </p>
      )}
      <label className="flex items-center justify-between gap-2 border-t border-separator py-1.5" title={t('optimize.merit.holdEflHelp')}>
        <span>{t('optimize.merit.holdEfl')}</span>
        <Switch
          aria-label={t('optimize.merit.holdEfl')}
          checked={merit.holdEfl}
          onCheckedChange={(v) => editMerit(t('optimize.undo.holdFocalLength'), (m) => ({ ...m, holdEfl: v }))}
        />
      </label>
    </Section>
  )
}

/** The merit at the current design, evaluated by the engine; dims from an edit until its answer. */
export function MeritReadout() {
  const t = useT()
  const evaluation = useOptimize((s) => s.evaluation)
  const evaluating = useOptimize((s) => s.evaluating)
  const error = useOptimize((s) => s.evaluationError)
  const value = evaluation ? sig(evaluation.merit, 4) : '–'
  return (
    <div
      className={cn('flex items-center gap-1.5 transition-opacity', evaluating && 'opacity-50')}
      title={evaluationHelp(evaluation)}
      data-testid="merit-readout"
      data-evaluating={evaluating ? '' : undefined}
      aria-label={t('optimize.merit.currentMerit', { value: evaluation ? value : t('optimize.merit.notEvaluated') })}
      aria-busy={evaluating}
    >
      {evaluating && <Loader2 className="size-3 animate-spin text-secondary" />}
      {error && !evaluating && <AlertTriangle className="size-3.5 text-warning" aria-label={t('optimize.merit.evaluationFailedShort', { error })} />}
      <div className="text-right leading-tight">
        <div className="text-[10px] text-secondary">{t('optimize.merit.merit')}</div>
        <div className="num text-[13px] font-medium">{value}</div>
      </div>
    </div>
  )
}

// MARK: Default merit

function DefaultMeritForm({ system, merit }: { system: System; merit: MeritSettings }) {
  const t = useT()
  const d = merit.defaults
  const set = (name: string, change: (d: DefaultMerit) => Partial<DefaultMerit>) =>
    editMerit(name, (m) => ({ ...m, defaults: { ...m.defaults, ...change(m.defaults) } }))
  const unit = fieldUnit(system.fields.kind)
  // The engine's symmetry from the summary run (paraxial_xy.symmetry); else the surface kinds.
  const summary = useWorkspace((s) => s.runs[SUMMARY]?.result?.json)
  const anamorphic = anamorphicSummary(summary, system).anamorphic

  return (
    <div className="border-t border-separator pt-1">
      <FormRow label={t('optimize.default.criterion')}>
        <Select aria-label={t('optimize.default.criterion')} value={d.criterion} onChange={(e) => set(t('optimize.undo.changeCriterion'), () => ({ criterion: e.target.value as Criterion }))} className="w-full">
          {CRITERIA.map((c) => (
            <option key={c.value} value={c.value}>
              {c.title}
            </option>
          ))}
        </Select>
      </FormRow>
      {d.criterion !== 'mtf' && (
        <FormRow label={t('optimize.default.reference')}>
          <Select aria-label={t('optimize.default.reference')} value={d.reference} onChange={(e) => set(t('optimize.undo.changeReference'), () => ({ reference: e.target.value }))} className="w-full">
            <option value="centroid">{d.criterion === 'wavefront' ? t('optimize.default.centroidWavefront') : t('optimize.default.centroid')}</option>
            <option value="chief">{d.criterion === 'wavefront' ? t('optimize.default.chiefWavefront') : t('optimize.default.chief')}</option>
          </Select>
        </FormRow>
      )}
      {d.criterion === 'rms_spot' && (
        <FormRow label={t('optimize.default.whiteLight')} help={t('optimize.default.whiteLightHelp')}>
          <Switch
            aria-label={t('optimize.default.whiteLightCentroid')}
            checked={d.scope === 'polychromatic'}
            onCheckedChange={(v) => set(t('optimize.undo.changeCentroid'), () => ({ scope: v ? 'polychromatic' : 'monochromatic' }))}
          />
        </FormRow>
      )}
      {d.criterion === 'mtf' && (
        <>
          <FormRow label={t('optimize.default.frequencies')} unit="cyc/mm">
            <TextInput
              ariaLabel={t('optimize.default.mtfFrequencies')}
              value={d.mtfFrequencies.map((f) => sig(f, 6)).join(', ')}
              placeholder="10, 30"
              onCommit={(text) => {
                const v = parseList(text)?.filter((f) => f > 0)
                if (v?.length) set(t('optimize.undo.changeMtfFrequencies'), () => ({ mtfFrequencies: v }))
              }}
            />
          </FormRow>
          <FormRow label={t('optimize.default.target')}>
            <NumberInput ariaLabel={t('optimize.default.mtfTarget')} value={d.mtfTarget} min={0} max={1} onCommit={(v) => set(t('optimize.undo.changeMtfTarget'), () => ({ mtfTarget: v ?? 0.5 }))} />
          </FormRow>
          <FormRow label={t('optimize.default.method')}>
            <Select aria-label={t('optimize.default.mtfMethod')} value={d.mtfMethod} onChange={(e) => set(t('optimize.undo.changeMtfMethod'), () => ({ mtfMethod: e.target.value }))} className="w-full">
              <option value="diffraction">{choiceTitle('diffraction')}</option>
              <option value="geometric">{choiceTitle('geometric')}</option>
            </Select>
          </FormRow>
          <FormRow label={t('optimize.default.weakestField')} help={t('optimize.default.weakestFieldHelp')}>
            <Switch aria-label={t('optimize.default.weakestField')} checked={d.worstField} onCheckedChange={(v) => set(t('optimize.undo.changeMtfFieldRule'), () => ({ worstField: v }))} />
          </FormRow>
        </>
      )}
      <FormRow label={t('optimize.default.sampling')} help={t('optimize.default.samplingHelp')}>
        <span className="flex w-full items-center gap-1">
          <NumberInput ariaLabel={t('optimize.default.pupilRings')} value={d.rings} integer min={1} max={12} onCommit={(v) => set(t('optimize.undo.changePupilRings'), () => ({ rings: v ?? 3 }))} />
          <span className="text-secondary">×</span>
          <NumberInput ariaLabel={t('optimize.default.pupilArms')} value={d.arms} integer min={3} max={24} onCommit={(v) => set(t('optimize.undo.changePupilArms'), () => ({ arms: v ?? 6 }))} />
        </span>
      </FormRow>
      <div className="flex flex-wrap gap-2 py-1">
        <IndexChecklist
          title={t('optimize.default.fields')}
          all={system.fields.fields.map((f, k) => fieldMenuLabel(k, f, unit))}
          selected={d.fields}
          onChange={(v) => set(t('optimize.undo.changeMeritFields'), () => ({ fields: v }))}
        />
        <IndexChecklist
          title={t('optimize.default.wavelengths')}
          all={system.wavelengths.wavelengths.map((w) => `${w.um.toFixed(4)} µm`)}
          selected={d.wavelengths}
          onChange={(v) => set(t('optimize.undo.changeMeritWavelengths'), () => ({ wavelengths: v }))}
        />
      </div>
      <ThicknessLimits d={d} set={set} />
      {(anamorphic || d.squeezeTarget != null) && (
        <FormRow label={t('optimize.default.squeezeTarget')} help={t('optimize.default.squeezeTargetHelp')}>
          <NumberInput ariaLabel={t('optimize.default.squeezeTarget')} value={d.squeezeTarget} nullable placeholder={t('optimize.default.nonePlaceholder')} onCommit={(v) => set(t('optimize.undo.changeSqueezeTarget'), () => ({ squeezeTarget: v }))} />
        </FormRow>
      )}
      <p className="py-1 text-[11px] text-secondary">{defaultMeritCaption(d.criterion)}</p>
    </div>
  )
}

function parseList(t: string): number[] | null {
  const parts = t.split(/[;,\s]+/).filter(Boolean)
  const v = parts.map(parseNumber)
  return v.every((x): x is number => x != null && Number.isFinite(x)) ? v : null
}

/** A checklist menu of fields or wavelengths; null is all. */
function IndexChecklist({ title, all, selected, onChange }: { title: string; all: string[]; selected: number[] | null; onChange: (v: number[] | null) => void }) {
  const t = useT()
  const on = (k: number) => selected?.includes(k) ?? true
  const summary = !selected || selected.length >= all.length ? t('optimize.default.all') : selected.length === 1 ? `${selected[0] + 1}` : t('optimize.default.someOf', { n: selected.length, total: all.length })
  const toggle = (k: number, v: boolean) => {
    const s = new Set(selected ?? all.map((_, i) => i))
    if (v) s.add(k)
    else s.delete(k)
    if (!s.size) return
    onChange(s.size === all.length ? null : [...s].sort((a, b) => a - b))
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="sm" aria-label={t('optimize.default.checklist', { title, summary })}>
          {t('optimize.default.checklist', { title, summary })}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuItem onSelect={() => onChange(null)}>{t('optimize.default.all')}</DropdownMenuItem>
        <DropdownMenuSeparator />
        {all.map((label, k) => (
          <DropdownMenuCheckboxItem key={k} checked={on(k)} onCheckedChange={(v) => toggle(k, v === true)} onSelect={(e) => e.preventDefault()}>
            <span className="num">{label}</span>
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

type LimitKey = 'glassMinCentre' | 'glassMaxCentre' | 'glassMinEdge' | 'airMinCentre' | 'airMaxCentre' | 'airMinEdge'

function ThicknessLimits({ d, set }: { d: DefaultMerit; set: (name: string, change: (d: DefaultMerit) => Partial<DefaultMerit>) => void }) {
  const t = useT()
  const field = (key: LimitKey, label: string) => (
    <NumberInput ariaLabel={label} value={d[key]} nullable placeholder={t('optimize.default.nonePlaceholder')} onCommit={(v) => set(t('optimize.undo.changeThicknessLimits'), () => ({ [key]: v }))} />
  )
  return (
    <div className="border-t border-separator py-1">
      <label className="flex items-center justify-between gap-2 py-0.5">
        <span>{t('optimize.default.thicknessLimits')}</span>
        <Switch aria-label={t('optimize.default.thicknessLimitsLabel')} checked={d.thickness} onCheckedChange={(v) => set(t('optimize.undo.changeThicknessLimits'), () => ({ thickness: v }))} />
      </label>
      {d.thickness && (
        <div className="grid grid-cols-[minmax(0,1fr)_4.5rem_4.5rem] items-center gap-x-1.5 gap-y-1 pt-1">
          <span />
          <span className="text-right text-[11px] text-secondary">{choiceTitle('glass')}</span>
          <span className="text-right text-[11px] text-secondary">{choiceTitle('air')}</span>
          <span className="text-secondary">{t('optimize.default.minCenter')}</span>
          {field('glassMinCentre', t('optimize.default.glassMinCenter'))}
          {field('airMinCentre', t('optimize.default.airMinCenter'))}
          <span className="text-secondary">{t('optimize.default.maxCenter')}</span>
          {field('glassMaxCentre', t('optimize.default.glassMaxCenter'))}
          {field('airMaxCentre', t('optimize.default.airMaxCenter'))}
          <span className="text-secondary">{t('optimize.default.minEdge')}</span>
          {field('glassMinEdge', t('optimize.default.glassMinEdge'))}
          {field('airMinEdge', t('optimize.default.airMinEdge'))}
        </div>
      )}
    </div>
  )
}

// MARK: Custom operands

function OperandTable({ system, merit }: { system: System; merit: MeritSettings }) {
  const t = useT()
  const ops = merit.operands
  // The selected operands (sorted) and the anchor of ⇧-click ranges.
  const [selection, setSelection] = useState<number[]>(ops.length ? [0] : [])
  const [anchor, setAnchor] = useState<number | null>(null)
  const [picker, setPicker] = useState(false)
  // The gap a drag would drop into (0…length), while dragging.
  const [dropGap, setDropGap] = useState<number | null>(null)
  const dragging = useRef<number[] | null>(null)
  const evaluation = useOptimize((s) => s.evaluation)
  const list = useRef<HTMLDivElement>(null)
  const selected = selection.filter((k) => k < ops.length)
  const sel = selected.length === 1 ? selected[0] : null
  const first = selected[0] ?? null
  const select = (v: number[], a?: number | null) => {
    setSelection([...new Set(v)].sort((x, y) => x - y))
    if (a !== undefined) setAnchor(a)
  }

  useEffect(() => {
    const last = selection.at(-1)
    if (last != null) list.current?.querySelector(`[data-index="${last}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [selection])

  const edit = (name: string, change: (ops: Operand[]) => Operand[]) => editMerit(name, (m) => ({ ...m, operands: change(m.operands) }))
  const add = (d: OperandDescriptor) => {
    let at = 0
    edit(t('optimize.undo.addOperand'), (o) => {
      const r = addOperand(o, newOperand(d), first ?? (o.length ? o.length - 1 : null))
      at = r.index
      return r.operands
    })
    select([at], at)
    setPicker(false)
  }
  const del = () => {
    if (!selected.length) return
    edit(selected.length === 1 ? t('optimize.undo.deleteOperand') : t('optimize.undo.deleteOperands'), (o) => deleteOperands(o, selected))
    const next = Math.min(selected[0], ops.length - selected.length - 1)
    select(next >= 0 ? [next] : [], null)
  }
  const dup = () => {
    if (!selected.length) return
    let at: number[] = []
    edit(selected.length === 1 ? t('optimize.undo.duplicateOperand') : t('optimize.undo.duplicateOperands'), (o) => {
      const r = duplicateOperands(o, selected)
      at = r.indices
      return r.operands
    })
    select(at, at[0] ?? null)
  }
  const moveTo = (from: number[], gap: number) => {
    let at: number[] = from
    const changed = edit(from.length === 1 ? t('optimize.undo.moveOperand') : t('optimize.undo.moveOperands'), (o) => {
      const r = moveOperandsTo(o, from, gap)
      at = r.indices
      return r.operands
    })
    if (changed) select(at, at[0] ?? null)
  }
  const move = (step: number) => {
    if (sel == null) return
    const to = sel + step
    if (to < 0 || to >= ops.length) return
    moveTo([sel], step > 0 ? to + 1 : to)
  }
  const update = (name: string, change: (op: Operand) => Operand) => {
    if (sel == null) return
    edit(name, (o) => o.map((op, k) => (k === sel ? change(structuredClone(op)) : op)))
  }
  const click = (k: number, e: React.MouseEvent) => {
    if (e.shiftKey) {
      const a = anchor ?? first ?? k
      const lo = Math.min(a, k), hi = Math.max(a, k)
      const range = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
      select(e.metaKey || e.ctrlKey ? [...selected, ...range] : range)
    } else if (isMac ? e.metaKey : e.ctrlKey) {
      select(selected.includes(k) ? selected.filter((x) => x !== k) : [...selected, k], k)
    } else select([k], k)
  }
  const gapAt = (k: number, e: React.DragEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    return e.clientY < r.top + r.height / 2 ? k : k + 1
  }

  const mod = isMac ? '⌥⌘' : 'Ctrl+Alt+'
  const many = selected.length > 1
  return (
    <div className="border-t border-separator pt-1.5">
      <div className="flex items-center gap-1 pb-1.5">
        <Popover open={picker} onOpenChange={setPicker}>
          <PopoverTrigger asChild>
            <Button size="iconSm" variant="ghost" aria-label={t('optimize.operand.add')} title={t('optimize.operand.addHelp')}>
              <Plus />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-[320px] p-0">
            <OperandPicker pick={add} />
          </PopoverContent>
        </Popover>
        <Button size="iconSm" variant="ghost" aria-label={t('common.delete')} title={many ? t('optimize.operand.deleteMany') : t('optimize.operand.deleteOne')} disabled={!selected.length} onClick={del}>
          <Minus />
        </Button>
        <Button size="iconSm" variant="ghost" aria-label={t('optimize.operand.duplicate')} title={many ? t('optimize.operand.duplicateMany') : t('optimize.operand.duplicateOne')} disabled={!selected.length} onClick={dup}>
          <Copy />
        </Button>
        <Button size="iconSm" variant="ghost" aria-label={t('optimize.operand.moveUp')} title={t('optimize.operand.moveUpHelp', { keys: `${mod}↑` })} disabled={sel == null || sel === 0} onClick={() => move(-1)}>
          <ArrowUp />
        </Button>
        <Button size="iconSm" variant="ghost" aria-label={t('optimize.operand.moveDown')} title={t('optimize.operand.moveDownHelp', { keys: `${mod}↓` })} disabled={sel == null || sel >= ops.length - 1} onClick={() => move(1)}>
          <ArrowDown />
        </Button>
        <span className="ml-auto text-[11px] text-secondary">
          {many ? t('optimize.operand.selectedOf', { n: selected.length, total: ops.length }) : t('optimize.merit.operandCount', { count: ops.length })}
        </span>
      </div>
      {ops.length === 0 ? (
        <p className="py-4 text-center text-secondary">{t('optimize.operand.none')}</p>
      ) : (
        <div
          ref={list}
          role="listbox"
          aria-label={t('optimize.operand.list')}
          aria-multiselectable
          tabIndex={0}
          title={t('optimize.operand.listHelp')}
          className="max-h-[260px] overflow-y-auto rounded-md border border-separator outline-none focus-visible:border-accent"
          data-testid="operand-list"
          onKeyDown={(e) => {
            if ((e.key === 'Backspace' || e.key === 'Delete') && !e.metaKey) {
              e.preventDefault()
              del()
            } else if (e.altKey && (e.metaKey || e.ctrlKey) && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
              e.preventDefault()
              move(e.key === 'ArrowUp' ? -1 : 1)
            } else if ((e.key === 'a' || e.key === 'A') && (isMac ? e.metaKey : e.ctrlKey)) {
              e.preventDefault()
              select(ops.map((_, k) => k), 0)
            } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
              e.preventDefault()
              const step = e.key === 'ArrowUp' ? -1 : 1
              const from = step < 0 ? selected[0] : selected.at(-1)
              const k = Math.min(Math.max((from ?? -step) + step, 0), ops.length - 1)
              if (e.shiftKey && selected.length) {
                const a = anchor ?? selected[0]
                const lo = Math.min(a, k), hi = Math.max(a, k)
                select(Array.from({ length: hi - lo + 1 }, (_, i) => lo + i))
              } else select([k], k)
            }
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDropGap(null)
          }}
        >
          {ops.map((op, k) => (
            <OperandRow
              key={k}
              index={k}
              operand={op}
              selected={selected.includes(k)}
              dropBefore={dropGap === k}
              dropAfter={dropGap === ops.length && k === ops.length - 1}
              row={evaluationRow(evaluation, k, op.kind.type)}
              issues={localIssues(op, system.fields.fields.length, system.wavelengths.wavelengths.length)}
              onClick={(e) => click(k, e)}
              onDragStart={(e) => {
                const from = selected.includes(k) ? selected : [k]
                if (!selected.includes(k)) select([k], k)
                dragging.current = from
                e.dataTransfer.effectAllowed = 'move'
                e.dataTransfer.setData('application/x-mklight-operands', from.join(','))
              }}
              onDragOver={(e) => {
                if (!dragging.current) return
                e.preventDefault()
                e.dataTransfer.dropEffect = 'move'
                setDropGap(gapAt(k, e))
              }}
              onDrop={(e) => {
                const from = dragging.current
                if (!from) return
                e.preventDefault()
                moveTo(from, gapAt(k, e))
                dragging.current = null
                setDropGap(null)
              }}
              onDragEnd={() => {
                dragging.current = null
                setDropGap(null)
              }}
            />
          ))}
        </div>
      )}
      {sel != null && ops[sel] && (
        <OperandDetail index={sel} operand={ops[sel]} system={system} row={evaluationRow(evaluation, sel, ops[sel].kind.type)} update={update} />
      )}
      {many && <p className="py-2 text-center text-[11px] text-secondary">{t('optimize.operand.manySelected', { n: selected.length })}</p>}
    </div>
  )
}

function OperandRow({
  index,
  operand,
  selected,
  dropBefore,
  dropAfter,
  row,
  issues,
  onClick,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
}: {
  index: number
  operand: Operand
  selected: boolean
  dropBefore: boolean
  dropAfter: boolean
  row: EvaluationRow | null
  issues: string[]
  onClick: (e: React.MouseEvent) => void
  onDragStart: (e: React.DragEvent) => void
  onDragOver: (e: React.DragEvent) => void
  onDrop: (e: React.DragEvent) => void
  onDragEnd: () => void
}) {
  const t = useT()
  const problem = row?.error ? t('optimize.operand.evaluationFailed', { error: engineText(row.error) }) : (issues[0] ?? null)
  const parts: string[] = []
  const params = operandParameters(operand)
  if (params) parts.push(params)
  parts.push(`${comparisonSymbol(operand.comparison)} ${sig(operand.target, 4)}`)
  if (!operand.constraint) parts.push(t('optimize.operand.weightShort', { value: sig(operand.weight, 4) }))
  const detail = problem ?? parts.join(' · ')
  const fraction = row?.fraction ?? NaN
  return (
    <div
      role="option"
      aria-selected={selected}
      data-index={index}
      draggable
      className={cn(
        'relative flex cursor-default items-center gap-2 border-b border-separator px-1.5 py-1 last:border-b-0',
        selected ? 'bg-accent-soft' : 'hover:bg-hover',
        !operand.enabled && 'opacity-45',
      )}
      onClick={onClick}
      onMouseDown={(e) => {
        // ⇧-click selects a range, not text.
        if (e.shiftKey) e.preventDefault()
      }}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onDragEnd={onDragEnd}
    >
      {dropBefore && <span className="pointer-events-none absolute inset-x-1 -top-px h-0.5 rounded-full bg-accent" aria-hidden />}
      {dropAfter && <span className="pointer-events-none absolute inset-x-1 -bottom-px h-0.5 rounded-full bg-accent" aria-hidden />}
      <span className="num w-5 shrink-0 text-right text-[11px] text-tertiary">{index + 1}</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1">
          <span className="truncate text-[13px]">{operandTitle(operand.kind.type)}</span>
          {operand.constraint && <span className="shrink-0 rounded-full bg-selected px-1.5 text-[10px] text-secondary">{t('optimize.operand.constraint')}</span>}
          {problem && <AlertTriangle className="size-3 shrink-0 text-warning" aria-label={problem} />}
        </div>
        <div className={cn('num truncate text-[11px]', problem ? 'text-warning' : 'text-secondary')} title={detail}>
          {detail}
        </div>
      </div>
      <div className="flex w-[84px] shrink-0 flex-col items-end gap-0.5">
        <span className="num text-[12px]">{row ? sig(row.value, 4) : '–'}</span>
        <span className="flex items-center gap-1" aria-hidden>
          <span className="relative h-1 w-[34px] overflow-hidden rounded-full bg-selected">
            {Number.isFinite(fraction) && (
              <span className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${Math.max(fraction > 0 ? 6 : 0, Math.min(fraction, 1) * 100)}%` }} />
            )}
          </span>
          <span className="num w-[34px] text-right text-[10px] text-secondary">{percentShare(fraction)}</span>
        </span>
      </div>
    </div>
  )
}

function OperandDetail({
  index,
  operand,
  system,
  row,
  update,
}: {
  index: number
  operand: Operand
  system: System
  row: EvaluationRow | null
  update: (name: string, change: (op: Operand) => Operand) => void
}) {
  const t = useT()
  const desc = operandDescriptor(operand.kind.type)
  const single = desc?.singleField ?? false
  const fieldValue = operand.field.field === 'index' ? String(operand.field.index ?? 0) : operand.field.field === 'all' ? 'all' : 'other'
  const w = operand.wavelength
  const wlValue = w.wavelength === 'index' ? String(w.index ?? 0) : w.wavelength === 'all' ? 'all' : 'primary'
  const issues = localIssues(operand, system.fields.fields.length, system.wavelengths.wavelengths.length)
  const unit = fieldUnit(system.fields.kind)

  return (
    <div className="mt-2 rounded-md bg-hover px-2 py-1.5" data-testid="operand-detail">
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="text-[13px] font-semibold">
          {index + 1}. {operandTitle(operand.kind.type)}
        </h4>
        {desc?.unit && <span className="text-[11px] text-secondary">{operandUnitText(desc.unit)}</span>}
      </div>
      {desc?.summary && <p className="pb-1 text-[11px] text-secondary">{desc.summary}</p>}
      {desc?.params.map((p) => <ParamEditor key={p.key} p={p} operand={operand} system={system} update={update} />)}
      {!desc && <p className="num truncate py-1 text-[11px] text-secondary">{t('optimize.operand.parameters', { json: JSON.stringify(operand.kind) })}</p>}
      <FormRow label={t('common.field')}>
        <Select
          aria-label={t('common.field')}
          className="w-full"
          value={fieldValue}
          onChange={(e) => {
            const v = e.target.value
            update(t('optimize.undo.changeOperandField'), (op) => ({ ...op, field: (v === 'all' ? { field: 'all' } : { field: 'index', index: Number(v) }) as JsonObject }))
          }}
        >
          {fieldValue === 'other' && <option value="other">{t('optimize.operand.normalized')}</option>}
          {(!single || fieldValue === 'all') && <option value="all">{t('optimize.operand.allFields')}</option>}
          {system.fields.fields.map((f, k) => (
            <option key={k} value={k}>
              {fieldMenuLabel(k, f, unit)}
            </option>
          ))}
        </Select>
      </FormRow>
      <FormRow label={t('common.wavelength')}>
        <Select
          aria-label={t('common.wavelength')}
          className="w-full"
          value={wlValue}
          onChange={(e) => {
            const v = e.target.value
            update(t('optimize.undo.changeOperandWavelength'), (op) => ({
              ...op,
              wavelength: (v === 'primary' ? { wavelength: 'primary' } : v === 'all' ? { wavelength: 'all' } : { wavelength: 'index', index: Number(v) }) as JsonObject,
            }))
          }}
        >
          <option value="primary">{t('optimize.operand.primary')}</option>
          <option value="all">{t('optimize.operand.allPolychromatic')}</option>
          {system.wavelengths.wavelengths.map((wv, k) => (
            <option key={k} value={k}>
              {wv.um.toFixed(4)} µm
            </option>
          ))}
        </Select>
      </FormRow>
      <FormRow label={t('optimize.operand.target')} help={t('optimize.operand.targetHelp')}>
        <span className="flex w-full items-center gap-1.5">
          <Segmented<Comparison>
            label={t('optimize.operand.comparison')}
            value={operand.comparison}
            onChange={(c) => update(t('optimize.undo.changeComparison'), (op) => ({ ...op, comparison: c }))}
            options={COMPARISONS.map((c) => ({ value: c.value, title: c.symbol, help: c.help }))}
          />
          <NumberInput ariaLabel={t('optimize.operand.target')} value={operand.target} onCommit={(v) => update(t('optimize.undo.changeTarget'), (op) => ({ ...op, target: v ?? 0 }))} />
        </span>
      </FormRow>
      <FormRow label={t('optimize.operand.constraintLabel')} help={t('optimize.operand.constraintHelp')}>
        <Switch
          aria-label={t('optimize.operand.constraintLabel')}
          checked={operand.constraint}
          onCheckedChange={(v) => update(v ? t('optimize.undo.makeConstraint') : t('optimize.undo.makeResidual'), (op) => ({ ...op, constraint: v }))}
        />
      </FormRow>
      {!operand.constraint && (
        <FormRow label={t('optimize.operand.weight')}>
          <NumberInput ariaLabel={t('optimize.operand.weight')} value={operand.weight} min={0} onCommit={(v) => update(t('optimize.undo.changeWeight'), (op) => ({ ...op, weight: Math.max(v ?? 1, 0) }))} />
        </FormRow>
      )}
      <FormRow label={t('optimize.operand.enabled')}>
        <Switch aria-label={t('optimize.operand.enabled')} checked={operand.enabled} onCheckedChange={(v) => update(v ? t('optimize.undo.enableOperand') : t('optimize.undo.disableOperand'), (op) => ({ ...op, enabled: v }))} />
      </FormRow>
      {row && (
        <p className="num py-0.5 text-right text-[12px]">
          {t('optimize.operand.value', { value: sig(row.value, 6) })}
          {Number.isFinite(row.fraction) && <span className="text-secondary"> · {percentShare(row.fraction)}</span>}
        </p>
      )}
      {row?.error && <p className="select-text py-0.5 text-[11px] text-warning">{engineText(row.error)}</p>}
      {issues.map((issue) => (
        <p key={issue} className="py-0.5 text-[11px] text-warning">
          {issue}
        </p>
      ))}
    </div>
  )
}

function ParamEditor({ p, operand, system, update }: { p: OperandParam; operand: Operand; system: System; update: (name: string, change: (op: Operand) => Operand) => void }) {
  const t = useT()
  const v = operand.kind[p.key]
  const name = t('optimize.undo.changeParam', { title: p.title })
  const setKind = (x: Json | undefined) =>
    update(name, (op) => {
      const kind = { ...op.kind }
      if (x === undefined) delete kind[p.key]
      else kind[p.key] = x
      return { ...op, kind }
    })
  const k = p.kind
  switch (k.type) {
    case 'number':
      return (
        <FormRow label={p.title} unit={k.unit}>
          <NumberInput
            ariaLabel={p.title}
            value={typeof v === 'number' ? v : null}
            nullable={!p.required}
            placeholder={p.required ? '0' : t('optimize.operand.autoPlaceholder')}
            onCommit={(x) => setKind(x ?? (p.required ? 0 : undefined))}
          />
        </FormRow>
      )
    case 'integer':
      return (
        <FormRow label={p.title}>
          <NumberInput ariaLabel={p.title} value={typeof v === 'number' ? v : null} integer min={k.min} onCommit={(x) => setKind(Math.max(x ?? k.min, k.min))} />
        </FormRow>
      )
    case 'surface': {
      const img = imageIndex(system)
      return (
        <FormRow label={p.title}>
          <Select aria-label={p.title} className="w-full" value={typeof v === 'number' ? String(v) : ''} onChange={(e) => setKind(e.target.value === '' ? undefined : Number(e.target.value))}>
            {!p.required && <option value="">{p.key === 'to' ? t('optimize.operand.next') : p.key === 'from' ? t('optimize.operand.first') : t('optimize.operand.image')}</option>}
            {p.required && typeof v !== 'number' && <option value="">{t('optimize.operand.choose')}</option>}
            {system.surfaces.slice(1).map((s, j) => (
              <option key={s.id} value={s.id}>
                {j + 1 === img ? t('optimize.operand.image') : t('common.surfaceN', { n: j + 1 }) + (s.label ? ' · ' + s.label : '')}
              </option>
            ))}
          </Select>
        </FormRow>
      )
    }
    case 'wavelength':
      return (
        <FormRow label={p.title}>
          <Select aria-label={p.title} className="w-full" value={String(typeof v === 'number' ? v : 0)} onChange={(e) => setKind(Number(e.target.value))}>
            {system.wavelengths.wavelengths.map((w, j) => (
              <option key={j} value={j}>
                {w.um.toFixed(4)} µm
              </option>
            ))}
          </Select>
        </FormRow>
      )
    case 'choice':
      return (
        <FormRow label={p.title}>
          <Select aria-label={p.title} className="w-full" value={typeof v === 'string' ? v : k.options[0]} onChange={(e) => setKind(e.target.value)}>
            {k.options.map((o) => (
              <option key={o} value={o}>
                {choiceTitle(o)}
              </option>
            ))}
          </Select>
        </FormRow>
      )
    case 'flag':
      return (
        <FormRow label={p.title}>
          <Switch aria-label={p.title} checked={v === true} onCheckedChange={(x) => setKind(x)} />
        </FormRow>
      )
    case 'sampling': {
      const o = (v && typeof v === 'object' && !Array.isArray(v) ? v : {}) as JsonObject
      const rings = typeof o.rings === 'number' ? o.rings : 3
      const arms = typeof o.arms === 'number' ? o.arms : 6
      return (
        <FormRow label={t('optimize.default.sampling')} help={t('optimize.default.samplingHelp')}>
          <span className="flex w-full items-center gap-1">
            <NumberInput ariaLabel={t('optimize.operand.rings')} value={rings} integer min={1} max={12} onCommit={(x) => setKind({ ...o, rings: x ?? 3, arms })} />
            <span className="text-secondary">×</span>
            <NumberInput ariaLabel={t('optimize.operand.arms')} value={arms} integer min={3} max={24} onCommit={(x) => setKind({ ...o, rings, arms: x ?? 6 })} />
          </span>
        </FormRow>
      )
    }
    case 'operandList': {
      const idx = Array.isArray(v) ? v.filter((x): x is number => typeof x === 'number') : []
      return (
        <FormRow label={p.title} help={t('optimize.operand.operandListHelp')}>
          <TextInput
            ariaLabel={p.title}
            value={idx.map((x) => x + 1).join(', ')}
            placeholder="1, 2"
            onCommit={(text) => {
              const parts = text.split(/[;,\s]+/).filter(Boolean)
              const vals = parts.map((s) => Number.parseInt(s, 10)).filter((x) => Number.isInteger(x) && x >= 1)
              if (vals.length === parts.length) setKind(vals.map((x) => x - 1))
            }}
          />
        </FormRow>
      )
    }
  }
}

/** The searchable operand picker, grouped by category; Return adds the first match. */
function OperandPicker({ pick }: { pick: (d: OperandDescriptor) => void }) {
  const t = useT()
  const [query, setQuery] = useState('')
  const groups = searchOperands(query)
  return (
    <div className="flex max-h-[380px] flex-col" data-testid="operand-picker">
      <div className="p-2">
        <input
          autoFocus
          aria-label={t('optimize.operand.search')}
          placeholder={t('optimize.operand.search')}
          className="h-7 w-full rounded-md border border-control-border bg-control px-2 text-[13px] outline-none focus:border-accent"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              const first = groups[0]?.items[0]
              if (first) pick(first)
            }
          }}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-1 pb-1">
        {groups.length === 0 && <p className="px-2 py-3 text-secondary">{t('optimize.operand.noMatch')}</p>}
        {groups.map((g) => (
          <div key={g.category}>
            <div className="px-2 pb-0.5 pt-1.5 text-[11px] font-semibold text-secondary">{categoryTitle(g.category)}</div>
            {g.items.map((d) => (
              <button
                key={d.type}
                type="button"
                className="block w-full cursor-default rounded-md px-2 py-1 text-left hover:bg-accent hover:text-on-accent focus-visible:bg-accent focus-visible:text-on-accent focus-visible:outline-none [&:hover_*]:text-on-accent"
                onClick={() => pick(d)}
              >
                <span className="flex items-baseline justify-between gap-2">
                  <span className="text-[13px]">{d.title}</span>
                  {d.unit && <span className="text-[11px] text-secondary">{operandUnitText(d.unit)}</span>}
                </span>
                {d.summary && <span className="block truncate text-[11px] text-secondary">{d.summary}</span>}
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
