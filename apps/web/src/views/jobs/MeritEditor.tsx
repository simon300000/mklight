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
import { parseNumber, sig } from '@/model/format'
import {
  COMPARISONS,
  CRITERIA,
  addOperand,
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
  const project = useDocument((s) => s.project)
  const merit = meritOf(project)
  const evaluating = useOptimize((s) => s.evaluating)
  const converting = useOptimize((s) => s.converting)
  const error = useOptimize((s) => s.evaluationError)
  const convert = useOptimize((s) => s.convertToCustom)
  const hasSnapshot = useWorkspace((s) => s.snapshot != null)

  return (
    <Section
      title="Merit Function"
      accessory={
        <span className="text-[11px] text-secondary" title="Default: the engine generates the merit from a few choices. Custom: edit every operand.">
          {merit.mode === 'default' ? 'Generated' : `${merit.operands.length} operand${merit.operands.length === 1 ? '' : 's'}`}
        </span>
      }
    >
      <div className="flex items-center gap-2 py-1">
        <Segmented<'default' | 'custom'>
          label="Merit mode"
          value={merit.mode}
          onChange={(m) => editMerit(m === 'custom' ? 'Use Custom Merit' : 'Use Default Merit', (x) => ({ ...x, mode: m }))}
          options={[
            { value: 'default', title: 'Default', help: 'The engine generates the merit from a few choices' },
            { value: 'custom', title: 'Custom', help: 'Edit every operand' },
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
              title="Start a custom merit from the operands the default generates for this design"
              onClick={() => void convert()}
            >
              {converting ? <Loader2 className="animate-spin" /> : <ListPlus />}
              Convert to Custom
            </Button>
          </div>
        </>
      ) : (
        <OperandTable system={project.system} merit={merit} />
      )}
      {error && !evaluating && (
        <p className="flex items-start gap-1 py-1 text-[11px] text-warning" role="alert">
          <AlertTriangle className="mt-px size-3.5 shrink-0" />
          <span className="select-text whitespace-pre-wrap">The engine could not evaluate this merit: {error}</span>
        </p>
      )}
      <label className="flex items-center justify-between gap-2 border-t border-separator py-1.5" title="Keep the EFL at its start value as a constraint, unless the merit has an EFL operand.">
        <span>Hold the focal length</span>
        <Switch
          aria-label="Hold the focal length"
          checked={merit.holdEfl}
          onCheckedChange={(v) => editMerit('Hold Focal Length', (m) => ({ ...m, holdEfl: v }))}
        />
      </label>
    </Section>
  )
}

/** The merit at the current design, evaluated by the engine; dims from an edit until its answer. */
export function MeritReadout() {
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
      aria-label={`Current merit ${evaluation ? value : 'not evaluated'}`}
      aria-busy={evaluating}
    >
      {evaluating && <Loader2 className="size-3 animate-spin text-secondary" />}
      {error && !evaluating && <AlertTriangle className="size-3.5 text-warning" aria-label={`Evaluation failed: ${error}`} />}
      <div className="text-right leading-tight">
        <div className="text-[10px] text-secondary">Merit</div>
        <div className="num text-[13px] font-medium">{value}</div>
      </div>
    </div>
  )
}

// MARK: Default merit

function DefaultMeritForm({ system, merit }: { system: System; merit: MeritSettings }) {
  const d = merit.defaults
  const set = (name: string, change: (d: DefaultMerit) => Partial<DefaultMerit>) =>
    editMerit(name, (m) => ({ ...m, defaults: { ...m.defaults, ...change(m.defaults) } }))
  const unit = fieldUnit(system.fields.kind)
  // The engine's symmetry from the summary run (paraxial_xy.symmetry); else the surface kinds.
  const summary = useWorkspace((s) => s.runs[SUMMARY]?.result?.json)
  const anamorphic = anamorphicSummary(summary, system).anamorphic

  return (
    <div className="border-t border-separator pt-1">
      <FormRow label="Criterion">
        <Select aria-label="Criterion" value={d.criterion} onChange={(e) => set('Change Criterion', () => ({ criterion: e.target.value as Criterion }))} className="w-full">
          {CRITERIA.map((c) => (
            <option key={c.value} value={c.value}>
              {c.title}
            </option>
          ))}
        </Select>
      </FormRow>
      {d.criterion !== 'mtf' && (
        <FormRow label="Reference">
          <Select aria-label="Reference" value={d.reference} onChange={(e) => set('Change Reference', () => ({ reference: e.target.value }))} className="w-full">
            <option value="centroid">{d.criterion === 'wavefront' ? 'Centroid (piston and tilt removed)' : 'Centroid'}</option>
            <option value="chief">{d.criterion === 'wavefront' ? 'Chief ray (piston removed)' : 'Chief ray'}</option>
          </Select>
        </FormRow>
      )}
      {d.criterion === 'rms_spot' && (
        <FormRow label="White light" help="One centroid for all wavelengths, so lateral color counts; otherwise one per wavelength.">
          <Switch
            aria-label="White-light centroid"
            checked={d.scope === 'polychromatic'}
            onCheckedChange={(v) => set('Change Centroid', () => ({ scope: v ? 'polychromatic' : 'monochromatic' }))}
          />
        </FormRow>
      )}
      {d.criterion === 'mtf' && (
        <>
          <FormRow label="Frequencies" unit="cyc/mm">
            <TextInput
              ariaLabel="MTF frequencies"
              value={d.mtfFrequencies.map((f) => sig(f, 6)).join(', ')}
              placeholder="10, 30"
              onCommit={(t) => {
                const v = parseList(t)?.filter((f) => f > 0)
                if (v?.length) set('Change MTF Frequencies', () => ({ mtfFrequencies: v }))
              }}
            />
          </FormRow>
          <FormRow label="Target">
            <NumberInput ariaLabel="MTF target" value={d.mtfTarget} min={0} max={1} onCommit={(v) => set('Change MTF Target', () => ({ mtfTarget: v ?? 0.5 }))} />
          </FormRow>
          <FormRow label="Method">
            <Select aria-label="MTF method" value={d.mtfMethod} onChange={(e) => set('Change MTF Method', () => ({ mtfMethod: e.target.value }))} className="w-full">
              <option value="diffraction">Diffraction</option>
              <option value="geometric">Geometric</option>
            </Select>
          </FormRow>
          <FormRow label="Weakest field" help="Maximise the weakest field and direction at each frequency (minimax), instead of the field-weighted bounds.">
            <Switch aria-label="Weakest field" checked={d.worstField} onCheckedChange={(v) => set('Change MTF Field Rule', () => ({ worstField: v }))} />
          </FormRow>
        </>
      )}
      <FormRow label="Sampling" help="Pupil sampling: rings (1…12) × arms (3…24)">
        <span className="flex w-full items-center gap-1">
          <NumberInput ariaLabel="Pupil rings" value={d.rings} integer min={1} max={12} onCommit={(v) => set('Change Pupil Rings', () => ({ rings: v ?? 3 }))} />
          <span className="text-secondary">×</span>
          <NumberInput ariaLabel="Pupil arms" value={d.arms} integer min={3} max={24} onCommit={(v) => set('Change Pupil Arms', () => ({ arms: v ?? 6 }))} />
        </span>
      </FormRow>
      <div className="flex flex-wrap gap-2 py-1">
        <IndexChecklist
          title="Fields"
          all={system.fields.fields.map((f, k) => fieldMenuLabel(k, f, unit))}
          selected={d.fields}
          onChange={(v) => set('Change Merit Fields', () => ({ fields: v }))}
        />
        <IndexChecklist
          title="Wavelengths"
          all={system.wavelengths.wavelengths.map((w) => `${w.um.toFixed(4)} µm`)}
          selected={d.wavelengths}
          onChange={(v) => set('Change Merit Wavelengths', () => ({ wavelengths: v }))}
        />
      </div>
      <ThicknessLimits d={d} set={set} />
      {(anamorphic || d.squeezeTarget != null) && (
        <FormRow label="Squeeze target" help="Hold the signed EFL_y / EFL_x at this value (anamorphic designs).">
          <NumberInput ariaLabel="Squeeze target" value={d.squeezeTarget} nullable placeholder="none" onCommit={(v) => set('Change Squeeze Target', () => ({ squeezeTarget: v }))} />
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
  const on = (k: number) => selected?.includes(k) ?? true
  const summary = !selected || selected.length >= all.length ? 'All' : selected.length === 1 ? `${selected[0] + 1}` : `${selected.length} of ${all.length}`
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
        <Button size="sm" aria-label={`${title}: ${summary}`}>
          {title}: {summary}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuItem onSelect={() => onChange(null)}>All</DropdownMenuItem>
        <DropdownMenuSeparator />
        {all.map((t, k) => (
          <DropdownMenuCheckboxItem key={k} checked={on(k)} onCheckedChange={(v) => toggle(k, v === true)} onSelect={(e) => e.preventDefault()}>
            <span className="num">{t}</span>
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

type LimitKey = 'glassMinCentre' | 'glassMaxCentre' | 'glassMinEdge' | 'airMinCentre' | 'airMaxCentre' | 'airMinEdge'

function ThicknessLimits({ d, set }: { d: DefaultMerit; set: (name: string, change: (d: DefaultMerit) => Partial<DefaultMerit>) => void }) {
  const field = (key: LimitKey, label: string) => (
    <NumberInput ariaLabel={label} value={d[key]} nullable placeholder="none" onCommit={(v) => set('Change Thickness Limits', () => ({ [key]: v }))} />
  )
  return (
    <div className="border-t border-separator py-1">
      <label className="flex items-center justify-between gap-2 py-0.5">
        <span>Thickness limits (mm)</span>
        <Switch aria-label="Thickness limits" checked={d.thickness} onCheckedChange={(v) => set('Change Thickness Limits', () => ({ thickness: v }))} />
      </label>
      {d.thickness && (
        <div className="grid grid-cols-[minmax(0,1fr)_4.5rem_4.5rem] items-center gap-x-1.5 gap-y-1 pt-1">
          <span />
          <span className="text-right text-[11px] text-secondary">Glass</span>
          <span className="text-right text-[11px] text-secondary">Air</span>
          <span className="text-secondary">Min center</span>
          {field('glassMinCentre', 'Glass min center')}
          {field('airMinCentre', 'Air min center')}
          <span className="text-secondary">Max center</span>
          {field('glassMaxCentre', 'Glass max center')}
          {field('airMaxCentre', 'Air max center')}
          <span className="text-secondary">Min edge</span>
          {field('glassMinEdge', 'Glass min edge')}
          {field('airMinEdge', 'Air min edge')}
        </div>
      )}
    </div>
  )
}

// MARK: Custom operands

function OperandTable({ system, merit }: { system: System; merit: MeritSettings }) {
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
    edit('Add Operand', (o) => {
      const r = addOperand(o, newOperand(d), first ?? (o.length ? o.length - 1 : null))
      at = r.index
      return r.operands
    })
    select([at], at)
    setPicker(false)
  }
  const del = () => {
    if (!selected.length) return
    edit(selected.length === 1 ? 'Delete Operand' : 'Delete Operands', (o) => deleteOperands(o, selected))
    const next = Math.min(selected[0], ops.length - selected.length - 1)
    select(next >= 0 ? [next] : [], null)
  }
  const dup = () => {
    if (!selected.length) return
    let at: number[] = []
    edit(selected.length === 1 ? 'Duplicate Operand' : 'Duplicate Operands', (o) => {
      const r = duplicateOperands(o, selected)
      at = r.indices
      return r.operands
    })
    select(at, at[0] ?? null)
  }
  const moveTo = (from: number[], gap: number) => {
    let at: number[] = from
    const changed = edit(from.length === 1 ? 'Move Operand' : 'Move Operands', (o) => {
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
            <Button size="iconSm" variant="ghost" aria-label="Add Operand" title="Add an operand after the selection">
              <Plus />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-[320px] p-0">
            <OperandPicker pick={add} />
          </PopoverContent>
        </Popover>
        <Button size="iconSm" variant="ghost" aria-label="Delete" title={many ? 'Delete the selected operands (⌫)' : 'Delete the selected operand (⌫)'} disabled={!selected.length} onClick={del}>
          <Minus />
        </Button>
        <Button size="iconSm" variant="ghost" aria-label="Duplicate" title={many ? 'Duplicate the selected operands' : 'Duplicate the selected operand'} disabled={!selected.length} onClick={dup}>
          <Copy />
        </Button>
        <Button size="iconSm" variant="ghost" aria-label="Move Up" title={`Move up (${mod}↑)`} disabled={sel == null || sel === 0} onClick={() => move(-1)}>
          <ArrowUp />
        </Button>
        <Button size="iconSm" variant="ghost" aria-label="Move Down" title={`Move down (${mod}↓)`} disabled={sel == null || sel >= ops.length - 1} onClick={() => move(1)}>
          <ArrowDown />
        </Button>
        <span className="ml-auto text-[11px] text-secondary">
          {many ? `${selected.length} of ${ops.length} selected` : `${ops.length} operand${ops.length === 1 ? '' : 's'}`}
        </span>
      </div>
      {ops.length === 0 ? (
        <p className="py-4 text-center text-secondary">No operands. Add one with +, or convert the default merit.</p>
      ) : (
        <div
          ref={list}
          role="listbox"
          aria-label="Merit operands"
          aria-multiselectable
          tabIndex={0}
          title="Drag to reorder; ⇧- or ⌘-click to select several"
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
      {many && <p className="py-2 text-center text-[11px] text-secondary">{selected.length} operands selected. Select one to edit it.</p>}
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
  const problem = row?.error ? `The engine could not evaluate it: ${row.error}` : (issues[0] ?? null)
  const parts: string[] = []
  const params = operandParameters(operand)
  if (params) parts.push(params)
  parts.push(`${comparisonSymbol(operand.comparison)} ${sig(operand.target, 4)}`)
  if (!operand.constraint) parts.push(`w ${sig(operand.weight, 4)}`)
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
          {operand.constraint && <span className="shrink-0 rounded-full bg-selected px-1.5 text-[10px] text-secondary">constraint</span>}
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
        {desc?.unit && <span className="text-[11px] text-secondary">{desc.unit}</span>}
      </div>
      {desc?.summary && <p className="pb-1 text-[11px] text-secondary">{desc.summary}</p>}
      {desc?.params.map((p) => <ParamEditor key={p.key} p={p} operand={operand} system={system} update={update} />)}
      {!desc && <p className="num truncate py-1 text-[11px] text-secondary">Parameters: {JSON.stringify(operand.kind)}</p>}
      <FormRow label="Field">
        <Select
          aria-label="Field"
          className="w-full"
          value={fieldValue}
          onChange={(e) => {
            const v = e.target.value
            update('Change Operand Field', (op) => ({ ...op, field: (v === 'all' ? { field: 'all' } : { field: 'index', index: Number(v) }) as JsonObject }))
          }}
        >
          {fieldValue === 'other' && <option value="other">Normalized</option>}
          {(!single || fieldValue === 'all') && <option value="all">All fields (weighted)</option>}
          {system.fields.fields.map((f, k) => (
            <option key={k} value={k}>
              {fieldMenuLabel(k, f, unit)}
            </option>
          ))}
        </Select>
      </FormRow>
      <FormRow label="Wavelength">
        <Select
          aria-label="Wavelength"
          className="w-full"
          value={wlValue}
          onChange={(e) => {
            const v = e.target.value
            update('Change Operand Wavelength', (op) => ({
              ...op,
              wavelength: (v === 'primary' ? { wavelength: 'primary' } : v === 'all' ? { wavelength: 'all' } : { wavelength: 'index', index: Number(v) }) as JsonObject,
            }))
          }}
        >
          <option value="primary">Primary</option>
          <option value="all">All (polychromatic)</option>
          {system.wavelengths.wavelengths.map((wv, k) => (
            <option key={k} value={k}>
              {wv.um.toFixed(4)} µm
            </option>
          ))}
        </Select>
      </FormRow>
      <FormRow label="Target" help="= target, ≤ an upper bound, ≥ a lower bound">
        <span className="flex w-full items-center gap-1.5">
          <Segmented<Comparison>
            label="Comparison"
            value={operand.comparison}
            onChange={(c) => update('Change Comparison', (op) => ({ ...op, comparison: c }))}
            options={COMPARISONS.map((c) => ({ value: c.value, title: c.symbol, help: c.help }))}
          />
          <NumberInput ariaLabel="Target" value={operand.target} onCommit={(v) => update('Change Target', (op) => ({ ...op, target: v ?? 0 }))} />
        </span>
      </FormRow>
      <FormRow label="Constraint" help="Hold it exactly (or as a bound) instead of weighting it in the merit">
        <Switch
          aria-label="Constraint"
          checked={operand.constraint}
          onCheckedChange={(v) => update(v ? 'Make Constraint' : 'Make Residual', (op) => ({ ...op, constraint: v }))}
        />
      </FormRow>
      {!operand.constraint && (
        <FormRow label="Weight">
          <NumberInput ariaLabel="Weight" value={operand.weight} min={0} onCommit={(v) => update('Change Weight', (op) => ({ ...op, weight: Math.max(v ?? 1, 0) }))} />
        </FormRow>
      )}
      <FormRow label="Enabled">
        <Switch aria-label="Enabled" checked={operand.enabled} onCheckedChange={(v) => update(v ? 'Enable Operand' : 'Disable Operand', (op) => ({ ...op, enabled: v }))} />
      </FormRow>
      {row && (
        <p className="num py-0.5 text-right text-[12px]">
          Value {sig(row.value, 6)}
          {Number.isFinite(row.fraction) && <span className="text-secondary"> · {percentShare(row.fraction)}</span>}
        </p>
      )}
      {row?.error && <p className="select-text py-0.5 text-[11px] text-warning">{row.error}</p>}
      {issues.map((t) => (
        <p key={t} className="py-0.5 text-[11px] text-warning">
          {t}
        </p>
      ))}
    </div>
  )
}

function ParamEditor({ p, operand, system, update }: { p: OperandParam; operand: Operand; system: System; update: (name: string, change: (op: Operand) => Operand) => void }) {
  const v = operand.kind[p.key]
  const name = `Change ${p.title}`
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
            placeholder={p.required ? '0' : 'auto'}
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
            {!p.required && <option value="">{p.key === 'to' ? 'Next' : p.key === 'from' ? 'First' : 'Image'}</option>}
            {p.required && typeof v !== 'number' && <option value="">Choose…</option>}
            {system.surfaces.slice(1).map((s, j) => (
              <option key={s.id} value={s.id}>
                {j + 1 === img ? 'Image' : `Surface ${j + 1}${s.label ? ' · ' + s.label : ''}`}
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
        <FormRow label="Sampling" help="Pupil sampling: rings (1…12) × arms (3…24)">
          <span className="flex w-full items-center gap-1">
            <NumberInput ariaLabel="Rings" value={rings} integer min={1} max={12} onCommit={(x) => setKind({ ...o, rings: x ?? 3, arms })} />
            <span className="text-secondary">×</span>
            <NumberInput ariaLabel="Arms" value={arms} integer min={3} max={24} onCommit={(x) => setKind({ ...o, rings, arms: x ?? 6 })} />
          </span>
        </FormRow>
      )
    }
    case 'operandList': {
      const idx = Array.isArray(v) ? v.filter((x): x is number => typeof x === 'number') : []
      return (
        <FormRow label={p.title} help="Numbers of earlier operands, e.g. 1, 3">
          <TextInput
            ariaLabel={p.title}
            value={idx.map((x) => x + 1).join(', ')}
            placeholder="1, 2"
            onCommit={(t) => {
              const parts = t.split(/[;,\s]+/).filter(Boolean)
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
  const [query, setQuery] = useState('')
  const groups = searchOperands(query)
  return (
    <div className="flex max-h-[380px] flex-col" data-testid="operand-picker">
      <div className="p-2">
        <input
          autoFocus
          aria-label="Search operands"
          placeholder="Search operands"
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
        {groups.length === 0 && <p className="px-2 py-3 text-secondary">No operand matches.</p>}
        {groups.map((g) => (
          <div key={g.category}>
            <div className="px-2 pb-0.5 pt-1.5 text-[11px] font-semibold text-secondary">{g.category}</div>
            {g.items.map((d) => (
              <button
                key={d.type}
                type="button"
                className="block w-full cursor-default rounded-md px-2 py-1 text-left hover:bg-accent hover:text-on-accent focus-visible:bg-accent focus-visible:text-on-accent focus-visible:outline-none [&:hover_*]:text-on-accent"
                onClick={() => pick(d)}
              >
                <span className="flex items-baseline justify-between gap-2">
                  <span className="text-[13px]">{d.title}</span>
                  {d.unit && <span className="text-[11px] text-secondary">{d.unit}</span>}
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
