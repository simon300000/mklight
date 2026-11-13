// The tolerance table (TolerancePanel.swift `ToleranceTableSection`,
// `ToleranceRowView`, `ToleranceGeneratorView`, `TestWavelengthField`): the
// default-tolerance generator, add / duplicate / delete, the test wavelength,
// and the rows grouped by element, each with kind, surface menu, limits and
// unit, its problem in red, and a context menu for unit, distribution and
// symmetry. Every change is one undoable step saved in the document.

import { ChevronRight, Copy, Link2, Minus, Plus, Waves, Wand2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuSeparator, ContextMenuTrigger } from '@/components/ui/context-menu'
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
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { cn, isMac } from '@/lib/utils'
import { surfaceIndexOfId } from '@/model/params'
import { imageIndex, type System } from '@/model/system'
import {
  ADDABLE_ELEMENT_KINDS,
  ADDABLE_SURFACE_KINDS,
  DEFAULT_TEST_WAVELENGTH_UM,
  DISTRIBUTIONS,
  KINDS,
  PRESETS,
  UNITS,
  addRows,
  convertRow,
  deleteRows,
  distributionJson,
  distributionTitle,
  duplicateRows,
  elementMenuTitle,
  elementTitle,
  generateRows,
  generatorValue,
  groupRows,
  hasGlass,
  isElementKind,
  isSolid,
  lensGroups,
  limitsText,
  makeRow,
  materialName,
  movingSurfaces,
  pivotIndex,
  presetGenerator,
  rowLabel,
  sameDistribution,
  setElement,
  short,
  surfaceLabel,
  surfacesFor,
  unitLabel,
  type Distribution,
  type Generator,
  type GeneratorValue,
  type LensGroup,
  type Preset,
  type ToleranceContext,
  type ToleranceKind,
  type ToleranceRow,
  type ToleranceSettings,
  type ToleranceUnit,
} from '@/model/tolerance'
import { editTolerances, useTolerance } from '@/state/tolerance'
import { NumberInput } from '../../SettingsForm'
import { Section } from '../../inspector/Form'
import { Hint } from './controls'

type Edit = typeof editTolerances

export function ToleranceTable({ settings, context, issues }: { settings: ToleranceSettings; context: ToleranceContext; issues: Map<string, string> }) {
  const system = context.system
  const rows = settings.rows
  const selection = new Set(useTolerance((s) => s.selection))
  const collapsed = new Set(useTolerance((s) => s.collapsed))
  const select = (ids: string[], anchorId?: string | null) => {
    useTolerance.setState({ selection: ids })
    if (anchorId !== undefined) anchor.current = anchorId
  }
  // The row a ⇧-click range starts from (the last plain or ⌘-click).
  const anchor = useRef<string | null>(null)
  const groups = groupRows(rows, system)
  // The rows in the order they are shown (collapsed groups left out).
  const visible = groups.flatMap((g) => (collapsed.has(g.key) ? [] : g.rows.map((r) => r.id)))
  const click = (id: string, mode: 'single' | 'toggle' | 'range') => {
    if (mode === 'single') return select([id], id)
    if (mode === 'toggle') return select(selection.has(id) ? [...selection].filter((x) => x !== id) : [...selection, id], id)
    const a = anchor.current && visible.includes(anchor.current) ? anchor.current : ([...selection].find((x) => visible.includes(x)) ?? id)
    const i = visible.indexOf(a), j = visible.indexOf(id)
    select(visible.slice(Math.min(i, j), Math.max(i, j) + 1))
  }
  // The first element row of each start surface shows the shared range and pivot.
  const leading = new Set<string>()
  const seen = new Set<number>()
  for (const r of rows) if (isElementKind(r.kind) && !seen.has(r.surface)) (seen.add(r.surface), leading.add(r.id))

  const duplicate = (ids: Set<string>) => {
    if (!ids.size) return
    let added: string[] = []
    editTolerances(ids.size === 1 ? 'Duplicate Tolerance' : 'Duplicate Tolerances', (t) => {
      added = duplicateRows(t, ids)
    })
    select(added)
  }
  const remove = (ids: Set<string>) => {
    if (!ids.size) return
    editTolerances(ids.size === 1 ? 'Delete Tolerance' : 'Delete Tolerances', (t) => deleteRows(t, ids))
    select([...selection].filter((id) => !ids.has(id)))
  }
  const add = (row: ToleranceRow, at: number) => {
    // After the last row on this surface or before it.
    const after = [...rows].reverse().find((r) => surfaceIndexOfId(system, r.surface) <= at)?.id ?? null
    editTolerances('Add Tolerance', (t) => addRows(t, [row], after))
    select([row.id], row.id)
  }

  return (
    <Section
      title="Tolerances"
      accessory={<span className="num text-[11px] text-secondary">{rows.length}</span>}
    >
      <div
        className="outline-none"
        tabIndex={-1}
        onKeyDown={(e) => {
          const t = e.target as HTMLElement
          if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA') return
          if ((e.key === 'Backspace' || e.key === 'Delete') && selection.size) {
            e.preventDefault()
            remove(selection)
          } else if (e.key.toLowerCase() === 'd' && (isMac ? e.metaKey : e.ctrlKey) && selection.size) {
            e.preventDefault()
            duplicate(selection)
          }
        }}
      >
        <div className="flex items-center gap-1 py-1">
          <GeneratorButton settings={settings} system={system} />
          <span className="flex-1" />
          <AddMenu system={system} settings={settings} onAdd={add} />
          <Button
            variant="ghost"
            size="iconSm"
            aria-label="Duplicate"
            title={`Duplicate the selected tolerances (${isMac ? '⌘D' : 'Ctrl+D'})`}
            disabled={!selection.size}
            onClick={() => duplicate(selection)}
          >
            <Copy />
          </Button>
          <Button variant="ghost" size="iconSm" aria-label="Delete" title="Delete the selected tolerances (⌫)" disabled={!selection.size} onClick={() => remove(selection)}>
            <Minus />
          </Button>
        </div>
        <TestWavelengthField value={settings.testWavelengthUm} onCommit={(v) => editTolerances('Change Test Wavelength', (t) => void (t.testWavelengthUm = v))} />
        {!rows.length ? (
          <p className="py-3 text-[12px] text-secondary">No tolerances. Use Default Tolerances, or add them one by one.</p>
        ) : (
          <div role="table" aria-label="Tolerances" className="pb-1">
            <div className="grid grid-cols-[minmax(0,1fr)_3.25rem_6.75rem_1.75rem] gap-x-1.5 px-1 pt-1 text-[10px] text-tertiary" aria-hidden>
              <span>Kind</span>
              <span>Surface</span>
              <span className="text-right">− / + limits</span>
              <span>Unit</span>
            </div>
            {groups.map((g) => {
              const closed = collapsed.has(g.key)
              const problems = g.rows.filter((r) => issues.has(r.id)).length
              return (
                <div key={g.key} role="rowgroup">
                  <button
                    type="button"
                    className="mt-1 flex w-full cursor-default items-center gap-1 rounded px-0.5 py-0.5 text-left text-[11px] font-semibold text-secondary hover:bg-hover"
                    aria-expanded={!closed}
                    title={closed ? `Show the ${g.rows.length} tolerances of ${g.key}` : `Hide the tolerances of ${g.key}`}
                    onClick={() => {
                      const next = new Set(collapsed)
                      if (closed) next.delete(g.key)
                      else next.add(g.key)
                      useTolerance.setState({ collapsed: [...next] })
                    }}
                  >
                    <ChevronRight className={cn('size-3 shrink-0 transition-transform', !closed && 'rotate-90')} />
                    <span className="min-w-0 flex-1 truncate">{g.title}</span>
                    {closed && problems > 0 && <span className="text-danger">{problems} to fix</span>}
                    {closed && <span className="num">{g.rows.length}</span>}
                  </button>
                  {!closed &&
                    g.rows.map((r) => (
                      <RowView
                        key={r.id}
                        row={r}
                        context={context}
                        issue={issues.get(r.id)}
                        showsElementLine={leading.has(r.id)}
                        selected={selection.has(r.id)}
                        onSelect={(mode) => click(r.id, mode)}
                        onDuplicate={() => duplicate(new Set([r.id]))}
                        onDelete={() => remove(new Set([r.id]))}
                      />
                    ))}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </Section>
  )
}

// MARK: Rows

function RowView({
  row,
  context,
  issue,
  showsElementLine,
  selected,
  onSelect,
  onDuplicate,
  onDelete,
}: {
  row: ToleranceRow
  context: ToleranceContext
  issue?: string
  showsElementLine: boolean
  selected: boolean
  onSelect: (mode: 'single' | 'toggle' | 'range') => void
  onDuplicate: () => void
  onDelete: () => void
}) {
  const system = context.system
  const symmetric = row.minus === row.plus
  const set = (name: string, change: (r: ToleranceRow) => void, edit: Edit = editTolerances) =>
    edit(name, (t) => {
      const r = t.rows.find((x) => x.id === row.id)
      if (r) change(r)
    })
  const convert = (u: ToleranceUnit) => {
    const c = convertRow(row, u, context)
    if (c) set('Change Tolerance Unit', (r) => Object.assign(r, { ...c, id: r.id }))
  }
  const units = KINDS[row.kind].units
  const label = rowLabel(row, system)
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div
          role="row"
          aria-selected={selected}
          aria-label={`${label}, ${limitsText(row)}`}
          className={cn('rounded-[5px] px-1 py-[1px]', selected && 'bg-accent-soft')}
          onContextMenu={() => !selected && onSelect('single')}
        >
          <div className="grid grid-cols-[minmax(0,1fr)_3.25rem_6.75rem_1.75rem] items-center gap-x-1.5">
            <span
              className="min-w-0 cursor-default truncate"
              title={label}
              onMouseDown={(e) => e.shiftKey && e.preventDefault()}
              onClick={(e) => onSelect(e.shiftKey ? 'range' : (isMac ? e.metaKey : e.ctrlKey) ? 'toggle' : 'single')}
            >
              {row.kind === 'other' ? label : KINDS[row.kind].short}
            </span>
            <SurfaceMenu row={row} system={system} set={set} />
            <span className="flex items-center justify-end gap-0.5">
              {symmetric ? (
                <>
                  <span className="text-secondary">±</span>
                  <NumberInput
                    className="w-[4.75rem]"
                    ariaLabel={`${label} limit`}
                    value={row.minus}
                    min={0}
                    onCommit={(v) => v != null && set('Change Tolerance', (r) => Object.assign(r, { minus: v, plus: v }))}
                  />
                </>
              ) : (
                <>
                  <span className="text-secondary">−</span>
                  <NumberInput className="w-[2.75rem]" ariaLabel={`${label} minus limit`} value={row.minus} min={0} onCommit={(v) => v != null && set('Change Tolerance', (r) => void (r.minus = v))} />
                  <span className="text-secondary">+</span>
                  <NumberInput className="w-[2.75rem]" ariaLabel={`${label} plus limit`} value={row.plus} min={0} onCommit={(v) => v != null && set('Change Tolerance', (r) => void (r.plus = v))} />
                </>
              )}
            </span>
            {units.length < 2 ? (
              <span className="truncate text-[12px] text-secondary" title={UNITS[row.unit].title}>
                {unitLabel(row, system)}
              </span>
            ) : (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button type="button" className="cursor-default truncate rounded px-0.5 text-left text-[12px] text-secondary hover:bg-hover" title={`Unit: ${UNITS[row.unit].title}. Changing it converts the limits.`}>
                    {unitLabel(row, system)}
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {units.map((u) => (
                    <DropdownMenuItem key={u} inset disabled={!convertRow(row, u, context)} onSelect={() => convert(u)}>
                      {u === row.unit && <span className="absolute left-2">✓</span>}
                      {UNITS[u].title}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
          {isElementKind(row.kind) && showsElementLine && <ElementLine row={row} system={system} />}
          {issue && <p className="py-0.5 pl-1 text-[11px] leading-snug text-danger">{issue}</p>}
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        {units.length > 1 && (
          <>
            <div className="px-2 pt-1 pb-0.5 text-[11px] font-semibold text-secondary">Unit</div>
            {units.map((u) => (
              <ContextMenuItem key={u} className="pl-6" disabled={!convertRow(row, u, context)} onSelect={() => convert(u)}>
                {u === row.unit && <span className="absolute left-2">✓</span>}
                {UNITS[u].title}
              </ContextMenuItem>
            ))}
            <ContextMenuSeparator />
          </>
        )}
        <div className="px-2 pt-1 pb-0.5 text-[11px] font-semibold text-secondary">Distribution</div>
        {DISTRIBUTIONS.map((d) => (
          <ContextMenuItem key={JSON.stringify(distributionJson(d))} className="pl-6" onSelect={() => set('Change Distribution', (r) => void (r.distribution = d))}>
            {sameDistribution(d, row.distribution) && <span className="absolute left-2">✓</span>}
            {distributionTitle(d)}
          </ContextMenuItem>
        ))}
        <ContextMenuSeparator />
        <ContextMenuItem
          onSelect={() =>
            set(symmetric ? 'Make Tolerance Asymmetric' : 'Make Tolerance Symmetric', (r) => {
              if (r.minus === r.plus) r.plus = r.minus * 2
              else r.plus = r.minus
            })
          }
        >
          {symmetric ? 'Make Asymmetric' : 'Make Symmetric'}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem shortcut="mod+d" onSelect={onDuplicate}>
          Duplicate
        </ContextMenuItem>
        <ContextMenuItem className="text-danger" onSelect={onDelete}>
          Delete
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

type SetRow = (name: string, change: (r: ToleranceRow) => void) => void

const surfaceOptions = (s: System) => Array.from({ length: Math.max(imageIndex(s) - 1, 0) }, (_, k) => k + 1)

/** The surface (a thickness is the gap after it), or for an element its range. */
function SurfaceMenu({ row, system, set }: { row: ToleranceRow; system: System; set: SetRow }) {
  const el = isElementKind(row.kind)
  const text = el ? (elementTitle(row, system) ?? surfaceLabel(row, system)) : surfaceLabel(row, system)
  if (row.kind === 'other') return <span className="num truncate text-[12px] text-secondary">{text}</span>
  const moving = movingSurfaces(row, system)
  const pivotChoice = pivotChoiceOf(row, system)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="num cursor-default truncate rounded px-0.5 text-left text-[12px] hover:bg-hover"
          title={el ? 'The element: the surfaces that move together' : 'The surface (a thickness is the gap after it)'}
        >
          {text}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {el ? (
          <>
            <DropdownMenuLabel>Element</DropdownMenuLabel>
            {lensGroups(system).map((g) => {
              const f = g.surfaces[0], l = g.surfaces[g.surfaces.length - 1]
              return (
                <DropdownMenuItem key={g.title} onSelect={() => moveElement(row, system, f, l, null)}>
                  {g.title} · S{f}–{l}
                </DropdownMenuItem>
              )
            })}
            <DropdownMenuSeparator />
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>First Surface</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {surfaceOptions(system).map((i) => (
                  <DropdownMenuItem key={i} disabled={i === moving[0]} onSelect={() => moveElement(row, system, i, Math.max(i, moving[moving.length - 1] ?? i), pivotChoice)}>
                    S{i}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>Last Surface</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {surfaceOptions(system)
                  .filter((i) => i >= (moving[0] ?? 1))
                  .map((i) => (
                    <DropdownMenuItem key={i} disabled={i === moving[moving.length - 1]} onSelect={() => moveElement(row, system, moving[0] ?? i, i, pivotChoice)}>
                      S{i}
                    </DropdownMenuItem>
                  ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          </>
        ) : (
          surfaceOptions(system).map((i) => (
            <DropdownMenuItem key={i} onSelect={() => set('Change Tolerance Surface', (r) => void (r.surface = system.surfaces[i].id))}>
              S{i}
              {i === system.stop ? ' (stop)' : ''}
            </DropdownMenuItem>
          ))
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** Where the pivot sits relative to the range, kept when the range moves. */
type PivotChoice = { at: 'first' } | { at: 'last' } | { at: 'surface'; index: number }

function pivotChoiceOf(row: ToleranceRow, s: System): PivotChoice {
  const m = movingSurfaces(row, s)
  const p = pivotIndex(row, s)
  if (p == null || p === m[0]) return { at: 'first' }
  return p === m[m.length - 1] ? { at: 'last' } : { at: 'surface', index: p }
}

function moveElement(row: ToleranceRow, s: System, f: number, l: number, c: PivotChoice | null, name = 'Change Element') {
  if (!s.surfaces[f] || !s.surfaces[l]) return
  const choice = c ?? { at: 'first' }
  const pivot = choice.at === 'first' ? null : choice.at === 'last' ? s.surfaces[l].id : (s.surfaces[choice.index]?.id ?? null)
  editTolerances(name, (t) => setElement(t, row.id, s.surfaces[f].id, s.surfaces[l].id, pivot))
}

/** "S3 S4 S5 move together · pivot S3, first vertex", the pivot a menu. */
function ElementLine({ row, system }: { row: ToleranceRow; system: System }) {
  const moving = movingSurfaces(row, system)
  const p = pivotIndex(row, system)
  const f = moving[0], l = moving[moving.length - 1]
  const pivotText = p == null ? '?' : p === f ? `S${p}, first vertex` : p === l ? `S${p}, last vertex` : `S${p}`
  const setPivot = (c: PivotChoice) => f != null && moveElement(row, system, f, l, c, 'Change Element Pivot')
  return (
    <div className="flex items-center gap-1 pl-1 text-[11px] text-secondary" aria-label={`Moves ${moving.map((i) => `surface ${i}`).join(', ')}, pivot ${pivotText}`}>
      <Link2 className="size-3 shrink-0 text-tertiary" />
      <span className="truncate">
        {moving.map((i) => `S${i}`).join(' ')}
        {moving.length > 1 ? ' move together' : ' moves'}
      </span>
      <span className="text-tertiary">·</span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            className="cursor-default whitespace-nowrap rounded px-0.5 text-accent hover:bg-hover"
            title={`The vertex the element tilts about; every element tolerance starting on S${f ?? 0} uses it`}
          >
            pivot {pivotText}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {f != null && <DropdownMenuItem onSelect={() => setPivot({ at: 'first' })}>First Vertex (S{f})</DropdownMenuItem>}
          {l != null && l !== f && <DropdownMenuItem onSelect={() => setPivot({ at: 'last' })}>Last Vertex (S{l})</DropdownMenuItem>}
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>Surface</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {surfaceOptions(system).map((i) => (
                <DropdownMenuItem key={i} onSelect={() => setPivot({ at: 'surface', index: i })}>
                  S{i}
                  {i === system.stop ? ' (stop)' : ''}
                </DropdownMenuItem>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

// MARK: Add

function surfaceMenuTitle(s: System, i: number): string {
  const sf = s.surfaces[i]
  let t = `S${i}`
  if (i === s.stop) t += ' (stop)'
  if (sf.label) t += ' ' + sf.label
  if (isSolid(s, i) && hasGlass(sf.material)) t += ' · ' + materialName(sf.material)
  return t
}

function AddMenu({ system, settings, onAdd }: { system: System; settings: ToleranceSettings; onAdd: (row: ToleranceRow, at: number) => void }) {
  const g = settings.generator
  const addSurface = (k: ToleranceKind, i: number) => onAdd(makeRow(k, system.surfaces[i].id, generatorValue(g, k), { distribution: g.distribution }), i)
  const addElement = (k: ToleranceKind, e: LensGroup) => {
    const f = e.surfaces[0], l = e.surfaces[e.surfaces.length - 1]
    onAdd(makeRow(k, system.surfaces[f].id, generatorValue(g, k), { last: system.surfaces[l].id, distribution: g.distribution }), l)
  }
  const elements = lensGroups(system)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" title="Add a tolerance on a surface or an element">
          <Plus className="size-3.5" />
          Add
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Surface</DropdownMenuLabel>
        {ADDABLE_SURFACE_KINDS.map((k) => {
          const surfaces = surfacesFor(k, system)
          return (
            <DropdownMenuSub key={k}>
              <DropdownMenuSubTrigger disabled={!surfaces.length}>{KINDS[k].title}</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {surfaces.map((i) => (
                  <DropdownMenuItem key={i} onSelect={() => addSurface(k, i)}>
                    {surfaceMenuTitle(system, i)}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          )
        })}
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Element</DropdownMenuLabel>
        {ADDABLE_ELEMENT_KINDS.map((k) => (
          <DropdownMenuSub key={k}>
            <DropdownMenuSubTrigger disabled={!elements.length}>{KINDS[k].title}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {elements.map((e) => (
                <DropdownMenuItem key={e.title} onSelect={() => addElement(k, e)}>
                  {elementMenuTitle(e)}
                </DropdownMenuItem>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// MARK: Test wavelength

function TestWavelengthField({ value, onCommit }: { value: number; onCommit: (um: number) => void }) {
  return (
    <div
      className="flex items-center gap-1.5 py-1 text-[12px]"
      title="The interferometer wavelength: one test fringe is half a wave of sag at it. Power and irregularity tolerances are counted in these fringes."
    >
      <Waves className="size-3.5 text-secondary" />
      <span>Test fringes at</span>
      <NumberInput
        className="w-[4.5rem]"
        ariaLabel="Test wavelength"
        value={Number((value * 1000).toPrecision(8))}
        placeholder="632.8"
        onCommit={(v) => v != null && v >= 200 && v <= 20000 && onCommit(v / 1000)}
      />
      <span className="text-secondary">nm</span>
      {Math.abs(value - DEFAULT_TEST_WAVELENGTH_UM) > 1e-9 && (
        <button type="button" className="cursor-default text-accent hover:underline" title="Back to 632.8 nm" onClick={() => onCommit(DEFAULT_TEST_WAVELENGTH_UM)}>
          He–Ne
        </button>
      )}
    </div>
  )
}

// MARK: Default tolerances

const GENERATOR_ROWS: { group?: string; title: string; key: GeneratorValue; unit: string; help: (nm: string) => string }[] = [
  { group: 'Surfaces', title: 'Radius (power)', key: 'radiusFringes', unit: 'fringes', help: (nm) => `Power error in test fringes at ${nm} nm over the clear aperture: Δc = N λ / h².` },
  { title: 'Irregularity', key: 'irregularityFringes', unit: 'fringes', help: (nm) => `Astigmatic and spherical irregularity of every lens surface, each this many test fringes at ${nm} nm (peak to valley over the clear aperture).` },
  { title: 'Thickness', key: 'thicknessMM', unit: 'mm', help: () => 'Every glass and air thickness inside the lens; the back focus is the compensator.' },
  { title: 'Decenter', key: 'decentreMM', unit: 'mm', help: () => 'Every lens surface alone, in x and in y; the rest of the system stays on the axis.' },
  { title: 'Tilt (wedge)', key: 'tiltArcmin', unit: '′', help: () => 'Every lens surface alone about x and about y, in arc minutes.' },
  { group: 'Elements', title: 'Decenter', key: 'elementDecentreMM', unit: 'mm', help: () => 'Every element as a rigid body (a cemented group together), in x and in y.' },
  { title: 'Tilt', key: 'elementTiltArcmin', unit: '′', help: () => 'Every element as a rigid body about its first vertex, about x and about y; the pivot can be changed per element.' },
  { group: 'Glasses', title: 'Index', key: 'index', unit: 'Δn_d', help: () => "n_d of every glass: a model glass's n_d, a catalog glass's melt offset Δn_d." },
  { title: 'Abbe number', key: 'abbePercent', unit: '% of V_d', help: () => "V_d of every glass: a model glass's V_d, a catalog glass's melt offset ΔV_d." },
]

function GeneratorButton({ settings, system }: { settings: ToleranceSettings; system: System }) {
  const [open, setOpen] = useState(false)
  const g = settings.generator
  const nm = short(settings.testWavelengthUm * 1000)
  const setG = (name: string, change: (g: Generator) => void) => editTolerances(name, (t) => change(t.generator))
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" title="Generate a tolerance set for every element from a class of manufacture">
          <Wand2 className="size-3.5" />
          Default Tolerances…
        </Button>
      </PopoverTrigger>
      <PopoverContent side="left" align="start" className="w-[330px]" aria-label="Default Tolerances">
        <h3 className="mb-2 text-[13px] font-semibold">Default Tolerances</h3>
        <Segmented<Preset | 'custom'>
          label="Class"
          className="mb-2 w-full [&>button]:flex-1"
          value={g.preset ?? 'custom'}
          options={[...PRESETS.map((p) => ({ value: p.value as Preset | 'custom', title: p.title })), ...(g.preset == null ? [{ value: 'custom' as const, title: 'Custom' }] : [])]}
          onChange={(p) => p !== 'custom' && setG('Choose Tolerance Class', (x) => Object.assign(x, presetGenerator(p, x.distribution)))}
        />
        <div className="grid grid-cols-[minmax(0,1fr)_5rem_3.5rem] items-center gap-x-2 gap-y-1 text-[12px]">
          {GENERATOR_ROWS.map((r) => (
            <div key={r.group ?? '' + r.key} className="contents">
              {r.group && <div className="col-span-3 pt-1.5 text-[11px] font-semibold text-secondary">{r.group}</div>}
              <span title={r.help(nm)}>{r.title}</span>
              <NumberInput
                ariaLabel={`${r.group ?? ''} ${r.title}`.trim()}
                value={g[r.key]}
                min={0}
                placeholder="0"
                onCommit={(v) =>
                  v != null &&
                  v >= 0 &&
                  setG('Change Default Tolerance', (x) => {
                    x[r.key] = v
                    x.preset = null
                  })
                }
              />
              <span className="truncate text-secondary">{r.unit}</span>
            </div>
          ))}
        </div>
        <label className="mt-2 flex items-center gap-2 text-[12px]" title="How Monte Carlo trials draw each perturbation within its limits">
          <span>Distribution</span>
          <Select
            className="flex-1"
            value={JSON.stringify(distributionJson(g.distribution))}
            onChange={(e) => setG('Change Distribution', (x) => void (x.distribution = JSON.parse(e.target.value) as Distribution))}
          >
            {DISTRIBUTIONS.map((d) => (
              <option key={JSON.stringify(distributionJson(d))} value={JSON.stringify(distributionJson(d))}>
                {distributionTitle(d)}
              </option>
            ))}
          </Select>
        </label>
        <Hint className="mt-1.5">A value of 0 leaves that kind out.</Hint>
        <div className="mt-3 flex justify-end gap-2">
          <Button size="sm" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            variant="prominent"
            title="Replace every tolerance by this class's set for each element"
            onClick={() => {
              editTolerances('Replace Tolerances', (t) => void (t.rows = generateRows(t.generator, system)))
              useTolerance.setState({ selection: [] })
              setOpen(false)
            }}
          >
            Replace Tolerances
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
