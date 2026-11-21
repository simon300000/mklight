// The tolerance table (TolerancePanel.swift `ToleranceTableSection`,
// `ToleranceRowView`, `ToleranceGeneratorView`, `TestWavelengthField`): the
// default-tolerance generator, add / duplicate / delete, the test wavelength,
// and the rows grouped by element, each with kind, surface menu, limits and
// unit, its problem in red, and a context menu for unit, distribution and
// symmetry. Every change is one undoable step saved in the document.

import { ChevronRight, Copy, Link2, Minus, Plus, Waves, Wand2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { t as translate, useT } from '@/i18n'
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
  const t = useT()
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
    editTolerances(t(ids.size === 1 ? 'tolerance.undo.duplicateTolerance' : 'tolerance.undo.duplicateTolerances'), (x) => {
      added = duplicateRows(x, ids)
    })
    select(added)
  }
  const remove = (ids: Set<string>) => {
    if (!ids.size) return
    editTolerances(t(ids.size === 1 ? 'tolerance.undo.deleteTolerance' : 'tolerance.undo.deleteTolerances'), (x) => deleteRows(x, ids))
    select([...selection].filter((id) => !ids.has(id)))
  }
  const add = (row: ToleranceRow, at: number) => {
    // After the last row on this surface or before it.
    const after = [...rows].reverse().find((r) => surfaceIndexOfId(system, r.surface) <= at)?.id ?? null
    editTolerances(t('tolerance.undo.addTolerance'), (x) => addRows(x, [row], after))
    select([row.id], row.id)
  }

  return (
    <Section
      title={t('tolerance.table.title')}
      accessory={<span className="num text-[11px] text-secondary">{rows.length}</span>}
    >
      <div
        className="outline-none"
        tabIndex={-1}
        onKeyDown={(e) => {
          const el = e.target as HTMLElement
          if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') return
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
            aria-label={t('tolerance.table.duplicate')}
            title={t('tolerance.table.duplicateHelp', { shortcut: isMac ? '⌘D' : 'Ctrl+D' })}
            disabled={!selection.size}
            onClick={() => duplicate(selection)}
          >
            <Copy />
          </Button>
          <Button variant="ghost" size="iconSm" aria-label={t('tolerance.table.delete')} title={t('tolerance.table.deleteHelp')} disabled={!selection.size} onClick={() => remove(selection)}>
            <Minus />
          </Button>
        </div>
        <TestWavelengthField value={settings.testWavelengthUm} onCommit={(v) => editTolerances(t('tolerance.undo.changeTestWavelength'), (x) => void (x.testWavelengthUm = v))} />
        {!rows.length ? (
          <p className="py-3 text-[12px] text-secondary">{t('tolerance.table.empty')}</p>
        ) : (
          <div role="table" aria-label={t('tolerance.table.title')} className="pb-1">
            <div className="grid grid-cols-[minmax(0,1fr)_3.25rem_6.75rem_1.75rem] gap-x-1.5 px-1 pt-1 text-[10px] text-tertiary" aria-hidden>
              <span>{t('tolerance.table.kind')}</span>
              <span>{t('tolerance.table.surface')}</span>
              <span className="text-right">{t('tolerance.table.limits')}</span>
              <span>{t('tolerance.table.unit')}</span>
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
                    title={closed ? t('tolerance.table.showGroup', { count: g.rows.length, group: g.key }) : t('tolerance.table.hideGroup', { group: g.key })}
                    onClick={() => {
                      const next = new Set(collapsed)
                      if (closed) next.delete(g.key)
                      else next.add(g.key)
                      useTolerance.setState({ collapsed: [...next] })
                    }}
                  >
                    <ChevronRight className={cn('size-3 shrink-0 transition-transform', !closed && 'rotate-90')} />
                    <span className="min-w-0 flex-1 truncate">{g.title}</span>
                    {closed && problems > 0 && <span className="text-danger">{t('tolerance.table.toFix', { count: problems })}</span>}
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
  const t = useT()
  const system = context.system
  const symmetric = row.minus === row.plus
  const set = (name: string, change: (r: ToleranceRow) => void, edit: Edit = editTolerances) =>
    edit(name, (ts) => {
      const r = ts.rows.find((x) => x.id === row.id)
      if (r) change(r)
    })
  const convert = (u: ToleranceUnit) => {
    const c = convertRow(row, u, context)
    if (c) set(t('tolerance.undo.changeToleranceUnit'), (r) => Object.assign(r, { ...c, id: r.id }))
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
                    ariaLabel={t('tolerance.table.limit', { label })}
                    value={row.minus}
                    min={0}
                    onCommit={(v) => v != null && set(t('tolerance.undo.changeTolerance'), (r) => Object.assign(r, { minus: v, plus: v }))}
                  />
                </>
              ) : (
                <>
                  <span className="text-secondary">−</span>
                  <NumberInput className="w-[2.75rem]" ariaLabel={t('tolerance.table.minusLimit', { label })} value={row.minus} min={0} onCommit={(v) => v != null && set(t('tolerance.undo.changeTolerance'), (r) => void (r.minus = v))} />
                  <span className="text-secondary">+</span>
                  <NumberInput className="w-[2.75rem]" ariaLabel={t('tolerance.table.plusLimit', { label })} value={row.plus} min={0} onCommit={(v) => v != null && set(t('tolerance.undo.changeTolerance'), (r) => void (r.plus = v))} />
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
                  <button type="button" className="cursor-default truncate rounded px-0.5 text-left text-[12px] text-secondary hover:bg-hover" title={t('tolerance.table.unitHelp', { unit: UNITS[row.unit].title })}>
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
            <div className="px-2 pt-1 pb-0.5 text-[11px] font-semibold text-secondary">{t('tolerance.table.unitMenu')}</div>
            {units.map((u) => (
              <ContextMenuItem key={u} className="pl-6" disabled={!convertRow(row, u, context)} onSelect={() => convert(u)}>
                {u === row.unit && <span className="absolute left-2">✓</span>}
                {UNITS[u].title}
              </ContextMenuItem>
            ))}
            <ContextMenuSeparator />
          </>
        )}
        <div className="px-2 pt-1 pb-0.5 text-[11px] font-semibold text-secondary">{t('tolerance.table.distribution')}</div>
        {DISTRIBUTIONS.map((d) => (
          <ContextMenuItem key={JSON.stringify(distributionJson(d))} className="pl-6" onSelect={() => set(t('tolerance.undo.changeDistribution'), (r) => void (r.distribution = d))}>
            {sameDistribution(d, row.distribution) && <span className="absolute left-2">✓</span>}
            {distributionTitle(d)}
          </ContextMenuItem>
        ))}
        <ContextMenuSeparator />
        <ContextMenuItem
          onSelect={() =>
            set(t(symmetric ? 'tolerance.undo.makeAsymmetric' : 'tolerance.undo.makeSymmetric'), (r) => {
              if (r.minus === r.plus) r.plus = r.minus * 2
              else r.plus = r.minus
            })
          }
        >
          {symmetric ? t('tolerance.table.makeAsymmetric') : t('tolerance.table.makeSymmetric')}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem shortcut="mod+d" onSelect={onDuplicate}>
          {t('tolerance.table.duplicate')}
        </ContextMenuItem>
        <ContextMenuItem className="text-danger" onSelect={onDelete}>
          {t('tolerance.table.delete')}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

type SetRow = (name: string, change: (r: ToleranceRow) => void) => void

const surfaceOptions = (s: System) => Array.from({ length: Math.max(imageIndex(s) - 1, 0) }, (_, k) => k + 1)

/** The surface (a thickness is the gap after it), or for an element its range. */
function SurfaceMenu({ row, system, set }: { row: ToleranceRow; system: System; set: SetRow }) {
  const t = useT()
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
          title={el ? t('tolerance.table.surfaceHelpElement') : t('tolerance.table.surfaceHelp')}
        >
          {text}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        {el ? (
          <>
            <DropdownMenuLabel>{t('tolerance.table.element')}</DropdownMenuLabel>
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
              <DropdownMenuSubTrigger>{t('tolerance.table.firstSurface')}</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {surfaceOptions(system).map((i) => (
                  <DropdownMenuItem key={i} disabled={i === moving[0]} onSelect={() => moveElement(row, system, i, Math.max(i, moving[moving.length - 1] ?? i), pivotChoice)}>
                    S{i}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>{t('tolerance.table.lastSurface')}</DropdownMenuSubTrigger>
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
            <DropdownMenuItem key={i} onSelect={() => set(t('tolerance.undo.changeToleranceSurface'), (r) => void (r.surface = system.surfaces[i].id))}>
              {i === system.stop ? t('tolerance.table.stopSuffix', { surface: `S${i}` }) : `S${i}`}
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

function moveElement(row: ToleranceRow, s: System, f: number, l: number, c: PivotChoice | null, name = translate('tolerance.undo.changeElement')) {
  if (!s.surfaces[f] || !s.surfaces[l]) return
  const choice = c ?? { at: 'first' }
  const pivot = choice.at === 'first' ? null : choice.at === 'last' ? s.surfaces[l].id : (s.surfaces[choice.index]?.id ?? null)
  editTolerances(name, (ts) => setElement(ts, row.id, s.surfaces[f].id, s.surfaces[l].id, pivot))
}

/** "S3 S4 S5 move together · pivot S3, first vertex", the pivot a menu. */
function ElementLine({ row, system }: { row: ToleranceRow; system: System }) {
  const t = useT()
  const moving = movingSurfaces(row, system)
  const p = pivotIndex(row, system)
  const f = moving[0], l = moving[moving.length - 1]
  const pivotText = p == null ? '?' : p === f ? t('tolerance.table.pivotFirst', { surface: `S${p}` }) : p === l ? t('tolerance.table.pivotLast', { surface: `S${p}` }) : `S${p}`
  const setPivot = (c: PivotChoice) => f != null && moveElement(row, system, f, l, c, t('tolerance.undo.changeElementPivot'))
  const surfaces = moving.map((i) => `S${i}`).join(' ')
  return (
    <div className="flex items-center gap-1 pl-1 text-[11px] text-secondary" aria-label={t('tolerance.table.movesAria', { surfaces: moving.map((i) => t('tolerance.table.surfaceLower', { n: i })).join(', '), pivot: pivotText })}>
      <Link2 className="size-3 shrink-0 text-tertiary" />
      <span className="truncate">{moving.length > 1 ? t('tolerance.table.movesTogether', { surfaces }) : t('tolerance.table.moves', { surfaces })}</span>
      <span className="text-tertiary">·</span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            className="cursor-default whitespace-nowrap rounded px-0.5 text-accent hover:bg-hover"
            title={t('tolerance.table.pivotHelp', { surface: `S${f ?? 0}` })}
          >
            {t('tolerance.table.pivotLabel', { pivot: pivotText })}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {f != null && <DropdownMenuItem onSelect={() => setPivot({ at: 'first' })}>{t('tolerance.table.firstVertex', { surface: `S${f}` })}</DropdownMenuItem>}
          {l != null && l !== f && <DropdownMenuItem onSelect={() => setPivot({ at: 'last' })}>{t('tolerance.table.lastVertex', { surface: `S${l}` })}</DropdownMenuItem>}
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>{t('tolerance.table.surfaceMenu')}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {surfaceOptions(system).map((i) => (
                <DropdownMenuItem key={i} onSelect={() => setPivot({ at: 'surface', index: i })}>
                  {i === system.stop ? t('tolerance.table.stopSuffix', { surface: `S${i}` }) : `S${i}`}
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
  let text = i === s.stop ? translate('tolerance.table.stopSuffix', { surface: `S${i}` }) : `S${i}`
  if (sf.label) text += ' ' + sf.label
  if (isSolid(s, i) && hasGlass(sf.material)) text += ' · ' + materialName(sf.material)
  return text
}

function AddMenu({ system, settings, onAdd }: { system: System; settings: ToleranceSettings; onAdd: (row: ToleranceRow, at: number) => void }) {
  const t = useT()
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
        <Button variant="ghost" size="sm" title={t('tolerance.table.addHelp')}>
          <Plus className="size-3.5" />
          {t('tolerance.table.add')}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>{t('tolerance.table.surface')}</DropdownMenuLabel>
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
        <DropdownMenuLabel>{t('tolerance.table.element')}</DropdownMenuLabel>
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
  const t = useT()
  return (
    <div className="flex items-center gap-1.5 py-1 text-[12px]" title={t('tolerance.table.testWavelengthHelp')}>
      <Waves className="size-3.5 text-secondary" />
      <span>{t('tolerance.table.testFringesAt')}</span>
      <NumberInput
        className="w-[4.5rem]"
        ariaLabel={t('tolerance.table.testWavelength')}
        value={Number((value * 1000).toPrecision(8))}
        placeholder="632.8"
        onCommit={(v) => v != null && v >= 200 && v <= 20000 && onCommit(v / 1000)}
      />
      <span className="text-secondary">nm</span>
      {Math.abs(value - DEFAULT_TEST_WAVELENGTH_UM) > 1e-9 && (
        <button type="button" className="cursor-default text-accent hover:underline" title={t('tolerance.table.backTo')} onClick={() => onCommit(DEFAULT_TEST_WAVELENGTH_UM)}>
          {t('tolerance.table.heNe')}
        </button>
      )}
    </div>
  )
}

// MARK: Default tolerances

/** The default-tolerance rows: keys into `tolerance.generator.*` (the title; `<key>Help` is its tooltip) and the unit. */
const GENERATOR_ROWS: { group?: 'groupSurfaces' | 'groupElements' | 'groupGlasses'; title: string; help: string; key: GeneratorValue; unit: string | 'unitFringes' | 'unitAbbe' }[] = [
  { group: 'groupSurfaces', title: 'radius', help: 'radiusHelp', key: 'radiusFringes', unit: 'unitFringes' },
  { title: 'irregularity', help: 'irregularityHelp', key: 'irregularityFringes', unit: 'unitFringes' },
  { title: 'thickness', help: 'thicknessHelp', key: 'thicknessMM', unit: 'mm' },
  { title: 'decenter', help: 'decenterHelp', key: 'decentreMM', unit: 'mm' },
  { title: 'tiltWedge', help: 'tiltWedgeHelp', key: 'tiltArcmin', unit: '′' },
  { group: 'groupElements', title: 'decenter', help: 'elementDecenterHelp', key: 'elementDecentreMM', unit: 'mm' },
  { title: 'tilt', help: 'elementTiltHelp', key: 'elementTiltArcmin', unit: '′' },
  { group: 'groupGlasses', title: 'index', help: 'indexHelp', key: 'index', unit: 'Δn_d' },
  { title: 'abbeNumber', help: 'abbeHelp', key: 'abbePercent', unit: 'unitAbbe' },
]

function GeneratorButton({ settings, system }: { settings: ToleranceSettings; system: System }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const g = settings.generator
  const nm = short(settings.testWavelengthUm * 1000)
  const setG = (name: string, change: (g: Generator) => void) => editTolerances(name, (x) => change(x.generator))
  const gt = (key: string, params?: Record<string, string | number>) => t(`tolerance.generator.${key}` as Parameters<typeof t>[0], params)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" title={t('tolerance.generator.buttonHelp')}>
          <Wand2 className="size-3.5" />
          {t('tolerance.generator.button')}
        </Button>
      </PopoverTrigger>
      <PopoverContent side="left" align="start" className="w-[330px]" aria-label={t('tolerance.generator.title')}>
        <h3 className="mb-2 text-[13px] font-semibold">{t('tolerance.generator.title')}</h3>
        <Segmented<Preset | 'custom'>
          label={t('tolerance.generator.class')}
          className="mb-2 w-full [&>button]:flex-1"
          value={g.preset ?? 'custom'}
          options={[...PRESETS.map((p) => ({ value: p.value as Preset | 'custom', title: p.title })), ...(g.preset == null ? [{ value: 'custom' as const, title: t('tolerance.presets.custom') }] : [])]}
          onChange={(p) => p !== 'custom' && setG(t('tolerance.undo.chooseToleranceClass'), (x) => Object.assign(x, presetGenerator(p, x.distribution)))}
        />
        <div className="grid grid-cols-[minmax(0,1fr)_5rem_3.5rem] items-center gap-x-2 gap-y-1 text-[12px]">
          {GENERATOR_ROWS.map((r) => (
            <div key={r.group ?? '' + r.key} className="contents">
              {r.group && <div className="col-span-3 pt-1.5 text-[11px] font-semibold text-secondary">{gt(r.group)}</div>}
              <span title={gt(r.help, { nm })}>{gt(r.title)}</span>
              <NumberInput
                ariaLabel={`${r.group ? gt(r.group) : ''} ${gt(r.title)}`.trim()}
                value={g[r.key]}
                min={0}
                placeholder="0"
                onCommit={(v) =>
                  v != null &&
                  v >= 0 &&
                  setG(t('tolerance.undo.changeDefaultTolerance'), (x) => {
                    x[r.key] = v
                    x.preset = null
                  })
                }
              />
              <span className="truncate text-secondary">{r.unit === 'unitFringes' || r.unit === 'unitAbbe' ? gt(r.unit) : r.unit}</span>
            </div>
          ))}
        </div>
        <label className="mt-2 flex items-center gap-2 text-[12px]" title={t('tolerance.generator.distributionHelp')}>
          <span>{t('tolerance.generator.distribution')}</span>
          <Select
            className="flex-1"
            value={JSON.stringify(distributionJson(g.distribution))}
            onChange={(e) => setG(t('tolerance.undo.changeDistribution'), (x) => void (x.distribution = JSON.parse(e.target.value) as Distribution))}
          >
            {DISTRIBUTIONS.map((d) => (
              <option key={JSON.stringify(distributionJson(d))} value={JSON.stringify(distributionJson(d))}>
                {distributionTitle(d)}
              </option>
            ))}
          </Select>
        </label>
        <Hint className="mt-1.5">{t('tolerance.generator.zeroHint')}</Hint>
        <div className="mt-3 flex justify-end gap-2">
          <Button size="sm" onClick={() => setOpen(false)}>
            {t('common.cancel')}
          </Button>
          <Button
            size="sm"
            variant="prominent"
            title={t('tolerance.generator.replaceHelp')}
            onClick={() => {
              editTolerances(t('tolerance.undo.replaceTolerances'), (x) => void (x.rows = generateRows(x.generator, system)))
              useTolerance.setState({ selection: [] })
              setOpen(false)
            }}
          >
            {t('tolerance.generator.replace')}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
