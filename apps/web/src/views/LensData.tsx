import {
  DataEditor,
  GridCellKind,
  getMiddleCenterBias,
  type CustomCell,
  type CustomRenderer,
  type DataEditorRef,
  type EditableGridCell,
  type GridCell,
  type GridColumn,
  type GridSelection,
  type Item,
  type ProvideEditorComponent,
  type Theme,
  CompactSelection,
} from '@glideapps/glide-data-grid'
import '@glideapps/glide-data-grid/dist/index.css'
import { Ellipsis, Minus, Plus, Table2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { t as tr, useT } from '@/i18n'
import { compact, exact, nf, parseNumber } from '@/model/format'
import { materialCompletions, offsetCompactText, offsetOf } from '@/model/materials'
import { canonicalParam, isVariable, paramKey, paramResolves, variableBoundsKind, variableRefusal, variablesOf, type ParamPath } from '@/model/params'
import { solveKindTitle, solveLetter, solvesOf } from '@/model/solves'
import {
  canDelete,
  conicOf,
  curvatureFromRadius,
  curvatureOf,
  imageIndex,
  materialText,
  parseMaterial,
  radiusFromCurvature,
  setConic,
  setCurvature,
  setMaterial,
  type Surface,
  type System,
} from '@/model/system'
import { CYLINDER_CHOICES, GROUPS, groupTitle, SURFACE_TYPES, cylinderAxis, tableTypeTitle, typeChoice } from '@/model/surfaceTypes'
import { command, deleteSurfaceAt, insertSurfaceAt, makeStop, setParamVariable, useValueFocus } from '@/state/commands'
import { LENS_DATA } from '@/state/arrangement'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { allIssues, LAYOUT, useWorkspace, type Issue } from '@/state/workspace'
import { cssVar, FONT_FAMILY, useDark } from '@/lib/theme'
import { cn, shortcutLabel } from '@/lib/utils'
import { PanelHeader } from './PanelHeader'
import { usePortalContainer } from './panels/context'
import { PanelMenuItems } from './panels/PanelArea'
import { useGlidePortal } from './lens/portal'
import { useColumnWidths } from './lens/columnWidths'
import { applyTypeChoice } from './inspector/SurfacePage'

type ColumnId = 'surf' | 'type' | 'comment' | 'radius' | 'thickness' | 'material' | 'semi' | 'conic'

/** Column widths are part of the e2e contract (Playwright clicks by offset). */
const COLUMNS: { id: ColumnId; width: number }[] = [
  { id: 'surf', width: 58 },
  { id: 'type', width: 80 },
  { id: 'comment', width: 110 },
  { id: 'radius', width: 110 },
  { id: 'thickness', width: 104 },
  { id: 'material', width: 100 },
  { id: 'semi', width: 112 },
  { id: 'conic', width: 84 },
]
/** A column's header, in the current language. */
const columnTitle = (id: ColumnId) => tr(`lens.grid.columns.${id}`)
/** The material column widens while a glass carries a melt offset. */
const MATERIAL_WIDE = 168
const VALUE_COLUMNS = new Set<ColumnId>(['comment', 'radius', 'thickness', 'material', 'semi', 'conic'])

// MARK: Cells

/** A small inline marker after a value: variable (bar when bounded, dot when
 *  pinned; orange when a solve sets it too) and the solve's letter. */
interface Marker {
  variable: boolean
  bounds: 'free' | 'bounded' | 'pinned'
  solve: string | null
  solveTitle: string | null
}

interface Part {
  /** A short axis hint before the value (`y`, `x`). */
  hint?: string
  text: string
  /** A cylinder's flat section, an automatic semi-diameter. */
  dim?: boolean
  marker?: Marker | null
}

interface LensCellData {
  kind: 'lens-cell'
  column: ColumnId
  row: number
  parts: Part[]
  /** A subtle note after the value (a melt offset `+0.0012/−0.8`). */
  note?: string
  align: 'left' | 'right'
  /** Text colour override (issues, the stop). */
  color?: string
  bold?: boolean
  /** Exact text an edit starts from, one per section; edited text replaces it. */
  edit: string[]
  original: string[]
  /** Labels of the sections' editors (`y`, `x`). */
  labels?: string[]
  /** Offer glass completions while typing. */
  completions?: boolean
  /** A menu cell (the type), drawn with a chevron. */
  menu?: boolean
}

type LensCell = CustomCell<LensCellData>

interface Palette {
  label: string
  secondary: string
  tertiary: string
  accent: string
  warning: string
  solve: string
  onAccent: string
}

let palette: Palette = { label: '#000', secondary: '#666', tertiary: '#999', accent: '#0a74ff', warning: '#c26a00', solve: '#7a4fe0', onAccent: '#fff' }

const SMALL = '11px'
const TINY_BOLD = '700 9px'

function markerWidth(m: Marker | null | undefined): number {
  if (!m) return 0
  return (m.variable ? 9 : 0) + (m.solve ? 9 : 0) + (m.variable && m.solve ? 1 : 0)
}

const lensRenderer: CustomRenderer<LensCell> = {
  kind: GridCellKind.Custom,
  isMatch: (c): c is LensCell => (c.data as LensCellData | undefined)?.kind === 'lens-cell',
  draw: (args, cell) => {
    const { ctx, rect, theme } = args
    const d = cell.data
    const family = theme.fontFamily
    const base = `${d.bold ? '600 ' : ''}${theme.baseFontStyle} ${family}`
    const small = `${SMALL} ${family}`
    const tiny = `${TINY_BOLD} ${family}`
    type Piece = { w: number; draw: (x: number, y: number) => void }
    const pieces: Piece[] = []
    const text = (s: string, font: string, color: string) => {
      ctx.font = font
      const w = ctx.measureText(s).width
      pieces.push({
        w,
        draw: (x, y) => {
          ctx.font = font
          ctx.fillStyle = color
          ctx.fillText(s, x, y + getMiddleCenterBias(ctx, font))
        },
      })
    }
    const gap = (w: number) => pieces.push({ w, draw: () => {} })
    const marker = (m: Marker) => {
      const w = markerWidth(m)
      pieces.push({
        w,
        draw: (x, y) => {
          ctx.font = tiny
          ctx.textBaseline = 'middle'
          let cx = x
          if (m.variable) {
            const color = m.solve ? palette.warning : palette.accent
            ctx.fillStyle = color
            ctx.fillText('V', cx + 1, y - 1 + getMiddleCenterBias(ctx, tiny))
            if (m.bounds === 'bounded') ctx.fillRect(cx + 1.5, y + 5, 6, 1.2)
            if (m.bounds === 'pinned') {
              ctx.beginPath()
              ctx.arc(cx + 4.5, y + 6, 1.4, 0, 2 * Math.PI)
              ctx.fill()
            }
            cx += 10
          }
          if (m.solve) {
            ctx.fillStyle = palette.solve
            ctx.fillText(m.solve, cx, y - 1 + getMiddleCenterBias(ctx, tiny))
          }
        },
      })
    }
    d.parts.forEach((p, k) => {
      if (k > 0) gap(10)
      if (p.hint) {
        text(p.hint, small, palette.tertiary)
        gap(3)
      }
      text(p.text, base, p.dim ? palette.tertiary : (d.color ?? palette.label))
      if (p.marker && markerWidth(p.marker)) {
        gap(3)
        marker(p.marker)
      }
    })
    if (d.note) {
      gap(5)
      text(d.note, small, palette.secondary)
    }
    if (d.menu) {
      gap(4)
      text('⌄', small, palette.tertiary)
    }
    const pad = theme.cellHorizontalPadding
    const total = pieces.reduce((a, p) => a + p.w, 0)
    let x = d.align === 'right' ? Math.max(rect.x + pad, rect.x + rect.width - pad - total) : rect.x + pad
    const y = rect.y + rect.height / 2
    ctx.save()
    ctx.beginPath()
    ctx.rect(rect.x, rect.y, rect.width, rect.height)
    ctx.clip()
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'left'
    for (const p of pieces) {
      p.draw(x, y)
      x += p.w
    }
    ctx.restore()
    return true
  },
  provideEditor: () => ({ editor: LensCellEditor, disablePadding: true }),
  onPaste: (val, data) => (data.column === 'type' ? undefined : { ...data, edit: [val, ...data.edit.slice(1)] }),
  onDelete: (cell) =>
    cell.data.column === 'comment' || cell.data.column === 'material' || cell.data.column === 'semi'
      ? { ...cell, data: { ...cell.data, edit: [''] } }
      : undefined,
}

/** The editor of a value cell: one field per section (y then x of an
 *  anamorphic radius; Tab moves y → x), glass completions in the material
 *  column. ↑/↓ commit and move between rows; Return commits and moves down,
 *  Tab across, Esc reverts. */
const LensCellEditor: ProvideEditorComponent<LensCell> = ({ value, onChange, onFinishedEditing, initialValue, target }) => {
  const d = value.data
  const refs = useRef<(HTMLInputElement | null)[]>([])
  const materials = useWorkspace((s) => s.materials)
  // The completions open in the window the table is in.
  const portalTo = usePortalContainer() ?? document.body
  const [hi, setHi] = useState(0)
  const [focused, setFocused] = useState(0)
  const started = useRef(false)
  const setText = (k: number, t: string) => {
    const edit = d.edit.slice()
    edit[k] = t
    onChange({ ...value, data: { ...d, edit } })
  }
  useEffect(() => {
    if (started.current) return
    started.current = true
    if (initialValue != null && initialValue !== '') setText(0, initialValue)
    const el = refs.current[0]
    if (el) {
      el.focus()
      if (initialValue == null || initialValue === '') el.select()
      else requestAnimationFrame(() => el.setSelectionRange(el.value.length, el.value.length))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const list = d.completions ? materialCompletions(d.edit[0] ?? '', materials) : []
  const finish = (edit: string[], movement: readonly [-1 | 0 | 1, -1 | 0 | 1]) => onFinishedEditing({ ...value, data: { ...d, edit } }, movement)
  return (
    <div className="flex items-stretch gap-1 bg-content" style={{ width: Math.max(target.width, d.edit.length > 1 ? 200 : 0), height: target.height }}>
      {d.edit.map((t, k) => (
        <label key={k} className="flex min-w-0 flex-1 items-center gap-1 px-1.5">
          {d.labels?.[k] && <span className="text-[11px] text-tertiary">{d.labels[k]}</span>}
          <input
            ref={(el) => void (refs.current[k] = el)}
            aria-label={`${columnTitle(d.column)}${d.labels?.[k] ? ' ' + d.labels[k] : ''}`}
            className={cn('h-full w-full min-w-0 bg-transparent text-[13px] text-label outline-none', d.align === 'right' && 'text-right')}
            value={t}
            spellCheck={false}
            autoComplete="off"
            onFocus={() => setFocused(k)}
            onChange={(e) => {
              setText(k, e.target.value)
              setHi(0)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Tab' && d.edit.length > 1 && ((k === 0 && !e.shiftKey) || (k === 1 && e.shiftKey))) {
                e.preventDefault()
                e.stopPropagation()
                refs.current[k === 0 ? 1 : 0]?.focus()
                refs.current[k === 0 ? 1 : 0]?.select()
                return
              }
              if (list.length && k === 0 && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
                e.preventDefault()
                e.stopPropagation()
                setHi((h) => (h + (e.key === 'ArrowDown' ? 1 : list.length - 1)) % list.length)
                return
              }
              if (list.length && k === 0 && e.key === 'Enter') {
                e.preventDefault()
                e.stopPropagation()
                finish([list[hi]], [0, 1])
                return
              }
              if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault()
                e.stopPropagation()
                finish(d.edit, [0, e.key === 'ArrowDown' ? 1 : -1])
              }
            }}
          />
        </label>
      ))}
      {list.length > 0 && focused === 0 && (
        createPortal(
        <ul
          role="listbox"
          aria-label={tr('lens.grid.glasses')}
          data-lens-completions
          className="glass-popover fixed z-[10000] max-h-56 w-[200px] overflow-y-auto rounded-md p-1 text-[12px] text-label"
          style={{ left: target.x, top: target.y + target.height + 4 }}
        >
          {list.map((g, k) => (
            <li
              key={g}
              role="option"
              aria-selected={k === hi}
              className={cn('cursor-default rounded px-1.5 py-[2px]', k === hi && 'bg-accent text-on-accent')}
              onMouseDown={(e) => {
                e.preventDefault()
                finish([g], [0, 0])
              }}
            >
              {g}
            </li>
          ))}
        </ul>,
        portalTo,
        )
      )}
    </div>
  )
}

const RENDERERS = [lensRenderer] as unknown as CustomRenderer<CustomCell>[]

// MARK: Theme

function useGridTheme(): Partial<Theme> {
  const dark = useDark()
  return useMemo(() => {
    const v = (n: string) => cssVar(n) || undefined
    palette = {
      label: v('--label') ?? palette.label,
      secondary: v('--secondary') ?? palette.secondary,
      tertiary: v('--tertiary') ?? palette.tertiary,
      accent: v('--accent') ?? palette.accent,
      warning: v('--warning') ?? palette.warning,
      solve: dark ? '#b694ff' : '#7a4fe0',
      onAccent: v('--on-accent') ?? '#fff',
    }
    return {
      accentColor: v('--accent'),
      accentLight: v('--accent-soft'),
      textDark: v('--label'),
      textMedium: v('--secondary'),
      textLight: v('--tertiary'),
      textHeader: v('--secondary'),
      textHeaderSelected: v('--on-accent'),
      bgCell: v('--content'),
      bgCellMedium: dark ? '#262628' : '#f7f7f9',
      bgHeader: v('--content'),
      bgHeaderHasFocus: dark ? '#2c2c2f' : '#efeff2',
      bgHeaderHovered: dark ? '#2c2c2f' : '#efeff2',
      borderColor: dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.06)',
      horizontalBorderColor: dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.06)',
      headerBottomBorderColor: dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.12)',
      fontFamily: FONT_FAMILY,
      baseFontStyle: '13px',
      headerFontStyle: '600 11px',
      editorFontSize: '13px',
      cellHorizontalPadding: 8,
      bgSearchResult: v('--accent-soft'),
    } satisfies Partial<Theme>
  }, [dark])
}

// MARK: Rows

/** The parameter behind a column for this surface (radius and conic of a
 *  toroid or biconic are its y section's; `x` addresses the x section). */
function columnParam(s: Surface, column: ColumnId, section: 'y' | 'x' = 'y'): ParamPath | null {
  const id = s.id
  const t = s.shape.type
  const anamorphic = t === 'toroid' || t === 'biconic'
  switch (column) {
    case 'radius':
      if (section === 'x') return anamorphic ? { param: 'shape_param', surface: id, parameter: 'curvature_x' } : null
      if (t === 'paraxial') return { param: 'paraxial_focal_length', surface: id }
      return t === 'coordinate_break' ? null : { param: 'curvature', surface: id }
    case 'thickness':
      return { param: 'thickness', surface: id }
    case 'conic':
      if (anamorphic) return section === 'x' ? (t === 'biconic' ? { param: 'shape_param', surface: id, parameter: 'conic_x' } : null) : { param: 'shape_param', surface: id, parameter: 'conic_y' }
      return section === 'x' ? null : { param: 'conic', surface: id }
    case 'semi':
      return { param: 'semi_diameter', surface: id }
    default:
      return null
  }
}

interface Context {
  system: System
  variables: Map<string, 'free' | 'bounded' | 'pinned'>
  solves: Map<string, JsonObject>
  issues: Map<number, Issue[]>
  autoSemi: Map<number, number>
}

function markerFor(ctx: Context, p: ParamPath | null): Marker | null {
  if (!p) return null
  const key = paramKey(canonicalParam(ctx.system, p))
  const v = ctx.variables.get(key)
  const solve = ctx.solves.get(key)
  if (!v && !solve) return null
  return { variable: !!v, bounds: v ?? 'free', solve: solve ? solveLetter(solve) : null, solveTitle: solve ? solveKindTitle(solve) : null }
}

/** One table cell of the design (`LensRow` of the macOS app). */
function cellFor(ctx: Context, col: number, row: number): GridCell {
  const { system } = ctx
  const s = system.surfaces[row]
  const id = COLUMNS[col].id
  const img = imageIndex(system)
  const readonly = (text = '', opts?: Partial<LensCellData>): GridCell => ({
    kind: GridCellKind.Custom,
    allowOverlay: false,
    readonly: true,
    copyData: text,
    data: { kind: 'lens-cell', column: id, row, parts: [{ text }], align: 'left', edit: [], original: [], ...opts },
  })
  const value = (parts: Part[], edit: string[], opts?: Partial<LensCellData>): GridCell => ({
    kind: GridCellKind.Custom,
    allowOverlay: true,
    copyData: parts.map((p) => p.text).join(' '),
    data: { kind: 'lens-cell', column: id, row, parts, align: 'right', edit, original: edit, ...opts },
  })
  switch (id) {
    case 'surf': {
      const label = row === 0 ? tr('lens.grid.rowObject') : row === img ? tr('lens.grid.rowImage') : row === system.stop ? tr('lens.grid.rowStop', { n: row }) : String(row)
      const list = ctx.issues.get(row)
      const worst = list?.some((i) => i.severity === 'error') ? 'error' : list?.length ? 'warning' : null
      return readonly(label, {
        parts: [{ text: worst ? `${label} ${worst === 'error' ? '⛔︎' : '⚠︎'}` : label }],
        color: worst ? cssVar(worst === 'error' ? '--danger' : '--warning') : row === system.stop ? palette.accent : row === 0 || row === img ? palette.secondary : undefined,
        bold: row === 0 || row === img || row === system.stop,
      })
    }
    case 'type': {
      const title = tableTypeTitle(s.shape)
      if (row === 0 || row === img) return readonly(title, { color: palette.secondary })
      return { ...(readonly(title, { menu: true }) as LensCell), allowOverlay: false, readonly: false }
    }
    case 'comment':
      return value([{ text: s.label ?? '' }], [s.label ?? ''], { align: 'left' })
    case 'radius': {
      const t = s.shape.type
      if (t === 'coordinate_break') return readonly('')
      if (t === 'paraxial') {
        const f = nf(s.shape.focal_length)
        return readonly(`f ${compact(f)}`, { align: 'right', parts: [{ text: `f ${compact(f)}`, marker: markerFor(ctx, columnParam(s, 'radius')) }] })
      }
      if (row === 0 && !Number.isFinite(nf(s.thickness))) return readonly('')
      const c = curvatureOf(s.shape)
      if (c == null) return readonly('')
      const ry = radiusFromCurvature(c)
      if (t === 'toroid' || t === 'biconic') {
        const rx = radiusFromCurvature(Number(s.shape.curvature_x ?? 0))
        const axis = cylinderAxis(s.shape)
        return value(
          [
            { hint: 'y', text: compact(ry), dim: axis === 'x', marker: markerFor(ctx, columnParam(s, 'radius')) },
            { hint: 'x', text: compact(rx), dim: axis === 'y', marker: markerFor(ctx, columnParam(s, 'radius', 'x')) },
          ],
          [exact(ry), exact(rx)],
          { labels: ['y', 'x'] },
        )
      }
      return value([{ text: compact(ry), marker: markerFor(ctx, columnParam(s, 'radius')) }], [exact(ry)])
    }
    case 'thickness': {
      if (row === img) return readonly('')
      const v = nf(s.thickness)
      return value([{ text: compact(v), marker: markerFor(ctx, columnParam(s, 'thickness')) }], [exact(v)])
    }
    case 'material': {
      if (row === img || s.shape.type === 'coordinate_break') return readonly(materialText(s))
      const m = materialText(s)
      const offset = s.interaction === 'refract' ? offsetCompactText(offsetOf(s.material)) : ''
      // Offset variables and solves mark the glass.
      const dn = markerFor(ctx, { param: 'material_dn_d', surface: s.id })
      const dv = markerFor(ctx, { param: 'material_dv_d', surface: s.id })
      const mk = dn || dv ? { variable: !!(dn?.variable || dv?.variable), bounds: dn?.bounds ?? dv?.bounds ?? 'free', solve: dn?.solve ?? dv?.solve ?? null, solveTitle: null } : null
      return value([{ text: m, marker: mk }], [m], { align: 'left', note: offset || undefined, completions: true })
    }
    case 'semi': {
      const marker = markerFor(ctx, columnParam(s, 'semi'))
      if (s.semi_diameter.mode === 'fixed') {
        const v = Number(s.semi_diameter.value)
        return value([{ text: `${compact(v)} U`, marker }], [exact(v)])
      }
      const a = ctx.autoSemi.get(row)
      return value([{ text: a != null && Number.isFinite(a) ? compact(a) : tr('lens.grid.auto'), dim: true, marker }], ['auto'])
    }
    case 'conic': {
      const t = s.shape.type
      const k = conicOf(s.shape)
      if (k == null) return readonly('')
      if (t === 'biconic') {
        const kx = Number(s.shape.conic_x ?? 0)
        const axis = cylinderAxis(s.shape)
        // A cylinder's flat section has no meaningful conic: only the powered one shows.
        if (axis === 'x') return value([{ hint: 'x', text: compact(kx), marker: markerFor(ctx, columnParam(s, 'conic', 'x')) }], [exact(kx)], { labels: ['x'] })
        if (axis === 'y') return value([{ hint: 'y', text: compact(k), marker: markerFor(ctx, columnParam(s, 'conic')) }], [exact(k)], { labels: ['y'] })
        return value(
          [
            { hint: 'y', text: compact(k), marker: markerFor(ctx, columnParam(s, 'conic')) },
            { hint: 'x', text: compact(kx), marker: markerFor(ctx, columnParam(s, 'conic', 'x')) },
          ],
          [exact(k), exact(kx)],
          { labels: ['y', 'x'] },
        )
      }
      if (t === 'toroid') return value([{ hint: 'y', text: compact(k), marker: markerFor(ctx, columnParam(s, 'conic')) }], [exact(k)], { labels: ['y'] })
      return value([{ text: compact(k), marker: markerFor(ctx, columnParam(s, 'conic')) }], [exact(k)])
    }
  }
}

// MARK: The table

/** The lens data table (`LensTableView`): what every surface has; everything
 *  type-specific lives in the inspector. Text cells edit in place (Return or
 *  a double click; Tab / ⇧Tab across, ↑/↓ and Return between rows,
 *  committing; Esc reverts). Small markers show variables and solves;
 *  validation issues badge their surface. */
export function LensData() {
  const t = useT()
  const system = useDocument((s) => s.project.system)
  const variables = useDocument((s) => s.project.variables)
  const edit = useDocument((s) => s.edit)
  const selection = useView((s) => s.selection)
  const select = useView((s) => s.select)
  const toast = useView((s) => s.toast)
  const materials = useWorkspace((s) => s.materials)
  const layout = useWorkspace((s) => s.runs[LAYOUT]?.result)
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const validationIssues = useWorkspace((s) => s.validationIssues)
  const setFocusParam = useValueFocus((s) => s.setParam)
  const theme = useGridTheme()
  const ref = useRef<DataEditorRef>(null)
  const box = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 600, h: 200 })
  const [menu, setMenu] = useState<{ row: number; x: number; y: number; context?: boolean } | null>(null)
  const [hoverTitle, setHoverTitle] = useState<string | undefined>()
  const [gridSel, setGridSel] = useState<GridSelection>({
    columns: CompactSelection.empty(),
    rows: CompactSelection.empty(),
    current: undefined,
  })

  const ctx: Context | null = useMemo(() => {
    if (!system?.surfaces) return null
    const vars = new Map<string, 'free' | 'bounded' | 'pinned'>()
    for (const v of variablesOf({ variables })) vars.set(paramKey(canonicalParam(system, v.param)), variableBoundsKind(v))
    const solves = new Map<string, JsonObject>()
    for (const s of solvesOf(system)) solves.set(paramKey(canonicalParam(system, s.target)), s.kind)
    const issues = new Map<number, Issue[]>()
    for (const i of allIssues({ modelIssues, validationIssues })) {
      if (i.surface == null || i.severity === 'info') continue
      issues.set(i.surface, [...(issues.get(i.surface) ?? []), i])
    }
    const autoSemi = new Map<number, number>()
    for (const s of (layout?.json.surfaces as JsonObject[] | undefined) ?? []) autoSemi.set(Number(s.index), nf(s.semi_diameter))
    return { system, variables: vars, solves, issues, autoSemi }
  }, [system, variables, modelIssues, validationIssues, layout])

  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // The view's selection follows the table's current cell and vice versa.
  useEffect(() => {
    setGridSel((g) => {
      if (g.current?.cell[1] === selection) return g
      const col = g.current?.cell[0] ?? 3
      return { ...g, current: { cell: [col, selection], range: { x: col, y: selection, width: 1, height: 1 }, rangeStack: [] } }
    })
  }, [selection])

  // Toggle Variable (⌃⌘V) acts on the focused value.
  const focusCol = gridSel.current?.cell[0]
  const focusRow = gridSel.current?.cell[1]
  useEffect(() => {
    if (focusCol == null || focusRow == null || !system?.surfaces?.[focusRow]) return
    const p = columnParam(system.surfaces[focusRow], COLUMNS[focusCol].id)
    setFocusParam(p && paramResolves(system, p) ? p : null)
  }, [focusCol, focusRow, system, setFocusParam])

  const rows = system?.surfaces?.length ?? 0
  const wideMaterial = !!system?.surfaces?.some((s) => offsetCompactText(offsetOf(s.material)) !== '')
  const [widths, setWidth, resetWidths] = useColumnWidths()
  const columns: GridColumn[] = useMemo(
    () => COLUMNS.map((c) => ({ id: c.id, title: t(`lens.grid.columns.${c.id}`), width: widths[c.id] ?? (c.id === 'material' && wideMaterial ? MATERIAL_WIDE : c.width) })),
    [wideMaterial, widths, t],
  )
  // Glide opens its cell editor in `#portal` of the main document; in a
  // separate window it must open in that window's document.
  const portalContainer = usePortalContainer()
  useGlidePortal(portalContainer ?? null)

  const getCellContent = useCallback(
    ([col, row]: Item): GridCell => (ctx ? cellFor(ctx, col, row) : { kind: GridCellKind.Loading, allowOverlay: false }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ctx, t],
  )

  const onCellEdited = useCallback(
    ([col, row]: Item, cell: EditableGridCell) => {
      if (cell.kind !== GridCellKind.Custom) return
      const d = (cell as unknown as LensCell).data
      if (d?.kind !== 'lens-cell') return
      const id = COLUMNS[col].id
      const changed = (k: number) => (d.edit[k] ?? '') !== (d.original[k] ?? '')
      const number = (text: string): number | null => {
        const v = parseNumber(text)
        if (v == null || Number.isNaN(v)) {
          toast(t('lens.grid.notNumber', { text }), 'warning')
          return null
        }
        return v
      }
      const finite = (text: string): number | null => {
        const v = number(text)
        if (v == null) return null
        if (!Number.isFinite(v)) {
          toast(t('lens.grid.conicFinite'), 'warning')
          return null
        }
        return v
      }
      const surface = useDocument.getState().project.system.surfaces[row]
      if (!surface) return
      const anamorphic = surface.shape.type === 'toroid' || surface.shape.type === 'biconic'
      switch (id) {
        case 'comment':
          if (changed(0)) edit(t('lens.grid.undo.label'), (s) => void (s.surfaces[row].label = d.edit[0]))
          break
        case 'radius': {
          if (changed(0)) {
            const r = number(d.edit[0])
            if (r != null) edit(t('lens.grid.undo.radius'), (s) => void (s.surfaces[row].shape = setCurvature(s.surfaces[row].shape, curvatureFromRadius(r))))
          }
          if (anamorphic && changed(1)) {
            const r = number(d.edit[1])
            if (r != null) edit(t('lens.grid.undo.radiusX'), (s) => void (s.surfaces[row].shape = { ...s.surfaces[row].shape, curvature_x: curvatureFromRadius(r) }))
          }
          break
        }
        case 'thickness': {
          if (!changed(0)) break
          const v = number(d.edit[0])
          if (v == null) return
          if (!Number.isFinite(v) && row !== 0) return toast(t('lens.grid.onlyObjectInfinite'), 'warning')
          edit(t('lens.grid.undo.thickness'), (s) => void (s.surfaces[row].thickness = Number.isFinite(v) ? v : 'inf'))
          break
        }
        case 'material': {
          if (!changed(0)) break
          const m = parseMaterial(d.edit[0], materials)
          if ('error' in m) return toast(m.error, 'warning')
          edit(t('lens.grid.undo.material'), (s) => setMaterial(s.surfaces[row], m))
          break
        }
        case 'semi': {
          if (!changed(0)) break
          const lower = d.edit[0].trim().toLowerCase()
          if (lower === '' || lower === 'auto' || lower === t('lens.grid.auto').toLowerCase()) {
            edit(t('lens.grid.undo.semi'), (s) => void (s.surfaces[row].semi_diameter = { mode: 'auto' }))
            break
          }
          const v = number(d.edit[0])
          if (v == null) return
          if (!(v > 0) || !Number.isFinite(v)) return toast(t('lens.grid.semiPositive'), 'warning')
          edit(t('lens.grid.undo.semi'), (s) => void (s.surfaces[row].semi_diameter = { mode: 'fixed', value: v }))
          break
        }
        case 'conic': {
          const type = surface.shape.type
          const axis = cylinderAxis(surface.shape)
          // Which section each edited field is: a cylinder shows only its powered conic.
          const sections: ('y' | 'x')[] = type === 'biconic' ? (axis === 'x' ? ['x'] : axis === 'y' ? ['y'] : ['y', 'x']) : ['y']
          sections.forEach((sec, k) => {
            if (!changed(k)) return
            const v = finite(d.edit[k])
            if (v == null) return
            if (!anamorphic) edit(t('lens.grid.undo.conic'), (s) => void (s.surfaces[row].shape = setConic(s.surfaces[row].shape, v)))
            else if (sec === 'y') edit(t('lens.grid.undo.conicY'), (s) => void (s.surfaces[row].shape = { ...s.surfaces[row].shape, conic_y: v }))
            else edit(t('lens.grid.undo.conicX'), (s) => void (s.surfaces[row].shape = { ...s.surfaces[row].shape, conic_x: v }))
          })
          break
        }
      }
    },
    [edit, materials, toast, t],
  )

  const latestSel = useRef(gridSel)
  latestSel.current = gridSel

  /** Editing continues in the cell a move lands on (Tab, Return, ↑/↓). */
  const onFinishedEditing = useCallback((cell: GridCell | undefined, movement: Item) => {
    if (!cell || (movement[0] === 0 && movement[1] === 0)) return
    requestAnimationFrame(() => {
      const target = ref.current
      const sel = latestSel.current.current?.cell
      if (!target || !sel || !VALUE_COLUMNS.has(COLUMNS[sel[0]]?.id)) return
      const canvas = box.current?.querySelector('canvas[data-testid="data-grid-canvas"]')
      canvas?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true }))
    })
  }, [])
  const onCellActivated = useCallback(([col, row]: Item) => {
    if (COLUMNS[col].id !== 'type') return
    const s = useDocument.getState().project.system
    if (row === 0 || row === imageIndex(s)) return
    const b = ref.current?.getBounds(col, row)
    if (b) setMenu({ row, x: b.x, y: b.y + b.height })
  }, [])

  if (!system?.surfaces || !ctx) return null
  const nVars = variablesOf({ variables }).length
  const caption = t('lens.grid.caption', { n: rows - 2, stop: system.stop }) + (nVars ? ` · ${t('lens.grid.variables', { count: nVars })}` : '')

  return (
    <section className="flex h-full min-h-0 flex-col bg-content" aria-label={t('lens.grid.title')} data-testid="lens-data">
      <PanelHeader icon={<Table2 />} title={t('lens.grid.title')} caption={caption}>
        <Tip label={`${t('lens.grid.insertSurface')} (${shortcutLabel(command('insertBefore').shortcut!)})`}>
          <Button variant="ghost" size="iconSm" aria-label={t('lens.grid.insertSurface')} onClick={() => insertSurfaceAt(Math.max(selection, 1))}>
            <Plus />
          </Button>
        </Tip>
        <Tip label={`${t('lens.grid.deleteSurface')} (${shortcutLabel(command('delete').shortcut!)})`}>
          <Button variant="ghost" size="iconSm" aria-label={t('lens.grid.deleteSurface')} disabled={!canDelete(system, selection)} onClick={() => deleteSurfaceAt(selection)}>
            <Minus />
          </Button>
        </Tip>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="iconSm" aria-label={t('common.more')}>
              <Ellipsis />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-[14rem]">
            <DropdownMenuLabel>{t('lens.grid.title')}</DropdownMenuLabel>
            <DropdownMenuItem onSelect={() => insertSurfaceAt(Math.max(selection, 1))} shortcut={command('insertBefore').shortcut}>
              {t('lens.grid.insertBefore')}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => insertSurfaceAt(Math.min(selection + 1, imageIndex(system)))} shortcut={command('insertAfter').shortcut}>
              {t('lens.grid.insertAfter')}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => deleteSurfaceAt(selection)} disabled={!canDelete(system, selection)} shortcut={command('delete').shortcut}>
              {t('lens.grid.deleteSurface')}
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() => makeStop(selection)}
              disabled={selection === 0 || selection === imageIndex(system) || selection === system.stop}
              shortcut={command('stop').shortcut}
            >
              {t('lens.grid.makeStop')}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={resetWidths} disabled={Object.keys(widths).length === 0}>
              {t('lens.grid.resetWidths')}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <PanelMenuItems panel={LENS_DATA} Item={DropdownMenuItem} Separator={DropdownMenuSeparator} />
          </DropdownMenuContent>
        </DropdownMenu>
      </PanelHeader>
      <div ref={box} className="relative min-h-0 flex-1" title={hoverTitle} onContextMenu={(e) => e.preventDefault()}>
        <DataEditor
          ref={ref}
          width={size.w}
          height={size.h}
          theme={theme}
          columns={columns}
          onColumnResize={(c, w) => setWidth(c.id as ColumnId, w)}
          minColumnWidth={40}
          maxColumnWidth={400}
          rows={rows}
          rowHeight={24}
          headerHeight={26}
          rowMarkers="none"
          smoothScrollY
          customRenderers={RENDERERS}
          getCellContent={getCellContent}
          onCellEdited={onCellEdited}
          onFinishedEditing={onFinishedEditing}
          onCellActivated={onCellActivated}
          onCellClicked={([col, row]) => {
            // The type cell is a menu: one click opens it on a selected row.
            if (COLUMNS[col].id === 'type' && row === useView.getState().selection) onCellActivated([col, row])
          }}
          getCellsForSelection
          onPaste
          gridSelection={gridSel}
          onGridSelectionChange={(g) => {
            setGridSel(g)
            const r = g.current?.cell[1]
            if (r != null && r !== useView.getState().selection) select(r)
          }}
          onItemHovered={(e) => {
            const [col, row] = e.location
            const list = e.kind === 'cell' && COLUMNS[col]?.id === 'surf' ? ctx.issues.get(row) : undefined
            const cellTitle = e.kind === 'cell' ? markerTitle(ctx, col, row) : undefined
            setHoverTitle(list?.length ? list.map((i) => i.message).join('\n') : cellTitle)
          }}
          onCellContextMenu={([, row], e) => {
            e.preventDefault()
            select(row)
            setMenu({ row, x: e.bounds.x + e.localEventX, y: e.bounds.y + e.localEventY, context: true })
          }}
          rangeSelect="cell"
          columnSelect="none"
          rowSelect="none"
          keybindings={{ search: false }}
          isOutsideClick={(e) => !(e.target instanceof Element && e.target.closest('[data-lens-completions]'))}
        />
        <RowMenu state={menu} onClose={() => setMenu(null)} />
      </div>
    </section>
  )
}

/** The help of a cell's markers (`Variable, bounded · Solve: Marginal ray height`). */
function markerTitle(ctx: Context, col: number, row: number): string | undefined {
  const s = ctx.system.surfaces[row]
  if (!s) return undefined
  const id = COLUMNS[col].id
  const out: string[] = []
  for (const sec of ['y', 'x'] as const) {
    const m = markerFor(ctx, columnParam(s, id, sec))
    if (!m) continue
    const pre = s.shape.type === 'toroid' || s.shape.type === 'biconic' ? `${sec}: ` : ''
    if (m.variable) {
      const base = tr(m.bounds === 'bounded' ? 'lens.grid.marker.variableBounded' : m.bounds === 'pinned' ? 'lens.grid.marker.variablePinned' : 'lens.grid.marker.variable')
      out.push(pre + (m.solve ? tr('lens.grid.marker.bySolve', { text: base }) : base))
    }
    if (m.solveTitle) out.push(pre + tr('lens.grid.marker.solve', { title: m.solveTitle }))
  }
  return out.length ? out.join(' · ') : undefined
}

/** The variables a row offers (radius, radius x, thickness, conic, conic x). */
function rowVariables(s: Surface, system: System): { title: string; path: ParamPath }[] {
  const anamorphic = s.shape.type === 'toroid' || s.shape.type === 'biconic'
  const list: [string, ParamPath | null][] = [
    [tr(anamorphic ? 'lens.grid.rowVariable.radiusY' : s.shape.type === 'paraxial' ? 'lens.grid.rowVariable.focalLength' : 'lens.grid.rowVariable.radius'), columnParam(s, 'radius')],
    [tr('lens.grid.rowVariable.radiusX'), columnParam(s, 'radius', 'x')],
    [tr('lens.grid.rowVariable.thickness'), columnParam(s, 'thickness')],
    [tr(anamorphic ? 'lens.grid.rowVariable.conicY' : 'lens.grid.rowVariable.conic'), columnParam(s, 'conic')],
    [tr('lens.grid.rowVariable.conicX'), columnParam(s, 'conic', 'x')],
  ]
  return list.flatMap(([title, path]) => (path && paramResolves(system, path) ? [{ title, path }] : []))
}

/** The type menu of the type column (grouped Spherical, Aspheric, Freeform,
 *  Anamorphic with the cylinder presets, Diffractive, Special) and the row's
 *  context menu (insert, delete, stop, variables, the type, Show in Inspector). */
function RowMenu({ state, onClose }: { state: { row: number; x: number; y: number; context?: boolean } | null; onClose: () => void }) {
  const t = useT()
  const project = useDocument((s) => s.project)
  if (!state) return null
  const system = project.system
  const s = system.surfaces[state.row]
  if (!s) return null
  const img = imageIndex(system)
  const fixedEnd = state.row === 0 || state.row === img
  const current = typeChoice(s.shape)
  const types = (
    <>
      {GROUPS.map((g) => (
        <div key={g}>
          <DropdownMenuLabel>{groupTitle(g)}</DropdownMenuLabel>
          {SURFACE_TYPES.filter((t) => t.group === g).map((t) => (
            <DropdownMenuCheckboxItem key={t.id} checked={t.id === current} onSelect={() => applyTypeChoice(state.row, t.id)}>
              {t.title}
            </DropdownMenuCheckboxItem>
          ))}
          {g === 'Anamorphic' &&
            CYLINDER_CHOICES.map((c) => (
              <DropdownMenuCheckboxItem key={c.id} checked={c.id === current} onSelect={() => applyTypeChoice(state.row, c.id)}>
                {c.title}
              </DropdownMenuCheckboxItem>
            ))}
        </div>
      ))}
    </>
  )
  return (
    <DropdownMenu open onOpenChange={(o) => !o && onClose()}>
      <DropdownMenuTrigger asChild>
        <span style={{ position: 'fixed', left: state.x, top: state.y, width: 1, height: 1 }} />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-[14rem]" aria-label={state.context ? t('lens.grid.surfaceMenu') : t('lens.grid.surfaceTypeMenu')}>
        {state.context ? (
          <>
            <DropdownMenuItem onSelect={() => insertSurfaceAt(Math.max(state.row, 1))} shortcut={command('insertBefore').shortcut}>
              {t('lens.grid.insertBefore')}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => insertSurfaceAt(Math.min(state.row + 1, img))} shortcut={command('insertAfter').shortcut}>
              {t('lens.grid.insertAfter')}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => deleteSurfaceAt(state.row)} disabled={!canDelete(system, state.row)} shortcut={command('delete').shortcut}>
              {t('lens.grid.deleteSurface')}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => makeStop(state.row)} disabled={fixedEnd || state.row === system.stop} shortcut={command('stop').shortcut}>
              {t('lens.grid.makeStop')}
            </DropdownMenuItem>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>{t('lens.grid.variablesMenu')}</DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {rowVariables(s, system).length === 0 && <DropdownMenuItem disabled>{t('lens.grid.noVariables')}</DropdownMenuItem>}
                {rowVariables(s, system).map(({ title, path }) => {
                  const on = isVariable(project, path)
                  const refusal = on ? null : variableRefusal(system, path)
                  return (
                    <DropdownMenuCheckboxItem
                      key={title}
                      checked={on}
                      disabled={!!refusal}
                      title={refusal ?? undefined}
                      onSelect={() => setParamVariable(path, !on)}
                      className={refusal ? 'line-through' : undefined}
                    >
                      {title}
                    </DropdownMenuCheckboxItem>
                  )
                })}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
            {!fixedEnd && (
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>{t('lens.grid.surfaceType')}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="min-w-[14rem]">{types}</DropdownMenuSubContent>
              </DropdownMenuSub>
            )}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => {
                useView.getState().select(state.row)
                useView.getState().setPage('surface')
              }}
            >
              {t('lens.grid.showInInspector')}
            </DropdownMenuItem>
          </>
        ) : (
          types
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
