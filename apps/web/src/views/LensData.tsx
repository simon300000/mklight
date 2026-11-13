import {
  DataEditor,
  GridCellKind,
  type DataEditorRef,
  type EditableGridCell,
  type GridCell,
  type GridColumn,
  type GridSelection,
  type Item,
  type Theme,
  CompactSelection,
} from '@glideapps/glide-data-grid'
import '@glideapps/glide-data-grid/dist/index.css'
import { Minus, Plus, Table2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { compact, exact, nf, parseNumber } from '@/model/format'
import {
  conicOf,
  curvatureFromRadius,
  curvatureOf,
  curvatureXOf,
  imageIndex,
  materialText,
  meltOffsetText,
  parseMaterial,
  radiusFromCurvature,
  setConic,
  setCurvature,
  setMaterial,
  type System,
} from '@/model/system'
import { GROUPS, SURFACE_TYPES, TYPE_SHORT, changeShapeType, makeCylinder } from '@/model/surfaceTypes'
import { deleteSelectedSurface, insertSurfaceAt, makeStop } from '@/state/commands'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { allIssues, LAYOUT, useWorkspace } from '@/state/workspace'
import { cssVar, useDark } from '@/lib/theme'
import { shortcutLabel } from '@/lib/utils'
import { PanelHeader } from './PanelHeader'

type ColumnId = 'surf' | 'type' | 'comment' | 'radius' | 'thickness' | 'material' | 'semi' | 'conic'

const COLUMNS: { id: ColumnId; title: string; width: number }[] = [
  { id: 'surf', title: 'Surf', width: 58 },
  { id: 'type', title: 'Type', width: 80 },
  { id: 'comment', title: 'Comment', width: 110 },
  { id: 'radius', title: 'Radius (mm)', width: 110 },
  { id: 'thickness', title: 'Thickness (mm)', width: 104 },
  { id: 'material', title: 'Material', width: 100 },
  { id: 'semi', title: 'Semi-Diam. (mm)', width: 112 },
  { id: 'conic', title: 'Conic', width: 84 },
]

function useGridTheme(): Partial<Theme> {
  const dark = useDark()
  return useMemo(() => {
    const v = (n: string) => cssVar(n) || undefined
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
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif',
      baseFontStyle: '13px',
      headerFontStyle: '600 11px',
      editorFontSize: '13px',
      cellHorizontalPadding: 8,
      bgSearchResult: v('--accent-soft'),
    } satisfies Partial<Theme>
  }, [dark])
}

/** Text of the radius cell: anamorphic surfaces show both sections. */
function radiusText(sys: System, i: number): { text: string; edit: string } {
  const shape = sys.surfaces[i].shape
  const c = curvatureOf(shape)
  if (c == null) return { text: '', edit: '' }
  const r = radiusFromCurvature(c)
  const cx = curvatureXOf(shape)
  if (cx != null) {
    const rx = radiusFromCurvature(cx)
    return { text: `y ${compact(r)}   x ${compact(rx)}`, edit: exact(r) }
  }
  return { text: compact(r), edit: exact(r) }
}

/** The lens data table (`LensTableView`): what every surface has; everything
 *  type-specific lives in the inspector. Edits go to the engine as a new
 *  snapshot; validation issues badge their surface. */
export function LensData() {
  const system = useDocument((s) => s.project.system)
  const edit = useDocument((s) => s.edit)
  const selection = useView((s) => s.selection)
  const select = useView((s) => s.select)
  const toast = useView((s) => s.toast)
  const materials = useWorkspace((s) => s.materials)
  const layout = useWorkspace((s) => s.runs[LAYOUT]?.result)
  const modelIssues = useWorkspace((s) => s.modelIssues)
  const validationIssues = useWorkspace((s) => s.validationIssues)
  const theme = useGridTheme()
  const ref = useRef<DataEditorRef>(null)
  const box = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 600, h: 200 })
  const [typeMenu, setTypeMenu] = useState<{ row: number; x: number; y: number } | null>(null)
  const [gridSel, setGridSel] = useState<GridSelection>({
    columns: CompactSelection.empty(),
    rows: CompactSelection.empty(),
    current: undefined,
  })

  const known = useMemo(() => new Set(materials), [materials])
  const issueSurfaces = useMemo(() => {
    const m = new Map<number, 'error' | 'warning'>()
    for (const i of allIssues({ modelIssues, validationIssues })) {
      if (i.surface == null || i.severity === 'info') continue
      if (m.get(i.surface) !== 'error') m.set(i.surface, i.severity)
    }
    return m
  }, [modelIssues, validationIssues])

  /** Automatic semi-diameters come from the engine's layout. */
  const autoSemi = useMemo(() => {
    const m = new Map<number, number>()
    for (const s of (layout?.json.surfaces as JsonObject[] | undefined) ?? []) m.set(Number(s.index), nf(s.semi_diameter))
    return m
  }, [layout])

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

  const rows = system?.surfaces?.length ?? 0
  const columns: GridColumn[] = useMemo(() => COLUMNS.map((c) => ({ id: c.id, title: c.title, width: c.width })), [])

  const getCellContent = useCallback(
    ([col, row]: Item): GridCell => {
      const s = system.surfaces[row]
      const id = COLUMNS[col].id
      const img = imageIndex(system)
      const text = (display: string, data = display, opts?: Partial<{ readonly: boolean; dim: boolean; align: 'left' | 'right' | 'center'; bold: boolean }>): GridCell => ({
        kind: GridCellKind.Text,
        data,
        displayData: display,
        allowOverlay: !opts?.readonly,
        readonly: opts?.readonly,
        contentAlign: opts?.align,
        themeOverride: opts?.dim ? { textDark: cssVar('--tertiary') } : undefined,
      })
      switch (id) {
        case 'surf': {
          const label = row === 0 ? 'OBJ' : row === img ? 'IMA' : row === system.stop ? `STO ${row}` : String(row)
          const issue = issueSurfaces.get(row)
          return {
            kind: GridCellKind.Text,
            data: label,
            displayData: issue ? `${label} ${issue === 'error' ? '⛔︎' : '⚠︎'}` : label,
            allowOverlay: false,
            readonly: true,
            themeOverride: issue
              ? { textDark: cssVar(issue === 'error' ? '--danger' : '--warning') }
              : row === system.stop
                ? { textDark: cssVar('--accent') }
                : undefined,
          }
        }
        case 'type':
          return text(TYPE_SHORT[s.shape.type] ?? s.shape.type, s.shape.type, { readonly: true })
        case 'comment':
          return text(s.label ?? '')
        case 'radius': {
          const r = radiusText(system, row)
          if (s.shape.type === 'coordinate_break' || s.shape.type === 'paraxial') return text('', '', { readonly: true })
          return text(r.text, r.edit, { align: 'right' })
        }
        case 'thickness': {
          if (row === img) return text('', '', { readonly: true })
          const t = nf(s.thickness)
          return text(compact(t), exact(t), { align: 'right' })
        }
        case 'material': {
          if (row === img) return text('', '', { readonly: true })
          const m = materialText(s)
          const off = meltOffsetText(s.material)
          return text(off ? `${m} ${off}` : m, m)
        }
        case 'semi': {
          if (s.semi_diameter.mode === 'fixed') {
            const v = Number(s.semi_diameter.value)
            return text(`${compact(v)} U`, exact(v), { align: 'right' })
          }
          const a = autoSemi.get(row)
          return text(a != null && Number.isFinite(a) ? compact(a) : 'auto', '', { align: 'right', dim: true })
        }
        case 'conic': {
          const k = conicOf(s.shape)
          if (k == null) return text('', '', { readonly: true })
          if (s.shape.type === 'biconic') {
            const kx = Number(s.shape.conic_x ?? 0)
            return text(`y ${compact(k)}   x ${compact(kx)}`, exact(k), { align: 'right' })
          }
          return text(compact(k), exact(k), { align: 'right' })
        }
      }
    },
    [system, issueSurfaces, autoSemi],
  )

  const onCellEdited = useCallback(
    ([col, row]: Item, value: EditableGridCell) => {
      if (value.kind !== GridCellKind.Text) return
      const t = value.data
      const id = COLUMNS[col].id
      const number = (allowEmpty = false): number | null | undefined => {
        if (allowEmpty && t.trim() === '') return undefined
        const v = parseNumber(t)
        if (v == null || Number.isNaN(v)) {
          toast(`"${t}" is not a number.`, 'warning')
          return null
        }
        return v
      }
      switch (id) {
        case 'comment':
          edit('Change Label', (s) => {
            s.surfaces[row].label = t
          })
          break
        case 'radius': {
          const r = number()
          if (r == null) return
          edit('Change Radius', (s) => {
            s.surfaces[row].shape = setCurvature(s.surfaces[row].shape, curvatureFromRadius(r))
          })
          break
        }
        case 'thickness': {
          const v = number()
          if (v == null) return
          if (!Number.isFinite(v) && row !== 0) {
            toast('Only the object distance can be infinite.', 'warning')
            return
          }
          edit('Change Thickness', (s) => {
            s.surfaces[row].thickness = Number.isFinite(v) ? v : 'inf'
          })
          break
        }
        case 'material': {
          const m = parseMaterial(t, known)
          if ('error' in m) {
            toast(m.error, 'warning')
            return
          }
          edit('Change Material', (s) => setMaterial(s.surfaces[row], m))
          break
        }
        case 'semi': {
          const lower = t.trim().toLowerCase()
          if (lower === '' || lower === 'auto') {
            edit('Change Semi-Diameter', (s) => {
              s.surfaces[row].semi_diameter = { mode: 'auto' }
            })
            break
          }
          const v = number()
          if (v == null) return
          if (!(v > 0) || !Number.isFinite(v)) {
            toast('A semi-diameter must be positive.', 'warning')
            return
          }
          edit('Change Semi-Diameter', (s) => {
            s.surfaces[row].semi_diameter = { mode: 'fixed', value: v }
          })
          break
        }
        case 'conic': {
          const v = number()
          if (v == null) return
          if (!Number.isFinite(v)) {
            toast('A conic constant must be finite.', 'warning')
            return
          }
          edit('Change Conic', (s) => {
            s.surfaces[row].shape = setConic(s.surfaces[row].shape, v)
          })
          break
        }
      }
    },
    [edit, known, toast],
  )

  const onCellActivated = useCallback(
    ([col, row]: Item) => {
      if (COLUMNS[col].id !== 'type') return
      const b = ref.current?.getBounds(col, row)
      if (b) setTypeMenu({ row, x: b.x, y: b.y + b.height })
    },
    [],
  )

  if (!system?.surfaces) return null

  return (
    <section className="flex h-full min-h-0 flex-col bg-content" aria-label="Lens Data" data-testid="lens-data">
      <PanelHeader icon={<Table2 />} title="Lens Data" caption={`${rows - 2} surfaces · stop ${system.stop}`}>
        <Tip label={`Insert Surface (${shortcutLabel('alt+mod+i')})`}>
          <Button variant="ghost" size="iconSm" aria-label="Insert Surface" onClick={() => insertSurfaceAt(Math.max(selection + 1, 1))}>
            <Plus />
          </Button>
        </Tip>
        <Tip label={`Delete Surface (${shortcutLabel('mod+Backspace')})`}>
          <Button variant="ghost" size="iconSm" aria-label="Delete Surface" onClick={deleteSelectedSurface}>
            <Minus />
          </Button>
        </Tip>
      </PanelHeader>
      <div ref={box} className="relative min-h-0 flex-1" onContextMenu={(e) => e.preventDefault()}>
        <DataEditor
          ref={ref}
          width={size.w}
          height={size.h}
          theme={theme}
          columns={columns}
          rows={rows}
          rowHeight={24}
          headerHeight={26}
          rowMarkers="none"
          smoothScrollY
          getCellContent={getCellContent}
          onCellEdited={onCellEdited}
          onCellActivated={onCellActivated}
          getCellsForSelection
          onPaste
          gridSelection={gridSel}
          onGridSelectionChange={(g) => {
            setGridSel(g)
            const r = g.current?.cell[1]
            if (r != null && r !== useView.getState().selection) select(r)
          }}
          onCellContextMenu={([col, row], e) => {
            e.preventDefault()
            select(row)
            const b = ref.current?.getBounds(col, row)
            if (b) setTypeMenu({ row, x: b.x, y: b.y + b.height, context: true } as never)
          }}
          rangeSelect="cell"
          columnSelect="none"
          rowSelect="none"
          keybindings={{ search: false }}
        />
        <TypeMenu
          state={typeMenu}
          onClose={() => setTypeMenu(null)}
          onPick={(row, type) => {
            if (type.startsWith('cylinder_')) {
              edit('Change Surface Type', (s) => makeCylinder(s, row, type === 'cylinder_x' ? 'x' : 'y'))
            } else {
              edit('Change Surface Type', (s) => changeShapeType(s, row, type))
            }
          }}
        />
      </div>
    </section>
  )
}

/** The type menu, grouped Spherical, Aspheric, Freeform, Anamorphic,
 *  Diffractive, Special (cylinder presets under Anamorphic); from the context
 *  menu it also has the surface commands. */
function TypeMenu({
  state,
  onClose,
  onPick,
}: {
  state: { row: number; x: number; y: number; context?: boolean } | null
  onClose: () => void
  onPick: (row: number, type: string) => void
}) {
  const system = useDocument((s) => s.project.system)
  if (!state) return null
  const cur = system.surfaces[state.row]?.shape.type
  const fixedEnd = state.row === 0 || state.row === imageIndex(system)
  return (
    <DropdownMenu open onOpenChange={(o) => !o && onClose()}>
      <DropdownMenuTrigger asChild>
        <span style={{ position: 'fixed', left: state.x, top: state.y, width: 1, height: 1 }} />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-[14rem]">
        {state.context && (
          <>
            <DropdownMenuItem onSelect={() => insertSurfaceAt(Math.max(state.row, 1))} shortcut="shift+mod+i">
              Insert Surface Before
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => insertSurfaceAt(state.row + 1)} shortcut="alt+mod+i">
              Insert Surface After
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={deleteSelectedSurface} disabled={fixedEnd} shortcut="mod+Backspace">
              Delete Surface
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => makeStop(state.row)} disabled={fixedEnd} shortcut="shift+mod+t">
              Make Aperture Stop
            </DropdownMenuItem>
            <DropdownMenuSeparator />
          </>
        )}
        {GROUPS.map((g) => (
          <div key={g}>
            <DropdownMenuLabel>{g}</DropdownMenuLabel>
            {SURFACE_TYPES.filter((t) => t.group === g).map((t) => (
              <DropdownMenuItem key={t.id} inset onSelect={() => onPick(state.row, t.id)} className={t.id === cur ? 'font-semibold' : ''}>
                {t.title}
              </DropdownMenuItem>
            ))}
            {g === 'Anamorphic' && (
              <>
                <DropdownMenuItem inset onSelect={() => onPick(state.row, 'cylinder_y')}>
                  Cylinder (y power)
                </DropdownMenuItem>
                <DropdownMenuItem inset onSelect={() => onPick(state.row, 'cylinder_x')}>
                  Cylinder (x power)
                </DropdownMenuItem>
              </>
            )}
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
