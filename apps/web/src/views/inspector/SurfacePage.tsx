import { MinusCircle, MousePointerClick, Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Json, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { nf, sig } from '@/model/format'
import { materialCompletions } from '@/model/materials'
import type { ParamPath } from '@/model/params'
import { previewKind } from '@/model/shapeSag'
import {
  curvatureFromRadius,
  imageIndex,
  materialText,
  parseMaterial,
  radiusFromCurvature,
  setMaterial,
  surfaceTitle,
  type Shape,
  type Surface,
} from '@/model/system'
import {
  CYLINDER_CHOICES,
  GROUPS,
  SURFACE_TYPES,
  changeShapeType,
  descriptorFor,
  getPath,
  makeCylinder,
  setDiffractiveBase,
  setPath,
  typeChoice,
  type CoefficientList,
  type ShapeParam,
} from '@/model/surfaceTypes'
import { makeStop } from '@/state/commands'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { LAYOUT, useWorkspace } from '@/state/workspace'
import { NumberInput } from '../SettingsForm'
import { DiffractionEfficiencySection } from './DiffractionEfficiency'
import { FormRow, NumberRow, Section, TextInput } from './Form'
import { MeltOffsetRows } from './MeltOffset'
import { ShapePreview } from './ShapePreview'
import { SolvesSection } from './SolvesSection'
import { SolveBadge, VariableToggle } from './VariableToggle'

type SurfaceEdit = (name: string, change: (s: Surface) => void | boolean) => void

/** Edits the surface of id `id` (by id, so a reordered list edits the right one). */
function useSurfaceEdit(id: number): SurfaceEdit {
  const edit = useDocument((s) => s.edit)
  return (name, change) =>
    edit(name, (sys) => {
      const sf = sys.surfaces.find((x) => x.id === id)
      if (!sf) return false
      return change(sf)
    })
}

/** The surface type, grouped by family, cylinder presets under Anamorphic. */
export function TypeSelect({ shape, onChange, disabled, className }: { shape: Shape; onChange: (choice: string) => void; disabled?: boolean; className?: string }) {
  return (
    <Select aria-label="Surface type" className={className} value={typeChoice(shape)} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
      {GROUPS.map((g) => (
        <optgroup key={g} label={g}>
          {SURFACE_TYPES.filter((t) => t.group === g).map((t) => (
            <option key={t.id} value={t.id}>
              {t.title}
            </option>
          ))}
          {g === 'Anamorphic' &&
            CYLINDER_CHOICES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title}
              </option>
            ))}
        </optgroup>
      ))}
    </Select>
  )
}

/** Applies a type-menu choice (a type, or a cylinder preset) as one step. */
export function applyTypeChoice(index: number, choice: string) {
  useDocument.getState().edit('Change Surface Type', (s) => {
    if (choice === 'cylinder_x' || choice === 'cylinder_y') {
      changeShapeType(s, index, 'biconic')
      makeCylinder(s, index, choice === 'cylinder_x' ? 'x' : 'y')
    } else changeShapeType(s, index, choice)
  })
}

function ParamControl({ surface, param, edit }: { surface: Surface; param: ShapeParam; edit: SurfaceEdit }) {
  const raw = getPath(surface.shape, param.key)
  const commit = (v: Json) => edit(`Change ${param.title}`, (s) => void (s.shape = setPath(s.shape, param.key, v)))
  if (param.options) {
    return (
      <FormRow label={param.title} help={param.help}>
        <Select aria-label={param.title} className="w-full" value={String(raw ?? param.options[0].value)} onChange={(e) => commit(e.target.value)}>
          {param.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.title}
            </option>
          ))}
        </Select>
      </FormRow>
    )
  }
  const value = typeof raw === 'number' ? (param.radius ? radiusFromCurvature(raw) : raw) : null
  return (
    <NumberRow
      label={param.title}
      help={param.help}
      unit={param.unit}
      integer={param.integer}
      value={value}
      param={param.path?.(surface.id)}
      onCommit={(v) => v != null && commit(param.radius ? curvatureFromRadius(v) : v)}
    />
  )
}

/** A type's coefficient list (A₄, A₆ … of an even asphere), each term with
 *  its meaning and a variable toggle. Long lists (XY, Zernike) show their
 *  nonzero terms and add any term from a menu. */
function Coefficients({ surface, list, edit }: { surface: Surface; list: CoefficientList; edit: SurfaceEdit }) {
  const shape = surface.shape
  const values = ((getPath(shape, list.key) as number[] | undefined) ?? []).map((v) => Number(v) || 0)
  const max = list.max(shape)
  const [revealed, setRevealed] = useState<Set<number>>(new Set())
  const [showAll, setShowAll] = useState(false)
  useEffect(() => {
    setRevealed(new Set())
    setShowAll(false)
  }, [surface.id])
  const visible = values.map((_, k) => k).filter((k) => !list.sparse || showAll || values[k] !== 0 || revealed.has(k) || k === values.length - 1)
  const write = (name: string, a: number[]) => edit(name, (s) => void (s.shape = setPath(s.shape, list.key, a)))
  const add = (k: number) => {
    const a = values.slice()
    while (a.length <= k) a.push(0)
    setRevealed((r) => new Set(r).add(k))
    write(`Add ${list.term(k, shape)}`, a)
  }
  const remove = (k: number) => {
    let a = values.slice()
    if (k === a.length - 1) {
      a.pop()
      if (list.sparse) while (a.length && a.at(-1) === 0 && !revealed.has(a.length - 1)) a = a.slice(0, -1)
    } else a[k] = 0
    setRevealed((r) => {
      const n = new Set(r)
      n.delete(k)
      return n
    })
    write(`Remove ${list.term(k, shape)}`, a)
  }
  return (
    <div className="mt-1" data-testid={`coefficients-${list.key}`}>
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_2.4rem_1rem] items-center gap-x-1.5 gap-y-[3px]">
        {visible.map((k) => {
          const path: ParamPath | undefined = list.path?.(surface.id, k)
          const removable = k === values.length - 1 || list.sparse
          return (
            <div key={k} className="contents">
              <span className="truncate text-secondary" title={list.meaning?.(k, shape)}>
                {list.term(k, shape)}
                {list.meaning && <span className="ml-1 text-[11px] text-tertiary">{list.meaning(k, shape)}</span>}
              </span>
              <span className="flex min-w-0 items-center gap-1">
                <NumberInput
                  ariaLabel={list.term(k, shape)}
                  value={values[k]}
                  onCommit={(v) => v != null && write(`Change ${list.term(k, shape)}`, values.map((x, i) => (i === k ? v : x)))}
                />
                {path && <SolveBadge path={path} />}
                {path && <VariableToggle path={path} />}
              </span>
              <span className="truncate text-[11px] text-secondary">{list.unit?.(k) ?? ''}</span>
              <button
                type="button"
                aria-label={`Remove ${list.term(k, shape)}`}
                title={k === values.length - 1 ? 'Remove this term' : 'Set this term to zero'}
                disabled={!removable}
                className="text-secondary hover:text-label disabled:invisible"
                onClick={() => remove(k)}
              >
                <MinusCircle className="size-3.5" />
              </button>
            </div>
          )
        })}
      </div>
      <div className="mt-0.5 flex items-center">
        {(values.length < max || (list.sparse && values.includes(0))) &&
          (list.sparse ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="-ml-1.5">
                  <Plus /> Add Term
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="max-h-72">
                {Array.from({ length: max }, (_, k) => k)
                  .filter((k) => k >= values.length || (values[k] === 0 && !revealed.has(k) && k !== values.length - 1))
                  .map((k) => (
                    <DropdownMenuItem key={k} onSelect={() => add(k)}>
                      {list.term(k, shape)}
                      {list.meaning && <span className="ml-auto pl-3 text-[11px] opacity-60">{list.meaning(k, shape)}</span>}
                    </DropdownMenuItem>
                  ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button variant="ghost" size="sm" className="-ml-1.5" onClick={() => add(values.length)}>
              <Plus /> Add {list.term(values.length, shape)}
              {list.meaning && <span className="text-[11px] text-tertiary">{list.meaning(values.length, shape)}</span>}
            </Button>
          ))}
        <span className="flex-1" />
        {list.sparse && values.slice(0, -1).includes(0) && (
          <Button variant="ghost" size="sm" aria-pressed={showAll} className={showAll ? 'text-accent' : ''} onClick={() => setShowAll((v) => !v)} title="Show the terms that are zero">
            Zero Terms
          </Button>
        )}
      </div>
      {values.length >= max && <p className="text-[11px] text-secondary">{max} terms, the most this type allows.</p>}
    </div>
  )
}

const PLACEMENT_KEYS: [string, string, string][] = [
  ['decenter_x', 'Decenter X', 'mm'],
  ['decenter_y', 'Decenter Y', 'mm'],
  ['tilt_x', 'Tilt X', '°'],
  ['tilt_y', 'Tilt Y', '°'],
  ['tilt_z', 'Tilt Z', '°'],
]
const IDENTITY = { decenter_x: 0, decenter_y: 0, tilt_x: 0, tilt_y: 0, tilt_z: 0, order: 'decenter_then_tilt', restore_after: false }

function PlacementSection({ surface, edit }: { surface: Surface; edit: SurfaceEdit }) {
  const [expanded, setExpanded] = useState(false)
  useEffect(() => setExpanded(false), [surface.id])
  const cb = surface.shape.type === 'coordinate_break'
  const p = { ...IDENTITY, ...((surface.placement as JsonObject | null) ?? {}) }
  const identity = PLACEMENT_KEYS.every(([k]) => Number(p[k as keyof typeof p] ?? 0) === 0)
  const set = (patch: JsonObject) =>
    edit('Change Placement', (s) => {
      const next = { ...IDENTITY, ...((s.placement as JsonObject | null) ?? {}), ...patch }
      const id = PLACEMENT_KEYS.every(([k]) => Number(next[k as keyof typeof next] ?? 0) === 0)
      s.placement = id && s.shape.type !== 'coordinate_break' && next.order === 'decenter_then_tilt' && !next.restore_after ? null : next
    })
  return (
    <Section title="Placement" testId="placement-section">
      {expanded || cb || !identity ? (
        <>
          {PLACEMENT_KEYS.map(([k, title, unit]) => (
            <NumberRow
              key={k}
              label={title}
              unit={unit}
              value={Number(p[k as keyof typeof p] ?? 0)}
              param={{ param: k, surface: surface.id }}
              onCommit={(v) => v != null && set({ [k]: v })}
            />
          ))}
          <FormRow label="Order">
            <Select aria-label="Placement order" className="w-full" value={String(p.order)} onChange={(e) => set({ order: e.target.value })}>
              <option value="decenter_then_tilt">Decenter, then tilt</option>
              <option value="tilt_then_decenter">Tilt, then decenter</option>
            </Select>
          </FormRow>
          <FormRow label="Return after surface">
            <Switch aria-label="Return after surface" checked={!!p.restore_after} onCheckedChange={(v) => set({ restore_after: v })} />
          </FormRow>
        </>
      ) : (
        <Button variant="ghost" size="sm" className="-ml-1.5 text-accent" onClick={() => setExpanded(true)}>
          Add Decenter or Tilt…
        </Button>
      )}
    </Section>
  )
}

const APERTURE_SHAPES = [
  { value: 'none', title: 'Semi-diameter' },
  { value: 'circular', title: 'Circular' },
  { value: 'annular', title: 'Annular' },
  { value: 'rectangular', title: 'Rectangular' },
  { value: 'elliptical', title: 'Elliptical' },
]
const APERTURE_FIELDS: Record<string, [string, string][]> = {
  circular: [['radius', 'Radius']],
  annular: [
    ['inner_radius', 'Inner radius'],
    ['outer_radius', 'Outer radius'],
  ],
  rectangular: [
    ['half_x', 'Half width x'],
    ['half_y', 'Half width y'],
  ],
  elliptical: [
    ['semi_x', 'Semi-axis x'],
    ['semi_y', 'Semi-axis y'],
  ],
}

/** Clear aperture: automatic or fixed semi-diameter, and an optional explicit
 *  aperture shape. */
function ApertureSection({ surface, autoValue, edit }: { surface: Surface; autoValue: number | null; edit: SurfaceEdit }) {
  const ap = surface.aperture as JsonObject | null
  const shape = ap ? String(ap.type) : 'none'
  const fixed = surface.semi_diameter.mode === 'fixed'
  const positive = (v: number) => (v > 0 && Number.isFinite(v) ? null : 'The value must be positive.')
  const setShape = (t: string) => {
    const r = (fixed ? Number(surface.semi_diameter.value) : autoValue) || 5
    const next: JsonObject | null =
      t === 'circular'
        ? { type: t, radius: r }
        : t === 'annular'
          ? { type: t, inner_radius: r / 3, outer_radius: r }
          : t === 'rectangular'
            ? { type: t, half_x: r, half_y: r }
            : t === 'elliptical'
              ? { type: t, semi_x: r, semi_y: r }
              : null
    edit('Change Aperture', (s) => void (s.aperture = next))
  }
  return (
    <Section title="Aperture">
      <FormRow label="Semi-diameter">
        <Segmented
          label="Semi-diameter"
          value={fixed ? 'fixed' : 'auto'}
          onChange={(m) =>
            edit('Change Semi-Diameter', (s) => {
              s.semi_diameter = m === 'auto' ? { mode: 'auto' } : { mode: 'fixed', value: Number(s.semi_diameter.value) || (autoValue != null && autoValue > 0 ? Math.round(autoValue * 1e4) / 1e4 : 1) }
            })
          }
          options={[
            { value: 'auto', title: 'Automatic' },
            { value: 'fixed', title: 'Fixed' },
          ]}
        />
      </FormRow>
      {fixed ? (
        <NumberRow
          label="Value"
          unit="mm"
          value={Number(surface.semi_diameter.value)}
          param={{ param: 'semi_diameter', surface: surface.id }}
          validate={positive}
          onCommit={(v) => v != null && edit('Change Semi-Diameter', (s) => void (s.semi_diameter = { mode: 'fixed', value: v }))}
        />
      ) : (
        autoValue != null &&
        Number.isFinite(autoValue) && (
          <FormRow label="Computed" unit="mm">
            <span className="num text-secondary">{sig(autoValue, 6)}</span>
          </FormRow>
        )
      )}
      <FormRow label="Clipping">
        <Select aria-label="Clipping" className="w-full" value={APERTURE_SHAPES.some((a) => a.value === shape) ? shape : 'none'} onChange={(e) => setShape(e.target.value)}>
          {APERTURE_SHAPES.map((a) => (
            <option key={a.value} value={a.value}>
              {a.title}
            </option>
          ))}
        </Select>
      </FormRow>
      {ap &&
        (APERTURE_FIELDS[shape] ?? []).map(([k, title]) => (
          <NumberRow
            key={k}
            label={title}
            unit="mm"
            value={Number(ap[k] ?? 0)}
            validate={positive}
            onCommit={(v) => v != null && edit('Change Aperture', (s) => void (s.aperture = { ...((s.aperture as JsonObject) ?? ap), [k]: v }))}
          />
        ))}
    </Section>
  )
}

/** The surface page (`SurfaceInspector`): title with a stop badge, the type
 *  menu, the type's summary and preview, then the data-driven sections,
 *  spacing and medium, aperture, placement, coating and solves. */
export function SurfacePage() {
  const system = useDocument((s) => s.project.system)
  const i = useView((s) => s.selection)
  const toast = useView((s) => s.toast)
  const materials = useWorkspace((s) => s.materials)
  const layout = useWorkspace((s) => s.runs[LAYOUT]?.result)
  const s = system?.surfaces?.[i]
  const edit = useSurfaceEdit(s?.id ?? -1)
  if (!system?.surfaces) return null
  if (!s) {
    return (
      <div className="flex flex-col items-center gap-1 py-10 text-center text-secondary" data-testid="surface-page">
        <MousePointerClick className="size-6" />
        <span className="font-medium text-label">No Surface Selected</span>
        <span className="text-[12px]">Select a row in the lens data to edit its surface.</span>
      </div>
    )
  }
  const d = descriptorFor(s.shape.type === 'sphere' ? 'conic' : s.shape.type)
  const img = imageIndex(system)
  const isEnd = i === 0 || i === img
  const cb = s.shape.type === 'coordinate_break'
  const autoSemi = nf((layout?.json.surfaces as JsonObject[] | undefined)?.find((x) => x.index === i)?.semi_diameter)
  const auto = Number.isFinite(autoSemi) ? autoSemi : null
  const semi = s.semi_diameter.mode === 'fixed' ? Number(s.semi_diameter.value) : auto
  const primaryUm = system.wavelengths.wavelengths[system.wavelengths.primary]?.um ?? 0.5875618
  const thicknessPath: ParamPath = { param: 'thickness', surface: s.id }
  const placementFirst = cb

  return (
    <div data-testid="surface-page">
      <div className="mb-2 flex items-center gap-2 px-1">
        <h2 className="text-[15px] font-semibold">{surfaceTitle(system, i)}</h2>
        {i === system.stop && <span className="rounded-full bg-accent-soft px-1.5 py-[1px] text-[11px] font-medium text-accent">Stop</span>}
        <span className="flex-1" />
        {!isEnd && i !== system.stop && (
          <Button size="sm" onClick={() => makeStop(i)}>
            Make Stop
          </Button>
        )}
      </div>
      <Section>
        <FormRow label="Type">
          <TypeSelect className="w-full" shape={s.shape} disabled={isEnd} onChange={(c) => applyTypeChoice(i, c)} />
        </FormRow>
        {d && <p className="py-1 text-[12px] text-secondary">{d.summary}</p>}
        {!isEnd && previewKind(s.shape.type) !== 'none' && <ShapePreview shape={s.shape} semiDiameter={semi} primaryUm={primaryUm} />}
        <FormRow label="Comment">
          <TextInput ariaLabel="Comment" value={s.label ?? ''} placeholder="none" onCommit={(v) => edit('Change Label', (sf) => void (sf.label = v))} />
        </FormRow>
      </Section>

      {placementFirst && <PlacementSection surface={s} edit={edit} />}

      {s.shape.type === 'diffractive' && (s.shape.base as Shape | undefined)?.type === 'plane' && (
        <Section title="Base Shape">
          <BasePicker surface={s} edit={edit} />
        </Section>
      )}

      {d?.sections
        .filter((sec) => !sec.applies || sec.applies(s.shape))
        .map((sec) => {
          const params = sec.params.filter((p) => p.options || getPath(s.shape, p.key) !== undefined)
          const list = sec.coefficients
          if (!params.length && !list) return null
          return (
            <Section key={sec.title} title={sec.title} footer={sec.footer}>
              {s.shape.type === 'diffractive' && sec.title === 'Base Shape' && <BasePicker surface={s} edit={edit} />}
              {params.map((p) => (
                <ParamControl key={p.key} surface={s} param={p} edit={edit} />
              ))}
              {list && <Coefficients surface={s} list={list} edit={edit} />}
            </Section>
          )
        })}

      {s.shape.type === 'diffractive' && <DiffractionEfficiencySection surface={s} index={i} />}

      {s.shape.type === 'biconic' && (
        <Section footer="A cylinder is a biconic with one section plane.">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="-ml-1.5 text-accent">
                Make Cylinder
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuItem onSelect={() => applyTypeChoice(i, 'cylinder_x')}>Power in x (plane in y)</DropdownMenuItem>
              <DropdownMenuItem onSelect={() => applyTypeChoice(i, 'cylinder_y')}>Power in y (plane in x)</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </Section>
      )}

      {i !== img && (
        <Section title="Spacing and Medium">
          <NumberRow
            label={i === 0 ? 'Object distance' : 'Thickness'}
            unit="mm"
            help="Distance to the next vertex."
            value={nf(s.thickness)}
            param={i === 0 ? undefined : thicknessPath}
            onCommit={(v) => {
              if (v == null) return
              if (!Number.isFinite(v) && i !== 0) return toast('Only the object distance can be infinite.', 'warning')
              edit('Change Thickness', (sf) => void (sf.thickness = Number.isFinite(v) ? v : 'inf'))
            }}
          />
          {!cb && i !== 0 && (
            <FormRow label="Interaction">
              <Segmented
                label="Interaction"
                value={(s.interaction as 'refract' | 'reflect' | 'dummy') ?? 'refract'}
                onChange={(v) =>
                  edit('Change Interaction', (sf) => {
                    sf.interaction = v
                  })
                }
                options={[
                  { value: 'refract', title: 'Refract' },
                  { value: 'reflect', title: 'Mirror' },
                  { value: 'dummy', title: 'Dummy' },
                ]}
              />
            </FormRow>
          )}
          {!cb && s.interaction === 'refract' && (
            <>
              <FormRow label="Material">
                <TextInput
                  ariaLabel="Material"
                  value={materialText(s) || 'AIR'}
                  placeholder="AIR"
                  suggestions={(t) => materialCompletions(t, materials)}
                  onCommit={(t) => {
                    const m = parseMaterial(t, materials)
                    if ('error' in m) return toast(m.error, 'warning')
                    edit('Change Material', (sf) => setMaterial(sf, m))
                  }}
                />
              </FormRow>
              {s.material.type === 'catalog' && <MeltOffsetRows surface={s} />}
              {s.material.type === 'model' && (
                <>
                  <NumberRow
                    label="n_d"
                    value={Number(s.material.nd)}
                    param={{ param: 'model_nd', surface: s.id }}
                    validate={(v) => (v >= 1 ? null : 'n_d must be at least 1.')}
                    onCommit={(v) => v != null && edit('Change n_d', (sf) => void (sf.material = { ...sf.material, nd: v }))}
                  />
                  <NumberRow
                    label="V_d"
                    value={Number(s.material.vd)}
                    param={{ param: 'model_vd', surface: s.id }}
                    validate={(v) => (v > 0 ? null : 'V_d must be positive.')}
                    onCommit={(v) => v != null && edit('Change V_d', (sf) => void (sf.material = { ...sf.material, vd: v }))}
                  />
                </>
              )}
            </>
          )}
        </Section>
      )}

      {!cb && <ApertureSection surface={s} autoValue={auto} edit={edit} />}

      {!placementFirst && !isEnd && <PlacementSection surface={s} edit={edit} />}

      {!isEnd && !cb && <CoatingSection surface={s} edit={edit} />}

      <SolvesSection index={i} />
    </div>
  )
}

function CoatingSection({ surface, edit }: { surface: Surface; edit: SurfaceEdit }) {
  const coatings = useDocument((s) => s.project.system.coatings as JsonObject | undefined)
  const names = Object.keys(coatings ?? {}).sort()
  const current = typeof surface.coating === 'string' ? surface.coating : ''
  return (
    <Section title="Coating">
      <FormRow label="Coating" help={names.length ? undefined : 'The design defines no coatings.'}>
        <Select aria-label="Coating" className="w-full" value={current} onChange={(e) => edit('Change Coating', (s) => void (s.coating = e.target.value || null))}>
          <option value="">Uncoated</option>
          {names.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
          {current && !names.includes(current) && <option value={current}>{current} (missing)</option>}
        </Select>
      </FormRow>
    </Section>
  )
}

/** A DOE's base: plane, sphere / conic or even asphere ("Change Base Shape"). */
function BasePicker({ surface, edit }: { surface: Surface; edit: SurfaceEdit }) {
  return (
    <FormRow label="Base">
      <Select
        aria-label="Base shape"
        className="w-full"
        value={String((surface.shape.base as Shape | undefined)?.type ?? 'plane')}
        onChange={(e) => edit('Change Base Shape', (sf) => void (sf.shape = setDiffractiveBase(sf.shape, e.target.value)))}
      >
        <option value="plane">Plane</option>
        <option value="conic">Sphere / Conic</option>
        <option value="even_asphere">Even Asphere</option>
      </Select>
    </FormRow>
  )
}
