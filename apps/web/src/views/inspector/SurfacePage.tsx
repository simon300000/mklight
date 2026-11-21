import { MinusCircle, MousePointerClick, Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Json, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Segmented } from '@/components/ui/segmented'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { t as tNow, tMaybe, useT } from '@/i18n'
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
  groupTitle,
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
  const t = useT()
  return (
    <Select aria-label={t('inspector.surface.typeLabel')} className={className} value={typeChoice(shape)} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
      {GROUPS.map((g) => (
        <optgroup key={g} label={groupTitle(g)}>
          {SURFACE_TYPES.filter((ty) => ty.group === g).map((ty) => (
            <option key={ty.id} value={ty.id}>
              {ty.title}
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
  useDocument.getState().edit(tNow('inspector.surface.undoType'), (s) => {
    if (choice === 'cylinder_x' || choice === 'cylinder_y') {
      changeShapeType(s, index, 'biconic')
      makeCylinder(s, index, choice === 'cylinder_x' ? 'x' : 'y')
    } else changeShapeType(s, index, choice)
  })
}

function ParamControl({ surface, param, edit }: { surface: Surface; param: ShapeParam; edit: SurfaceEdit }) {
  const t = useT()
  const raw = getPath(surface.shape, param.key)
  const commit = (v: Json) => edit(t('inspector.surface.changeNamed', { name: param.title }), (s) => void (s.shape = setPath(s.shape, param.key, v)))
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
  const t = useT()
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
    write(t('inspector.surface.addNamed', { name: list.term(k, shape) }), a)
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
    write(t('inspector.surface.removeNamed', { name: list.term(k, shape) }), a)
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
                  onCommit={(v) => v != null && write(t('inspector.surface.changeNamed', { name: list.term(k, shape) }), values.map((x, i) => (i === k ? v : x)))}
                />
                {path && <SolveBadge path={path} />}
                {path && <VariableToggle path={path} />}
              </span>
              <span className="truncate text-[11px] text-secondary">{list.unit?.(k) ?? ''}</span>
              <button
                type="button"
                aria-label={t('inspector.surface.removeNamed', { name: list.term(k, shape) })}
                title={k === values.length - 1 ? t('inspector.surface.removeTerm') : t('inspector.surface.zeroTerm')}
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
                  <Plus /> {t('inspector.surface.addTerm')}
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
              <Plus /> {t('inspector.surface.addNamed', { name: list.term(values.length, shape) })}
              {list.meaning && <span className="text-[11px] text-tertiary">{list.meaning(values.length, shape)}</span>}
            </Button>
          ))}
        <span className="flex-1" />
        {list.sparse && values.slice(0, -1).includes(0) && (
          <Button variant="ghost" size="sm" aria-pressed={showAll} className={showAll ? 'text-accent' : ''} onClick={() => setShowAll((v) => !v)} title={t('inspector.surface.zeroTermsHelp')}>
            {t('inspector.surface.zeroTerms')}
          </Button>
        )}
      </div>
      {values.length >= max && <p className="text-[11px] text-secondary">{t('inspector.surface.maxTerms', { max })}</p>}
    </div>
  )
}

const PLACEMENT_KEYS: [string, 'decenterX' | 'decenterY' | 'tiltX' | 'tiltY' | 'tiltZ', string][] = [
  ['decenter_x', 'decenterX', 'mm'],
  ['decenter_y', 'decenterY', 'mm'],
  ['tilt_x', 'tiltX', '°'],
  ['tilt_y', 'tiltY', '°'],
  ['tilt_z', 'tiltZ', '°'],
]
const IDENTITY = { decenter_x: 0, decenter_y: 0, tilt_x: 0, tilt_y: 0, tilt_z: 0, order: 'decenter_then_tilt', restore_after: false }

function PlacementSection({ surface, edit }: { surface: Surface; edit: SurfaceEdit }) {
  const t = useT()
  const [expanded, setExpanded] = useState(false)
  useEffect(() => setExpanded(false), [surface.id])
  const cb = surface.shape.type === 'coordinate_break'
  const p = { ...IDENTITY, ...((surface.placement as JsonObject | null) ?? {}) }
  const identity = PLACEMENT_KEYS.every(([k]) => Number(p[k as keyof typeof p] ?? 0) === 0)
  const set = (patch: JsonObject) =>
    edit(t('inspector.placement.undo'), (s) => {
      const next = { ...IDENTITY, ...((s.placement as JsonObject | null) ?? {}), ...patch }
      const id = PLACEMENT_KEYS.every(([k]) => Number(next[k as keyof typeof next] ?? 0) === 0)
      s.placement = id && s.shape.type !== 'coordinate_break' && next.order === 'decenter_then_tilt' && !next.restore_after ? null : next
    })
  return (
    <Section title={t('inspector.placement.title')} testId="placement-section">
      {expanded || cb || !identity ? (
        <>
          {PLACEMENT_KEYS.map(([k, title, unit]) => (
            <NumberRow
              key={k}
              label={t(`inspector.placement.${title}`)}
              unit={unit}
              value={Number(p[k as keyof typeof p] ?? 0)}
              param={{ param: k, surface: surface.id }}
              onCommit={(v) => v != null && set({ [k]: v })}
            />
          ))}
          <FormRow label={t('inspector.placement.order')}>
            <Select aria-label={t('inspector.placement.orderLabel')} className="w-full" value={String(p.order)} onChange={(e) => set({ order: e.target.value })}>
              <option value="decenter_then_tilt">{t('inspector.placement.decenterThenTilt')}</option>
              <option value="tilt_then_decenter">{t('inspector.placement.tiltThenDecenter')}</option>
            </Select>
          </FormRow>
          <FormRow label={t('inspector.placement.restore')}>
            <Switch aria-label={t('inspector.placement.restore')} checked={!!p.restore_after} onCheckedChange={(v) => set({ restore_after: v })} />
          </FormRow>
        </>
      ) : (
        <Button variant="ghost" size="sm" className="-ml-1.5 text-accent" onClick={() => setExpanded(true)}>
          {t('inspector.placement.add')}
        </Button>
      )}
    </Section>
  )
}

const APERTURE_SHAPES = ['none', 'circular', 'annular', 'rectangular', 'elliptical']
const APERTURE_FIELDS: Record<string, string[]> = {
  circular: ['radius'],
  annular: ['inner_radius', 'outer_radius'],
  rectangular: ['half_x', 'half_y'],
  elliptical: ['semi_x', 'semi_y'],
}

/** Clear aperture: automatic or fixed semi-diameter, and an optional explicit
 *  aperture shape. */
function ApertureSection({ surface, autoValue, edit }: { surface: Surface; autoValue: number | null; edit: SurfaceEdit }) {
  const t = useT()
  const ap = surface.aperture as JsonObject | null
  const shape = ap ? String(ap.type) : 'none'
  const fixed = surface.semi_diameter.mode === 'fixed'
  const positive = (v: number) => (v > 0 && Number.isFinite(v) ? null : t('inspector.aperture.positive'))
  const setShape = (kind: string) => {
    const r = (fixed ? Number(surface.semi_diameter.value) : autoValue) || 5
    const next: JsonObject | null =
      kind === 'circular'
        ? { type: kind, radius: r }
        : kind === 'annular'
          ? { type: kind, inner_radius: r / 3, outer_radius: r }
          : kind === 'rectangular'
            ? { type: kind, half_x: r, half_y: r }
            : kind === 'elliptical'
              ? { type: kind, semi_x: r, semi_y: r }
              : null
    edit(t('inspector.aperture.undo'), (s) => void (s.aperture = next))
  }
  return (
    <Section title={t('inspector.aperture.title')}>
      <FormRow label={t('inspector.aperture.semiDiameter')}>
        <Segmented
          label={t('inspector.aperture.semiDiameter')}
          value={fixed ? 'fixed' : 'auto'}
          onChange={(m) =>
            edit(t('inspector.aperture.undoSemiDiameter'), (s) => {
              s.semi_diameter = m === 'auto' ? { mode: 'auto' } : { mode: 'fixed', value: Number(s.semi_diameter.value) || (autoValue != null && autoValue > 0 ? Math.round(autoValue * 1e4) / 1e4 : 1) }
            })
          }
          options={[
            { value: 'auto', title: t('common.auto') },
            { value: 'fixed', title: t('inspector.aperture.fixed') },
          ]}
        />
      </FormRow>
      {fixed ? (
        <NumberRow
          label={t('inspector.aperture.value')}
          unit="mm"
          value={Number(surface.semi_diameter.value)}
          param={{ param: 'semi_diameter', surface: surface.id }}
          validate={positive}
          onCommit={(v) => v != null && edit(t('inspector.aperture.undoSemiDiameter'), (s) => void (s.semi_diameter = { mode: 'fixed', value: v }))}
        />
      ) : (
        autoValue != null &&
        Number.isFinite(autoValue) && (
          <FormRow label={t('inspector.aperture.computed')} unit="mm">
            <span className="num text-secondary">{sig(autoValue, 6)}</span>
          </FormRow>
        )
      )}
      <FormRow label={t('inspector.aperture.clipping')}>
        <Select aria-label={t('inspector.aperture.clipping')} className="w-full" value={APERTURE_SHAPES.includes(shape) ? shape : 'none'} onChange={(e) => setShape(e.target.value)}>
          {APERTURE_SHAPES.map((a) => (
            <option key={a} value={a}>
              {tMaybe(`inspector.aperture.shapes.${a}`)}
            </option>
          ))}
        </Select>
      </FormRow>
      {ap &&
        (APERTURE_FIELDS[shape] ?? []).map((k) => (
          <NumberRow
            key={k}
            label={tMaybe(`inspector.aperture.fields.${k}`) ?? k}
            unit="mm"
            value={Number(ap[k] ?? 0)}
            validate={positive}
            onCommit={(v) => v != null && edit(t('inspector.aperture.undo'), (s) => void (s.aperture = { ...((s.aperture as JsonObject) ?? ap), [k]: v }))}
          />
        ))}
    </Section>
  )
}

/** The surface page (`SurfaceInspector`): title with a stop badge, the type
 *  menu, the type's summary and preview, then the data-driven sections,
 *  spacing and medium, aperture, placement, coating and solves. */
export function SurfacePage() {
  const t = useT()
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
        <span className="font-medium text-label">{t('inspector.surface.noneSelected')}</span>
        <span className="text-[12px]">{t('inspector.surface.noneHelp')}</span>
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
        {i === system.stop && <span className="rounded-full bg-accent-soft px-1.5 py-[1px] text-[11px] font-medium text-accent">{t('inspector.surface.stop')}</span>}
        <span className="flex-1" />
        {!isEnd && i !== system.stop && (
          <Button size="sm" onClick={() => makeStop(i)}>
            {t('inspector.surface.makeStop')}
          </Button>
        )}
      </div>
      <Section>
        <FormRow label={t('inspector.surface.type')}>
          <TypeSelect className="w-full" shape={s.shape} disabled={isEnd} onChange={(c) => applyTypeChoice(i, c)} />
        </FormRow>
        {d && <p className="py-1 text-[12px] text-secondary">{d.summary}</p>}
        {!isEnd && previewKind(s.shape.type) !== 'none' && <ShapePreview shape={s.shape} semiDiameter={semi} primaryUm={primaryUm} />}
        <FormRow label={t('inspector.surface.comment')}>
          <TextInput ariaLabel={t('inspector.surface.comment')} value={s.label ?? ''} placeholder={t('inspector.surface.none')} onCommit={(v) => edit(t('inspector.surface.undoLabel'), (sf) => void (sf.label = v))} />
        </FormRow>
      </Section>

      {placementFirst && <PlacementSection surface={s} edit={edit} />}

      {s.shape.type === 'diffractive' && (s.shape.base as Shape | undefined)?.type === 'plane' && (
        <Section title={t('inspector.sections.baseShape')}>
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
            <Section key={sec.id} title={sec.title} footer={sec.footer}>
              {s.shape.type === 'diffractive' && sec.id === 'baseShape' && <BasePicker surface={s} edit={edit} />}
              {params.map((p) => (
                <ParamControl key={p.key} surface={s} param={p} edit={edit} />
              ))}
              {list && <Coefficients surface={s} list={list} edit={edit} />}
            </Section>
          )
        })}

      {s.shape.type === 'diffractive' && <DiffractionEfficiencySection surface={s} index={i} />}

      {s.shape.type === 'biconic' && (
        <Section footer={t('inspector.surface.cylinderFooter')}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="-ml-1.5 text-accent">
                {t('inspector.surface.makeCylinder')}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuItem onSelect={() => applyTypeChoice(i, 'cylinder_x')}>{t('inspector.surface.powerInX')}</DropdownMenuItem>
              <DropdownMenuItem onSelect={() => applyTypeChoice(i, 'cylinder_y')}>{t('inspector.surface.powerInY')}</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </Section>
      )}

      {i !== img && (
        <Section title={t('inspector.surface.spacing')}>
          <NumberRow
            label={i === 0 ? t('inspector.surface.objectDistance') : t('inspector.surface.thickness')}
            unit="mm"
            help={t('inspector.surface.thicknessHelp')}
            value={nf(s.thickness)}
            param={i === 0 ? undefined : thicknessPath}
            onCommit={(v) => {
              if (v == null) return
              if (!Number.isFinite(v) && i !== 0) return toast(t('inspector.surface.onlyObjectInfinite'), 'warning')
              edit(t('inspector.surface.undoThickness'), (sf) => void (sf.thickness = Number.isFinite(v) ? v : 'inf'))
            }}
          />
          {!cb && i !== 0 && (
            <FormRow label={t('inspector.surface.interaction')}>
              <Segmented
                label={t('inspector.surface.interaction')}
                value={(s.interaction as 'refract' | 'reflect' | 'dummy') ?? 'refract'}
                onChange={(v) =>
                  edit(t('inspector.surface.undoInteraction'), (sf) => {
                    sf.interaction = v
                  })
                }
                options={[
                  { value: 'refract', title: t('inspector.surface.refract') },
                  { value: 'reflect', title: t('inspector.surface.mirror') },
                  { value: 'dummy', title: t('inspector.surface.dummy') },
                ]}
              />
            </FormRow>
          )}
          {!cb && s.interaction === 'refract' && (
            <>
              <FormRow label={t('inspector.surface.material')}>
                <TextInput
                  ariaLabel={t('inspector.surface.material')}
                  value={materialText(s) || 'AIR'}
                  placeholder="AIR"
                  suggestions={(q) => materialCompletions(q, materials)}
                  onCommit={(text) => {
                    const m = parseMaterial(text, materials)
                    if ('error' in m) return toast(m.error, 'warning')
                    edit(t('inspector.surface.undoMaterial'), (sf) => setMaterial(sf, m))
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
                    validate={(v) => (v >= 1 ? null : t('inspector.surface.ndMin'))}
                    onCommit={(v) => v != null && edit(t('inspector.surface.undoNd'), (sf) => void (sf.material = { ...sf.material, nd: v }))}
                  />
                  <NumberRow
                    label="V_d"
                    value={Number(s.material.vd)}
                    param={{ param: 'model_vd', surface: s.id }}
                    validate={(v) => (v > 0 ? null : t('inspector.surface.vdPositive'))}
                    onCommit={(v) => v != null && edit(t('inspector.surface.undoVd'), (sf) => void (sf.material = { ...sf.material, vd: v }))}
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
  const t = useT()
  const coatings = useDocument((s) => s.project.system.coatings as JsonObject | undefined)
  const names = Object.keys(coatings ?? {}).sort()
  const current = typeof surface.coating === 'string' ? surface.coating : ''
  return (
    <Section title={t('inspector.coating.title')}>
      <FormRow label={t('inspector.coating.title')} help={names.length ? undefined : t('inspector.coating.none')}>
        <Select aria-label={t('inspector.coating.title')} className="w-full" value={current} onChange={(e) => edit(t('inspector.coating.undo'), (s) => void (s.coating = e.target.value || null))}>
          <option value="">{t('inspector.coating.uncoated')}</option>
          {names.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
          {current && !names.includes(current) && <option value={current}>{t('inspector.coating.missing', { name: current })}</option>}
        </Select>
      </FormRow>
    </Section>
  )
}

/** A DOE's base: plane, sphere / conic or even asphere ("Change Base Shape"). */
function BasePicker({ surface, edit }: { surface: Surface; edit: SurfaceEdit }) {
  const t = useT()
  return (
    <FormRow label={t('inspector.surface.base')}>
      <Select
        aria-label={t('inspector.surface.baseLabel')}
        className="w-full"
        value={String((surface.shape.base as Shape | undefined)?.type ?? 'plane')}
        onChange={(e) => edit(t('inspector.surface.undoBase'), (sf) => void (sf.shape = setDiffractiveBase(sf.shape, e.target.value)))}
      >
        <option value="plane">{t('inspector.types.plane.title')}</option>
        <option value="conic">{t('inspector.types.conic.title')}</option>
        <option value="even_asphere">{t('inspector.types.even_asphere.title')}</option>
      </Select>
    </FormRow>
  )
}
