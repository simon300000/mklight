import { MinusCircle, Plus } from 'lucide-react'
import { useEffect, useRef } from 'react'
import type { Json, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { nf, sig } from '@/model/format'
import {
  conicOf,
  curvatureFromRadius,
  curvatureOf,
  imageIndex,
  materialText,
  parseMaterial,
  radiusFromCurvature,
  setMaterial,
  surfaceTitle,
  type Shape,
} from '@/model/system'
import { GROUPS, SURFACE_TYPES, changeShapeType, descriptorFor, getPath, setPath, type CoefficientList, type ShapeParam } from '@/model/surfaceTypes'
import { makeStop } from '@/state/commands'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { LAYOUT, useWorkspace } from '@/state/workspace'
import { cssVar, useDark } from '@/lib/theme'
import { NumberInput } from '../SettingsForm'
import { FormRow, NumberRow, Section, TextInput } from './Form'

/** Sag of a rotational shape at height r (display geometry of the preview,
 *  from the documented formulas; docs/SURFACES.md). */
function sag(shape: Shape, r: number): number | null {
  const base = shape.type === 'diffractive' ? (shape.base as Shape) : shape
  const c = curvatureOf(base) ?? 0
  const k = conicOf(base) ?? 0
  const q = 1 - (1 + k) * c * c * r * r
  if (q < 0) return null
  let z = (c * r * r) / (1 + Math.sqrt(q))
  if (base.type === 'even_asphere' || shape.type === 'diffractive') {
    ;((base.coefficients as number[]) ?? []).forEach((a, i) => (z += a * r ** (2 * i + 4)))
  }
  if (base.type === 'odd_asphere') ((base.coefficients as number[]) ?? []).forEach((b, i) => (z += b * r ** (i + 1)))
  return z
}

function SagPreview({ shape, semi }: { shape: Shape; semi: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const dark = useDark()
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const w = c.clientWidth, h = 90
    const dpr = devicePixelRatio || 1
    c.width = w * dpr
    c.height = h * dpr
    const g = c.getContext('2d')
    if (!g) return
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, w, h)
    const pts: [number, number][] = []
    for (let i = 0; i <= 80; i++) {
      const r = -semi + (2 * semi * i) / 80
      const z = sag(shape, Math.abs(r))
      if (z != null && Number.isFinite(z)) pts.push([z, r])
    }
    if (!pts.length) return
    const zs = pts.map((p) => p[0])
    const zmin = Math.min(...zs, 0), zmax = Math.max(...zs, 0)
    const scale = Math.min((h - 16) / (2 * semi), (w - 40) / Math.max(zmax - zmin, 1e-9))
    const x0 = w / 2 - ((zmin + zmax) / 2) * scale
    g.strokeStyle = cssVar('--grid')
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke()
    g.strokeStyle = cssVar('--accent')
    g.lineWidth = 1.5
    g.beginPath()
    pts.forEach(([z, r], i) => (i ? g.lineTo(x0 + z * scale, h / 2 - r * scale) : g.moveTo(x0 + z * scale, h / 2 - r * scale)))
    g.stroke()
    g.fillStyle = cssVar('--secondary')
    g.font = '10px -apple-system, system-ui, sans-serif'
    g.fillText(`sag ${sig(sag(shape, semi), 4)} mm at r ${sig(semi, 4)}`, 6, h - 4)
  }, [shape, semi, dark])
  return <canvas ref={ref} className="h-[90px] w-full" role="img" aria-label="Sag profile" />
}

function ParamControl({ index, param, shape }: { index: number; param: ShapeParam; shape: Shape }) {
  const edit = useDocument((s) => s.edit)
  const raw = getPath(shape, param.key)
  const commit = (v: Json) =>
    edit(`Change ${param.title}`, (s) => {
      s.surfaces[index].shape = setPath(s.surfaces[index].shape, param.key, v)
    })
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
      onCommit={(v) => v != null && commit(param.radius ? curvatureFromRadius(v) : v)}
    />
  )
}

function Coefficients({ index, list, shape }: { index: number; list: CoefficientList; shape: Shape }) {
  const edit = useDocument((s) => s.edit)
  const coeffs = ((getPath(shape, list.key) as number[] | undefined) ?? []).slice()
  const max = list.max(shape)
  const set = (name: string, next: number[]) =>
    edit(name, (s) => {
      s.surfaces[index].shape = setPath(s.surfaces[index].shape, list.key, next)
    })
  return (
    <div className="mt-1">
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_2rem_1rem] items-center gap-x-2 gap-y-[3px]">
        {coeffs.map((c, k) => (
          <div key={k} className="contents">
            <span className="truncate text-secondary" title={list.meaning?.(k, shape)}>
              {list.term(k, shape)}
              {list.meaning && <span className="ml-1 text-[11px] text-tertiary">{list.meaning(k, shape)}</span>}
            </span>
            <NumberInput ariaLabel={list.term(k, shape)} value={c} onCommit={(v) => v != null && set(`Change ${list.title}`, coeffs.map((x, i) => (i === k ? v : x)))} />
            <span className="truncate text-[11px] text-secondary">{list.unit?.(k) ?? ''}</span>
            <button
              type="button"
              aria-label="Remove this term"
              title="Remove this term"
              className="text-secondary hover:text-label"
              onClick={() => set('Remove Term', coeffs.filter((_, i) => i !== k))}
            >
              <MinusCircle className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
      <Button variant="ghost" size="sm" className="-ml-1.5 mt-0.5" disabled={coeffs.length >= max} onClick={() => set('Add Term', [...coeffs, 0])}>
        <Plus /> Add {list.term(coeffs.length, shape)}
      </Button>
    </div>
  )
}

const PLACEMENT_KEYS: [string, string, string][] = [
  ['decenter_x', 'Decenter x', 'mm'],
  ['decenter_y', 'Decenter y', 'mm'],
  ['tilt_x', 'Tilt x', '°'],
  ['tilt_y', 'Tilt y', '°'],
  ['tilt_z', 'Tilt z', '°'],
]

/** The surface page: title with a stop badge, the type menu, the type's
 *  summary and preview, then the data-driven sections, spacing and medium,
 *  aperture and placement. */
export function SurfacePage() {
  const system = useDocument((s) => s.project.system)
  const edit = useDocument((s) => s.edit)
  const i = useView((s) => s.selection)
  const toast = useView((s) => s.toast)
  const materials = useWorkspace((s) => s.materials)
  const layout = useWorkspace((s) => s.runs[LAYOUT]?.result)
  if (!system?.surfaces?.[i]) return null
  const s = system.surfaces[i]
  const d = descriptorFor(s.shape.type === 'sphere' ? 'conic' : s.shape.type)
  const img = imageIndex(system)
  const isEnd = i === 0 || i === img
  const autoSemi = (layout?.json.surfaces as JsonObject[] | undefined)?.find((x) => x.index === i)?.semi_diameter
  const semi = s.semi_diameter.mode === 'fixed' ? Number(s.semi_diameter.value) : nf(autoSemi)
  const placement = s.placement as JsonObject | null
  const rotational = ['plane', 'conic', 'sphere', 'even_asphere', 'odd_asphere', 'diffractive'].includes(s.shape.type)

  return (
    <div data-testid="surface-page">
      <div className="mb-3 flex items-center gap-2 px-1">
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
          <Select
            aria-label="Surface type"
            className="w-full"
            value={s.shape.type === 'sphere' ? 'conic' : s.shape.type}
            disabled={isEnd}
            onChange={(e) => edit('Change Surface Type', (sys) => changeShapeType(sys, i, e.target.value))}
          >
            {GROUPS.map((g) => (
              <optgroup key={g} label={g}>
                {SURFACE_TYPES.filter((t) => t.group === g).map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.title}
                  </option>
                ))}
              </optgroup>
            ))}
          </Select>
        </FormRow>
        <FormRow label="Comment">
          <TextInput ariaLabel="Comment" value={s.label ?? ''} onCommit={(v) => edit('Change Label', (sys) => void (sys.surfaces[i].label = v))} />
        </FormRow>
        {d && <p className="py-1 text-[12px] text-secondary">{d.summary}</p>}
        {rotational && semi > 0 && Number.isFinite(semi) && <SagPreview shape={s.shape} semi={semi} />}
      </Section>

      {d?.sections.map((sec) => (
        <Section key={sec.title} title={sec.title} footer={sec.footer}>
          {sec.params.map((p) => (
            <ParamControl key={p.key} index={i} param={p} shape={s.shape} />
          ))}
          {sec.coefficients && <Coefficients index={i} list={sec.coefficients} shape={s.shape} />}
        </Section>
      ))}

      {i !== img && (
        <Section title="Spacing and Medium">
          <NumberRow
            label={i === 0 ? 'Object distance' : 'Thickness'}
            unit="mm"
            value={nf(s.thickness)}
            onCommit={(v) => {
              if (v == null) return
              if (!Number.isFinite(v) && i !== 0) return toast('Only the object distance can be infinite.', 'warning')
              edit('Change Thickness', (sys) => void (sys.surfaces[i].thickness = Number.isFinite(v) ? v : 'inf'))
            }}
          />
          <FormRow label="Material">
            <TextInput
              ariaLabel="Material"
              value={materialText(s)}
              placeholder="air"
              onCommit={(t) => {
                const m = parseMaterial(t, new Set(materials))
                if ('error' in m) return toast(m.error, 'warning')
                edit('Change Material', (sys) => setMaterial(sys.surfaces[i], m))
              }}
            />
          </FormRow>
        </Section>
      )}

      <Section title="Aperture">
        <FormRow label="Automatic semi-diameter">
          <Switch
            aria-label="Automatic semi-diameter"
            checked={s.semi_diameter.mode === 'auto'}
            onCheckedChange={(auto) =>
              edit('Change Semi-Diameter', (sys) => {
                sys.surfaces[i].semi_diameter = auto ? { mode: 'auto' } : { mode: 'fixed', value: Number.isFinite(semi) && semi > 0 ? Math.round(semi * 1e4) / 1e4 : 10 }
              })
            }
          />
        </FormRow>
        <NumberRow
          label="Semi-diameter"
          unit="mm"
          value={Number.isFinite(semi) ? semi : null}
          nullable
          min={0}
          onCommit={(v) =>
            edit('Change Semi-Diameter', (sys) => {
              sys.surfaces[i].semi_diameter = v == null || !(v > 0) ? { mode: 'auto' } : { mode: 'fixed', value: v }
            })
          }
        />
      </Section>

      {(placement || s.shape.type === 'coordinate_break') && (
        <Section title="Placement">
          {PLACEMENT_KEYS.map(([k, title, unit]) => (
            <NumberRow
              key={k}
              label={title}
              unit={unit}
              value={Number(placement?.[k] ?? 0)}
              onCommit={(v) =>
                v != null &&
                edit('Change Placement', (sys) => {
                  const p = (sys.surfaces[i].placement as JsonObject | null) ?? {
                    decenter_x: 0, decenter_y: 0, tilt_x: 0, tilt_y: 0, tilt_z: 0, order: 'decenter_then_tilt', restore_after: false,
                  }
                  sys.surfaces[i].placement = { ...p, [k]: v }
                })
              }
            />
          ))}
          <FormRow label="Order">
            <Select
              aria-label="Placement order"
              className="w-full"
              value={String(placement?.order ?? 'decenter_then_tilt')}
              onChange={(e) => edit('Change Placement', (sys) => void (sys.surfaces[i].placement = { ...((sys.surfaces[i].placement as JsonObject) ?? {}), order: e.target.value }))}
            >
              <option value="decenter_then_tilt">Decenter, then tilt</option>
              <option value="tilt_then_decenter">Tilt, then decenter</option>
            </Select>
          </FormRow>
        </Section>
      )}
    </div>
  )
}
