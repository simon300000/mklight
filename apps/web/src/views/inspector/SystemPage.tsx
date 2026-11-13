import { AlertTriangle, CircleDot, Circle, MinusCircle, Plus } from 'lucide-react'
import { useEffect } from 'react'
import type { JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { APERTURE_TYPES, FIELD_KINDS, WAVELENGTH_PRESETS, fieldUnit, type System } from '@/model/system'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { SUMMARY, useWorkspace } from '@/state/workspace'
import { catalogReferences } from '@/model/materials'
import { loadGlassCatalog, removeGlassCatalog, useCatalogs } from '@/state/commands'
import { anamorphicSummary, squeezeText } from '../StatusCapsule'
import { seriesColor, useDark, wavelengthColor } from '@/lib/theme'
import { sig } from '@/model/format'
import { NumberInput } from '../SettingsForm'
import { FormRow, NumberRow, Section, TextInput } from './Form'

/** Converts field or aperture values with the engine's `convert_spec`, so the
 *  system stays the same; without it the numbers stay (validation says why). */
async function convertSpec(system: System, change: JsonObject): Promise<JsonObject | null> {
  const ws = useWorkspace.getState()
  if (!ws.engine) return null
  try {
    // The snapshot of exactly this system (the debounced one may lag an edit).
    const snap = ws.snapshot && JSON.stringify(ws.snapshot.system) === JSON.stringify(system) ? ws.snapshot : await ws.engine.openSnapshot(system)
    const job = ws.engine.submit({ analysis: 'convert_spec', settings: change }, snap)
    const r = await job.result
    return r.json
  } catch {
    return null
  }
}

/** Applies a conversion only when the design is still the one converted. */
const unchanged = (a: System, b: System) => JSON.stringify(a) === JSON.stringify(b)

/** System settings (`SystemInspector`): name and notes, aperture, fields,
 *  wavelengths (with presets), environment and ray aiming. */
export function SystemPage() {
  const system = useDocument((s) => s.project.system)
  const edit = useDocument((s) => s.edit)
  const dark = useDark()
  if (!system?.surfaces) return null
  const ap = system.aperture
  const apType = APERTURE_TYPES.find((a) => a.type === ap.type)
  const unit = fieldUnit(system.fields.kind)

  const setApertureType = async (type: string) => {
    if (type === ap.type) return
    const before = system
    const conv0 = await convertSpec(system, { aperture: type })
    edit('Change Aperture Type', (s) => {
      const conv = unchanged(s, before) ? conv0 : null
      const a = conv?.aperture && typeof conv.aperture === 'object' ? (conv.aperture as JsonObject) : null
      const stopSD = conv?.stop_semi_diameter
      if (type === 'float_by_stop_size') {
        s.aperture = { type }
        // "Float by stop size" fixes an automatic stop semi-diameter at the beam's current size.
        const stop = s.surfaces[s.stop]
        if (stop && stop.semi_diameter.mode === 'auto' && !stop.aperture && typeof stopSD === 'number' && stopSD > 0 && Number.isFinite(stopSD)) {
          stop.semi_diameter = { mode: 'fixed', value: stopSD }
        }
        return
      }
      if (a && a.type === type && typeof a.value === 'number') {
        s.aperture = { type, value: a.value }
        return
      }
      const v = s.aperture.value ?? (type === 'object_space_na' ? 0.1 : type === 'entrance_pupil_diameter' ? 10 : 4)
      s.aperture = { type, value: v }
    })
  }

  const setFieldKind = async (kind: string) => {
    if (kind === system.fields.kind) return
    const before = system
    const conv = await convertSpec(system, { field_kind: kind })
    edit('Change Field Type', (s) => {
      const values = unchanged(s, before) ? (conv?.fields as JsonObject | null | undefined)?.values : null
      s.fields.kind = kind
      // Each field keeps its chief ray; weights and vignetting stay.
      if (Array.isArray(values) && values.length === s.fields.fields.length) {
        values.forEach((v, k) => {
          const [x, y] = v as number[]
          if (typeof x === 'number' && typeof y === 'number') Object.assign(s.fields.fields[k], { x, y })
        })
      }
    })
  }

  return (
    <div data-testid="system-page">
      <Section>
        <FormRow label="Name">
          <TextInput ariaLabel="Name" value={system.name} onCommit={(v) => edit('Rename', (s) => void (s.name = v))} />
        </FormRow>
        <FormRow label="Notes">
          <TextInput ariaLabel="Notes" value={system.notes ?? ''} placeholder="none" onCommit={(v) => edit('Change Notes', (s) => void (s.notes = v))} />
        </FormRow>
      </Section>

      <Section title="Aperture">
        <FormRow label="Type">
          <Select aria-label="Aperture type" className="w-full" value={ap.type} onChange={(e) => void setApertureType(e.target.value)}>
            {APERTURE_TYPES.map((a) => (
              <option key={a.type} value={a.type}>
                {a.title}
              </option>
            ))}
          </Select>
        </FormRow>
        {ap.type !== 'float_by_stop_size' && (
          <NumberRow
            label="Value"
            unit={apType?.unit}
            value={ap.value ?? null}
            param={{ param: 'aperture_value' }}
            min={0}
            onCommit={(v) => {
              if (v == null || !(v > 0)) return useView.getState().toast('The aperture value must be positive.', 'warning')
              edit('Change Aperture', (s) => void (s.aperture = { type: s.aperture.type, value: v }))
            }}
          />
        )}
      </Section>

      <AnamorphicSection />

      <Section
        title="Fields"
        accessory={
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              edit('Add Field', (s) => {
                const last = s.fields.fields.at(-1) ?? { x: 0, y: 0, weight: 1 }
                s.fields.fields.push({ x: last.x, y: last.y * 1.25 + (last.y === 0 ? 5 : 0), weight: 1, vignetting: { vdx: 0, vdy: 0, vcx: 0, vcy: 0 } })
              })
            }
          >
            <Plus /> Add Field
          </Button>
        }
      >
        <FormRow label="Type">
          <Select aria-label="Field type" className="w-full" value={system.fields.kind} onChange={(e) => void setFieldKind(e.target.value)}>
            {FIELD_KINDS.map((k) => (
              <option key={k.kind} value={k.kind}>
                {k.title}
              </option>
            ))}
          </Select>
        </FormRow>
        <div className="mt-1 grid grid-cols-[2.2rem_1fr_1fr_1fr_1.2rem] items-center gap-x-1.5 gap-y-1 text-[11px] text-secondary">
          <span />
          <span className="text-right">X ({unit})</span>
          <span className="text-right">Y ({unit})</span>
          <span className="text-right">Weight</span>
          <span />
          {system.fields.fields.map((f, k) => (
            <div key={k} className="contents text-[13px] text-label">
              <span className="flex items-center gap-1" aria-label={`Field ${k + 1}`}>
                <span className="size-[7px] rounded-full" style={{ background: seriesColor(k, dark) }} />
                <span className="num">{k + 1}</span>
              </span>
              <NumberInput ariaLabel={`Field ${k + 1} x`} value={f.x} onCommit={(v) => v != null && edit('Change Field', (s) => void (s.fields.fields[k].x = v))} />
              <NumberInput ariaLabel={`Field ${k + 1} y`} value={f.y} onCommit={(v) => v != null && edit('Change Field', (s) => void (s.fields.fields[k].y = v))} />
              <NumberInput ariaLabel={`Field ${k + 1} weight`} value={f.weight} min={0} onCommit={(v) => v != null && edit('Change Field Weight', (s) => void (s.fields.fields[k].weight = v))} />
              <button
                type="button"
                aria-label="Remove Field"
                title="Remove Field"
                disabled={system.fields.fields.length <= 1}
                className="text-secondary hover:text-label disabled:opacity-30"
                onClick={() => edit('Remove Field', (s) => void s.fields.fields.splice(k, 1))}
              >
                <MinusCircle className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
      </Section>

      <Section
        title="Wavelengths"
        accessory={
          <div className="flex items-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => edit('Add Wavelength', (s) => void s.wavelengths.wavelengths.push({ um: Math.round(((s.wavelengths.wavelengths.at(-1)?.um ?? 0.55) + 0.05) * 1e6) / 1e6, weight: 1 }))}
            >
              <Plus /> Add
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm">
                  Presets
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {WAVELENGTH_PRESETS.map((p) => (
                  <DropdownMenuItem
                    key={p.name}
                    onSelect={() =>
                      edit('Apply Wavelength Preset', (s) => {
                        s.wavelengths = { wavelengths: p.list.map(([um, weight]) => ({ um, weight })), primary: p.primary }
                      })
                    }
                  >
                    {p.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        }
      >
        <div className="grid grid-cols-[3.4rem_1fr_1fr_1.2rem] items-center gap-x-1.5 gap-y-1 text-[11px] text-secondary">
          <span>Primary</span>
          <span className="text-right">λ (µm)</span>
          <span className="text-right">Weight</span>
          <span />
          {system.wavelengths.wavelengths.map((w, k) => (
            <div key={k} className="contents text-[13px] text-label">
              <button
                type="button"
                aria-label={system.wavelengths.primary === k ? 'Primary wavelength' : 'Make primary'}
                className="flex justify-center"
                style={{ color: wavelengthColor(w.um, dark) }}
                onClick={() => edit('Change Primary Wavelength', (s) => void (s.wavelengths.primary = k))}
              >
                {system.wavelengths.primary === k ? <CircleDot className="size-4" /> : <Circle className="size-4" />}
              </button>
              <NumberInput
                ariaLabel={`Wavelength ${k + 1}`}
                value={w.um}
                min={1e-6}
                onCommit={(v) => v != null && v > 0 && edit('Change Wavelength', (s) => void (s.wavelengths.wavelengths[k].um = v))}
              />
              <NumberInput
                ariaLabel={`Wavelength ${k + 1} weight`}
                value={w.weight}
                min={0}
                onCommit={(v) => v != null && edit('Change Wavelength Weight', (s) => void (s.wavelengths.wavelengths[k].weight = v))}
              />
              <button
                type="button"
                aria-label="Remove Wavelength"
                title="Remove Wavelength"
                disabled={system.wavelengths.wavelengths.length <= 1}
                className="text-secondary hover:text-label disabled:opacity-30"
                onClick={() =>
                  edit('Remove Wavelength', (s) => {
                    s.wavelengths.wavelengths.splice(k, 1)
                    const p = s.wavelengths.primary
                    s.wavelengths.primary = Math.min(p > k ? p - 1 : p, s.wavelengths.wavelengths.length - 1)
                  })
                }
              >
                <MinusCircle className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Environment">
        <NumberRow
          label="Temperature"
          unit="°C"
          value={Number((system.environment as JsonObject)?.temperature_c ?? 20)}
          onCommit={(v) => v != null && edit('Change Temperature', (s) => void ((s.environment as JsonObject).temperature_c = v))}
        />
        <NumberRow
          label="Pressure"
          unit="atm"
          min={0}
          value={Number((system.environment as JsonObject)?.pressure_atm ?? 1)}
          onCommit={(v) => v != null && v > 0 && edit('Change Pressure', (s) => void ((s.environment as JsonObject).pressure_atm = v))}
        />
      </Section>

      <Section title="Ray Aiming" footer="Aims every ray through the real stop instead of the paraxial entrance pupil. Slower, needed for wide fields and pupil aberration.">
        <FormRow label="Real ray aiming">
          <Switch
            aria-label="Real ray aiming"
            checked={(system.ray_aiming as JsonObject)?.mode === 'real'}
            onCheckedChange={(on) => edit('Change Ray Aiming', (s) => void (s.ray_aiming = on ? { mode: 'real', tolerance: 1e-10 } : { mode: 'off' }))}
          />
        </FormRow>
      </Section>
      {Array.isArray(system.configurations) && system.configurations.length > 0 && (
        <Section title="Configurations" testId="configurations">
          {(system.configurations as JsonObject[]).map((c, k) => {
            const n = ((c.overrides as unknown[]) ?? []).length
            return (
              <FormRow key={k} label={String(c.name || `Configuration ${k + 1}`)}>
                <span className="text-secondary">
                  {n} override{n === 1 ? '' : 's'}
                </span>
              </FormRow>
            )
          })}
        </Section>
      )}
      <CatalogsSection />
    </div>
  )
}

/** The x / y first order of a design that is not rotationally symmetric
 *  (the engine's `paraxial_xy` at the primary wavelength). */
function AnamorphicSection() {
  const system = useDocument((s) => s.project.system)
  const run = useWorkspace((s) => s.runs[SUMMARY])
  const a = anamorphicSummary(run?.result?.json, system)
  if (!a.anamorphic) return null
  const p = a.primary
  const mm = (v: number) => `${sig(v, 6)} mm`
  return (
    <Section title="First Order" testId="anamorphic-first-order">
      <div className={run?.status === 'running' ? 'opacity-50' : undefined}>
        {p ? (
          <>
            <FormRow label="EFL x / y">
              <span className="num">
                {sig(p.eflX, 6)} / {mm(p.eflY)}
              </span>
            </FormRow>
            <FormRow label="BFL x / y">
              <span className="num">
                {sig(p.bflX, 6)} / {mm(p.bflY)}
              </span>
            </FormRow>
            <FormRow label="Squeeze" help={`Squeeze: ${a.squeezeDefinition ?? 'signed EFL y / EFL x'}`}>
              <span className="num">{squeezeText(p.squeeze)}</span>
            </FormRow>
            {a.symmetry && (
              <FormRow label="Symmetry">
                <span>{a.symmetry === 'meridional' ? 'Meridional plane' : a.symmetry === 'none' ? 'None' : 'Rotational'}</span>
              </FormRow>
            )}
          </>
        ) : (
          <p className="py-1 text-[12px] text-secondary">Not rotationally symmetric. The engine’s first-order data covers the y section.</p>
        )}
      </div>
    </Section>
  )
}

/** The session's glass catalogs (built-in first) and the project's catalog
 *  references, with Load Glass Catalog… and Remove. */
function CatalogsSection() {
  const engine = useWorkspace((s) => s.engine)
  const list = useCatalogs((s) => s.list)
  const refresh = useCatalogs((s) => s.refresh)
  const refs = catalogReferences(useDocument((s) => s.project.catalogs))
  useEffect(() => {
    void refresh()
  }, [engine, refresh])
  const loaded = new Set(list.map((c) => c.name.toUpperCase()))
  return (
    <Section
      title="Glass Catalogs"
      testId="glass-catalogs"
      accessory={
        <Button variant="ghost" size="sm" disabled={!engine} onClick={() => void loadGlassCatalog()}>
          <Plus /> Load Glass Catalog…
        </Button>
      }
      footer="AGF or GlassCatalog JSON files. The project records the catalog by name; the browser cannot reopen it from its path, so a project's catalogs are loaded again by hand."
    >
      {list.length === 0 && refs.length === 0 && <p className="py-1 text-[12px] text-secondary">No catalogs.</p>}
      {list.map((c) => {
        const ref = refs.find((r) => r.name.toUpperCase() === c.name.toUpperCase())
        return (
          <FormRow key={c.name} label={c.name} help={c.source === 'builtin' ? 'Built into the engine' : ref?.path ?? undefined}>
            <span className="flex items-center gap-1.5 text-secondary">
              <span className="num">{c.glass_count} glasses</span>
              {ref && (
                <button type="button" aria-label={`Remove ${c.name}`} title="Remove from the project" className="hover:text-label" onClick={() => removeGlassCatalog(ref.name)}>
                  <MinusCircle className="size-3.5" />
                </button>
              )}
            </span>
          </FormRow>
        )
      })}
      {refs
        .filter((r) => !loaded.has(r.name.toUpperCase()))
        .map((r) => (
          <FormRow key={r.name} label={r.name} help={r.path ?? undefined}>
            <span className="flex items-center gap-1.5">
              <span className="flex items-center gap-1 text-[12px] text-warning">
                <AlertTriangle className="size-3.5" /> Not loaded
              </span>
              <Button variant="ghost" size="sm" onClick={() => void loadGlassCatalog()}>
                Locate…
              </Button>
              <button type="button" aria-label={`Remove ${r.name}`} title="Remove from the project" className="text-secondary hover:text-label" onClick={() => removeGlassCatalog(r.name)}>
                <MinusCircle className="size-3.5" />
              </button>
            </span>
          </FormRow>
        ))}
    </Section>
  )
}
