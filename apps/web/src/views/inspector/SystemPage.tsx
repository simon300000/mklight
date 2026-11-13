import { CircleDot, Circle, MinusCircle, Plus } from 'lucide-react'
import type { JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { APERTURE_TYPES, FIELD_KINDS, WAVELENGTH_PRESETS, fieldUnit, type System } from '@/model/system'
import { useDocument } from '@/state/document'
import { useView } from '@/state/view'
import { useWorkspace } from '@/state/workspace'
import { seriesColor, useDark, wavelengthColor } from '@/lib/theme'
import { NumberInput } from '../SettingsForm'
import { FormRow, NumberRow, Section, TextInput } from './Form'

/** Converts field or aperture values with the engine's `convert_spec`, so the
 *  system stays the same; without it the numbers stay (validation says why). */
async function convertSpec(system: System, change: JsonObject): Promise<JsonObject | null> {
  const ws = useWorkspace.getState()
  if (!ws.engine || !ws.snapshot) return null
  try {
    const job = ws.engine.submit({ analysis: 'convert_spec', settings: change }, ws.snapshot)
    const r = await job.result
    return r.json
  } catch {
    void system
    return null
  }
}

/** System settings (`SystemInspector`): name and notes, aperture, fields,
 *  wavelengths (with presets), environment and ray aiming. */
export function SystemPage() {
  const system = useDocument((s) => s.project.system)
  const edit = useDocument((s) => s.edit)
  const dark = useDark()
  const catalogs = useWorkspace((s) => s.descriptors.length)
  if (!system?.surfaces) return null
  const ap = system.aperture
  const apType = APERTURE_TYPES.find((a) => a.type === ap.type)
  const unit = fieldUnit(system.fields.kind)

  const setApertureType = async (type: string) => {
    const conv = await convertSpec(system, { aperture: type })
    const value = conv?.aperture && typeof conv.aperture === 'object' ? (conv.aperture as JsonObject).value : undefined
    const stopSD = conv?.stop_semi_diameter
    edit('Change Aperture Type', (s) => {
      s.aperture = type === 'float_by_stop_size' ? { type } : { type, value: typeof value === 'number' ? value : (s.aperture.value ?? 10) }
      // "Float by stop size" fixes an automatic stop semi-diameter at the beam's current size.
      const stop = s.surfaces[s.stop]
      if (type === 'float_by_stop_size' && stop.semi_diameter.mode === 'auto' && typeof stopSD === 'number') {
        stop.semi_diameter = { mode: 'fixed', value: stopSD }
      }
    })
  }

  const setFieldKind = async (kind: string) => {
    const conv = await convertSpec(system, { field_kind: kind })
    const values = (conv?.fields as JsonObject | null | undefined)?.values
    edit('Change Field Type', (s) => {
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
            min={0}
            onCommit={(v) => {
              if (v == null || !(v > 0)) return useView.getState().toast('The aperture value must be positive.', 'warning')
              edit('Change Aperture', (s) => void (s.aperture = { type: s.aperture.type, value: v }))
            }}
          />
        )}
      </Section>

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
      {catalogs > 0 && Array.isArray(system.configurations) && system.configurations.length > 0 && (
        <Section title="Configurations">
          {(system.configurations as JsonObject[]).map((c, k) => (
            <FormRow key={k} label={String(c.name || `Configuration ${k + 1}`)}>
              <span className="text-secondary">{((c.overrides as unknown[]) ?? []).length} overrides</span>
            </FormRow>
          ))}
        </Section>
      )}
    </div>
  )
}
