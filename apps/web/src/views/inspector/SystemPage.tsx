import { AlertTriangle, CircleDot, Circle, MinusCircle, Plus } from 'lucide-react'
import { useEffect } from 'react'
import type { JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Select } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { tMaybe, useT } from '@/i18n'
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

const PRESET_KEYS: Record<string, string> = {
  'F, d, C (visible)': 'fdc',
  'F′, e, C′ (visible)': 'fec',
  'g, F, e, d, C': 'gFedC',
  'd only': 'dOnly',
  'e only': 'eOnly',
  'Photopic (5)': 'photopic',
  'Near infrared': 'nearInfrared',
}

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
  const t = useT()
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
    edit(t('inspector.system.undoApertureType'), (s) => {
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
    edit(t('inspector.system.undoFieldType'), (s) => {
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
        <FormRow label={t('inspector.system.name')}>
          <TextInput ariaLabel={t('inspector.system.name')} value={system.name} onCommit={(v) => edit(t('inspector.system.undoRename'), (s) => void (s.name = v))} />
        </FormRow>
        <FormRow label={t('inspector.system.notes')}>
          <TextInput ariaLabel={t('inspector.system.notes')} value={system.notes ?? ''} placeholder={t('inspector.system.none')} onCommit={(v) => edit(t('inspector.system.undoNotes'), (s) => void (s.notes = v))} />
        </FormRow>
      </Section>

      <Section title={t('inspector.aperture.title')}>
        <FormRow label={t('inspector.surface.type')}>
          <Select aria-label={t('inspector.system.apertureType')} className="w-full" value={ap.type} onChange={(e) => void setApertureType(e.target.value)}>
            {APERTURE_TYPES.map((a) => (
              <option key={a.type} value={a.type}>
                {tMaybe(`inspector.system.apertureTypes.${a.type}`) ?? a.title}
              </option>
            ))}
          </Select>
        </FormRow>
        {ap.type !== 'float_by_stop_size' && (
          <NumberRow
            label={t('inspector.aperture.value')}
            unit={apType?.unit}
            value={ap.value ?? null}
            param={{ param: 'aperture_value' }}
            min={0}
            onCommit={(v) => {
              if (v == null || !(v > 0)) return useView.getState().toast(t('inspector.system.apertureValuePositive'), 'warning')
              edit(t('inspector.aperture.undo'), (s) => void (s.aperture = { type: s.aperture.type, value: v }))
            }}
          />
        )}
      </Section>

      <AnamorphicSection />

      <Section
        title={t('inspector.system.fields')}
        accessory={
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              edit(t('inspector.system.addField'), (s) => {
                const last = s.fields.fields.at(-1) ?? { x: 0, y: 0, weight: 1 }
                s.fields.fields.push({ x: last.x, y: last.y * 1.25 + (last.y === 0 ? 5 : 0), weight: 1, vignetting: { vdx: 0, vdy: 0, vcx: 0, vcy: 0 } })
              })
            }
          >
            <Plus /> {t('inspector.system.addField')}
          </Button>
        }
      >
        <FormRow label={t('inspector.surface.type')}>
          <Select aria-label={t('inspector.system.fieldType')} className="w-full" value={system.fields.kind} onChange={(e) => void setFieldKind(e.target.value)}>
            {FIELD_KINDS.map((k) => (
              <option key={k.kind} value={k.kind}>
                {tMaybe(`inspector.system.fieldKinds.${k.kind}`) ?? k.title}
              </option>
            ))}
          </Select>
        </FormRow>
        <div className="mt-1 grid grid-cols-[2.2rem_1fr_1fr_1fr_1.2rem] items-center gap-x-1.5 gap-y-1 text-[11px] text-secondary">
          <span />
          <span className="text-right">X ({unit})</span>
          <span className="text-right">Y ({unit})</span>
          <span className="text-right">{t('inspector.system.weight')}</span>
          <span />
          {system.fields.fields.map((f, k) => (
            <div key={k} className="contents text-[13px] text-label">
              <span className="flex items-center gap-1" aria-label={t('common.fieldN', { n: k + 1 })}>
                <span className="size-[7px] rounded-full" style={{ background: seriesColor(k, dark) }} />
                <span className="num">{k + 1}</span>
              </span>
              <NumberInput ariaLabel={t('inspector.system.fieldX', { n: k + 1 })} value={f.x} onCommit={(v) => v != null && edit(t('inspector.system.undoField'), (s) => void (s.fields.fields[k].x = v))} />
              <NumberInput ariaLabel={t('inspector.system.fieldY', { n: k + 1 })} value={f.y} onCommit={(v) => v != null && edit(t('inspector.system.undoField'), (s) => void (s.fields.fields[k].y = v))} />
              <NumberInput ariaLabel={t('inspector.system.fieldWeight', { n: k + 1 })} value={f.weight} min={0} onCommit={(v) => v != null && edit(t('inspector.system.undoFieldWeight'), (s) => void (s.fields.fields[k].weight = v))} />
              <button
                type="button"
                aria-label={t('inspector.system.removeField')}
                title={t('inspector.system.removeField')}
                disabled={system.fields.fields.length <= 1}
                className="text-secondary hover:text-label disabled:opacity-30"
                onClick={() => edit(t('inspector.system.removeField'), (s) => void s.fields.fields.splice(k, 1))}
              >
                <MinusCircle className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
      </Section>

      <Section
        title={t('inspector.system.wavelengths')}
        accessory={
          <div className="flex items-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => edit(t('inspector.system.addWavelength'), (s) => void s.wavelengths.wavelengths.push({ um: Math.round(((s.wavelengths.wavelengths.at(-1)?.um ?? 0.55) + 0.05) * 1e6) / 1e6, weight: 1 }))}
            >
              <Plus /> {t('common.add')}
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm">
                  {t('inspector.system.presets')}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {WAVELENGTH_PRESETS.map((p) => (
                  <DropdownMenuItem
                    key={p.name}
                    onSelect={() =>
                      edit(t('inspector.system.undoPreset'), (s) => {
                        s.wavelengths = { wavelengths: p.list.map(([um, weight]) => ({ um, weight })), primary: p.primary }
                      })
                    }
                  >
                    {PRESET_KEYS[p.name] ? tMaybe(`inspector.system.presetNames.${PRESET_KEYS[p.name]}`) : p.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        }
      >
        <div className="grid grid-cols-[3.4rem_1fr_1fr_1.2rem] items-center gap-x-1.5 gap-y-1 text-[11px] text-secondary">
          <span>{t('inspector.system.primary')}</span>
          <span className="text-right">λ (µm)</span>
          <span className="text-right">{t('inspector.system.weight')}</span>
          <span />
          {system.wavelengths.wavelengths.map((w, k) => (
            <div key={k} className="contents text-[13px] text-label">
              <button
                type="button"
                aria-label={system.wavelengths.primary === k ? t('inspector.system.primaryWavelength') : t('inspector.system.makePrimary')}
                className="flex justify-center"
                style={{ color: wavelengthColor(w.um, dark) }}
                onClick={() => edit(t('inspector.system.undoPrimary'), (s) => void (s.wavelengths.primary = k))}
              >
                {system.wavelengths.primary === k ? <CircleDot className="size-4" /> : <Circle className="size-4" />}
              </button>
              <NumberInput
                ariaLabel={t('common.wavelengthN', { n: k + 1 })}
                value={w.um}
                min={1e-6}
                onCommit={(v) => v != null && v > 0 && edit(t('inspector.system.undoWavelength'), (s) => void (s.wavelengths.wavelengths[k].um = v))}
              />
              <NumberInput
                ariaLabel={t('inspector.system.wavelengthWeight', { n: k + 1 })}
                value={w.weight}
                min={0}
                onCommit={(v) => v != null && edit(t('inspector.system.undoWavelengthWeight'), (s) => void (s.wavelengths.wavelengths[k].weight = v))}
              />
              <button
                type="button"
                aria-label={t('inspector.system.removeWavelength')}
                title={t('inspector.system.removeWavelength')}
                disabled={system.wavelengths.wavelengths.length <= 1}
                className="text-secondary hover:text-label disabled:opacity-30"
                onClick={() =>
                  edit(t('inspector.system.removeWavelength'), (s) => {
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

      <Section title={t('inspector.system.environment')}>
        <NumberRow
          label={t('inspector.system.temperature')}
          unit="°C"
          value={Number((system.environment as JsonObject)?.temperature_c ?? 20)}
          onCommit={(v) => v != null && edit(t('inspector.system.undoTemperature'), (s) => void ((s.environment as JsonObject).temperature_c = v))}
        />
        <NumberRow
          label={t('inspector.system.pressure')}
          unit="atm"
          min={0}
          value={Number((system.environment as JsonObject)?.pressure_atm ?? 1)}
          onCommit={(v) => v != null && v > 0 && edit(t('inspector.system.undoPressure'), (s) => void ((s.environment as JsonObject).pressure_atm = v))}
        />
      </Section>

      <Section title={t('inspector.system.rayAiming')} footer={t('inspector.system.rayAimingFooter')}>
        <FormRow label={t('inspector.system.realRayAiming')}>
          <Switch
            aria-label={t('inspector.system.realRayAiming')}
            checked={(system.ray_aiming as JsonObject)?.mode === 'real'}
            onCheckedChange={(on) => edit(t('inspector.system.undoRayAiming'), (s) => void (s.ray_aiming = on ? { mode: 'real', tolerance: 1e-10 } : { mode: 'off' }))}
          />
        </FormRow>
      </Section>
      {Array.isArray(system.configurations) && system.configurations.length > 0 && (
        <Section title={t('inspector.system.configurations')} testId="configurations">
          {(system.configurations as JsonObject[]).map((c, k) => {
            const n = ((c.overrides as unknown[]) ?? []).length
            return (
              <FormRow key={k} label={String(c.name || t('inspector.system.configurationN', { n: k + 1 }))}>
                <span className="text-secondary">
                  {t('inspector.system.overrides', { count: n })}
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
  const t = useT()
  const system = useDocument((s) => s.project.system)
  const run = useWorkspace((s) => s.runs[SUMMARY])
  const a = anamorphicSummary(run?.result?.json, system)
  if (!a.anamorphic) return null
  const p = a.primary
  const mm = (v: number) => `${sig(v, 6)} mm`
  return (
    <Section title={t('inspector.system.firstOrder')} testId="anamorphic-first-order">
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
            <FormRow label={t('inspector.system.squeeze')} help={t('inspector.system.squeezeHelp', { definition: a.squeezeDefinition ?? t('inspector.system.squeezeDefinition') })}>
              <span className="num">{squeezeText(p.squeeze)}</span>
            </FormRow>
            {a.symmetry && (
              <FormRow label={t('inspector.system.symmetry')}>
                <span>{a.symmetry === 'meridional' ? t('inspector.system.meridional') : a.symmetry === 'none' ? t('common.none') : t('inspector.system.rotational')}</span>
              </FormRow>
            )}
          </>
        ) : (
          <p className="py-1 text-[12px] text-secondary">{t('inspector.system.notSymmetric')}</p>
        )}
      </div>
    </Section>
  )
}

/** The session's glass catalogs (built-in first) and the project's catalog
 *  references, with Load Glass Catalog… and Remove. */
function CatalogsSection() {
  const t = useT()
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
      title={t('inspector.catalogs.title')}
      testId="glass-catalogs"
      accessory={
        <Button variant="ghost" size="sm" disabled={!engine} onClick={() => void loadGlassCatalog()}>
          <Plus /> {t('inspector.catalogs.load')}
        </Button>
      }
      footer={t('inspector.catalogs.footer')}
    >
      {list.length === 0 && refs.length === 0 && <p className="py-1 text-[12px] text-secondary">{t('inspector.catalogs.none')}</p>}
      {list.map((c) => {
        const ref = refs.find((r) => r.name.toUpperCase() === c.name.toUpperCase())
        return (
          <FormRow key={c.name} label={c.name} help={c.source === 'builtin' ? t('inspector.catalogs.builtin') : ref?.path ?? undefined}>
            <span className="flex items-center gap-1.5 text-secondary">
              <span className="num">{t('inspector.catalogs.glasses', { count: c.glass_count })}</span>
              {ref && (
                <button type="button" aria-label={t('inspector.catalogs.remove', { name: c.name })} title={t('inspector.catalogs.removeHelp')} className="hover:text-label" onClick={() => removeGlassCatalog(ref.name)}>
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
                <AlertTriangle className="size-3.5" /> {t('inspector.catalogs.notLoaded')}
              </span>
              <Button variant="ghost" size="sm" onClick={() => void loadGlassCatalog()}>
                {t('inspector.catalogs.locate')}
              </Button>
              <button type="button" aria-label={t('inspector.catalogs.remove', { name: r.name })} title={t('inspector.catalogs.removeHelp')} className="text-secondary hover:text-label" onClick={() => removeGlassCatalog(r.name)}>
                <MinusCircle className="size-3.5" />
              </button>
            </span>
          </FormRow>
        ))}
    </Section>
  )
}
