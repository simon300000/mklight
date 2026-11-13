// The run-local spectrum control (`SpectrumControl`): a compact "Spectrum: …"
// button in the settings form that opens the editor — design wavelengths,
// photopic, scotopic, blackbody (quick picks 5500 K, 3200 K), explicit
// wavelengths seeded from the design, or a sensor response imported from CSV —
// with the engine's sampling and a small chart of the weights the run will
// use, resolved by the engine (`system_summary` with the spectrum) or,
// until it answers, a local preview labelled approximate.

import { Minus, Plus } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Json, JsonObject } from '@/engine/types'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select } from '@/components/ui/select'
import { wavelengthColor, useDark } from '@/lib/theme'
import { sig } from '@/model/format'
import type { System } from '@/model/system'
import {
  BLACKBODY_PRESETS,
  PHOTOPIC_RANGE,
  SCOTOPIC_RANGE,
  SPACING_TITLES,
  SPECTRUM_KINDS,
  parseSpectrum,
  parseSpectrumCsv,
  previewCurve,
  previewSamples,
  primaryIndex,
  resolvedSamples,
  samplingOf,
  spectrumIssues,
  spectrumJson,
  spectrumOfKind,
  spectrumSummary,
  withSampling,
  type BadRow,
  type Sample,
  type Spacing,
  type Spectrum,
  type SpectrumKind,
} from '@/model/spectrum'
import { activeSnapshot, stableJson, useWorkspace } from '@/state/workspace'
import { NumberInput } from '../SettingsForm'

const designSamples = (system: System): Sample[] => (system.wavelengths?.wavelengths ?? []).map((w) => ({ um: w.um, weight: w.weight }))

/** The compact button and its popover editor. `value` is the setting
 *  (absent/null: the design); `onChange(null)` removes it. */
export function SpectrumButton({
  value,
  system,
  onChange,
  compact,
}: {
  value: Json | undefined
  system: System
  onChange: (v: JsonObject | null) => void
  /** The pinned bar's form: a small "Spectrum: …" capsule. */
  compact?: boolean
}) {
  const s = parseSpectrum(value)
  const kindTitle = s ? SPECTRUM_KINDS.find((k) => k.kind === s.kind)?.title : 'Unreadable'
  const text = s ? (s.kind === 'design' ? 'Design' : spectrumSummary(s)) : 'Unreadable'
  return (
    <Popover>
      <PopoverTrigger asChild>
        {compact ? (
          <button
            type="button"
            className="flex h-[22px] max-w-full min-w-0 cursor-default items-center gap-1 rounded-full bg-selected px-2 text-[11px] hover:bg-hover focus-visible:outline-2"
            aria-label={`Spectrum: ${s ? spectrumSummary(s) : 'unreadable'}`}
            title={`The wavelengths this analysis runs with: ${kindTitle}`}
            data-testid="pinned-spectrum"
          >
            <span className="shrink-0 text-secondary">Spectrum:</span>
            <span className="truncate font-medium">{text}</span>
          </button>
        ) : (
          <button
            type="button"
            className="flex h-6 w-full min-w-0 cursor-default items-center justify-end gap-1 truncate rounded-md border border-control-border bg-control px-2 text-[13px] hover:bg-hover"
            aria-label={`Spectrum: ${s ? spectrumSummary(s) : 'unreadable'}`}
            title={kindTitle}
          >
            <span className="truncate">{text}</span>
          </button>
        )}
      </PopoverTrigger>
      <PopoverContent className="w-[340px] max-w-[calc(100vw-16px)]" align={compact ? 'start' : 'end'}>
        <SpectrumEditor value={value} system={system} onChange={onChange} />
      </PopoverContent>
    </Popover>
  )
}

export function SpectrumEditor({ value, system, onChange }: { value: Json | undefined; system: System; onChange: (v: JsonObject | null) => void }) {
  const parsed = parseSpectrum(value)
  const s: Spectrum = parsed ?? { kind: 'design' }
  const set = (n: Spectrum) => onChange(spectrumJson(n))
  const issues = spectrumIssues(s)
  const design = designSamples(system)
  const sm = samplingOf(s)
  const [csv, setCsv] = useState<{ name: string; bad: BadRow[]; unit: string } | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <div className="flex flex-col gap-2 text-[12px]" data-testid="spectrum-editor">
      <h3 className="text-[13px] font-semibold">Spectrum</h3>
      {!parsed && <p className="text-danger">The current spectrum setting is not readable; choosing a kind replaces it.</p>}
      <Select aria-label="Spectrum kind" value={s.kind} onChange={(e) => set(spectrumOfKind(e.target.value as SpectrumKind, s, design))}>
        {SPECTRUM_KINDS.map((k) => (
          <option key={k.kind} value={k.kind}>
            {k.title}
          </option>
        ))}
      </Select>

      {s.kind === 'blackbody' && (
        <div className="grid grid-cols-[1fr_6rem_auto] items-center gap-2">
          <span className="text-secondary">Temperature</span>
          <NumberInput ariaLabel="Temperature" value={s.temperatureK} min={1} onCommit={(v) => v != null && set({ ...s, temperatureK: v })} />
          <span className="text-secondary">K</span>
          <span />
          <span className="col-span-2 flex gap-1">
            {BLACKBODY_PRESETS.map((t) => (
              <Button key={t} size="sm" variant={s.temperatureK === t ? 'prominent' : 'default'} onClick={() => set({ ...s, temperatureK: t })}>
                {t} K
              </Button>
            ))}
          </span>
        </div>
      )}

      {s.kind === 'explicit' && (
        <div className="max-h-48 overflow-y-auto rounded-md border border-separator">
          <table className="w-full text-[12px]">
            <thead className="text-secondary">
              <tr>
                <th className="px-2 py-1 text-left font-normal">λ (µm)</th>
                <th className="px-2 py-1 text-left font-normal">Weight</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {s.wavelengths.map((w, k) => (
                <tr key={k}>
                  <td className="px-1 py-[2px]">
                    <NumberInput ariaLabel={`Wavelength ${k + 1}`} value={w.um} min={0} onCommit={(v) => v != null && set({ ...s, wavelengths: s.wavelengths.map((x, i) => (i === k ? { ...x, um: v } : x)) })} />
                  </td>
                  <td className="px-1 py-[2px]">
                    <NumberInput ariaLabel={`Weight ${k + 1}`} value={w.weight} min={0} onCommit={(v) => v != null && set({ ...s, wavelengths: s.wavelengths.map((x, i) => (i === k ? { ...x, weight: v } : x)) })} />
                  </td>
                  <td className="w-6">
                    <Button variant="ghost" size="iconSm" aria-label={`Remove wavelength ${k + 1}`} onClick={() => set({ ...s, wavelengths: s.wavelengths.filter((_, i) => i !== k) })}>
                      <Minus />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex gap-1 border-t border-separator p-1">
            <Button size="sm" variant="ghost" onClick={() => set({ ...s, wavelengths: [...s.wavelengths, { um: s.wavelengths.at(-1)?.um ?? 0.55, weight: 1 }] })}>
              <Plus /> Add
            </Button>
            <Button size="sm" variant="ghost" onClick={() => set({ ...s, wavelengths: design })}>
              From Design
            </Button>
          </div>
        </div>
      )}

      {s.kind === 'sensor' && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={() => fileRef.current?.click()}>
              Import CSV…
            </Button>
            <span className="text-secondary">{s.table.length ? `${s.table.length} rows${csv ? ` from ${csv.name} (${csv.unit})` : ''}` : 'No response table'}</span>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,.txt,.tsv,text/csv,text/plain"
              className="hidden"
              aria-label="Sensor response CSV file"
              onChange={async (e) => {
                const f = e.target.files?.[0]
                e.target.value = ''
                if (!f) return
                const r = parseSpectrumCsv(await f.text())
                setCsv({ name: f.name, bad: r.badRows, unit: r.unit })
                if (r.table.length >= 2) {
                  const lo = r.table[0].um, hi = r.table[r.table.length - 1].um
                  set({ ...s, table: r.table, sampling: { ...s.sampling, lowerUm: lo, upperUm: hi } })
                }
              }}
            />
          </div>
          {csv && csv.bad.length > 0 && (
            <ul className="max-h-24 overflow-y-auto text-[11px] text-secondary">
              {csv.bad.map((b) => (
                <li key={b.line}>
                  Line {b.line}: {b.reason} <span className="num opacity-70">({b.text})</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {sm && (
        <fieldset className="rounded-md border border-separator px-2 py-1">
          <legend className="px-1 text-[11px] text-secondary">Sampling</legend>
          <div className="grid grid-cols-[1fr_5rem_5rem] items-center gap-x-2 gap-y-1">
            <span className="text-secondary">Samples</span>
            <span className="col-span-2">
              <NumberInput ariaLabel="Samples" value={sm.samples} integer min={1} max={4096} onCommit={(v) => v != null && set(withSampling(s, { ...sm, samples: v }))} />
            </span>
            <span className="text-secondary">Range (µm)</span>
            <NumberInput ariaLabel="Range from" value={sm.lowerUm} min={0} onCommit={(v) => v != null && set(withSampling(s, { ...sm, lowerUm: v }))} />
            <NumberInput ariaLabel="Range to" value={sm.upperUm} min={0} onCommit={(v) => v != null && set(withSampling(s, { ...sm, upperUm: v }))} />
            <span className="text-secondary">Spacing</span>
            <span className="col-span-2">
              <Select aria-label="Spacing" className="w-full" value={sm.spacing} onChange={(e) => set(withSampling(s, { ...sm, spacing: e.target.value as Spacing }))}>
                {(Object.keys(SPACING_TITLES) as Spacing[]).map((k) => (
                  <option key={k} value={k}>
                    {SPACING_TITLES[k]}
                  </option>
                ))}
              </Select>
            </span>
          </div>
          {(s.kind === 'photopic' || s.kind === 'scotopic') && (
            <p className="mt-1 text-[11px] text-secondary">
              The {s.kind} table covers {(s.kind === 'photopic' ? PHOTOPIC_RANGE : SCOTOPIC_RANGE).map((x) => x.toFixed(2)).join('–')} µm.
            </p>
          )}
        </fieldset>
      )}

      {issues.length > 0 && (
        <ul className="text-[11px] text-danger">
          {issues.map((i) => (
            <li key={i}>{i}</li>
          ))}
        </ul>
      )}

      <SpectrumChart spectrum={s} system={system} valid={issues.length === 0} />
    </div>
  )
}

/** The weights the run will use, as engine-resolved or approximate samples. */
function useResolved(s: Spectrum, valid: boolean): Sample[] | null {
  const engine = useWorkspace((w) => w.engine)
  const snap = useWorkspace((w) => activeSnapshot(w))
  const json = spectrumJson(s)
  const key = json ? stableJson(json) : ''
  const [res, setRes] = useState<{ key: string; samples: Sample[] } | null>(null)
  useEffect(() => {
    if (!engine || !snap || !json || !valid) return
    let alive = true
    const t = setTimeout(() => {
      let job: ReturnType<typeof engine.submit> | null = null
      try {
        job = engine.submit({ analysis: 'system_summary', settings: { spectrum: json } }, snap)
      } catch {
        return
      }
      job.result.then(
        (r) => {
          const samples = resolvedSamples(r.json)
          if (alive && samples) setRes({ key, samples })
        },
        () => {},
      )
    }, 250)
    return () => {
      alive = false
      clearTimeout(t)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine, snap, key, valid])
  return res && res.key === key ? res.samples : null
}

export function SpectrumChart({ spectrum, system, valid }: { spectrum: Spectrum; system: System; valid: boolean }) {
  const dark = useDark()
  const engineSamples = useResolved(spectrum, valid)
  const local = useMemo(() => (spectrum.kind === 'design' ? designSamples(system) : previewSamples(spectrum)), [spectrum, system])
  const curve = useMemo(() => previewCurve(spectrum), [spectrum])
  const samples = engineSamples ?? local
  const source = spectrum.kind === 'design' ? 'Design wavelengths' : engineSamples ? 'Engine-resolved' : spectrum.kind === 'explicit' ? 'As entered' : 'Approximate preview'
  if (!samples?.length) return <p className="text-[11px] text-secondary">No weights to show.</p>
  const primary = spectrum.kind === 'design' ? system.wavelengths?.primary ?? 0 : primaryIndex(samples)
  const W = 300, H = 84, ml = 4, mr = 4, mt = 6, mb = 16
  const xs = [...samples.map((w) => w.um), ...(curve ?? []).map((c) => c.um)]
  let lo = Math.min(...xs), hi = Math.max(...xs)
  if (!(hi > lo)) [lo, hi] = [lo - 0.01, hi + 0.01]
  const pad = (hi - lo) * 0.04
  lo -= pad
  hi += pad
  const wmax = Math.max(...samples.map((w) => w.weight), 1e-300)
  const px = (um: number) => ml + ((um - lo) / (hi - lo)) * (W - ml - mr)
  const py = (v: number) => mt + (1 - v) * (H - mt - mb)
  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-[84px] w-full" role="img" aria-label={`${source}: ${samples.length} weighted wavelengths`}>
        <line x1={ml} x2={W - mr} y1={py(0)} y2={py(0)} stroke="var(--grid)" />
        {curve && (
          <polyline
            fill="none"
            stroke="var(--secondary)"
            strokeWidth={1}
            strokeDasharray="3 2"
            points={curve.map((c) => `${px(c.um).toFixed(1)},${py(c.weight).toFixed(1)}`).join(' ')}
          />
        )}
        {samples.map((w, k) => (
          <g key={k}>
            <line x1={px(w.um)} x2={px(w.um)} y1={py(0)} y2={py(w.weight / wmax)} stroke={wavelengthColor(w.um, dark)} strokeWidth={k === primary ? 3 : 2} />
            {k === primary && <circle cx={px(w.um)} cy={py(w.weight / wmax) - 4} r={2} fill="var(--label)" />}
          </g>
        ))}
        <text x={ml} y={H - 3} fontSize={10} fill="var(--secondary)">
          {sig(lo + pad, 3)} µm
        </text>
        <text x={W - mr} y={H - 3} fontSize={10} fill="var(--secondary)" textAnchor="end">
          {sig(hi - pad, 3)} µm
        </text>
      </svg>
      <figcaption className="text-[11px] text-secondary">
        {source} · {samples.length} wavelength{samples.length === 1 ? '' : 's'}
        {primary != null && samples[primary] ? ` · primary ${sig(samples[primary].um, 4)} µm` : ''}
      </figcaption>
    </figure>
  )
}
