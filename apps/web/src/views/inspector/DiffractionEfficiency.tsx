import { Minus, Plus } from 'lucide-react'
import type { Json } from '@/engine/types'
import { Segmented } from '@/components/ui/segmented'
import { fmt, sig } from '@/model/format'
import { efficiencyModel, phaseOf, traceEfficiency, type EfficiencyModel } from '@/model/shapeSag'
import type { Surface } from '@/model/system'
import { useDocument } from '@/state/document'
import { useDark, wavelengthColor } from '@/lib/theme'
import { FormRow, Section } from './Form'

const pct = (v: number) => (Number.isFinite(v) ? `${fmt(v * 100, 1)} %` : '–')

/** A diffractive surface's order efficiency model (SURFACES §3): ideal,
 *  kinoform or an N-level staircase, with the trace order's efficiency at the
 *  design wavelength and, estimated without the relief's dispersion, at the
 *  system wavelengths. The engine traces with its own model. */
export function DiffractionEfficiencySection({ surface, index }: { surface: Surface; index: number }) {
  const edit = useDocument((s) => s.edit)
  const wavelengths = useDocument((s) => s.project.system.wavelengths.wavelengths)
  const dark = useDark()
  const p = phaseOf(surface.shape)
  if (!p) return null
  const model = efficiencyModel(surface.diffraction_efficiency as Json)
  const set = (m: EfficiencyModel) => edit('Change Diffraction Efficiency', (s) => void (s.surfaces[index].diffraction_efficiency = m))
  const levels = model.type === 'multilevel' ? model.levels : 8
  return (
    <Section
      title="Diffraction Efficiency"
      testId="diffraction-efficiency"
      footer={model.type !== 'ideal' ? 'System wavelengths estimated with blaze m₀λ₀/λ, ignoring the relief material’s dispersion. The engine traces the exact model.' : undefined}
    >
      <FormRow label="Model" help="Ideal: every photon in the trace order. Kinoform: a continuous blazed relief. Multilevel: a staircase of N etch levels.">
        <Segmented
          label="Efficiency model"
          value={model.type}
          onChange={(t) => set(t === 'multilevel' ? { type: 'multilevel', levels } : { type: t })}
          options={[
            { value: 'ideal', title: 'Ideal' },
            { value: 'kinoform', title: 'Kinoform' },
            { value: 'multilevel', title: 'Multilevel' },
          ]}
        />
      </FormRow>
      {model.type === 'multilevel' && (
        <FormRow label="Levels" help="Number of etch levels of the staircase (2 binary, 4, 8, 16 …).">
          <span className="flex items-center gap-1">
            <button type="button" aria-label="Fewer levels" className="rounded p-0.5 text-secondary hover:bg-hover" onClick={() => set({ type: 'multilevel', levels: Math.max(1, levels - 1) })}>
              <Minus className="size-3.5" />
            </button>
            <span className="num w-8 text-center">{levels}</span>
            <button type="button" aria-label="More levels" className="rounded p-0.5 text-secondary hover:bg-hover" onClick={() => set({ type: 'multilevel', levels: Math.min(256, levels + 1) })}>
              <Plus className="size-3.5" />
            </button>
          </span>
        </FormRow>
      )}
      <FormRow label="Trace order at λ₀" help={`Efficiency of the trace order (${p.traceOrder}) at the design wavelength ${sig(p.designWavelengthUm, 5)} µm.`}>
        <span className="num">{pct(traceEfficiency(model, p, p.designWavelengthUm))}</span>
      </FormRow>
      {model.type !== 'ideal' && (
        <div className="space-y-[3px] py-1">
          {wavelengths.map((w, k) => {
            const e = traceEfficiency(model, p, w.um)
            return (
              <div key={k} className="flex items-center gap-2 text-[11px]" aria-label={`${sig(w.um, 4)} micrometres: ${pct(e)}`}>
                <span className="num w-[62px] shrink-0 text-secondary">{sig(w.um, 4)} µm</span>
                <span className="relative h-[5px] flex-1 rounded-full bg-selected">
                  <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.max(2, (Number.isFinite(e) ? e : 0) * 100)}%`, background: wavelengthColor(w.um, dark) }} />
                </span>
                <span className="num w-[46px] shrink-0 text-right">{pct(e)}</span>
              </div>
            )
          })}
        </div>
      )}
    </Section>
  )
}
