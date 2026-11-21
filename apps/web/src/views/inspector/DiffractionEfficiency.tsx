import { Minus, Plus } from 'lucide-react'
import type { Json } from '@/engine/types'
import { useT } from '@/i18n'
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
  const t = useT()
  const edit = useDocument((s) => s.edit)
  const wavelengths = useDocument((s) => s.project.system.wavelengths.wavelengths)
  const dark = useDark()
  const p = phaseOf(surface.shape)
  if (!p) return null
  const model = efficiencyModel(surface.diffraction_efficiency as Json)
  const set = (m: EfficiencyModel) => edit(t('inspector.efficiency.undo'), (s) => void (s.surfaces[index].diffraction_efficiency = m))
  const levels = model.type === 'multilevel' ? model.levels : 8
  return (
    <Section
      title={t('inspector.efficiency.title')}
      testId="diffraction-efficiency"
      footer={model.type !== 'ideal' ? t('inspector.efficiency.footer') : undefined}
    >
      <FormRow label={t('inspector.efficiency.model')} help={t('inspector.efficiency.modelHelp')}>
        <Segmented
          label={t('inspector.efficiency.modelLabel')}
          value={model.type}
          onChange={(m) => set(m === 'multilevel' ? { type: 'multilevel', levels } : { type: m })}
          options={[
            { value: 'ideal', title: t('inspector.efficiency.ideal') },
            { value: 'kinoform', title: t('inspector.efficiency.kinoform') },
            { value: 'multilevel', title: t('inspector.efficiency.multilevel') },
          ]}
        />
      </FormRow>
      {model.type === 'multilevel' && (
        <FormRow label={t('inspector.efficiency.levels')} help={t('inspector.efficiency.levelsHelp')}>
          <span className="flex items-center gap-1">
            <button type="button" aria-label={t('inspector.efficiency.fewer')} className="rounded p-0.5 text-secondary hover:bg-hover" onClick={() => set({ type: 'multilevel', levels: Math.max(1, levels - 1) })}>
              <Minus className="size-3.5" />
            </button>
            <span className="num w-8 text-center">{levels}</span>
            <button type="button" aria-label={t('inspector.efficiency.more')} className="rounded p-0.5 text-secondary hover:bg-hover" onClick={() => set({ type: 'multilevel', levels: Math.min(256, levels + 1) })}>
              <Plus className="size-3.5" />
            </button>
          </span>
        </FormRow>
      )}
      <FormRow label={t('inspector.efficiency.traceOrder')} help={t('inspector.efficiency.traceOrderHelp', { order: p.traceOrder, wavelength: sig(p.designWavelengthUm, 5) })}>
        <span className="num">{pct(traceEfficiency(model, p, p.designWavelengthUm))}</span>
      </FormRow>
      {model.type !== 'ideal' && (
        <div className="space-y-[3px] py-1">
          {wavelengths.map((w, k) => {
            const e = traceEfficiency(model, p, w.um)
            return (
              <div key={k} className="flex items-center gap-2 text-[11px]" aria-label={t('inspector.efficiency.wavelengthLabel', { um: sig(w.um, 4), value: pct(e) })}>
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
