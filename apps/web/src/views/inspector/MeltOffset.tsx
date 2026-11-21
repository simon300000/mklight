import { useEffect, useState } from 'react'
import type { JsonObject } from '@/engine/types'
import { useT } from '@/i18n'
import { Button } from '@/components/ui/button'
import { isVariable, type ParamPath } from '@/model/params'
import { glassNumbers, isZeroOffset, offsetOf, offsetRejection, withOffset, type GlassNumbers, type MaterialOffset } from '@/model/materials'
import type { Material, Surface } from '@/model/system'
import { useDocument } from '@/state/document'
import { useWorkspace } from '@/state/workspace'
import { FormRow, NumberRow } from './Form'

const cache = new Map<string, GlassNumbers | null>()

/** A catalogue glass's d-line numbers with its offset and as catalogued, from
 *  the engine's `glass_info` (cached per glass and offset). */
export function useGlassNumbers(m: Material | null): GlassNumbers | null {
  const engine = useWorkspace((s) => s.engine)
  const snapshot = useWorkspace((s) => s.snapshot)
  const libraryKey = useWorkspace((s) => s.materials.length)
  const o = offsetOf(m)
  const key = m && m.type === 'catalog' ? JSON.stringify([m.catalog ?? null, m.name, o, libraryKey]) : null
  const [, bump] = useState(0)
  useEffect(() => {
    if (!key || !m || !engine || !snapshot || cache.has(key)) return
    let live = true
    const settings: JsonObject = { name: String(m.name), catalog: (m.catalog as string | null) ?? null, offset: { dn_d: o?.dn_d ?? 0, dv_d: o?.dv_d ?? 0 } }
    try {
      const job = engine.submit({ analysis: 'glass_info', settings }, snapshot)
      job.result.then(
        (r) => {
          cache.set(key, glassNumbers(r.json))
          if (live) bump((n) => n + 1)
        },
        () => {
          cache.set(key, null)
          if (live) bump((n) => n + 1)
        },
      )
    } catch {
      cache.set(key, null)
    }
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, engine, snapshot])
  return key ? (cache.get(key) ?? null) : null
}

/** A catalogue glass's melt offset (docs/APP.md "Glass offsets in the
 *  inspector"): "Add Melt Offset…", then Δn_d and ΔV_d validated against the
 *  engine's ±0.05 and ±10, each a possible variable, the melt glass's n_d and
 *  V_d against the catalogue's, and Remove Melt Offset. */
export function MeltOffsetRows({ surface }: { surface: Surface }) {
  const t = useT()
  const edit = useDocument((s) => s.edit)
  const project = useDocument((s) => s.project)
  const [expanded, setExpanded] = useState(false)
  const offset = offsetOf(surface.material) ?? { dn_d: 0, dv_d: 0 }
  const dn: ParamPath = { param: 'material_dn_d', surface: surface.id }
  const dv: ParamPath = { param: 'material_dv_d', surface: surface.id }
  const g = useGlassNumbers(surface.material)
  useEffect(() => setExpanded(false), [surface.id])
  const showing = expanded || !isZeroOffset(offset) || isVariable(project, dn) || isVariable(project, dv)
  const set = (name: string, change: (o: MaterialOffset) => MaterialOffset) =>
    edit(name, (s) => {
      const sf = s.surfaces.find((x) => x.id === surface.id)
      if (!sf || sf.material.type !== 'catalog') return false
      sf.material = withOffset(sf.material, change(offsetOf(sf.material) ?? { dn_d: 0, dv_d: 0 }))
    })
  if (!showing) {
    return (
      <div className="py-[3px]">
        <Button variant="ghost" size="sm" className="-ml-1.5 text-accent" onClick={() => setExpanded(true)} title={t('inspector.melt.addHelp')}>
          {t('inspector.melt.add')}
        </Button>
      </div>
    )
  }
  return (
    <div data-testid="melt-offset">
      <NumberRow
        label="Δn_d"
        detail={t('inspector.melt.detail')}
        value={offset.dn_d}
        param={dn}
        help={t('inspector.melt.dnHelp')}
        validate={(v) => offsetRejection('dn_d', v)}
        onCommit={(v) => v != null && set(t('inspector.melt.undoDn'), (o) => ({ ...o, dn_d: v }))}
      />
      <NumberRow
        label="ΔV_d"
        detail={t('inspector.melt.detail')}
        value={offset.dv_d}
        param={dv}
        help={t('inspector.melt.dvHelp')}
        validate={(v) => offsetRejection('dv_d', v)}
        onCommit={(v) => v != null && set(t('inspector.melt.undoDv'), (o) => ({ ...o, dv_d: v }))}
      />
      {g && (
        <FormRow label={t('inspector.melt.glass')}>
          <span className="num text-right text-[12px] leading-tight">
            <span className="block">
              n_d {g.nd.toFixed(5)}  V_d {g.vd.toFixed(2)}
            </span>
            {!isZeroOffset(offset) && (
              <span className="block text-[11px] text-secondary">
                {t('inspector.melt.catalog')} {g.catalogNd.toFixed(5)}  {g.catalogVd.toFixed(2)}
              </span>
            )}
          </span>
        </FormRow>
      )}
      {!isZeroOffset(offset) && (
        <Button
          variant="ghost"
          size="sm"
          className="-ml-1.5 text-accent"
          title={t('inspector.melt.removeHelp')}
          onClick={() => {
            setExpanded(false)
            set(t('inspector.melt.remove'), () => ({ dn_d: 0, dv_d: 0 }))
          }}
        >
          {t('inspector.melt.remove')}
        </Button>
      )}
    </div>
  )
}
