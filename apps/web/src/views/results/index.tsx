// Analysis result views: a dedicated renderer per known analysis, the generic
// data view for everything else, and the key numbers of the pane footer.
import type { JSX } from 'react'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import type { System } from '@/model/system'
import { fmt, nf, sig } from '@/model/format'
import { arr, isObj, maxAbs, nums, obj } from './common'
import { FansView, fanMax } from './Fans'
import { FieldCurvatureView, maxDistortion } from './FieldCurvature'
import { FirstOrderView, valueText } from './FirstOrder'
import { GenericView } from './Generic'
import { MtfView } from './Mtf'
import { PsfView } from './Psf'
import { SpotGrid } from './SpotGrid'
import { WavefrontView } from './Wavefront'
import { CURVE_RENDERERS } from '../analysis/curves'
import { MAP_RENDERERS } from '../analysis/maps'
import type { KeyNumber, Renderers } from '../analysis/registry'

export type { KeyNumber }

/** Dedicated renderers of the analysis groups; they take precedence over the
 *  built-in views below. */
export const RENDERERS: Renderers = { ...CURVE_RENDERERS, ...MAP_RENDERERS }

export function ResultView(props: {
  analysis: string
  result: AnalysisResult
  system: System
  settings?: JsonObject
  onSettingsChange?: (patch: JsonObject) => void
}): JSX.Element {
  const { analysis, result, system } = props
  const j = isObj(result.json) ? result.json : {}
  const r = RENDERERS[analysis]
  if (r && (!r.accepts || r.accepts(j))) {
    const settings = props.settings ?? (isObj(result.runInfo.settings) ? result.runInfo.settings : {})
    return <r.View analysis={analysis} result={result} system={system} settings={settings} onSettingsChange={props.onSettingsChange} />
  }
  switch (analysis) {
    case 'spot_diagram':
      if (Array.isArray(j.fields)) return <SpotGrid result={result} system={system} />
      break
    case 'ray_fan':
    case 'opd_fan':
      if (Array.isArray(j.fields)) return <FansView result={result} system={system} opd={analysis === 'opd_fan'} />
      break
    case 'mtf':
      if (Array.isArray(j.fields)) return <MtfView result={result} system={system} />
      break
    case 'psf':
      if (Array.isArray(j.psfs)) return <PsfView result={result} system={system} />
      break
    case 'field_curvature_distortion':
      if (Array.isArray(j.curves)) return <FieldCurvatureView result={result} system={system} />
      break
    case 'wavefront_map':
      if (Array.isArray(j.maps)) return <WavefrontView result={result} system={system} />
      break
    case 'first_order':
      if (Array.isArray(j.entries)) return <FirstOrderView result={result} />
      break
  }
  return <GenericView analysis={analysis} result={result} />
}

const um = (mm: number, d = 2) => `${fmt(mm * 1000, d)} µm`

export function keyNumbers(analysis: string, result: AnalysisResult, system?: System): KeyNumber[] {
  const j = isObj(result.json) ? result.json : {}
  const r = RENDERERS[analysis]
  if (r?.keyNumbers && (!r.accepts || r.accepts(j))) return r.keyNumbers(result, system)
  const out: KeyNumber[] = []
  switch (analysis) {
    case 'spot_diagram': {
      const rms = arr(j.fields).filter(isObj).map((f) => nf(f.rms_radius)).filter(Number.isFinite)
      if (rms.length) out.push({ label: 'Max RMS radius', value: um(Math.max(...rms)), help: 'Largest polychromatic RMS spot radius over the fields' })
      const airy = nf(j.airy_radius)
      if (Number.isFinite(airy)) out.push({ label: 'Airy radius', value: um(airy), help: '1.22 λ N at the primary wavelength' })
      break
    }
    case 'ray_fan': {
      const m = fanMax(j)
      out.push({ label: 'Max |ε|', value: m < 0.1 ? um(m) : `${fmt(m, 4)} mm`, help: 'Largest transverse ray aberration over all fans' })
      break
    }
    case 'opd_fan':
      out.push({ label: 'Max |OPD|', value: `${fmt(fanMax(j), 3)} λ`, help: 'Largest optical path difference over all fans' })
      break
    case 'mtf': {
      const c = nf(obj(arr(j.fields)[0])?.cutoff)
      if (Number.isFinite(c)) out.push({ label: 'Cutoff', value: `${sig(c, 4)} cycles/mm`, help: 'Diffraction cutoff frequency' })
      break
    }
    case 'psf': {
      const p = obj(arr(j.psfs)[0])
      const s = nf(p?.strehl)
      if (Number.isFinite(s)) out.push({ label: 'Strehl', value: fmt(s, 3), help: 'Strehl ratio of the first PSF (field 1)' })
      break
    }
    case 'wavefront_map': {
      const m = obj(arr(j.maps)[0])
      if (m) {
        out.push({ label: 'RMS', value: `${fmt(nf(m.rms_waves), 4)} λ`, help: 'RMS wavefront error of the first map, piston removed' })
        out.push({ label: 'PV', value: `${fmt(nf(m.pv_waves), 4)} λ`, help: 'Peak-to-valley wavefront error of the first map' })
      }
      break
    }
    case 'field_curvature_distortion': {
      const curves = arr(j.curves).filter(isObj)
      if (curves.length) {
        out.push({ label: 'Max |distortion|', value: `${fmt(maxDistortion(j), 3)} %`, help: 'Largest distortion over the field' })
        const fc = Math.max(...curves.map((c) => Math.max(maxAbs(nums(c.tangential)), maxAbs(nums(c.sagittal)))))
        out.push({ label: 'Max focus shift', value: `${fmt(fc, 4)} mm`, help: 'Largest tangential or sagittal focus shift' })
      }
      break
    }
    case 'first_order': {
      const es = arr(j.entries).filter(isObj)
      const find = (...keys: string[]) => es.find((e) => keys.includes(String(e.key)))
      const efl = find('efl')
      if (efl) out.push({ label: 'EFL', value: valueText(efl.value, efl.unit), help: String(efl.label ?? 'Effective focal length') })
      const f = find('image_f_number', 'working_f_number', 'f_number')
      if (f) out.push({ label: 'F/#', value: valueText(f.value, ''), help: String(f.label ?? 'F-number') })
      break
    }
    case 'validate':
      if (typeof j.errors === 'number') out.push({ label: 'Errors', value: String(j.errors) })
      if (typeof j.warnings === 'number') out.push({ label: 'Warnings', value: String(j.warnings) })
      break
  }
  return out
}
