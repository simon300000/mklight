// Settings the engine chooses from the system when left at 0, null or empty
// (`AutoSettings` of the macOS app, docs/WAVE.md): the form shows them as
// "Auto (n)" with the value the last run used, read from its result, never as
// a bare 0.

import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { nf } from '@/model/format'

export type Sentinel = 'zero' | 'null' | 'empty'

/** The sentinel of an automatic setting, by key path (`pupil.grid`). */
export function autoSentinel(key: string, analysis: string): Sentinel | null {
  switch (key) {
    case 'pupil.grid':
    case 'optics.pupil.grid':
    case 'optics.psf_size':
      return 'zero'
    case 'grid_size':
      return ['psf', 'diffraction_encircled_energy', 'line_spread'].includes(analysis) ? 'zero' : null
    case 'max_frequency':
      return analysis === 'mtf' ? 'null' : null
    case 'focus_range_mm':
      return analysis === 'through_focus_mtf' ? 'null' : null
    case 'spacing_mm':
      return analysis === 'psf' ? 'null' : null
    case 'optics.psf_spacing_mm':
      return 'null'
    case 'through_focus.min':
    case 'through_focus.max':
      return analysis === 'spot_diagram' ? 'null' : null
    case 'frequencies':
      return ['mtf_vs_field', 'through_focus_mtf'].includes(analysis) ? 'empty' : null
    default:
      return null
  }
}

export function isAuto(value: Json | undefined, s: Sentinel): boolean {
  if (s === 'zero') return value === undefined || value === null || value === 0
  if (s === 'null') return value === undefined || value === null
  return value === undefined || value === null || (Array.isArray(value) && value.length === 0)
}

/** "Auto (282)", or "Auto" before a run has said what it used. */
export const autoDisplay = (used: string | null) => (used ? `Auto (${used})` : 'Auto')

/** `%.3g` as Swift prints it. */
export function g3(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (v === 0) return '0'
  const e = Math.floor(Math.log10(Math.abs(v)))
  if (e < -4 || e >= 3) return v.toExponential(2).replace(/\.?0+e/, 'e').replace(/e([+-])(\d)$/, 'e$10$2')
  return String(Number(v.toPrecision(3)))
}

const whole = (v: number) => (Number.isFinite(v) ? Math.trunc(v) : null)

/** "282", or "254–371" when the values differ. */
export function rangeText(v: number[]): string | null {
  const f = v.filter(Number.isFinite)
  if (!f.length) return null
  const lo = whole(Math.min(...f)), hi = whole(Math.max(...f))
  if (lo == null || hi == null) return null
  return lo === hi ? String(lo) : `${lo}–${hi}`
}

function micrometres(mm: number, upper: number | null): string {
  const a = g3(mm * 1000)
  if (upper == null || !(upper > mm * (1 + 1e-6))) return `${a} µm`
  return `${a}–${g3(upper * 1000)} µm`
}

const objs = (v: Json | undefined): JsonObject[] =>
  Array.isArray(v) ? v.filter((x): x is JsonObject => x !== null && typeof x === 'object' && !Array.isArray(x)) : []

function buf(r: AnalysisResult, ref: Json | undefined, fallback?: string): number[] | null {
  const name = typeof ref === 'string' ? ref : fallback
  const b = name ? r.buffers[name] : undefined
  return b ? Array.from(b as ArrayLike<number | bigint>, Number) : null
}

/** The value the engine used for an automatic setting, read from the result. */
export function autoUsed(key: string, analysis: string, r: AnalysisResult | undefined | null): string | null {
  if (!r) return null
  const j = r.json
  switch (`${analysis} ${key}`) {
    case 'psf grid_size':
      return rangeText(objs(j.psfs).map((p) => nf((p.psf as JsonObject | undefined)?.size)))
    case 'psf spacing_mm': {
      const s = objs(j.psfs).map((p) => nf((p.psf as JsonObject | undefined)?.spacing_mm)).filter(Number.isFinite)
      return s.length ? micrometres(Math.min(...s), Math.max(...s)) : null
    }
    case 'line_spread grid_size':
      return rangeText(objs(j.fields).map((f) => buf(r, f.positions_mm)?.length ?? NaN))
    case 'mtf max_frequency': {
      const f = buf(r, j.frequencies, 'frequencies')
      return f?.length ? g3(f[f.length - 1]) : null
    }
    case 'mtf_vs_field frequencies':
    case 'through_focus_mtf frequencies': {
      const f = buf(r, j.frequencies, 'frequencies')
      return f?.length ? f.map(g3).join(', ') : null
    }
    case 'through_focus_mtf focus_range_mm': {
      const f = buf(r, j.focus_mm, 'focus_mm')
      if (!f?.length) return null
      const half = Math.max(...f.map(Math.abs))
      return Number.isFinite(half) ? '±' + g3(half) : null
    }
    case 'spot_diagram through_focus': {
      const first = objs(j.fields)[0]
      const planes = objs(first?.through_focus).map((p) => nf(p.defocus)).filter(Number.isFinite)
      if (!planes.length) return null
      const lo = Math.min(...planes), hi = Math.max(...planes)
      if (Math.abs(lo + hi) <= 1e-9 * Math.max(1, Math.abs(hi))) return '±' + g3(hi)
      return `${g3(lo)} to ${g3(hi)}`
    }
    case 'image_simulation optics.psf_size': {
      const v = whole(nf(j.psf_size))
      return v == null ? null : String(v)
    }
    case 'image_simulation optics.psf_spacing_mm': {
      const v = nf(j.psf_spacing_mm)
      return Number.isFinite(v) ? micrometres(v, null) : null
    }
  }
  if (key.endsWith('pupil.grid')) return rangeText(objs(j.maps).map((m) => nf(m.grid)))
  return null
}
