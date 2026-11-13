// MOCK ENGINE — more synthetic wave-optics results: line spread, MTF vs field,
// through-focus MTF and diffraction encircled energy. Built from traced spot
// sizes and the wavefront model like mtf/psf; not diffraction computations.
import type { JsonObject } from '../engine/types'
import { MockOptics, linspace } from './optics'
import { addBuf, airyRadius, newOut, nf, pair, selectIndices, type Out } from './analyses'
import { buildMap, diffractionMtf, j1, spotFor } from './waves'
import type { Obj } from './model'

const fieldSel = (o: MockOptics, v: unknown) => selectIndices(v, o.sys.fields.length, 'all')
const wlSel = (o: MockOptics, v: unknown, d: 'all' | 'primary') => selectIndices(v, o.sys.wavelengths.length, d, o.primary)

/** Bessel J0 (Numerical Recipes). */
function j0(x: number): number {
  const ax = Math.abs(x)
  if (ax < 8) {
    const y = x * x
    const a1 = 57568490411.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))))
    const a2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))))
    return a1 / a2
  }
  const z = 8 / ax
  const y = z * z
  const xx = ax - 0.785398164
  const a1 = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
  const a2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * -0.934935152e-7)))
  return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * a1 - z * Math.sin(xx) * a2)
}

const airyEE = (x: number) => (x < 1e-9 ? 0 : Math.min(1, Math.max(0, 1 - j0(x) ** 2 - j1(x) ** 2)))

function workingN(o: MockOptics): number {
  const n = o.workingFNumber(o.primary)
  return Number.isFinite(n) ? n : 10
}

function sigmas(o: MockOptics, f: number, wls: number[]): { sx: number; sy: number } {
  const st = spotFor(o, f, wls).poly
  return { sx: Number.isFinite(st.rms_x) ? st.rms_x : 0, sy: Number.isFinite(st.rms_y) ? st.rms_y : 0 }
}

const pupilSampling = (wls: number[]) => wls.map((w) => ({ grid: 65, wavelength: w }))
const fieldValue = (o: MockOptics, f: number) => pair(o.sys.fields[f].x, o.sys.fields[f].y)

function cutoffOf(o: MockOptics, wls: number[]): number {
  const lam = Math.min(...wls.map((w) => o.wavelengthsUm[w])) * 1e-3
  return 1 / (lam * workingN(o))
}

/** A round 1-2-5 number at or below x. */
function nice(x: number): number {
  if (!(x > 0)) return 1
  const e = Math.pow(10, Math.floor(Math.log10(x)))
  const m = x / e
  return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * e
}

function autoFrequencies(cutoff: number): number[] {
  const base = nice(cutoff / 12)
  return [base, 2 * base, 4 * base].filter((f) => f < cutoff)
}

// --- Line spread ---------------------------------------------------------------------

export function lineSpread(o: MockOptics, s: Obj): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const nw = workingN(o)
  const cutoff = cutoffOf(o, wls)
  const lamN = Math.min(...wls.map((w) => o.wavelengthsUm[w])) * 1e-3 * nw
  const N = Number(s.grid_size) > 0 ? Math.min(Math.max(Math.floor(Number(s.grid_size)), 16), 513) | 1 : 129
  const out = newOut({})
  const groups: JsonObject[] = []
  fields.forEach((f, k) => {
    const { sx, sy } = sigmas(o, f, wls)
    const half = Math.max(6 * lamN, 4 * Math.max(sx, sy))
    const pos = linspace(-half, half, N)
    const lsf = (sigma: number) => {
      const steps = 160
      const v = pos.map((x) => {
        let acc = 0
        for (let i = 0; i < steps; i++) {
          const nu = ((i + 0.5) / steps) * cutoff
          acc += diffractionMtf(nu / cutoff) * Math.exp(-2 * Math.PI * Math.PI * nu * nu * sigma * sigma) * Math.cos(2 * Math.PI * nu * x)
        }
        return Math.max(acc, 0)
      })
      const dx = pos[1] - pos[0]
      const area = v.reduce((a, b) => a + b, 0) * dx || 1
      return Float64Array.from(v, (a) => a / area)
    }
    addBuf(out, `positions_mm_${k}`, Float64Array.from(pos), [N])
    addBuf(out, `x_${k}`, lsf(sx), [N])
    addBuf(out, `y_${k}`, lsf(sy), [N])
    groups.push({ field: f, field_value: fieldValue(o, f), positions_mm: `positions_mm_${k}`, pupil_sampling: pupilSampling(wls), x: `x_${k}`, y: `y_${k}` })
  })
  out.json = { fields: groups }
  return out
}

// --- MTF vs field -----------------------------------------------------------------------

export function mtfVsField(o: MockOptics, s: Obj): Out {
  const wls = wlSel(o, s.wavelengths, 'all')
  const cutoff = cutoffOf(o, wls)
  const freqs = Array.isArray(s.frequencies) && s.frequencies.length ? (s.frequencies as number[]) : autoFrequencies(cutoff)
  const nS = Math.max(Math.floor(Number(s.field_samples ?? 11)), 1)
  const dir = Array.isArray(s.direction) ? (s.direction as number[]) : [0, 1]
  const dl = Math.hypot(dir[0], dir[1]) || 1
  const [ux, uy] = [dir[0] / dl, dir[1] / dl]
  const mf = o.maxField
  const ts = nS === 1 ? [0] : linspace(0, 1, nS)
  const fv = new Float64Array(nS * 2)
  const sig = ts.map((t, i) => {
    const x = ux * t * mf
    const y = uy * t * mf
    fv[i * 2] = x
    fv[i * 2 + 1] = y
    // Spot at an arbitrary field point through the nearest system field's blur, scaled by (field/maxfield)².
    const fi = o.sys.fields.reduce((best, f, k, a) => (Math.hypot(f.x - x, f.y - y) < Math.hypot(a[best].x - x, a[best].y - y) ? k : best), 0)
    return sigmas(o, fi, wls)
  })
  const out = newOut({})
  addBuf(out, 'frequencies', Float64Array.from(freqs), [freqs.length])
  addBuf(out, 'normalized_field', Float64Array.from(ts), [nS])
  addBuf(out, 'field_values', fv, [nS, 2])
  const curves: JsonObject[] = freqs.map((nu, k) => {
    const d = diffractionMtf(nu / cutoff)
    const tg = Float64Array.from(sig, (g) => d * Math.exp(-2 * Math.PI * Math.PI * nu * nu * g.sy * g.sy))
    const sg = Float64Array.from(sig, (g) => d * Math.exp(-2 * Math.PI * Math.PI * nu * nu * g.sx * g.sx))
    addBuf(out, `tangential_${k}`, tg, [nS])
    addBuf(out, `sagittal_${k}`, sg, [nS])
    return { frequency: nu, tangential: `tangential_${k}`, sagittal: `sagittal_${k}` }
  })
  out.json = {
    curves,
    field_values: 'field_values',
    frequencies: 'frequencies',
    normalized_field: 'normalized_field',
    pupil_sampling: ts.map(() => pupilSampling(wls)),
  }
  return out
}

// --- Through-focus MTF ---------------------------------------------------------------------

export function throughFocusMtf(o: MockOptics, s: Obj): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const cutoff = cutoffOf(o, wls)
  const freqs = Array.isArray(s.frequencies) && s.frequencies.length ? (s.frequencies as number[]) : autoFrequencies(cutoff)
  const nF = Math.max(Math.floor(Number(s.focus_samples ?? 21)), 1)
  const nw = workingN(o)
  const lam = o.wavelengthsUm[o.primary] * 1e-3
  const range = Array.isArray(s.focus_range_mm) && s.focus_range_mm.length === 2 ? (s.focus_range_mm as number[]) : [-8 * lam * nw * nw, 8 * lam * nw * nw]
  const focus = nF === 1 ? [0] : linspace(range[0], range[1], nF)
  const out = newOut({})
  addBuf(out, 'focus_mm', Float64Array.from(focus), [nF])
  addBuf(out, 'frequencies', Float64Array.from(freqs), [freqs.length])
  let k = 0
  const groups: JsonObject[] = []
  for (const f of fields) {
    const sp = spotFor(o, f, wls)
    const pts = sp.pts
    const n = pts.length || 1
    const spread = (axis: 'x' | 'y', delta: number) => {
      const slope = axis === 'x' ? 'sx' : 'sy'
      const vals = pts.map((p) => p[axis] + delta * p[slope])
      const m = vals.reduce((a, b) => a + b, 0) / n
      return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / n)
    }
    const curves: JsonObject[] = []
    for (const nu of freqs) {
      const d = diffractionMtf(nu / cutoff)
      const tg = Float64Array.from(focus, (df) => d * Math.exp(-2 * Math.PI * Math.PI * nu * nu * spread('y', df) ** 2))
      const sg = Float64Array.from(focus, (df) => d * Math.exp(-2 * Math.PI * Math.PI * nu * nu * spread('x', df) ** 2))
      addBuf(out, `tangential_${k}`, tg, [nF])
      addBuf(out, `sagittal_${k}`, sg, [nF])
      curves.push({ frequency: nu, tangential: `tangential_${k}`, sagittal: `sagittal_${k}` })
      k++
    }
    groups.push({ curves, field: f, field_value: fieldValue(o, f), pupil_sampling: pupilSampling(wls) })
  }
  out.json = { fields: groups, focus_mm: 'focus_mm', frequencies: 'frequencies' }
  return out
}

// --- Diffraction encircled energy --------------------------------------------------------------

export function diffractionEncircledEnergy(o: MockOptics, s: Obj): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const nw = workingN(o)
  const lam = Math.min(...wls.map((w) => o.wavelengthsUm[w])) * 1e-3
  const lamN = lam * nw
  const nR = Math.max(Math.floor(Number(s.samples ?? 51)), 2)
  const rmax = typeof s.max_radius_mm === 'number' && s.max_radius_mm > 0 ? s.max_radius_mm : 6 * lamN + 2 * airyRadius(o)
  const radius = linspace(0, rmax, nR)
  const padding = typeof s.padding === 'number' && s.padding > 0 ? s.padding : 2.5
  const out = newOut({})
  addBuf(out, 'radius_mm', Float64Array.from(radius), [nR])
  const first = (arr: number[], frac: number): number | null => {
    for (let i = 1; i < arr.length; i++)
      if (arr[i] >= frac) {
        const t = (frac - arr[i - 1]) / (arr[i] - arr[i - 1] || 1)
        return radius[i - 1] + t * (radius[i] - radius[i - 1])
      }
    return null
  }
  const groups: JsonObject[] = []
  fields.forEach((f, k) => {
    const { sx, sy } = sigmas(o, f, wls)
    const sigma = Math.max(Math.hypot(sx, sy) / Math.SQRT2, 0.3 * lamN)
    const S = Math.exp(-((2 * Math.PI * buildMap(o, f, o.primary, 17).rmsTR) ** 2))
    const ee = radius.map((r) => S * airyEE((Math.PI * r) / lamN) + (1 - S) * (1 - Math.exp(-(r * r) / (2 * sigma * sigma))))
    const ens = radius.map((r) => {
      const re = r * 1.128
      return S * airyEE((Math.PI * re) / lamN) + (1 - S) * (1 - Math.exp(-(re * re) / (2 * sigma * sigma)))
    })
    const dl = radius.map((r) => airyEE((Math.PI * r) / lamN))
    addBuf(out, `encircled_${k}`, Float64Array.from(ee), [nR])
    addBuf(out, `ensquared_${k}`, Float64Array.from(ens), [nR])
    addBuf(out, `diffraction_limit_${k}`, Float64Array.from(dl), [nR])
    groups.push({
      center_mm: pair(0, 0),
      diffraction_limit: `diffraction_limit_${k}`,
      encircled: `encircled_${k}`,
      ensquared: `ensquared_${k}`,
      field: f,
      field_value: fieldValue(o, f),
      grid_size: Number(s.grid_size) > 0 ? Number(s.grid_size) : 128,
      pupil_sampling: pupilSampling(wls),
      r50_mm: nf(first(ee, 0.5)),
      r80_mm: nf(first(ee, 0.8)),
      r90_mm: nf(first(ee, 0.9)),
      spacing_mm: lamN / padding,
    })
  })
  out.json = { fields: groups, radius_mm: 'radius_mm' }
  return out
}
