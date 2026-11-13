// MOCK ENGINE — the remaining geometric analyses on MockOptics: seidel,
// chromatic focal shift, longitudinal aberration, lateral colour, encircled
// energy, geometric MTF, best focus, ray grid, footprint, grid distortion,
// relative illumination and the field-map family. Shapes follow docs/RESULTS.md;
// numbers are illustrative.
import type { Json, JsonObject } from '../engine/types'
import { MockOptics, linspace, shapeCurvature, type Ray } from './optics'
import { addBuf, airyRadius, newOut, nf, nfs, pair, pupilSamples, selectIndices, spotData, type Out } from './analyses'
import type { Obj } from './model'

const fieldSel = (o: MockOptics, v: unknown) => selectIndices(v, o.sys.fields.length, 'all')
const wlSel = (o: MockOptics, v: unknown, d: 'all' | 'primary') => selectIndices(v, o.sys.wavelengths.length, d, o.primary)
const wlIndex = (o: MockOptics, v: unknown): number => (typeof v === 'number' && Number.isInteger(v) && v >= 0 && v < o.sys.wavelengths.length ? v : o.primary)

/** A copy of the optics with another wavelength list (primary = `primary`). */
export function withWavelengths(o: MockOptics, ums: number[], primary: number): MockOptics {
  return new MockOptics({ ...o.sys, wavelengths: ums.map((um) => ({ um, weight: 1 })), primary })
}

/** Paraxial focus relative to the image surface for wavelength index w. */
export function paraxialFocus(o: MockOptics, w: number): number | null {
  const m = o.marginalStart()
  const pr = o.paraxial(m.y1, m.u0, w)
  const k = o.sys.surfaces.length - 2
  const nu = pr.nu[k]
  if (!(Math.abs(nu) > 1e-15)) return null
  return (-pr.y[k] * o.imageN(w)) / nu - o.sys.surfaces[k].thickness
}

export const isRotational = (o: MockOptics) => !o.sys.surfaces.some((s) => ['toroid', 'biconic', 'xy_polynomial', 'zernike_sag'].includes(s.shape.type)) && !o.sys.surfaces.some((s) => s.hasPlacement)

function chiefHit(o: MockOptics, x: number, y: number, w: number): Ray | null {
  return o.trace(o.makeRay(o.fieldPointXY(x, y), 0, 0), w).atImage
}

/** Unit chief-ray start: height at the first surface and slope. */
function chiefStart(o: MockOptics): { y1: number; u0: number } {
  const d = o.objectDistance
  if (d !== null) {
    const u0 = -1 / o.objectToPupil(d)
    return { y1: 1 + u0 * d, u0 }
  }
  return { y1: -o.epZ, u0: 1 }
}

export function fieldScale(o: MockOptics): number {
  if (o.sys.fieldKind === 'object_angle') return Math.tan((o.maxField * Math.PI) / 180)
  if (o.sys.fieldKind === 'object_height') return o.maxField
  return Math.abs(o.chiefImagePerUnit) > 1e-12 ? o.maxField / o.chiefImagePerUnit : 1
}

// --- Seidel ---------------------------------------------------------------------

export function seidel(o: MockOptics, s: Obj): Out {
  const sys = o.sys
  const n = sys.surfaces.length
  const w = wlIndex(o, s.wavelength)
  const wls = o.wavelengthsUm
  const shortW = typeof s.short_wavelength === 'number' ? s.short_wavelength : wls.indexOf(Math.min(...wls))
  const longW = typeof s.long_wavelength === 'number' ? s.long_wavelength : wls.indexOf(Math.max(...wls))
  if (!isRotational(o))
    return newOut({ status: 'not_applicable', reason: 'Seidel coefficients need a rotationally symmetric system', short_wavelength: shortW, long_wavelength: longW, wavelength: w, wavelength_um: wls[w] })
  const m0 = o.marginalStart()
  const mg = o.paraxial(m0.y1, m0.u0, w)
  const c0 = chiefStart(o)
  const k = fieldScale(o)
  const ch = o.paraxial(c0.y1 * k, c0.u0 * k, w)
  const nSigned = (i: number, ww: number) => o.nAfter[Math.max(i, 0)][ww] * o.travelSign[Math.max(i, 0)]
  const dn = (i: number) => nSigned(i, shortW) - nSigned(i, longW)
  const rows: JsonObject[] = []
  const sums = { s1: 0, s2: 0, s3: 0, s4: 0, s5: 0, cl: 0, ct: 0 }
  let petz = 0
  // Lagrange invariant at the first surface.
  const n0 = nSigned(0, w)
  const H = n0 * (ch.y[1] * (mg.nu[0] / n0) - mg.y[1] * (ch.nu[0] / n0))
  for (let i = 1; i < n - 1; i++) {
    const sf = sys.surfaces[i]
    if (sf.isCB || sf.interaction === 'dummy') continue
    const c = shapeCurvature(sf.shape)
    const nb = i === 1 ? n0 : nSigned(i - 1, w)
    const na = nSigned(i, w)
    const y = mg.y[i]
    const yb = ch.y[i]
    const u = mg.nu[i - 1] / nb
    const ub = ch.nu[i - 1] / nb
    const A = nb * (c * y + u)
    const Ab = nb * (c * yb + ub)
    const d1n = 1 / na - 1 / nb
    const dun = mg.nu[i] / na / na - u / nb
    const s1 = -A * A * y * dun
    const s2 = -A * Ab * y * dun
    const s3 = -Ab * Ab * y * dun
    const s4 = -H * H * c * d1n
    const s5 = Math.abs(A) > 1e-12 ? (Ab / A) * (s3 + s4) : 0
    const ddn = dn(i) / na - dn(i - 1 >= 0 ? i - 1 : 0) / nb
    const cl = -A * y * ddn
    const ct = -A * yb * ddn
    petz += (c * (na - nb)) / (nb * na)
    sums.s1 += s1
    sums.s2 += s2
    sums.s3 += s3
    sums.s4 += s4
    sums.s5 += s5
    sums.cl += cl
    sums.ct += ct
    rows.push({ surface: i, s1: nf(s1), s2: nf(s2), s3: nf(s3), s4: nf(s4), s5: nf(s5), cl: nf(cl), ct: nf(ct) })
  }
  const kImg = n - 2
  const nk = nSigned(kImg, w)
  const lam = wls[w] * 1e-3
  const wm = { w020: 0, w040: sums.s1 / 8, w111: 0, w131: sums.s2 / 2, w220: (sums.s3 + sums.s4) / 4, w220p: sums.s4 / 4, w222: sums.s3 / 2, w311: sums.s5 / 2 }
  const ww = Object.fromEntries(Object.entries(wm).map(([key, v]) => [key, nf(v / lam)]))
  const wmm = Object.fromEntries(Object.entries(wm).map(([key, v]) => [key, nf(v)]))
  return newOut({
    status: 'applicable',
    reason: null,
    image_marginal_slope: nf(mg.nu[kImg]),
    lagrange_invariant: nf(H),
    long_wavelength: longW,
    short_wavelength: shortW,
    petzval_radius: nf(Math.abs(petz) > 1e-15 ? -1 / (nk * petz) : Infinity),
    petzval_sum: nf(petz),
    sums: Object.fromEntries(Object.entries(sums).map(([key, v]) => [key, nf(v)])),
    surfaces: rows,
    wavefront_mm: wmm,
    wavefront_waves: ww,
    wavelength: w,
    wavelength_um: wls[w],
  })
}

// --- Chromatic and longitudinal -------------------------------------------------------

export function chromaticFocalShift(o: MockOptics, s: Obj): Out {
  const wls = o.wavelengthsUm
  const lo = typeof s.min_um === 'number' ? s.min_um : Math.min(...wls)
  const hi = typeof s.max_um === 'number' ? s.max_um : Math.max(...wls)
  const samples = Math.max(Math.floor(Number(s.samples ?? 41)), 2)
  const ums = linspace(lo, hi, samples)
  const pri = wls[o.primary]
  const oc = withWavelengths(o, [...ums, pri], samples)
  const focus = ums.map((_, i) => paraxialFocus(oc, i))
  const ref = paraxialFocus(oc, samples)
  const shift = focus.map((f) => (f === null || ref === null ? null : f - ref))
  const finite = shift.filter((v): v is number => v !== null && Number.isFinite(v))
  return newOut({
    focal_shift: nfs(shift),
    focal_shift_x: nfs(shift),
    paraxial_focus: nfs(focus),
    paraxial_focus_x: nfs(focus),
    primary_wavelength_um: pri,
    range: finite.length ? Math.max(...finite) - Math.min(...finite) : null,
    wavelengths_um: ums,
  })
}

export function longitudinalAberration(o: MockOptics, s: Obj): Out {
  const wls = wlSel(o, s.wavelengths, 'all')
  const n = Math.max(Math.floor(Number(s.pupil_samples ?? 21)), 2)
  const zones = linspace(0, 1, n).map((z) => Math.max(z, 1e-4))
  const pri = paraxialFocus(o, o.primary)
  const fp = o.fieldPointXY(0, 0)
  const axes = s.axes === 'both' || (s.axes === 'auto' && !isRotational(o)) ? ['y', 'x'] : [s.axes === 'x' ? 'x' : 'y']
  const curves: JsonObject[] = []
  for (const axis of axes)
    for (const w of wls) {
      const par = paraxialFocus(o, w)
      const focus = zones.map((z) => {
        const r = o.trace(o.makeRay(fp, axis === 'x' ? z : 0, axis === 'y' ? z : 0), w).atImage
        if (!r) return null
        const a = axis === 'x' ? 0 : 1
        if (!(Math.abs(r.d[a]) > 1e-12)) return null
        const t = -r.p[a] / r.d[a]
        return t * r.d[2]
      })
      curves.push({
        axis,
        wavelength: w,
        wavelength_um: o.wavelengthsUm[w],
        focus: nfs(focus),
        aberration: nfs(focus.map((f) => (f === null || pri === null ? null : f - pri))),
        paraxial_focus: nf(par),
      })
    }
  return newOut({ curves, primary_paraxial_focus: nf(pri), pupil: zones })
}

function fieldSweep(o: MockOptics, samples: number, axes: unknown): { values: number[]; coords: [number, number][]; xAxis: boolean } {
  const rot = isRotational(o)
  const xAxis = axes === 'x'
  void rot
  const values = linspace(0, o.maxField, Math.max(Math.floor(samples), 2))
  return { values, coords: values.map((v) => (xAxis ? [v, 0] : [0, v]) as [number, number]), xAxis }
}

export function lateralColour(o: MockOptics, s: Obj): Out {
  const wls = o.wavelengthsUm
  const shortW = typeof s.short_wavelength === 'number' ? s.short_wavelength : wls.indexOf(Math.min(...wls))
  const longW = typeof s.long_wavelength === 'number' ? s.long_wavelength : wls.indexOf(Math.max(...wls))
  const { values, coords, xAxis } = fieldSweep(o, Number(s.field_samples ?? 21), s.field_axes)
  const axis = xAxis ? 0 : 1
  const hit = (w: number) => coords.map(([x, y]) => chiefHit(o, x, y, w)?.p[axis] ?? null)
  const base = hit(o.primary)
  const curves = wls.map((_, w) => {
    const h = hit(w)
    return { wavelength: w, wavelength_um: wls[w], offset: nfs(h.map((v, i) => (v === null || base[i] === null ? null : v - (base[i] as number)))) }
  })
  const hs = hit(shortW)
  const hl = hit(longW)
  const mf = Math.max(...values, 1e-12)
  return newOut({
    airy_radius: nf(airyRadius(o)),
    curves,
    field_coordinates: coords.map(([x, y]) => pair(x, y)),
    field_unit: o.fieldUnit,
    field_values: values,
    lateral_colour: nfs(hs.map((v, i) => (v === null || hl[i] === null ? null : v - (hl[i] as number)))),
    long_wavelength: longW,
    normalized_field: values.map((v) => v / mf),
    short_wavelength: shortW,
  })
}

// --- Encircled energy and geometric MTF ----------------------------------------------

function spotOptsFrom(o: MockOptics, s: Obj, ringCap: number, defaultRef: string) {
  return {
    fields: fieldSel(o, s.fields),
    wls: wlSel(o, s.wavelengths, 'all'),
    pattern: String(s.pattern ?? 'square'),
    rings: Math.min(Number(s.rings ?? 8), ringCap),
    seed: Number(s.seed ?? 1),
    defocus: Number(s.defocus ?? 0),
    reference: String(s.reference ?? defaultRef),
  }
}

function crossing(sorted: number[], total: number, frac: number): number | null {
  if (!sorted.length || total <= 0) return null
  const target = frac * sorted.length
  const i = Math.ceil(target) - 1
  return sorted[Math.min(Math.max(i, 0), sorted.length - 1)]
}

export function encircledEnergy(o: MockOptics, s: Obj): Out {
  const so = spotOptsFrom(o, s, 14, 'centroid')
  const data = spotData(o, so)
  const samples = Math.max(Math.floor(Number(s.samples ?? 101)), 2)
  const geoMax = Math.max(0, ...data.map((d) => d.poly.geo_c).filter(Number.isFinite))
  const rmax = typeof s.max_radius === 'number' && s.max_radius > 0 ? s.max_radius : geoMax > 0 ? geoMax : airyRadius(o) * 3
  const radius = linspace(0, rmax, samples)
  const fields = data.map((f) => {
    const dist = f.pts.map((p) => Math.hypot(p.x, p.y)).sort((a, b) => a - b)
    const sq = f.pts.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y))).sort((a, b) => a - b)
    const frac = (arr: number[]) => {
      let j = 0
      return radius.map((r) => {
        while (j < arr.length && arr[j] <= r) j++
        return arr.length ? j / arr.length : 0
      })
    }
    return {
      field: f.field,
      field_value: pair(o.sys.fields[f.field].x, o.sys.fields[f.field].y),
      encircled: frac(dist),
      ensquared: frac(sq),
      r50: nf(crossing(dist, 1, 0.5)),
      r80: nf(crossing(dist, 1, 0.8)),
      r90: nf(crossing(dist, 1, 0.9)),
      half_width80: nf(crossing(sq, 1, 0.8)),
      rays_arrived: f.pts.length,
    }
  })
  return newOut({ fields, radius, wavelengths: so.wls })
}

export function geometricMtf(o: MockOptics, s: Obj): Out {
  const so = spotOptsFrom(o, s, 14, 'centroid')
  const data = spotData(o, so)
  let nw = o.workingFNumber(o.primary)
  if (!Number.isFinite(nw)) nw = 10
  const cutoff = 1 / (o.wavelengthsUm[o.primary] * 1e-3 * nw)
  const maxF = typeof s.max_frequency === 'number' && s.max_frequency > 0 ? s.max_frequency : cutoff
  const nS = Math.max(Math.floor(Number(s.frequency_samples ?? 41)), 2)
  const freqs = linspace(0, maxF, nS)
  const curves: JsonObject[] = []
  for (const f of data)
    for (const orient of ['tangential', 'sagittal'] as const) {
      const pos = f.pts.map((p) => (orient === 'tangential' ? p.y : p.x))
      const mod = freqs.map((nu) => {
        if (!pos.length) return null
        let c = 0
        let sn = 0
        for (const x of pos) {
          c += Math.cos(2 * Math.PI * nu * x)
          sn += Math.sin(2 * Math.PI * nu * x)
        }
        return Math.hypot(c, sn) / pos.length
      })
      curves.push({ field: f.field, orientation: orient, modulation: nfs(mod) })
    }
  return newOut({
    curves,
    cutoff_frequency: nf(cutoff),
    diffraction_limit: null,
    field_values: data.map((f) => pair(o.sys.fields[f.field].x, o.sys.fields[f.field].y)),
    frequencies: freqs,
  })
}

// --- Best focus ----------------------------------------------------------------------

export function bestFocus(o: MockOptics, s: Obj): Out {
  const so = { ...spotOptsFrom(o, s, 8, 'centroid'), defocus: 0 }
  const data = spotData(o, so)
  const weights = data.map((d) => o.sys.fields[d.field].weight)
  const wsum = weights.reduce((a, b) => a + b, 0) || 1
  // RMS of x + δ·slope about the centroid.
  const rmsAt = (d: (typeof data)[number], delta: number): number => {
    const n = d.pts.length
    if (!n) return NaN
    let mx = 0
    let my = 0
    for (const p of d.pts) {
      mx += p.x + delta * p.sx
      my += p.y + delta * p.sy
    }
    mx /= n
    my /= n
    let v = 0
    for (const p of d.pts) {
      const dx = p.x + delta * p.sx - mx
      const dy = p.y + delta * p.sy - my
      v += dx * dx + dy * dy
    }
    return Math.sqrt(v / n)
  }
  const merit = (delta: number) => data.reduce((a, d, i) => a + (weights[i] / wsum) * (rmsAt(d, delta) || 0), 0)
  let nw = o.workingFNumber(o.primary)
  if (!Number.isFinite(nw)) nw = 5
  const lam = o.wavelengthsUm[o.primary] * 1e-3
  const rms0 = Math.max(0, ...data.map((d) => rmsAt(d, 0)).filter(Number.isFinite))
  const auto = Math.max(8 * lam * nw * nw, 4 * nw * rms0, 1e-3)
  const range = Array.isArray(s.range) && s.range.length === 2 ? (s.range as number[]) : [-auto, auto]
  let a = range[0]
  let b = range[1]
  const g = (Math.sqrt(5) - 1) / 2
  let c = b - g * (b - a)
  let dd = a + g * (b - a)
  let fc = merit(c)
  let fd = merit(dd)
  let evals = 2
  const maxEval = Math.min(Number(s.max_evaluations ?? 200), 60)
  const tol = Math.max(Number(s.tolerance ?? 1e-8), 1e-7)
  while (b - a > tol && evals < maxEval) {
    if (fc < fd) {
      b = dd
      dd = c
      fd = fc
      c = b - g * (b - a)
      fc = merit(c)
    } else {
      a = c
      c = dd
      fc = fd
      dd = a + g * (b - a)
      fd = merit(dd)
    }
    evals++
  }
  const best = (a + b) / 2
  // Closed-form minimiser of the mean square RMS.
  let num = 0
  let den = 0
  for (const [i, d] of data.entries()) {
    const n = d.pts.length || 1
    const mx = d.pts.reduce((x, p) => x + p.x, 0) / n
    const my = d.pts.reduce((x, p) => x + p.y, 0) / n
    const ms = d.pts.reduce((x, p) => x + p.sx, 0) / n
    const mys = d.pts.reduce((x, p) => x + p.sy, 0) / n
    let cov = 0
    let vs = 0
    for (const p of d.pts) {
      cov += (p.x - mx) * (p.sx - ms) + (p.y - my) * (p.sy - mys)
      vs += (p.sx - ms) ** 2 + (p.sy - mys) ** 2
    }
    num += (weights[i] / wsum) * (cov / n)
    den += (weights[i] / wsum) * (vs / n)
  }
  const tfN = Math.floor(Number(s.through_focus_samples ?? 0))
  const tfD = tfN > 0 ? linspace(range[0], range[1], tfN) : []
  return newOut({
    converged: true,
    defocus: best,
    evaluations: evals,
    field_rms: nfs(data.map((d) => rmsAt(d, best))),
    field_rms_at_zero: nfs(data.map((d) => rmsAt(d, 0))),
    field_values: data.map((d) => pair(o.sys.fields[d.field].x, o.sys.fields[d.field].y)),
    fields: data.map((d) => d.field),
    mean_square_defocus: den > 0 ? -num / den : 0,
    merit: merit(best),
    merit_at_zero: merit(0),
    range: [range[0], range[1]],
    through_focus_defocus: tfD,
    through_focus_merit: tfD.map(merit),
  })
}

// --- Ray grid and footprint ----------------------------------------------------------

export function rayGrid(o: MockOptics, s: Obj): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const pupil = pupilSamples(String(s.pattern ?? 'hexapolar'), Number(s.rings ?? 6), Number(s.seed ?? 1))
  const F = fields.length
  const W = wls.length
  const P = pupil.length
  const pts = new Float64Array(F * W * P * 2).fill(NaN)
  const dirs = new Float64Array(F * W * P * 3).fill(NaN)
  const chiefB = new Float64Array(F * W * 2).fill(NaN)
  let arrived = 0
  fields.forEach((f, fi) => {
    const fp = o.fieldPoint(f)
    wls.forEach((w, wi) => {
      const c = o.trace(o.makeRay(fp, 0, 0), w).atImage
      if (c) {
        chiefB[(fi * W + wi) * 2] = c.p[0]
        chiefB[(fi * W + wi) * 2 + 1] = c.p[1]
      }
      pupil.forEach(([px, py], pi) => {
        const r = o.trace(o.makeRay(fp, px, py), w).atImage
        if (!r) return
        const at = ((fi * W + wi) * P + pi) * 2
        pts[at] = r.p[0]
        pts[at + 1] = r.p[1]
        const dt = ((fi * W + wi) * P + pi) * 3
        dirs[dt] = r.d[0]
        dirs[dt + 1] = r.d[1]
        dirs[dt + 2] = r.d[2]
        arrived++
      })
    })
  })
  const out = newOut({
    arrived,
    field_values: fields.map((f) => pair(o.sys.fields[f].x, o.sys.fields[f].y)),
    fields,
    launched: F * W * P,
    pupil_count: P,
    wavelengths: wls,
    wavelengths_um: wls.map((w) => o.wavelengthsUm[w]),
  })
  addBuf(out, 'pupil', Float64Array.from(pupil.flat()), [P, 2])
  addBuf(out, 'points', pts, [F, W, P, 2])
  addBuf(out, 'directions', dirs, [F, W, P, 3])
  addBuf(out, 'chief', chiefB, [F, W, 2])
  return out
}

function hull(pts: [number, number, number][]): [number, number, number][] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (p.length < 3) return p
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const lower: [number, number, number][] = []
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop()
    lower.push(q)
  }
  const upper: [number, number, number][] = []
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop()
    upper.push(q)
  }
  lower.pop()
  upper.pop()
  return [...lower, ...upper]
}

export function footprint(o: MockOptics, s: Obj): Out {
  const n = o.sys.surfaces.length
  const surf = Math.min(Math.max(Math.floor(Number(s.surface ?? 1)), 1), n - 1)
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const pupil = pupilSamples(String(s.pattern ?? 'square'), Math.min(Number(s.rings ?? 8), 16), Number(s.seed ?? 1))
  const local: number[] = []
  const clipped: number[] = []
  const gpts: number[] = []
  const gout: number[] = []
  const sets: JsonObject[] = []
  const overlays: JsonObject[] = []
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity
  const traces = new Map<string, { p: number[][]; failedAt: number | null }[]>()
  for (const f of fields)
    for (const w of wls) {
      const fp = o.fieldPoint(f)
      traces.set(
        `${f}:${w}`,
        pupil.map(([px, py]) => {
          const t = o.trace(o.makeRay(fp, px, py), w)
          return { p: t.points, failedAt: t.failedAt }
        }),
      )
    }
  for (const f of fields)
    for (const w of wls) {
      const list = traces.get(`${f}:${w}`)!
      let lost = 0
      const pOff = local.length / 2
      const cOff = clipped.length / 2
      let np = 0
      let nc = 0
      for (const t of list) {
        const reached = t.p.length > surf && (t.failedAt === null || t.failedAt >= surf)
        if (!reached) {
          lost++
          continue
        }
        const h = t.p[surf]
        if (t.failedAt === surf) {
          clipped.push(h[0], h[1])
          nc++
        } else {
          local.push(h[0], h[1])
          np++
          xmin = Math.min(xmin, h[0]); xmax = Math.max(xmax, h[0]); ymin = Math.min(ymin, h[1]); ymax = Math.max(ymax, h[1])
        }
      }
      sets.push({ field: f, wavelength: w, launched: list.length, lost_before: lost, points: [pOff, np], clipped: [cOff, nc] })
    }
  for (let k = 1; k < n; k++)
    for (const f of fields)
      for (const w of wls) {
        const list = traces.get(`${f}:${w}`)!
        const hits: [number, number, number][] = []
        for (const t of list) if (t.p.length > k && (t.failedAt === null || t.failedAt >= k)) hits.push([t.p[k][0], t.p[k][1], t.p[k][2]])
        const gOff = gpts.length / 3
        for (const h of hits) gpts.push(h[0], h[1], h[2])
        const hl = hull(hits)
        const qOff = gout.length / 3
        for (const h of hl) gout.push(h[0], h[1], h[2])
        overlays.push({ surface: k, field: f, wavelength: w, points: [gOff, hits.length], outline: [qOff, hl.length] })
      }
  const sd = o.sys.surfaces[surf].sdFixed
  const maxR = local.length ? Math.max(...Array.from({ length: local.length / 2 }, (_, i) => Math.hypot(local[2 * i], local[2 * i + 1]))) : 0
  const out = newOut({
    aperture: (o.sys.surfaces[surf].raw.aperture ?? null) as Json,
    extent: local.length ? [xmin, xmax, ymin, ymax].map(nf) : [null, null, null, null],
    field_values: fields.map((f) => pair(o.sys.fields[f].x, o.sys.fields[f].y)),
    max_radius: nf(maxR),
    overlays,
    semi_diameter: nf(sd ?? maxR),
    sets,
    surface: surf,
  })
  addBuf(out, 'points', Float64Array.from(local), [local.length / 2, 2])
  addBuf(out, 'clipped', Float64Array.from(clipped), [clipped.length / 2, 2])
  addBuf(out, 'global_points', Float64Array.from(gpts), [gpts.length / 3, 3])
  addBuf(out, 'global_outlines', Float64Array.from(gout), [gout.length / 3, 3])
  return out
}

// --- Grid distortion -----------------------------------------------------------------

export function gridDistortion(o: MockOptics, s: Obj): Out {
  const n = Math.min(Math.max(Math.floor(Number(s.grid_size ?? 11)), 2), 201)
  const w = wlIndex(o, s.wavelength)
  const hw = Number(s.half_width ?? Math.SQRT1_2)
  const [cx, cy] = Array.isArray(s.field_center) ? (s.field_center as number[]) : [0, 0]
  const field = new Float64Array(n * n * 2)
  const par = new Float64Array(n * n * 2)
  const real = new Float64Array(n * n * 2).fill(NaN)
  const dist = new Float64Array(n * n)
  let worst = 0
  const mf = o.maxField
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const hx = cx + hw * (-1 + (2 * i) / (n - 1))
      const hy = cy + hw * (-1 + (2 * j) / (n - 1))
      const k = j * n + i
      field[k * 2] = hx
      field[k * 2 + 1] = hy
      const fp = o.fieldPointXY(hx * mf, hy * mf)
      const [px, py] = fp.kind === 'angle' ? [fp.tx, fp.ty] : [fp.x, fp.y]
      par[k * 2] = o.chiefImagePerUnit * px
      par[k * 2 + 1] = o.chiefImagePerUnit * py
      const r = o.trace(o.makeRay(fp, 0, 0), w).atImage
      if (r) {
        real[k * 2] = r.p[0]
        real[k * 2 + 1] = r.p[1]
        const rp = Math.hypot(par[k * 2], par[k * 2 + 1])
        dist[k] = rp > 1e-12 ? (100 * (Math.hypot(r.p[0], r.p[1]) - rp)) / rp : 0
        worst = Math.max(worst, Math.abs(dist[k]))
      } else dist[k] = NaN
    }
  const out = newOut({ grid_size: n, max_abs_distortion_percent: nf(worst), wavelength: w })
  addBuf(out, 'field', field, [n * n, 2])
  addBuf(out, 'paraxial', par, [n * n, 2])
  addBuf(out, 'real', real, [n * n, 2])
  addBuf(out, 'distortion_percent', dist, [n * n])
  return out
}

// --- Relative illumination and the field-map family ---------------------------------------

/** Fraction of pupil points blocked for a field point, and the chief hit. */
function pupilFill(o: MockOptics, x: number, y: number, w: number, dense: boolean) {
  const fp = o.fieldPointXY(x, y)
  const pts = pupilSamples('hexapolar', dense ? 3 : 1, 1)
  let blocked = 0
  for (const [px, py] of pts) if (!o.trace(o.makeRay(fp, px, py), w).atImage) blocked++
  const chief = o.trace(o.makeRay(fp, 0, 0), w)
  return { vig: blocked / pts.length, chief: chief.atImage, trace: chief, fp }
}

function chiefAngleDeg(r: Ray | null): number | null {
  if (!r) return null
  return (Math.atan2(Math.hypot(r.d[0], r.d[1]), Math.abs(r.d[2])) * 180) / Math.PI
}

function objectChiefAngle(o: MockOptics, fp: ReturnType<MockOptics['fieldPointXY']>): number {
  if (fp.kind === 'angle') return Math.atan(Math.hypot(fp.tx, fp.ty))
  const q = Math.abs(o.epZ + (o.objectDistance ?? 0))
  return q > 1e-12 ? Math.atan(Math.hypot(fp.x, fp.y) / q) : 0
}

export function relativeIllumination(o: MockOptics, s: Obj): Out {
  const w = wlIndex(o, s.wavelength)
  const { values, coords } = fieldSweep(o, Number(s.field_samples ?? 11), s.field_axes)
  const rows = coords.map(([x, y]) => {
    const r = pupilFill(o, x, y, w, true)
    const cra = chiefAngleDeg(r.chief)
    const cos4 = Math.cos(objectChiefAngle(o, r.fp)) ** 4
    const img = cra === null ? NaN : Math.cos((cra * Math.PI) / 180) ** 4
    return { vig: r.vig, cra, cos4, img }
  })
  const v0 = rows[0]
  const ri = rows.map((r) => (Number.isFinite(r.img) && v0 && 1 - v0.vig > 0 ? (r.img * (1 - r.vig)) / (1 - v0.vig) : NaN))
  let nw = o.workingFNumber(w)
  if (!Number.isFinite(nw)) nw = 10
  const omega0 = Math.PI * (1 / (2 * nw)) ** 2
  const mf = Math.max(...values, 1e-12)
  return newOut({
    chief_ray_angle_deg: nfs(rows.map((r) => r.cra)),
    cos4_object: nfs(rows.map((r) => r.cos4)),
    field_coordinates: coords.map(([x, y]) => pair(x, y)),
    field_unit: o.fieldUnit,
    field_values: values,
    normalized_field: values.map((v) => v / mf),
    projected_solid_angle: nfs(ri.map((v) => v * omega0)),
    relative_illumination: nfs(ri),
    vignetting_fraction: nfs(rows.map((r) => r.vig)),
    wavelength: w,
  })
}

interface MapColumn {
  name: string
  definition: string
  unit: string
}

function mapGrid(o: MockOptics, s: Obj, columns: MapColumn[], compute: (x: number, y: number, hx: number, hy: number, w: number) => (number | null)[], w: number): Out {
  const mode = String(s.mode ?? 'grid')
  const g = Array.isArray(s.grid) ? (s.grid as number[]) : [5, 5]
  const ext = Array.isArray(s.extent) ? (s.extent as number[]) : [1, 1]
  const nx = mode === 'y' ? 1 : Math.min(Math.max(Math.floor(g[0] ?? 5), 1), 41)
  const ny = mode === 'x' ? 1 : Math.min(Math.max(Math.floor(g[1] ?? 5), 1), 41)
  const axis = (n: number, e: number) => (n === 1 ? [0] : linspace(-e, e, n))
  const xs = axis(nx, ext[0] ?? 1)
  const ys = axis(ny, ext[1] ?? 1)
  const C = columns.length
  const fc = new Float64Array(nx * ny * 2)
  const im = new Float64Array(nx * ny * 2).fill(NaN)
  const vals = new Float64Array(nx * ny * C).fill(NaN)
  const mf = o.maxField
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i
      fc[k * 2] = xs[i]
      fc[k * 2 + 1] = ys[j]
      const chief = chiefHit(o, xs[i] * mf, ys[j] * mf, w)
      if (chief) {
        im[k * 2] = chief.p[0]
        im[k * 2 + 1] = chief.p[1]
      }
      const row = compute(xs[i] * mf, ys[j] * mf, xs[i], ys[j], w)
      for (let c = 0; c < C; c++) vals[k * C + c] = row[c] ?? NaN
    }
  const out = newOut({ columns: columns as unknown as Json, field_coordinates: 'field_coordinates', image_mm: 'image_mm', shape: [nx, ny], values: 'values', wavelength: w })
  addBuf(out, 'field_coordinates', fc, [ny, nx, 2])
  addBuf(out, 'image_mm', im, [ny, nx, 2])
  addBuf(out, 'values', vals, [ny, nx, C])
  return out
}

export function chiefRayAngle(o: MockOptics, s: Obj): Out {
  const w = wlIndex(o, s.wavelength)
  const lim = Array.isArray(s.sensor_limit) ? (s.sensor_limit as number[][]) : []
  const limitAt = (r: number): number | null => {
    if (!lim.length) return null
    if (r <= lim[0][0]) return lim[0][1]
    for (let i = 1; i < lim.length; i++) if (r <= lim[i][0]) return lim[i - 1][1] + ((r - lim[i - 1][0]) / (lim[i][0] - lim[i - 1][0])) * (lim[i][1] - lim[i - 1][1])
    return lim[lim.length - 1][1]
  }
  const columns: MapColumn[] = [
    { name: 'cra', unit: 'deg', definition: 'Angle between the chief ray and the image-surface normal.' },
    { name: 'cra_x', unit: 'deg', definition: 'Signed angle of the chief ray in the xz plane.' },
    { name: 'cra_y', unit: 'deg', definition: 'Signed angle of the chief ray in the yz plane.' },
    { name: 'azimuth', unit: 'deg', definition: 'Azimuth of the chief ray direction in the image plane.' },
    { name: 'image_radius', unit: 'mm', definition: 'Radial distance of the chief hit from the axis.' },
    { name: 'margin', unit: 'deg', definition: 'Sensor limit minus the chief ray angle; NaN without a limit.' },
  ]
  return mapGrid(o, s, columns, (x, y) => {
    const c = chiefHit(o, x, y, w)
    if (!c) return []
    const cra = chiefAngleDeg(c) as number
    const rad = Math.hypot(c.p[0], c.p[1])
    const lm = limitAt(rad)
    return [cra, (Math.atan2(c.d[0], Math.abs(c.d[2])) * 180) / Math.PI, (Math.atan2(c.d[1], Math.abs(c.d[2])) * 180) / Math.PI, (Math.atan2(c.d[1], c.d[0]) * 180) / Math.PI, rad, lm === null ? null : lm - cra]
  }, w)
}

export function pupilAberration(o: MockOptics, s: Obj): Out {
  const w = wlIndex(o, s.wavelength)
  const stop = Math.min(Math.max(o.sys.stop, 1), o.sys.surfaces.length - 2)
  const stopSD = o.sys.surfaces[stop].sdFixed ?? o.epRadius
  const columns: MapColumn[] = [
    { name: 'pupil_shift_x', unit: 'mm', definition: 'x offset of the chief ray from the pupil centre, referred to the entrance pupil.' },
    { name: 'pupil_shift_y', unit: 'mm', definition: 'y offset of the chief ray from the pupil centre, referred to the entrance pupil.' },
    { name: 'pupil_scale_x', unit: '1', definition: 'Real pupil half-width in x relative to the paraxial pupil.' },
    { name: 'pupil_scale_y', unit: '1', definition: 'Real pupil half-height in y relative to the paraxial pupil.' },
    { name: 'pupil_z', unit: 'mm', definition: 'Axial position of the entrance pupil seen from this field, from surface 1.' },
  ]
  return mapGrid(o, s, columns, (x, y, hx, hy) => {
    const fp = o.fieldPointXY(x, y)
    const t = o.trace(o.makeRay(fp, 0, 0), w)
    const at = t.points[stop]
    if (!t.ok && !at) return []
    const k = o.epRadius / Math.max(stopSD, 1e-9)
    const edge = (px: number, py: number) => (o.trace(o.makeRay(fp, px, py), w).atImage ? 1 : 0)
    const sx = 0.5 * (edge(1, 0) + edge(-1, 0))
    const sy = 0.5 * (edge(0, 1) + edge(0, -1))
    const h2 = hx * hx + hy * hy
    return [at ? at[0] * k : null, at ? at[1] * k : null, sx || 0.5, sy || 0.5, o.epZ * (1 + 0.04 * h2)]
  }, w)
}

export function relativeIlluminationMap(o: MockOptics, s: Obj): Out {
  const w = wlIndex(o, s.wavelength)
  const columns: MapColumn[] = [
    { name: 'relative_illumination', unit: '1', definition: 'Projected solid angle of the real exit pupil relative to the axis.' },
    { name: 'vignetting_fraction', unit: '1', definition: 'Clipped share of the stop area.' },
    { name: 'cos4_object', unit: '1', definition: 'cos^4 of the object-space chief ray angle.' },
    { name: 'chief_ray_angle', unit: 'deg', definition: 'Chief ray angle of incidence on the image surface.' },
  ]
  const g = Array.isArray(s.grid) ? (s.grid as number[]) : [5, 5]
  const dense = (g[0] ?? 5) * (g[1] ?? 5) <= 100
  const axis = pupilFill(o, 0, 0, w, dense)
  return mapGrid(o, s, columns, (x, y) => {
    const r = pupilFill(o, x, y, w, dense)
    const cra = chiefAngleDeg(r.chief)
    const img = cra === null ? NaN : Math.cos((cra * Math.PI) / 180) ** 4
    const ri = (img * (1 - r.vig)) / Math.max(1 - axis.vig, 1e-9)
    return [Number.isFinite(ri) ? ri : null, r.vig, Math.cos(objectChiefAngle(o, r.fp)) ** 4, cra]
  }, w)
}
