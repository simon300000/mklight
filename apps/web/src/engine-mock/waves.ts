// MOCK ENGINE — synthetic wave-optics results (mtf, psf, wavefront_map).
// They are built from the mock's traced spot sizes and OPD fans so they
// respond to edits; they are not diffraction computations.
import type { Json, JsonObject } from '../engine/types'
import type { MockOptics } from './optics'
import { addBuf, exitPupil, newOut, nf, opdFanData, pair, selectIndices, spotData, type Curve, type Out } from './analyses'
import type { Obj } from './model'

const fieldSel = (o: MockOptics, v: unknown) => selectIndices(v, o.sys.fields.length, 'all')
const wlSel = (o: MockOptics, v: unknown, d: 'all' | 'primary') => selectIndices(v, o.sys.wavelengths.length, d, o.primary)

export function spotFor(o: MockOptics, f: number, wls: number[]) {
  return spotData(o, { fields: [f], wls, pattern: 'hexapolar', rings: 4, seed: 1, defocus: 0, reference: 'chief' })[0]
}

// --- MTF ------------------------------------------------------------------

export function diffractionMtf(s: number): number {
  if (s >= 1) return 0
  if (s <= 0) return 1
  return (2 / Math.PI) * (Math.acos(s) - s * Math.sqrt(1 - s * s))
}

export function mtf(o: MockOptics, s: Obj): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const nS = Math.max(Math.floor(Number(s.frequency_samples ?? 41)), 2)
  let nw = o.workingFNumber(o.primary)
  if (!Number.isFinite(nw)) nw = 10
  const lamMin = Math.min(...wls.map((w) => o.wavelengthsUm[w] * 1e-3))
  const cutoff = 1 / (lamMin * nw)
  // Automatic limit: where the sharpest field's estimate falls to 10 % contrast
  // (the engine estimates 10 % polychromatic contrast).
  let autoF = 0
  for (const f of fields) {
    const st = spotFor(o, f, wls).poly
    const sigma = Math.max(Number(st.rms_x) || 0, Number(st.rms_y) || 0)
    autoF = Math.max(autoF, sigma > 0 ? Math.sqrt(Math.log(10) / 2) / (Math.PI * sigma) : cutoff)
  }
  autoF = Math.min(Math.max(autoF, cutoff * 0.02), cutoff)
  const maxF = typeof s.max_frequency === 'number' && s.max_frequency > 0 ? s.max_frequency : autoF
  const freqs = Float64Array.from({ length: nS }, (_, i) => (i / (nS - 1)) * maxF)
  const out = newOut({})
  addBuf(out, 'frequencies', freqs, [nS])
  const groups: JsonObject[] = []
  fields.forEach((f, k) => {
    const st = spotFor(o, f, wls).poly
    const sx = Number.isFinite(st.rms_x) ? st.rms_x : 0
    const sy = Number.isFinite(st.rms_y) ? st.rms_y : 0
    const dl = new Float64Array(nS), tg = new Float64Array(nS), sg = new Float64Array(nS)
    for (let i = 0; i < nS; i++) {
      const nu = freqs[i]
      const d = diffractionMtf(nu / cutoff)
      dl[i] = d
      tg[i] = d * Math.exp(-2 * Math.PI * Math.PI * nu * nu * sy * sy)
      sg[i] = d * Math.exp(-2 * Math.PI * Math.PI * nu * nu * sx * sx)
    }
    addBuf(out, `tangential_${k}`, tg, [nS])
    addBuf(out, `sagittal_${k}`, sg, [nS])
    addBuf(out, `diffraction_limit_${k}`, dl, [nS])
    const fx = o.sys.fields[f].x
    const fy = o.sys.fields[f].y
    const len = Math.hypot(fx, fy)
    groups.push({
      cutoff,
      diffraction_limit: `diffraction_limit_${k}`,
      field: f,
      field_value: pair(fx, fy),
      pupil_sampling: wls.map((w) => ({ grid: 65, wavelength: w })),
      sagittal: `sagittal_${k}`,
      tangential: `tangential_${k}`,
      tangential_direction: len > 0 ? pair(fx / len, fy / len) : pair(0, 1),
    })
  })
  out.json = { fields: groups, frequencies: 'frequencies', nyquist: null }
  return out
}

// --- Wavefront model ------------------------------------------------------

function interp(c: Curve, p: number): number {
  const xs = c.pupil
  if (p <= xs[0]) return c.values[0] ?? 0
  for (let i = 1; i < xs.length; i++) {
    if (p <= xs[i]) {
      const a = c.values[i - 1] ?? 0
      const b = c.values[i] ?? 0
      const t = (p - xs[i - 1]) / (xs[i] - xs[i - 1])
      return a + t * (b - a)
    }
  }
  return c.values[xs.length - 1] ?? 0
}

/** W(ρ, θ) in waves: even tangential/sagittal parts plus the odd (coma-like) tangential part. */
function wavefrontFn(o: MockOptics, f: number, w: number): (u: number, v: number) => number {
  const fan = opdFanData(o, [f], [w], 41)[0]
  const T = fan.tangential[0]
  const S = fan.sagittal[0]
  return (u, v) => {
    const r = Math.hypot(u, v)
    if (r === 0) return 0
    const c2 = (u * u) / (r * r)
    const s2 = (v * v) / (r * r)
    const sinT = v / r
    const et = (interp(T, r) + interp(T, -r)) / 2
    const ot = (interp(T, r) - interp(T, -r)) / 2
    const es = (interp(S, r) + interp(S, -r)) / 2
    return es * c2 + et * s2 + ot * sinT
  }
}

function solve(A: number[][], b: number[]): number[] {
  const n = b.length
  const M = A.map((r, i) => [...r, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    ;[M[c], M[p]] = [M[p], M[c]]
    const d = M[c][c] || 1e-30
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / d
      if (f !== 0) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  const x = new Array<number>(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let v = M[r][n]
    for (let k = r + 1; k < n; k++) v -= M[r][k] * x[k]
    x[r] = v / (M[r][r] || 1e-30)
  }
  return x
}

const fact = (n: number): number => (n <= 1 ? 1 : n * fact(n - 1))

interface ZTerm {
  n: number
  m: number
  sin: boolean
}
function zernikeTerms(count: number): ZTerm[] {
  const out: ZTerm[] = []
  for (let d = 0; out.length < count; d += 2) {
    for (let n = Math.ceil(d / 2); n <= d && out.length < count; n++) {
      const m = d - n
      if (m > n || (n - m) % 2) continue
      out.push({ n, m, sin: false })
      if (m > 0 && out.length < count) out.push({ n, m, sin: true })
    }
  }
  return out
}
function zernikeEval(t: ZTerm, rho: number, theta: number): number {
  let r = 0
  for (let s = 0; s <= (t.n - t.m) / 2; s++)
    r += ((s % 2 ? -1 : 1) * fact(t.n - s)) / (fact(s) * fact((t.n + t.m) / 2 - s) * fact((t.n - t.m) / 2 - s)) * Math.pow(rho, t.n - 2 * s)
  if (t.m === 0) return r
  return r * (t.sin ? Math.sin(t.m * theta) : Math.cos(t.m * theta))
}

interface WfMap {
  n: number
  coords: Float64Array
  opd: Float64Array
  amp: Float64Array
  rms: number
  rmsTR: number
  pv: number
  pvTR: number
  pts: { u: number; v: number; w: number }[]
}

export function buildMap(o: MockOptics, f: number, w: number, n: number): WfMap {
  const fn = wavefrontFn(o, f, w)
  const coords = Float64Array.from({ length: n }, (_, i) => -1 + (2 * i + 1) / n)
  const opd = new Float64Array(n * n).fill(NaN)
  const amp = new Float64Array(n * n)
  const pts: { u: number; v: number; w: number }[] = []
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const u = coords[i], v = coords[j]
      if (u * u + v * v > 1) continue
      const val = fn(u, v)
      opd[j * n + i] = val
      amp[j * n + i] = 1
      pts.push({ u, v, w: val })
    }
  const N = Math.max(pts.length, 1)
  const mean = pts.reduce((a, p) => a + p.w, 0) / N
  const rms = Math.sqrt(pts.reduce((a, p) => a + (p.w - mean) ** 2, 0) / N)
  const ws = pts.map((p) => p.w)
  const pv = Math.max(...ws) - Math.min(...ws)
  // Remove piston and best-fit tilt.
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  const b = [0, 0, 0]
  for (const p of pts) {
    const q = [1, p.u, p.v]
    for (let r = 0; r < 3; r++) {
      b[r] += q[r] * p.w
      for (let c = 0; c < 3; c++) A[r][c] += q[r] * q[c]
    }
  }
  const [a0, a1, a2] = solve(A, b)
  const res = pts.map((p) => p.w - (a0 + a1 * p.u + a2 * p.v))
  const rm = res.reduce((a, x) => a + x, 0) / N
  const rmsTR = Math.sqrt(res.reduce((a, x) => a + (x - rm) ** 2, 0) / N)
  const pvTR = Math.max(...res) - Math.min(...res)
  return { n, coords, opd, amp, rms, rmsTR, pv, pvTR, pts }
}

export function wavefrontMap(o: MockOptics, s: Obj): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'primary')
  const nTerms = Math.min(Math.max(Math.floor(Number(s.zernike_terms ?? 37)), 1), 231)
  const grid = Number(s.pupil?.grid) > 0 ? Math.min(Number(s.pupil.grid), 129) : 41
  const ordering = s.zernike_ordering === 'standard' ? 'standard' : 'fringe'
  const terms = zernikeTerms(nTerms)
  const out = newOut({})
  const maps: JsonObject[] = []
  const xp = exitPupil(o)
  let k = 0
  for (const f of fields)
    for (const w of wls) {
      const m = buildMap(o, f, w, grid)
      // Zernike least-squares fit on the lattice.
      const basis = m.pts.map((p) => {
        const r = Math.hypot(p.u, p.v), th = Math.atan2(p.v, p.u)
        return terms.map((t) => zernikeEval(t, r, th))
      })
      const A = terms.map(() => new Array<number>(terms.length).fill(0))
      const b = new Array<number>(terms.length).fill(0)
      basis.forEach((q, i) => {
        for (let r = 0; r < terms.length; r++) {
          b[r] += q[r] * m.pts[i].w
          for (let c = 0; c <= r; c++) A[r][c] += q[r] * q[c]
        }
      })
      for (let r = 0; r < terms.length; r++) {
        for (let c = r + 1; c < terms.length; c++) A[r][c] = A[c][r]
        A[r][r] += 1e-12
      }
      const coef = solve(A, b)
      const resid = Math.sqrt(m.pts.reduce((a, p, i) => a + (p.w - basis[i].reduce((x, v, j) => x + v * coef[j], 0)) ** 2, 0) / Math.max(m.pts.length, 1))
      const c = (i: number) => coef[i] ?? 0
      const ch = o.trace(o.makeRay(o.fieldPoint(f), 0, 0), w).atImage?.p ?? [0, 0, 0]
      addBuf(out, `opd_waves_${k}`, m.opd, [m.n, m.n])
      addBuf(out, `coordinates_${k}`, m.coords, [m.n])
      addBuf(out, `amplitude_${k}`, m.amp, [m.n, m.n])
      addBuf(out, `coefficients_${k}`, Float64Array.from(coef), [nTerms])
      const imgN = Math.abs(o.imageN(w))
      const nwf = o.workingFNumber(w)
      maps.push({
        amplitude: `amplitude_${k}`,
        coordinates: `coordinates_${k}`,
        exit_pupil: {
          chief_direction: [0, 0, 1],
          chief_point_mm: [nf(ch[0]), nf(ch[1]), 0],
          clear_area: 1,
          effective_f_number: nf(nwf),
          exit_pupil_point_mm: null,
          image_index: imgN,
          lattice_rays: m.pts.length,
          numerical_aperture: nf(Number.isFinite(nwf) ? 1 / (2 * nwf) : 0),
          paraxial_position_mm: nf(xp.distance),
          paraxial_radius_mm: nf(xp.radius),
          propagation_sign: 1,
          reference: 'chief',
          reference_point_mm: [nf(ch[0]), nf(ch[1]), 0],
          reference_radius_mm: nf(xp.distance),
          rim_points: 0,
          sin_max: Number.isFinite(nwf) ? 1 / (2 * nwf * imgN) : 0,
          surviving_rays: m.pts.length,
          telecentric: !Number.isFinite(xp.distance),
          transmitted_fraction: 1,
          wavelength_um: o.wavelengthsUm[w],
        },
        field: f,
        field_value: pair(o.sys.fields[f].x, o.sys.fields[f].y),
        grid: m.n,
        opd_waves: `opd_waves_${k}`,
        pv_waves: nf(m.pv),
        pv_waves_tilt_removed: nf(m.pvTR),
        rim: [],
        rms_waves: nf(m.rms),
        rms_waves_tilt_removed: nf(m.rmsTR),
        seidel: {
          astigmatism: Math.hypot(c(4), c(5)),
          astigmatism_angle_deg: (Math.atan2(c(5), c(4)) * 90) / Math.PI,
          coma: Math.hypot(c(6), c(7)),
          coma_angle_deg: (Math.atan2(c(6), c(7)) * 180) / Math.PI,
          focus: c(3),
          piston: c(0),
          spherical: c(8),
          tilt: Math.hypot(c(1), c(2)),
          tilt_angle_deg: (Math.atan2(c(2), c(1)) * 180) / Math.PI,
        },
        strehl_at_reference: Math.exp(-((2 * Math.PI * m.rmsTR) ** 2)),
        strehl_marechal: Math.exp(-((2 * Math.PI * m.rmsTR) ** 2)),
        wavelength: w,
        wavelength_um: o.wavelengthsUm[w],
        zernike: { coefficients: `coefficients_${k}`, ordering },
        zernike_condition: 1,
        zernike_residual_rms: resid,
      })
      k++
    }
  out.json = { maps }
  return out
}

// --- PSF ------------------------------------------------------------------

export function j1(x: number): number {
  const ax = Math.abs(x)
  if (ax < 8) {
    const y = x * x
    const a1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.4826 + y * -30.16036606)))))
    const a2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))))
    return a1 / a2
  }
  const z = 8 / ax
  const y = z * z
  const xx = ax - 2.356194491
  const a1 = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)))
  const a2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
  const ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * a1 - z * Math.sin(xx) * a2)
  return x < 0 ? -ans : ans
}

export function psf(o: MockOptics, s: Obj): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const poly = s.polychromatic !== false
  const padding = typeof s.padding === 'number' && s.padding > 0 ? s.padding : 2.5
  const size = Number(s.grid_size) > 0 ? Math.min(Math.max(Math.floor(Number(s.grid_size)), 8), 512) : 64
  const unitEnergy = s.normalization === 'unit_energy'
  let nw = o.workingFNumber(o.primary)
  if (!Number.isFinite(nw)) nw = 10
  const out = newOut({})
  const psfs: JsonObject[] = []
  let k = 0
  const groups: (number | null)[] = poly ? [null] : wls
  for (const f of fields)
    for (const g of groups) {
      const wlist = g === null ? wls : [g]
      const w0 = g === null ? o.primary : g
      const lamUm = g === null ? Math.min(...wls.map((w) => o.wavelengthsUm[w])) : o.wavelengthsUm[g]
      const lam = lamUm * 1e-3
      const lamN = lam * nw
      const spacing = typeof s.spacing_mm === 'number' && s.spacing_mm > 0 ? s.spacing_mm : lamN / padding
      const sp = spotFor(o, f, wlist)
      const rmsW = buildMap(o, f, w0, 25).rmsTR
      const S = Math.exp(-((2 * Math.PI * rmsW) ** 2))
      const sigma = Math.max(Number.isFinite(sp.poly.rms_x) ? sp.poly.rms_x : 0, 0.5 * lamN)
      const Ea = (4 * lamN * lamN) / Math.PI
      const haloW = ((1 - S) * Ea) / (2 * Math.PI * sigma * sigma)
      const half = Math.floor(size / 2)
      const scale = unitEnergy ? 1 / Ea : 1
      const data = new Float64Array(size * size)
      for (let j = 0; j < size; j++)
        for (let i = 0; i < size; i++) {
          const r = Math.hypot((i - half) * spacing, (j - half) * spacing)
          const x = (Math.PI * r) / lamN
          const air = x < 1e-9 ? 1 : (2 * j1(x) / x) ** 2
          data[j * size + i] = scale * (S * air + haloW * Math.exp(-(r * r) / (2 * sigma * sigma)))
        }
      const peak = data[half * size + half]
      addBuf(out, `data_${k}`, data, [size, size])
      psfs.push({
        centroid_mm: pair(0, 0),
        field: f,
        field_value: pair(o.sys.fields[f].x, o.sys.fields[f].y),
        peak_mm: pair(0, 0),
        psf: {
          center_mm: pair(0, 0),
          data: `data_${k}`,
          diffraction_peak: scale,
          normalization: unitEnergy ? 'unit_energy' : 'strehl',
          size,
          spacing_mm: spacing,
          total_energy: unitEnergy ? 1 : Ea,
          warnings: [] as Json[],
          wavelength_um: g === null ? null : lamUm,
        },
        pupil_sampling: wlist.map((w) => ({ grid: 65, wavelength: w })),
        reference_mm: pair(sp.chief[0], sp.chief[1]),
        strehl: Math.min(Math.max(peak / scale, 0), 1),
        wavelength: g,
      })
      k++
    }
  out.json = { psfs }
  return out
}
