// MOCK ENGINE — materials, radiometry, polarisation, ghosts, stray light,
// image simulation and the rng stream check. Correctly shaped (docs/RESULTS.md),
// synthetic numbers that respond to the design; none is an optical computation.
import { EngineError, type Json, type JsonObject } from '../engine/types'
import { MockOptics, cauchy, linspace, materialNdVd, shapeCurvature } from './optics'
import { addBuf, airyRadius, newOut, nf, pair, pupilSamples, spotData, type Out } from './analyses'
import { CATALOG, type Obj } from './model'
import { LINE_E, LINE_G, catalogOf, glassEntry, pgfOf } from './glass'
import { fieldScale, paraxialFocus } from './geo'

const wlIndex = (o: MockOptics, v: unknown): number => (typeof v === 'number' && Number.isInteger(v) && v >= 0 && v < o.sys.wavelengths.length ? v : o.primary)

// --- Glass map and glass info ------------------------------------------------------------

export function glassMap(o: MockOptics, s: Obj): Out {
  const meta: Obj[] = Array.isArray(s.metadata) ? (s.metadata as Obj[]) : []
  const users = new Map<string, number[]>()
  o.sys.surfaces.forEach((sf, i) => {
    const m = sf.material
    if (sf.interaction === 'reflect' || m.kind === 'air' || m.kind === 'vacuum' || m.kind === 'constant') return
    const key = m.kind === 'catalog' ? `${m.name.toUpperCase()}|${m.dnd}|${m.dvd}` : `MODEL|${i}`
    users.set(key, [...(users.get(key) ?? []), i])
  })
  const glasses: JsonObject[] = []
  const coords: number[] = []
  const add = (g: JsonObject, nd: number, vd: number, pgf: number) => {
    glasses.push({ ...g, nd: nf(nd), vd: nf(vd), pgf: nf(pgf) })
    coords.push(nd, vd, pgf)
  }
  for (const [name, g] of Object.entries(CATALOG)) {
    const ov = meta.find((m) => String(m.name).toUpperCase() === name)
    add(
      {
        catalog: catalogOf(g),
        climate_resistance: null,
        exclude_substitution: ov ? ov.exclude_substitution === true : false,
        kind: 'catalog',
        name,
        offset: { dn_d: 0, dv_d: 0 },
        relative_cost: null,
        stain_resistance: null,
        status: ov?.status ?? g.status ?? 'standard',
        surfaces: users.get(`${name}|0|0`) ?? [],
        transmittance: [{ um: 0.4, thickness_mm: 10, tau: 0.99 }, { um: 0.55, thickness_mm: 10, tau: 0.999 }, { um: 0.7, thickness_mm: 10, tau: 0.9995 }],
      },
      g.nd,
      g.vd,
      pgfOf(name, g.vd),
    )
  }
  o.sys.surfaces.forEach((sf, i) => {
    const m = sf.material
    if (m.kind === 'catalog') {
      if (m.dnd === 0 && m.dvd === 0) return
      const g = glassEntry(m.name)
      if (!g) return
      add(
        { catalog: catalogOf(g), climate_resistance: null, exclude_substitution: false, kind: 'offset', name: m.name.toUpperCase(), offset: { dn_d: m.dnd, dv_d: m.dvd }, relative_cost: null, stain_resistance: null, status: null, surfaces: [i], transmittance: [] },
        g.nd + m.dnd,
        g.vd + m.dvd,
        pgfOf(m.name, g.vd + m.dvd),
      )
    } else if (m.kind === 'model') {
      add({ catalog: null, kind: 'model', name: `MODEL ${i}`, offset: { dn_d: 0, dv_d: 0 }, relative_cost: null, status: null, surfaces: [i], transmittance: [] }, m.nd, m.vd, pgfOf(`MODEL ${i}`, m.vd))
    }
  })
  const out = newOut({ coordinates: 'coordinates', glasses })
  addBuf(out, 'coordinates', Float64Array.from(coords), [glasses.length, 3])
  return out
}

const RANGE_UM: [number, number] = [0.31, 2.5]

export function glassInfo(o: MockOptics, s: Obj): Out {
  const name = String(s.name ?? '').toUpperCase()
  const g = glassEntry(name)
  if (!g || (typeof s.catalog === 'string' && s.catalog.toUpperCase() !== catalogOf(g).toUpperCase())) throw new EngineError('failed', `unknown glass '${String(s.name ?? '')}'`)
  const dn = Number(s.offset?.dn_d ?? 0)
  const dv = Number(s.offset?.dv_d ?? 0)
  const nd = g.nd + dn
  const vd = g.vd + dv
  const n = (um: number) => cauchy(nd, Math.max(vd, 1), um)
  const env = (o.sys.raw.environment ?? {}) as Obj
  const wls = Array.isArray(s.wavelengths_um) ? (s.wavelengths_um as number[]) : o.wavelengthsUm
  const alpha = (um: number) => 0.0005 * Math.pow(0.55 / um, 3) * (1 + (nd - 1.5) * 3)
  const samples = Number(s.curve_samples) > 0 ? Math.floor(Number(s.curve_samples)) : 101
  const curve = new Float64Array(samples * 2)
  linspace(RANGE_UM[0], RANGE_UM[1], samples).forEach((um, i) => {
    curve[i * 2] = um
    curve[i * 2 + 1] = n(um)
  })
  const pgf = (n(LINE_G) - n(0.4861327)) / (n(0.4861327) - n(0.6562725))
  const out = newOut({
    catalog: catalogOf(g),
    catalog_nd: g.nd,
    catalog_vd: g.vd,
    comment: null,
    cte: null,
    density: null,
    dispersion: { formula: 'cauchy_fit', coefficients: { nd: g.nd, vd: g.vd } },
    environment: { temperature_c: typeof s.temperature_c === 'number' ? s.temperature_c : Number(env.temperature_c ?? 20), pressure_atm: typeof s.pressure_atm === 'number' ? s.pressure_atm : Number(env.pressure_atm ?? 1) },
    has_thermal_data: false,
    indices: wls.map((um) => ({ wavelength_um: um, n: nf(n(um)), n_catalog: nf(n(um)), in_range: um >= RANGE_UM[0] && um <= RANGE_UM[1], absorption_per_mm: alpha(um) })),
    max_um: RANGE_UM[1],
    min_um: RANGE_UM[0],
    name,
    nd,
    ne: n(LINE_E),
    offset: { dn_d: dn, dv_d: dv },
    pgf: nf(Number.isFinite(pgf) ? pgf : pgfOf(name, vd)),
    relative_cost: null,
    vd: nf(vd),
  })
  addBuf(out, 'dispersion_curve', curve, [samples, 2])
  return out
}

// --- Transmission and polarisation ----------------------------------------------------------

function fresnel(n1: number, n2: number, sinT: number): { ts: number; tp: number } {
  const s1 = Math.min(Math.max(sinT, 0), 0.999)
  const c1 = Math.sqrt(1 - s1 * s1)
  const sin2 = (n1 / n2) * s1
  if (sin2 >= 1) return { ts: 0, tp: 0 }
  const c2 = Math.sqrt(1 - sin2 * sin2)
  const rs = ((n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)) ** 2
  const rp = ((n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)) ** 2
  return { ts: 1 - rs, tp: 1 - rp }
}

const bulkAlpha = (nd: number, um: number) => 0.0005 * Math.pow(0.55 / um, 3) * (1 + (nd - 1.5) * 3)

/** Interface and bulk transmission of one traced ray, normal-incidence Fresnel with an incidence-angle estimate. */
function rayTransmission(o: MockOptics, w: number, heights: Map<number, number>): { ts: number; tp: number; sinMax: number; retard: number } {
  let ts = 1
  let tp = 1
  let sinMax = 0
  let retard = 0
  const n = o.sys.surfaces.length
  for (let i = 1; i < n - 1; i++) {
    const sf = o.sys.surfaces[i]
    if (sf.isCB || sf.interaction === 'dummy') continue
    const c = Math.abs(shapeCurvature(sf.shape))
    const sinT = Math.min((heights.get(i) ?? 0) * c, 0.95)
    if (sf.interaction === 'reflect') {
      ts *= 0.93
      tp *= 0.92
      continue
    }
    const n1 = o.nBefore(i, w)
    const n2 = o.nAfter[i][w]
    if (Math.abs(n1 - n2) > 1e-9) {
      const f = fresnel(n1, n2, sinT)
      ts *= f.ts
      tp *= f.tp
      retard += 0.02 * sinT * sinT
    }
    sinMax = Math.max(sinMax, sinT)
    if (n2 > 1.0001) {
      const nd = materialNdVd(sf.material).nd
      const um = o.wavelengthsUm[w]
      const t = Number.isFinite(sf.thickness) ? sf.thickness : 0
      const a = Math.exp(-bulkAlpha(nd, um) * t)
      ts *= a
      tp *= a
    }
  }
  return { ts, tp, sinMax, retard }
}

export function systemTransmission(o: MockOptics, s: Obj): Out {
  const mf = o.maxField
  const given: Obj[] = Array.isArray(s.fields) ? (s.fields as Obj[]) : []
  const pts: { index: number; hx: number; hy: number; x: number; y: number }[] = given.length
    ? given.map((p, i) => ({ index: i, hx: Number(p.hx ?? 0), hy: Number(p.hy ?? 0), x: Number(p.hx ?? 0) * mf, y: Number(p.hy ?? 0) * mf }))
    : o.sys.fields.map((f, i) => ({ index: i, hx: f.x / mf, hy: f.y / mf, x: f.x, y: f.y }))
  const wls: number[] = Array.isArray(s.wavelengths) && s.wavelengths.length ? (s.wavelengths as number[]).filter((w) => w >= 0 && w < o.wavelengthsUm.length) : o.wavelengthsUm.map((_, i) => i)
  const grid = Math.min(Math.max(Math.floor(Number((s.pupil as Obj | undefined)?.n ?? 32)), 3), 16)
  const pupil = pupilSamples('square', Math.ceil(grid / 2), 1)
  const points: JsonObject[] = []
  const poly: number[] = []
  for (const p of pts) {
    const fp = o.fieldPointXY(p.x, p.y)
    let acc = 0
    let wsum = 0
    for (const w of wls) {
      let sum = 0
      let reached = 0
      let sumReached = 0
      for (const [px, py] of pupil) {
        const t = o.trace(o.makeRay(fp, px, py), w)
        if (!t.atImage) continue
        reached++
        const r = rayTransmission(o, w, t.heights)
        const v = 0.5 * (r.ts + r.tp)
        sum += v
        sumReached += v
      }
      const tr = sum / pupil.length
      const weight = o.sys.wavelengths[w].weight
      acc += tr * weight
      wsum += weight
      points.push({
        field: p.index,
        wavelength: w,
        wavelength_um: o.wavelengthsUm[w],
        transmission: tr,
        mean_unvignetted: reached ? sumReached / reached : 0,
        unvignetted_fraction: reached / pupil.length,
        samples: pupil.length,
        failed: 0,
      })
    }
    poly.push(wsum > 0 ? acc / wsum : 0)
  }
  return newOut({
    fields: pts.map((p) => ({ index: p.index, hx: p.hx, hy: p.hy, field_value: pair(p.x, p.y) })),
    points,
    polychromatic: poly,
    wavelengths: wls,
  })
}

export function polarizationPupilMap(o: MockOptics, s: Obj): Out {
  const w = wlIndex(o, s.wavelength)
  const n = Math.min(Math.max(Math.floor(Number(s.grid ?? 33)), 1), 33)
  const mf = o.maxField
  let hx = 0
  let hy = 0
  if (Array.isArray(s.point) && s.point.length === 2) [hx, hy] = s.point as number[]
  else {
    const f = o.sys.fields[Math.min(Math.max(Math.floor(Number(s.field ?? 0)), 0), o.sys.fields.length - 1)]
    hx = f.x / mf
    hy = f.y / mf
  }
  const fp = o.fieldPointXY(hx * mf, hy * mf)
  const axis = n === 1 ? [0] : linspace(-1, 1, n)
  const pupil: [number, number][] = []
  for (const y of axis) for (const x of axis) if (x * x + y * y <= 1 + 1e-12) pupil.push([x, y])
  const N = pupil.length
  const jones = new Float64Array(N * 8)
  const reached = new Uint8Array(N)
  const inten = new Float64Array(N)
  const dia = new Float64Array(N)
  const diaAxis = new Float64Array(N * 3)
  const ret = new Float64Array(N)
  const fast = new Float64Array(N * 3)
  const defined = new Uint8Array(N)
  const retJ = new Float64Array(N)
  const tmax = new Float64Array(N)
  const tmin = new Float64Array(N)
  let maxD = 0, maxR = 0, sumI = 0, sumR2 = 0, nReach = 0
  pupil.forEach(([px, py], k) => {
    const t = o.trace(o.makeRay(fp, px, py), w)
    if (!t.atImage) return
    const r = rayTransmission(o, w, t.heights)
    reached[k] = 1
    defined[k] = 1
    const hi = Math.max(r.ts, r.tp)
    const lo = Math.min(r.ts, r.tp)
    tmax[k] = hi
    tmin[k] = lo
    dia[k] = hi + lo > 0 ? (hi - lo) / (hi + lo) : 0
    const phi = Math.atan2(py, px)
    diaAxis[k * 3] = Math.cos(2 * phi)
    diaAxis[k * 3 + 1] = Math.sin(2 * phi)
    fast[k * 3] = Math.cos(2 * phi)
    fast[k * 3 + 1] = Math.sin(2 * phi)
    ret[k] = Math.min(r.retard, Math.PI)
    retJ[k] = ret[k]
    inten[k] = 0.5 * (hi + lo)
    const a = Math.sqrt(r.ts)
    const b = Math.sqrt(r.tp)
    const c2 = Math.cos(phi) ** 2
    const s2 = Math.sin(phi) ** 2
    const cs = Math.sin(phi) * Math.cos(phi)
    const half = ret[k] / 2
    const m = [a * c2 + b * s2, (a - b) * cs, (a - b) * cs, a * s2 + b * c2]
    for (let q = 0; q < 4; q++) {
      jones[k * 8 + q * 2] = m[q] * Math.cos(half)
      jones[k * 8 + q * 2 + 1] = m[q] * Math.sin(half)
    }
    maxD = Math.max(maxD, dia[k])
    maxR = Math.max(maxR, ret[k])
    sumI += inten[k]
    sumR2 += ret[k] * ret[k]
    nReach++
  })
  const out = newOut({
    field: { hx, hy, vignetting: { vdx: 0, vdy: 0, vcx: 0, vcy: 0 } },
    grid: n,
    max_diattenuation: maxD,
    max_retardance: maxR,
    mean_intensity: nReach ? sumI / nReach : 0,
    rms_retardance: nReach ? Math.sqrt(sumR2 / nReach) : 0,
    sample_count: N,
    undefined_retarder: 0,
    wavelength: w,
    wavelength_um: o.wavelengthsUm[w],
  })
  addBuf(out, 'pupil', Float64Array.from(pupil.flat()), [N, 2])
  addBuf(out, 'reached', reached, [N])
  addBuf(out, 'jones', jones, [N, 2, 2, 2])
  addBuf(out, 'intensity', inten, [N])
  addBuf(out, 'diattenuation', dia, [N])
  addBuf(out, 'diattenuation_axis', diaAxis, [N, 3])
  addBuf(out, 'retardance', ret, [N])
  addBuf(out, 'fast_axis', fast, [N, 3])
  addBuf(out, 'retarder_defined', defined, [N])
  addBuf(out, 'retardance_jones', retJ, [N])
  addBuf(out, 't_max', tmax, [N])
  addBuf(out, 't_min', tmin, [N])
  return out
}

// --- Ghosts -----------------------------------------------------------------------------------

interface GhostPath {
  label: string
  path: JsonObject
  reflections: number[]
  rProduct: number
  transmittance: number
  focus: number
  spot: number
  imageHeight: number
  relative: number
}

/** Paraxial trace of a ghost path (reflect at j going out, at i coming back). */
function ghostRay(o: MockOptics, w: number, refl: [number, number], y1: number, u0: number): { yImg: number; focus: number } | null {
  const surfaces = o.sys.surfaces
  const n = surfaces.length
  const image = n - 1
  const [j, i] = refl
  const seq: { s: number; r: boolean }[] = []
  for (let k = 1; k < j; k++) seq.push({ s: k, r: false })
  seq.push({ s: j, r: true })
  for (let k = j - 1; k > i; k--) seq.push({ s: k, r: false })
  seq.push({ s: i, r: true })
  for (let k = i + 1; k < j; k++) seq.push({ s: k, r: false })
  if (j < image) seq.push({ s: j, r: false })
  for (let k = Math.max(j, i) + 1; k < image; k++) seq.push({ s: k, r: false })
  const medium = (idx: number) => o.nAfter[Math.max(idx, 0)][w]
  let dir = 1
  let nIn = medium(0)
  let y = y1
  let nu = nIn * u0
  let z = o.vertexZ[1]
  let first = true
  for (const st of seq) {
    const sf = surfaces[st.s]
    const zs = o.vertexZ[st.s]
    if (!first) y += (Math.abs(zs - z) / nIn) * nu
    first = false
    z = zs
    const c = sf.isCB || sf.interaction === 'dummy' ? 0 : shapeCurvature(sf.shape)
    let nOut: number
    if (st.r) nOut = -nIn
    else nOut = dir * (dir > 0 ? medium(st.s) : medium(st.s - 1))
    nu -= y * c * (nOut - nIn)
    if (st.r) dir = -dir
    nIn = nOut
  }
  const k = n - 2
  const lastZ = o.vertexZ[k]
  y += (Math.abs(lastZ - z) / nIn) * nu
  const t = surfaces[k].thickness
  const yImg = y + (t / nIn) * nu
  if (!(Math.abs(nu) > 1e-15)) return { yImg, focus: Infinity }
  const zf = (-y * nIn) / nu
  return { yImg, focus: zf - t }
}

export function enumerateGhosts(o: MockOptics, s: Obj, w: number): { ghosts: GhostPath[]; primaryT: number; signal: number; primaryFocus: number; primaryHeight: number; maxField: number; interfaces: string[] } {
  const surfaces = o.sys.surfaces
  const n = surfaces.length
  const rOf = (i: number): number => {
    const sf = surfaces[i]
    if (sf.isCB || sf.interaction === 'dummy') return 0
    if (sf.interaction === 'reflect') return 0.9
    const n1 = o.nBefore(i, w)
    const n2 = o.nAfter[i][w]
    return ((n1 - n2) / (n1 + n2)) ** 2
  }
  const sensorR = typeof s.sensor_reflectance === 'number' ? s.sensor_reflectance : 0
  const enumeration = (s.enumeration ?? {}) as Obj
  const exclude = new Set<number>(Array.isArray(enumeration.exclude_surfaces) ? (enumeration.exclude_surfaces as number[]) : [])
  const minProd = Number(enumeration.min_reflectance_product ?? 0)
  const T = (i: number) => 1 - rOf(i)
  let primaryT = 1
  for (let i = 1; i < n - 1; i++) primaryT *= T(i)
  const m0 = o.marginalStart()
  const k = fieldScale(o)
  const c0 = o.objectDistance !== null ? { y1: k * (1 + (-1 / o.objectToPupil(o.objectDistance)) * o.objectDistance), u0: (k * -1) / o.objectToPupil(o.objectDistance) } : { y1: -o.epZ * k, u0: k }
  const primaryFocus = paraxialFocus(o, w) ?? 0
  const primaryHeight = Math.abs(o.chiefImagePerUnit * o.fieldParameterY(o.fieldPointXY(0, o.maxField)))
  const signal = typeof s.signal_diameter_mm === 'number' && s.signal_diameter_mm > 0 ? s.signal_diameter_mm : Math.max(2 * airyRadius(o), 0.02)
  const nearFactor = Number(s.near_focus_factor ?? 10)
  const ghosts: GhostPath[] = []
  const explicit: number[][] | null = Array.isArray(s.paths) ? (s.paths as unknown[]).map((p) => (Array.isArray(p) ? (p as number[]) : ((p as Obj)?.reflections as number[]) ?? [])) : null
  const pairs: [number, number][] = []
  if (explicit) for (const e of explicit) if (e.length === 2 && e[0] > e[1]) pairs.push([e[0], e[1]])
  if (!explicit) {
    for (let j = 2; j <= n - 2 + (sensorR > 0 ? 1 : 0); j++)
      for (let i = 1; i < j; i++) if (!exclude.has(i) && !exclude.has(j)) pairs.push([j, i])
  }
  for (const [j, i] of pairs) {
    const rj = j === n - 1 ? sensorR : rOf(j)
    const ri = rOf(i)
    const rProd = rj * ri
    if (!(rProd > 0) || rProd < minProd) continue
    let tr = 1
    for (let q = 1; q < n - 1; q++) {
      const passes = q < i ? 1 : q === i ? 1 : q < j ? 3 : q === j ? 1 : 1
      tr *= T(q) ** passes
    }
    const mg = ghostRay(o, w, [j, i], m0.y1, m0.u0)
    const ch = ghostRay(o, w, [j, i], c0.y1, c0.u0)
    if (!mg) continue
    const spot = 2 * Math.abs(mg.yImg)
    const eff = Math.max(spot, signal)
    ghosts.push({
      label: `R${j}·R${i}`,
      path: { reflections: [j, i] },
      reflections: [j, i],
      rProduct: rProd,
      transmittance: tr,
      focus: mg.focus,
      spot,
      imageHeight: ch ? Math.abs(ch.yImg) : 0,
      relative: (rProd * tr * (signal / eff) ** 2) / Math.max(primaryT, 1e-12),
    })
  }
  void nearFactor
  ghosts.sort((a, b) => b.relative - a.relative)
  const interfaces = surfaces.map((sf, i) => (i === 0 ? 'object' : i === n - 1 ? 'image' : sf.interaction === 'reflect' ? 'mirror' : sf.shape.type === 'paraxial' ? 'ideal_lens' : Math.abs(o.nBefore(i, w) - o.nAfter[i][w]) > 1e-9 ? 'refractive' : 'passive'))
  return { ghosts, primaryT, signal, primaryFocus, primaryHeight, maxField: o.maxField, interfaces }
}

function ghostCoverage(o: MockOptics, ghosts: GhostPath[], interfaces: string[], s: Obj): JsonObject {
  const clip = (s.clip ?? {}) as Obj
  return {
    apertures: o.sys.surfaces.map((sf, i) => ({ surface: i, clipped: i > 0 && i < o.sys.surfaces.length - 1, semi_diameter: sf.sdFixed, margin_mm: Number(clip.margin_mm ?? 0) })),
    interfaces,
    paths: ghosts.map((g) => g.path),
  }
}

export function ghostPaths(o: MockOptics, s: Obj): Out {
  const w = wlIndex(o, s.wavelength)
  const e = enumerateGhosts(o, s, w)
  const near = Number(s.near_focus_factor ?? 10)
  return newOut({
    coverage: ghostCoverage(o, e.ghosts, e.interfaces, s),
    ghosts: e.ghosts.map((g) => ({
      focus_from_sensor: nf(g.focus),
      image_height: nf(g.imageHeight),
      label: g.label,
      near_focus: g.spot <= near * e.signal,
      path: g.path,
      reflectance_product: g.rProduct,
      relative_irradiance: nf(g.relative),
      spot_diameter: nf(g.spot),
      transmittance: g.transmittance,
      vignetting: 1,
    })),
    primary_focus_from_sensor: nf(e.primaryFocus),
    primary_image_height: nf(e.primaryHeight),
    primary_transmittance: e.primaryT,
    signal_diameter: nf(e.signal),
    truncated: false,
    wavelength: w,
  })
}

function blob(map: Float64Array, nx: number, ny: number, hx: number, hy: number, cx: number, cy: number, flux: number, sigma: number): void {
  const norm = flux / (2 * Math.PI * sigma * sigma)
  const px = (2 * hx) / nx
  const py = (2 * hy) / ny
  for (let j = 0; j < ny; j++) {
    const y = -hy + (j + 0.5) * py
    const dy = y - cy
    if (Math.abs(dy) > 5 * sigma) continue
    for (let i = 0; i < nx; i++) {
      const x = -hx + (i + 0.5) * px
      const dx = x - cx
      if (Math.abs(dx) > 5 * sigma) continue
      map[j * nx + i] += norm * Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    }
  }
}

export function ghostIrradiance(o: MockOptics, s: Obj): Out {
  const w = o.primary
  const e = enumerateGhosts(o, { ...s, sensor_reflectance: s.sensor_reflectance }, w)
  const bins = Array.isArray(s.map_bins) ? (s.map_bins as number[]) : [64, 64]
  const nx = Math.min(Math.max(Math.floor(bins[0] ?? 64), 2), 128)
  const ny = Math.min(Math.max(Math.floor(bins[1] ?? 64), 2), 128)
  const half = Array.isArray(s.map_half_size) && s.map_half_size.length === 2 ? (s.map_half_size as number[]) : [Math.max(e.primaryHeight * 1.3, 1), Math.max(e.primaryHeight * 1.3, 1)]
  const mf = o.maxField
  const sel: number[] = Array.isArray(s.fields) && s.fields.length ? (s.fields as number[]).filter((f) => f >= 0 && f < o.sys.fields.length) : o.sys.fields.map((_, i) => i)
  const top = e.ghosts.slice(0, 12)
  const pixel = ((2 * half[0]) / nx) * ((2 * half[1]) / ny)
  const sigmaP = Math.max(airyRadius(o) * 0.6, 0.004)
  const primary = new Float64Array(nx * ny)
  const total = new Float64Array(nx * ny)
  const pathMaps = top.map(() => new Float64Array(nx * ny))
  const centreOf = (f: number, h: number): [number, number] => {
    const fp = o.fieldPointXY(o.sys.fields[f].x, o.sys.fields[f].y)
    const par = o.chiefImagePerUnit * (fp.kind === 'angle' ? fp.ty : fp.y)
    const frac = mf > 0 ? o.sys.fields[f].y / mf : 0
    return [0, h > 0 ? h * frac : par]
  }
  const sum = (m: Float64Array) => m.reduce((a, b) => a + b, 0) * pixel
  const stats = (map: Float64Array, f: number, flux: number, centre: [number, number]): JsonObject => {
    const peak = map.reduce((a, b) => Math.max(a, b), 0)
    return { field: { type: 'system', index: f }, flux, flux_on_map: sum(map), peak, rays_detected: Math.round(flux * 4000) + 1, rays_traced: 4608, rms_radius: nf(Math.hypot(0, 0.5 * centre[1] * 0.05) + sigmaP), centroid: pair(centre[0], centre[1]) }
  }
  const primaryFields: JsonObject[] = []
  for (const f of sel) {
    const m = new Float64Array(nx * ny)
    const c = centreOf(f, e.primaryHeight)
    blob(m, nx, ny, half[0], half[1], c[0], c[1], 1, sigmaP)
    for (let q = 0; q < m.length; q++) primary[q] += m[q]
    primaryFields.push(stats(m, f, 1, c))
  }
  const primPeak = primary.reduce((a, b) => Math.max(a, b), 0) || 1
  const ghostsJson: JsonObject[] = top.map((g, gi) => {
    const flux = (g.rProduct * g.transmittance) / Math.max(e.primaryT, 1e-12)
    const sigma = Math.max(g.spot / 4, sigmaP)
    const fields: JsonObject[] = []
    let peak = 0
    let onMap = 0
    for (const f of sel) {
      const m = new Float64Array(nx * ny)
      const c = centreOf(f, g.imageHeight)
      blob(m, nx, ny, half[0], half[1], c[0], c[1], flux, sigma)
      for (let q = 0; q < m.length; q++) {
        total[q] += m[q]
        pathMaps[gi][q] += m[q]
      }
      const st = stats(m, f, flux, c)
      fields.push(st)
      peak = Math.max(peak, Number(st.peak))
      onMap += Number(st.flux_on_map)
    }
    return { label: g.label, path: g.path, map_index: gi, flux: onMap, flux_ratio: onMap / Math.max(sel.length, 1), peak_ratio: peak / primPeak, fields }
  })
  const totalFlux = ghostsJson.reduce((a, g) => a + Number(g.flux_ratio), 0)
  const out = newOut({
    coverage: ghostCoverage(o, e.ghosts, e.interfaces, s),
    ghosts: ghostsJson,
    map: { bins: [nx, ny], half_size: half },
    primary: { label: 'primary', path: null, map_index: null, flux: sum(primary), flux_ratio: 1, peak_ratio: 1, fields: primaryFields },
    total_flux_ratio: totalFlux,
    total_peak_ratio: total.reduce((a, b) => Math.max(a, b), 0) / primPeak,
    truncated: false,
  })
  addBuf(out, 'primary_map', primary, [ny, nx])
  addBuf(out, 'total_ghost_map', total, [ny, nx])
  if (s.keep_path_maps === true && top.length) {
    const all = new Float64Array(top.length * nx * ny)
    pathMaps.forEach((m, g) => all.set(m, g * nx * ny))
    addBuf(out, 'path_maps', all, [top.length, ny, nx])
  }
  return out
}

// --- Stray light ------------------------------------------------------------------------------------

const STRAY_CLASSES = ['primary', 'ghost', 'scatter', 'mechanical', 'sneak'] as const

function rngOf(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function strayLight(o: MockOptics, s: Obj): Out {
  const px = Array.isArray(s.detector_pixels) ? (s.detector_pixels as number[]) : [128, 128]
  const nx = Math.min(Math.max(Math.floor(px[0] ?? 128), 4), 128)
  const ny = Math.min(Math.max(Math.floor(px[1] ?? 128), 4), 128)
  const mask = Math.floor(Number(s.classes ?? 30)) & 31
  const rays = Math.max(Math.floor(Number(s.rays ?? 20000)), 1)
  const e = enumerateGhosts(o, {}, o.primary)
  const margin = Number(s.detector_margin ?? 0.05)
  const halfY = Math.max(e.primaryHeight * (1 + margin), 1)
  const halfX = halfY
  const rnd = rngOf(Math.floor(Number(s.seed ?? 1)))
  const src = (s.source ?? {}) as Obj
  const fIdx = Math.min(Math.max(Math.floor(Number(src.index ?? 0)), 0), o.sys.fields.length - 1)
  const cy = o.sys.fields[fIdx].y / Math.max(o.maxField, 1e-12) * e.primaryHeight
  const power = 1 * Number(src.strength ?? 1)
  const maps = STRAY_CLASSES.map(() => new Float64Array(nx * ny))
  const sigmaP = Math.max(airyRadius(o) * 0.6, 0.004)
  blob(maps[0], nx, ny, halfX, halfY, 0, cy, power * e.primaryT, sigmaP)
  for (const g of e.ghosts.slice(0, 8)) blob(maps[1], nx, ny, halfX, halfY, 0, g.imageHeight * (o.maxField > 0 ? o.sys.fields[fIdx].y / o.maxField : 0), power * g.rProduct * g.transmittance, Math.max(g.spot / 4, sigmaP))
  // Scatter: a broad halo; mechanical: light near the detector edge; sneak: weak speckle.
  blob(maps[2], nx, ny, halfX, halfY, 0, cy, power * 2e-4, halfY * 0.45)
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) / nx - 0.5
      const y = (j + 0.5) / ny - 0.5
      const edge = Math.max(Math.abs(x), Math.abs(y)) * 2
      maps[3][j * nx + i] = power * 1e-5 * Math.exp(-12 * (1 - edge)) / (4 * halfX * halfY)
      maps[4][j * nx + i] = power * 4e-7 * rnd() / (4 * halfX * halfY)
    }
  const pixel = ((2 * halfX) / nx) * ((2 * halfY) / ny)
  const flux = maps.map((m) => m.reduce((a, b) => a + b, 0) * pixel)
  const included = STRAY_CLASSES.map((_, k) => (mask & (1 << k)) !== 0)
  const owned = STRAY_CLASSES.map((c, k) => (included[k] ? c : null)).filter((c): c is (typeof STRAY_CLASSES)[number] => c !== null)
  const C = owned.length
  const cls = new Float64Array(C * nx * ny)
  const clsErr = new Float64Array(C * nx * ny)
  const tot = new Float64Array(nx * ny)
  const totErr = new Float64Array(nx * ny)
  owned.forEach((c, r) => {
    const m = maps[STRAY_CLASSES.indexOf(c)]
    for (let q = 0; q < m.length; q++) {
      cls[r * nx * ny + q] = m[q]
      clsErr[r * nx * ny + q] = (m[q] / Math.sqrt(rays)) * 1.5
    }
    if (c !== 'primary') for (let q = 0; q < m.length; q++) {
      tot[q] += m[q]
      totErr[q] += (m[q] / Math.sqrt(rays)) ** 2
    }
  })
  for (let q = 0; q < totErr.length; q++) totErr[q] = Math.sqrt(totErr[q]) * 1.5
  const strayFlux = flux.reduce((a, f, k) => (included[k] && k > 0 ? a + f : a), 0)
  const primaryFlux = flux[0]
  const err = (f: number) => (rays > 1 ? (f / Math.sqrt(rays)) * 1.5 : null)
  const totals = STRAY_CLASSES.map((c, k) => ({
    class: c,
    flux: flux[k],
    fraction: primaryFlux > 0 ? flux[k] / primaryFlux : null,
    hits: Math.round(flux[k] * rays),
    included: included[k],
    moments: { flux: flux[k], centroid: pair(0, cy), centroid_std_error: err(0.001), rms_radius: k === 0 ? sigmaP : halfY * (0.3 + 0.1 * k), rms_radius_std_error: err(0.001) },
    std_error: err(flux[k]),
  }))
  const sigs: JsonObject[] = []
  for (const g of e.ghosts.slice(0, Math.min(Number(s.top_signatures ?? 20), 12))) {
    const f = power * g.rProduct * g.transmittance
    sigs.push({ class: 'ghost', signature: g.label, flux: f, fraction: primaryFlux > 0 ? f / primaryFlux : null, std_error: err(f), hits: Math.round(f * rays) })
  }
  sigs.push({ class: 'scatter', signature: 'lens scatter', flux: flux[2], fraction: primaryFlux > 0 ? flux[2] / primaryFlux : null, std_error: err(flux[2]), hits: Math.round(flux[2] * rays) })
  sigs.sort((a, b) => Number(b.flux) - Number(a.flux))
  const out = newOut({
    classes: mask,
    clear_semi_diameters: o.sys.surfaces.map((sf) => nf(sf.sdFixed)),
    detector: { surface: o.sys.surfaces.length - 1, nx, ny, half_x: halfX, half_y: halfY, accept: 90 },
    emitted: power,
    energy: { emitted: power, detected: flux.reduce((a, b) => a + b, 0), absorbed: power * 0.04, escaped: power * 0.02, balance_residual: 0, detected_std_error: err(power), absorbed_std_error: err(power * 0.04), escaped_std_error: err(power * 0.02) },
    map_classes: owned,
    partition: null,
    primary_obstruction: null,
    source: { spec: { type: String(src.type ?? 'system_field'), index: fIdx, margin: Number(src.margin ?? 0.1), strength: power }, reference_flux: power, description: `system field ${fIdx + 1}` },
    stray_flux: strayFlux,
    stray_fraction: primaryFlux > 0 ? strayFlux / primaryFlux : null,
    stray_std_error: err(strayFlux),
    top_signatures: sigs.slice(0, Math.floor(Number(s.top_signatures ?? 20))),
    totals,
    warnings: ['MOCK ENGINE: stray light is synthetic'],
  })
  addBuf(out, 'class_irradiance', cls, [C, ny, nx])
  addBuf(out, 'class_irradiance_std_error', clsErr, [C, ny, nx])
  addBuf(out, 'total_irradiance', tot, [ny, nx])
  addBuf(out, 'total_irradiance_std_error', totErr, [ny, nx])
  return out
}

// --- Image simulation ----------------------------------------------------------------------------------

function patternAt(kind: string, u: number, v: number): number {
  switch (kind) {
    case 'siemens_star': {
      const r = Math.hypot(u, v)
      if (r > 0.9 || r < 0.03) return 0.5
      return Math.sin(24 * Math.atan2(v, u)) > 0 ? 1 : 0
    }
    case 'slanted_edge':
      return u * Math.cos(0.09) + v * Math.sin(0.09) > 0 ? 1 : 0.05
    case 'text_chart': {
      // Rows of bars whose pitch shrinks down the chart.
      const row = Math.floor(((v + 1) / 2) * 6)
      const pitch = 0.16 / (1 + row * 0.55)
      return Math.abs(u) < 0.8 && ((v + 1) / 2) * 6 - row > 0.25 && Math.floor((u + 1) / pitch) % 2 === 0 ? 1 : 0.1
    }
    default:
      return Math.abs(((u * 4 + 8) % 1) - 0.5) < 0.04 || Math.abs(((v * 4 + 8) % 1) - 0.5) < 0.04 ? 1 : 0.15
  }
}

export function imageSimulation(o: MockOptics, s: Obj): Out {
  const size = Array.isArray(s.size) ? (s.size as number[]) : [64, 64]
  const W = Math.min(Math.max(Math.floor(size[0] ?? 64), 2), 1024)
  const H = Math.min(Math.max(Math.floor(size[1] ?? 64), 2), 1024)
  const mono = s.monochrome !== false
  const C = mono ? 1 : 3
  const optics = (s.optics ?? {}) as Obj
  const grid = Math.min(Math.max(Math.floor(Number(optics.field_grid ?? 3)), 2), 9)
  const src = (s.source ?? {}) as Obj
  const kind = src.kind === 'pattern' ? String(src.pattern ?? 'grid') : 'grid'
  const warnings: string[] = []
  if (src.kind && src.kind !== 'pattern') warnings.push('mock engine: only pattern sources are simulated, a grid is used')
  const maxH = Math.max(Math.abs(o.chiefImagePerUnit * o.fieldParameterY(o.fieldPointXY(0, o.maxField))), 1e-3)
  const half = maxH * 1.0
  const pixel: [number, number] = [(2 * half) / W, (2 * half) / H]
  // Blur sigma (mm) and relative illumination at the field nodes along the diagonal.
  const nodes = Array.from({ length: grid * grid }, (_, k) => [(k % grid) / (grid - 1), Math.floor(k / grid) / (grid - 1)])
  const mapping: JsonObject[] = []
  const sig: number[] = []
  const ri: number[] = []
  const lam = o.wavelengthsUm[o.primary] * 1e-3
  const nw = Number.isFinite(o.workingFNumber(o.primary)) ? o.workingFNumber(o.primary) : 10
  nodes.forEach(([gx, gy]) => {
    const hx = -1 + 2 * gx
    const hy = -1 + 2 * gy
    const r = Math.min(Math.hypot(hx, hy), 1)
    const x = hx * o.maxField * 0.7071
    const y = hy * o.maxField * 0.7071
    const fp = o.fieldPointXY(x, y)
    const chief = o.trace(o.makeRay(fp, 0, 0), o.primary).atImage
    const rms = o.sys.fields.length ? spotRms(o, Math.min(Math.round(r * (o.sys.fields.length - 1)), o.sys.fields.length - 1)) : 0
    sig.push(Math.max(rms, 0.4 * lam * nw))
    const cos4 = Math.cos(Math.atan(r * Math.tan((Math.min(o.maxField, 40) * Math.PI) / 180))) ** 4
    ri.push(cos4)
    mapping.push({
      field: pair(hx, hy),
      ideal_mm: pair(o.chiefImagePerUnit * (fp.kind === 'angle' ? fp.tx : fp.x), o.chiefImagePerUnit * (fp.kind === 'angle' ? fp.ty : fp.y)),
      pupil_sampling: [{ grid: 33, wavelength: o.primary }],
      real_mm: pair(chief ? chief.p[0] : NaN, chief ? chief.p[1] : NaN),
      relative_illumination: cos4,
    })
  })
  const img = new Float64Array(W * H * C)
  const radiusPx = Math.min(W * H > 262144 ? 2 : 5, 8)
  const rnd = rngOf(Math.floor(Number(s.seed ?? 1)))
  const readSigma = Number(s.read_noise_sigma ?? 0)
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd())
  const src2 = (px: number, py: number) => patternAt(kind, (px / W) * 2 - 1, (py / H) * 2 - 1)
  for (let j = 0; j < H; j++) {
    const v = ((j + 0.5) / H) * 2 - 1
    for (let i = 0; i < W; i++) {
      const u = ((i + 0.5) / W) * 2 - 1
      const r = Math.min(Math.hypot(u, v), 1)
      // Nearest node blur.
      const gi = Math.min(Math.round(((u + 1) / 2) * (grid - 1)), grid - 1)
      const gj = Math.min(Math.round(((v + 1) / 2) * (grid - 1)), grid - 1)
      const sg = sig[gj * grid + gi] / pixel[0]
      const k = Math.max(sg, 0.35)
      let acc = 0
      let ws = 0
      for (let dj = -radiusPx; dj <= radiusPx; dj++)
        for (let di = -radiusPx; di <= radiusPx; di++) {
          const wgt = Math.exp(-(di * di + dj * dj) / (2 * k * k))
          const pi = i + di
          const pj = j + dj
          if (pi < 0 || pi >= W || pj < 0 || pj >= H) {
            ws += wgt
            continue
          }
          acc += wgt * src2(pi + 0.5, pj + 0.5)
          ws += wgt
        }
      const vig = ri[gj * grid + gi] * (1 - 0.1 * r * r)
      for (let c = 0; c < C; c++) {
        const tint = C === 3 ? [1, 0.97, 0.92][c] : 1
        img[(j * W + i) * C + c] = (acc / ws) * vig * tint + (readSigma > 0 ? readSigma * gauss() : 0)
      }
    }
  }
  const out = newOut({
    channels: C,
    field_grid: grid,
    field_mapping: mapping,
    image: 'image',
    origin_mm: pair(-half, -half),
    pixel_size_mm: pair(pixel[0], pixel[1]),
    psf_size: Number(optics.psf_size) > 0 ? Number(optics.psf_size) : 32,
    psf_spacing_mm: typeof optics.psf_spacing_mm === 'number' ? optics.psf_spacing_mm : (lam * nw) / 2.5,
    size: [W, H],
    warnings: warnings as Json,
  })
  addBuf(out, 'image', img, [H, W, C])
  return out
}

function spotRms(o: MockOptics, f: number): number {
  const sp = spotData(o, { fields: [f], wls: [o.primary], pattern: 'hexapolar', rings: 3, seed: 1, defocus: 0, reference: 'chief' })[0]
  return Number.isFinite(sp.poly.rms_c) ? sp.poly.rms_c : 0
}

// --- rng_stream -------------------------------------------------------------------------------------------

export function rngStream(s: Obj): Out {
  const count = Math.min(Math.max(Math.floor(Number(s.count ?? 100000)), 0), 4_000_000)
  const g = rngOf((Math.floor(Number(s.seed ?? 0)) * 2654435761 + Math.floor(Number(s.stream ?? 0)) * 40503) >>> 0)
  const data = new Float64Array(count)
  let sum = 0
  let sq = 0
  let mn = Infinity
  let mx = -Infinity
  for (let i = 0; i < count; i++) {
    const v = g()
    data[i] = v
    sum += v
    sq += v * v
    if (v < mn) mn = v
    if (v > mx) mx = v
  }
  const mean = count ? sum / count : 0
  const out = newOut({ count, first: Array.from(data.subarray(0, 4)), max: count ? mx : 0, mean, min: count ? mn : 0, variance: count ? sq / count - mean * mean : 0 })
  addBuf(out, 'uniform', data, [count])
  return out
}

