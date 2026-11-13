// MOCK ENGINE — synthetic analyses on MockOptics, written out in the engine's
// result format (docs/RESULTS.md). Numbers are illustrative, not optical truth.
import type { Json, JsonObject, TypedArray } from '../engine/types'
import { MockOptics, linspace, materialNdVd, sag, shapeConic, shapeCurvature, type Ray, type V3 } from './optics'
import { structuralIssues, type Obj } from './model'

export interface Out {
  json: JsonObject
  buffers: Record<string, TypedArray>
  shapes: Record<string, number[]>
}

export function newOut(json: JsonObject = {}): Out {
  return { json, buffers: {}, shapes: {} }
}
export function addBuf(o: Out, name: string, data: TypedArray, shape: number[]): void {
  o.buffers[name] = data
  o.shapes[name] = shape
}

/** Engine `nf`: NaN -> null, ±inf -> "inf"/"-inf". */
export function nf(d: number | null | undefined): Json {
  if (d === null || d === undefined || Number.isNaN(d)) return null
  if (!Number.isFinite(d)) return d > 0 ? 'inf' : '-inf'
  return d
}
export const nfs = (a: (number | null | undefined)[]): Json[] => a.map(nf)
export const pair = (x: number, y: number): Json => [nf(x), nf(y)]

// --- Selections -----------------------------------------------------------

export function selectIndices(v: unknown, count: number, dflt: 'all' | 'primary', primary = 0): number[] {
  const all = Array.from({ length: count }, (_, i) => i)
  const o = (v ?? {}) as Obj
  const type = typeof o.type === 'string' ? o.type : dflt
  switch (type) {
    case 'primary':
      return primary >= 0 && primary < count ? [primary] : []
    case 'index':
      return Number.isInteger(o.index) && o.index >= 0 && o.index < count ? [o.index] : []
    case 'list':
      return (Array.isArray(o.indices) ? (o.indices as number[]) : []).filter((i) => Number.isInteger(i) && i >= 0 && i < count)
    default:
      return all
  }
}
const fieldSel = (o: MockOptics, v: unknown) => selectIndices(v, o.sys.fields.length, 'all')
const wlSel = (o: MockOptics, v: unknown, d: 'all' | 'primary') =>
  selectIndices(v, o.sys.wavelengths.length, d, o.primary)

function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function pupilSamples(pattern: string, rings: number, seed: number): [number, number][] {
  const r = Math.max(Math.floor(rings), 1)
  if (pattern === 'square') {
    const out: [number, number][] = []
    for (let i = -r; i <= r; i++)
      for (let j = -r; j <= r; j++) {
        const x = i / r
        const y = j / r
        if (x * x + y * y <= 1 + 1e-12) out.push([x, y])
      }
    return out
  }
  if (pattern === 'random' || pattern === 'dithered') {
    const g = rng(seed)
    return Array.from({ length: 3 * r * (r + 1) + 1 }, () => {
      const rad = Math.sqrt(g())
      const a = 2 * Math.PI * g()
      return [rad * Math.cos(a), rad * Math.sin(a)] as [number, number]
    })
  }
  const out: [number, number][] = [[0, 0]]
  for (let k = 1; k <= r; k++) {
    const rad = k / r
    for (let j = 0; j < 6 * k; j++) {
      const a = (2 * Math.PI * j) / (6 * k)
      out.push([rad * Math.cos(a), rad * Math.sin(a)])
    }
  }
  return out
}

// --- Semi-diameters -------------------------------------------------------

export function semiDiameters(o: MockOptics): number[] {
  const surfaces = o.sys.surfaces
  const traced = new Array<number>(surfaces.length).fill(0)
  for (let f = 0; f < o.sys.fields.length; f++) {
    const fp = o.fieldPoint(f)
    for (const [px, py] of [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]) {
      for (const [i, h] of o.trace(o.makeRay(fp, px, py), o.primary).heights) traced[i] = Math.max(traced[i], h)
    }
  }
  return surfaces.map((s, i) => {
    if (s.sdFixed !== null) return s.sdFixed
    if (s.aperture) {
      const a = s.aperture
      return a.type === 'circular' ? a.radius : a.type === 'annular' ? a.outer : a.type === 'rectangular' ? Math.hypot(a.hx, a.hy) : Math.max(a.sx, a.sy)
    }
    return traced[i] > 0 ? traced[i] : o.epRadius
  })
}

// --- Layout ---------------------------------------------------------------

interface Poly {
  pts: V3[]
  kind: number
  surface: number
  field: number
  wl: number
  flags: number
}

export function layout(o: MockOptics, s: Obj): Out {
  const sys = o.sys
  const surfaces = sys.surfaces
  const n = surfaces.length
  const sd = semiDiameters(o)
  const margin = typeof s.edge_margin === 'number' ? s.edge_margin : 0.5
  const m = Math.max(Math.floor(Number(s.profile_samples ?? 41)) | 1, 5)
  const raysPer = Math.max(Math.floor(Number(s.rays_per_field ?? 7)), 1)
  const include3D = s.include_3d !== false
  const xz = s.section === 'xz'
  const ax: 0 | 1 = xz ? 0 : 1
  const ox: 0 | 1 = xz ? 1 : 0
  const ringCount = Math.max(Math.floor(Number(s.rings ?? 3)), 1)
  const ringSamples = Math.max(Math.floor(Number(s.ring_samples ?? 64)), 3)
  const meshRings = include3D ? Math.max(Math.floor(Number(s.mesh_rings ?? 0)), 0) : 0
  const image = n - 1
  const lines: Poly[] = []
  const mech = surfaces.map((x, i) => x.mechSD ?? sd[i] + margin)

  const profile = (i: number, from: number, to: number, clamp: number, axis: 0 | 1): V3[] => {
    const out: V3[] = []
    for (const r of linspace(from, to, m)) {
      const rc = Math.min(Math.abs(r), clamp)
      const sg = sagOf(o, i, rc * rc)
      if (sg === null) continue
      const zz = o.vertexZ[i] + sg
      out.push(axis === 1 ? [0, r, zz] : [r, 0, zz])
    }
    return out
  }

  let imageHeight = 0
  const wls = wlSel(o, s.wavelengths, 'primary')
  for (const f of fieldSel(o, s.fields)) {
    const fp = o.fieldPoint(f)
    const meridional = xz ? sys.fields[f].y === 0 : sys.fields[f].x === 0
    for (const w of wls) {
      const pupil: [number, number][] = linspace(-1, 1, raysPer).map((v) => (xz ? [v, 0] : [0, v]) as [number, number])
      if (include3D) pupil.push(...((xz ? [[0, -1], [0, 1]] : [[-1, 0], [1, 0]]) as [number, number][]))
      for (const [px, py] of pupil) {
        const t = o.trace(o.makeRay(fp, px, py), w)
        const h = t.heights.get(image)
        if (h !== undefined) imageHeight = Math.max(imageHeight, h)
        lines.push({
          pts: t.points,
          kind: 5,
          surface: -1,
          field: f,
          wl: w,
          flags: (meridional && (xz ? py === 0 : px === 0) ? 1 : 0) | (t.ok ? 0 : 2),
        })
      }
    }
  }

  const fpMax = o.fieldPointXY(0, o.maxField)
  for (let i = 0; i < n; i++) {
    if (surfaces[i].isCB) continue
    if (i === 0 && o.objectDistance === null) continue
    let r = sd[i]
    if (i === image) r = Math.max(imageHeight, Math.abs(o.chiefImagePerUnit * o.fieldParameterY(fpMax)))
    if (i === 0)
      r =
        Math.max(
          0,
          ...sys.fields.map((f) => {
            const fp = o.fieldPointXY(f.x, f.y)
            return fp.kind === 'height' ? Math.hypot(fp.x, fp.y) : 0
          }),
        ) || 1
    r = Math.max(r, 1e-3)
    const kind = i === sys.stop ? 3 : i === image ? 4 : 0
    const dflag = surfaces[i].shape.type === 'diffractive' ? 4 : 0
    lines.push({ pts: profile(i, -r, r, r, ax), kind, surface: i, field: -1, wl: -1, flags: 1 | dflag })
    if (i === sys.stop) {
      for (const sg of [-1, 1])
        lines.push({
          pts: (xz ? [[sg * r, 0, o.vertexZ[i]], [sg * r * 1.3, 0, o.vertexZ[i]]] : [[0, sg * r, o.vertexZ[i]], [0, sg * r * 1.3, o.vertexZ[i]]]) as V3[],
          kind: 3,
          surface: i,
          field: -1,
          wl: -1,
          flags: 1,
        })
    }
    if (include3D) {
      lines.push({ pts: profile(i, -r, r, r, ox), kind, surface: i, field: -1, wl: -1, flags: dflag })
      for (let k = 1; k <= ringCount; k++) {
        const rk = (r * k) / ringCount
        const z = sagOf(o, i, rk * rk)
        if (z === null) continue
        lines.push({
          pts: linspace(0, 2 * Math.PI, ringSamples + 1).map((a) => [rk * Math.cos(a), rk * Math.sin(a), o.vertexZ[i] + z] as V3),
          kind: 1,
          surface: i,
          field: -1,
          wl: -1,
          flags: dflag,
        })
      }
    }
  }

  const elements: { first: number; last: number; front: V3[]; back: V3[]; mech: number }[] = []
  for (let i = 1; i < Math.max(image - 1, 1); i++) {
    const a = surfaces[i]
    const b = surfaces[i + 1]
    if (a.interaction !== 'refract' || a.isCB || b.interaction !== 'refract' || b.isCB) continue
    if (a.material.kind === 'air' || a.material.kind === 'vacuum') continue
    const h = Math.max(sd[i], sd[i + 1])
    const front = profile(i, -h, h, sd[i], ax)
    const back = profile(i + 1, -h, h, sd[i + 1], ax)
    if (front.length !== back.length || front.length === 0) continue
    elements.push({ first: i, last: i + 1, front, back, mech: Math.max(mech[i], mech[i + 1]) })
    const f0 = front[0], f1 = front[front.length - 1], b0 = back[0], b1 = back[back.length - 1]
    lines.push({ pts: [f0, b0], kind: 2, surface: i, field: -1, wl: -1, flags: 1 })
    lines.push({ pts: [f1, b1], kind: 2, surface: i, field: -1, wl: -1, flags: 1 })
  }

  // Bounds, then the axis.
  const mn: V3 = [Infinity, Infinity, Infinity]
  const mx: V3 = [-Infinity, -Infinity, -Infinity]
  for (const l of lines)
    for (const p of l.pts)
      for (let k = 0; k < 3; k++) {
        if (p[k] < mn[k]) mn[k] = p[k]
        if (p[k] > mx[k]) mx[k] = p[k]
      }
  if (!Number.isFinite(mn[0])) {
    mn.fill(-1)
    mx.fill(1)
  }
  lines.push({ pts: [[0, 0, mn[2]], [0, 0, mx[2]]], kind: 6, surface: -1, field: -1, wl: -1, flags: 1 })
  mn[0] = Math.min(mn[0], 0)
  mn[1] = Math.min(mn[1], 0)
  mx[0] = Math.max(mx[0], 0)
  mx[1] = Math.max(mx[1], 0)

  // Element outlines (kind 7, closed).
  const outlineIndex: number[] = []
  for (const e of elements) {
    outlineIndex.push(lines.length)
    const pts = [...e.front, ...[...e.back].reverse(), e.front[0]]
    lines.push({ pts, kind: 7, surface: e.first, field: -1, wl: -1, flags: 1 })
  }

  const P = lines.length
  const V = lines.reduce((a, l) => a + l.pts.length, 0)
  const origin: V3 = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2]
  const v64 = new Float64Array(V * 3)
  const v32 = new Float32Array(V * 3)
  const offsets = new Uint32Array(P + 1)
  const kinds = new Uint8Array(P)
  const surf = new Int32Array(P)
  const fld = new Int32Array(P)
  const wav = new Int32Array(P)
  const flags = new Uint8Array(P)
  let vi = 0
  lines.forEach((l, k) => {
    for (const p of l.pts) {
      for (let c = 0; c < 3; c++) {
        v64[vi * 3 + c] = p[c]
        v32[vi * 3 + c] = p[c] - origin[c]
      }
      vi++
    }
    offsets[k + 1] = vi
    kinds[k] = l.kind
    surf[k] = l.surface
    fld[k] = l.field
    wav[k] = l.wl
    flags[k] = l.flags
  })

  // Optional 3-D mesh: a fan at the vertex and quad strips per radial subdivision.
  const meshTri: number[] = []
  const meshSurf: number[] = []
  if (meshRings > 0) {
    const K = Math.max(ringSamples, 3)
    for (let i = 1; i < n; i++) {
      if (surfaces[i].isCB) continue
      let r = sd[i]
      if (i === image) r = Math.max(imageHeight, Math.abs(o.chiefImagePerUnit * o.fieldParameterY(fpMax)), 1e-3)
      if (!(r > 0)) continue
      const vert = (ring: number, k: number): V3 => {
        const rr = (r * ring) / meshRings
        const a = (2 * Math.PI * k) / K
        const zz = sagOf(o, i, rr * rr)
        return [rr * Math.cos(a) - origin[0], rr * Math.sin(a) - origin[1], o.vertexZ[i] + (zz ?? 0) - origin[2]]
      }
      const tri = (a: V3, b: V3, c: V3) => {
        meshTri.push(...a, ...b, ...c)
        meshSurf.push(i)
      }
      const centre: V3 = [-origin[0], -origin[1], o.vertexZ[i] + (sagOf(o, i, 0) ?? 0) - origin[2]]
      for (let k = 0; k < K; k++) tri(centre, vert(1, k), vert(1, k + 1))
      for (let ring = 1; ring < meshRings; ring++)
        for (let k = 0; k < K; k++) {
          const a = vert(ring, k), b = vert(ring + 1, k), c = vert(ring + 1, k + 1), d = vert(ring, k + 1)
          tri(a, b, c)
          tri(a, c, d)
        }
    }
  }

  const surfInfo = surfaces.map((s, i) => ({
    id: s.id,
    index: i,
    label: s.label,
    vertex: [0, 0, nf(i === 0 && o.objectDistance === null ? 0 : o.vertexZ[i])],
    axis: [0, 0, 1],
    semi_diameter: nf(sd[i]),
    semi_diameter_is_auto: s.sdFixed === null && !s.aperture,
    mechanical_semi_diameter: nf(mech[i]),
    is_stop: i === sys.stop,
    is_mirror: s.interaction === 'reflect',
    is_diffractive: s.shape.type === 'diffractive',
  }))
  const out = newOut({
    bounds: { min: mn.map(nf), max: mx.map(nf) },
    origin,
    surfaces: surfInfo,
    elements: elements.map((e, k) => ({
      first_surface: e.first,
      last_surface: e.last,
      mechanical_semi_diameter: nf(e.mech),
      outline_polyline: outlineIndex[k],
    })),
    kind_names: ['surface_profile', 'surface_rim', 'element_edge', 'stop', 'image_plane', 'ray', 'axis', 'element_outline'],
    polyline_count: P,
    vertex_count: V,
    section: xz ? 'xz' : 'yz',
    field_count: sys.fields.length,
    field_values: sys.fields.map((f) => pair(f.x, f.y)),
    wavelengths_um: o.wavelengthsUm,
    mesh_triangle_count: meshSurf.length,
  })
  addBuf(out, 'vertices', v32, [V, 3])
  addBuf(out, 'vertices_f64', v64, [V, 3])
  addBuf(out, 'offsets', offsets, [P + 1])
  addBuf(out, 'kinds', kinds, [P])
  addBuf(out, 'surfaces', surf, [P])
  addBuf(out, 'fields', fld, [P])
  addBuf(out, 'wavelengths', wav, [P])
  addBuf(out, 'flags', flags, [P])
  if (meshSurf.length) {
    addBuf(out, 'surface_mesh', Float32Array.from(meshTri), [meshSurf.length, 3, 3])
    addBuf(out, 'mesh_surfaces', Uint32Array.from(meshSurf), [meshSurf.length])
  }
  return out
}

function sagOf(o: MockOptics, i: number, rho: number): number | null {
  const r = sag(o.sys.surfaces[i].shape, rho)
  return r ? r.z : null
}

// --- Spot diagram ---------------------------------------------------------

interface RawPt {
  x: number
  y: number
  sx: number
  sy: number
  w: number
}
export interface SpotStats {
  centroid: [number, number]
  rms_c: number
  rms_chief: number
  geo_c: number
  geo_chief: number
  rms_x: number
  rms_y: number
  arrived: number
  launched: number
}
export interface FieldSpot {
  field: number
  ref: [number, number]
  chief: [number, number]
  centroid: [number, number]
  pts: RawPt[]
  perWl: { w: number; chief: [number, number]; stats: SpotStats }[]
  poly: SpotStats
  launched: number
}

function stats(pts: { x: number; y: number }[], chief: [number, number], launched: number): SpotStats {
  const n = pts.length
  if (n === 0) {
    const nan = NaN
    return { centroid: [nan, nan], rms_c: nan, rms_chief: nan, geo_c: nan, geo_chief: nan, rms_x: nan, rms_y: nan, arrived: 0, launched }
  }
  let cx = 0, cy = 0
  for (const p of pts) {
    cx += p.x
    cy += p.y
  }
  cx /= n
  cy /= n
  let sc = 0, sch = 0, gc = 0, gch = 0, sx = 0, sy = 0
  for (const p of pts) {
    const dx = p.x - cx, dy = p.y - cy
    const ex = p.x - chief[0], ey = p.y - chief[1]
    sc += dx * dx + dy * dy
    sch += ex * ex + ey * ey
    sx += dx * dx
    sy += dy * dy
    gc = Math.max(gc, Math.hypot(dx, dy))
    gch = Math.max(gch, Math.hypot(ex, ey))
  }
  return {
    centroid: [cx - chief[0], cy - chief[1]],
    rms_c: Math.sqrt(sc / n),
    rms_chief: Math.sqrt(sch / n),
    geo_c: gc,
    geo_chief: gch,
    rms_x: Math.sqrt(sx / n),
    rms_y: Math.sqrt(sy / n),
    arrived: n,
    launched,
  }
}

export interface SpotOpts {
  fields: number[]
  wls: number[]
  pattern: string
  rings: number
  seed: number
  defocus: number
  reference: string
}

export function spotData(o: MockOptics, so: SpotOpts): FieldSpot[] {
  const out: FieldSpot[] = []
  const pupil = (f: number) => pupilSamples(so.pattern, so.rings, (so.seed ^ Math.imul(f, 0x9e3779b9)) >>> 0)
  const at = (r: Ray) => {
    const sx = Math.abs(r.d[2]) > 1e-12 ? r.d[0] / r.d[2] : 0
    const sy = Math.abs(r.d[2]) > 1e-12 ? r.d[1] / r.d[2] : 0
    return { x: r.p[0] + so.defocus * sx, y: r.p[1] + so.defocus * sy, sx, sy }
  }
  for (const f of so.fields) {
    const fp = o.fieldPoint(f)
    const pu = pupil(f)
    const raw: RawPt[] = []
    const perW: { w: number; pts: RawPt[] }[] = []
    for (const w of so.wls) {
      const pts: RawPt[] = []
      for (const [px, py] of pu) {
        const r = o.trace(o.makeRay(fp, px, py), w).atImage
        if (r) pts.push({ ...at(r), w })
      }
      perW.push({ w, pts })
      raw.push(...pts)
    }
    const chiefOf = (w: number): [number, number] => {
      const r = o.trace(o.makeRay(fp, 0, 0), w).atImage
      if (!r) return [NaN, NaN]
      const a = at(r)
      return [a.x, a.y]
    }
    let cx = 0, cy = 0
    for (const p of raw) {
      cx += p.x
      cy += p.y
    }
    const cen: [number, number] = raw.length ? [cx / raw.length, cy / raw.length] : [NaN, NaN]
    let chief = chiefOf(o.primary)
    if (Number.isNaN(chief[0])) chief = cen
    const ref = so.reference === 'centroid' ? cen : chief
    const launched = pu.length
    out.push({
      field: f,
      ref,
      chief,
      centroid: cen,
      pts: raw.map((p) => ({ ...p, x: p.x - ref[0], y: p.y - ref[1] })),
      perWl: perW.map(({ w, pts }) => {
        const c = chiefOf(w)
        return { w, chief: c, stats: stats(pts, Number.isNaN(c[0]) ? chief : c, launched) }
      }),
      poly: stats(raw, chief, launched * so.wls.length),
      launched,
    })
  }
  return out
}

function statsJson(s: SpotStats): JsonObject {
  return {
    centroid: pair(s.centroid[0], s.centroid[1]),
    geo_radius_centroid: nf(s.geo_c),
    geo_radius_chief: nf(s.geo_chief),
    rays_arrived: s.arrived,
    rays_launched: s.launched,
    rms_radius_centroid: nf(s.rms_c),
    rms_radius_chief: nf(s.rms_chief),
    rms_x: nf(s.rms_x),
    rms_y: nf(s.rms_y),
    vignetted_fraction: nf(s.launched > 0 ? 1 - s.arrived / s.launched : 0),
  }
}

export function airyRadius(o: MockOptics): number {
  return 1.22 * o.wavelengthsUm[o.primary] * 1e-3 * o.workingFNumber(o.primary)
}

export function spotDiagram(o: MockOptics, s: Obj): Out {
  const so: SpotOpts = {
    fields: fieldSel(o, s.fields),
    wls: wlSel(o, s.wavelengths, 'all'),
    pattern: String(s.pattern ?? 'hexapolar'),
    rings: Number(s.rings ?? 6),
    seed: Number(s.seed ?? 1),
    defocus: Number(s.defocus ?? 0),
    reference: String(s.reference ?? 'chief'),
  }
  const data = spotData(o, so)
  const includePts = s.include_points !== false
  const tf = s.through_focus && typeof s.through_focus === 'object' ? (s.through_focus as Obj) : null
  const N = includePts ? data.reduce((a, f) => a + f.pts.length, 0) : 0
  const points = new Float64Array(N * 2)
  const pf = new Uint32Array(N)
  const pw = new Uint32Array(N)
  const slope = tf && includePts ? new Float64Array(N * 2) : null

  let planes: number[] = []
  if (tf) {
    const steps = Math.max(Math.floor(Number(tf.steps ?? tf.samples ?? 11)), 1)
    const lam = o.wavelengthsUm[o.primary] * 1e-3
    let nw = o.workingFNumber(o.primary)
    if (!Number.isFinite(nw)) nw = 1
    const rms = Math.max(0, ...data.map((d) => (Number.isFinite(d.poly.rms_c) ? d.poly.rms_c : 0)))
    const auto = Math.max(8 * lam * nw * nw, 4 * nw * rms)
    const lo = typeof tf.min === 'number' ? tf.min : so.defocus - auto
    const hi = typeof tf.max === 'number' ? tf.max : so.defocus + auto
    planes = linspace(lo, hi, steps)
  }

  let off = 0
  const fieldsJson = data.map((f, k) => {
    const range = [off, includePts ? f.pts.length : 0]
    if (includePts) {
      // Slope of each point relative to the reference ray's slope.
      const fp = o.fieldPoint(f.field)
      const refSlope = (() => {
        const r = o.trace(o.makeRay(fp, 0, 0), o.primary).atImage
        return r && Math.abs(r.d[2]) > 1e-12 ? [r.d[0] / r.d[2], r.d[1] / r.d[2]] : [0, 0]
      })()
      f.pts.forEach((p, j) => {
        points[(off + j) * 2] = p.x
        points[(off + j) * 2 + 1] = p.y
        pf[off + j] = k
        pw[off + j] = p.w
        if (slope) {
          slope[(off + j) * 2] = p.sx - refSlope[0]
          slope[(off + j) * 2 + 1] = p.sy - refSlope[1]
        }
      })
      off += f.pts.length
    }
    const j: JsonObject = {
      centroid: pair(f.centroid[0], f.centroid[1]),
      chief: pair(f.chief[0], f.chief[1]),
      field: f.field,
      field_value: pair(o.sys.fields[f.field].x, o.sys.fields[f.field].y),
      geo_radius: nf(so.reference === 'centroid' ? f.poly.geo_c : f.poly.geo_chief),
      per_wavelength: f.perWl.map((w) => ({
        chief: pair(w.chief[0], w.chief[1]),
        stats: statsJson(w.stats),
        wavelength: w.w,
        wavelength_um: o.wavelengthsUm[w.w],
        weight: o.sys.wavelengths[w.w].weight,
      })),
      point_range: range,
      polychromatic: statsJson(f.poly),
      reference: pair(f.ref[0], f.ref[1]),
      rms_radius: nf(so.reference === 'centroid' ? f.poly.rms_c : f.poly.rms_chief),
    }
    if (tf) {
      j.through_focus = planes.map((d) => {
        const sh = d - so.defocus
        const raw = f.pts.map((p) => ({ x: p.x + sh * p.sx, y: p.y + sh * p.sy }))
        // Points are relative to the reference; shift the reference too (its slope is the chief's).
        const st = stats(raw, [0, 0], f.launched)
        return { defocus: d, rms_radius_centroid: nf(st.rms_c), rms_radius_chief: nf(st.rms_chief) }
      })
    }
    return j
  })

  const out = newOut({
    airy_radius: nf(airyRadius(o)),
    defocus: so.defocus,
    fields: fieldsJson,
    wavelengths: so.wls,
    wavelengths_um: o.wavelengthsUm,
  })
  addBuf(out, 'points', points, [N, 2])
  addBuf(out, 'point_field', pf, [N])
  addBuf(out, 'point_wavelength', pw, [N])
  if (slope) addBuf(out, 'point_slope', slope, [N, 2])
  return out
}

// --- Fans -----------------------------------------------------------------

export interface Curve {
  wavelength: number
  pupil: number[]
  values: (number | null)[]
  cross: (number | null)[]
}
export interface FieldFan {
  field: number
  tangential: Curve[]
  sagittal: Curve[]
}

export function rayFanData(o: MockOptics, fields: number[], wls: number[], samples: number): FieldFan[] {
  const pupil = linspace(-1, 1, Math.max(Math.floor(samples) | 1, 3))
  return fields.map((f) => {
    const fp = o.fieldPoint(f)
    const tan: Curve[] = []
    const sagC: Curve[] = []
    for (const w of wls) {
      const chief = o.trace(o.makeRay(fp, 0, 0), w).atImage?.p
      const eps = (px: number, py: number, axis: 0 | 1): number | null => {
        const p = o.trace(o.makeRay(fp, px, py), w).atImage?.p
        if (!chief || !p) return null
        const e = p[axis] - chief[axis]
        return Number.isFinite(e) ? e : null
      }
      tan.push({ wavelength: w, pupil, values: pupil.map((v) => eps(0, v, 1)), cross: pupil.map((v) => eps(0, v, 0)) })
      sagC.push({ wavelength: w, pupil, values: pupil.map((v) => eps(v, 0, 0)), cross: pupil.map((v) => eps(v, 0, 1)) })
    }
    return { field: f, tangential: tan, sagittal: sagC }
  })
}

export function exitPupil(o: MockOptics): { radius: number; distance: number } {
  const w = o.primary
  const k = o.sys.surfaces.length - 2
  const nk = o.imageN(w)
  const chief = o.paraxial(-o.epZ, 1, w)
  const m0 = o.marginalStart()
  const marg = o.paraxial(m0.y1, m0.u0, w)
  const tk = o.sys.surfaces[k].thickness
  const yk = chief.y[k]
  const uk = chief.nu[k] / nk
  if (!(Math.abs(uk) > 1e-15)) return { radius: o.epRadius, distance: Infinity }
  const dxp = -yk / uk
  const ym = marg.y[k] + (dxp * marg.nu[k]) / nk
  return { radius: Math.abs(ym), distance: Math.abs(tk - dxp) }
}

export function opdFanData(o: MockOptics, fields: number[], wls: number[], samples: number): FieldFan[] {
  const fan = rayFanData(o, fields, wls, samples)
  const { radius, distance } = exitPupil(o)
  const nImg = Math.abs(o.imageN(o.primary))
  const scale = Number.isFinite(distance) && distance > 0 ? (nImg * radius) / distance : 0
  const integrate = (c: Curve): Curve => {
    const lambdaMM = o.wavelengthsUm[c.wavelength] * 1e-3
    const mid = Math.floor(c.pupil.length / 2)
    const vals: (number | null)[] = new Array(c.pupil.length).fill(null)
    vals[mid] = c.values[mid] === null ? null : 0
    for (const dir of [1, -1]) {
      let acc = 0
      let k = mid
      while (k + dir >= 0 && k + dir < c.pupil.length) {
        const a = c.values[k]
        const b = c.values[k + dir]
        if (a === null || b === null) break
        acc += 0.5 * (a + b) * (c.pupil[k + dir] - c.pupil[k])
        k += dir
        const wv = (-scale * acc) / lambdaMM
        vals[k] = Number.isFinite(wv) ? wv : null
      }
    }
    return { wavelength: c.wavelength, pupil: c.pupil, values: vals, cross: [] }
  }
  return fan.map((f) => ({ field: f.field, tangential: f.tangential.map(integrate), sagittal: f.sagittal.map(integrate) }))
}

export function fans(o: MockOptics, s: Obj, opd: boolean): Out {
  const fields = fieldSel(o, s.fields)
  const wls = wlSel(o, s.wavelengths, 'all')
  const samples = Number(s.samples ?? 41)
  const data = opd ? opdFanData(o, fields, wls, samples) : rayFanData(o, fields, wls, samples)
  const curves = (cs: Curve[]): Json =>
    cs.map((c) => {
      const j: JsonObject = { wavelength: c.wavelength, pupil: c.pupil, values: nfs(c.values) }
      if (!opd) j.cross = nfs(c.cross)
      return j
    })
  return newOut({
    quantity: opd ? 'opd' : 'transverse_aberration',
    fields: data.map((f) => ({
      field: f.field,
      field_value: pair(o.sys.fields[f.field].x, o.sys.fields[f.field].y),
      tangential: curves(f.tangential),
      sagittal: curves(f.sagittal),
    })),
    wavelengths_um: o.wavelengthsUm,
  })
}

// --- Field curvature and distortion ---------------------------------------

export function fieldCurvature(o: MockOptics, s: Obj): Out {
  const values = linspace(0, o.maxField, Math.max(Math.floor(Number(s.field_samples ?? 21)), 2))
  const wls = wlSel(o, s.wavelengths, 'all')
  const delta = 0.02
  const crossing = (a: Ray | null, b: Ray | null, axis: 0 | 1): number | null => {
    if (!a || !b || !(Math.abs(a.d[2]) > 1e-12) || !(Math.abs(b.d[2]) > 1e-12)) return null
    const sa = a.d[axis] / a.d[2]
    const sb = b.d[axis] / b.d[2]
    if (!(Math.abs(sa - sb) > 1e-15)) return null
    return -(a.p[axis] - b.p[axis]) / (sa - sb)
  }
  const isAngle = o.sys.fieldKind === 'object_angle'
  const efl = o.efl(o.primary)
  const curves = wls.map((w) => {
    const t: (number | null)[] = [], sg: (number | null)[] = [], dist: (number | null)[] = []
    const par: (number | null)[] = [], real: (number | null)[] = [], ftan: (number | null)[] = [], fth: (number | null)[] = []
    for (const v of values) {
      const fp = o.fieldPointXY(0, v)
      const tr = (px: number, py: number) => o.trace(o.makeRay(fp, px, py), w).atImage
      const chief = tr(0, 0)
      t.push(crossing(tr(0, delta), tr(0, -delta), 1))
      sg.push(crossing(tr(delta, 0), tr(-delta, 0), 0))
      const yPar = o.chiefImagePerUnit * o.fieldParameterY(fp)
      par.push(yPar)
      real.push(chief ? chief.p[1] : null)
      dist.push(chief && Math.abs(yPar) > 1e-12 ? (100 * (chief.p[1] - yPar)) / yPar : chief ? 0 : null)
      const ref1 = isAngle && efl !== null ? efl * Math.tan((v * Math.PI) / 180) : null
      const ref2 = isAngle && efl !== null ? (efl * v * Math.PI) / 180 : null
      ftan.push(chief && ref1 !== null && Math.abs(ref1) > 1e-12 ? (100 * (chief.p[1] - ref1)) / ref1 : null)
      fth.push(chief && ref2 !== null && Math.abs(ref2) > 1e-12 ? (100 * (chief.p[1] - ref2)) / ref2 : null)
    }
    return {
      wavelength: w,
      wavelength_um: o.wavelengthsUm[w],
      tangential: nfs(t),
      sagittal: nfs(sg),
      tangential_close_rays: nfs(t),
      sagittal_close_rays: nfs(sg),
      distortion_percent: nfs(dist),
      f_tan_theta_distortion_percent: nfs(ftan),
      f_theta_distortion_percent: nfs(fth),
      paraxial_height: nfs(par),
      real_height: nfs(real),
    }
  })
  const mf = Math.max(...values, 1e-12)
  return newOut({
    curves,
    field_coordinates: values.map((v) => pair(0, v)),
    field_unit: o.fieldUnit,
    field_values: values,
    normalized_field: values.map((v) => v / mf),
    wavelengths_um: o.wavelengthsUm,
  })
}

// --- First order ----------------------------------------------------------

interface Entry {
  key: string
  label: string
  value: number
  unit: string
}

export function firstOrderEntries(o: MockOptics, w: number): Entry[] {
  const e: Entry[] = []
  const k = o.sys.surfaces.length - 2
  const nk = o.imageN(w)
  const add = (key: string, label: string, v: number | null, unit: string) => {
    if (v !== null && !Number.isNaN(v)) e.push({ key, label, value: v, unit })
  }
  const efl = o.efl(w)
  add('efl', 'Effective focal length', efl ?? Infinity, 'mm')
  const a = o.paraxial(1, 0, w)
  if (Math.abs(a.nu[k]) > 1e-15) add('bfl', 'Back focal length', (-a.y[k] * nk) / a.nu[k], 'mm')
  add('entrance_pupil_diameter', 'Entrance pupil diameter', 2 * o.epRadius, 'mm')
  add('ep_position', 'Entrance pupil position (from surface 1)', o.epZ, 'mm')
  const xp = exitPupil(o)
  add('xp_diameter', 'Exit pupil diameter', 2 * xp.radius, 'mm')
  add('xp_distance', 'Exit pupil distance (to image)', xp.distance, 'mm')
  if (o.objectDistance === null && efl !== null) add('image_f_number', 'Image-space F/#', Math.abs(efl) / (2 * o.epRadius), '')
  const nw = o.workingFNumber(w)
  add('working_f_number', 'Paraxial working F/#', nw, '')
  add('image_na', 'Image-space NA', 1 / (2 * nw), '')
  add('max_field', 'Maximum field', o.maxField, o.fieldUnit)
  add('paraxial_image_height', 'Paraxial image height', o.chiefImagePerUnit * o.fieldParameterY(o.fieldPointXY(0, o.maxField)), 'mm')
  add('total_track', 'Total track (surface 1 → image)', o.trackLength, 'mm')
  if (o.objectDistance !== null) {
    const d = o.objectDistance
    const u0 = o.epRadius / o.objectToPupil(d)
    const m = o.paraxial(u0 * d, u0, w)
    if (Math.abs(m.nu[k]) > 1e-15) add('magnification', 'Paraxial magnification', (o.nAfter[0][w] * u0) / m.nu[k], '')
  }
  let petz = 0
  for (let i = 1; i < o.sys.surfaces.length - 1; i++) {
    const sf = o.sys.surfaces[i]
    if (sf.isCB || sf.interaction === 'dummy') continue
    const n1 = o.nBefore(i, w) * (i > 1 ? o.travelSign[i - 1] : 1)
    const n2 = o.nAfter[i][w] * o.travelSign[i]
    petz += (shapeCurvature(sf.shape) * (n2 - n1)) / (n1 * n2)
  }
  if (Math.abs(petz) > 1e-15) add('petzval_radius', 'Petzval radius', -1 / (nk * petz), 'mm')
  return e
}

const entriesJson = (es: Entry[]): Json[] => es.map((x) => ({ key: x.key, label: x.label, value: nf(x.value), unit: x.unit }))

function paraxialXY(o: MockOptics): JsonObject {
  const k = o.sys.surfaces.length - 2
  return {
    primary_wavelength: o.primary,
    squeeze_definition: 'Signed EFL_y / EFL_x; unit-index referenced.',
    symmetry: 'rotational',
    warnings: [],
    wavelengths: o.wavelengthsUm.map((um, w) => {
      const efl = o.efl(w)
      const a = o.paraxial(1, 0, w)
      const bfl = Math.abs(a.nu[k]) > 1e-15 ? (-a.y[k] * o.imageN(w)) / a.nu[k] : Infinity
      return {
        wavelength: w,
        wavelength_um: um,
        efl_x: nf(efl ?? Infinity),
        efl_y: nf(efl ?? Infinity),
        bfl_x: nf(bfl),
        bfl_y: nf(bfl),
        squeeze: 1,
        doe_power: [],
      }
    }),
  }
}

export function firstOrder(o: MockOptics, s: Obj): Out {
  const wls = wlSel(o, s.wavelength, 'primary')
  const use = wls.length ? wls : [o.primary]
  const tables = use.map((w) => ({
    wavelength: w,
    wavelength_um: o.wavelengthsUm[w],
    entries: entriesJson(firstOrderEntries(o, w)),
  }))
  return newOut({
    entries: tables[0].entries,
    paraxial_xy: paraxialXY(o),
    system: {
      name: o.sys.name,
      surfaces: o.sys.surfaces.length,
      stop: o.sys.stop,
      wavelengths_um: o.wavelengthsUm,
      primary_wavelength: o.primary,
      field_kind: o.sys.fieldKind,
      field_unit: o.fieldUnit,
      max_field: o.maxField,
      object_at_infinity: o.sys.objectAtInfinity,
      ray_aiming: (o.sys.raw.ray_aiming ?? { mode: 'off' }) as Json,
    },
    tables,
    wavelength_um: tables[0].wavelength_um,
  })
}

// --- System summary and validate ------------------------------------------

export function systemSummary(o: MockOptics): Out {
  const sys = o.sys
  const n = sys.surfaces.length
  const nw = o.wavelengthsUm.length
  const raw = sys.raw
  const nAfter = new Float64Array(n * nw)
  const vg = new Float64Array(n * 3)
  const surfaces = sys.surfaces.map((s, i) => {
    const c = shapeCurvature(s.shape)
    const nv = materialNdVd(s.interaction === 'refract' && !s.isCB ? s.material : sys.surfaces[Math.max(i - 1, 0)].material)
    for (let w = 0; w < nw; w++) nAfter[i * nw + w] = o.nAfter[i][w]
    const infObj = i === 0 && o.objectDistance === null
    vg[i * 3] = infObj ? NaN : 0
    vg[i * 3 + 1] = infObj ? NaN : 0
    vg[i * 3 + 2] = infObj ? NaN : o.vertexZ[i]
    const t = s.shape.type
    return {
      id: s.id,
      index: i,
      label: s.label,
      type: t === 'conic' ? (shapeConic(s.shape) === 0 && c !== 0 ? 'sphere' : c === 0 && shapeConic(s.shape) === 0 ? 'plane' : 'conic') : t,
      shape: s.shape as Json,
      curvature: c,
      conic: shapeConic(s.shape),
      radius: nf(c !== 0 ? 1 / c : Infinity),
      thickness: nf(s.thickness),
      to_next_is_axial: true,
      material: s.material.kind === 'catalog' ? s.material.name : s.material.kind === 'air' ? null : s.material.name,
      material_ref: s.material.raw,
      interaction: s.interaction === 'reflect' ? 'reflect' : 'refract',
      is_stop: i === sys.stop,
      coating: (s.raw.coating ?? null) as Json,
      semi_diameter: (s.raw.semi_diameter ?? { mode: 'auto' }) as Json,
      aperture: (s.raw.aperture ?? null) as Json,
      n_after: o.nAfter[i],
      alpha_after_per_mm: o.nAfter[i].map(() => 0),
      nd_after: nf(nv.nd),
      vd_after: nf(nv.vd),
      frame: infObj ? null : { origin: [0, 0, o.vertexZ[i]], rot: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] },
    }
  })
  const out = newOut({
    aperture: (raw.aperture ?? {}) as Json,
    environment: (raw.environment ?? { temperature_c: 20, pressure_atm: 1 }) as Json,
    fields: (raw.fields ?? {}) as Json,
    first_to_image_distance: o.trackLength,
    n_object: o.nAfter[0],
    name: sys.name,
    num_configurations: Array.isArray(raw.configurations) ? raw.configurations.length : 0,
    num_surfaces: n,
    object_at_infinity: sys.objectAtInfinity,
    paraxial_xy: paraxialXY(o),
    primary_wavelength: o.primary,
    ray_aiming: (raw.ray_aiming ?? { mode: 'off' }) as Json,
    stop: sys.stop,
    surfaces: surfaces as Json,
    symmetry: 'rotational',
    warnings: o.warnings,
    wavelength_weights: sys.wavelengths.map((w) => w.weight),
    wavelengths_um: o.wavelengthsUm,
  })
  addBuf(out, 'n_after', nAfter, [n, nw])
  addBuf(out, 'vertex_global', vg, [n, 3])
  return out
}

export function validate(o: MockOptics): Out {
  const issues = structuralIssues(o.sys.raw as JsonObject).map((i) => ({
    severity: i.severity,
    code: i.code,
    surface: i.surface,
    message: i.message,
  }))
  for (const w of o.warnings) issues.push({ severity: 'warning', code: 'mock', surface: null, message: w })
  const errors = issues.filter((i) => i.severity === 'error').length
  return newOut({ ok: errors === 0, errors, warnings: issues.length - errors, issues })
}

export function glassInfo(s: Obj, catalog: Record<string, { nd: number; vd: number }>): Out | null {
  const name = String(s.name ?? '').toUpperCase()
  const g = catalog[name]
  if (!g) return null
  const dn = Number(s.offset?.dn_d ?? 0)
  const dv = Number(s.offset?.dv_d ?? 0)
  return newOut({
    name,
    catalog: s.catalog ?? 'MOCK',
    catalog_nd: g.nd,
    catalog_vd: g.vd,
    nd: g.nd + dn,
    vd: g.vd + dv,
    offset: { dn_d: dn, dv_d: dv },
  })
}
