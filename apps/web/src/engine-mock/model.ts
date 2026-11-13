// MOCK ENGINE — parsing and structural validation of the engine's
// `OpticalSystem` JSON (port of the Swift model's `structuralIssues()` subset).
import type { Json, JsonObject } from '../engine/types'

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Obj = Record<string, any>

export const CATALOG: Record<string, { nd: number; vd: number }> = {
  'N-BK7': { nd: 1.5168, vd: 64.17 },
  'N-BAK1': { nd: 1.5725, vd: 57.55 },
  'N-SK16': { nd: 1.62041, vd: 60.32 },
  'N-LAK9': { nd: 1.691, vd: 54.71 },
  'N-BAF10': { nd: 1.67003, vd: 47.11 },
  'N-LAF2': { nd: 1.74397, vd: 44.85 },
  F2: { nd: 1.62004, vd: 36.37 },
  SF5: { nd: 1.6727, vd: 32.21 },
  'N-SF6': { nd: 1.80518, vd: 25.36 },
  'N-SF11': { nd: 1.78472, vd: 25.68 },
  'N-LASF9': { nd: 1.85025, vd: 32.17 },
  FUSED_SILICA: { nd: 1.45846, vd: 67.82 },
  CAF2: { nd: 1.43385, vd: 94.99 },
}

export const MOCK_WARNING = 'MOCK ENGINE: synthetic data, not an optical computation'

/** A JSON number, or the engine's "inf" / "-inf" / "nan" strings. */
export function parseNum(v: unknown): number | undefined {
  if (typeof v === 'number') return v
  if (v === 'inf') return Infinity
  if (v === '-inf') return -Infinity
  if (v === 'nan') return NaN
  return undefined
}

export type MatKind = 'air' | 'vacuum' | 'constant' | 'model' | 'catalog'
export interface Mat {
  kind: MatKind
  n: number
  nd: number
  vd: number
  name: string
  dnd: number
  dvd: number
  raw: Json
}

export type Shape = Obj & { type: string }

export type Aperture =
  | { type: 'circular'; radius: number }
  | { type: 'annular'; inner: number; outer: number }
  | { type: 'rectangular'; hx: number; hy: number }
  | { type: 'elliptical'; sx: number; sy: number }

export interface Surf {
  id: number
  label: string
  shape: Shape
  thickness: number
  material: Mat
  interaction: 'refract' | 'reflect' | 'dummy'
  sdFixed: number | null
  aperture: Aperture | null
  mechSD: number | null
  isCB: boolean
  hasPlacement: boolean
  raw: Obj
}

export interface FieldDef {
  x: number
  y: number
  weight: number
}

export interface Sys {
  raw: Obj
  name: string
  surfaces: Surf[]
  stop: number
  aperture: { type: string; value: number | null }
  fieldKind: 'object_angle' | 'object_height' | 'paraxial_image_height' | 'real_image_height'
  fields: FieldDef[]
  wavelengths: { um: number; weight: number }[]
  primary: number
  rayAimingReal: boolean
  objectAtInfinity: boolean
}

function num(v: unknown, d: number): number {
  const x = parseNum(v)
  return x === undefined ? d : x
}

function parseMaterial(m: Obj | null | undefined): Mat {
  const t = m?.type ?? 'air'
  const base = { n: 1, nd: 1, vd: 0, name: 'AIR', dnd: 0, dvd: 0, raw: (m ?? { type: 'air' }) as Json }
  switch (t) {
    case 'vacuum':
      return { kind: 'vacuum', ...base, name: 'VACUUM' }
    case 'constant':
      return { kind: 'constant', ...base, n: num(m?.n, 1), nd: num(m?.n, 1), name: `n=${num(m?.n, 1)}` }
    case 'model':
      return { kind: 'model', ...base, nd: num(m?.nd, 1.5), vd: num(m?.vd, 50), name: 'MODEL' }
    case 'catalog':
      return {
        kind: 'catalog',
        ...base,
        name: String(m?.name ?? ''),
        dnd: num(m?.offset?.dn_d, 0),
        dvd: num(m?.offset?.dv_d, 0),
      }
    default:
      return { kind: 'air', ...base }
  }
}

function parseAperture(a: Obj | null | undefined): Aperture | null {
  if (!a || typeof a !== 'object') return null
  switch (a.type) {
    case 'circular':
      return { type: 'circular', radius: num(a.radius, 0) }
    case 'annular':
      return { type: 'annular', inner: num(a.inner_radius, 0), outer: num(a.outer_radius, 0) }
    case 'rectangular':
      return { type: 'rectangular', hx: num(a.half_x, 0), hy: num(a.half_y, 0) }
    case 'elliptical':
      return { type: 'elliptical', sx: num(a.semi_x, 0), sy: num(a.semi_y, 0) }
    default:
      return null
  }
}

export function apertureBoundingRadius(a: Aperture): number {
  switch (a.type) {
    case 'circular':
      return a.radius
    case 'annular':
      return a.outer
    case 'rectangular':
      return Math.hypot(a.hx, a.hy)
    case 'elliptical':
      return Math.max(a.sx, a.sy)
  }
}

function nonZeroDeep(v: unknown): boolean {
  if (typeof v === 'number') return v !== 0
  if (Array.isArray(v)) return v.some(nonZeroDeep)
  if (v && typeof v === 'object') return Object.entries(v).some(([k, x]) => k !== 'order' && k !== 'restore_after' && nonZeroDeep(x))
  return false
}

export function parseSystem(system: JsonObject): Sys {
  const s = system as Obj
  const rawSurfaces: Obj[] = Array.isArray(s.surfaces) ? s.surfaces : []
  const surfaces: Surf[] = rawSurfaces.map((r, i) => {
    const shape: Shape = (r.shape && typeof r.shape === 'object' ? r.shape : { type: 'plane' }) as Shape
    const sd = r.semi_diameter
    const interaction = r.interaction === 'reflect' || r.interaction === 'dummy' ? r.interaction : 'refract'
    return {
      id: typeof r.id === 'number' ? r.id : i,
      label: typeof r.label === 'string' ? r.label : '',
      shape,
      thickness: num(r.thickness, 0),
      material: parseMaterial(r.material),
      interaction,
      sdFixed: sd?.mode === 'fixed' ? num(sd.value, NaN) : null,
      aperture: parseAperture(r.aperture),
      mechSD: parseNum(r.edge?.mechanical_semi_diameter) ?? null,
      isCB: shape.type === 'coordinate_break',
      hasPlacement: r.placement != null && nonZeroDeep(r.placement),
      raw: r,
    }
  })
  const fkind = s.fields?.kind
  const wl: { um: number; weight: number }[] = (s.wavelengths?.wavelengths ?? []).map((w: Obj) => ({
    um: num(w.um, NaN),
    weight: num(w.weight, 1),
  }))
  const ap = s.aperture ?? {}
  return {
    raw: s,
    name: typeof s.name === 'string' ? s.name : '',
    surfaces,
    stop: num(s.stop, 1),
    aperture: { type: String(ap.type ?? 'entrance_pupil_diameter'), value: parseNum(ap.value) ?? null },
    fieldKind:
      fkind === 'object_height' || fkind === 'paraxial_image_height' || fkind === 'real_image_height'
        ? fkind
        : 'object_angle',
    fields: (s.fields?.fields ?? []).map((f: Obj) => ({ x: num(f.x, 0), y: num(f.y, 0), weight: num(f.weight, 1) })),
    wavelengths: wl,
    primary: num(s.wavelengths?.primary, 0),
    rayAimingReal: s.ray_aiming?.mode === 'real',
    objectAtInfinity: surfaces.length > 0 && !Number.isFinite(surfaces[0].thickness),
  }
}

export interface ModelIssue {
  severity: 'error' | 'warning'
  surface: number | null
  message: string
  code: string
}

/** Errors the engine's structural validation would report (subset). */
export function structuralIssues(system: JsonObject): ModelIssue[] {
  const out: ModelIssue[] = []
  const err = (message: string, surface: number | null = null, code = 'structure') =>
    out.push({ severity: 'error', surface, message, code })
  const sys = parseSystem(system)
  const n = sys.surfaces.length

  // Non-finite numbers (the JSON check). Thickness strings are handled below.
  const walk = (v: unknown, path: string, surface: number | null) => {
    if (typeof v === 'number') {
      if (!Number.isFinite(v)) err(`${path} is not a finite number`, surface)
    } else if (Array.isArray(v)) v.forEach((x, i) => walk(x, `${path}[${i}]`, surface))
    else if (v && typeof v === 'object')
      for (const [k, x] of Object.entries(v)) walk(x, path ? `${path}.${k}` : k, surface)
  }
  const raw = system as Obj
  for (const [k, v] of Object.entries(raw)) {
    if (k === 'surfaces') {
      ;(Array.isArray(v) ? v : []).forEach((s: Obj, i: number) => {
        for (const [sk, sv] of Object.entries(s ?? {})) {
          if (sk === 'thickness') continue
          walk(sv, `surfaces[${i}].${sk}`, i)
        }
      })
    } else walk(v, k, null)
  }

  if (n < 3) {
    err('a system needs object, at least one surface, and image')
    return out
  }
  if (!(sys.stop >= 1 && sys.stop <= n - 2)) err(`stop index ${sys.stop} must be between 1 and ${n - 2}`)
  if (sys.aperture.value !== null && !(sys.aperture.value > 0 && Number.isFinite(sys.aperture.value)))
    err(`[invalid_aperture_value] aperture.value: must be positive and finite (got ${sys.aperture.value})`)
  if (sys.wavelengths.length === 0 || !(sys.primary >= 0 && sys.primary < sys.wavelengths.length))
    err('wavelength set empty or primary out of range')
  if (sys.wavelengths.some((w) => !(w.um > 0 && Number.isFinite(w.um)) || w.weight < 0))
    err('wavelengths must be positive and finite with weights ≥ 0')
  if (sys.fields.length === 0) err('at least one field is required')
  if (sys.objectAtInfinity && sys.fieldKind === 'object_height')
    err('object-height fields require a finite object distance')
  if (sys.objectAtInfinity && sys.aperture.type === 'object_space_na')
    err('object-space NA requires a finite object distance')
  if (sys.objectAtInfinity && sys.fieldKind === 'object_angle' && sys.fields.some((f) => Math.hypot(f.x, f.y) >= 90))
    err('field angles must be below 90°')
  sys.fields.forEach((f, k) => {
    if (f.weight < 0) err(`[negative_weight] fields.fields[${k}].weight: must not be negative`)
  })

  const ids = new Set<number>()
  sys.surfaces.forEach((s, i) => {
    if (ids.has(s.id)) err(`duplicate surface id ${s.id}`, i)
    ids.add(s.id)
    const rawT = (s.raw as Obj).thickness
    const t = parseNum(rawT)
    if (t === undefined || Number.isNaN(t)) err(`surfaces[${i}].thickness is not a finite number`, i)
    else if (i > 0 && !Number.isFinite(t) && i !== n - 1) err('only the object may have infinite thickness', i)
    if (s.sdFixed !== null && !(s.sdFixed > 0)) err('fixed semi-diameter must be > 0', i)
    if ((i === 0 || i === n - 1) && s.isCB) err('object/image cannot be coordinate breaks', i)
    if (s.material.kind === 'catalog' && !CATALOG[s.material.name.toUpperCase()])
      err(`unknown material '${s.material.name}' on surface ${i}`, i)
    if (s.material.kind === 'model' && !(s.material.nd > 0 && s.material.vd > 0))
      err('model glass needs positive n_d and V_d', i)
  })
  if (sys.surfaces[n - 1].interaction === 'reflect') err('the image surface cannot be a mirror', n - 1)
  return out
}

/** JSON with every object's keys sorted (the hash input). */
export function canonicalJson(v: unknown): string {
  if (v === null || typeof v !== 'object') {
    if (typeof v === 'number' && !Number.isFinite(v)) return JSON.stringify(String(v))
    return JSON.stringify(v) ?? 'null'
  }
  if (Array.isArray(v)) return `[${v.map(canonicalJson).join(',')}]`
  const o = v as Obj
  return `{${Object.keys(o)
    .filter((k) => o[k] !== undefined)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${canonicalJson(o[k])}`)
    .join(',')}}`
}

// --- SHA-256 (fallback for contexts without crypto.subtle) ---

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
  0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
  0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
])

function sha256Fallback(data: Uint8Array): string {
  const h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ])
  const len = data.length
  const padded = new Uint8Array(((len + 9 + 63) >> 6) << 6)
  padded.set(data)
  padded[len] = 0x80
  const dv = new DataView(padded.buffer)
  dv.setUint32(padded.length - 8, Math.floor((len * 8) / 0x100000000))
  dv.setUint32(padded.length - 4, (len * 8) >>> 0)
  const w = new Uint32Array(64)
  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n))
  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = dv.getUint32(off + i * 4)
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0
    }
    let [a, b, c, d, e, f, g, hh] = h
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
      const ch = (e & f) ^ (~e & g)
      const t1 = (hh + S1 + ch + K[i] + w[i]) | 0
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
      const maj = (a & b) ^ (a & c) ^ (b & c)
      const t2 = (S0 + maj) | 0
      hh = g
      g = f
      f = e
      e = (d + t1) | 0
      d = c
      c = b
      b = a
      a = (t1 + t2) | 0
    }
    h[0] += a
    h[1] += b
    h[2] += c
    h[3] += d
    h[4] += e
    h[5] += f
    h[6] += g
    h[7] += hh
  }
  return Array.from(h, (x) => x.toString(16).padStart(8, '0')).join('')
}

export async function sha256Hex(text: string): Promise<string> {
  const data = new TextEncoder().encode(text)
  const subtle = (globalThis as { crypto?: Crypto }).crypto?.subtle
  if (subtle) {
    try {
      const d = await subtle.digest('SHA-256', data)
      return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, '0')).join('')
    } catch {
      // fall through
    }
  }
  return sha256Fallback(data)
}
