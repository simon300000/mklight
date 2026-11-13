// The engine's `layout` scene and `footprint` overlays (docs/RESULTS.md), read
// into typed arrays for drawing (as EngineLayout.swift, SceneGeometry.swift
// and FootprintOverlay.swift).

import type { AnalysisResult, JsonObject } from '@/engine/types'
import { nf } from '@/model/format'
import type { LayoutMode } from './camera'

/** Polylines of the engine's `layout` scene. */
export interface Scene {
  /** Absolute global x, y, z per vertex. */
  xyz: Float64Array
  offsets: Uint32Array
  kinds: Uint8Array
  surfaces: Int32Array
  fields: Int32Array
  wavelengths: Int32Array
  /** Bit 0: in the section; bit 1: vignetted ray; bit 2: DOE surface. */
  flags: Uint8Array
  bounds: { min: number[]; max: number[] }
  surfaceInfo: JsonObject[]
  wavelengthsUm: number[]
  section: 'yz' | 'xz'
  /** The optional surface mesh: absolute vertices (9 per triangle) and the surface of each triangle. */
  mesh: { xyz: Float64Array; surfaces: Int32Array } | null
}

export const KIND = { profile: 0, rim: 1, edge: 2, stop: 3, image: 4, ray: 5, axis: 6, outline: 7 } as const

export function readScene(r: AnalysisResult): Scene | null {
  const b = r.buffers
  const offsets = b.offsets as Uint32Array | undefined
  const kinds = b.kinds as Uint8Array | undefined
  if (!offsets || !kinds) return null
  const origin = ((r.json.origin as number[] | undefined) ?? [0, 0, 0]).map((x) => Number(x) || 0)
  let xyz = b.vertices_f64 as Float64Array | undefined
  if (!xyz && b.vertices) {
    const v = b.vertices as Float32Array
    xyz = new Float64Array(v.length)
    for (let i = 0; i < v.length; i++) xyz[i] = v[i] + origin[i % 3]
  }
  if (!xyz) return null
  const n = kinds.length
  const bounds = r.json.bounds as JsonObject | undefined
  let mesh: Scene['mesh'] = null
  const m = b.surface_mesh as Float32Array | undefined
  if (m && m.length >= 9) {
    const t = Math.floor(m.length / 9)
    const mx = new Float64Array(t * 9)
    for (let i = 0; i < t * 9; i++) mx[i] = m[i] + origin[i % 3]
    const ids = b.mesh_surfaces as ArrayLike<number> | undefined
    const ms = new Int32Array(t)
    for (let k = 0; k < t; k++) ms[k] = ids && k < ids.length ? Number(ids[k]) : -1
    mesh = { xyz: mx, surfaces: ms }
  }
  const scene: Scene = {
    xyz,
    offsets,
    kinds,
    surfaces: (b.surfaces as Int32Array) ?? new Int32Array(n).fill(-1),
    fields: (b.fields as Int32Array) ?? new Int32Array(n).fill(-1),
    wavelengths: (b.wavelengths as Int32Array) ?? new Int32Array(n).fill(-1),
    flags: (b.flags as Uint8Array) ?? new Uint8Array(n).fill(1),
    bounds: {
      min: ((bounds?.min as unknown[]) ?? [0, 0, 0]).map(nf),
      max: ((bounds?.max as unknown[]) ?? [0, 0, 0]).map(nf),
    },
    surfaceInfo: (r.json.surfaces as JsonObject[]) ?? [],
    wavelengthsUm: ((r.json.wavelengths_um as number[] | undefined) ?? []).map(Number),
    section: r.json.section === 'xz' ? 'xz' : 'yz',
    mesh,
  }
  markNearSection(scene)
  return scene
}

/** For a design whose surfaces are all off the section plane (a tilted or
 *  decentred system), profiles, stops, edges and rays within 2 % of the
 *  scene's height of the plane count as in the section, so the section view
 *  still draws the build. */
function markNearSection(s: Scene) {
  const lens = new Set<number>([KIND.profile, KIND.stop, KIND.edge, KIND.ray])
  for (let p = 0; p < s.kinds.length; p++) if (s.flags[p] & 1 && lens.has(s.kinds[p])) return
  const offAxis = s.section === 'xz' ? 1 : 0
  const upAxis = s.section === 'xz' ? 0 : 1
  let height = 0
  for (let i = 0; i < s.xyz.length; i += 3) {
    const h = Math.abs(s.xyz[i + upAxis])
    if (Number.isFinite(h)) height = Math.max(height, h)
  }
  if (!(height > 0)) return
  const tol = 0.02 * height
  const flags = new Uint8Array(s.flags)
  for (let p = 0; p < s.kinds.length; p++) {
    if (flags[p] & 1 || !(lens.has(s.kinds[p]) || s.kinds[p] === KIND.axis)) continue
    let near = true
    for (let i = s.offsets[p]; i < s.offsets[p + 1]; i++) if (Math.abs(s.xyz[3 * i + offAxis]) > tol) near = false
    if (near) flags[p] |= 1
  }
  s.flags = flags
}

/** Whether polyline `p` is drawn in `mode`: sections draw what is in the section plane. */
export const drawn = (s: Scene, p: number, mode: LayoutMode) => mode === '3d' || (s.flags[p] & 1) !== 0

const isLens = (k: number) => k !== KIND.ray && k !== KIND.axis && k !== KIND.image

/** Points to frame (flat xyz): everything drawn, or the lens with room for the rays around it. */
export function fitPoints(s: Scene, mode: LayoutMode, lensOnly = false): number[] {
  const pts: number[] = []
  for (let p = 0; p < s.kinds.length; p++) {
    if (!drawn(s, p, mode) || (lensOnly && !isLens(s.kinds[p]))) continue
    if (s.kinds[p] === KIND.axis) continue
    for (let i = s.offsets[p]; i < s.offsets[p + 1]; i++) {
      const x = s.xyz[3 * i], y = s.xyz[3 * i + 1], z = s.xyz[3 * i + 2]
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) pts.push(x, y, z)
    }
  }
  if (!pts.length && lensOnly) return fitPoints(s, mode)
  if (lensOnly && pts.length) {
    let z0 = Infinity, z1 = -Infinity, h = 0
    for (let i = 0; i < pts.length; i += 3) {
      z0 = Math.min(z0, pts[i + 2])
      z1 = Math.max(z1, pts[i + 2])
      h = Math.max(h, Math.abs(pts[i]), Math.abs(pts[i + 1]))
    }
    const m = 0.6 * Math.max(z1 - z0, 2 * h)
    const up = mode === 'xz' ? [1.3 * h, 0, 0] : [0, 1.3 * h, 0]
    pts.push(0, 0, z0 - m, 0, 0, z1 + m, up[0], up[1], z0 + up[2], -up[0], -up[1], z0 - up[2])
  }
  return pts
}

/** A slim, long scene (a slow lens far from its image): the first fit frames the lens. */
export function isSlim(s: Scene, mode: LayoutMode): boolean {
  const ext = (p: number[]) => {
    let z0 = Infinity, z1 = -Infinity, h = 0
    for (let i = 0; i < p.length; i += 3) {
      z0 = Math.min(z0, p[i + 2])
      z1 = Math.max(z1, p[i + 2])
      h = Math.max(h, Math.abs(p[i]), Math.abs(p[i + 1]))
    }
    return p.length ? { z: z1 - z0, h: 2 * h } : { z: 0, h: 0 }
  }
  const a = ext(fitPoints(s, mode)), l = ext(fitPoints(s, mode, true))
  return a.h > 0 && a.z > 12 * a.h && l.z * 5 < a.z
}

// MARK: Footprints

export interface Footprint {
  surface: number
  field: number
  wavelength: number
  /** Counterclockwise, unclosed, global (flat xyz). */
  points: number[]
}

/** Beam outlines from a `footprint` result's `overlays` and `global_outlines`. */
export function readFootprints(r: AnalysisResult | undefined): Footprint[] {
  const overlays = r?.json.overlays as JsonObject[] | undefined
  const buf = r?.buffers.global_outlines as Float64Array | undefined
  if (!overlays?.length || !buf) return []
  const rows = Math.floor(buf.length / 3)
  const out: Footprint[] = []
  for (const o of overlays) {
    const range = o.outline as number[] | undefined
    if (!range || range.length !== 2 || range[1] < 3) continue
    const [start, n] = range
    if (start < 0 || start + n > rows) continue
    const pts = Array.from(buf.subarray(start * 3, (start + n) * 3))
    if (!pts.every(Number.isFinite)) continue
    out.push({ surface: Number(o.surface ?? 0), field: Number(o.field ?? 0), wavelength: Number(o.wavelength ?? 0), points: pts })
  }
  return out
}

// MARK: Symmetry

const NON_ROTATIONAL = new Set(['toroid', 'biconic', 'xy_polynomial', 'zernike_sag'])

/** The design has no rotational symmetry (both sections matter): the
 *  engine's symmetry class from the first-order result, or a surface kind
 *  without rotational symmetry. */
export function isNonRotational(summary: JsonObject | undefined, surfaces: { shape: { type: string } }[] | undefined): boolean {
  const sym = summary?.symmetry
  if (typeof sym === 'string' && sym !== 'rotational') return true
  return (surfaces ?? []).some((s) => NON_ROTATIONAL.has(s.shape?.type))
}
