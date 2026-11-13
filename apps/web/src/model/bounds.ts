// Variable bounds (apps/macos/Sources/MKLightUI/ViewModels/BoundsDisplay.swift,
// Document/DesignDocument.swift `DesignVariable`): radius ↔ curvature bounds,
// sensible bounds for new variables, pins and out-of-bounds checks.
//
// The engine bounds the curvature c = 1/R; the app lets the designer type a
// radius window instead. A radius window `min … max` is read on the projective
// line of radii, rising from `min` to `max` and wrapping through ±∞ (a flat
// surface) when `min > max`: its curvatures form the interval [1/max, 1/min].
//
// * 40 … 60 → c ∈ [1/60, 1/40]; −60 … −40 → c ∈ [−1/40, −1/60].
// * 100 … −100 passes through ∞: |R| ≥ 100, c ∈ [−0.01, 0.01].
// * An empty side is open: an empty max is +∞ (R ≥ 50 is c ∈ [0, 0.02]), an
//   empty min −∞; both empty is unbounded.
// * 0 is the limit R → 0 (no bound on the curvature on that side):
//   0 … 50 is c ≥ 0.02.
// * ∞ … ∞ holds the surface flat (c = 0).
//
// A window that contains R = 0 itself (−50 … 50, or 60 … 40, which wraps
// through ∞ and then 0) is not an interval of curvatures and is refused.

import type { ParamPath, VariableSpec } from './params'
import { surfaceIndexOfId } from './params'
import { sig } from './format'
import { nf } from './format'
import type { System } from './system'

export type CurvatureBoundsUnit = 'radius' | 'curvature'

export interface Bounds {
  lower: number | null
  upper: number | null
}

export interface RadiusWindow {
  min: number | null
  max: number | null
}

export const CONTAINS_ZERO =
  'This radius window passes through R = 0, which is not a range of curvatures. For |R| ≥ r, enter min r and max −r (through ∞).'

export const RADIUS_HELP =
  'Bounds as a radius window in mm; the optimizer bounds the curvature c = 1/R. Empty is open on that side. Min above max wraps through flat (R = ∞): min 100, max −100 is |R| ≥ 100. 0 leaves the curvature unbounded on that side.'

/** −0 → 0, so stored bounds read 0. */
const clean = (v: number) => (v === 0 ? 0 : v)

/** Curvature bounds for a typed radius window (null = an empty field), or an
 *  error when the window contains R = 0. */
export function curvatureBounds(minRadius: number | null, maxRadius: number | null): { ok: true; bounds: Bounds } | { ok: false; error: string } {
  if (minRadius == null && maxRadius == null) return { ok: true, bounds: { lower: null, upper: null } }
  const lo = minRadius ?? -Infinity
  const hi = maxRadius ?? Infinity
  // c_upper = 1/min (min → 0⁺: unbounded above); c_lower = 1/max (max → 0⁻: unbounded below).
  const upper = lo === 0 ? null : clean(1 / lo)
  const lower = hi === 0 ? null : clean(1 / hi)
  if (lower != null && upper != null && lower > upper) return { ok: false, error: CONTAINS_ZERO }
  return { ok: true, bounds: { lower, upper } }
}

/** The radius window that shows curvature bounds (null = an empty field);
 *  the inverse of `curvatureBounds`. */
export function radiusWindow(lower: number | null, upper: number | null): RadiusWindow {
  if (lower == null && upper == null) return { min: null, max: null }
  // A pinned flat surface: ∞ … ∞.
  if (lower === 0 && upper === 0) return { min: Infinity, max: Infinity }
  const minR = upper == null ? 0 : upper === 0 ? -Infinity : 1 / upper
  const maxR = lower == null ? 0 : lower === 0 ? Infinity : 1 / lower
  return { min: minR === -Infinity ? null : minR, max: maxR === Infinity ? null : maxR }
}

/** Whether the window wraps through a flat surface (min > max, both set). */
export function passesThroughFlat(w: RadiusWindow): boolean {
  const { min: a, max: b } = w
  if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) return false
  return a > b && a > 0 && b < 0
}

/** Compact: 4 significant digits, ∞ for infinities. */
export const compactBound = (v: number) => sig(v, 4)

/** The bounds in the other unit, e.g. `c 0.0167 … 0.025 1/mm` or `R 40 … 60 mm`. */
export function counterpart(b: Bounds, shownAs: CurvatureBoundsUnit): string | null {
  if (b.lower == null && b.upper == null) return null
  if (shownAs === 'radius') {
    return `c ${b.lower == null ? '−∞' : compactBound(b.lower)} … ${b.upper == null ? '∞' : compactBound(b.upper)} 1/mm`
  }
  const r = radiusWindow(b.lower, b.upper)
  return `R ${r.min == null ? '−∞' : compactBound(r.min)}${passesThroughFlat(r) ? ' … ∞ … ' : ' … '}${r.max == null ? '∞' : compactBound(r.max)} mm`
}

export const isCurvature = (p: ParamPath) => p.param === 'curvature' || (p.param === 'shape_param' && p.parameter === 'curvature_y')

export const isPinned = (v: Bounds) => v.lower != null && v.lower === v.upper

export function outOfBounds(v: Bounds, value: number): boolean {
  return (v.lower != null && value < v.lower - 1e-12) || (v.upper != null && value > v.upper + 1e-12)
}

/** The smallest glass centre thickness a new glass-thickness variable gets. */
export const GLASS_MINIMUM = 1.0
const MAX_DND = 0.05
const MAX_DVD = 10

/** Sensible bounds for a new variable: air gaps and the back focus ≥ 0, glass
 *  thickness ≥ 1 mm, model n_d 1.4…2.1 and V_d 15…100, melt offsets Δn_d ±0.05
 *  and ΔV_d ±10, semi-diameters and normalisation radii > 0. */
export function defaultBounds(p: ParamPath, s: System): Bounds {
  switch (p.param) {
    case 'thickness': {
      const i = typeof p.surface === 'number' ? surfaceIndexOfId(s, p.surface) : -1
      const image = s.surfaces.length - 1
      if (i <= 0 || i >= image) return { lower: null, upper: null }
      const surf = s.surfaces[i]
      if (surf.shape?.type === 'coordinate_break') return { lower: null, upper: null }
      if (i === image - 1) return { lower: 0, upper: null }
      const air = !surf.material || surf.material.type === 'air' || surf.interaction === 'reflect'
      return air ? { lower: 0, upper: null } : { lower: GLASS_MINIMUM, upper: null }
    }
    case 'semi_diameter':
      return { lower: 0, upper: null }
    case 'model_nd':
      return { lower: 1.4, upper: 2.1 }
    case 'model_vd':
      return { lower: 15, upper: 100 }
    case 'material_dn_d':
      return { lower: -MAX_DND, upper: MAX_DND }
    case 'material_dv_d':
      return { lower: -MAX_DVD, upper: MAX_DVD }
    case 'shape_param':
      return p.parameter === 'norm_radius' ? { lower: 1e-3, upper: null } : { lower: null, upper: null }
    default:
      return { lower: null, upper: null }
  }
}

/** A saved bound: the engine may write "inf" or null. */
export function boundValue(v: unknown): number | null {
  if (v == null) return null
  const n = nf(v)
  return Number.isFinite(n) ? n : null
}

export function boundsOf(v: VariableSpec): Bounds {
  return { lower: boundValue(v.lower), upper: boundValue(v.upper) }
}

/** Action name of a bounds change (macOS `DesignDocument.setBounds`). */
export function boundsActionName(before: Bounds, after: Bounds): string {
  if (after.lower != null && after.lower === after.upper) return 'Pin Variable'
  return isPinned(before) ? 'Unpin Variable' : 'Change Bounds'
}
