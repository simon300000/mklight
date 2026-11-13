import { describe, expect, it } from 'vitest'
import { CONTAINS_ZERO, boundsActionName, counterpart, curvatureBounds, defaultBounds, outOfBounds, passesThroughFlat, radiusWindow } from './bounds'
import type { System } from './system'

const ok = (min: number | null, max: number | null) => {
  const r = curvatureBounds(min, max)
  if (!r.ok) throw new Error(r.error)
  return r.bounds
}

describe('radius window ↔ curvature bounds', () => {
  it('40 … 60 is c ∈ [1/60, 1/40]', () => {
    const b = ok(40, 60)
    expect(b.lower).toBeCloseTo(1 / 60, 15)
    expect(b.upper).toBeCloseTo(1 / 40, 15)
  })

  it('−60 … −40 is c ∈ [−1/40, −1/60]', () => {
    const b = ok(-60, -40)
    expect(b.lower).toBeCloseTo(-1 / 40, 15)
    expect(b.upper).toBeCloseTo(-1 / 60, 15)
  })

  it('100 … −100 wraps through ∞: c ∈ [−0.01, 0.01]', () => {
    const b = ok(100, -100)
    expect(b.lower).toBeCloseTo(-0.01, 15)
    expect(b.upper).toBeCloseTo(0.01, 15)
    expect(passesThroughFlat(radiusWindow(b.lower, b.upper))).toBe(true)
  })

  it('an empty side is open', () => {
    expect(ok(50, null)).toEqual({ lower: 0, upper: 0.02 })
    expect(ok(null, -50)).toEqual({ lower: -0.02, upper: 0 })
    expect(ok(null, null)).toEqual({ lower: null, upper: null })
  })

  it('0 is the limit R → 0: the curvature is unbounded on that side', () => {
    expect(ok(0, 50)).toEqual({ lower: 0.02, upper: null })
    expect(ok(-50, 0)).toEqual({ lower: null, upper: -0.02 })
  })

  it('∞ … ∞ holds the surface flat', () => {
    expect(ok(Infinity, Infinity)).toEqual({ lower: 0, upper: 0 })
    expect(radiusWindow(0, 0)).toEqual({ min: Infinity, max: Infinity })
  })

  it('refuses a window containing R = 0', () => {
    expect(curvatureBounds(-50, 50)).toEqual({ ok: false, error: CONTAINS_ZERO })
    expect(curvatureBounds(60, 40).ok).toBe(false)
  })

  it('round-trips through the radius window', () => {
    for (const [a, b] of [
      [40, 60],
      [-60, -40],
      [100, -100],
      [50, null],
      [null, -50],
    ] as [number | null, number | null][]) {
      const c = ok(a, b)
      const w = radiusWindow(c.lower, c.upper)
      if (a == null) expect(w.min).toBeNull()
      else expect(w.min).toBeCloseTo(a, 9)
      if (b == null) expect(w.max).toBeNull()
      else expect(w.max).toBeCloseTo(b, 9)
    }
  })

  it('shows the other unit', () => {
    expect(counterpart({ lower: 0.02, upper: 0.025 }, 'radius')).toBe('c 0.02 … 0.025 1/mm')
    expect(counterpart({ lower: -0.01, upper: 0.01 }, 'curvature')).toBe('R 100 … ∞ … −100 mm')
    expect(counterpart({ lower: null, upper: null }, 'radius')).toBeNull()
  })
})

describe('variable bounds', () => {
  const system = {
    surfaces: [
      { id: 0, material: { type: 'air' }, interaction: 'refract', shape: { type: 'sphere' } },
      { id: 1, material: { type: 'catalog' }, interaction: 'refract', shape: { type: 'sphere' } },
      { id: 2, material: { type: 'air' }, interaction: 'refract', shape: { type: 'sphere' } },
      { id: 3, material: { type: 'air' }, interaction: 'refract', shape: { type: 'sphere' } },
      { id: 4, material: { type: 'air' }, interaction: 'refract', shape: { type: 'plane' } },
    ],
  } as unknown as System

  it('gives new variables sensible bounds', () => {
    expect(defaultBounds({ param: 'thickness', surface: 1 }, system)).toEqual({ lower: 1, upper: null })
    expect(defaultBounds({ param: 'thickness', surface: 2 }, system)).toEqual({ lower: 0, upper: null })
    expect(defaultBounds({ param: 'thickness', surface: 3 }, system)).toEqual({ lower: 0, upper: null })
    expect(defaultBounds({ param: 'model_nd', surface: 1 }, system)).toEqual({ lower: 1.4, upper: 2.1 })
    expect(defaultBounds({ param: 'model_vd', surface: 1 }, system)).toEqual({ lower: 15, upper: 100 })
    expect(defaultBounds({ param: 'material_dn_d', surface: 1 }, system)).toEqual({ lower: -0.05, upper: 0.05 })
    expect(defaultBounds({ param: 'curvature', surface: 1 }, system)).toEqual({ lower: null, upper: null })
  })

  it('flags out-of-bounds values and names the action', () => {
    expect(outOfBounds({ lower: 0, upper: null }, -0.1)).toBe(true)
    expect(outOfBounds({ lower: 0, upper: 1 }, 0.5)).toBe(false)
    expect(boundsActionName({ lower: 0, upper: null }, { lower: 2, upper: 2 })).toBe('Pin Variable')
    expect(boundsActionName({ lower: 2, upper: 2 }, { lower: 0, upper: null })).toBe('Unpin Variable')
    expect(boundsActionName({ lower: 0, upper: null }, { lower: 1, upper: null })).toBe('Change Bounds')
  })
})
