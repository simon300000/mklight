import { describe, expect, it } from 'vitest'
import { automaticStretch, defaultCamera, fit, niceLength, pan, projector, zoom } from './camera'

describe('layout camera', () => {
  it('projects the sections with the axis to the right and the section height up', () => {
    const cam = { ...defaultCamera(), center: [0, 0, 10] as [number, number, number], ppmm: 2 }
    const size = { w: 200, h: 100 }
    const yz = projector(cam, 'yz', size, 1)
    expect(yz([0, 5, 20]).slice(0, 2)).toEqual([120, 40])
    const xz = projector(cam, 'xz', size, 1)
    expect(xz([5, 0, 10]).slice(0, 2)).toEqual([100, 40])
    // A vertical stretch draws heights taller.
    expect(projector(cam, 'yz', size, 5)([0, 5, 10])[1]).toBe(0)
  })

  it('fits points with a margin', () => {
    const pts = [0, -10, 0, 0, 10, 100]
    const c = fit(defaultCamera(), 'yz', pts, { w: 560, h: 400 }, 1)
    expect(c.center[2]).toBeCloseTo(50, 9)
    expect(c.ppmm).toBeCloseTo(5, 9)
    // Stretched: the height counts five times.
    const s = fit(defaultCamera(), 'yz', pts, { w: 560, h: 400 }, 5)
    expect(s.ppmm).toBeCloseTo(400 / (100 * 1.12), 9)
  })

  it('pans and zooms about the pointer', () => {
    const cam = { ...defaultCamera(), center: [0, 0, 0] as [number, number, number], ppmm: 1 }
    expect(pan(cam, 'yz', 10, 0, 1).center[2]).toBeCloseTo(-10, 9)
    expect(pan(cam, 'yz', 0, 10, 1).center[1]).toBeCloseTo(10, 9)
    const size = { w: 200, h: 200 }
    const before = projector(cam, 'yz', size, 1)([0, 0, 30])
    const z = zoom(cam, 'yz', 2, 30, 0, 1)
    const after = projector(z, 'yz', size, 1)([0, 0, 30])
    expect(after[0]).toBeCloseTo(before[0], 9)
  })

  it('picks a vertical stretch for slim sections only', () => {
    expect(automaticStretch(100, 400)).toBe(1)
    expect(automaticStretch(40, 400)).toBe(5)
    expect(automaticStretch(1, 400)).toBe(50)
    expect(niceLength(90)).toBe(100)
    expect(niceLength(0.03)).toBeCloseTo(0.02, 12)
  })
})
