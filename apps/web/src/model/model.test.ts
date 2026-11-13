import { describe, expect, it } from 'vitest'
import { compact, fmt, parseNumber, sig } from './format'
import { readProject, writeProject, projectFromSystem, SAMPLES, isSurfaceTypeSample } from './project'
import { canDelete, clone, deleteSurface, insertSurface, newSystem, parseMaterial, setCurvature, materialText } from './system'
import { changeShapeType, xyTerm, zernikeTerm, zernikeName, setPath, getPath } from './surfaceTypes'

describe('number format', () => {
  it('uses a decimal point, no grouping, no negative zero', () => {
    expect(fmt(12345.678, 3)).toBe('12345.678')
    expect(fmt(-0.00001, 4)).toBe('−1.000e-5')
    expect(fmt(-0.00000, 4)).toBe('0.0000')
    expect(fmt(Infinity)).toBe('∞')
    expect(sig(1.5)).toBe('1.5')
    expect(compact(-50)).toBe('−50')
  })
  it('parses ∞, inf, decimal commas and the minus sign', () => {
    expect(parseNumber('∞')).toBe(Infinity)
    expect(parseNumber('-inf')).toBe(-Infinity)
    expect(parseNumber('1,5')).toBe(1.5)
    expect(parseNumber('−2.5')).toBe(-2.5)
    expect(parseNumber('1e-3')).toBe(0.001)
    expect(parseNumber('abc')).toBeNull()
    expect(parseNumber('')).toBeNull()
  })
})

describe('surfaces', () => {
  it('inserts before the stop and keeps the stop on its surface', () => {
    const s = newSystem()
    s.stop = 2
    const at = insertSurface(s, 1)
    expect(at).toBe(1)
    expect(s.stop).toBe(3)
    expect(new Set(s.surfaces.map((x) => x.id)).size).toBe(s.surfaces.length)
  })
  it('never deletes the object, the image or the last surface', () => {
    const s = newSystem()
    expect(canDelete(s, 0)).toBe(false)
    expect(canDelete(s, 3)).toBe(false)
    expect(deleteSurface(s, 1)).toBe(true)
    expect(deleteSurface(s, 1)).toBe(false)
  })
  it('makes a plane a conic when it is curved', () => {
    expect(setCurvature({ type: 'plane' }, 0.02)).toEqual({ type: 'conic', curvature: 0.02, conic: 0 })
  })
  it('parses materials', () => {
    expect(parseMaterial('')).toEqual({ interaction: 'refract', material: { type: 'air' } })
    expect(parseMaterial('mirror')).toEqual({ interaction: 'reflect', material: null })
    expect(parseMaterial('n=1.5')).toMatchObject({ material: { type: 'constant', n: 1.5 } })
    expect(parseMaterial('1.5168/64.17')).toMatchObject({ material: { type: 'model', nd: 1.5168, vd: 64.17 } })
    expect(parseMaterial('n-bk7', new Set(['N-BK7']))).toMatchObject({ material: { type: 'catalog', name: 'N-BK7' } })
    expect(parseMaterial('XYZ', new Set(['N-BK7']))).toHaveProperty('error')
  })
  it('changes the type keeping what carries over', () => {
    const s = newSystem()
    changeShapeType(s, 1, 'even_asphere')
    expect(s.surfaces[1].shape).toMatchObject({ type: 'even_asphere', curvature: 0.02, conic: 0 })
    changeShapeType(s, 1, 'diffractive')
    expect(s.surfaces[1].shape.base).toMatchObject({ type: 'even_asphere', curvature: 0.02 })
    changeShapeType(s, 1, 'toroid')
    expect(s.surfaces[1].shape).toMatchObject({ type: 'toroid', curvature_x: 0.02, curvature_y: 0.02 })
  })
  it('names terms', () => {
    expect(xyTerm(0)).toEqual({ m: 1, n: 0 })
    expect(xyTerm(2)).toEqual({ m: 2, n: 0 })
    expect(xyTerm(4)).toEqual({ m: 0, n: 2 })
    expect(zernikeName(zernikeTerm(4, 'fringe'))).toBe('defocus')
    expect(zernikeName(zernikeTerm(9, 'fringe'))).toBe('spherical')
    expect(zernikeName(zernikeTerm(11, 'standard'))).toBe('spherical')
  })
  it('sets key paths immutably', () => {
    const shape = { type: 'diffractive', base: { type: 'conic', curvature: 0, conic: 0 } }
    const next = setPath(shape, 'base.curvature', 0.1)
    expect(getPath(next, 'base.curvature')).toBe(0.1)
    expect(shape.base.curvature).toBe(0)
  })
})

describe('project file', () => {
  it('round-trips a sample and keeps unknown keys', () => {
    const p = projectFromSystem(clone(SAMPLES[0].system))
    p.extra = { future_section: { a: 1 } }
    const back = readProject(writeProject(p, '0.1.0'))
    expect(back.system).toEqual(p.system)
    expect(back.extra).toEqual({ future_section: { a: 1 } })
  })
  it('refuses other files', () => {
    expect(() => readProject('{}')).toThrow()
    expect(() => readProject('not json')).toThrow()
    expect(() => readProject(JSON.stringify({ format: 'mklight-project', schema_version: 2, system: {} }))).toThrow(/schema version/)
  })
  it('lists the samples, extended surface types apart', () => {
    expect(SAMPLES.length).toBeGreaterThan(10)
    expect(isSurfaceTypeSample(SAMPLES.find((s) => s.id === 'double_gauss')!)).toBe(false)
    expect(isSurfaceTypeSample(SAMPLES.find((s) => s.id === 'toroid')!)).toBe(true)
    expect(materialText(SAMPLES[0].system.surfaces[1])).toBe('N-BK7')
  })
})
