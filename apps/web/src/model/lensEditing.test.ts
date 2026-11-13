import { describe, expect, it } from 'vitest'
import { projectFromSystem } from './project'
import { clone, newSystem, materialText, meltOffsetText, setMaterial, type Surface } from './system'
import { isVariable, setVariable, variableBoundsKind, variableOf, variableRefusal, variablesOf } from './params'
import { kindFits, newSolveKind, removeSolve, setSolve, solveFor, solveSummary, solveTargets, solvesOf, solveValue } from './solves'
import { addCatalogReference, catalogReference, materialCompletions, offsetCompactText, offsetRejection, parseMaterial, replacedMaterial, withOffset } from './materials'
import {
  conicSag,
  departure,
  departureMap,
  efficiency,
  minimumPeriod,
  phase,
  phaseOf,
  phasePower,
  previewKind,
  previewRadius,
  profile,
  qBfs,
  qCon,
  sag,
  traceEfficiency,
  zernike,
} from './shapeSag'
import { changeShapeType, cylinderAxis, makeCylinder, setDiffractiveBase, tableTypeTitle, typeChoice } from './surfaceTypes'

const sys = () => newSystem()

describe('variables', () => {
  it('makes a value a variable with default bounds and removes it', () => {
    const p = projectFromSystem(sys())
    const t = { param: 'thickness', surface: p.system.surfaces[1].id }
    expect(setVariable(p, t, true)).toBe(true)
    // Glass keeps a minimum centre thickness.
    expect(variableOf(p, t)).toMatchObject({ lower: 1 })
    expect(variableBoundsKind(variableOf(p, t)!)).toBe('bounded')
    expect(setVariable(p, t, true)).toBe(false)
    expect(setVariable(p, t, false)).toBe(true)
    expect(variablesOf(p)).toHaveLength(0)
  })
  it('leaves curvatures unbounded and omits empty bounds', () => {
    const p = projectFromSystem(sys())
    const c = { param: 'curvature', surface: p.system.surfaces[1].id }
    setVariable(p, c, true)
    expect(variablesOf(p)[0]).toEqual({ param: c })
    expect(variableBoundsKind(variablesOf(p)[0])).toBe('free')
  })
  it('refuses a solved parameter, and matches curvature_y with curvature on a toroid', () => {
    const p = projectFromSystem(sys())
    const id = p.system.surfaces[2].id
    setSolve(p.system, { param: 'thickness', surface: id }, { type: 'marginal_ray_height', height: 0 })
    expect(variableRefusal(p.system, { param: 'thickness', surface: id })).toMatch(/Marginal ray height/)
    expect(setVariable(p, { param: 'thickness', surface: id }, true)).toBe(false)
    changeShapeType(p.system, 1, 'toroid')
    const s1 = p.system.surfaces[1].id
    setVariable(p, { param: 'shape_param', surface: s1, parameter: 'curvature_y' }, true)
    expect(isVariable(p, { param: 'curvature', surface: s1 })).toBe(true)
  })
  it('pins with equal bounds', () => {
    expect(variableBoundsKind({ param: { param: 'conic', surface: 1 }, lower: 0, upper: 0 })).toBe('pinned')
  })
})

describe('solves', () => {
  it('adds, replaces and removes the solve of a parameter', () => {
    const s = sys()
    const t = { param: 'thickness', surface: s.surfaces[2].id }
    setSolve(s, t, newSolveKind('marginal_ray_height', s, t))
    expect(solveFor(s, t)?.kind).toEqual({ type: 'marginal_ray_height', height: 0 })
    setSolve(s, t, { type: 'position', from: 1, to: 3, distance: 99 })
    expect(solvesOf(s)).toHaveLength(1)
    expect(solveSummary(s, solvesOf(s)[0].kind)).toBe('Position S1 → S3 = 99 mm')
    removeSolve(s, t)
    expect(s.solves).toBeUndefined()
  })
  it('offers the kinds that fit each target', () => {
    const s = sys()
    const targets = solveTargets(s, 1)
    const thickness = targets.find((t) => t.title === 'Thickness')!
    const radius = targets.find((t) => t.title === 'Radius')!
    expect(thickness.kinds).toContain('marginal_ray_height')
    expect(thickness.kinds).not.toContain('f_number')
    expect(radius.kinds).toEqual(expect.arrayContaining(['marginal_ray_angle', 'f_number', 'center_of_curvature', 'pickup']))
    expect(kindFits('f_number', { param: 'paraxial_focal_length', surface: 1 })).toBe(true)
    expect(kindFits('center_of_curvature', { param: 'paraxial_focal_length', surface: 1 })).toBe(false)
    expect(solveTargets(s, 0)).toEqual([])
  })
  it('starts pickups from the surface before and reads main numbers', () => {
    const s = sys()
    const t = { param: 'curvature', surface: s.surfaces[2].id }
    const k = newSolveKind('pickup', s, t)
    expect(k.source).toEqual({ param: 'curvature', surface: s.surfaces[1].id })
    expect(solveValue({ type: 'f_number', value: 4 })).toBe(4)
    expect(solveValue(k)).toBeNull()
  })
})

describe('materials', () => {
  it('parses the material column like the macOS app', () => {
    expect(parseMaterial('dummy')).toEqual({ interaction: 'dummy', material: null })
    expect(parseMaterial('=1.6')).toMatchObject({ material: { type: 'constant', n: 1.6 } })
    expect(parseMaterial('1,5')).toMatchObject({ material: { type: 'constant', n: 1.5 } })
    expect(parseMaterial('1.5168 / 64.17')).toMatchObject({ material: { type: 'model', nd: 1.5168, vd: 64.17 } })
    expect(parseMaterial('schott:n-bk7')).toMatchObject({ material: { type: 'catalog', catalog: 'SCHOTT', name: 'N-BK7' } })
    expect(parseMaterial('0.9/20')).toHaveProperty('error')
    expect(parseMaterial('a b')).toHaveProperty('error')
    expect(parseMaterial('n-sk16')).toMatchObject({ material: { name: 'N-SK16' } })
  })
  it('completes glass names and keywords', () => {
    const g = ['N-BK7', 'N-SF6', 'N-SK16', 'F2']
    expect(materialCompletions('n-s', g)).toEqual(['N-SF6', 'N-SK16'])
    expect(materialCompletions('mi', g)).toEqual(['MIRROR'])
    expect(materialCompletions('1.5', g)).toEqual([])
    expect(materialCompletions('N-BK7', g)).toEqual([])
  })
  it('keeps a retyped glass’s melt offset and drops it for another glass', () => {
    const s = sys()
    const surf: Surface = clone(s.surfaces[1])
    surf.material = withOffset(surf.material, { dn_d: 0.0012, dv_d: -0.8 })
    expect(meltOffsetText(surf.material)).toBe('+0.0012/−0.8')
    const same = parseMaterial('n-bk7', ['N-BK7', 'F2'])
    if ('error' in same) throw new Error(same.error)
    setMaterial(surf, same)
    expect(surf.material.offset).toEqual({ dn_d: 0.0012, dv_d: -0.8 })
    expect(replacedMaterial(surf.material, { type: 'catalog', catalog: null, name: 'F2' }).offset).toBeUndefined()
    expect(withOffset(surf.material, { dn_d: 0, dv_d: 0 }).offset).toBeUndefined()
    expect(materialText({ ...surf, interaction: 'dummy' })).toBe('DUMMY')
  })
  it('validates offsets and formats them', () => {
    expect(offsetRejection('dn_d', 0.05)).toBeNull()
    expect(offsetRejection('dn_d', 0.051)).toMatch(/±0.05/)
    expect(offsetRejection('dv_d', -10.5)).toMatch(/±10/)
    expect(offsetCompactText({ dn_d: 0, dv_d: 1.25 })).toBe('0/+1.25')
  })
  it('records catalogs by name, replacing one of the same name', () => {
    const a = catalogReference('mycat.agf')
    expect(a).toMatchObject({ name: 'MYCAT', format: 'agf' })
    const list = addCatalogReference(addCatalogReference([], a), catalogReference('MyCat.AGF'))
    expect(list).toHaveLength(1)
  })
})

describe('shape sag', () => {
  it('evaluates conics, aspheres and their departures', () => {
    expect(conicSag(0.02, 0, 25)).toBeCloseTo(50 - Math.sqrt(2500 - 25), 12)
    expect(conicSag(0.1, 0, 200)).toBeNull()
    const asph = { type: 'even_asphere', curvature: 0, conic: 0, coefficients: [1e-4] }
    expect(sag(asph, 0, 2)).toBeCloseTo(1e-4 * 16, 15)
    expect(departure(asph, 2, 0)).toBeCloseTo(1.6e-3, 15)
    expect(sag({ type: 'odd_asphere', curvature: 0, conic: 0, coefficients: [0.1, 0.01] }, 3, 0)).toBeCloseTo(0.3 + 0.09, 12)
  })
  it('evaluates the Q polynomials at their known values', () => {
    // Q₀ᶜᵒⁿ = 1, Q₁ᶜᵒⁿ = −(5 − 6x) at u² = x.
    expect(qCon([1], 0.3)).toBe(1)
    expect(qCon([0, 1], 0.5)).toBeCloseTo(-(5 - 6 * 0.5), 12)
    // Q₀ᵇᶠˢ = 1.
    expect(qBfs([1], 0.4)).toBeCloseTo(1, 12)
    expect(sag({ type: 'q_con', curvature: 0, conic: 0, norm_radius: 10, coefficients: [1] }, 11, 0)).toBeNull()
  })
  it('gives toroids and biconics different sections', () => {
    const toroid = { type: 'toroid', curvature_x: 0, curvature_y: 0.02, conic_y: 0, y_asphere: [] }
    expect(sag(toroid, 5, 0)).toBe(0)
    expect(sag(toroid, 0, 5)).toBeCloseTo(conicSag(0.02, 0, 25)!, 12)
    const bic = { type: 'biconic', curvature_x: 0.01, curvature_y: 0, conic_x: 0, conic_y: 0, x_asphere: [], y_asphere: [] }
    expect(sag(bic, 0, 4)).toBe(0)
    expect(sag(bic, 4, 0)).toBeCloseTo(conicSag(0.01, 0, 16)!, 12)
    expect(profile(toroid, 5, 'x')?.every(([, z]) => z === 0)).toBe(true)
  })
  it('evaluates freeform departures over the normalisation disk', () => {
    // Fringe Z₄ (defocus) = 2ρ² − 1.
    expect(zernike({ n: 2, m: 0 }, 'fringe', 0.5, 0)).toBeCloseTo(-0.5, 12)
    const z = { type: 'zernike_sag', curvature: 0, conic: 0, norm_radius: 10, ordering: 'fringe', coefficients: [0, 0, 0, 0.001] }
    expect(departure(z, 10, 0)).toBeCloseTo(0.001, 12)
    const xy = { type: 'xy_polynomial', curvature: 0, conic: 0, norm_radius: 10, coefficients: [0, 0, 0.002] }
    expect(sag(xy, 5, 0)).toBeCloseTo(0.002 * 0.25, 12)
    const map = departureMap(xy, 10, 8)
    expect(map).toHaveLength(64)
    expect(map[0]).toBeNull()
    expect(previewKind('xy_polynomial')).toBe('freeform')
    expect(previewRadius(xy, null)).toBe(10)
  })
  it('gives a DOE its phase, power and zone period', () => {
    const shape = { type: 'diffractive', base: { type: 'plane' }, phase: { norm_radius: 10, coefficients: [-100], design_order: 1, design_wavelength_um: 0.5, trace_order: 1 } }
    const p = phaseOf(shape)!
    expect(phase(p, 10)).toBeCloseTo(-100, 12)
    // φ = −m λ A₁ / (π r²) with λ in mm.
    expect(phasePower(p, 1, 0.5)).toBeCloseTo((0.5e-3 * 100) / (Math.PI * 100), 15)
    // Φ = −100 ρ², |dΦ/dr| = 2 r: the period 2π / 20 at the edge.
    expect(minimumPeriod(p, 10)!.period).toBeCloseTo((2 * Math.PI) / 20, 9)
    expect(sag(shape, 3, 4)).toBe(0)
  })
  it('estimates diffraction efficiency', () => {
    expect(efficiency({ type: 'ideal' }, 1, 1, 0.9)).toBe(1)
    expect(efficiency({ type: 'kinoform' }, 1, 1, 1)).toBeCloseTo(1, 12)
    expect(efficiency({ type: 'multilevel', levels: 8 }, 1, 1, 1)).toBeCloseTo((Math.sin(Math.PI / 8) / (Math.PI / 8)) ** 2, 12)
    const p = phaseOf({ type: 'diffractive', base: { type: 'plane' }, phase: { norm_radius: 1, coefficients: [], design_order: 1, design_wavelength_um: 0.55, trace_order: 1 } })!
    expect(traceEfficiency({ type: 'kinoform' }, p, 0.55)).toBeCloseTo(1, 12)
    expect(traceEfficiency({ type: 'kinoform' }, p, 0.45)).toBeLessThan(1)
  })
})

describe('surface types', () => {
  it('recognises cylinders and names table types', () => {
    const s = sys()
    changeShapeType(s, 1, 'biconic')
    makeCylinder(s, 1, 'x')
    const sh = s.surfaces[1].shape
    expect(cylinderAxis(sh)).toBe('x')
    expect(typeChoice(sh)).toBe('cylinder_x')
    expect(tableTypeTitle(sh)).toBe('Cylinder')
    expect(tableTypeTitle({ type: 'conic', curvature: 0.1, conic: 0 })).toBe('Sphere')
    expect(tableTypeTitle({ type: 'conic', curvature: 0.1, conic: -1 })).toBe('Conic')
  })
  it('changes a DOE’s base keeping its curvature', () => {
    const s = sys()
    changeShapeType(s, 1, 'diffractive')
    const plane = setDiffractiveBase(s.surfaces[1].shape, 'plane')
    expect(plane.base).toEqual({ type: 'plane' })
    const asph = setDiffractiveBase(s.surfaces[1].shape, 'even_asphere')
    expect(asph.base).toMatchObject({ type: 'even_asphere', curvature: 0.02 })
  })
})
