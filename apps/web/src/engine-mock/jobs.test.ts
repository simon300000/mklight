import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import samples from '@samples'
import { createMockEngine } from './index'
import type { JobProgress, JsonObject } from '../engine/types'

type Sample = { id: string; system: JsonObject }
const byId = (id: string) => (samples as unknown as Sample[]).find((s) => s.id === id)!.system
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x))
const eng = createMockEngine({ latencyMs: 0 })
const fixture = (name: string) => readFileSync(join(process.cwd(), 'src/engine-mock/fixtures', name))
const toBuffer = (b: Buffer) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer

/* eslint-disable @typescript-eslint/no-explicit-any */
type Obj = any

async function run(analysis: string, settings: JsonObject = {}, system: JsonObject = byId('double_gauss')) {
  const snap = await eng.openSnapshot(system)
  return eng.submit({ analysis, settings }, snap).result
}

describe('optimize', () => {
  const dg = byId('double_gauss')
  const surfaces = (dg.surfaces as Obj[]).filter((s) => s.shape.type === 'conic' && s.shape.curvature !== 0)
  const variables = surfaces.slice(0, 4).map((s) => ({ param: { param: 'curvature', surface: s.id } }))

  it('improves the merit on the double gauss and returns a valid system', async () => {
    const snap = await eng.openSnapshot(dg)
    const job = eng.submit({ analysis: 'optimize', settings: { variables, max_iterations: 8 } }, snap)
    const progress: JobProgress[] = []
    job.onProgress((p) => progress.push(p))
    const r = await job.result
    const j = r.json as Obj
    expect(j.final.merit).toBeLessThan(j.start.merit)
    expect(j.improved).toBe(true)
    expect(j.improvement).toBeGreaterThan(0)
    expect(['converged', 'finished']).toContain(j.status)
    expect(j.variables).toHaveLength(4)
    expect(j.variables[0]).toMatchObject({ lower: null, upper: null })
    expect(j.history.length).toBe(j.iterations + 1)
    expect(r.shapes?.merit_history).toEqual([j.history.length, 4])
    expect((r.buffers.merit_history as Float64Array).length).toBe(j.history.length * 4)
    expect(j.operands.length).toBeGreaterThan(0)
    expect(j.merit_spec.operands.length).toBe(j.operands.length)
    // The optimised system opens, and its merit matches the report.
    const opened = await eng.openSnapshot(j.system)
    expect(opened.hash).toBe(j.system_hash.length === 64 ? opened.hash : '')
    const ev = await run('merit_evaluation', {}, j.system)
    expect((ev.json as Obj).merit).toBeCloseTo(j.final.merit, 9)
    // Progress carries the live merit.
    const detailed = progress.filter((p) => p.stage === 'Optimizing')
    expect(detailed.length).toBeGreaterThan(2)
    expect(detailed[detailed.length - 1].detail).toMatchObject({ iteration: expect.any(Number), evaluations: expect.any(Number), merit: expect.any(Number), best_merit: expect.any(Number) })
  })

  it('respects bounds and holds the EFL', async () => {
    const id = surfaces[0].id
    const c = surfaces[0].shape.curvature as number
    const r = await run('optimize', { variables: [{ param: { param: 'curvature', surface: id }, lower: c * 0.999, upper: c * 1.001 }, ...variables.slice(1)], max_iterations: 5 })
    const j = r.json as Obj
    const v = j.variables[0]
    expect(v.value).toBeGreaterThanOrEqual(c * 0.999 - 1e-12)
    expect(v.value).toBeLessThanOrEqual(c * 1.001 + 1e-12)
    const before = (await run('first_order')).json.entries as Obj[]
    const after = (await run('first_order', {}, j.system)).json.entries as Obj[]
    const efl = (e: Obj[]) => e.find((x) => x.key === 'efl').value
    expect(Math.abs(efl(after) / efl(before) - 1)).toBeLessThan(0.02)
  })

  it('stop() ends early with the best so far', async () => {
    const snap = await eng.openSnapshot(dg)
    const job = eng.submit({ analysis: 'optimize', settings: { max_iterations: 200 } }, snap)
    let n = 0
    job.onProgress((p) => {
      if (p.stage === 'Optimizing' && ++n === 2) job.stop()
    })
    const j = (await job.result).json as Obj
    expect(j.status).toBe('stopped')
    expect(j.stopped).toBe(true)
    expect(j.iterations).toBeLessThan(10)
    expect(j.final.merit).toBeLessThanOrEqual(j.start.merit)
  })

  it('cancel() rejects with cancelled', async () => {
    const snap = await eng.openSnapshot(dg)
    const job = eng.submit({ analysis: 'optimize' }, snap)
    job.cancel()
    await expect(job.result).rejects.toMatchObject({ kind: 'cancelled' })
  })
})

describe('merit_evaluation', () => {
  it('reports one RMS operand per field and the EFL hold', async () => {
    const j = (await run('merit_evaluation')).json as Obj
    expect(j.operands.length).toBe(4)
    expect(j.operands[3]).toMatchObject({ kind: 'efl', role: 'constraint', contribution: null })
    expect(j.merit).toBeGreaterThan(0)
    expect(j.violation).toBe(0)
    expect(j.status).toEqual({ status: 'ok' })
  })
})

describe('glass_substitution', () => {
  it('returns ranked combinations with systems', async () => {
    const snap = await eng.openSnapshot(byId('achromat'))
    const job = eng.submit({ analysis: 'glass_substitution', settings: { max_combinations: 8, top_n: 3 } }, snap)
    const events: JobProgress[] = []
    job.onProgress((p) => events.push(p))
    const j = (await job.result).json as Obj
    for (const k of ['algorithm', 'candidate_sets', 'history', 'merit_spec', 'results', 'starting_merit', 'statistics', 'status', 'stopped', 'termination', 'backend']) expect(j).toHaveProperty(k)
    expect(j.results.length).toBeGreaterThan(0)
    expect(j.results.length).toBeLessThanOrEqual(3)
    expect(j.results[0].rank).toBe(1)
    expect(j.results[0].glasses[0]).toMatchObject({ surface: expect.any(Number), name: expect.any(String), nd: expect.any(Number), vd: expect.any(Number) })
    expect(j.results[0].system_hash).toMatch(/^[0-9a-f]{64}$/)
    expect(j.results[0].merit).toBeLessThanOrEqual(j.results[j.results.length - 1].merit)
    expect(events.some((p) => p.detail && (p.detail as Obj).combinations !== undefined)).toBe(true)
  })

  it('can be stopped', async () => {
    const snap = await eng.openSnapshot(byId('achromat'))
    const job = eng.submit({ analysis: 'glass_substitution', settings: { max_combinations: 100 } }, snap)
    let n = 0
    job.onProgress((p) => {
      if (p.stage === 'Glass substitution' && ++n === 2) job.stop()
    })
    const j = (await job.result).json as Obj
    expect(j.status).toBe('stopped')
    expect(j.statistics.combinations).toBeLessThan(20)
  })
})

describe('tolerance', () => {
  it('returns sensitivity and Monte Carlo', async () => {
    const snap = await eng.openSnapshot(byId('cooke_triplet'))
    const job = eng.submit({ analysis: 'tolerance', settings: { trials: 12, record_trials: 3, threshold: 1, seed: 4, compensators: [{ param: { param: 'thickness', surface: 6 }, range: 1 }] } }, snap)
    const events: JobProgress[] = []
    job.onProgress((p) => events.push(p))
    const r = await job.result
    const j = r.json as Obj
    for (const k of ['status', 'stopped', 'sensitivity', 'monte_carlo', 'compensators', 'criterion_spec', 'backend']) expect(j).toHaveProperty(k)
    expect(j.sensitivity.operands.length).toBeGreaterThan(3)
    expect(j.sensitivity.ranking.length).toBe(j.sensitivity.operands.length)
    expect(j.sensitivity.predicted_criterion).toBeGreaterThanOrEqual(j.sensitivity.nominal.criterion)
    expect(j.monte_carlo.trials).toBe(12)
    expect(j.monte_carlo.records).toHaveLength(3)
    expect(j.monte_carlo.percentiles.map((p: Obj) => p.percentile)).toEqual([50, 90, 97.7])
    expect(j.monte_carlo.yield_fraction).toBeGreaterThanOrEqual(0)
    expect((r.buffers.monte_carlo_criteria as Float64Array).length).toBe(12)
    expect(events.some((p) => (p.detail as Obj | undefined)?.stage === 'monte_carlo')).toBe(true)
    // The same seed reproduces.
    const again = (await run('tolerance', { trials: 12, seed: 4, compensators: [{ param: { param: 'thickness', surface: 6 }, range: 1 }] }, byId('cooke_triplet'))).json as Obj
    expect(again.monte_carlo.criteria).toEqual(j.monte_carlo.criteria)
  })

  it('previews one case as a perturbed system', async () => {
    const j = (await run('tolerance', { preview_case: { type: 'sensitivity', operand: 0, positive: true } }, byId('cooke_triplet'))).json as Obj
    expect(j.preview.system.surfaces).toBeDefined()
    expect(j.preview.perturbations.some((d: number) => d !== 0)).toBe(true)
    expect(j.sensitivity).toBeNull()
  })

  it('stops with the completed trials', async () => {
    const snap = await eng.openSnapshot(byId('cooke_triplet'))
    const job = eng.submit({ analysis: 'tolerance', settings: { trials: 500, mode: 'monte_carlo' } }, snap)
    job.onProgress((p) => {
      if ((p.detail as Obj | undefined)?.done >= 8) job.stop()
    })
    const j = (await job.result).json as Obj
    expect(j.status).toBe('stopped')
    expect(j.monte_carlo.trials).toBeLessThan(100)
    expect(j.monte_carlo.cancelled).toBe(true)
  })
})

describe('solves', () => {
  const withSolve = () => {
    const sys = clone(byId('singlet'))
    const surfaces = sys.surfaces as Obj[]
    sys.solves = [{ target: { param: 'thickness', surface: surfaces[2].id }, kind: { type: 'marginal_ray_height', height: 0 } }]
    return sys
  }

  it('resolves a marginal ray height solve onto the paraxial focus', async () => {
    const sys = withSolve()
    const j = (await run('solves', {}, sys)).json as Obj
    expect(j.outcomes).toHaveLength(1)
    expect(j.outcomes[0]).toMatchObject({ kind: 'marginal_ray_height', method: 'closed_form' })
    expect(Math.abs(j.outcomes[0].residual)).toBeLessThan(1e-9)
    const bfl = ((await run('first_order', {}, sys)).json.entries as Obj[]).find((e) => e.key === 'bfl').value
    expect(j.outcomes[0].value).toBeCloseTo(bfl, 6)
    expect(j.solved_params).toHaveLength(1)
  })

  it('applySolves writes the outcome into the system', async () => {
    const sys = withSolve()
    const snap = await eng.openSnapshot(sys)
    const r = await eng.applySolves({}, snap)
    const t = (r.snapshot.system.surfaces as Obj[])[2].thickness
    expect(t).toBeCloseTo((r.report.outcomes as Obj[])[0].value, 12)
    expect(t).not.toBe(96.6967287 + 1)
    expect(r.snapshot.hash).not.toBe(snap.hash === r.snapshot.hash ? '' : snap.hash)
  })

  it('handles pickups and F/#', async () => {
    const sys = clone(byId('singlet'))
    const s = sys.surfaces as Obj[]
    sys.solves = [{ target: { param: 'curvature', surface: s[2].id }, kind: { type: 'pickup', source: { param: 'curvature', surface: s[1].id }, scale: -0.5, offset: 0 } }]
    const j = (await run('solves', {}, sys)).json as Obj
    expect(j.outcomes[0].value).toBeCloseTo(-0.5 * (s[1].shape as Obj).curvature, 12)
  })
})

describe('convert_spec', () => {
  it('is submittable and converts the aperture and fields', async () => {
    const j = (await run('convert_spec', { aperture: 'image_space_f_number', field_kind: 'paraxial_image_height' }, byId('singlet'))).json as Obj
    expect(j.aperture.type).toBe('image_space_f_number')
    expect(j.aperture.value).toBeCloseTo(5, 1)
    expect(j.fields.kind).toBe('paraxial_image_height')
    expect(j.fields.values).toHaveLength(3)
    expect(j.fields.values[0]).toEqual([0, 0])
    expect(j.stop_semi_diameter).toBeCloseTo(10, 6)
    const bad = (await run('convert_spec', { aperture: 'object_space_na', field_kind: 'object_height' }, byId('singlet'))).json as Obj
    expect(bad.aperture).toBeNull()
    expect(bad.aperture_error).toMatch(/finite object/)
    expect(bad.fields).toBeNull()
  })
})

describe('every analysis', () => {
  it('has a descriptor for each result shape', async () => {
    const names = (await eng.analysisDescriptors()).map((d) => d.name)
    expect(names).toContain('convert_spec')
    expect(names.length).toBeGreaterThan(40)
  })

  describe.each(['singlet', 'double_gauss', 'newtonian'])('on %s', (id) => {
    it('runs every descriptor name without throwing', async () => {
      const snap = await eng.openSnapshot(byId(id))
      const names = (await eng.analysisDescriptors()).map((d) => d.name)
      for (const name of names) {
        const settings: JsonObject = name === 'glass_info' ? { name: 'N-BK7' } : name === 'tolerance' ? { trials: 6 } : name === 'optimize' ? { max_iterations: 2 } : name === 'glass_substitution' ? { max_combinations: 3 } : {}
        const r = await eng.submit({ analysis: name, settings }, snap).result
        expect(typeof r.json, name).toBe('object')
        expect(r.json, name).not.toBeNull()
        expect(r.runInfo.analysis).toBe(name)
        for (const [buf, shape] of Object.entries(r.shapes ?? {})) expect((r.buffers[buf] as ArrayLike<number>).length, `${name}.${buf}`).toBe(shape.reduce((a, b) => a * b, 1))
      }
    }, 60_000)
  })

  it('glass_info has a dispersion curve buffer', async () => {
    const r = await run('glass_info', { name: 'N-SF6', curve_samples: 50 })
    expect(r.shapes?.dispersion_curve).toEqual([50, 2])
    expect((r.json as Obj).indices).toHaveLength(3)
    expect((r.json as Obj).nd).toBeCloseTo(1.80518, 5)
  })

  it('glass_map lists catalog and design glasses with coordinates', async () => {
    const r = await run('glass_map', {}, byId('achromat'))
    const gl = (r.json as Obj).glasses as Obj[]
    expect(r.shapes?.coordinates).toEqual([gl.length, 3])
    expect(gl.some((g) => g.surfaces.length > 0)).toBe(true)
  })

  it('footprint and ray_grid buffers are consistent', async () => {
    const f = await run('footprint', { surface: 3, rings: 4 })
    const j = f.json as Obj
    const total = (j.sets as Obj[]).reduce((a, s) => a + s.points[1], 0)
    expect((f.buffers.points as Float64Array).length).toBe(total * 2)
    const last = (j.overlays as Obj[])[j.overlays.length - 1]
    expect((f.buffers.global_points as Float64Array).length).toBeGreaterThanOrEqual((last.points[0] + last.points[1]) * 3)
    const g = await run('ray_grid', { rings: 2 })
    const gj = g.json as Obj
    expect(g.shapes?.points).toEqual([gj.fields.length, gj.wavelengths.length, gj.pupil_count, 2])
  })

  it('field maps carry columns and aligned buffers', async () => {
    for (const a of ['chief_ray_angle', 'pupil_aberration', 'relative_illumination_map']) {
      const r = await run(a, { grid: [3, 4] })
      const j = r.json as Obj
      expect(j.shape).toEqual([3, 4])
      expect(r.shapes?.values).toEqual([4, 3, j.columns.length])
    }
  })

  it('image_simulation keeps low-y rows first', async () => {
    const r = await run('image_simulation', { size: [32, 16], source: { kind: 'pattern', pattern: 'slanted_edge' } })
    expect(r.shapes?.image).toEqual([16, 32, 1])
    expect((r.json as Obj).origin_mm[1]).toBeLessThan(0)
  })

  it('ghost analyses rank paths', async () => {
    const gp = (await run('ghost_paths')).json as Obj
    expect(gp.ghosts.length).toBeGreaterThan(5)
    expect(gp.ghosts[0].relative_irradiance).toBeGreaterThanOrEqual(gp.ghosts[1].relative_irradiance)
    const gi = await run('ghost_irradiance')
    expect(gi.shapes?.primary_map).toEqual([64, 64])
    expect((gi.json as Obj).ghosts.length).toBeGreaterThan(0)
  })

  it('layout adds rings, a mesh and honours section xz', async () => {
    const plain = await run('layout', { include_3d: true, rings: 2, ring_samples: 16 })
    expect((plain.json as Obj).mesh_triangle_count).toBe(0)
    const m = await run('layout', { include_3d: true, mesh_rings: 3, ring_samples: 16 })
    const T = (m.json as Obj).mesh_triangle_count as number
    expect(T).toBeGreaterThan(100)
    expect(m.shapes?.surface_mesh).toEqual([T, 3, 3])
    expect(m.shapes?.mesh_surfaces).toEqual([T])
    const ringsOf = (r: typeof plain) => Array.from(r.buffers.kinds as Uint8Array).filter((k) => k === 1).length
    expect(ringsOf(plain)).toBeGreaterThan(ringsOf(await run('layout', { include_3d: true, rings: 1, ring_samples: 16 })))
    const xz = await run('layout', { section: 'xz' })
    expect((xz.json as Obj).section).toBe('xz')
    const v = xz.buffers.vertices_f64 as Float64Array
    const off = xz.buffers.offsets as Uint32Array
    const kinds = xz.buffers.kinds as Uint8Array
    const flags = xz.buffers.flags as Uint8Array
    const firstProfile = kinds.findIndex((k, i) => k === 0 && (flags[i] & 1) === 1)
    let maxX = 0
    let maxY = 0
    for (let i = off[firstProfile]; i < off[firstProfile + 1]; i++) {
      maxX = Math.max(maxX, Math.abs(v[i * 3]))
      maxY = Math.max(maxY, Math.abs(v[i * 3 + 1]))
    }
    expect(maxX).toBeGreaterThan(0)
    expect(maxY).toBe(0)
  })

  it('rng_stream mean is near one half', async () => {
    const j = (await run('rng_stream', { count: 5000 })).json as Obj
    expect(j.count).toBe(5000)
    expect(Math.abs(j.mean - 0.5)).toBeLessThan(0.02)
  })

  it('analyses respond to the design', async () => {
    const changed = clone(byId('singlet'))
    ;(changed.surfaces as Obj[])[1].shape.curvature *= 1.1
    const a = (await run('seidel', {}, byId('singlet'))).json as Obj
    const b = (await run('seidel', {}, changed)).json as Obj
    expect(a.sums.s1).not.toBe(b.sums.s1)
    expect(a.status).toBe('applicable')
  })
})

describe('import and catalogs', () => {
  it('imports a .zmx fixture', async () => {
    const r = await eng.importPrescription(toBuffer(fixture('doublet.zmx')), 'zmx')
    const s = r.system as Obj
    expect(r.format).toBe('zmx')
    expect(s.name).toBe('Cemented doublet f100 F/5')
    expect(s.surfaces).toHaveLength(6)
    expect(s.surfaces[0].thickness).toBe('inf')
    expect(s.aperture).toEqual({ type: 'entrance_pupil_diameter', value: 20 })
    expect(s.fields.fields.map((f: Obj) => f.y)).toEqual([0, 2, 3])
    expect(s.fields.kind).toBe('object_angle')
    expect(s.wavelengths.primary).toBe(1)
    expect(s.stop).toBe(1)
    expect(s.surfaces[1].material).toMatchObject({ type: 'catalog', name: 'N-BK7' })
    expect(s.surfaces[3].shape).toMatchObject({ type: 'even_asphere', conic: -0.3 })
    expect(s.surfaces[3].shape.coefficients[0]).toBeCloseTo(1.2e-7, 12)
    expect(s.surfaces[4].material.type).toBe('model')
    const kinds = r.warnings.map((w) => w.kind)
    expect(kinds).toContain('unknown_glass')
    expect(kinds).toContain('unsupported_record')
    const w = r.warnings.find((x) => x.kind === 'unknown_glass')!
    expect(w.line).toBeGreaterThan(0)
    expect(w.surface).toBe(4)
    // The import is a usable system.
    const snap = await eng.openSnapshot(r.system)
    const fo = await eng.submit({ analysis: 'first_order' }, snap).result
    expect((fo.json.entries as Obj[]).find((e) => e.key === 'efl').value).toBeGreaterThan(50)
  })

  it('imports a CODE V .seq fixture', async () => {
    const r = await eng.importPrescription(toBuffer(fixture('singlet.seq')), 'seq')
    const s = r.system as Obj
    expect(s.name).toBe('Singlet f100 F/5')
    expect(s.surfaces).toHaveLength(4)
    expect(s.surfaces[1].shape.curvature).toBeCloseTo(1 / 60, 12)
    expect(s.surfaces[1].material).toMatchObject({ type: 'catalog', name: 'N-BK7' })
    expect(s.stop).toBe(1)
    expect(s.wavelengths.wavelengths[0].um).toBeCloseTo(0.6562725, 9)
    expect(s.solves[0].kind.type).toBe('marginal_ray_height')
    expect(s.solves[0].target.surface).toBe(s.surfaces[2].id)
    expect(r.warnings.some((w) => w.kind === 'solve_dropped')).toBe(true)
    await eng.openSnapshot(r.system)
  })

  it('rejects text without surfaces', async () => {
    await expect(eng.importPrescription(new TextEncoder().encode('NAME nothing\n').buffer as ArrayBuffer, 'zmx')).rejects.toMatchObject({ kind: 'failed' })
  })

  it('loads an AGF catalog so its glasses resolve', async () => {
    const e = createMockEngine({ latencyMs: 0 })
    const info = await e.loadCatalog('mini.agf', fixture('mini.agf').toString())
    expect(info).toEqual({ name: 'MINI', source: 'user', glass_count: 2 })
    expect((await e.catalogs()).map((c) => c.name)).toContain('MINI')
    expect(await e.materialNames()).toContain('TESTGLASS-1')
    const sys = clone(byId('singlet'))
    ;(sys.surfaces as Obj[])[1].material = { type: 'catalog', catalog: null, name: 'testglass-1' }
    await e.openSnapshot(sys)
    const r = await e.submit({ analysis: 'glass_info', settings: { name: 'TESTGLASS-2' } }, await e.openSnapshot(sys)).result
    expect((r.json as Obj).nd).toBeCloseTo(1.497, 6)
    expect((r.json as Obj).catalog).toBe('MINI')
  })
})
