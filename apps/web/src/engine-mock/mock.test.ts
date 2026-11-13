import { describe, expect, it } from 'vitest'
import samples from '@samples'
import { createMockEngine } from './index'
import type { JsonObject } from '../engine/types'

type Sample = { id: string; system: JsonObject }
const byId = (id: string) => (samples as unknown as Sample[]).find((s) => s.id === id)!.system
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x))
const eng = createMockEngine({ latencyMs: 0 })

describe.each(['singlet', 'double_gauss'])('mock engine on %s', (id) => {
  const system = byId(id)

  it('opens a stable snapshot', async () => {
    const a = await eng.openSnapshot(system)
    const b = await eng.openSnapshot(clone(system))
    expect(a.hash).toMatch(/^[0-9a-f]{64}$/)
    expect(a.hash).toBe(b.hash)
  })

  it('rejects a bad stop index', async () => {
    const bad = clone(system)
    bad.stop = 99
    await expect(eng.openSnapshot(bad)).rejects.toMatchObject({ kind: 'invalidModel' })
  })

  it('layout buffers are consistent', async () => {
    const snap = await eng.openSnapshot(system)
    const r = await eng.submit({ analysis: 'layout' }, snap).result
    const P = r.json.polyline_count as number
    const V = r.json.vertex_count as number
    const off = r.buffers.offsets as Uint32Array
    expect(off.length).toBe(P + 1)
    expect(off[P]).toBe(V)
    expect((r.buffers.vertices as Float32Array).length).toBe(V * 3)
    expect((r.buffers.kinds as Uint8Array).length).toBe(P)
  })

  it('spot_diagram points align with point_range', async () => {
    const snap = await eng.openSnapshot(system)
    const r = await eng.submit({ analysis: 'spot_diagram' }, snap).result
    const fields = r.json.fields as { point_range: [number, number] }[]
    const N = (r.buffers.points as Float64Array).length / 2
    const last = fields[fields.length - 1].point_range
    expect(last[0] + last[1]).toBe(N)
    expect((r.buffers.point_field as Uint32Array).length).toBe(N)
  })

  it('ray_fan curves are consistent', async () => {
    const snap = await eng.openSnapshot(system)
    const r = await eng.submit({ analysis: 'ray_fan' }, snap).result
    const f = (r.json.fields as { tangential: { pupil: number[]; values: unknown[] }[] }[])[0]
    expect(f.tangential[0].pupil.length).toBe(f.tangential[0].values.length)
  })

  it('curvature changes the spot RMS', async () => {
    const rms = async (sys: JsonObject) => {
      const snap = await eng.openSnapshot(sys)
      const r = await eng.submit({ analysis: 'spot_diagram', settings: { fields: { type: 'index', index: 0 } } }, snap).result
      return (r.json.fields as { rms_radius: number }[])[0].rms_radius
    }
    const changed = clone(system)
    const surf = (changed.surfaces as { shape: { curvature?: number } }[]).find((s) => typeof s.shape.curvature === 'number' && s.shape.curvature !== 0)!
    surf.shape.curvature = surf.shape.curvature! * 1.05
    expect(await rms(changed)).not.toBeCloseTo(await rms(system), 8)
  })

  it('runs every implemented analysis', async () => {
    const snap = await eng.openSnapshot(system)
    for (const a of ['validate', 'system_summary', 'first_order', 'opd_fan', 'field_curvature_distortion', 'mtf', 'psf', 'wavefront_map', 'seidel']) {
      const r = await eng.submit({ analysis: a }, snap).result
      expect(r.runInfo.backend).toBe('mock')
    }
    const tf = await eng.submit({ analysis: 'spot_diagram', settings: { through_focus: { min: null, max: null, steps: 5 } } }, snap).result
    expect(tf.buffers.point_slope).toBeDefined()
  })

  it('cancel rejects with kind cancelled', async () => {
    const slow = createMockEngine({ latencyMs: 200 })
    const snap = await slow.openSnapshot(system)
    const job = slow.submit({ analysis: 'spot_diagram' }, snap)
    job.cancel()
    await expect(job.result).rejects.toMatchObject({ kind: 'cancelled' })
  })
})

describe('settings', () => {
  it('rejects unknown analyses and keys', async () => {
    const snap = await eng.openSnapshot(byId('singlet'))
    await expect(eng.submit({ analysis: 'nope' }, snap).result).rejects.toMatchObject({ kind: 'invalidSettings' })
    await expect(eng.submit({ analysis: 'layout', settings: { bogus: 1 } }, snap).result).rejects.toMatchObject({ kind: 'invalidSettings' })
  })
})
