import { describe, expect, it } from 'vitest'
import type { JsonObject } from '@/engine/types'
import { sampleById } from '@/model/project'
import { clone } from '@/model/system'
import { solveRows, solveWarnings } from './solves'

describe('solves dry run', () => {
  const s = clone(sampleById('double_gauss')!.system)
  const report: JsonObject = {
    outcomes: [
      { index: 0, kind: 'marginal_ray_height', method: 'closed_form', previous: 40, value: 41.25, residual: 1e-13, target: { param: 'thickness', surface: s.surfaces[10].id }, evaluations: 2 },
      { index: 1, kind: 'f_number', method: 'root_finding', previous: 0.02, value: 0.025, residual: -2e-12, target: { param: 'curvature', surface: s.surfaces[9].id }, evaluations: 9 },
      { index: 2, kind: 'pickup', method: 'unchanged', previous: 3, value: 3, residual: 0, target: { param: 'thickness', surface: s.surfaces[2].id }, evaluations: 0 },
      { kind: 'new_kind', value: 'nan', residual: 'inf', target: 'bad' },
    ],
    warnings: ['solve 2 converged slowly', 7],
  }
  it('lists now → solved with titles, radii and residuals', () => {
    const rows = solveRows(report, s)
    expect(rows.map((r) => r.target)).toEqual(['T10', 'R9', 'T2', '?'])
    expect(rows.map((r) => r.kind)).toEqual(['Marginal ray height', 'F-number', 'Pickup', 'new kind'])
    expect(rows[0]).toMatchObject({ previous: 40, value: 41.25, changed: true, method: 'closed_form' })
    // Curvatures are shown as radii.
    expect(rows[1].previous).toBeCloseTo(50)
    expect(rows[1].value).toBeCloseTo(40)
    expect(rows[2].changed).toBe(false)
    expect(rows[3]).toMatchObject({ id: 3, previous: null, changed: true, residual: Infinity })
    expect(Number.isNaN(rows[3].value)).toBe(true)
  })
  it('tolerates a missing report and keeps string warnings', () => {
    expect(solveRows(null, s)).toEqual([])
    expect(solveRows(report, null)[0].target).toBe('thickness')
    expect(solveWarnings(report)).toEqual(['solve 2 converged slowly'])
  })
})
