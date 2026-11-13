import { describe, expect, it, vi } from 'vitest'

// uPlot reads matchMedia when imported.
vi.hoisted(() => {
  window.matchMedia ??= ((q: string) => ({ matches: false, media: q, onchange: null, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent: () => false })) as unknown as typeof window.matchMedia
})
import { fieldIndexIn, type Issue } from '@/state/workspace'
import { fieldOf, issueText } from './Issues'
import { seriesAt } from './analysis/curves/kit'
import { loadWidths } from './lens/columnWidths'
import { readoutAt } from './results/UPlot'

describe('validation issues', () => {
  it('reads the field index as the engine writes it (from 0)', () => {
    expect(fieldIndexIn('field 0: the chief ray is stopped (clipped) at surface 3')).toBe(0)
    expect(fieldIndexIn('Field index 2 has no rays')).toBe(2)
    expect(fieldIndexIn('surface 3 is too thin')).toBeNull()
  })
  it('numbers fields from 1 in the text and prefers the engine field', () => {
    const i: Issue = { message: 'field 0: the chief ray is stopped at surface 3', severity: 'error', surface: 3, code: 'chief_ray_stopped' }
    expect(fieldOf(i)).toBe(0)
    expect(issueText(i)).toBe('Field 1: the chief ray is stopped at surface 3')
    expect(fieldOf({ ...i, field: 4 })).toBe(4)
    expect(issueText({ message: 'surface 2: thickness is negative', severity: 'warning', surface: 2 })).toBe('surface 2: thickness is negative')
  })
})

describe('hover readouts', () => {
  it('interpolates a line series and leaves gaps out', () => {
    expect(seriesAt([0, 1, 2], [0, 10, 20], 1.5)).toBeCloseTo(15)
    expect(seriesAt([2, 1, 0], [20, 10, 0], 0.5)).toBeCloseTo(5)
    expect(seriesAt([0, 1, 2], [0, NaN, 20], 0.5)).toBeNull()
    expect(seriesAt([0, 1], [0, 1], 3)).toBeNull()
  })
  it('lists x and each shown series once', () => {
    const u = {
      data: [
        [0, 10, 20],
        [1, 0.8, 0.5],
        [1, 0.8, 0.5],
        [null, 0.7, 0.4],
      ],
      series: [{}, { label: 'Incidence', stroke: '#00f', width: 1.8 }, { stroke: '#00f', width: 0, points: { fill: '#00f' } }, { label: 'Limit', stroke: '#888', show: true }],
      axes: [{ label: 'Spatial frequency (cycles/mm)' }],
    } as unknown as Parameters<typeof readoutAt>[0]
    const r = readoutAt(u, 1)!
    expect(r.title).toBe('Spatial frequency 10')
    expect(r.rows.map((x) => [x.label, x.value])).toEqual([
      ['Incidence', '0.8'],
      ['Limit', '0.7'],
    ])
    expect(readoutAt(u, 0)!.rows).toHaveLength(1)
  })
})

describe('lens data column widths', () => {
  it('keeps only sane stored widths and defaults otherwise', () => {
    expect(loadWidths({ getItem: () => JSON.stringify({ surf: 90, type: 'x', comment: 5000 }) })).toEqual({ surf: 90 })
    expect(loadWidths({ getItem: () => 'not json' })).toEqual({})
    expect(loadWidths(null)).toEqual({})
  })
})
