import { describe, expect, it } from 'vitest'
import type { AnalysisResult, JsonObject, TypedArray } from '@/engine/types'
import { autoDisplay, autoSentinel, autoUsed, g3, isAuto, rangeText } from './autoSettings'
import { attempts, badgeHelp, durationText, groupWarnings, remedyFor, routeLine, runFacts, splitWarning, tierText, warningLine } from './runInfo'
import { controlFor, pairLabels, rowKeys, variantsOf } from './settingControl'
import { subscriptPieces } from './Subscripted'
import { settingsSummary } from '../SettingsForm'

const res = (json: JsonObject, buffers: Record<string, TypedArray> = {}, extra: Partial<AnalysisResult['runInfo']> = {}, envelope?: JsonObject): AnalysisResult => ({
  json,
  buffers,
  runInfo: { snapshot_hash: 'abcdef0123456789', analysis: 'x', settings: {}, backend: 'cpu_reference', warnings: [], ...extra },
  envelope,
})

describe('warning grouping', () => {
  it('splits scopes and numbers fields and wavelengths from 1', () => {
    expect(splitWarning('field 2, wavelength 0: psf truncated')).toEqual(['Field 3 · λ1', 'psf truncated'])
    expect(splitWarning('surface 4: clipped')).toEqual(['Surface 4', 'clipped'])
    expect(splitWarning('field (-1.0, 1.0): no rays')).toEqual(['field (−1.0, 1.0)', 'no rays'])
    expect(splitWarning('note: something')).toEqual([null, 'note: something'])
    expect(splitWarning('plain message')).toEqual([null, 'plain message'])
  })

  it('groups by message with numbers masked, first-seen order, capitalised', () => {
    const g = groupWarnings([
      'field 0: pupil sampling coarse (0.58 waves per sample); increase the pupil sampling',
      'field 2: pupil sampling coarse (0.72 waves per sample); increase the pupil sampling',
      'field 2: pupil sampling coarse (0.73 waves per sample); increase the pupil sampling',
      'image grid too small; enlarge the grid',
    ])
    expect(g).toHaveLength(2)
    expect(g[0].scopes).toEqual(['Field 1', 'Field 3'])
    expect(g[0].message).toMatch(/^Pupil sampling coarse \(0.58/)
    expect(g[0].remedy).toBe('pupilSampling')
    expect(g[1]).toEqual({ message: 'Image grid too small; enlarge the grid', scopes: [], remedy: 'imageGrid' })
    expect(warningLine(g[0])).toMatch(/^Field 1, Field 3: Pupil/)
  })
})

describe('remedies', () => {
  it('sends an explicit pupil grid back to automatic', () => {
    const r = remedyFor('pupilSampling', { pupil: { grid: 32, rays: 65 } })!
    expect(r.title).toBe('Sample the Pupil Automatically (grid 32 → auto)')
    expect(r).toMatchObject({ key: 'pupil', value: { grid: 0, rays: 65 } })
    expect(remedyFor('pupilSampling', { pupil: { grid: 0 } })).toBeNull()
  })

  it('image grid and the image simulation PSF window', () => {
    expect(remedyFor('imageGrid', { grid_size: 128 })).toMatchObject({ title: 'Size the Image Grid Automatically (128 → auto)', key: 'grid_size', value: 0 })
    expect(remedyFor('imageGrid', { grid_size: 0 })).toBeNull()
    const o = remedyFor('imageGrid', { optics: { psf_size: 63, pupil: { grid: 64 } } })!
    expect(o).toMatchObject({ title: 'Size the PSF Window Automatically (63 → auto)', key: 'optics', value: { psf_size: 0, pupil: { grid: 64 } } })
    const p = remedyFor('pupilSampling', { optics: { psf_size: 63, pupil: { grid: 64 } } })!
    expect(p).toMatchObject({ key: 'optics', value: { psf_size: 63, pupil: { grid: 0 } } })
    expect(remedyFor(null, { grid_size: 3 })).toBeNull()
  })
})

describe('route, tier and attempts', () => {
  it('describes cost-model decisions with both predicted times', () => {
    const d = { decision: 'qualified_metal', rule: 'cost model', predicted_ms: { metal: 312, cpu: 557 } }
    expect(routeLine('auto', d)).toBe('GPU by predicted time (GPU predicted 312 ms, CPU 557 ms)')
    expect(routeLine('default:auto', { ...d, decision: 'cpu_reference', predicted_ms: { metal: 400, cpu: 450 } })).toBe(
      "CPU: the GPU's predicted gain is too small (GPU predicted 400 ms, CPU 450 ms)",
    )
    expect(routeLine('auto', { ...d, decision: 'cpu_reference', predicted_ms: { metal: 900, cpu: 450 } })).toMatch(/^CPU by predicted time/)
    expect(routeLine('auto', { ...d, repeated: true })).toMatch(/; kept from the first identical request$/)
  })

  it('names the other rules and leaves explicit or unknown ones out', () => {
    expect(routeLine('auto', { decision: 'cpu_reference', rule: 'accuracy exclusion' })).toBe('CPU: this analysis is not accurate enough on the GPU')
    expect(routeLine('auto', { decision: 'cpu_reference', rule: 'known failure' })).toBe('CPU: this request already failed on the GPU')
    expect(routeLine('auto', { decision: 'qualified_metal', reason: 'no cost calibration for this device' })).toBe('GPU: no timing data for this device, so no prediction')
    expect(routeLine('gpu:metal', { decision: 'explicit_backend' })).toBeNull()
    expect(routeLine('auto', { decision: 'something_new' })).toBeNull()
    expect(routeLine('auto', null)).toBeNull()
  })

  it('tier and attempts', () => {
    expect(tierText('design')).toBeNull()
    expect(tierText('preview')).toMatch(/^Preview/)
    expect(
      attempts([
        { backend: 'gpu:metal', outcome: 'failed', reason: 'device lost', elapsed_ms: 12 },
        { backend: 'gpu:metal', outcome: 'skipped', reason: 'earlier failure', elapsed_ms: 0 },
        { backend: 'gpu:metal', outcome: 'reference_completion', reason: 'partial', elapsed_ms: 1800 },
      ]).map((a) => a.title),
    ).toEqual([
      'GPU failed after 12 ms; completed on the CPU',
      'GPU not tried: the same input already failed on it',
      'GPU path completed the result on the CPU (1.8 s)',
    ])
    expect(durationText(0.2)).toBe('<1 ms')
  })

  it('run facts leave empty rows out and the badge help lists warnings', () => {
    const r = res({}, {}, { duration_ms: 4.25, warnings: ['field 1: a', 'field 0: a', 'b'], failed_attempts: [{ backend: 'gpu:metal', outcome: 'failed', reason: 'x', elapsed_ms: 3 }] }, {
      backend: { requested: 'auto', used: 'cpu_reference', precision: 'f64_reference', fallback_reason: 'device lost', cost_decision: { decision: 'qualified_metal', rule: 'cost model', predicted_ms: { metal: 1, cpu: 2 } } },
    })
    const f = runFacts(r)
    expect(f.rows.map(([k]) => k)).toEqual(['Computed on', 'Requested', 'Route', 'Ran on the CPU', 'Precision', 'Snapshot', 'Time'])
    expect(f.rows[0][1]).toBe('CPU reference · f64')
    expect(f.groups.map((g) => g.scopes)).toEqual([['Field 2', 'Field 1'], []])
    expect(f.attempts).toHaveLength(1)
    expect(badgeHelp(f)).toMatch(/^Ran on the CPU: device lost\nField 2, Field 1: A\nB\nClick for details\.$/)
  })
})

describe('automatic settings', () => {
  it('knows the sentinels', () => {
    expect(autoSentinel('pupil.grid', 'mtf')).toBe('zero')
    expect(autoSentinel('grid_size', 'psf')).toBe('zero')
    expect(autoSentinel('grid_size', 'spot_diagram')).toBeNull()
    expect(autoSentinel('frequencies', 'mtf_vs_field')).toBe('empty')
    expect(autoSentinel('through_focus.min', 'spot_diagram')).toBe('null')
    expect(isAuto(0, 'zero')).toBe(true)
    expect(isAuto(32, 'zero')).toBe(false)
    expect(isAuto([], 'empty')).toBe(true)
    expect(isAuto(null, 'null')).toBe(true)
  })

  it('reads what the run used', () => {
    expect(rangeText([282, 282])).toBe('282')
    expect(rangeText([371, 254.6])).toBe('254–371')
    expect(g3(0.4371)).toBe('0.437')
    const psf = res({ psfs: [{ psf: { size: 282, spacing_mm: 0.0004 } }, { psf: { size: 371, spacing_mm: 0.0005 } }] })
    expect(autoDisplay(autoUsed('grid_size', 'psf', psf))).toBe('Auto (282–371)')
    expect(autoUsed('spacing_mm', 'psf', psf)).toBe('0.4–0.5 µm')
    const tf = res({ focus_mm: 'focus_mm', frequencies: 'frequencies' }, { focus_mm: new Float64Array([-0.437, 0, 0.437]), frequencies: new Float64Array([2, 5, 10]) })
    expect(autoDisplay(autoUsed('focus_range_mm', 'through_focus_mtf', tf))).toBe('Auto (±0.437)')
    expect(autoDisplay(autoUsed('frequencies', 'through_focus_mtf', tf))).toBe('Auto (2, 5, 10)')
    const spot = res({ fields: [{ through_focus: [{ defocus: -0.193 }, { defocus: 0 }, { defocus: 0.193 }] }] })
    expect(autoUsed('through_focus', 'spot_diagram', spot)).toBe('±0.193')
    expect(autoUsed('pupil.grid', 'wavefront_map', res({ maps: [{ grid: 128 }, { grid: 256 }] }))).toBe('128–256')
    expect(autoUsed('pupil.grid', 'mtf', res({}))).toBeNull()
    expect(autoDisplay(null)).toBe('Auto')
    expect(autoUsed('optics.psf_size', 'image_simulation', res({ psf_size: 505 }))).toBe('505')
  })
})

describe('setting controls from the schema', () => {
  const a = { analysis: 'x' }
  it('selections, indices, spectrum', () => {
    expect(controlFor('fields', { type: 'object', default: { type: 'all' } }, a).kind).toBe('fieldSelection')
    expect(controlFor('wavelength', { type: ['integer', 'null'], default: null }, a)).toEqual({ kind: 'index', domain: 'wavelengths' })
    expect(controlFor('spectrum', { type: ['object', 'null'] }, a).kind).toBe('spectrum')
    expect(controlFor('configuration', { type: 'integer' }, a).kind).toBe('hidden')
    expect(controlFor('seed', { type: 'integer' }, { analysis: 'x', pattern: 'hexapolar' }).kind).toBe('hidden')
    expect(controlFor('seed', { type: 'integer' }, { analysis: 'x', pattern: 'random' }).kind).toBe('number')
  })

  it('pairs, index lists, number lists, tables', () => {
    expect(controlFor('grid', { type: 'array', default: [5, 5] }, a)).toEqual({ kind: 'pair', integer: true, nullable: false })
    expect(controlFor('extent', { type: 'array', default: [1, 1] }, a)).toEqual({ kind: 'pair', integer: false, nullable: false })
    expect(controlFor('range', { type: ['array', 'null'], default: null, items: { type: 'number' } }, a)).toEqual({ kind: 'pair', integer: false, nullable: true })
    expect(controlFor('wavelengths', { type: 'array', default: [], items: { type: 'integer' } }, a)).toEqual({ kind: 'indexList', domain: 'wavelengths' })
    expect(controlFor('frequencies', { type: 'array', default: [], items: { type: 'number' } }, a).kind).toBe('numberList')
    const t = controlFor('points', { type: 'array', default: [], description: 'Extra points [{"hx", "hy"}], weight 1.' }, a)
    expect(t.kind === 'table' && t.columns.map((c) => c.key)).toEqual(['hx', 'hy'])
    expect(rowKeys('no rows')).toBeNull()
    expect(controlFor('sensor_limit', { type: 'array', default: [] }, a).kind).toBe('sensorLimit')
    expect(controlFor('paths', { type: ['array', 'null'], default: null }, a).kind).toBe('hidden')
    expect(pairLabels('field_center', {})[0]).toBe('Hx')
    expect(pairLabels('size', {})[2]).toBe('×')
  })

  it('groups and tagged unions', () => {
    const g = controlFor('detector', { type: ['object', 'null'], default: null, properties: { pitch_mm: { type: 'array', default: [0.005, 0.005] } } }, a)
    expect(g.kind === 'group' && g.nullable).toBe(true)
    const byDefault = controlFor('enumeration', { type: 'object', default: { max_reflections: 2, exclude_surfaces: [] } }, a)
    expect(byDefault.kind === 'group' && Object.keys(byDefault.properties)).toEqual(['max_reflections', 'exclude_surfaces'])
    const schema: JsonObject = {
      type: 'object',
      default: { type: 'grid', n: 32 },
      oneOf: [
        { title: 'Grid', properties: { type: { const: 'grid' }, n: { type: 'integer', default: 32 } } },
        { title: 'Random', properties: { type: { const: 'random' }, count: { type: 'integer', default: 1024 }, seed: { type: 'integer', default: 1 } } },
      ],
    }
    expect(variantsOf(schema).map((v) => [v.tagKey, v.tag, v.title, Object.keys(v.properties)])).toEqual([
      ['type', 'grid', 'Grid', ['n']],
      ['type', 'random', 'Random', ['count', 'seed']],
    ])
    expect(controlFor('pupil', schema, a).kind).toBe('variants')
  })
})

describe('summary and subscripts', () => {
  it('summarises what differs from a new pane', () => {
    expect(settingsSummary({ fields: { type: 'all' }, rays: 7 }, { fields: { type: 'index', index: 1 }, rays: 7 })).toBe('field 2')
    expect(settingsSummary({}, { spectrum: { kind: 'photopic' } })).toBe('Spectrum photopic')
    expect(settingsSummary({ through_focus: null }, { through_focus: { min: -0.193, max: 0.193, steps: 3 } })).toBe('±0.193 mm · 3 planes')
    expect(settingsSummary({ grid: [5, 5] }, { grid: [9, 9] })).toBe('grid 9 × 9')
  })

  it('splits subscripts', () => {
    expect(subscriptPieces('Σ S_III')).toEqual([
      { text: 'Σ S', sub: false },
      { text: 'III', sub: true },
    ])
    expect(subscriptPieces('C_L sum')).toEqual([
      { text: 'C', sub: false },
      { text: 'L', sub: true },
      { text: ' sum', sub: false },
    ])
  })
})
