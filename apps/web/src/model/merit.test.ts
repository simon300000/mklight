import { describe, expect, it } from 'vitest'
import {
  addOperand,
  defaultMerit,
  defaultMeritJson,
  deleteOperands,
  meritJobSettings,
  meritSection,
  moveOperand,
  moveOperandsTo,
  duplicateOperands,
  newOperand,
  operandDescriptor,
  readDefaultMerit,
  readEvaluation,
  readMerit,
  searchOperands,
  specOperands,
  isEflHold,
  type Operand,
} from './merit'

describe('default merit', () => {
  it('builds the engine’s default_merit settings', () => {
    expect(defaultMeritJson(defaultMerit())).toEqual({
      criterion: { type: 'rms_spot', reference: 'centroid', scope: 'monochromatic' },
      sampling: { rings: 3, arms: 6 },
      fields: null,
      wavelengths: null,
      thickness: { glass_min_centre: 1, glass_max_centre: 30, glass_min_edge: 1, air_min_centre: 0.1, air_max_centre: null, air_min_edge: 0.1 },
    })
  })

  it('builds the MTF criterion, selections and no thickness limits', () => {
    const m = { ...defaultMerit(), criterion: 'mtf' as const, mtfFrequencies: [20, 40], worstField: true, fields: [0, 2], thickness: false, squeezeTarget: 1.5 }
    const j = defaultMeritJson(m)
    expect(j.criterion).toEqual({ type: 'mtf', frequencies: [20, 40], target: 0.5, method: 'diffraction', worst_field: true })
    expect(j.fields).toEqual([0, 2])
    expect(j.thickness).toBeNull()
    expect(j.squeeze_target).toBe(1.5)
    expect(readDefaultMerit(j)).toEqual(m)
  })

  it('round-trips the saved section and builds job settings', () => {
    const m = readMerit(null)
    expect(m.mode).toBe('default')
    expect(m.holdEfl).toBe(true)
    expect(meritJobSettings(m)).toEqual({ hold_efl: true, default_merit: defaultMeritJson(defaultMerit()) })
    const custom = { ...m, mode: 'custom' as const, operands: [newOperand(operandDescriptor('efl')!)], specOptions: { on_failure: 'penalty' } }
    const back = readMerit(meritSection(custom))
    expect(back).toEqual({ ...custom, specOptions: { name: '', on_failure: 'penalty' } })
    const s = meritJobSettings(custom)
    expect(s.merit).toMatchObject({ on_failure: 'penalty', schema_version: 1, operands: [{ kind: { type: 'efl' }, role: 'residual' }] })
    expect(s.default_merit).toBeUndefined()
  })
})

describe('custom operands', () => {
  const op = (type: string): Operand => newOperand(operandDescriptor(type)!)
  const combine = (args: number[]): Operand => ({ ...op('combine'), kind: { type: 'combine', op: 'sum', args } })

  it('creates operands with their defaults', () => {
    const r = op('real_ray')
    expect(r.field).toEqual({ field: 'index', index: 0 })
    expect(r.kind).toMatchObject({ type: 'real_ray', quantity: 'y', px: 0, py: 1 })
    expect(op('rms_spot').kind.sampling).toEqual({ rings: 3, arms: 6 })
  })

  it('keeps combine references across add, delete and move', () => {
    let ops = [op('efl'), op('bfl'), combine([0, 1])]
    ops = addOperand(ops, op('rms_spot'), 0).operands
    expect(ops[3].kind.args).toEqual([0, 2])
    ops = deleteOperands(ops, [0])
    expect(ops[2].kind.args).toEqual([1])
    ops = moveOperand(ops, 1, 0)
    expect(ops.map((o) => o.kind.type)).toEqual(['bfl', 'rms_spot', 'combine'])
    expect(ops[2].kind.args).toEqual([0])
  })

  it('moves a multi-selection to a gap and duplicates a selection', () => {
    const ops = [op('efl'), op('bfl'), op('rms_spot'), op('mtf'), combine([0, 2])]
    const m = moveOperandsTo(ops, [0, 2], 4)
    expect(m.operands.map((o) => o.kind.type)).toEqual(['bfl', 'mtf', 'efl', 'rms_spot', 'combine'])
    expect(m.indices).toEqual([2, 3])
    expect(m.operands[4].kind.args).toEqual([2, 3])
    const up = moveOperandsTo(ops, [3], 0)
    expect(up.operands.map((o) => o.kind.type)).toEqual(['mtf', 'efl', 'bfl', 'rms_spot', 'combine'])
    expect(up.operands[4].kind.args).toEqual([1, 3])
    expect(moveOperandsTo(ops, [1], 2).operands).toBe(ops)
    const d = duplicateOperands(ops, [0, 2])
    expect(d.operands.map((o) => o.kind.type)).toEqual(['efl', 'efl', 'bfl', 'rms_spot', 'rms_spot', 'mtf', 'combine'])
    expect(d.indices).toEqual([1, 4])
    expect(d.operands[6].kind.args).toEqual([0, 3])
  })

  it('searches by category', () => {
    const g = searchOperands('mtf')
    expect(g.map((x) => x.category)).toEqual(['Image quality'])
    expect(searchOperands('').reduce((n, x) => n + x.items.length, 0)).toBeGreaterThan(30)
  })

  it('reads an evaluation and drops the EFL hold on conversion', () => {
    const e = readEvaluation({
      merit: 1e-3,
      violation: 0,
      status: { status: 'ok' },
      backend: { reported_on: 'cpu-f64-reference' },
      operands: [{ index: 0, kind: 'rms_spot', value: 0.01, contribution_fraction: 1 }],
      merit_spec: {
        name: '',
        on_failure: 'penalty',
        schema_version: 1,
        operands: [
          { kind: { type: 'rms_spot' }, target: 0, weight: 1 },
          { kind: { type: 'efl' }, target: 50, weight: 1, role: 'constraint', label: 'hold EFL' },
        ],
      },
    })!
    expect(e.merit).toBe(1e-3)
    expect(e.rows[0].fraction).toBe(1)
    const s = specOperands(e)
    expect(s.options).toEqual({ name: '', on_failure: 'penalty' })
    expect(s.operands.filter((o) => !isEflHold(o)).map((o) => o.kind.type)).toEqual(['rms_spot'])
  })
})
