import { describe, expect, it } from 'vitest'
import samples from '@samples'
import type { JsonObject } from '@/engine/types'
import type { System } from './system'
import {
  DISPLAY,
  addRows,
  buildChanges,
  convertRow,
  defaultSettings,
  duplicateRows,
  engineLimits,
  generateRows,
  groupRows,
  jobSettings,
  keepingIds,
  lensGroups,
  makeContext,
  makeRow,
  presetGenerator,
  readMeritInfo,
  readResult,
  readSettings,
  rowFromOperand,
  rowLabel,
  rowOperand,
  setElement,
  settingsIssues,
  settingsKey,
  settingsSection,
  share,
  timing,
  trialsCsv,
  validPercentiles,
  worstOffenders,
  type ToleranceKind,
} from './tolerance'

const sample = (id: string) => structuredClone((samples as unknown as { id: string; system: System }[]).find((s) => s.id === id)!.system)
const gauss = sample('double_gauss')
const id = (s: System, i: number) => s.surfaces[i].id

describe('default tolerances', () => {
  const rows = generateRows(presetGenerator('precision'), gauss)
  const count = (k: ToleranceKind) => rows.filter((r) => r.kind === k).length
  const on = (k: ToleranceKind) => rows.filter((r) => r.kind === k).map((r) => gauss.surfaces.findIndex((s) => s.id === r.surface))

  it('covers every lens surface, gap, element and glass of the double Gauss', () => {
    // Lens surfaces: 1–5 and 7–11 (the stop S6 has air on both sides).
    const lens = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    for (const k of ['radius', 'irregularity_astigmatism', 'irregularity_spherical', 'decenter_x', 'decenter_y', 'tilt_x', 'tilt_y'] as ToleranceKind[]) expect(on(k)).toEqual(lens)
    // Every gap inside the lens, the stop's included; not the back focus (S11).
    expect(on('thickness')).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    // Every glass: N-SK16 S1, N-SK16 S3, F2 S4, F2 S7, N-SK16 S8, N-SK16 S10.
    expect(on('index')).toEqual([1, 3, 4, 7, 8, 10])
    expect(count('abbe')).toBe(6)
    // Four elements (two cemented doublets), each moved as a rigid body.
    expect(lensGroups(gauss).map((g) => g.surfaces)).toEqual([[1, 2], [3, 4, 5], [7, 8, 9], [10, 11]])
    for (const k of ['element_decenter_x', 'element_decenter_y', 'element_tilt_x', 'element_tilt_y'] as ToleranceKind[]) expect(count(k)).toBe(4)
    const l2 = rows.filter((r) => r.kind.startsWith('element_') && r.surface === id(gauss, 3))
    expect(l2.length).toBe(4)
    expect(l2.every((r) => r.last === id(gauss, 5) && r.pivot == null)).toBe(true)
    expect(rowLabel(l2[0], gauss)).toBe('Element decenter x L2')
    const all = (v: number) => Object.fromEntries(gauss.surfaces.map((s) => [s.id, v]))
    expect(settingsIssues({ ...defaultSettings(gauss), rows }, makeContext(gauss, { semiDiameters: all(10), abbeNumbers: all(50) })).size).toBe(0)
  })

  it('leaves out kinds at 0 and uses the class values', () => {
    const g = presetGenerator('commercial')
    expect(g.radiusFringes).toBe(5)
    expect(g.thicknessMM).toBe(0.2)
    g.irregularityFringes = 0
    const fewer = generateRows(g, gauss)
    expect(fewer.some((r) => r.kind.startsWith('irregularity'))).toBe(false)
    expect(fewer.find((r) => r.kind === 'thickness')?.minus).toBe(0.2)
  })

  it('groups rows by element, air spaces and the stop', () => {
    const titles = groupRows(rows, gauss).map((g) => g.key)
    expect(titles).toEqual(['L1', 'Air after L1', 'L2', 'Air after L2', 'Stop S6', 'L3', 'Air after L3', 'L4'])
    expect(groupRows(rows, gauss)[0].title).toBe('L1 · N-SK16')
  })

  it('a new design starts from the default set with the back focus', () => {
    const t = readSettings(null, gauss)
    expect(t.rows.length).toBe(rows.length)
    expect(t.compensators.length).toBe(1)
    expect(t.compensators[0].param).toEqual({ param: 'thickness', surface: id(gauss, 11) })
    expect(t.compensators[0].range).toBe(0.5)
  })
})

describe('units', () => {
  const h = 8
  const c = makeContext(gauss, { semiDiameters: { [id(gauss, 1)]: h } })
  const curv = gauss.surfaces[1].shape.curvature as number
  const R = 1 / curv

  it('counts fringes of power over the clear aperture: Δc = Nλ/h²', () => {
    const l = engineLimits(makeRow('radius', id(gauss, 1), 3), c)
    expect(l.ok).toBe(true)
    if (l.ok) {
      expect(l.minus).toBeCloseTo((3 * 0.6328e-3) / (h * h), 15)
      expect(l.plus).toBe(l.minus)
    }
  })

  it('reads ΔR in mm and % of R', () => {
    const mm = engineLimits(makeRow('radius', id(gauss, 1), 0.1, { unit: 'radius_mm' }), c)
    expect(mm.ok && Math.abs(mm.minus - (curv - 1 / (R + 0.1))) < 1e-15 && Math.abs(mm.plus - (1 / (R - 0.1) - curv)) < 1e-15).toBe(true)
    const pc = engineLimits(makeRow('radius', id(gauss, 1), 0.5, { unit: 'percent' }), c)
    expect(pc.ok && Math.abs(pc.plus - (1 / (R - Math.abs(R) * 0.005) - curv)) < 1e-15).toBe(true)
  })

  it('converts fringes ↔ ΔR ↔ % of R and back with the same engine limits', () => {
    const fr = makeRow('radius', id(gauss, 1), 3)
    const base = engineLimits(fr, c)
    for (const u of ['radius_mm', 'percent'] as const) {
      const conv = convertRow(fr, u, c)!
      expect(conv.unit).toBe(u)
      const l = engineLimits(conv, c)
      expect(l.ok && base.ok).toBe(true)
      if (l.ok && base.ok) {
        expect(Math.abs(l.minus - base.minus)).toBeLessThan(1e-8 * base.minus)
        expect(Math.abs(l.plus - base.plus)).toBeLessThan(1e-8 * base.plus)
      }
      const back = convertRow(conv, 'fringes', c)!
      expect(back.minus).toBeCloseTo(3, 6)
      expect(back.plus).toBeCloseTo(3, 6)
    }
    // ΔR of 0.1 mm as % of R.
    const pct = convertRow(makeRow('radius', id(gauss, 1), 0.1, { unit: 'radius_mm' }), 'percent', c)!
    expect(pct.minus).toBeCloseTo((0.1 / Math.abs(R)) * 100, 6)
  })

  it('needs a clear aperture for fringes and a curvature for ΔR', () => {
    expect(convertRow(makeRow('radius', id(gauss, 2), 3, { unit: 'radius_mm' }), 'fringes', makeContext(gauss))).toBeNull()
    const flat = makeRow('radius', id(gauss, 4), 3)
    expect(convertRow(flat, 'radius_mm', makeContext(gauss, { semiDiameters: { [id(gauss, 4)]: 8 } }))).toBeNull()
    const l = engineLimits(makeRow('radius', id(gauss, 4), 1, { unit: 'radius_mm' }), c)
    expect(!l.ok && l.issue).toContain('flat')
  })

  it('tilts go in degrees, Abbe % needs V_d, catalogue glasses address their melt offset', () => {
    const t = engineLimits(makeRow('tilt_x', id(gauss, 1), 1.5), c)
    expect(t.ok && t.minus === 0.025 && t.plus === 0.025).toBe(true)
    expect(convertRow(makeRow('tilt_x', id(gauss, 1), 1), 'degrees', c)!.minus).toBeCloseTo(1 / 60, 9)
    const abbe = makeRow('abbe', id(gauss, 1), 0.5)
    expect(engineLimits(abbe, c).ok).toBe(false)
    const withVd = makeContext(gauss, { abbeNumbers: { [id(gauss, 1)]: 60 } })
    const l = engineLimits(abbe, withVd)
    expect(l.ok && Math.abs(l.minus - 0.3) < 1e-12).toBe(true)
    expect(convertRow(abbe, 'abbe', withVd)!.minus).toBeCloseTo(0.3, 12)
    const op = rowOperand(makeRow('index', id(gauss, 1), 0.001), c)
    expect(op.ok && op.operand.param).toEqual({ param: 'material_dn_d', surface: id(gauss, 1) })
  })
})

describe('the saved set', () => {
  const ctx = makeContext(gauss, { semiDiameters: Object.fromEntries(gauss.surfaces.map((s) => [s.id, 10])), abbeNumbers: Object.fromEntries(gauss.surfaces.map((s) => [s.id, 50])) })

  it('round-trips through the project section', () => {
    const t = defaultSettings(gauss)
    t.rows[0].distribution = { type: 'truncated_normal', sigmas: 3 }
    t.rows[1].plus = 0.9
    t.testWavelengthUm = 0.5461
    t.percentiles = [10, 50, 99]
    t.trials = 120
    t.recordTrials = 500
    t.mode = 'monte_carlo'
    t.criterion.kind = 'mtf'
    t.criterion.threshold = 0.4
    t.criterion.defaults.mtfFrequencies = [10, 30]
    t.criterion.defaults.fields = [0, 2]
    t.generator = presetGenerator('high_precision')
    t.compensators[0].range = null
    const section = JSON.parse(JSON.stringify(settingsSection(t, ctx, readMeritInfo(null)))) as JsonObject
    expect((section.operands as unknown[]).length).toBe(t.rows.length)
    expect((section.monte_carlo as JsonObject).record_trials).toBe(120)
    const back = readSettings(section, gauss)
    expect(settingsKey(back)).toBe(settingsKey(t))
    expect(back.testWavelengthUm).toBe(0.5461)
    expect(back.rows[0].distribution).toEqual({ type: 'truncated_normal', sigmas: 3 })
    expect(back.criterion.defaults.fields).toEqual([0, 2])
  })

  it('reads a spec written elsewhere as rows', () => {
    const c1 = gauss.surfaces[1].shape.curvature as number
    const row = rowFromOperand({ param: { param: 'curvature', surface: id(gauss, 1) }, minus: 1e-5, plus: 1e-5 }, gauss)!
    expect(row.kind).toBe('radius')
    expect(row.unit).toBe('radius_mm')
    expect(row.minus).toBeCloseTo(1 / c1 - 1 / (c1 + 1e-5), 9)
    const tilt = rowFromOperand({ param: { param: 'element_placement', surface: id(gauss, 3), last: id(gauss, 5), pivot: id(gauss, 5), component: 'tilt_y' }, minus: 1 / 60, plus: 1 / 60 }, gauss)!
    expect(tilt.kind).toBe('element_tilt_y')
    expect(tilt.last).toBe(id(gauss, 5))
    expect(tilt.pivot).toBe(id(gauss, 5))
    expect(tilt.minus).toBeCloseTo(1, 12)
    const t = readSettings({ operands: [{ param: { param: 'thickness', surface: id(gauss, 2) }, minus: 0.02, plus: 0.03 }], compensators: [] }, gauss)
    expect(t.rows.length).toBe(1)
    expect(t.rows[0].kind).toBe('thickness')
  })

  it('keeps row identities across an undo by content', () => {
    const a = defaultSettings(gauss)
    const b = structuredClone(a)
    const dup = duplicateRows(b, new Set([b.rows[0].id]))
    expect(b.rows[1].id).toBe(dup[0])
    const reread = keepingIds(readSettings(settingsSection(a, ctx), gauss), b)
    expect(reread.rows.map((r) => r.id)).toEqual(a.rows.map((r) => r.id))
  })

  it('element rows on one surface share their range and pivot', () => {
    const t = defaultSettings(gauss)
    t.rows = []
    const r1 = makeRow('element_decenter_x', id(gauss, 3), 0.03, { last: id(gauss, 5) })
    addRows(t, [r1])
    const r2 = makeRow('element_tilt_x', id(gauss, 3), 1, { last: id(gauss, 4) })
    addRows(t, [r2])
    expect(t.rows[1].last).toBe(id(gauss, 5))
    setElement(t, r1.id, id(gauss, 3), id(gauss, 5), id(gauss, 5))
    expect(t.rows.every((r) => r.pivot === id(gauss, 5))).toBe(true)
    t.rows[1].pivot = null
    expect(settingsIssues(t, ctx).size).toBe(2)
  })

  it('builds the job settings', () => {
    const t = defaultSettings(gauss)
    t.trials = 50
    t.recordTrials = 200
    t.criterion.threshold = 20
    const { settings, rows } = jobSettings(t, ctx, readMeritInfo(null))
    expect(rows.length).toBe(t.rows.length)
    expect(settings.record_trials).toBe(50)
    expect(settings.irregularity_wavelength_um).toBe(0.6328)
    expect(settings.threshold).toBeCloseTo(DISPLAY.criterion('spot', 20), 15)
    expect((settings.default_merit as JsonObject).thickness).toBeNull()
    expect(validPercentiles([97.7, 50, 50, 120, -1, NaN])).toEqual([50, 97.7])
  })
})

describe('results', () => {
  const json: JsonObject = {
    status: 'finished',
    stopped: false,
    compensators: [{ param: { param: 'thickness', surface: 11 }, range: 0.5 }],
    sensitivity: {
      nominal: { criterion: 1e-4, compensators: [28.9], ok: true },
      operands: [
        { index: 0, label: 'a', param: { param: 'thickness', surface: 1 }, minus: 0.05, plus: 0.05, criterion_minus: 2e-4, criterion_plus: 1.5e-4, change_minus: 1e-4, change_plus: 5e-5, worst: 1e-4, ok: true, compensators_minus: [28.8], compensators_plus: [29] },
        { index: 1, label: 'b', param: { param: 'thickness', surface: 2 }, minus: 0.05, plus: 0.05, criterion_minus: 1.1e-4, criterion_plus: 1.1e-4, change_minus: 1e-5, change_plus: 1e-5, worst: 1e-5, ok: true, compensators_minus: [28.9], compensators_plus: [28.9] },
      ],
      ranking: [0, 1],
      rss_change: 1.005e-4,
      predicted_criterion: 2.005e-4,
      cancelled: false,
      evaluations: 5,
    },
    monte_carlo: { trials: 4, failures: 1, seed: 0, criteria: [1e-4, 'inf', 4e-4, 2e-4], percentiles: [{ percentile: 50, value: 2e-4 }], records: [{ trial: 2, criterion: 4e-4, ok: true, compensators: [29.1], perturbations: [0.01, -0.02] }], nominal: { criterion: 1e-4, compensators: [28.9], ok: true } },
  }
  const r = readResult(json, gauss, 'spot')

  it('ranks, shares and finds the worst offenders', () => {
    expect(r.sensitivities.map((s) => s.id)).toEqual([0, 1])
    expect(share(r, r.sensitivities[0])).toBeCloseTo(1 / 1.01, 9)
    expect([...worstOffenders(r)]).toEqual([0])
    expect(r.criteria.length).toBe(3)
    expect(DISPLAY.value('spot', 1e-4)).toBeCloseTo(10, 9)
  })

  it('exports trials with recorded perturbations', () => {
    const rows = [makeRow('thickness', 1, 0.05), makeRow('thickness', 2, 0.05)]
    const text = trialsCsv(r, rows, gauss)
    const lines = text.trim().split('\n')
    expect(lines[0]).toBe('trial,criterion,rms_spot_um,compensator_1,Thickness S1–2,Thickness S2–3')
    expect(lines[2]).toBe('2,,,,,')
    expect(lines[3].endsWith('29.1,0.01,-0.02')).toBe(true)
    const changes = buildChanges([0.025, -0.05], rows, { tolerances: [{ minus: 0.05, plus: 0.05 }, { minus: 0.05, plus: 0.05 }] }, gauss)
    expect(changes[0].fraction).toBe(-1)
    expect(changes[1].text).toBe('+0.025 mm')
  })

  it('estimates the time left from the progress', () => {
    const t = timing({ elapsed: 10, monteCarloElapsed: 8, fraction: 0.6, mode: 'both', trials: 100, operands: 10, lastSecondsPerTrial: null })
    expect(t.trialsDone).toBe(50)
    expect(t.secondsPerTrial).toBeCloseTo(0.16, 12)
    expect(t.secondsLeft).toBeCloseTo(8, 9)
  })
})
