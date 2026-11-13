import { describe, expect, it } from 'vitest'
import type { AnalysisResult, JsonObject } from '@/engine/types'
import {
  candidatesJson,
  candidatesSummary,
  changeSummary,
  combinationHighlights,
  defaultCandidates,
  designHighlights,
  glassSlots,
  glassSubstitutionSettings,
  isSubstitutedMaterialParam,
  mapStatuses,
  meritJobSettings,
  meritSummary,
  offsetNote,
  parseGlassProgress,
  parseNames,
  readGlassInfo,
  readGlassMap,
  readSubstitution,
  runningMinimum,
  startGlasses,
  statusText,
  type GlassSearchSettings,
} from './glass'
import { sampleById } from './project'
import { clone, meltOffsetText } from './system'

const doubleGauss = () => clone(sampleById('double_gauss')!.system)

function search(p: Partial<GlassSearchSettings> = {}): GlassSearchSettings {
  return {
    selected: [],
    candidates: defaultCandidates(),
    overrides: {},
    algorithm: 'tabu',
    maxCombinations: 100,
    maxSeconds: null,
    localIterations: 100,
    topN: 10,
    useDesignVariables: true,
    ...p,
  }
}

describe('candidates', () => {
  it('writes the engine candidates object', () => {
    const c = defaultCandidates()
    expect(candidatesJson(c)).toEqual({
      catalogs: [],
      statuses: ['preferred', 'standard', 'melt'],
      include: [],
      exclude: [],
      nearest: 10,
      window: null,
    })
    const w = candidatesJson({ ...c, rule: 'window', windowNd: 0.02, windowVd: 3, maxRelativeCost: 2, statuses: ['obsolete', 'standard'] })
    expect(w.nearest).toBeNull()
    expect(w.window).toEqual({ nd: 0.02, vd: 3 })
    expect(w.max_relative_cost).toBe(2)
    expect(w.statuses).toEqual(['standard', 'obsolete'])
    expect('min_transmission' in w).toBe(false)
  })
  it('summarises in plain words', () => {
    expect(candidatesSummary(defaultCandidates())).toBe('10 nearest · all catalogs')
    const s = candidatesSummary({ ...defaultCandidates(), catalogs: ['SCHOTT'], include: ['N-BK7'], exclude: ['F2', 'SF5'], minTransmission: 0.9 })
    expect(s).toBe('10 nearest · SCHOTT · +1 · −2 · limits')
  })
  it('splits name lists', () => {
    expect(parseNames('N-BK7, SCHOTT:F2;\n  SF5 ,,')).toEqual(['N-BK7', 'SCHOTT:F2', 'SF5'])
  })
})

describe('glass gaps', () => {
  it('lists the glasses of a design with their surface ids', () => {
    const s = doubleGauss()
    const slots = glassSlots(s)
    expect(slots.map((x) => x.index)).toEqual([1, 3, 4, 7, 8, 10])
    expect(slots[0].material).toBe('N-SK16')
    expect(slots.every((x) => !x.isModel && x.offset === '')).toBe(true)
  })
  it('notes melt offsets of ticked glasses and keeps them on the glass they leave', () => {
    const s = doubleGauss()
    s.surfaces[1].material = { type: 'catalog', catalog: null, name: 'N-SK16', offset: { dn_d: 0.001, dv_d: -0.5 } }
    const id = s.surfaces[1].id
    const text = meltOffsetText(s.surfaces[1].material)
    expect(text).not.toBe('')
    expect(glassSlots(s)[0].offset).toBe(text)
    expect(startGlasses(s)[id]).toBe(`N-SK16 ${text}`)
    expect(offsetNote(s, [])).toBeNull()
    expect(offsetNote(s, [id])).toBe(
      `S1 N-SK16 ${text} carries a melt offset. A replacement glass is installed without it, even the same glass: the offset belonged to the original melt.`,
    )
  })
  it('leaves material variables of ticked glasses out', () => {
    expect(isSubstitutedMaterialParam({ param: 'material_dn_d', surface: 3 }, [3])).toBe(true)
    expect(isSubstitutedMaterialParam({ param: 'model_nd', surface: 3 }, [1])).toBe(false)
    expect(isSubstitutedMaterialParam({ param: 'curvature', surface: 3 }, [3])).toBe(false)
  })
})

describe('settings', () => {
  it('takes the merit of the Optimize panel', () => {
    expect(meritJobSettings(null)).toEqual({})
    const def = { type: 'rms_spot' }
    expect(meritJobSettings({ mode: 'default', default_merit: { criterion: def }, hold_efl: false, spec: {} })).toEqual({
      hold_efl: false,
      default_merit: { criterion: def },
    })
    const spec = { name: '', operands: [{ kind: { type: 'efl' } }] }
    expect(meritJobSettings({ mode: 'custom', spec })).toEqual({ hold_efl: true, merit: spec })
    expect(meritSummary({ mode: 'custom', spec })).toBe('Custom merit, 1 operand, focal length held.')
    expect(meritSummary(null)).toBe('Default merit: RMS spot, focal length held.')
  })
  it('builds the glass_substitution settings in surface order with per-lens candidates', () => {
    const s = doubleGauss()
    const ids = glassSlots(s).map((x) => x.id)
    const own = { ...defaultCandidates(), nearest: 3 }
    const variables = [
      { param: { param: 'curvature', surface: ids[0] }, lower: null, upper: null },
      { param: { param: 'material_dn_d', surface: ids[2] }, lower: -0.01, upper: 0.01 },
      { param: { param: 'thickness', surface: ids[1] }, lower: 2, upper: 2 },
    ]
    const out = glassSubstitutionSettings(s, null, variables, search({ selected: [ids[2], ids[0]], overrides: { [ids[2]]: own }, maxSeconds: 30 }))
    expect(out.algorithm).toBe('tabu')
    expect(out.max_seconds).toBe(30)
    expect((out.material_variables as JsonObject[]).map((m) => m.surface)).toEqual([ids[0], ids[2]])
    expect(((out.material_variables as JsonObject[])[1].candidates as JsonObject).nearest).toBe(3)
    // The offset of a ticked glass and the pinned thickness are not varied.
    expect(out.variables).toEqual([variables[0]])
    const auto = glassSubstitutionSettings(s, null, variables, search({ useDesignVariables: false }))
    expect('variables' in auto).toBe(false)
    expect(auto.material_variables).toEqual([])
  })
})

const RESULT: JsonObject = {
  status: 'stopped',
  stopped: true,
  algorithm: 'tabu',
  termination: 'stopped',
  starting_merit: 0.02,
  statistics: { combinations: 7, failed: 1 },
  history: [{ best_merit: 0.03 }, { best_merit: 0.018 }, { best_merit: 0.019 }, { best_merit: 'nan' }],
  candidate_sets: [{ surface: 1, glasses: [{}, {}, {}] }],
  results: [
    {
      rank: 1,
      merit: 0.015,
      violation: 0,
      feasible: true,
      glasses: [
        { surface: 1, catalog: 'SCHOTT', name: 'N-SK2', nd: 1.607, vd: 56.7, dpgf: 0 },
        { surface: 3, catalog: 'SCHOTT', name: 'N-SK2', nd: 1.607, vd: 56.7, dpgf: 0 },
      ],
      variables: [
        { param: { param: 'curvature', surface: 1 }, start: 0.01, value: 0.0102, change: 0.0002 },
        { param: { param: 'thickness', surface: 2 }, start: 5, value: 5, change: 0 },
      ],
      system: { surfaces: [] },
      local_termination: 'cancelled',
    },
    { merit: 'inf', feasible: false, glasses: [] },
  ],
}

describe('result', () => {
  it('reads the ranking', () => {
    const r = readSubstitution(RESULT)!
    expect(r.stopped).toBe(true)
    expect(r.combinations).toBe(7)
    expect(r.history).toEqual([0.03, 0.018, 0.018])
    expect(r.candidateCounts).toEqual({ 1: 3 })
    expect(r.ranked[1].rank).toBe(2)
    expect(r.ranked[1].merit).toBe(Infinity)
    expect(r.ranked[1].system).toBeNull()
    expect(changeSummary(r.ranked[0])).toBe('1 variable re-optimized · cut short')
    expect(statusText(r, false)).toBe('7 combinations re-optimized, 1 failed. Stopped early: the ranking so far. Select a combination to preview it in the window.')
    expect(readSubstitution({})).toBeNull()
  })
  it('rings a candidate glass once with all its surfaces', () => {
    const s = doubleGauss()
    const r = readSubstitution(RESULT)!
    expect(combinationHighlights(r.ranked[0], s)).toEqual([{ id: 'SCHOTT/N-SK2', label: 'S1, S3' }])
  })
  it('keeps a running minimum', () => {
    expect(runningMinimum([3, 1, 2, 0.5])).toEqual([3, 1, 1, 0.5])
  })
  it('parses the live stage text', () => {
    const p = parseGlassProgress('glass_substitution: combinations 12; best merit 1.234500e-2; glasses [{"catalog":"S","name":"N-BK7"},{"name":"F2"}]')!
    expect(p.combinations).toBe(12)
    expect(p.bestMerit).toBeCloseTo(0.012345)
    expect(p.glasses).toEqual(['N-BK7', 'F2'])
    expect(parseGlassProgress('Preparing candidates')).toBeNull()
    expect(parseGlassProgress(null, { combinations: 3, best_merit: 0.5, glasses: ['A'] })).toEqual({ combinations: 3, bestMerit: 0.5, glasses: ['A'] })
  })
})

describe('glass map', () => {
  const json: JsonObject = {
    coordinates: 'coordinates',
    glasses: [
      { kind: 'catalog', catalog: 'SCHOTT', name: 'N-BK7', status: 'preferred', surfaces: [1], nd: 0, vd: 0, pgf: 0 },
      { kind: 'catalog', catalog: 'SCHOTT', name: 'F2', status: 'obsolete', surfaces: [] },
      { kind: 'catalog', catalog: 'OHARA', name: 'S-X', status: { other: 9 }, surfaces: [] },
      { kind: 'offset', catalog: 'SCHOTT', name: 'N-BK7', status: 'preferred', surfaces: [3], offset: { dn_d: 0.001, dv_d: 0 } },
      { kind: 'model', name: '1.6/40', status: null, surfaces: [5] },
    ],
  }
  const result: AnalysisResult = {
    json,
    buffers: { coordinates: new Float64Array([1.5168, 64.17, 0.5349, 1.62, 36.37, 0.5828, 1.7, 30, 0.6, 1.5178, 64.17, 0.5349, 1.6, 40, 0.58]) },
    runInfo: { snapshot_hash: 'h', analysis: 'glass_map', settings: {}, backend: 'cpu', warnings: [] },
  }
  it('reads points from the coordinates buffer with ids, status and design use', () => {
    const g = readGlassMap(result)
    expect(g.map((x) => x.id)).toEqual(['SCHOTT/N-BK7', 'SCHOTT/F2', 'OHARA/S-X', 'offset:SCHOTT/N-BK7:3', 'model:1.6/40:5'])
    expect(g[0].nd).toBe(1.5168)
    expect(g[2].statusLabel).toBe('Status 9')
    expect(g[2].status).toBe('unknown')
    expect(g[3].offset).toEqual({ dn_d: 0.001, dv_d: 0 })
    expect(designHighlights(g).map((h) => h.label)).toEqual(['S1', 'S3', 'S5'])
    expect(mapStatuses(g)).toEqual(['preferred', 'obsolete', 'unknown'])
    // JSON only: coordinates from the glass fields.
    expect(readGlassMap(json)[0].nd).toBe(0)
  })
  it('reads glass data', () => {
    const r: AnalysisResult = {
      json: {
        name: 'N-BK7', catalog: 'SCHOTT', nd: 1.5168, vd: 64.17, pgf: 0.5349, ne: 1.5187, offset: { dn_d: 0, dv_d: 0 },
        dispersion: { formula: 'sellmeier' }, cte: 7.1e-6, density: null, comment: ' ',
        indices: [{ wavelength_um: 0.5876, n: 1.5168, n_catalog: 1.5168, absorption_per_mm: 0, in_range: true }],
      },
      buffers: { dispersion_curve: new Float64Array([0.4, 1.53, 0.7, 1.51]) },
      runInfo: { snapshot_hash: 'h', analysis: 'glass_info', settings: {}, backend: 'cpu', warnings: [] },
    }
    const d = readGlassInfo(r)!
    expect(d.formula).toBe('Sellmeier')
    expect(d.curve).toEqual([[0.4, 1.53], [0.7, 1.51]])
    expect(d.comment).toBeNull()
    expect(d.density).toBeNull()
    expect(d.indices[0].inRange).toBe(true)
  })
})
