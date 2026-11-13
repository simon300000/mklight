import { describe, expect, it } from 'vitest'
import {
  parseSpectrum,
  parseSpectrumCsv,
  previewSamples,
  primaryIndex,
  resolvedSamples,
  spectrumIssues,
  spectrumJson,
  spectrumSummary,
  type Spectrum,
} from './spectrum'

describe('spectrum setting', () => {
  it('round-trips every kind, keeping sampling short', () => {
    const cases: [unknown, string][] = [
      [null, 'design'],
      [{ kind: 'photopic' }, 'photopic'],
      [{ kind: 'scotopic', sampling: { samples: 5 } }, 'scotopic ×5'],
      [{ kind: 'blackbody', temperature_k: 5500 }, 'blackbody 5500 K'],
      [{ kind: 'explicit', wavelengths: [{ um: 0.55, weight: 1 }, { um: 0.65 }] }, '2 wavelengths'],
      [{ kind: 'sensor', table: [[0.4, 0.1], [0.7, 1]] }, 'sensor (2 rows)'],
    ]
    for (const [json, summary] of cases) {
      const s = parseSpectrum(json as never)!
      expect(s).not.toBeNull()
      expect(spectrumSummary(s)).toBe(summary)
      if (json && (json as { kind: string }).kind !== 'explicit') expect(spectrumJson(s)).toEqual(json)
    }
    expect(spectrumJson({ kind: 'design' })).toBeNull()
  })

  it('rejects what the engine would reject structurally', () => {
    expect(parseSpectrum({ kind: 'laser' })).toBeNull()
    expect(parseSpectrum({ kind: 'photopic', extra: 1 })).toBeNull()
    expect(parseSpectrum({ kind: 'photopic', sampling: { spacing: 'log' } })).toBeNull()
  })

  it('validates ranges and tables', () => {
    const p: Spectrum = { kind: 'photopic', sampling: { samples: 9, lowerUm: 0.3, upperUm: 0.8, spacing: 'equal_weight' } }
    expect(spectrumIssues(p)[0]).toMatch(/inside the photopic table/)
    expect(spectrumIssues({ kind: 'explicit', wavelengths: [] })).toEqual(['Add at least one wavelength.'])
    expect(spectrumIssues({ kind: 'sensor', table: [{ um: 0.5, weight: 1 }], sampling: p.sampling })).toContain('The response table needs at least two rows.')
  })

  it('previews the engine sampling with equal energy per sample', () => {
    const s = previewSamples({ kind: 'photopic', sampling: { samples: 5, lowerUm: 0.38, upperUm: 0.78, spacing: 'equal_weight' } })!
    expect(s).toHaveLength(5)
    const w = s.map((x) => x.weight)
    for (const x of w) expect(x / w[0]).toBeCloseTo(1, 3)
    expect(s[2].um).toBeCloseTo(0.559, 2)
    expect(primaryIndex(s)).toBe(2)
    const u = previewSamples({ kind: 'blackbody', temperatureK: 3200, sampling: { samples: 4, lowerUm: 0.4, upperUm: 0.7, spacing: 'uniform' } })!
    expect(u.map((x) => x.um)).toEqual([0.4, 0.5, 0.6, 0.7].map((x) => expect.closeTo(x, 9)))
    // A 3200 K source rises through the visible.
    expect(u[3].weight).toBeGreaterThan(u[1].weight)
  })

  it('reads resolved samples from system_summary', () => {
    expect(resolvedSamples({ wavelengths_um: [0.5, 0.6], wavelength_weights: [1, 2] })).toEqual([
      { um: 0.5, weight: 1 },
      { um: 0.6, weight: 2 },
    ])
    expect(resolvedSamples({})).toBeNull()
  })
})

describe('sensor CSV', () => {
  it('reads a header, nm by magnitude, decimal commas with semicolons, and reports bad rows', () => {
    const r = parseSpectrumCsv('wavelength;response\n400;0,1\n# comment\n500;0,8\nabc;1\n450;0,5\n500;0,9\n')
    expect(r.header).toBe('wavelength;response')
    expect(r.unit).toBe('nm')
    expect(r.table).toEqual([
      { um: 0.4, weight: 0.1 },
      { um: 0.45, weight: 0.5 },
      { um: 0.5, weight: 0.8 },
    ])
    expect(r.badRows.map((b) => [b.line, b.reason])).toEqual([
      [5, 'not a number'],
      [7, 'repeats a wavelength'],
    ])
  })

  it('keeps µm when the header says so', () => {
    const r = parseSpectrumCsv('lambda (um),qe\n0.4,0.2\n0.6,0.9')
    expect(r.unit).toBe('µm')
    expect(r.table[1]).toEqual({ um: 0.6, weight: 0.9 })
    expect(parseSpectrumCsv('0.4\t1\n0.5\t-1').badRows[0].reason).toBe('response must not be negative')
  })
})
