// The run-local `spectrum` setting (`SpectrumSetting` of the macOS app):
// absent (the design's wavelengths) or one of the engine's kinds
//   {kind: explicit, wavelengths: [{um, weight}]}
//   {kind: photopic | scotopic, sampling?}
//   {kind: blackbody, temperature_k, sampling?}
//   {kind: sensor, table: [[um, response]], sampling?}
// with sampling `{samples, range_um: [min, max], spacing: equal_weight | uniform}`.
// Also a local preview of the weights a run will use (labelled approximate
// until the engine resolves them) and the sensor CSV reader.

import type { Json, JsonObject } from '@/engine/types'
import { nf } from './format'

export interface Sample {
  um: number
  weight: number
}

export type Spacing = 'equal_weight' | 'uniform'

export interface Sampling {
  samples: number
  lowerUm: number
  upperUm: number
  spacing: Spacing
}

export const DEFAULT_SAMPLING: Sampling = { samples: 9, lowerUm: 0.38, upperUm: 0.78, spacing: 'equal_weight' }

export type Spectrum =
  | { kind: 'design' }
  | { kind: 'photopic'; sampling: Sampling }
  | { kind: 'scotopic'; sampling: Sampling }
  | { kind: 'blackbody'; temperatureK: number; sampling: Sampling }
  | { kind: 'explicit'; wavelengths: Sample[] }
  | { kind: 'sensor'; table: Sample[]; sampling: Sampling }

export type SpectrumKind = Spectrum['kind']

export const SPECTRUM_KINDS: { kind: SpectrumKind; title: string }[] = [
  { kind: 'design', title: 'Design wavelengths' },
  { kind: 'photopic', title: 'Photopic V(λ)' },
  { kind: 'scotopic', title: 'Scotopic V′(λ)' },
  { kind: 'blackbody', title: 'Blackbody' },
  { kind: 'explicit', title: 'Explicit wavelengths' },
  { kind: 'sensor', title: 'Sensor response' },
]

export const PHOTOPIC_RANGE: [number, number] = [0.36, 0.83]
export const SCOTOPIC_RANGE: [number, number] = [0.38, 0.78]
export const BLACKBODY_PRESETS = [5500, 3200]
export const SPACING_TITLES: Record<Spacing, string> = { equal_weight: 'Equal energy', uniform: 'Uniform' }

const obj = (v: Json | undefined): JsonObject | null => (v !== null && typeof v === 'object' && !Array.isArray(v) ? v : null)
const num = (v: Json | undefined): number | null => (typeof v === 'number' ? v : null)

/** Reads `{samples, range_um, spacing}`; missing keys take the defaults,
 *  unknown keys or wrong types give null. */
export function parseSampling(v: Json | undefined): Sampling | null {
  const s = { ...DEFAULT_SAMPLING }
  if (v === undefined || v === null) return s
  const o = obj(v)
  if (!o || Object.keys(o).some((k) => !['samples', 'range_um', 'spacing'].includes(k))) return null
  if (o.samples !== undefined) {
    if (typeof o.samples !== 'number' || !Number.isInteger(o.samples)) return null
    s.samples = o.samples
  }
  if (o.range_um !== undefined) {
    const r = Array.isArray(o.range_um) ? o.range_um.map(num) : []
    if (r.length !== 2 || r[0] == null || r[1] == null) return null
    s.lowerUm = r[0]
    s.upperUm = r[1]
  }
  if (o.spacing !== undefined) {
    if (o.spacing !== 'equal_weight' && o.spacing !== 'uniform') return null
    s.spacing = o.spacing
  }
  return s
}

/** Only what differs from the defaults (null when nothing does). */
export function samplingJson(s: Sampling): JsonObject | null {
  const o: JsonObject = {}
  if (s.samples !== DEFAULT_SAMPLING.samples) o.samples = s.samples
  if (s.lowerUm !== DEFAULT_SAMPLING.lowerUm || s.upperUm !== DEFAULT_SAMPLING.upperUm) o.range_um = [s.lowerUm, s.upperUm]
  if (s.spacing !== DEFAULT_SAMPLING.spacing) o.spacing = s.spacing
  return Object.keys(o).length ? o : null
}

/** Reads the setting; absent/null is the design; null for anything the
 *  engine would reject structurally. */
export function parseSpectrum(v: Json | undefined): Spectrum | null {
  if (v === undefined || v === null) return { kind: 'design' }
  const o = obj(v)
  if (!o || typeof o.kind !== 'string') return null
  const only = (...allowed: string[]) => Object.keys(o).every((k) => k === 'kind' || allowed.includes(k))
  switch (o.kind) {
    case 'photopic':
    case 'scotopic': {
      const s = only('sampling') ? parseSampling(o.sampling) : null
      return s ? { kind: o.kind, sampling: s } : null
    }
    case 'blackbody': {
      const t = num(o.temperature_k)
      const s = only('temperature_k', 'sampling') ? parseSampling(o.sampling) : null
      return s && t != null ? { kind: 'blackbody', temperatureK: t, sampling: s } : null
    }
    case 'explicit': {
      if (!only('wavelengths') || !Array.isArray(o.wavelengths)) return null
      const out: Sample[] = []
      for (const w of o.wavelengths) {
        const wo = obj(w)
        const um = num(wo?.um)
        if (!wo || um == null || Object.keys(wo).some((k) => k !== 'um' && k !== 'weight')) return null
        const weight = wo.weight === undefined ? 1 : num(wo.weight)
        if (weight == null) return null
        out.push({ um, weight })
      }
      return { kind: 'explicit', wavelengths: out }
    }
    case 'sensor': {
      if (!only('table', 'sampling') || !Array.isArray(o.table)) return null
      const s = parseSampling(o.sampling)
      if (!s) return null
      const table: Sample[] = []
      for (const r of o.table) {
        if (!Array.isArray(r) || r.length !== 2 || typeof r[0] !== 'number' || typeof r[1] !== 'number') return null
        table.push({ um: r[0], weight: r[1] })
      }
      return { kind: 'sensor', table, sampling: s }
    }
  }
  return null
}

/** The engine's JSON; null for the design (the setting is left out). */
export function spectrumJson(s: Spectrum): JsonObject | null {
  const withSampling = (o: JsonObject, sm: Sampling): JsonObject => {
    const j = samplingJson(sm)
    return j ? { ...o, sampling: j } : o
  }
  switch (s.kind) {
    case 'design':
      return null
    case 'photopic':
    case 'scotopic':
      return withSampling({ kind: s.kind }, s.sampling)
    case 'blackbody':
      return withSampling({ kind: 'blackbody', temperature_k: s.temperatureK }, s.sampling)
    case 'explicit':
      return { kind: 'explicit', wavelengths: s.wavelengths.map((w) => ({ um: w.um, weight: w.weight })) }
    case 'sensor':
      return withSampling({ kind: 'sensor', table: s.table.map((r) => [r.um, r.weight]) }, s.sampling)
  }
}

export const samplingOf = (s: Spectrum): Sampling | null => ('sampling' in s ? s.sampling : null)

export function withSampling(s: Spectrum, sm: Sampling): Spectrum {
  return 'sampling' in s ? { ...s, sampling: sm } : s
}

/** A new spectrum of `kind`, seeded from the current one and the design. */
export function spectrumOfKind(kind: SpectrumKind, cur: Spectrum, design: Sample[]): Spectrum {
  const sm = samplingOf(cur) ?? { ...DEFAULT_SAMPLING }
  switch (kind) {
    case 'design':
      return { kind }
    case 'photopic':
    case 'scotopic':
      return { kind, sampling: sm }
    case 'blackbody':
      return { kind, temperatureK: 5500, sampling: sm }
    case 'explicit':
      return { kind, wavelengths: design.length ? design.map((w) => ({ ...w })) : [{ um: 0.5875618, weight: 1 }] }
    case 'sensor':
      return { kind, table: cur.kind === 'sensor' ? cur.table : [], sampling: sm }
  }
}

// MARK: Validation

function samplingIssues(s: Sampling): string[] {
  const out: string[] = []
  if (!(Number.isInteger(s.samples) && s.samples >= 1 && s.samples <= 4096)) out.push('Samples must be between 1 and 4096.')
  if (!(s.lowerUm > 0 && s.upperUm > s.lowerUm && Number.isFinite(s.upperUm))) out.push('The range must be positive and increasing.')
  return out
}

function rangeIssue(s: Sampling, r: [number, number], what: string): string[] {
  if (!(s.lowerUm > 0 && s.upperUm > s.lowerUm)) return []
  if (s.lowerUm < r[0] - 1e-12 || s.upperUm > r[1] + 1e-12) return [`The range must lie inside ${what} (${r[0].toFixed(3)}–${r[1].toFixed(3)} µm).`]
  return []
}

export function tableIssue(t: Sample[]): string | null {
  if (t.length < 2) return 'The response table needs at least two rows.'
  if (t.some((r) => !(r.um > 0 && Number.isFinite(r.um) && r.weight >= 0 && Number.isFinite(r.weight)))) return 'Responses must not be negative.'
  for (let i = 1; i < t.length; i++) if (t[i - 1].um >= t[i].um) return 'Wavelengths must increase.'
  return null
}

/** What the engine would refuse, in words, before a run. */
export function spectrumIssues(s: Spectrum): string[] {
  switch (s.kind) {
    case 'design':
      return []
    case 'photopic':
      return [...samplingIssues(s.sampling), ...rangeIssue(s.sampling, PHOTOPIC_RANGE, 'the photopic table')]
    case 'scotopic':
      return [...samplingIssues(s.sampling), ...rangeIssue(s.sampling, SCOTOPIC_RANGE, 'the scotopic table')]
    case 'blackbody':
      return [...(s.temperatureK > 0 && Number.isFinite(s.temperatureK) ? [] : ['The temperature must be positive.']), ...samplingIssues(s.sampling)]
    case 'explicit': {
      const out: string[] = []
      if (!s.wavelengths.length) out.push('Add at least one wavelength.')
      if (s.wavelengths.some((w) => !(w.um > 0 && Number.isFinite(w.um)))) out.push('Wavelengths must be positive.')
      if (s.wavelengths.some((w) => !(w.weight >= 0 && Number.isFinite(w.weight)))) out.push('Weights must not be negative.')
      return out
    }
    case 'sensor': {
      const out = samplingIssues(s.sampling)
      const p = tableIssue(s.table)
      if (p) out.push(p)
      else out.push(...rangeIssue(s.sampling, [s.table[0].um, s.table[s.table.length - 1].um], 'the response table'))
      return out
    }
  }
}

// MARK: Text

const kelvin = (t: number) => (Number.isInteger(t) ? `${t} K` : `${t.toFixed(1)} K`)
const samplesSuffix = (s: Sampling) => (s.samples === DEFAULT_SAMPLING.samples ? '' : ` ×${s.samples}`)

/** `photopic`, `blackbody 5500 K`, `3 wavelengths`, `sensor (42 rows)`. */
export function spectrumSummary(s: Spectrum): string {
  switch (s.kind) {
    case 'design':
      return 'design'
    case 'photopic':
    case 'scotopic':
      return s.kind + samplesSuffix(s.sampling)
    case 'blackbody':
      return `blackbody ${kelvin(s.temperatureK)}` + samplesSuffix(s.sampling)
    case 'explicit':
      return s.wavelengths.length === 1 ? `${Number(s.wavelengths[0].um.toPrecision(4))} µm` : `${s.wavelengths.length} wavelengths`
    case 'sensor':
      return `sensor (${s.table.length} rows)` + samplesSuffix(s.sampling)
  }
}

// MARK: Preview

const logExpm1 = (x: number) => (x > 50 ? x : Math.log(Math.expm1(x)))

function interpolate(t: Sample[], x: number): number {
  if (!t.length || x < t[0].um || x > t[t.length - 1].um) return 0
  let hi = t.findIndex((r) => r.um >= x)
  if (hi < 0) hi = t.length - 1
  hi = Math.min(Math.max(hi, 1), t.length - 1)
  const a = t[hi - 1], b = t[hi]
  return a.weight + ((b.weight - a.weight) * (x - a.um)) / (b.um - a.um)
}

/** The continuous response sampled, peak 1 (photopic and scotopic as
 *  Gaussian fits — a display approximation); null for design and explicit. */
export function previewCurve(s: Spectrum, points = 161): Sample[] | null {
  const sm = samplingOf(s)
  if (!sm || !(sm.lowerUm > 0 && sm.upperUm > sm.lowerUm && Number.isFinite(sm.upperUm)) || points < 2) return null
  let f: (l: number) => number
  switch (s.kind) {
    case 'photopic':
      f = (l) => 1.019 * Math.exp(-285.4 * (l - 0.559) ** 2)
      break
    case 'scotopic':
      f = (l) => 0.992 * Math.exp(-321.9 * (l - 0.503) ** 2)
      break
    case 'blackbody': {
      const t = s.temperatureK
      if (!(t > 0 && Number.isFinite(t))) return null
      f = (l) => Math.exp(-5 * Math.log(l) - logExpm1(14387.768775039337 / (l * t)))
      break
    }
    case 'sensor':
      if (tableIssue(s.table)) return null
      f = (l) => interpolate(s.table, l)
      break
    default:
      return null
  }
  const xs = Array.from({ length: points }, (_, i) => sm.lowerUm + ((sm.upperUm - sm.lowerUm) * i) / (points - 1))
  const ys = xs.map(f)
  const peak = Math.max(0, ...ys.filter(Number.isFinite))
  if (!(peak > 0)) return null
  return xs.map((um, i) => ({ um, weight: Number.isFinite(ys[i]) ? ys[i] / peak : 0 }))
}

/** The engine's two spacings on a finely tabulated curve (trapezoids). */
export function sampleCurve(curve: Sample[], s: Sampling): Sample[] | null {
  if (curve.length < 2) return null
  const energy = [0], moment = [0]
  for (let i = 1; i < curve.length; i++) {
    const p = curve[i - 1], q = curve[i]
    const e = ((q.um - p.um) * (p.weight + q.weight)) / 2
    energy.push(energy[i - 1] + e)
    moment.push(moment[i - 1] + (e * (p.um + q.um)) / 2)
  }
  const total = energy[energy.length - 1]
  if (!(total > 0)) return null
  const cumulative = (x: number): [number, number] => {
    let i = curve.findIndex((c) => c.um >= x)
    if (i < 0) i = curve.length - 1
    i = Math.min(Math.max(i, 1), curve.length - 1)
    const a = curve[i - 1], b = curve[i]
    const f = (x - a.um) / (b.um - a.um)
    const vx = a.weight + (b.weight - a.weight) * f
    const e = ((x - a.um) * (a.weight + vx)) / 2
    return [energy[i - 1] + e, moment[i - 1] + (e * (a.um + x)) / 2]
  }
  const n = s.samples
  const out: Sample[] = []
  if (s.spacing === 'uniform') {
    const d = (s.upperUm - s.lowerUm) / (Math.max(n, 2) - 1)
    for (let i = 0; i < n; i++) {
      const um = n === 1 ? (s.lowerUm + s.upperUm) / 2 : s.lowerUm + d * i
      const half = n === 1 ? (s.upperUm - s.lowerUm) / 2 : d / 2
      out.push({ um, weight: cumulative(Math.min(um + half, s.upperUm))[0] - cumulative(Math.max(um - half, s.lowerUm))[0] })
    }
  } else {
    let left = s.lowerUm
    for (let i = 0; i < n; i++) {
      const target = (total * (i + 1)) / n
      let lo = left, hi = s.upperUm
      if (i + 1 < n)
        for (let k = 0; k < 60; k++) {
          const mid = (lo + hi) / 2
          if (cumulative(mid)[0] < target) lo = mid
          else hi = mid
        }
      const right = i + 1 === n ? s.upperUm : hi
      const [e0, m0] = cumulative(left), [e1, m1] = cumulative(right)
      const e = e1 - e0
      out.push({ um: e > 0 ? (m1 - m0) / e : (left + right) / 2, weight: e })
      left = right
    }
  }
  return out
}

/** The weighted wavelengths a run would use, approximated locally; explicit
 *  wavelengths are exact; null for the design. */
export function previewSamples(s: Spectrum): Sample[] | null {
  if (s.kind === 'design') return null
  if (s.kind === 'explicit') return s.wavelengths
  if (samplingIssues(s.sampling).length) return null
  const curve = previewCurve(s, 801)
  return curve ? sampleCurve(curve, s.sampling) : null
}

/** The index the engine makes primary: nearest the energy-weighted mean. */
export function primaryIndex(samples: Sample[]): number | null {
  if (!samples.length) return null
  const total = samples.reduce((a, w) => a + w.weight, 0)
  const mean = total > 0 ? samples.reduce((a, w) => a + w.um * w.weight, 0) / total : samples[0].um
  let best = 0
  samples.forEach((w, i) => {
    if (Math.abs(w.um - mean) < Math.abs(samples[best].um - mean)) best = i
  })
  return best
}

/** The run's wavelengths as the engine resolves them, from a `system_summary`
 *  result (`wavelengths_um`, `wavelength_weights`). */
export function resolvedSamples(json: JsonObject | null | undefined): Sample[] | null {
  const um = Array.isArray(json?.wavelengths_um) ? json.wavelengths_um.map(nf) : []
  if (!um.length) return null
  const w = Array.isArray(json?.wavelength_weights) ? json.wavelength_weights.map(nf) : um.map(() => 1)
  if (w.length !== um.length) return null
  return um.map((x, i) => ({ um: x, weight: w[i] }))
}

// MARK: CSV

export interface BadRow {
  line: number
  text: string
  reason: string
}

export interface SpectrumCsv {
  table: Sample[]
  unit: 'µm' | 'nm'
  header: string | null
  badRows: BadRow[]
}

/** Two columns (λ, response); optional header; comma, semicolon or tab (a
 *  decimal comma with the latter two); `#` comments; nm detected by header
 *  or magnitude; unreadable rows reported by line; sorted by wavelength. */
export function parseSpectrumCsv(text: string): SpectrumCsv {
  const rows: { line: number; text: string; um: number; value: number }[] = []
  const bad: BadRow[] = []
  let header: string | null = null
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  lines.forEach((raw, k) => {
    const line = raw.trim().replace(/^﻿/, '').trim()
    if (!line || line.startsWith('#')) return
    const sep = line.includes('\t') ? '\t' : line.includes(';') ? ';' : ','
    const comma = sep !== ','
    const cells = line.split(sep).map((c) => c.trim().replace(/^"|"$/g, ''))
    const number = (s: string): number | null => {
      const t = comma ? s.replace(',', '.') : s
      if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return null
      return Number(t)
    }
    const a = cells.length > 0 ? number(cells[0]) : null
    const b = cells.length > 1 ? number(cells[1]) : null
    if (a == null || b == null) {
      if (!rows.length && !bad.length && header == null && cells.some((c) => number(c) == null)) header = line
      else bad.push({ line: k + 1, text: line, reason: cells.length < 2 ? 'needs two columns' : 'not a number' })
      return
    }
    if (!(Number.isFinite(a) && a > 0)) bad.push({ line: k + 1, text: line, reason: 'wavelength must be positive' })
    else if (!(Number.isFinite(b) && b >= 0)) bad.push({ line: k + 1, text: line, reason: 'response must not be negative' })
    else rows.push({ line: k + 1, text: line, um: a, value: b })
  })
  const words = new Set((header ?? '').toLowerCase().split(/[^a-zµ]+/).filter(Boolean))
  const has = (ws: string[]) => ws.some((w) => words.has(w))
  const unit: 'µm' | 'nm' = has(['nm', 'nanometer', 'nanometers', 'nanometre', 'nanometres'])
    ? 'nm'
    : has(['um', 'µm', 'micron', 'microns', 'micrometer', 'micrometers', 'micrometre', 'micrometres'])
      ? 'µm'
      : Math.max(0, ...rows.map((r) => r.um)) > 50
        ? 'nm'
        : 'µm'
  const scale = unit === 'nm' ? 0.001 : 1
  const table: Sample[] = []
  for (const r of [...rows].sort((p, q) => p.um - q.um || p.line - q.line)) {
    const um = r.um * scale
    if (table.length && Math.abs(table[table.length - 1].um - um) < 1e-12) {
      bad.push({ line: r.line, text: r.text, reason: 'repeats a wavelength' })
      continue
    }
    table.push({ um, weight: r.value })
  }
  return { table, unit, header, badRows: bad.sort((p, q) => p.line - q.line) }
}
