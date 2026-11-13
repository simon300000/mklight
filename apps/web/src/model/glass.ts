// Glass substitution and glass map helpers (apps/macos/Sources/MKLightUI/ViewModels/
// GlassSubstitution.swift, GlassOffsets.swift and MapReaders.swift "Glass map"):
// the candidate rules of a lens in plain terms and as the engine's `candidates`
// object, the design's glass gaps, the `glass_substitution` settings and result,
// its live progress, and the `glass_map` / `glass_info` results. Pure functions.

import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { nf, sig } from './format'
import { offsetCompactText } from './materials'
import { solveFor, type ParamPath, type VariableSpec } from './params'
import { imageIndex, materialText, meltOffsetText, type System } from './system'

const arr = (v: Json | undefined): Json[] => (Array.isArray(v) ? v : [])
const obj = (v: Json | undefined): JsonObject | null => (v !== null && typeof v === 'object' && !Array.isArray(v) ? v : null)
const str = (v: Json | undefined): string | null => (typeof v === 'string' ? v : null)
const int = (v: Json | undefined): number | null => (typeof v === 'number' && Number.isFinite(v) ? Math.round(v) : null)

// MARK: Candidates

export const ALL_STATUSES = ['preferred', 'standard', 'melt', 'special', 'obsolete'] as const
export type CandidateStatus = (typeof ALL_STATUSES)[number]
const DEFAULT_STATUSES: CandidateStatus[] = ['standard', 'preferred', 'melt']

/** Which catalogue glasses may replace a lens's glass (the engine's `CandidateSet`). */
export interface GlassCandidates {
  /** Catalogues to draw from; empty: every loaded catalogue. */
  catalogs: string[]
  statuses: CandidateStatus[]
  rule: 'nearest' | 'window'
  /** The K glasses nearest the current one. */
  nearest: number
  /** Half-widths around the current glass. */
  windowNd: number
  windowVd: number
  /** Names or `catalog:name`. */
  include: string[]
  exclude: string[]
  minTransmission: number | null
  maxRelativeCost: number | null
  maxClimateClass: number | null
  maxStainClass: number | null
}

export function defaultCandidates(): GlassCandidates {
  return {
    catalogs: [],
    statuses: [...DEFAULT_STATUSES],
    rule: 'nearest',
    nearest: 10,
    windowNd: 0.05,
    windowVd: 5,
    include: [],
    exclude: [],
    minTransmission: null,
    maxRelativeCost: null,
    maxClimateClass: null,
    maxStainClass: null,
  }
}

/** The engine's `candidates` object. */
export function candidatesJson(c: GlassCandidates): JsonObject {
  const o: JsonObject = {
    catalogs: [...c.catalogs],
    statuses: ALL_STATUSES.filter((s) => c.statuses.includes(s)),
    include: [...c.include],
    exclude: [...c.exclude],
  }
  if (c.rule === 'nearest') {
    o.nearest = c.nearest
    o.window = null
  } else {
    o.nearest = null
    o.window = { nd: c.windowNd, vd: c.windowVd }
  }
  const put = (k: string, v: number | null) => {
    if (v != null && Number.isFinite(v)) o[k] = v
  }
  put('min_transmission', c.minTransmission)
  put('max_relative_cost', c.maxRelativeCost)
  put('max_climate_class', c.maxClimateClass)
  put('max_stain_class', c.maxStainClass)
  return o
}

const sameSet = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x) => b.includes(x))

/** A short description, e.g. "10 nearest · all catalogs". */
export function candidatesSummary(c: GlassCandidates): string {
  const parts: string[] = []
  parts.push(c.rule === 'nearest' ? `${c.nearest} nearest` : `n_d ±${sig(c.windowNd, 4)}, V_d ±${sig(c.windowVd, 4)}`)
  parts.push(c.catalogs.length ? c.catalogs.join(', ') : 'all catalogs')
  if (!sameSet(c.statuses, DEFAULT_STATUSES)) parts.push(ALL_STATUSES.filter((s) => c.statuses.includes(s)).join(', '))
  if (c.include.length) parts.push(`+${c.include.length}`)
  if (c.exclude.length) parts.push(`−${c.exclude.length}`)
  if ([c.minTransmission, c.maxRelativeCost, c.maxClimateClass, c.maxStainClass].some((v) => v != null)) parts.push('limits')
  return parts.join(' · ')
}

/** Splits "N-BK7, SCHOTT:F2" into names. */
export function parseNames(text: string): string[] {
  return text
    .split(/[,;\n]/)
    .map((s) => s.trim())
    .filter(Boolean)
}

// MARK: Glass gaps

/** A glass gap the substitution may change: the stable id of the surface in
 *  front of the glass and the glass now there. */
export interface GlassSlot {
  id: number
  index: number
  material: string
  isModel: boolean
  /** A catalogue glass's melt offset text (`+0.0012/−0.8`), or ''. */
  offset: string
}

/** Every glass gap of a design: refracting surfaces followed by glass. */
export function glassSlots(system: System): GlassSlot[] {
  const out: GlassSlot[] = []
  const last = imageIndex(system)
  system.surfaces?.forEach((s, i) => {
    if (i <= 0 || i >= last) return
    if (s.shape?.type === 'coordinate_break' || s.interaction !== 'refract') return
    const t = s.material?.type
    if (!t || t === 'air' || t === 'vacuum') return
    out.push({
      id: s.id,
      index: i,
      material: materialText(s),
      isModel: t === 'model',
      offset: t === 'catalog' ? meltOffsetText(s.material) : '',
    })
  })
  return out
}

/** The design's glasses by surface id, a melt offset shown on the glass it leaves. */
export function startGlasses(system: System): Record<number, string> {
  const o: Record<number, string> = {}
  for (const s of glassSlots(system)) o[s.id] = s.material + (s.offset ? ' ' + s.offset : '')
  return o
}

/** The note on ticked glasses with melt offsets, or null. */
export function offsetNote(system: System, selected: readonly number[]): string | null {
  const slots = glassSlots(system).filter((s) => selected.includes(s.id) && s.offset)
  if (!slots.length) return null
  const names = slots.map((s) => `S${s.index} ${s.material} ${s.offset}`)
  const list = names.length < 3 ? names.join(' and ') : names.slice(0, -1).join(', ') + ', and ' + names.at(-1)
  const one = slots.length === 1
  return (
    `${list} ${one ? 'carries a melt offset' : 'carry melt offsets'}. A replacement glass is installed without ` +
    `${one ? 'it' : 'them'}, even the same glass: the offset belonged to the original melt.`
  )
}

const MATERIAL_PARAMS = new Set(['material_dn_d', 'material_dv_d', 'model_nd', 'model_vd', 'model_dpgf'])

/** Material parameters substitution may not vary: the offsets and model numbers
 *  of the glasses it replaces (the engine refuses them). */
export function isSubstitutedMaterialParam(p: ParamPath, selected: readonly number[]): boolean {
  return MATERIAL_PARAMS.has(p.param) && typeof p.surface === 'number' && selected.includes(p.surface)
}

/** A variable with equal bounds is pinned. */
export const isPinned = (v: VariableSpec) => v.lower != null && v.upper != null && v.lower === v.upper

/** The design variables a job varies: not pinned, not solve-controlled. */
export function freeVariables(system: System, variables: VariableSpec[]): VariableSpec[] {
  return variables.filter((v) => !isPinned(v) && !solveFor(system, v.param))
}

// MARK: Merit (the Optimize panel's, from the project's `merit` section)

/** The merit part of `optimize` / `glass_substitution` settings. */
export function meritJobSettings(merit: Json): JsonObject {
  const m = obj(merit)
  if (!m) return {}
  const s: JsonObject = { hold_efl: typeof m.hold_efl === 'boolean' ? m.hold_efl : true }
  if (m.mode === 'custom' && obj(m.spec)) s.merit = m.spec
  else if (obj(m.default_merit)) s.default_merit = m.default_merit
  return s
}

/** "Default merit: rms spot, focal length held." */
export function meritSummary(merit: Json): string {
  const m = obj(merit)
  const hold = m && typeof m.hold_efl === 'boolean' ? m.hold_efl : true
  let base: string
  if (m?.mode === 'custom') {
    const n = arr(obj(m.spec)?.operands).length
    base = `Custom merit, ${n} operand${n === 1 ? '' : 's'}`
  } else {
    const crit = obj(obj(m?.default_merit)?.criterion)
    const type = str(crit?.type) ?? 'rms_spot'
    const title: Record<string, string> = { rms_spot: 'RMS spot', wavefront: 'RMS wavefront', mtf: 'MTF' }
    base = `Default merit: ${title[type] ?? type.replace(/_/g, ' ')}`
  }
  return base + (hold ? ', focal length held.' : '.')
}

// MARK: Settings

export type GlassAlgorithm = 'tabu' | 'hammer'

export const ALGORITHMS: { value: GlassAlgorithm; title: string; detail: string }[] = [
  { value: 'tabu', title: 'Quick (tabu search)', detail: 'Tries glass combinations, each re-optimized from the current shape.' },
  { value: 'hammer', title: 'Thorough (global search)', detail: 'Hops over glasses and shape together to escape local minima; slower.' },
]

export interface GlassSearchSettings {
  /** Surface ids whose glass may change. */
  selected: number[]
  candidates: GlassCandidates
  /** Per-lens candidates replacing `candidates`. */
  overrides: Record<number, GlassCandidates>
  algorithm: GlassAlgorithm
  maxCombinations: number
  maxSeconds: number | null
  localIterations: number
  topN: number
  /** Re-optimise the design's variables (else every curvature and the back focus). */
  useDesignVariables: boolean
}

/** The `glass_substitution` settings. */
export function glassSubstitutionSettings(system: System, merit: Json, variables: VariableSpec[], g: GlassSearchSettings): JsonObject {
  const s: JsonObject = {
    algorithm: g.algorithm,
    max_combinations: g.maxCombinations,
    local_iterations: g.localIterations,
    top_n: g.topN,
  }
  if (g.maxSeconds != null && g.maxSeconds > 0) s.max_seconds = g.maxSeconds
  Object.assign(s, meritJobSettings(merit))
  const free = freeVariables(system, variables.filter((v) => !isSubstitutedMaterialParam(v.param, g.selected)))
  if (g.useDesignVariables && free.length) s.variables = free
  const order = glassSlots(system)
    .map((x) => x.id)
    .filter((id) => g.selected.includes(id))
  s.material_variables = order.map((id) => ({ surface: id, candidates: candidatesJson(g.overrides[id] ?? g.candidates) }))
  return s
}

// MARK: Result

export interface SubstitutedGlass {
  surface: number
  catalog: string
  name: string
  nd: number
  vd: number
  dpgf: number
}

/** The glass map's id of a catalogue glass. */
export const mapId = (catalog: string | null | undefined, name: string) => `${catalog ?? ''}/${name}`

export interface VariableChange {
  param: ParamPath | null
  start: number
  value: number
}

export interface GlassCombination {
  rank: number
  merit: number
  violation: number
  feasible: boolean
  glasses: SubstitutedGlass[]
  changes: VariableChange[]
  system: System | null
  /** The re-optimisation was cut short by Stop or the time limit. */
  cutShort: boolean
}

export interface SubstitutionResult {
  status: string
  stopped: boolean
  algorithm: string
  termination: string
  startingMerit: number
  combinations: number
  failed: number
  ranked: GlassCombination[]
  /** The best merit after each re-optimisation (running minimum). */
  history: number[]
  candidateCounts: Record<number, number>
}

export function runningMinimum(v: number[]): number[] {
  const out: number[] = []
  for (const x of v) out.push(out.length ? Math.min(x, out[out.length - 1]) : x)
  return out
}

function readGlass(g: Json): SubstitutedGlass {
  const o = obj(g) ?? {}
  return {
    surface: int(o.surface) ?? 0,
    catalog: str(o.catalog) ?? '',
    name: str(o.name) ?? '?',
    nd: nf(o.nd),
    vd: nf(o.vd),
    dpgf: nf(o.dpgf),
  }
}

/** Reads the `glass_substitution` result; null when it has no ranking. */
export function readSubstitution(json: JsonObject | null | undefined): SubstitutionResult | null {
  if (!json || !Array.isArray(json.results)) return null
  const stats = obj(json.statistics)
  const ranked = json.results.map((r, k): GlassCombination => {
    const o = obj(r) ?? {}
    return {
      rank: int(o.rank) ?? k + 1,
      merit: nf(o.merit),
      violation: nf(o.violation),
      feasible: typeof o.feasible === 'boolean' ? o.feasible : true,
      glasses: arr(o.glasses).map(readGlass),
      changes: arr(o.variables).map((v) => {
        const c = obj(v) ?? {}
        const p = obj(c.param)
        return { param: p && typeof p.param === 'string' ? (p as ParamPath) : null, start: nf(c.start), value: nf(c.value) }
      }),
      system: obj(o.system) && Array.isArray(obj(o.system)!.surfaces) ? (o.system as System) : null,
      cutShort: o.local_termination === 'cancelled',
    }
  })
  const counts: Record<number, number> = {}
  for (const c of arr(json.candidate_sets)) {
    const o = obj(c)
    const s = int(o?.surface)
    if (o && s != null) counts[s] = arr(o.glasses).length
  }
  return {
    status: str(json.status) ?? '',
    stopped: json.stopped === true,
    algorithm: str(json.algorithm) ?? '',
    termination: str(json.termination) ?? '',
    startingMerit: nf(json.starting_merit),
    combinations: int(stats?.combinations) ?? 0,
    failed: int(stats?.failed) ?? 0,
    ranked,
    history: runningMinimum(
      arr(json.history)
        .map((h) => nf(obj(h)?.best_merit))
        .filter(Number.isFinite),
    ),
    candidateCounts: counts,
  }
}

/** Variables that moved noticeably. */
export function movedChanges(c: GlassCombination): VariableChange[] {
  return c.changes.filter(
    (x) => Number.isFinite(x.value) && Number.isFinite(x.start) && Math.abs(x.value - x.start) > 1e-9 * Math.max(Math.abs(x.start), 1),
  )
}

/** "3 variables re-optimized · cut short". */
export function changeSummary(c: GlassCombination): string {
  const n = movedChanges(c).length
  let s = n === 0 ? 'shape unchanged' : `${n} variable${n === 1 ? '' : 's'} re-optimized`
  if (c.cutShort) s += ' · cut short'
  return s
}

/** The ranking's status line. */
export function statusText(r: SubstitutionResult, applied: boolean): string {
  let s = `${r.combinations} combination${r.combinations === 1 ? '' : 's'} re-optimized`
  if (r.failed > 0) s += `, ${r.failed} failed`
  s += '. '
  if (r.stopped) s += 'Stopped early: the ranking so far. '
  if (r.termination === 'time_limit') s += 'The time limit ended the search. '
  if (!applied) s += 'Select a combination to preview it in the window.'
  return s.trim()
}

/** Ringed glasses of a combination on the map: id → "S1, S3". */
export function combinationHighlights(c: GlassCombination, system: System): GlassHighlight[] {
  const by = new Map<string, string[]>()
  for (const g of c.glasses) {
    const i = system.surfaces.findIndex((s) => s.id === g.surface)
    const id = mapId(g.catalog, g.name)
    by.set(id, [...(by.get(id) ?? []), `S${i >= 0 ? i : '?'}`])
  }
  return [...by.entries()].map(([id, l]) => ({ id, label: l.join(', ') })).sort((a, b) => a.id.localeCompare(b.id))
}

// MARK: Live progress

export interface GlassProgress {
  combinations: number
  bestMerit: number
  glasses: string[]
}

/** Live progress from the engine's stage text
 *  `glass_substitution: combinations N; best merit M; glasses [...]`, or from a
 *  progress detail object with the same fields. */
export function parseGlassProgress(stage: string | null | undefined, detail?: Json | null): GlassProgress | null {
  const d = obj(detail ?? null)
  if (d && (typeof d.combinations === 'number' || d.best_merit != null)) {
    return {
      combinations: int(d.combinations) ?? 0,
      bestMerit: nf(d.best_merit),
      glasses: arr(d.glasses).map((g) => (typeof g === 'string' ? g : (str(obj(g)?.name) ?? '?'))),
    }
  }
  if (!stage || !stage.startsWith('glass_substitution:')) return null
  const body = stage.slice('glass_substitution:'.length)
  const info: GlassProgress = { combinations: 0, bestMerit: NaN, glasses: [] }
  const c = /combinations (\d+)/.exec(body)
  if (c) info.combinations = Number(c[1])
  const m = /best merit ([^\s;]+)/.exec(body)
  if (m) info.bestMerit = Number(m[1])
  const g = body.indexOf('glasses ')
  if (g >= 0) {
    try {
      const list = JSON.parse(body.slice(g + 'glasses '.length)) as Json
      info.glasses = arr(list).map((x) => (typeof x === 'string' ? x : (str(obj(x)?.name) ?? '?')))
    } catch {
      // A truncated list: no glasses.
    }
  }
  return info
}

// MARK: Glass map

export type GlassStatus = 'preferred' | 'standard' | 'obsolete' | 'special' | 'melt' | 'unknown'

export const STATUS_ORDER: GlassStatus[] = ['preferred', 'standard', 'special', 'melt', 'obsolete', 'unknown']

export function readStatus(v: Json | undefined): { status: GlassStatus; label: string } {
  if (typeof v === 'string') {
    const s = v.toLowerCase()
    if ((['preferred', 'standard', 'obsolete', 'special', 'melt'] as string[]).includes(s)) {
      return { status: s as GlassStatus, label: s.charAt(0).toUpperCase() + s.slice(1) }
    }
  }
  const o = obj(v ?? null)
  if (o && typeof o.other === 'number') return { status: 'unknown', label: `Status ${o.other}` }
  return { status: 'unknown', label: 'Unknown' }
}

export const statusTitle = (s: GlassStatus) => s.charAt(0).toUpperCase() + s.slice(1)

const SLOTS_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
const SLOTS_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
const STATUS_SLOT: Record<GlassStatus, number> = { preferred: 5, standard: 0, special: 6, melt: 3, obsolete: 7, unknown: -1 }

/** The status colour: a categorical slot, neutral grey when unknown. */
export function statusColor(s: GlassStatus, dark: boolean): string {
  const k = STATUS_SLOT[s]
  if (k < 0) return dark ? '#9a9a95' : '#8a8a85'
  return (dark ? SLOTS_DARK : SLOTS_LIGHT)[k]
}

/** Obsolete glasses are drawn hollow, so status does not rely on colour alone. */
export const statusHollow = (s: GlassStatus) => s === 'obsolete'

export interface GlassMapPoint {
  /** `catalog/name`; `model:name:S` and `offset:catalog/name:S` for design glasses. */
  id: string
  catalog: string | null
  name: string
  kind: 'catalog' | 'offset' | 'model'
  nd: number
  vd: number
  pgf: number
  status: GlassStatus
  statusLabel: string
  /** Surface indices whose following medium is this glass. */
  surfaces: number[]
  excludeSubstitution: boolean | null
  offset: { dn_d: number; dv_d: number } | null
  relativeCost: number | null
  climate: number | null
  stain: number | null
}

export interface GlassHighlight {
  id: string
  label: string
}

/** The design uses the glass. */
export const inDesign = (g: GlassMapPoint) => g.kind !== 'catalog' || g.surfaces.length > 0
export const surfaceLabel = (g: GlassMapPoint) => g.surfaces.map((s) => `S${s}`).join(', ')

/** A melt offset as the lens table shows it (`+0.0012/−0.8`), '' when zero. */
export const offsetText = (o: { dn_d: number; dv_d: number } | null): string => offsetCompactText(o)

export function displayName(g: GlassMapPoint): string {
  if (g.kind === 'model') return `${g.name} (model)`
  if (g.kind === 'offset') return `${g.name} ${offsetText(g.offset) || '(melt)'}`
  return g.name
}

/** The P_g,F normal line through K7 and F2. */
export const normalLine = (vd: number) => 0.6438 - 0.001682 * vd
export const deltaPgF = (g: GlassMapPoint) => g.pgf - normalLine(g.vd)
/** The id of the catalogue point an offset glass is shifted from. */
export const nominalId = (g: GlassMapPoint) => mapId(g.catalog, g.name)

function pointId(catalog: string | null, name: string, kind: GlassMapPoint['kind'], surfaces: number[]): string {
  if (kind === 'model') return `model:${name}:${surfaces[0] ?? -1}`
  if (kind === 'offset') return `offset:${catalog ?? ''}/${name}:${surfaces[0] ?? -1}`
  return mapId(catalog, name)
}

/** The glasses of a `glass_map` result (or its JSON), coordinates from the
 *  `coordinates` buffer when present. */
export function readGlassMap(result: AnalysisResult | JsonObject | null | undefined): GlassMapPoint[] {
  if (!result) return []
  const isResult = 'json' in result && 'buffers' in result && obj(result.json as Json) !== null
  const json = isResult ? (result as AnalysisResult).json : (result as JsonObject)
  const list = arr(json.glasses)
  let coords: ArrayLike<number> | null = null
  if (isResult) {
    const r = result as AnalysisResult
    const name = typeof json.coordinates === 'string' ? json.coordinates : 'coordinates'
    const b = r.buffers[name]
    if (b && !(b instanceof BigInt64Array) && !(b instanceof BigUint64Array) && b.length === list.length * 3) coords = b
  }
  return list.map((x, k): GlassMapPoint => {
    const g = obj(x) ?? {}
    const kind = g.kind === 'offset' || g.kind === 'model' ? g.kind : 'catalog'
    const name = str(g.name) ?? '?'
    const catalog = str(g.catalog)
    const surfaces = arr(g.surfaces).filter((s): s is number => typeof s === 'number')
    const c = (i: number, key: string) => (coords ? Number(coords[k * 3 + i]) : nf(g[key]))
    const st = kind === 'model' ? { status: 'unknown' as GlassStatus, label: 'Model' } : readStatus(g.status)
    const off = obj(g.offset)
    const num = (v: Json | undefined) => (typeof v === 'number' && Number.isFinite(v) ? v : null)
    return {
      id: pointId(catalog, name, kind, surfaces),
      catalog,
      name,
      kind,
      nd: c(0, 'nd'),
      vd: c(1, 'vd'),
      pgf: c(2, 'pgf'),
      status: st.status,
      statusLabel: st.label,
      surfaces,
      excludeSubstitution: typeof g.exclude_substitution === 'boolean' ? g.exclude_substitution : null,
      offset: kind === 'offset' ? { dn_d: nf(off?.dn_d ?? 0), dv_d: nf(off?.dv_d ?? 0) } : null,
      relativeCost: num(g.relative_cost),
      climate: num(g.climate_resistance),
      stain: num(g.stain_resistance),
    }
  })
}

/** The design's glasses, labelled with their surfaces. */
export function designHighlights(glasses: GlassMapPoint[]): GlassHighlight[] {
  return glasses.filter(inDesign).map((g) => ({ id: g.id, label: surfaceLabel(g) }))
}

export function mapCatalogs(glasses: GlassMapPoint[]): string[] {
  const seen = new Set<string>()
  for (const g of glasses) if (g.catalog) seen.add(g.catalog)
  return [...seen]
}

/** The statuses present among catalogue glasses, in legend order. */
export function mapStatuses(glasses: GlassMapPoint[]): GlassStatus[] {
  const present = new Set(glasses.filter((g) => g.kind === 'catalog').map((g) => g.status))
  return STATUS_ORDER.filter((s) => present.has(s))
}

/** Axis domain of a list with padding (x: V_d; y: n_d or P_g,F). */
export function domain(values: number[], minPad: number, fallback: [number, number]): [number, number] {
  const v = values.filter(Number.isFinite)
  if (!v.length) return fallback
  const lo = Math.min(...v), hi = Math.max(...v)
  const pad = Math.max((hi - lo) * 0.06, minPad)
  return [lo - pad, hi + pad]
}

// MARK: Glass data (`glass_info`)

export interface GlassInfo {
  name: string
  catalog: string
  nd: number
  ne: number
  vd: number
  pgf: number
  catalogNd: number
  catalogVd: number
  dnd: number
  dvd: number
  minUm: number
  maxUm: number
  formula: string | null
  hasThermalData: boolean
  cte: number | null
  density: number | null
  relativeCost: number | null
  comment: string | null
  temperatureC: number | null
  pressureAtm: number | null
  indices: { um: number; n: number; nCatalog: number; absorption: number; inRange: boolean }[]
  curve: [number, number][]
}

const FORMULAS: Record<string, string> = {
  schott: 'Schott', sellmeier: 'Sellmeier', sellmeier2: 'Sellmeier 2', herzberger: 'Herzberger',
  conrady: 'Conrady', extended: 'Extended', cauchy: 'Cauchy', tabulated: 'Tabulated',
}

export function formulaTitle(key: string): string {
  return FORMULAS[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export function readGlassInfo(r: AnalysisResult): GlassInfo | null {
  const j = r.json
  const name = str(j.name)
  if (!name || j.nd == null) return null
  const finite = (v: Json | undefined) => {
    const x = typeof v === 'number' ? v : NaN
    return Number.isFinite(x) ? x : null
  }
  const env = obj(j.environment)
  const off = obj(j.offset)
  const b = r.buffers.dispersion_curve
  const curve: [number, number][] = []
  if (b && !(b instanceof BigInt64Array) && !(b instanceof BigUint64Array)) {
    for (let i = 0; i + 1 < b.length; i += 2) curve.push([Number(b[i]), Number(b[i + 1])])
  }
  const formula = str(obj(j.dispersion)?.formula)
  const comment = str(j.comment)
  return {
    name,
    catalog: str(j.catalog) ?? '',
    nd: nf(j.nd),
    ne: nf(j.ne),
    vd: nf(j.vd),
    pgf: nf(j.pgf),
    catalogNd: nf(j.catalog_nd),
    catalogVd: nf(j.catalog_vd),
    dnd: nf(off?.dn_d ?? 0),
    dvd: nf(off?.dv_d ?? 0),
    minUm: nf(j.min_um),
    maxUm: nf(j.max_um),
    formula: formula ? formulaTitle(formula) : null,
    hasThermalData: j.has_thermal_data === true,
    cte: finite(j.cte),
    density: finite(j.density),
    relativeCost: finite(j.relative_cost),
    comment: comment && comment.trim() ? comment : null,
    temperatureC: finite(env?.temperature_c),
    pressureAtm: finite(env?.pressure_atm),
    indices: arr(j.indices).flatMap((x) => {
      const o = obj(x)
      if (!o || typeof o.wavelength_um !== 'number') return []
      return [{ um: o.wavelength_um, n: nf(o.n), nCatalog: nf(o.n_catalog), absorption: nf(o.absorption_per_mm), inRange: o.in_range !== false }]
    }),
    curve,
  }
}
