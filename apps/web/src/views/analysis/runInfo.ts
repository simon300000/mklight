// What a result says about its run, in words (`RunWarnings`, `RunRoute`,
// `RunDetails.remedy` of the macOS app): warnings grouped by message with the
// fields and wavelengths numbered from 1, the route of an automatic request
// (`backend.cost_decision`, docs/RESULTS.md §5), the tier of a non-design run,
// the backend attempts that did not compute the result
// (`run_info.failed_attempts`) and a one-click remedy for coarse sampling.

import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { nf } from '@/model/format'

const obj = (v: Json | undefined): JsonObject | null => (v !== null && typeof v === 'object' && !Array.isArray(v) ? v : null)

// MARK: Warnings

export type RemedyKind = 'pupilSampling' | 'imageGrid'

export interface WarningGroup {
  /** The first occurrence's message, capitalised, without its scope. */
  message: string
  /** `Field 1`, `Field 3 · λ2`, `field (1, −1)` (empty: the whole run). */
  scopes: string[]
  remedy: RemedyKind | null
}

/** One line for tooltips and screen readers: "Field 1, Field 3: message". */
export const warningLine = (g: WarningGroup) => (g.scopes.length ? `${g.scopes.join(', ')}: ${g.message}` : g.message)

/** `field 2, wavelength 1: text` → [`Field 3 · λ2`, `text`]; also `field (-1.0, 1.0): …`. */
export function splitWarning(w: string): [string | null, string] {
  const t = w.trim()
  const colon = t.indexOf(': ')
  if (colon < 0) return [null, t]
  const head = t.slice(0, colon)
  const rest = t.slice(colon + 2)
  if (head.startsWith('field (')) return ['field ' + head.slice(6).replace(/-/g, '−'), rest]
  const scope: string[] = []
  for (const p of head.split(',').map((x) => x.trim())) {
    const words = p.split(' ')
    const k = Number(words[1])
    if (words.length !== 2 || !/^\d+$/.test(words[1])) return [null, t]
    if (words[0] === 'field') scope.push(`Field ${k + 1}`)
    else if (words[0] === 'wavelength') scope.push(`λ${k + 1}`)
    else if (words[0] === 'surface') scope.push(`Surface ${k}`)
    else return [null, t]
  }
  return scope.length ? [scope.join(' · '), rest] : [null, t]
}

/** The message with numbers masked, so "0.58 waves" and "0.72 waves" group. */
export const maskNumbers = (s: string) => s.replace(/[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?/g, '#')

function remedyKind(s: string): RemedyKind | null {
  const l = s.toLowerCase()
  if (l.includes('increase the pupil sampling')) return 'pupilSampling'
  if (l.includes('enlarge the grid')) return 'imageGrid'
  return null
}

/** Groups warnings by message (numbers masked), in first-seen order. */
export function groupWarnings(warnings: readonly string[]): WarningGroup[] {
  const groups = new Map<string, WarningGroup>()
  for (const w of warnings) {
    const [scope, message] = splitWarning(w)
    const key = maskNumbers(message)
    const g = groups.get(key)
    if (g) {
      if (scope && !g.scopes.includes(scope)) g.scopes.push(scope)
    } else {
      groups.set(key, { message: message.charAt(0).toUpperCase() + message.slice(1), scopes: scope ? [scope] : [], remedy: remedyKind(message) })
    }
  }
  return [...groups.values()]
}

// MARK: Route, tier, attempts

/** Whole milliseconds below a second, then seconds; `<1 ms` for nothing. */
export function durationText(ms: number): string {
  if (ms < 0.5) return '<1 ms'
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

function predictedTimes(d: JsonObject): { gpu: number; cpu: number } | null {
  const p = obj(d.predicted_ms)
  if (!p) return null
  const gpu = nf(p.metal), cpu = nf(p.cpu)
  return Number.isFinite(gpu) && Number.isFinite(cpu) && gpu >= 0 && cpu >= 0 ? { gpu, cpu } : null
}

/** The route of an `auto` request in one line; null for explicit requests,
 *  runs without a decision and decisions this app does not know. */
export function routeLine(requested: Json | undefined, decision: Json | undefined): string | null {
  const asked = typeof requested === 'string' ? requested.replace(/^default:/, '') : ''
  const d = obj(decision)
  if (asked !== 'auto' || !d || typeof d.decision !== 'string') return null
  const rule = typeof d.rule === 'string' ? d.rule : null
  const t = predictedTimes(d)
  const pred = t ? ` (GPU predicted ${durationText(t.gpu)}, CPU ${durationText(t.cpu)})` : ''
  let line: string
  if (d.decision === 'qualified_metal' && rule === 'cost model') line = 'GPU by predicted time' + pred
  else if (d.decision === 'cpu_reference' && rule === 'cost model')
    line = (t && t.gpu < t.cpu ? "CPU: the GPU's predicted gain is too small" : 'CPU by predicted time') + pred
  else if (d.decision === 'cpu_reference' && rule === 'accuracy exclusion') line = 'CPU: this analysis is not accurate enough on the GPU'
  else if (d.decision === 'cpu_reference' && rule === 'no device share') line = 'CPU: the GPU takes no part of a run of this size and shape'
  else if (d.decision === 'cpu_reference' && rule === 'known failure') line = 'CPU: this request already failed on the GPU'
  else if (d.decision === 'qualified_metal' && rule === null)
    line =
      d.reason === 'no cost calibration for this device'
        ? 'GPU: no timing data for this device, so no prediction'
        : d.reason === 'no calibration for this analysis'
          ? 'GPU: no timing data for this analysis, so no prediction'
          : 'GPU, without a prediction'
  else return null
  return d.repeated === true ? line + '; kept from the first identical request' : line
}

/** The tier of a run that was not design tier; null for design tier. */
export const tierText = (tier: Json | undefined): string | null =>
  tier === 'preview' ? 'Preview: cheaper sampling, not for final decisions' : typeof tier === 'string' && tier && tier !== 'design' ? tier : null

export interface Attempt {
  title: string
  reason: string
}

const backendName = (id: string) => (id.startsWith('gpu') ? 'GPU' : id === 'cpu_reference' ? 'CPU reference' : id)

/** One entry per backend attempt that did not compute the result. */
export function attempts(list: Json | undefined): Attempt[] {
  if (!Array.isArray(list)) return []
  return list.map(obj).filter((a): a is JsonObject => a !== null).map((a) => {
    const name = backendName(String(a.backend ?? ''))
    const ms = nf(a.elapsed_ms)
    const took = ms > 0 ? durationText(ms) : null
    const outcome = String(a.outcome ?? '')
    const title =
      outcome === 'failed'
        ? `${name} failed${took ? ` after ${took}` : ''}; completed on the CPU`
        : outcome === 'skipped'
          ? `${name} not tried: the same input already failed on it`
          : outcome === 'reference_completion'
            ? `${name} path completed the result on the CPU${took ? ` (${took})` : ''}`
            : `${name} not used (${outcome.replace(/_/g, ' ')})`
    return { title, reason: String(a.reason ?? '') }
  })
}

/** "CPU reference · f64", "Metal (Apple M2 Pro) · f32". */
export function backendSummary(r: AnalysisResult): string {
  const be = obj(r.envelope?.backend)
  const used = String(be?.used ?? r.runInfo.backend ?? '')
  const name = used === 'cpu_reference' ? 'CPU reference' : used.startsWith('gpu:') ? used.charAt(4).toUpperCase() + used.slice(5) : used
  const dev = typeof be?.device === 'string' && be.device ? ` (${be.device})` : ''
  const p = String(be?.precision ?? r.runInfo.precision ?? '')
  const prec = p === 'f64_reference' ? 'f64' : p === 'f32_gpu' ? 'f32' : p
  return prec ? `${name}${dev} · ${prec}` : name + dev
}

export interface RunFacts {
  rows: [string, string][]
  attempts: Attempt[]
  groups: WarningGroup[]
  /** Raw engine messages. */
  raw: string[]
  fallback: string | null
}

/** Everything the run details show, rows that carry nothing left out. */
export function runFacts(r: AnalysisResult): RunFacts {
  const ri = r.runInfo
  const be = obj(r.envelope?.backend)
  const requested = be?.requested
  const fallback = typeof be?.fallback_reason === 'string' && be.fallback_reason ? be.fallback_reason : null
  const precision = String(be?.precision ?? ri.precision ?? '')
  const rows: [string, string][] = [
    ['Computed on', backendSummary(r)],
    ['Requested', typeof requested === 'string' ? requested : ''],
    ['Route', routeLine(requested, be?.cost_decision) ?? ''],
    ['Ran on the CPU', fallback ?? ''],
    ['Precision', precision === 'f64_reference' ? 'f64 reference' : precision === 'f32_gpu' ? 'f32 (GPU, qualified)' : precision],
    ['Tier', tierText(ri.tier) ?? ''],
    ['Seed', ri.seed == null ? '' : String(ri.seed)],
    ['Snapshot', String(ri.snapshot_hash ?? '').slice(0, 12)],
    ['Engine', String(ri.engine_version ?? '')],
    ['Time', ri.duration_ms != null ? `${Number(ri.duration_ms).toFixed(1)} ms` : ''],
  ]
  const raw = ri.warnings ?? []
  return { rows: rows.filter(([, v]) => v), attempts: attempts(ri.failed_attempts), groups: groupWarnings(raw), raw, fallback }
}

/** The badge's tooltip: the fallback and up to three warning lines. */
export function badgeHelp(f: RunFacts): string {
  const parts: string[] = []
  if (f.fallback) parts.push('Ran on the CPU: ' + f.fallback)
  parts.push(...f.groups.slice(0, 3).map(warningLine))
  if (f.groups.length > 3) parts.push(`… ${f.groups.length - 3} more`)
  return parts.length ? parts.join('\n') + '\nClick for details.' : 'How this result was computed'
}

// MARK: Remedies

export interface Remedy {
  title: string
  help: string
  /** The top-level setting to replace, and its new value. */
  key: string
  value: Json
}

const int = (v: Json | undefined): number | null => (typeof v === 'number' && Number.isInteger(v) ? v : null)

/** The setting change that answers a warning, given the pane's effective
 *  settings: an explicit coarse pupil grid, image grid or PSF window goes back
 *  to the engine's automatic choice. Automatic settings that still warn have
 *  reached the engine's cap, so nothing is offered. */
export function remedyFor(kind: RemedyKind | null, settings: JsonObject): Remedy | null {
  if (!kind) return null
  const optics = settings.pupil === undefined && settings.grid_size === undefined ? obj(settings.optics) : null
  if (kind === 'pupilSampling') {
    const pupil = obj(optics ? optics.pupil : settings.pupil)
    const grid = int(pupil?.grid)
    if (!pupil || grid == null || grid <= 0) return null
    const title = `Sample the Pupil Automatically (grid ${grid} → auto)`
    const help = `The pupil grid is set to ${grid}. Automatic sampling picks a grid fine enough for the wavefront's slope (64 to 1024), then runs again.`
    const np = { ...pupil, grid: 0 }
    return optics ? { title, help, key: 'optics', value: { ...optics, pupil: np } } : { title, help, key: 'pupil', value: np }
  }
  if (optics) {
    const size = int(optics.psf_size)
    if (size == null || size <= 0) return null
    return {
      title: `Size the PSF Window Automatically (${size} → auto)`,
      help: `The PSF window is set to ${size} samples. The automatic window holds the blur and its diffraction wings, then runs again.`,
      key: 'optics',
      value: { ...optics, psf_size: 0 },
    }
  }
  const size = int(settings.grid_size)
  if (size == null || size <= 0) return null
  return {
    title: `Size the Image Grid Automatically (${size} → auto)`,
    help: `The image grid is set to ${size}. The automatic grid holds the blur and its diffraction wings, then runs again.`,
    key: 'grid_size',
    value: 0,
  }
}
