// What a result says about its run, in words (`RunWarnings`, `RunRoute`,
// `RunDetails.remedy` of the macOS app): warnings grouped by message with the
// fields and wavelengths numbered from 1, the route of an automatic request
// (`backend.cost_decision`, docs/RESULTS.md §5), the tier of a non-design run,
// the backend attempts that did not compute the result
// (`run_info.failed_attempts`) and a one-click remedy for coarse sampling.

import type { AnalysisResult, Json, JsonObject } from '@/engine/types'
import { engineText, t } from '@/i18n'
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
  const text = w.trim()
  const colon = text.indexOf(': ')
  if (colon < 0) return [null, text]
  const head = text.slice(0, colon)
  const rest = text.slice(colon + 2)
  if (head.startsWith('field (')) return [t('analysis.runInfo.fieldXY', { xy: head.slice(6).replace(/-/g, '−') }), rest]
  const scope: string[] = []
  for (const p of head.split(',').map((x) => x.trim())) {
    const words = p.split(' ')
    const k = Number(words[1])
    if (words.length !== 2 || !/^\d+$/.test(words[1])) return [null, text]
    if (words[0] === 'field') scope.push(t('common.fieldN', { n: k + 1 }))
    else if (words[0] === 'wavelength') scope.push(`λ${k + 1}`)
    else if (words[0] === 'surface') scope.push(t('common.surfaceN', { n: k }))
    else return [null, text]
  }
  return scope.length ? [scope.join(' · '), rest] : [null, text]
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
      groups.set(key, { message: engineText(message.charAt(0).toUpperCase() + message.slice(1)), scopes: scope ? [scope] : [], remedy: remedyKind(message) })
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
  const times = predictedTimes(d)
  const pred = times ? t('analysis.runInfo.predicted', { gpu: durationText(times.gpu), cpu: durationText(times.cpu) }) : ''
  const R = (k: string) => t(`analysis.runInfo.${k}` as never)
  let line: string
  if (d.decision === 'qualified_metal' && rule === 'cost model') line = R('routeGpu') + pred
  else if (d.decision === 'cpu_reference' && rule === 'cost model')
    line = (times && times.gpu < times.cpu ? R('routeCpuSmallGain') : R('routeCpu')) + pred
  else if (d.decision === 'cpu_reference' && rule === 'accuracy exclusion') line = R('routeCpuAccuracy')
  else if (d.decision === 'cpu_reference' && rule === 'no device share') line = R('routeCpuNoShare')
  else if (d.decision === 'cpu_reference' && rule === 'known failure') line = R('routeCpuKnownFailure')
  else if (d.decision === 'qualified_metal' && rule === null)
    line =
      d.reason === 'no cost calibration for this device'
        ? R('routeGpuNoDevice')
        : d.reason === 'no calibration for this analysis'
          ? R('routeGpuNoAnalysis')
          : R('routeGpuNoPrediction')
  else return null
  return d.repeated === true ? t('analysis.runInfo.repeated', { line }) : line
}

/** The tier of a run that was not design tier; null for design tier. */
export const tierText = (tier: Json | undefined): string | null =>
  tier === 'preview' ? t('analysis.runInfo.tierPreview') : typeof tier === 'string' && tier && tier !== 'design' ? tier : null

export interface Attempt {
  title: string
  reason: string
}

const backendName = (id: string) => (id.startsWith('gpu') ? 'GPU' : id === 'cpu_reference' ? t('analysis.runInfo.cpuReference') : id)

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
        ? took
          ? t('analysis.runInfo.attemptFailedAfter', { name, took })
          : t('analysis.runInfo.attemptFailed', { name })
        : outcome === 'skipped'
          ? t('analysis.runInfo.attemptSkipped', { name })
          : outcome === 'reference_completion'
            ? took
              ? t('analysis.runInfo.attemptCompletedIn', { name, took })
              : t('analysis.runInfo.attemptCompleted', { name })
            : t('analysis.runInfo.attemptNotUsed', { name, outcome: outcome.replace(/_/g, ' ') })
    return { title, reason: engineText(String(a.reason ?? '')) }
  })
}

/** "CPU reference · f64", "Metal (Apple M2 Pro) · f32". */
export function backendSummary(r: AnalysisResult): string {
  const be = obj(r.envelope?.backend)
  const used = String(be?.used ?? r.runInfo.backend ?? '')
  const name = used === 'cpu_reference' ? t('analysis.runInfo.cpuReference') : used.startsWith('gpu:') ? used.charAt(4).toUpperCase() + used.slice(5) : used
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
  const fallback = typeof be?.fallback_reason === 'string' && be.fallback_reason ? engineText(be.fallback_reason) : null
  const precision = String(be?.precision ?? ri.precision ?? '')
  const rows: [string, string][] = [
    [t('analysis.runInfo.computedOn'), backendSummary(r)],
    [t('analysis.runInfo.requested'), typeof requested === 'string' ? requested : ''],
    [t('analysis.runInfo.route'), routeLine(requested, be?.cost_decision) ?? ''],
    [t('analysis.runInfo.ranOnCpu'), fallback ?? ''],
    [t('analysis.runInfo.precision'), precision === 'f64_reference' ? t('analysis.runInfo.f64Reference') : precision === 'f32_gpu' ? t('analysis.runInfo.f32Qualified') : precision],
    [t('analysis.runInfo.tier'), tierText(ri.tier) ?? ''],
    [t('analysis.runInfo.seed'), ri.seed == null ? '' : String(ri.seed)],
    [t('analysis.runInfo.snapshot'), String(ri.snapshot_hash ?? '').slice(0, 12)],
    [t('analysis.runInfo.engine'), String(ri.engine_version ?? '')],
    [t('analysis.runInfo.time'), ri.duration_ms != null ? `${Number(ri.duration_ms).toFixed(1)} ms` : ''],
  ]
  const raw = ri.warnings ?? []
  return { rows: rows.filter(([, v]) => v), attempts: attempts(ri.failed_attempts), groups: groupWarnings(raw), raw: raw.map((w) => engineText(w)), fallback }
}

/** The badge's tooltip: the fallback and up to three warning lines. */
export function badgeHelp(f: RunFacts): string {
  const parts: string[] = []
  if (f.fallback) parts.push(t('analysis.runInfo.ranOnCpuColon', { reason: f.fallback }))
  parts.push(...f.groups.slice(0, 3).map(warningLine))
  if (f.groups.length > 3) parts.push(t('analysis.runInfo.moreWarnings', { count: f.groups.length - 3 }))
  return parts.length ? parts.join('\n') + '\n' + t('analysis.runInfo.clickForDetails') : t('analysis.runInfo.howComputed')
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
    const title = t('analysis.runInfo.pupilTitle', { grid })
    const help = t('analysis.runInfo.pupilHelp', { grid })
    const np = { ...pupil, grid: 0 }
    return optics ? { title, help, key: 'optics', value: { ...optics, pupil: np } } : { title, help, key: 'pupil', value: np }
  }
  if (optics) {
    const size = int(optics.psf_size)
    if (size == null || size <= 0) return null
    return {
      title: t('analysis.runInfo.psfTitle', { size }),
      help: t('analysis.runInfo.psfHelp', { size }),
      key: 'optics',
      value: { ...optics, psf_size: 0 },
    }
  }
  const size = int(settings.grid_size)
  if (size == null || size <= 0) return null
  return {
    title: t('analysis.runInfo.gridTitle', { size }),
    help: t('analysis.runInfo.gridHelp', { size }),
    key: 'grid_size',
    value: 0,
  }
}
