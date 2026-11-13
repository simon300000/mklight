// MOCK ENGINE — a synthetic EngineClient for UI development. Port of the Swift
// `MockEngine`: correctly shaped results that respond to edits, never an
// optical computation. Every result says so in run_info.
import {
  EngineError,
  type AnalysisDescriptor,
  type AnalysisJob,
  type AnalysisRequest,
  type AnalysisResult,
  type CatalogInfo,
  type EngineClient,
  type EngineInfo,
  type JobProgress,
  type JsonObject,
  type ModelSnapshot,
  type OpticalSystem,
  type RunInfo,
} from '../engine/types'
import descriptorsJson from './descriptors.json'
import { CATALOG, MOCK_WARNING, canonicalJson, parseSystem, sha256Hex, structuralIssues, type Obj } from './model'
import { MockOptics } from './optics'
import * as A from './analyses'
import * as W from './waves'

const VERSION = '0.1.0-mock'
const descriptors = descriptorsJson as unknown as AnalysisDescriptor[]

function engineInfo(): EngineInfo {
  const nav = (typeof navigator !== 'undefined' ? navigator : undefined) as
    | (Navigator & { gpu?: unknown })
    | undefined
  return {
    name: 'Mock engine (synthetic data)',
    version: VERSION,
    backends: ['cpu_reference'],
    threads: nav?.hardwareConcurrency ?? 1,
    isolated: (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated ?? false,
    webgpu: !!nav && 'gpu' in nav,
    isMock: true,
  }
}

/** Light schema check: enum, numeric bounds and basic types of provided keys. */
function checkSettings(d: AnalysisDescriptor, settings: JsonObject): string[] {
  const props = ((d.settings_schema as Obj).properties ?? {}) as Record<string, Obj>
  const known = new Set([...Object.keys(props), ...Object.keys(d.default_settings ?? {})])
  const errs: string[] = []
  for (const [k, v] of Object.entries(settings)) {
    if (!known.has(k)) {
      errs.push(`unknown setting '${k}'`)
      continue
    }
    const p = props[k]
    if (!p || v === null) continue
    const types: string[] = Array.isArray(p.type) ? p.type : p.type ? [p.type] : []
    if (typeof v === 'number') {
      if (types.length && !types.includes('number') && !types.includes('integer')) errs.push(`'${k}' must be ${types.join(' or ')}`)
      if (types.includes('integer') && !types.includes('number') && !Number.isInteger(v)) errs.push(`'${k}' must be an integer`)
      if (typeof p.minimum === 'number' && v < p.minimum) errs.push(`'${k}' must be at least ${p.minimum}`)
      if (typeof p.maximum === 'number' && v > p.maximum) errs.push(`'${k}' must be at most ${p.maximum}`)
    } else if (typeof v === 'string' && Array.isArray(p.enum) && !p.enum.includes(v)) {
      errs.push(`'${k}' must be one of ${p.enum.join(', ')}`)
    }
  }
  return errs
}

function compute(name: string, s: Obj, snap: ModelSnapshot): { out: A.Out; warnings: string[] } {
  const o = new MockOptics(parseSystem(snap.system))
  let out: A.Out
  switch (name) {
    case 'validate':
      out = A.validate(o)
      break
    case 'system_summary':
      out = A.systemSummary(o)
      break
    case 'first_order':
      out = A.firstOrder(o, s)
      break
    case 'layout':
      out = A.layout(o, s)
      break
    case 'spot_diagram':
      out = A.spotDiagram(o, s)
      break
    case 'ray_fan':
      out = A.fans(o, s, false)
      break
    case 'opd_fan':
      out = A.fans(o, s, true)
      break
    case 'field_curvature_distortion':
      out = A.fieldCurvature(o, s)
      break
    case 'mtf':
      out = W.mtf(o, s)
      break
    case 'psf':
      out = W.psf(o, s)
      break
    case 'wavefront_map':
      out = W.wavefrontMap(o, s)
      break
    case 'glass_info': {
      const g = A.glassInfo(s, CATALOG)
      if (!g) throw new EngineError('failed', `unknown glass '${String(s.name ?? '')}'`)
      out = g
      break
    }
    case 'solves':
      out = A.newOut({ outcomes: [], order: [], sweeps: 0, max_change: 0, warnings: [], solved_params: [], configuration: null })
      break
    default:
      out = A.newOut({ note: 'mock', analysis: name })
  }
  return { out, warnings: o.warnings }
}

export function createMockEngine(opts: { latencyMs?: number } = {}): EngineClient {
  const latency = opts.latencyMs ?? 300
  let nextId = 1
  const engine: EngineClient = {
    info: engineInfo(),

    async openSnapshot(system: OpticalSystem): Promise<ModelSnapshot> {
      const errors = structuralIssues(system).filter((i) => i.severity === 'error')
      if (errors.length) {
        const issues = errors.map((i) => (i.surface !== null && !i.message.includes(`surface ${i.surface}`) ? `surface ${i.surface}: ${i.message}` : i.message))
        const msg = issues.length === 1 ? issues[0] : `${issues[0]} (and ${issues.length - 1} more)`
        throw new EngineError('invalidModel', msg, issues)
      }
      const hash = await sha256Hex(canonicalJson(system))
      return { hash, system, libraryHash: 'mock' }
    },

    async analysisDescriptors() {
      return descriptors
    },

    submit(req: AnalysisRequest, snap: ModelSnapshot): AnalysisJob {
      const id = nextId++
      const listeners: ((p: JobProgress) => void)[] = []
      const timers: ReturnType<typeof setTimeout>[] = []
      let settled = false
      let finish: () => void = () => {}
      let fail: (e: unknown) => void = () => {}
      const result = new Promise<AnalysisResult>((resolve, reject) => {
        const run = () => {
          if (settled) return
          settled = true
          timers.forEach(clearTimeout)
          try {
            const started = Date.now()
            const t0 = typeof performance !== 'undefined' ? performance.now() : started
            const d = descriptors.find((x) => x.name === req.analysis)
            if (!d) throw new EngineError('invalidSettings', `unknown analysis '${req.analysis}'`)
            const given = req.settings ?? {}
            const errs = checkSettings(d, given)
            if (errs.length) throw new EngineError('invalidSettings', errs[0], errs)
            const settings: JsonObject = { ...(d.default_settings ?? {}), ...given }
            const { out, warnings } = compute(req.analysis, settings as Obj, snap)
            const dur = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0
            const runInfo: RunInfo = {
              engine_version: VERSION,
              snapshot_hash: snap.hash,
              analysis: req.analysis,
              settings,
              backend: 'mock',
              precision: 'f64_reference',
              seed: typeof settings.seed === 'number' ? settings.seed : null,
              started_unix_ms: started,
              duration_ms: dur,
              tier: 'design',
              warnings: [MOCK_WARNING, ...warnings],
            }
            for (const l of listeners) l({ progress: 1, stage: 'Done' })
            resolve({
              json: out.json,
              buffers: out.buffers,
              shapes: out.shapes,
              runInfo,
              envelope: {
                backend: { requested: 'auto', used: 'mock', precision: 'f64_reference' },
                library_hash: 'mock',
                result_version: d.result_version ?? 1,
              },
            })
          } catch (e) {
            reject(e)
          }
        }
        finish = run
        fail = (e) => {
          if (settled) return
          settled = true
          timers.forEach(clearTimeout)
          reject(e)
        }
      })
      result.catch(() => {}) // callers observe rejection themselves

      const stages: [number, string][] = [
        [0.1, 'Resolving system'],
        [0.4, 'Tracing rays'],
        [0.8, 'Collecting results'],
      ]
      stages.forEach(([p, stage]) => {
        timers.push(
          setTimeout(() => {
            if (!settled) for (const l of listeners) l({ progress: p, stage })
          }, latency * p),
        )
      })
      timers.push(setTimeout(() => finish(), latency))

      return {
        id,
        onProgress: (cb) => {
          listeners.push(cb)
        },
        result,
        cancel: () => fail(new EngineError('cancelled', 'Cancelled')),
        stop: () => finish(),
      }
    },

    async materialNames() {
      return Object.keys(CATALOG).sort()
    },

    async catalogs(): Promise<CatalogInfo[]> {
      return [{ name: 'MOCK', source: 'builtin', glass_count: Object.keys(CATALOG).length }]
    },

    async loadCatalog() {
      throw new EngineError('unsupported', 'The mock engine cannot load glass catalogs')
    },

    async importPrescription() {
      throw new EngineError('unsupported', 'The mock engine cannot import lens prescription files')
    },

    async applySolves(_settings, snap) {
      return { report: { outcomes: [], warnings: [] }, snapshot: snap }
    },

    makeDocumentEngine() {
      return engine
    },
  }
  return engine
}
