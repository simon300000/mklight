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
import { BUILTIN_CATALOG, CATALOG, MOCK_WARNING, canonicalJson, parseSystem, sha256Hex, structuralIssues, type Obj } from './model'
import { MockOptics } from './optics'
import * as A from './analyses'
import * as W from './waves'
import * as W2 from './waves2'
import * as G from './geo'
import * as R from './radio'
import * as J from './jobs'
import * as S from './solves'
import { importPrescription as importFile, loadAgf } from './importers'
import { clone } from './params'

const VERSION = '0.1.0-mock'

const ENUM_APERTURE = ['entrance_pupil_diameter', 'image_space_f_number', 'object_space_na', 'paraxial_working_f_number', 'float_by_stop_size', null]
const ENUM_FIELD = ['object_angle', 'object_height', 'paraxial_image_height', 'real_image_height', null]

const CONVERT_SPEC: AnalysisDescriptor = {
  name: 'convert_spec',
  title: 'Convert field and aperture types',
  category: 'model',
  description:
    "The system's fields in another field kind and its aperture in another aperture type, each giving the same system: every field keeps its chief ray, and the aperture keeps the paraxial marginal ray.",
  available: true,
  unavailable_reason: null,
  settings_schema: {
    type: 'object',
    properties: {
      aperture: { type: ['string', 'null'], enum: ENUM_APERTURE, default: null, description: 'Aperture type to convert to; null skips.' },
      field_kind: { type: ['string', 'null'], enum: ENUM_FIELD, default: null, description: 'Field kind to convert the fields to; null skips.' },
    },
  },
  default_settings: { aperture: null, field_kind: null },
  result_version: 1,
  gpu_path: false,
  stoppable: false,
}

/** The descriptors of the real engine, plus convert_spec; mock-sized optimiser budgets. */
const descriptors: AnalysisDescriptor[] = [
  ...(descriptorsJson as unknown as AnalysisDescriptor[]).map((d) => {
    if (d.name !== 'optimize' && d.name !== 'glass_substitution') return d
    const copy = clone(d) as AnalysisDescriptor
    const props = ((copy.settings_schema as Obj).properties ?? {}) as Record<string, Obj>
    if (copy.default_settings && typeof copy.default_settings.max_iterations === 'number') copy.default_settings.max_iterations = 30
    if (props.max_iterations) props.max_iterations.default = 30
    return copy
  }),
  CONVERT_SPEC,
]

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

const SYNC: Record<string, (o: MockOptics, s: Obj, system: Obj) => A.Out> = {
  validate: (o) => A.validate(o),
  system_summary: (o) => A.systemSummary(o),
  first_order: (o, s) => A.firstOrder(o, s),
  layout: (o, s) => A.layout(o, s),
  spot_diagram: (o, s) => A.spotDiagram(o, s),
  ray_fan: (o, s) => A.fans(o, s, false),
  opd_fan: (o, s) => A.fans(o, s, true),
  field_curvature_distortion: (o, s) => A.fieldCurvature(o, s),
  mtf: (o, s) => W.mtf(o, s),
  psf: (o, s) => W.psf(o, s),
  wavefront_map: (o, s) => W.wavefrontMap(o, s),
  line_spread: (o, s) => W2.lineSpread(o, s),
  mtf_vs_field: (o, s) => W2.mtfVsField(o, s),
  through_focus_mtf: (o, s) => W2.throughFocusMtf(o, s),
  diffraction_encircled_energy: (o, s) => W2.diffractionEncircledEnergy(o, s),
  glass_info: (o, s) => R.glassInfo(o, s),
  glass_map: (o, s) => R.glassMap(o, s),
  seidel: (o, s) => G.seidel(o, s),
  chromatic_focal_shift: (o, s) => G.chromaticFocalShift(o, s),
  longitudinal_aberration: (o, s) => G.longitudinalAberration(o, s),
  lateral_colour: (o, s) => G.lateralColour(o, s),
  encircled_energy: (o, s) => G.encircledEnergy(o, s),
  geometric_mtf: (o, s) => G.geometricMtf(o, s),
  best_focus: (o, s) => G.bestFocus(o, s),
  ray_grid: (o, s) => G.rayGrid(o, s),
  footprint: (o, s) => G.footprint(o, s),
  grid_distortion: (o, s) => G.gridDistortion(o, s),
  relative_illumination: (o, s) => G.relativeIllumination(o, s),
  relative_illumination_map: (o, s) => G.relativeIlluminationMap(o, s),
  pupil_aberration: (o, s) => G.pupilAberration(o, s),
  chief_ray_angle: (o, s) => G.chiefRayAngle(o, s),
  system_transmission: (o, s) => R.systemTransmission(o, s),
  polarization_pupil_map: (o, s) => R.polarizationPupilMap(o, s),
  ghost_paths: (o, s) => R.ghostPaths(o, s),
  ghost_irradiance: (o, s) => R.ghostIrradiance(o, s),
  stray_light: (o, s) => R.strayLight(o, s),
  image_simulation: (o, s) => R.imageSimulation(o, s),
  rng_stream: (_o, s) => R.rngStream(s),
  solves: (_o, s, system) => S.solves(system, s),
  convert_spec: (_o, s, system) => S.convertSpec(system, s),
  merit_evaluation: (_o, s, system) => J.meritEvaluation(s, system),
}

function compute(name: string, s: Obj, snap: ModelSnapshot): { out: A.Out; warnings: string[] } {
  const o = new MockOptics(parseSystem(snap.system))
  const fn = SYNC[name]
  const out = fn ? fn(o, s, snap.system as Obj) : A.newOut({ note: 'mock', analysis: name })
  return { out, warnings: o.warnings }
}

const ASYNC: Record<string, (s: Obj, system: Obj, ctx: J.JobCtx) => Promise<A.Out>> = {
  optimize: J.optimize,
  glass_substitution: J.glassSubstitution,
  tolerance: J.tolerance,
}

export function createMockEngine(opts: { latencyMs?: number } = {}): EngineClient {
  const latency = opts.latencyMs ?? 300
  let nextId = 1
  const userCatalogs: CatalogInfo[] = []
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
      let stopFlag = false
      let cancelled = false
      let finish: () => void = () => {}
      let fail: (e: unknown) => void = () => {}
      const emit = (p: JobProgress) => {
        if (!settled) for (const l of listeners) l(p)
      }
      const started = Date.now()
      const t0 = typeof performance !== 'undefined' ? performance.now() : started
      const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())

      /** Validates the request; returns its descriptor and complete settings. */
      const prepare = (): { d: AnalysisDescriptor; settings: JsonObject } => {
        const d = descriptors.find((x) => x.name === req.analysis)
        if (!d) throw new EngineError('invalidSettings', `unknown analysis '${req.analysis}'`)
        const given = req.settings ?? {}
        const errs = checkSettings(d, given)
        if (errs.length) throw new EngineError('invalidSettings', errs[0], errs)
        return { d, settings: { ...(d.default_settings ?? {}), ...given } }
      }
      const pack = (d: AnalysisDescriptor, settings: JsonObject, out: A.Out, warnings: string[], stopped: boolean): AnalysisResult => {
        const runInfo: RunInfo = {
          engine_version: VERSION,
          snapshot_hash: snap.hash,
          analysis: req.analysis,
          settings,
          backend: 'mock',
          precision: 'f64_reference',
          seed: typeof settings.seed === 'number' ? settings.seed : null,
          started_unix_ms: started,
          duration_ms: now() - t0,
          tier: 'design',
          warnings: [MOCK_WARNING, ...warnings],
        }
        void stopped
        return {
          json: out.json,
          buffers: out.buffers,
          shapes: out.shapes,
          runInfo,
          envelope: {
            backend: { requested: 'auto', used: 'mock', precision: 'f64_reference' },
            library_hash: 'mock',
            result_version: d.result_version ?? 1,
          },
        }
      }

      let result: Promise<AnalysisResult>
      if (ASYNC[req.analysis]) {
        // A design job: runs as a real search in small steps, stoppable.
        result = new Promise<AnalysisResult>((resolve, reject) => {
          fail = (e) => {
            if (settled) return
            settled = true
            cancelled = true
            reject(e)
          }
          void (async () => {
            try {
              const { d, settings } = prepare()
              const optics = new MockOptics(parseSystem(snap.system))
              const ctx: J.JobCtx = {
                report: (progress, stage, detail) => emit({ progress, stage, detail }),
                stopped: () => stopFlag,
                check: () => {
                  if (cancelled) throw new EngineError('cancelled', 'Cancelled')
                },
                tick: () => new Promise<void>((r) => setTimeout(r, latency > 0 ? 1 : 0)),
                elapsedMs: () => now() - t0,
                budgetMs: 1800,
              }
              emit({ progress: 0, stage: 'Starting' })
              await ctx.tick()
              ctx.check()
              const out = await ASYNC[req.analysis](settings as Obj, clone(snap.system) as Obj, ctx)
              ctx.check()
              if (settled) return
              settled = true
              const stopped = out.json.stopped === true
              emit({ progress: 1, stage: stopped ? 'Stopped' : 'Done' })
              resolve(pack(d, settings, out, optics.warnings, stopped))
            } catch (e) {
              if (!settled) {
                settled = true
                reject(e)
              }
            }
          })()
        })
      } else {
        result = new Promise<AnalysisResult>((resolve, reject) => {
          const run = () => {
            if (settled) return
            settled = true
            timers.forEach(clearTimeout)
            try {
              const { d, settings } = prepare()
              const { out, warnings } = compute(req.analysis, settings as Obj, snap)
              emit({ progress: 1, stage: 'Done' })
              settled = true
              resolve(pack(d, settings, out, warnings, false))
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
        const stages: [number, string][] = [
          [0.1, 'Resolving system'],
          [0.4, 'Tracing rays'],
          [0.8, 'Collecting results'],
        ]
        stages.forEach(([p, stage]) => {
          timers.push(setTimeout(() => emit({ progress: p, stage }), latency * p))
        })
        timers.push(setTimeout(() => finish(), latency))
      }
      result.catch(() => {}) // callers observe rejection themselves

      return {
        id,
        onProgress: (cb) => {
          listeners.push(cb)
        },
        result,
        cancel: () => fail(new EngineError('cancelled', 'Cancelled')),
        stop: () => {
          stopFlag = true
          finish()
        },
      }
    },

    async materialNames() {
      return Object.keys(CATALOG).sort()
    },

    async catalogs(): Promise<CatalogInfo[]> {
      const builtin = Object.values(CATALOG).filter((g) => !g.catalog || g.catalog === BUILTIN_CATALOG).length
      return [{ name: BUILTIN_CATALOG, source: 'builtin', glass_count: builtin }, ...userCatalogs]
    },

    async loadCatalog(fileName: string, text: string): Promise<CatalogInfo> {
      const info = loadAgf(fileName, text)
      const k = userCatalogs.findIndex((c) => c.name === info.name)
      if (k >= 0) userCatalogs[k] = info
      else userCatalogs.push(info)
      return info
    },

    async importPrescription(data, format) {
      return importFile(data, format)
    },

    async applySolves(settings, snap) {
      const { report, solved } = S.evaluateSolves(clone(snap.system) as Obj, settings as Obj)
      return { report, snapshot: await engine.openSnapshot(solved as OpticalSystem) }
    },

    makeDocumentEngine() {
      return engine
    },
  }
  return engine
}
