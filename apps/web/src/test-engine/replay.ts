// A replay engine for tests: an EngineClient that serves results recorded from
// the real engine (recordings.json, made by record.mjs), never computed ones.
// It exists for unit tests that need engine data without a browser or wasm;
// the app itself only ever runs the real engine (src/engine/index.ts) and does
// not import this directory.
//
// A request is answered when the recordings hold the analysis with exactly
// those settings (`{}` or none: the default run); anything else rejects with
// `notFound`, so a test never gets a result the engine did not produce for it.
import {
  EngineError,
  type AnalysisDescriptor,
  type AnalysisJob,
  type AnalysisRequest,
  type AnalysisResult,
  type EngineClient,
  type JobProgress,
  type JsonObject,
  type ModelSnapshot,
  type OpticalSystem,
  type TypedArray,
} from '../engine/types'
import recordings from './recordings.json'

interface RecordedBuffer {
  name: string
  dtype: string
  shape: number[]
  data: number[]
}
interface RecordedRun {
  analysis: string
  settings: JsonObject | null
  json: JsonObject
  buffers: RecordedBuffer[]
  run_info: AnalysisResult['runInfo']
  envelope: JsonObject
  progress: JobProgress[]
}
interface Recordings {
  descriptors: AnalysisDescriptor[]
  /** The error the engine gave for a system whose stop surface does not exist. */
  invalidStop: { message: string; issues: string[] }
  runs: RecordedRun[]
}
const rec = recordings as unknown as Recordings

const ARRAYS: Record<string, (d: number[]) => TypedArray> = {
  f64: (d) => Float64Array.from(d),
  f32: (d) => Float32Array.from(d),
  i32: (d) => Int32Array.from(d),
  u32: (d) => Uint32Array.from(d),
  u8: (d) => Uint8Array.from(d),
}

function canonical(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`
  if (v && typeof v === 'object') {
    const o = v as Record<string, unknown>
    return `{${Object.keys(o).sort().map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`).join(',')}}`
  }
  return JSON.stringify(v)
}

const keyOf = (analysis: string, settings: JsonObject | null | undefined) =>
  `${analysis}|${settings && Object.keys(settings).length ? canonical(settings) : ''}`

const byKey = new Map(rec.runs.map((r) => [keyOf(r.analysis, r.settings), r]))

/** The requests this engine can answer, as `analysis` or `analysis {settings}`. */
export const recordedRuns = (): { analysis: string; settings: JsonObject | null }[] =>
  rec.runs.map((r) => ({ analysis: r.analysis, settings: r.settings }))

export function createReplayEngine(opts: { latencyMs?: number } = {}): EngineClient {
  const latency = opts.latencyMs ?? 0
  let nextId = 1

  const result = (r: RecordedRun, snap: ModelSnapshot): AnalysisResult => {
    const buffers: AnalysisResult['buffers'] = {}
    const shapes: Record<string, number[]> = {}
    for (const b of r.buffers) {
      buffers[b.name] = ARRAYS[b.dtype](b.data)
      shapes[b.name] = b.shape
    }
    return {
      json: structuredClone(r.json),
      buffers,
      shapes,
      runInfo: { ...structuredClone(r.run_info), snapshot_hash: snap.hash },
      envelope: structuredClone(r.envelope),
    }
  }

  const engine: EngineClient = {
    info: { name: 'Replay engine (recorded results)', version: 'recorded', backends: ['cpu_reference'], threads: 1, isolated: false, webgpu: false },

    async openSnapshot(system: OpticalSystem): Promise<ModelSnapshot> {
      const surfaces = (system.surfaces as unknown[] | undefined) ?? []
      if (typeof system.stop === 'number' && system.stop >= surfaces.length) {
        throw new EngineError('invalidModel', rec.invalidStop.message, rec.invalidStop.issues)
      }
      return { hash: canonical(system).length.toString(16).padStart(8, '0') + String(surfaces.length), system, libraryHash: null }
    },

    async analysisDescriptors() {
      return rec.descriptors
    },

    submit(req: AnalysisRequest, snap: ModelSnapshot): AnalysisJob {
      const id = nextId++
      const listeners: ((p: JobProgress) => void)[] = []
      const timers: ReturnType<typeof setTimeout>[] = []
      let settled = false
      let fail: (e: unknown) => void = () => {}
      let finish: () => void = () => {}
      const emit = (p: JobProgress) => {
        if (!settled) for (const l of listeners) l(p)
      }
      const run = byKey.get(keyOf(req.analysis, req.settings))
      const promise = new Promise<AnalysisResult>((resolve, reject) => {
        fail = (e) => {
          if (settled) return
          settled = true
          timers.forEach(clearTimeout)
          reject(e)
        }
        finish = () => {
          if (settled) return
          if (!run) return fail(new EngineError('notFound', `no recording for ${keyOf(req.analysis, req.settings)} (src/test-engine/record.mjs lists what is recorded)`))
          for (const p of run.progress) emit(p)
          settled = true
          timers.forEach(clearTimeout)
          resolve(result(run, snap))
        }
        timers.push(setTimeout(finish, latency))
      })
      promise.catch(() => {})
      return {
        id,
        onProgress: (cb) => void listeners.push(cb),
        result: promise,
        cancel: () => fail(new EngineError('cancelled', 'Cancelled')),
        stop: () => finish(),
      }
    },

    async materialNames() {
      return []
    },
    async catalogs() {
      return []
    },
    async loadCatalog() {
      throw new EngineError('unsupported', 'the replay engine loads no catalogs')
    },
    async applySolves() {
      throw new EngineError('unsupported', 'the replay engine applies no solves')
    },
    async importPrescription() {
      throw new EngineError('unsupported', 'the replay engine imports nothing')
    },
    makeDocumentEngine() {
      return engine
    },
  }
  return engine
}
