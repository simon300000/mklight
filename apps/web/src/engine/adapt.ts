// Adapts the WASM engine module (`@engine`, apps/web/engine/src/types.ts) to the
// contract the UI codes against (./types.ts). The two agree on the JSON (the
// engine's snake_case keys) and on the calls; they differ only in a few
// TS-side wrappers, mapped here so the UI and the mock stay as they are:
//
// * progress: `{fraction, stage}` → `{progress, stage, detail}`, with `detail`
//   parsed from the engine's stage text as the macOS app does
//   (`lm 12: merit 3.4e-4`; `glass_substitution: combinations N; best merit M;
//   glasses [...]`; tolerance stages);
// * results: `bufferInfo` → `shapes`, and `backend` / `libraryHash` /
//   `derivedSnapshotHash` / `resultVersion` → `envelope` (the FFI envelope's keys);
// * backend requests: `{kind: 'auto' | 'cpu_reference'}` → `'auto' | 'cpu_reference'`,
//   `{kind: 'gpu', name}` → the engine's object form.

import type {
  AnalysisJob,
  AnalysisRequest,
  AnalysisResult,
  EngineClient,
  EngineInfo,
  Json,
  JsonObject,
  JobProgress,
} from './types'
import { EngineError } from './types'

/* The real module's shapes, as far as the adapter reads them. */
interface RealResult {
  json: Json
  buffers: AnalysisResult['buffers']
  bufferInfo?: { name: string; shape: number[]; dtype: string }[]
  runInfo: AnalysisResult['runInfo']
  libraryHash?: string | null
  backend?: JsonObject | null
  derivedSnapshotHash?: string | null
  resultVersion?: number
}
interface RealJob {
  readonly id: number
  onProgress(cb: (p: { fraction: number; stage: string }) => void): unknown
  result: Promise<RealResult>
  cancel(): void
  stop(): void
}
export interface RealEngine {
  readonly info: Omit<EngineInfo, 'backends'> & { backends: { id: string; title?: string; available?: boolean }[]; threaded?: boolean }
  submit(req: JsonObject & { analysis: string }, snap: unknown): RealJob
  [key: string]: unknown
}

/** Live detail from the engine's progress stage text (DesignJobs.swift, GlassSubstitution.swift). */
export function stageDetail(stage: string): JsonObject | null {
  const lm = /^\s*([a-z_]+)\s+(\d+):\s*merit\s+([-+.\deEinfa]+)/i.exec(stage)
  if (lm) {
    const merit = Number(lm[3])
    return { algorithm: lm[1], iteration: Number(lm[2]), merit, best_merit: merit }
  }
  if (stage.startsWith('glass_substitution:')) {
    const body = stage.slice('glass_substitution:'.length)
    const c = /combinations\s+(\d+)/.exec(body)
    const b = /best merit\s+([^\s;]+)/.exec(body)
    const g = /glasses\s+(\[.*\])\s*$/.exec(body)
    let glasses: Json = []
    if (g) {
      try {
        glasses = JSON.parse(g[1]) as Json
      } catch {
        glasses = []
      }
    }
    return { combinations: c ? Number(c[1]) : 0, best_merit: b ? Number(b[1]) : null, glasses }
  }
  const mc = /monte carlo\s+(\d+)\s*\/\s*(\d+)/i.exec(stage)
  if (mc) return { stage: 'monte_carlo', done: Number(mc[1]), trials: Number(mc[2]) }
  if (/sensitivity/i.test(stage)) {
    const s = /(\d+)\s*\/\s*(\d+)/.exec(stage)
    return { stage: 'sensitivity', ...(s ? { done: Number(s[1]), total: Number(s[2]) } : {}) }
  }
  return null
}

/** The module's own EngineError class → the UI's (same kind and issues). */
export function convertError(e: unknown): unknown {
  if (e instanceof EngineError) return e
  if (e && typeof e === 'object' && (e as { name?: string }).name === 'EngineError') {
    const x = e as { kind: string; message: string; issues?: string[] }
    return new EngineError(x.kind as EngineError['kind'], x.message, x.issues ?? [])
  }
  return e
}
const rethrow = (e: unknown): never => {
  throw convertError(e)
}

function backendWire(b: AnalysisRequest['backend']): Json | undefined {
  if (!b) return undefined
  if (b.kind === 'auto' || b.kind === 'cpu_reference') return b.kind
  return { kind: 'gpu', name: b.name }
}

function adaptResult(r: RealResult): AnalysisResult {
  const shapes: Record<string, number[]> = {}
  for (const b of r.bufferInfo ?? []) shapes[b.name] = b.shape
  const json = (r.json && typeof r.json === 'object' && !Array.isArray(r.json) ? r.json : { value: r.json }) as JsonObject
  return {
    json,
    buffers: r.buffers,
    shapes,
    runInfo: r.runInfo,
    envelope: {
      backend: r.backend ?? null,
      library_hash: r.libraryHash ?? null,
      derived_snapshot_hash: r.derivedSnapshotHash ?? null,
      result_version: r.resultVersion ?? null,
    },
  }
}

function adaptJob(job: RealJob): AnalysisJob {
  return {
    id: job.id,
    onProgress(cb: (p: JobProgress) => void) {
      job.onProgress((p) => cb({ progress: p.fraction, stage: p.stage, detail: stageDetail(p.stage ?? '') }))
    },
    result: job.result.then(adaptResult, rethrow),
    cancel: () => job.cancel(),
    stop: () => job.stop(),
  }
}

/** The WASM engine as the UI's `EngineClient`. */
export function adaptEngine(real: RealEngine): EngineClient {
  const call = <T,>(name: string, ...args: unknown[]): T => {
    let out: unknown
    try {
      out = (real[name] as (...a: unknown[]) => unknown).apply(real, args)
    } catch (e) {
      throw convertError(e)
    }
    return (out instanceof Promise ? out.catch(rethrow) : out) as T
  }
  const client: EngineClient = {
    info: {
      ...real.info,
      backends: real.info.backends,
      isMock: false,
    },
    openSnapshot: (system) => call('openSnapshot', system),
    analysisDescriptors: () => call('analysisDescriptors'),
    submit(req, snap) {
      const wire: JsonObject & { analysis: string } = { analysis: req.analysis }
      if (req.settings) wire.settings = req.settings
      const be = backendWire(req.backend)
      if (be !== undefined) wire.backend = be
      if (req.fresh) wire.fresh = true
      try {
        return adaptJob(real.submit(wire, snap))
      } catch (e) {
        throw convertError(e)
      }
    },
    materialNames: () => call('materialNames'),
    catalogs: () => call('catalogs'),
    loadCatalog: (fileName, text) => call('loadCatalog', fileName, text),
    applySolves: (settings, snap) => call('applySolves', settings, snap),
    importPrescription: (data, format) => call('importPrescription', data, format),
    makeDocumentEngine: () => adaptEngine(call<RealEngine>('makeDocumentEngine')),
  }
  return client
}
