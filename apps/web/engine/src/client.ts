// The page-side EngineClient: typed calls over postMessage to the engine
// worker (worker.ts). Mirrors the Swift FFIEngine: one worker (engine) per
// `createEngine`, one engine session per client; `makeDocumentEngine` adds
// sessions on the same worker. docs/WEB.md.

import type { Calls, FromWorker, Op, RawResult, WireError } from './protocol'
import { ready } from './ready'
import {
  type AnalysisDescriptor,
  type AnalysisJob,
  type AnalysisRequest,
  type AnalysisResult,
  type BackendInfo,
  type BackendRequest,
  type BufferInfo,
  type CatalogInfo,
  type CreateEngineOptions,
  type EngineClient,
  type EngineInfo,
  EngineError,
  type GlassInfo,
  type JobProgress,
  type JSONObject,
  type ModelSnapshot,
  type OpticalSystemJSON,
  type PrescriptionFormat,
  type PrescriptionImport,
  type RunInfo,
  type SessionInfo,
  type SolveApplication,
  type TypedArray,
} from './types'

/** Web defaults of the engine configuration (docs/WEB.md "Memory"). */
const WEB_CONFIG: JSONObject = {
  app_name: 'MKLight Web',
  // The analysis graph's budget: 64 MiB of the 2 GiB shared memory
  // (native: 256 MiB).
  analysis_graph_bytes: 64 * 1024 * 1024,
}

function toError(e: WireError): EngineError {
  return new EngineError(e.kind, e.message)
}

interface JobSink {
  progress: Set<(p: JobProgress) => void>
  last: JobProgress
  resolve: (r: AnalysisResult) => void
  reject: (e: EngineError) => void
}

/** The worker connection shared by an engine's sessions. */
class Connection {
  private worker: Worker
  private nextId = 1
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: EngineError) => void }>()
  readonly jobs = new Map<number, JobSink>()
  nextJob = 1
  private dead: EngineError | null = null

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'mklight-engine' })
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => this.receive(ev.data)
    this.worker.onerror = (ev) => {
      ev.preventDefault?.()
      this.fail(new EngineError('unavailable', `engine worker failed: ${ev.message || 'unknown error'}`))
    }
  }

  private fail(e: EngineError) {
    this.dead = e
    for (const p of this.pending.values()) p.reject(e)
    this.pending.clear()
    for (const j of this.jobs.values()) j.reject(e)
    this.jobs.clear()
  }

  call<K extends Op>(op: K, args: Calls[K]['args'], transfer: Transferable[] = []): Promise<Calls[K]['result']> {
    if (this.dead) return Promise.reject(this.dead)
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject })
      this.worker.postMessage({ id, op, args }, transfer)
    })
  }

  private receive(m: FromWorker) {
    switch (m.type) {
      case 'reply': {
        const p = this.pending.get(m.id)
        if (!p) return
        this.pending.delete(m.id)
        if (m.ok) p.resolve(m.value)
        else p.reject(toError(m.error))
        return
      }
      case 'progress': {
        const j = this.jobs.get(m.job)
        if (!j) return
        j.last = { fraction: m.fraction, stage: m.stage }
        for (const cb of j.progress) cb(j.last)
        return
      }
      case 'result': {
        const j = this.jobs.get(m.job)
        if (!j) return
        this.jobs.delete(m.job)
        try {
          const r = decodeResult(m.result)
          const done = { fraction: 1, stage: 'Done' }
          for (const cb of j.progress) cb(done)
          j.resolve(r)
        } catch (e) {
          j.reject(new EngineError('parse', `unexpected result envelope: ${(e as Error).message}`))
        }
        return
      }
      case 'failed': {
        const j = this.jobs.get(m.job)
        if (!j) return
        this.jobs.delete(m.job)
        j.reject(toError(m.error))
        return
      }
    }
  }

  terminate() {
    this.worker.terminate()
    this.fail(new EngineError('unavailable', 'the engine was shut down'))
  }
}

/** Envelope text + buffers → AnalysisResult (docs/RESULTS.md §1). */
export function decodeResult(raw: RawResult): AnalysisResult {
  const env = JSON.parse(raw.envelope)
  const bufferInfo = (env.buffers ?? []) as BufferInfo[]
  const buffers: Record<string, TypedArray> = {}
  for (const b of bufferInfo) buffers[b.name] = raw.buffers[b.index]
  return {
    json: env.result ?? null,
    buffers,
    bufferInfo,
    runInfo: env.run_info as RunInfo,
    libraryHash: env.library_hash ?? null,
    backend: env.backend ?? null,
    derivedSnapshotHash: env.derived_snapshot_hash ?? null,
    resultVersion: env.result_version ?? 0,
  }
}

/** The request as the engine takes it (docs/FFI.md "Analysis request"). */
export function wireRequest(req: AnalysisRequest): JSONObject {
  const o: JSONObject = { analysis: req.analysis, settings: req.settings ?? {} }
  const backend = wireBackend(req.backend)
  if (backend) {
    o.backend = backend
    if ((backend as { kind?: string }).kind === 'gpu') o.allow_backend_fallback = true
  }
  if (req.fresh) o.fresh = true
  if (req.tier) o.tier = req.tier
  if (req.aim_stop !== undefined) o.aim_stop = req.aim_stop
  if (req.route !== undefined) o.route = req.route
  return o
}

function wireBackend(b: BackendRequest | undefined): JSONObject | null {
  if (b === undefined || b === 'auto') return null
  if (b === 'cpu_reference') return { kind: 'cpu_reference' }
  if (b === 'gpu') return { kind: 'gpu', name: 'webgpu' }
  return b
}

class Client implements EngineClient {
  constructor(
    private conn: Connection,
    readonly info: EngineInfo,
    private sid: Promise<number>,
    private descriptors: AnalysisDescriptor[],
  ) {}

  private snapshot(r: { hash: string; library_hash: string; system: JSONObject }): ModelSnapshot {
    return { hash: r.hash, system: r.system, libraryHash: r.library_hash }
  }

  async openSnapshot(system: OpticalSystemJSON): Promise<ModelSnapshot> {
    const sid = await this.sid
    try {
      return this.snapshot(await this.conn.call('setSystem', { sid, system }))
    } catch (e) {
      // As the Swift client: what the engine cannot take as a system is an
      // invalid model.
      if (e instanceof EngineError && (e.kind === 'parse' || e.kind === 'invalidArgument' || e.kind === 'invalidModel')) {
        throw new EngineError('invalidModel', e.message, [e.message])
      }
      throw e
    }
  }

  async analysisDescriptors(): Promise<AnalysisDescriptor[]> {
    return this.descriptors
  }

  submit(req: AnalysisRequest, snap: ModelSnapshot): AnalysisJob {
    const conn = this.conn
    const job = conn.nextJob++
    let resolve!: (r: AnalysisResult) => void
    let reject!: (e: EngineError) => void
    const result = new Promise<AnalysisResult>((res, rej) => {
      resolve = res
      reject = rej
    })
    const sink: JobSink = {
      progress: new Set(),
      last: { fraction: 0, stage: 'Queued' },
      resolve,
      reject,
    }
    conn.jobs.set(job, sink)
    let settled = false
    result.then(
      () => (settled = true),
      () => (settled = true),
    )
    const request = wireRequest(req)
    const submitted = this.sid
      .then((sid) => conn.call('submit', { sid, job, hash: snap.hash, system: snap.system, request }))
      .then(
        () => true,
        (e: EngineError) => {
          conn.jobs.delete(job)
          reject(e.kind === 'invalidArgument' ? new EngineError('invalidSettings', e.message, [e.message]) : e)
          return false
        },
      )
    // Cancel and stop follow the submission (calls are handled in order).
    const after = (op: 'cancel' | 'stop') =>
      submitted.then((ok) => (ok && !settled ? conn.call(op, { job }) : null)).catch(() => {})
    return {
      id: job,
      analysis: req.analysis,
      snapshotHash: snap.hash,
      result,
      onProgress(cb) {
        sink.progress.add(cb)
        cb(sink.last)
        return () => sink.progress.delete(cb)
      },
      cancel() {
        if (!settled) after('cancel')
      },
      stop() {
        if (!settled) after('stop')
      },
    }
  }

  async materialNames(): Promise<string[]> {
    const g = await this.glasses()
    return [...new Set(g.map((x) => x.name))].sort()
  }

  async glasses(catalog?: string): Promise<GlassInfo[]> {
    return (await this.conn.call('glasses', { sid: await this.sid, catalog: catalog ?? '' })) as unknown as GlassInfo[]
  }

  async catalogs(): Promise<CatalogInfo[]> {
    return (await this.conn.call('catalogs', { sid: await this.sid })) as unknown as CatalogInfo[]
  }

  async loadCatalog(fileName: string, text: string): Promise<CatalogInfo> {
    const bytes = new TextEncoder().encode(text)
    const r = await this.conn.call('loadCatalog', { sid: await this.sid, fileName, bytes }, [bytes.buffer])
    return r as unknown as CatalogInfo
  }

  async applySolves(settings: JSONObject, snap: ModelSnapshot): Promise<SolveApplication> {
    const r = await this.conn.call('applySolves', {
      sid: await this.sid,
      settings,
      hash: snap.hash,
      system: snap.system,
    })
    return { report: r.report, snapshot: this.snapshot(r) }
  }

  async importPrescription(data: ArrayBuffer, format: PrescriptionFormat): Promise<PrescriptionImport> {
    const bytes = new Uint8Array(data.slice(0))
    const r = await this.conn.call('importPrescription', { sid: await this.sid, bytes, format }, [bytes.buffer])
    return r as unknown as PrescriptionImport
  }

  async openProject(text: string): Promise<{ snapshot: ModelSnapshot; info: SessionInfo }> {
    const r = await this.conn.call('openProject', { sid: await this.sid, text })
    return { snapshot: this.snapshot(r), info: r.info as unknown as SessionInfo }
  }

  async saveProject(): Promise<string> {
    return this.conn.call('saveProject', { sid: await this.sid })
  }

  async sessionInfo(): Promise<SessionInfo> {
    return (await this.conn.call('sessionInfo', { sid: await this.sid })) as unknown as SessionInfo
  }

  async webgpuCalibration(): Promise<JSONObject> {
    return (await this.conn.call('webgpuCalibration', {})) as JSONObject
  }

  async setWebgpuCalibration(calibration: JSONObject | null): Promise<void> {
    await this.conn.call('setWebgpuCalibration', { calibration })
  }

  makeDocumentEngine(): EngineClient {
    return new Client(this.conn, this.info, this.conn.call('sessionNew', {}), this.descriptors)
  }

  dispose(): void {
    this.sid.then((sid) => this.conn.call('sessionFree', { sid })).catch(() => {})
  }

  /** Test support: run a request in the engine worker's own thread. */
  async runSync(req: AnalysisRequest): Promise<AnalysisResult> {
    return decodeResult(await this.conn.call('runSync', { sid: await this.sid, request: wireRequest(req) }))
  }

  /** Shut the engine worker down (every session of this engine). */
  terminate(): void {
    this.conn.terminate()
  }
}

/** Start an engine worker and return a client for its first session.
 * Rejects with EngineError `unavailable` when this page cannot run the
 * threaded engine (no cross-origin isolation): see `ready()`. */
export async function createEngine(opts: CreateEngineOptions = {}): Promise<EngineClient> {
  const r = await ready()
  if (!r.threaded) {
    throw new EngineError('unavailable', r.reason ?? 'this browser cannot run the engine')
  }
  const threads = Math.max(1, Math.floor(opts.threads ?? r.threads))
  const conn = new Connection()
  const init = await conn.call('init', {
    threads,
    cores: r.hardwareConcurrency,
    config: { ...WEB_CONFIG, ...(opts.config ?? {}) },
    webgpuCalibration: opts.webgpuCalibration ?? 'auto',
  })
  const sid = await conn.call('sessionNew', {})
  const info: EngineInfo = {
    name: 'MKLight engine',
    version: init.version,
    backends: init.backends as unknown as BackendInfo[],
    threads: init.threads,
    isolated: r.isolated,
    threaded: true,
    webgpu: r.webgpu,
  }
  return new Client(conn, info, Promise.resolve(sid), init.analyses as unknown as AnalysisDescriptor[])
}
