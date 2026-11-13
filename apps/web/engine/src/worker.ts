// The engine worker: owns the wasm engine, its rayon pool and the host
// workers of the engine's own threads. The page talks to it through
// `client.ts` (protocol.ts). Calls are answered in order; analyses run on
// the engine's scheduler threads, and a 30 ms poll while jobs are active
// turns their status into progress and result messages. docs/WEB.md.

/// <reference lib="webworker" />

import init, * as wasm from '../pkg/mklight_wasm.js'
import type { Calls, FromWorker, Op, RawResult, Request, WireError } from './protocol'
import type { EngineErrorKind, JSONObject, TypedArray } from './types'

declare const self: DedicatedWorkerGlobalScope

const POLL_MS = 30
/** Shadow stack of each engine thread (a multiple of 64 KiB). */
const THREAD_STACK = 2 * 1024 * 1024

/** A submitted job: client id → engine id, last progress sent. */
interface Tracked {
  engine: number
  fraction: number
  stage: string
}
const jobs = new Map<number, Tracked>()
let poller: ReturnType<typeof setInterval> | null = null
let ready = false

function post(msg: FromWorker, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer)
}

function wireError(e: unknown): WireError {
  if (e && typeof e === 'object' && 'kind' in e && typeof (e as { kind: unknown }).kind === 'string') {
    return { kind: (e as { kind: EngineErrorKind }).kind, message: String((e as { message?: unknown }).message) }
  }
  if (e instanceof WebAssembly.RuntimeError) {
    // A Rust panic (panic = abort traps): the engine may be inconsistent now.
    return { kind: 'panic', message: `internal error (panic): ${e.message}` }
  }
  return { kind: 'internal', message: e instanceof Error ? e.message : String(e) }
}

function transferables(r: RawResult): Transferable[] {
  return r.buffers.map((b) => (b as TypedArray).buffer as ArrayBuffer)
}

// ---------------------------------------------------------------------------
// Host workers for the engine's own threads (mklight_core::platform)

const hosts: Worker[] = []

/** Start one host worker; resolves once it runs (about to take work). */
function startHost(): Promise<void> {
  const w = new Worker(new URL('./thread-host.ts', import.meta.url), {
    type: 'module',
    name: `mklight-thread-${hosts.length}`,
  })
  hosts.push(w)
  return new Promise((resolve, reject) => {
    w.onmessage = () => resolve()
    w.onerror = (e) => reject(new Error(`thread host failed: ${e.message}`))
    w.postMessage({ module: wasm.wasm_module(), memory: wasm.wasm_memory(), stack: THREAD_STACK })
  })
}

/**
 * Start the GPU worker (docs/GPU.md §11) when this browser exposes WebGPU to
 * workers; resolves once it runs. The engine then registers `gpu:webgpu`,
 * whose device the GPU worker opens and serves.
 */
async function startGpu(): Promise<boolean> {
  if (!('gpu' in navigator) || typeof wasm.gpu_thread !== 'function') return false
  const w = new Worker(new URL('./gpu-host.ts', import.meta.url), { type: 'module', name: 'mklight-gpu' })
  hosts.push(w)
  await new Promise<void>((resolve, reject) => {
    w.onmessage = () => resolve()
    w.onerror = (e) => reject(new Error(`GPU worker failed: ${e.message}`))
    w.postMessage({ module: wasm.wasm_module(), memory: wasm.wasm_memory(), stack: THREAD_STACK })
  })
  return true
}

/** Start the host workers the engine's queued threads still need. */
function topUpHosts() {
  let n = wasm.wanted_host_threads()
  while (n-- > 0) void startHost()
}

// ---------------------------------------------------------------------------
// Jobs

function startPolling() {
  if (poller === null) poller = setInterval(poll, POLL_MS)
}

function poll() {
  if (jobs.size === 0) {
    if (poller !== null) clearInterval(poller)
    poller = null
    return
  }
  topUpHosts()
  const ids = [...jobs.keys()]
  const statuses = JSON.parse(wasm.job_statuses(new Float64Array(ids.map((j) => jobs.get(j)!.engine)))) as Array<{
    state: string
    progress: number
    stage: string
  } | null>
  ids.forEach((job, i) => {
    const t = jobs.get(job)!
    const st = statuses[i]
    if (st === null) {
      jobs.delete(job)
      post({ type: 'failed', job, error: { kind: 'notFound', message: 'the engine lost the job' } })
      return
    }
    if (st.state === 'queued' || st.state === 'running') {
      const fraction = Math.max(t.fraction, st.progress ?? 0)
      const stage = st.stage || (st.state === 'queued' ? 'Queued' : 'Running')
      if (fraction !== t.fraction || stage !== t.stage) {
        t.fraction = fraction
        t.stage = stage
        post({ type: 'progress', job, fraction, stage })
      }
      return
    }
    jobs.delete(job)
    try {
      const r = wasm.job_take_result(t.engine) as RawResult
      post({ type: 'result', job, result: r }, transferables(r))
    } catch (e) {
      post({ type: 'failed', job, error: wireError(e) })
      try {
        wasm.job_release(t.engine)
      } catch {
        // already gone
      }
    }
  })
}

// ---------------------------------------------------------------------------
// Calls

type Handler<K extends Op> = (args: Calls[K]['args']) => Calls[K]['result'] | Promise<Calls[K]['result']>

const json = (s: string) => JSON.parse(s)

function snapshotOf(sid: number) {
  const h = json(wasm.session_hashes(sid)) as { hash: string; library_hash: string }
  return { ...h, system: json(wasm.session_get_system(sid)) as JSONObject }
}

const handlers: { [K in Op]: Handler<K> } = {
  async init({ threads, cores, config }) {
    await init()
    // rayon's pool build waits until all its threads run, and a worker can
    // only start while this one is idle: start their hosts first.
    const [, gpu] = await Promise.all([
      Promise.all(Array.from({ length: threads }, () => startHost())),
      startGpu().catch(() => false),
    ])
    wasm.init_thread_pool(threads)
    wasm.engine_create(JSON.stringify(config), cores)
    if (gpu) wasm.register_webgpu()
    topUpHosts()
    ready = true
    return {
      version: wasm.engine_version(),
      backends: json(wasm.list_backends()),
      threads,
      analyses: json(wasm.list_analyses()),
    }
  },
  listAnalyses: () => json(wasm.list_analyses()),
  sessionNew: () => wasm.session_new(),
  sessionFree: ({ sid }) => (wasm.session_free(sid), null),
  sessionInfo: ({ sid }) => json(wasm.session_info(sid)),
  setSystem({ sid, system }) {
    wasm.session_set_system(sid, JSON.stringify(system))
    return snapshotOf(sid)
  },
  catalogs: ({ sid }) => json(wasm.session_list_catalogs(sid)),
  glasses: ({ sid, catalog }) => json(wasm.session_list_glasses(sid, catalog)),
  loadCatalog: ({ sid, fileName, bytes }) => json(wasm.session_load_catalog(sid, fileName, bytes)),
  applySolves({ sid, settings, hash, system }) {
    const now = json(wasm.session_hashes(sid)) as { hash: string }
    if (now.hash !== hash) wasm.session_set_system(sid, JSON.stringify(system))
    const r = json(wasm.session_apply_solves(sid, JSON.stringify(settings)))
    return { report: r.report, ...snapshotOf(sid) }
  },
  importPrescription: ({ sid, bytes, format }) => json(wasm.import_prescription_bytes(sid, bytes, format)),
  openProject({ sid, text }) {
    const info = json(wasm.session_open_project(sid, text))
    return { info, ...snapshotOf(sid) }
  },
  saveProject: ({ sid }) => wasm.session_save_project(sid),
  submit({ sid, job, hash, system, request }) {
    const engine = wasm.session_submit_on(sid, hash, JSON.stringify(system), JSON.stringify(request))
    jobs.set(job, { engine, fraction: 0, stage: 'Queued' })
    topUpHosts()
    startPolling()
    return engine
  },
  cancel({ job }) {
    const t = jobs.get(job)
    if (t) wasm.job_cancel(t.engine)
    return null
  },
  stop({ job }) {
    const t = jobs.get(job)
    if (t) wasm.job_stop(t.engine)
    return null
  },
  runSync: ({ sid, request }) => wasm.session_run(sid, JSON.stringify(request)) as RawResult,
}

async function handle({ id, op, args }: Request) {
  try {
    if (!ready && op !== 'init') throw { kind: 'notReady', message: 'the engine is not initialised' }
    const value = await (handlers[op] as Handler<Op>)(args)
    const transfer = op === 'runSync' ? transferables(value as RawResult) : []
    post({ type: 'reply', id, ok: true, value }, transfer)
  } catch (e) {
    post({ type: 'reply', id, ok: false, error: wireError(e) })
  }
}

// One call at a time, in arrival order (init is asynchronous).
let chain: Promise<void> = Promise.resolve()
self.onmessage = (ev: MessageEvent<Request>) => {
  chain = chain.then(() => handle(ev.data))
}
