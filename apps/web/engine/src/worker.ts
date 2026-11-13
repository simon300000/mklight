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
/** Attempts to start one worker before giving up (delays 50 ms … 6.4 s). */
const START_ATTEMPTS = 8

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

/**
 * Start a worker from `make` and hand it the module and the shared memory;
 * resolves once it reports that it runs. A worker the browser refuses to
 * start (an `error` event before it runs: WebKit does this when a worker
 * script's response lacks the COEP header, as a server's `304 Not Modified`
 * often does) is replaced, with growing delays, up to `START_ATTEMPTS`
 * times. Every engine thread waits for its host, so a refused start must
 * never be final while a retry can succeed.
 */
async function startWorker(make: () => Worker, what: string): Promise<void> {
  for (let attempt = 1; ; attempt++) {
    const w = make()
    hosts.push(w)
    const ok = await new Promise<boolean>((resolve) => {
      w.onmessage = () => resolve(true)
      w.onerror = (e) => {
        e.preventDefault()
        resolve(false)
      }
      w.postMessage({ module: wasm.wasm_module(), memory: wasm.wasm_memory(), stack: THREAD_STACK })
    })
    if (ok) return
    w.terminate()
    hosts.splice(hosts.indexOf(w), 1)
    if (attempt >= START_ATTEMPTS) throw new Error(`${what} could not be started (${attempt} attempts)`)
    console.warn(`MKLight: the browser refused to start ${what} (attempt ${attempt}); retrying`)
    await new Promise((r) => setTimeout(r, 25 * 2 ** attempt))
  }
}

/** Start one host worker; resolves once it runs (about to take work). */
function startHost(): Promise<void> {
  return startWorker(
    () =>
      new Worker(new URL('./thread-host.ts', import.meta.url), {
        type: 'module',
        name: `mklight-thread-${hosts.length}`,
      }),
    'an engine thread',
  )
}

/**
 * Start the GPU worker (docs/GPU.md §11) when this browser exposes WebGPU to
 * workers; resolves once it runs. The engine then registers `gpu:webgpu`,
 * whose device the GPU worker opens and serves.
 */
async function startGpu(): Promise<boolean> {
  if (!('gpu' in navigator) || typeof wasm.gpu_thread !== 'function') return false
  await startWorker(
    () => new Worker(new URL('./gpu-host.ts', import.meta.url), { type: 'module', name: 'mklight-gpu' }),
    'the GPU worker',
  )
  return true
}

// ---------------------------------------------------------------------------
// WebGPU routing calibration (docs/GPU.md §11.6), cached in IndexedDB

const CAL_DB = 'mklight-webgpu-calibration'
const CAL_STORE = 'calibrations'

function idb<T>(mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> {
  return new Promise((resolve) => {
    let open: IDBOpenDBRequest
    try {
      open = indexedDB.open(CAL_DB, 1)
    } catch {
      return resolve(undefined)
    }
    open.onupgradeneeded = () => open.result.createObjectStore(CAL_STORE)
    open.onerror = () => resolve(undefined)
    open.onsuccess = () => {
      try {
        const r = f(open.result.transaction(CAL_STORE, mode).objectStore(CAL_STORE))
        r.onsuccess = () => resolve(r.result)
        r.onerror = () => resolve(undefined)
      } catch {
        resolve(undefined)
      }
    }
  })
}

/** What a calibration is valid for: the adapter, the CPU threads and the
 * engine version. */
async function calibrationKey(threads: number): Promise<string> {
  let adapter = 'unknown'
  try {
    const a = await (navigator as unknown as { gpu: { requestAdapter(): Promise<{ info?: Record<string, string> } | null> } }).gpu.requestAdapter()
    const i = a?.info ?? {}
    adapter = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join('/') || 'unknown'
  } catch {
    /* keyed as unknown */
  }
  return `${adapter}|threads=${threads}|engine=${wasm.engine_version()}`
}

let calibrationTimer: ReturnType<typeof setInterval> | null = null

/** Install the cached calibration, or measure one in the background and
 * cache it when it is ready. */
async function startCalibration(threads: number) {
  const key = await calibrationKey(threads)
  const cached = await idb<string>('readonly', (s) => s.get(key))
  if (cached) {
    const c = JSON.parse(cached)
    c.source = 'cached'
    wasm.webgpu_calibration_set(JSON.stringify(c))
    return
  }
  if (!wasm.webgpu_calibration_start()) return
  topUpHosts()
  calibrationTimer = setInterval(() => {
    // The calibration's analyses start threads (plan workers, scoped
    // threads) like any job: give them hosts.
    topUpHosts()
    const s = json(wasm.webgpu_calibration_state()) as { state: string; calibration?: JSONObject }
    if (s.state === 'running') return
    clearInterval(calibrationTimer!)
    calibrationTimer = null
    if (s.state === 'ready' && s.calibration) {
      void idb('readwrite', (st) => st.put(JSON.stringify(s.calibration), key))
    }
  }, 100)
}

/** Start the host workers the engine's queued threads still need. */
function topUpHosts() {
  let n = wasm.wanted_host_threads()
  while (n-- > 0) {
    startHost().catch((e: Error) => {
      // Asked for again by the next top-up (every poll while jobs run).
      console.error(`MKLight: ${e.message}`)
      wasm.host_start_failed()
    })
  }
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
  async init({ threads, cores, config, webgpuCalibration }) {
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
    if (gpu && webgpuCalibration !== 'off') void startCalibration(threads)
    ready = true
    return {
      version: wasm.engine_version(),
      backends: json(wasm.list_backends()),
      threads,
      analyses: json(wasm.list_analyses()),
    }
  },
  listAnalyses: () => json(wasm.list_analyses()),
  webgpuCalibration: () =>
    typeof wasm.webgpu_calibration_state === 'function' ? json(wasm.webgpu_calibration_state()) : { state: 'idle' },
  setWebgpuCalibration({ calibration }) {
    if (calibrationTimer !== null) {
      clearInterval(calibrationTimer)
      calibrationTimer = null
    }
    wasm.webgpu_calibration_set(calibration ? JSON.stringify(calibration) : '')
    return null
  },
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
