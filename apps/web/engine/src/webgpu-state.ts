// What this browser remembers about WebGPU, and the bounded adapter request
// (docs/GPU.md §11.4, "Pipeline watchdog"). Shared by the page (`ready.ts`)
// and the engine worker (`worker.ts`).
//
// A pipeline the watchdog gave up on keeps the browser's GPU process busy:
// its compilation cannot be cancelled and may outlive the page. A reload
// that then calls `navigator.gpu.requestAdapter()` waits behind it, so the
// remembered failure is keyed without the adapter (user agent and kernel
// digest) and looked up before any WebGPU call, and every WebGPU call on a
// load path is bounded by a timer.

/** How long one `requestAdapter` may take before WebGPU counts as
 * unavailable for this load. */
export const ADAPTER_TIMEOUT_MS = 4000

/** How long a remembered pipeline failure holds: browsers do not expose the
 * driver version, so a driver update is noticed only by trying again. */
export const PIPELINE_FAILURE_DAYS = 14

const CAL_DB = 'mklight-webgpu-calibration'
const CAL_STORE = 'calibrations'
const FAILURE_PREFIX = 'pipeline-failure|'

/** One request on the calibration database (`undefined` on any error). */
export function idb<T>(mode: IDBTransactionMode, f: (s: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> {
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
      const db = open.result
      try {
        const tx = db.transaction(CAL_STORE, mode)
        const r = f(tx.objectStore(CAL_STORE))
        r.onsuccess = () => resolve(r.result)
        r.onerror = () => resolve(undefined)
        tx.oncomplete = tx.onabort = () => db.close()
      } catch {
        db.close()
        resolve(undefined)
      }
    }
  })
}

/** A compute pipeline that never finished compiling in an earlier load. */
export interface PipelineFailure {
  reason: string
  /** When it was recorded (ms since the epoch). */
  at: number
  /** The adapter it happened on (vendor/architecture/device/description),
   * for the message only. */
  adapter?: string
}

/** The key of a pipeline failure: the browser (its user agent names the
 * version, so an update tries again) and the kernels (a new build tries
 * again). Not the adapter: finding it would need `requestAdapter`. */
export function pipelineFailureKey(kernels: string): string {
  return `${FAILURE_PREFIX}${navigator.userAgent}|kernels=${kernels}`
}

function parse(text: unknown): PipelineFailure | null {
  if (typeof text !== 'string') return null
  try {
    const f = JSON.parse(text) as PipelineFailure
    if (typeof f.reason === 'string' && typeof f.at === 'number') return f
  } catch {
    /* unreadable */
  }
  return null
}

const live = (f: PipelineFailure | null): f is PipelineFailure =>
  f !== null && Date.now() - f.at < PIPELINE_FAILURE_DAYS * 864e5

/**
 * The unexpired pipeline failure remembered for this browser and `kernels`,
 * without any WebGPU call. Records of other browsers or kernel builds are
 * stale and removed, as are expired ones. Records of an earlier format
 * (keyed by adapter too) count as well.
 */
export async function rememberedPipelineFailure(kernels: string): Promise<PipelineFailure | null> {
  const keys = (await idb<IDBValidKey[]>('readonly', (s) => s.getAllKeys())) ?? []
  const mine = `|${navigator.userAgent}|kernels=${kernels}`
  let found: PipelineFailure | null = null
  for (const k of keys) {
    if (typeof k !== 'string' || !k.startsWith(FAILURE_PREFIX)) continue
    const f = k.endsWith(mine) ? parse(await idb('readonly', (s) => s.get(k))) : null
    if (live(f)) found ??= f
    else void idb('readwrite', (s) => s.delete(k))
  }
  return found
}

/** Whether a pipeline failure is remembered for this browser, whatever the
 * kernel build: the page cannot tell the digest without the engine, so
 * `ready()` stays off WebGPU after any. The engine worker removes records
 * of other builds when it starts, so this errs at most for one load. */
export async function anyPipelineFailure(): Promise<PipelineFailure | null> {
  const keys = (await idb<IDBValidKey[]>('readonly', (s) => s.getAllKeys())) ?? []
  const mine = `|${navigator.userAgent}|kernels=`
  for (const k of keys) {
    if (typeof k !== 'string' || !k.startsWith(FAILURE_PREFIX) || !k.includes(mine)) continue
    const f = parse(await idb('readonly', (s) => s.get(k)))
    if (live(f)) return f
  }
  return null
}

/** Remember a pipeline failure for this browser and `kernels`. */
export function recordPipelineFailure(kernels: string, f: PipelineFailure): Promise<unknown> {
  return idb('readwrite', (s) => s.put(JSON.stringify(f), pipelineFailureKey(kernels)))
}

/** The adapter, or why WebGPU is unavailable for this load. */
export type AdapterProbe = { ok: true; key: string } | { ok: false; reason: string }

type GpuLike = { requestAdapter(): Promise<{ info?: Record<string, string> } | null> }

/**
 * Request the WebGPU adapter, bounded by `timeoutMs`: an adapter that does
 * not answer in time (the browser's GPU process busy, e.g. with a
 * compilation an earlier page left behind) makes WebGPU unavailable for
 * this load, with the reason. `key` names the adapter: vendor,
 * architecture, device and description.
 */
export async function probeAdapter(timeoutMs = ADAPTER_TIMEOUT_MS): Promise<AdapterProbe> {
  const gpu = (navigator as Navigator & { gpu?: GpuLike }).gpu
  if (!gpu) return { ok: false, reason: 'this browser offers no WebGPU' }
  let timer: ReturnType<typeof setTimeout> | undefined
  const late = new Promise<'late'>((resolve) => {
    timer = setTimeout(() => resolve('late'), timeoutMs)
  })
  try {
    const a = await Promise.race([gpu.requestAdapter(), late])
    if (a === 'late') {
      return {
        ok: false,
        reason:
          `the browser did not answer the WebGPU adapter request within ${timeoutMs / 1000} s ` +
          '(its GPU process may still be busy), so WebGPU is not used in this page load',
      }
    }
    if (a === null) return { ok: false, reason: 'this browser offers no WebGPU adapter' }
    const i = a.info ?? {}
    return { ok: true, key: [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join('/') || 'unknown' }
  } catch (e) {
    return { ok: false, reason: `the WebGPU adapter request failed: ${e instanceof Error ? e.message : String(e)}` }
  } finally {
    clearTimeout(timer)
  }
}
