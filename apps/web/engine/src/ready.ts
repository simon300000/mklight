// What this page can run, before any engine exists (docs/WEB.md
// "Isolation"). The threaded engine needs SharedArrayBuffer, which browsers
// expose only to cross-origin isolated pages (COOP + COEP headers, or the
// service worker of `isolation.ts`).

import type { Readiness } from './types'
import { anyPipelineFailure, probeAdapter } from './webgpu-state'

type Isolation = Omit<Readiness, 'webgpu' | 'webgpuReason'>

let isolationCached: Isolation | null = null
let cached: Promise<Readiness> | null = null

/** Isolation and threads of this page, without any WebGPU call: what
 * `createEngine` checks. */
export function isolation(): Isolation {
  if (isolationCached) return isolationCached
  const isolated = globalThis.crossOriginIsolated === true
  const sab = typeof SharedArrayBuffer === 'function'
  const hardwareConcurrency = Math.max(1, navigator.hardwareConcurrency || 1)
  let reason: string | null = null
  if (!globalThis.isSecureContext) {
    reason = 'The page is not served over HTTPS (or localhost), so it cannot use the multithreaded engine.'
  } else if (!isolated) {
    reason =
      'The page is not cross-origin isolated (COOP/COEP headers), so the multithreaded engine cannot start. ' +
      'Reload the page; if it persists, private browsing or a browser setting may be blocking the service worker.'
  } else if (!sab) {
    reason = 'This browser does not provide SharedArrayBuffer.'
  }
  const threaded = reason === null
  isolationCached = { isolated, threaded, threads: threaded ? hardwareConcurrency : 1, hardwareConcurrency, reason }
  return isolationCached
}

/** Whether WebGPU can be used, and why not. No WebGPU call when a pipeline
 * failure is remembered for this browser; the adapter request is bounded
 * (`ADAPTER_TIMEOUT_MS`). */
async function webgpu(): Promise<Pick<Readiness, 'webgpu' | 'webgpuReason'>> {
  if (!('gpu' in navigator)) return { webgpu: false, webgpuReason: 'this browser offers no WebGPU' }
  const f = await anyPipelineFailure()
  if (f) return { webgpu: false, webgpuReason: `${f.reason} (remembered for this browser)` }
  const a = await probeAdapter()
  return a.ok ? { webgpu: true, webgpuReason: null } : { webgpu: false, webgpuReason: a.reason }
}

/** Isolation, thread count and WebGPU presence of this page. Cached.
 * Resolves within a few seconds even when the browser's GPU process does
 * not answer. */
export function ready(): Promise<Readiness> {
  cached ??= (async () => ({ ...isolation(), ...(await webgpu()) }))()
  return cached
}
