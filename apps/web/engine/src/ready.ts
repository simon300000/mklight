// What this page can run, before any engine exists (docs/WEB.md
// "Isolation"). The threaded engine needs SharedArrayBuffer, which browsers
// expose only to cross-origin isolated pages (COOP + COEP headers, or the
// service worker of `isolation.ts`).

import type { Readiness } from './types'

let cached: Promise<Readiness> | null = null

async function hasWebGPU(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown | null> } }).gpu
  if (!gpu) return false
  try {
    return (await gpu.requestAdapter()) !== null
  } catch {
    return false
  }
}

/** Isolation, thread count and WebGPU presence of this page. Cached. */
export function ready(): Promise<Readiness> {
  cached ??= (async () => {
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
    return {
      isolated,
      threaded,
      threads: threaded ? hardwareConcurrency : 1,
      hardwareConcurrency,
      webgpu: await hasWebGPU(),
      reason,
    }
  })()
  return cached
}
