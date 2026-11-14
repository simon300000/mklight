// Loads the WASM engine (`@engine`). There is no other engine: without the wasm
// package (apps/web/engine/pkg, built by `npm run build:wasm`) `@engine` is the
// stub of ./missing.ts, and loadEngine rejects with a message the app shows as
// its engine error instead of any result.

import * as engineModule from '@engine'
import { adaptEngine, type RealEngine } from './adapt'
import type { EngineClient } from './types'

const real = engineModule as unknown as {
  createEngine: () => Promise<RealEngine>
  ready?: () => Promise<{ threaded: boolean; reason: string | null }>
  engineMissing?: boolean
}

export * from './types'

export const ENGINE_NOT_BUILT = 'Engine not built: run npm run build:wasm in apps/web, then reload.'

export async function loadEngine(): Promise<EngineClient> {
  if ('engineMissing' in real) throw new Error(ENGINE_NOT_BUILT)
  // The engine runs only multithreaded: it needs a cross-origin isolated page
  // (COOP/COEP headers, or the isolation service worker in index.html).
  const r = await real.ready?.()
  if (r && !r.threaded) throw new Error(r.reason ?? 'This browser cannot run the multithreaded engine.')
  return adaptEngine(await real.createEngine())
}
