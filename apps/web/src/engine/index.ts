// Loads the WASM engine (`@engine`). There is no other engine: without the wasm
// package (apps/web/engine/pkg, built by `npm run build:wasm`) `@engine` is the
// stub of ./missing.ts, and loadEngine rejects with a message the app shows as
// its engine error instead of any result.

import * as engineModule from '@engine'
import { adaptEngine, type RealEngine } from './adapt'
import type { EngineClient } from './types'
import { t } from '@/i18n'

const real = engineModule as unknown as {
  createEngine: () => Promise<RealEngine>
  engineMissing?: boolean
}

export * from './types'

export async function loadEngine(): Promise<EngineClient> {
  if ('engineMissing' in real) throw new Error(t('app.engine.notBuilt'))
  // The engine runs only multithreaded: it needs a cross-origin isolated page
  // (COOP/COEP headers, or the isolation service worker in index.html), and
  // createEngine rejects with the reason otherwise. Not `ready()`: it asks
  // for the WebGPU adapter, and the app's start must not wait on the
  // browser's GPU process (the engine worker bounds its own WebGPU calls).
  return adaptEngine(await real.createEngine())
}
