// Chooses the engine: the WASM engine (`@engine`) when it is part of the build,
// else the mock. `?engine=mock` or `?engine=wasm` in the URL, or VITE_ENGINE at
// build time, choose explicitly.

import * as engineModule from '@engine'
import { adaptEngine, type RealEngine } from './adapt'
import type { EngineClient } from './types'

const real = engineModule as unknown as {
  createEngine: () => Promise<RealEngine>
  ready?: () => Promise<{ threaded: boolean; reason: string | null }>
  engineMissing?: boolean
}

export * from './types'

export type EngineChoice = 'mock' | 'wasm'

export function requestedEngine(): EngineChoice {
  const q = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('engine') : null
  const v = (q ?? import.meta.env.VITE_ENGINE ?? '').toLowerCase()
  if (v === 'mock' || v === 'wasm') return v
  return 'engineMissing' in real ? 'mock' : 'wasm'
}

export async function loadEngine(choice = requestedEngine()): Promise<EngineClient> {
  if (choice === 'wasm' && !('engineMissing' in real)) {
    // The engine runs only multithreaded: it needs a cross-origin isolated page
    // (COOP/COEP headers, or the isolation service worker in index.html).
    const r = await real.ready?.()
    if (r && !r.threaded) throw new Error(r.reason ?? 'This browser cannot run the multithreaded engine.')
    return adaptEngine(await real.createEngine())
  }
  const { createMockEngine } = await import('@/engine-mock')
  return createMockEngine()
}
