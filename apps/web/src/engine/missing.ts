// Stand-in for `@engine` while apps/web/engine (the WASM engine module) is not
// in this checkout. vite.config.ts points `@engine` here only in that case.
import type { EngineClient } from './types'

export const engineMissing = true

export async function createEngine(): Promise<EngineClient> {
  throw new Error('The WASM engine module (apps/web/engine) is not part of this build.')
}
