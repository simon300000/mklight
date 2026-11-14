// Stand-in for `@engine` while the wasm package (apps/web/engine/pkg, built by
// `npm run build:wasm`) is not there. vite.config.ts points `@engine` here only
// in that case; loadEngine (./index.ts) then reports the engine as not built.
export const engineMissing = true

export async function createEngine(): Promise<never> {
  throw new Error('The WASM engine is not built.')
}
