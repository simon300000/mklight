/// <reference types="vite/client" />

declare module '@samples' {
  const samples: { id: string; system: Record<string, unknown> }[]
  export default samples
}

interface ImportMetaEnv {
  /** `mock` or `wasm`; unset picks the WASM engine when it is in the build. */
  readonly VITE_ENGINE?: string
}
