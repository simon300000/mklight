/// <reference types="vite/client" />

declare module '@samples' {
  const samples: { id: string; system: Record<string, unknown> }[]
  export default samples
}
