// Messages between the page (`client.ts`) and the engine worker
// (`worker.ts`). Internal to @engine; docs/WEB.md describes the design.

import type { EngineErrorKind, JSONObject, JSONValue, TypedArray } from './types'

/** Every call the client makes, with its arguments and result type. */
export interface Calls {
  init: {
    args: { threads: number; cores: number; config: JSONObject }
    result: { version: string; backends: JSONValue; threads: number; analyses: JSONValue }
  }
  listAnalyses: { args: Record<string, never>; result: JSONValue }
  sessionNew: { args: Record<string, never>; result: number }
  sessionFree: { args: { sid: number }; result: null }
  sessionInfo: { args: { sid: number }; result: JSONValue }
  setSystem: {
    args: { sid: number; system: JSONObject }
    result: { hash: string; library_hash: string; system: JSONObject }
  }
  catalogs: { args: { sid: number }; result: JSONValue }
  glasses: { args: { sid: number; catalog: string }; result: JSONValue }
  loadCatalog: { args: { sid: number; fileName: string; bytes: Uint8Array }; result: JSONValue }
  applySolves: {
    args: { sid: number; settings: JSONObject; hash: string; system: JSONObject }
    result: { report: JSONValue; hash: string; library_hash: string; system: JSONObject }
  }
  importPrescription: { args: { sid: number; bytes: Uint8Array; format: string }; result: JSONValue }
  openProject: {
    args: { sid: number; text: string }
    result: { info: JSONValue; hash: string; library_hash: string; system: JSONObject }
  }
  saveProject: { args: { sid: number }; result: string }
  /** Submit; the result arrives as a `job` event. Returns the engine job id. */
  submit: {
    args: { sid: number; job: number; hash: string; system: JSONObject; request: JSONObject }
    result: number
  }
  cancel: { args: { job: number }; result: null }
  stop: { args: { job: number }; result: null }
  /** Test support: run in the worker thread itself (blocking it). */
  runSync: { args: { sid: number; request: JSONObject }; result: RawResult }
}

export type Op = keyof Calls

export interface Request<K extends Op = Op> {
  id: number
  op: K
  args: Calls[K]['args']
}

export interface WireError {
  kind: EngineErrorKind
  message: string
}

/** The envelope as text plus its buffers, in envelope order. */
export interface RawResult {
  envelope: string
  buffers: TypedArray[]
}

export type FromWorker =
  | { type: 'reply'; id: number; ok: true; value: unknown }
  | { type: 'reply'; id: number; ok: false; error: WireError }
  | { type: 'progress'; job: number; fraction: number; stage: string }
  | { type: 'result'; job: number; result: RawResult }
  | { type: 'failed'; job: number; error: WireError }
