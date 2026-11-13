// The `@engine` contract: the types the web UI codes against.
//
// Mirrors the Swift `EngineClient` (apps/macos/Sources/MKLightEngine/
// EngineClient.swift) and the engine's JSON shapes (docs/FFI.md,
// docs/RESULTS.md). JSON objects keep the engine's snake_case keys exactly
// as the engine writes them; only the TS-side wrappers (EngineInfo,
// ModelSnapshot, AnalysisResult, …) use camelCase. See docs/WEB.md.

// ---------------------------------------------------------------------------
// JSON

export type JSONValue =
  | null
  | boolean
  | number
  | string
  | JSONValue[]
  | { [key: string]: JSONValue }
export type JSONObject = { [key: string]: JSONValue }

/** An `OpticalSystem` as mklight-core serialises it (docs/FFI.md
 * "Optical system"). `±∞` are the strings "inf" / "-inf", NaN is null. */
export type OpticalSystemJSON = JSONObject

// ---------------------------------------------------------------------------
// Engine info and readiness

/** What `ready()` reports before an engine exists. */
export interface Readiness {
  /** `globalThis.crossOriginIsolated`: SharedArrayBuffer and threads usable. */
  isolated: boolean
  /** Whether the threaded wasm build will be used (isolated + SAB + atomics). */
  threaded: boolean
  /** Threads the rayon pool will get (1 for the single-threaded fallback). */
  threads: number
  /** `navigator.hardwareConcurrency`. */
  hardwareConcurrency: number
  /** `navigator.gpu` exists and an adapter could be requested. */
  webgpu: boolean
  /** When `threaded` is false: why (no isolation, no SAB, …), for a banner. */
  reason: string | null
}

/** One entry of `mk_engine_list_backends` (docs/FFI.md "Backends"). */
export interface BackendInfo {
  backend: JSONObject
  id: string
  title: string
  available: boolean
  reason: string | null
  precision: Precision
  capabilities: JSONObject
}

export interface EngineInfo {
  /** "MKLight engine". */
  name: string
  /** Engine crate version (`run_info.engine_version`). */
  version: string
  /** `mk_engine_list_backends` at creation time. */
  backends: BackendInfo[]
  /** Rayon pool size in the engine worker. */
  threads: number
  /** Cross-origin isolated (threads possible). */
  isolated: boolean
  /** Running the threaded build. */
  threaded: boolean
  /** WebGPU present in this browser (not: a WebGPU backend is registered;
   * see `backends` for that). */
  webgpu: boolean
}

// ---------------------------------------------------------------------------
// Errors

/** The engine's status codes (docs/FFI.md "Status codes"), by name, plus
 * the web-only `unavailable` (the wasm module failed to load). */
export type EngineErrorKind =
  | 'invalidArgument'
  | 'invalidSettings'
  | 'parse'
  | 'invalidModel'
  | 'notFound'
  | 'io'
  | 'unsupported'
  | 'cancelled'
  | 'timeout'
  | 'notReady'
  | 'jobFailed'
  | 'compute'
  | 'panic'
  | 'internal'
  | 'unavailable'

/** Thrown (rejected) by every client call. `issues` is set for
 * `invalidModel` / `invalidSettings` (one message per problem). */
export class EngineError extends Error {
  readonly kind: EngineErrorKind
  readonly issues: string[]
  constructor(kind: EngineErrorKind, message: string, issues: string[] = []) {
    super(message)
    this.name = 'EngineError'
    this.kind = kind
    this.issues = issues.length > 0 ? issues : kind === 'invalidModel' || kind === 'invalidSettings' ? [message] : []
  }
}

// ---------------------------------------------------------------------------
// Snapshots

/** An immutable system the engine accepted, identified by its BLAKE3 hash. */
export interface ModelSnapshot {
  /** 64 hex digits (`mk_session_current_snapshot_hash`). */
  hash: string
  /** The system as the engine holds it (read back after acceptance). */
  system: OpticalSystemJSON
  /** The session's glass-library hash at that time. */
  libraryHash: string | null
}

// ---------------------------------------------------------------------------
// Analyses

/** `mk_engine_list_analyses` entry (docs/FFI.md "Analyses"). */
export interface AnalysisDescriptor {
  name: string
  title: string
  category: string
  description: string
  available: boolean
  unavailable_reason: string | null
  settings_schema: JSONObject
  default_settings: JSONObject | null
  result_schema: JSONObject
  result_version: number
  gpu_path: boolean
  stoppable: boolean
}

/** A backend request (docs/FFI.md): `"auto"`, `"cpu_reference"`, `"gpu"`
 * (the browser's GPU backend, falls back to the CPU with a warning), or
 * the engine's JSON object form. */
export type BackendRequest = 'auto' | 'cpu_reference' | 'gpu' | JSONObject

export interface AnalysisRequest {
  analysis: string
  /** Partial settings; missing keys take the schema defaults. */
  settings?: JSONObject
  backend?: BackendRequest
  /** Recompute, bypassing the analysis graph. */
  fresh?: boolean
  tier?: Tier
  aim_stop?: AimStopKind | null
  route?: JSONObject | null
}

export type Precision = 'f64_reference' | 'f32_gpu'
export type Tier = 'design' | 'preview'
export type AimStopKind = 'requested' | 'conditioned'
export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

export interface BackendAttempt {
  backend: string
  outcome: string
  reason: string
  elapsed_ms: number
}

/** `run_info` of the result envelope (docs/RESULTS.md §1). */
export interface RunInfo {
  engine_version: string
  snapshot_hash: string
  analysis: string
  settings: JSONObject
  backend: string
  precision: Precision
  seed: number | null
  started_unix_ms: number
  duration_ms: number
  tier: Tier
  aim_stop: AimStopKind
  warnings: string[]
  failed_attempts?: BackendAttempt[]
}

export type BufferDType = 'f64' | 'f32' | 'i32' | 'u32' | 'i64' | 'u64' | 'u8'
export type TypedArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array
  | Uint8Array

/** The envelope's `buffers` entry, plus the array itself in `buffers`. */
export interface BufferInfo {
  index: number
  name: string
  dtype: BufferDType
  shape: number[]
}

export interface AnalysisResult {
  /** The analysis' own result (`result` of the envelope). */
  json: JSONValue
  /** Typed arrays by buffer name, row-major (shapes in `bufferInfo`).
   * Transferred out of the worker, owned by the caller. */
  buffers: Record<string, TypedArray>
  bufferInfo: BufferInfo[]
  runInfo: RunInfo
  libraryHash: string | null
  /** `backend` of the envelope (requested, used, fallback reason, …). */
  backend: JSONObject | null
  derivedSnapshotHash: string | null
  resultVersion: number
}

export interface JobProgress {
  /** Monotone, 0…1. */
  fraction: number
  stage: string
}

export interface AnalysisJob {
  /** Client-side id, unique per engine worker. */
  readonly id: number
  readonly analysis: string
  readonly snapshotHash: string
  /** Subscribe to progress; returns the unsubscribe function. */
  onProgress(cb: (p: JobProgress) => void): () => void
  /** Rejects with EngineError (`cancelled` after `cancel()`). */
  result: Promise<AnalysisResult>
  /** Discard the job. Idempotent. */
  cancel(): void
  /** End early keeping the best result so far (stoppable analyses). */
  stop(): void
}

// ---------------------------------------------------------------------------
// Catalogs, solves, import, projects

export interface CatalogInfo {
  name: string
  /** `builtin` or `user`. */
  source: string
  path: string | null
  glass_count: number
}

export interface GlassInfo {
  catalog: string
  name: string
  nd: number
  vd: number
  min_um: number
  max_um: number
}

export interface SolveApplication {
  report: JSONValue
  snapshot: ModelSnapshot
}

export type PrescriptionFormat = 'zmx' | 'seq'

export interface ImportWarning {
  kind: string
  line: number | null
  surface: number | null
  message: string
}

export interface PrescriptionImport {
  format: PrescriptionFormat
  encoding: string
  system: OpticalSystemJSON
  warnings: ImportWarning[]
}

/** `mk_session_info_json`. */
export interface SessionInfo {
  session_id: number
  path: string | null
  dirty: boolean
  snapshot_hash: string
  library_hash: string
  name: string
  open_warnings: string[]
  catalogs: CatalogInfo[]
}

// ---------------------------------------------------------------------------
// The client

export interface EngineClient {
  readonly info: EngineInfo
  /** Validate and adopt `system` as this session's current model. Rejects
   * with EngineError `invalidModel` (with `issues`). */
  openSnapshot(system: OpticalSystemJSON): Promise<ModelSnapshot>
  analysisDescriptors(): Promise<AnalysisDescriptor[]>
  /** Runs against `snap` (the session is set to it first). Never throws
   * synchronously: submission errors reject `result`. */
  submit(req: AnalysisRequest, snap: ModelSnapshot): AnalysisJob
  /** Sorted unique glass names over every loaded catalog. */
  materialNames(): Promise<string[]>
  glasses(catalog?: string): Promise<GlassInfo[]>
  catalogs(): Promise<CatalogInfo[]>
  /** AGF text, or GlassCatalog JSON when `fileName` ends in `.json`. */
  loadCatalog(fileName: string, text: string): Promise<CatalogInfo>
  applySolves(settings: JSONObject, snap: ModelSnapshot): Promise<SolveApplication>
  importPrescription(data: ArrayBuffer, format: PrescriptionFormat): Promise<PrescriptionImport>
  /** Replace this session with a `.mklight` project (docs/FFI.md "Project
   * file"). Catalog references by path cannot be resolved in the browser;
   * they come back in `open_warnings`. Returns the new current snapshot. */
  openProject(text: string): Promise<{ snapshot: ModelSnapshot; info: SessionInfo }>
  /** The session as `.mklight` project text (pretty JSON). */
  saveProject(): Promise<string>
  sessionInfo(): Promise<SessionInfo>
  /** A new session (own system and catalogs) on the same worker/engine. */
  makeDocumentEngine(): EngineClient
  /** Release this session (the engine worker stays while others use it). */
  dispose(): void
}

export interface CreateEngineOptions {
  /** Rayon threads; default `navigator.hardwareConcurrency`. */
  threads?: number
  /** Extra `EngineConfig` keys (docs/FFI.md), e.g. `analysis_graph_bytes`. */
  config?: JSONObject
}
