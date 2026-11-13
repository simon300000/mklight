// The engine boundary, as the web UI codes against it.
//
// Mirrors the Swift `EngineClient` (apps/macos/Sources/MKLightEngine/EngineClient.swift)
// and the FFI JSON shapes (docs/FFI.md, docs/RESULTS.md). JSON that crosses the
// boundary keeps the engine's snake_case keys; the UI never computes optics.
//
// The real module (apps/web/engine, `@engine`) exports `createEngine()`; the mock
// in src/engine-mock implements the same interface with synthetic data.

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
export type JsonObject = { [key: string]: Json }

/** An optical system in the engine's `OpticalSystem` serialisation. */
export type OpticalSystem = JsonObject

export interface EngineInfo {
  name: string
  version: string
  /** Backend ids the engine can run (`cpu_reference`, `gpu:webgpu`, …). */
  backends: string[]
  /** Worker threads of the engine's pool. */
  threads: number
  /** `crossOriginIsolated` in the page (needed for threads). */
  isolated: boolean
  /** WebGPU adapter available. */
  webgpu: boolean
  /** True for engines that return synthetic data; the UI labels them. */
  isMock?: boolean
}

export interface ModelSnapshot {
  hash: string
  system: OpticalSystem
  libraryHash?: string | null
}

export type EngineErrorKind =
  | 'cancelled'
  | 'invalidModel'
  | 'invalidSettings'
  | 'unsupported'
  | 'failed'
  | 'engine'
  | 'unavailable'

export class EngineError extends Error {
  kind: EngineErrorKind
  issues: string[]
  constructor(kind: EngineErrorKind, message: string, issues: string[] = []) {
    super(message)
    this.name = 'EngineError'
    this.kind = kind
    this.issues = issues
  }
}

/** One entry of `mk_engine_list_analyses` (docs/FFI.md). */
export interface AnalysisDescriptor {
  name: string
  title: string
  category: string
  description: string
  available: boolean
  unavailable_reason: string | null
  settings_schema: JsonObject
  default_settings: JsonObject | null
  result_version?: number
  gpu_path?: boolean
  stoppable: boolean
}

export type BackendRequest =
  | { kind: 'auto' }
  | { kind: 'cpu_reference' }
  | { kind: 'gpu'; name: string }

export interface AnalysisRequest {
  analysis: string
  settings?: JsonObject
  backend?: BackendRequest
  /** Recompute instead of answering from the analysis graph (Run / Try Again). */
  fresh?: boolean
}

export interface JobProgress {
  progress: number
  stage?: string | null
  /** Analysis-specific live data (optimisation merit, trials, …). */
  detail?: Json
}

export type TypedArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Uint8Array
  | BigInt64Array
  | BigUint64Array

export interface RunInfo {
  engine_version?: string
  snapshot_hash: string
  analysis: string
  settings: JsonObject
  backend: string
  precision?: string
  seed?: number | null
  started_unix_ms?: number
  duration_ms?: number
  tier?: string
  aim_stop?: string
  warnings: string[]
  [key: string]: Json | undefined
}

/** A finished analysis: the envelope's `result`, its buffers by name, and `run_info`. */
export interface AnalysisResult {
  json: JsonObject
  buffers: Record<string, TypedArray>
  /** Buffer shapes by name, when the engine reports them. */
  shapes?: Record<string, number[]>
  runInfo: RunInfo
  /** The rest of the envelope (`backend`, `library_hash`, …), when available. */
  envelope?: JsonObject
}

export interface AnalysisJob {
  id: number
  onProgress(cb: (p: JobProgress) => void): void
  result: Promise<AnalysisResult>
  /** Discard the job. Its result promise rejects with `cancelled`. */
  cancel(): void
  /** Stop early keeping the best result so far (stoppable analyses). */
  stop(): void
}

export interface CatalogInfo {
  name: string
  source: string
  path?: string | null
  glass_count: number
}

export interface ImportWarning {
  /** `unsupported_record`, `unsupported_surface`, `unknown_glass`, `solve_dropped`,
   *  `non_sequential`, `approximated`, `malformed`, `defaulted`, `validation`, `other`. */
  kind?: string
  category?: string
  message: string
  surface?: number | null
  line?: number | null
  [key: string]: Json | undefined
}

export interface PrescriptionImport {
  format: 'zmx' | 'seq'
  encoding?: string
  system: OpticalSystem
  warnings: ImportWarning[]
}

export interface SolveApplication {
  report: JsonObject
  snapshot: ModelSnapshot
}

export interface EngineClient {
  info: EngineInfo
  openSnapshot(system: OpticalSystem): Promise<ModelSnapshot>
  analysisDescriptors(): Promise<AnalysisDescriptor[]>
  submit(req: AnalysisRequest, snap: ModelSnapshot): AnalysisJob
  materialNames(): Promise<string[]>
  catalogs(): Promise<CatalogInfo[]>
  loadCatalog(fileName: string, text: string): Promise<CatalogInfo>
  applySolves(settings: JsonObject, snap: ModelSnapshot): Promise<SolveApplication>
  importPrescription(data: ArrayBuffer, format: 'zmx' | 'seq'): Promise<PrescriptionImport>
  makeDocumentEngine(): EngineClient | Promise<EngineClient>
}
