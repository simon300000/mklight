import Foundation
import MKLightEngine
import MKLightModel
import Synchronization

/// The Rust engine over its C ABI (`crates/mklight-ffi`, docs/FFI.md).
///
/// An `FFIEngine` is one engine handle plus one **session** (the current
/// system, user glass catalogs, library hash). `makeDocumentEngine()` gives
/// each document its own session on the shared engine.
///
/// Lifetimes: the engine handle (`mk_engine_destroy`) and each session
/// (`mk_session_destroy`) are freed when the last `FFIEngine` value and the
/// last running job referring to them are gone. A session keeps its engine
/// alive.
///
/// Threading: `FFIEngine` is `Sendable` and may be used from any task. The C
/// engine and sessions are thread-safe. The asynchronous `EngineClient`
/// methods never block their caller: their C calls (including waits for the
/// session lock) run on a global dispatch queue, never on the main thread.
/// Job events arrive on the engine's `mklight-events` thread and are
/// forwarded to each job's `AsyncStream`s; consumers hop to the main actor
/// themselves. A job runs against the snapshot it was submitted with: the
/// session's system is set to that snapshot and the job submitted in one
/// locked step.
///
/// When the library was not linked into the build, `init` throws
/// `EngineError.libraryUnavailable` (see `FFIEngine.isLinked`).
public struct FFIEngine: EngineClient {
    let core: FFIEngineCore
    let session: FFISessionHandle

    /// Whether `libmklight_ffi.a` was linked into this build.
    public static var isLinked: Bool { CABI.isLinked }

    /// Engine configuration keys (docs/FFI.md, `mk_engine_create`), e.g.
    /// `worker_threads`, `max_retained_jobs`, `progress_interval_ms`.
    public init(configuration: [String: JSONValue] = [:]) throws {
        var config = configuration
        config["app_name"] = config["app_name"] ?? .string("MKLight")
        let text = String(decoding: try MKLightJSON.encode(JSONValue.object(config)), as: UTF8.self)
        let core = try FFIEngineCore(configuration: text)
        self.init(core: core, session: try FFISessionHandle(core: core))
    }

    init(core: FFIEngineCore, session: FFISessionHandle) {
        self.core = core
        self.session = session
    }

    public var info: EngineInfo {
        EngineInfo(name: "MKLight engine", version: core.version, isMock: false)
    }

    /// A preparation the engine refuses to start is reported by
    /// `gpuPreparation()` as failed.
    public func prepareGPU(cacheDirectory: URL?) {
        let refusal: String?
        do {
            try CABI.prepareGPU(core.pointer, cacheDirectory: cacheDirectory?.path)
            refusal = nil
        } catch {
            refusal = Self.describe(error)
        }
        core.gpuPreparationError.withLock { $0 = refusal }
    }

    /// The engine's state; `failed` when `prepareGPU` was refused or the
    /// state cannot be read.
    public func gpuPreparation() -> GPUPreparation {
        if let refusal = core.gpuPreparationError.withLock({ $0 }) { return .failed(device: "", message: refusal) }
        do {
            return try decodeJSON(GPUPreparation.self, try CABI.gpuPreparation(core.pointer))
        } catch {
            return .failed(device: "", message: Self.describe(error))
        }
    }

    private static func describe(_ error: any Error) -> String {
        (error as? EngineError).map(message(of:)) ?? error.localizedDescription
    }

    // MARK: Sessions

    /// A new session on the same engine.
    public func newSession() throws -> FFIEngine {
        FFIEngine(core: core, session: try FFISessionHandle(core: core))
    }

    public func makeDocumentEngine() -> any EngineClient {
        (try? newSession()) ?? self
    }

    /// Glasses of one catalog, or of every catalog.
    public func glasses(in catalog: String? = nil) async throws -> [GlassInfo] {
        let session = self.session
        let text = try await offMain { try session.locked { try CABI.sessionListGlasses($0, catalog: catalog) } }
        return try decodeJSON([GlassInfo].self, text)
    }

    // MARK: EngineClient: snapshots

    public func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot {
        let local = system.structuralIssues()
        let json: String
        do {
            json = try encodeSystem(system)
        } catch {
            throw EngineError.invalidModel(local.filter { $0.severity == .error }.map(\.description))
        }
        let session = self.session
        do {
            let (hash, library) = try await offMain {
                try session.locked { s in (try CABI.sessionSetSystem(s, json: json), try CABI.sessionLibraryHash(s)) }
            }
            return ModelSnapshot(system: system, hash: SnapshotHash(hex: hash), libraryHash: library)
        } catch let error as EngineError
            where error.code == .invalidModel || error.code == .parse || error.code == .invalidArgument
        {
            // The engine stops at its first error (a system it cannot read
            // as one, or one that fails validation); add the mirror's list.
            let message = error.code == .invalidModel ? Self.message(of: error) : (error.errorDescription ?? "")
            let extra = local.filter { $0.severity == .error }.map(\.description).filter { $0 != message }
            throw EngineError.invalidModel([message] + extra)
        }
    }

    public func materialNames() async -> [String] {
        let names = ((try? await glasses()) ?? []).map(\.name)
        return Array(Set(names)).sorted()
    }

    /// Solves the snapshot's system in the engine (`mk_session_apply_solves`)
    /// and reads the solved system back.
    public func applySolves(settings: [String: JSONValue], to snapshot: ModelSnapshot) async throws -> SolveApplication {
        let json = try encodeSystem(snapshot.system)
        let settingsJSON = String(decoding: try MKLightJSON.encode(JSONValue.object(settings)), as: UTF8.self)
        let session = self.session
        let (report, hash, systemJSON, library) = try await offMain {
            try session.locked { s in
                if try CABI.sessionSnapshotHash(s) != snapshot.hash.hex { _ = try CABI.sessionSetSystem(s, json: json) }
                let (report, hash) = try CABI.sessionApplySolves(s, settings: settingsJSON)
                return (report, hash, try CABI.sessionSystem(s), try CABI.sessionLibraryHash(s))
            }
        }
        let system = try decodeJSON(OpticalSystem.self, systemJSON)
        return SolveApplication(
            report: try decodeJSON(JSONValue.self, report),
            snapshot: ModelSnapshot(system: system, hash: SnapshotHash(hex: hash), libraryHash: library))
    }

    // MARK: EngineClient: catalogs

    public func catalogs() async -> [CatalogInfo] {
        let session = self.session
        guard let text = try? await offMain({ try session.locked { try CABI.sessionListCatalogs($0) } }) else { return [] }
        return (try? decodeJSON([CatalogInfo].self, text)) ?? []
    }

    public func loadCatalog(at url: URL) async throws -> CatalogInfo {
        let session = self.session
        let text = try await offMain { try session.locked { try CABI.sessionLoadCatalog($0, path: url.path) } }
        return try decodeJSON(CatalogInfo.self, text)
    }

    // MARK: EngineClient: lens prescription files

    public func importPrescription(_ data: Data, format: PrescriptionFormat) throws -> PrescriptionImport {
        try decodeJSON(PrescriptionImport.self, try CABI.importPrescription(data, format: format.rawValue))
    }

    public func importPrescription(at url: URL) async throws -> PrescriptionImport {
        let session = self.session
        let text = try await offMain { try session.locked { try CABI.sessionImportPrescription($0, path: url.path) } }
        return try decodeJSON(PrescriptionImport.self, text)
    }

    // MARK: EngineClient: analyses

    public func analysisDescriptors() async -> [AnalysisDescriptor] { core.descriptors }

    /// Runs the analysis in the engine. The settings go as given: what the
    /// request leaves out (a seed included) takes the engine's default, the
    /// one its schema declares and the settings form shows.
    ///
    /// Sets the session to `snapshot`, submits, and bridges the job: engine
    /// events → `JobReporter` progress; the terminal state → the result,
    /// taken (and its buffers wrapped) right away; `cancel()` →
    /// `mk_job_cancel`. Submission errors (unknown analysis, invalid
    /// settings, unavailable backend) fail the job at once. A job that
    /// fails while running throws the error of its own status code (an
    /// unknown glass is `notFound`, a numerical failure `compute`), as a
    /// submission error does.
    ///
    /// Nothing here blocks the caller (usually the main actor): encoding
    /// the system, waiting for the session lock (held during e.g. a catalog
    /// load) and the set-system-and-submit step all run in the job's task,
    /// the C calls through `offMain`. A job cancelled before it reached the
    /// engine is never submitted.
    public func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let core = self.core, session = self.session
        let requestJSON = Self.wireJSON(request)
        let submission = Submission()
        let (progress, continuation) = AsyncStream.makeStream(of: JobProgress.self, bufferingPolicy: .bufferingNewest(8))
        let reporter = JobReporter(continuation: continuation)
        let task = Task.detached(priority: .userInitiated) { () throws -> AnalysisOutput in
            defer { reporter.finish() }
            reporter.report(0, stage: "Queued")
            let (id, events): (UInt64, AsyncStream<JobEvent>)
            do {
                let json = try encodeSystem(snapshot.system)
                (id, events) = try await offMain {
                    try session.locked { s in
                        if submission.isCancelled { throw EngineError.cancelled }
                        if try CABI.sessionSnapshotHash(s) != snapshot.hash.hex { _ = try CABI.sessionSetSystem(s, json: json) }
                        return try core.hub.submit { try CABI.submit(s, requestJSON: requestJSON) }
                    }
                }
            } catch let error as EngineError where error.code == .invalidArgument {
                throw EngineError.invalidSettings([Self.message(of: error)])
            }
            // `cancel()` raced with the submission: cancel the engine job.
            if !submission.record(id) { try? CABI.jobCancel(core.pointer, id) }
            // `stop()` came before the engine job existed.
            if submission.stopRequested { try? CABI.jobStop(core.pointer, id) }
            // Safety net: should an event be missed, poll the job state.
            let poller = Task.detached {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(250))
                    if let (state, fraction) = try? CABI.jobStatus(core.pointer, id), state.isTerminal {
                        core.hub.deliver(id, JobEvent(state: state, progress: fraction, stage: ""))
                        return
                    }
                }
            }
            defer { poller.cancel() }
            for await event in events {
                switch event.state {
                case .queued: reporter.report(0, stage: "Queued")
                case .running: reporter.report(event.progress, stage: event.stage.isEmpty ? "Running" : event.stage)
                default: break
                }
            }
            if Task.isCancelled {
                core.hub.remove(id)
                try? CABI.jobRelease(core.pointer, id)
                throw EngineError.cancelled
            }
            let (text, buffers) = try await offMain { try CABI.jobTakeResult(core.pointer, id) }
            let envelope = try decodeJSON(JSONValue.self, text)
            guard let runInfoJSON = envelope["run_info"] else {
                throw EngineError.engine(.parse, "result envelope without run_info")
            }
            let runInfo = try decodeJSON(RunInfo.self, String(decoding: try MKLightJSON.encode(runInfoJSON), as: UTF8.self))
            let library: String? = if case .string(let h)? = envelope["library_hash"] { h } else { nil }
            reporter.report(1, stage: "Done")
            return AnalysisOutput(
                runInfo: runInfo, json: envelope["result"] ?? .null, buffers: buffers, libraryHash: library,
                engineRequest: request, backend: BackendReport(json: envelope["backend"]))
        }
        return AnalysisJob(
            engineRequest: request, snapshotHash: snapshot.hash, progress: progress, task: task,
            onCancel: { if let id = submission.cancel() { try? CABI.jobCancel(core.pointer, id) } },
            onStop: core.stoppable.contains(request.analysis)
                ? { @Sendable in
                    guard let id = submission.requestStop() else { return true }
                    return (try? CABI.jobStop(core.pointer, id)) != nil
                } : nil)
    }

    /// The request as the C ABI takes it: `{"analysis", "settings",
    /// "backend", "allow_backend_fallback", "fresh"}` (`fresh` only when set).
    static func wireJSON(_ request: EngineAnalysisRequest) -> String {
        var o: [String: JSONValue] = ["analysis": .string(request.analysis), "settings": .object(request.settings)]
        if request.fresh { o["fresh"] = .bool(true) }
        switch request.execution.backend {
        case .auto: break
        case .cpuReference: o["backend"] = .object(["kind": .string("cpu_reference")])
        case .gpu:
            o["backend"] = .object(["kind": .string("gpu"), "name": .string("metal")])
            o["allow_backend_fallback"] = .bool(true)
        }
        return (try? String(decoding: MKLightJSON.encode(JSONValue.object(o)), as: UTF8.self)) ?? "{}"
    }

    static func message(of error: EngineError) -> String {
        switch error {
        case .invalidModel(let m): m.joined(separator: "; ")
        case .unsupported(let m), .failed(let m), .engine(_, let m), .libraryUnavailable(let m): m
        case .invalidSettings(let m): m.joined(separator: "; ")
        case .cancelled: "cancelled"
        }
    }
}

/// The engine job id of an `AnalysisJob`, filled in once the job's task has
/// submitted it, and whether `cancel()` came first.
final class Submission: Sendable {
    private enum State { case pending, submitted(UInt64), cancelled }
    private let state = Mutex(State.pending)
    private let stop = Mutex(false)

    /// Whether `stop()` was requested (before or after submission).
    var stopRequested: Bool { stop.withLock { $0 } }

    /// Records a stop request; returns the engine job id when submitted.
    func requestStop() -> UInt64? {
        stop.withLock { $0 = true }
        return state.withLock { if case .submitted(let id) = $0 { id } else { nil } }
    }

    var isCancelled: Bool { state.withLock { if case .cancelled = $0 { true } else { false } } }

    /// Records the engine job id. `false` when the job was cancelled first
    /// (the caller then cancels the engine job).
    func record(_ id: UInt64) -> Bool {
        state.withLock { s in
            if case .cancelled = s { return false }
            s = .submitted(id)
            return true
        }
    }

    /// Marks the job cancelled; returns the engine job id if it was
    /// submitted already.
    func cancel() -> UInt64? {
        state.withLock { s in
            defer { s = .cancelled }
            if case .submitted(let id) = s { return id }
            return nil
        }
    }
}

func encodeSystem(_ system: OpticalSystem) throws -> String {
    String(decoding: try MKLightJSON.encode(system), as: UTF8.self)
}

/// Decodes engine JSON; a mismatch is an `EngineError` with code `parse`.
func decodeJSON<T: Decodable>(_ type: T.Type, _ text: String) throws -> T {
    do {
        return try MKLightJSON.decode(T.self, from: Data(text.utf8))
    } catch {
        throw EngineError.engine(.parse, "unexpected engine JSON for \(T.self): \(error)")
    }
}
