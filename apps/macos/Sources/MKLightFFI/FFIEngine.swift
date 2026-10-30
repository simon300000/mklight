import Foundation
import MKLightEngine
import MKLightModel
import Synchronization

/// The Rust engine over its C ABI (`crates/mklight-ffi`, docs/FFI.md).
///
/// An `FFIEngine` is one engine handle plus one **session** (a project: the
/// current system, user glass catalogs, library hash). `makeDocumentEngine()`
/// gives each document its own session on the shared engine.
///
/// Lifetimes: the engine handle (`mk_engine_destroy`) and each session
/// (`mk_session_destroy`) are freed when the last `FFIEngine` value and the
/// last running job referring to them are gone. A session keeps its engine
/// alive.
///
/// Threading: `FFIEngine` is `Sendable` and may be used from any task. The C
/// engine and sessions are thread-safe. The `EngineClient` methods never
/// block their caller: their C calls (including waits for the session lock)
/// run on a global dispatch queue, never on the main thread. The synchronous
/// project helpers (`openProject`, `save`, `setSystem`, …) do block. Job events arrive on the
/// engine's `mklight-events` thread and are forwarded to each job's
/// `AsyncStream`s; consumers hop to the main actor themselves. A job runs
/// against the snapshot it was submitted with: the session's system is set
/// to that snapshot and the job submitted in one locked step.
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
        let backends: [String] =
            if case .array(let list) = core.backendsJSON.value {
                list.compactMap { b in
                    guard b["available"] == .bool(true), case .string(let id)? = b["id"] else { return nil }
                    return id
                }
            } else { [] }
        return EngineInfo(name: "MKLight engine", version: core.version, backends: backends, isMock: false)
    }

    /// `mk_engine_list_backends`, verbatim.
    public var backends: JSONValue { core.backendsJSON.value }

    // MARK: Sessions and projects

    /// A new untitled session on the same engine.
    public func newSession() throws -> FFIEngine {
        FFIEngine(core: core, session: try FFISessionHandle(core: core))
    }

    /// A new session holding `system` (e.g. a sample).
    public func newSession(system: OpticalSystem) throws -> FFIEngine {
        let e = try newSession()
        _ = try e.setSystem(system)
        return e
    }

    public func makeDocumentEngine() -> any EngineClient {
        (try? newSession()) ?? self
    }

    /// Opens a `.mklight` project (older schema versions are migrated, user
    /// catalogs loaded; missing ones are listed in `info.openWarnings`) as a
    /// new session.
    public func openProject(at url: URL) throws -> FFIEngine {
        let s = try CABI.sessionOpen(core.pointer, path: url.path)
        return FFIEngine(core: core, session: FFISessionHandle(core: core, pointer: s))
    }

    /// Atomically saves the session's project (`nil`: where it was opened
    /// from or last saved to).
    public func save(to url: URL? = nil) throws {
        try session.locked { try CABI.sessionSave($0, path: url?.path) }
    }

    /// The session's project document (same schema as the file).
    public func project() throws -> ProjectFile {
        let text = try session.locked { try CABI.sessionProject($0) }
        do {
            return try ProjectFile.decode(Data(text.utf8))
        } catch {
            throw EngineError.engine(.parse, "project from the engine: \(error.localizedDescription)")
        }
    }

    /// `mk_session_info_json`: path, dirty flag, hashes, open warnings,
    /// migration report, catalogs.
    public func sessionInfo() throws -> FFISessionInfo {
        try decodeJSON(FFISessionInfo.self, try session.locked { try CABI.sessionInfo($0) })
    }

    /// The session's current system.
    public func system() throws -> OpticalSystem {
        try decodeJSON(OpticalSystem.self, try session.locked { try CABI.sessionSystem($0) })
    }

    /// Replaces the session's system (validated by the engine) and returns
    /// its snapshot hash.
    @discardableResult
    public func setSystem(_ system: OpticalSystem) throws -> SnapshotHash {
        let json = try encodeSystem(system)
        return SnapshotHash(hex: try session.locked { try CABI.sessionSetSystem($0, json: json) })
    }

    /// Replaces a project section: `views`, `merit`, `variables`,
    /// `tolerances` or `metadata`.
    public func setProjectSection(_ section: String, _ value: JSONValue) throws {
        let json = String(decoding: try MKLightJSON.encode(value), as: UTF8.self)
        try session.locked { try CABI.sessionSetSection($0, section: section, json: json) }
    }

    public func currentSnapshotHash() throws -> SnapshotHash {
        SnapshotHash(hex: try session.locked { try CABI.sessionSnapshotHash($0) })
    }

    public func currentLibraryHash() throws -> String {
        try session.locked { try CABI.sessionLibraryHash($0) }
    }

    /// Glasses of one catalog, or of every catalog.
    public func glasses(in catalog: String? = nil) async throws -> [GlassInfo] {
        let session = self.session
        let text = try await offMain { try session.locked { try CABI.sessionListGlasses($0, catalog: catalog) } }
        return try decodeJSON([GlassInfo].self, text)
    }

    // MARK: EngineClient: snapshots and validation

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
            return ModelSnapshot(
                system: system, hash: SnapshotHash(hex: hash), warnings: local.map(\.description), libraryHash: library)
        } catch let error as EngineError where error.code == .invalidModel || error.code == .parse {
            // The engine stops at its first error; add the mirror's list.
            let message = error.code == .parse ? (error.errorDescription ?? "") : Self.message(of: error)
            let extra = local.filter { $0.severity == .error }.map(\.description).filter { $0 != message }
            throw EngineError.invalidModel([message] + extra)
        }
    }

    /// Every problem: the engine's `validate` analysis when the engine
    /// accepts the system (authoritative, with severities and surfaces),
    /// else the engine's rejection plus the mirror's structural checks.
    public func validate(_ system: OpticalSystem) async -> [ModelIssue] {
        let local = system.structuralIssues()
        guard let json = try? encodeSystem(system) else { return local }
        let session = self.session, core = self.core
        do {
            let text = try await offMain { () throws -> String in
                let id = try session.locked { s -> UInt64 in
                    _ = try CABI.sessionSetSystem(s, json: json)
                    return try CABI.submit(s, requestJSON: #"{"analysis":"validate"}"#)
                }
                _ = try CABI.jobWait(core.pointer, id, timeoutMs: -1)
                return try CABI.jobTakeResult(core.pointer, id).json
            }
            let envelope = try decodeJSON(JSONValue.self, text)
            guard case .array(let issues)? = envelope["result"]?["issues"] else { return local }
            return issues.map { i in
                let severity: ModelIssue.Severity = i["severity"] == .string("error") ? .error : .warning
                let surface = i["surface"]?.doubleValue.map { Int($0) }
                let code = if case .string(let c)? = i["code"] { "[\(c)] " } else { "" }
                let message = if case .string(let m)? = i["message"] { m } else { "" }
                return ModelIssue(severity, surface: surface, code + message)
            }
        } catch let error as EngineError {
            let message = Self.message(of: error)
            return [ModelIssue(.error, message)] + local.filter { $0.description != message }
        } catch {
            return local
        }
    }

    public func materialNames() async -> [String] {
        let names = ((try? await glasses()) ?? []).map(\.name)
        return Array(Set(names)).sorted()
    }

    public func applyParamEdits(_ edits: [ParamEdit], to snapshot: ModelSnapshot) async throws -> ModelSnapshot {
        let json = try encodeSystem(snapshot.system)
        let editsJSON = String(decoding: try MKLightJSON.encode(edits), as: UTF8.self)
        let session = self.session
        let (hash, systemJSON, library) = try await offMain {
            try session.locked { s in
                if try CABI.sessionSnapshotHash(s) != snapshot.hash.hex { _ = try CABI.sessionSetSystem(s, json: json) }
                let hash = try CABI.sessionApplyEdits(s, json: editsJSON)
                return (hash, try CABI.sessionSystem(s), try CABI.sessionLibraryHash(s))
            }
        }
        let system = try decodeJSON(OpticalSystem.self, systemJSON)
        return ModelSnapshot(
            system: system, hash: SnapshotHash(hex: hash),
            warnings: system.structuralIssues().map(\.description), libraryHash: library)
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
            snapshot: ModelSnapshot(
                system: system, hash: SnapshotHash(hex: hash), warnings: system.structuralIssues().map(\.description),
                libraryHash: library))
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

    // MARK: EngineClient: analyses

    public func analysisDescriptors() async -> [AnalysisDescriptor] { core.descriptors }

    /// A typed analysis runs as the engine analysis
    /// `kind.engineName(in: descriptors)` (`first_order`, or `paraxial` on
    /// engines that predate it) with the settings its schema accepts
    /// (`AnalysisRequest.engineSettings`).
    /// Its result decodes into the typed mirror when the JSON matches, and
    /// is shown generically otherwise. An analysis the engine lists as
    /// unavailable fails with `EngineError.unsupported` and its reason.
    public func submit(_ request: AnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let problems = request.problems(for: snapshot.system)
        let name = request.kind.engineName(in: core.descriptors)
        guard problems.isEmpty else {
            return AnalysisJob.run(request: request, snapshot: snapshot) { _ in throw EngineError.invalidSettings(problems) }
        }
        guard let descriptor = core.descriptor(name), descriptor.available else {
            let why = core.descriptor(name)?.unavailableReason ?? "the engine has no analysis '\(name)'"
            return AnalysisJob.run(request: request, snapshot: snapshot) { _ in
                throw EngineError.unsupported("\(request.kind.title) is not available in this engine build: \(why)")
            }
        }
        let wire = WireRequest(
            analysis: name, settings: request.engineSettings(for: descriptor), execution: request.execution)
        let kind = request.kind
        return start(wire, snapshot: snapshot, request: request, engineRequest: nil) { json, buffers in
            AnalysisResult.fromEngine(kind: kind, json: json, buffers: buffers)
        }
    }

    public func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        var settings = request.settings
        if let d = core.descriptor(request.analysis), settings["seed"] == nil,
            d.settingsFields.contains(where: { $0.name == "seed" })
        {
            settings["seed"] = Int64(exactly: request.execution.seed).map(JSONValue.integer) ?? .unsigned(request.execution.seed)
        }
        let wire = WireRequest(analysis: request.analysis, settings: settings, execution: request.execution)
        let name = request.analysis
        return start(wire, snapshot: snapshot, request: nil, engineRequest: request) { json, buffers in
            .generic(GenericResult(analysis: name, json: json, buffers: buffers))
        }
    }

    // MARK: Jobs

    /// `{"analysis", "settings", "backend", "allow_backend_fallback"}`.
    struct WireRequest: Sendable {
        var analysis: String
        var settings: [String: JSONValue]
        var execution: ExecutionSettings

        var json: String {
            var o: [String: JSONValue] = ["analysis": .string(analysis), "settings": .object(settings)]
            switch execution.backend {
            case .auto: break
            case .cpuReference: o["backend"] = .object(["kind": .string("cpu_reference")])
            case .gpu:
                o["backend"] = .object(["kind": .string("gpu"), "name": .string("metal")])
                o["allow_backend_fallback"] = .bool(true)
            }
            return (try? String(decoding: MKLightJSON.encode(JSONValue.object(o)), as: UTF8.self)) ?? "{}"
        }
    }

    /// Sets the session to `snapshot`, submits, and bridges the job: engine
    /// events → `JobReporter` progress; the terminal state → the result,
    /// taken (and its buffers wrapped) right away; `cancel()` →
    /// `mk_job_cancel`. Submission errors (unknown analysis, invalid
    /// settings, unavailable backend) fail the job at once.
    ///
    /// Nothing here blocks the caller (usually the main actor): encoding
    /// the system, waiting for the session lock (held during e.g. a catalog
    /// load) and the set-system-and-submit step all run in the job's task,
    /// the C calls through `offMain`. A job cancelled before it reached the
    /// engine is never submitted.
    private func start(
        _ wire: WireRequest, snapshot: ModelSnapshot, request: AnalysisRequest?, engineRequest: EngineAnalysisRequest?,
        decode: @escaping @Sendable (JSONValue, [ResultBuffer]) -> AnalysisResult
    ) -> AnalysisJob {
        let core = self.core, session = self.session
        let requestJSON = wire.json
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
                runInfo: runInfo, result: decode(envelope["result"] ?? .null, buffers), libraryHash: library,
                request: request, engineRequest: engineRequest, backend: BackendReport(json: envelope["backend"]))
        }
        return AnalysisJob(
            request: request, engineRequest: engineRequest, snapshotHash: snapshot.hash, progress: progress, task: task,
            onCancel: { if let id = submission.cancel() { try? CABI.jobCancel(core.pointer, id) } },
            onStop: Self.stoppable.contains(wire.analysis)
                ? { @Sendable in
                    guard let id = submission.requestStop() else { return true }
                    return (try? CABI.jobStop(core.pointer, id)) != nil
                } : nil)
    }

    /// Analyses `mk_job_stop` can stop (ABI 1.2).
    static let stoppable: Set<String> = ["optimize", "tolerance"]

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

/// `mk_session_info_json`.
public struct FFISessionInfo: Codable, Sendable, Hashable {
    public struct Migration: Codable, Sendable, Hashable {
        public var fromVersion: Int
        public var toVersion: Int
        public var notes: [String]
        private enum CodingKeys: String, CodingKey { case fromVersion = "from_version", toVersion = "to_version", notes }
    }
    public var sessionID: UInt64
    public var path: String?
    public var dirty: Bool
    public var snapshotHash: String
    public var libraryHash: String
    public var name: String
    public var openWarnings: [String]
    public var migration: Migration?
    public var catalogs: [CatalogInfo]

    private enum CodingKeys: String, CodingKey {
        case sessionID = "session_id", path, dirty, snapshotHash = "snapshot_hash", libraryHash = "library_hash", name
        case openWarnings = "open_warnings", migration, catalogs
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
