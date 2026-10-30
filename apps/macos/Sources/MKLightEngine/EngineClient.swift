import Foundation
import MKLightModel

/// Content hash of a model snapshot, as hex. Produced by the engine (BLAKE3 of
/// the canonical JSON in the Rust engine); front ends only compare hashes.
public struct SnapshotHash: Codable, Sendable, Hashable, CustomStringConvertible {
    public var hex: String
    public init(hex: String) { self.hex = hex }
    public init(from decoder: any Decoder) throws { hex = try decoder.singleValueContainer().decode(String.self) }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(hex)
    }
    /// First 12 hex digits, for display.
    public var short: String { String(hex.prefix(12)) }
    public var description: String { hex }
}

/// An immutable, engine-validated model snapshot. Analyses only ever run
/// against snapshots, never against the live document.
public struct ModelSnapshot: Sendable, Hashable {
    public let system: OpticalSystem
    public let hash: SnapshotHash
    /// Non-fatal problems the engine found while validating.
    public let warnings: [String]
    /// Hash of the glass library the snapshot was taken with (the engine's
    /// `library_hash`; `nil` for engines without user catalogs). A result is
    /// current only while both hashes match (docs/FFI.md, result envelope).
    public let libraryHash: String?

    /// Engine implementations create snapshots; front ends receive them from
    /// `EngineClient.openSnapshot`.
    public init(system: OpticalSystem, hash: SnapshotHash, warnings: [String] = [], libraryHash: String? = nil) {
        self.system = system
        self.hash = hash
        self.warnings = warnings
        self.libraryHash = libraryHash
    }

    public static func == (a: ModelSnapshot, b: ModelSnapshot) -> Bool {
        a.hash == b.hash && a.libraryHash == b.libraryHash
    }
    public func hash(into h: inout Hasher) {
        h.combine(hash)
        h.combine(libraryHash)
    }
}

public struct EngineInfo: Sendable, Hashable {
    public var name: String
    public var version: String
    /// Compute backends available, e.g. `["cpu_reference", "metal"]`.
    public var backends: [String]
    /// True for engines that return synthetic data (`MockEngine`). Front ends
    /// must label such results.
    public var isMock: Bool

    public init(name: String, version: String, backends: [String], isMock: Bool) {
        self.name = name
        self.version = version
        self.backends = backends
        self.isMock = isMock
    }
}

/// Status codes of the engine's C ABI (`MKStatus`, docs/FFI.md).
public enum EngineErrorCode: Int32, Sendable, Hashable, CaseIterable, CustomStringConvertible {
    case invalidArgument = 1
    case parse = 2
    case invalidModel = 3
    case notFound = 4
    case io = 5
    case unsupported = 6
    case cancelled = 7
    case timeout = 8
    case notReady = 9
    case jobFailed = 10
    case compute = 11
    case panic = 12
    case `internal` = 13

    public var description: String {
        switch self {
        case .invalidArgument: "invalid argument"
        case .parse: "parse error"
        case .invalidModel: "invalid model"
        case .notFound: "not found"
        case .io: "I/O error"
        case .unsupported: "unsupported"
        case .cancelled: "cancelled"
        case .timeout: "timeout"
        case .notReady: "not ready"
        case .jobFailed: "job failed"
        case .compute: "computation failed"
        case .panic: "internal panic"
        case .internal: "internal error"
        }
    }
}

public enum EngineError: Error, Sendable, Hashable, LocalizedError {
    case cancelled
    case invalidModel([String])
    /// The analysis settings do not fit the snapshot (see
    /// `AnalysisRequest.problems(for:)`).
    case invalidSettings([String])
    case unsupported(String)
    case failed(String)
    /// Any other engine status, with the engine's message.
    case engine(EngineErrorCode, String)
    /// The engine library is not linked into this build, or its ABI major
    /// version differs from the one the app was compiled against.
    case libraryUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .cancelled: "Cancelled"
        case .invalidModel(let issues): "Invalid model: " + issues.joined(separator: "; ")
        case .invalidSettings(let issues): "Invalid analysis settings: " + issues.joined(separator: "; ")
        case .unsupported(let s): "Unsupported: \(s)"
        case .failed(let s): s
        case .engine(let code, let s): s.isEmpty ? "Engine error: \(code)" : s
        case .libraryUnavailable(let s): "Engine library unavailable: \(s)"
        }
    }

    /// The engine status code this error corresponds to (`nil` for errors
    /// raised by the front end itself).
    public var code: EngineErrorCode? {
        switch self {
        case .cancelled: .cancelled
        case .invalidModel: .invalidModel
        case .invalidSettings: .invalidArgument
        case .unsupported: .unsupported
        case .failed: .jobFailed
        case .engine(let c, _): c
        case .libraryUnavailable: nil
        }
    }

    /// Maps an engine status and message to the most specific case:
    /// cancellation, invalid model and unsupported get their own cases, a
    /// failed job keeps its message, everything else keeps its code.
    public static func from(code: EngineErrorCode, message: String) -> EngineError {
        switch code {
        case .cancelled: .cancelled
        case .invalidModel: .invalidModel([message])
        case .unsupported: .unsupported(message)
        case .jobFailed: .failed(message)
        default: .engine(code, message)
        }
    }
}

/// The boundary between front ends and the optical engine.
///
/// Every computation goes through this protocol; the UI never computes optics
/// itself. Implementations: `MockEngine` (synthetic data, for UI work) and
/// `FFIEngine` (the Rust engine over its C ABI, module `MKLightFFI`). All
/// values crossing the boundary are `Sendable` value types with the engine's
/// JSON representation.
///
/// The requirements after `materialNames()` have default implementations
/// (in an extension) for engines without those capabilities.
public protocol EngineClient: Sendable {
    var info: EngineInfo { get }

    /// Validates `system` and freezes it into a hashed snapshot. Throws
    /// `EngineError.invalidModel` with every structural problem found.
    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot

    /// Validation only: every problem, without throwing.
    func validate(_ system: OpticalSystem) async -> [ModelIssue]

    /// Starts an analysis of `snapshot`. The job runs asynchronously; observe
    /// its progress and await its result through the returned handle. A
    /// request whose `problems(for:)` is not empty fails immediately with
    /// `EngineError.invalidSettings`.
    func submit(_ request: AnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob

    /// Names of the glasses the engine can resolve (for completion and
    /// validation in editors).
    func materialNames() async -> [String]

    /// An engine for one document: same engine, its own session state (user
    /// glass catalogs, library hash). Engines without per-document state
    /// return themselves.
    func makeDocumentEngine() -> any EngineClient

    /// Every analysis the engine knows, with its settings schema. Analyses
    /// of later phases are listed with `available == false`.
    func analysisDescriptors() async -> [AnalysisDescriptor]

    /// Starts any engine analysis by name, with JSON settings. Unknown
    /// analyses and invalid settings fail the job at once.
    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob

    /// Applies parameter edits atomically (all or none) and returns the
    /// snapshot of the edited system.
    func applyParamEdits(_ edits: [ParamEdit], to snapshot: ModelSnapshot) async throws -> ModelSnapshot

    /// Glass catalogs of this (document) engine, built-in first.
    func catalogs() async -> [CatalogInfo]

    /// Loads a user glass catalog (AGF text, or JSON when the file ends in
    /// `.json`), replacing one of the same name. Changes the library hash.
    func loadCatalog(at url: URL) async throws -> CatalogInfo

    /// Solves the snapshot's system (docs/SOLVES.md) and returns the solve
    /// report (the `solves` result) with the snapshot of the solved system.
    /// The `solves` analysis is the dry run of this.
    func applySolves(settings: [String: JSONValue], to snapshot: ModelSnapshot) async throws -> SolveApplication
}

/// The outcome of `EngineClient.applySolves`.
public struct SolveApplication: Sendable {
    /// The `solves` report (`outcomes`, `warnings`, …; docs/RESULTS.md).
    public var report: JSONValue
    /// The solved system's snapshot.
    public var snapshot: ModelSnapshot

    public init(report: JSONValue, snapshot: ModelSnapshot) {
        self.report = report
        self.snapshot = snapshot
    }
}

extension EngineClient {
    public func makeDocumentEngine() -> any EngineClient { self }

    /// The analyses of `AnalysisKind`, all available, without schemas.
    public func analysisDescriptors() async -> [AnalysisDescriptor] {
        AnalysisKind.allCases.map {
            AnalysisDescriptor(name: $0.engineName, title: $0.title, category: "analysis", available: true)
        }
    }

    public func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let message = "\(info.name) cannot run '\(request.analysis)' by name"
        return AnalysisJob.run(engineRequest: request, snapshot: snapshot) { _ in throw EngineError.unsupported(message) }
    }

    public func applyParamEdits(_ edits: [ParamEdit], to snapshot: ModelSnapshot) async throws -> ModelSnapshot {
        throw EngineError.unsupported("\(info.name) does not apply parameter edits")
    }

    public func catalogs() async -> [CatalogInfo] { [] }

    public func loadCatalog(at url: URL) async throws -> CatalogInfo {
        throw EngineError.unsupported("\(info.name) cannot load glass catalogs")
    }

    /// Runs the `solves` analysis and writes every outcome's value into the
    /// system (engines with a native apply-solves call override this).
    public func applySolves(settings: [String: JSONValue], to snapshot: ModelSnapshot) async throws -> SolveApplication {
        let job = submit(EngineAnalysisRequest(analysis: "solves", settings: settings), on: snapshot)
        let report = try await job.result().json
        var system = snapshot.system
        for outcome in report["outcomes"]?.array ?? [] {
            guard let target = outcome["target"], let value = outcome["value"]?.nf, value.isFinite,
                let path = try? MKLightJSON.decode(ParamPath.self, from: MKLightJSON.encode(target))
            else { continue }
            system.setValue(value, at: path)
        }
        return SolveApplication(report: report, snapshot: try await openSnapshot(system))
    }
}
