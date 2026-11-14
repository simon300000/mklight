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
    /// Hash of the glass library the snapshot was taken with (the engine's
    /// `library_hash`; `nil` for engines without user catalogs). A result is
    /// current only while both hashes match (docs/FFI.md, result envelope).
    public let libraryHash: String?

    /// Engine implementations create snapshots; front ends receive them from
    /// `EngineClient.openSnapshot`.
    public init(system: OpticalSystem, hash: SnapshotHash, libraryHash: String? = nil) {
        self.system = system
        self.hash = hash
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
    /// True for engines that return synthetic data (`MockEngine`). Front ends
    /// must label such results.
    public var isMock: Bool

    public init(name: String, version: String, isMock: Bool) {
        self.name = name
        self.version = version
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
    /// The engine refused the analysis settings.
    case invalidSettings([String])
    case unsupported(String)
    /// A job ended with an internal error (status `jobFailed`).
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
    /// cancellation, invalid model and unsupported get their own cases,
    /// everything else keeps its code. A job that failed carries the status
    /// of its own error; `jobFailed` (`failed`) is a job that ended with an
    /// internal error.
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
/// values crossing the boundary are `Sendable` value types: analyses are
/// requested by name with JSON settings, and results come back as the
/// engine's JSON with typed buffers.
///
/// The requirements after `materialNames()` have default implementations
/// (in an extension) for engines without those capabilities.
public protocol EngineClient: Sendable {
    var info: EngineInfo { get }

    /// Validates `system` and freezes it into a hashed snapshot. Throws
    /// `EngineError.invalidModel` with every structural problem found.
    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot

    /// Every analysis the engine knows, with its settings schema. Analyses
    /// of later phases are listed with `available == false`.
    func analysisDescriptors() async -> [AnalysisDescriptor]

    /// Starts an engine analysis of `snapshot` by name, with JSON settings.
    /// The job runs asynchronously; observe its progress and await its
    /// result through the returned handle. Unknown analyses and invalid
    /// settings fail the job at once.
    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob

    /// Names of the glasses the engine can resolve (for completion and
    /// validation in editors).
    func materialNames() async -> [String]

    /// An engine for one document: same engine, its own session state (user
    /// glass catalogs, library hash). Engines without per-document state
    /// return themselves.
    func makeDocumentEngine() -> any EngineClient

    /// Glass catalogs of this (document) engine, built-in first.
    func catalogs() async -> [CatalogInfo]

    /// Loads a user glass catalog (AGF text, or JSON when the file ends in
    /// `.json`), replacing one of the same name. Changes the library hash.
    func loadCatalog(at url: URL) async throws -> CatalogInfo

    /// Removes a catalog the project records from this session.
    func removeCatalog(named name: String) async throws

    /// Reads a catalog again from its file and reports what changed.
    func reloadCatalog(named name: String) async throws -> CatalogChanges

    /// Makes the session's app-wide library catalogs exactly these files,
    /// in order (unchanged files are not read again).
    func setLibraryCatalogs(_ files: [URL]) async -> LibrarySync

    /// The user's availability overrides (status, exclusion from glass
    /// substitution), applied to the session's glass library.
    func setGlassOverrides(_ overrides: [GlassOverride]) async throws

    /// Glass table rows of one catalog, or of every catalog.
    func glassRows(in catalog: String?) async throws -> [GlassRow]

    /// Everything about one glass (`catalog` nil: the first that has it).
    func glassDetails(catalog: String?, name: String) async throws -> GlassDetails

    /// The library catalogs the design uses that the project does not
    /// record yet; a saved project records them.
    func libraryCatalogsInUse() async -> [CatalogReference]

    /// A catalog glass (engine JSON) from an editor's specification
    /// (`GlassSpec`, docs/FFI.md), with n_d and V_d computed.
    func glass(from spec: JSONValue) throws -> JSONValue

    /// A catalog in the engine's JSON form as AGF text.
    func agf(of catalog: JSONValue) throws -> AGFExport

    /// Solves the snapshot's system (docs/SOLVES.md) and returns the solve
    /// report (the `solves` result) with the snapshot of the solved system.
    /// The `solves` analysis is the dry run of this.
    func applySolves(settings: [String: JSONValue], to snapshot: ModelSnapshot) async throws -> SolveApplication

    /// Imports a lens prescription file (`.zmx` or `.seq`, any common text
    /// encoding) with the built-in glass catalog. Content that does not map
    /// is listed in the result's warnings; only text without surfaces
    /// throws. Synchronous, because documents read their files
    /// synchronously.
    func importPrescription(_ data: Data, format: PrescriptionFormat) throws -> PrescriptionImport

    /// Imports a lens prescription file into this (document) engine's
    /// session, resolving glasses against its user catalogs.
    func importPrescription(at url: URL) async throws -> PrescriptionImport

    /// Starts creating the GPU's compute kernels in the background (once
    /// per engine), saving the compiled kernels in `cacheDirectory` so later
    /// launches load them. Progress: `gpuPreparation()`.
    func prepareGPU(cacheDirectory: URL?)

    /// The state of `prepareGPU`.
    func gpuPreparation() -> GPUPreparation
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

    /// Engines without a GPU have nothing to prepare.
    public func prepareGPU(cacheDirectory: URL?) {}
    public func gpuPreparation() -> GPUPreparation { .idle }

    public func catalogs() async -> [CatalogInfo] { [] }

    public func loadCatalog(at url: URL) async throws -> CatalogInfo {
        throw EngineError.unsupported("\(info.name) cannot load glass catalogs")
    }

    public func removeCatalog(named name: String) async throws {
        throw EngineError.unsupported("\(info.name) cannot remove glass catalogs")
    }

    public func reloadCatalog(named name: String) async throws -> CatalogChanges {
        throw EngineError.unsupported("\(info.name) cannot reload glass catalogs")
    }

    public func setLibraryCatalogs(_ files: [URL]) async -> LibrarySync { LibrarySync() }

    public func setGlassOverrides(_ overrides: [GlassOverride]) async throws {}

    public func glassRows(in catalog: String?) async throws -> [GlassRow] { [] }

    public func glassDetails(catalog: String?, name: String) async throws -> GlassDetails {
        throw EngineError.unsupported("\(info.name) has no glass data")
    }

    public func libraryCatalogsInUse() async -> [CatalogReference] { [] }

    public func glass(from spec: JSONValue) throws -> JSONValue {
        throw EngineError.unsupported("\(info.name) cannot make glasses")
    }

    public func agf(of catalog: JSONValue) throws -> AGFExport {
        throw EngineError.unsupported("\(info.name) cannot write AGF catalogs")
    }

    public func importPrescription(_ data: Data, format: PrescriptionFormat) throws -> PrescriptionImport {
        throw EngineError.unsupported("\(info.name) cannot import lens prescription files")
    }

    public func importPrescription(at url: URL) async throws -> PrescriptionImport {
        guard let format = PrescriptionFormat(fileExtension: url.pathExtension) else {
            throw EngineError.unsupported("\(url.lastPathComponent) is not a lens prescription file")
        }
        return try importPrescription(try Data(contentsOf: url), format: format)
    }

    /// Runs the `solves` analysis and writes every outcome's value into the
    /// system (engines with a native apply-solves call override this).
    public func applySolves(settings: [String: JSONValue], to snapshot: ModelSnapshot) async throws -> SolveApplication {
        let job = submit(EngineAnalysisRequest(analysis: "solves", settings: settings), on: snapshot)
        let report = try await job.result().json
        var system = snapshot.system
        for outcome in report["outcomes"]?.array ?? [] {
            guard let target = outcome["target"], let value = outcome["value"]?.nf, value.isFinite,
                let path = try? target.decode(ParamPath.self)
            else { continue }
            system.setValue(value, at: path)
        }
        return SolveApplication(report: report, snapshot: try await openSnapshot(system))
    }
}
