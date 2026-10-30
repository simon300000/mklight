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

    /// Engine implementations create snapshots; front ends receive them from
    /// `EngineClient.openSnapshot`.
    public init(system: OpticalSystem, hash: SnapshotHash, warnings: [String] = []) {
        self.system = system
        self.hash = hash
        self.warnings = warnings
    }

    public static func == (a: ModelSnapshot, b: ModelSnapshot) -> Bool { a.hash == b.hash }
    public func hash(into h: inout Hasher) { h.combine(hash) }
}

public struct EngineInfo: Sendable, Hashable {
    public var name: String
    public var version: String
    /// Compute backends available, e.g. `["cpu-f64", "metal"]`.
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

public enum EngineError: Error, Sendable, Hashable, LocalizedError {
    case cancelled
    case invalidModel([String])
    /// The analysis settings do not fit the snapshot (see
    /// `AnalysisRequest.problems(for:)`).
    case invalidSettings([String])
    case unsupported(String)
    case failed(String)

    public var errorDescription: String? {
        switch self {
        case .cancelled: "Cancelled"
        case .invalidModel(let issues): "Invalid model: " + issues.joined(separator: "; ")
        case .invalidSettings(let issues): "Invalid analysis settings: " + issues.joined(separator: "; ")
        case .unsupported(let s): "Unsupported: \(s)"
        case .failed(let s): s
        }
    }
}

/// The boundary between front ends and the optical engine.
///
/// Every computation goes through this protocol; the UI never computes optics
/// itself. Implementations: `MockEngine` (synthetic data, for UI work) and,
/// later, the Rust engine over its C ABI. All values crossing the boundary are
/// `Sendable` value types with the engine's JSON representation, so an FFI
/// implementation only has to move JSON (and bulk buffers) across.
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
}
