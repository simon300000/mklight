import Foundation
import MKLightModel

/// Floating-point model used for a computation (engine `Precision`).
public enum Precision: String, Codable, Sendable, Hashable {
    /// CPU double precision, the reference.
    case f64Reference = "f64_reference"
    /// GPU (or emulated) single precision within the documented error budget.
    case f32GPU = "f32_gpu"
}

/// Metadata attached to every result (mirror of the engine's `RunInfo`): what
/// was computed, from which snapshot, with which settings, backend and seed.
/// It makes results reproducible and tells the UI when a result is stale.
public struct RunInfo: Codable, Sendable, Hashable {
    public var engineVersion: String
    public var snapshotHash: SnapshotHash
    public var analysis: String
    public var settings: JSONValue
    public var backend: String
    public var precision: Precision
    public var seed: UInt64?
    public var startedUnixMs: UInt64
    public var durationMs: Double
    public var warnings: [String]

    public init(
        engineVersion: String, snapshotHash: SnapshotHash, analysis: String, settings: JSONValue, backend: String,
        precision: Precision, seed: UInt64?, startedUnixMs: UInt64, durationMs: Double, warnings: [String] = []
    ) {
        self.engineVersion = engineVersion
        self.snapshotHash = snapshotHash
        self.analysis = analysis
        self.settings = settings
        self.backend = backend
        self.precision = precision
        self.seed = seed
        self.startedUnixMs = startedUnixMs
        self.durationMs = durationMs
        self.warnings = warnings
    }

    private enum CodingKeys: String, CodingKey {
        case engineVersion = "engine_version"
        case snapshotHash = "snapshot_hash"
        case analysis, settings, backend, precision, seed
        case startedUnixMs = "started_unix_ms"
        case durationMs = "duration_ms"
        case warnings
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        engineVersion = try c.decode(String.self, forKey: .engineVersion)
        snapshotHash = try c.decode(SnapshotHash.self, forKey: .snapshotHash)
        analysis = try c.decode(String.self, forKey: .analysis)
        settings = try c.decode(JSONValue.self, forKey: .settings)
        backend = try c.decode(String.self, forKey: .backend)
        precision = try c.decode(Precision.self, forKey: .precision)
        seed = try c.decodeIfPresent(UInt64.self, forKey: .seed)
        startedUnixMs = try c.decode(UInt64.self, forKey: .startedUnixMs)
        durationMs = try c.decode(Double.self, forKey: .durationMs)
        warnings = try c.decodeIfPresent([String].self, forKey: .warnings) ?? []
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(engineVersion, forKey: .engineVersion)
        try c.encode(snapshotHash, forKey: .snapshotHash)
        try c.encode(analysis, forKey: .analysis)
        try c.encode(settings, forKey: .settings)
        try c.encode(backend, forKey: .backend)
        try c.encode(precision, forKey: .precision)
        try c.encode(seed, forKey: .seed)
        try c.encode(startedUnixMs, forKey: .startedUnixMs)
        try c.encode(durationMs, forKey: .durationMs)
        try c.encode(warnings, forKey: .warnings)
    }

    /// Whether this result was computed from `snapshot` with `request`'s
    /// settings, including its execution block (backend preference,
    /// precision and seed). A result that is not current is stale. Settings
    /// compare as exact JSON values, so 64-bit seeds are not rounded.
    public func isCurrent(for snapshot: ModelSnapshot?, request: AnalysisRequest) -> Bool {
        guard let snapshot, snapshot.hash == snapshotHash, analysis == request.kind.rawValue, seed == request.seed
        else { return false }
        return (try? request.settingsJSON()) == settings
    }
}
