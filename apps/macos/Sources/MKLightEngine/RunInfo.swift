import Foundation
import MKLightModel

/// Floating-point model used for a computation (engine `Precision`).
public enum Precision: String, Codable, Sendable, Hashable {
    /// CPU double precision, the reference.
    case f64Reference = "f64_reference"
    /// GPU (or emulated) single precision within the documented error budget.
    case f32GPU = "f32_gpu"
}

/// Accuracy tier of a run (engine `Tier`, CONVENTIONS §7).
public enum Tier: String, Codable, Sendable, Hashable {
    /// Design-grade budgets and sampling.
    case design
    /// Cheaper sampling and aiming within the preview budgets; not for
    /// final decisions.
    case preview
}

/// How the real-aim solves of a run stopped (engine `AimStopKind`).
public enum AimStopKind: String, Codable, Sendable, Hashable {
    /// At the system's requested aiming tolerance.
    case requested
    /// At the residual derived per field and wavelength from the propagated
    /// error.
    case conditioned
}

/// One backend attempt that did not produce the published result (engine
/// `BackendAttempt`).
public struct BackendAttempt: Codable, Sendable, Hashable {
    /// Backend id (`gpu:metal`).
    public var backend: String
    /// `failed`, `skipped` or `reference_completion`.
    public var outcome: String
    public var reason: String
    /// Wall time spent in the attempt; 0 when it never started.
    public var elapsedMs: Double

    public init(backend: String, outcome: String, reason: String, elapsedMs: Double) {
        self.backend = backend
        self.outcome = outcome
        self.reason = reason
        self.elapsedMs = elapsedMs
    }

    private enum CodingKeys: String, CodingKey { case backend, outcome, reason, elapsedMs = "elapsed_ms" }
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
    /// Accuracy tier the result was computed at.
    public var tier: Tier
    /// Aim-stop policy the run's real-aim solves used.
    public var aimStop: AimStopKind
    public var warnings: [String]
    /// Backend attempts that did not compute this result, in order; their
    /// time is part of `durationMs`.
    public var failedAttempts: [BackendAttempt]

    public init(
        engineVersion: String, snapshotHash: SnapshotHash, analysis: String, settings: JSONValue, backend: String,
        precision: Precision, seed: UInt64?, startedUnixMs: UInt64, durationMs: Double, tier: Tier = .design,
        aimStop: AimStopKind = .requested, warnings: [String] = [], failedAttempts: [BackendAttempt] = []
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
        self.tier = tier
        self.aimStop = aimStop
        self.warnings = warnings
        self.failedAttempts = failedAttempts
    }

    private enum CodingKeys: String, CodingKey {
        case engineVersion = "engine_version"
        case snapshotHash = "snapshot_hash"
        case analysis, settings, backend, precision, seed
        case startedUnixMs = "started_unix_ms"
        case durationMs = "duration_ms"
        case tier
        case aimStop = "aim_stop"
        case warnings
        case failedAttempts = "failed_attempts"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        engineVersion = try c.decode(String.self, forKey: .engineVersion)
        snapshotHash = try c.decode(SnapshotHash.self, forKey: .snapshotHash)
        analysis = try c.decode(String.self, forKey: .analysis)
        settings = try c.decode(JSONValue.self, forKey: .settings)
        backend = try c.decode(String.self, forKey: .backend)
        // A precision this app does not know (a newer engine) is shown as
        // not-the-reference, with a warning, rather than failing the result.
        let precisionName = try c.decode(String.self, forKey: .precision)
        precision = Precision(rawValue: precisionName) ?? .f32GPU
        seed = try c.decodeIfPresent(UInt64.self, forKey: .seed)
        startedUnixMs = try c.decode(UInt64.self, forKey: .startedUnixMs)
        durationMs = try c.decode(Double.self, forKey: .durationMs)
        // Likewise a tier or aim stop this app does not know: shown as the
        // one that is not the default, with a warning.
        let tierName = try c.decodeIfPresent(String.self, forKey: .tier) ?? Tier.design.rawValue
        tier = Tier(rawValue: tierName) ?? .preview
        let aimStopName = try c.decodeIfPresent(String.self, forKey: .aimStop) ?? AimStopKind.requested.rawValue
        aimStop = AimStopKind(rawValue: aimStopName) ?? .conditioned
        warnings = try c.decodeIfPresent([String].self, forKey: .warnings) ?? []
        failedAttempts = try c.decodeIfPresent([BackendAttempt].self, forKey: .failedAttempts) ?? []
        if Precision(rawValue: precisionName) == nil { warnings.append("unknown precision '\(precisionName)'") }
        if Tier(rawValue: tierName) == nil { warnings.append("unknown tier '\(tierName)'") }
        if AimStopKind(rawValue: aimStopName) == nil { warnings.append("unknown aim stop '\(aimStopName)'") }
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
        try c.encode(tier, forKey: .tier)
        try c.encode(aimStop, forKey: .aimStop)
        try c.encode(warnings, forKey: .warnings)
        if !failedAttempts.isEmpty { try c.encode(failedAttempts, forKey: .failedAttempts) }
    }
}
