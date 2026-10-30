import CryptoKit
import Foundation
import MKLightModel

/// MOCK ENGINE — returns synthetic, illustrative data for UI development.
///
/// It honours the full `EngineClient` contract (validation, snapshots, async
/// jobs with progress and cancellation, reproducible seeds, run metadata), but
/// its numbers come from `MockOptics`, not from the Rust engine. Every result's
/// `RunInfo` has `backend == "mock"` and a warning saying so, and
/// `info.isMock` is true so front ends can label the data.
public struct MockEngine: EngineClient {
    /// Simulated computation time per analysis, spread over its stages.
    public var simulatedLatency: Duration

    public init(simulatedLatency: Duration = .milliseconds(300)) {
        self.simulatedLatency = simulatedLatency
    }

    public static let version = "0.1.0-mock"
    public static let mockWarning = "MOCK ENGINE: synthetic data, not an optical computation"

    public var info: EngineInfo {
        EngineInfo(name: "Mock engine (synthetic data)", version: Self.version, backends: ["mock"], isMock: true)
    }

    public func validate(_ system: OpticalSystem) async -> [ModelIssue] {
        system.structuralIssues()
    }

    public func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot {
        let issues = system.structuralIssues()
        let errors = issues.filter { $0.severity == .error }
        if !errors.isEmpty { throw EngineError.invalidModel(errors.map(\.description)) }
        return ModelSnapshot(
            system: system, hash: try Self.hash(system), warnings: issues.map(\.description))
    }

    /// SHA-256 of the sorted-key JSON. (The Rust engine uses BLAKE3 of its
    /// own canonical JSON; hashes are only compared, never interpreted.)
    public static func hash(_ system: OpticalSystem) throws -> SnapshotHash {
        let digest = SHA256.hash(data: try MKLightJSON.encode(system))
        return SnapshotHash(hex: digest.map { String(format: "%02x", $0) }.joined())
    }

    public func materialNames() async -> [String] {
        MockOptics.catalog.keys.sorted()
    }

    public func submit(_ request: AnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let latency = simulatedLatency
        return AnalysisJob.run(request: request, snapshot: snapshot) { reporter in
            let started = Date()
            let stages = Self.stages(for: request.kind)
            let step = latency / max(stages.count, 1)
            for (k, stage) in stages.enumerated() {
                try reporter.checkCancelled()
                reporter.report(Double(k) / Double(stages.count + 1), stage: stage)
                if step > .zero { try await Task.sleep(for: step) }
            }
            try reporter.checkCancelled()
            reporter.report(Double(stages.count) / Double(stages.count + 1), stage: "Assembling result")
            let optics = MockOptics(system: snapshot.system)
            let result = Self.compute(request, optics)
            var warnings = [Self.mockWarning] + optics.warnings
            if Self.configuration(of: request) != 0 { warnings.append("mock engine ignores configurations") }
            let info = RunInfo(
                engineVersion: Self.version, snapshotHash: snapshot.hash, analysis: request.kind.rawValue,
                settings: try request.settingsJSON(), backend: "mock", precision: .f64Reference, seed: request.seed,
                startedUnixMs: UInt64(started.timeIntervalSince1970 * 1000),
                durationMs: Date().timeIntervalSince(started) * 1000, warnings: warnings)
            return AnalysisOutput(runInfo: info, result: result)
        }
    }

    static func stages(for kind: AnalysisKind) -> [String] {
        switch kind {
        case .layout: ["Resolving system", "Tracing layout rays"]
        case .spotDiagram: ["Resolving system", "Aiming rays", "Tracing pupil grid", "Reducing"]
        case .rayFan, .opdFan: ["Resolving system", "Tracing fans", "Reducing"]
        case .fieldCurvature: ["Resolving system", "Tracing field samples", "Locating foci"]
        case .mtf: ["Resolving system", "Sampling wavefront", "Autocorrelating pupil", "Averaging fields"]
        case .psf: ["Resolving system", "Sampling wavefront", "Propagating to image"]
        case .firstOrder: ["Paraxial trace"]
        }
    }

    static func configuration(of r: AnalysisRequest) -> Int {
        switch r {
        case .layout(let s): s.configuration
        case .spotDiagram(let s): s.configuration
        case .rayFan(let s), .opdFan(let s): s.configuration
        case .fieldCurvature(let s): s.configuration
        case .mtf(let s): s.configuration
        case .psf(let s): s.configuration
        case .firstOrder(let s): s.configuration
        }
    }

    static func compute(_ r: AnalysisRequest, _ o: MockOptics) -> AnalysisResult {
        switch r {
        case .layout(let s): .layout(o.layout(s))
        case .spotDiagram(let s): .spotDiagram(o.spot(s))
        case .rayFan(let s): .rayFan(o.rayFan(s))
        case .opdFan(let s): .opdFan(o.opdFan(s))
        case .fieldCurvature(let s): .fieldCurvature(o.fieldCurvature(s))
        case .mtf(let s): .mtf(o.mtf(s))
        case .psf(let s): .psf(o.psf(s))
        case .firstOrder(let s): .firstOrder(o.firstOrder(s))
        }
    }
}
