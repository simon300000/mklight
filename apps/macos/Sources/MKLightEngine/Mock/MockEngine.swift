import CryptoKit
import Foundation
import MKLightModel

/// MOCK ENGINE — returns synthetic, illustrative data for UI development.
///
/// It honours the core of the `EngineClient` contract (validated snapshots,
/// analyses by name as async jobs with progress and cancellation,
/// reproducible seeds, run metadata; `MockByName.swift`), but its numbers
/// come from `MockOptics`, not from the Rust engine. Every result's `RunInfo`
/// has `backend == "mock"` and a warning saying so, and `info.isMock` is true
/// so front ends can label the data.
public struct MockEngine: EngineClient {
    /// Simulated computation time per analysis, spread over its stages.
    public var simulatedLatency: Duration

    public init(simulatedLatency: Duration = .milliseconds(300)) {
        self.simulatedLatency = simulatedLatency
    }

    public static let version = "0.1.0-mock"
    public static let mockWarning = "MOCK ENGINE: synthetic data, not an optical computation"

    public var info: EngineInfo {
        EngineInfo(name: "Mock engine (synthetic data)", version: Self.version, isMock: true)
    }

    public func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot {
        let issues = system.structuralIssues()
        let errors = issues.filter { $0.severity == .error }
        if !errors.isEmpty { throw EngineError.invalidModel(errors.map(\.description)) }
        let hash: SnapshotHash
        do {
            hash = try Self.hash(system)
        } catch {
            throw EngineError.invalidModel(["model cannot be serialised: \(error.localizedDescription)"])
        }
        return ModelSnapshot(system: system, hash: hash)
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
}
