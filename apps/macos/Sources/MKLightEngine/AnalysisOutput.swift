import Foundation
import MKLightModel

/// A finished analysis: the engine's result (`result` of the envelope,
/// docs/RESULTS.md) as JSON with its binary buffers, plus the run metadata.
/// Views read both through tolerant accessors (`ResultReading.swift`).
public struct AnalysisOutput: Sendable, Hashable {
    public var runInfo: RunInfo
    /// The analysis-specific result JSON.
    public var json: JSONValue
    /// The result's binary buffers.
    public var buffers: [ResultBuffer]
    /// The glass library the result was computed with (engine envelope's
    /// `library_hash`), if the engine reports one.
    public var libraryHash: String?
    /// The request that produced this output. Engines that record their own
    /// normalised settings in `RunInfo.settings` (the Rust engine) set it,
    /// so staleness compares like with like.
    public var engineRequest: EngineAnalysisRequest?
    /// The envelope's backend report (requested, used, fallback reason), if
    /// the engine sent one.
    public var backend: BackendReport?

    public init(
        runInfo: RunInfo, json: JSONValue, buffers: [ResultBuffer] = [], libraryHash: String? = nil,
        engineRequest: EngineAnalysisRequest? = nil, backend: BackendReport? = nil
    ) {
        self.runInfo = runInfo
        self.json = json
        self.buffers = buffers
        self.libraryHash = libraryHash
        self.engineRequest = engineRequest
        self.backend = backend
    }

    public func buffer(named name: String) -> ResultBuffer? { buffers.first { $0.name == name } }

    /// Whether this output is current for `snapshot` and `request`: computed
    /// from the same system **and** glass library, with the same request
    /// (settings and backend preference). Without the request that produced
    /// it, the engine's recorded settings must contain every requested
    /// value. Otherwise it is stale.
    public func isCurrent(for snapshot: ModelSnapshot?, request: EngineAnalysisRequest) -> Bool {
        guard let snapshot, libraryMatches(snapshot), runInfo.snapshotHash == snapshot.hash else { return false }
        if let own = engineRequest { return own == request }
        return request.matches(runInfo)
    }

    private func libraryMatches(_ snapshot: ModelSnapshot) -> Bool {
        libraryHash == nil || snapshot.libraryHash == nil || libraryHash == snapshot.libraryHash
    }
}
