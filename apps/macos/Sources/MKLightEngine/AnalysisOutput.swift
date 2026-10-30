import Foundation
import MKLightModel

// MARK: - Envelope

public enum AnalysisResult: Codable, Sendable, Hashable {
    case layout(LayoutScene)
    case spotDiagram(SpotDiagramResult)
    case rayFan(FanResult)
    case opdFan(FanResult)
    case fieldCurvature(FieldCurvatureResult)
    case mtf(MTFResult)
    case psf(PSFResult)
    case firstOrder(FirstOrderResult)
    /// Any other engine result (or a typed one the mirror cannot decode).
    case generic(GenericResult)

    /// The typed analysis, `nil` for generic results.
    public var kind: AnalysisKind? {
        switch self {
        case .layout: .layout
        case .spotDiagram: .spotDiagram
        case .rayFan: .rayFan
        case .opdFan: .opdFan
        case .fieldCurvature: .fieldCurvature
        case .mtf: .mtf
        case .psf: .psf
        case .firstOrder: .firstOrder
        case .generic: nil
        }
    }

    /// The analysis name (`AnalysisKind.rawValue`, or the engine name of a
    /// generic result).
    public var analysisName: String {
        if case .generic(let g) = self { return g.analysis }
        return kind!.rawValue
    }

    private enum CodingKeys: String, CodingKey { case analysis, data }

    /// Unknown analysis names decode as `.generic` (without buffers, which
    /// do not travel through JSON).
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let name = try c.decode(String.self, forKey: .analysis)
        guard let kind = AnalysisKind(rawValue: name) else {
            self = .generic(GenericResult(analysis: name, json: try c.decode(JSONValue.self, forKey: .data)))
            return
        }
        switch kind {
        case .layout: self = .layout(try c.decode(LayoutScene.self, forKey: .data))
        case .spotDiagram: self = .spotDiagram(try c.decode(SpotDiagramResult.self, forKey: .data))
        case .rayFan: self = .rayFan(try c.decode(FanResult.self, forKey: .data))
        case .opdFan: self = .opdFan(try c.decode(FanResult.self, forKey: .data))
        case .fieldCurvature: self = .fieldCurvature(try c.decode(FieldCurvatureResult.self, forKey: .data))
        case .mtf: self = .mtf(try c.decode(MTFResult.self, forKey: .data))
        case .psf: self = .psf(try c.decode(PSFResult.self, forKey: .data))
        case .firstOrder: self = .firstOrder(try c.decode(FirstOrderResult.self, forKey: .data))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(analysisName, forKey: .analysis)
        switch self {
        case .layout(let v): try c.encode(v, forKey: .data)
        case .spotDiagram(let v): try c.encode(v, forKey: .data)
        case .rayFan(let v), .opdFan(let v): try c.encode(v, forKey: .data)
        case .fieldCurvature(let v): try c.encode(v, forKey: .data)
        case .mtf(let v): try c.encode(v, forKey: .data)
        case .psf(let v): try c.encode(v, forKey: .data)
        case .firstOrder(let v): try c.encode(v, forKey: .data)
        case .generic(let g): try c.encode(g.json, forKey: .data)
        }
    }
}

/// A finished analysis: its data plus the run metadata.
public struct AnalysisOutput: Codable, Sendable, Hashable {
    public var runInfo: RunInfo
    public var result: AnalysisResult
    /// The glass library the result was computed with (engine envelope's
    /// `library_hash`), if the engine reports one.
    public var libraryHash: String?
    /// The front-end request that produced this output. Engines that record
    /// their own normalised settings in `RunInfo.settings` (the Rust engine)
    /// set it, so staleness compares like with like. Not serialised.
    public var request: AnalysisRequest?
    /// Same, for a request by name.
    public var engineRequest: EngineAnalysisRequest?

    public init(
        runInfo: RunInfo, result: AnalysisResult, libraryHash: String? = nil, request: AnalysisRequest? = nil,
        engineRequest: EngineAnalysisRequest? = nil
    ) {
        self.runInfo = runInfo
        self.result = result
        self.libraryHash = libraryHash
        self.request = request
        self.engineRequest = engineRequest
    }

    private enum CodingKeys: String, CodingKey { case runInfo = "run_info", result, libraryHash = "library_hash" }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        runInfo = try c.decode(RunInfo.self, forKey: .runInfo)
        result = try c.decode(AnalysisResult.self, forKey: .result)
        libraryHash = try c.decodeIfPresent(String.self, forKey: .libraryHash)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(runInfo, forKey: .runInfo)
        try c.encode(result, forKey: .result)
        try c.encodeIfPresent(libraryHash, forKey: .libraryHash)
    }

    /// Whether this output is current for `snapshot` and `request`: computed
    /// from the same system **and** glass library, with the same settings
    /// (including backend, precision and seed). Otherwise it is stale.
    public func isCurrent(for snapshot: ModelSnapshot?, request: AnalysisRequest) -> Bool {
        guard let snapshot, libraryMatches(snapshot) else { return false }
        if let own = self.request {
            return own == request && runInfo.snapshotHash == snapshot.hash
        }
        return runInfo.isCurrent(for: snapshot, request: request)
    }

    /// Same for a request by name: the engine's recorded settings contain
    /// every requested value.
    public func isCurrent(for snapshot: ModelSnapshot?, request: EngineAnalysisRequest) -> Bool {
        guard let snapshot, libraryMatches(snapshot), runInfo.snapshotHash == snapshot.hash else { return false }
        if let own = engineRequest { return own == request }
        return request.matches(runInfo)
    }

    private func libraryMatches(_ snapshot: ModelSnapshot) -> Bool {
        libraryHash == nil || snapshot.libraryHash == nil || libraryHash == snapshot.libraryHash
    }
}
