import Foundation

/// A solve or pickup: `target` follows from the rule `kind` (mirror of the
/// engine's `Solve`, docs/SOLVES.md). The stored target value is only the
/// starting point; the engine's solver returns a new system with every
/// target set.
public struct Solve: Codable, Sendable, Hashable {
    public var target: ParamPath
    public var kind: SolveKind

    public init(target: ParamPath, kind: SolveKind) {
        self.target = target
        self.kind = kind
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly("target", "kind")
        target = try c.value("target")
        kind = try c.value("kind")
    }

    private enum CodingKeys: String, CodingKey { case target, kind }

    /// Surfaces the solve refers to (target, pickup source, position range,
    /// centre of curvature).
    public var surfaces: [SurfaceID] {
        var out: [SurfaceID] = target.surface.map { [$0] } ?? []
        switch kind {
        case .pickup(let source, _, _): if let s = source.surface { out.append(s) }
        case .position(let from, let to, _): out += [from, to]
        case .centerOfCurvature(let s): out.append(s)
        default: break
        }
        return out
    }
}

/// The rule that sets a solve's target (engine `SolveKind`, tagged by
/// `"type"`). Angle solves take the paraxial **slope** `u = dy/dz`, which is
/// dimensionless: UI labels call it "paraxial angle u" and it is never shown
/// or converted as degrees (CONVENTIONS §8).
public enum SolveKind: Codable, Sendable, Hashable {
    /// On a thickness: marginal-ray height (mm) on the next vertex.
    case marginalRayHeight(height: Double)
    /// On a curvature or ideal-lens focal length: marginal-ray slope `u` after the surface.
    case marginalRayAngle(slope: Double)
    case chiefRayHeight(height: Double)
    case chiefRayAngle(slope: Double)
    /// Image-space F/# `f'/EPD`.
    case fNumber(value: Double)
    /// `target = scale · source + offset`.
    case pickup(source: ParamPath, scale: Double, offset: Double)
    /// Axial distance (mm) from the vertex of `from` to that of `to`.
    case position(from: SurfaceID, to: SurfaceID, distance: Double)
    /// Centre of curvature on the vertex of `surface`.
    case centerOfCurvature(surface: SurfaceID)
    /// Axial edge thickness (mm) at `radius` (`nil`: from the semi-diameters).
    case edgeThickness(value: Double, radius: Double?)

    public enum Kind: String, CaseIterable, Sendable, Identifiable {
        case marginalRayHeight = "marginal_ray_height"
        case marginalRayAngle = "marginal_ray_angle"
        case chiefRayHeight = "chief_ray_height"
        case chiefRayAngle = "chief_ray_angle"
        case fNumber = "f_number"
        case pickup
        case position
        case centerOfCurvature = "center_of_curvature"
        case edgeThickness = "edge_thickness"
        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .marginalRayHeight: "Marginal ray height"
            case .marginalRayAngle: "Marginal ray angle"
            case .chiefRayHeight: "Chief ray height"
            case .chiefRayAngle: "Chief ray angle"
            case .fNumber: "F-number"
            case .pickup: "Pickup"
            case .position: "Position"
            case .centerOfCurvature: "Centre of curvature"
            case .edgeThickness: "Edge thickness"
            }
        }

        /// Label and unit of the kind's main number. Slopes are "paraxial
        /// angle u", dimensionless (CONVENTIONS §8).
        public var valueLabel: (label: String, unit: String)? {
            switch self {
            case .marginalRayHeight, .chiefRayHeight: ("height", "mm")
            case .marginalRayAngle, .chiefRayAngle: ("paraxial angle u", "")
            case .fNumber: ("F/#", "")
            case .position: ("distance", "mm")
            case .edgeThickness: ("edge thickness", "mm")
            case .pickup, .centerOfCurvature: nil
            }
        }

        /// Whether this kind can control a thickness (else a curvature or
        /// focal length, except pickups which take any parameter).
        public var targetsThickness: Bool {
            switch self {
            case .marginalRayHeight, .chiefRayHeight, .position, .edgeThickness: true
            default: false
            }
        }
    }

    public var kind: Kind {
        switch self {
        case .marginalRayHeight: .marginalRayHeight
        case .marginalRayAngle: .marginalRayAngle
        case .chiefRayHeight: .chiefRayHeight
        case .chiefRayAngle: .chiefRayAngle
        case .fNumber: .fNumber
        case .pickup: .pickup
        case .position: .position
        case .centerOfCurvature: .centerOfCurvature
        case .edgeThickness: .edgeThickness
        }
    }

    /// The kind's main number (height, slope, F/#, distance, edge thickness).
    public var value: Double? {
        get {
            switch self {
            case .marginalRayHeight(let v), .chiefRayHeight(let v), .marginalRayAngle(let v), .chiefRayAngle(let v),
                .fNumber(let v), .position(_, _, let v), .edgeThickness(let v, _):
                v
            case .pickup, .centerOfCurvature: nil
            }
        }
        set {
            guard let v = newValue else { return }
            switch self {
            case .marginalRayHeight: self = .marginalRayHeight(height: v)
            case .chiefRayHeight: self = .chiefRayHeight(height: v)
            case .marginalRayAngle: self = .marginalRayAngle(slope: v)
            case .chiefRayAngle: self = .chiefRayAngle(slope: v)
            case .fNumber: self = .fNumber(value: v)
            case .position(let a, let b, _): self = .position(from: a, to: b, distance: v)
            case .edgeThickness(_, let r): self = .edgeThickness(value: v, radius: r)
            case .pickup, .centerOfCurvature: break
            }
        }
    }

    /// Every number of the kind (for finiteness checks).
    var numbers: [Double] {
        switch self {
        case .pickup(_, let s, let o): [s, o]
        case .centerOfCurvature: []
        case .edgeThickness(let v, let r): [v] + (r.map { [$0] } ?? [])
        default: value.map { [$0] } ?? []
        }
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        let t = try c.value("type", as: String.self)
        guard let kind = Kind(rawValue: t) else {
            throw DecodingError.dataCorruptedError(forKey: AnyKey("type"), in: c, debugDescription: "unknown solve '\(t)'")
        }
        switch kind {
        case .marginalRayHeight, .chiefRayHeight: try c.allowOnly("type", "height")
        case .marginalRayAngle, .chiefRayAngle: try c.allowOnly("type", "slope")
        case .fNumber: try c.allowOnly("type", "value")
        case .pickup: try c.allowOnly("type", "source", "scale", "offset")
        case .position: try c.allowOnly("type", "from", "to", "distance")
        case .centerOfCurvature: try c.allowOnly("type", "surface")
        case .edgeThickness: try c.allowOnly("type", "value", "radius")
        }
        switch kind {
        case .marginalRayHeight: self = .marginalRayHeight(height: try c.value("height"))
        case .marginalRayAngle: self = .marginalRayAngle(slope: try c.value("slope"))
        case .chiefRayHeight: self = .chiefRayHeight(height: try c.value("height"))
        case .chiefRayAngle: self = .chiefRayAngle(slope: try c.value("slope"))
        case .fNumber: self = .fNumber(value: try c.value("value"))
        case .pickup:
            self = .pickup(
                source: try c.value("source"), scale: try c.value("scale", default: 1), offset: try c.value("offset", default: 0))
        case .position: self = .position(from: try c.value("from"), to: try c.value("to"), distance: try c.value("distance"))
        case .centerOfCurvature: self = .centerOfCurvature(surface: try c.value("surface"))
        case .edgeThickness: self = .edgeThickness(value: try c.value("value"), radius: try c.optional("radius"))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(kind.rawValue, "type")
        switch self {
        case .marginalRayHeight(let h), .chiefRayHeight(let h): try c.put(h, "height")
        case .marginalRayAngle(let u), .chiefRayAngle(let u): try c.put(u, "slope")
        case .fNumber(let v): try c.put(v, "value")
        case .pickup(let source, let scale, let offset):
            try c.put(source, "source")
            try c.put(scale, "scale")
            try c.put(offset, "offset")
        case .position(let from, let to, let d):
            try c.put(from, "from")
            try c.put(to, "to")
            try c.put(d, "distance")
        case .centerOfCurvature(let s): try c.put(s, "surface")
        case .edgeThickness(let v, let r):
            try c.put(v, "value")
            try c.putOptional(r, "radius")
        }
    }
}

extension OpticalSystem {
    /// Parameters controlled by solves (exclude them from optimisation; the
    /// editor marks them).
    public var solvedParams: Set<ParamPath> { Set(solves.map { canonical($0.target) }) }

    /// The solve controlling `path` (by any path naming the same
    /// parameter), if any.
    public func solve(for path: ParamPath) -> Solve? {
        let p = canonical(path)
        return solves.first { canonical($0.target) == p }
    }

    /// Whether a solve sets `path`. `optimize` applies solves first and
    /// refuses such a parameter as a variable.
    public func isSolved(_ path: ParamPath) -> Bool { solve(for: path) != nil }

    /// One path per parameter: on a toroid or biconic the `curvature` path
    /// is the y section's curvature, so `shape_param curvature_y` there
    /// becomes `curvature`. Every other path is its own.
    public func canonical(_ path: ParamPath) -> ParamPath {
        guard case .shapeParam(let id, .curvatureY) = path, let i = surfaceIndex(of: id) else { return path }
        switch surfaces[i].shape {
        case .toroid, .biconic: return .curvature(surface: id)
        default: return path
        }
    }

    /// Structural checks of the engine's solve validation that need no
    /// optics: every target and source exists, no parameter has two solves,
    /// no pickup of itself, finite numbers, positive F/# and edge radius.
    /// The engine's validation is authoritative (type rules, cycles).
    func solveIssues() -> [ModelIssue] {
        var out: [ModelIssue] = []
        var seen = Set<ParamPath>()
        for (k, s) in solves.enumerated() {
            let name = "solve \(k) (\(s.kind.kind.rawValue))"
            if !seen.insert(s.target).inserted {
                out.append(ModelIssue(.error, "\(name): \(s.target) already has a solve"))
            }
            // An automatic semi-diameter can be a target: the solve fixes it.
            let fixesSemiDiameter = if case .semiDiameter = s.target { true } else { false }
            if !fixesSemiDiameter || s.target.surface.flatMap(surfaceIndex(of:)) == nil,
                let why = s.target.unresolvedReason(in: self)
            {
                out.append(ModelIssue(.error, "\(name): invalid target \(s.target): \(why)"))
            }
            if case .pickup(let source, _, _) = s.kind {
                if source == s.target { out.append(ModelIssue(.error, "\(name): a pickup cannot read its own target")) }
                if solve(for: source) == nil, let why = source.unresolvedReason(in: self) {
                    out.append(ModelIssue(.error, "\(name): invalid source \(source): \(why)"))
                }
            }
            for id in s.surfaces where surfaceIndex(of: id) == nil {
                out.append(ModelIssue(.error, "\(name): no surface with id \(id.rawValue)"))
            }
            if s.kind.numbers.contains(where: { !$0.isFinite }) {
                out.append(ModelIssue(.error, "\(name): values must be finite"))
            }
            if case .fNumber(let v) = s.kind, !(v > 0) { out.append(ModelIssue(.error, "\(name): F/# must be > 0")) }
            if case .edgeThickness(_, let r?) = s.kind, !(r > 0) {
                out.append(ModelIssue(.error, "\(name): edge radius must be > 0"))
            }
        }
        return out
    }
}

/// How a solve's value was found (engine `SolveMethod`).
public enum SolveMethod: String, Codable, Sendable, Hashable {
    case unchanged
    case closedForm = "closed_form"
    case fixedPoint = "fixed_point"
    case rootFinding = "root_finding"
}

/// One solve's outcome (engine `SolveOutcome`).
public struct SolveOutcome: Codable, Sendable, Hashable {
    /// Index into `OpticalSystem.solves`.
    public var index: Int
    public var kind: String
    public var target: ParamPath
    /// Value in the input system (`nil` for an automatic semi-diameter).
    public var previous: Double?
    public var value: Double
    /// `measured − goal` on the solved system, in the condition's units
    /// (mm, slope, F/#, or dimensionless).
    public var residual: Double
    public var method: SolveMethod
    public var evaluations: Int

    private enum CodingKeys: String, CodingKey {
        case index, kind, target, previous, value, residual, method, evaluations
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(index, forKey: .index)
        try c.encode(kind, forKey: .kind)
        try c.encode(target, forKey: .target)
        try c.encode(previous, forKey: .previous)  // explicit null, as serde writes it
        try c.encode(value, forKey: .value)
        try c.encode(residual, forKey: .residual)
        try c.encode(method, forKey: .method)
        try c.encode(evaluations, forKey: .evaluations)
    }
}

/// Result of applying a system's solves (engine `SolveReport`). The run
/// metadata stays a raw JSON tree here; `MKLightEngine.RunInfo` decodes it.
public struct SolveReport: Codable, Sendable, Hashable {
    public var outcomes: [SolveOutcome]
    /// Evaluation order (indices into `solves`).
    public var order: [Int]
    public var sweeps: Int
    public var maxChange: Double
    public var solvedParams: [ParamPath]
    public var warnings: [String]
    /// The configuration that was solved (the returned system is that
    /// configuration, flattened); `nil` for the system as given.
    public var configuration: Int?
    public var runInfo: JSONValue

    private enum CodingKeys: String, CodingKey {
        case outcomes, order, sweeps, maxChange = "max_change", solvedParams = "solved_params", warnings, configuration
        case runInfo = "run_info"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        outcomes = try c.decode([SolveOutcome].self, forKey: .outcomes)
        order = try c.decode([Int].self, forKey: .order)
        sweeps = try c.decode(Int.self, forKey: .sweeps)
        maxChange = try c.decode(Double.self, forKey: .maxChange)
        solvedParams = try c.decode([ParamPath].self, forKey: .solvedParams)
        warnings = try c.decodeIfPresent([String].self, forKey: .warnings) ?? []
        configuration = try c.decodeIfPresent(Int.self, forKey: .configuration)
        runInfo = try c.decode(JSONValue.self, forKey: .runInfo)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(outcomes, forKey: .outcomes)
        try c.encode(order, forKey: .order)
        try c.encode(sweeps, forKey: .sweeps)
        try c.encode(maxChange, forKey: .maxChange)
        try c.encode(solvedParams, forKey: .solvedParams)
        try c.encode(warnings, forKey: .warnings)
        try c.encode(configuration, forKey: .configuration)
        try c.encode(runInfo, forKey: .runInfo)
    }
}
