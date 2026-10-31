import Foundation

/// Address of one numeric parameter of an `OpticalSystem` (mirror of the
/// engine's `ParamPath`). Surfaces are addressed by `SurfaceID`, so paths
/// survive inserts and deletes.
public enum ParamPath: Codable, Sendable, Hashable {
    case curvature(surface: SurfaceID)
    case thickness(surface: SurfaceID)
    case conic(surface: SurfaceID)
    /// Even-asphere coefficient `index` (0 → A4).
    case asphere(surface: SurfaceID, index: Int)
    /// Coefficient `index` (zero-based) of an extended shape's coefficient
    /// family (SURFACES §7). A missing in-range coefficient reads 0.
    case shapeCoefficient(surface: SurfaceID, set: CoefficientSet, index: Int)
    /// A scalar parameter of an extended shape. JSON names the selector
    /// `parameter`, since `param` is the tag.
    case shapeParam(surface: SurfaceID, parameter: ShapeParameter)
    case paraxialFocalLength(surface: SurfaceID)
    case modelNd(surface: SurfaceID)
    case modelVd(surface: SurfaceID)
    case modelDpgf(surface: SurfaceID)
    /// `Δn_d` / `ΔV_d` of a catalog glass's `MaterialOffset`.
    case materialDnD(surface: SurfaceID)
    case materialDvD(surface: SurfaceID)
    case decenterX(surface: SurfaceID)
    case decenterY(surface: SurfaceID)
    case tiltX(surface: SurfaceID)
    case tiltY(surface: SurfaceID)
    case tiltZ(surface: SurfaceID)
    case semiDiameter(surface: SurfaceID)
    case apertureValue
    case fieldX(index: Int)
    case fieldY(index: Int)
    case fieldWeight(index: Int)
    case wavelengthUm(index: Int)
    case wavelengthWeight(index: Int)
    case temperatureC
    case pressureAtm
    /// Test fringes of a sag overlay (SURFACES §10); 0 without one.
    case irregularity(surface: SurfaceID, mode: IrregularityMode)
    /// One component of the rigid motion of the range `surface…last` about
    /// `pivot` (`nil`: `surface`); mm or degrees.
    case elementPlacement(surface: SurfaceID, last: SurfaceID, pivot: SurfaceID?, component: PlacementComponent)

    /// The `param` tag of each surface-addressed case.
    private static let surfaceTags: [String: @Sendable (SurfaceID) -> ParamPath] = [
        "curvature": { .curvature(surface: $0) },
        "thickness": { .thickness(surface: $0) },
        "conic": { .conic(surface: $0) },
        "paraxial_focal_length": { .paraxialFocalLength(surface: $0) },
        "model_nd": { .modelNd(surface: $0) },
        "model_vd": { .modelVd(surface: $0) },
        "model_dpgf": { .modelDpgf(surface: $0) },
        "material_dn_d": { .materialDnD(surface: $0) },
        "material_dv_d": { .materialDvD(surface: $0) },
        "decenter_x": { .decenterX(surface: $0) },
        "decenter_y": { .decenterY(surface: $0) },
        "tilt_x": { .tiltX(surface: $0) },
        "tilt_y": { .tiltY(surface: $0) },
        "tilt_z": { .tiltZ(surface: $0) },
        "semi_diameter": { .semiDiameter(surface: $0) },
    ]
    private static let indexTags: [String: @Sendable (Int) -> ParamPath] = [
        "field_x": { .fieldX(index: $0) },
        "field_y": { .fieldY(index: $0) },
        "field_weight": { .fieldWeight(index: $0) },
        "wavelength_um": { .wavelengthUm(index: $0) },
        "wavelength_weight": { .wavelengthWeight(index: $0) },
    ]

    /// `(tag, surface, index)` of this path.
    private var parts: (String, SurfaceID?, Int?) {
        switch self {
        case .curvature(let s): ("curvature", s, nil)
        case .thickness(let s): ("thickness", s, nil)
        case .conic(let s): ("conic", s, nil)
        case .asphere(let s, let i): ("asphere", s, i)
        case .shapeCoefficient(let s, _, let i): ("shape_coefficient", s, i)
        case .shapeParam(let s, _): ("shape_param", s, nil)
        case .paraxialFocalLength(let s): ("paraxial_focal_length", s, nil)
        case .modelNd(let s): ("model_nd", s, nil)
        case .modelVd(let s): ("model_vd", s, nil)
        case .modelDpgf(let s): ("model_dpgf", s, nil)
        case .materialDnD(let s): ("material_dn_d", s, nil)
        case .materialDvD(let s): ("material_dv_d", s, nil)
        case .decenterX(let s): ("decenter_x", s, nil)
        case .decenterY(let s): ("decenter_y", s, nil)
        case .tiltX(let s): ("tilt_x", s, nil)
        case .tiltY(let s): ("tilt_y", s, nil)
        case .tiltZ(let s): ("tilt_z", s, nil)
        case .semiDiameter(let s): ("semi_diameter", s, nil)
        case .apertureValue: ("aperture_value", nil, nil)
        case .fieldX(let i): ("field_x", nil, i)
        case .fieldY(let i): ("field_y", nil, i)
        case .fieldWeight(let i): ("field_weight", nil, i)
        case .wavelengthUm(let i): ("wavelength_um", nil, i)
        case .wavelengthWeight(let i): ("wavelength_weight", nil, i)
        case .temperatureC: ("temperature_c", nil, nil)
        case .pressureAtm: ("pressure_atm", nil, nil)
        case .irregularity(let s, _): ("irregularity", s, nil)
        case .elementPlacement(let s, _, _, _): ("element_placement", s, nil)
        }
    }

    public var surface: SurfaceID? { parts.1 }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        let tag = try c.value("param", as: String.self)
        if let make = Self.surfaceTags[tag] {
            try c.allowOnly("param", "surface")
            self = make(try c.value("surface"))
        } else if let make = Self.indexTags[tag] {
            try c.allowOnly("param", "index")
            self = make(try c.index("index"))
        } else {
            switch tag {
            case "asphere":
                try c.allowOnly("param", "surface", "index")
                self = .asphere(surface: try c.value("surface"), index: try c.index("index"))
            case "shape_coefficient":
                try c.allowOnly("param", "surface", "set", "index")
                self = .shapeCoefficient(
                    surface: try c.value("surface"), set: try c.value("set"), index: try c.index("index"))
            case "shape_param":
                try c.allowOnly("param", "surface", "parameter")
                self = .shapeParam(surface: try c.value("surface"), parameter: try c.value("parameter"))
            case "irregularity":
                try c.allowOnly("param", "surface", "mode")
                self = .irregularity(surface: try c.value("surface"), mode: try c.value("mode"))
            case "element_placement":
                try c.allowOnly("param", "surface", "last", "pivot", "component")
                self = .elementPlacement(
                    surface: try c.value("surface"), last: try c.value("last"), pivot: try c.optional("pivot"),
                    component: try c.value("component"))
            case "aperture_value", "temperature_c", "pressure_atm":
                try c.allowOnly("param")
                self = tag == "aperture_value" ? .apertureValue : tag == "temperature_c" ? .temperatureC : .pressureAtm
            default:
                throw DecodingError.dataCorruptedError(forKey: AnyKey("param"), in: c, debugDescription: "unknown parameter '\(tag)'")
            }
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        let (tag, surface, index) = parts
        try c.put(tag, "param")
        if let surface { try c.put(surface, "surface") }
        switch self {
        case .shapeCoefficient(_, let set, _): try c.put(set, "set")
        case .shapeParam(_, let p): try c.put(p, "parameter")
        case .irregularity(_, let mode): try c.put(mode, "mode")
        case .elementPlacement(_, let last, let pivot, let component):
            try c.put(last, "last")
            try c.putOptional(pivot, "pivot")
            try c.put(component, "component")
        default: break
        }
        if let index { try c.put(index, "index") }
    }
}

/// Part of a sag overlay's fringe irregularity.
public enum IrregularityMode: String, Codable, Sendable, Hashable, CaseIterable {
    case astigmatism, spherical
}

/// One component of a rigid `ElementPlacement` motion.
public enum PlacementComponent: String, Codable, Sendable, Hashable, CaseIterable {
    case decenterX = "decenter_x"
    case decenterY = "decenter_y"
    case tiltX = "tilt_x"
    case tiltY = "tilt_y"
    case tiltZ = "tilt_z"

    public var title: String {
        switch self {
        case .decenterX: "Decenter X"
        case .decenterY: "Decenter Y"
        case .tiltX: "Tilt X"
        case .tiltY: "Tilt Y"
        case .tiltZ: "Tilt Z"
        }
    }

    public var keyPath: WritableKeyPath<Placement, Double> {
        switch self {
        case .decenterX: \.decenterX
        case .decenterY: \.decenterY
        case .tiltX: \.tiltX
        case .tiltY: \.tiltY
        case .tiltZ: \.tiltZ
        }
    }
}

/// One configuration of a multi-configuration system: parameter overrides on
/// top of the base system.
public struct Configuration: Codable, Sendable, Hashable {
    public var name: String
    public var overrides: [Pair<ParamPath, Double>]

    public init(name: String = "", overrides: [Pair<ParamPath, Double>] = []) {
        self.name = name
        self.overrides = overrides
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly("name", "overrides")
        name = try c.value("name", default: "")
        overrides = try c.value("overrides", default: [])
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(name, "name")
        try c.put(overrides, "overrides")
    }
}
