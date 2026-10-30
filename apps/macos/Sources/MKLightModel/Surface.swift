import Foundation

/// Stable identity of a surface, independent of its index. The UI keys
/// selection and view state on it so they survive inserts and deletes.
public struct SurfaceID: RawRepresentable, Codable, Sendable, Hashable, Comparable, CustomStringConvertible {
    public var rawValue: UInt64
    public init(rawValue: UInt64) { self.rawValue = rawValue }
    public init(_ v: UInt64) { rawValue = v }
    public init(from decoder: any Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(UInt64.self)
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(rawValue)
    }
    public static func < (a: SurfaceID, b: SurfaceID) -> Bool { a.rawValue < b.rawValue }
    public var description: String { "#\(rawValue)" }
}

/// What the surface does to light.
public enum Interaction: String, Codable, Sendable, Hashable, CaseIterable {
    case refract
    /// Specular mirror; the medium after is the medium before.
    case reflect
    /// No interaction (dummy/reference surface).
    case dummy
}

/// Clear-aperture size.
public enum SemiDiameter: Codable, Sendable, Hashable {
    /// Computed by the tracer from the largest ray height; does not clip.
    case auto
    /// User-fixed radius (mm) that clips rays.
    case fixed(Double)

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("mode", as: String.self) {
        case "auto":
            try c.allowOnly("mode")
            self = .auto
        case "fixed":
            try c.allowOnly("mode", "value")
            self = .fixed(try c.value("value"))
        case let m:
            throw DecodingError.dataCorruptedError(forKey: AnyKey("mode"), in: c, debugDescription: "unknown semi-diameter mode '\(m)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .auto: try c.put("auto", "mode")
        case .fixed(let v):
            try c.put("fixed", "mode")
            try c.put(v, "value")
        }
    }

    public var fixedValue: Double? {
        if case .fixed(let v) = self { return v }
        return nil
    }
}

/// Explicit clipping aperture; overrides the circular semi-diameter.
public enum ApertureShape: Codable, Sendable, Hashable {
    case circular(radius: Double)
    case annular(innerRadius: Double, outerRadius: Double)
    case rectangular(halfX: Double, halfY: Double)
    case elliptical(semiX: Double, semiY: Double)

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("type", as: String.self) {
        case "circular":
            try c.allowOnly("type", "radius")
            self = .circular(radius: try c.value("radius"))
        case "annular":
            try c.allowOnly("type", "inner_radius", "outer_radius")
            self = .annular(innerRadius: try c.value("inner_radius"), outerRadius: try c.value("outer_radius"))
        case "rectangular":
            try c.allowOnly("type", "half_x", "half_y")
            self = .rectangular(halfX: try c.value("half_x"), halfY: try c.value("half_y"))
        case "elliptical":
            try c.allowOnly("type", "semi_x", "semi_y")
            self = .elliptical(semiX: try c.value("semi_x"), semiY: try c.value("semi_y"))
        case let t:
            throw DecodingError.dataCorruptedError(forKey: AnyKey("type"), in: c, debugDescription: "unknown aperture '\(t)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .circular(let r):
            try c.put("circular", "type")
            try c.put(r, "radius")
        case .annular(let a, let b):
            try c.put("annular", "type")
            try c.put(a, "inner_radius")
            try c.put(b, "outer_radius")
        case .rectangular(let x, let y):
            try c.put("rectangular", "type")
            try c.put(x, "half_x")
            try c.put(y, "half_y")
        case .elliptical(let x, let y):
            try c.put("elliptical", "type")
            try c.put(x, "semi_x")
            try c.put(y, "semi_y")
        }
    }

    /// Radius of the smallest enclosing circle.
    public var boundingRadius: Double {
        switch self {
        case .circular(let r): r
        case .annular(_, let r): r
        case .rectangular(let x, let y): hypot(x, y)
        case .elliptical(let x, let y): max(x, y)
        }
    }
}

public enum TransformOrder: String, Codable, Sendable, Hashable, CaseIterable {
    case decenterThenTilt = "decenter_then_tilt"
    case tiltThenDecenter = "tilt_then_decenter"
}

/// Rigid placement change applied before the surface (CONVENTIONS §2):
/// decenters in mm, tilts in degrees about x, then y, then z.
public struct Placement: Codable, Sendable, Hashable {
    public var decenterX = 0.0
    public var decenterY = 0.0
    public var tiltX = 0.0
    public var tiltY = 0.0
    public var tiltZ = 0.0
    public var order = TransformOrder.decenterThenTilt
    /// Return to the pre-placement frame after the surface.
    public var restoreAfter = false

    public init(
        decenterX: Double = 0, decenterY: Double = 0, tiltX: Double = 0, tiltY: Double = 0, tiltZ: Double = 0,
        order: TransformOrder = .decenterThenTilt, restoreAfter: Bool = false
    ) {
        self.decenterX = decenterX
        self.decenterY = decenterY
        self.tiltX = tiltX
        self.tiltY = tiltY
        self.tiltZ = tiltZ
        self.order = order
        self.restoreAfter = restoreAfter
    }

    public var isIdentity: Bool {
        decenterX == 0 && decenterY == 0 && tiltX == 0 && tiltY == 0 && tiltZ == 0
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly("decenter_x", "decenter_y", "tilt_x", "tilt_y", "tilt_z", "order", "restore_after")
        decenterX = try c.value("decenter_x", default: 0)
        decenterY = try c.value("decenter_y", default: 0)
        tiltX = try c.value("tilt_x", default: 0)
        tiltY = try c.value("tilt_y", default: 0)
        tiltZ = try c.value("tilt_z", default: 0)
        order = try c.value("order", default: .decenterThenTilt)
        restoreAfter = try c.value("restore_after", default: false)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(decenterX, "decenter_x")
        try c.put(decenterY, "decenter_y")
        try c.put(tiltX, "tilt_x")
        try c.put(tiltY, "tilt_y")
        try c.put(tiltZ, "tilt_z")
        try c.put(order, "order")
        try c.put(restoreAfter, "restore_after")
    }
}

/// Mechanical description of a lens edge (stray-light geometry).
public struct EdgeSpec: Codable, Sendable, Hashable {
    /// `nil` means the clear aperture plus the default edge margin.
    public var mechanicalSemiDiameter: Double?
    public var edgeScatter: ScatterModel?

    public init(mechanicalSemiDiameter: Double? = nil, edgeScatter: ScatterModel? = nil) {
        self.mechanicalSemiDiameter = mechanicalSemiDiameter
        self.edgeScatter = edgeScatter
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly("mechanical_semi_diameter", "edge_scatter")
        mechanicalSemiDiameter = try c.optional("mechanical_semi_diameter")
        edgeScatter = try c.optional("edge_scatter")
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.putOptional(mechanicalSemiDiameter, "mechanical_semi_diameter")
        try c.putOptional(edgeScatter, "edge_scatter")
    }
}

/// Scalar order efficiency of a diffractive surface (SURFACES §3): which
/// share of the light goes into each diffraction order. `ideal` puts it all
/// into the trace order; a `kinoform` (continuous blaze) and a `multilevel`
/// staircase of N levels spread it by the thin-element Fourier model.
public enum DiffractionEfficiency: Codable, Sendable, Hashable {
    case ideal
    case kinoform
    case multilevel(levels: Int)

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        let type: String = try c.value("type")
        switch type {
        case "ideal":
            try c.allowOnly("type")
            self = .ideal
        case "kinoform":
            try c.allowOnly("type")
            self = .kinoform
        case "multilevel":
            try c.allowOnly("type", "levels")
            let n: Int = try c.value("levels")
            guard n >= 1, n <= Int(UInt16.max) else {
                throw DecodingError.dataCorruptedError(
                    forKey: AnyKey("levels"), in: c, debugDescription: "levels must be 1 … 65535")
            }
            self = .multilevel(levels: n)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: AnyKey("type"), in: c, debugDescription: "unknown diffraction efficiency '\(type)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .ideal: try c.put("ideal", "type")
        case .kinoform: try c.put("kinoform", "type")
        case .multilevel(let n):
            try c.put("multilevel", "type")
            try c.put(n, "levels")
        }
    }

    /// The efficiency of `order` for a relief blazed at `blaze` (α·m₀: the
    /// design order times the relief's optical path at this wavelength over
    /// its design path, SURFACES §3.1), `selected` being the trace order.
    /// Mirrors the engine's `DiffractionEfficiency::efficiency` for the
    /// inspector's readout; the engine traces with its own.
    public func efficiency(order: Int, selected: Int, blaze: Double) -> Double {
        func sinc(_ x: Double) -> Double {
            if x == 0 { return 1 }
            let r = x - x.rounded()
            return sin(Double.pi * r) / (Double.pi * x)
        }
        let m = Double(order)
        switch self {
        case .ideal: return order == selected ? 1 : 0
        case .kinoform: return pow(sinc(m - blaze), 2)
        case .multilevel(let levels):
            let n = Double(levels)
            var r = (blaze - m) / n
            r -= r.rounded()
            return pow(sinc(m / n) * sinc(n * r) / sinc(r), 2)
        }
    }
}

/// One surface of the sequential train.
public struct Surface: Codable, Sendable, Hashable, Identifiable {
    public var id: SurfaceID
    public var label: String
    public var shape: SurfaceShape
    /// Distance to the next vertex (CONVENTIONS §2); may be `±∞` for the object.
    public var thickness: Double
    /// Medium after this surface.
    public var material: MaterialRef
    public var interaction: Interaction
    public var semiDiameter: SemiDiameter
    public var aperture: ApertureShape?
    public var placement: Placement?
    /// Name of a coating in `OpticalSystem.coatings`; `nil` is a bare interface.
    public var coating: String?
    public var scatter: ScatterModel?
    /// Order efficiency model; meaningful on diffractive surfaces only.
    public var diffractionEfficiency: DiffractionEfficiency
    public var edge: EdgeSpec

    public init(
        id: SurfaceID, label: String = "", shape: SurfaceShape = .plane, thickness: Double = 0,
        material: MaterialRef = .air, interaction: Interaction = .refract, semiDiameter: SemiDiameter = .auto,
        aperture: ApertureShape? = nil, placement: Placement? = nil, coating: String? = nil,
        scatter: ScatterModel? = nil, diffractionEfficiency: DiffractionEfficiency = .ideal, edge: EdgeSpec = EdgeSpec()
    ) {
        self.id = id
        self.label = label
        self.shape = shape
        self.thickness = thickness
        self.material = material
        self.interaction = interaction
        self.semiDiameter = semiDiameter
        self.aperture = aperture
        self.placement = placement
        self.coating = coating
        self.scatter = scatter
        self.diffractionEfficiency = diffractionEfficiency
        self.edge = edge
    }

    public var isCoordinateBreak: Bool { shape == .coordinateBreak }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly(
            "id", "label", "shape", "thickness", "material", "interaction", "semi_diameter", "aperture", "placement",
            "coating", "scatter", "diffraction_efficiency", "edge")
        id = try c.value("id")
        label = try c.value("label", default: "")
        shape = try c.value("shape")
        thickness = try c.extendedDouble("thickness")
        material = try c.value("material")
        interaction = try c.value("interaction", default: .refract)
        semiDiameter = try c.value("semi_diameter", default: .auto)
        aperture = try c.optional("aperture")
        placement = try c.optional("placement")
        coating = try c.optional("coating")
        scatter = try c.optional("scatter")
        diffractionEfficiency = try c.value("diffraction_efficiency", default: .ideal)
        edge = try c.value("edge", default: EdgeSpec())
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(id, "id")
        try c.put(label, "label")
        try c.put(shape, "shape")
        try c.putExtendedDouble(thickness, "thickness")
        try c.put(material, "material")
        try c.put(interaction, "interaction")
        try c.put(semiDiameter, "semi_diameter")
        try c.putOptional(aperture, "aperture")
        try c.putOptional(placement, "placement")
        try c.putOptional(coating, "coating")
        try c.putOptional(scatter, "scatter")
        try c.put(diffractionEfficiency, "diffraction_efficiency")
        try c.put(edge, "edge")
    }
}
