import Foundation

/// Surface shape in its local frame (CONVENTIONS §2, SURFACES.md). Curvature
/// `c = 1/R` is stored, never the radius.
///
/// The JSON form is the engine's serde form: decoding is strict (unknown keys
/// are errors), fields the engine defaults (`conic`, `conic_x`, `conic_y`,
/// `x_asphere`, `y_asphere`, `ordering`) may be absent, and encoding always
/// writes every field.
public enum SurfaceShape: Codable, Sendable, Hashable {
    /// Flat surface `z = 0`.
    case plane
    /// Sphere or conic: `z = c r² / (1 + √(1 − (1+k) c² r²))`.
    case conic(curvature: Double, conic: Double)
    /// Conic plus `Σ a_i r^(2i+4)`; `coefficients[0]` is A4.
    case evenAsphere(curvature: Double, conic: Double, coefficients: [Double])
    /// Conic plus `Σ β_i r^i` from i = 1; `coefficients[0]` is β1, in mm^(1−i).
    case oddAsphere(curvature: Double, conic: Double, coefficients: [Double])
    /// Forbes conic-based Q departure; coefficients a₀, a₁, … in mm (SURFACES §2).
    case qCon(curvature: Double, conic: Double, normRadius: Double, coefficients: [Double])
    /// Forbes best-fit-sphere Q departure; `curvature` is that of the sphere.
    case qBfs(curvature: Double, normRadius: Double, coefficients: [Double])
    /// Fixed-period phase structure on a plane, conic or even-asphere base
    /// (SURFACES §3). Curvature, conic and asphere parameters address the base.
    indirect case diffractive(base: SurfaceShape, phase: DiffractivePhase)
    /// The y profile (conic plus `yAsphere` from y⁴) rotated about a line
    /// parallel to y at the x radius (SURFACES §4).
    case toroid(curvatureX: Double, curvatureY: Double, conicY: Double, yAsphere: [Double])
    /// Independent x and y conics plus axis polynomials from x⁴ and y⁴
    /// (SURFACES §4). A cylinder is a biconic preset: see `cylinder(curvature:axis:)`.
    case biconic(
        curvatureX: Double, curvatureY: Double, conicX: Double, conicY: Double, xAsphere: [Double],
        yAsphere: [Double])
    /// Conic plus `Σ C_j X^m Y^n` in normalised coordinates, coefficients in
    /// mm; the term order is `XYTerm` (SURFACES §5).
    case xyPolynomial(curvature: Double, conic: Double, normRadius: Double, coefficients: [Double])
    /// Conic plus Zernike sag terms in mm, from Z1 (piston); the term order is
    /// `ZernikeTerm` (SURFACES §6).
    case zernikeSag(
        curvature: Double, conic: Double, normRadius: Double, ordering: ZernikeOrdering, coefficients: [Double])
    /// Ideal thin lens of the given focal length (mm).
    case paraxial(focalLength: Double)
    /// Changes the frame only (uses the surface's `placement`).
    case coordinateBreak

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("type", as: String.self) {
        case "plane":
            try c.allowOnly("type")
            self = .plane
        case "conic":
            try c.allowOnly("type", "curvature", "conic")
            self = .conic(curvature: try c.value("curvature"), conic: try c.value("conic", default: 0))
        case "even_asphere":
            try c.allowOnly("type", "curvature", "conic", "coefficients")
            self = .evenAsphere(
                curvature: try c.value("curvature"), conic: try c.value("conic", default: 0),
                coefficients: try c.value("coefficients"))
        case "odd_asphere":
            try c.allowOnly("type", "curvature", "conic", "coefficients")
            self = .oddAsphere(
                curvature: try c.value("curvature"), conic: try c.value("conic", default: 0),
                coefficients: try c.value("coefficients"))
        case "q_con":
            try c.allowOnly("type", "curvature", "conic", "norm_radius", "coefficients")
            self = .qCon(
                curvature: try c.value("curvature"), conic: try c.value("conic", default: 0),
                normRadius: try c.value("norm_radius"), coefficients: try c.value("coefficients"))
        case "q_bfs":
            try c.allowOnly("type", "curvature", "norm_radius", "coefficients")
            self = .qBfs(
                curvature: try c.value("curvature"), normRadius: try c.value("norm_radius"),
                coefficients: try c.value("coefficients"))
        case "diffractive":
            try c.allowOnly("type", "base", "phase")
            self = .diffractive(base: try c.value("base"), phase: try c.value("phase"))
        case "toroid":
            try c.allowOnly("type", "curvature_x", "curvature_y", "conic_y", "y_asphere")
            self = .toroid(
                curvatureX: try c.value("curvature_x"), curvatureY: try c.value("curvature_y"),
                conicY: try c.value("conic_y", default: 0), yAsphere: try c.value("y_asphere", default: []))
        case "biconic":
            try c.allowOnly("type", "curvature_x", "curvature_y", "conic_x", "conic_y", "x_asphere", "y_asphere")
            self = .biconic(
                curvatureX: try c.value("curvature_x"), curvatureY: try c.value("curvature_y"),
                conicX: try c.value("conic_x", default: 0), conicY: try c.value("conic_y", default: 0),
                xAsphere: try c.value("x_asphere", default: []), yAsphere: try c.value("y_asphere", default: []))
        case "xy_polynomial":
            try c.allowOnly("type", "curvature", "conic", "norm_radius", "coefficients")
            self = .xyPolynomial(
                curvature: try c.value("curvature"), conic: try c.value("conic", default: 0),
                normRadius: try c.value("norm_radius"), coefficients: try c.value("coefficients"))
        case "zernike_sag":
            try c.allowOnly("type", "curvature", "conic", "norm_radius", "ordering", "coefficients")
            self = .zernikeSag(
                curvature: try c.value("curvature"), conic: try c.value("conic", default: 0),
                normRadius: try c.value("norm_radius"), ordering: try c.value("ordering", default: .fringe),
                coefficients: try c.value("coefficients"))
        case "paraxial":
            try c.allowOnly("type", "focal_length")
            self = .paraxial(focalLength: try c.value("focal_length"))
        case "coordinate_break":
            try c.allowOnly("type")
            self = .coordinateBreak
        case let t:
            throw DecodingError.dataCorruptedError(
                forKey: AnyKey("type"), in: c, debugDescription: "unknown surface shape '\(t)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(typeName, "type")
        switch self {
        case .plane, .coordinateBreak: break
        case .conic(let cv, let k):
            try c.put(cv, "curvature")
            try c.put(k, "conic")
        case .evenAsphere(let cv, let k, let a), .oddAsphere(let cv, let k, let a):
            try c.put(cv, "curvature")
            try c.put(k, "conic")
            try c.put(a, "coefficients")
        case .qCon(let cv, let k, let r, let a), .xyPolynomial(let cv, let k, let r, let a):
            try c.put(cv, "curvature")
            try c.put(k, "conic")
            try c.put(r, "norm_radius")
            try c.put(a, "coefficients")
        case .qBfs(let cv, let r, let a):
            try c.put(cv, "curvature")
            try c.put(r, "norm_radius")
            try c.put(a, "coefficients")
        case .diffractive(let base, let phase):
            try c.put(base, "base")
            try c.put(phase, "phase")
        case .toroid(let cx, let cy, let ky, let ya):
            try c.put(cx, "curvature_x")
            try c.put(cy, "curvature_y")
            try c.put(ky, "conic_y")
            try c.put(ya, "y_asphere")
        case .biconic(let cx, let cy, let kx, let ky, let xa, let ya):
            try c.put(cx, "curvature_x")
            try c.put(cy, "curvature_y")
            try c.put(kx, "conic_x")
            try c.put(ky, "conic_y")
            try c.put(xa, "x_asphere")
            try c.put(ya, "y_asphere")
        case .zernikeSag(let cv, let k, let r, let o, let a):
            try c.put(cv, "curvature")
            try c.put(k, "conic")
            try c.put(r, "norm_radius")
            try c.put(o, "ordering")
            try c.put(a, "coefficients")
        case .paraxial(let f):
            try c.put(f, "focal_length")
        }
    }

    /// The JSON `type` tag (the engine's `SurfaceShape::type_name`).
    public var typeName: String { kind.typeName }

    /// Whether the shape belongs to the engine's original solver subset
    /// (plane, conic, even asphere, paraxial, coordinate break). The engine's
    /// layout, Seidel, field curvature, ghost and merit evaluation reject
    /// systems with other shapes (`OpticalSystem::require_legacy_shapes`).
    public var isLegacy: Bool { kind.isLegacy }

    // MARK: Cylinder preset

    /// The axis along which a cylinder has power.
    public enum CylinderAxis: String, Sendable, Hashable, CaseIterable {
        /// Power in x (curved in the x-z plane, straight along y).
        case x
        /// Power in y (curved in the y-z plane, straight along x).
        case y
    }

    /// A cylinder: the biconic preset with curvature `curvature` along `axis`
    /// and a flat other axis (SURFACES §4). `.x` matches the engine sample's
    /// `cylinder(curvature_x)`.
    public static func cylinder(curvature: Double, axis: CylinderAxis) -> SurfaceShape {
        .biconic(
            curvatureX: axis == .x ? curvature : 0, curvatureY: axis == .y ? curvature : 0, conicX: 0, conicY: 0,
            xAsphere: [], yAsphere: [])
    }

    /// The power axis when the shape is a cylinder: a biconic with one curved
    /// axis and the other axis flat (zero curvature, zero polynomial). The
    /// curved axis may carry a conic and polynomial. `nil` otherwise.
    public var cylinderAxis: CylinderAxis? {
        guard case .biconic(let cx, let cy, _, _, let xa, let ya) = self else { return nil }
        let flatX = cx == 0 && xa.allSatisfy { $0 == 0 }
        let flatY = cy == 0 && ya.allSatisfy { $0 == 0 }
        if flatY && cx != 0 { return .x }
        if flatX && cy != 0 { return .y }
        return nil
    }

    /// Whether the shape is the cylinder preset of a biconic.
    public var isCylinder: Bool { cylinderAxis != nil }
}

/// Phase lag of a diffractive surface at order +1, `Φ = Σ A_i ρ^(2i)` in
/// radians with `ρ = r / normRadius` (SURFACES §3). The design order and
/// wavelength are metadata; they do not rescale the traced phase.
public struct DiffractivePhase: Codable, Sendable, Hashable {
    /// Normalisation radius (mm), > 0.
    public var normRadius: Double
    /// A₁, A₂, … (radians), at most `CoefficientSet.phase` limit (10).
    public var coefficients: [Double]
    /// Nonzero signed design order (an `i32` in the engine).
    public var designOrder: Int
    /// Design wavelength λ₀ (µm), > 0.
    public var designWavelengthUm: Double
    /// Signed order that is traced; 0 is ordinary refraction or reflection
    /// (an `i32` in the engine).
    public var traceOrder: Int

    public init(
        normRadius: Double, coefficients: [Double] = [], designOrder: Int = 1, designWavelengthUm: Double,
        traceOrder: Int = 1
    ) {
        self.normRadius = normRadius
        self.coefficients = coefficients
        self.designOrder = designOrder
        self.designWavelengthUm = designWavelengthUm
        self.traceOrder = traceOrder
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly("norm_radius", "coefficients", "design_order", "design_wavelength_um", "trace_order")
        normRadius = try c.value("norm_radius")
        coefficients = try c.value("coefficients")
        // Orders are i32 in the engine: out-of-range values fail to decode.
        designOrder = Int(try c.value("design_order", as: Int32.self))
        designWavelengthUm = try c.value("design_wavelength_um")
        traceOrder = Int(try c.value("trace_order", as: Int32.self))
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(normRadius, "norm_radius")
        try c.put(coefficients, "coefficients")
        try c.put(designOrder, "design_order")
        try c.put(designWavelengthUm, "design_wavelength_um")
        try c.put(traceOrder, "trace_order")
    }
}

/// Term ordering and normalisation of a `zernikeSag` surface (SURFACES §6).
/// `"noll"` decodes as `.standard`, which always encodes as `"standard"`.
public enum ZernikeOrdering: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
    /// Unnormalised Fringe order, Z1–Z37.
    case fringe
    /// Noll-normalised standard order, Z1–Z45 (n ≤ 8).
    case standard

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .fringe: "Fringe"
        case .standard: "Standard (Noll)"
        }
    }

    /// Number of supported terms (the engine's `ZernikeOrdering::max_terms`).
    public var maxTerms: Int {
        switch self {
        case .fringe: 37
        case .standard: 45
        }
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        switch try c.decode(String.self) {
        case "fringe": self = .fringe
        case "standard", "noll": self = .standard
        case let s:
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "unknown Zernike ordering '\(s)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(rawValue)
    }
}
