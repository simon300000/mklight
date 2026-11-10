import Foundation

/// Reference to an optical medium. The engine resolves it to numbers.
public enum MaterialRef: Codable, Sendable, Hashable {
    /// Air at the system temperature and pressure (index 1 by convention).
    case air
    /// Vacuum (index `1/n_air`).
    case vacuum
    /// A catalog glass; `catalog == nil` searches every loaded catalog.
    /// `offset` shifts its `n_d` and `V_d` (docs/MATERIALS.md).
    case catalog(catalog: String?, name: String, offset: MaterialOffset = .zero)
    /// Model glass from `n_d`, `V_d` and `ΔP_g,F`.
    case model(nd: Double, vd: Double, dpgf: Double)
    /// Constant index at every wavelength.
    case constant(n: Double)

    public static func glass(_ name: String) -> MaterialRef { .catalog(catalog: nil, name: name) }

    public var isAir: Bool { self == .air }

    /// Short name shown in tables (matches the engine's `display_name`).
    public var displayName: String {
        switch self {
        case .air: "AIR"
        case .vacuum: "VACUUM"
        case .catalog(_, let name, _): name
        case .model(let nd, let vd, _): String(format: "%.5f/%.2f", nd, vd)
        case .constant(let n): "n=" + NumberDisplay.significant(n)
        }
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("type", as: String.self) {
        case "air":
            try c.allowOnly("type")
            self = .air
        case "vacuum":
            try c.allowOnly("type")
            self = .vacuum
        case "catalog":
            try c.allowOnly("type", "catalog", "name", "offset")
            self = .catalog(
                catalog: try c.optional("catalog"), name: try c.value("name"),
                offset: try c.value("offset", default: .zero))
        case "model":
            try c.allowOnly("type", "nd", "vd", "dpgf")
            self = .model(nd: try c.value("nd"), vd: try c.value("vd"), dpgf: try c.value("dpgf", default: 0))
        case "constant":
            try c.allowOnly("type", "n")
            self = .constant(n: try c.value("n"))
        case let t:
            throw DecodingError.dataCorruptedError(forKey: AnyKey("type"), in: c, debugDescription: "unknown material '\(t)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .air: try c.put("air", "type")
        case .vacuum: try c.put("vacuum", "type")
        case .catalog(let cat, let name, let offset):
            try c.put("catalog", "type")
            try c.putOptional(cat, "catalog")
            try c.put(name, "name")
            if !offset.isZero { try c.put(offset, "offset") }
        case .model(let nd, let vd, let dpgf):
            try c.put("model", "type")
            try c.put(nd, "nd")
            try c.put(vd, "vd")
            try c.put(dpgf, "dpgf")
        case .constant(let n):
            try c.put("constant", "type")
            try c.put(n, "n")
        }
    }
}

/// Changes `Δn_d`, `ΔV_d` to a catalog glass; its dispersion is scaled about
/// d, keeping partial-dispersion ratios. Omitted from JSON when zero.
public struct MaterialOffset: Codable, Sendable, Hashable {
    public var dnD: Double
    public var dvD: Double

    public static let zero = MaterialOffset()
    /// Structural bounds on `|Δn_d|` and `|ΔV_d|`.
    public static let maxDnD = 0.05
    public static let maxDvD = 10.0

    public init(dnD: Double = 0, dvD: Double = 0) {
        self.dnD = dnD
        self.dvD = dvD
    }

    public var isZero: Bool { dnD == 0 && dvD == 0 }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly("dn_d", "dv_d")
        dnD = try c.value("dn_d", default: 0)
        dvD = try c.value("dv_d", default: 0)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(dnD, "dn_d")
        try c.put(dvD, "dv_d")
    }
}

/// One thin-film layer.
public struct CoatingLayer: Codable, Sendable, Hashable {
    public var material: MaterialRef
    /// Physical thickness (µm); superseded by `quarterWaves` when set.
    public var thicknessUm: Double
    /// Optical thickness in quarter waves at the reference wavelength.
    public var quarterWaves: Double?
    /// Constant extinction coefficient `k`.
    public var extinction: Double

    public init(material: MaterialRef, thicknessUm: Double = 0, quarterWaves: Double? = nil, extinction: Double = 0) {
        self.material = material
        self.thicknessUm = thicknessUm
        self.quarterWaves = quarterWaves
        self.extinction = extinction
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        try c.allowOnly("material", "thickness_um", "quarter_waves", "extinction")
        material = try c.value("material")
        thicknessUm = try c.value("thickness_um", default: 0)
        quarterWaves = try c.optional("quarter_waves")
        extinction = try c.value("extinction", default: 0)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(material, "material")
        try c.put(thicknessUm, "thickness_um")
        try c.putOptional(quarterWaves, "quarter_waves")
        try c.put(extinction, "extinction")
    }
}

/// Complex refractive index `n + ik`, serialised as `[n, k]`.
public typealias ComplexIndex = Pair<Double, Double>

/// A named coating definition.
public enum CoatingDef: Codable, Sendable, Hashable {
    /// Multilayer stack, layers listed outward from the substrate.
    case thinFilm(referenceWavelengthUm: Double, layers: [CoatingLayer], mirrorSubstrate: ComplexIndex?)
    /// Fixed reflectance and transmittance; absorption `1 − R − T`.
    case ideal(reflectance: Double, transmittance: Double)
    /// Tabulated normal-incidence reflectance `(λ µm, R)`; `T = 1 − R`.
    case tabulated(reflectance: [SpectralSample])

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("type", as: String.self) {
        case "thin_film":
            try c.allowOnly("type", "reference_wavelength_um", "layers", "mirror_substrate")
            self = .thinFilm(
                referenceWavelengthUm: try c.value("reference_wavelength_um"), layers: try c.value("layers"),
                mirrorSubstrate: try c.optional("mirror_substrate"))
        case "ideal":
            try c.allowOnly("type", "reflectance", "transmittance")
            self = .ideal(reflectance: try c.value("reflectance"), transmittance: try c.value("transmittance"))
        case "tabulated":
            try c.allowOnly("type", "reflectance")
            self = .tabulated(reflectance: try c.value("reflectance"))
        case let t:
            throw DecodingError.dataCorruptedError(forKey: AnyKey("type"), in: c, debugDescription: "unknown coating '\(t)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .thinFilm(let ref, let layers, let sub):
            try c.put("thin_film", "type")
            try c.put(ref, "reference_wavelength_um")
            try c.put(layers, "layers")
            try c.putOptional(sub, "mirror_substrate")
        case .ideal(let r, let t):
            try c.put("ideal", "type")
            try c.put(r, "reflectance")
            try c.put(t, "transmittance")
        case .tabulated(let table):
            try c.put("tabulated", "type")
            try c.put(table, "reflectance")
        }
    }
}
