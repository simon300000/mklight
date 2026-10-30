import Foundation

/// Surface scatter (BSDF) models; values per unit projected solid angle (sr⁻¹).
public enum ScatterModel: Codable, Sendable, Hashable {
    /// Lambertian with total integrated scatter `tis`.
    case lambertian(tis: Double)
    /// Harvey–Shack ABg: `A / (B + |β−β₀|^g)`.
    case abg(a: Double, b: Double, g: Double)
    /// Gaussian microfacets with RMS slope (radians).
    case gaussian(tis: Double, sigmaSlope: Double)

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("type", as: String.self) {
        case "lambertian": self = .lambertian(tis: try c.value("tis"))
        case "abg": self = .abg(a: try c.value("a"), b: try c.value("b"), g: try c.value("g"))
        case "gaussian": self = .gaussian(tis: try c.value("tis"), sigmaSlope: try c.value("sigma_slope"))
        case let t:
            throw DecodingError.dataCorruptedError(forKey: AnyKey("type"), in: c, debugDescription: "unknown scatter model '\(t)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .lambertian(let tis):
            try c.put("lambertian", "type")
            try c.put(tis, "tis")
        case .abg(let a, let b, let g):
            try c.put("abg", "type")
            try c.put(a, "a")
            try c.put(b, "b")
            try c.put(g, "g")
        case .gaussian(let tis, let s):
            try c.put("gaussian", "type")
            try c.put(tis, "tis")
            try c.put(s, "sigma_slope")
        }
    }
}

/// Non-optical geometry for the stray-light solver.
public enum MechanicalPart: Codable, Sendable, Hashable {
    /// Cylindrical tube between two surfaces' vertex planes.
    case barrel(from: SurfaceID, to: SurfaceID, innerRadius: Double, finish: ScatterModel)
    /// Flat annular baffle at local `z` in a surface's frame.
    case annularBaffle(surface: SurfaceID, z: Double, innerRadius: Double, outerRadius: Double, finish: ScatterModel)

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("type", as: String.self) {
        case "barrel":
            self = .barrel(
                from: try c.value("from"), to: try c.value("to"), innerRadius: try c.value("inner_radius"),
                finish: try c.value("finish"))
        case "annular_baffle":
            self = .annularBaffle(
                surface: try c.value("surface"), z: try c.value("z"), innerRadius: try c.value("inner_radius"),
                outerRadius: try c.value("outer_radius"), finish: try c.value("finish"))
        case let t:
            throw DecodingError.dataCorruptedError(forKey: AnyKey("type"), in: c, debugDescription: "unknown mechanical part '\(t)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .barrel(let from, let to, let r, let finish):
            try c.put("barrel", "type")
            try c.put(from, "from")
            try c.put(to, "to")
            try c.put(r, "inner_radius")
            try c.put(finish, "finish")
        case .annularBaffle(let s, let z, let ri, let ro, let finish):
            try c.put("annular_baffle", "type")
            try c.put(s, "surface")
            try c.put(z, "z")
            try c.put(ri, "inner_radius")
            try c.put(ro, "outer_radius")
            try c.put(finish, "finish")
        }
    }
}

/// Image sensor at the image surface.
public struct Sensor: Codable, Sendable, Hashable {
    public var pixelsX: UInt32
    public var pixelsY: UInt32
    /// Pixel pitch (mm).
    public var pitchX: Double
    public var pitchY: Double
    /// Largest accepted chief-ray angle (degrees).
    public var maxCRADeg: Double?
    /// Relative spectral response `(λ µm, response)`; empty means flat.
    public var spectralResponse: [SpectralSample]

    public init(
        pixelsX: UInt32, pixelsY: UInt32, pitchX: Double, pitchY: Double, maxCRADeg: Double? = nil,
        spectralResponse: [SpectralSample] = []
    ) {
        self.pixelsX = pixelsX
        self.pixelsY = pixelsY
        self.pitchX = pitchX
        self.pitchY = pitchY
        self.maxCRADeg = maxCRADeg
        self.spectralResponse = spectralResponse
    }

    public var halfWidth: Double { 0.5 * Double(pixelsX) * pitchX }
    public var halfHeight: Double { 0.5 * Double(pixelsY) * pitchY }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        pixelsX = try c.value("pixels_x")
        pixelsY = try c.value("pixels_y")
        pitchX = try c.value("pitch_x")
        pitchY = try c.value("pitch_y")
        maxCRADeg = try c.optional("max_cra_deg")
        spectralResponse = try c.value("spectral_response", default: [])
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(pixelsX, "pixels_x")
        try c.put(pixelsY, "pixels_y")
        try c.put(pitchX, "pitch_x")
        try c.put(pitchY, "pitch_y")
        try c.putOptional(maxCRADeg, "max_cra_deg")
        try c.put(spectralResponse, "spectral_response")
    }
}
