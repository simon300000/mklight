import Foundation

/// How the system aperture is specified.
public enum ApertureSpec: Codable, Sendable, Hashable {
    /// Entrance pupil diameter (mm).
    case entrancePupilDiameter(Double)
    /// Paraxial image-space F/# (`f'/EPD`), object at infinity.
    case imageSpaceFNumber(Double)
    /// Object-space NA `n·sinθ` (finite objects).
    case objectSpaceNA(Double)
    /// Paraxial working F/# `1/(2 n' sinθ')`.
    case paraxialWorkingFNumber(Double)
    /// The stop's fixed semi-diameter sets the aperture.
    case floatByStopSize

    public static let `default` = ApertureSpec.entrancePupilDiameter(10)

    public enum Kind: String, CaseIterable, Sendable, Identifiable {
        case entrancePupilDiameter = "entrance_pupil_diameter"
        case imageSpaceFNumber = "image_space_f_number"
        case objectSpaceNA = "object_space_na"
        case paraxialWorkingFNumber = "paraxial_working_f_number"
        case floatByStopSize = "float_by_stop_size"
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .entrancePupilDiameter: "Entrance pupil diameter"
            case .imageSpaceFNumber: "Image-space F/#"
            case .objectSpaceNA: "Object-space NA"
            case .paraxialWorkingFNumber: "Paraxial working F/#"
            case .floatByStopSize: "Float by stop size"
            }
        }
        public var unit: String {
            switch self {
            case .entrancePupilDiameter: "mm"
            default: ""
            }
        }
    }

    public var kind: Kind {
        switch self {
        case .entrancePupilDiameter: .entrancePupilDiameter
        case .imageSpaceFNumber: .imageSpaceFNumber
        case .objectSpaceNA: .objectSpaceNA
        case .paraxialWorkingFNumber: .paraxialWorkingFNumber
        case .floatByStopSize: .floatByStopSize
        }
    }

    /// The numeric value, `nil` for `floatByStopSize`.
    public var value: Double? {
        switch self {
        case .entrancePupilDiameter(let v), .imageSpaceFNumber(let v), .objectSpaceNA(let v),
            .paraxialWorkingFNumber(let v):
            v
        case .floatByStopSize: nil
        }
    }

    public init(kind: Kind, value: Double) {
        switch kind {
        case .entrancePupilDiameter: self = .entrancePupilDiameter(value)
        case .imageSpaceFNumber: self = .imageSpaceFNumber(value)
        case .objectSpaceNA: self = .objectSpaceNA(value)
        case .paraxialWorkingFNumber: self = .paraxialWorkingFNumber(value)
        case .floatByStopSize: self = .floatByStopSize
        }
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        let t = try c.value("type", as: String.self)
        guard let kind = Kind(rawValue: t) else {
            throw DecodingError.dataCorruptedError(forKey: AnyKey("type"), in: c, debugDescription: "unknown aperture spec '\(t)'")
        }
        self = kind == .floatByStopSize ? .floatByStopSize : ApertureSpec(kind: kind, value: try c.value("value"))
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(kind.rawValue, "type")
        if let v = value { try c.put(v, "value") }
    }
}

public enum FieldKind: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
    /// Object-space field angle (degrees).
    case objectAngle = "object_angle"
    /// Object height (mm), finite objects only.
    case objectHeight = "object_height"
    /// Paraxial image height (mm).
    case paraxialImageHeight = "paraxial_image_height"
    /// Real image height (mm).
    case realImageHeight = "real_image_height"

    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .objectAngle: "Object angle"
        case .objectHeight: "Object height"
        case .paraxialImageHeight: "Paraxial image height"
        case .realImageHeight: "Real image height"
        }
    }
    public var unit: String { self == .objectAngle ? "°" : "mm" }
}

/// Vignetting factors (CONVENTIONS §3): `Py' = vdy + (1 − vcy)·Py`.
public struct Vignetting: Codable, Sendable, Hashable {
    public var vdx = 0.0, vdy = 0.0, vcx = 0.0, vcy = 0.0

    public init(vdx: Double = 0, vdy: Double = 0, vcx: Double = 0, vcy: Double = 0) {
        self.vdx = vdx
        self.vdy = vdy
        self.vcx = vcx
        self.vcy = vcy
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        vdx = try c.value("vdx", default: 0)
        vdy = try c.value("vdy", default: 0)
        vcx = try c.value("vcx", default: 0)
        vcy = try c.value("vcy", default: 0)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(vdx, "vdx")
        try c.put(vdy, "vdy")
        try c.put(vcx, "vcx")
        try c.put(vcy, "vcy")
    }
}

public struct Field: Codable, Sendable, Hashable {
    public var x: Double
    public var y: Double
    public var weight: Double
    public var vignetting: Vignetting

    public init(x: Double = 0, y: Double = 0, weight: Double = 1, vignetting: Vignetting = Vignetting()) {
        self.x = x
        self.y = y
        self.weight = weight
        self.vignetting = vignetting
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        x = try c.value("x")
        y = try c.value("y")
        weight = try c.value("weight", default: 1)
        vignetting = try c.value("vignetting", default: Vignetting())
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(x, "x")
        try c.put(y, "y")
        try c.put(weight, "weight")
        try c.put(vignetting, "vignetting")
    }
}

public struct FieldSet: Codable, Sendable, Hashable {
    public var kind: FieldKind
    public var fields: [Field]

    public init(kind: FieldKind = .objectAngle, fields: [Field] = [Field()]) {
        self.kind = kind
        self.fields = fields
    }

    /// Normaliser for `(Hx, Hy)`; `1` when every field is on axis.
    public var maxRadial: Double {
        let m = fields.map { hypot($0.x, $0.y) }.max() ?? 0
        return m > 0 ? m : 1
    }

    private enum CodingKeys: String, CodingKey { case kind, fields }
}

public struct Wavelength: Codable, Sendable, Hashable {
    /// Micrometres.
    public var um: Double
    public var weight: Double

    public init(um: Double, weight: Double = 1) {
        self.um = um
        self.weight = weight
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        um = try c.value("um")
        weight = try c.value("weight", default: 1)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(um, "um")
        try c.put(weight, "weight")
    }
}

public struct WavelengthSet: Codable, Sendable, Hashable {
    public var wavelengths: [Wavelength]
    /// Index into `wavelengths`.
    public var primary: Int

    /// F, d, C with d primary (the engine's default).
    public init(
        wavelengths: [Wavelength] = [
            Wavelength(um: SpectralLine.F.um), Wavelength(um: SpectralLine.d.um), Wavelength(um: SpectralLine.C.um),
        ],
        primary: Int = 1
    ) {
        self.wavelengths = wavelengths
        self.primary = primary
    }

    public var primaryUm: Double? { wavelengths.indices.contains(primary) ? wavelengths[primary].um : nil }

    private enum CodingKeys: String, CodingKey { case wavelengths, primary }
}

public enum RayAiming: Codable, Sendable, Hashable {
    /// Pupil coordinates map onto the paraxial entrance pupil.
    case off
    /// Real-ray aiming onto the stop; tolerance as a fraction of the stop radius.
    case real(tolerance: Double)

    public static let defaultTolerance = 1e-10

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        switch try c.value("mode", as: String.self) {
        case "off": self = .off
        case "real": self = .real(tolerance: try c.value("tolerance", default: RayAiming.defaultTolerance))
        case let m:
            throw DecodingError.dataCorruptedError(forKey: AnyKey("mode"), in: c, debugDescription: "unknown ray aiming '\(m)'")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        switch self {
        case .off: try c.put("off", "mode")
        case .real(let tol):
            try c.put("real", "mode")
            try c.put(tol, "tolerance")
        }
    }
}

public struct EnvironmentConditions: Codable, Sendable, Hashable {
    public var temperatureC: Double
    public var pressureAtm: Double

    public init(temperatureC: Double = 20, pressureAtm: Double = 1) {
        self.temperatureC = temperatureC
        self.pressureAtm = pressureAtm
    }

    private enum CodingKeys: String, CodingKey {
        case temperatureC = "temperature_c"
        case pressureAtm = "pressure_atm"
    }
}

/// Standard spectral lines (µm), identical to the engine's `materials::lines`.
public enum SpectralLine: String, CaseIterable, Sendable {
    case g, F, e, d, C

    public var um: Double {
        switch self {
        case .d: 0.5875618
        case .F: 0.4861327
        case .C: 0.6562725
        case .g: 0.4358343
        case .e: 0.5460740
        }
    }
}
