import Foundation

/// The complete physical description of an optical system: a value-type
/// mirror of the engine's `OpticalSystem`, encoded with exactly the engine's
/// JSON representation. It holds no UI state.
public struct OpticalSystem: Codable, Sendable, Hashable {
    public var name: String
    public var notes: String
    public var aperture: ApertureSpec
    public var fields: FieldSet
    public var wavelengths: WavelengthSet
    /// `surfaces[0]` is the object and `surfaces.last` the image.
    public var surfaces: [Surface]
    /// Index of the aperture stop.
    public var stop: Int
    public var rayAiming: RayAiming
    public var environment: EnvironmentConditions
    /// Named coatings referenced by `Surface.coating`.
    public var coatings: [String: CoatingDef]
    public var mechanics: [MechanicalPart]
    public var sensor: Sensor?
    public var configurations: [Configuration]

    public init(
        name: String, notes: String = "", aperture: ApertureSpec = .default, fields: FieldSet = FieldSet(),
        wavelengths: WavelengthSet = WavelengthSet(), surfaces: [Surface], stop: Int, rayAiming: RayAiming = .off,
        environment: EnvironmentConditions = EnvironmentConditions(), coatings: [String: CoatingDef] = [:],
        mechanics: [MechanicalPart] = [], sensor: Sensor? = nil, configurations: [Configuration] = []
    ) {
        self.name = name
        self.notes = notes
        self.aperture = aperture
        self.fields = fields
        self.wavelengths = wavelengths
        self.surfaces = surfaces
        self.stop = stop
        self.rayAiming = rayAiming
        self.environment = environment
        self.coatings = coatings
        self.mechanics = mechanics
        self.sensor = sensor
        self.configurations = configurations
    }

    /// The engine's minimal valid system: object at infinity, a plane stop
    /// 10 mm before the image.
    public static func newEmpty() -> OpticalSystem {
        OpticalSystem(
            name: "Untitled",
            surfaces: [
                Surface(id: SurfaceID(0), shape: .plane, thickness: .infinity),
                Surface(id: SurfaceID(1), label: "STOP", shape: .plane, thickness: 10),
                Surface(id: SurfaceID(2), shape: .plane, thickness: 0),
            ],
            stop: 1)
    }

    public var imageIndex: Int { surfaces.count - 1 }
    public var objectAtInfinity: Bool { surfaces.first?.thickness.isInfinite ?? false }

    public func surfaceIndex(of id: SurfaceID) -> Int? { surfaces.firstIndex { $0.id == id } }

    /// A `SurfaceID` no surface uses yet.
    public var nextSurfaceID: SurfaceID {
        SurfaceID((surfaces.map(\.id.rawValue).max()).map { $0 + 1 } ?? 0)
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        name = try c.value("name", default: "")
        notes = try c.value("notes", default: "")
        aperture = try c.value("aperture")
        fields = try c.value("fields")
        wavelengths = try c.value("wavelengths")
        surfaces = try c.value("surfaces")
        stop = try c.value("stop")
        rayAiming = try c.value("ray_aiming", default: .off)
        environment = try c.value("environment", default: EnvironmentConditions())
        coatings = try c.value("coatings", default: [:])
        mechanics = try c.value("mechanics", default: [])
        sensor = try c.optional("sensor")
        configurations = try c.value("configurations", default: [])
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(name, "name")
        try c.put(notes, "notes")
        try c.put(aperture, "aperture")
        try c.put(fields, "fields")
        try c.put(wavelengths, "wavelengths")
        try c.put(surfaces, "surfaces")
        try c.put(stop, "stop")
        try c.put(rayAiming, "ray_aiming")
        try c.put(environment, "environment")
        try c.put(coatings, "coatings")
        try c.put(mechanics, "mechanics")
        try c.putOptional(sensor, "sensor")
        try c.put(configurations, "configurations")
    }
}

// MARK: - Structural validation

/// A structural problem with a model. The engine's validation is
/// authoritative; this mirror exists so a mock or offline engine and the
/// editor can report the same problems without tracing.
public struct ModelIssue: Sendable, Hashable, CustomStringConvertible {
    public enum Severity: Sendable, Hashable { case error, warning }
    public var severity: Severity
    /// Surface index the issue concerns, if any.
    public var surface: Int?
    public var message: String
    public init(_ severity: Severity, surface: Int? = nil, _ message: String) {
        self.severity = severity
        self.surface = surface
        self.message = message
    }
    public var description: String { surface.map { "surface \($0): \(message)" } ?? message }
}

extension OpticalSystem {
    /// The checks of the engine's `OpticalSystem::validate`, collected rather
    /// than stopping at the first.
    public func structuralIssues() -> [ModelIssue] {
        var out: [ModelIssue] = []
        let n = surfaces.count
        guard n >= 3 else {
            return [ModelIssue(.error, "a system needs object, at least one surface, and image")]
        }
        if stop < 1 || stop > n - 2 {
            out.append(ModelIssue(.error, "stop index \(stop) must be between 1 and \(n - 2)"))
        }
        if wavelengths.wavelengths.isEmpty || !wavelengths.wavelengths.indices.contains(wavelengths.primary) {
            out.append(ModelIssue(.error, "wavelength set empty or primary out of range"))
        }
        if wavelengths.wavelengths.contains(where: { !($0.um > 0 && $0.um.isFinite) || $0.weight < 0 }) {
            out.append(ModelIssue(.error, "wavelengths must be positive and finite with weights ≥ 0"))
        }
        if fields.fields.isEmpty {
            out.append(ModelIssue(.error, "at least one field is required"))
        }
        if objectAtInfinity && fields.kind == .objectHeight {
            out.append(ModelIssue(.error, "object-height fields require a finite object distance"))
        }
        if objectAtInfinity, case .objectSpaceNA = aperture {
            out.append(ModelIssue(.error, "object-space NA requires a finite object distance"))
        }
        if objectAtInfinity && fields.kind == .objectAngle && fields.fields.contains(where: { hypot($0.x, $0.y) >= 90 }) {
            out.append(ModelIssue(.error, "field angles must be below 90°"))
        }
        var ids = Set<SurfaceID>()
        for (i, s) in surfaces.enumerated() {
            if !ids.insert(s.id).inserted {
                out.append(ModelIssue(.error, surface: i, "duplicate surface id \(s.id.rawValue)"))
            }
            if i > 0 && !s.thickness.isFinite && i != n - 1 {
                out.append(ModelIssue(.error, surface: i, "only the object may have infinite thickness"))
            }
            if let c = s.coating, coatings[c] == nil {
                out.append(ModelIssue(.error, surface: i, "unknown coating '\(c)'"))
            }
            if case .fixed(let v) = s.semiDiameter, !(v > 0) {
                out.append(ModelIssue(.error, surface: i, "fixed semi-diameter must be > 0"))
            }
            if (i == 0 || i == n - 1) && s.isCoordinateBreak {
                out.append(ModelIssue(.error, surface: i, "object/image cannot be coordinate breaks"))
            }
        }
        if surfaces[n - 1].interaction == .reflect {
            out.append(ModelIssue(.error, surface: n - 1, "the image surface cannot be a mirror"))
        }
        for (k, cfg) in configurations.enumerated() {
            for o in cfg.overrides where !o.first.resolves(in: self) {
                out.append(ModelIssue(.error, "configuration \(k): parameter \(o.first) does not exist"))
            }
        }
        return out
    }
}

// MARK: - Project file

/// On-disk `.mklight` project: a versioned envelope around the model. Bare
/// `OpticalSystem` JSON (as written by the engine and CLI) is also accepted.
public struct ProjectFile: Codable, Sendable, Hashable {
    public static let formatName = "mklight-project"
    public static let currentVersion = 1

    public var format: String
    public var version: Int
    public var system: OpticalSystem

    public init(system: OpticalSystem) {
        format = Self.formatName
        version = Self.currentVersion
        self.system = system
    }

    public enum Error: Swift.Error, LocalizedError {
        case unsupportedVersion(Int)
        public var errorDescription: String? {
            switch self {
            case .unsupportedVersion(let v):
                "This project was written by a newer MKLight (format version \(v))."
            }
        }
    }

    public static func decode(_ data: Data) throws -> ProjectFile {
        let dec = MKLightJSON.decoder()
        if let envelope = try? dec.decode(Envelope.self, from: data), envelope.format == formatName {
            guard envelope.version <= currentVersion else { throw Error.unsupportedVersion(envelope.version) }
            return try dec.decode(ProjectFile.self, from: data)
        }
        return ProjectFile(system: try dec.decode(OpticalSystem.self, from: data))
    }

    public func encoded() throws -> Data {
        try MKLightJSON.encode(self, pretty: true)
    }

    private struct Envelope: Decodable {
        var format: String
        var version: Int
    }
}
