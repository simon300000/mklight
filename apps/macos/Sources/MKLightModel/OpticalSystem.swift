import Foundation

/// The complete physical description of an optical system: a value-type
/// mirror of the engine's `OpticalSystem`, encoded with exactly the engine's
/// JSON representation. It holds no UI state.
///
/// Decoding is strict, like the engine's `deny_unknown_fields`: an unknown
/// key anywhere in the model is an error rather than silently dropped
/// (every snapshot and save re-encodes the mirror).
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
    /// Solves and pickups (docs/SOLVES.md). Omitted from the JSON when empty,
    /// as the engine does, so systems without solves hash as before.
    public var solves: [Solve]

    public init(
        name: String, notes: String = "", aperture: ApertureSpec = .default, fields: FieldSet = FieldSet(),
        wavelengths: WavelengthSet = WavelengthSet(), surfaces: [Surface], stop: Int, rayAiming: RayAiming = .off,
        environment: EnvironmentConditions = EnvironmentConditions(), coatings: [String: CoatingDef] = [:],
        mechanics: [MechanicalPart] = [], sensor: Sensor? = nil, configurations: [Configuration] = [],
        solves: [Solve] = []
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
        self.solves = solves
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
        try c.allowOnly(
            "name", "notes", "aperture", "fields", "wavelengths", "surfaces", "stop", "ray_aiming", "environment",
            "coatings", "mechanics", "sensor", "configurations", "solves")
        name = try c.value("name", default: "")
        notes = try c.value("notes", default: "")
        aperture = try c.value("aperture")
        fields = try c.value("fields")
        wavelengths = try c.value("wavelengths")
        surfaces = try c.value("surfaces")
        stop = try c.index("stop")
        rayAiming = try c.value("ray_aiming", default: .off)
        environment = try c.value("environment", default: EnvironmentConditions())
        coatings = try c.value("coatings", default: [:])
        mechanics = try c.value("mechanics", default: [])
        sensor = try c.optional("sensor")
        configurations = try c.value("configurations", default: [])
        solves = try c.value("solves", default: [])
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
        if !solves.isEmpty { try c.put(solves, "solves") }
    }
}

// MARK: - Structural validation

/// A structural problem with a model. The engine's validation is
/// authoritative; this mirror exists so a mock or offline engine and the
/// editor can report the same problems without tracing. Every structural
/// problem is an error, as in the engine; warnings come from the engine's
/// `validate` analysis.
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
    /// The rules of the engine's `OpticalSystem::validation_issues`
    /// (CONVENTIONS §13): every problem found, each an error.
    /// `validation_cases.json` holds the engine's verdicts the rules are
    /// tested against. Values JSON cannot carry (NaN, and ±∞ outside the
    /// thicknesses that allow it) are errors, of which the JSON check
    /// reports the first. Of a configuration override this checks that its
    /// path resolves and that the parameter takes its value; the engine
    /// also validates the system each override configures.
    public func structuralIssues() -> [ModelIssue] {
        let json = nonRepresentableValueIssues()
        var out: [ModelIssue] = json
        let n = surfaces.count
        guard n >= 3 else {
            return out + [ModelIssue(.error, "a system needs object, at least one surface, and image")]
        }
        if stop < 1 || stop > n - 2 {
            out.append(ModelIssue(.error, "stop index \(stop) must be between 1 and \(n - 2)"))
        }
        var rules = RuleIssues()
        if let v = aperture.value { rules.positive("invalid_aperture_value", "aperture.value", v) }
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
        for (k, f) in fields.fields.enumerated() {
            let p = "fields.fields[\(k)]"
            rules.nonNegative("negative_weight", "\(p).weight", f.weight)
            // `P' = vd + (1 − vc)·P`: a compression of 1 leaves no pupil.
            for (name, x) in [("vcx", f.vignetting.vcx), ("vcy", f.vignetting.vcy)] where x.isFinite && !(0..<1).contains(x) {
                rules.add("invalid_vignetting", "\(p).vignetting.\(name)", "a pupil compression must lie in [0, 1) (got \(x))")
            }
        }
        if environment.temperatureC.isFinite && environment.temperatureC <= -273.15 {
            rules.add("invalid_environment", "environment.temperature_c", "the temperature must be above absolute zero")
        }
        rules.nonNegative("invalid_environment", "environment.pressure_atm", environment.pressureAtm)
        if case .real(let tolerance) = rayAiming {
            rules.positive("invalid_ray_aiming", "ray_aiming.tolerance", tolerance)
        }
        out += rules.take()
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
            let shape = Self.shapeIssues(s.shape, path: "surfaces[\(i)].shape", surface: i)
            // The JSON check stops at the first non-finite value; the shape
            // rules report each one with its full path, so they replace it.
            if shape.contains(where: { $0.message.hasSuffix(Self.nonFiniteSuffix) }) {
                out.removeAll { j in json.contains(j) && j.surface == i && j.message.hasPrefix("surfaces[\(i)].shape") }
            }
            out += shape
            let p = "surfaces[\(i)]"
            rules.surface = i
            rules.material(s.material, "\(p).material")
            if let a = s.aperture { rules.aperture(a, "\(p).aperture") }
            if let e = s.elementPlacement {
                let end = surfaceIndex(of: e.last)
                if i == 0 || end.map({ $0 < i || $0 + 1 == n }) ?? true {
                    rules.add("invalid_element_range", "\(p).element_placement.last", "expected an interior ordered range")
                }
                if let pivot = e.pivot, surfaceIndex(of: pivot) == nil {
                    rules.add("missing_surface", "\(p).element_placement.pivot", "missing pivot vertex")
                }
            }
            if let m = s.scatter { rules.scatter(m, "\(p).scatter") }
            if let m = s.edge.mechanicalSemiDiameter {
                rules.positive("invalid_mechanical_semi_diameter", "\(p).edge.mechanical_semi_diameter", m)
            }
            if let m = s.edge.edgeScatter { rules.scatter(m, "\(p).edge.edge_scatter") }
            out += rules.take()
        }
        if surfaces[n - 1].interaction == .reflect {
            out.append(ModelIssue(.error, surface: n - 1, "the image surface cannot be a mirror"))
        }
        rules.surface = nil
        for name in coatings.keys.sorted() {
            rules.coating(coatings[name]!, "coatings[\(RuleIssues.quoted(name))]")
        }
        for (k, m) in mechanics.enumerated() {
            let p = "mechanics[\(k)]"
            func attached(_ key: String, _ id: SurfaceID) {
                if surfaceIndex(of: id) == nil {
                    rules.add("missing_surface", "\(p).\(key)", "no surface with id \(id.rawValue)")
                }
            }
            switch m {
            case .barrel(let from, let to, let innerRadius, let finish):
                attached("from", from)
                attached("to", to)
                rules.positive("invalid_mechanics", "\(p).inner_radius", innerRadius)
                rules.scatter(finish, "\(p).finish")
            case .annularBaffle(let surface, _, let innerRadius, let outerRadius, let finish):
                attached("surface", surface)
                rules.nonNegative("invalid_mechanics", "\(p).inner_radius", innerRadius)
                rules.positive("invalid_mechanics", "\(p).outer_radius", outerRadius)
                if innerRadius >= outerRadius {
                    rules.add("invalid_mechanics", p, "the inner radius must be below the outer radius")
                }
                rules.scatter(finish, "\(p).finish")
            }
        }
        if let sensor {
            if sensor.pixelsX == 0 || sensor.pixelsY == 0 { rules.add("invalid_sensor", "sensor", "the sensor needs pixels") }
            rules.positive("invalid_sensor", "sensor.pitch_x", sensor.pitchX)
            rules.positive("invalid_sensor", "sensor.pitch_y", sensor.pitchY)
            rules.increasing("invalid_sensor", "sensor.spectral_response", sensor.spectralResponse)
            for (k, sample) in sensor.spectralResponse.enumerated() {
                rules.positive("invalid_sensor", "sensor.spectral_response[\(k)][0]", sample.um)
                rules.nonNegative("invalid_sensor", "sensor.spectral_response[\(k)][1]", sample.value)
            }
        }
        out += rules.take()
        for (k, cfg) in configurations.enumerated() {
            for o in cfg.overrides {
                if let why = o.first.unresolvedReason(in: self) {
                    out.append(ModelIssue(.error, "configuration \(k): invalid parameter path \(o.first): \(why)"))
                } else if o.second.isFinite, let why = o.first.valueRejection(o.second) {
                    out.append(ModelIssue(.error, "configuration \(k): invalid value \(o.second) for \(o.first): \(why)"))
                }
            }
        }
        out += solveIssues()
        return out
    }

    static let nonFiniteSuffix = " is not a finite number"

    /// The engine's structural rules of a surface shape (CONVENTIONS §13,
    /// SURFACES §7): finite values, positive normalisation radius and design
    /// wavelength, nonzero design order, a plane/conic/even-asphere
    /// diffractive base beneath any sag overlays, and the coefficient
    /// limits. Messages follow the engine's `[code] path: message`
    /// (non-finite values use the JSON check's wording).
    static func shapeIssues(_ shape: SurfaceShape, path: String, surface i: Int) -> [ModelIssue] {
        var out: [ModelIssue] = []
        func issue(_ code: String, _ p: String, _ message: String) {
            out.append(ModelIssue(.error, surface: i, "[\(code)] \(p): \(message)"))
        }
        @discardableResult
        func finite(_ name: String, _ v: Double) -> Bool {
            if v.isFinite { return true }
            out.append(ModelIssue(.error, surface: i, "\(path).\(name) = \(v) is not a finite number"))
            return false
        }
        func positive(_ code: String, _ name: String, _ v: Double) {
            if finite(name, v) && v <= 0 { issue(code, "\(path).\(name)", "must be > 0 (got \(v))") }
        }
        func series(_ name: String, _ a: [Double], _ max: Int) {
            if a.count > max { issue("coefficient_count", "\(path).\(name)", "at most \(max) coefficients") }
            for (k, v) in a.enumerated() { finite("\(name)[\(k)]", v) }
        }
        func normRadius(_ name: String, _ r: Double) { positive("invalid_norm_radius", name, r) }
        switch shape {
        case .plane, .coordinateBreak: break
        case .paraxial(let f):
            if finite("focal_length", f) && f == 0 {
                issue("invalid_focal_length", "\(path).focal_length", "an ideal lens needs a non-zero focal length")
            }
        case .conic(let c, let k):
            finite("curvature", c)
            finite("conic", k)
        case .evenAsphere(let c, let k, let a):
            finite("curvature", c)
            finite("conic", k)
            series("coefficients", a, CoefficientSet.maxEvenTerms)
        case .oddAsphere(let c, let k, let a):
            finite("curvature", c)
            finite("conic", k)
            series("coefficients", a, CoefficientSet.maxOddTerms)
        case .qCon(let c, let k, let r, let a):
            finite("curvature", c)
            finite("conic", k)
            normRadius("norm_radius", r)
            series("coefficients", a, CoefficientSet.maxQTerms)
        case .qBfs(let c, let r, let a):
            finite("curvature", c)
            normRadius("norm_radius", r)
            series("coefficients", a, CoefficientSet.maxQTerms)
        case .xyPolynomial(let c, let k, let r, let a):
            finite("curvature", c)
            finite("conic", k)
            normRadius("norm_radius", r)
            series("coefficients", a, CoefficientSet.maxXYTerms)
        case .zernikeSag(let c, let k, let r, let o, let a):
            finite("curvature", c)
            finite("conic", k)
            normRadius("norm_radius", r)
            series("coefficients", a, o.maxTerms)
        case .sagOverlay(let base, let r, let w, let a, let angle, let sph):
            switch base {
            case .coordinateBreak, .paraxial, .diffractive:
                issue("invalid_sag_overlay_base", "\(path).base", "expected a geometric sag base")
            default: break
            }
            out += shapeIssues(base, path: "\(path).base", surface: i)
            normRadius("norm_radius", r)
            positive("invalid_wavelength", "test_wavelength_um", w)
            finite("astigmatism", a)
            finite("astigmatism_angle_deg", angle)
            finite("spherical", sph)
        case .diffractive(let base, let phase):
            var geometry = base
            while case .sagOverlay(let inner, _, _, _, _, _) = geometry { geometry = inner }
            switch geometry {
            case .plane, .conic, .evenAsphere: out += shapeIssues(base, path: "\(path).base", surface: i)
            default: issue("invalid_diffractive_base", "\(path).base", "expected plane, conic or even_asphere")
            }
            normRadius("phase.norm_radius", phase.normRadius)
            positive("invalid_design_wavelength", "phase.design_wavelength_um", phase.designWavelengthUm)
            if phase.designOrder == 0 {
                issue("invalid_diffraction_order", "\(path).phase.design_order", "design order must be nonzero")
            }
            series("phase.coefficients", phase.coefficients, CoefficientSet.maxPhaseTerms)
        case .toroid(let cx, let cy, let ky, let ya):
            finite("curvature_x", cx)
            finite("curvature_y", cy)
            finite("conic_y", ky)
            series("y_asphere", ya, CoefficientSet.maxAxisTerms)
        case .biconic(let cx, let cy, let kx, let ky, let xa, let ya):
            finite("curvature_x", cx)
            finite("curvature_y", cy)
            finite("conic_x", kx)
            finite("conic_y", ky)
            series("x_asphere", xa, CoefficientSet.maxAxisTerms)
            series("y_asphere", ya, CoefficientSet.maxAxisTerms)
        }
        return out
    }

    /// The first value that has no JSON form (the encoder stops there), as an
    /// error located by its coding path.
    func nonRepresentableValueIssues() -> [ModelIssue] {
        do {
            _ = try MKLightJSON.encode(self)
            return []
        } catch EncodingError.invalidValue(let value, let context) {
            let path = context.codingPath
            let surface = path.first?.stringValue == "surfaces" ? path.dropFirst().first?.intValue : nil
            let field = path.map { $0.intValue.map { "[\($0)]" } ?? ".\($0.stringValue)" }.joined()
            return [ModelIssue(.error, surface: surface, "\(field.drop { $0 == "." }) = \(value) is not a finite number")]
        } catch {
            return [ModelIssue(.error, "model cannot be serialised: \(error.localizedDescription)")]
        }
    }
}


/// Collects problems of numbers and references in the engine's
/// `[code] path: message` form, with the engine's codes and JSON paths
/// (`validate.rs`). A value that is not finite is left to the JSON check.
struct RuleIssues {
    private var issues: [ModelIssue] = []
    /// Surface index of the issues being added, if they concern one.
    var surface: Int?

    /// The issues added so far, which it forgets.
    mutating func take() -> [ModelIssue] {
        defer { issues = [] }
        return issues
    }

    mutating func add(_ code: String, _ path: String, _ message: String) {
        issues.append(ModelIssue(.error, surface: surface, "[\(code)] \(path): \(message)"))
    }

    mutating func positive(_ code: String, _ path: String, _ v: Double) {
        if v.isFinite && v <= 0 { add(code, path, "must be > 0 (got \(v))") }
    }

    mutating func nonNegative(_ code: String, _ path: String, _ v: Double) {
        if v.isFinite && v < 0 { add(code, path, "must be ≥ 0 (got \(v))") }
    }

    /// The wavelengths of a `(λ, value)` table increase strictly: the
    /// engine's interpolants search the table in that order.
    mutating func increasing(_ code: String, _ path: String, _ table: [SpectralSample]) {
        for k in table.indices.dropFirst() where table[k].um <= table[k - 1].um {
            add(code, "\(path)[\(k)][0]", "wavelengths must increase (\(table[k].um) follows \(table[k - 1].um))")
        }
    }

    mutating func material(_ m: MaterialRef, _ path: String) {
        switch m {
        case .air, .vacuum: break
        case .catalog(_, _, let offset):
            for (name, v, max) in [("dn_d", offset.dnD, MaterialOffset.maxDnD), ("dv_d", offset.dvD, MaterialOffset.maxDvD)]
            where v.isFinite && abs(v) > max {
                add("invalid_material", "\(path).offset.\(name)", "must be within ±\(max) (got \(v))")
            }
        case .model(let nd, let vd, _):
            // Its dispersion is the refractivity `n_d − 1` over `V_d`.
            if nd.isFinite && nd <= 1 { add("invalid_material", "\(path).nd", "must be > 1 (got \(nd))") }
            positive("invalid_material", "\(path).vd", vd)
        case .constant(let n): positive("invalid_material", "\(path).n", n)
        }
    }

    mutating func aperture(_ a: ApertureShape, _ path: String) {
        switch a {
        case .circular(let radius): positive("invalid_aperture", "\(path).radius", radius)
        case .annular(let inner, let outer):
            nonNegative("invalid_aperture", "\(path).inner_radius", inner)
            positive("invalid_aperture", "\(path).outer_radius", outer)
            if inner >= outer {
                add("invalid_aperture", path, "inner radius \(inner) must be below the outer radius \(outer)")
            }
        case .rectangular(let halfX, let halfY):
            positive("invalid_aperture", "\(path).half_x", halfX)
            positive("invalid_aperture", "\(path).half_y", halfY)
        case .elliptical(let semiX, let semiY):
            positive("invalid_aperture", "\(path).semi_x", semiX)
            positive("invalid_aperture", "\(path).semi_y", semiY)
        }
    }

    mutating func scatter(_ m: ScatterModel, _ path: String) {
        func tis(_ v: Double) {
            if v.isFinite && !(0...1).contains(v) {
                add("invalid_scatter", "\(path).tis", "a total integrated scatter must lie in [0, 1] (got \(v))")
            }
        }
        switch m {
        case .lambertian(let t): tis(t)
        case .abg(let a, let b, _):
            nonNegative("invalid_scatter", "\(path).a", a)
            nonNegative("invalid_scatter", "\(path).b", b)
        case .gaussian(let t, let sigmaSlope):
            tis(t)
            positive("invalid_scatter", "\(path).sigma_slope", sigmaSlope)
        }
    }

    mutating func coating(_ c: CoatingDef, _ path: String) {
        switch c {
        case .thinFilm(let reference, let layers, let substrate):
            positive("invalid_coating", "\(path).reference_wavelength_um", reference)
            for (j, layer) in layers.enumerated() {
                let p = "\(path).layers[\(j)]"
                material(layer.material, "\(p).material")
                nonNegative("invalid_coating", "\(p).thickness_um", layer.thicknessUm)
                if let q = layer.quarterWaves { nonNegative("invalid_coating", "\(p).quarter_waves", q) }
                nonNegative("invalid_coating", "\(p).extinction", layer.extinction)
            }
            if let substrate {
                positive("invalid_coating", "\(path).mirror_substrate[0]", substrate.first)
                nonNegative("invalid_coating", "\(path).mirror_substrate[1]", substrate.second)
            }
        case .ideal(let r, let t):
            if r.isFinite && t.isFinite && !(r >= 0 && t >= 0 && r + t <= 1 + 1e-12) {
                add("invalid_coating", path, "R = \(r) and T = \(t) must be ≥ 0 with R + T ≤ 1")
            }
        case .tabulated(let table):
            if table.isEmpty { add("invalid_coating", "\(path).reflectance", "a tabulated coating needs at least one sample") }
            increasing("invalid_coating", "\(path).reflectance", table)
            for (j, sample) in table.enumerated() {
                let p = "\(path).reflectance[\(j)]"
                positive("invalid_coating", "\(p)[0]", sample.um)
                if sample.value.isFinite && !(0...1).contains(sample.value) {
                    add("invalid_coating", p, "reflectance \(sample.value) outside [0, 1]")
                }
            }
        }
    }

    /// A map key as the engine's paths write it: in double quotes, with
    /// quotes, backslashes, newlines and tabs escaped.
    static func quoted(_ key: String) -> String {
        var out = "\""
        for c in key.unicodeScalars {
            switch c {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default: out.unicodeScalars.append(c)
            }
        }
        return out + "\""
    }
}
