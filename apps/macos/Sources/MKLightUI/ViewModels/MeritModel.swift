import Foundation
import MKLightEngine
import MKLightModel

// MARK: - Operands

/// One line of a custom merit function: the engine's `Operand`
/// (docs/OPTIMIZATION.md) with its kind kept as JSON, so parameters the app
/// does not edit (ray groups, detectors, ghost paths) survive a round trip.
public struct MeritOperand: Identifiable, Hashable, Sendable {
    public enum Comparison: String, CaseIterable, Sendable, Identifiable {
        case equal, lessThan = "less_than", greaterThan = "greater_than"
        public var id: String { rawValue }
        public var symbol: String {
            switch self {
            case .equal: "="
            case .lessThan: "≤"
            case .greaterThan: "≥"
            }
        }
    }

    /// Session identity (not saved).
    public var id = UUID()
    /// `{"type": "efl", …}`: the operand kind and its parameters.
    public var kind: [String: JSONValue]
    public var target = 0.0
    public var weight = 1.0
    public var comparison = Comparison.equal
    /// An explicit constraint rather than a weighted residual.
    public var isConstraint = false
    /// `{"field": "all"}`, `{"field": "index", "index": i}` or normalised.
    public var field: JSONValue = .object(["field": .string("all")])
    /// `{"wavelength": "primary" | "all" | "index", …}`.
    public var wavelength: JSONValue = .object(["wavelength": .string("primary")])
    public var config = 0
    public var enabled = true
    public var label = ""

    public init(kind: [String: JSONValue]) {
        self.kind = kind
        if let d = OperandCatalog.descriptor(kind["type"]?.string ?? ""), d.singleField {
            field = .object(["field": .string("index"), "index": .integer(0)])
        }
    }

    public var type: String { kind["type"]?.string ?? "?" }
    public var descriptor: OperandDescriptor? { OperandCatalog.descriptor(type) }

    /// The engine's JSON form.
    public var json: JSONValue {
        var o: [String: JSONValue] = [
            "kind": .object(kind), "target": .number(target), "weight": .number(weight),
            "comparison": .string(comparison.rawValue), "role": .string(isConstraint ? "constraint" : "residual"),
            "field": field, "wavelength": wavelength, "config": .integer(Int64(config)), "enabled": .bool(enabled),
        ]
        if !label.isEmpty { o["label"] = .string(label) }
        return .object(o)
    }

    /// Reads an engine operand (a spec's or a result's operand table).
    public init?(json: JSONValue) {
        guard let k = json["kind"]?.object, k["type"]?.string != nil else { return nil }
        kind = k
        target = json["target"]?.nf ?? 0
        weight = json["weight"]?.nf ?? 1
        comparison = json["comparison"]?.string.flatMap(Comparison.init(rawValue:)) ?? .equal
        isConstraint = json["role"]?.string == "constraint"
        field = json["field"] ?? .object(["field": .string("all")])
        wavelength = json["wavelength"] ?? .object(["wavelength": .string("primary")])
        config = json["config"]?.int ?? 0
        enabled = json["enabled"]?.bool ?? true
        label = json["label"]?.string ?? ""
    }

    /// Residual before weighting (`v − t`, one-sided for inequalities).
    public func residual(_ value: Double) -> Double {
        let d = value - target
        switch comparison {
        case .equal: return d
        case .lessThan: return max(d, 0)
        case .greaterThan: return min(d, 0)
        }
    }

    /// Problems the app can see before the engine does.
    public func localIssues(fieldCount: Int, wavelengthCount: Int) -> [String] {
        var out: [String] = []
        guard let d = descriptor else { return [] }
        if d.singleField && field["field"]?.string == "all" { out.append("Needs one field, not all fields.") }
        if let i = field["index"]?.int, field["field"]?.string == "index", i >= fieldCount { out.append("Field \(i + 1) does not exist.") }
        if let i = wavelength["index"]?.int, wavelength["wavelength"]?.string == "index", i >= wavelengthCount {
            out.append("Wavelength \(i + 1) does not exist.")
        }
        for p in d.params where p.required && (kind[p.key] == nil || kind[p.key] == .null) { out.append("\(p.title) is required.") }
        if !weight.isFinite || weight < 0 { out.append("The weight must be zero or positive.") }
        return out
    }
}

/// How an operand parameter is edited.
public enum OperandParamKind: Sendable, Hashable {
    case number(unit: String)
    case integer(min: Int)
    /// A stable surface id; optional ones default to the engine's choice
    /// (usually the image).
    case surface
    /// A system wavelength index.
    case wavelength
    case choice([String])
    case flag
    /// `rings` / `arms` pupil sampling.
    case sampling
    /// Earlier operand indices (`combine`).
    case operandList
}

public struct OperandParam: Sendable, Hashable, Identifiable {
    public var key: String
    public var title: String
    public var kind: OperandParamKind
    public var required = false
    public var defaultValue: JSONValue?
    public var id: String { key }
}

/// An operand kind of the engine's operand reference.
public struct OperandDescriptor: Sendable, Identifiable {
    public var type: String
    public var title: String
    public var category: String
    public var unit: String
    public var summary: String
    public var params: [OperandParam] = []
    /// Defined at one field point (`f1` in the operand reference).
    public var singleField = false
    public var id: String { type }

    /// The kind object for a new operand.
    public var newKind: [String: JSONValue] {
        var k: [String: JSONValue] = ["type": .string(type)]
        for p in params { if let v = p.defaultValue { k[p.key] = v } }
        return k
    }
}

/// The engine's operands (docs/OPTIMIZATION.md, operand reference), grouped
/// by category for the searchable picker.
public enum OperandCatalog {
    public static let categories = ["First order", "Geometry", "Real rays", "Image quality", "Ghosts and stray light", "Manufacturing", "Composition"]

    static func n(_ key: String, _ title: String, unit: String = "", def: Double = 0, required: Bool = true) -> OperandParam {
        OperandParam(key: key, title: title, kind: .number(unit: unit), required: required, defaultValue: required ? .number(def) : nil)
    }
    static func surface(_ key: String = "surface", _ title: String = "Surface", required: Bool = false) -> OperandParam {
        OperandParam(key: key, title: title, kind: .surface, required: required)
    }
    static func choice(_ key: String, _ title: String, _ options: [String], def: String? = nil) -> OperandParam {
        OperandParam(key: key, title: title, kind: .choice(options), defaultValue: def.map(JSONValue.string))
    }
    static func wl(_ key: String, _ title: String, def: Int) -> OperandParam {
        OperandParam(key: key, title: title, kind: .wavelength, required: true, defaultValue: .integer(Int64(def)))
    }
    static let pupil: [OperandParam] = [n("px", "Pupil x", def: 0), n("py", "Pupil y", def: 1)]
    static let sampling = OperandParam(key: "sampling", title: "Pupil sampling", kind: .sampling,
                                       defaultValue: .object(["rings": .integer(3), "arms": .integer(6)]))

    public static let all: [OperandDescriptor] = [
        // First order
        .init(type: "efl", title: "EFL", category: "First order", unit: "mm", summary: "Effective focal length (of the y section)"),
        .init(type: "efl_x", title: "EFL x", category: "First order", unit: "mm", summary: "Focal length of the x section"),
        .init(type: "efl_y", title: "EFL y", category: "First order", unit: "mm", summary: "Focal length of the y section"),
        .init(type: "bfl", title: "BFL", category: "First order", unit: "mm", summary: "Back focal length: last surface to paraxial focus"),
        .init(type: "bfl_x", title: "BFL x", category: "First order", unit: "mm", summary: "Back focus of the x section"),
        .init(type: "bfl_y", title: "BFL y", category: "First order", unit: "mm", summary: "Back focus of the y section"),
        .init(type: "anamorphic_squeeze", title: "Anamorphic squeeze", category: "First order", unit: "", summary: "Signed EFL_y / EFL_x"),
        .init(type: "f_number", title: "F-number", category: "First order", unit: "", summary: "Image-space, working or real working F/#",
              params: [choice("kind", "Kind", ["paraxial_image_space", "paraxial_working", "real_working"], def: "paraxial_image_space")]),
        .init(type: "magnification", title: "Magnification", category: "First order", unit: "", summary: "Paraxial lateral magnification"),
        .init(type: "entrance_pupil_diameter", title: "Entrance pupil diameter", category: "First order", unit: "mm", summary: ""),
        .init(type: "entrance_pupil_position", title: "Entrance pupil position", category: "First order", unit: "mm", summary: "From the first surface"),
        .init(type: "exit_pupil_position", title: "Exit pupil position", category: "First order", unit: "mm", summary: "From the image surface"),
        .init(type: "paraxial_ray", title: "Paraxial ray", category: "First order", unit: "mm / – / °",
              summary: "Marginal or chief ray height, slope or angle after a surface",
              params: [choice("ray", "Ray", ["marginal", "chief"], def: "marginal"), choice("quantity", "Quantity", ["height", "slope", "angle"], def: "height"), surface()]),
        .init(type: "petzval_sum", title: "Petzval sum", category: "First order", unit: "1/mm", summary: "Σ c (N′ − N)/(N N′)"),
        .init(type: "axial_colour", title: "Axial colour", category: "First order", unit: "mm", summary: "Paraxial focus at short minus long λ",
              params: [wl("short", "Short λ", def: 0), wl("long", "Long λ", def: 2)]),
        // Geometry
        .init(type: "total_track", title: "Total track", category: "Geometry", unit: "mm", summary: "Sum of thicknesses",
              params: [surface("from", "From"), surface("to", "To")]),
        .init(type: "centre_thickness", title: "Centre thickness", category: "Geometry", unit: "mm", summary: "Axial gap after a surface",
              params: [surface(required: true), surface("to", "To")]),
        .init(type: "edge_thickness", title: "Edge thickness", category: "Geometry", unit: "mm", summary: "Gap at the rim (or a height)",
              params: [surface(required: true), surface("to", "To"), n("height", "Height", unit: "mm", required: false)]),
        .init(type: "min_thickness", title: "Minimum thickness", category: "Geometry", unit: "mm", summary: "Smallest gap of a medium",
              params: [choice("measure", "Measure", ["centre", "edge"], def: "centre"), choice("medium", "Medium", ["glass", "air", "any"], def: "glass"),
                       surface("from", "From"), surface("to", "To")]),
        .init(type: "max_thickness", title: "Maximum thickness", category: "Geometry", unit: "mm", summary: "Largest gap of a medium",
              params: [choice("measure", "Measure", ["centre", "edge"], def: "centre"), choice("medium", "Medium", ["glass", "air", "any"], def: "glass"),
                       surface("from", "From"), surface("to", "To")]),
        // Real rays
        .init(type: "real_ray", title: "Real ray", category: "Real rays", unit: "mm / – / °", summary: "Position, direction or angle of one ray",
              params: [choice("quantity", "Quantity", ["x", "y", "z", "l", "m", "n", "angle_x", "angle_y", "incidence_angle", "optical_path"], def: "y"),
                       surface()] + pupil + [OperandParam(key: "global", title: "Global frame", kind: .flag)], singleField: true),
        .init(type: "ray_aberration", title: "Ray aberration", category: "Real rays", unit: "mm", summary: "Transverse aberration of one ray",
              params: [choice("axis", "Axis", ["x", "y"], def: "y")] + pupil + [choice("reference", "Reference", ["chief", "centroid"], def: "chief")],
              singleField: true),
        .init(type: "wavefront_aberration", title: "Wavefront aberration", category: "Real rays", unit: "waves", summary: "OPD of one ray",
              params: pupil + [choice("reference", "Reference", ["chief", "centroid"], def: "chief")], singleField: true),
        .init(type: "chief_ray_angle", title: "Chief ray angle", category: "Real rays", unit: "°", summary: "Incidence of the chief ray (0 telecentric)",
              params: [surface()], singleField: true),
        // Image quality
        .init(type: "rms_spot", title: "RMS spot radius", category: "Image quality", unit: "mm", summary: "About the centroid or the chief ray",
              params: [choice("reference", "Reference", ["centroid", "chief"], def: "centroid"), sampling]),
        .init(type: "spot_centroid", title: "Spot centroid", category: "Image quality", unit: "mm", summary: "Centroid minus the chief ray",
              params: [choice("axis", "Axis", ["x", "y"], def: "y")], singleField: true),
        .init(type: "wavefront_rms", title: "RMS wavefront", category: "Image quality", unit: "waves", summary: "Piston (and tilt) removed",
              params: [choice("reference", "Reference", ["centroid", "chief"], def: "centroid"), sampling]),
        .init(type: "mtf", title: "MTF", category: "Image quality", unit: "", summary: "Modulation at one frequency; use ≥ for a lower bound",
              params: [n("frequency", "Frequency", unit: "cyc/mm", def: 30), choice("orientation", "Orientation", ["tangential", "sagittal", "average"], def: "average"),
                       choice("method", "Method", ["diffraction", "geometric"], def: "diffraction"), n("defocus_mm", "Defocus", unit: "mm", required: false)]),
        .init(type: "distortion", title: "Distortion", category: "Image quality", unit: "%", summary: "Real chief ray against the mapping",
              params: [choice("mapping", "Mapping", ["f_tan_theta", "f_theta"], def: "f_tan_theta")], singleField: true),
        .init(type: "lateral_colour", title: "Lateral colour", category: "Image quality", unit: "mm", summary: "Chief height at short minus long λ",
              params: [wl("short", "Short λ", def: 0), wl("long", "Long λ", def: 2)], singleField: true),
        .init(type: "relative_illumination", title: "Relative illumination", category: "Image quality", unit: "", summary: "Against on axis",
              singleField: true),
        // Ghosts and stray light
        .init(type: "ghost_focus", title: "Ghost focus distance", category: "Ghosts and stray light", unit: "mm",
              summary: "Distance of the strongest ghosts' focus from the image (larger is better)",
              params: [OperandParam(key: "worst", title: "Worst ghosts", kind: .integer(min: 1), defaultValue: .integer(1)),
                       choice("metric", "Metric", ["distance", "peak"], def: "distance")]),
        .init(type: "ghost_irradiance", title: "Ghost irradiance", category: "Ghosts and stray light", unit: "rel.", summary: "Ghost against primary",
              params: [choice("metric", "Metric", ["peak", "mean", "power"], def: "peak"),
                       OperandParam(key: "max_reflections", title: "Max reflections", kind: .integer(min: 2), defaultValue: .integer(2))]),
        .init(type: "stray_light_power", title: "Stray light power", category: "Ghosts and stray light", unit: "rel.",
              summary: "Monte-Carlo fraction; not for least squares",
              params: [choice("class", "Class", ["ghost", "scatter", "mechanical", "sneak", "all_non_primary"], def: "all_non_primary"),
                       OperandParam(key: "rays", title: "Rays", kind: .integer(min: 1000), defaultValue: .integer(100_000))]),
        // Manufacturing
        .init(type: "doe_min_zone_period", title: "DOE minimum zone period", category: "Manufacturing", unit: "mm", summary: "", params: [surface(required: true)]),
        .init(type: "doe_max_phase_slope", title: "DOE maximum phase slope", category: "Manufacturing", unit: "rad/mm", summary: "", params: [surface(required: true)]),
        .init(type: "doe_total_phase", title: "DOE total phase", category: "Manufacturing", unit: "rad", summary: "", params: [surface(required: true)]),
        .init(type: "q_max_departure", title: "Q-type maximum departure", category: "Manufacturing", unit: "mm", summary: "From the best-fit sphere",
              params: [surface(required: true)]),
        .init(type: "q_max_departure_slope", title: "Q-type departure slope", category: "Manufacturing", unit: "mm/mm", summary: "",
              params: [surface(required: true)]),
        .init(type: "freeform_max_departure", title: "Freeform maximum departure", category: "Manufacturing", unit: "mm",
              summary: "XY or Zernike sag from its base conic", params: [surface(required: true)]),
        // Composition
        .init(type: "combine", title: "Combine", category: "Composition", unit: "", summary: "Arithmetic on earlier operands",
              params: [choice("op", "Operation", ["sum", "difference", "product", "quotient", "abs", "min", "max"], def: "sum"),
                       OperandParam(key: "args", title: "Operands", kind: .operandList, required: true, defaultValue: .array([]))]),
        .init(type: "param_value", title: "Parameter value", category: "Composition", unit: "", summary: "Any model parameter"),
        .init(type: "tolerance_sensitivity", title: "Tolerance sensitivity", category: "Composition", unit: "", summary: "Of a criterion to a parameter"),
    ]

    static let byType = Dictionary(all.map { ($0.type, $0) }, uniquingKeysWith: { a, _ in a })

    public static func descriptor(_ type: String) -> OperandDescriptor? { byType[type] }

    /// Operands a user adds from the picker (the composition kinds that
    /// need a model path or a nested criterion are read and kept, not added).
    public static var addable: [OperandDescriptor] { all.filter { !["param_value", "tolerance_sensitivity"].contains($0.type) } }

    public static func title(_ type: String) -> String {
        descriptor(type)?.title ?? type.replacingOccurrences(of: "_", with: " ").capitalized
    }

    /// Descriptors matching a search, by category.
    public static func search(_ text: String) -> [(category: String, items: [OperandDescriptor])] {
        let q = text.trimmingCharacters(in: .whitespaces).lowercased()
        let hits = addable.filter {
            q.isEmpty || $0.title.lowercased().contains(q) || $0.type.contains(q) || $0.summary.lowercased().contains(q)
                || $0.category.lowercased().contains(q)
        }
        return categories.compactMap { c in
            let items = hits.filter { $0.category == c }
            return items.isEmpty ? nil : (c, items)
        }
    }
}

// MARK: - Default merit

/// The engine's default merit generator (`DefaultMeritSpec`): criterion,
/// pupil sampling, fields and wavelengths, thickness boundaries and an
/// optional anamorphic squeeze target.
public struct DefaultMeritSettings: Hashable, Sendable {
    public enum Criterion: String, CaseIterable, Sendable, Identifiable {
        case rmsSpot = "rms_spot", wavefront, mtf
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .rmsSpot: "RMS spot radius"
            case .wavefront: "RMS wavefront"
            case .mtf: "MTF"
            }
        }
    }
    public var criterion = Criterion.rmsSpot
    /// Spot reference: centroid or chief.
    public var reference = "centroid"
    /// Spot centroid per wavelength (`monochromatic`) or of the white-light spot.
    public var scope = "monochromatic"
    public var mtfFrequencies: [Double] = [10, 30]
    public var mtfTarget = 0.5
    public var mtfMethod = "diffraction"
    public var worstField = false
    public var rings = 3
    public var arms = 6
    /// `nil`: every field (or wavelength) with positive weight.
    public var fields: [Int]?
    public var wavelengths: [Int]?
    public var thickness = true
    public var glassMinCentre: Double? = 1.0
    public var glassMaxCentre: Double? = 30.0
    public var glassMinEdge: Double? = 1.0
    public var airMinCentre: Double? = 0.1
    public var airMaxCentre: Double?
    public var airMinEdge: Double? = 0.1
    public var squeezeTarget: Double?

    public init() {}

    /// The `default_merit` settings object (laid over the engine's defaults).
    public var json: JSONValue {
        var crit: [String: JSONValue] = ["type": .string(criterion.rawValue)]
        switch criterion {
        case .rmsSpot:
            crit["reference"] = .string(reference)
            crit["scope"] = .string(scope)
        case .wavefront:
            crit["reference"] = .string(reference)
        case .mtf:
            crit["frequencies"] = .array(mtfFrequencies.map(JSONValue.number))
            crit["target"] = .number(mtfTarget)
            crit["method"] = .string(mtfMethod)
            crit["worst_field"] = .bool(worstField)
        }
        var o: [String: JSONValue] = [
            "criterion": .object(crit),
            "sampling": .object(["rings": .integer(Int64(rings)), "arms": .integer(Int64(arms))]),
            "fields": fields.map { .array($0.map { .integer(Int64($0)) }) } ?? .null,
            "wavelengths": wavelengths.map { .array($0.map { .integer(Int64($0)) }) } ?? .null,
        ]
        if thickness {
            func v(_ d: Double?) -> JSONValue { d.map(JSONValue.number) ?? .null }
            o["thickness"] = .object([
                "glass_min_centre": v(glassMinCentre), "glass_max_centre": v(glassMaxCentre), "glass_min_edge": v(glassMinEdge),
                "air_min_centre": v(airMinCentre), "air_max_centre": v(airMaxCentre), "air_min_edge": v(airMinEdge),
            ])
        } else {
            o["thickness"] = .null
        }
        if let squeezeTarget { o["squeeze_target"] = .number(squeezeTarget) }
        return .object(o)
    }

    public init(json: JSONValue?) {
        guard let j = json else { return }
        let c = j["criterion"]
        criterion = c?["type"]?.string.flatMap(Criterion.init(rawValue:)) ?? .rmsSpot
        reference = c?["reference"]?.string ?? reference
        scope = c?["scope"]?.string ?? scope
        mtfFrequencies = c?["frequencies"]?.nfArray ?? mtfFrequencies
        mtfTarget = c?["target"]?.nf ?? mtfTarget
        mtfMethod = c?["method"]?.string ?? mtfMethod
        worstField = c?["worst_field"]?.bool ?? false
        rings = j[path: "sampling", "rings"]?.int ?? rings
        arms = j[path: "sampling", "arms"]?.int ?? arms
        fields = j["fields"]?.intArray
        wavelengths = j["wavelengths"]?.intArray
        if let t = j["thickness"] {
            thickness = t != .null
            if thickness {
                glassMinCentre = Self.number(t["glass_min_centre"])
                glassMaxCentre = Self.number(t["glass_max_centre"])
                glassMinEdge = Self.number(t["glass_min_edge"])
                airMinCentre = Self.number(t["air_min_centre"])
                airMaxCentre = Self.number(t["air_max_centre"])
                airMinEdge = Self.number(t["air_min_edge"])
            }
        }
        squeezeTarget = Self.number(j["squeeze_target"])
    }

    /// An optional number: absent or `null` is `nil`.
    static func number(_ v: JSONValue?) -> Double? {
        guard let v, v != .null, let d = v.nf, d.isFinite else { return nil }
        return d
    }

    /// A one-line description for the panel.
    public var summary: String {
        var s: String
        switch criterion {
        case .rmsSpot: s = "RMS spot radius"
        case .wavefront: s = "RMS wavefront"
        case .mtf:
            s = "MTF ≥ \(Fmt.short(mtfTarget)) at " + mtfFrequencies.map { Fmt.short($0) }.joined(separator: ", ") + " cyc/mm"
            if worstField { s += ", weakest field" }
        }
        s += " · \(rings) rings × \(arms) arms"
        if thickness { s += " · thickness limits" }
        if let squeezeTarget { s += " · squeeze \(Fmt.short(squeezeTarget))" }
        return s
    }
}

// MARK: - Saved merit

/// The merit function of a design, saved in the project's `merit` section:
/// the default generator's settings, or a custom operand list (an engine
/// `MeritFunctionSpec`), and whether the focal length is held.
public struct MeritSettings: Hashable, Sendable {
    public enum Mode: String, Sendable { case `default`, custom }
    public var mode = Mode.default
    public var defaults = DefaultMeritSettings()
    public var operands: [MeritOperand] = []
    public var holdEFL = true

    public init() {}

    public init(section: JSONValue?) {
        guard let s = section, s.object != nil else { return }
        mode = s["mode"]?.string.flatMap(Mode.init(rawValue:)) ?? .default
        defaults = DefaultMeritSettings(json: s["default_merit"])
        operands = (s[path: "spec", "operands"]?.array ?? []).compactMap(MeritOperand.init(json:))
        holdEFL = s["hold_efl"]?.bool ?? true
    }

    public var section: JSONValue {
        .object([
            "mode": .string(mode.rawValue), "default_merit": defaults.json, "hold_efl": .bool(holdEFL),
            "spec": spec,
        ])
    }

    /// The custom operands as a `MeritFunctionSpec`.
    public var spec: JSONValue {
        .object(["schema_version": .integer(1), "name": .string(""), "operands": .array(operands.map(\.json))])
    }

    /// The merit part of `optimize` / `glass_substitution` settings.
    public var jobSettings: [String: JSONValue] {
        var s: [String: JSONValue] = ["hold_efl": .bool(holdEFL)]
        switch mode {
        case .default: s["default_merit"] = defaults.json
        case .custom: s["merit"] = spec
        }
        return s
    }

    /// Hashable ignores the operands' session ids.
    public static func == (a: MeritSettings, b: MeritSettings) -> Bool { a.section == b.section }
    public func hash(into h: inout Hasher) { h.combine(section) }

    // MARK: Editing

    public mutating func add(_ d: OperandDescriptor, after index: Int? = nil) -> UUID {
        let op = MeritOperand(kind: d.newKind)
        let at = index.map { min($0 + 1, operands.count) } ?? operands.count
        operands.insert(op, at: at)
        remapCombine { $0 >= at ? $0 + 1 : $0 }
        return op.id
    }

    public mutating func delete(_ ids: Set<UUID>) {
        let removed = operands.indices.filter { ids.contains(operands[$0].id) }
        guard !removed.isEmpty else { return }
        operands.removeAll { ids.contains($0.id) }
        remapCombine { old in
            if removed.contains(old) { return nil }
            return old - removed.filter { $0 < old }.count
        }
    }

    public mutating func duplicate(_ ids: Set<UUID>) -> [UUID] {
        var out: [UUID] = []
        for i in operands.indices.reversed() where ids.contains(operands[i].id) {
            var copy = operands[i]
            copy.id = UUID()
            operands.insert(copy, at: i + 1)
            let at = i + 1
            remapCombine { $0 >= at ? $0 + 1 : $0 }
            out.append(copy.id)
        }
        return out
    }

    /// Moves operands (List `onMove` semantics), keeping `combine` references.
    public mutating func move(fromOffsets source: IndexSet, toOffset destination: Int) {
        let before = operands.map(\.id)
        operands.move(fromOffsets: source, toOffset: destination)
        let newIndex = Dictionary(uniqueKeysWithValues: operands.enumerated().map { ($0.element.id, $0.offset) })
        remapCombine { old in old < before.count ? newIndex[before[old]] : nil }
    }

    /// Rewrites `combine` argument indices (`nil` drops a reference).
    mutating func remapCombine(_ f: (Int) -> Int?) {
        for i in operands.indices where operands[i].type == "combine" {
            let args = operands[i].kind["args"]?.intArray ?? []
            operands[i].kind["args"] = .array(args.compactMap(f).map { .integer(Int64($0)) })
        }
    }
}

// MARK: - Evaluation

/// The merit and its operands at the current design, from a one-iteration
/// `optimize` run (the engine has no evaluate-only call): its `start`
/// values are the design's, on the CPU f64 reference.
public struct MeritEvaluation: Sendable, Equatable {
    public struct Row: Sendable, Equatable {
        public var index: Int
        public var kind: String
        public var label: String
        public var isConstraint: Bool
        public var value: Double
        public var target: Double
        public var contribution: Double
        /// Share of the merit (0…1); NaN for constraints.
        public var fraction: Double
        public var error: String?
    }
    public var merit: Double
    public var violation: Double
    public var rows: [Row]
    /// The effective spec the engine used (generated operands included).
    public var spec: JSONValue?

    public init?(json: JSONValue) {
        guard json.object != nil else { return nil }
        merit = json[path: "start", "merit"]?.nf ?? .nan
        violation = json[path: "start", "violation"]?.nf ?? .nan
        spec = json["merit_spec"]
        var rows: [Row] = []
        for (k, o) in (json["operands"]?.array ?? []).enumerated() {
            let value = o["start"]?.nf ?? .nan
            let target = o["target"]?.nf ?? 0
            let weight = o["weight"]?.nf ?? 0
            let constraint = o["role"]?.string == "constraint"
            let enabled = o["enabled"]?.bool ?? true
            var op = MeritOperand(kind: ["type": .string(o["kind"]?.string ?? "")])
            op.target = target
            op.comparison = o["comparison"]?.string.flatMap(MeritOperand.Comparison.init(rawValue:)) ?? .equal
            let r = op.residual(value)
            let c = constraint || !enabled ? Double.nan : weight * r * r
            rows.append(
                Row(
                    index: o["index"]?.int ?? k, kind: o["kind"]?.string ?? "", label: o["label"]?.string ?? "", isConstraint: constraint,
                    value: value, target: target, contribution: c, fraction: .nan, error: o["error"]?.string))
        }
        let total = rows.map(\.contribution).filter(\.isFinite).reduce(0, +)
        for i in rows.indices where rows[i].contribution.isFinite {
            rows[i].fraction = total > 0 ? rows[i].contribution / total : 0
        }
        self.rows = rows
    }

    public func row(_ index: Int) -> Row? { rows.first { $0.index == index } }

    /// The generated (or custom) operands of the effective spec.
    public var specOperands: [MeritOperand] { (spec?["operands"]?.array ?? []).compactMap(MeritOperand.init(json:)) }
}
