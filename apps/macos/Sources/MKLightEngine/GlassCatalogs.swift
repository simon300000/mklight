import Foundation
import MKLightModel

// The engine's glass catalog management (docs/FFI.md, "Glass catalog
// management"): table rows, glass details, reload reports, availability
// overrides and user glasses. Readers are tolerant: a missing optional value
// is `nil`.

/// A catalog glass's availability (AGF status code).
public enum GlassStatus: Hashable, Sendable, Codable, Identifiable {
    case standard, preferred, obsolete, special, melt
    case other(Int)

    /// The statuses a user chooses from.
    public static let choices: [GlassStatus] = [.preferred, .standard, .special, .melt, .obsolete]

    public var id: Int { code }

    public var code: Int {
        switch self {
        case .standard: 0
        case .preferred: 1
        case .obsolete: 2
        case .special: 3
        case .melt: 4
        case .other(let c): c
        }
    }

    public var title: String {
        switch self {
        case .standard: "Standard"
        case .preferred: "Preferred"
        case .obsolete: "Obsolete"
        case .special: "Special"
        case .melt: "Melt"
        case .other(let c): "Status \(c)"
        }
    }

    /// The engine's JSON form (`"preferred"`, or `{"other": 7}`).
    public var json: JSONValue {
        switch self {
        case .standard: .string("standard")
        case .preferred: .string("preferred")
        case .obsolete: .string("obsolete")
        case .special: .string("special")
        case .melt: .string("melt")
        case .other(let c): .object(["other": .integer(Int64(c))])
        }
    }

    public init?(json: JSONValue?) {
        switch json {
        case .string("standard")?: self = .standard
        case .string("preferred")?: self = .preferred
        case .string("obsolete")?: self = .obsolete
        case .string("special")?: self = .special
        case .string("melt")?: self = .melt
        case let v?: guard let c = v["other"]?.int else { return nil }
            self = .other(c)
        case nil: return nil
        }
    }

    public init(from decoder: any Decoder) throws {
        let v = try JSONValue(from: decoder)
        guard let s = GlassStatus(json: v) else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "not a glass status"))
        }
        self = s
    }

    public func encode(to encoder: any Encoder) throws { try json.encode(to: encoder) }
}

/// The user's availability setting for one catalog glass. `nil` keeps the
/// catalog's value.
public struct GlassOverride: Codable, Hashable, Sendable {
    public var catalog: String
    public var name: String
    public var status: GlassStatus?
    public var excludeSubstitution: Bool?

    public init(catalog: String, name: String, status: GlassStatus? = nil, excludeSubstitution: Bool? = nil) {
        self.catalog = catalog
        self.name = name
        self.status = status
        self.excludeSubstitution = excludeSubstitution
    }

    public var isEmpty: Bool { status == nil && excludeSubstitution == nil }

    /// Whether this override is for `catalog`'s glass `name` (case-insensitive, as the engine matches).
    public func matches(catalog c: String, name n: String) -> Bool {
        catalog.caseInsensitiveCompare(c) == .orderedSame && name.caseInsensitiveCompare(n) == .orderedSame
    }

    private enum CodingKeys: String, CodingKey { case catalog, name, status, excludeSubstitution = "exclude_substitution" }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        catalog = try c.decode(String.self, forKey: .catalog)
        name = try c.decode(String.self, forKey: .name)
        status = try c.decodeIfPresent(GlassStatus.self, forKey: .status)
        excludeSubstitution = try c.decodeIfPresent(Bool.self, forKey: .excludeSubstitution)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(catalog, forKey: .catalog)
        try c.encode(name, forKey: .name)
        try c.encodeIfPresent(status, forKey: .status)
        try c.encodeIfPresent(excludeSubstitution, forKey: .excludeSubstitution)
    }
}

private func finite(_ v: JSONValue?) -> Double? {
    guard let d = v?.doubleValue, d.isFinite else { return nil }
    return d
}

/// One row of a glass table (`mk_session_list_glasses`).
public struct GlassRow: Hashable, Sendable, Identifiable {
    public var catalog: String
    public var name: String
    public var nd: Double
    public var vd: Double
    public var pgf: Double?
    public var dpgf: Double?
    public var minUm: Double
    public var maxUm: Double
    /// In effect, after the user's overrides.
    public var status: GlassStatus
    public var excludeSubstitution: Bool
    /// The user's overrides changed status or exclusion.
    public var overridden: Bool
    public var density: Double?
    public var relativeCost: Double?
    /// 10⁻⁶/K, −30…+70 °C.
    public var tce: Double?
    public var climateResistance: Double?
    public var stainResistance: Double?
    public var transmissionMinUm: Double?
    public var transmissionMaxUm: Double?
    public var lambda80Um: Double?

    public var id: String { catalog + ":" + name }

    public init?(json v: JSONValue) {
        guard let catalog = v["catalog"]?.string, let name = v["name"]?.string else { return nil }
        self.catalog = catalog
        self.name = name
        nd = v["nd"]?.doubleValue ?? .nan
        vd = v["vd"]?.doubleValue ?? .nan
        pgf = finite(v["pgf"])
        dpgf = finite(v["dpgf"])
        minUm = v["min_um"]?.doubleValue ?? 0
        maxUm = v["max_um"]?.doubleValue ?? .infinity
        status = GlassStatus(json: v["status"]) ?? .standard
        excludeSubstitution = v["exclude_substitution"]?.bool ?? false
        overridden = v["overridden"]?.bool ?? false
        density = finite(v["density"])
        relativeCost = finite(v["relative_cost"])
        tce = finite(v["tce"])
        climateResistance = finite(v["climate_resistance"])
        stainResistance = finite(v["stain_resistance"])
        transmissionMinUm = finite(v["transmission_min_um"])
        transmissionMaxUm = finite(v["transmission_max_um"])
        lambda80Um = finite(v["lambda80_um"])
    }

    /// Whether the valid range is unrestricted (no `LD` record).
    public var unrestrictedRange: Bool { maxUm > 1e100 }
}

/// Everything about one glass (`mk_session_glass_details`).
public struct GlassDetails: Sendable, Hashable {
    public struct LineIndex: Sendable, Hashable {
        public var line: String
        public var um: Double
        public var n: Double
    }

    public struct Model: Sendable, Hashable {
        public var nd: Double
        public var vd: Double
        public var dpgf: Double
    }

    public struct Transmittance: Sendable, Hashable {
        public var um: Double
        public var tau: Double
        public var thicknessMm: Double
    }

    public var row: GlassRow
    public var catalogStatus: GlassStatus
    public var catalogExcludeSubstitution: Bool
    public var formula: Int?
    public var formulaName: String
    public var coefficients: [Double]
    public var computedNd: Double?
    public var computedVd: Double?
    public var model: Model?
    public var catalogDpgf: Double?
    public var tce100To300: Double?
    public var milCode: String?
    public var meltFrequency: Int?
    public var acidResistance: Double?
    public var alkaliResistance: Double?
    public var phosphateResistance: Double?
    public var hasThermalData: Bool
    public var transmittance: [Transmittance]
    public var comment: String
    public var lines: [LineIndex]
    /// `(λ µm, n)` across the glass's range.
    public var indexCurve: [(um: Double, n: Double)]
    /// The glass in the engine's catalog JSON form (before overrides).
    public var glass: JSONValue

    public init?(json v: JSONValue) {
        guard let row = GlassRow(json: v), let glass = v["glass"], glass.object != nil else { return nil }
        self.row = row
        catalogStatus = GlassStatus(json: v["catalog_status"]) ?? row.status
        catalogExcludeSubstitution = v["catalog_exclude_substitution"]?.bool ?? row.excludeSubstitution
        formula = v["formula"]?.int
        formulaName = v["formula_name"]?.string ?? ""
        coefficients = (v["coefficients"]?.array ?? []).compactMap(\.doubleValue)
        computedNd = finite(v["computed_nd"])
        computedVd = finite(v["computed_vd"])
        if let m = v["model"], let nd = finite(m["nd"]), let vd = finite(m["vd"]) {
            model = Model(nd: nd, vd: vd, dpgf: finite(m["dpgf"]) ?? 0)
        }
        catalogDpgf = finite(v["catalog_dpgf"])
        tce100To300 = finite(v["tce_100_300"])
        milCode = v["mil_code"]?.string
        meltFrequency = v["melt_frequency"]?.int
        acidResistance = finite(v["acid_resistance"])
        alkaliResistance = finite(v["alkali_resistance"])
        phosphateResistance = finite(v["phosphate_resistance"])
        hasThermalData = v["thermal"]?.object != nil
        transmittance = (v["transmittance"]?.array ?? []).compactMap { t in
            guard let um = finite(t["um"]), let tau = finite(t["tau"]), let th = finite(t["thickness_mm"]) else { return nil }
            return Transmittance(um: um, tau: tau, thicknessMm: th)
        }
        comment = v["comment"]?.string ?? ""
        lines = (v["lines"]?.array ?? []).compactMap { l in
            guard let name = l["line"]?.string, let um = finite(l["um"]), let n = finite(l["n"]) else { return nil }
            return LineIndex(line: name, um: um, n: n)
        }
        indexCurve = (v["index_curve"]?.array ?? []).compactMap { p in
            guard let a = p.array, a.count == 2, let l = a[0].doubleValue, let n = a[1].doubleValue else { return nil }
            return (l, n)
        }
        self.glass = glass
    }

    public static func == (a: Self, b: Self) -> Bool { a.row == b.row && a.glass == b.glass }
    public func hash(into h: inout Hasher) {
        h.combine(row)
        h.combine(glass)
    }
}

/// What reloading a catalog changed (`mk_session_reload_catalog`).
public struct CatalogChanges: Sendable, Hashable {
    public struct Change: Sendable, Hashable {
        public var name: String
        public var ndBefore: Double
        public var ndAfter: Double
        public var vdBefore: Double
        public var vdAfter: Double
        public var fields: [String]
    }

    public var catalog: String
    public var added: [String]
    public var removed: [String]
    public var changed: [Change]
    public var unchanged: Int

    public init(catalog: String, added: [String] = [], removed: [String] = [], changed: [Change] = [], unchanged: Int = 0) {
        self.catalog = catalog
        self.added = added
        self.removed = removed
        self.changed = changed
        self.unchanged = unchanged
    }

    public init?(json v: JSONValue) {
        guard let catalog = v["catalog"]?.string else { return nil }
        self.catalog = catalog
        added = (v["added"]?.array ?? []).compactMap(\.string)
        removed = (v["removed"]?.array ?? []).compactMap(\.string)
        changed = (v["changed"]?.array ?? []).compactMap { c in
            guard let name = c["name"]?.string else { return nil }
            return Change(
                name: name, ndBefore: c["nd_before"]?.doubleValue ?? .nan, ndAfter: c["nd_after"]?.doubleValue ?? .nan,
                vdBefore: c["vd_before"]?.doubleValue ?? .nan, vdAfter: c["vd_after"]?.doubleValue ?? .nan,
                fields: (c["fields"]?.array ?? []).compactMap(\.string))
        }
        unchanged = v["unchanged"]?.int ?? 0
    }

    public var isEmpty: Bool { added.isEmpty && removed.isEmpty && changed.isEmpty }

    /// One line, e.g. "2 added, 1 removed, 3 changed".
    public var summary: String {
        if isEmpty { return "No changes" }
        var parts: [String] = []
        if !added.isEmpty { parts.append("\(added.count) added") }
        if !removed.isEmpty { parts.append("\(removed.count) removed") }
        if !changed.isEmpty { parts.append("\(changed.count) changed") }
        return parts.joined(separator: ", ")
    }
}

/// The outcome of `EngineClient.setLibraryCatalogs`.
public struct LibrarySync: Sendable, Hashable {
    public struct Failure: Sendable, Hashable {
        public var path: String
        public var message: String
        public init(path: String, message: String) {
            self.path = path
            self.message = message
        }
    }

    public var catalogs: [CatalogInfo]
    public var errors: [Failure]
    /// Whether the session's glass library changed.
    public var changed: Bool

    public init(catalogs: [CatalogInfo] = [], errors: [Failure] = [], changed: Bool = false) {
        self.catalogs = catalogs
        self.errors = errors
        self.changed = changed
    }
}

/// A catalog written as AGF text (`mk_catalog_to_agf`).
public struct AGFExport: Sendable, Hashable {
    public var text: String
    /// Glasses the writer could not express or had to rename.
    public var warnings: [String]
    public init(text: String, warnings: [String]) {
        self.text = text
        self.warnings = warnings
    }
}
