import Foundation
import MKLightEngine
import MKLightModel

/// Readers of engine result JSON (docs/RESULTS.md) into small values the
/// views plot. Each returns `nil` when the JSON does not have the expected
/// shape; the pane then shows the result generically.

/// `first_order`: the keyed report table.
public struct FirstOrderData: Sendable {
    public struct Entry: Identifiable, Sendable {
        public var key: String
        public var label: String
        public var value: Double
        public var unit: String
        public var id: String { key }
        public var formatted: String { Fmt.value(value, unit: unit) }
    }
    public var entries: [Entry]
    public var wavelengthUm: Double?

    public init?(_ json: JSONValue?) {
        guard let list = json?["entries"]?.array else { return nil }
        entries = list.compactMap { e in
            guard let key = e["key"]?.string, let v = e["value"]?.nf else { return nil }
            return Entry(key: key, label: e["label"]?.string ?? key, value: v, unit: e["unit"]?.string ?? "")
        }
        wavelengthUm = json?["wavelength_um"]?.nf
        if entries.isEmpty { return nil }
    }

    public func entry(_ keys: [String]) -> Entry? {
        for k in keys { if let e = entries.first(where: { $0.key == k }) { return e } }
        return nil
    }
}

/// A field as specified, for labels: `0°`, `14°`, `(3, 4) mm`.
public enum FieldLabel {
    public static func text(_ v: [Double]?, unit: String) -> String {
        guard let v, v.count == 2 else { return "—" }
        let u = unit == "°" ? "°" : " " + unit
        if v[0] == 0 { return Fmt.short(v[1]) + u }
        return "(\(Fmt.short(v[0])), \(Fmt.short(v[1])))" + u
    }

    public static func unit(_ system: OpticalSystem?) -> String { system?.fields.kind.unit ?? "" }
}

/// Number formats of analysis panes.
public enum Fmt {
    /// Up to 6 significant digits, scientific for very small or large.
    public static func short(_ v: Double) -> String {
        if v.isNaN { return "—" }
        if v.isInfinite { return v > 0 ? "∞" : "−∞" }
        if v == 0 { return "0" }
        let a = abs(v)
        if a < 1e-3 || a >= 1e6 { return String(format: "%.3e", v) }
        return String(format: "%.6g", v)
    }

    public static func value(_ v: Double, unit: String) -> String {
        let s = short(v)
        guard !unit.isEmpty, v.isFinite else { return s }
        return unit == "°" ? s + "°" : s + " " + unit
    }

    /// A length in mm shown in µm below 0.1 mm (never "−0.00").
    public static func length(_ mm: Double) -> String {
        guard mm.isFinite else { return short(mm) }
        let s = abs(mm) < 0.1 ? String(format: "%.2f µm", mm * 1000) : String(format: "%.4f mm", mm)
        if s.hasPrefix("-"), s.dropFirst().allSatisfy({ "0. µm".contains($0) }) { return String(s.dropFirst()) }
        return s
    }

    /// A signed value with fixed decimals (`+0.0123`, `−1.2000`); a value
    /// that rounds to zero is `0.0000` with a figure space for the sign, so
    /// columns stay aligned and no "−0.0000" appears.
    public static func signed(_ v: Double, decimals: Int = 4) -> String {
        guard v.isFinite else { return "—" }
        let s = String(format: "%+.\(decimals)f", v)
        if s.dropFirst().allSatisfy({ $0 == "0" || $0 == "." }) { return "\u{2007}" + s.dropFirst() }
        return s
    }

    public static func percent(_ f: Double, digits: Int = 1) -> String {
        f.isFinite ? String(format: "%.\(digits)f %%", f * 100) : "—"
    }

    /// A run time: whole milliseconds below a second, then seconds.
    public static func duration(ms: Double) -> String {
        ms < 1000 ? String(format: "%.0f ms", ms) : String(format: "%.1f s", ms / 1000)
    }
}

extension AnalysisOutput {
    /// A buffer named by a JSON string value (the wave analyses refer to
    /// their buffers by name).
    func buffer(ref: JSONValue?) -> ResultBuffer? {
        ref?.string.flatMap { buffer(named: $0) }
    }

    /// The buffer of the result's key `key`: the one its string value
    /// names, else the one named `key` itself.
    func buffer(key: String) -> ResultBuffer? {
        buffer(ref: json[key]) ?? buffer(named: key)
    }
}

/// The engine's first-order data of both sections (`paraxial_xy` of
/// `first_order` and `system_summary`, docs/RESULTS.md): EFL and BFL in x
/// and y, the signed squeeze `EFL_y / EFL_x` and DOE phase powers at every
/// system wavelength, the design's symmetry class and the engine's
/// warnings. `nil` when the result has no `paraxial_xy`.
public struct ParaxialXYData: Sendable, Equatable {
    /// The geometry's symmetry (placements and apertures included).
    public enum Symmetry: String, Sendable {
        case rotational, meridional, none

        public var title: String {
            switch self {
            case .rotational: "Rotational"
            case .meridional: "Meridional plane"
            case .none: "None"
            }
        }
    }

    public struct DOEPower: Sendable, Equatable {
        public var surface: Int
        /// 1/mm, the selected order's phase power (base power excluded).
        public var power: Double
    }

    public struct Row: Sendable, Equatable, Identifiable {
        public var wavelength: Int
        public var wavelengthUm: Double
        public var eflX: Double
        public var eflY: Double
        public var bflX: Double
        public var bflY: Double
        public var squeeze: Double
        public var doePower: [DOEPower]
        public var id: Int { wavelength }
    }

    public var symmetry: Symmetry?
    public var primaryWavelength: Int?
    public var squeezeDefinition: String?
    public var warnings: [String]
    /// One row per system wavelength, in the engine's order.
    public var rows: [Row]

    public init?(_ json: JSONValue?) {
        guard let xy = json?["paraxial_xy"], xy.object != nil else { return nil }
        symmetry = (xy["symmetry"]?.string ?? json?["symmetry"]?.string).flatMap(Symmetry.init(rawValue:))
        primaryWavelength = xy["primary_wavelength"]?.int
        squeezeDefinition = xy["squeeze_definition"]?.string
        warnings = xy["warnings"]?.array?.compactMap(\.string) ?? []
        rows = (xy["wavelengths"]?.array ?? []).enumerated().compactMap { k, w in
            guard let efly = w["efl_y"]?.nf else { return nil }
            let eflx = w["efl_x"]?.nf ?? .nan
            return Row(
                wavelength: w["wavelength"]?.int ?? k,
                wavelengthUm: w["wavelength_um"]?.nf ?? .nan,
                eflX: eflx, eflY: efly,
                bflX: w["bfl_x"]?.nf ?? .nan, bflY: w["bfl_y"]?.nf ?? .nan,
                squeeze: w["squeeze"]?.nf ?? (eflx != 0 ? efly / eflx : .nan),
                doePower: (w["doe_power"]?.array ?? []).compactMap { d in
                    guard let s = d["surface"]?.int, let p = d["power"]?.nf else { return nil }
                    return DOEPower(surface: s, power: p)
                })
        }
    }

    /// The primary wavelength's row (else the first).
    public var primary: Row? { rows.first { $0.wavelength == primaryWavelength } ?? rows.first }

    /// The other wavelengths' rows.
    public var others: [Row] {
        guard let p = primary else { return [] }
        return rows.filter { $0.wavelength != p.wavelength }
    }

    /// Whether the design lacks rotational symmetry, so that its x and y
    /// first order differ. `nil` when the engine does not say.
    public var isAnamorphic: Bool? { symmetry.map { $0 != .rotational } }

    /// Whether any DOE is listed.
    public var hasDOEs: Bool { rows.contains { !$0.doePower.isEmpty } }

    /// The engine's symmetry class when it reports one; otherwise the
    /// surface kinds of `system` (toroid, biconic, freeform) decide.
    public static func isAnamorphic(_ json: JSONValue?, system: OpticalSystem?) -> Bool {
        if let a = ParaxialXYData(json)?.isAnamorphic { return a }
        if let s = json?["symmetry"]?.string.flatMap(Symmetry.init(rawValue:)) { return s != .rotational }
        return system?.hasNonRotationalSurfaces ?? false
    }

    /// `1.5000×`, signed.
    public static func squeezeText(_ v: Double, digits: Int = 4) -> String {
        v.isFinite ? String(format: "%.\(digits)f×", v) : Fmt.short(v)
    }
}

extension OpticalSystem {
    /// Whether any surface lacks rotational symmetry (toroid, biconic, XY
    /// or Zernike freeform): first-order data then differs between x and y.
    /// Only a fallback for engines that do not report `symmetry`
    /// (`ParaxialXYData.isAnamorphic(_:system:)`).
    public var hasNonRotationalSurfaces: Bool {
        surfaces.contains { [.toroid, .biconic, .xyPolynomial, .zernikeSag].contains($0.shape.kind) }
    }
}

/// Units of a wave result's image-space quantities (the result's `units`,
/// docs/WAVE.md §5): lengths on the image surface, or, for an afocal image
/// space, angles about the chief ray. Fields named `*_mm` then hold mrad,
/// frequencies are cycles/mrad and focus shifts are vergence in mD.
public struct ImageUnits: Sendable, Equatable {
    public var afocal: Bool

    public init(afocal: Bool = false) { self.afocal = afocal }
    /// From a result JSON (`units.afocal`; absent: focal).
    public init(_ result: JSONValue) { afocal = result["units"]?["afocal"]?.bool ?? false }

    public var frequency: String { afocal ? "cycles/mrad" : "cycles/mm" }
    /// The unit of a `*_mm` value × 1000.
    public var micro: String { afocal ? "µrad" : "µm" }
    /// Per unit length (or angle), as a line-spread density.
    public var density: String { afocal ? "mrad⁻¹" : "mm⁻¹" }
    public var focusAxis: String { afocal ? "Vergence (mD)" : "Focus shift (mm)" }

    /// A scale-bar label for a `*_mm` value.
    public func barLabel(_ v: Double) -> String {
        guard afocal else {
            return v >= 1 ? String(format: "%g mm", v) : v >= 0.001 ? String(format: "%g µm", v * 1000) : String(format: "%g nm", v * 1e6)
        }
        return v >= 1 ? String(format: "%g mrad", v) : v >= 0.001 ? String(format: "%g µrad", v * 1000) : String(format: "%g nrad", v * 1e6)
    }

    /// A `*_mm` value as `Fmt.length`, or as an angle.
    public func length(_ v: Double) -> String {
        guard afocal else { return Fmt.length(v) }
        guard v.isFinite else { return Fmt.short(v) }
        return abs(v) < 0.1 ? String(format: "%.2f µrad", v * 1000) : String(format: "%.4f mrad", v)
    }
}
