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

    /// A length in mm shown in µm below 0.1 mm.
    public static func length(_ mm: Double) -> String {
        guard mm.isFinite else { return short(mm) }
        return abs(mm) < 0.1 ? String(format: "%.2f µm", mm * 1000) : String(format: "%.4f mm", mm)
    }

    public static func percent(_ f: Double, digits: Int = 1) -> String {
        f.isFinite ? String(format: "%.\(digits)f %%", f * 100) : "—"
    }
}

extension AnalysisOutput {
    /// A buffer named by a JSON string value (the wave analyses refer to
    /// their buffers by name).
    func buffer(ref: JSONValue?) -> ResultBuffer? {
        ref?.string.flatMap { buffer(named: $0) }
    }
}

/// Anamorphic first-order data: EFL in x and y and the signed squeeze
/// `EFL_y / EFL_x` (SURFACES §8, the engine's `ParaxialXY`). Read from the
/// first-order result when the engine reports it (`efl_x`, `efl_y`,
/// `squeeze` entries, or a `paraxial_xy` object); `nil` otherwise.
public struct AnamorphicFirstOrder: Sendable, Equatable {
    public var eflX: Double?
    public var eflY: Double?
    public var squeeze: Double?

    public init?(_ json: JSONValue?) {
        guard let json else { return nil }
        let entries = FirstOrderData(json)
        let xy = json["paraxial_xy"]
        eflX = entries?.entry(["efl_x"])?.value ?? xy?["efl_x"]?.nf
        eflY = entries?.entry(["efl_y"])?.value ?? xy?["efl_y"]?.nf
        squeeze = entries?.entry(["squeeze"])?.value ?? xy?["squeeze"]?.nf
        if squeeze == nil, let x = eflX, let y = eflY, x != 0 { squeeze = y / x }
        if eflX == nil && eflY == nil && squeeze == nil { return nil }
    }
}

extension OpticalSystem {
    /// Whether any surface lacks rotational symmetry (toroid, biconic, XY
    /// or Zernike freeform): first-order data then differs between x and y.
    public var hasNonRotationalSurfaces: Bool {
        surfaces.contains { [.toroid, .biconic, .xyPolynomial, .zernikeSag].contains($0.shape.kind) }
    }
}
