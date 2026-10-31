import Foundation

/// Radius ↔ curvature conversion for display. The model stores curvature
/// `c = 1/R`; a plane is `c = 0`, shown as an infinite radius.
public enum RadiusDisplay {
    /// Radius for a curvature (`±∞` for `c = 0`).
    public static func radius(fromCurvature c: Double) -> Double {
        c == 0 ? .infinity : 1 / c
    }

    /// Curvature for a radius; `±∞` and `0` both mean a plane.
    public static func curvature(fromRadius r: Double) -> Double {
        (r.isInfinite || r == 0) ? 0 : 1 / r
    }
}

extension SurfaceShape {
    /// Radius of curvature (mm), `∞` for planes and shapes without curvature.
    public var radius: Double { RadiusDisplay.radius(fromCurvature: curvature) }
}

extension Surface {
    /// Radius of curvature for display; setting it updates the curvature.
    public var radius: Double {
        get { shape.radius }
        set { shape.setCurvature(RadiusDisplay.curvature(fromRadius: newValue)) }
    }
}

/// Number formatting for tables and inspectors.
///
/// **One rule app-wide:** numbers are shown with a decimal point and no
/// digit grouping, whatever the system region, as in lens prescriptions,
/// glass catalogues and the engine's files (`1.5`, `12345.678`). The app's
/// views run with `NumberDisplay.locale` (en_US_POSIX), so Swift Charts
/// axes, `formatted()` and number fields follow the same rule; the
/// `String(format:)` paths are locale-free. Typed numbers also accept a
/// decimal comma (`LensValueParser.number`).
public enum NumberDisplay {
    /// The locale every view formats numbers with.
    public static let locale = Locale(identifier: "en_US_POSIX")

    /// Up to `digits` significant digits, trailing zeros dropped, without
    /// exponent for ordinary magnitudes: `1.5`, `1.7552`, `0.0001`.
    public static func significant(_ v: Double, digits: Int = 10) -> String {
        if v.isNaN { return "NaN" }
        if v.isInfinite { return v > 0 ? "∞" : "−∞" }
        return String(format: "%.\(max(digits, 1))g", v)
    }

    /// Fixed decimals for ordinary magnitudes, scientific outside them; `∞`
    /// for infinities.
    public static func format(_ v: Double, decimals: Int = 4) -> String {
        if v.isNaN { return "NaN" }
        if v.isInfinite { return v > 0 ? "Infinity" : "-Infinity" }
        let a = abs(v)
        if a != 0 && (a < 1e-3 || a >= 1e7) {
            return String(format: "%.\(max(decimals, 1))e", v)
        }
        return String(format: "%.\(decimals)f", v)
    }

    /// Shortest representation that parses back to the same `Double`.
    public static func exact(_ v: Double) -> String {
        if v.isInfinite { return v > 0 ? "inf" : "-inf" }
        return "\(v)"
    }
}
