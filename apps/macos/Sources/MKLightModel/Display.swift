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
public enum NumberDisplay {
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
