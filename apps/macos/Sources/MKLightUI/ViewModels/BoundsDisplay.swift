import Foundation
import MKLightModel

/// How curvature variables show their value and bounds: as a radius in mm
/// (the lens designer's unit, the default) or as the curvature `c = 1/R`
/// the engine optimises.
public enum CurvatureBoundsUnit: String, CaseIterable, Identifiable, Sendable {
    case radius, curvature

    public var id: String { rawValue }
    public var title: String { self == .radius ? "Radius" : "Curvature" }
    public var unit: String { self == .radius ? "mm" : "1/mm" }
}

/// Radius ↔ curvature bounds. The engine bounds the curvature `c = 1/R`;
/// the app lets the designer type a radius window instead.
///
/// A radius window `min … max` is read on the projective line of radii,
/// rising from `min` to `max` and wrapping through ±∞ (a flat surface)
/// when `min > max`: its curvatures form the interval `[1/max, 1/min]`.
///
/// * `40 … 60` → `c ∈ [1/60, 1/40]`; `−60 … −40` → `c ∈ [−1/40, −1/60]`.
/// * `100 … −100` passes through ∞: `|R| ≥ 100`, `c ∈ [−0.01, 0.01]`.
/// * An empty side is open: an empty max is `+∞` (`R ≥ 50` is
///   `c ∈ [0, 0.02]`), an empty min `−∞`; both empty is unbounded.
/// * `0` is the limit `R → 0` (no bound on the curvature on that side):
///   `0 … 50` is `c ≥ 0.02`.
/// * `∞ … ∞` holds the surface flat (`c = 0`).
///
/// A window that contains `R = 0` itself (`−50 … 50`, or `60 … 40`, which
/// wraps through ∞ and then 0) is not an interval of curvatures and is
/// refused.
public enum RadiusBounds {
    public enum Problem: Error, Equatable, CustomStringConvertible {
        /// The window contains R = 0 (an infinite curvature).
        case containsZero

        public var description: String {
            switch self {
            case .containsZero:
                "This radius window passes through R = 0, which is not a range of curvatures. For |R| ≥ r, enter min r and max −r (through ∞)."
            }
        }
    }

    /// Curvature bounds for a typed radius window (`nil` = an empty field).
    public static func curvature(minRadius: Double?, maxRadius: Double?) -> Result<(lower: Double?, upper: Double?), Problem> {
        if minRadius == nil && maxRadius == nil { return .success((nil, nil)) }
        let lo = minRadius ?? -.infinity, hi = maxRadius ?? .infinity
        // c_upper = 1/min (min → 0⁺: unbounded above); c_lower = 1/max (max → 0⁻: unbounded below).
        let upper: Double? = lo == 0 ? nil : clean(1 / lo)
        let lower: Double? = hi == 0 ? nil : clean(1 / hi)
        if let l = lower, let u = upper, l > u { return .failure(.containsZero) }
        return .success((lower, upper))
    }

    /// The radius window (`nil` = an empty field, open on that side) that
    /// shows curvature bounds; the inverse of `curvature(minRadius:maxRadius:)`.
    public static func radius(lower: Double?, upper: Double?) -> (min: Double?, max: Double?) {
        if lower == nil && upper == nil { return (nil, nil) }
        // A pinned flat surface: ∞ … ∞.
        if lower == 0 && upper == 0 { return (.infinity, .infinity) }
        let minR: Double = upper.map { $0 == 0 ? -.infinity : 1 / $0 } ?? 0
        let maxR: Double = lower.map { $0 == 0 ? .infinity : 1 / $0 } ?? 0
        return (minR == -.infinity ? nil : minR, maxR == .infinity ? nil : maxR)
    }

    /// Whether the window wraps through a flat surface (min > max, both set).
    public static func passesThroughFlat(minRadius: Double?, maxRadius: Double?) -> Bool {
        guard let a = minRadius, let b = maxRadius, a.isFinite, b.isFinite else { return false }
        return a > b && a > 0 && b < 0
    }

    /// `-0` → `0`, so the stored bounds read `0`.
    static func clean(_ v: Double) -> Double { v == 0 ? 0 : v }

    /// The short help shown with radius bounds.
    public static let help =
        "Bounds as a radius window in mm; the optimizer bounds the curvature c = 1/R. Empty is open on that side. Min above max wraps through flat (R = ∞): min 100, max −100 is |R| ≥ 100. 0 leaves the curvature unbounded on that side."
}

/// Formatting of variable values and bounds.
public enum BoundsFormat {
    /// Compact: 4 significant digits, `∞` for infinities.
    public static func compact(_ v: Double) -> String {
        if v.isInfinite { return v > 0 ? "∞" : "−∞" }
        if v.isNaN { return "—" }
        return String(format: "%.4g", v)
    }

    /// A one-line summary of curvature bounds in the other unit, e.g.
    /// `c 0.0167 … 0.025` or `R 40 … 60`.
    public static func counterpart(lower: Double?, upper: Double?, shownAs unit: CurvatureBoundsUnit) -> String? {
        guard lower != nil || upper != nil else { return nil }
        switch unit {
        case .radius:
            return "c " + (lower.map(compact) ?? "−∞") + " … " + (upper.map(compact) ?? "∞") + " 1/mm"
        case .curvature:
            let r = RadiusBounds.radius(lower: lower, upper: upper)
            let wraps = RadiusBounds.passesThroughFlat(minRadius: r.min, maxRadius: r.max)
            return "R " + (r.min.map(compact) ?? "−∞") + (wraps ? " … ∞ … " : " … ") + (r.max.map(compact) ?? "∞") + " mm"
        }
    }
}
