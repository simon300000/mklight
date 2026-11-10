import Foundation

/// Coefficient families of the extended surface shapes, addressed by
/// `ParamPath.shapeCoefficient` with zero-based indices (SURFACES §7). Which
/// family belongs to which shape, and its limit, mirror the engine's
/// `coefficient_slot!`.
public enum CoefficientSet: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
    /// Odd asphere β1, β2, … (index 0 is β1).
    case odd
    /// Forbes Q a₀, a₁, … of `qCon` and `qBfs`.
    case q
    /// Diffractive phase A₁, A₂, … (index 0 is A₁).
    case phase
    /// Biconic x polynomial from x⁴.
    case xAsphere = "x_asphere"
    /// Toroid or biconic y polynomial from y⁴.
    case yAsphere = "y_asphere"
    /// XY polynomial terms in `XYTerm` order.
    case xy
    /// Zernike sag terms; index 0 is Z1.
    case zernike

    public var id: String { rawValue }

    /// Even asphere A4 … A42, addressed by `ParamPath.asphere`.
    public static let maxEvenTerms = 20
    public static let maxOddTerms = 20
    public static let maxQTerms = 20
    public static let maxPhaseTerms = 10
    public static let maxAxisTerms = 10
    /// Degrees 1 through 10.
    public static let maxXYTerms = 65

    /// Number of coefficients `shape` allows in this set, or `nil` when the
    /// set does not belong to the shape.
    public func maxCount(for shape: SurfaceShape) -> Int? {
        if case .sagOverlay(let base, _, _, _, _, _) = shape { return maxCount(for: base) }
        return switch (shape, self) {
        case (.oddAsphere, .odd): Self.maxOddTerms
        case (.qCon, .q), (.qBfs, .q): Self.maxQTerms
        case (.diffractive, .phase): Self.maxPhaseTerms
        case (.biconic, .xAsphere), (.biconic, .yAsphere), (.toroid, .yAsphere): Self.maxAxisTerms
        case (.xyPolynomial, .xy): Self.maxXYTerms
        case (.zernikeSag(_, _, _, let ordering, _), .zernike): ordering.maxTerms
        default: nil
        }
    }

    /// Short label of coefficient `index` of this set on `shape`, e.g. `β3`,
    /// `Q2`, `A1`, `A4x`, `A6y`, `C(x²y)`, `Z7`.
    public func termLabel(_ index: Int) -> String {
        switch self {
        case .odd: "β\(index + 1)"
        case .q: "Q\(index)"
        case .phase: "A\(index + 1)"
        case .xAsphere: "A\(2 * index + 4)x"
        case .yAsphere: "A\(2 * index + 4)y"
        case .xy: XYTerm(index: index).map { "C(\($0.monomial))" } ?? "C\(index)"
        case .zernike: "Z\(index + 1)"
        }
    }
}

/// Scalar parameters of the extended shapes, addressed by
/// `ParamPath.shapeParam` (mirror of the engine's `parameter_slot!`).
public enum ShapeParameter: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
    case curvatureX = "curvature_x"
    case conicX = "conic_x"
    case curvatureY = "curvature_y"
    case conicY = "conic_y"
    case normRadius = "norm_radius"
    /// A sag overlay's astigmatism axis (degrees from +x toward +y).
    case astigmatismAngleDeg = "astigmatism_angle_deg"
    case designOrder = "design_order"
    case designWavelengthUm = "design_wavelength_um"
    case traceOrder = "trace_order"

    public var id: String { rawValue }

    /// Diffraction orders: integral `i32` values, never continuous variables.
    public var isOrder: Bool { self == .designOrder || self == .traceOrder }

    public var title: String {
        switch self {
        case .curvatureX: "R_x"
        case .conicX: "k_x"
        case .curvatureY: "R_y"
        case .conicY: "k_y"
        case .normRadius: "Norm radius"
        case .astigmatismAngleDeg: "Astigmatism axis"
        case .designOrder: "Design order"
        case .designWavelengthUm: "Design λ"
        case .traceOrder: "Trace order"
        }
    }
}

/// A term `X^m Y^n` of an XY polynomial (SURFACES §5). Total degree
/// `d = m + n` rises from 1 to 10 and, within a degree, `m` falls from `d`
/// to 0: zero-based index `j = d(d+1)/2 − 1 + n`.
public struct XYTerm: Sendable, Hashable {
    /// Power of X.
    public var m: Int
    /// Power of Y.
    public var n: Int

    public init(m: Int, n: Int) {
        self.m = m
        self.n = n
    }

    /// The term at zero-based coefficient `index`, or `nil` outside 0..<65.
    public init?(index: Int) {
        guard index >= 0, index < CoefficientSet.maxXYTerms else { return nil }
        var d = 1
        while (d + 1) * (d + 2) / 2 - 1 <= index { d += 1 }
        n = index - (d * (d + 1) / 2 - 1)
        m = d - n
    }

    public var degree: Int { m + n }
    /// Zero-based coefficient index.
    public var index: Int { degree * (degree + 1) / 2 - 1 + n }

    /// E.g. `x`, `y²`, `x²y`.
    public var monomial: String {
        func power(_ v: String, _ p: Int) -> String {
            p == 0 ? "" : p == 1 ? v : v + Self.superscript(p)
        }
        return power("x", m) + power("y", n)
    }

    static func superscript(_ n: Int) -> String {
        let digits = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]
        return String(n).compactMap { $0.wholeNumberValue.map { digits[$0] } }.joined()
    }
}

/// A Zernike term `(n, m)` (SURFACES §6): `R_n^|m|(ρ)` times `cos(mθ)` for
/// m > 0, `sin(|m|θ)` for m < 0. Term numbers `j` start at 1 (piston).
public struct ZernikeTerm: Sendable, Hashable {
    /// Radial order.
    public var n: Int
    /// Signed azimuthal frequency (negative for sine terms).
    public var m: Int

    public init(n: Int, m: Int) {
        self.n = n
        self.m = m
    }

    /// Term `j` (1-based) in `ordering`, or `nil` outside 1…`maxTerms`.
    public init?(j: Int, ordering: ZernikeOrdering) {
        guard j >= 1, j <= ordering.maxTerms else { return nil }
        switch ordering {
        case .fringe:
            if j == 37 {
                (n, m) = (12, 0)
                return
            }
            let k = Int(Double(j - 1).squareRoot())
            let o = j - 1 - k * k
            let am = k - o / 2
            n = 2 * k - am
            m = o % 2 == 0 ? am : -am
        case .standard:
            var nn = 0
            while (nn + 1) * (nn + 2) / 2 < j { nn += 1 }
            let q = j - nn * (nn + 1) / 2 - 1
            let am = nn % 2 == 0 ? 2 * ((q + 1) / 2) : 2 * (q / 2) + 1
            n = nn
            m = am == 0 ? 0 : (j % 2 == 0 ? am : -am)
        }
    }
}

// MARK: - Coefficient and parameter access

extension SurfaceShape {
    /// The coefficient sets this shape has (empty for legacy shapes).
    public var coefficientSets: [CoefficientSet] {
        CoefficientSet.allCases.filter { $0.maxCount(for: self) != nil }
    }

    /// The scalar shape parameters this shape has.
    public var shapeParameters: [ShapeParameter] {
        ShapeParameter.allCases.filter { shapeParameter($0) != nil }
    }

    /// The stored coefficients of `set` (possibly shorter than its limit;
    /// missing trailing terms are zero), or `nil` when the set does not
    /// belong to the shape.
    public func coefficients(in set: CoefficientSet) -> [Double]? {
        if case .sagOverlay(let base, _, _, _, _, _) = self { return base.coefficients(in: set) }
        return switch (self, set) {
        case (.oddAsphere(_, _, let a), .odd), (.qCon(_, _, _, let a), .q), (.qBfs(_, _, let a), .q),
            (.biconic(_, _, _, _, let a, _), .xAsphere), (.biconic(_, _, _, _, _, let a), .yAsphere),
            (.toroid(_, _, _, let a), .yAsphere), (.xyPolynomial(_, _, _, let a), .xy),
            (.zernikeSag(_, _, _, _, let a), .zernike):
            a
        case (.diffractive(_, let p), .phase): p.coefficients
        default: nil
        }
    }

    /// Replaces the coefficients of `set`. Returns false when the set does
    /// not belong to the shape (the count is not checked here; structural
    /// validation reports it).
    @discardableResult
    public mutating func setCoefficients(_ a: [Double], in set: CoefficientSet) -> Bool {
        if let done = editOverlayBase({ $0.setCoefficients(a, in: set) }) { return done }
        switch (self, set) {
        case (.oddAsphere(let c, let k, _), .odd): self = .oddAsphere(curvature: c, conic: k, coefficients: a)
        case (.qCon(let c, let k, let r, _), .q): self = .qCon(curvature: c, conic: k, normRadius: r, coefficients: a)
        case (.qBfs(let c, let r, _), .q): self = .qBfs(curvature: c, normRadius: r, coefficients: a)
        case (.diffractive(let base, var p), .phase):
            p.coefficients = a
            self = .diffractive(base: base, phase: p)
        case (.biconic(let cx, let cy, let kx, let ky, _, let ya), .xAsphere):
            self = .biconic(curvatureX: cx, curvatureY: cy, conicX: kx, conicY: ky, xAsphere: a, yAsphere: ya)
        case (.biconic(let cx, let cy, let kx, let ky, let xa, _), .yAsphere):
            self = .biconic(curvatureX: cx, curvatureY: cy, conicX: kx, conicY: ky, xAsphere: xa, yAsphere: a)
        case (.toroid(let cx, let cy, let ky, _), .yAsphere):
            self = .toroid(curvatureX: cx, curvatureY: cy, conicY: ky, yAsphere: a)
        case (.xyPolynomial(let c, let k, let r, _), .xy):
            self = .xyPolynomial(curvature: c, conic: k, normRadius: r, coefficients: a)
        case (.zernikeSag(let c, let k, let r, let o, _), .zernike):
            self = .zernikeSag(curvature: c, conic: k, normRadius: r, ordering: o, coefficients: a)
        default: return false
        }
        return true
    }

    /// The value of a scalar shape parameter (orders as `Double`), or `nil`
    /// when the parameter does not belong to the shape.
    public func shapeParameter(_ p: ShapeParameter) -> Double? {
        // The angle belongs to the overlay of the sag shape (as in the engine).
        if p == .astigmatismAngleDeg {
            guard case .sagOverlay(_, _, _, _, let angle, _) = sagShape else { return nil }
            return angle
        }
        if case .sagOverlay(let base, _, _, _, _, _) = self { return base.shapeParameter(p) }
        return switch (self, p) {
        case (.toroid(let cx, _, _, _), .curvatureX), (.biconic(let cx, _, _, _, _, _), .curvatureX): cx
        case (.toroid(_, let cy, _, _), .curvatureY), (.biconic(_, let cy, _, _, _, _), .curvatureY): cy
        case (.biconic(_, _, let kx, _, _, _), .conicX): kx
        case (.toroid(_, _, let ky, _), .conicY), (.biconic(_, _, _, let ky, _, _), .conicY): ky
        case (.qCon(_, _, let r, _), .normRadius), (.qBfs(_, let r, _), .normRadius),
            (.xyPolynomial(_, _, let r, _), .normRadius), (.zernikeSag(_, _, let r, _, _), .normRadius):
            r
        case (.diffractive(_, let ph), .normRadius): ph.normRadius
        case (.diffractive(_, let ph), .designWavelengthUm): ph.designWavelengthUm
        case (.diffractive(_, let ph), .designOrder): Double(ph.designOrder)
        case (.diffractive(_, let ph), .traceOrder): Double(ph.traceOrder)
        default: nil
        }
    }

    /// Why `v` cannot be stored in parameter `p` (an order must be an
    /// integer `i32`, as in the engine's setter), or `nil`.
    static func orderRejection(_ p: ShapeParameter, _ v: Double) -> String? {
        guard p.isOrder else { return nil }
        // The engine's check: `v.fract() != 0 || |v| > i32::MAX`.
        return v.rounded(.towardZero) != v || abs(v) > Double(Int32.max)
            ? "[invalid_diffraction_order] expected an i32" : nil
    }

    /// Sets a scalar shape parameter. Returns false when it does not belong
    /// to the shape, or for an order that is not an integral `i32`. Range
    /// rules (positive radius, nonzero design order) are structural
    /// validation, as in the engine.
    @discardableResult
    public mutating func setShapeParameter(_ p: ShapeParameter, _ v: Double) -> Bool {
        guard shapeParameter(p) != nil, Self.orderRejection(p, v) == nil else { return false }
        if p == .astigmatismAngleDeg {
            if case .diffractive(var base, let phase) = self {
                base.setShapeParameter(p, v)
                self = .diffractive(base: base, phase: phase)
            } else if case .sagOverlay(let base, let r, let w, let a, _, let s) = self {
                self = .sagOverlay(
                    base: base, normRadius: r, testWavelengthUm: w, astigmatism: a, astigmatismAngleDeg: v, spherical: s)
            }
            return true
        }
        if let done = editOverlayBase({ $0.setShapeParameter(p, v) }) { return done }
        switch self {
        case .toroid(var cx, var cy, var ky, let ya):
            switch p {
            case .curvatureX: cx = v
            case .curvatureY: cy = v
            default: ky = v
            }
            self = .toroid(curvatureX: cx, curvatureY: cy, conicY: ky, yAsphere: ya)
        case .biconic(var cx, var cy, var kx, var ky, let xa, let ya):
            switch p {
            case .curvatureX: cx = v
            case .curvatureY: cy = v
            case .conicX: kx = v
            default: ky = v
            }
            self = .biconic(curvatureX: cx, curvatureY: cy, conicX: kx, conicY: ky, xAsphere: xa, yAsphere: ya)
        case .qCon(let c, let k, _, let a): self = .qCon(curvature: c, conic: k, normRadius: v, coefficients: a)
        case .qBfs(let c, _, let a): self = .qBfs(curvature: c, normRadius: v, coefficients: a)
        case .xyPolynomial(let c, let k, _, let a):
            self = .xyPolynomial(curvature: c, conic: k, normRadius: v, coefficients: a)
        case .zernikeSag(let c, let k, _, let o, let a):
            self = .zernikeSag(curvature: c, conic: k, normRadius: v, ordering: o, coefficients: a)
        case .diffractive(let base, var ph):
            switch p {
            case .normRadius: ph.normRadius = v
            case .designWavelengthUm: ph.designWavelengthUm = v
            case .designOrder: ph.designOrder = Int(v)
            default: ph.traceOrder = Int(v)
            }
            self = .diffractive(base: base, phase: ph)
        default: return false
        }
        return true
    }
}
