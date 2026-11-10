import Foundation
import MKLightModel

/// Sag `z(x, y)` of every surface shape, for the inspector's profile
/// previews and sag maps only (docs/SURFACES.md §1–6). Display geometry:
/// no ray tracing, no first order. The engine's `surface_math` is the
/// definition; these follow the same formulas and agree with it where
/// tested (`SurfaceTypeTests`).
public enum ShapeSag {
    /// Rationalised conic sag `B(c, k, s)` with `s = r²`, or `nil` outside
    /// its real domain.
    public static func conic(_ c: Double, _ k: Double, _ s: Double) -> Double? {
        let q = 1 - (1 + k) * c * c * s
        guard q >= 0 else { return nil }
        return c * s / (1 + q.squareRoot())
    }

    /// The sag at local `(x, y)` in mm, or `nil` outside the shape's domain
    /// (including normalised polynomials beyond their radius).
    public static func sag(_ shape: SurfaceShape, x: Double, y: Double) -> Double? {
        let s = x * x + y * y
        switch shape {
        case .plane, .paraxial, .coordinateBreak:
            return 0
        case .conic(let c, let k):
            return conic(c, k, s)
        case .evenAsphere(let c, let k, let a):
            guard let b = conic(c, k, s) else { return nil }
            return b + horner(a, s) * s * s
        case .oddAsphere(let c, let k, let a):
            guard let b = conic(c, k, s) else { return nil }
            let r = s.squareRoot()
            return b + a.enumerated().reduce(0) { $0 + $1.element * pow(r, Double($1.offset + 1)) }
        case .qCon(let c, let k, let rn, let a):
            guard let b = conic(c, k, s), let t = normalised(s, rn) else { return nil }
            return b + t * t * qCon(a, t)
        case .qBfs(let c, let rn, let a):
            guard let b = conic(c, 0, s), let t = normalised(s, rn) else { return nil }
            if a.allSatisfy({ $0 == 0 }) { return b }
            let w = 1 - c * c * s
            guard w > 0 else { return nil }
            return b + t * (1 - t) * qBfs(a, t) / w.squareRoot()
        case .diffractive(let base, _):
            return sag(base, x: x, y: y)
        case .sagOverlay(let base, let r, let w, let a, let angle, let sph):
            // SURFACES §10: one fringe is λ/2 PV sag.
            guard let b = sag(base, x: x, y: y) else { return nil }
            let u = x / r, v = y / r, t = u * u + v * v
            let theta = 2 * angle * .pi / 180
            let astig = (u - v) * (u + v) * cos(theta) + 2 * u * v * sin(theta)
            return b + w * 1e-3 * (a * astig / 4 + 2 * sph * (t * t - t))
        case .toroid(let cx, let cy, let ky, let b):
            let yy = y * y
            guard let zb = conic(cy, ky, yy) else { return nil }
            let zy = zb + horner(b, yy) * yy * yy
            if cx == 0 { return zy }
            let a = 1 - cx * zy
            let d = a * a - cx * cx * x * x
            guard a > 0, d >= 0 else { return nil }
            return zy + cx * x * x / (a + d.squareRoot())
        case .biconic(let cx, let cy, let kx, let ky, let ax, let ay):
            let xx = x * x, yy = y * y
            let w = 1 - (1 + kx) * cx * cx * xx - (1 + ky) * cy * cy * yy
            guard w >= 0 else { return nil }
            return (cx * xx + cy * yy) / (1 + w.squareRoot()) + horner(ax, xx) * xx * xx + horner(ay, yy) * yy * yy
        case .xyPolynomial(let c, let k, let rn, let a):
            guard let b = conic(c, k, s), normalised(s, rn) != nil else { return nil }
            return b + xyDeparture(a, x / rn, y / rn)
        case .zernikeSag(let c, let k, let rn, let ordering, let a):
            guard let b = conic(c, k, s), normalised(s, rn) != nil else { return nil }
            return b + zernikeDeparture(a, ordering: ordering, x / rn, y / rn)
        }
    }

    /// The sag minus its base conic: what the freeform terms add.
    public static func departure(_ shape: SurfaceShape, x: Double, y: Double) -> Double? {
        guard let z = sag(shape, x: x, y: y) else { return nil }
        let s = x * x + y * y
        switch shape {
        case .xyPolynomial(let c, let k, _, _), .zernikeSag(let c, let k, _, _, _), .qCon(let c, let k, _, _),
            .oddAsphere(let c, let k, _), .evenAsphere(let c, let k, _):
            return conic(c, k, s).map { z - $0 }
        case .qBfs(let c, _, _):
            return conic(c, 0, s).map { z - $0 }
        default:
            return z
        }
    }

    // MARK: Polynomials

    static func horner(_ a: [Double], _ s: Double) -> Double {
        a.reversed().reduce(0) { $0 * s + $1 }
    }

    /// `t = (r / r_norm)²`, or `nil` beyond the normalisation radius.
    static func normalised(_ s: Double, _ rn: Double) -> Double? {
        guard rn > 0 else { return nil }
        let t = s / (rn * rn)
        return t <= 1 + 1e-12 ? t : nil
    }

    /// `Σ a_m Q_m^con(t)` with `Q_m^con(t) = P_m^(0,4)(2t − 1)`.
    public static func qCon(_ a: [Double], _ t: Double) -> Double {
        let x = 2 * t - 1, alpha = 0.0, beta = 4.0
        var sum = 0.0, p0 = 1.0, p1 = (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2
        for (m, am) in a.enumerated() {
            if m == 0 {
                sum += am * p0
            } else if m == 1 {
                sum += am * p1
            } else {
                let n = Double(m), ab = alpha + beta
                let c1 = 2 * n * (n + ab) * (2 * n + ab - 2)
                let c2 = (2 * n + ab - 1) * ((2 * n + ab) * (2 * n + ab - 2) * x + alpha * alpha - beta * beta)
                let c3 = 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + ab)
                let p2 = (c2 * p1 - c3 * p0) / c1
                p0 = p1
                p1 = p2
                sum += am * p2
            }
        }
        return sum
    }

    /// `Σ a_m Q_m^bfs(t)` through Forbes' auxiliary basis (SURFACES §2).
    public static func qBfs(_ a: [Double], _ t: Double) -> Double {
        let n = a.count
        guard n > 0 else { return 0 }
        var f = [Double](repeating: 0, count: n), g = [Double](repeating: 0, count: n), h = [Double](repeating: 0, count: n)
        f[0] = 2
        if n > 1 { f[1] = 19.0.squareRoot() / 2 }
        g[0] = -0.5
        if n > 2 {
            for m in 2..<n {
                h[m - 2] = -Double(m * (m - 1)) / (2 * f[m - 2])
                g[m - 1] = -(1 + g[m - 2] * h[m - 2]) / f[m - 1]
                f[m] = (Double(m * (m + 1)) + 3 - g[m - 1] * g[m - 1] - h[m - 2] * h[m - 2]).squareRoot()
            }
        }
        var b = [Double](repeating: 0, count: n + 2)
        for m in stride(from: n - 1, through: 0, by: -1) {
            b[m] = (a[m] - (m + 1 < n ? g[m] * b[m + 1] : 0) - (m + 2 < n ? h[m] * b[m + 2] : 0)) / f[m]
        }
        var alpha = [Double](repeating: 0, count: n + 2)
        for m in stride(from: n - 1, through: 0, by: -1) {
            alpha[m] = b[m] + (2 - 4 * t) * alpha[m + 1] - alpha[m + 2]
        }
        return 2 * (alpha[0] + (n > 1 ? alpha[1] : 0))
    }

    /// `Σ C_j X^m Y^n` in the engine's term order (SURFACES §5).
    static func xyDeparture(_ c: [Double], _ x: Double, _ y: Double) -> Double {
        var z = 0.0
        for (j, cj) in c.enumerated() where cj != 0 {
            guard let term = XYTerm(index: j) else { break }
            z += cj * pow(x, Double(term.m)) * pow(y, Double(term.n))
        }
        return z
    }

    /// `Σ c_j Z_j(ρ, θ)` with θ from +x toward +y (SURFACES §6).
    static func zernikeDeparture(_ c: [Double], ordering: ZernikeOrdering, _ x: Double, _ y: Double) -> Double {
        let rho = (x * x + y * y).squareRoot(), theta = atan2(y, x)
        var z = 0.0
        for (k, cj) in c.enumerated() where cj != 0 {
            guard let term = ZernikeTerm(j: k + 1, ordering: ordering) else { break }
            z += cj * zernike(term, ordering: ordering, rho: rho, theta: theta)
        }
        return z
    }

    /// One Zernike term: Fringe unnormalised, Standard (Noll) normalised.
    public static func zernike(_ t: ZernikeTerm, ordering: ZernikeOrdering, rho: Double, theta: Double) -> Double {
        let m = abs(t.m)
        var r = 0.0
        for k in 0...((t.n - m) / 2) {
            let num = factorial(t.n - k)
            let den = factorial(k) * factorial((t.n + m) / 2 - k) * factorial((t.n - m) / 2 - k)
            r += (k % 2 == 0 ? 1 : -1) * num / den * pow(rho, Double(t.n - 2 * k))
        }
        let angular = t.m > 0 ? cos(Double(m) * theta) : t.m < 0 ? sin(Double(m) * theta) : 1
        let norm = ordering == .standard ? (t.m == 0 ? Double(t.n + 1).squareRoot() : (2 * Double(t.n + 1)).squareRoot()) : 1
        return norm * r * angular
    }

    static func factorial(_ n: Int) -> Double { n <= 1 ? 1 : (2...n).reduce(1.0) { $0 * Double($1) } }

    // MARK: Diffractive phase

    /// Phase `Φ(r) = Σ A_i (r/r_norm)^(2i)` in radians (order +1).
    public static func phase(_ p: DiffractivePhase, r: Double) -> Double {
        guard p.normRadius > 0 else { return 0 }
        let rho2 = (r / p.normRadius) * (r / p.normRadius)
        return p.coefficients.enumerated().reduce(0) { $0 + $1.element * pow(rho2, Double($1.offset + 1)) }
    }

    /// `dΦ/dr` in rad/mm.
    public static func phaseSlope(_ p: DiffractivePhase, r: Double) -> Double {
        guard p.normRadius > 0 else { return 0 }
        let rn2 = p.normRadius * p.normRadius, rho2 = r * r / rn2
        return p.coefficients.enumerated().reduce(0) { acc, e in
            let i = Double(e.offset + 1)
            return acc + e.element * i * pow(rho2, i - 1) * 2 * r / rn2
        }
    }

    /// Paraxial reduced power of the phase in order `m` at `wavelengthUm`,
    /// `φ = −m λ A₁ / (π r_norm²)` in mm⁻¹ (SURFACES §3, +z in air).
    public static func phasePower(_ p: DiffractivePhase, order m: Int, wavelengthUm: Double) -> Double {
        guard let a1 = p.coefficients.first, p.normRadius > 0 else { return 0 }
        return -Double(m) * wavelengthUm * 1e-3 * a1 / (Double.pi * p.normRadius * p.normRadius)
    }

    /// The smallest zone period `2π / |dΦ/dr|` (mm) over `0 < r ≤ h`, and
    /// where it occurs; `nil` without a phase gradient.
    public static func minimumPeriod(_ p: DiffractivePhase, upTo h: Double, samples: Int = 400) -> (period: Double, radius: Double)? {
        guard h > 0, h.isFinite else { return nil }
        var best: (Double, Double)?
        for k in 1...samples {
            let r = h * Double(k) / Double(samples)
            let slope = abs(phaseSlope(p, r: r))
            guard slope > 0, slope.isFinite else { continue }
            let period = 2 * Double.pi / slope
            if best == nil || period < best!.0 { best = (period, r) }
        }
        return best.map { (period: $0.0, radius: $0.1) }
    }
}
