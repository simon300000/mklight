import Foundation
import MKLightModel

/// MOCK OPTICS — illustrative only, not the engine.
///
/// A deliberately small stand-in that lets the UI be developed against data
/// that responds to edits: nominal Cauchy indices fitted to `n_d`/`V_d`, a
/// rotationally symmetric real-ray trace that ignores placements, ray aiming
/// and configurations, and closed-form approximations for MTF and PSF. Every
/// `MockEngine` result says so in its `RunInfo`. The real numbers come from
/// the Rust engine (`mklight-trace`, `mklight-analysis`, `mklight-wave`).
struct MockOptics {
    typealias V3 = SIMD3<Double>

    struct Ray {
        var p: V3
        var d: V3
    }

    struct Trace {
        /// Global points: start, then every surface hit.
        var points: [V3]
        /// Surface index where the ray was lost (`nil` if it reached the image).
        var failedAt: Int?
        /// Ray at the image surface (position and direction), if it got there.
        var atImage: Ray?
        /// Radial height at each surface (index → r), for auto semi-diameters.
        var heights: [Int: Double]
        var ok: Bool { failedAt == nil }
    }

    /// Object-space description of a field point.
    enum FieldPoint {
        /// Direction tangents (object at infinity, or finite object by angle).
        case angle(tanX: Double, tanY: Double)
        /// Object height (finite object).
        case height(x: Double, y: Double)
    }

    let system: OpticalSystem
    let wavelengthsUm: [Double]
    let primary: Int
    private(set) var warnings: [String] = []
    /// Global z of each surface vertex (surface 1 at z = 0).
    let vertexZ: [Double]
    /// Positive index of the medium after surface i at wavelength w.
    let nAfter: [[Double]]
    /// Propagation sign (±1 along z) after surface i.
    let travelSign: [Double]
    let objectDistance: Double?

    // First-order pupil geometry at the primary wavelength.
    private(set) var epZ = 0.0
    private(set) var epRadius = 1.0
    /// Paraxial image height per unit field parameter (tan or object height).
    private(set) var chiefImagePerUnit = 0.0

    static let catalog: [String: (nd: Double, vd: Double)] = [
        "N-BK7": (1.51680, 64.17), "N-BAK1": (1.57250, 57.55), "N-SK16": (1.62041, 60.32),
        "N-LAK9": (1.69100, 54.71), "N-BAF10": (1.67003, 47.11), "N-LAF2": (1.74397, 44.85),
        "F2": (1.62004, 36.37), "SF5": (1.67270, 32.21), "N-SF6": (1.80518, 25.36),
        "N-SF11": (1.78472, 25.68), "N-LASF9": (1.85025, 32.17), "FUSED_SILICA": (1.45846, 67.82),
        "CAF2": (1.43385, 94.99),
    ]

    init(system: OpticalSystem) {
        self.system = system
        let wls = system.wavelengths.wavelengths.map(\.um)
        wavelengthsUm = wls
        primary = system.wavelengths.primary
        let n = system.surfaces.count
        var warn = Set<String>()

        var z = [Double](repeating: 0, count: n)
        for i in 1..<max(n - 1, 1) { z[i + 1] = z[i] + system.surfaces[i].thickness }
        objectDistance = system.objectAtInfinity ? nil : system.surfaces[0].thickness
        z[0] = -(objectDistance ?? 0)
        vertexZ = z

        var media = [[Double]](repeating: [], count: n)
        var signs = [Double](repeating: 1, count: n)
        var before = system.surfaces[0].material
        var sign = 1.0
        for i in 0..<n {
            let s = system.surfaces[i]
            let inherit = i > 0 && (s.interaction != .refract || s.isCoordinateBreak)
            let after = inherit ? before : s.material
            media[i] = wls.map { Self.nominalIndex(after, $0, warnings: &warn) }
            if i > 0 && s.interaction == .reflect && !s.isCoordinateBreak { sign = -sign }
            signs[i] = sign
            before = after
        }
        nAfter = media
        travelSign = signs
        if system.surfaces.contains(where: { !($0.placement?.isIdentity ?? true) }) {
            warn.insert("mock engine ignores tilts and decenters")
        }
        if case .real = system.rayAiming { warn.insert("mock engine ignores ray aiming") }
        warnings = warn.sorted()
        computePupils()
    }

    // MARK: Materials

    /// Cauchy `n = A + B/λ²` through `n_d` with `n_F − n_C = (n_d − 1)/V_d`.
    static func cauchy(nd: Double, vd: Double, um: Double) -> Double {
        let f = SpectralLine.F.um, c = SpectralLine.C.um, d = SpectralLine.d.um
        let b = (nd - 1) / vd / (1 / (f * f) - 1 / (c * c))
        return nd + b * (1 / (um * um) - 1 / (d * d))
    }

    static func nominalIndex(_ m: MaterialRef, _ um: Double, warnings: inout Set<String>) -> Double {
        switch m {
        case .air, .vacuum: return 1
        case .constant(let n): return n
        case .model(let nd, let vd, _): return cauchy(nd: nd, vd: max(vd, 1), um: um)
        case .catalog(_, let name, let o):
            if let g = catalog[name.uppercased()] { return cauchy(nd: g.nd + o.dnD, vd: g.vd + o.dvD, um: um) }
            warnings.insert("unknown glass '\(name)': mock uses n_d = 1.5, V_d = 50")
            return cauchy(nd: 1.5, vd: 50, um: um)
        }
    }

    func nBefore(_ i: Int, _ w: Int) -> Double { nAfter[max(i - 1, 0)][w] }

    // MARK: Paraxial (y–nu) trace with signed reduced indices

    struct Paraxial {
        var y: [Double]
        /// Reduced slope `n·u` after each surface (signed index).
        var nu: [Double]
    }

    func paraxial(y1: Double, u0: Double, w: Int) -> Paraxial {
        let n = system.surfaces.count
        var ys = [Double](repeating: 0, count: n), nus = [Double](repeating: 0, count: n)
        var bigN = nAfter[0][w]
        var nu = bigN * u0, y = y1
        nus[0] = nu
        for i in 1..<n {
            ys[i] = y
            let s = system.surfaces[i]
            let nPrime = nAfter[i][w] * travelSign[i]
            var phi = 0.0
            if !s.isCoordinateBreak && s.interaction != .dummy {
                switch s.shape {
                case .conic(let c, _), .evenAsphere(let c, _, _): phi = c * (nPrime - bigN)
                case .paraxial(let f): phi = f != 0 ? 1 / f : 0
                case .plane, .coordinateBreak: phi = 0
                case .diffractive(let base, let ph):
                    // Base power plus the order's phase power −mλA₁/(π r_n²).
                    let a1 = ph.coefficients.first ?? 0
                    let lambdaMM = wavelengthsUm[w] * 1e-3
                    phi = base.curvature * (nPrime - bigN)
                        - Double(ph.traceOrder) * lambdaMM * a1 / (Double.pi * ph.normRadius * ph.normRadius)
                default:
                    // Extended shapes: the mock uses the y vertex curvature
                    // of the base conic (toroid/biconic: c_y), ignoring
                    // the departure's quadratic terms.
                    phi = s.shape.curvature * (nPrime - bigN)
                }
            }
            nu -= y * phi
            nus[i] = nu
            if i < n - 1 { y += s.thickness / nPrime * nu }
            bigN = nPrime
        }
        return Paraxial(y: ys, nu: nus)
    }

    /// Image-space reduced index (signed) at wavelength w.
    func imageN(_ w: Int) -> Double {
        let k = system.surfaces.count - 2
        return nAfter[k][w] * travelSign[k]
    }

    private mutating func computePupils() {
        let stop = min(max(system.stop, 1), system.surfaces.count - 2)
        let w = primary
        let a = paraxial(y1: 1, u0: 0, w: w).y[stop]
        let b = paraxial(y1: 0, u0: 1, w: w).y[stop]
        epZ = abs(a) > 1e-12 ? b / a : 0
        let d = objectDistance
        let stopSD = system.surfaces[stop].semiDiameter.fixedValue ?? system.surfaces[stop].aperture?.boundingRadius
        switch system.aperture {
        case .entrancePupilDiameter(let v): epRadius = v / 2
        case .imageSpaceFNumber(let f):
            epRadius = abs(efl(w) ?? 100) / (2 * max(f, 1e-6))
            if d != nil { warnings.append("image-space F/# assumes an object at infinity") }
        case .objectSpaceNA(let na):
            let u = na / nAfter[0][w]
            epRadius = abs(u * (d.map(objectToPupil) ?? epZ))
        case .paraxialWorkingFNumber(let nw):
            let uTarget = 1 / (2 * max(nw, 1e-6) * abs(imageN(w)))
            if let d {
                let pr = paraxial(y1: d, u0: 1, w: w)
                let u = abs(pr.nu[pr.nu.count - 2] / imageN(w))
                epRadius = u > 0 ? uTarget / u * abs(objectToPupil(d)) : 1
            } else {
                let pr = paraxial(y1: 1, u0: 0, w: w)
                let u = abs(pr.nu[pr.nu.count - 2] / imageN(w))
                epRadius = u > 0 ? uTarget / u : 1
            }
        case .floatByStopSize:
            let rs = stopSD ?? 1
            if stopSD == nil { warnings.append("float-by-stop without a fixed stop size: mock uses 1 mm") }
            if let d {
                let u = abs(a * d + b) > 1e-12 ? rs / (a * d + b) : 0
                epRadius = abs(u * objectToPupil(d))
            } else {
                epRadius = abs(a) > 1e-12 ? abs(rs / a) : 1
            }
        }
        if let d, abs(d + epZ) <= 1e-9 * max(1, abs(d), abs(epZ)) {
            warnings.append("object lies on the entrance pupil: paraxial pupil slopes are degenerate")
        }
        if !(epRadius.isFinite && epRadius > 0) {
            warnings.append("entrance pupil undefined for this aperture specification: mock uses a 1 mm radius")
            epRadius = 1
        }

        // Paraxial chief-ray image height per unit field parameter.
        let unit = chiefStart()
        let pr = paraxial(y1: unit.y1, u0: unit.u0, w: w)
        chiefImagePerUnit = pr.y[pr.y.count - 1]
    }

    /// Paraxial chief-ray start `(y1, u0)` for a unit tangent (infinite
    /// object) or unit object height (finite object).
    private func chiefStart() -> (y1: Double, u0: Double) {
        if let d = objectDistance {
            let u0 = -1 / objectToPupil(d)
            return (1 + u0 * d, u0)
        }
        return (-epZ, 1)
    }

    /// Object-to-entrance-pupil distance `d + z_EP`, kept away from zero so
    /// that an object on the pupil (warned about in `computePupils`) gives
    /// large but finite slopes instead of NaN.
    func objectToPupil(_ d: Double) -> Double {
        let l = d + epZ, floor = 1e-9 * max(1, abs(d), abs(epZ))
        return abs(l) > floor ? l : (l < 0 ? -floor : floor)
    }

    /// Paraxial marginal-ray start `(y1, u0)` from the axial object point to
    /// the edge of the entrance pupil.
    func marginalStart() -> (y1: Double, u0: Double) {
        guard let d = objectDistance else { return (epRadius, 0) }
        let u0 = epRadius / objectToPupil(d)
        return (u0 * d, u0)
    }

    func efl(_ w: Int) -> Double? {
        let pr = paraxial(y1: 1, u0: 0, w: w)
        let nu = pr.nu[pr.nu.count - 2]
        return abs(nu) > 1e-15 ? -1 / nu : nil
    }

    // MARK: Fields

    var maxField: Double { system.fields.maxRadial }

    /// Object-space field point for a field value `(x, y)` in the system's
    /// field unit.
    func fieldPoint(x: Double, y: Double) -> FieldPoint {
        switch system.fields.kind {
        case .objectAngle:
            let t = FieldPoint.angle(tanX: tan(x * .pi / 180), tanY: tan(y * .pi / 180))
            if let d = objectDistance, case .angle(let tx, let ty) = t {
                return .height(x: -(epZ + d) * tx, y: -(epZ + d) * ty)
            }
            return t
        case .objectHeight:
            return .height(x: x, y: y)
        case .paraxialImageHeight, .realImageHeight:
            let k = abs(chiefImagePerUnit) > 1e-12 ? 1 / chiefImagePerUnit : 0
            return objectDistance == nil ? .angle(tanX: x * k, tanY: y * k) : .height(x: x * k, y: y * k)
        }
    }

    func fieldPoint(_ i: Int) -> FieldPoint {
        let f = system.fields.fields[i]
        return fieldPoint(x: f.x, y: f.y)
    }

    /// Field parameter along y (tan or object height) for paraxial heights.
    func fieldParameterY(_ fp: FieldPoint) -> Double {
        switch fp {
        case .angle(_, let ty): ty
        case .height(_, let y): y
        }
    }

    var trackLength: Double { abs(vertexZ[vertexZ.count - 1] - vertexZ[min(1, vertexZ.count - 1)]) }

    // MARK: Real rays

    func makeRay(_ fp: FieldPoint, px: Double, py: Double) -> Ray {
        let pupil = V3(px * epRadius, py * epRadius, epZ)
        switch fp {
        case .angle(let tx, let ty):
            let d = simd_normalize3(V3(tx, ty, 1))
            let zStart = min(0, epZ) - max(0.15 * trackLength, 5)
            return Ray(p: pupil + d * ((zStart - epZ) / d.z), d: d)
        case .height(let hx, let hy):
            let o = V3(hx, hy, vertexZ[0])
            return Ray(p: o, d: simd_normalize3(pupil - o))
        }
    }

    func trace(_ start: Ray, w: Int) -> Trace {
        var out = Trace(points: [start.p], failedAt: nil, atImage: nil, heights: [:])
        var ray = start
        let n = system.surfaces.count
        for i in 1..<n {
            let s = system.surfaces[i]
            if s.isCoordinateBreak { continue }
            let q = ray.p - V3(0, 0, vertexZ[i])
            guard let t = Self.intersect(s.shape, q, ray.d) else {
                out.failedAt = i
                return out
            }
            let hit = q + t * ray.d
            let global = hit + V3(0, 0, vertexZ[i])
            out.points.append(global)
            out.heights[i] = (hit.x * hit.x + hit.y * hit.y).squareRoot()
            if !Self.passes(s, hit.x, hit.y) {
                out.failedAt = i
                return out
            }
            if i == n - 1 {
                out.atImage = Ray(p: global, d: ray.d)
                break
            }
            guard let d = Self.interact(s, hit, ray.d, n1: nBefore(i, w), n2: nAfter[i][w]) else {
                out.failedAt = i
                return out
            }
            ray = Ray(p: global, d: d)
        }
        return out
    }

    static func passes(_ s: Surface, _ x: Double, _ y: Double) -> Bool {
        let r2 = x * x + y * y
        if let ap = s.aperture {
            switch ap {
            case .circular(let r): return r2 <= r * r
            case .annular(let a, let b): return r2 >= a * a && r2 <= b * b
            case .rectangular(let hx, let hy): return abs(x) <= hx && abs(y) <= hy
            case .elliptical(let a, let b):
                let u = x / a, v = y / b
                return u * u + v * v <= 1
            }
        }
        if case .fixed(let r) = s.semiDiameter { return r2 <= r * r }
        return true
    }

    /// Sag `z(ρ)` and `dz/dρ` with `ρ = r²`, in the cancellation-free form.
    static func sag(_ shape: SurfaceShape, _ rho: Double) -> (z: Double, dz: Double)? {
        func conic(_ c: Double, _ k: Double) -> (Double, Double)? {
            let arg = 1 - (1 + k) * c * c * rho
            guard arg >= 0 else { return nil }
            let s = arg.squareRoot()
            return (c * rho / (1 + s), s > 0 ? c / (2 * s) : .infinity)
        }
        switch shape {
        case .plane, .paraxial, .coordinateBreak: return (0, 0)
        case .conic(let c, let k): return conic(c, k).map { (z: $0.0, dz: $0.1) }
        case .evenAsphere(let c, let k, let a):
            guard let (z0, d0) = conic(c, k) else { return nil }
            var poly = 0.0, dpoly = 0.0
            for (i, ai) in a.enumerated().reversed() {
                poly = poly * rho + ai
                dpoly = dpoly * rho + Double(i + 2) * ai
            }
            return (z0 + poly * rho * rho, d0 + dpoly * rho)
        case .oddAsphere(let c, let k, let a):
            guard let (z0, d0) = conic(c, k) else { return nil }
            let r = rho.squareRoot()
            var z = z0, dz = d0
            for (i, b) in a.enumerated() {
                let p = Double(i + 1)
                z += b * pow(r, p)
                // d(r^p)/dρ = (p/2) r^(p−2); the cone term's normal is +z at the vertex.
                if r > 0 { dz += b * p / 2 * pow(r, p - 2) }
            }
            return (z, dz)
        // The mock ignores fringe irregularity.
        case .diffractive(let base, _), .sagOverlay(let base, _, _, _, _): return sag(base, rho)
        case .toroid(_, let cy, let ky, let ya), .biconic(_, let cy, _, let ky, _, let ya):
            // The y meridian is exactly the y profile; the mock is rotational.
            return sag(.evenAsphere(curvature: cy, conic: ky, coefficients: ya), rho)
        case .qCon(let c, let k, _, _), .xyPolynomial(let c, let k, _, _), .zernikeSag(let c, let k, _, _, _):
            // Mock approximation: the base conic without the departure.
            return conic(c, k).map { (z: $0.0, dz: $0.1) }
        case .qBfs(let c, _, _):
            return conic(c, 0).map { (z: $0.0, dz: $0.1) }
        }
    }

    /// Ray–surface intersection distance in the surface's local frame.
    static func intersect(_ shape: SurfaceShape, _ q: V3, _ d: V3) -> Double? {
        func planeT() -> Double? { abs(d.z) > 1e-15 ? -q.z / d.z : nil }
        func conicT(_ c: Double, _ k: Double) -> Double? {
            // c(x² + y² + (1+k) z²) − 2z = 0 along q + t d, root in the stable
            // form t = C / (−B ∓ √(B² − AC)) that stays finite as c → 0.
            let a = c * (d.x * d.x + d.y * d.y + (1 + k) * d.z * d.z)
            let b = c * (q.x * d.x + q.y * d.y + (1 + k) * q.z * d.z) - d.z
            let cc = c * (q.x * q.x + q.y * q.y + (1 + k) * q.z * q.z) - 2 * q.z
            let disc = b * b - a * cc
            guard disc >= 0 else { return nil }
            let root = disc.squareRoot()
            let denom = -b + (b <= 0 ? root : -root)
            guard abs(denom) > 1e-300 else { return nil }
            let t = cc / denom
            let p = q + t * d
            guard sag(.conic(curvature: c, conic: k), p.x * p.x + p.y * p.y) != nil else { return nil }
            return t
        }
        switch shape {
        case .plane, .paraxial, .coordinateBreak: return planeT()
        case .conic(let c, let k): return conicT(c, k)
        case .diffractive(let base, _): return intersect(base, q, d)
        default:
            // Newton on the mock's rotational sag from the base conic's root.
            guard var t = conicT(shape.curvature, shape.conicConstant) ?? planeT() else { return nil }
            for _ in 0..<30 {
                let p = q + t * d
                let rho = p.x * p.x + p.y * p.y
                guard let (z, dz) = sag(shape, rho) else { return nil }
                let g = p.z - z
                let gp = d.z - dz * 2 * (p.x * d.x + p.y * d.y)
                guard abs(gp) > 1e-14, gp.isFinite else { return nil }
                let step = g / gp
                t -= step
                if abs(step) <= 1e-13 * (1 + abs(t)) { return t }
            }
            return nil
        }
    }

    /// New direction after the surface interaction (`nil` on TIR or failure).
    static func interact(_ s: Surface, _ hit: V3, _ d: V3, n1: Double, n2: Double) -> V3? {
        if s.interaction == .dummy { return d }
        if case .paraxial(let f) = s.shape {
            guard abs(d.z) > 1e-12, f != 0 else { return nil }
            let sz = d.z > 0 ? 1.0 : -1.0
            let target = V3(f * d.x / abs(d.z), f * d.y / abs(d.z), sz * f)
            return simd_normalize3((f > 0 ? 1 : -1) * (target - V3(hit.x, hit.y, 0)))
        }
        guard let (_, dz) = sag(s.shape, hit.x * hit.x + hit.y * hit.y), dz.isFinite else { return nil }
        var nrm = simd_normalize3(V3(-2 * hit.x * dz, -2 * hit.y * dz, 1))
        var cosI = simd_dot3(d, nrm)
        if cosI < 0 {
            nrm = -nrm
            cosI = -cosI
        }
        if s.interaction == .reflect { return simd_normalize3(d - 2 * cosI * nrm) }
        // Vector Snell law with 1 − cos² = (1 − cos)(1 + cos) (CONVENTIONS §7.3).
        let mu = n1 / n2
        let rad = 1 - mu * mu * ((1 - cosI) * (1 + cosI))
        guard rad >= 0 else { return nil }
        return simd_normalize3(mu * d + (rad.squareRoot() - mu * cosI) * nrm)
    }
}

@inline(__always) func simd_dot3(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
    (a.x * b.x).addingProduct(a.y, b.y).addingProduct(a.z, b.z)
}

@inline(__always) func simd_normalize3(_ v: SIMD3<Double>) -> SIMD3<Double> {
    let l = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
    return l > 0 ? v / l : v
}

/// SplitMix64: a small deterministic generator for the mock's random pupil
/// sampling. (The engine uses Philox4x32-10, CONVENTIONS §6.)
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [0, 1) with 53 random bits.
    mutating func unit() -> Double { Double(next() >> 11) * 0x1p-53 }
}

/// Neumaier-compensated sum.
struct CompensatedSum {
    private var sum = 0.0, c = 0.0
    mutating func add(_ x: Double) {
        let t = sum + x
        c += abs(sum) >= abs(x) ? (sum - t) + x : (x - t) + sum
        sum = t
    }
    var value: Double { sum + c }
}
