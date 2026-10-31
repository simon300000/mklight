import Foundation

/// Reading and writing the value a `ParamPath` names, with the same rules as
/// the engine's `ParamPath::get` / `set` (no optics). Used for variables,
/// solve outcomes and generic parameter forms.
extension OpticalSystem {
    /// The current value of `path`, or `nil` when the system has no such
    /// parameter (an automatic semi-diameter has no value).
    public func value(at path: ParamPath) -> Double? {
        guard path.resolves(in: self) else { return nil }
        if let id = path.surface {
            guard let i = surfaceIndex(of: id) else { return nil }
            let s = surfaces[i]
            switch path {
            case .curvature: return s.shape.curvature
            case .thickness: return s.thickness
            case .conic: return s.shape.conicConstant
            case .asphere(_, let k):
                let a = s.shape.asphereCoefficients
                return a.indices.contains(k) ? a[k] : 0
            case .shapeCoefficient(_, let set, let k):
                let a = s.shape.coefficients(in: set) ?? []
                return a.indices.contains(k) ? a[k] : 0
            case .shapeParam(_, let p): return s.shape.shapeParameter(p)
            case .paraxialFocalLength:
                if case .paraxial(let f) = s.shape { return f }
                return nil
            case .modelNd, .modelVd, .modelDpgf:
                guard case .model(let nd, let vd, let dpgf) = s.material else { return nil }
                if case .modelNd = path { return nd }
                if case .modelVd = path { return vd }
                return dpgf
            case .materialDnD, .materialDvD:
                guard case .catalog(_, _, let o) = s.material else { return nil }
                if case .materialDnD = path { return o.dnD }
                return o.dvD
            case .decenterX: return s.placement?.decenterX ?? 0
            case .decenterY: return s.placement?.decenterY ?? 0
            case .tiltX: return s.placement?.tiltX ?? 0
            case .tiltY: return s.placement?.tiltY ?? 0
            case .tiltZ: return s.placement?.tiltZ ?? 0
            case .semiDiameter: return s.semiDiameter.fixedValue
            default: return nil
            }
        }
        switch path {
        case .apertureValue: return aperture.value
        case .fieldX(let i): return fields.fields[i].x
        case .fieldY(let i): return fields.fields[i].y
        case .fieldWeight(let i): return fields.fields[i].weight
        case .wavelengthUm(let i): return wavelengths.wavelengths[i].um
        case .wavelengthWeight(let i): return wavelengths.wavelengths[i].weight
        case .temperatureC: return environment.temperatureC
        case .pressureAtm: return environment.pressureAtm
        default: return nil
        }
    }

    /// Sets the value `path` names. Returns false when the system has no
    /// such parameter. Like the engine, setting a curvature on a plane makes
    /// it a conic (a conic on a plane is refused), curvature, conic and
    /// asphere paths address a diffractive surface's base, a coefficient past
    /// the end extends the list with zeros, an index at or beyond the shape's
    /// limit is refused, and a diffraction order must be an integral `i32`.
    @discardableResult
    public mutating func setValue(_ v: Double, at path: ParamPath) -> Bool {
        if let id = path.surface {
            guard let i = surfaceIndex(of: id) else { return false }
            var s = surfaces[i]
            func placed(_ f: (inout Placement) -> Void) {
                var p = s.placement ?? Placement()
                f(&p)
                s.placement = p
            }
            switch path {
            case .curvature: guard s.shape.setCurvature(v) else { return false }
            case .conic: guard s.shape.hasConic, s.shape.setConic(v) else { return false }
            case .thickness: s.thickness = v
            case .asphere(_, let k): guard s.shape.setAsphereCoefficient(v, at: k) else { return false }
            case .shapeCoefficient(_, let set, let k):
                guard let max = set.maxCount(for: s.shape), k >= 0, k < max, var a = s.shape.coefficients(in: set)
                else { return false }
                if k >= a.count { a += Array(repeating: 0, count: k - a.count + 1) }
                a[k] = v
                s.shape.setCoefficients(a, in: set)
            case .shapeParam(_, let p): guard s.shape.setShapeParameter(p, v) else { return false }
            case .paraxialFocalLength:
                guard case .paraxial = s.shape else { return false }
                s.shape = .paraxial(focalLength: v)
            case .modelNd, .modelVd, .modelDpgf:
                guard case .model(var nd, var vd, var dpgf) = s.material else { return false }
                if case .modelNd = path { nd = v } else if case .modelVd = path { vd = v } else { dpgf = v }
                s.material = .model(nd: nd, vd: vd, dpgf: dpgf)
            case .materialDnD, .materialDvD:
                guard case .catalog(let cat, let name, var o) = s.material else { return false }
                if case .materialDnD = path { o.dnD = v } else { o.dvD = v }
                s.material = .catalog(catalog: cat, name: name, offset: o)
            case .decenterX: placed { $0.decenterX = v }
            case .decenterY: placed { $0.decenterY = v }
            case .tiltX: placed { $0.tiltX = v }
            case .tiltY: placed { $0.tiltY = v }
            case .tiltZ: placed { $0.tiltZ = v }
            case .semiDiameter: s.semiDiameter = .fixed(v)
            default: return false
            }
            surfaces[i] = s
            return true
        }
        switch path {
        case .apertureValue:
            guard aperture.value != nil else { return false }
            aperture = ApertureSpec(kind: aperture.kind, value: v)
        case .fieldX(let i): guard fields.fields.indices.contains(i) else { return false }; fields.fields[i].x = v
        case .fieldY(let i): guard fields.fields.indices.contains(i) else { return false }; fields.fields[i].y = v
        case .fieldWeight(let i): guard fields.fields.indices.contains(i) else { return false }; fields.fields[i].weight = v
        case .wavelengthUm(let i):
            guard wavelengths.wavelengths.indices.contains(i) else { return false }
            wavelengths.wavelengths[i].um = v
        case .wavelengthWeight(let i):
            guard wavelengths.wavelengths.indices.contains(i) else { return false }
            wavelengths.wavelengths[i].weight = v
        case .temperatureC: environment.temperatureC = v
        case .pressureAtm: environment.pressureAtm = v
        default: return false
        }
        return true
    }
}

extension ParamPath {
    /// A short human label, e.g. `R 3`, `T 4`, `A6 2`, `field 2 y`, with the
    /// surface given by its index in `system`.
    public func label(in system: OpticalSystem) -> String {
        let n = surface.flatMap(system.surfaceIndex(of:)).map(String.init) ?? "?"
        switch self {
        case .curvature(let id):
            // The y section of a toroid or biconic.
            let anamorphic = system.surfaceIndex(of: id).map { system.surfaces[$0].shape.kind == .toroid || system.surfaces[$0].shape.kind == .biconic } ?? false
            return anamorphic ? "Radius y \(n)" : "Radius \(n)"
        case .thickness: return "Thickness \(n)"
        case .conic: return "Conic \(n)"
        case .asphere(_, let k): return "A\(2 * k + 4) on \(n)"
        case .shapeCoefficient(_, let set, let k):
            let family =
                switch set {
                case .odd: "Odd "
                case .phase: "Phase "
                default: ""
                }
            return "\(family)\(set.termLabel(k)) on \(n)"
        case .shapeParam(_, let p): return "\(p.title) on \(n)"
        case .paraxialFocalLength: return "Focal length \(n)"
        case .modelNd: return "n_d \(n)"
        case .modelVd: return "V_d \(n)"
        case .modelDpgf: return "ΔP_g,F \(n)"
        case .materialDnD: return "Δn_d \(n)"
        case .materialDvD: return "ΔV_d \(n)"
        case .decenterX: return "Decenter X \(n)"
        case .decenterY: return "Decenter Y \(n)"
        case .tiltX: return "Tilt X \(n)"
        case .tiltY: return "Tilt Y \(n)"
        case .tiltZ: return "Tilt Z \(n)"
        case .semiDiameter: return "Semi-diameter \(n)"
        case .apertureValue: return "Aperture"
        case .fieldX(let i): return "Field \(i + 1) x"
        case .fieldY(let i): return "Field \(i + 1) y"
        case .fieldWeight(let i): return "Field \(i + 1) weight"
        case .wavelengthUm(let i): return "λ\(i + 1)"
        case .wavelengthWeight(let i): return "λ\(i + 1) weight"
        case .temperatureC: return "Temperature"
        case .pressureAtm: return "Pressure"
        }
    }

    /// Whether the value is a curvature `c = 1/R` (1/mm): shown and bounded
    /// as a radius by default.
    public var isCurvature: Bool {
        switch self {
        case .curvature, .shapeParam(_, .curvatureX), .shapeParam(_, .curvatureY): true
        default: false
        }
    }

    /// The unit of the parameter's value (and of its bounds), `""` when it
    /// is dimensionless: `mm`, `1/mm`, `mm⁻³` (A₄), `°`, `µm`, `rad`, …
    /// Polynomial coefficients carry the inverse powers that make their
    /// term a length (docs/SURFACES.md).
    public func unit(in system: OpticalSystem) -> String {
        switch self {
        case .curvature: return "1/mm"
        case .thickness, .decenterX, .decenterY, .semiDiameter, .paraxialFocalLength: return "mm"
        case .conic, .modelNd, .modelVd, .modelDpgf, .materialDnD, .materialDvD, .fieldWeight, .wavelengthWeight:
            return ""
        case .asphere(_, let k): return Self.inverseMM(2 * k + 3)
        case .shapeCoefficient(_, let set, let k):
            switch set {
            case .odd: return Self.inverseMM(k)
            case .xAsphere, .yAsphere: return Self.inverseMM(2 * k + 3)
            case .q, .xy, .zernike: return "mm"
            case .phase: return "rad"
            }
        case .shapeParam(_, let p):
            switch p {
            case .curvatureX, .curvatureY: return "1/mm"
            case .conicX, .conicY, .designOrder, .traceOrder: return ""
            case .normRadius: return "mm"
            case .designWavelengthUm: return "µm"
            }
        case .tiltX, .tiltY, .tiltZ: return "°"
        case .apertureValue: return system.aperture.kind.unit
        case .fieldX, .fieldY: return system.fields.kind.unit
        case .wavelengthUm: return "µm"
        case .temperatureC: return "°C"
        case .pressureAtm: return "atm"
        }
    }

    /// `mm⁻ⁿ` with a superscript exponent; `""` for n = 0, `1/mm` for n = 1.
    static func inverseMM(_ n: Int) -> String {
        if n <= 0 { return "" }
        if n == 1 { return "1/mm" }
        let sup: [Character] = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]
        return "mm⁻" + String(String(n).map { sup[$0.wholeNumberValue ?? 0] })
    }
}
