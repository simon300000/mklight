import Foundation

extension SurfaceShape {
    /// The shape types, without their parameters. The raw values are stable
    /// identifiers; `typeName` is the JSON `type` tag.
    public enum Kind: String, CaseIterable, Sendable, Identifiable {
        case plane, conic, evenAsphere, oddAsphere, qCon, qBfs, diffractive, toroid, biconic, xyPolynomial, zernikeSag
        case paraxial, coordinateBreak

        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .plane: "Plane"
            case .conic: "Standard"
            case .evenAsphere: "Even Asphere"
            case .oddAsphere: "Odd Asphere"
            case .qCon: "Q-Type (Conic)"
            case .qBfs: "Q-Type (Best-Fit Sphere)"
            case .diffractive: "Diffractive"
            case .toroid: "Toroid"
            case .biconic: "Biconic"
            case .xyPolynomial: "XY Polynomial"
            case .zernikeSag: "Zernike Sag"
            case .paraxial: "Paraxial"
            case .coordinateBreak: "Coord Break"
            }
        }

        /// The JSON `type` tag.
        public var typeName: String {
            switch self {
            case .plane: "plane"
            case .conic: "conic"
            case .evenAsphere: "even_asphere"
            case .oddAsphere: "odd_asphere"
            case .qCon: "q_con"
            case .qBfs: "q_bfs"
            case .diffractive: "diffractive"
            case .toroid: "toroid"
            case .biconic: "biconic"
            case .xyPolynomial: "xy_polynomial"
            case .zernikeSag: "zernike_sag"
            case .paraxial: "paraxial"
            case .coordinateBreak: "coordinate_break"
            }
        }

        /// The engine's original solver subset (`SurfaceShape::is_legacy`).
        public var isLegacy: Bool {
            switch self {
            case .plane, .conic, .evenAsphere, .paraxial, .coordinateBreak: true
            default: false
            }
        }

        /// Types with a normalisation radius (Q, XY, Zernike and the
        /// diffractive phase).
        public var isNormalised: Bool {
            switch self {
            case .qCon, .qBfs, .xyPolynomial, .zernikeSag, .diffractive: true
            default: false
            }
        }
    }

    public var kind: Kind {
        switch self {
        case .plane: .plane
        case .conic: .conic
        case .evenAsphere: .evenAsphere
        case .oddAsphere: .oddAsphere
        case .qCon: .qCon
        case .qBfs: .qBfs
        case .diffractive: .diffractive
        case .toroid: .toroid
        case .biconic: .biconic
        case .xyPolynomial: .xyPolynomial
        case .zernikeSag: .zernikeSag
        case .sagOverlay(let base, _, _, _, _, _): base.kind
        case .paraxial: .paraxial
        case .coordinateBreak: .coordinateBreak
        }
    }

    /// The shape carrying the sag: a diffractive surface's base, else itself
    /// (the engine's `sag_shape`).
    public var sagShape: SurfaceShape {
        if case .diffractive(let base, _) = self { return base }
        return self
    }

    /// The base of a sag overlay, else the shape itself.
    var sagShapeBase: SurfaceShape {
        if case .sagOverlay(let base, _, _, _, _, _) = self { return base }
        return self
    }

    /// Sets fringe irregularity `mode`, wrapping the sag shape in a zero
    /// overlay over `clearRadius` first, like the engine's
    /// `ParamPath::Irregularity` setter. False for a paraxial lens or
    /// coordinate break, or when a new overlay has no clear radius.
    mutating func setIrregularity(
        _ mode: IrregularityMode, _ v: Double, clearRadius: Double?, testWavelengthUm: Double
    ) -> Bool {
        if case .diffractive(var base, let phase) = self {
            guard base.setIrregularity(mode, v, clearRadius: clearRadius, testWavelengthUm: testWavelengthUm)
            else { return false }
            self = .diffractive(base: base, phase: phase)
            return true
        }
        switch self {
        case .paraxial, .coordinateBreak: return false
        case .sagOverlay(let base, let r, let w, let a, let angle, let s):
            self = .sagOverlay(
                base: base, normRadius: r, testWavelengthUm: w, astigmatism: mode == .astigmatism ? v : a,
                astigmatismAngleDeg: angle, spherical: mode == .spherical ? v : s)
        default:
            guard let clearRadius else { return false }
            self = .sagOverlay(
                base: self, normRadius: clearRadius, testWavelengthUm: testWavelengthUm,
                astigmatism: mode == .astigmatism ? v : 0, spherical: mode == .spherical ? v : 0)
        }
        return true
    }

    /// The geometric surface without a diffractive phase or sag overlay (the
    /// engine's `geometry`). Curvature, conic and asphere paths address it.
    public var geometry: SurfaceShape {
        switch self {
        case .diffractive(let base, _), .sagOverlay(let base, _, _, _, _, _): base.geometry
        default: self
        }
    }

    /// Vertex curvature as the engine's `SurfaceShape::curvature()`: the y
    /// curvature of a toroid or biconic, the base's for a diffractive
    /// surface, 0 for shapes without one.
    public var curvature: Double {
        switch self {
        case .conic(let c, _), .evenAsphere(let c, _, _), .oddAsphere(let c, _, _), .qCon(let c, _, _, _),
            .qBfs(let c, _, _), .xyPolynomial(let c, _, _, _), .zernikeSag(let c, _, _, _, _):
            c
        case .diffractive(let base, _), .sagOverlay(let base, _, _, _, _, _): base.curvature
        case .toroid(_, let cy, _, _), .biconic(_, let cy, _, _, _, _): cy
        default: 0
        }
    }

    /// Conic constant as the engine's `SurfaceShape::conic()`: the y conic
    /// of a toroid or biconic, the base's for a diffractive surface, 0 for
    /// shapes without one (including `qBfs`).
    public var conicConstant: Double {
        switch self {
        case .conic(_, let k), .evenAsphere(_, let k, _), .oddAsphere(_, let k, _), .qCon(_, let k, _, _),
            .xyPolynomial(_, let k, _, _), .zernikeSag(_, let k, _, _, _):
            k
        case .diffractive(let base, _), .sagOverlay(let base, _, _, _, _, _): base.conicConstant
        case .toroid(_, _, let ky, _), .biconic(_, _, _, let ky, _, _): ky
        default: 0
        }
    }

    /// Even-asphere coefficients of the geometry (A4, A6, …), as the
    /// `asphere` path reads them; empty otherwise.
    public var asphereCoefficients: [Double] {
        if case .evenAsphere(_, _, let a) = geometry { return a }
        return []
    }

    /// The normalisation radius of a Q, XY, Zernike or diffractive shape.
    public var normRadius: Double? { shapeParameter(.normRadius) }

    /// Whether the `curvature` path resolves (the engine's `Curvature` get):
    /// the geometry is a plane or has a `curvature` field, which on a toroid
    /// or biconic is the y section. Paraxial and coordinate break have none.
    public var hasCurvature: Bool {
        switch geometry {
        case .plane, .conic, .evenAsphere, .oddAsphere, .qCon, .qBfs, .xyPolynomial, .zernikeSag, .toroid, .biconic: true
        default: false
        }
    }

    /// Whether the `conic` path resolves (the engine's `Conic` get). A plane
    /// and `qBfs` have no conic.
    public var hasConic: Bool {
        switch geometry {
        case .conic, .evenAsphere, .oddAsphere, .qCon, .xyPolynomial, .zernikeSag: true
        default: false
        }
    }

    /// Whether the `asphere` path resolves: the geometry is an even asphere.
    public var hasAsphereCoefficients: Bool {
        if case .evenAsphere = geometry { return true }
        return false
    }

    /// Applies `edit` to the base of a sag overlay; `nil` when not an overlay.
    mutating func editOverlayBase(_ edit: (inout SurfaceShape) -> Bool) -> Bool? {
        guard case .sagOverlay(var base, let r, let w, let a, let angle, let s) = self else { return nil }
        guard edit(&base) else { return false }
        self = .sagOverlay(
            base: base, normRadius: r, testWavelengthUm: w, astigmatism: a, astigmatismAngleDeg: angle, spherical: s)
        return true
    }

    /// Applies `edit` to the geometry (inside a diffractive phase or overlay).
    private mutating func editGeometry(_ edit: (inout SurfaceShape) -> Bool) -> Bool {
        switch self {
        case .diffractive(var base, let phase):
            guard base.editGeometry(edit) else { return false }
            self = .diffractive(base: base, phase: phase)
            return true
        case .sagOverlay:
            return editOverlayBase { $0.editGeometry(edit) } ?? false
        default:
            return edit(&self)
        }
    }

    /// Sets the curvature of the geometry, promoting a plane to a sphere,
    /// exactly like the engine's `ParamPath::Curvature` setter (the y section
    /// of a toroid or biconic). Returns false for shapes without one.
    @discardableResult
    public mutating func setCurvature(_ c: Double) -> Bool {
        editGeometry { s in
            switch s {
            case .plane: s = .conic(curvature: c, conic: 0)
            case .conic(_, let k): s = .conic(curvature: c, conic: k)
            case .evenAsphere(_, let k, let a): s = .evenAsphere(curvature: c, conic: k, coefficients: a)
            case .oddAsphere(_, let k, let a): s = .oddAsphere(curvature: c, conic: k, coefficients: a)
            case .qCon(_, let k, let r, let a): s = .qCon(curvature: c, conic: k, normRadius: r, coefficients: a)
            case .qBfs(_, let r, let a): s = .qBfs(curvature: c, normRadius: r, coefficients: a)
            case .xyPolynomial(_, let k, let r, let a):
                s = .xyPolynomial(curvature: c, conic: k, normRadius: r, coefficients: a)
            case .zernikeSag(_, let k, let r, let o, let a):
                s = .zernikeSag(curvature: c, conic: k, normRadius: r, ordering: o, coefficients: a)
            case .toroid(let cx, _, let k, let a): s = .toroid(curvatureX: cx, curvatureY: c, conicY: k, yAsphere: a)
            case .biconic(let cx, _, let kx, let ky, let ax, let ay):
                s = .biconic(curvatureX: cx, curvatureY: c, conicX: kx, conicY: ky, xAsphere: ax, yAsphere: ay)
            default: return false
            }
            return true
        }
    }

    /// Sets the conic constant of the geometry, as the engine's
    /// `ParamPath::Conic` setter does for shapes that have one. As an editor
    /// convenience it also promotes a plane (or a diffractive plane base) to
    /// a conic of zero curvature, which the `conic` path does not:
    /// `OpticalSystem.setValue(_:at:)` refuses that, like the engine.
    @discardableResult
    public mutating func setConic(_ k: Double) -> Bool {
        editGeometry { s in
            switch s {
            case .plane: s = .conic(curvature: 0, conic: k)
            case .conic(let c, _): s = .conic(curvature: c, conic: k)
            case .evenAsphere(let c, _, let a): s = .evenAsphere(curvature: c, conic: k, coefficients: a)
            case .oddAsphere(let c, _, let a): s = .oddAsphere(curvature: c, conic: k, coefficients: a)
            case .qCon(let c, _, let r, let a): s = .qCon(curvature: c, conic: k, normRadius: r, coefficients: a)
            case .xyPolynomial(let c, _, let r, let a):
                s = .xyPolynomial(curvature: c, conic: k, normRadius: r, coefficients: a)
            case .zernikeSag(let c, _, let r, let o, let a):
                s = .zernikeSag(curvature: c, conic: k, normRadius: r, ordering: o, coefficients: a)
            default: return false
            }
            return true
        }
    }

    /// Sets even-asphere coefficient `index` of the geometry, extending the
    /// list with zeros (the engine's `ParamPath::Asphere` setter).
    @discardableResult
    public mutating func setAsphereCoefficient(_ v: Double, at index: Int) -> Bool {
        guard index >= 0 else { return false }
        return editGeometry { s in
            guard case .evenAsphere(let c, let k, var a) = s else { return false }
            if index >= a.count { a += Array(repeating: 0, count: index - a.count + 1) }
            a[index] = v
            s = .evenAsphere(curvature: c, conic: k, coefficients: a)
            return true
        }
    }
}

extension OpticalSystem {
    /// The primary wavelength in µm (the d line when the set is invalid).
    var primaryWavelengthUm: Double {
        let w = wavelengths.wavelengths
        return w.indices.contains(wavelengths.primary) ? w[wavelengths.primary].um : 0.5875618
    }

    /// Changes a surface's shape type, keeping what carries over: curvature
    /// and conic where the new type has them (a toroid or biconic gets
    /// `c_x = c_y = c`), the normalisation radius between normalised types,
    /// x/y values between toroid and biconic. A diffractive surface wraps the
    /// old shape as its base when it is a plane, conic or even asphere (else
    /// a conic with the old c and k), with an empty order +1 phase at the
    /// primary wavelength; converting a diffractive surface uses its base.
    /// A new normalisation radius is the fixed semi-diameter, else the
    /// aperture's bounding radius, else 10 mm. Overrides that no longer
    /// resolve are dropped.
    public mutating func setShapeKind(_ kind: SurfaceShape.Kind, at index: Int) {
        guard surfaces.indices.contains(index) else { return }
        let surface = surfaces[index]
        let old = surface.shape
        if old.kind != kind || kind == .coordinateBreak {
            let source = old.kind == .diffractive ? old.geometry : old
            let c = source.curvature, k = source.conicConstant
            let r: Double = {
                for v in [old.normRadius, surface.semiDiameter.fixedValue, surface.aperture?.boundingRadius] {
                    if let v, v.isFinite, v > 0 { return v }
                }
                return 10
            }()
            surfaces[index].shape =
                switch kind {
                case .plane: .plane
                case .conic: .conic(curvature: c, conic: k)
                case .evenAsphere: .evenAsphere(curvature: c, conic: k, coefficients: source.asphereCoefficients)
                case .oddAsphere: .oddAsphere(curvature: c, conic: k, coefficients: [])
                case .qCon: .qCon(curvature: c, conic: k, normRadius: r, coefficients: [])
                case .qBfs: .qBfs(curvature: c, normRadius: r, coefficients: [])
                case .diffractive:
                    .diffractive(
                        base: {
                            switch source {
                            case .plane, .conic, .evenAsphere: source
                            default: .conic(curvature: c, conic: k)
                            }
                        }(),
                        phase: DiffractivePhase(
                            normRadius: r, coefficients: [], designOrder: 1, designWavelengthUm: primaryWavelengthUm,
                            traceOrder: 1))
                case .toroid:
                    if case .biconic(let cx, let cy, _, let ky, _, let ya) = source {
                        .toroid(curvatureX: cx, curvatureY: cy, conicY: ky, yAsphere: ya)
                    } else {
                        .toroid(curvatureX: c, curvatureY: c, conicY: k, yAsphere: [])
                    }
                case .biconic:
                    if case .toroid(let cx, let cy, let ky, let ya) = source {
                        .biconic(curvatureX: cx, curvatureY: cy, conicX: 0, conicY: ky, xAsphere: [], yAsphere: ya)
                    } else {
                        .biconic(curvatureX: c, curvatureY: c, conicX: k, conicY: k, xAsphere: [], yAsphere: [])
                    }
                case .xyPolynomial: .xyPolynomial(curvature: c, conic: k, normRadius: r, coefficients: [])
                case .zernikeSag: .zernikeSag(curvature: c, conic: k, normRadius: r, ordering: .fringe, coefficients: [])
                case .paraxial:
                    if case .paraxial(let f) = old { .paraxial(focalLength: f) } else { .paraxial(focalLength: 100) }
                case .coordinateBreak: .coordinateBreak
                }
        }
        if kind == .coordinateBreak && surfaces[index].placement == nil {
            surfaces[index].placement = Placement()
        }
        // e.g. a conic override on a surface that became a plane.
        let id = surfaces[index].id, current = self
        for k in configurations.indices {
            configurations[k].overrides.removeAll { $0.first.surface == id && $0.first.unresolvedReason(in: current) != nil }
        }
    }
}

extension ParamPath {
    /// Whether the path names an existing parameter of `system`.
    public func resolves(in system: OpticalSystem) -> Bool { unresolvedReason(in: system) == nil }

    /// Why the path does not name a parameter of `system` (`nil` when it
    /// does). The same rules and messages as the engine's `ParamPath::get`,
    /// which its `validate` applies to every configuration override.
    public func unresolvedReason(in system: OpticalSystem) -> String? {
        if let id = surface {
            guard let i = system.surfaceIndex(of: id) else { return "no surface with id \(id.rawValue)" }
            let s = system.surfaces[i]
            switch self {
            case .curvature:
                return s.shape.hasCurvature ? nil : "surface has no curvature"
            case .conic:
                return s.shape.hasConic ? nil : "surface has no conic"
            case .asphere:
                return s.shape.hasAsphereCoefficients ? nil : "surface is not an even asphere"
            case .shapeCoefficient(_, let set, let index):
                guard let max = set.maxCount(for: s.shape) else { return "coefficient set does not belong to shape" }
                return index < max ? nil : "[coefficient_count] index exceeds shape limit"
            case .shapeParam(_, let p):
                return s.shape.shapeParameter(p) == nil ? "scalar does not belong to shape" : nil
            case .paraxialFocalLength:
                if case .paraxial = s.shape { return nil }
                return "surface is not paraxial"
            case .modelNd, .modelVd, .modelDpgf:
                if case .model = s.material { return nil }
                return "material is not a model glass"
            case .materialDnD, .materialDvD:
                if case .catalog = s.material { return nil }
                return "material is not a catalogue glass"
            case .semiDiameter:
                return s.semiDiameter == .auto ? "semi-diameter is automatic" : nil
            case .elementPlacement(_, let last, let pivot, _):
                guard let end = system.surfaceIndex(of: last), i > 0, i <= end, end + 1 < system.surfaces.count,
                    pivot.map({ system.surfaceIndex(of: $0) != nil }) ?? true
                else { return "expected an interior, ordered range and existing pivot" }
                if let e = s.elementPlacement, e.last != last || e.pivot != pivot {
                    return "another range or pivot starts here"
                }
                return nil
            default:
                return nil  // thickness, decenters, tilts and irregularity exist on every surface
            }
        }
        switch self {
        case .fieldX(let i), .fieldY(let i), .fieldWeight(let i):
            return system.fields.fields.indices.contains(i) ? nil : "no such field"
        case .wavelengthUm(let i), .wavelengthWeight(let i):
            return system.wavelengths.wavelengths.indices.contains(i) ? nil : "no such wavelength"
        case .apertureValue:
            return system.aperture.value == nil ? "aperture floats by stop size" : nil
        default:
            return nil
        }
    }

    /// Why the engine's `ParamPath::set` would refuse the finite value `v`
    /// for a path that resolves: a diffraction order must be an integral
    /// `i32`, a fixed semi-diameter positive. `nil` when it accepts it.
    public func valueRejection(_ v: Double) -> String? {
        switch self {
        case .shapeParam(_, let p): SurfaceShape.orderRejection(p, v)
        case .semiDiameter: v > 0 ? nil : "semi-diameter must be finite and > 0"
        default: nil
        }
    }
}
