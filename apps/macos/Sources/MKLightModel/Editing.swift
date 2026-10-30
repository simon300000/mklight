import Foundation

/// Structural edits of the surface list. They keep the stop index and surface
/// identities consistent; they never compute optics.
extension OpticalSystem {
    /// Inserts a plane air surface at `index` (1…image index) and returns its
    /// id. Surfaces at and after `index` move down by one.
    @discardableResult
    public mutating func insertSurface(at index: Int, thickness: Double = 0) -> SurfaceID {
        let i = min(max(index, 1), imageIndex)
        let id = nextSurfaceID
        surfaces.insert(Surface(id: id, shape: .plane, thickness: thickness, material: .air), at: i)
        if i <= stop { stop += 1 }
        return id
    }

    /// Removes the surface at `index`. The object, the image and the last
    /// remaining surface between them cannot be removed. Deleting the stop
    /// keeps the stop at the same position (clamped). Configuration overrides
    /// and mechanical parts that refer to the removed surface are removed too.
    @discardableResult
    public mutating func deleteSurface(at index: Int) -> Bool {
        guard index > 0, index < imageIndex, surfaces.count > 3 else { return false }
        let removed = surfaces.remove(at: index)
        if index < stop { stop -= 1 }
        stop = min(max(stop, 1), imageIndex - 1)
        for k in configurations.indices {
            configurations[k].overrides.removeAll { $0.first.surface == removed.id }
        }
        mechanics.removeAll { $0.surfaces.contains(removed.id) }
        return true
    }

    /// Removes field `index` (at least one field remains). Overrides of that
    /// field are dropped and those of later fields renumbered, so every
    /// index-addressed `ParamPath` keeps naming the same field.
    @discardableResult
    public mutating func removeField(at index: Int) -> Bool {
        guard fields.fields.indices.contains(index), fields.fields.count > 1 else { return false }
        fields.fields.remove(at: index)
        remapIndexedOverrides(removed: index) { $0.fieldIndex }
        return true
    }

    /// Removes wavelength `index` (at least one remains), keeping the primary
    /// wavelength and index-addressed overrides pointing at the same entries.
    @discardableResult
    public mutating func removeWavelength(at index: Int) -> Bool {
        guard wavelengths.wavelengths.indices.contains(index), wavelengths.wavelengths.count > 1 else { return false }
        wavelengths.wavelengths.remove(at: index)
        if wavelengths.primary > index || (wavelengths.primary == index && index == wavelengths.wavelengths.count) {
            wavelengths.primary -= 1
        }
        remapIndexedOverrides(removed: index) { $0.wavelengthIndex }
        return true
    }

    private mutating func remapIndexedOverrides(removed: Int, index: (ParamPath) -> Int?) {
        for k in configurations.indices {
            configurations[k].overrides = configurations[k].overrides.compactMap { o in
                guard let i = index(o.first) else { return o }
                if i == removed { return nil }
                return i > removed ? Pair(o.first.withIndex(i - 1), o.second) : o
            }
        }
    }

    /// Drops configuration overrides and mechanical parts that resolved in
    /// `previous` but no longer resolve here. Call after an edit (shape,
    /// material, semi-diameter or aperture changes can invalidate them); the
    /// edit stays undoable as a whole. References that were already broken in
    /// `previous` are kept, so `structuralIssues()` still reports them.
    public mutating func dropReferencesInvalidated(since previous: OpticalSystem) {
        let current = self
        for k in configurations.indices {
            configurations[k].overrides.removeAll { o in
                o.first.unresolvedReason(in: current) != nil && o.first.unresolvedReason(in: previous) == nil
            }
        }
        let ids = Set(surfaces.map(\.id)), before = Set(previous.surfaces.map(\.id))
        mechanics.removeAll { m in
            m.surfaces.contains { !ids.contains($0) } && m.surfaces.allSatisfy(before.contains)
        }
    }

    /// Makes `index` the aperture stop if it is between object and image.
    public mutating func setStop(_ index: Int) {
        guard index > 0, index < imageIndex else { return }
        stop = index
    }

    /// Changes a surface's shape type, keeping curvature and conic when the
    /// new type has them.
    public mutating func setShapeKind(_ kind: SurfaceShape.Kind, at index: Int) {
        guard surfaces.indices.contains(index) else { return }
        let old = surfaces[index].shape
        let c = old.curvature, k = old.conicConstant
        surfaces[index].shape =
            switch kind {
            case .plane: .plane
            case .conic: .conic(curvature: c, conic: k)
            case .evenAsphere: .evenAsphere(curvature: c, conic: k, coefficients: old.asphereCoefficients)
            case .paraxial:
                if case .paraxial(let f) = old { .paraxial(focalLength: f) } else { .paraxial(focalLength: 100) }
            case .coordinateBreak: .coordinateBreak
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

extension SurfaceShape {
    public enum Kind: String, CaseIterable, Sendable, Identifiable {
        case plane, conic, evenAsphere, paraxial, coordinateBreak
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .plane: "Plane"
            case .conic: "Standard"
            case .evenAsphere: "Even Asphere"
            case .paraxial: "Paraxial"
            case .coordinateBreak: "Coord Break"
            }
        }
    }

    public var kind: Kind {
        switch self {
        case .plane: .plane
        case .conic: .conic
        case .evenAsphere: .evenAsphere
        case .paraxial: .paraxial
        case .coordinateBreak: .coordinateBreak
        }
    }

    public var curvature: Double {
        switch self {
        case .conic(let c, _), .evenAsphere(let c, _, _): c
        default: 0
        }
    }

    public var conicConstant: Double {
        switch self {
        case .conic(_, let k), .evenAsphere(_, let k, _): k
        default: 0
        }
    }

    public var asphereCoefficients: [Double] {
        if case .evenAsphere(_, _, let a) = self { return a }
        return []
    }

    /// Whether the shape has a curvature that can be edited.
    public var hasCurvature: Bool {
        switch self {
        case .plane, .conic, .evenAsphere: true
        default: false
        }
    }

    /// Sets the curvature, promoting a plane to a sphere (like the engine's
    /// `ParamPath::Curvature` setter). Returns false for shapes without one.
    @discardableResult
    public mutating func setCurvature(_ c: Double) -> Bool {
        switch self {
        case .plane: self = .conic(curvature: c, conic: 0)
        case .conic(_, let k): self = .conic(curvature: c, conic: k)
        case .evenAsphere(_, let k, let a): self = .evenAsphere(curvature: c, conic: k, coefficients: a)
        default: return false
        }
        return true
    }

    @discardableResult
    public mutating func setConic(_ k: Double) -> Bool {
        switch self {
        case .plane: self = .conic(curvature: 0, conic: k)
        case .conic(let c, _): self = .conic(curvature: c, conic: k)
        case .evenAsphere(let c, _, let a): self = .evenAsphere(curvature: c, conic: k, coefficients: a)
        default: return false
        }
        return true
    }
}

extension ParamPath {
    /// Whether the path names an existing parameter of `system`.
    public func resolves(in system: OpticalSystem) -> Bool { unresolvedReason(in: system) == nil }

    /// Why the path does not name a parameter of `system` (`nil` when it
    /// does). The same rules as the engine's `ParamPath::get`, which its
    /// `validate` applies to every configuration override.
    public func unresolvedReason(in system: OpticalSystem) -> String? {
        if let id = surface {
            guard let i = system.surfaceIndex(of: id) else { return "no surface with id \(id.rawValue)" }
            let s = system.surfaces[i]
            switch self {
            case .curvature:
                return s.shape.hasCurvature ? nil : "surface has no curvature"
            case .conic:
                switch s.shape {
                case .conic, .evenAsphere: return nil
                default: return "surface has no conic"
                }
            case .asphere:
                if case .evenAsphere = s.shape { return nil }
                return "surface is not an even asphere"
            case .paraxialFocalLength:
                if case .paraxial = s.shape { return nil }
                return "surface is not paraxial"
            case .modelNd, .modelVd, .modelDpgf:
                if case .model = s.material { return nil }
                return "material is not a model glass"
            case .semiDiameter:
                return s.semiDiameter == .auto ? "semi-diameter is automatic" : nil
            default:
                return nil  // thickness, decenters and tilts exist on every surface
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

    /// Field index of a field-addressed path.
    public var fieldIndex: Int? {
        switch self {
        case .fieldX(let i), .fieldY(let i), .fieldWeight(let i): i
        default: nil
        }
    }

    /// Wavelength index of a wavelength-addressed path.
    public var wavelengthIndex: Int? {
        switch self {
        case .wavelengthUm(let i), .wavelengthWeight(let i): i
        default: nil
        }
    }

    /// The same parameter for another field or wavelength index.
    func withIndex(_ i: Int) -> ParamPath {
        switch self {
        case .fieldX: .fieldX(index: i)
        case .fieldY: .fieldY(index: i)
        case .fieldWeight: .fieldWeight(index: i)
        case .wavelengthUm: .wavelengthUm(index: i)
        case .wavelengthWeight: .wavelengthWeight(index: i)
        default: self
        }
    }
}

extension MechanicalPart {
    /// Surfaces the part is attached to.
    public var surfaces: [SurfaceID] {
        switch self {
        case .barrel(let from, let to, _, _): [from, to]
        case .annularBaffle(let s, _, _, _, _): [s]
        }
    }
}
