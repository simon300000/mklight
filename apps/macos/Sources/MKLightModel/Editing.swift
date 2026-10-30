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
    /// keeps the stop at the same position (clamped).
    @discardableResult
    public mutating func deleteSurface(at index: Int) -> Bool {
        guard index > 0, index < imageIndex, surfaces.count > 3 else { return false }
        let removed = surfaces.remove(at: index)
        if index < stop { stop -= 1 }
        stop = min(max(stop, 1), imageIndex - 1)
        // Configuration overrides of the removed surface no longer resolve.
        for k in configurations.indices {
            configurations[k].overrides.removeAll { $0.first.surface == removed.id }
        }
        return true
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
    /// Whether the path names an existing parameter of `system` (the shape
    /// and material checks are left to the engine).
    public func resolves(in system: OpticalSystem) -> Bool {
        if let s = surface { return system.surfaceIndex(of: s) != nil }
        switch self {
        case .fieldX(let i), .fieldY(let i), .fieldWeight(let i): return system.fields.fields.indices.contains(i)
        case .wavelengthUm(let i), .wavelengthWeight(let i): return system.wavelengths.wavelengths.indices.contains(i)
        case .apertureValue: return system.aperture.value != nil
        default: return true
        }
    }
}
