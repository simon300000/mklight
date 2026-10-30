import Foundation
import MKLightModel

/// A user edit of the lens data, addressed by surface id where possible.
/// Applying it is pure model manipulation (no optics).
public enum LensEdit: Equatable, Sendable {
    case setCurvature(SurfaceID, Double)
    case setThickness(SurfaceID, Double)
    case setConic(SurfaceID, Double)
    case setSemiDiameter(SurfaceID, SemiDiameter)
    case setMaterial(SurfaceID, LensValueParser.MaterialEdit)
    case setLabel(SurfaceID, String)
    case setShapeKind(SurfaceID, SurfaceShape.Kind)
    /// A cylinder: a biconic with one plane axis.
    case setCylinder(SurfaceID, SurfaceShape.CylinderAxis)
    case setShapeParameter(SurfaceID, ShapeParameter, Double)
    case setCoating(SurfaceID, String?)
    case insert(at: Int)
    case delete(at: Int)
    case makeStop(at: Int)

    public var actionName: String {
        switch self {
        case .setCurvature: "Change Radius"
        case .setThickness: "Change Thickness"
        case .setConic: "Change Conic"
        case .setSemiDiameter: "Change Semi-Diameter"
        case .setMaterial: "Change Material"
        case .setLabel: "Change Label"
        case .setShapeKind, .setCylinder: "Change Surface Type"
        case .setShapeParameter(_, let p, _): "Change \(p.title)"
        case .setCoating: "Change Coating"
        case .insert: "Insert Surface"
        case .delete: "Delete Surface"
        case .makeStop: "Set Stop"
        }
    }

    public func apply(to s: inout OpticalSystem) {
        func at(_ id: SurfaceID, _ f: (inout Surface) -> Void) {
            if let i = s.surfaceIndex(of: id) { f(&s.surfaces[i]) }
        }
        switch self {
        case .setCurvature(let id, let c): at(id) { $0.shape.setCurvature(c) }
        case .setThickness(let id, let t): at(id) { $0.thickness = t }
        case .setConic(let id, let k): at(id) { $0.shape.setConic(k) }
        case .setSemiDiameter(let id, let sd): at(id) { $0.semiDiameter = sd }
        case .setMaterial(let id, let m):
            at(id) {
                $0.interaction = m.interaction
                if let mat = m.material { $0.material = mat }
            }
        case .setLabel(let id, let l): at(id) { $0.label = l }
        case .setShapeKind(let id, let k): if let i = s.surfaceIndex(of: id) { s.setShapeKind(k, at: i) }
        case .setCylinder(let id, let axis):
            if let i = s.surfaceIndex(of: id) {
                let c = s.surfaces[i].shape.curvature
                s.setShapeKind(.biconic, at: i)
                s.surfaces[i].shape = .cylinder(curvature: c, axis: axis)
            }
        case .setShapeParameter(let id, let p, let v): at(id) { $0.shape.setShapeParameter(p, v) }
        case .setCoating(let id, let c): at(id) { $0.coating = c }
        case .insert(let i): s.insertSurface(at: i)
        case .delete(let i): s.deleteSurface(at: i)
        case .makeStop(let i): s.setStop(i)
        }
    }
}
