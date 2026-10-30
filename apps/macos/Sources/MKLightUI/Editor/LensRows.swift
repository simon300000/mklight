import Foundation
import MKLightEngine
import MKLightModel

/// Columns of the lens data editor.
enum LensColumn: String, CaseIterable {
    case index, type, label, radius, thickness, material, semiDiameter, conic, coating

    var title: String {
        switch self {
        case .index: "Surf"
        case .type: "Type"
        case .label: "Comment"
        case .radius: "Radius"
        case .thickness: "Thickness"
        case .material: "Material"
        case .semiDiameter: "Semi-Diameter"
        case .conic: "Conic"
        case .coating: "Coating"
        }
    }

    var width: CGFloat {
        switch self {
        case .index: 46
        case .type: 112
        case .label: 100
        case .radius, .thickness: 104
        case .material: 104
        case .semiDiameter: 118
        case .conic: 84
        case .coating: 96
        }
    }

    var isNumeric: Bool { [.radius, .thickness, .semiDiameter, .conic].contains(self) }
    var isPopup: Bool { self == .type || self == .coating }
}

/// Display values of one surface row. Computed from the model (and, for
/// automatic semi-diameters, from the engine's layout result).
struct LensRow: Equatable {
    enum Role: Equatable { case object, stop, image, surface }

    var id: SurfaceID
    var index: Int
    var role: Role
    var shapeKind: SurfaceShape.Kind
    var label: String
    var radius: String
    var radiusExact: String?
    var thickness: String
    var thicknessExact: String
    var material: String
    var semiDiameter: String
    var semiDiameterExact: String
    var semiDiameterIsAuto: Bool
    var conic: String
    var conicExact: String?
    var coating: String?

    var indexText: String {
        switch role {
        case .object: "OBJ"
        case .stop: "STO"
        case .image: "IMA"
        case .surface: "\(index)"
        }
    }

    /// Text shown in `column`, and the full-precision text used while editing.
    func text(_ c: LensColumn) -> (display: String, exact: String?) {
        switch c {
        case .index: (indexText, nil)
        case .type: (shapeKind.title, nil)
        case .label: (label, label)
        case .radius: (radius, radiusExact)
        case .thickness: (thickness, thicknessExact)
        case .material: (material, role == .image ? nil : material)
        case .semiDiameter: (semiDiameter, semiDiameterExact)
        case .conic: (conic, conicExact)
        case .coating: (coating ?? "", nil)
        }
    }

    func isEditable(_ c: LensColumn) -> Bool {
        switch c {
        case .index: false
        case .type: role != .object && role != .image
        case .label: true
        case .radius: radiusExact != nil
        case .thickness: role != .image
        case .material: role != .image
        case .semiDiameter: true
        case .conic: conicExact != nil
        case .coating: role != .object && role != .image
        }
    }

    static func rows(for s: OpticalSystem, layout: LayoutScene?) -> [LensRow] {
        let autoSD = Dictionary(
            (layout?.surfaces ?? []).map { ($0.id, $0.semiDiameter) }, uniquingKeysWith: { a, _ in a })
        return s.surfaces.enumerated().map { i, sf in
            let role: Role = i == 0 ? .object : (i == s.imageIndex ? .image : (i == s.stop ? .stop : .surface))
            let hasCurv = sf.shape.hasCurvature
            let radius = sf.shape.radius
            let material: String =
                switch sf.interaction {
                case .reflect: "MIRROR"
                case .dummy: "DUMMY"
                case .refract: sf.isCoordinateBreak ? "" : sf.material.displayName
                }
            let sdText: String, sdExact: String
            switch sf.semiDiameter {
            case .fixed(let v):
                sdText = NumberDisplay.format(v)
                sdExact = NumberDisplay.exact(v)
            case .auto:
                sdText = autoSD[sf.id].map { NumberDisplay.format($0) + " A" } ?? "auto"
                sdExact = "auto"
            }
            return LensRow(
                id: sf.id, index: i, role: role, shapeKind: sf.shape.kind, label: sf.label,
                radius: hasCurv ? NumberDisplay.format(radius) : (sf.isCoordinateBreak ? "" : "—"),
                radiusExact: hasCurv ? NumberDisplay.exact(radius) : nil,
                thickness: NumberDisplay.format(sf.thickness), thicknessExact: NumberDisplay.exact(sf.thickness),
                material: material, semiDiameter: sdText, semiDiameterExact: sdExact,
                semiDiameterIsAuto: sf.semiDiameter == .auto,
                conic: hasCurv ? NumberDisplay.format(sf.shape.conicConstant, decimals: 6) : "",
                conicExact: hasCurv ? NumberDisplay.exact(sf.shape.conicConstant) : nil,
                coating: sf.coating)
        }
    }

    /// Parses `text` typed into `column` into an edit of this row.
    func edit(_ column: LensColumn, text: String) -> Result<LensEdit, LensValueParser.Failure>? {
        switch column {
        case .label: return .success(.setLabel(id, text))
        case .radius: return LensValueParser.curvature(fromRadiusText: text).map { .setCurvature(id, $0) }
        case .thickness: return LensValueParser.thickness(text, allowInfinite: role == .object).map { .setThickness(id, $0) }
        case .conic: return LensValueParser.finite(text).map { .setConic(id, $0) }
        case .semiDiameter: return LensValueParser.semiDiameter(text).map { .setSemiDiameter(id, $0) }
        case .material: return LensValueParser.material(text).map { .setMaterial(id, $0) }
        case .index, .type, .coating: return nil
        }
    }
}
