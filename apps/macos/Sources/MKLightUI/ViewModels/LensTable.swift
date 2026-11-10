import Foundation
import MKLightEngine
import MKLightModel

/// Columns of the lens data table: the parameters every surface type has.
/// Type-specific parameters live in the surface inspector.
///
/// `radiusX` and `conicX` are the x sections of anamorphic surfaces
/// (toroid, biconic, cylinder): they have no column of their own but sit
/// beside the y section in the Radius and Conic cells, so both sections
/// show and edit in place.
public enum LensColumn: String, CaseIterable, Identifiable, Sendable {
    case surface, type, comment, radius, radiusX, thickness, material, semiDiameter, conic, conicX

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .surface: "Surf"
        case .type: "Type"
        case .comment: "Comment"
        case .radius: "Radius"
        case .radiusX: "Radius x"
        case .thickness: "Thickness"
        case .material: "Material"
        case .semiDiameter: "Semi-Diameter"
        case .conic: "Conic"
        case .conicX: "Conic x"
        }
    }

    public var isNumeric: Bool { [.radius, .radiusX, .thickness, .semiDiameter, .conic, .conicX].contains(self) }

    /// The table column a value shows in (an x section in its y section's).
    public var base: LensColumn {
        switch self {
        case .radiusX: .radius
        case .conicX: .conic
        default: self
        }
    }

    /// Columns a text field edits, in keyboard order.
    public static let editable: [LensColumn] = [.comment, .radius, .radiusX, .thickness, .material, .semiDiameter, .conic, .conicX]

    /// The model parameter behind a numeric column (for variables and solves).
    public func param(_ id: SurfaceID) -> ParamPath? {
        switch self {
        case .radius: .curvature(surface: id)
        case .thickness: .thickness(surface: id)
        case .conic: .conic(surface: id)
        case .semiDiameter: .semiDiameter(surface: id)
        default: nil
        }
    }

    /// The parameter behind the column for this surface's shape: radius and
    /// conic of a toroid or biconic are its y section's (the `curvature`
    /// path addresses the y section), `radiusX` and `conicX` its x
    /// section's (a toroid's x section has no conic).
    public func param(for surface: Surface) -> ParamPath? {
        guard surface.shape.isAnamorphic else { return self == .radiusX || self == .conicX ? nil : param(surface.id) }
        switch self {
        case .radiusX: return .shapeParam(surface: surface.id, parameter: .curvatureX)
        case .conic: return .shapeParam(surface: surface.id, parameter: .conicY)
        case .conicX: return surface.shape.kind == .biconic ? .shapeParam(surface: surface.id, parameter: .conicX) : nil
        default: return param(surface.id)
        }
    }
}

/// A small inline marker next to a value: variable, or controlled by a solve.
public struct CellMarker: Equatable, Sendable {
    public enum Bounds: Sendable, Equatable { case free, bounded, pinned }
    public var isVariable = false
    /// A variable's bounds: none, a range, or pinned (equal bounds).
    public var bounds = Bounds.free
    /// Short letter of the solve kind (`M` marginal, `C` chief, `F` F/#,
    /// `P` pickup, …) and its title.
    public var solve: (letter: String, title: String)?

    public static func == (a: CellMarker, b: CellMarker) -> Bool {
        a.isVariable == b.isVariable && a.bounds == b.bounds && a.solve?.letter == b.solve?.letter && a.solve?.title == b.solve?.title
    }
    public var isEmpty: Bool { !isVariable && solve == nil }
}

/// What one table row shows. Computed from the design, plus the engine's
/// automatic semi-diameters from the layout result.
public struct LensRow: Identifiable, Equatable, Sendable {
    public enum Role: Equatable, Sendable { case object, stop, image, surface }

    public var id: SurfaceID
    public var index: Int
    public var role: Role
    public var kind: SurfaceShape.Kind
    public var typeTitle: String
    public var isMirror: Bool
    public var cells: [LensColumn: String]
    /// Full-precision text an edit starts from.
    public var exact: [LensColumn: String]
    public var editable: Set<LensColumn>
    public var markers: [LensColumn: CellMarker]
    public var semiDiameterIsAuto: Bool
    /// A short axis hint before a value (`y` and `x` for the two sections
    /// of an anamorphic surface).
    public var hints: [LensColumn: String] = [:]
    /// A subtle note after a value: a catalogue glass's melt offset
    /// (`+0.0012/−0.8`).
    public var notes: [LensColumn: String] = [:]
    /// The section with power of a cylinder (its other section is flat,
    /// shown dimmed).
    public var poweredAxis: SurfaceShape.CylinderAxis?

    /// The values shown in a table column's cell: the y and x sections of
    /// an anamorphic surface (both, or a cylinder's powered conic alone),
    /// else the column itself.
    public func shown(_ column: LensColumn) -> [LensColumn] {
        let candidates: [LensColumn] =
            switch column.base {
            case .radius: [.radius, .radiusX]
            case .conic: [.conic, .conicX]
            default: [column]
            }
        let present = candidates.filter { cells[$0] != nil }
        return present.isEmpty ? [column.base] : present
    }

    /// Whether the row shows two sections in `column`'s cell.
    public func hasSections(_ column: LensColumn) -> Bool { shown(column).count > 1 }

    /// Whether `column` is the flat section of a cylinder (dimmed).
    public func isFlatSection(_ column: LensColumn) -> Bool {
        guard let a = poweredAxis else { return false }
        switch column {
        case .radius, .conic: return a == .x
        case .radiusX, .conicX: return a == .y
        default: return false
        }
    }

    public var surfaceText: String {
        switch role {
        case .object: "OBJ"
        case .stop: "STO"
        case .image: "IMA"
        case .surface: "\(index)"
        }
    }

    public func text(_ c: LensColumn) -> String { c == .surface ? surfaceText : c == .type ? typeTitle : cells[c] ?? "" }

    public static func rows(
        for s: OpticalSystem, autoSemiDiameters: [SurfaceID: Double] = [:], variables: Set<ParamPath> = [],
        bounds: [ParamPath: CellMarker.Bounds] = [:]
    ) -> [LensRow] {
        let solves = Dictionary(s.solves.map { (s.canonical($0.target), $0.kind.kind) }, uniquingKeysWith: { a, _ in a })
        return s.surfaces.enumerated().map { i, sf in
            let role: Role = i == 0 ? .object : (i == s.imageIndex ? .image : (i == s.stop ? .stop : .surface))
            var cells: [LensColumn: String] = [.comment: sf.label]
            var exact: [LensColumn: String] = [.comment: sf.label]
            var editable: Set<LensColumn> = [.comment]
            // An object at infinity has no meaningful curvature.
            let hasCurvature = sf.shape.hasCurvature && !(role == .object && sf.thickness.isInfinite)
            var hints: [LensColumn: String] = [:]
            if case .biconic(let cx, let cy, let kx, let ky, _, _) = sf.shape {
                // Both sections, y first (the meridional section).
                for (c, v) in [(LensColumn.radius, RadiusDisplay.radius(fromCurvature: cy)), (.radiusX, RadiusDisplay.radius(fromCurvature: cx))] {
                    cells[c] = RowFormat.compact(v)
                    exact[c] = NumberDisplay.exact(v)
                    editable.insert(c)
                }
                let axis = sf.shape.cylinderAxis
                // A cylinder's flat section has no meaningful conic: only the powered one shows.
                for (c, v, a) in [(LensColumn.conic, ky, SurfaceShape.CylinderAxis.y), (.conicX, kx, .x)] where axis == nil || axis == a {
                    cells[c] = RowFormat.plain(v)
                    exact[c] = NumberDisplay.exact(v)
                    editable.insert(c)
                    hints[c] = a.rawValue
                }
                hints[.radius] = "y"
                hints[.radiusX] = "x"
            } else if case .toroid(let cx, let cy, let ky, _) = sf.shape {
                for (c, v) in [(LensColumn.radius, RadiusDisplay.radius(fromCurvature: cy)), (.radiusX, RadiusDisplay.radius(fromCurvature: cx))] {
                    cells[c] = RowFormat.compact(v)
                    exact[c] = NumberDisplay.exact(v)
                    editable.insert(c)
                }
                cells[.conic] = RowFormat.plain(ky)
                exact[.conic] = NumberDisplay.exact(ky)
                editable.insert(.conic)
                hints[.radius] = "y"
                hints[.radiusX] = "x"
                hints[.conic] = "y"
            } else if hasCurvature {
                cells[.radius] = RowFormat.length(sf.shape.radius)
                exact[.radius] = NumberDisplay.exact(sf.shape.radius)
                editable.insert(.radius)
                if sf.shape.hasConic {
                    cells[.conic] = RowFormat.plain(sf.shape.conicConstant)
                    exact[.conic] = NumberDisplay.exact(sf.shape.conicConstant)
                    editable.insert(.conic)
                }
            } else if case .paraxial(let f) = sf.shape {
                cells[.radius] = "f " + RowFormat.length(f)
            }
            if role != .image {
                cells[.thickness] = RowFormat.length(sf.thickness)
                exact[.thickness] = NumberDisplay.exact(sf.thickness)
                editable.insert(.thickness)
            }
            let material: String =
                switch sf.interaction {
                case .reflect: "MIRROR"
                case .dummy: "DUMMY"
                case .refract: sf.isCoordinateBreak ? "" : (sf.material.isAir ? "" : sf.material.displayName)
                }
            cells[.material] = material
            exact[.material] = sf.material.isAir && sf.interaction == .refract ? "" : material
            if role != .image && !sf.isCoordinateBreak { editable.insert(.material) }
            switch sf.semiDiameter {
            case .fixed(let v):
                cells[.semiDiameter] = RowFormat.length(v, decimals: 3)
                exact[.semiDiameter] = NumberDisplay.exact(v)
            case .auto:
                cells[.semiDiameter] = autoSemiDiameters[sf.id].map { RowFormat.length($0, decimals: 3) } ?? "auto"
                exact[.semiDiameter] = "auto"
            }
            if !sf.isCoordinateBreak { editable.insert(.semiDiameter) }
            var markers: [LensColumn: CellMarker] = [:]
            for c in LensColumn.allCases {
                guard let p = c.param(for: sf).map(s.canonical) else { continue }
                var m = CellMarker(isVariable: variables.contains(p))
                if m.isVariable { m.bounds = bounds[p] ?? .free }
                if let k = solves[p] { m.solve = (k.letter, k.title) }
                if c == .radius, let k = solves[.paraxialFocalLength(surface: sf.id)] { m.solve = (k.letter, k.title) }
                if !m.isEmpty { markers[c] = m }
            }
            var notes: [LensColumn: String] = [:]
            if sf.interaction == .refract, let o = sf.material.catalogOffset, !o.isZero {
                notes[.material] = o.compactText
                // Offset variables and solves mark the glass.
                var m = CellMarker()
                for p in [ParamPath.materialDnD(surface: sf.id), .materialDvD(surface: sf.id)] {
                    if variables.contains(p) {
                        m.isVariable = true
                        m.bounds = bounds[p] ?? .free
                    }
                    if let k = solves[p] { m.solve = (k.letter, k.title) }
                }
                if !m.isEmpty { markers[.material] = m }
            }
            return LensRow(
                id: sf.id, index: i, role: role, kind: sf.shape.kind, typeTitle: Self.typeTitle(sf.shape),
                isMirror: sf.interaction == .reflect, cells: cells, exact: exact, editable: editable, markers: markers,
                semiDiameterIsAuto: sf.semiDiameter == .auto, hints: hints, notes: notes, poweredAxis: sf.shape.cylinderAxis)
        }
    }

    /// Short type name for the table: a conic of k = 0 is a sphere.
    static func typeTitle(_ shape: SurfaceShape) -> String {
        switch shape {
        case .conic(_, let k): return k == 0 ? "Sphere" : "Conic"
        case .qCon: return "Q-Con"
        case .qBfs: return "Q-BFS"
        case .biconic where shape.isCylinder: return "Cylinder"
        default: return SurfaceTypes.descriptor(for: shape).title
        }
    }

    /// Parses `text` typed into `column` into an edit of this row.
    public func edit(_ column: LensColumn, text: String) -> Result<LensEdit, LensValueParser.Failure>? {
        switch column {
        case .comment: return .success(.setLabel(id, text))
        case .radiusX where kind == .toroid || kind == .biconic:
            return LensValueParser.curvature(fromRadiusText: text).map { .setShapeParameter(id, .curvatureX, $0) }
        case .conic where kind == .toroid || kind == .biconic:
            return LensValueParser.finite(text).map { .setShapeParameter(id, .conicY, $0) }
        case .conicX where kind == .biconic:
            return LensValueParser.finite(text).map { .setShapeParameter(id, .conicX, $0) }
        case .radiusX, .conicX: return nil
        case .radius: return LensValueParser.curvature(fromRadiusText: text).map { .setCurvature(id, $0) }
        case .thickness: return LensValueParser.thickness(text, allowInfinite: role == .object).map { .setThickness(id, $0) }
        case .conic: return LensValueParser.finite(text).map { .setConic(id, $0) }
        case .semiDiameter: return LensValueParser.semiDiameter(text).map { .setSemiDiameter(id, $0) }
        case .material: return LensValueParser.material(text).map { .setMaterial(id, $0) }
        case .surface, .type: return nil
        }
    }
}

extension SolveKind.Kind {
    /// One-letter marker shown next to a solved value.
    public var letter: String {
        switch self {
        case .marginalRayHeight, .marginalRayAngle: "M"
        case .chiefRayHeight, .chiefRayAngle: "C"
        case .fNumber: "F"
        case .pickup: "P"
        case .position: "D"
        case .centerOfCurvature: "O"
        case .edgeThickness: "E"
        }
    }
}

/// Number formats of the lens table.
enum RowFormat {
    static func length(_ v: Double, decimals: Int = 4) -> String {
        if v.isNaN { return "—" }
        if v.isInfinite { return v > 0 ? "∞" : "−∞" }
        let a = abs(v)
        if a != 0 && (a < 1e-4 || a >= 1e7) { return String(format: "%.4e", v) }
        return String(format: "%.\(decimals)f", v)
    }

    /// Up to 6 significant digits without trailing zeros, for the two
    /// sections that share an anamorphic cell (the exact value edits).
    static func compact(_ v: Double) -> String {
        if v.isNaN { return "—" }
        if v.isInfinite { return v > 0 ? "∞" : "−∞" }
        return plain(v)
    }

    static func plain(_ v: Double) -> String {
        if v == 0 { return "0" }
        let a = abs(v)
        if a < 1e-3 || a >= 1e6 { return String(format: "%.4e", v) }
        return String(format: "%.6g", v)
    }
}

/// Edits of the lens table, applied to the design as undoable steps.
@MainActor
public struct LensEditor {
    public let document: DesignDocument
    public let undoManager: UndoManager?

    public init(document: DesignDocument, undoManager: UndoManager?) {
        self.document = document
        self.undoManager = undoManager
    }

    /// Commits typed text. Returns an error message when the text was not
    /// accepted (the model is then unchanged).
    @discardableResult
    public func commit(_ row: LensRow, _ column: LensColumn, text: String) -> String? {
        guard row.exact[column] != text else { return nil }
        guard let parsed = row.edit(column, text: text) else { return nil }
        switch parsed {
        case .success(let e):
            apply(e)
            return nil
        case .failure(let f):
            return f.description
        }
    }

    @discardableResult
    public func apply(_ e: LensEdit) -> Bool {
        document.edit(e.actionName, undoManager: undoManager) { e.apply(to: &$0) }
    }

    /// Inserts a surface before `index` (1 … image); returns its id.
    @discardableResult
    public func insert(at index: Int) -> SurfaceID? {
        let i = min(max(index, 1), document.system.imageIndex)
        guard apply(.insert(at: i)) else { return nil }
        return document.system.surfaces[i].id
    }

    /// Deletes the surface `id`; returns the id to select next.
    @discardableResult
    public func delete(_ id: SurfaceID) -> SurfaceID? {
        guard let i = document.system.surfaceIndex(of: id), apply(.delete(at: i)) else { return nil }
        let s = document.system
        return s.surfaces[min(i, s.imageIndex)].id
    }

    public func makeStop(_ id: SurfaceID) {
        if let i = document.system.surfaceIndex(of: id) { apply(.makeStop(at: i)) }
    }

    /// Applies a type-menu choice (a type, or a preset such as a cylinder).
    public func setType(_ choice: SurfaceTypeChoice, _ id: SurfaceID) {
        switch choice {
        case .kind(let k): apply(.setShapeKind(id, k))
        case .cylinder(let axis): apply(.setCylinder(id, axis))
        }
    }

    public func toggleVariable(_ path: ParamPath) {
        document.setVariable(path, !document.isVariable(path), undoManager: undoManager)
    }
}
