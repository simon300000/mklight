import MKLightEngine
import MKLightModel
import SwiftUI

/// Solves on the selected surface's thickness and curvature (docs/SOLVES.md):
/// pick a rule and its value; pickups and other solves on this surface are
/// listed with a remove button. Angle solves take the paraxial slope, labelled
/// "paraxial angle u" and passed through unconverted (CONVENTIONS §8). The
/// engine applies solves; the stored target value is only its start point.
struct SolvesSection: View {
    let system: OpticalSystem
    let index: Int
    let edit: ModelEditor

    private var id: SurfaceID { system.surfaces[index].id }

    static let thicknessKinds: [SolveKind.Kind] = [.marginalRayHeight, .chiefRayHeight, .edgeThickness]
    static let curvatureKinds: [SolveKind.Kind] = [.marginalRayAngle, .chiefRayAngle, .fNumber]

    var body: some View {
        Section("Solves") {
            solveRow("Thickness", target: .thickness(surface: id), kinds: Self.thicknessKinds)
            if system.surfaces[index].shape.hasCurvature {
                solveRow("Curvature", target: .curvature(surface: id), kinds: Self.curvatureKinds)
            }
            ForEach(others, id: \.offset) { item in
                HStack {
                    Text(describe(item.element)).font(.caption).lineLimit(2)
                    Spacer()
                    Button("Remove", systemImage: "minus.circle") {
                        let s = item.element
                        edit("Remove Solve") { $0.solves.removeAll { $0 == s } }
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                }
            }
        }
    }

    /// Solves that refer to this surface but are not edited by the rows.
    private var others: [EnumeratedSequence<[Solve]>.Element] {
        let rowTargets: Set<ParamPath> = [.thickness(surface: id), .curvature(surface: id)]
        return Array(system.solves.filter { s in
            s.surfaces.contains(id) && !(rowTargets.contains(s.target) && Self.editable(s.kind.kind))
        }.enumerated())
    }

    private static func editable(_ k: SolveKind.Kind) -> Bool {
        (thicknessKinds + curvatureKinds).contains(k)
    }

    @ViewBuilder private func solveRow(_ title: String, target: ParamPath, kinds: [SolveKind.Kind]) -> some View {
        let current = system.solve(for: target).flatMap { Self.editable($0.kind.kind) ? $0 : nil }
        Picker(title, selection: Binding(
            get: { current?.kind.kind.rawValue ?? "" },
            set: { raw in
                edit(raw.isEmpty ? "Remove Solve" : "Set Solve") { sys in
                    sys.solves.removeAll { $0.target == target }
                    if let k = SolveKind.Kind(rawValue: raw) {
                        sys.solves.append(Solve(target: target, kind: Self.defaultKind(k)))
                    }
                }
            })
        ) {
            Text("None").tag("")
            ForEach(kinds) { Text($0.title).tag($0.rawValue) }
        }
        if let current, let label = current.kind.kind.valueLabel, let value = current.kind.value {
            NumericField(title: label.label, value: value, unit: label.unit) { v in
                edit("Change Solve") { sys in
                    guard let i = sys.solves.firstIndex(where: { $0.target == target }) else { return }
                    sys.solves[i].kind.value = v
                }
            }
            .help(current.kind.kind == .marginalRayAngle || current.kind.kind == .chiefRayAngle
                ? "Paraxial slope u = dy/dz after the surface (dimensionless, not degrees)" : "")
        }
    }

    static func defaultKind(_ k: SolveKind.Kind) -> SolveKind {
        switch k {
        case .marginalRayHeight: .marginalRayHeight(height: 0)
        case .chiefRayHeight: .chiefRayHeight(height: 0)
        case .marginalRayAngle: .marginalRayAngle(slope: 0)
        case .chiefRayAngle: .chiefRayAngle(slope: 0)
        case .fNumber: .fNumber(value: 4)
        case .edgeThickness: .edgeThickness(value: 1, radius: nil)
        case .pickup: .pickup(source: .thickness(surface: SurfaceID(0)), scale: 1, offset: 0)
        case .position: .position(from: SurfaceID(0), to: SurfaceID(0), distance: 0)
        case .centerOfCurvature: .centerOfCurvature(surface: SurfaceID(0))
        }
    }

    private func describe(_ s: Solve) -> String {
        let target = "\(s.target)"
        switch s.kind {
        case .pickup(let source, let scale, let offset):
            return "\(target) = \(NumberDisplay.format(scale, decimals: 4)) × \(source) + \(NumberDisplay.format(offset, decimals: 4))"
        case .position(let from, let to, let d):
            return "\(target): distance \(from) → \(to) = \(NumberDisplay.format(d)) mm"
        case .centerOfCurvature(let surface):
            return "\(target): centre of curvature on \(surface)"
        default:
            let v = s.kind.value.map { NumberDisplay.format($0) } ?? ""
            return "\(target): \(s.kind.kind.title) \(s.kind.kind.valueLabel?.label ?? "") \(v)"
        }
    }
}
