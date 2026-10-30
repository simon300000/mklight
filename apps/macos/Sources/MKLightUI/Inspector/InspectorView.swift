import MKLightModel
import SwiftUI

/// Applies a named, undoable change to the model.
typealias ModelEditor = (_ actionName: String, _ change: (inout OpticalSystem) -> Void) -> Void

/// System settings (aperture, fields, wavelengths, environment, ray aiming)
/// and the selected surface's full parameters.
struct InspectorView: View {
    let system: OpticalSystem
    @Bindable var viewState: WindowViewState
    let edit: ModelEditor

    var body: some View {
        VStack(spacing: 0) {
            Picker("Inspector", selection: $viewState.inspectorTab) {
                ForEach(InspectorTab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(8)
            Divider()
            switch viewState.inspectorTab {
            case .system: SystemInspector(system: system, edit: edit)
            case .surface:
                if let id = viewState.selection, let i = system.surfaceIndex(of: id) {
                    SurfaceInspector(system: system, index: i, edit: edit)
                } else {
                    ContentUnavailableView("No Surface Selected", systemImage: "square.dashed", description: Text("Select a row in the lens data editor."))
                }
            }
        }
    }
}

// MARK: - System

struct SystemInspector: View {
    let system: OpticalSystem
    let edit: ModelEditor

    var body: some View {
        Form {
            Section("System") {
                TextField("Name", text: Binding(get: { system.name }, set: { v in edit("Rename System") { $0.name = v } }))
                TextField("Notes", text: Binding(get: { system.notes }, set: { v in edit("Edit Notes") { $0.notes = v } }), axis: .vertical)
                    .lineLimit(1...4)
            }
            apertureSection
            fieldsSection
            wavelengthsSection
            Section("Environment") {
                NumericField(title: "Temperature", value: system.environment.temperatureC, unit: "°C") { v in
                    edit("Change Temperature") { $0.environment.temperatureC = v }
                }
                NumericField(title: "Pressure", value: system.environment.pressureAtm, unit: "atm", parse: LensValueParser.positive) { v in
                    edit("Change Pressure") { $0.environment.pressureAtm = v }
                }
            }
            Section("Ray Aiming") {
                Picker("Mode", selection: Binding(get: { system.rayAiming == .off ? 0 : 1 }, set: { v in
                    edit("Change Ray Aiming") { $0.rayAiming = v == 0 ? .off : .real(tolerance: RayAiming.defaultTolerance) }
                })) {
                    Text("Off (paraxial pupil)").tag(0)
                    Text("Real").tag(1)
                }
                if case .real(let tol) = system.rayAiming {
                    NumericField(title: "Tolerance", value: tol, parse: LensValueParser.positive) { v in
                        edit("Change Aiming Tolerance") { $0.rayAiming = .real(tolerance: v) }
                    }
                }
            }
            Section("Coatings") {
                if system.coatings.isEmpty { Text("None").foregroundStyle(.secondary) }
                ForEach(system.coatings.keys.sorted(), id: \.self) { name in
                    LabeledContent(name, value: system.coatings[name]?.kindName ?? "")
                }
                Button("Add Ideal AR Coating (R = 0.5 %)") {
                    edit("Add Coating") { s in
                        var name = "AR", k = 2
                        while s.coatings[name] != nil {
                            name = "AR\(k)"
                            k += 1
                        }
                        s.coatings[name] = .ideal(reflectance: 0.005, transmittance: 0.995)
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private var apertureSection: some View {
        Section("Aperture") {
            Picker("Type", selection: Binding(get: { system.aperture.kind }, set: { k in
                edit("Change Aperture Type") { $0.aperture = ApertureSpec(kind: k, value: $0.aperture.value ?? 10) }
            })) {
                ForEach(ApertureSpec.Kind.allCases) { Text($0.title).tag($0) }
            }
            if let v = system.aperture.value {
                NumericField(title: "Value", value: v, unit: system.aperture.kind.unit, parse: LensValueParser.positive) { nv in
                    edit("Change Aperture") { $0.aperture = ApertureSpec(kind: $0.aperture.kind, value: nv) }
                }
            }
        }
    }

    private var fieldsSection: some View {
        Section {
            Picker("Type", selection: Binding(get: { system.fields.kind }, set: { k in edit("Change Field Type") { $0.fields.kind = k } })) {
                ForEach(FieldKind.allCases) { Text($0.title).tag($0) }
            }
            Grid(alignment: .trailing, horizontalSpacing: 6, verticalSpacing: 4) {
                GridRow {
                    Text("#")
                    Text("X (\(system.fields.kind.unit))")
                    Text("Y (\(system.fields.kind.unit))")
                    Text("Weight")
                    Text("")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(system.fields.fields.indices, id: \.self) { i in
                    let f = system.fields.fields[i]
                    GridRow {
                        Circle().fill(Palette.fieldColor(i)).frame(width: 8, height: 8)
                        CompactNumber(value: f.x) { v in edit("Change Field") { $0.fields.fields[i].x = v } }
                        CompactNumber(value: f.y) { v in edit("Change Field") { $0.fields.fields[i].y = v } }
                        CompactNumber(value: f.weight, parse: LensValueParser.finite) { v in
                            edit("Change Field Weight") { $0.fields.fields[i].weight = max(v, 0) }
                        }
                        Button {
                            edit("Remove Field") { $0.removeField(at: i) }
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .disabled(system.fields.fields.count <= 1)
                    }
                }
            }
            Button("Add Field") {
                edit("Add Field") { s in
                    let last = s.fields.fields.last ?? Field()
                    s.fields.fields.append(Field(x: last.x, y: last.y + (last.y == 0 ? 1 : last.y / 2)))
                }
            }
        } header: {
            Text("Fields")
        }
    }

    private var wavelengthsSection: some View {
        Section {
            Grid(alignment: .trailing, horizontalSpacing: 6, verticalSpacing: 4) {
                GridRow {
                    Text("Primary")
                    Text("λ (µm)")
                    Text("Weight")
                    Text("")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(system.wavelengths.wavelengths.indices, id: \.self) { i in
                    let w = system.wavelengths.wavelengths[i]
                    GridRow {
                        Button {
                            edit("Set Primary Wavelength") { $0.wavelengths.primary = i }
                        } label: {
                            Image(systemName: system.wavelengths.primary == i ? "largecircle.fill.circle" : "circle")
                                .foregroundStyle(Palette.wavelengthColor(w.um))
                        }
                        .buttonStyle(.borderless)
                        CompactNumber(value: w.um, parse: LensValueParser.positive) { v in edit("Change Wavelength") { $0.wavelengths.wavelengths[i].um = v } }
                        CompactNumber(value: w.weight) { v in edit("Change Wavelength Weight") { $0.wavelengths.wavelengths[i].weight = max(v, 0) } }
                        Button {
                            edit("Remove Wavelength") { $0.removeWavelength(at: i) }
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .disabled(system.wavelengths.wavelengths.count <= 1)
                    }
                }
            }
            HStack {
                Button("Add Wavelength") { edit("Add Wavelength") { $0.wavelengths.wavelengths.append(Wavelength(um: SpectralLine.e.um)) } }
                Menu("Presets") {
                    Button("F, d, C (visible)") { preset([.F, .d, .C], primary: 1) }
                    Button("g, e, C") { preset([.g, .e, .C], primary: 1) }
                    Button("d only") { preset([.d], primary: 0) }
                }
                .fixedSize()
            }
        } header: {
            Text("Wavelengths")
        }
    }

    private func preset(_ lines: [SpectralLine], primary: Int) {
        edit("Wavelength Preset") { $0.wavelengths = WavelengthSet(wavelengths: lines.map { Wavelength(um: $0.um) }, primary: primary) }
    }
}

/// A compact numeric cell for grids (no label).
struct CompactNumber: View {
    let value: Double
    var parse: (String) -> Result<Double, LensValueParser.Failure> = LensValueParser.finite
    let commit: (Double) -> Void

    var body: some View {
        NumericField(title: "", value: value, parse: parse, commit: commit)
            .labelsHidden()
            .frame(minWidth: 56, maxWidth: 90)
    }
}

// MARK: - Surface

struct SurfaceInspector: View {
    let system: OpticalSystem
    let index: Int
    let edit: ModelEditor

    private var s: Surface { system.surfaces[index] }

    private func change(_ name: String, _ f: @escaping (inout Surface) -> Void) {
        let id = s.id
        edit(name) { sys in
            if let i = sys.surfaceIndex(of: id) { f(&sys.surfaces[i]) }
        }
    }

    var body: some View {
        Form {
            Section("Surface \(index)\(index == system.stop ? " (stop)" : "")") {
                TextField("Comment", text: Binding(get: { s.label }, set: { v in change("Change Label") { $0.label = v } }))
                LabeledContent("Type", value: s.shape.kind.title)
                Picker("Interaction", selection: Binding(get: { s.interaction }, set: { v in change("Change Interaction") { $0.interaction = v } })) {
                    Text("Refract").tag(Interaction.refract)
                    Text("Reflect (mirror)").tag(Interaction.reflect)
                    Text("Dummy").tag(Interaction.dummy)
                }
                .disabled(index == 0 || index == system.imageIndex)
                NumericField(title: "Thickness", value: s.thickness, unit: "mm", parse: { LensValueParser.thickness($0, allowInfinite: index == 0) }) { v in
                    change("Change Thickness") { $0.thickness = v }
                }
                if index > 0 && index < system.imageIndex && index != system.stop {
                    Button("Make Stop") { edit("Set Stop") { $0.setStop(index) } }
                }
            }
            shapeSection
            apertureSection
            placementSection
            Section("Coating") {
                Picker("Coating", selection: Binding(get: { s.coating ?? "" }, set: { v in change("Change Coating") { $0.coating = v.isEmpty ? nil : v } })) {
                    Text("None (bare)").tag("")
                    ForEach(system.coatings.keys.sorted(), id: \.self) { Text($0).tag($0) }
                }
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder private var shapeSection: some View {
        Section("Shape") {
            switch s.shape {
            case .plane, .conic, .evenAsphere:
                NumericField(title: "Radius", value: s.shape.radius, unit: "mm", parse: LensValueParser.radius) { r in
                    change("Change Radius") { $0.shape.setCurvature(RadiusDisplay.curvature(fromRadius: r)) }
                }
                NumericField(title: "Curvature", value: s.shape.curvature, unit: "1/mm") { c in change("Change Curvature") { $0.shape.setCurvature(c) } }
                NumericField(title: "Conic", value: s.shape.conicConstant) { k in change("Change Conic") { $0.shape.setConic(k) } }
                if case .evenAsphere(_, _, let a) = s.shape {
                    ForEach(a.indices, id: \.self) { i in
                        NumericField(title: "A\(2 * i + 4)", value: a[i], unit: "mm^\(-(2 * i + 3))") { v in
                            change("Change Asphere Coefficient") { sf in
                                if case .evenAsphere(let c, let k, var co) = sf.shape {
                                    co[i] = v
                                    sf.shape = .evenAsphere(curvature: c, conic: k, coefficients: co)
                                }
                            }
                        }
                    }
                    HStack {
                        Button("Add Term") {
                            change("Add Asphere Term") { sf in
                                if case .evenAsphere(let c, let k, let co) = sf.shape { sf.shape = .evenAsphere(curvature: c, conic: k, coefficients: co + [0]) }
                            }
                        }
                        Button("Remove Last") {
                            change("Remove Asphere Term") { sf in
                                if case .evenAsphere(let c, let k, let co) = sf.shape { sf.shape = .evenAsphere(curvature: c, conic: k, coefficients: Array(co.dropLast())) }
                            }
                        }
                        .disabled(a.isEmpty)
                    }
                }
            case .paraxial(let f):
                NumericField(title: "Focal length", value: f, unit: "mm") { v in change("Change Focal Length") { $0.shape = .paraxial(focalLength: v) } }
            case .coordinateBreak:
                Text("Changes the coordinate frame only; see Placement.").foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var apertureSection: some View {
        Section("Clear Aperture") {
            Picker("Semi-diameter", selection: Binding(get: { s.semiDiameter == .auto ? 0 : 1 }, set: { v in
                change("Change Semi-Diameter") { $0.semiDiameter = v == 0 ? .auto : .fixed($0.semiDiameter.fixedValue ?? 10) }
            })) {
                Text("Automatic").tag(0)
                Text("Fixed").tag(1)
            }
            if let v = s.semiDiameter.fixedValue {
                NumericField(title: "Value", value: v, unit: "mm", parse: LensValueParser.positive) { nv in change("Change Semi-Diameter") { $0.semiDiameter = .fixed(nv) } }
            }
            Picker("Aperture", selection: Binding(get: { apertureKind }, set: { k in change("Change Aperture") { $0.aperture = Self.aperture(k, size: $0.semiDiameter.fixedValue ?? 10) } })) {
                Text("None").tag(0)
                Text("Circular").tag(1)
                Text("Annular").tag(2)
                Text("Rectangular").tag(3)
                Text("Elliptical").tag(4)
            }
            switch s.aperture {
            case .circular(let r):
                NumericField(title: "Radius", value: r, unit: "mm", parse: LensValueParser.positive) { v in change("Change Aperture") { $0.aperture = .circular(radius: v) } }
            case .annular(let a, let b):
                NumericField(title: "Inner radius", value: a, unit: "mm") { v in change("Change Aperture") { $0.aperture = .annular(innerRadius: v, outerRadius: b) } }
                NumericField(title: "Outer radius", value: b, unit: "mm", parse: LensValueParser.positive) { v in change("Change Aperture") { $0.aperture = .annular(innerRadius: a, outerRadius: v) } }
            case .rectangular(let x, let y):
                NumericField(title: "Half width (x)", value: x, unit: "mm", parse: LensValueParser.positive) { v in change("Change Aperture") { $0.aperture = .rectangular(halfX: v, halfY: y) } }
                NumericField(title: "Half height (y)", value: y, unit: "mm", parse: LensValueParser.positive) { v in change("Change Aperture") { $0.aperture = .rectangular(halfX: x, halfY: v) } }
            case .elliptical(let x, let y):
                NumericField(title: "Semi-axis x", value: x, unit: "mm", parse: LensValueParser.positive) { v in change("Change Aperture") { $0.aperture = .elliptical(semiX: v, semiY: y) } }
                NumericField(title: "Semi-axis y", value: y, unit: "mm", parse: LensValueParser.positive) { v in change("Change Aperture") { $0.aperture = .elliptical(semiX: x, semiY: v) } }
            case nil:
                EmptyView()
            }
        }
    }

    private var apertureKind: Int {
        switch s.aperture {
        case nil: 0
        case .circular: 1
        case .annular: 2
        case .rectangular: 3
        case .elliptical: 4
        }
    }

    private static func aperture(_ k: Int, size: Double) -> ApertureShape? {
        switch k {
        case 1: .circular(radius: size)
        case 2: .annular(innerRadius: size / 4, outerRadius: size)
        case 3: .rectangular(halfX: size, halfY: size)
        case 4: .elliptical(semiX: size, semiY: size)
        default: nil
        }
    }

    @ViewBuilder private var placementSection: some View {
        let p = s.placement ?? Placement()
        Section("Placement (applied before the surface)") {
            NumericField(title: "Decenter X", value: p.decenterX, unit: "mm") { v in place { $0.decenterX = v } }
            NumericField(title: "Decenter Y", value: p.decenterY, unit: "mm") { v in place { $0.decenterY = v } }
            NumericField(title: "Tilt X", value: p.tiltX, unit: "°") { v in place { $0.tiltX = v } }
            NumericField(title: "Tilt Y", value: p.tiltY, unit: "°") { v in place { $0.tiltY = v } }
            NumericField(title: "Tilt Z", value: p.tiltZ, unit: "°") { v in place { $0.tiltZ = v } }
            Picker("Order", selection: Binding(get: { p.order }, set: { v in place { $0.order = v } })) {
                Text("Decenter, then tilt").tag(TransformOrder.decenterThenTilt)
                Text("Tilt, then decenter").tag(TransformOrder.tiltThenDecenter)
            }
            Toggle("Restore frame after surface", isOn: Binding(get: { p.restoreAfter }, set: { v in place { $0.restoreAfter = v } }))
        }
        .disabled(index == 0 && system.objectAtInfinity)
    }

    private func place(_ f: @escaping (inout Placement) -> Void) {
        change("Change Placement") { sf in
            var p = sf.placement ?? Placement()
            f(&p)
            sf.placement = (p.isIdentity && !p.restoreAfter && p.order == .decenterThenTilt && !sf.isCoordinateBreak) ? nil : p
        }
    }
}
