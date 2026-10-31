import MKLightEngine
import MKLightModel
import SwiftUI

/// The inspector: the selected surface's parameters, the system's, or the
/// open design job (Optimize, Tolerance, Solves). It is a column of the
/// window's glass layer: the Surface / System switch floats in a bar over
/// the form, which scrolls beneath it.
struct InspectorView: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Group {
            if let job = window.activeJob {
                JobPanel(kind: job, design: design, workspace: workspace, window: window)
                    .id(job)
                    .transition(.opacity)
            } else {
                tabs.transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.25), value: window.activeJob)
    }

    private var tabs: some View {
        Group {
            switch window.inspectorTab {
            case .surface:
                if let id = window.selection, let i = design.system.surfaceIndex(of: id) {
                    SurfaceInspector(design: design, workspace: workspace, window: window, index: i)
                        .id(id)
                } else {
                    ContentUnavailableView(
                        "No Surface Selected", systemImage: "cursorarrow.click",
                        description: Text("Select a row in the lens data to edit its surface."))
                }
            case .system:
                SystemInspector(design: design, workspace: workspace, window: window)
            }
        }
        .frame(maxHeight: .infinity, alignment: .top)
        .scrollEdgeEffectStyle(.soft, for: .top)
        .safeAreaBar(edge: .top) {
            Picker("Inspector", selection: $window.inspectorTab) {
                ForEach(InspectorTab.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.large)
            .padding(.horizontal, Space.l)
            .padding(.vertical, Space.s)
        }
    }
}

/// The selected surface, built from its type's descriptor: only the
/// parameters of that type show, with a preview of its shape. Every
/// continuous value can be made a variable.
struct SurfaceInspector: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    let index: Int
    @Environment(\.undoManager) private var undoManager

    private var system: OpticalSystem { design.system }
    private var surface: Surface { system.surfaces[index] }
    private var type: SurfaceTypeDescriptor { SurfaceTypes.descriptor(for: surface.shape) }
    private var isEnd: Bool { index == 0 || index == system.imageIndex }
    private var title: String {
        index == 0 ? "Object" : index == system.imageIndex ? "Image" : "Surface \(index)"
    }

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: Space.s) {
                    HStack(alignment: .firstTextBaseline, spacing: Space.s) {
                        Text(title).font(.title3.weight(.semibold))
                        if index == system.stop {
                            Label("Stop", systemImage: "camera.aperture")
                                .font(.caption.weight(.medium))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.tint.opacity(0.15), in: Capsule())
                                .foregroundStyle(.tint)
                        }
                        Spacer()
                        if !isEnd { TypeMenu(shape: surface.shape) { setType($0) } }
                    }
                    Text(type.summary).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, Space.xs)
                if !isEnd && type.preview != .idealLens && type.preview != .coordinateBreak {
                    ShapePreview(shape: surface.shape, semiDiameter: semiDiameter, primaryWavelengthUm: primaryWavelength)
                        .padding(.vertical, Space.xs)
                }
                TextValueField(title: "Comment", value: surface.label, prompt: "none") { v in
                    edit("Change Comment") { $0.label = v }
                }
            }
            if type.placementFirst { PlacementSection(surface: surface, edit: edit, variable: variable) }
            if case .diffractive(let base, _) = surface.shape {
                Section("Base Shape") {
                    Picker("Base", selection: Binding(get: { base.kind }, set: setDiffractiveBase)) {
                        Text("Plane").tag(SurfaceShape.Kind.plane)
                        Text("Sphere / Conic").tag(SurfaceShape.Kind.conic)
                        Text("Even Asphere").tag(SurfaceShape.Kind.evenAsphere)
                    }
                }
            }
            ForEach(type.sections) { section in
                let params = section.params.filter { SurfaceTypes.value(surface.shape, at: $0.key) != nil }
                let list = section.coefficients.flatMap { SurfaceTypes.value(surface.shape, at: $0.key) != nil ? $0 : nil }
                if section.applies(surface.shape) && (!params.isEmpty || list != nil || !section.choices.isEmpty) {
                    Section {
                        ForEach(section.choices) { c in choiceField(c) }
                        ForEach(params) { p in paramField(p) }
                        if let list { CoefficientRows(list: list, surface: surface, edit: edit, variable: variable) }
                    } header: {
                        Text(section.title)
                    } footer: {
                        if !section.footer.isEmpty { Text(section.footer).font(.caption).foregroundStyle(.secondary) }
                    }
                }
            }
            if type.orderEfficiency, case .diffractive(_, let phase) = surface.shape {
                DiffractionEfficiencySection(
                    model: surface.diffractionEfficiency, phase: phase, wavelengths: system.wavelengths.wavelengths.map(\.um)
                ) { m in edit("Change Diffraction Efficiency") { $0.diffractionEfficiency = m } }
            }
            if surface.shape.kind == .biconic {
                Section {
                    Menu("Make Cylinder") {
                        Button("Power in x (plane in y)") { setType(.cylinder(.x)) }
                        Button("Power in y (plane in x)") { setType(.cylinder(.y)) }
                    }
                    .fixedSize()
                } footer: {
                    Text("A cylinder is a biconic with one section plane.").font(.caption).foregroundStyle(.secondary)
                }
            }
            if index != system.imageIndex {
                Section("Spacing and Medium") {
                    NumberField(
                        title: "Thickness", value: surface.thickness, unit: "mm", help: "Distance to the next vertex.",
                        parse: { LensValueParser.thickness($0, allowInfinite: index == 0) },
                        variable: variable(.thickness(surface: surface.id)),
                        solve: system.solve(for: .thickness(surface: surface.id))?.kind.kind.title
                    ) { v in edit("Change Thickness") { $0.thickness = v } }
                    if !surface.isCoordinateBreak && index != 0 {
                        Picker("Interaction", selection: Binding(get: { surface.interaction }, set: { v in edit("Change Interaction") { $0.interaction = v } })) {
                            Text("Refract").tag(Interaction.refract)
                            Text("Mirror").tag(Interaction.reflect)
                            Text("Dummy").tag(Interaction.dummy)
                        }
                    }
                    if !surface.isCoordinateBreak && surface.interaction == .refract {
                        TextValueField(
                            title: "Material", value: surface.material.isAir ? "AIR" : surface.material.displayName, prompt: "AIR",
                            suggestions: workspace.materialNames
                        ) { text in
                            switch LensValueParser.material(text) {
                            case .success(let m): edit("Change Material") { s in s.interaction = m.interaction; if let mat = m.material { s.material = mat } }
                            case .failure(let f): window.flash(f.description)
                            }
                        }
                        if case .model(let nd, let vd, let dpgf) = surface.material {
                            NumberField(title: "n_d", value: nd, variable: variable(.modelNd(surface: surface.id))) { v in
                                edit("Change n_d") { $0.material = .model(nd: v, vd: vd, dpgf: dpgf) }
                            }
                            NumberField(title: "V_d", value: vd, variable: variable(.modelVd(surface: surface.id))) { v in
                                edit("Change V_d") { $0.material = .model(nd: nd, vd: v, dpgf: dpgf) }
                            }
                        }
                    }
                }
            }
            if !surface.isCoordinateBreak {
                ApertureSection(surface: surface, autoValue: semiDiameter, edit: edit, variable: variable)
            }
            if !type.placementFirst && !isEnd { PlacementSection(surface: surface, edit: edit, variable: variable) }
            if !isEnd && !surface.isCoordinateBreak {
                Section("Coating") {
                    Picker("Coating", selection: Binding(get: { surface.coating ?? "" }, set: { v in edit("Change Coating") { $0.coating = v.isEmpty ? nil : v } })) {
                        Text("Uncoated").tag("")
                        ForEach(system.coatings.keys.sorted(), id: \.self) { Text($0).tag($0) }
                    }
                }
            }
            SolvesSection(design: design, surface: surface)
        }
        .formStyle(.grouped)
    }

    /// The drawn semi-diameter: fixed, or the engine's automatic one.
    private var semiDiameter: Double? {
        surface.semiDiameter.fixedValue ?? EngineLayout.autoSemiDiameters(EngineLayout.scene(from: workspace.layout.output))[surface.id]
    }

    private var primaryWavelength: Double {
        let w = system.wavelengths
        return w.wavelengths.indices.contains(w.primary) ? w.wavelengths[w.primary].um : 0.5875618
    }

    @ViewBuilder private func paramField(_ p: ShapeParam) -> some View {
        let path = p.path?(surface.id)
        NumberField(
            title: p.title, value: SurfaceTypes.displayValue(p, in: surface.shape), unit: p.unit, help: p.help,
            parse: { text in
                switch SurfaceTypes.storedValue(p, text: text) {
                case .success(let j): .success(j.doubleValue ?? 0)
                case .failure(let f): .failure(f)
                }
            },
            variable: path.map(variable), solve: path.flatMap { system.solve(for: $0)?.kind.kind.title }
        ) { stored in
            let value: JSONValue = p.display == .integer ? .integer(Int64(stored)) : .number(stored)
            edit("Change \(p.title)") { s in
                if let shape = SurfaceTypes.shape(s.shape, setting: p.key, to: value) { s.shape = shape }
            }
        }
    }

    private func choiceField(_ c: ShapeChoice) -> some View {
        let current = SurfaceTypes.value(surface.shape, at: c.key)?.string ?? c.options.first?.value ?? ""
        return Picker(
            c.title,
            selection: Binding(
                get: { current },
                set: { v in
                    edit("Change \(c.title)") { s in
                        if let shape = SurfaceTypes.shape(s.shape, setting: c.key, to: .string(v)) { s.shape = shape }
                    }
                })
        ) {
            ForEach(c.options, id: \.value) { Text($0.title).tag($0.value) }
        }
        .help(c.help)
    }

    private func edit(_ name: String, _ change: @escaping (inout Surface) -> Void) {
        let id = surface.id
        design.edit(name, undoManager: undoManager) { sys in
            if let i = sys.surfaceIndex(of: id) { change(&sys.surfaces[i]) }
        }
    }

    private func setType(_ choice: SurfaceTypeChoice) {
        LensEditor(document: design, undoManager: undoManager).setType(choice, surface.id)
    }

    private func setDiffractiveBase(_ k: SurfaceShape.Kind) {
        edit("Change Base Shape") { s in
            guard case .diffractive(let base, let phase) = s.shape else { return }
            let c = base.curvature, kk = base.conicConstant
            let newBase: SurfaceShape =
                switch k {
                case .conic: .conic(curvature: c, conic: kk)
                case .evenAsphere: .evenAsphere(curvature: c, conic: kk, coefficients: base.asphereCoefficients)
                default: .plane
                }
            s.shape = .diffractive(base: newBase, phase: phase)
        }
    }

    private func variable(_ p: ParamPath) -> FieldVariable {
        FieldVariable(
            isOn: Binding(get: { design.isVariable(p) }, set: { design.setVariable(p, $0, undoManager: undoManager) }),
            solve: design.system.solve(for: p)?.kind.kind.title)
    }
}

/// The surface type as a menu grouped by family (Spherical, Aspheric,
/// Freeform, Anamorphic, Diffractive, Special), with presets such as the
/// cylinder. Shared by the inspector and the lens table.
struct TypeMenu: View {
    let shape: SurfaceShape
    var compact = false
    let choose: (SurfaceTypeChoice) -> Void

    var body: some View {
        let current = SurfaceTypes.choice(for: shape)
        Menu {
            ForEach(SurfaceTypes.menu, id: \.group) { g in
                Section(g.group.title) {
                    ForEach(g.choices) { c in
                        Toggle(isOn: Binding(get: { c == current }, set: { _ in choose(c) })) {
                            Label(SurfaceTypes.title(of: c), systemImage: SurfaceTypes.symbol(of: c))
                        }
                    }
                }
            }
        } label: {
            if compact {
                Label(LensRow.typeTitle(shape), systemImage: SurfaceTypes.symbol(of: current))
            } else {
                Label(SurfaceTypes.title(of: current), systemImage: SurfaceTypes.symbol(of: current))
            }
        }
        .accessibilityLabel("Surface type")
        .accessibilityValue(SurfaceTypes.title(of: current))
    }
}

/// The coefficient list of a type (e.g. A₄, A₆, … of an even asphere),
/// each labelled with its meaning and with a variable toggle. Long lists
/// (XY, Zernike) show their nonzero terms and add any term from a menu.
private struct CoefficientRows: View {
    let list: CoefficientList
    let surface: Surface
    let edit: (String, @escaping (inout Surface) -> Void) -> Void
    let variable: (ParamPath) -> FieldVariable
    @State private var revealed: Set<Int> = []
    @State private var showAll = false

    var body: some View {
        let values = SurfaceTypes.coefficients(list, in: surface.shape)
        let limit = list.maxCount(surface.shape)
        let visible = values.indices.filter { k in
            !list.sparse || showAll || values[k] != 0 || revealed.contains(k) || k == values.count - 1
        }
        ForEach(visible, id: \.self) { k in
            HStack(spacing: Space.xs) {
                NumberField(
                    title: list.term(k, surface.shape), detail: list.meaning(k, surface.shape), value: values[k],
                    unit: list.unit(k), variable: list.path.map { variable($0(surface.id, k)) }
                ) { nv in set(values, k, nv) }
                let removable = k == values.count - 1 || list.sparse
                Button("Remove \(list.term(k, surface.shape))", systemImage: "minus.circle") { remove(values, k) }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .opacity(removable ? 1 : 0)
                    .disabled(!removable)
                    .help(k == values.count - 1 ? "Remove this term" : "Set this term to zero")
            }
        }
        HStack {
            if values.count < limit || (list.sparse && values.contains(0)) {
                if list.sparse {
                    Menu {
                        ForEach(0..<limit, id: \.self) { k in
                            if k >= values.count || (values[k] == 0 && !revealed.contains(k)) {
                                Button("\(list.term(k, surface.shape))   \(list.meaning(k, surface.shape))") { add(values, k) }
                            }
                        }
                    } label: {
                        Label("Add Term", systemImage: "plus")
                    }
                    .fixedSize()
                } else {
                    Button {
                        add(values, values.count)
                    } label: {
                        Label("Add \(list.term(values.count, surface.shape))  \(list.meaning(values.count, surface.shape))", systemImage: "plus")
                    }
                }
            }
            Spacer()
            if list.sparse && values.dropLast().contains(0) {
                Toggle("Zero Terms", isOn: $showAll)
                    .toggleStyle(.button)
                    .help("Show the terms that are zero")
            }
        }
        .buttonStyle(.borderless)
        if values.count >= limit {
            Text("\(limit) terms, the most this type allows.").font(.caption).foregroundStyle(.secondary)
        }
    }

    private func set(_ values: [Double], _ k: Int, _ v: Double) {
        var a = values
        a[k] = v
        write(a, "Change \(list.term(k, surface.shape))")
    }

    private func add(_ values: [Double], _ k: Int) {
        var a = values
        if k >= a.count { a += Array(repeating: 0, count: k - a.count + 1) }
        revealed.insert(k)
        write(a, "Add \(list.term(k, surface.shape))")
    }

    private func remove(_ values: [Double], _ k: Int) {
        var a = values
        if k == a.count - 1 {
            a.removeLast()
            if list.sparse {
                while let last = a.last, last == 0, !revealed.contains(a.count - 1) { a.removeLast() }
            }
        } else {
            a[k] = 0
        }
        revealed.remove(k)
        write(a, "Remove \(list.term(k, surface.shape))")
    }

    private func write(_ a: [Double], _ name: String) {
        let key = list.key
        edit(name) { s in
            if let shape = SurfaceTypes.shape(s.shape, setting: key, to: .array(a.map(JSONValue.number))) { s.shape = shape }
        }
    }
}

/// Clear aperture: automatic or fixed semi-diameter, and an optional
/// explicit aperture shape.
private struct ApertureSection: View {
    let surface: Surface
    let autoValue: Double?
    let edit: (String, @escaping (inout Surface) -> Void) -> Void
    let variable: (ParamPath) -> FieldVariable

    enum Shape: String, CaseIterable { case none, circular, annular, rectangular, elliptical }

    var body: some View {
        Section("Aperture") {
            Picker("Semi-diameter", selection: Binding(get: { surface.semiDiameter == .auto }, set: { auto in
                edit("Change Semi-Diameter") { s in s.semiDiameter = auto ? .auto : .fixed(s.semiDiameter.fixedValue ?? autoValue ?? 1) }
            })) {
                Text("Automatic").tag(true)
                Text("Fixed").tag(false)
            }
            if case .fixed(let v) = surface.semiDiameter {
                NumberField(title: "Value", value: v, unit: "mm", parse: LensValueParser.positiveLength, variable: variable(.semiDiameter(surface: surface.id))) { nv in
                    edit("Change Semi-Diameter") { $0.semiDiameter = .fixed(nv) }
                }
            } else if let autoValue {
                LabeledContent("Computed") { Text(Fmt.short(autoValue) + " mm").numeric().foregroundStyle(.secondary) }
            }
            Picker("Clipping", selection: Binding(get: { shape }, set: setShape)) {
                Text("Semi-diameter").tag(Shape.none)
                Text("Circular").tag(Shape.circular)
                Text("Annular").tag(Shape.annular)
                Text("Rectangular").tag(Shape.rectangular)
                Text("Elliptical").tag(Shape.elliptical)
            }
            switch surface.aperture {
            case .circular(let r)?:
                NumberField(title: "Radius", value: r, unit: "mm", parse: LensValueParser.positiveLength) { v in setAperture(.circular(radius: v)) }
            case .annular(let a, let b)?:
                NumberField(title: "Inner radius", value: a, unit: "mm", parse: LensValueParser.positiveLength) { v in setAperture(.annular(innerRadius: v, outerRadius: b)) }
                NumberField(title: "Outer radius", value: b, unit: "mm", parse: LensValueParser.positiveLength) { v in setAperture(.annular(innerRadius: a, outerRadius: v)) }
            case .rectangular(let x, let y)?:
                NumberField(title: "Half width x", value: x, unit: "mm", parse: LensValueParser.positiveLength) { v in setAperture(.rectangular(halfX: v, halfY: y)) }
                NumberField(title: "Half width y", value: y, unit: "mm", parse: LensValueParser.positiveLength) { v in setAperture(.rectangular(halfX: x, halfY: v)) }
            case .elliptical(let x, let y)?:
                NumberField(title: "Semi-axis x", value: x, unit: "mm", parse: LensValueParser.positiveLength) { v in setAperture(.elliptical(semiX: v, semiY: y)) }
                NumberField(title: "Semi-axis y", value: y, unit: "mm", parse: LensValueParser.positiveLength) { v in setAperture(.elliptical(semiX: x, semiY: v)) }
            case nil:
                EmptyView()
            }
        }
    }

    private var shape: Shape {
        switch surface.aperture {
        case .circular?: .circular
        case .annular?: .annular
        case .rectangular?: .rectangular
        case .elliptical?: .elliptical
        case nil: .none
        }
    }

    private func setShape(_ s: Shape) {
        let r = surface.semiDiameter.fixedValue ?? autoValue ?? 5
        let a: ApertureShape? =
            switch s {
            case .none: nil
            case .circular: .circular(radius: r)
            case .annular: .annular(innerRadius: r / 3, outerRadius: r)
            case .rectangular: .rectangular(halfX: r, halfY: r)
            case .elliptical: .elliptical(semiX: r, semiY: r)
            }
        setAperture(a)
    }

    private func setAperture(_ a: ApertureShape?) { edit("Change Aperture") { $0.aperture = a } }
}

/// Decenters and tilts before the surface.
private struct PlacementSection: View {
    let surface: Surface
    let edit: (String, @escaping (inout Surface) -> Void) -> Void
    let variable: (ParamPath) -> FieldVariable
    @State private var expanded = false

    var body: some View {
        let p = surface.placement ?? Placement()
        Section {
            if expanded || surface.isCoordinateBreak || !p.isIdentity {
                NumberField(title: "Decenter X", value: p.decenterX, unit: "mm", variable: variable(.decenterX(surface: surface.id))) { v in set { $0.decenterX = v } }
                NumberField(title: "Decenter Y", value: p.decenterY, unit: "mm", variable: variable(.decenterY(surface: surface.id))) { v in set { $0.decenterY = v } }
                NumberField(title: "Tilt X", value: p.tiltX, unit: "°", variable: variable(.tiltX(surface: surface.id))) { v in set { $0.tiltX = v } }
                NumberField(title: "Tilt Y", value: p.tiltY, unit: "°", variable: variable(.tiltY(surface: surface.id))) { v in set { $0.tiltY = v } }
                NumberField(title: "Tilt Z", value: p.tiltZ, unit: "°", variable: variable(.tiltZ(surface: surface.id))) { v in set { $0.tiltZ = v } }
                Picker("Order", selection: Binding(get: { p.order }, set: { o in set { $0.order = o } })) {
                    Text("Decenter, then tilt").tag(TransformOrder.decenterThenTilt)
                    Text("Tilt, then decenter").tag(TransformOrder.tiltThenDecenter)
                }
                Toggle("Return after surface", isOn: Binding(get: { p.restoreAfter }, set: { r in set { $0.restoreAfter = r } }))
            } else {
                Button("Add Decenter or Tilt…") { expanded = true }.buttonStyle(.borderless)
            }
        } header: {
            Text("Placement")
        }
    }

    private func set(_ f: @escaping (inout Placement) -> Void) {
        edit("Change Placement") { s in
            var p = s.placement ?? Placement()
            f(&p)
            s.placement = p.isIdentity && !s.isCoordinateBreak && p.order == .decenterThenTilt && !p.restoreAfter ? nil : p
        }
    }
}

/// Solves on this surface's parameters.
private struct SolvesSection: View {
    let design: DesignDocument
    let surface: Surface
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        let solves = design.system.solves.enumerated().filter { $0.element.target.surface == surface.id }
        if !solves.isEmpty || canAdd {
            Section("Solves") {
                ForEach(solves, id: \.offset) { k, s in
                    HStack {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(s.kind.kind.title)
                            Text(s.target.label(in: design.system)).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if let label = s.kind.kind.valueLabel, let v = s.kind.value {
                            Text("\(Fmt.short(v)) \(label.unit)").numeric().foregroundStyle(.secondary)
                        }
                        Button("Remove", systemImage: "minus.circle") {
                            design.edit("Remove Solve", undoManager: undoManager) { $0.solves.remove(at: k) }
                        }
                        .labelStyle(.iconOnly).buttonStyle(.borderless).foregroundStyle(.secondary)
                    }
                }
                if canAdd {
                    Menu("Add Solve") {
                        Button("Marginal ray height on thickness") { add(.thickness(surface: surface.id), .marginalRayHeight(height: 0)) }
                        Button("Chief ray height on thickness") { add(.thickness(surface: surface.id), .chiefRayHeight(height: 0)) }
                        if surface.shape.hasCurvature {
                            Button("Marginal ray angle on curvature") { add(.curvature(surface: surface.id), .marginalRayAngle(slope: 0)) }
                            Button("F-number on curvature") { add(.curvature(surface: surface.id), .fNumber(value: 4)) }
                        }
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                }
            }
        }
    }

    private var canAdd: Bool {
        let i = design.system.surfaceIndex(of: surface.id) ?? 0
        return i > 0 && i < design.system.imageIndex
    }

    private func add(_ target: ParamPath, _ kind: SolveKind) {
        design.edit("Add Solve", undoManager: undoManager) { s in
            s.solves.removeAll { $0.target == target }
            s.solves.append(Solve(target: target, kind: kind))
        }
    }
}

/// A small drawing of the surface's sag profile (display only).
struct SagPreview: View {
    let shape: SurfaceShape
    let semiDiameter: Double?

    var body: some View {
        Canvas { ctx, size in
            let rect = CGRect(origin: .zero, size: size).insetBy(dx: 6, dy: 6)
            var axis = Path()
            axis.move(to: CGPoint(x: rect.midX, y: rect.minY))
            axis.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
            ctx.stroke(axis, with: .color(.secondary.opacity(0.3)), style: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
            switch shape {
            case .coordinateBreak:
                let c = CGPoint(x: rect.midX, y: rect.midY)
                for (dx, dy) in [(1.0, 0.0), (0.0, -1.0)] {
                    var p = Path()
                    p.move(to: c)
                    p.addLine(to: CGPoint(x: c.x + dx * rect.width * 0.4, y: c.y + dy * rect.height * 0.4))
                    ctx.stroke(p, with: .color(.secondary), lineWidth: 1.5)
                }
            case .paraxial(let f):
                var p = Path()
                p.move(to: CGPoint(x: rect.midX, y: rect.minY))
                p.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
                ctx.stroke(p, with: .color(.primary), lineWidth: 1.5)
                let s: CGFloat = f >= 0 ? 1 : -1
                for (y, dir) in [(rect.minY, CGFloat(1)), (rect.maxY, CGFloat(-1))] {
                    var a = Path()
                    a.move(to: CGPoint(x: rect.midX - 6, y: y + dir * 6 * s))
                    a.addLine(to: CGPoint(x: rect.midX, y: y))
                    a.addLine(to: CGPoint(x: rect.midX + 6, y: y + dir * 6 * s))
                    ctx.stroke(a, with: .color(.primary), lineWidth: 1.5)
                }
            default:
                let h = semiDiameter ?? 1
                guard let pts = SurfaceTypes.profile(shape, semiDiameter: h) else { return }
                // Height to fit; sag exaggerated so a shallow curve still shows.
                let zMax = max(pts.map { abs($0.1) }.max() ?? 0, 1e-12)
                let rScale = rect.height / (2 * h)
                let zScale = max(rScale, min(rect.width * 0.35 / zMax, rScale * 20))
                let flip: CGFloat = (pts.map(\.1).max() ?? 0) > 0 ? -1 : 1
                var p = Path()
                for (k, q) in pts.enumerated() {
                    let pt = CGPoint(x: rect.midX + flip * rect.width * 0.15 + CGFloat(q.1) * zScale, y: rect.midY - CGFloat(q.0) * rScale)
                    if k == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
                }
                ctx.stroke(p, with: .color(.primary), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
            }
        }
        .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.5)))
        .accessibilityLabel("Surface profile preview")
    }
}
