import MKLightEngine
import MKLightModel
import SwiftUI

/// System settings: aperture, fields, wavelengths (with presets),
/// environment, ray aiming, configurations and glass catalogs.
struct SystemInspector: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    @Environment(\.undoManager) private var undoManager

    private var system: OpticalSystem { design.system }

    var body: some View {
        Form {
            Section {
                TextValueField(title: "Name", value: system.name) { v in edit("Rename") { $0.name = v } }
                TextValueField(title: "Notes", value: system.notes, prompt: "none") { v in edit("Change Notes") { $0.notes = v } }
            }
            Section("Aperture") {
                Picker("Type", selection: Binding(get: { system.aperture.kind }, set: setApertureKind)) {
                    ForEach(ApertureSpec.Kind.allCases) { Text($0.title).tag($0) }
                }
                if let v = system.aperture.value {
                    NumberField(
                        title: "Value", value: v, unit: system.aperture.kind.unit, parse: LensValueParser.positive,
                        variable: variable(.apertureValue)
                    ) { nv in edit("Change Aperture") { $0.aperture = ApertureSpec(kind: $0.aperture.kind, value: nv) } }
                }
            }
            anamorphicSection
            fieldsSection
            wavelengthsSection
            Section("Environment") {
                NumberField(title: "Temperature", value: system.environment.temperatureC, unit: "°C") { v in
                    edit("Change Temperature") { $0.environment.temperatureC = v }
                }
                NumberField(title: "Pressure", value: system.environment.pressureAtm, unit: "atm", parse: LensValueParser.positive) { v in
                    edit("Change Pressure") { $0.environment.pressureAtm = v }
                }
            }
            Section {
                Toggle("Real ray aiming", isOn: Binding(get: { system.rayAiming != .off }, set: { on in
                    edit("Change Ray Aiming") { $0.rayAiming = on ? .real(tolerance: RayAiming.defaultTolerance) : .off }
                }))
            } header: {
                Text("Ray Aiming")
            } footer: {
                Text("Aims every ray through the real stop instead of the paraxial entrance pupil. Slower, needed for wide fields and pupil aberration.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !system.configurations.isEmpty {
                Section("Configurations") {
                    ForEach(Array(system.configurations.enumerated()), id: \.offset) { k, c in
                        LabeledContent(c.name.isEmpty ? "Configuration \(k + 1)" : c.name) {
                            Text("\(c.overrides.count) override\(c.overrides.count == 1 ? "" : "s")").foregroundStyle(.secondary)
                        }
                    }
                }
            }
            Section("Glass Catalogs") {
                ForEach(workspace.catalogs) { c in
                    LabeledContent(c.name) {
                        Text("\(c.glassCount) glasses").foregroundStyle(.secondary)
                    }
                }
                ForEach(workspace.catalogWarnings, id: \.self) { w in
                    Label(w, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                }
            }
        }
        .formStyle(.grouped)
    }

    // MARK: First order

    /// The x / y first order of a design that is not rotationally symmetric
    /// (the engine's `paraxial_xy` at the primary wavelength).
    @ViewBuilder private var anamorphicSection: some View {
        let json = workspace.summary.output?.json
        if ParaxialXYData.isAnamorphic(json, system: system) {
            Section {
                if let xy = ParaxialXYData(json), let p = xy.primary {
                    LabeledContent("EFL x / y", value: Fmt.short(p.eflX) + " / " + Fmt.short(p.eflY) + " mm")
                        .monospacedDigit()
                    LabeledContent("BFL x / y", value: Fmt.short(p.bflX) + " / " + Fmt.short(p.bflY) + " mm")
                        .monospacedDigit()
                    LabeledContent("Squeeze", value: ParaxialXYData.squeezeText(p.squeeze))
                        .monospacedDigit()
                        .help("Squeeze: " + (xy.squeezeDefinition ?? "signed EFL y / EFL x"))
                    if let s = xy.symmetry { LabeledContent("Symmetry", value: s.title) }
                } else {
                    Text("Not rotationally symmetric. The engine's first-order data covers the y section.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } header: {
                Text("First Order")
            }
            .opacity(workspace.summary.isStale ? 0.5 : 1)
        }
    }

    // MARK: Fields

    private var fieldsSection: some View {
        Section {
            Picker("Type", selection: Binding(get: { system.fields.kind }, set: { k in edit("Change Field Type") { $0.fields.kind = k } })) {
                ForEach(FieldKind.allCases) { Text($0.title).tag($0) }
            }
            Grid(alignment: .trailing, horizontalSpacing: Space.m, verticalSpacing: Space.xs) {
                GridRow {
                    Text("")
                    Text("X (\(system.fields.kind.unit))")
                    Text("Y (\(system.fields.kind.unit))")
                    Text("Weight")
                    Text("")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(Array(system.fields.fields.enumerated()), id: \.offset) { k, f in
                    GridRow {
                        HStack(spacing: Space.xs) {
                            Circle().fill(Palette.fieldColor(k)).frame(width: 7, height: 7)
                            Text("\(k + 1)").numeric()
                        }
                        .accessibilityElement(children: .combine)
                        .accessibilityLabel("Field \(k + 1)")
                        CompactNumber(value: f.x) { v in edit("Change Field") { $0.fields.fields[k].x = v } }
                        CompactNumber(value: f.y) { v in edit("Change Field") { $0.fields.fields[k].y = v } }
                        CompactNumber(value: f.weight, parse: LensValueParser.finite) { v in edit("Change Field Weight") { $0.fields.fields[k].weight = v } }
                        Button("Remove Field", systemImage: "minus.circle") {
                            edit("Remove Field") { $0.removeField(at: k) }
                        }
                        .labelStyle(.iconOnly).buttonStyle(.borderless).foregroundStyle(.secondary)
                        .disabled(system.fields.fields.count <= 1)
                    }
                    .padding(.vertical, 1)
                    .background(window.highlightedField == k ? Color.accentColor.opacity(0.15) : .clear)
                }
            }
            Button {
                edit("Add Field") { s in
                    let last = s.fields.fields.last ?? Field()
                    s.fields.fields.append(Field(x: last.x, y: last.y * 1.25 + (last.y == 0 ? 5 : 0)))
                }
            } label: {
                Label("Add Field", systemImage: "plus")
            }
            .buttonStyle(.borderless)
        } header: {
            Text("Fields")
        }
        .onDisappear { window.highlightedField = nil }
    }

    // MARK: Wavelengths

    static let presets: [(String, [(Double, Double)], Int)] = [
        ("F, d, C (visible)", [(0.4861327, 1), (0.5875618, 1), (0.6562725, 1)], 1),
        ("F′, e, C′ (visible)", [(0.4799914, 1), (0.5460740, 1), (0.6438469, 1)], 1),
        ("g, F, e, d, C", [(0.4358343, 1), (0.4861327, 1), (0.5460740, 1), (0.5875618, 1), (0.6562725, 1)], 3),
        ("d only", [(0.5875618, 1)], 0),
        ("e only", [(0.5460740, 1)], 0),
        ("Photopic (5)", [(0.47, 0.091), (0.51, 0.503), (0.555, 1), (0.61, 0.503), (0.65, 0.107)], 2),
        ("Near infrared", [(0.75, 1), (0.85, 1), (0.95, 1)], 1),
    ]

    private var wavelengthsSection: some View {
        Section {
            Grid(alignment: .trailing, horizontalSpacing: Space.m, verticalSpacing: Space.xs) {
                GridRow {
                    Text("Primary")
                    Text("λ (µm)")
                    Text("Weight")
                    Text("")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(Array(system.wavelengths.wavelengths.enumerated()), id: \.offset) { k, w in
                    GridRow {
                        Button {
                            edit("Change Primary Wavelength") { $0.wavelengths.primary = k }
                        } label: {
                            Image(systemName: system.wavelengths.primary == k ? "largecircle.fill.circle" : "circle")
                                .foregroundStyle(Palette.wavelengthColor(w.um))
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel(system.wavelengths.primary == k ? "Primary wavelength" : "Make primary")
                        CompactNumber(value: w.um, parse: LensValueParser.positive) { v in edit("Change Wavelength") { $0.wavelengths.wavelengths[k].um = v } }
                        CompactNumber(value: w.weight, parse: LensValueParser.finite) { v in edit("Change Wavelength Weight") { $0.wavelengths.wavelengths[k].weight = v } }
                        Button("Remove Wavelength", systemImage: "minus.circle") {
                            edit("Remove Wavelength") { $0.removeWavelength(at: k) }
                        }
                        .labelStyle(.iconOnly).buttonStyle(.borderless).foregroundStyle(.secondary)
                        .disabled(system.wavelengths.wavelengths.count <= 1)
                    }
                }
            }
            HStack {
                Button {
                    edit("Add Wavelength") { s in s.wavelengths.wavelengths.append(Wavelength(um: (s.wavelengths.wavelengths.last?.um ?? 0.55) + 0.05)) }
                } label: {
                    Label("Add Wavelength", systemImage: "plus")
                }
                .buttonStyle(.borderless)
                Spacer()
                Menu("Presets") {
                    ForEach(Self.presets, id: \.0) { name, list, primary in
                        Button(name) {
                            edit("Wavelength Preset") { s in
                                s.wavelengths = WavelengthSet(wavelengths: list.map { Wavelength(um: $0.0, weight: $0.1) }, primary: primary)
                            }
                        }
                    }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }
        } header: {
            Text("Wavelengths")
        }
    }

    // MARK: Helpers

    private func setApertureKind(_ k: ApertureSpec.Kind) {
        edit("Change Aperture Type") { s in
            let v = s.aperture.value ?? (k == .objectSpaceNA ? 0.1 : k == .entrancePupilDiameter ? 10 : 4)
            s.aperture = ApertureSpec(kind: k, value: v)
        }
    }

    private func edit(_ name: String, _ change: (inout OpticalSystem) -> Void) {
        design.edit(name, undoManager: undoManager, change)
    }

    private func variable(_ p: ParamPath) -> Binding<Bool> {
        Binding(get: { design.isVariable(p) }, set: { design.setVariable(p, $0, undoManager: undoManager) })
    }
}

/// A number field inside a grid row (no label).
struct CompactNumber: View {
    let value: Double
    var parse: (String) -> Result<Double, LensValueParser.Failure> = LensValueParser.finite
    let commit: (Double) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField("", text: $text)
            .labelsHidden()
            .multilineTextAlignment(.trailing)
            .font(.body.monospacedDigit())
            .frame(minWidth: 52, maxWidth: 76)
            .focused($focused)
            .onSubmit(finish)
            .onChange(of: focused) { _, f in if f { text = NumberDisplay.exact(value) } else { finish() } }
            .onAppear { text = Fmt.short(value) }
            .onChange(of: value) { _, v in if !focused { text = Fmt.short(v) } }
    }

    private func finish() {
        guard text != NumberDisplay.exact(value), text != Fmt.short(value) else { return }
        if case .success(let v) = parse(text) { commit(v) } else { text = Fmt.short(value) }
    }
}
