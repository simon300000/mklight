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
    @Environment(\.openWindow) private var openWindow
    @Environment(\.designSession) private var designSession

    /// Where a catalog comes from, for its row.
    static func origin(of c: CatalogInfo) -> String {
        switch (c.isBuiltin, c.library, c.inProject) {
        case (true, _, _): String(localized: "Built into MKLight")
        case (_, true, true): String(localized: "Library, recorded in the design")
        case (_, true, false): String(localized: "Library")
        default: String(localized: "This design only (not in the library)")
        }
    }

    private func showBrowser(_ catalog: String?) {
        if let catalog { GlassBrowserModel.shared.scope = .catalog(catalog) }
        openWindow(id: GlassBrowserView.windowID)
    }

    private func reload(_ c: CatalogInfo) {
        Task {
            do {
                let changes = try await workspace.reloadCatalog(named: c.name)
                window.flash(String(localized: "\(c.name) reloaded: \(changes.localizedSummary.lowercased()).", locale: NumberDisplay.locale))
            } catch {
                window.flash(String(localized: "\(c.name): \(error.localizedDescription)", locale: NumberDisplay.locale))
            }
        }
    }

    var body: some View {
        Form {
            Section {
                TextValueField(title: String(localized: "Name"), value: system.name) { v in edit(String(localized: "Rename")) { $0.name = v } }
                TextValueField(title: String(localized: "Notes"), value: system.notes, prompt: String(localized: "none")) { v in edit(String(localized: "Change Notes")) { $0.notes = v } }
            }
            Section("Aperture") {
                Picker("Type", selection: Binding(get: { system.aperture.kind }, set: setApertureKind)) {
                    ForEach(ApertureSpec.Kind.allCases) { Text($0.title).tag($0) }
                }
                if let v = system.aperture.value {
                    NumberField(
                        title: String(localized: "Value"), value: v, unit: system.aperture.kind.unit, parse: LensValueParser.positive,
                        variable: variable(.apertureValue)
                    ) { nv in edit(String(localized: "Change Aperture")) { $0.aperture = ApertureSpec(kind: $0.aperture.kind, value: nv) } }
                }
            }
            anamorphicSection
            fieldsSection
            wavelengthsSection
            Section("Environment") {
                NumberField(title: String(localized: "Temperature"), value: system.environment.temperatureC, unit: "°C") { v in
                    edit(String(localized: "Change Temperature")) { $0.environment.temperatureC = v }
                }
                NumberField(title: String(localized: "Pressure"), value: system.environment.pressureAtm, unit: "atm", parse: LensValueParser.positive) { v in
                    edit(String(localized: "Change Pressure")) { $0.environment.pressureAtm = v }
                }
            }
            Section {
                Toggle("Real ray aiming", isOn: Binding(get: { system.rayAiming != .off }, set: { on in
                    edit(String(localized: "Change Ray Aiming")) { $0.rayAiming = on ? .real(tolerance: RayAiming.defaultTolerance) : .off }
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
                        LabeledContent(c.name.isEmpty ? String(localized: "Configuration \(k + 1)", locale: NumberDisplay.locale) : c.name) {
                            Text(c.overrides.count == 1 ? String(localized: "1 override") : String(localized: "\(c.overrides.count) overrides", locale: NumberDisplay.locale)).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            Section {
                ForEach(workspace.catalogs) { c in
                    LabeledContent {
                        Text(glassCountText(c.glassCount)).foregroundStyle(.secondary)
                    } label: {
                        Text(c.isBuiltin ? String(localized: "Built-in") : c.name)
                        Text(Self.origin(of: c)).font(.caption).foregroundStyle(.secondary)
                    }
                    .help(c.path ?? "")
                    .contextMenu {
                        Button("Show in Glass Catalogs") { showBrowser(c.name) }
                        if c.path != nil {
                            Button("Reload from Disk") { reload(c) }
                        }
                        if c.inProject {
                            Button("Remove from Design") { design.removeCatalog(named: c.name, undoManager: undoManager) }
                        }
                    }
                }
                ForEach(workspace.catalogWarnings + workspace.libraryWarnings, id: \.self) { w in
                    WarningNote(w)
                }
                HStack {
                    Button("Glass Catalogs…") { showBrowser(nil) }
                    Button("Add to Design…") { designSession?.requestCatalog() }
                }
            } header: {
                Text("Glass Catalogs")
            } footer: {
                Text("Library catalogs load into every design; a design records the ones it uses when saved. Right-click a catalog to reload or remove it.")
                    .font(.caption).foregroundStyle(.secondary)
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
                        .help(String(localized: "Squeeze: \(xy.squeezeDefinition.map { EngineText.string($0) } ?? String(localized: "signed EFL y / EFL x"))", locale: NumberDisplay.locale))
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
            Picker("Type", selection: Binding(get: { system.fields.kind }, set: setFieldKind)) {
                ForEach(FieldKind.allCases) { Text($0.title).tag($0) }
            }
            Grid(alignment: .trailing, horizontalSpacing: Space.m, verticalSpacing: Space.xs) {
                GridRow {
                    Text("")
                    Text(String(localized: "X (\(system.fields.kind.unit))", locale: NumberDisplay.locale))
                    Text(String(localized: "Y (\(system.fields.kind.unit))", locale: NumberDisplay.locale))
                    Text("Weight")
                    Text("")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(Array(system.fields.fields.enumerated()), id: \.offset) { k, f in
                    GridRow {
                        HStack(spacing: Space.xs) {
                            Circle().fill(Palette.fieldColor(k)).frame(width: 7, height: 7)
                            Text(String(localized: "\(k + 1)", locale: NumberDisplay.locale)).numeric()
                        }
                        .accessibilityElement(children: .combine)
                        .accessibilityLabel(String(localized: "Field \(k + 1)", locale: NumberDisplay.locale))
                        CompactNumber(value: f.x) { v in edit(String(localized: "Change Field")) { $0.fields.fields[k].x = v } }
                        CompactNumber(value: f.y) { v in edit(String(localized: "Change Field")) { $0.fields.fields[k].y = v } }
                        CompactNumber(value: f.weight, parse: LensValueParser.finite) { v in edit(String(localized: "Change Field Weight")) { $0.fields.fields[k].weight = v } }
                        Button("Remove Field", systemImage: "minus.circle") {
                            edit(String(localized: "Remove Field")) { $0.removeField(at: k) }
                        }
                        .labelStyle(.iconOnly).buttonStyle(.borderless).foregroundStyle(.secondary)
                        .disabled(system.fields.fields.count <= 1)
                    }
                    .padding(.vertical, 1)
                    .background(window.highlightedField == k ? Color.accentColor.opacity(0.15) : .clear)
                }
            }
            Button {
                edit(String(localized: "Add Field")) { s in
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
        (String(localized: "F, d, C (visible)"), [(0.4861327, 1), (0.5875618, 1), (0.6562725, 1)], 1),
        (String(localized: "F′, e, C′ (visible)"), [(0.4799914, 1), (0.5460740, 1), (0.6438469, 1)], 1),
        (String(localized: "g, F, e, d, C"), [(0.4358343, 1), (0.4861327, 1), (0.5460740, 1), (0.5875618, 1), (0.6562725, 1)], 3),
        (String(localized: "d only"), [(0.5875618, 1)], 0),
        (String(localized: "e only"), [(0.5460740, 1)], 0),
        (String(localized: "Photopic (5)"), [(0.47, 0.091), (0.51, 0.503), (0.555, 1), (0.61, 0.503), (0.65, 0.107)], 2),
        (String(localized: "Near infrared"), [(0.75, 1), (0.85, 1), (0.95, 1)], 1),
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
                            edit(String(localized: "Change Primary Wavelength")) { $0.wavelengths.primary = k }
                        } label: {
                            Image(systemName: system.wavelengths.primary == k ? "largecircle.fill.circle" : "circle")
                                .foregroundStyle(Palette.wavelengthColor(w.um))
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel(system.wavelengths.primary == k ? String(localized: "Primary wavelength") : String(localized: "Make primary"))
                        CompactNumber(value: w.um, parse: LensValueParser.positive) { v in edit(String(localized: "Change Wavelength")) { $0.wavelengths.wavelengths[k].um = v } }
                        CompactNumber(value: w.weight, parse: LensValueParser.finite) { v in edit(String(localized: "Change Wavelength Weight")) { $0.wavelengths.wavelengths[k].weight = v } }
                        Button("Remove Wavelength", systemImage: "minus.circle") {
                            edit(String(localized: "Remove Wavelength")) { $0.removeWavelength(at: k) }
                        }
                        .labelStyle(.iconOnly).buttonStyle(.borderless).foregroundStyle(.secondary)
                        .disabled(system.wavelengths.wavelengths.count <= 1)
                    }
                }
            }
            HStack {
                Button {
                    edit(String(localized: "Add Wavelength")) { s in s.wavelengths.wavelengths.append(Wavelength(um: (s.wavelengths.wavelengths.last?.um ?? 0.55) + 0.05)) }
                } label: {
                    Label("Add Wavelength", systemImage: "plus")
                }
                .buttonStyle(.borderless)
                Spacer()
                Menu("Presets") {
                    ForEach(Self.presets, id: \.0) { name, list, primary in
                        Button(name) {
                            edit(String(localized: "Apply Wavelength Preset")) { s in
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

    /// Switches the field kind, converting every field's value so it keeps
    /// its chief ray (the engine's `convert_spec`). Where the engine cannot
    /// convert (e.g. an object height at infinity) the numbers stay and
    /// validation says why.
    private func setFieldKind(_ k: FieldKind) {
        let before = system
        Task {
            let c = await workspace.convertSpec(of: before, fieldKind: k)
            edit(String(localized: "Change Field Type")) { $0.setFieldKind(k, conversion: $0 == before ? c : nil) }
        }
    }

    /// Switches the aperture type, converting the value so the paraxial
    /// marginal ray stays (see `setFieldKind`).
    private func setApertureKind(_ k: ApertureSpec.Kind) {
        let before = system
        Task {
            let c = await workspace.convertSpec(of: before, aperture: k)
            edit(String(localized: "Change Aperture Type")) { $0.setApertureKind(k, conversion: $0 == before ? c : nil) }
        }
    }

    private func edit(_ name: String, _ change: (inout OpticalSystem) -> Void) {
        design.edit(name, undoManager: undoManager, change)
    }

    private func variable(_ p: ParamPath) -> FieldVariable {
        FieldVariable(
            isOn: Binding(get: { design.isVariable(p) }, set: { design.setVariable(p, $0, undoManager: undoManager) }),
            solve: design.system.solve(for: p)?.kind.kind.title)
    }
}

/// A number field inside a grid row (no label): shows the formatted value,
/// edits the exact one, and commits on Return or when focus leaves, if the
/// text was edited; rejected text reverts.
struct CompactNumber: View {
    let value: Double
    var parse: (String) -> Result<Double, LensValueParser.Failure> = LensValueParser.finite
    let commit: (Double) -> Void

    var body: some View {
        CommittingField(value: value, format: entry, commit: commit) { text, _ in
            TextField("", text: text)
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .font(.body.monospacedDigit())
                .frame(minWidth: 52, maxWidth: 76)
        }
    }

    private var entry: FieldEdit<Double>.Format {
        FieldEdit.Format(exact: NumberDisplay.exact, display: Fmt.short, parse: { [parse] in try? parse($0).get() })
    }
}
