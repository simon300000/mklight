import MKLightEngine
import MKLightModel
import SwiftUI

/// The settings of a pane, generated from the analysis' settings schema
/// (`SettingControls`): a two-column form, labels on the left, controls on
/// the right. Objects are groups of their properties, tagged unions a
/// variant picker, lists menus, chips or small tables; nothing is typed as
/// JSON. Unset settings show the engine default.
struct SettingsBar: View {
    let pane: AnalysisPane
    let descriptor: AnalysisDescriptor
    let system: OpticalSystem?

    var body: some View {
        let fields = visibleFields
        // The spectrum, the largest control, comes last on its own.
        let simple = fields.filter { control(of: $0) != .spectrum }
        let spectrum = fields.filter { control(of: $0) == .spectrum }
        VStack(alignment: .leading, spacing: Space.m) {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: Space.m, verticalSpacing: Space.s) {
                ForEach(simple) { f in
                    GridRow {
                        rowLabel(f)
                        SettingControlView(field: f, control: control(of: f), slot: slot(f), context: context)
                            .gridColumnAlignment(.leading)
                    }
                }
            }
            ForEach(spectrum) { f in
                Divider()
                SpectrumControl(
                    value: spectrumBinding(f), systemWavelengths: systemSpectrum, systemPrimary: system?.wavelengths.primary,
                    resolve: spectrumResolver)
            }
        }
        .controlSize(.small)
    }

    private func rowLabel(_ f: SettingsField) -> some View {
        Text(Self.label(f))
            .foregroundStyle(.secondary)
            .gridColumnAlignment(.trailing)
            .lineLimit(1)
            .help(f.description.isEmpty ? f.title : f.description)
            .accessibilityHidden(true)
    }

    /// The settings with a control.
    var visibleFields: [SettingsField] { descriptor.settingsFields.filter { control(of: $0) != .hidden } }

    /// The control of a top-level setting (`SettingControls`), given the
    /// system and the pane's pattern.
    func control(of f: SettingsField) -> SettingControl {
        SettingControls.control(
            for: f, analysis: pane.analysis, hasConfigurations: (system?.configurations.count ?? 0) > 0,
            pattern: descriptor.settingsFields.contains(where: { $0.name == "pattern" }) ? pane.value("pattern")?.string : nil)
    }

    /// The top-level value and setter of a setting.
    func slot(_ f: SettingsField) -> SettingSlot {
        SettingSlot(path: f.name, value: pane.value(f.name), fallback: f.defaultValue, set: { pane.set(f.name, $0) })
    }

    var context: SettingContext {
        SettingContext(
            analysis: pane.analysis, output: pane.output, fieldOptions: fieldOptions, wavelengthOptions: wavelengthOptions,
            surfaceOptions: surfaceOptions, glassNames: pane.analysis == "glass_info" ? pane.workspace?.materialNames ?? [] : [])
    }

    /// The values that differ from the engine's defaults, for a one-line
    /// summary under the pane's title.
    static func summary(_ pane: AnalysisPane, _ descriptor: AnalysisDescriptor?) -> String {
        guard let descriptor else { return "" }
        return descriptor.settingsFields.compactMap { f -> String? in
            guard let v = pane.settings[f.name], v != f.defaultValue, v != descriptor.defaultSettings[f.name] else { return nil }
            if f.name == "spectrum" { return SpectrumSetting(json: v).map { $0.kind == .design ? nil : "Spectrum \($0.summary)" } ?? "Spectrum custom" }
            if f.name == "optics", pane.analysis == "image_simulation" {
                let e = v["field_extent"]?.doubleValue ?? ImageSimulationSettings.fullExtent
                let c = v["field_center"]?.array?.compactMap(\.doubleValue) ?? []
                let at = c.count == 2 && (c[0] != 0 || c[1] != 0) ? " at " + ImageSimulationSettings.coordinates(SIMD2(c[0], c[1])) : ""
                return "View \(ImageSimulationSettings.extentTitle(e))" + at
            }
            if f.name == "pupil", let g = v["grid"]?.int { return g == 0 ? nil : "Pupil \(g)" }
            if f.name == "through_focus", pane.analysis == "spot_diagram" {
                return v == .null ? "Single plane" : ThroughFocusControl.summary(v, output: pane.output)
            }
            if let a = AutoSettings.sentinel(f.name, analysis: pane.analysis), AutoSettings.isAuto(v, a) { return "\(label(f)) auto" }
            if let t = arraySummary(f, v) { return "\(label(f)) \(t)" }
            if let variant = f.variant(of: v) { return "\(label(f)) \(variant.title.lowercased())" }
            let value: String =
                switch v["type"]?.string {
                case "all": "all"
                case "primary": "primary"
                case "index": "\((v["index"]?.int ?? 0) + 1)"
                case "list": "\(v["indices"]?.array?.count ?? 0)"
                default:
                    switch v {
                    case .null: "off"
                    case .object: "on"
                    case .array(let a): "\(a.count)"
                    case .bool(let b): b ? "on" : "off"
                    default: v.string.map { title(.string($0)) } ?? v.doubleValue.map(Fmt.short) ?? v.displayText
                    }
                }
            return "\(label(f)) \(value)"
        }.joined(separator: " · ")
    }

    /// The pane's `spectrum`: absent (or null) is the design's wavelengths.
    func spectrumBinding(_ f: SettingsField) -> Binding<JSONValue?> {
        Binding(get: { pane.settings[f.name].flatMap { $0 == .null ? nil : $0 } }, set: { pane.set(f.name, $0) })
    }

    var systemSpectrum: [SpectrumSample] {
        system?.wavelengths.wavelengths.map { SpectrumSample(um: $0.um, weight: $0.weight) } ?? []
    }

    /// The engine's resolution of a spectrum on the pane's snapshot
    /// (`system_summary`), for the preview.
    var spectrumResolver: SpectrumResolver? {
        guard let w = pane.workspace, let snapshot = w.snapshot else { return nil }
        let engine = w.documentEngine
        return { try await SpectrumSetting.resolve($0, engine: engine, snapshot: snapshot) }
    }

    var fieldOptions: [(Int, String)] {
        guard let s = system else { return [] }
        return s.fields.fields.enumerated().map { k, f in (k, "\(k + 1): " + FieldLabel.text([f.x, f.y], unit: s.fields.kind.unit)) }
    }

    var wavelengthOptions: [(Int, String)] {
        guard let s = system else { return [] }
        return s.wavelengths.wavelengths.enumerated().map { k, w in
            (k, String(format: "%.4f µm", w.um) + (k == s.wavelengths.primary ? " (primary)" : ""))
        }
    }

    var surfaceOptions: [(Int, String)] {
        guard let s = system else { return [] }
        return (1...max(s.imageIndex, 1)).map { i in
            (i, i == s.imageIndex ? "Image" : i == s.stop ? "\(i) (stop)" : "\(i)" + (s.surfaces[i].label.isEmpty ? "" : " " + s.surfaces[i].label))
        }
    }

    static func label(_ f: SettingsField) -> String {
        var name = f.name
        for suffix in ["_mm", "_um", "_deg"] where name.hasSuffix(suffix) { name.removeLast(suffix.count) }
        let t = f.title == f.name ? name.replacingOccurrences(of: "_", with: " ").capitalized : f.title
        return t.count > 22 ? String(t.prefix(20)) + "…" : t
    }

    static func title(_ v: JSONValue) -> String {
        guard case .string(let s) = v else { return v.displayText }
        if s.count <= 3 { return s.uppercased() }
        // A name with capitals (a glass such as N-SK16) is shown as written;
        // only snake_case identifiers are turned into words.
        if s.contains(where: \.isUppercase) { return s }
        return s.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

/// Where a control reads and writes its value: a top-level setting of the
/// pane, or a key of an enclosing object. `path` is dotted (`pupil.grid`),
/// for the automatic-setting lookups; `fallback` is the engine default
/// shown when nothing is set.
struct SettingSlot {
    var path: String
    var value: JSONValue?
    var fallback: JSONValue?
    var set: (JSONValue?) -> Void

    /// The value shown: the set one, else the default.
    var current: JSONValue? { value ?? fallback }

    /// The slot of `key` inside this object slot. Setting a child to nil
    /// restores the child's default inside the object (a partial object is
    /// never sent).
    func child(_ f: SettingsField) -> SettingSlot {
        let object = current?.object ?? [:]
        return SettingSlot(path: path + "." + f.name, value: object[f.name], fallback: f.defaultValue) { v in
            var o = (current?.object ?? [:])
            if let v { o[f.name] = v } else { o.removeValue(forKey: f.name); if let d = f.defaultValue { o[f.name] = d } }
            set(.object(o))
        }
    }
}

/// What the generated controls need from the pane and the system.
struct SettingContext {
    var analysis: String
    var output: AnalysisOutput?
    var fieldOptions: [(Int, String)] = []
    var wavelengthOptions: [(Int, String)] = []
    var surfaceOptions: [(Int, String)] = []
    /// Glass names to complete a glass `name` setting with.
    var glassNames: [String] = []

    func options(_ domain: SettingControl.IndexDomain) -> [(Int, String)] {
        switch domain {
        case .fields: fieldOptions
        case .wavelengths: wavelengthOptions
        case .surfaces: surfaceOptions
        }
    }
}

/// One setting's control, chosen by `control`; recursive for groups and
/// variants.
struct SettingControlView: View {
    let field: SettingsField
    let control: SettingControl
    let slot: SettingSlot
    let context: SettingContext

    var body: some View {
        let f = field, value = slot.current, label = SettingsBar.label(f)
        switch control {
        case .fieldSelection:
            SelectionMenu(label: "Fields", value: value, options: context.fieldOptions, allowsPrimary: false, set: { slot.set($0) })
        case .wavelengthSelection:
            SelectionMenu(label: "λ", value: value, options: context.wavelengthOptions, allowsPrimary: true, set: { slot.set($0) })
        case .fieldIndex:
            IndexMenu(label: "Field", value: value, options: context.fieldOptions, nullTitle: nil, set: { slot.set($0) })
        case .wavelengthIndex:
            IndexMenu(label: label, value: value, options: context.wavelengthOptions, nullTitle: f.name == "wavelength" ? "Primary" : "Auto", set: { slot.set($0) })
        case .surfaceIndex:
            IndexMenu(label: "Surface", value: value, options: context.surfaceOptions, nullTitle: nil, set: { slot.set($0) })
        case .picker:
            Picker(label, selection: Binding(get: { value ?? .null }, set: { slot.set($0) })) {
                ForEach(f.enumValues, id: \.self) { v in Text(SettingsBar.title(v)).tag(v) }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .fixedSize()
            .help(f.description)
            .accessibilityLabel(label)
        case .toggle:
            Toggle(label, isOn: Binding(get: { value?.bool ?? false }, set: { slot.set(.bool($0)) }))
                .labelsHidden()
                .help(f.description)
                .accessibilityLabel(label)
        case .number:
            SettingNumberField(
                field: f, label: "", value: value, auto: AutoSettings.sentinel(slot.path, analysis: context.analysis),
                autoUsed: AutoSettings.used(slot.path, analysis: context.analysis, output: context.output), set: { slot.set($0) }
            )
            .accessibilityLabel(label)
        case .text:
            SettingTextField(
                field: f, label: label, value: value, completions: f.name == "name" ? context.glassNames : [], set: { slot.set($0) })
        case .pair(let integer, let nullable):
            SettingPairField(field: f, label: label, value: value, integer: integer, nullable: nullable, set: { slot.set($0) })
        case .indexList(let domain):
            IndexListMenu(label: label, value: value, options: context.options(domain), help: f.description, set: { slot.set($0) })
        case .numberList:
            NumberListField(field: f, label: label, value: value, set: { slot.set($0) })
        case .table(let columns, let rowsAreArrays):
            TableSettingControl(field: f, label: label, columns: columns, rowsAreArrays: rowsAreArrays, value: value, set: { slot.set($0) })
        case .group(let nullable):
            SettingGroupControl(field: f, nullable: nullable, slot: slot, context: context)
        case .variants(let nullable):
            VariantControl(field: f, nullable: nullable, slot: slot, context: context)
        case .pupil:
            PupilSamplingControl(field: f, slot: slot, context: context)
        case .frequencies:
            FrequencyListSetting(
                field: f, value: value, autoUsed: AutoSettings.used(f.name, analysis: context.analysis, output: context.output), set: { slot.set($0) })
        case .throughFocus:
            ThroughFocusControl(field: f, value: value, output: context.output, set: { slot.set($0) })
        case .sensorLimit:
            SensorLimitControl(value: value, help: f.description, set: { slot.set($0) })
        case .imageSource, .imageOptics:
            SpecialImageControl(field: f, control: control, context: context)
        case .spectrum, .hidden:
            EmptyView()
        }
    }
}

/// The settings inside an object: its properties as rows, indented under
/// the group's label; an optional object gets an on/off switch first.
struct SettingGroupControl: View {
    let field: SettingsField
    let nullable: Bool
    let slot: SettingSlot
    let context: SettingContext

    private var isOn: Bool { !nullable || (slot.current.map { $0 != .null } ?? false) }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            if nullable {
                Toggle(isOn: Binding(get: { isOn }, set: { slot.set($0 ? SettingControls.enabledValue(field) : .null) })) {
                    Text(isOn ? "On" : "Off").foregroundStyle(.secondary)
                }
                .help(field.description)
                .accessibilityLabel(SettingsBar.label(field))
            }
            if isOn {
                NestedSettings(fields: field.properties, slot: slot, context: context)
            }
        }
    }
}

/// The rows of an object's properties (groups and variants).
struct NestedSettings: View {
    let fields: [SettingsField]
    let slot: SettingSlot
    let context: SettingContext

    var body: some View {
        Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: Space.m, verticalSpacing: Space.xs) {
            ForEach(fields.filter { $0.constValue == nil }) { f in
                let control = SettingControls.control(for: f, analysis: context.analysis, nested: true)
                if control != .hidden {
                    GridRow {
                        Text(SettingsBar.label(f))
                            .foregroundStyle(.secondary)
                            .gridColumnAlignment(.trailing)
                            .lineLimit(1)
                            .help(f.description.isEmpty ? f.title : f.description)
                        SettingControlView(field: f, control: control, slot: slot.child(f), context: context)
                            .gridColumnAlignment(.leading)
                    }
                }
            }
        }
    }
}

/// A tagged union: a picker of the variants (with "Off" when null is
/// allowed), then the chosen variant's own settings.
struct VariantControl: View {
    let field: SettingsField
    let nullable: Bool
    let slot: SettingSlot
    let context: SettingContext

    private var current: SettingsField.Variant? { field.variant(of: slot.current) }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            Picker(SettingsBar.label(field), selection: Binding(get: { current?.id ?? "" }, set: choose)) {
                if nullable { Text("Off").tag("") }
                ForEach(field.variants) { v in Text(v.title).tag(v.id) }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .fixedSize()
            .help(field.description)
            .accessibilityLabel(SettingsBar.label(field))
            if let v = current, v.properties.contains(where: { $0.constValue == nil }) {
                NestedSettings(fields: v.properties, slot: slot, context: context)
            }
        }
    }

    /// Switches the variant, keeping the values of properties both have.
    private func choose(_ id: String) {
        guard let v = field.variants.first(where: { $0.id == id }) else {
            slot.set(.null)
            return
        }
        var o = v.defaultObject.object ?? [:]
        for (k, x) in slot.current?.object ?? [:] where o[k] != nil && v.properties.first { $0.name == k }?.constValue == nil { o[k] = x }
        slot.set(.object(o))
    }
}

/// `{"type": "all" | "primary" | "index" | "list"}` selections as a menu.
struct SelectionMenu: View {
    let label: String
    let value: JSONValue?
    let options: [(Int, String)]
    let allowsPrimary: Bool
    let set: (JSONValue) -> Void

    var body: some View {
        Menu {
            Button("All") { set(.object(["type": .string("all")])) }
            if allowsPrimary { Button("Primary") { set(.object(["type": .string("primary")])) } }
            Divider()
            ForEach(options, id: \.0) { k, title in
                Button(title) { set(.object(["type": .string("index"), "index": .integer(Int64(k))])) }
            }
        } label: {
            Text(current)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .accessibilityLabel(label == "λ" ? "Wavelengths" : label)
        .accessibilityValue(current)
    }

    private var current: String {
        switch value?["type"]?.string {
        case "primary": return "Primary"
        case "index":
            let i = value?["index"]?.int ?? 0
            return options.first { $0.0 == i }.map { String($0.1.split(separator: ":").first ?? "") } ?? "\(i + 1)"
        case "list": return "\(value?["indices"]?.array?.count ?? 0) selected"
        default: return "All"
        }
    }
}

/// An index setting (`integer` or `integer|null`) as a menu.
struct IndexMenu: View {
    let label: String
    let value: JSONValue?
    let options: [(Int, String)]
    let nullTitle: String?
    let set: (JSONValue) -> Void

    var body: some View {
        Menu {
            if let nullTitle { Button(nullTitle) { set(.null) } }
            ForEach(options, id: \.0) { k, title in Button(title) { set(.integer(Int64(k))) } }
        } label: {
            Text(current)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .accessibilityLabel(label)
        .accessibilityValue(current)
    }

    private var current: String {
        guard let i = value?.int else { return nullTitle ?? "—" }
        let title = options.first { $0.0 == i }?.1 ?? "\(i)"
        return String(title.split(separator: ":").first ?? Substring(title))
    }
}

/// A list of indices (`[i, …]`, empty = all) as a menu of check marks.
struct IndexListMenu: View {
    let label: String
    let value: JSONValue?
    let options: [(Int, String)]
    let help: String
    let set: (JSONValue?) -> Void

    private var selected: Set<Int> { Set(value?.array?.compactMap(\.int) ?? []) }

    var body: some View {
        Menu {
            Button("All") { set(.array([])) }
            Divider()
            ForEach(options, id: \.0) { k, title in
                Toggle(title, isOn: Binding(get: { selected.contains(k) }, set: { on in
                    var s = selected
                    if on { s.insert(k) } else { s.remove(k) }
                    set(.array(s.sorted().map { .integer(Int64($0)) }))
                }))
            }
        } label: {
            Text(current)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help(help)
        .accessibilityLabel(label)
        .accessibilityValue(current)
    }

    private var current: String {
        let s = selected
        if s.isEmpty || s.count == options.count && !options.isEmpty { return "All" }
        if s.count == 1, let k = s.first, let t = options.first(where: { $0.0 == k }) { return String(t.1.split(separator: ":").first ?? Substring(t.1)) }
        return "\(s.count) of \(options.count)"
    }
}

/// A free text setting (a glass name, a catalogue); empty restores the
/// default, or null when allowed. Only edited text is committed.
struct SettingTextField: View {
    let field: SettingsField
    let label: String
    let value: JSONValue?
    /// Values to suggest while typing (glass names).
    var completions: [String] = []
    let set: (JSONValue?) -> Void

    var body: some View {
        CommittingField(value: value, format: entry, commit: set) { text, focused in
            TextField(field.nullable ? "any" : label, text: text)
                .textFieldStyle(.roundedBorder)
                .frame(width: 140)
                .materialSuggestions(focused ? suggestions(text.wrappedValue) : [])
        }
        .help(field.description.isEmpty ? field.title : field.description)
        .accessibilityLabel(label)
    }

    private func suggestions(_ text: String) -> [String] {
        let q = text.trimmingCharacters(in: .whitespaces).uppercased()
        guard !q.isEmpty else { return [] }
        return Array(completions.filter { $0.uppercased().hasPrefix(q) && $0.uppercased() != q }.prefix(12))
    }

    private var entry: FieldEdit<JSONValue?>.Format {
        let text: (JSONValue?) -> String = { $0?.string ?? "" }
        return FieldEdit.Format(
            exact: text, display: text,
            parse: { [field, value] typed in
                let t = typed.trimmingCharacters(in: .whitespaces)
                if !t.isEmpty { return .some(.string(t)) }
                // A setting already at its default keeps it.
                return .some(field.nullable ? .null : value == field.defaultValue ? value : nil)
            })
    }
}

/// A compact number setting; empty text restores the default (or null).
/// An automatic setting (`auto`: 0 or null asks the engine to choose)
/// shows "Auto (n)" with the value the last run used, never a bare 0;
/// typing "auto" or nothing makes it automatic again, as does 0 where 0
/// is what asks the engine. Only edited text is committed.
struct SettingNumberField: View {
    let field: SettingsField
    let label: String
    let value: JSONValue?
    var auto: AutoSettings.Sentinel? = nil
    var autoUsed: String? = nil
    let set: (JSONValue?) -> Void

    var body: some View {
        HStack(spacing: Space.xs) {
            if !label.isEmpty { Text(label).foregroundStyle(.secondary) }
            CommittingField(value: value, format: entry, commit: set) { text, _ in
                TextField(placeholder, text: text)
                    .textFieldStyle(.roundedBorder)
                    .numeric()
                    .multilineTextAlignment(.trailing)
                    .frame(width: isAuto ? max(64, CGFloat(placeholder.count) * 7 + 8) : 64)
            }
            if !field.unit.isEmpty { Text(field.unit).foregroundStyle(.secondary) }
        }
        .help(helpText)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label.isEmpty ? SettingsBar.label(field) : label)
        .accessibilityValue(isAuto ? placeholder : display)
    }

    private var isInteger: Bool { field.types.contains("integer") && !field.types.contains("number") }
    private var nullable: Bool { field.types.contains("null") }
    private var isAuto: Bool { auto.map { AutoSettings.isAuto(value, $0) } ?? false }
    private var placeholder: String { auto != nil ? AutoSettings.display(used: autoUsed) : nullable ? "auto" : "" }

    private var helpText: String {
        var h = field.description.isEmpty ? field.title : field.description
        if auto != nil {
            h +=
                isAuto
                ? "\nAutomatic: the engine chooses from the system."
                : auto == .zero ? "\nClear the field (or type 0) to let the engine choose." : "\nClear the field to let the engine choose."
        }
        return h
    }

    /// The text of `value`, empty for an automatic one: compact, or with
    /// `exact` (the focused field) a text that reads back as the value.
    static func display(_ value: JSONValue?, isInteger: Bool, auto: AutoSettings.Sentinel?, exact: Bool = false) -> String {
        if let auto, AutoSettings.isAuto(value, auto) { return "" }
        guard let v = value, let d = v.doubleValue else { return "" }
        return isInteger ? String(Int(saturating: d)) : exact ? FieldText.exact(d) : Fmt.short(d)
    }

    private var display: String { Self.display(value, isInteger: isInteger, auto: auto) }

    /// The typed setting: the automatic value for nothing, "auto", or the
    /// 0 of a setting that is automatic at 0; else the number, held to the
    /// setting's range.
    var entry: FieldEdit<JSONValue?>.Format {
        FieldEdit.Format(
            exact: { [isInteger, auto] in Self.display($0, isInteger: isInteger, auto: auto, exact: true) },
            display: { [isInteger, auto] in Self.display($0, isInteger: isInteger, auto: auto) },
            parse: { [field, value, auto, isInteger, nullable] text in
                let t = text.trimmingCharacters(in: .whitespaces)
                let typed = LensValueParser.number(t)
                if t.isEmpty || (auto != nil && t.lowercased().hasPrefix("auto")) || (auto == .zero && typed == 0) {
                    let cleared: JSONValue? =
                        switch auto {
                        case .zero?: .integer(0)
                        case .null?: .null
                        default: nullable ? .null : nil
                        }
                    // A setting already at its default keeps it.
                    return .some(cleared == nil && value == field.defaultValue ? value : cleared)
                }
                guard var d = typed, d.isFinite else { return nil }
                if let lo = field.minimum { d = max(d, lo) }
                if let hi = field.maximum { d = min(d, hi) }
                return .some(isInteger ? .integer(Int64(Int(saturating: d))) : .number(d))
            })
    }
}

/// The pupil sampling of the wave analyses (`pupil`): the grid, automatic
/// by default ("Auto (128)"), and the traced rays per diameter. The other
/// pupil keys are kept as they are.
struct PupilSamplingControl: View {
    let field: SettingsField
    let slot: SettingSlot
    let context: SettingContext

    var body: some View {
        HStack(spacing: Space.s) {
            SettingNumberField(
                field: Self.field("grid", "Pupil grid samples per side; 0 chooses from the wavefront slope (64 to 1024)."),
                label: "Grid", value: slot.current?["grid"], auto: .zero,
                autoUsed: AutoSettings.used(slot.path + ".grid", analysis: context.analysis, output: context.output)
            ) { set("grid", $0) }
            SettingNumberField(
                field: Self.field("rays", "Rays traced per pupil diameter (the lattice the wavefront is interpolated from)."),
                label: "Rays", value: slot.current?["rays"]
            ) { set("rays", $0) }
        }
    }

    static func field(_ name: String, _ help: String) -> SettingsField {
        var f = SettingsField(name: name, schema: .object(["type": .string("integer"), "description": .string(help)]), required: false)
        f.minimum = 0
        return f
    }

    private func set(_ key: String, _ v: JSONValue?) {
        var p = slot.current?.object ?? [:]
        p[key] = v ?? field.defaultValue?[key]
        slot.set(.object(p))
    }
}

/// Numbers in a comma-separated field (`[x, …]`), kept in the order typed;
/// empty is the empty list. Only edited text is committed.
struct NumberListField: View {
    let field: SettingsField
    let label: String
    let value: JSONValue?
    let set: (JSONValue?) -> Void

    private var isInteger: Bool { field.items?["type"]?.string == "integer" }

    var body: some View {
        HStack(spacing: Space.xs) {
            CommittingField(value: value, format: entry, commit: set) { text, _ in
                TextField(field.nullable ? "auto" : "none", text: text)
                    .textFieldStyle(.roundedBorder)
                    .numeric()
                    .frame(width: max(96, CGFloat(text.wrappedValue.count) * 7 + 8))
            }
            if !field.unit.isEmpty { Text(field.unit).foregroundStyle(.secondary) }
        }
        .help(field.description + "\nComma separated.")
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label)
        .accessibilityValue(entry.display(value))
    }

    private var entry: FieldEdit<JSONValue?>.Format {
        func text(_ value: JSONValue?, exact: Bool) -> String {
            let v = value?.array?.compactMap(\.doubleValue) ?? []
            return isInteger ? v.map { String(Int(saturating: $0)) }.joined(separator: ", ") : FrequencyListSetting.text(v, exact: exact)
        }
        return FieldEdit.Format(
            exact: { text($0, exact: true) }, display: { text($0, exact: false) },
            parse: { [field, isInteger] typed in
                let t = typed.trimmingCharacters(in: .whitespaces)
                if t.isEmpty { return .some(field.nullable ? .null : .array([])) }
                guard let v = FrequencyListSetting.numbers(t) else { return nil }
                return .some(.array(v.map { isInteger ? JSONValue.integer(Int64(Int(saturating: $0))) : .number($0) }))
            })
    }
}

/// Fixed MTF frequencies (cycles/mm), comma separated; empty lets the
/// engine choose three round 1-2-5 frequencies ("Auto (2, 5, 10)"). Only
/// edited text is committed.
struct FrequencyListSetting: View {
    let field: SettingsField
    let value: JSONValue?
    let autoUsed: String?
    let set: (JSONValue?) -> Void

    private var values: [Double] { value?.array?.compactMap(\.doubleValue) ?? [] }
    private var placeholder: String { AutoSettings.display(used: autoUsed) }

    var body: some View {
        HStack(spacing: Space.xs) {
            CommittingField(value: values, format: Self.entry, commit: { set(.array($0.map(JSONValue.number))) }) { text, _ in
                TextField(placeholder, text: text)
                    .textFieldStyle(.roundedBorder)
                    .numeric()
                    .frame(width: max(96, CGFloat(max(placeholder.count, text.wrappedValue.count)) * 7 + 8))
            }
            Text(field.unit.isEmpty ? "cycles/mm" : field.unit).foregroundStyle(.secondary)
        }
        .help(field.description + "\nComma separated; empty lets the engine choose.")
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Frequencies")
        .accessibilityValue(values.isEmpty ? placeholder : Self.text(values))
    }

    /// "10, 20, 30": compact, or with `exact` (a focused field) texts that
    /// read back as the values.
    static func text(_ v: [Double], exact: Bool = false) -> String {
        v.map(exact ? FieldText.exact : Fmt.short).joined(separator: ", ")
    }

    /// The finite numbers of "10, 20 30" (commas, semicolons or spaces), in
    /// the order typed; `nil` when a part is not one.
    static func numbers(_ t: String) -> [Double]? {
        let parts = t.split(whereSeparator: { $0 == "," || $0 == ";" || $0 == " " })
        let v = parts.compactMap { LensValueParser.number(String($0)) }
        return v.count == parts.count && v.allSatisfy(\.isFinite) ? v : nil
    }

    /// The frequencies of "10, 20 30": positive, sorted and unique.
    static func parse(_ t: String) -> [Double]? {
        guard let v = numbers(t), v.allSatisfy({ $0 > 0 }) else { return nil }
        return Array(Set(v)).sorted()
    }

    /// Nothing or "auto" is the empty list (the engine chooses).
    static let entry = FieldEdit<[Double]>.Format(
        exact: { text($0, exact: true) }, display: { text($0) },
        parse: { typed in
            let t = typed.trimmingCharacters(in: .whitespaces)
            if t.isEmpty || t.lowercased().hasPrefix("auto") { return [] }
            return parse(t).flatMap { $0.isEmpty ? nil : $0 }
        })
}

/// The spot diagram's through-focus sampling: off (one plane), or planes
/// from −Δ to +Δ with Δ automatic ("Auto (±0.08)") or typed, and the number
/// of planes. A range that is not symmetric (from a file) shows its bounds.
struct ThroughFocusControl: View {
    let field: SettingsField
    let value: JSONValue?
    let output: AnalysisOutput?
    let set: (JSONValue?) -> Void

    private var isOn: Bool { value.map { $0 != .null } ?? false }
    private var min: Double? { value?["min"]?.doubleValue }
    private var max: Double? { value?["max"]?.doubleValue }
    private var steps: Int { value?["steps"]?.int ?? 3 }
    private var symmetric: Bool { (min == nil && max == nil) || (min != nil && max != nil && abs(min! + max!) <= 1e-12 * Swift.max(1, abs(max!))) }

    /// "±0.08 mm · 3 planes" (the range the run used when automatic).
    static func summary(_ v: JSONValue, output: AnalysisOutput?) -> String {
        let steps = v["steps"]?.int ?? 3
        let planes = steps == 1 ? "1 plane" : "\(steps) planes"
        if let a = v["min"]?.doubleValue, let b = v["max"]?.doubleValue {
            if abs(a + b) <= 1e-12 * Swift.max(1, abs(b)) { return "±\(Fmt.short(b)) mm · " + planes }
            return "\(Fmt.short(a)) to \(Fmt.short(b)) mm · " + planes
        }
        if let used = AutoSettings.used("through_focus", analysis: "spot_diagram", output: output) { return "\(used) mm · " + planes }
        return "Auto range · " + planes
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            Toggle(isOn: Binding(get: { isOn }, set: { on in set(on ? .object(["min": .null, "max": .null, "steps": .integer(Int64(steps))]) : .null) })) {
                Text(isOn ? "Planes before and after focus" : "Image surface only").foregroundStyle(.secondary)
            }
            .help(field.description)
            .accessibilityLabel("Through focus")
            if isOn {
                HStack(spacing: Space.s) {
                    if symmetric {
                        Text("±").foregroundStyle(.secondary)
                        SettingNumberField(
                            field: bound("max"), label: "", value: max.map(JSONValue.number) ?? .null, auto: .null,
                            autoUsed: AutoSettings.used("through_focus", analysis: "spot_diagram", output: output)?.replacingOccurrences(of: "±", with: "")
                        ) { v in
                            if let d = v?.doubleValue { update(min: -abs(d), max: abs(d)) } else { update(min: nil, max: nil) }
                        }
                        .accessibilityLabel("Focus range")
                    } else {
                        SettingNumberField(field: bound("min"), label: "", value: min.map(JSONValue.number) ?? .null, auto: .null) { v in
                            update(min: v?.doubleValue, max: max)
                        }
                        .accessibilityLabel("First plane")
                        Text("to").foregroundStyle(.secondary)
                        SettingNumberField(field: bound("max"), label: "", value: max.map(JSONValue.number) ?? .null, auto: .null) { v in
                            update(min: min, max: v?.doubleValue)
                        }
                        .accessibilityLabel("Last plane")
                    }
                    Text("mm").foregroundStyle(.secondary)
                    Stepper(value: Binding(get: { steps }, set: { update(steps: $0) }), in: 1...15) {
                        Text("\(steps) planes").numeric()
                    }
                    .help("Planes from the first to the last; 3 shows before, on and after focus.")
                    .accessibilityLabel("Planes")
                }
            }
        }
    }

    private func bound(_ key: String) -> SettingsField {
        SettingsField(name: key, schema: .object(["type": .array([.string("number"), .string("null")]), "description": .string("Half-range of the planes; empty lets the engine choose from the depth of focus and the geometric blur.")]), required: false)
    }

    private func update(min newMin: Double? = nil, max newMax: Double? = nil, steps newSteps: Int? = nil) {
        var o = value?.object ?? [:]
        if newSteps == nil {
            o["min"] = newMin.map(JSONValue.number) ?? .null
            o["max"] = newMax.map(JSONValue.number) ?? .null
        } else {
            o["min"] = min.map(JSONValue.number) ?? .null
            o["max"] = max.map(JSONValue.number) ?? .null
        }
        o["steps"] = .integer(Int64(newSteps ?? steps))
        set(.object(o))
    }
}

/// The image simulation's source and optics keep their dedicated controls;
/// they need the pane itself.
struct SpecialImageControl: View {
    let field: SettingsField
    let control: SettingControl
    let context: SettingContext
    @Environment(\.analysisPane) private var pane

    var body: some View {
        if let pane {
            switch control {
            case .imageSource: ImageSourceControl(pane: pane, field: field)
            case .imageOptics: ImageOpticsControl(pane: pane, field: field)
            default: EmptyView()
            }
        }
    }
}

extension EnvironmentValues {
    /// The pane whose settings are being edited.
    @Entry var analysisPane: AnalysisPane?
}

/// Text of a JSON value, for messages about values a control cannot edit.
enum JSONText {
    static func text(_ v: JSONValue?) -> String {
        guard let v, let d = try? MKLightJSON.encode(v) else { return "null" }
        return String(decoding: d, as: UTF8.self)
    }
}

/// Lays children out in rows, wrapping when the width runs out.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    var lineSpacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, line: CGFloat = 0, maxX: CGFloat = 0
        for s in subviews {
            let size = Self.size(s, width: width)
            if x > 0 && x + size.width > width {
                y += line + lineSpacing
                x = 0
                line = 0
            }
            x += size.width + spacing
            maxX = max(maxX, x - spacing)
            line = max(line, size.height)
        }
        return CGSize(width: proposal.width ?? maxX, height: y + line)
    }

    /// The item's ideal size, or the line's width when it is wider (its
    /// text then wraps instead of running out of the pane).
    static func size(_ s: LayoutSubview, width: CGFloat) -> CGSize {
        let ideal = s.sizeThatFits(.unspecified)
        guard width.isFinite, ideal.width > width else { return ideal }
        return s.sizeThatFits(ProposedViewSize(width: width, height: nil))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, line: CGFloat = 0
        for s in subviews {
            let size = Self.size(s, width: bounds.width)
            if x > bounds.minX && x + size.width > bounds.maxX {
                y += line + lineSpacing
                x = bounds.minX
                line = 0
            }
            s.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            line = max(line, size.height)
        }
    }
}
