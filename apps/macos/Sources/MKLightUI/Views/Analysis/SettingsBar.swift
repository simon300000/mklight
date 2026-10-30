import MKLightEngine
import MKLightModel
import SwiftUI

/// The compact settings bar of a pane, generated from the analysis'
/// settings schema: menus for enums and field / wavelength / surface
/// selections, checkboxes, number fields. Nested objects and lists go to an
/// "Advanced" popover as JSON. Unset settings show the engine default.
struct SettingsBar: View {
    let pane: AnalysisPane
    let descriptor: AnalysisDescriptor
    let system: OpticalSystem?
    @State private var showAdvanced = false

    var body: some View {
        let fields = descriptor.settingsFields.filter { kind(of: $0) != .hidden }
        let simple = fields.filter { kind(of: $0) != .advanced }
        let advanced = fields.filter { kind(of: $0) == .advanced }
        FlowLayout(spacing: Space.m, lineSpacing: Space.xs) {
            ForEach(simple) { f in control(f) }
            if !advanced.isEmpty {
                Button {
                    showAdvanced = true
                } label: {
                    Label("More", systemImage: "ellipsis.circle").labelStyle(.titleAndIcon)
                }
                .buttonStyle(.borderless)
                .help("More settings: " + advanced.map(\.title).joined(separator: ", "))
                .popover(isPresented: $showAdvanced) {
                    AdvancedSettings(pane: pane, fields: advanced).frame(width: 380)
                }
            }
            if !pane.settings.isEmpty {
                Button("Defaults") { pane.setSettings([:]) }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Restore the engine's default settings")
            }
        }
        .font(.callout)
        .controlSize(.small)
        .padding(.horizontal, Space.m)
        .padding(.vertical, Space.xs + 1)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    enum Kind { case fieldSelection, wavelengthSelection, fieldIndex, wavelengthIndex, surfaceIndex, enumeration, boolean, number, hidden, advanced }

    func kind(of f: SettingsField) -> Kind {
        let types = Set(f.types)
        let isSelection = types.contains("object") && f.defaultValue?["type"]?.string != nil
        let integer = types.contains("integer")
        if f.name == "fields" && isSelection { return .fieldSelection }
        if ["wavelengths", "wavelength"].contains(f.name) && isSelection { return .wavelengthSelection }
        if f.name == "field" && integer { return .fieldIndex }
        if ["wavelength", "long_wavelength", "short_wavelength"].contains(f.name) && integer { return .wavelengthIndex }
        if f.name == "surface" && integer { return .surfaceIndex }
        switch f.name {
        case "configuration":
            return (system?.configurations.count ?? 0) > 0 ? .number : .hidden
        case "stop_key", "delay_ms", "chunk": return .hidden
        default: break
        }
        if !f.enumValues.isEmpty { return .enumeration }
        if types.subtracting(["null"]) == ["boolean"] { return .boolean }
        if !types.subtracting(["null"]).isEmpty && types.subtracting(["null"]).isSubset(of: ["number", "integer"]) { return .number }
        return .advanced
    }

    @ViewBuilder
    func control(_ f: SettingsField) -> some View {
        let value = pane.value(f.name)
        let label = Self.label(f)
        switch kind(of: f) {
        case .fieldSelection:
            SelectionMenu(
                label: "Fields", value: value, options: fieldOptions, allowsPrimary: false,
                set: { pane.set(f.name, $0) })
        case .wavelengthSelection:
            SelectionMenu(
                label: "λ", value: value, options: wavelengthOptions, allowsPrimary: true,
                set: { pane.set(f.name, $0) })
        case .fieldIndex:
            IndexMenu(label: "Field", value: value, options: fieldOptions, nullTitle: nil, set: { pane.set(f.name, $0) })
        case .wavelengthIndex:
            IndexMenu(
                label: label, value: value, options: wavelengthOptions,
                nullTitle: f.name == "wavelength" ? "Primary" : "Auto", set: { pane.set(f.name, $0) })
        case .surfaceIndex:
            IndexMenu(label: "Surface", value: value, options: surfaceOptions, nullTitle: nil, set: { pane.set(f.name, $0) })
        case .enumeration:
            Picker(label, selection: Binding(get: { value ?? .null }, set: { pane.set(f.name, $0) })) {
                ForEach(f.enumValues, id: \.self) { v in Text(Self.title(v)).tag(v) }
            }
            .pickerStyle(.menu)
            .fixedSize()
            .help(f.description)
        case .boolean:
            Toggle(label, isOn: Binding(get: { value?.bool ?? false }, set: { pane.set(f.name, .bool($0)) }))
                .help(f.description)
        case .number:
            SettingNumberField(field: f, label: label, value: value, set: { pane.set(f.name, $0) })
        case .hidden, .advanced:
            EmptyView()
        }
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
        let t = f.title == f.name ? f.name.replacingOccurrences(of: "_", with: " ").capitalized : f.title
        return t.count > 22 ? String(t.prefix(20)) + "…" : t
    }

    static func title(_ v: JSONValue) -> String {
        guard case .string(let s) = v else { return v.displayText }
        return s.replacingOccurrences(of: "_", with: " ").capitalized
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
            Text("\(label): \(current)")
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
            Text("\(label): \(current)")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    private var current: String {
        guard let i = value?.int else { return nullTitle ?? "—" }
        let title = options.first { $0.0 == i }?.1 ?? "\(i)"
        return String(title.split(separator: ":").first ?? Substring(title))
    }
}

/// A compact number setting; empty text restores the default (or null).
struct SettingNumberField: View {
    let field: SettingsField
    let label: String
    let value: JSONValue?
    let set: (JSONValue?) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: Space.xs) {
            Text(label).foregroundStyle(.secondary)
            TextField(placeholder, text: $text)
                .textFieldStyle(.roundedBorder)
                .numeric()
                .multilineTextAlignment(.trailing)
                .frame(width: 64)
                .focused($focused)
                .onSubmit(commit)
                .onChange(of: focused) { _, f in if !f { commit() } }
            if !field.unit.isEmpty { Text(field.unit).foregroundStyle(.secondary) }
        }
        .help(field.description.isEmpty ? field.title : field.description)
        .onAppear { text = display }
        .onChange(of: value) { _, _ in if !focused { text = display } }
    }

    private var isInteger: Bool { field.types.contains("integer") && !field.types.contains("number") }
    private var nullable: Bool { field.types.contains("null") }
    private var placeholder: String { nullable ? "auto" : "" }

    private var display: String {
        guard let v = value, let d = v.doubleValue else { return "" }
        return isInteger ? String(Int(d)) : Fmt.short(d)
    }

    private func commit() {
        let t = text.trimmingCharacters(in: .whitespaces)
        if t.isEmpty {
            set(nullable ? .null : nil)
            return
        }
        guard var d = LensValueParser.number(t), d.isFinite else {
            text = display
            return
        }
        if let lo = field.minimum { d = max(d, lo) }
        if let hi = field.maximum { d = min(d, hi) }
        set(isInteger ? .integer(Int64(d.rounded())) : .number(d))
        text = isInteger ? String(Int(d.rounded())) : Fmt.short(d)
    }
}

/// Settings without a compact control, edited as JSON.
struct AdvancedSettings: View {
    let pane: AnalysisPane
    let fields: [SettingsField]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.l) {
                Text("More Settings").font(.headline)
                ForEach(fields) { f in JSONSettingEditor(pane: pane, field: f) }
            }
            .padding(Space.l)
        }
        .frame(maxHeight: 420)
    }
}

struct JSONSettingEditor: View {
    let pane: AnalysisPane
    let field: SettingsField
    @State private var text = ""
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            HStack {
                Text(SettingsBar.label(field)).font(.subheadline.weight(.medium))
                Spacer()
                if pane.settings[field.name] != nil {
                    Button("Default") {
                        pane.set(field.name, nil)
                        text = Self.text(pane.value(field.name))
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                }
            }
            if !field.description.isEmpty {
                Text(field.description).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            TextField("JSON", text: $text, axis: .vertical)
                .font(.system(.caption, design: .monospaced))
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...6)
                .onSubmit(commit)
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
        }
        .onAppear { text = Self.text(pane.value(field.name)) }
    }

    static func text(_ v: JSONValue?) -> String {
        guard let v, let d = try? MKLightJSON.encode(v) else { return "null" }
        return String(decoding: d, as: UTF8.self)
    }

    private func commit() {
        guard let v = try? JSONValue.parse(Data(text.utf8)) else {
            error = "Not valid JSON"
            return
        }
        error = field.accepts(v) ? nil : "The engine expects \(field.types.joined(separator: " or "))"
        if error == nil { pane.set(field.name, v) }
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
            let size = s.sizeThatFits(.unspecified)
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

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, line: CGFloat = 0
        for s in subviews {
            let size = s.sizeThatFits(.unspecified)
            if x > bounds.minX && x + size.width > bounds.maxX {
                y += line + lineSpacing
                x = bounds.minX
                line = 0
            }
            s.place(at: CGPoint(x: x, y: y + (line > 0 ? 0 : 0)), proposal: ProposedViewSize(size))
            x += size.width + spacing
            line = max(line, size.height)
        }
    }
}
