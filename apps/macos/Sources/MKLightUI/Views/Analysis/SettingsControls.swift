import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

// Compact settings controls for array settings: two-element numeric
// arrays (`grid`, `extent`, `size`, optional ranges), rows of the same
// columns as a small table, the chief-ray sensor limit table and the image
// simulation source.

extension SettingsBar {
    /// The summary text of an array or object setting ("9 × 9", "2 points").
    static func arraySummary(_ f: SettingsField, _ v: JSONValue) -> String? {
        if f.name == "sensor_limit" {
            let n = FieldMapData.sensorLimit(v).count
            return n == 0 ? "none" : n == 1 ? "1 point" : "\(n) points"
        }
        if f.name == "source", let kind = v["kind"]?.string {
            switch kind {
            case "pattern": return SettingsBar.title(v["pattern"] ?? .string("grid"))
            case "buffer": return "image"
            default: return ((v["path"]?.string ?? "") as NSString).lastPathComponent
            }
        }
        if let a = v.array, a.count == 2, let x = a[0].doubleValue, let y = a[1].doubleValue {
            if f.name.hasSuffix("center") { return "(\(Fmt.short(x)), \(Fmt.short(y)))" }
            if f.name.contains("range") { return "\(Fmt.short(x)) to \(Fmt.short(y))" }
            return "\(Fmt.short(x)) × \(Fmt.short(y))"
        }
        return nil
    }
}

/// A two-element numeric setting as two small fields: `Grid 5 × 5`.
struct SettingPairField: View {
    let field: SettingsField
    let label: String
    let value: JSONValue?
    /// Whole numbers (`grid`, `size`); by default when both defaults are.
    var integer: Bool? = nil
    /// `null` allowed: both fields empty is the automatic state.
    var nullable = false
    let set: (JSONValue?) -> Void
    @State private var a = ""
    @State private var b = ""
    @FocusState private var focused: Int?

    var isInteger: Bool { integer ?? (field.defaultValue?.array?.allSatisfy { if case .integer = $0 { true } else { false } } ?? false) }

    /// `[min, max]` ranges read "from … to".
    var isRange: Bool { field.name.contains("range") || field.description.lowercased().contains("[min, max]") }

    /// A point of the normalised field (`field_center`): `Hx, Hy`, signed.
    var isFieldPoint: Bool { field.name.hasSuffix("center") || field.name.hasSuffix("centre") }

    /// Bounds the description states ("1..101", "2..4096"), else positive
    /// (a field point: within ±2 field radii).
    var bounds: ClosedRange<Double> {
        if isFieldPoint { return -2...2 }
        let text = field.description
        if let r = text.range(of: #"(\d+)\.\.(\d+)"#, options: .regularExpression) {
            let parts = text[r].components(separatedBy: "..").compactMap(Double.init)
            if parts.count == 2, parts[0] <= parts[1] { return parts[0]...parts[1] }
        }
        if isRange || field.description.contains("signed") { return -1e12...1e12 }
        return isInteger ? 1...1e9 : 1e-9...1e12
    }

    var body: some View {
        HStack(spacing: Space.xs) {
            if isFieldPoint { Text("Hx").foregroundStyle(.secondary) }
            box($a, index: 0)
            Text(isFieldPoint ? "Hy" : isRange ? "to" : "×").foregroundStyle(.secondary)
            box($b, index: 1)
            if !field.unit.isEmpty { Text(field.unit).foregroundStyle(.secondary) }
        }
        .help(field.description.isEmpty ? field.title : field.description)
        .onAppear(perform: load)
        .onChange(of: value) { _, _ in if focused == nil { load() } }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label)
    }

    private func box(_ text: Binding<String>, index: Int) -> some View {
        TextField(nullable ? "auto" : index == 0 ? "x" : "y", text: text)
            .textFieldStyle(.roundedBorder)
            .numeric()
            .multilineTextAlignment(.trailing)
            .frame(width: nullable ? 52 : isInteger ? 40 : 48)
            .focused($focused, equals: index)
            .onSubmit(commit)
            .onChange(of: focused) { old, new in if old == index && new != index { commit() } }
            .accessibilityLabel("\(label) \(index == 0 ? "x" : "y")")
    }

    private func display(_ v: JSONValue?) -> String {
        guard let d = v?.doubleValue else { return "" }
        return isInteger ? String(Int(d)) : Fmt.short(d)
    }

    private func load() {
        let arr = value?.array ?? []
        a = display(arr.first)
        b = display(arr.count > 1 ? arr[1] : nil)
    }

    private func commit() {
        if nullable, a.trimmingCharacters(in: .whitespaces).isEmpty, b.trimmingCharacters(in: .whitespaces).isEmpty {
            if value != nil && value != .null { set(.null) }
            return
        }
        guard var x = LensValueParser.number(a), var y = LensValueParser.number(b), x.isFinite, y.isFinite else {
            load()
            return
        }
        x = min(max(x, bounds.lowerBound), bounds.upperBound)
        y = min(max(y, bounds.lowerBound), bounds.upperBound)
        let v: JSONValue =
            isInteger
            ? .array([.integer(Int64(x.rounded())), .integer(Int64(y.rounded()))]) : .array([.number(x), .number(y)])
        if v != value { set(v) }
        load()
    }
}

/// Rows of the same scalar columns (`[{"hx", "hy"}, …]` or `[[r, θ], …]`),
/// edited as a small table in a popover; the row count is the button.
struct TableSettingControl: View {
    let field: SettingsField
    let label: String
    let columns: [SettingsField]
    let rowsAreArrays: Bool
    let value: JSONValue?
    let set: (JSONValue?) -> Void
    @State private var open = false

    private var count: Int { value?.array?.count ?? 0 }

    /// What an empty list means, from the description's "empty = …" clause
    /// ("the system fields"), else "Automatic" or "None".
    static func emptyTitle(_ f: SettingsField) -> String {
        if let r = f.description.range(of: #"empty = ([^;(.]+)"#, options: .regularExpression) {
            let t = f.description[r].dropFirst("empty = ".count).trimmingCharacters(in: .whitespaces)
            if !t.isEmpty { return t.prefix(1).uppercased() + t.dropFirst() }
        }
        return f.nullable ? "Automatic" : "None"
    }

    var body: some View {
        Button {
            open = true
        } label: {
            Text(count == 0 ? Self.emptyTitle(field) : count == 1 ? "1 row" : "\(count) rows")
        }
        .buttonStyle(.borderless)
        .help(field.description.isEmpty ? field.title : field.description)
        .accessibilityLabel(label)
        .popover(isPresented: $open) {
            TableSettingEditor(title: label, help: field.description, columns: columns, rowsAreArrays: rowsAreArrays, value: value, set: set)
                .padding(Space.l)
                .frame(width: CGFloat(72 * columns.count + 90).clamped(to: 240...520))
        }
    }
}

/// The rows of a table setting: one text field per column, add and delete.
struct TableSettingEditor: View {
    struct Row: Identifiable, Equatable {
        var id = UUID()
        var cells: [String]
    }
    let title: String
    let help: String
    let columns: [SettingsField]
    let rowsAreArrays: Bool
    let set: (JSONValue?) -> Void
    @State private var rows: [Row]

    init(title: String, help: String, columns: [SettingsField], rowsAreArrays: Bool, value: JSONValue?, set: @escaping (JSONValue?) -> Void) {
        self.title = title
        self.help = help
        self.columns = columns
        self.rowsAreArrays = rowsAreArrays
        self.set = set
        _rows = State(initialValue: Self.rows(of: value, columns: columns, rowsAreArrays: rowsAreArrays))
    }

    /// The rows of a value as text, one cell per column.
    static func rows(of value: JSONValue?, columns: [SettingsField], rowsAreArrays: Bool) -> [Row] {
        (value?.array ?? []).map { r in
            Row(cells: columns.enumerated().map { k, c in
                let v = rowsAreArrays ? r.array.flatMap { k < $0.count ? $0[k] : nil } : r[c.name]
                return text(v)
            })
        }
    }

    static func text(_ v: JSONValue?) -> String {
        guard let v else { return "" }
        if let d = v.doubleValue { return Fmt.short(d) }
        return v.string ?? v.displayText
    }

    /// The value of the rows, dropping rows with an unreadable number.
    static func value(of rows: [Row], columns: [SettingsField], rowsAreArrays: Bool) -> JSONValue {
        .array(rows.compactMap { r in
            var cells: [JSONValue] = []
            for (k, c) in columns.enumerated() {
                let t = r.cells.indices.contains(k) ? r.cells[k].trimmingCharacters(in: .whitespaces) : ""
                if c.baseTypes == ["string"] {
                    cells.append(.string(t))
                } else if c.baseTypes == ["boolean"] {
                    cells.append(.bool(["true", "yes", "1", "on"].contains(t.lowercased())))
                } else {
                    guard let d = LensValueParser.number(t), d.isFinite else { return nil }
                    cells.append(c.baseTypes == ["integer"] ? .integer(Int64(d.rounded())) : .number(d))
                }
            }
            return rowsAreArrays ? .array(cells) : .object(Dictionary(uniqueKeysWithValues: zip(columns.map(\.name), cells)))
        })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            Text(title).font(.headline)
            if !help.isEmpty {
                Text(help).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Grid(alignment: .leading, horizontalSpacing: Space.s, verticalSpacing: Space.xs) {
                GridRow {
                    ForEach(columns) { c in
                        Text(c.unit.isEmpty ? SettingsBar.label(c) : "\(SettingsBar.label(c)) (\(c.unit))")
                    }
                    Color.clear.frame(width: 16, height: 1)
                }
                .font(.caption).foregroundStyle(.secondary)
                ForEach($rows) { $r in
                    GridRow {
                        ForEach(columns.indices, id: \.self) { k in
                            TextField(SettingsBar.label(columns[k]), text: $r.cells[k])
                                .textFieldStyle(.roundedBorder)
                                .numeric()
                                .multilineTextAlignment(.trailing)
                                .frame(width: 72)
                                .onSubmit(commit)
                                .accessibilityLabel(SettingsBar.label(columns[k]))
                        }
                        Button {
                            rows.removeAll { $0.id == r.id }
                            commit()
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .help("Delete this row")
                        .accessibilityLabel("Delete row")
                    }
                }
            }
            HStack {
                Button {
                    rows.append(rows.last.map { Row(cells: $0.cells) } ?? Row(cells: columns.map { Self.text($0.defaultValue) }))
                    commit()
                } label: {
                    Label("Add Row", systemImage: "plus")
                }
                Spacer()
                if !rows.isEmpty {
                    Button("Clear") {
                        rows = []
                        commit()
                    }
                }
            }
            .buttonStyle(.borderless)
        }
        .controlSize(.small)
    }

    private func commit() {
        set(Self.value(of: rows, columns: columns, rowsAreArrays: rowsAreArrays))
    }
}

extension Comparable {
    func clamped(to r: ClosedRange<Self>) -> Self { min(max(self, r.lowerBound), r.upperBound) }
}

/// The chief-ray sensor limit, `[[image radius mm, max incidence °], …]`,
/// edited as a small table in a popover.
struct SensorLimitControl: View {
    let value: JSONValue?
    let help: String
    let set: (JSONValue?) -> Void
    @State private var open = false

    private var rows: [(Double, Double)] { FieldMapData.sensorLimit(value) }

    var body: some View {
        Button {
            open = true
        } label: {
            Text("Sensor limit: " + (rows.isEmpty ? "None" : rows.count == 1 ? String(format: "%.1f°", rows[0].1) : "\(rows.count) points"))
        }
        .buttonStyle(.borderless)
        .help(help)
        .popover(isPresented: $open) {
            SensorLimitEditor(rows: rows, set: set).padding(Space.l).frame(width: 300)
        }
    }
}

/// The rows of a sensor limit: radius and angle, add and delete; rows are
/// kept sorted by radius (the engine interpolates an increasing table).
struct SensorLimitEditor: View {
    struct Row: Identifiable, Equatable {
        var id = UUID()
        var radius: String
        var angle: String
    }
    let set: (JSONValue?) -> Void
    @State private var rows: [Row]
    @FocusState private var focus: UUID?

    init(rows: [(Double, Double)], set: @escaping (JSONValue?) -> Void) {
        self.set = set
        _rows = State(initialValue: rows.map { Row(radius: Fmt.short($0.0), angle: Fmt.short($0.1)) })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            Text("Sensor CRA Limit").font(.headline)
            Text("Maximum chief-ray incidence against image radius; linear between points, held beyond the ends.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Grid(alignment: .leading, horizontalSpacing: Space.s, verticalSpacing: Space.xs) {
                GridRow {
                    Text("Radius (mm)")
                    Text("Max angle (°)")
                    Color.clear.frame(width: 16, height: 1)
                }
                .font(.caption).foregroundStyle(.secondary)
                ForEach($rows) { $r in
                    GridRow {
                        field($r.radius, id: r.id, label: "Radius")
                        field($r.angle, id: r.id, label: "Maximum angle")
                        Button {
                            rows.removeAll { $0.id == r.id }
                            commit()
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .help("Delete this point")
                        .accessibilityLabel("Delete point")
                    }
                }
            }
            HStack {
                Button {
                    let last = rows.last.flatMap { LensValueParser.number($0.radius) } ?? -1
                    let angle = rows.last?.angle ?? "30"
                    rows.append(Row(radius: Fmt.short(last < 0 ? 0 : last + 1), angle: angle))
                    commit()
                } label: {
                    Label("Add Point", systemImage: "plus")
                }
                Spacer()
                if !rows.isEmpty {
                    Button("No Limit") {
                        rows = []
                        commit()
                    }
                }
            }
            .buttonStyle(.borderless)
        }
        .controlSize(.small)
    }

    private func field(_ text: Binding<String>, id: UUID, label: String) -> some View {
        TextField(label, text: text)
            .textFieldStyle(.roundedBorder)
            .numeric()
            .multilineTextAlignment(.trailing)
            .frame(width: 90)
            .onSubmit(commit)
            .accessibilityLabel(label)
    }

    /// Valid rows, sorted by radius, as the setting; then the rows shown again.
    private func commit() {
        let parsed = rows.compactMap { r -> (Double, Double)? in
            guard let a = LensValueParser.number(r.radius), let b = LensValueParser.number(r.angle), a.isFinite, b.isFinite, a >= 0
            else { return nil }
            return (a, b)
        }
        .sorted { $0.0 < $1.0 }
        var unique: [(Double, Double)] = []
        for p in parsed where unique.last?.0 != p.0 { unique.append(p) }
        set(unique.isEmpty ? nil : .array(unique.map { .array([.number($0.0), .number($0.1)]) }))
        rows = unique.map { Row(radius: Fmt.short($0.0), angle: Fmt.short($0.1)) }
    }
}

/// The image simulation's source: a built-in pattern or an image file.
struct ImageSourceControl: View {
    let pane: AnalysisPane
    let field: SettingsField
    @State private var importing = false
    @State private var missing: String?

    private var file: String? { pane.settings[ImageSourceImport.fileKey]?.string }

    private var title: String {
        if let file { return (file as NSString).lastPathComponent }
        let v = pane.value(field.name)
        switch v?["kind"]?.string {
        case "buffer": return "Image buffer"
        case "path": return ((v?["path"]?.string ?? "") as NSString).lastPathComponent
        default:
            let p = v?["pattern"]?.string ?? "grid"
            return ImageSimulationData.patterns.first { $0.id == p }?.title ?? p
        }
    }

    var body: some View {
        Menu {
            ForEach(ImageSimulationData.patterns, id: \.id) { p in
                Button {
                    missing = nil
                    ImageSourceImport.use(pattern: p.id, in: pane)
                } label: {
                    Label(p.title, systemImage: p.symbol)
                }
            }
            Divider()
            Button("Choose Image…") { importing = true }
        } label: {
            if missing != nil {
                Label("Source: \(title) (missing)", systemImage: "exclamationmark.triangle")
            } else {
                Text("Source: \(title)")
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help(missing.map { "\($0) could not be read; the simulation uses the grid pattern. Choose the image again." } ?? field.description)
        .accessibilityLabel("Source")
        .accessibilityValue(title)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result {
                missing = ImageSourceImport.use(url, in: pane) ? nil : url.lastPathComponent
            }
        }
        .task(id: refreshKey) {
            missing = ImageSourceImport.refresh(pane) ? nil : file.map { ($0 as NSString).lastPathComponent }
        }
    }

    /// Re-derive the samples when the file or the output size changes.
    private var refreshKey: String {
        let size = (try? MKLightJSON.encode(pane.value("size") ?? .null)).map { String(decoding: $0, as: UTF8.self) } ?? ""
        let aspect = ImageSourceImport.fieldAspect(pane).map { String(format: "%.6f", $0) } ?? ""
        return (file ?? "") + "|" + size + "|" + aspect
    }
}

/// The image simulation's optics sampling (`optics`): the PSF window and
/// spacing, the pupil grid and rays (automatic by default, shown with the
/// values the run used) and the field grid of PSFs. The patch's extent and
/// centre are set in the view; other keys are kept as they are.
struct ImageOpticsControl: View {
    let pane: AnalysisPane
    let field: SettingsField

    private var optics: [String: JSONValue] { pane.value(field.name)?.object ?? field.defaultValue?.object ?? [:] }
    private var pupil: [String: JSONValue] { optics["pupil"]?.object ?? [:] }

    private func used(_ key: String) -> String? { AutoSettings.used(key, analysis: pane.analysis, output: pane.output) }

    var body: some View {
        FlowLayout(spacing: Space.m, lineSpacing: Space.xs) {
            SettingNumberField(
                field: number("psf_size", integer: true, "PSF window samples per side (odd); 0 sizes it from the blur and diffraction."),
                label: "PSF window", value: optics["psf_size"], auto: .zero, autoUsed: used("optics.psf_size")
            ) { set("psf_size", $0) }
            SettingNumberField(
                field: number("psf_spacing_mm", integer: false, "PSF sample spacing; empty uses the smallest λ·F/# over the patch divided by 2.5.", unit: "mm"),
                label: "Spacing", value: optics["psf_spacing_mm"], auto: .null, autoUsed: used("optics.psf_spacing_mm")
            ) { set("psf_spacing_mm", $0) }
            SettingNumberField(
                field: number("grid", integer: true, "Pupil grid samples per side; 0 chooses from the wavefront slope (64 to 1024)."),
                label: "Pupil", value: pupil["grid"], auto: .zero, autoUsed: used("optics.pupil.grid")
            ) { setPupil("grid", $0) }
            SettingNumberField(
                field: number("rays", integer: true, "Rays traced per pupil diameter."), label: "Rays", value: pupil["rays"]
            ) { setPupil("rays", $0) }
            SettingNumberField(
                field: number("field_grid", integer: true, "PSFs per side of the patch; the image is mapped between them."),
                label: "PSF grid", value: optics["field_grid"]
            ) { set("field_grid", $0) }
        }
    }

    private func number(_ name: String, integer: Bool, _ help: String, unit: String = "") -> SettingsField {
        var schema: [String: JSONValue] = ["type": integer ? .string("integer") : .array([.string("number"), .string("null")]), "description": .string(help)]
        if !unit.isEmpty { schema["unit"] = .string(unit) }
        var f = SettingsField(name: name, schema: .object(schema), required: false)
        f.minimum = 0
        return f
    }

    private func set(_ key: String, _ v: JSONValue?) {
        var o = optics
        o[key] = v ?? field.defaultValue?[key]
        pane.set(field.name, .object(o))
    }

    private func setPupil(_ key: String, _ v: JSONValue?) {
        var p = pupil
        p[key] = v ?? field.defaultValue?["pupil"]?[key]
        set("pupil", .object(p))
    }
}
