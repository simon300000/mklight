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

/// The edit of a two-number setting in two boxes, each a `FieldEdit`. The
/// pair is one value: it is committed on Return and when the focus leaves
/// a box, from the typed numbers and, for a box that was not edited, its
/// stored one. Half a pair waits for its other number while the focus is
/// in the pair, and is given up when the focus leaves it.
struct PairEdit {
    typealias Format = FieldEdit<Double?>.Format

    /// The two boxes; the fields show and set their texts.
    var boxes: [FieldEdit<Double?>]
    /// `null` allowed: both boxes empty is the automatic state.
    let nullable: Bool
    /// The box with the focus.
    private(set) var focus: Int?

    /// Two empty boxes; `valueChanged` fills them.
    init(nullable: Bool = false) {
        let empty = Format(exact: { _ in "" }, display: { _ in "" }, parse: { _ in nil })
        boxes = [FieldEdit(nil, empty), FieldEdit(nil, empty)]
        self.nullable = nullable
    }

    /// The setting is `pair` (`nil` where it has no number): boxes that
    /// were not edited follow it.
    mutating func valueChanged(_ pair: [Double?], _ format: Format) {
        for i in boxes.indices { boxes[i].valueChanged(pair.indices.contains(i) ? pair[i] : nil, format) }
    }

    /// The focus moved to box `new`, or out of the pair (`nil`).
    mutating func focusChanged(to new: Int?, _ format: Format) -> FieldEdit<[Double]?>.Commit {
        let old = focus
        focus = new
        var result = FieldEdit<[Double]?>.Commit.unchanged
        if old != nil, old != new { result = commit(format) }
        if let new, new != old { boxes[new].beginEditing(format) }
        return result
    }

    /// Return in a box.
    mutating func submit(_ format: Format) -> FieldEdit<[Double]?>.Commit { commit(format) }

    /// The pair to write (`nil`: automatic), when it was edited, is
    /// complete and is not the stored one.
    private mutating func commit(_ format: Format) -> FieldEdit<[Double]?>.Commit {
        defer {
            // A box without the focus shows its number at rest, unless it waits.
            for i in boxes.indices where i != focus && !boxes[i].isEdited { _ = boxes[i].endEditing(format) }
        }
        guard boxes.contains(where: \.isEdited) else { return .unchanged }
        let stored = boxes.map(\.value)
        let blank = boxes.map { $0.text.trimmingCharacters(in: .whitespaces).isEmpty }
        if nullable, !blank.contains(false) {
            show([nil, nil], format)
            return stored.contains { $0 != nil } ? .value(nil) : .unchanged
        }
        // Half a pair with the focus still in it waits for its other number.
        if blank[0] != blank[1], focus != nil { return .unchanged }
        var typed: [Double] = []
        for i in boxes.indices {
            if case .invalid = boxes[i].submit(format) { continue }
            if let v = boxes[i].value { typed.append(v) }
        }
        guard typed.count == boxes.count else {
            show(stored, format)
            return .unchanged
        }
        return typed == stored ? .unchanged : .value(typed)
    }

    /// Both boxes show `pair`, whatever was typed.
    private mutating func show(_ pair: [Double?], _ format: Format) {
        for i in boxes.indices {
            boxes[i].valueChanged(pair[i], format)
            boxes[i].revert(format)
        }
    }
}

/// A two-element numeric setting as two small fields: `Grid 5 × 5`. Only
/// edited text is committed; a box left as it was keeps its stored value.
struct SettingPairField: View {
    let field: SettingsField
    let label: String
    let value: JSONValue?
    /// Whole numbers (`grid`, `size`); by default when both defaults are.
    var integer: Bool? = nil
    /// `null` allowed: both fields empty is the automatic state.
    var nullable = false
    let set: (JSONValue?) -> Void
    @State private var edit: PairEdit
    @FocusState private var focused: Int?

    init(field: SettingsField, label: String, value: JSONValue?, integer: Bool? = nil, nullable: Bool = false, set: @escaping (JSONValue?) -> Void) {
        self.field = field
        self.label = label
        self.value = value
        self.integer = integer
        self.nullable = nullable
        self.set = set
        _edit = State(initialValue: PairEdit(nullable: nullable))
    }

    var isInteger: Bool { integer ?? (field.defaultValue?.array?.allSatisfy { if case .integer = $0 { true } else { false } } ?? false) }

    /// `[min, max]` ranges read "from … to".
    var isRange: Bool { field.name.contains("range") || field.description.lowercased().contains("[min, max]") }

    /// A point of the normalised field (`field_center`): `Hx, Hy`, signed.
    var isFieldPoint: Bool { field.name.hasSuffix("center") || field.name.hasSuffix("centre") }

    /// The bounds of both numbers: the schema's `minimum` and `maximum` (of
    /// the setting or of its items), else those the description states
    /// ("1..101", "2..4096"); unbounded without either.
    var bounds: ClosedRange<Double> {
        let lo = field.minimum ?? field.items?["minimum"]?.doubleValue, hi = field.maximum ?? field.items?["maximum"]?.doubleValue
        if lo != nil || hi != nil {
            let r = (lo ?? -.infinity, hi ?? .infinity)
            if r.0 <= r.1 { return r.0...r.1 }
        }
        let text = field.description
        if let r = text.range(of: #"(\d+)\.\.(\d+)"#, options: .regularExpression) {
            let parts = text[r].components(separatedBy: "..").compactMap(Double.init)
            if parts.count == 2, parts[0] <= parts[1] { return parts[0]...parts[1] }
        }
        return -.infinity...(.infinity)
    }

    var body: some View {
        HStack(spacing: Space.xs) {
            if isFieldPoint { Text("Hx").foregroundStyle(.secondary) }
            box(0)
            Text(isFieldPoint ? "Hy" : isRange ? "to" : "×").foregroundStyle(.secondary)
            box(1)
            if !field.unit.isEmpty { Text(field.unit).foregroundStyle(.secondary) }
        }
        .help(field.description.isEmpty ? field.title : field.description)
        .onAppear(perform: sync)
        .onChange(of: value) { _, _ in sync() }
        .onChange(of: focused) { _, new in
            sync()
            write(edit.focusChanged(to: new, entry))
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label)
    }

    private func box(_ index: Int) -> some View {
        TextField(nullable ? "auto" : index == 0 ? "x" : "y", text: $edit.boxes[index].text)
            .textFieldStyle(.roundedBorder)
            .numeric()
            .multilineTextAlignment(.trailing)
            .frame(width: nullable ? 52 : isInteger ? 40 : 48)
            .focused($focused, equals: index)
            .onSubmit {
                sync()
                write(edit.submit(entry))
            }
            .accessibilityLabel("\(label) \(index == 0 ? "x" : "y")")
    }

    /// One number of the pair: whole where the setting is, held to its
    /// bounds; no number shows as an empty box.
    private var entry: PairEdit.Format {
        let isInteger = isInteger, bounds = bounds
        return PairEdit.Format(
            exact: { v in v.map { isInteger ? String(Int(saturating: $0)) : FieldText.exact($0) } ?? "" },
            display: { v in v.map { isInteger ? String(Int(saturating: $0)) : Fmt.short($0) } ?? "" },
            parse: { text in
                guard let d = LensValueParser.number(text), d.isFinite else { return nil }
                let held = min(max(d, bounds.lowerBound), bounds.upperBound)
                return .some(isInteger ? Double(Int(saturating: held)) : held)
            })
    }

    /// Reports the setting's two numbers to the boxes.
    private func sync() {
        let pair = value?.array ?? []
        edit.valueChanged([0, 1].map { pair.indices.contains($0) ? pair[$0].doubleValue : nil }, entry)
    }

    private func write(_ result: FieldEdit<[Double]?>.Commit) {
        guard case .value(let pair) = result else { return }
        set(pair.map { p in .array(p.map { isInteger ? .integer(Int64(Int(saturating: $0))) : .number($0) }) } ?? .null)
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

/// The rows of a table setting: one field per column, add and delete. A
/// cell commits on Return or when focus leaves, if its text was edited; a
/// row with an empty number is not part of the setting until it is filled.
struct TableSettingEditor: View {
    struct Row: Identifiable, Equatable {
        var id = UUID()
        /// One value per column; `nil` is an empty cell.
        var cells: [JSONValue?]
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

    /// The rows of a value, one cell per column.
    static func rows(of value: JSONValue?, columns: [SettingsField], rowsAreArrays: Bool) -> [Row] {
        (value?.array ?? []).map { r in
            Row(cells: columns.enumerated().map { k, c in rowsAreArrays ? r.array.flatMap { k < $0.count ? $0[k] : nil } : r[c.name] })
        }
    }

    static func text(_ v: JSONValue?) -> String {
        guard let v else { return "" }
        if let d = v.doubleValue { return Fmt.short(d) }
        return v.string ?? v.displayText
    }

    /// The value of the rows, dropping rows with a number missing.
    static func value(of rows: [Row], columns: [SettingsField], rowsAreArrays: Bool) -> JSONValue {
        .array(rows.compactMap { r in
            var cells: [JSONValue] = []
            for (k, c) in columns.enumerated() {
                let v = r.cells.indices.contains(k) ? r.cells[k] : nil
                if c.baseTypes == ["string"] {
                    cells.append(v ?? .string(""))
                } else if c.baseTypes == ["boolean"] {
                    cells.append(v ?? .bool(false))
                } else {
                    guard let v, v.doubleValue != nil else { return nil }
                    cells.append(v)
                }
            }
            return rowsAreArrays ? .array(cells) : .object(Dictionary(uniqueKeysWithValues: zip(columns.map(\.name), cells)))
        })
    }

    /// A cell of `column`: text, yes/no, or a finite number (whole for an
    /// integer column; empty is an empty cell).
    static func entry(_ column: SettingsField) -> FieldEdit<JSONValue?>.Format {
        if column.baseTypes == ["string"] {
            return FieldEdit.Format(exact: text, display: text, parse: { .some(.string($0.trimmingCharacters(in: .whitespaces))) })
        }
        if column.baseTypes == ["boolean"] {
            return FieldEdit.Format(
                exact: text, display: text,
                parse: { .some(.bool(["true", "yes", "1", "on"].contains($0.trimmingCharacters(in: .whitespaces).lowercased()))) })
        }
        let isInteger = column.baseTypes == ["integer"]
        return FieldEdit.Format(
            exact: { v in
                guard let d = v?.doubleValue else { return text(v) }
                return isInteger ? String(Int(saturating: d)) : FieldText.exact(d)
            },
            display: text,
            parse: { typed in
                let t = typed.trimmingCharacters(in: .whitespaces)
                if t.isEmpty { return .some(nil) }
                guard let d = LensValueParser.number(t), d.isFinite else { return nil }
                return .some(isInteger ? .integer(Int64(Int(saturating: d))) : .number(d))
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
                ForEach(rows) { r in
                    GridRow {
                        ForEach(columns.indices, id: \.self) { k in
                            CommittingField(
                                value: r.cells[k], format: Self.entry(columns[k]), endsOnDisappear: true,
                                commit: { v in
                                    guard let i = rows.firstIndex(where: { $0.id == r.id }) else { return }
                                    rows[i].cells[k] = v
                                    commit()
                                }
                            ) { text, _ in
                                TextField(SettingsBar.label(columns[k]), text: text)
                                    .textFieldStyle(.roundedBorder)
                                    .numeric()
                                    .multilineTextAlignment(.trailing)
                                    .frame(width: 72)
                            }
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
                    rows.append(Row(cells: rows.last?.cells ?? columns.map(\.defaultValue)))
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
/// kept sorted by radius (the engine interpolates an increasing table). A
/// cell commits on Return or when focus leaves, if its text was edited;
/// text that is not a number reverts.
struct SensorLimitEditor: View {
    struct Row: Identifiable, Equatable {
        var id = UUID()
        var radius: Double
        var angle: Double
    }
    let set: (JSONValue?) -> Void
    @State private var rows: [Row]

    init(rows: [(Double, Double)], set: @escaping (JSONValue?) -> Void) {
        self.set = set
        _rows = State(initialValue: rows.map { Row(radius: $0.0, angle: $0.1) })
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
                ForEach(rows) { r in
                    GridRow {
                        field(\.radius, of: r, label: "Radius", format: Self.radius)
                        field(\.angle, of: r, label: "Maximum angle", format: Self.angle)
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
                    let last = rows.last?.radius ?? -1
                    rows.append(Row(radius: last < 0 ? 0 : last + 1, angle: rows.last?.angle ?? 30))
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

    /// A finite number, at least `minimum`.
    private static func number(minimum: Double) -> FieldEdit<Double>.Format {
        FieldEdit.Format(
            exact: FieldText.exact, display: Fmt.short,
            parse: { text in LensValueParser.number(text).flatMap { $0.isFinite && $0 >= minimum ? $0 : nil } })
    }

    private static let radius = number(minimum: 0)
    private static let angle = number(minimum: -.infinity)

    private func field(_ key: WritableKeyPath<Row, Double>, of row: Row, label: String, format: FieldEdit<Double>.Format) -> some View {
        CommittingField(
            value: row[keyPath: key], format: format, endsOnDisappear: true,
            commit: { v in
                guard let i = rows.firstIndex(where: { $0.id == row.id }) else { return }
                rows[i][keyPath: key] = v
                commit()
            }
        ) { text, _ in
            TextField(label, text: text)
                .textFieldStyle(.roundedBorder)
                .numeric()
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
        }
        .accessibilityLabel(label)
    }

    /// The rows sorted by radius, one per radius, as the setting.
    private func commit() {
        var unique: [Row] = []
        for r in rows.sorted(by: { $0.radius < $1.radius }) where unique.last?.radius != r.radius { unique.append(r) }
        set(unique.isEmpty ? nil : .array(unique.map { .array([.number($0.radius), .number($0.angle)]) }))
        rows = unique
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
