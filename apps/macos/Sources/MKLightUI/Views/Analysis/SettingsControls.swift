import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

// Compact settings controls for array and object settings the settings bar
// would otherwise leave to the JSON editor: two-element numeric arrays
// (`grid`, `extent`, `size`), the chief-ray sensor limit table and the
// image simulation source.

extension SettingsBar {
    /// The kind of the settings handled here, or nil.
    func arrayKind(of f: SettingsField) -> Kind? {
        let types = Set(f.types).subtracting(["null"])
        if f.name == "sensor_limit" && types == ["array"] { return .sensorLimit }
        if f.name == "source" && types == ["object"] && f.defaultValue?["kind"]?.string == "pattern" { return .imageSource }
        if types == ["array"], let d = f.defaultValue?.array, d.count == 2, d.allSatisfy({ $0.doubleValue != nil }) { return .pair }
        return nil
    }

    @ViewBuilder
    func arrayControl(_ f: SettingsField, kind: Kind) -> some View {
        switch kind {
        case .pair:
            SettingPairField(field: f, label: Self.label(f), value: pane.value(f.name), set: { pane.set(f.name, $0) })
        case .sensorLimit:
            SensorLimitControl(value: pane.value(f.name), help: f.description, set: { pane.set(f.name, $0) })
        case .imageSource:
            ImageSourceControl(pane: pane, field: f)
        default:
            EmptyView()
        }
    }
}

/// A two-element numeric setting as two small fields: `Grid 5 × 5`.
struct SettingPairField: View {
    let field: SettingsField
    let label: String
    let value: JSONValue?
    let set: (JSONValue?) -> Void
    @State private var a = ""
    @State private var b = ""
    @FocusState private var focused: Int?

    /// Integers when the default is (`grid`, `size`).
    var isInteger: Bool { field.defaultValue?.array?.allSatisfy { if case .integer = $0 { true } else { false } } ?? false }

    /// Bounds the description states ("1..101", "2..4096"), else positive.
    var bounds: ClosedRange<Double> {
        let text = field.description
        if let r = text.range(of: #"(\d+)\.\.(\d+)"#, options: .regularExpression) {
            let parts = text[r].components(separatedBy: "..").compactMap(Double.init)
            if parts.count == 2, parts[0] <= parts[1] { return parts[0]...parts[1] }
        }
        return isInteger ? 1...1e9 : 1e-9...1e12
    }

    var body: some View {
        HStack(spacing: Space.xs) {
            Text(label).foregroundStyle(.secondary)
            box($a, index: 0)
            Text("×").foregroundStyle(.secondary)
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
        TextField(index == 0 ? "x" : "y", text: text)
            .textFieldStyle(.roundedBorder)
            .numeric()
            .multilineTextAlignment(.trailing)
            .frame(width: isInteger ? 40 : 48)
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
        (file ?? "") + "|" + ((try? MKLightJSON.encode(pane.value("size") ?? .null)).map { String(decoding: $0, as: UTF8.self) } ?? "")
    }
}
