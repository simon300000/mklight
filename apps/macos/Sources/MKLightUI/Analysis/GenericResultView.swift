import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

/// Shows any engine result without knowing its schema: scalar fields as a
/// list, arrays of records as tables, everything else as an outline, and the
/// binary buffers with their shapes, ranges and a plot of the selected one.
/// Unknown analyses and fields of a newer engine therefore always display.
struct GenericResultView: View {
    let result: GenericResult
    @State private var plotted: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let note = result.note {
                    Label(note, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
                }
                let layout = GenericLayout(result.json)
                if !layout.scalars.isEmpty {
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 3) {
                        ForEach(layout.scalars, id: \.key) { item in
                            GridRow {
                                Text(item.key).foregroundStyle(.secondary)
                                Text(item.value.displayText).monospacedDigit().textSelection(.enabled)
                            }
                        }
                    }
                    .font(.callout)
                }
                ForEach(layout.tables, id: \.key) { table in
                    RecordTable(title: table.key, rows: table.rows)
                }
                if !layout.others.isEmpty {
                    GroupBox("Other data") {
                        OutlineGroup(layout.others.map { JSONNode(key: $0.key, value: $0.value) }, children: \.children) { node in
                            HStack {
                                Text(node.key).foregroundStyle(.secondary)
                                Text(node.summary).monospacedDigit().lineLimit(1).textSelection(.enabled)
                            }
                            .font(.caption)
                        }
                    }
                }
                if !result.buffers.isEmpty { buffers }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var buffers: some View {
        GroupBox("Buffers") {
            VStack(alignment: .leading, spacing: 8) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                    GridRow {
                        Text("name")
                        Text("type")
                        Text("min … max")
                        Text("")
                    }
                    .foregroundStyle(.secondary)
                    ForEach(result.buffers, id: \.name) { b in
                        GridRow {
                            Text(b.name).monospaced()
                            Text("\(b.dtype.name)[\(b.shape.map(String.init).joined(separator: "×"))]").monospaced()
                            Text(range(b)).monospacedDigit()
                            Button("Plot") { plotted = b.name }
                                .buttonStyle(.link)
                                .disabled(!BufferPlot.canPlot(b))
                        }
                    }
                }
                .font(.caption)
                if let name = plotted ?? result.buffers.first(where: BufferPlot.canPlot)?.name,
                    let b = result.buffer(named: name)
                {
                    BufferPlot(buffer: b).frame(height: 180)
                }
            }
        }
    }

    private func range(_ b: ResultBuffer) -> String {
        guard b.count > 0 else { return "empty" }
        var lo = Double.infinity, hi = -Double.infinity
        for i in 0..<min(b.count, 1_000_000) where b[i].isFinite {
            lo = min(lo, b[i])
            hi = max(hi, b[i])
        }
        return lo <= hi ? "\(NumberDisplay.format(lo)) … \(NumberDisplay.format(hi))" : "no finite values"
    }
}

/// Splits a JSON object into scalars, record tables and the rest.
struct GenericLayout {
    var scalars: [(key: String, value: JSONValue)] = []
    var tables: [(key: String, rows: [[String: JSONValue]])] = []
    var others: [(key: String, value: JSONValue)] = []

    init(_ json: JSONValue) {
        guard case .object(let o) = json else {
            others = [("result", json)]
            return
        }
        for (k, v) in o.sorted(by: { $0.key < $1.key }) {
            switch v {
            case .array(let items) where !items.isEmpty && items.allSatisfy({ if case .object = $0 { true } else { false } }):
                tables.append((k, items.compactMap { if case .object(let r) = $0 { r } else { nil } }))
            case .object, .array: others.append((k, v))
            default: scalars.append((k, v))
            }
        }
    }
}

/// An array of JSON objects as a grid: one column per key (union, in first
/// appearance order), nested values summarised.
struct RecordTable: View {
    let title: String
    let rows: [[String: JSONValue]]
    static let maxRows = 500

    var body: some View {
        let columns = Self.columns(rows)
        GroupBox(title) {
            ScrollView(.horizontal) {
                Grid(alignment: .trailing, horizontalSpacing: 12, verticalSpacing: 2) {
                    GridRow {
                        ForEach(columns, id: \.self) { Text($0).foregroundStyle(.secondary) }
                    }
                    ForEach(Array(rows.prefix(Self.maxRows).enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(columns, id: \.self) { c in
                                Text(row[c].map { $0.displayText } ?? "").monospacedDigit().lineLimit(1)
                            }
                        }
                    }
                }
                .font(.caption)
                .textSelection(.enabled)
            }
            if rows.count > Self.maxRows {
                Text("… \(rows.count - Self.maxRows) more rows").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    static func columns(_ rows: [[String: JSONValue]]) -> [String] {
        var seen = Set<String>(), out: [String] = []
        // Keys in a stable order: common identifying keys first, then sorted.
        let preferred = ["index", "id", "label", "name", "severity", "code", "surface"]
        let all = rows.prefix(maxRows).flatMap(\.keys)
        for k in preferred where all.contains(k) && seen.insert(k).inserted { out.append(k) }
        for k in Set(all).sorted() where seen.insert(k).inserted { out.append(k) }
        return out
    }
}

/// A JSON value in an outline.
struct JSONNode: Identifiable {
    static let maxChildren = 200
    let id = UUID()
    let key: String
    let value: JSONValue

    var children: [JSONNode]? {
        switch value {
        case .object(let o): o.sorted { $0.key < $1.key }.map { JSONNode(key: $0.key, value: $0.value) }
        case .array(let a):
            Array(a.prefix(Self.maxChildren).enumerated().map { JSONNode(key: "[\($0.offset)]", value: $0.element) })
                + (a.count > Self.maxChildren
                    ? [JSONNode(key: "…", value: .string("\(a.count - Self.maxChildren) more"))] : [])
        default: nil
        }
    }

    var summary: String {
        switch value {
        case .object(let o): "{\(o.count) fields}"
        case .array(let a): "[\(a.count)]"
        default: value.displayText
        }
    }
}

extension JSONValue {
    /// Short text for tables (CONVENTIONS §11: `null` is an undefined value).
    var displayText: String {
        switch self {
        case .null: "—"
        case .bool(let b): b ? "true" : "false"
        case .integer(let i): "\(i)"
        case .unsigned(let u): "\(u)"
        case .number(let d): NumberDisplay.format(d, decimals: 6)
        case .string(let s): s
        case .array(let a):
            a.count <= 6 ? "[" + a.map(\.displayText).joined(separator: ", ") + "]" : "[\(a.count) values]"
        case .object(let o): "{" + o.keys.sorted().prefix(4).joined(separator: ", ") + (o.count > 4 ? ", …}" : "}")
        }
    }
}

/// Line plot of a buffer: a 1-D buffer against its index, an `n × 2` buffer
/// as (x, y), and an `n × k` buffer as k series against the row index.
struct BufferPlot: View {
    let buffer: ResultBuffer
    static let maxPoints = 4000

    static func canPlot(_ b: ResultBuffer) -> Bool {
        b.count > 1 && (b.shape.count == 1 || (b.shape.count == 2 && b.shape[1] <= 8))
    }

    struct Point: Identifiable {
        var id: Int
        var series: String
        var x: Double
        var y: Double
    }

    var points: [Point] {
        var out: [Point] = []
        if buffer.shape.count == 1 {
            let step = max(1, buffer.count / Self.maxPoints)
            for i in stride(from: 0, to: buffer.count, by: step) where buffer[i].isFinite {
                out.append(Point(id: i, series: buffer.name, x: Double(i), y: buffer[i]))
            }
        } else if buffer.shape[1] == 2 {
            let n = buffer.shape[0], step = max(1, n / Self.maxPoints)
            for i in stride(from: 0, to: n, by: step) where buffer[[i, 0]].isFinite && buffer[[i, 1]].isFinite {
                out.append(Point(id: i, series: buffer.name, x: buffer[[i, 0]], y: buffer[[i, 1]]))
            }
        } else {
            let n = buffer.shape[0], k = buffer.shape[1], step = max(1, n / Self.maxPoints)
            for j in 0..<k {
                for i in stride(from: 0, to: n, by: step) where buffer[[i, j]].isFinite {
                    out.append(Point(id: j * n + i, series: "column \(j)", x: Double(i), y: buffer[[i, j]]))
                }
            }
        }
        return out
    }

    var body: some View {
        Chart(points) { p in
            LineMark(x: .value("x", p.x), y: .value(buffer.name, p.y))
                .foregroundStyle(by: .value("series", p.series))
                .interpolationMethod(.linear)
        }
        .chartYScale(domain: .automatic(includesZero: false))
        .chartXScale(domain: .automatic(includesZero: false))
    }
}
