import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

/// Any result, shown from its JSON and buffers alone: scalars as a table,
/// arrays of records as tables, numeric arrays and 1-D buffers as curves,
/// 2-D buffers as maps, and the full JSON as an outline. Used for analyses
/// without a dedicated view, so new engine analyses show up without code
/// changes.
struct GenericResultView: View {
    let output: AnalysisOutput

    var body: some View {
        let json = output.json
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                let scalars = Self.scalars(json)
                if !scalars.isEmpty {
                    section("Values") {
                        KeyValueGrid(rows: scalars.map { ($0.0, $0.1.displayText) })
                    }
                }
                ForEach(Self.recordTables(json), id: \.0) { name, rows in
                    section(Self.title(name)) { RecordTable(rows: rows) }
                }
                ForEach(Self.curves(json), id: \.0) { name, values in
                    section(Self.title(name)) {
                        LineChart(
                            series: [Series(id: name, label: name, color: Palette.fieldColor(0), points: values.enumerated().map { (Double($0.offset), $0.element) })],
                            xLabel: "index", yLabel: "", zeroLine: false
                        )
                        .frame(height: 120)
                    }
                }
                ForEach(output.buffers, id: \.name) { b in
                    section(b.name) { BufferPreview(buffer: b) }
                }
                if json != .null {
                    section("Data") { JSONOutline(value: json) }
                }
            }
            .padding(Space.l)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func section<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: Space.s) {
            Text(title).font(.subheadline.weight(.semibold))
            content()
        }
    }

    static func title(_ key: String) -> String { key.replacingOccurrences(of: "_", with: " ").capitalized }

    /// Scalars at the top level and one object level down (`a.b`).
    static func scalars(_ json: JSONValue) -> [(String, JSONValue)] {
        guard let o = json.object else { return [] }
        var out: [(String, JSONValue)] = []
        for (k, v) in o.sorted(by: { $0.key < $1.key }) {
            switch v {
            case .object(let inner):
                for (k2, v2) in inner.sorted(by: { $0.key < $1.key }) where isScalar(v2) { out.append(("\(k).\(k2)", v2)) }
            default: if isScalar(v) { out.append((k, v)) }
            }
        }
        return out
    }

    static func isScalar(_ v: JSONValue) -> Bool {
        switch v {
        case .array, .object: false
        default: true
        }
    }

    /// Top-level arrays of objects.
    static func recordTables(_ json: JSONValue) -> [(String, [[String: JSONValue]])] {
        guard let o = json.object else { return [] }
        return o.sorted(by: { $0.key < $1.key }).compactMap { k, v in
            guard let a = v.array, !a.isEmpty else { return nil }
            let rows = a.compactMap(\.object)
            return rows.count == a.count ? (k, rows) : nil
        }
    }

    /// Top-level numeric arrays longer than four values.
    static func curves(_ json: JSONValue) -> [(String, [Double])] {
        guard let o = json.object else { return [] }
        return o.sorted(by: { $0.key < $1.key }).compactMap { k, v in
            guard let a = v.nfArray, a.count > 4 else { return nil }
            return (k, a)
        }
    }
}

/// Records as a table of their scalar columns (first 200 rows).
struct RecordTable: View {
    let rows: [[String: JSONValue]]

    var body: some View {
        let cols = Array(Set(rows.prefix(50).flatMap { $0.filter { GenericResultView.isScalar($0.value) }.keys })).sorted()
        ScrollView(.horizontal) {
            Grid(alignment: .trailing, horizontalSpacing: Space.l, verticalSpacing: 3) {
                GridRow {
                    ForEach(cols, id: \.self) { Text(GenericResultView.title($0)).foregroundStyle(.secondary) }
                }
                .font(.caption.weight(.medium))
                Divider()
                ForEach(Array(rows.prefix(200).enumerated()), id: \.offset) { _, r in
                    GridRow {
                        ForEach(cols, id: \.self) { c in Text(r[c]?.displayText ?? "").numeric() }
                    }
                    .font(.caption)
                }
            }
            .textSelection(.enabled)
        }
        if rows.count > 200 { Text("\(rows.count - 200) more rows").font(.caption).foregroundStyle(.secondary) }
    }
}

/// A buffer: its shape and range, a curve for 1-D data, a map for 2-D.
struct BufferPreview: View {
    let buffer: ResultBuffer

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(buffer.description).font(.caption.monospaced()).foregroundStyle(.secondary)
            if buffer.shape.count == 1, buffer.count > 1, buffer.count <= 20000 {
                let v = buffer.doubles
                LineChart(
                    series: [Series(id: buffer.name, label: buffer.name, color: Palette.fieldColor(0), points: v.enumerated().map { (Double($0.offset), $0.element) })],
                    xLabel: "index", yLabel: "", zeroLine: false
                )
                .frame(height: 110)
            } else if buffer.shape.count == 2, buffer.shape[1] > 3, buffer.count <= 4_000_000, let g = Grid2D(buffer) {
                HStack(alignment: .top) {
                    HeatMapImage(grid: g, scale: .perceptual).aspectRatio(CGFloat(g.nx) / CGFloat(max(g.ny, 1)), contentMode: .fit).frame(maxHeight: 200)
                    ColorBar(scale: .perceptual, range: g.range).frame(height: 160)
                }
            } else if buffer.count > 0, buffer.count <= 2_000_000 {
                let f = buffer.doubles.filter(\.isFinite)
                if let lo = f.min(), let hi = f.max() {
                    Text("range \(Fmt.short(lo)) … \(Fmt.short(hi)), \(buffer.count - f.count) without a value")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// The JSON tree as disclosure groups.
struct JSONOutline: View {
    let value: JSONValue

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            switch value {
            case .object(let o):
                ForEach(o.keys.sorted(), id: \.self) { k in node(k, o[k]!) }
            case .array(let a):
                ForEach(Array(a.prefix(100).enumerated()), id: \.offset) { i, v in node("[\(i)]", v) }
                if a.count > 100 { Text("… \(a.count - 100) more").foregroundStyle(.secondary) }
            default:
                Text(value.displayText)
            }
        }
        .font(.caption.monospaced())
    }

    @ViewBuilder private func node(_ key: String, _ v: JSONValue) -> some View {
        if GenericResultView.isScalar(v) {
            HStack(alignment: .firstTextBaseline) {
                Text(key).foregroundStyle(.secondary)
                Text(v.displayText).textSelection(.enabled)
            }
        } else {
            DisclosureGroup {
                JSONOutline(value: v).padding(.leading, Space.m)
            } label: {
                HStack {
                    Text(key)
                    Text(v.displayText).foregroundStyle(.tertiary)
                }
            }
        }
    }
}
