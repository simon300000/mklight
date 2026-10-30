import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

// Dedicated views of the wave-optics analyses (docs/RESULTS.md, "Wave
// optics"). Large arrays travel as buffers named in the JSON.

/// A menu choosing one of several items (field / wavelength) of a result.
struct ItemPicker: View {
    let titles: [String]
    @Binding var selection: Int

    var body: some View {
        if titles.count > 1 {
            Picker("Show", selection: $selection) {
                ForEach(titles.indices, id: \.self) { Text(titles[$0]).tag($0) }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .fixedSize()
        } else if let t = titles.first {
            Text(t).font(.caption.weight(.semibold))
        }
    }
}

// MARK: - Wavefront map

struct WavefrontView: View {
    struct Map {
        var title: String
        var grid: Grid2D
        var rms: Double
        var pv: Double
        var strehl: Double
        var zernike: [Double]
        var ordering: String
        var rim: [[(Double, Double)]]
    }
    let maps: [Map]
    @State private var selected = 0

    init?(output: AnalysisOutput, context: AnalysisContext) {
        guard let list = output.json["maps"]?.array, !list.isEmpty else { return nil }
        maps = list.compactMap { m in
            guard let g = Grid2D(output.buffer(ref: m["opd_waves"])) else { return nil }
            let wl = m["wavelength_um"]?.nf ?? .nan
            let field = m["field"]?.int ?? 0
            return Map(
                title: "Field \(field + 1) · \(context.fieldLabel(m["field_value"]?.nfArray)) · \(String(format: "%.4f µm", wl))",
                grid: g, rms: m["rms_waves"]?.nf ?? .nan, pv: m["pv_waves"]?.nf ?? .nan,
                strehl: m["strehl_at_reference"]?.nf ?? .nan,
                zernike: output.buffer(ref: m[path: "zernike", "coefficients"])?.doubles ?? m[path: "zernike", "coefficients"]?.nfArray ?? [],
                ordering: m[path: "zernike", "ordering"]?.string ?? "fringe",
                rim: (m["rim"]?.array ?? []).compactMap { r in
                    output.buffer(ref: r).map { b in (0..<b.rows).map { (b[$0 * 2], b[$0 * 2 + 1]) } }
                })
        }
        if maps.isEmpty { return nil }
    }

    var body: some View {
        let m = maps[min(selected, maps.count - 1)]
        VStack(alignment: .leading, spacing: Space.s) {
            ItemPicker(titles: maps.map(\.title), selection: $selected)
            Text(String(format: "RMS %.4f λ   P-V %.4f λ   Strehl %.3f", m.rms, m.pv, m.strehl))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            GeometryReader { geo in
                let wide = geo.size.width >= 500 && !m.zernike.isEmpty
                let tableWidth: CGFloat = wide ? 200 : 0
                let side = max(min(wide ? geo.size.height : geo.size.height * 0.55, geo.size.width - tableWidth - 70), 60)
                let map = HStack(alignment: .center, spacing: Space.s) {
                    ZStack {
                        HeatMapImage(grid: m.grid, scale: .diverging)
                        Canvas { ctx, size in
                            for loop in m.rim where loop.count > 2 {
                                var p = Path()
                                for (k, q) in loop.enumerated() {
                                    let pt = CGPoint(x: (q.0 + 1) / 2 * size.width, y: (1 - q.1) / 2 * size.height)
                                    if k == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
                                }
                                p.closeSubpath()
                                ctx.stroke(p, with: .color(.secondary), lineWidth: 0.8)
                            }
                        }
                    }
                    .frame(width: side, height: side)
                    .accessibilityLabel(String(format: "Wavefront map, RMS %.3f waves, peak to valley %.3f waves", m.rms, m.pv))
                    ColorBar(scale: .diverging, range: m.grid.range, unit: "waves").frame(height: min(side, 240))
                }
                if wide {
                    HStack(alignment: .top, spacing: Space.l) {
                        map
                        Spacer(minLength: 0)
                        ZernikeTable(coefficients: m.zernike, ordering: m.ordering).frame(width: tableWidth)
                    }
                } else {
                    VStack(alignment: .leading, spacing: Space.m) {
                        map
                        if !m.zernike.isEmpty { ZernikeTable(coefficients: m.zernike, ordering: m.ordering) }
                    }
                }
            }
        }
        .padding(Space.m)
    }
}

struct ZernikeTable: View {
    let coefficients: [Double]
    let ordering: String

    static let fringeNames = [
        "Piston", "Tilt x", "Tilt y", "Defocus", "Astig 0°", "Astig 45°", "Coma x", "Coma y", "Spherical", "Trefoil x",
        "Trefoil y",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text("Zernike (\(ordering.capitalized), waves)").font(.caption.weight(.semibold))
            ScrollView {
                Grid(alignment: .leading, horizontalSpacing: Space.m, verticalSpacing: 2) {
                    ForEach(coefficients.indices, id: \.self) { k in
                        GridRow {
                            Text("Z\(k + 1)").foregroundStyle(.secondary)
                            Text(ordering == "fringe" && k < Self.fringeNames.count ? Self.fringeNames[k] : "")
                                .foregroundStyle(.secondary).lineLimit(1).fixedSize()
                            Text(String(format: "%+.4f", coefficients[k])).numeric().fixedSize().gridColumnAlignment(.trailing)
                                .foregroundStyle(abs(coefficients[k]) >= 0.05 ? .primary : .secondary)
                        }
                    }
                }
                .font(.caption)
                .textSelection(.enabled)
            }
        }
    }
}

// MARK: - PSF

struct PSFView: View {
    struct Item { var title: String; var grid: Grid2D; var strehl: Double; var spacing: Double; var size: Int }
    let items: [Item]
    @State private var selected = 0
    @State private var log = false

    init?(output: AnalysisOutput, context: AnalysisContext) {
        guard let list = output.json["psfs"]?.array, !list.isEmpty else { return nil }
        items = list.compactMap { p in
            let psf = p["psf"]
            guard let n = psf?["size"]?.int, let h = psf?["spacing_mm"]?.nf,
                var g = Grid2D(output.buffer(ref: psf?["data"]))
            else { return nil }
            let c = psf?["center_mm"]?.nfArray ?? [0, 0]
            let half = Double(n / 2) * h
            g.xRange = (c[0] - half)...(c[0] + half)
            g.yRange = (c[1] - half)...(c[1] + half)
            let wl = psf?["wavelength_um"]?.nf
            let field = p["field"]?.int ?? 0
            return Item(
                title: "Field \(field + 1) · \(context.fieldLabel(p["field_value"]?.nfArray))"
                    + (wl.map { $0.isFinite ? String(format: " · %.4f µm", $0) : " · polychromatic" } ?? " · polychromatic"),
                grid: g, strehl: p["strehl"]?.nf ?? .nan, spacing: h, size: n)
        }
        if items.isEmpty { return nil }
    }

    var body: some View {
        let it = items[min(selected, items.count - 1)]
        VStack(alignment: .leading, spacing: Space.s) {
            ItemPicker(titles: items.map(\.title), selection: $selected)
            HStack {
                Text(String(format: "Strehl %.3f", it.strehl)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
                Toggle("Log scale", isOn: $log).toggleStyle(.checkbox).controlSize(.small).fixedSize()
            }
            GeometryReader { geo in
                let side = max(min(geo.size.height, geo.size.width - 70), 60)
                HStack(spacing: Space.s) {
                    ZStack(alignment: .bottomLeading) {
                        HeatMapImage(grid: it.grid, scale: .sequential(log: log))
                        let width = Double(it.size) * it.spacing
                        let bar = Nice.length(width / 4)
                        VStack(alignment: .leading, spacing: 2) {
                            Rectangle().frame(width: side * CGFloat(bar / width), height: 1.5)
                            Text(Nice.lengthLabel(bar)).font(.caption2)
                        }
                        .foregroundStyle(.secondary)
                        .padding(Space.s)
                    }
                    .frame(width: side, height: side)
                    .accessibilityLabel(String(format: "Point spread function, Strehl ratio %.3f", it.strehl))
                    ColorBar(scale: .sequential(log: log), range: it.grid.range).frame(height: min(side, 240))
                    Spacer(minLength: 0)
                }
            }
        }
        .padding(Space.m)
    }
}

// MARK: - MTF vs field, through focus, line spread

struct MTFVsFieldView: View {
    let field: [Double]
    let curves: [(Double, [Double], [Double])]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let f = output.buffer(ref: j["normalized_field"])?.doubles ?? j["normalized_field"]?.nfArray,
            let list = j["curves"]?.array, !list.isEmpty
        else { return nil }
        field = f
        curves = list.map { c in
            (
                c["frequency"]?.nf ?? .nan, output.buffer(ref: c["tangential"])?.doubles ?? [],
                output.buffer(ref: c["sagittal"])?.doubles ?? []
            )
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            FlowLayout(spacing: Space.m) {
                ForEach(Array(curves.enumerated()), id: \.offset) { k, c in
                    LegendItem(color: Palette.fieldColor(k), label: String(format: "%.0f cycles/mm", c.0))
                }
                LegendItem(color: .secondary, label: "T solid, S dashed")
            }
            LineChart(
                series: curves.enumerated().flatMap { k, c in
                    [
                        Series(id: "t\(k)", label: "", color: Palette.fieldColor(k), points: zip(field, c.1).map { ($0, $1) }),
                        Series(id: "s\(k)", label: "", color: Palette.fieldColor(k), points: zip(field, c.2).map { ($0, $1) }, dashed: true),
                    ]
                }, xLabel: "Normalized field", yLabel: "Modulation", xDomain: 0...1, yDomain: 0...1.02, zeroLine: false)
        }
        .padding(Space.m)
    }
}

struct ThroughFocusMTFView: View {
    struct FieldCurves { var title: String; var curves: [(Double, [Double], [Double])] }
    let focus: [Double]
    let fields: [FieldCurves]
    @State private var selected = 0

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let f = output.buffer(ref: j["focus_mm"])?.doubles ?? j["focus_mm"]?.nfArray, let list = j["fields"]?.array, !list.isEmpty
        else { return nil }
        focus = f
        fields = list.map { fl in
            FieldCurves(
                title: "Field \((fl["field"]?.int ?? 0) + 1) · " + context.fieldLabel(fl["field_value"]?.nfArray),
                curves: (fl["curves"]?.array ?? []).map { c in
                    (c["frequency"]?.nf ?? .nan, output.buffer(ref: c["tangential"])?.doubles ?? [], output.buffer(ref: c["sagittal"])?.doubles ?? [])
                })
        }
    }

    var body: some View {
        let f = fields[min(selected, fields.count - 1)]
        VStack(alignment: .leading, spacing: Space.s) {
            HStack {
                ItemPicker(titles: fields.map(\.title), selection: $selected)
                Spacer()
                FlowLayout(spacing: Space.m) {
                    ForEach(Array(f.curves.enumerated()), id: \.offset) { k, c in
                        LegendItem(color: Palette.fieldColor(k), label: String(format: "%.0f c/mm", c.0))
                    }
                }
                .fixedSize()
            }
            LineChart(
                series: f.curves.enumerated().flatMap { k, c in
                    [
                        Series(id: "t\(k)", label: "", color: Palette.fieldColor(k), points: zip(focus, c.1).map { ($0, $1) }),
                        Series(id: "s\(k)", label: "", color: Palette.fieldColor(k), points: zip(focus, c.2).map { ($0, $1) }, dashed: true),
                    ]
                }, xLabel: "Focus shift (mm)", yLabel: "Modulation", yDomain: 0...1.02, zeroLine: false, markers: [(x: 0, label: "Image")])
        }
        .padding(Space.m)
    }
}

struct LineSpreadView: View {
    struct Item { var field: Int; var x: [Double]; var lx: [Double]; var ly: [Double] }
    let items: [Item]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        guard let list = output.json["fields"]?.array, !list.isEmpty else { return nil }
        items = list.enumerated().compactMap { k, f in
            guard let x = output.buffer(ref: f["positions_mm"])?.doubles else { return nil }
            return Item(
                field: f["field"]?.int ?? k, x: x.map { $0 * 1000 }, lx: output.buffer(ref: f["x"])?.doubles ?? [],
                ly: output.buffer(ref: f["y"])?.doubles ?? [])
        }
        if items.isEmpty { return nil }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            FlowLayout(spacing: Space.m) {
                ForEach(items, id: \.field) { LegendItem(color: Palette.fieldColor($0.field), label: "Field \($0.field + 1)") }
                LegendItem(color: .secondary, label: "x solid, y dashed")
            }
            LineChart(
                series: items.flatMap { i in
                    [
                        Series(id: "x\(i.field)", label: "", color: Palette.fieldColor(i.field), points: zip(i.x, i.lx).map { ($0, $1) }),
                        Series(id: "y\(i.field)", label: "", color: Palette.fieldColor(i.field), points: zip(i.x, i.ly).map { ($0, $1) }, dashed: true),
                    ]
                }, xLabel: "Position (µm)", yLabel: "Line spread (mm⁻¹)", zeroLine: false)
        }
        .padding(Space.m)
    }
}
