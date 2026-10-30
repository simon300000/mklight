import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

// Dedicated views of the signed Cartesian field maps (docs/RESULTS.md):
// `pupil_aberration`, `relative_illumination_map` and `chief_ray_angle`.
// They share `FieldMapData`; a grid run is drawn as a map over the
// normalised field, an x / y sweep as a curve.

// MARK: - Shared pieces

/// A menu choosing a column of a field map; the selected column's
/// definition is the help text.
struct FieldColumnPicker: View {
    let columns: [FieldMapData.Column]
    @Binding var selection: Int

    var body: some View {
        Picker("Quantity", selection: $selection) {
            ForEach(columns.indices, id: \.self) { k in
                Text(columns[k].titleWithUnit).tag(k).help(columns[k].definition)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .fixedSize()
        .help(columns.indices.contains(selection) ? columns[selection].definition : "")
        .accessibilityLabel("Quantity")
    }
}

/// A heat map over the normalised field with axes: the samples' values
/// interpolated bilinearly between the nodes (dots), optional iso-lines,
/// and the value of the nearest sample on hover.
struct FieldMapPlot: View {
    let grid: Grid2D
    var scale: HeatScale = .sequential(log: false)
    var range: ClosedRange<Double>
    var contours: [Double] = []
    /// A sample to ring (the worst one), as a node index.
    var marked: Int?
    var title: String
    var format: (Double) -> String = { Fmt.short($0) }
    @State private var hover: Int?
    @Environment(\.colorScheme) private var colorScheme

    private var count: Int { grid.nx * grid.ny }
    private var showsNodes: Bool { count <= 441 }

    private func node(_ k: Int) -> SIMD2<Double> {
        let i = k % grid.nx, j = k / grid.nx
        let dx = grid.nx > 1 ? (grid.xRange.upperBound - grid.xRange.lowerBound) / Double(grid.nx - 1) : 0
        let dy = grid.ny > 1 ? (grid.yRange.upperBound - grid.yRange.lowerBound) / Double(grid.ny - 1) : 0
        return SIMD2(grid.xRange.lowerBound + Double(i) * dx, grid.yRange.lowerBound + Double(j) * dy)
    }

    var aspect: CGFloat {
        CGFloat((grid.xRange.upperBound - grid.xRange.lowerBound) / max(grid.yRange.upperBound - grid.yRange.lowerBound, 1e-9))
    }

    var body: some View {
        let display = grid.upsampled(to: 192)
        let segments = contours.map { (level: $0, segments: Contours.segments(grid, level: $0)) }
        Chart {
            PointMark(x: .value("Hx", grid.xRange.lowerBound), y: .value("Hy", grid.yRange.lowerBound)).opacity(0)
            if showsNodes {
                ForEach(0..<count, id: \.self) { k in
                    let p = node(k)
                    PointMark(x: .value("Hx", p.x), y: .value("Hy", p.y))
                        .symbolSize(count > 121 ? 4 : 9)
                        .foregroundStyle(Color.primary.opacity(grid.values[k].isFinite ? 0.35 : 0.15))
                }
            }
            if let m = marked, m < count {
                let p = node(m)
                PointMark(x: .value("Hx", p.x), y: .value("Hy", p.y))
                    .symbol(Circle().strokeBorder(lineWidth: 1.5))
                    .symbolSize(90)
                    .foregroundStyle(Color.primary)
            }
        }
        .chartXScale(domain: grid.xRange)
        .chartYScale(domain: grid.yRange)
        .chartXAxis { AxisMarks(values: .automatic(desiredCount: 5)) }
        .chartYAxis { AxisMarks(values: .automatic(desiredCount: 5)) }
        .chartXAxisLabel("Hx", alignment: .center)
        .chartYAxisLabel("Hy")
        .chartPlotStyle { $0.aspectRatio(aspect, contentMode: .fit) }
        .chartBackground { proxy in
            GeometryReader { geo in
                if let f = proxy.plotFrame {
                    let r = geo[f]
                    // Pixels are centred on the nodes' interpolated values: extend by half a pixel.
                    let px = r.width / CGFloat(max(display.nx - 1, 1)), py = r.height / CGFloat(max(display.ny - 1, 1))
                    HeatMapImage(grid: display, scale: scale, range: range)
                        .frame(width: r.width + px, height: r.height + py)
                        .position(x: r.midX, y: r.midY)
                        .clipShape(Rectangle().path(in: r))
                }
            }
        }
        .chartOverlay { proxy in
            GeometryReader { geo in
                if let f = proxy.plotFrame {
                    let r = geo[f]
                    ZStack(alignment: .topLeading) {
                        Canvas { ctx, _ in drawContours(segments, in: r, ctx: &ctx) }
                            .allowsHitTesting(false)
                        Rectangle().fill(.clear).contentShape(Rectangle())
                            .onContinuousHover { phase in
                                switch phase {
                                case .active(let p): hover = nearest(to: p, in: r)
                                case .ended: hover = nil
                                }
                            }
                        if let h = hover { tooltip(h, in: r) }
                    }
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title) map over the normalised field, \(grid.nx) by \(grid.ny) samples")
        .accessibilityValue(accessibilityValue)
    }

    private var accessibilityValue: String {
        let f = grid.finite
        guard let lo = f.min(), let hi = f.max() else { return "No values" }
        return "From \(format(lo)) to \(format(hi))"
    }

    private func point(_ p: SIMD2<Double>, in r: CGRect) -> CGPoint {
        let x = (p.x - grid.xRange.lowerBound) / (grid.xRange.upperBound - grid.xRange.lowerBound)
        let y = (p.y - grid.yRange.lowerBound) / (grid.yRange.upperBound - grid.yRange.lowerBound)
        return CGPoint(x: r.minX + CGFloat(x) * r.width, y: r.maxY - CGFloat(y) * r.height)
    }

    private func nearest(to p: CGPoint, in r: CGRect) -> Int? {
        guard r.insetBy(dx: -6, dy: -6).contains(p) else { return nil }
        var best: (Int, CGFloat)?
        for k in 0..<count {
            let q = point(node(k), in: r)
            let d = hypot(q.x - p.x, q.y - p.y)
            if best == nil || d < best!.1 { best = (k, d) }
        }
        return best?.0
    }

    private func drawContours(_ levels: [(level: Double, segments: [Contours.Segment])], in r: CGRect, ctx: inout GraphicsContext) {
        let ink = Color.primary.opacity(0.65)
        for (level, segs) in levels where !segs.isEmpty {
            var path = Path()
            for s in segs {
                path.move(to: point(s.a, in: r))
                path.addLine(to: point(s.b, in: r))
            }
            ctx.stroke(path, with: .color(ink), style: StrokeStyle(lineWidth: 0.9, lineCap: .round))
            // Label each level once, at its segment nearest the upper right.
            if let s = segs.max(by: { ($0.a.x + $0.a.y + $0.b.x + $0.b.y) < ($1.a.x + $1.a.y + $1.b.x + $1.b.y) }) {
                let mid = point((s.a + s.b) / 2, in: r)
                let text = ctx.resolve(Text(format(level)).font(.caption2.monospacedDigit()).foregroundStyle(Color.primary))
                let size = text.measure(in: CGSize(width: 80, height: 20))
                let box = CGRect(x: mid.x - size.width / 2 - 2, y: mid.y - size.height / 2, width: size.width + 4, height: size.height)
                    .offsetBy(dx: min(0, r.maxX - (mid.x + size.width / 2 + 2)), dy: max(0, r.minY - (mid.y - size.height / 2)))
                ctx.fill(Path(roundedRect: box, cornerRadius: 3), with: .color(colorScheme == .dark ? .black.opacity(0.6) : .white.opacity(0.75)))
                ctx.draw(text, at: CGPoint(x: box.midX, y: box.midY))
            }
        }
    }

    @ViewBuilder private func tooltip(_ k: Int, in r: CGRect) -> some View {
        let p = node(k)
        let q = point(p, in: r)
        let v = grid.values[k]
        VStack(alignment: .leading, spacing: 1) {
            Text(v.isFinite ? format(v) : "No value").font(.caption.weight(.semibold))
            Text(String(format: "Hx %.3g  Hy %.3g", p.x, p.y)).font(.caption2).foregroundStyle(.secondary)
        }
        .monospacedDigit()
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(.background, in: .rect(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(.separator))
        .fixedSize()
        .offset(x: min(q.x + 8, r.maxX - 90), y: max(q.y - 34, r.minY))
        .allowsHitTesting(false)
    }
}

/// A field map with its colour bar: as large as fits, square in field units.
struct FieldMapPanel: View {
    let plot: FieldMapPlot
    let unit: String

    var body: some View {
        HStack(alignment: .center, spacing: Space.s) {
            plot
            ColorBar(scale: plot.scale, range: plot.range, unit: unit).frame(maxHeight: 220)
        }
    }
}

/// The value format of a column: degrees, fractions, lengths.
func fieldValueFormat(_ c: FieldMapData.Column) -> (Double) -> String {
    switch c.unit {
    case "deg": { $0.isFinite ? String(format: "%.2f°", $0) : "—" }
    case "mm": { Fmt.length($0) }
    case "1": { $0.isFinite ? String(format: "%.4g", $0) : "—" }
    default: { Fmt.value($0, unit: c.unit) }
    }
}

// MARK: - Pupil aberration

/// `pupil_aberration`: a chosen column as a map over the field (diverging
/// for signed quantities), or as a curve for a sweep.
struct PupilAberrationView: View {
    let data: FieldMapData
    @State private var column = 0

    init?(output: AnalysisOutput) {
        guard let d = FieldMapData(output: output) else { return nil }
        data = d
    }

    var body: some View {
        let c = min(column, data.columns.count - 1)
        let col = data.columns[c]
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(alignment: .firstTextBaseline, spacing: Space.m) {
                FieldColumnPicker(columns: data.columns, selection: $column)
                Text(col.definition).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            FieldMapContent(data: data, column: c)
        }
        .padding(Space.m)
    }
}

/// A column of a field map: the map (grid runs) or the curve (sweeps).
struct FieldMapContent: View {
    let data: FieldMapData
    let column: Int
    var contours: [Double] = []
    var range: ClosedRange<Double>?

    var body: some View {
        let col = data.columns[column]
        let format = fieldValueFormat(col)
        if data.layout == .map {
            let signed = data.isSigned(column)
            let r = range ?? data.range(column) ?? 0...1
            let marked = signed ? data.column(column).indices.max { abs(data.value($0, column)).nanToZero < abs(data.value($1, column)).nanToZero } : nil
            FieldMapPanel(
                plot: FieldMapPlot(
                    grid: data.grid(column), scale: signed ? .diverging : .sequential(log: false), range: r, contours: contours,
                    marked: marked, title: col.title, format: format),
                unit: col.unitLabel)
        } else {
            LineChart(
                series: [Series(id: "v", label: col.title, color: Palette.fieldColor(0), points: data.sweep(column))],
                xLabel: data.layout == .sweepX ? "Hx" : "Hy", yLabel: col.titleWithUnit, zeroLine: data.isSigned(column))
                .accessibilityValue(data.range(column).map { "From \(format($0.lowerBound)) to \(format($0.upperBound))" } ?? "")
        }
    }
}

extension Double {
    fileprivate var nanToZero: Double { isFinite ? self : 0 }
}

// MARK: - Relative illumination map

/// `relative_illumination_map`: the map with iso-illumination lines and
/// the illumination against image radius alongside.
struct RelativeIlluminationMapView: View {
    let data: FieldMapData
    @State private var column: Int

    init?(output: AnalysisOutput) {
        guard let d = FieldMapData(output: output) else { return nil }
        data = d
        _column = State(initialValue: d.columnIndex("relative_illumination") ?? 0)
    }

    private var isFraction: Bool { ["relative_illumination", "vignetting_fraction"].contains(data.columns[min(column, data.columns.count - 1)].name) }

    var body: some View {
        let c = min(column, data.columns.count - 1)
        let col = data.columns[c]
        let isRI = col.name == "relative_illumination"
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(alignment: .firstTextBaseline, spacing: Space.m) {
                FieldColumnPicker(columns: data.columns, selection: $column)
                Text(col.definition).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            if data.layout == .map {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .center, spacing: Space.l) {
                        map(c, isRI: isRI)
                        radial(c).frame(minWidth: 240, maxWidth: 360)
                    }
                    .frame(minWidth: 560)
                    VStack(alignment: .leading, spacing: Space.m) {
                        map(c, isRI: isRI)
                        radial(c).frame(minHeight: 140, maxHeight: 200)
                    }
                }
            } else {
                FieldMapContent(data: data, column: c)
            }
        }
        .padding(Space.m)
    }

    private func map(_ c: Int, isRI: Bool) -> some View {
        let r = isFraction ? 0...1 : (data.range(c) ?? 0...1)
        let levels = isRI ? Contours.illuminationLevels(in: data.range(c) ?? 0...1) : []
        return FieldMapContent(data: data, column: c, contours: levels, range: r)
    }

    private func radial(_ c: Int) -> some View {
        let col = data.columns[c]
        let pts = data.radial(c)
        let line = data.isRadial(c)
        let format = fieldValueFormat(col)
        return VStack(alignment: .leading, spacing: Space.xs) {
            Text("\(col.title) vs image radius").font(.caption.weight(.semibold))
            Chart {
                ForEach(Array(pts.enumerated()), id: \.offset) { _, p in
                    if line {
                        LineMark(x: .value("Image radius (mm)", p.0), y: .value(col.title, p.1))
                            .foregroundStyle(Palette.fieldColor(0))
                            .lineStyle(StrokeStyle(lineWidth: 1.6))
                    }
                    PointMark(x: .value("Image radius (mm)", p.0), y: .value(col.title, p.1))
                        .foregroundStyle(Palette.fieldColor(0))
                        .symbolSize(line ? 14 : 18)
                }
            }
            .chartXAxisLabel("Image radius (mm)", alignment: .center)
            .chartYAxisLabel(col.titleWithUnit)
            .chartYScale(domain: isFraction ? 0...1.02 : FieldMapData.range(pts.map(\.1)))
            .chartYAxis { AxisMarks(values: .automatic(desiredCount: 4)) }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("\(col.title) versus image radius, \(pts.count) samples")
            .accessibilityValue(pts.isEmpty ? "No values" : "From \(format(pts.map(\.1).min()!)) to \(format(pts.map(\.1).max()!))")
        }
    }
}

// MARK: - Chief ray angle

/// `chief_ray_angle`: the incidence against image height with the sensor
/// limit and the margin between them (red where violated); a map over the
/// field for asymmetric systems.
struct ChiefRayAngleView: View {
    enum Display: String, CaseIterable { case curve = "Curve", map = "Map" }
    let data: FieldMapData
    let limit: [(Double, Double)]
    let incidence: Int
    @State private var display: Display
    @State private var column: Int

    init?(output: AnalysisOutput) {
        guard let d = FieldMapData(output: output), let inc = d.columnIndex("incidence") else { return nil }
        data = d
        incidence = inc
        limit = FieldMapData.sensorLimit(output.runInfo.settings["sensor_limit"])
        _display = State(initialValue: d.layout == .map && !d.isRadial(inc, tolerance: 1e-3) ? .map : .curve)
        _column = State(initialValue: inc)
    }

    /// Image height along the sweep (signed) or the image radius (map runs).
    private var heights: [Double] {
        switch data.layout {
        case .sweepX: data.image.map(\.x)
        case .sweepY: data.image.map(\.y)
        case .map: data.imageRadius
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.m) {
                if data.layout == .map {
                    Picker("Display", selection: $display) {
                        ForEach(Display.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                }
                if display == .map && data.layout == .map {
                    FieldColumnPicker(columns: data.columns.filter { $0.name != "limit" }, selection: filteredColumn)
                } else {
                    legend
                }
                Spacer(minLength: 0)
            }
            if display == .map && data.layout == .map {
                FieldMapContent(data: data, column: column)
            } else {
                curve
            }
        }
        .padding(Space.m)
    }

    /// The picker over the columns without `limit` (constant in a map).
    private var filteredColumn: Binding<Int> {
        let shown = data.columns.indices.filter { data.columns[$0].name != "limit" }
        return Binding(
            get: { shown.firstIndex(of: column) ?? 0 },
            set: { column = shown[min(max($0, 0), shown.count - 1)] })
    }

    private var legend: some View {
        FlowLayout(spacing: Space.m) {
            LegendItem(color: Palette.fieldColor(0), label: "Chief ray incidence")
            if !limit.isEmpty {
                LegendItem(color: .secondary, label: "Sensor limit", dashed: true)
                HStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 2).fill(Palette.fieldColor(5).opacity(0.3)).frame(width: 12, height: 8)
                    Text("Margin")
                }
                .font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 2).fill(Palette.fieldColor(7).opacity(0.45)).frame(width: 12, height: 8)
                    Text("Violation")
                }
                .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    struct Sample: Identifiable {
        var id: Int
        var h: Double
        var cra: Double
        var margin: Double
    }

    private var samples: [Sample] {
        let m = data.columnIndex("margin")
        return (0..<data.count).compactMap { k in
            let h = heights[k], v = data.value(k, incidence)
            guard h.isFinite, v.isFinite else { return nil }
            return Sample(id: k, h: h, cra: v, margin: m.map { data.value(k, $0) } ?? .nan)
        }
        .sorted { $0.h < $1.h }
    }

    /// The curve through the samples: a sweep in order, a symmetric map's
    /// unique radii; nil when the incidence is not a function of radius.
    private var curvePoints: [(Double, Double)]? {
        let s = samples
        if data.layout != .map { return s.map { ($0.h, $0.cra) } }
        guard data.isRadial(incidence) else { return nil }
        var out: [(Double, Double)] = []
        for p in s where out.last.map({ abs($0.0 - p.h) > 1e-9 }) ?? true { out.append((p.h, p.cra)) }
        return out
    }

    /// Area segments between the curve and the limit, split where they cross.
    struct Band: Identifiable {
        var id: Int
        var violated: Bool
        var points: [(x: Double, lo: Double, hi: Double)]
    }

    static func bands(curve: [(Double, Double)], limit: [(Double, Double)]) -> [Band] {
        guard curve.count >= 2, !limit.isEmpty else { return [] }
        // Densify at the curve's and the limit's own break points.
        let lo = curve.first!.0, hi = curve.last!.0
        var xs = Set(curve.map(\.0))
        for (r, _) in limit {
            for x in [r, -r] where x > lo && x < hi { xs.insert(x) }
        }
        let sorted = xs.sorted()
        func cra(_ x: Double) -> Double {
            for k in 1..<curve.count where x <= curve[k].0 {
                let (x0, y0) = curve[k - 1], (x1, y1) = curve[k]
                return x1 > x0 ? y0 + (y1 - y0) * (x - x0) / (x1 - x0) : y1
            }
            return curve.last!.1
        }
        var pts: [(x: Double, margin: Double, c: Double, l: Double)] = sorted.map { x in
            let c = cra(x), l = FieldMapData.limit(limit, at: abs(x))
            return (x, l - c, c, l)
        }
        // Insert the crossings.
        var k = 1
        while k < pts.count {
            let a = pts[k - 1], b = pts[k]
            if (a.margin < 0) != (b.margin < 0), a.margin != 0, b.margin != 0 {
                let t = a.margin / (a.margin - b.margin)
                let x = a.x + (b.x - a.x) * t
                let c = cra(x)
                pts.insert((x, 0, c, c), at: k)
                k += 1
            }
            k += 1
        }
        var bands: [Band] = []
        for k in 1..<max(pts.count, 1) {
            let a = pts[k - 1], b = pts[k]
            let violated = (a.margin + b.margin) / 2 < 0
            let ia = (x: a.x, lo: min(a.c, a.l), hi: max(a.c, a.l)), ib = (x: b.x, lo: min(b.c, b.l), hi: max(b.c, b.l))
            if let last = bands.last, last.violated == violated {
                bands[bands.count - 1].points.append(ib)
            } else {
                bands.append(Band(id: bands.count, violated: violated, points: [ia, ib]))
            }
        }
        return bands.filter { $0.points.count >= 2 }
    }

    private var curve: some View {
        let s = samples
        let line = curvePoints
        let bands = line.map { Self.bands(curve: $0, limit: limit) } ?? []
        let xs = s.map(\.h)
        let xr = FieldMapData.range(xs + (data.layout == .map ? [0] : []))
        let limitPts = limitCurve(over: xr)
        let ys = s.map(\.cra) + limitPts.map(\.1)
        let yr = 0...max((ys.filter(\.isFinite).max() ?? 1) * 1.08, 1e-3)
        let xLabel = data.layout == .map ? "Image radius (mm)" : data.layout == .sweepX ? "Image height x (mm)" : "Image height y (mm)"
        return Chart {
            ForEach(bands) { b in
                ForEach(Array(b.points.enumerated()), id: \.offset) { _, p in
                    AreaMark(x: .value("h", p.x), yStart: .value("lo", p.lo), yEnd: .value("hi", p.hi), series: .value("band", "b\(b.id)"))
                        .foregroundStyle(b.violated ? Palette.fieldColor(7).opacity(0.45) : Palette.fieldColor(5).opacity(0.22))
                        .interpolationMethod(.linear)
                }
            }
            ForEach(Array(limitPts.enumerated()), id: \.offset) { _, p in
                LineMark(x: .value("h", p.0), y: .value("deg", p.1), series: .value("s", "limit"))
                    .foregroundStyle(Color.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1.4, dash: [5, 3]))
            }
            if let line {
                ForEach(Array(line.enumerated()), id: \.offset) { _, p in
                    LineMark(x: .value("h", p.0), y: .value("deg", p.1), series: .value("s", "cra"))
                        .foregroundStyle(Palette.fieldColor(0))
                        .lineStyle(StrokeStyle(lineWidth: 1.8, lineCap: .round))
                }
            }
            ForEach(s) { p in
                let bad = p.margin < 0
                PointMark(x: .value("h", p.h), y: .value("deg", p.cra))
                    .symbol(bad ? .triangle : .circle)
                    .symbolSize(bad ? 40 : (line == nil ? 22 : 14))
                    .foregroundStyle(bad ? Palette.fieldColor(7) : Palette.fieldColor(0))
            }
        }
        .chartXScale(domain: xr)
        .chartYScale(domain: yr)
        .chartXAxisLabel(xLabel, alignment: .center)
        .chartYAxisLabel("Chief ray angle (°)")
        .chartYAxis { AxisMarks(values: .automatic(desiredCount: 5)) }
        .environment(\.locale, Locale(identifier: "en_US_POSIX"))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Chief ray angle versus image height" + (limit.isEmpty ? "" : ", with the sensor limit"))
        .accessibilityValue(accessibilitySummary(s))
    }

    private func accessibilitySummary(_ s: [Sample]) -> String {
        guard let worst = s.max(by: { $0.cra < $1.cra }) else { return "No values" }
        var t = String(format: "Largest %.2f degrees at %.3g millimetres", worst.cra, worst.h)
        let bad = s.filter { $0.margin < 0 }.count
        if !limit.isEmpty { t += bad == 0 ? ", within the sensor limit" : ", \(bad) samples exceed the sensor limit" }
        return t
    }

    /// The limit table over the plotted heights (both signs for sweeps),
    /// held at its end points.
    private func limitCurve(over xr: ClosedRange<Double>) -> [(Double, Double)] {
        guard !limit.isEmpty else { return [] }
        var xs = Set([xr.lowerBound, xr.upperBound, 0].filter { xr.contains($0) })
        for (r, _) in limit { for x in [r, -r] where xr.contains(x) { xs.insert(x) } }
        return xs.sorted().map { ($0, FieldMapData.limit(limit, at: abs($0))) }
    }
}
