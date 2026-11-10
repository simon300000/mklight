import Charts
import CoreGraphics
import MKLightEngine
import MKLightModel
import SwiftUI

/// A row-major grid of values with its extent in plot units.
struct Grid2D: Sendable {
    var nx: Int
    var ny: Int
    /// Row 0 at the bottom (`y = yRange.lowerBound`), x fastest.
    var values: [Double]
    var xRange: ClosedRange<Double>
    var yRange: ClosedRange<Double>
    /// Where the map has no data by definition (`unitDisk`: outside the
    /// normalised field, hypot(Hx, Hy) > 1). Upsampling then interpolates
    /// cells with missing corners from their finite corners and clips at
    /// the boundary, so the map reaches the field's edge exactly.
    var noData: NoData?

    enum NoData: Sendable, Equatable {
        case unitDisk

        func excludes(_ x: Double, _ y: Double) -> Bool {
            switch self {
            case .unitDisk: (x * x + y * y).squareRoot() > 1 + 1e-9
            }
        }
    }

    init(nx: Int, ny: Int, values: [Double], xRange: ClosedRange<Double> = -1...1, yRange: ClosedRange<Double> = -1...1) {
        self.nx = nx
        self.ny = ny
        self.values = values
        self.xRange = xRange
        self.yRange = yRange
    }

    /// A `[ny, nx]` (or `[k, ny, nx]` slice `k`) buffer.
    init?(_ b: ResultBuffer?, slice k: Int = 0, xRange: ClosedRange<Double> = -1...1, yRange: ClosedRange<Double> = -1...1) {
        guard let b, b.shape.count >= 2 else { return nil }
        let ny = b.shape[b.shape.count - 2], nx = b.shape[b.shape.count - 1]
        let start = k * nx * ny
        guard nx > 0, ny > 0, start + nx * ny <= b.count else { return nil }
        self.init(nx: nx, ny: ny, values: (start..<(start + nx * ny)).map { b[$0] }, xRange: xRange, yRange: yRange)
    }

    var finite: [Double] { values.filter(\.isFinite) }
    var range: ClosedRange<Double> {
        let f = finite
        guard let lo = f.min(), let hi = f.max(), lo < hi else { return 0...1 }
        return lo...hi
    }
}

extension ResultBuffer {
    /// The rows of an `[n, 2]` buffer as pairs; `nil` for any other shape
    /// (an empty buffer has no rows).
    var pairs: [(Double, Double)]? {
        if count == 0 { return [] }
        guard shape.count == 2, shape[1] == 2, count == shape[0] * 2 else { return nil }
        return (0..<shape[0]).map { (self[$0 * 2], self[$0 * 2 + 1]) }
    }
}

enum HeatScale: Equatable {
    /// Magnitude from `lo` to `hi` (optionally logarithmic); zero recedes
    /// into the plot surface (PSF, irradiance).
    case sequential(log: Bool)
    /// Signed values, symmetric about zero.
    case diverging
    /// A quantity over its range: the perceptual multi-hue ramp, every
    /// value visible (field maps, pupil maps).
    case perceptual
}

/// A heat map drawn from a `Grid2D` (NaN cells transparent), with the
/// one-hue or diverging scale of `Palette`.
struct HeatMapImage: View {
    let grid: Grid2D
    var scale: HeatScale = .sequential(log: false)
    var range: ClosedRange<Double>?
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        if let image = makeImage(dark: colorScheme == .dark) {
            Image(decorative: image, scale: 1)
                .resizable()
                .interpolation(grid.nx > 96 ? .medium : .none)
        } else {
            Color.clear
        }
    }

    /// The smallest value a logarithmic scale tells apart: five decades
    /// below the largest.
    static func logFloor(_ r: ClosedRange<Double>) -> Double { max(r.upperBound * 1e-5, 1e-300) }

    static func normalise(_ v: Double, scale: HeatScale, range r: ClosedRange<Double>) -> Double {
        switch scale {
        case .sequential(let log):
            if log {
                let lo = logFloor(r)
                return (Foundation.log10(max(v, lo)) - Foundation.log10(lo)) / max(Foundation.log10(r.upperBound) - Foundation.log10(lo), 1e-12)
            }
            return (v - r.lowerBound) / max(r.upperBound - r.lowerBound, 1e-300)
        case .diverging:
            let m = max(abs(r.lowerBound), abs(r.upperBound), 1e-300)
            return v / m
        case .perceptual:
            return (v - r.lowerBound) / max(r.upperBound - r.lowerBound, 1e-300)
        }
    }

    private func makeImage(dark: Bool) -> CGImage? {
        let r = range ?? grid.range
        var px = [UInt8](repeating: 0, count: grid.nx * grid.ny * 4)
        for j in 0..<grid.ny {
            let row = grid.ny - 1 - j  // image row 0 at the top
            for i in 0..<grid.nx {
                let v = grid.values[j * grid.nx + i]
                let o = (row * grid.nx + i) * 4
                guard v.isFinite else { continue }
                let t = Self.normalise(v, scale: scale, range: r)
                let c =
                    switch scale {
                    case .diverging: Palette.diverging(t, dark: dark)
                    case .perceptual: Palette.perceptual(t)
                    case .sequential: Palette.sequential(t, dark: dark)
                    }
                px[o] = UInt8(min(max(c.x, 0), 1) * 255)
                px[o + 1] = UInt8(min(max(c.y, 0), 1) * 255)
                px[o + 2] = UInt8(min(max(c.z, 0), 1) * 255)
                px[o + 3] = 255
            }
        }
        let data = Data(px) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(
            width: grid.nx, height: grid.ny, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: grid.nx * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

/// A vertical colour bar with end labels. A logarithmic scale runs from
/// its floor (`HeatMapImage.logFloor`) to the largest value, evenly in
/// decades.
struct ColorBar: View {
    var scale: HeatScale
    var range: ClosedRange<Double>
    var unit: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if !unit.isEmpty { Text(unit) }
            Text(label(scale == .diverging ? max(abs(range.lowerBound), abs(range.upperBound)) : range.upperBound))
            HeatMapImage(grid: bar, scale: scale, range: barRange)
                .frame(width: 10)
                .clipShape(RoundedRectangle(cornerRadius: 2))
            Text(label(scale == .diverging ? -max(abs(range.lowerBound), abs(range.upperBound)) : low))
        }
        .fixedSize(horizontal: true, vertical: false)
        .font(.caption2.monospacedDigit())
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Color scale from \(label(low)) to \(label(range.upperBound))" + (isLog ? ", logarithmic" : ""))
    }

    private var isLog: Bool { scale == .sequential(log: true) }

    /// The value at the bar's lower end.
    private var low: Double { isLog ? HeatMapImage.logFloor(range) : range.lowerBound }

    private var barRange: ClosedRange<Double> {
        scale == .diverging ? -1...1 : 0...1
    }

    private var bar: Grid2D {
        let n = 64
        let floor = HeatMapImage.logFloor(barRange)
        let values = (0..<n).map { k -> Double in
            let t = Double(k) / Double(n - 1)
            return scale == .diverging ? 2 * t - 1 : isLog ? pow(floor, 1 - t) : t
        }
        return Grid2D(nx: 1, ny: n, values: values)
    }

    private func label(_ v: Double) -> String {
        guard v.isFinite else { return Fmt.short(v) }
        let a = abs(v)
        return a != 0 && (a < 1e-2 || a >= 1e4) ? String(format: "%.1e", v) : String(format: "%.3g", v)
    }
}

/// A legend entry: a short line or dot in the series colour plus its name.
struct LegendItem: View {
    let color: Color
    let label: String
    var dashed = false

    var body: some View {
        HStack(spacing: 4) {
            Canvas { ctx, size in
                var p = Path()
                p.move(to: CGPoint(x: 0, y: size.height / 2))
                p.addLine(to: CGPoint(x: size.width, y: size.height / 2))
                ctx.stroke(p, with: .color(color), style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: dashed ? [3, 2] : []))
            }
            .frame(width: 14, height: 8)
            Text(label).lineLimit(1)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

/// Legends for fields or wavelengths, wrapped, with optional further
/// entries (line styles, reference circles) in the same flow, so a narrow
/// pane wraps them instead of clipping.
struct WavelengthLegend<Extra: View>: View {
    let wavelengths: [Int]
    let um: [Double]
    @ViewBuilder var extra: Extra

    var body: some View {
        FlowLayout(spacing: Space.m) {
            ForEach(wavelengths, id: \.self) { w in
                if um.indices.contains(w) {
                    LegendItem(color: Palette.wavelengthColor(um[w]), label: String(format: "%.4f µm", um[w]))
                }
            }
            extra
        }
    }
}

extension WavelengthLegend where Extra == EmptyView {
    init(wavelengths: [Int], um: [Double]) {
        self.init(wavelengths: wavelengths, um: um) { EmptyView() }
    }
}

/// A segmented control that becomes a pop-up menu when its pane is too
/// narrow for the segments (narrow panes never overflow into their
/// neighbours).
struct SegmentedOrMenu<Value: Hashable>: View {
    let title: String
    let options: [(value: Value, label: String)]
    @Binding var selection: Value

    var body: some View {
        ViewThatFits(in: .horizontal) {
            picker.pickerStyle(.segmented).fixedSize()
            picker.pickerStyle(.menu).fixedSize()
        }
        .accessibilityLabel(title)
    }

    private var picker: some View {
        Picker(title, selection: $selection) {
            ForEach(options.indices, id: \.self) { k in Text(options[k].label).tag(options[k].value) }
        }
        .labelsHidden()
    }
}

/// A titled small chart frame used in grids of plots.
struct PlotCell<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(title).font(.caption.weight(.semibold)).lineLimit(1).truncationMode(.middle)
            content
        }
    }
}

/// One `(x, y)` series of a line chart.
struct Series: Identifiable {
    var id: String
    var label: String
    var color: Color
    var points: [(Double, Double)]
    var dashed = false
}

/// A line chart of several series with axis titles; NaN breaks a line.
struct LineChart: View {
    let series: [Series]
    var xLabel: String
    var yLabel: String
    var xDomain: ClosedRange<Double>?
    var yDomain: ClosedRange<Double>?
    var zeroLine = true
    var markers: [(x: Double, label: String)] = []
    var xTicks: [Double]?

    var body: some View {
        // About one x label per 60 pt (at least three), so narrow panes do not overlap them.
        GeometryReader { geo in
            axes(chart, xCount: max(3, min(6, Int(geo.size.width / 60))))
        }
    }

    private func axes(_ chart: some View, xCount: Int) -> some View {
        chart
            .chartXAxis {
                // `.aligned` keeps the first and last labels inside the plot,
                // so narrow panes do not drop the end labels.
                if let xTicks {
                    AxisMarks(values: xTicks) { v in
                        AxisGridLine()
                        AxisTick()
                        // End labels grow inwards, so neither is clipped.
                        AxisValueLabel(anchor: v.index == 0 ? .topLeading : v.index == v.count - 1 ? .topTrailing : .top)
                    }
                } else {
                    AxisMarks(preset: .aligned, values: .automatic(desiredCount: xCount))
                }
            }
            // Few y labels: fan and profile plots are often short.
            .chartYAxis { AxisMarks(values: .automatic(desiredCount: 3)) }
            .environment(\.locale, NumberDisplay.locale)
    }

    private var chart: some View {
        Chart {
            if zeroLine {
                RuleMark(y: .value("zero", 0)).foregroundStyle(.secondary.opacity(0.4)).lineStyle(StrokeStyle(lineWidth: 0.5))
            }
            ForEach(markers, id: \.label) { m in
                RuleMark(x: .value(m.label, m.x))
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [2, 3]))
                    .annotation(position: .top, alignment: .leading, spacing: 2) {
                        Text(m.label).font(.caption2).foregroundStyle(.secondary)
                    }
            }
            ForEach(series) { s in
                ForEach(Array(segments(s.points).enumerated()), id: \.offset) { k, seg in
                    ForEach(Array(seg.enumerated()), id: \.offset) { _, p in
                        LineMark(x: .value(xLabel, p.0), y: .value(yLabel, p.1), series: .value("s", "\(s.id)-\(k)"))
                            .foregroundStyle(s.color)
                            .lineStyle(StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round, dash: s.dashed ? [4, 3] : []))
                            .interpolationMethod(.linear)
                    }
                }
            }
        }
        .chartXAxisLabel(xLabel, alignment: .center)
        .chartYAxisLabel(yLabel)
        .chartXScale(domain: xDomain ?? autoDomain(\.0))
        .chartYScale(domain: yDomain ?? autoDomain(\.1))
        .chartLegend(.hidden)
        .accessibilityLabel(yLabel + " versus " + xLabel)
    }

    /// Splits a series at non-finite values.
    private func segments(_ pts: [(Double, Double)]) -> [[(Double, Double)]] {
        var out: [[(Double, Double)]] = [[]]
        for p in pts {
            if p.0.isFinite && p.1.isFinite { out[out.count - 1].append(p) } else if !(out.last?.isEmpty ?? true) { out.append([]) }
        }
        return out.filter { !$0.isEmpty }
    }

    private func autoDomain(_ key: KeyPath<(Double, Double), Double>) -> ClosedRange<Double> {
        let v = series.flatMap { $0.points.map { $0[keyPath: key] } }.filter(\.isFinite)
        guard var lo = v.min(), var hi = v.max() else { return 0...1 }
        if zeroLine && key == \.1 { lo = min(lo, 0); hi = max(hi, 0) }
        if lo == hi { lo -= 1; hi += 1 }
        let pad = (hi - lo) * 0.04
        return (lo - (key == \.1 ? pad : 0))...(hi + (key == \.1 ? pad : 0))
    }
}

/// The merit history of an optimisation, on a logarithmic scale.
struct Sparkline: View {
    let values: [Double]

    var body: some View {
        let logs = values.map { $0 > 0 ? log10($0) : .nan }
        Chart {
            ForEach(Array(logs.enumerated()), id: \.offset) { k, v in
                if v.isFinite {
                    LineMark(x: .value("Iteration", k), y: .value("log₁₀ merit", v))
                        .interpolationMethod(.monotone)
                        .lineStyle(StrokeStyle(lineWidth: 1.6))
                }
            }
        }
        .chartXAxis(.hidden)
        .chartYAxis {
            AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) { v in
                AxisGridLine()
                AxisValueLabel {
                    if let d = v.as(Double.self) { Text(String(format: "%.1e", pow(10, d))).font(.caption2) }
                }
            }
        }
        .chartYScale(domain: domain(logs))
        .environment(\.locale, NumberDisplay.locale)
        .accessibilityLabel("Merit history, \(values.count) iterations, logarithmic scale")
    }

    private func domain(_ v: [Double]) -> ClosedRange<Double> {
        let f = v.filter(\.isFinite)
        guard let lo = f.min(), let hi = f.max(), lo < hi else { return (f.first ?? 0) - 1...(f.first ?? 0) + 1 }
        let pad = (hi - lo) * 0.05
        return (lo - pad)...(hi + pad)
    }
}

/// A small key / value table.
struct KeyValueGrid: View {
    let rows: [(String, String)]

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: Space.xs) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                GridRow {
                    Text(r.0).foregroundStyle(.secondary)
                    Text(r.1).numeric().gridColumnAlignment(.trailing)
                }
            }
        }
        .font(.callout)
        .textSelection(.enabled)
    }
}

/// Nice tick step and "scale bar" length.
enum Nice {
    static func length(_ x: Double) -> Double {
        guard x > 0, x.isFinite else { return 1 }
        let e = pow(10, floor(log10(x)))
        let m = x / e
        return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * e
    }

    static func lengthLabel(_ mm: Double) -> String {
        mm >= 1 ? String(format: "%g mm", mm) : mm >= 0.001 ? String(format: "%g µm", mm * 1000) : String(format: "%g nm", mm * 1e6)
    }
}

/// A map (or pupil plot) with its colour bar and a side table: the map is as
/// large as fits; the table moves below it in narrow panes.
struct MapLayout<Plot: View, Bar: View, Side: View>: View {
    var aspect: CGFloat = 1
    @ViewBuilder var plot: Plot
    @ViewBuilder var bar: Bar
    @ViewBuilder var side: Side

    var body: some View {
        GeometryReader { geo in
            let wide = geo.size.width > 560
            let sideW: CGFloat = wide ? 210 : 0
            let availW = max(geo.size.width - sideW - 70 - (wide ? Space.l : 0), 40)
            let availH = max(geo.size.height - (wide ? 0 : 150), 40)
            let w = min(availW, availH * aspect)
            let h = w / max(aspect, 0.01)
            if wide {
                HStack(alignment: .top, spacing: Space.l) {
                    HStack(spacing: Space.s) {
                        plot.frame(width: w, height: h)
                        bar.frame(height: min(h, 240))
                    }
                    Spacer(minLength: 0)
                    ScrollView { side }.frame(width: sideW)
                }
            } else {
                VStack(alignment: .leading, spacing: Space.m) {
                    HStack(spacing: Space.s) {
                        plot.frame(width: w, height: h)
                        bar.frame(height: min(h, 240))
                    }
                    ScrollView { side }
                }
            }
        }
    }
}
