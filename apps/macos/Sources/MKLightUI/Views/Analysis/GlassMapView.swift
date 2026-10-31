import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

// The glass map (`glass_map`, docs/RESULTS.md): catalogue glasses in the
// n_d – V_d or P_g,F – V_d plane, V_d decreasing to the right as glass maps
// are drawn. `GlassMapChart` is reusable: give it points, the glasses to
// mark with their labels, and a selection.

/// The plane a glass map shows.
enum GlassMapPlane: String, CaseIterable, Identifiable, Sendable {
    case ndVd, pgfVd
    var id: String { rawValue }

    var title: String { self == .ndVd ? "n_d" : "P_g,F" }
    var segmentTitle: String { self == .ndVd ? "Index" : "Partial Dispersion" }

    /// The y axis title with its subscript.
    var axisLabel: Text {
        self == .ndVd
            ? Text("Refractive index n") + GlassMapPlane.sub("d") : Text("Partial dispersion P") + GlassMapPlane.sub("g,F")
    }

    static func sub(_ s: String) -> Text { Text(s).font(.caption2).baselineOffset(-3) }

    func y(_ g: GlassMapPoint) -> Double { self == .ndVd ? g.nd : g.pgf }
}

/// An interactive glass map: catalogue glasses as small points coloured
/// and shaped by status, highlighted glasses larger with an accent ring and
/// their label, hover for a glass's data, click to select. Hover and hit
/// testing search the points in screen space (cheap for thousands).
struct GlassMapChart: View {
    let glasses: [GlassMapPoint]
    var highlighted: [GlassHighlight] = []
    @Binding var selection: String?
    var plane: GlassMapPlane = .ndVd

    private var plotted: [GlassMapPoint] { glasses.filter { $0.vd.isFinite && plane.y($0).isFinite } }

    /// V_d is plotted negated, so the axis decreases to the right.
    var xDomain: ClosedRange<Double> {
        let v = plotted.map(\.vd)
        let lo = v.min() ?? 20, hi = v.max() ?? 90
        let pad = max((hi - lo) * 0.05, 2)
        return -(hi + pad)...(-(lo - pad))
    }

    var yDomain: ClosedRange<Double> {
        let v = plotted.map { plane.y($0) }
        let lo = v.min() ?? (plane == .ndVd ? 1.4 : 0.5), hi = v.max() ?? (plane == .ndVd ? 2 : 0.65)
        let pad = max((hi - lo) * 0.06, plane == .ndVd ? 0.02 : 0.005)
        return (lo - pad)...(hi + pad)
    }

    var body: some View {
        let xd = xDomain, yd = yDomain
        Chart {
            PointMark(x: .value("V_d", xd.lowerBound), y: .value(plane.title, yd.lowerBound)).opacity(0)
            if plane == .pgfVd {
                // The normal line through K7 and F2.
                ForEach([xd.lowerBound, xd.upperBound], id: \.self) { x in
                    LineMark(x: .value("V_d", x), y: .value(plane.title, GlassMapPoint.normalLine(-x)), series: .value("s", "normal"))
                        .foregroundStyle(Color.secondary.opacity(0.7))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                }
            }
        }
        .chartXScale(domain: xd)
        .chartYScale(domain: yd)
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 8)) { v in
                AxisGridLine()
                AxisTick()
                AxisValueLabel { if let d = v.as(Double.self) { Text(String(format: "%g", -d)) } }
            }
        }
        .chartYAxis { AxisMarks(values: .automatic(desiredCount: 6)) }
        .chartXAxisLabel(alignment: .center) { Text("Abbe number V") + GlassMapPlane.sub("d") }
        .chartYAxisLabel { plane.axisLabel }
        .chartOverlay { proxy in
            GeometryReader { geo in
                if let f = proxy.plotFrame {
                    GlassMapLayer(
                        glasses: plotted, highlighted: highlighted, selection: $selection, plane: plane,
                        frame: geo[f], xDomain: xd, yDomain: yd)
                }
            }
        }
        .environment(\.locale, Locale(identifier: "en_US_POSIX"))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Glass map, \(plane.title) against Abbe number, \(plotted.count) glasses")
        .accessibilityValue(accessibilityValue)
    }

    private var accessibilityValue: String {
        let marked = highlighted.compactMap { h in glasses.first { $0.id == h.id }.map { "\($0.name) \(h.label)" } }
        var t = marked.isEmpty ? "" : "Marked: " + marked.joined(separator: "; ")
        if let s = selection, let g = glasses.first(where: { $0.id == s }) {
            t += (t.isEmpty ? "" : ". ") + "Selected \(g.name)"
        }
        return t
    }
}

/// The points, labels, hover card and click handling of a glass map, over
/// the chart's plot area.
private struct GlassMapLayer: View {
    let glasses: [GlassMapPoint]
    let highlighted: [GlassHighlight]
    @Binding var selection: String?
    let plane: GlassMapPlane
    let frame: CGRect
    let xDomain: ClosedRange<Double>
    let yDomain: ClosedRange<Double>
    @State private var hover: String?
    @Environment(\.colorScheme) private var colorScheme

    private func point(_ g: GlassMapPoint) -> CGPoint {
        let x = (-g.vd - xDomain.lowerBound) / (xDomain.upperBound - xDomain.lowerBound)
        let y = (plane.y(g) - yDomain.lowerBound) / (yDomain.upperBound - yDomain.lowerBound)
        return CGPoint(x: frame.minX + CGFloat(x) * frame.width, y: frame.maxY - CGFloat(y) * frame.height)
    }

    var body: some View {
        let labels = Dictionary(highlighted.map { ($0.id, $0.label) }, uniquingKeysWith: { a, _ in a })
        ZStack(alignment: .topLeading) {
            Canvas { ctx, _ in draw(&ctx, labels: labels) }
                .allowsHitTesting(false)
            Rectangle().fill(.clear).contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let p): hover = nearest(p)
                    case .ended: hover = nil
                    }
                }
                .gesture(SpatialTapGesture().onEnded { e in selection = nearest(e.location) })
            if let h = hover, let g = glasses.first(where: { $0.id == h }) {
                GlassCard(glass: g, compact: true)
                    .padding(.horizontal, Space.s)
                    .padding(.vertical, Space.xs)
                    .background(.background, in: .rect(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.separator))
                    .fixedSize()
                    .offset(tooltipOffset(point(g)))
                    .allowsHitTesting(false)
            }
        }
    }

    private func tooltipOffset(_ p: CGPoint) -> CGSize {
        let w: CGFloat = 170, h: CGFloat = 84
        let x = p.x + 12 + w > frame.maxX ? p.x - 12 - w : p.x + 12
        let y = min(max(p.y - h / 2, frame.minY), frame.maxY - h)
        return CGSize(width: x, height: y)
    }

    private func nearest(_ p: CGPoint) -> String? {
        guard frame.insetBy(dx: -8, dy: -8).contains(p) else { return nil }
        var best: (String, CGFloat)?
        for g in glasses {
            let q = point(g)
            let d = (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y)
            // Design glasses win ties: they are drawn larger.
            let r = d - (g.inDesign ? 9 : 0)
            if best == nil || r < best!.1 { best = (g.id, r) }
        }
        guard let b = best, b.1 <= 100 else { return nil }
        return b.0
    }

    private func draw(_ ctx: inout GraphicsContext, labels: [String: String]) {
        let dark = colorScheme == .dark
        ctx.clip(to: Path(frame.insetBy(dx: -4, dy: -4)))
        // Catalogue glasses first, the marked ones on top.
        for g in glasses where labels[g.id] == nil {
            let p = point(g)
            let c = Palette.color(g.status.rgb(dark: dark))
            let r: CGFloat = 2.6
            let dot = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
            if g.status.hollow {
                ctx.stroke(dot, with: .color(c), lineWidth: 1)
            } else {
                ctx.fill(dot, with: .color(c.opacity(0.85)))
            }
        }
        var placed: [CGRect] = []
        let accent = Color.accentColor
        let marked = glasses.filter { labels[$0.id] != nil }
        for g in marked {
            let p = point(g)
            let r: CGFloat = 5
            let dot = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
            ctx.fill(dot, with: .color(Palette.color(g.status.rgb(dark: dark))))
            ctx.stroke(Path(ellipseIn: CGRect(x: p.x - r - 2, y: p.y - r - 2, width: 2 * r + 4, height: 2 * r + 4)), with: .color(accent), lineWidth: 2)
            placed.append(CGRect(x: p.x - r - 2, y: p.y - r - 2, width: 2 * r + 4, height: 2 * r + 4))
        }
        for g in marked {
            let p = point(g)
            let label = labels[g.id] ?? ""
            let text = ctx.resolve(
                Text(label.isEmpty ? g.name : "\(g.name)  \(label)").font(.caption2.weight(.semibold)).foregroundStyle(Color.primary))
            let size = text.measure(in: CGSize(width: 240, height: 30))
            let candidates = [
                CGPoint(x: p.x + 9, y: p.y - size.height / 2), CGPoint(x: p.x - 9 - size.width, y: p.y - size.height / 2),
                CGPoint(x: p.x - size.width / 2, y: p.y - 9 - size.height), CGPoint(x: p.x - size.width / 2, y: p.y + 9),
            ]
            let boxes = candidates.map { CGRect(origin: $0, size: size).insetBy(dx: -2, dy: 0) }
            let fits = boxes.filter { b in frame.contains(b) && !placed.contains { $0.intersects(b) } }
            let box = fits.first ?? boxes.first { frame.contains($0) } ?? boxes[0]
            placed.append(box)
            ctx.fill(Path(roundedRect: box, cornerRadius: 3), with: .color(dark ? .black.opacity(0.55) : .white.opacity(0.8)))
            ctx.draw(text, at: CGPoint(x: box.midX, y: box.midY))
        }
        if let s = selection, let g = glasses.first(where: { $0.id == s }) {
            let p = point(g)
            let r: CGFloat = g.inDesign ? 10 : 7
            ctx.stroke(
                Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)), with: .color(.primary),
                style: StrokeStyle(lineWidth: 1.5, dash: [3, 2]))
        }
    }
}

/// A glass's data: name, catalogue, status, n_d, V_d, P_g,F (and, full,
/// ΔP_g,F, its surfaces and whether substitution skips it).
struct GlassCard: View {
    let glass: GlassMapPoint
    var compact = false

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 1 : Space.xs) {
            Text(glass.displayName).font(compact ? .caption.weight(.semibold) : .headline)
            Text([glass.catalog ?? (glass.kind == .model ? "Model glass" : ""), glass.status == .unknown ? "" : glass.status.label]
                .filter { !$0.isEmpty }.joined(separator: " · "))
                .font(.caption2).foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: Space.m, verticalSpacing: 1) {
                row(Text("n") + GlassMapPlane.sub("d"), String(format: "%.5f", glass.nd))
                row(Text("V") + GlassMapPlane.sub("d"), String(format: "%.2f", glass.vd))
                row(Text("P") + GlassMapPlane.sub("g,F"), glass.pgf.isFinite ? String(format: "%.4f", glass.pgf) : "—")
                if !compact {
                    row(Text("ΔP") + GlassMapPlane.sub("g,F"), glass.pgf.isFinite ? String(format: "%+.4f", glass.deltaPgF) : "—")
                    if glass.status != .unknown { row(Text("Status"), glass.status.label) }
                    if !glass.surfaces.isEmpty { row(Text("Surfaces"), glass.surfaceLabel) }
                    if glass.excludeSubstitution == true { row(Text("Substitution"), "Excluded") }
                }
            }
            .font(compact ? .caption2 : .callout)
        }
        .accessibilityElement(children: .combine)
    }

    private func row(_ k: Text, _ v: String) -> some View {
        GridRow {
            k.foregroundStyle(.secondary)
            Text(v).numeric().gridColumnAlignment(.trailing)
        }
    }
}

/// The status legend of a glass map, for the statuses present.
struct GlassStatusLegend: View {
    let statuses: [GlassStatus]
    var showsDesign = true
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        FlowLayout(spacing: Space.m) {
            ForEach(statuses, id: \.self) { s in
                HStack(spacing: 4) {
                    let c = Palette.color(s.rgb(dark: colorScheme == .dark))
                    if s.hollow {
                        Circle().strokeBorder(c, lineWidth: 1).frame(width: 7, height: 7)
                    } else {
                        Circle().fill(c).frame(width: 6, height: 6)
                    }
                    Text(statuses == [.unknown] ? "Catalogue glass" : s.label)
                }
            }
            if showsDesign {
                HStack(spacing: 4) {
                    Circle().fill(.secondary).frame(width: 8, height: 8)
                        .overlay(Circle().strokeBorder(Color.accentColor, lineWidth: 2).frame(width: 12, height: 12))
                        .frame(width: 12, height: 12)
                    Text("In the design")
                }
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .combine)
    }
}

extension GlassStatus {
    /// The status colour: a categorical slot, or neutral grey when unknown.
    func rgb(dark: Bool) -> SIMD3<Double> {
        slot < 0 ? (dark ? Palette.rgb(0x9a9a95) : Palette.rgb(0x8a8a85)) : Palette.slotRGB(slot, dark: dark)
    }
}

/// `glass_map`: the map, a plane switch, a catalogue filter, the selected
/// glass and the design's glasses.
struct GlassMapView: View {
    let data: GlassMapData
    var selectSurface: (Int) -> Void = { _ in }
    @State private var plane = GlassMapPlane.ndVd
    @State private var selection: String?
    @State private var catalog: String?

    init?(output: AnalysisOutput, context: AnalysisContext, selection: String? = nil) {
        guard let d = GlassMapData(output: output), !d.glasses.isEmpty else { return nil }
        data = d
        selectSurface = context.selectSurface
        _selection = State(initialValue: selection)
    }

    private var shown: [GlassMapPoint] {
        guard let catalog else { return data.glasses }
        return data.glasses.filter { $0.catalog == catalog || $0.inDesign }
    }

    var body: some View {
        let statuses = GlassStatus.allCases.filter { s in data.glasses.contains { $0.status.group == s && !($0.kind == .model) } }
        VStack(alignment: .leading, spacing: Space.s) {
            FlowLayout(spacing: Space.m, lineSpacing: Space.xs) {
                Picker("Plane", selection: $plane) {
                    ForEach(GlassMapPlane.allCases) { Text($0.segmentTitle).tag($0) }
                }
                .help("Refractive index n_d or partial dispersion P_g,F against Abbe number V_d")
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                if data.catalogs.count > 1 {
                    Picker("Catalogue", selection: $catalog) {
                        Text("All catalogues").tag(String?.none)
                        ForEach(data.catalogs, id: \.self) { Text($0).tag(String?.some($0)) }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .fixedSize()
                }
                GlassStatusLegend(statuses: statuses.isEmpty ? [.unknown] : statuses)
                    .fixedSize()
            }
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: Space.l) {
                    chart.frame(minWidth: 360)
                    VStack(alignment: .leading, spacing: Space.m) {
                        selected
                        designList
                        Spacer(minLength: 0)
                    }
                    .frame(width: 200)
                }
                // Narrow: the chart, then the selected glass only (the chart
                // labels the design's glasses).
                VStack(alignment: .leading, spacing: Space.m) {
                    chart.frame(minHeight: 200)
                    selected.fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(Space.m)
    }

    private var chart: some View {
        GlassMapChart(glasses: shown, highlighted: data.designHighlights, selection: $selection, plane: plane)
    }

    @ViewBuilder private var selected: some View {
        if let s = selection, let g = data.glasses.first(where: { $0.id == s }) {
            VStack(alignment: .leading, spacing: Space.s) {
                GlassCard(glass: g)
                if !g.surfaces.isEmpty {
                    HStack(spacing: Space.xs) {
                        Text("Select").font(.caption).foregroundStyle(.secondary)
                        ForEach(g.surfaces, id: \.self) { s in
                            Button("S\(s)") { selectSurface(s) }
                                .help("Select surface \(s) in the lens table")
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
            }
            .padding(Space.m)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.fill.quinary, in: .rect(cornerRadius: 8))
        } else {
            Text("Click a glass to inspect it.").font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var designList: some View {
        let design = data.glasses.filter(\.inDesign)
        if !design.isEmpty {
            VStack(alignment: .leading, spacing: Space.xs) {
                Text("In the design").font(.caption.weight(.semibold))
                ForEach(design) { g in
                    Button {
                        selection = g.id
                    } label: {
                        HStack {
                            Text(g.displayName).lineLimit(1)
                            Spacer(minLength: Space.s)
                            Text(g.surfaceLabel).foregroundStyle(.secondary).lineLimit(1)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .accessibilityLabel("\(g.name), surfaces \(g.surfaceLabel)")
                }
            }
        }
    }
}
