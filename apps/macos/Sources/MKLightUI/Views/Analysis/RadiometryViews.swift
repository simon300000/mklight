import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

// Ghosts, stray light, polarisation and transmission.

// MARK: - Ghost paths

struct GhostPathsView: View {
    struct Ghost: Identifiable {
        var id: Int
        var label: String
        var path: [Int]
        var irradiance: Double
        var focus: Double
        var spot: Double
        var reflectance: Double
        var nearFocus: Bool
    }
    let ghosts: [Ghost]
    let truncated: Bool

    static func sci(_ v: Double) -> String { v.isFinite ? String(format: "%.2e", v) : "—" }
    let context: AnalysisContext

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let list = j["ghosts"]?.array else { return nil }
        ghosts = list.enumerated().map { k, g in
            Ghost(
                id: k, label: g["label"]?.string ?? "", path: g["path"]?.intArray ?? [],
                irradiance: g["relative_irradiance"]?.nf ?? .nan, focus: g["focus_from_sensor"]?.nf ?? .nan,
                spot: g["spot_diameter"]?.nf ?? .nan, reflectance: g["reflectance_product"]?.nf ?? .nan,
                nearFocus: g["near_focus"]?.bool ?? false)
        }
        truncated = j["truncated"]?.bool ?? false
        self.context = context
    }

    var body: some View {
        if ghosts.isEmpty {
            ContentUnavailableView("No Ghost Paths", systemImage: "checkmark.circle", description: Text("The enumeration found no ghost reaching the sensor."))
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ScrollView {
                    Grid(alignment: .trailing, horizontalSpacing: Space.l, verticalSpacing: 3) {
                        GridRow {
                            Text("Ghost").gridColumnAlignment(.leading)
                            Text("Rel. irradiance")
                            Text("Focus (mm)")
                            Text("Blur ⌀ (mm)")
                            Text("ΠR")
                        }
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        Divider().gridCellUnsizedAxes(.horizontal)
                        ForEach(ghosts) { g in
                            GridRow {
                                HStack(spacing: Space.xs) {
                                    Text(g.label.isEmpty ? g.path.map(String.init).joined(separator: "·") : g.label)
                                    if g.nearFocus {
                                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                                            .help("Focused near the sensor").accessibilityLabel("near focus")
                                    }
                                }
                                Text(Self.sci(g.irradiance))
                                Text(String(format: "%.2f", g.focus))
                                Text(String(format: "%.2f", g.spot))
                                Text(Self.sci(g.reflectance))
                            }
                            .font(.callout.monospacedDigit())
                            .lineLimit(1)
                            .contentShape(Rectangle())
                            .onTapGesture { if let s = g.path.first { context.selectSurface(s) } }
                        }
                    }
                    .padding(Space.m)
                    .textSelection(.enabled)
                }
                if truncated {
                    Label("Higher-order paths were dropped by the path budget.", systemImage: "info.circle")
                        .font(.caption).foregroundStyle(.secondary).padding(Space.m)
                }
            }
        }
    }
}

// MARK: - Maps (ghost irradiance, stray light)

/// A map with a selector, log scale and colour bar, in mm on the sensor.
struct SensorMapView: View {
    let maps: [(title: String, grid: Grid2D)]
    let unit: String
    var rows: [(String, String)] = []
    @State private var selected = 0
    @State private var log = true

    var body: some View {
        let m = maps[min(selected, maps.count - 1)]
        VStack(alignment: .leading, spacing: Space.s) {
            HStack {
                ItemPicker(titles: maps.map(\.title), selection: $selected)
                Spacer()
                Toggle("Log scale", isOn: $log).toggleStyle(.checkbox).controlSize(.small)
            }
            MapLayout(aspect: CGFloat((m.grid.xRange.upperBound - m.grid.xRange.lowerBound) / max(m.grid.yRange.upperBound - m.grid.yRange.lowerBound, 1e-9))) {
                HeatMapImage(grid: m.grid, scale: .sequential(log: log))
                    .overlay(Rectangle().strokeBorder(.separator))
                    .accessibilityLabel("\(m.title) map")
            } bar: {
                ColorBar(scale: .sequential(log: log), range: m.grid.range, unit: unit)
            } side: {
                if !rows.isEmpty { KeyValueGrid(rows: rows) }
            }
            Text(String(format: "Sensor %.2f × %.2f mm", m.grid.xRange.upperBound - m.grid.xRange.lowerBound, m.grid.yRange.upperBound - m.grid.yRange.lowerBound))
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(Space.m)
    }
}

struct GhostIrradianceView: View {
    let map: SensorMapView

    init?(output: AnalysisOutput) {
        let j = output.json
        let half = j[path: "map", "half_size"]?.nfArray ?? [1, 1]
        let xr = -half[0]...half[0], yr = -half[1]...half[1]
        var maps: [(String, Grid2D)] = []
        if let g = Grid2D(output.buffer(named: "total_ghost_map"), xRange: xr, yRange: yr) { maps.append(("All ghosts", g)) }
        if let g = Grid2D(output.buffer(named: "primary_map"), xRange: xr, yRange: yr) { maps.append(("Primary image", g)) }
        guard !maps.isEmpty else { return nil }
        var rows: [(String, String)] = [
            ("Ghost / primary flux", Fmt.short(j["total_flux_ratio"]?.nf ?? .nan)),
            ("Ghost / primary peak", Fmt.short(j["total_peak_ratio"]?.nf ?? .nan)),
        ]
        for g in (j["ghosts"]?.array ?? []).prefix(6) {
            rows.append((g["label"]?.string ?? "", Fmt.short(g["peak_ratio"]?.nf ?? .nan)))
        }
        map = SensorMapView(maps: maps.map { (title: $0.0, grid: $0.1) }, unit: "mm⁻²", rows: rows)
    }

    var body: some View { map }
}

struct StrayLightView: View {
    let map: SensorMapView

    init?(output: AnalysisOutput) {
        let j = output.json
        let d = j["detector"]
        let hx = d?["half_x"]?.nf ?? 1, hy = d?["half_y"]?.nf ?? 1
        let xr = -hx...hx, yr = -hy...hy
        var maps: [(String, Grid2D)] = []
        if let g = Grid2D(output.buffer(named: "total_irradiance"), xRange: xr, yRange: yr) { maps.append(("Stray total", g)) }
        let classes = j["map_classes"]?.array?.compactMap(\.string) ?? []
        for (k, c) in classes.enumerated() {
            if let g = Grid2D(output.buffer(named: "class_irradiance"), slice: k, xRange: xr, yRange: yr) { maps.append((c.capitalized, g)) }
        }
        guard !maps.isEmpty else { return nil }
        var rows: [(String, String)] = [
            ("Stray flux", Fmt.short(j["stray_flux"]?.nf ?? .nan) + " W"),
            ("± standard error", Fmt.short(j["stray_std_error"]?.nf ?? .nan) + " W"),
        ]
        if let f = j["stray_fraction"]?.nf, f.isFinite { rows.append(("Stray fraction", Fmt.short(f))) }
        for t in j["totals"]?.array ?? [] {
            rows.append(((t["class"]?.string ?? "").capitalized, Fmt.short(t["flux"]?.nf ?? .nan) + " W"))
        }
        map = SensorMapView(maps: maps.map { (title: $0.0, grid: $0.1) }, unit: "W/mm²", rows: rows)
    }

    var body: some View { map }
}

// MARK: - Polarisation pupil map

struct PolarizationView: View {
    enum Quantity: String, CaseIterable { case intensity = "Intensity", diattenuation = "Diattenuation", retardance = "Retardance" }
    let pupil: [(Double, Double)]
    let reached: [Bool]
    let values: [Quantity: [Double]]
    let stats: [(String, String)]
    @State private var quantity = Quantity.intensity
    @Environment(\.colorScheme) private var scheme

    init?(output: AnalysisOutput) {
        guard let p = output.buffer(named: "pupil"), let i = output.buffer(named: "intensity") else { return nil }
        pupil = (0..<p.rows).map { (p[$0 * 2], p[$0 * 2 + 1]) }
        reached = output.buffer(named: "reached").map { b in (0..<b.count).map { b[$0] != 0 } } ?? Array(repeating: true, count: p.rows)
        values = [
            .intensity: i.doubles, .diattenuation: output.buffer(named: "diattenuation")?.doubles ?? [],
            .retardance: output.buffer(named: "retardance")?.doubles ?? [],
        ]
        let j = output.json
        stats = [
            ("Mean intensity", Fmt.short(j["mean_intensity"]?.nf ?? .nan)),
            ("Max diattenuation", Fmt.short(j["max_diattenuation"]?.nf ?? .nan)),
            ("Max retardance", Fmt.short(j["max_retardance"]?.nf ?? .nan) + " rad"),
            ("RMS retardance", Fmt.short(j["rms_retardance"]?.nf ?? .nan) + " rad"),
        ]
    }

    var body: some View {
        let v = values[quantity] ?? []
        let finite = v.enumerated().filter { reached.indices.contains($0.offset) && reached[$0.offset] && $0.element.isFinite }.map(\.element)
        let range = (finite.min() ?? 0)...max(finite.max() ?? 1, (finite.min() ?? 0) + 1e-12)
        VStack(alignment: .leading, spacing: Space.s) {
            Picker("Quantity", selection: $quantity) {
                ForEach(Quantity.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            MapLayout {
                    Canvas { ctx, size in
                        let r = min(size.width, size.height) / 2
                        let c = CGPoint(x: size.width / 2, y: size.height / 2)
                        let cell = max(2 * r / CGFloat(max(sqrt(Double(pupil.count) * 4 / .pi), 1)), 2)
                        ctx.stroke(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(.secondary.opacity(0.4)))
                        for (k, p) in pupil.enumerated() where v.indices.contains(k) && reached[k] && v[k].isFinite {
                            let t = HeatMapImage.normalise(v[k], scale: .sequential(log: false), range: range)
                            let rgb = Palette.sequential(t, dark: scheme == .dark)
                            let q = CGPoint(x: c.x + CGFloat(p.0) * r, y: c.y - CGFloat(p.1) * r)
                            ctx.fill(Path(CGRect(x: q.x - cell / 2, y: q.y - cell / 2, width: cell, height: cell)), with: .color(Color(.sRGB, red: rgb.x, green: rgb.y, blue: rgb.z)))
                        }
                    }
                    .accessibilityLabel("\(quantity.rawValue) over the pupil")
            } bar: {
                ColorBar(scale: .sequential(log: false), range: range, unit: quantity == .retardance ? "rad" : "")
            } side: {
                KeyValueGrid(rows: stats)
            }
        }
        .padding(Space.m)
    }
}

// MARK: - System transmission

struct TransmissionView: View {
    struct Point { var field: Int; var um: Double; var t: Double }
    let points: [Point]
    let polychromatic: [Double]
    let fieldLabels: [String]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let list = j["points"]?.array, !list.isEmpty else { return nil }
        points = list.map { Point(field: $0["field"]?.int ?? 0, um: $0["wavelength_um"]?.nf ?? .nan, t: $0["transmission"]?.nf ?? .nan) }
        polychromatic = j["polychromatic"]?.nfArray ?? []
        fieldLabels = (j["fields"]?.array ?? []).map { f in
            String(format: "H (%.2f, %.2f)", f["hx"]?.nf ?? 0, f["hy"]?.nf ?? 0)
        }
    }

    var body: some View {
        let fields = Array(Set(points.map(\.field))).sorted()
        VStack(alignment: .leading, spacing: Space.s) {
            FlowLayout(spacing: Space.m) {
                ForEach(fields, id: \.self) { f in
                    LegendItem(
                        color: Palette.fieldColor(f),
                        label: (fieldLabels.indices.contains(f) ? fieldLabels[f] : "Field \(f + 1)")
                            + (polychromatic.indices.contains(f) ? " · " + Fmt.percent(polychromatic[f]) : ""))
                }
            }
            if Set(points.map(\.um)).count > 1 {
                LineChart(
                    series: fields.map { f in
                        Series(id: "\(f)", label: "", color: Palette.fieldColor(f), points: points.filter { $0.field == f }.map { ($0.um, $0.t) })
                    }, xLabel: "Wavelength (µm)", yLabel: "Transmission", yDomain: 0...1.02, zeroLine: false)
            } else {
                Chart(points, id: \.field) { p in
                    BarMark(x: .value("Field", "\(p.field + 1)"), y: .value("Transmission", p.t)).foregroundStyle(Palette.fieldColor(p.field)).cornerRadius(3)
                }
                .chartYScale(domain: 0...1)
            }
        }
        .padding(Space.m)
    }
}
