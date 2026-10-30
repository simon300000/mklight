import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

// Dedicated views of the geometric analyses (docs/RESULTS.md §8). Each
// reads the engine JSON in its initialiser and returns nil when the shape
// is not what it expects, so the pane falls back to the generic view.

// MARK: - Spot diagram

struct SpotDiagramView: View {
    struct FieldSpot: Identifiable {
        var id: Int
        var label: String
        var rms: Double
        var geo: Double
        var points: [(x: Double, y: Double, w: Int)]
        var vignetted: Double?
    }
    let fields: [FieldSpot]
    let airy: Double
    let wavelengths: [Int]
    let wavelengthsUm: [Double]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let list = j["fields"]?.array, !list.isEmpty else { return nil }
        let pts = output.buffer(named: "points"), pw = output.buffer(named: "point_wavelength")
        wavelengthsUm = j["wavelengths_um"]?.nfArray ?? context.wavelengthsUm
        wavelengths = j["wavelengths"]?.intArray ?? Array(wavelengthsUm.indices)
        airy = j["airy_radius"]?.nf ?? .nan
        fields = list.enumerated().map { k, f in
            var points: [(Double, Double, Int)] = []
            if let pts, let range = f["point_range"]?.intArray, range.count == 2 {
                let lo = max(range[0], 0), hi = min(range[0] + range[1], pts.rows)
                if lo < hi {
                    points.reserveCapacity(hi - lo)
                    for i in lo..<hi {
                        let w = pw.map { Int($0[i]) } ?? 0
                        points.append((pts[i * 2], pts[i * 2 + 1], w))
                    }
                }
            }
            let index = f["field"]?.int ?? k
            return FieldSpot(
                id: index, label: "Field \(index + 1) · " + context.fieldLabel(f["field_value"]?.nfArray),
                rms: f["rms_radius"]?.nf ?? .nan, geo: f["geo_radius"]?.nf ?? .nan,
                points: points.map { (x: $0.0, y: $0.1, w: $0.2) },
                vignetted: f[path: "polychromatic", "vignetted_fraction"]?.nf)
        }
    }

    /// One scale for every field: the largest GEO radius (or the Airy disk).
    private var halfWidth: Double {
        let geo = fields.map(\.geo).filter(\.isFinite).max() ?? 0
        let a = airy.isFinite ? airy : 0
        return max(geo, a, 1e-6) * 1.15
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack {
                WavelengthLegend(wavelengths: wavelengths, um: wavelengthsUm)
                Spacer()
                if airy.isFinite {
                    LegendItem(color: .secondary, label: "Airy disk \(Fmt.length(airy))", dashed: true)
                }
            }
            GeometryReader { geo in
                let cols = max(1, min(fields.count, Int(geo.size.width / 170)))
                let rows = Int(ceil(Double(fields.count) / Double(cols)))
                let cell = min(geo.size.width / CGFloat(cols), geo.size.height / CGFloat(max(rows, 1))) - Space.m
                ScrollView {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: Space.m), count: cols), spacing: Space.m) {
                        ForEach(fields) { f in
                            SpotCell(field: f, halfWidth: halfWidth, airy: airy, wavelengthsUm: wavelengthsUm, size: max(cell - 34, 80))
                        }
                    }
                }
            }
            ScaleNote(halfWidth: halfWidth)
        }
        .padding(Space.m)
    }
}

private struct ScaleNote: View {
    let halfWidth: Double
    var body: some View {
        Text("Box half-width \(Fmt.length(halfWidth)), common to all fields")
            .font(.caption2).foregroundStyle(.secondary)
    }
}

private struct SpotCell: View {
    let field: SpotDiagramView.FieldSpot
    let halfWidth: Double
    let airy: Double
    let wavelengthsUm: [Double]
    let size: CGFloat
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(spacing: 2) {
            Text(field.label).font(.caption.weight(.semibold)).lineLimit(1)
            Canvas { ctx, sz in
                let s = min(sz.width, sz.height) / 2 / CGFloat(halfWidth)
                let c = CGPoint(x: sz.width / 2, y: sz.height / 2)
                ctx.stroke(Path(CGRect(origin: .zero, size: sz)), with: .color(.secondary.opacity(0.25)), lineWidth: 0.5)
                var cross = Path()
                cross.move(to: CGPoint(x: c.x, y: 0)); cross.addLine(to: CGPoint(x: c.x, y: sz.height))
                cross.move(to: CGPoint(x: 0, y: c.y)); cross.addLine(to: CGPoint(x: sz.width, y: c.y))
                ctx.stroke(cross, with: .color(.secondary.opacity(0.2)), lineWidth: 0.5)
                if airy.isFinite {
                    let r = CGFloat(airy) * s
                    ctx.stroke(
                        Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                        with: .color(.secondary), style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
                }
                let dot: CGFloat = field.points.count > 2000 ? 1.2 : 1.8
                let dark = scheme == .dark
                for p in field.points where p.x.isFinite && p.y.isFinite {
                    let um = wavelengthsUm.indices.contains(p.w) ? wavelengthsUm[p.w] : 0.55
                    let rgb = Palette.wavelengthRGB(um, dark: dark)
                    let q = CGPoint(x: c.x + CGFloat(p.x) * s, y: c.y - CGFloat(p.y) * s)
                    ctx.fill(
                        Path(ellipseIn: CGRect(x: q.x - dot, y: q.y - dot, width: 2 * dot, height: 2 * dot)),
                        with: .color(Color(.sRGB, red: rgb.x, green: rgb.y, blue: rgb.z, opacity: 0.85)))
                }
            }
            .frame(width: size, height: size)
            .accessibilityLabel("\(field.label) spot, RMS \(Fmt.length(field.rms)), GEO \(Fmt.length(field.geo))")
            HStack(spacing: Space.m) {
                Text("RMS \(Fmt.length(field.rms))")
                Text("GEO \(Fmt.length(field.geo))")
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Ray and OPD fans

struct FanView: View {
    struct Curve { var wavelength: Int; var points: [(Double, Double)] }
    struct FieldFan: Identifiable { var id: Int; var label: String; var tangential: [Curve]; var sagittal: [Curve] }
    let fields: [FieldFan]
    let opd: Bool
    let wavelengthsUm: [Double]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let list = j["fields"]?.array, !list.isEmpty else { return nil }
        opd = j["quantity"]?.string == "opd"
        wavelengthsUm = j["wavelengths_um"]?.nfArray ?? context.wavelengthsUm
        let scale = opd ? 1.0 : 1000.0  // transverse in µm
        func curves(_ v: JSONValue?) -> [Curve] {
            (v?.array ?? []).compactMap { c in
                guard let p = c["pupil"]?.nfArray, let y = c["values"]?.nfArray, p.count == y.count else { return nil }
                return Curve(wavelength: c["wavelength"]?.int ?? 0, points: zip(p, y).map { ($0, $1 * scale) })
            }
        }
        fields = list.enumerated().map { k, f in
            let i = f["field"]?.int ?? k
            return FieldFan(
                id: i, label: "Field \(i + 1) · " + context.fieldLabel(f["field_value"]?.nfArray),
                tangential: curves(f["tangential"]), sagittal: curves(f["sagittal"]))
        }
    }

    private var unit: String { opd ? "waves" : "µm" }

    private var yDomain: ClosedRange<Double> {
        let v = fields.flatMap { ($0.tangential + $0.sagittal).flatMap { $0.points.map(\.1) } }.filter(\.isFinite)
        let m = max(v.map(abs).max() ?? 1, 1e-9) * 1.08
        return -m...m
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            WavelengthLegend(wavelengths: Array(Set(fields.flatMap { $0.tangential.map(\.wavelength) })).sorted(), um: wavelengthsUm)
            ScrollView {
                VStack(spacing: Space.m) {
                    ForEach(fields) { f in
                        HStack(spacing: Space.l) {
                            PlotCell(title: f.label, subtitle: "Tangential") { chart(f.tangential, x: "Py", y: opd ? "W (waves)" : "εy (µm)") }
                            PlotCell(title: " ", subtitle: "Sagittal") { chart(f.sagittal, x: "Px", y: opd ? "W (waves)" : "εx (µm)") }
                        }
                        .frame(height: 150)
                    }
                }
            }
        }
        .padding(Space.m)
    }

    private func chart(_ curves: [Curve], x: String, y: String) -> some View {
        LineChart(
            series: curves.map { c in
                Series(id: "\(c.wavelength)", label: "", color: color(c.wavelength), points: c.points)
            }, xLabel: x, yLabel: y, xDomain: -1...1, yDomain: yDomain)
    }

    private func color(_ w: Int) -> Color { Palette.wavelengthColor(wavelengthsUm.indices.contains(w) ? wavelengthsUm[w] : 0.55) }
}

// MARK: - Field curvature and distortion

struct FieldCurvatureView: View {
    struct Curve { var wavelength: Int; var t: [Double]; var s: [Double]; var d: [Double] }
    let field: [Double]
    let fieldUnit: String
    let curves: [Curve]
    let wavelengthsUm: [Double]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let f = j["field_values"]?.nfArray, let list = j["curves"]?.array, !list.isEmpty else { return nil }
        field = f
        fieldUnit = j["field_unit"]?.string ?? context.fieldUnit
        wavelengthsUm = j["wavelengths_um"]?.nfArray ?? context.wavelengthsUm
        curves = list.map {
            Curve(
                wavelength: $0["wavelength"]?.int ?? 0, t: $0["tangential"]?.nfArray ?? [], s: $0["sagittal"]?.nfArray ?? [],
                d: $0["distortion_percent"]?.nfArray ?? [])
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.l) {
                WavelengthLegend(wavelengths: curves.map(\.wavelength), um: wavelengthsUm)
                Spacer()
                LegendItem(color: .secondary, label: "Tangential")
                LegendItem(color: .secondary, label: "Sagittal", dashed: true)
            }
            HStack(spacing: Space.l) {
                PlotCell(title: "Field Curvature") {
                    LineChart(
                        series: curves.flatMap { c in
                            [
                                Series(id: "t\(c.wavelength)", label: "T", color: color(c.wavelength), points: zip(c.t, field).map { ($0, $1) }),
                                Series(id: "s\(c.wavelength)", label: "S", color: color(c.wavelength), points: zip(c.s, field).map { ($0, $1) }, dashed: true),
                            ]
                        }, xLabel: "Focus shift (mm)", yLabel: "Field (\(fieldUnit))", zeroLine: false)
                }
                PlotCell(title: "Distortion") {
                    LineChart(
                        series: curves.map { c in
                            Series(id: "d\(c.wavelength)", label: "", color: color(c.wavelength), points: zip(c.d, field).map { ($0, $1) })
                        }, xLabel: "Distortion (%)", yLabel: "Field (\(fieldUnit))", zeroLine: false)
                }
            }
        }
        .padding(Space.m)
    }

    private func color(_ w: Int) -> Color { Palette.wavelengthColor(wavelengthsUm.indices.contains(w) ? wavelengthsUm[w] : 0.55) }
}

// MARK: - Grid distortion

struct GridDistortionView: View {
    let n: Int
    let paraxial: [(Double, Double)]
    let real: [(Double, Double)]
    let maxDistortion: Double

    init?(output: AnalysisOutput) {
        let j = output.json
        guard let n = j["grid_size"]?.int, n >= 2, let p = output.buffer(named: "paraxial"), let r = output.buffer(named: "real"),
            p.count == n * n * 2, r.count == n * n * 2
        else { return nil }
        self.n = n
        paraxial = (0..<(n * n)).map { (p[$0 * 2], p[$0 * 2 + 1]) }
        real = (0..<(n * n)).map { (r[$0 * 2], r[$0 * 2 + 1]) }
        maxDistortion = j["max_abs_distortion_percent"]?.nf ?? .nan
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.l) {
                LegendItem(color: .secondary.opacity(0.5), label: "Paraxial grid", dashed: true)
                LegendItem(color: .primary, label: "Real chief rays")
                Spacer()
                Text("Max \(String(format: "%.3f", maxDistortion)) %").font(.caption).foregroundStyle(.secondary).numeric()
            }
            Canvas { ctx, size in
                let all = (paraxial + real).filter { $0.0.isFinite && $0.1.isFinite }
                let m = max(all.map { max(abs($0.0), abs($0.1)) }.max() ?? 1, 1e-9) * 1.06
                let s = min(size.width, size.height) / 2 / CGFloat(m)
                let c = CGPoint(x: size.width / 2, y: size.height / 2)
                func pt(_ p: (Double, Double)) -> CGPoint { CGPoint(x: c.x + CGFloat(p.0) * s, y: c.y - CGFloat(p.1) * s) }
                func grid(_ pts: [(Double, Double)]) -> Path {
                    var path = Path()
                    for line in 0..<n {
                        for dir in 0..<2 {
                            var started = false
                            for k in 0..<n {
                                let p = dir == 0 ? pts[line * n + k] : pts[k * n + line]
                                guard p.0.isFinite && p.1.isFinite else { started = false; continue }
                                if started { path.addLine(to: pt(p)) } else { path.move(to: pt(p)); started = true }
                            }
                        }
                    }
                    return path
                }
                ctx.stroke(grid(paraxial), with: .color(.secondary.opacity(0.5)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                ctx.stroke(grid(real), with: .color(.primary), lineWidth: 1.2)
            }
            .aspectRatio(1, contentMode: .fit)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityLabel("Grid distortion, maximum \(String(format: "%.3f", maxDistortion)) percent")
        }
        .padding(Space.m)
    }
}

// MARK: - Seidel

struct SeidelView: View {
    struct Bar: Identifiable { var id: String { "\(surface)-\(term)" }; var surface: String; var term: String; var value: Double }
    static let terms = [("s1", "S_I spherical"), ("s2", "S_II coma"), ("s3", "S_III astigmatism"), ("s4", "S_IV Petzval"), ("s5", "S_V distortion")]
    static let colour = [("cl", "C_L axial"), ("ct", "C_T lateral")]
    let surfaces: [[String: Double]]
    let surfaceIndex: [Int]
    let sums: [String: Double]
    let wavefront: [(String, Double)]
    let petzvalRadius: Double
    @State private var chromatic = false

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let list = j["surfaces"]?.array, let s = j["sums"]?.object else { return nil }
        let keys = (Self.terms + Self.colour).map(\.0)
        surfaces = list.map { o in Dictionary(uniqueKeysWithValues: keys.map { ($0, o[$0]?.nf ?? .nan) }) }
        surfaceIndex = list.enumerated().map { $0.element["surface"]?.int ?? $0.offset }
        sums = Dictionary(uniqueKeysWithValues: keys.map { ($0, s[$0]?.nf ?? .nan) })
        wavefront = (j["wavefront_waves"]?.object ?? [:]).sorted { $0.key < $1.key }.compactMap { k, v in v.nf.map { (k.uppercased(), $0) } }
        petzvalRadius = j["petzval_radius"]?.nf ?? .nan
    }

    private var bars: [Bar] {
        let terms = chromatic ? Self.colour : Self.terms
        var out: [Bar] = []
        for (k, s) in surfaces.enumerated() {
            for (key, title) in terms { out.append(Bar(surface: "\(surfaceIndex[k])", term: title, value: s[key] ?? .nan)) }
        }
        for (key, title) in terms { out.append(Bar(surface: "Σ", term: title, value: sums[key] ?? .nan)) }
        return out.filter { $0.value.isFinite }
    }

    var body: some View {
        let terms = (chromatic ? Self.colour : Self.terms).map(\.1)
        VStack(alignment: .leading, spacing: Space.s) {
            HStack {
                Picker("Coefficients", selection: $chromatic) {
                    Text("Seidel").tag(false)
                    Text("Chromatic").tag(true)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                Spacer()
                if petzvalRadius.isFinite {
                    Text("Petzval radius \(Fmt.short(petzvalRadius)) mm").font(.caption).foregroundStyle(.secondary).numeric()
                }
            }
            Chart(bars) { b in
                BarMark(x: .value("Surface", b.surface), y: .value("Coefficient (mm)", b.value))
                    .position(by: .value("Term", b.term))
                    .foregroundStyle(by: .value("Term", b.term))
                    .cornerRadius(2)
            }
            .chartForegroundStyleScale(domain: terms, range: terms.indices.map { Palette.fieldColor($0 + (chromatic ? 5 : 0)) })
            .chartLegend(position: .top, alignment: .leading)
            .chartYAxisLabel("mm")
            .chartXAxisLabel("Surface", alignment: .center)
            .accessibilityLabel("Seidel coefficients per surface and their sum")
            if !wavefront.isEmpty {
                ScrollView(.horizontal) {
                    HStack(spacing: Space.l) {
                        ForEach(wavefront, id: \.0) { k, v in
                            VStack(spacing: 1) {
                                Text(k).font(.caption2).foregroundStyle(.secondary)
                                Text(String(format: "%.4f", v)).font(.caption.monospacedDigit())
                            }
                        }
                        Text("waves").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(Space.m)
    }
}

// MARK: - Longitudinal aberration, chromatic focal shift, lateral colour

struct LongitudinalView: View {
    let pupil: [Double]
    let curves: [(Int, [Double])]
    let um: [Double]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let p = j["pupil"]?.nfArray, let list = j["curves"]?.array, !list.isEmpty else { return nil }
        pupil = p
        curves = list.map { ($0["wavelength"]?.int ?? 0, $0["aberration"]?.nfArray ?? []) }
        um = context.wavelengthsUm
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            WavelengthLegend(wavelengths: curves.map(\.0), um: um)
            LineChart(
                series: curves.map { w, a in
                    Series(id: "\(w)", label: "", color: Palette.wavelengthColor(um.indices.contains(w) ? um[w] : 0.55), points: zip(a, pupil).map { ($0, $1) })
                }, xLabel: "Longitudinal aberration (mm)", yLabel: "Pupil zone", yDomain: 0...1, zeroLine: false)
        }
        .padding(Space.m)
    }
}

struct ChromaticFocalShiftView: View {
    let um: [Double]
    let shift: [Double]
    let primary: Double

    init?(output: AnalysisOutput) {
        let j = output.json
        guard let w = j["wavelengths_um"]?.nfArray, let s = j["focal_shift"]?.nfArray, w.count == s.count else { return nil }
        um = w
        shift = s
        primary = j["primary_wavelength_um"]?.nf ?? .nan
    }

    var body: some View {
        LineChart(
            series: [Series(id: "shift", label: "", color: Palette.fieldColor(0), points: zip(shift.map { $0 * 1000 }, um).map { ($0, $1) })],
            xLabel: "Focal shift (µm)", yLabel: "Wavelength (µm)", zeroLine: false,
            markers: [])
            .padding(Space.m)
    }
}

struct LateralColourView: View {
    let field: [Double]
    let unit: String
    let curves: [(Int, [Double])]
    let airy: Double
    let um: [Double]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let f = j["field_values"]?.nfArray, let list = j["curves"]?.array else { return nil }
        field = f
        unit = j["field_unit"]?.string ?? context.fieldUnit
        curves = list.map { ($0["wavelength"]?.int ?? 0, $0["offset"]?.nfArray ?? []) }
        airy = j["airy_radius"]?.nf ?? .nan
        um = context.wavelengthsUm
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack {
                WavelengthLegend(wavelengths: curves.map(\.0), um: um)
                Spacer()
                if airy.isFinite { LegendItem(color: .secondary, label: "± Airy radius", dashed: true) }
            }
            Chart {
                if airy.isFinite {
                    RuleMark(x: .value("Airy", airy * 1000)).lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3])).foregroundStyle(.secondary)
                    RuleMark(x: .value("Airy", -airy * 1000)).lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3])).foregroundStyle(.secondary)
                }
                ForEach(curves, id: \.0) { w, off in
                    ForEach(Array(zip(off, field).enumerated()), id: \.offset) { _, p in
                        if p.0.isFinite && p.1.isFinite {
                            LineMark(x: .value("Offset (µm)", p.0 * 1000), y: .value("Field", p.1), series: .value("λ", w))
                                .foregroundStyle(Palette.wavelengthColor(um.indices.contains(w) ? um[w] : 0.55))
                        }
                    }
                }
            }
            .chartXAxisLabel("Lateral color (µm)", alignment: .center)
            .chartYAxisLabel("Field (\(unit))")
            .accessibilityLabel("Lateral color versus field")
        }
        .padding(Space.m)
    }
}

// MARK: - Relative illumination

struct RelativeIlluminationView: View {
    let field: [Double]
    let unit: String
    let ri: [Double]
    let vignetting: [Double]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let f = j["field_values"]?.nfArray, let r = j["relative_illumination"]?.nfArray else { return nil }
        field = f
        ri = r
        vignetting = j["vignetting_fraction"]?.nfArray ?? []
        unit = j["field_unit"]?.string ?? context.fieldUnit
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.l) {
                LegendItem(color: Palette.fieldColor(0), label: "Relative illumination")
                if !vignetting.isEmpty { LegendItem(color: Palette.fieldColor(1), label: "Vignetted fraction", dashed: true) }
            }
            LineChart(
                series: [
                    Series(id: "ri", label: "", color: Palette.fieldColor(0), points: zip(field, ri).map { ($0, $1) }),
                    Series(id: "v", label: "", color: Palette.fieldColor(1), points: zip(field, vignetting).map { ($0, $1) }, dashed: true),
                ], xLabel: "Field (\(unit))", yLabel: "Fraction", yDomain: 0...1.05, zeroLine: false)
        }
        .padding(Space.m)
    }
}

// MARK: - Footprint

struct FootprintView: View {
    let points: [(Double, Double, Int)]
    let clipped: [(Double, Double)]
    let semiDiameter: Double
    let surface: Int
    let um: [Double]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let sets = j["sets"]?.array, let p = output.buffer(named: "points") else { return nil }
        var pts: [(Double, Double, Int)] = []
        for s in sets {
            guard let r = s["points"]?.intArray, r.count == 2 else { continue }
            let w = s["wavelength"]?.int ?? 0
            for i in r[0]..<min(r[0] + r[1], p.rows) { pts.append((p[i * 2], p[i * 2 + 1], w)) }
        }
        points = pts
        if let c = output.buffer(named: "clipped") {
            clipped = (0..<c.rows).map { (c[$0 * 2], c[$0 * 2 + 1]) }
        } else {
            clipped = []
        }
        semiDiameter = j["semi_diameter"]?.nf ?? .nan
        surface = j["surface"]?.int ?? 0
        um = context.wavelengthsUm
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.l) {
                Text("Surface \(surface)").font(.caption.weight(.semibold))
                if semiDiameter.isFinite { LegendItem(color: .secondary, label: "Clear aperture \(Fmt.short(semiDiameter)) mm", dashed: true) }
                if !clipped.isEmpty { LegendItem(color: Color(.sRGB, red: 0.89, green: 0.29, blue: 0.28), label: "\(clipped.count) clipped") }
            }
            Canvas { ctx, size in
                let all = points.map { max(abs($0.0), abs($0.1)) } + clipped.map { max(abs($0.0), abs($0.1)) }
                let m = max(all.filter(\.isFinite).max() ?? 1, semiDiameter.isFinite ? semiDiameter : 0, 1e-9) * 1.05
                let s = min(size.width, size.height) / 2 / CGFloat(m)
                let c = CGPoint(x: size.width / 2, y: size.height / 2)
                if semiDiameter.isFinite {
                    let r = CGFloat(semiDiameter) * s
                    ctx.stroke(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(.secondary), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                }
                for p in points where p.0.isFinite {
                    let q = CGPoint(x: c.x + CGFloat(p.0) * s, y: c.y - CGFloat(p.1) * s)
                    let color = Palette.wavelengthColor(um.indices.contains(p.2) ? um[p.2] : 0.55)
                    ctx.fill(Path(ellipseIn: CGRect(x: q.x - 1.3, y: q.y - 1.3, width: 2.6, height: 2.6)), with: .color(color.opacity(0.8)))
                }
                for p in clipped where p.0.isFinite {
                    let q = CGPoint(x: c.x + CGFloat(p.0) * s, y: c.y - CGFloat(p.1) * s)
                    ctx.fill(Path(ellipseIn: CGRect(x: q.x - 1.3, y: q.y - 1.3, width: 2.6, height: 2.6)), with: .color(Color(.sRGB, red: 0.89, green: 0.29, blue: 0.28)))
                }
            }
            .aspectRatio(1, contentMode: .fit)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityLabel("Footprint on surface \(surface), \(points.count) rays pass, \(clipped.count) clipped")
        }
        .padding(Space.m)
    }
}

// MARK: - Encircled energy (geometric and diffraction)

struct EncircledEnergyView: View {
    struct Curve { var field: Int; var label: String; var radius: [Double]; var encircled: [Double]; var limit: [Double]; var r80: Double }
    let curves: [Curve]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let list = j["fields"]?.array, !list.isEmpty else { return nil }
        let shared = j["radius"]?.nfArray ?? output.buffer(ref: j["radius_mm"])?.doubles ?? []
        curves = list.enumerated().compactMap { k, f in
            let enc = f["encircled"]?.nfArray ?? output.buffer(ref: f["encircled"])?.doubles ?? []
            guard !enc.isEmpty, enc.count == shared.count else { return nil }
            let i = f["field"]?.int ?? k
            return Curve(
                field: i, label: "Field \(i + 1) · " + context.fieldLabel(f["field_value"]?.nfArray), radius: shared, encircled: enc,
                limit: output.buffer(ref: f["diffraction_limit"])?.doubles ?? [], r80: (f["r80"] ?? f["r80_mm"])?.nf ?? .nan)
        }
        if curves.isEmpty { return nil }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            FlowLayout(spacing: Space.m) {
                ForEach(curves, id: \.field) { c in LegendItem(color: Palette.fieldColor(c.field), label: c.label + " · r80 \(Fmt.length(c.r80))") }
                if curves.contains(where: { !$0.limit.isEmpty }) { LegendItem(color: .secondary, label: "Diffraction limit", dashed: true) }
            }
            LineChart(
                series: curves.map { c in
                    Series(id: "\(c.field)", label: c.label, color: Palette.fieldColor(c.field), points: zip(c.radius.map { $0 * 1000 }, c.encircled).map { ($0, $1) })
                } + curves.prefix(1).filter { !$0.limit.isEmpty }.map { c in
                    Series(id: "limit", label: "", color: .secondary, points: zip(c.radius.map { $0 * 1000 }, c.limit).map { ($0, $1) }, dashed: true)
                }, xLabel: "Radius (µm)", yLabel: "Encircled energy", yDomain: 0...1.02, zeroLine: false)
        }
        .padding(Space.m)
    }
}

// MARK: - MTF (geometric and wave)

struct MTFView: View {
    struct Curve { var field: Int; var label: String; var tangential: [Double]; var sagittal: [Double] }
    let frequencies: [Double]
    let curves: [Curve]
    let limit: [Double]
    let nyquist: Double?
    let cutoff: Double?

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        let freq = j["frequencies"]?.nfArray ?? output.buffer(ref: j["frequencies"])?.doubles ?? []
        guard !freq.isEmpty else { return nil }
        frequencies = freq
        var out: [Int: Curve] = [:]
        var lim: [Double] = []
        if let list = j["curves"]?.array {  // geometric_mtf
            for c in list {
                let f = c["field"]?.int ?? 0
                var cur = out[f] ?? Curve(field: f, label: "Field \(f + 1)", tangential: [], sagittal: [])
                let m = c["modulation"]?.nfArray ?? []
                if c["orientation"]?.string == "sagittal" { cur.sagittal = m } else { cur.tangential = m }
                out[f] = cur
            }
            let values = j["field_values"]?.array ?? []
            for (k, f) in out.keys.sorted().enumerated() where values.indices.contains(k) {
                out[f]?.label = "Field \(f + 1) · " + context.fieldLabel(values[k].nfArray)
            }
            cutoff = j["cutoff_frequency"]?.nf
        } else if let list = j["fields"]?.array {  // wave mtf
            for (k, f) in list.enumerated() {
                let i = f["field"]?.int ?? k
                out[i] = Curve(
                    field: i, label: "Field \(i + 1) · " + context.fieldLabel(f["field_value"]?.nfArray),
                    tangential: output.buffer(ref: f["tangential"])?.doubles ?? f["tangential"]?.nfArray ?? [],
                    sagittal: output.buffer(ref: f["sagittal"])?.doubles ?? f["sagittal"]?.nfArray ?? [])
                if lim.isEmpty { lim = output.buffer(ref: f["diffraction_limit"])?.doubles ?? [] }
            }
            cutoff = list.first?["cutoff"]?.nf
        } else {
            return nil
        }
        curves = out.keys.sorted().compactMap { out[$0] }
        limit = lim
        nyquist = j["nyquist"]?.nfArray?.first.flatMap { $0.isFinite ? $0 : nil }
        if curves.isEmpty { return nil }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            FlowLayout(spacing: Space.m) {
                ForEach(curves, id: \.field) { c in LegendItem(color: Palette.fieldColor(c.field), label: c.label) }
                LegendItem(color: .secondary, label: "T solid, S dashed")
                if !limit.isEmpty { LegendItem(color: .primary.opacity(0.5), label: "Diffraction limit") }
            }
            LineChart(
                series: (limit.isEmpty ? [] : [Series(id: "limit", label: "", color: .primary.opacity(0.45), points: zip(frequencies, limit).map { ($0, $1) })])
                    + curves.flatMap { c in
                        [
                            Series(id: "t\(c.field)", label: "", color: Palette.fieldColor(c.field), points: zip(frequencies, c.tangential).map { ($0, $1) }),
                            Series(id: "s\(c.field)", label: "", color: Palette.fieldColor(c.field), points: zip(frequencies, c.sagittal).map { ($0, $1) }, dashed: true),
                        ]
                    },
                xLabel: "Spatial frequency (cycles/mm)", yLabel: "Modulation", xDomain: 0...(frequencies.last ?? 1), yDomain: 0...1.02,
                zeroLine: false, markers: nyquist.map { [(x: $0, label: "Nyquist")] } ?? [])
        }
        .padding(Space.m)
    }
}

// MARK: - Best focus

struct BestFocusView: View {
    let defocus: Double
    let merit: Double
    let meritAtZero: Double
    let throughFocus: [(Double, Double)]

    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let d = j["defocus"]?.nf else { return nil }
        defocus = d
        merit = j["merit"]?.nf ?? .nan
        meritAtZero = j["merit_at_zero"]?.nf ?? .nan
        let x = j["through_focus_defocus"]?.nfArray ?? [], y = j["through_focus_merit"]?.nfArray ?? []
        throughFocus = Array(zip(x, y))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            KeyValueGrid(rows: [
                ("Best defocus", Fmt.length(defocus)), ("RMS at best focus", Fmt.length(merit)),
                ("RMS at the image surface", Fmt.length(meritAtZero)),
            ])
            if throughFocus.count > 2 {
                LineChart(
                    series: [Series(id: "tf", label: "", color: Palette.fieldColor(0), points: throughFocus.map { ($0.0, $0.1 * 1000) })],
                    xLabel: "Defocus (mm)", yLabel: "RMS radius (µm)", zeroLine: false, markers: [(x: defocus, label: "Best")])
            } else {
                Text("Set Through Focus Samples to plot the merit against defocus.").font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
        }
        .padding(Space.l)
    }
}
