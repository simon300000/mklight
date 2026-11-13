import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

/// The engine's `glass_info` result, read tolerantly.
struct GlassInfoData {
    struct Index: Identifiable {
        var id: Int
        var um: Double
        /// At the analysis environment (relative to air).
        var n: Double
        /// At the catalogue's reference conditions.
        var nCatalog: Double
        var absorptionPerMm: Double
        var inRange: Bool
    }

    var name: String
    var catalog: String
    var nd: Double
    var ne: Double
    var vd: Double
    var pgf: Double
    var catalogNd: Double
    var catalogVd: Double
    var dnd: Double
    var dvd: Double
    var minUm: Double
    var maxUm: Double
    var formula: String?
    var hasThermalData: Bool
    var cte: Double?
    var density: Double?
    var relativeCost: Double?
    var comment: String?
    var temperatureC: Double?
    var pressureAtm: Double?
    var indices: [Index]
    /// `(λ µm, n)` over the glass's range, empty when not requested.
    var curve: [(Double, Double)]

    init?(output: AnalysisOutput) {
        let j = output.json
        guard let name = j["name"]?.string, let nd = j["nd"]?.nf else { return nil }
        self.name = name
        catalog = j["catalog"]?.string ?? ""
        self.nd = nd
        ne = j["ne"]?.nf ?? .nan
        vd = j["vd"]?.nf ?? .nan
        pgf = j["pgf"]?.nf ?? .nan
        catalogNd = j["catalog_nd"]?.nf ?? .nan
        catalogVd = j["catalog_vd"]?.nf ?? .nan
        dnd = j[path: "offset", "dn_d"]?.nf ?? 0
        dvd = j[path: "offset", "dv_d"]?.nf ?? 0
        minUm = j["min_um"]?.nf ?? .nan
        maxUm = j["max_um"]?.nf ?? .nan
        formula = j[path: "dispersion", "formula"]?.string.map(Self.formulaTitle)
        hasThermalData = j["has_thermal_data"]?.bool ?? false
        cte = j["cte"]?.nf.flatMap { $0.isFinite ? $0 : nil }
        density = j["density"]?.nf.flatMap { $0.isFinite ? $0 : nil }
        relativeCost = j["relative_cost"]?.nf.flatMap { $0.isFinite ? $0 : nil }
        comment = j["comment"]?.string.flatMap { $0.trimmingCharacters(in: .whitespaces).isEmpty ? nil : $0 }
        temperatureC = j[path: "environment", "temperature_c"]?.nf
        pressureAtm = j[path: "environment", "pressure_atm"]?.nf
        indices = (j["indices"]?.array ?? []).enumerated().compactMap { k, r in
            guard let um = r["wavelength_um"]?.nf else { return nil }
            return Index(
                id: k, um: um, n: r["n"]?.nf ?? .nan, nCatalog: r["n_catalog"]?.nf ?? .nan,
                absorptionPerMm: r["absorption_per_mm"]?.nf ?? .nan, inRange: r["in_range"]?.bool ?? true)
        }
        if let b = output.buffer(named: "dispersion_curve"), b.shape.count == 2, b.shape[1] == 2 {
            curve = (0..<b.rows).map { (b[2 * $0], b[2 * $0 + 1]) }
        } else {
            curve = []
        }
    }

    var hasOffset: Bool { dnd != 0 || dvd != 0 }

    /// The dispersion formula's usual name.
    static func formulaTitle(_ key: String) -> String {
        switch key {
        case "schott": "Schott"
        case "sellmeier": "Sellmeier"
        case "sellmeier2": "Sellmeier 2"
        case "herzberger": "Herzberger"
        case "conrady": "Conrady"
        case "extended": "Extended"
        case "cauchy": "Cauchy"
        case "tabulated": "Tabulated"
        default: key.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}

/// `glass_info`: the glass's catalogue data, its dispersion curve with the
/// system wavelengths marked, and its indices at those wavelengths.
struct GlassDataView: View {
    let data: GlassInfoData
    let primaryUm: Double?

    init?(output: AnalysisOutput, context: AnalysisContext) {
        guard let d = GlassInfoData(output: output) else { return nil }
        data = d
        // The design's primary is the run's unless the run had its own spectrum.
        let um = context.wavelengthsUm
        primaryUm =
            !AnalysisContext.hasRunSpectrum(output) && um.indices.contains(context.primaryWavelength) ? um[context.primaryWavelength] : nil
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                header
                curveSection
                if !data.indices.isEmpty { indicesSection }
                propertiesSection
            }
            .padding(Space.l)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            HStack(alignment: .firstTextBaseline, spacing: Space.s) {
                Text(data.name).font(.title3.weight(.semibold)).textSelection(.enabled)
                if !data.catalog.isEmpty { Text(data.catalog).font(.callout).foregroundStyle(.secondary) }
                if data.hasOffset {
                    Text("Offset")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, Space.s)
                        .padding(.vertical, Space.xxs)
                        .background(.tint.opacity(0.15), in: .capsule)
                        .help("Δn_d \(Fmt.signed(data.dnd, decimals: 5)), ΔV_d \(Fmt.signed(data.dvd, decimals: 2)) from the catalog values")
                        .accessibilityLabel("Index and Abbe offset applied")
                }
            }
            .accessibilityElement(children: .combine)
            FlowLayout(spacing: Space.xl, lineSpacing: Space.m) {
                stat("n_d", NumberDisplay.format(data.nd, decimals: 6), help: "Refractive index at the d line (587.56 nm)", spoken: "n d")
                stat("V_d", NumberDisplay.format(data.vd, decimals: 2), help: "Abbe number (n_d − 1) / (n_F − n_C)", spoken: "Abbe number")
                stat("P_g,F", NumberDisplay.format(data.pgf, decimals: 4), help: "Relative partial dispersion (n_g − n_F) / (n_F − n_C)", spoken: "Partial dispersion P g F")
                stat("n_e", NumberDisplay.format(data.ne, decimals: 6), help: "Refractive index at the e line (546.07 nm)", spoken: "n e")
            }
            if data.hasOffset {
                let line = "Catalog n_d \(NumberDisplay.format(data.catalogNd, decimals: 6)), V_d \(NumberDisplay.format(data.catalogVd, decimals: 2))"
                Self.captionSubscripted(line)
                    .font(.caption).foregroundStyle(.secondary)
                    .accessibilityLabel(Subscripted.spoken(line))
            }
        }
        .padding(Space.l)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.fill.quinary, in: .rect(cornerRadius: 10))
    }

    private func stat(_ label: String, _ value: String, help: String, spoken: String) -> some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            Self.captionSubscripted(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3).numeric().textSelection(.enabled)
        }
        .help(help)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(spoken)
        .accessibilityValue(value)
    }

    /// A caption-size label with `_` subscripts a step smaller (the shared
    /// `Subscripted.text` is sized for body text).
    static func captionSubscripted(_ label: String) -> Text {
        Subscripted.pieces(label).reduce(Text(verbatim: "")) { t, p in
            let piece = p.isSubscript ? Text(verbatim: p.text).font(.system(size: 8.5)).baselineOffset(-2) : Text(verbatim: p.text)
            return Text("\(t)\(piece)")
        }
    }

    // MARK: Dispersion curve

    @ViewBuilder private var curveSection: some View {
        section("Dispersion", subtitle: rangeText) {
            if data.curve.count > 1 {
                VStack(alignment: .leading, spacing: Space.s) {
                    DispersionChart(curve: data.curve, indices: data.indices)
                        .frame(height: 220)
                    if !data.indices.isEmpty {
                        WavelengthLegend(wavelengths: Array(data.indices.indices), um: data.indices.map(\.um))
                    }
                }
            } else {
                Text("Set Curve Samples above zero to plot the index over the glass's range.")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    private var rangeText: String {
        guard data.minUm.isFinite, data.maxUm.isFinite else { return "" }
        return "valid \(NumberDisplay.format(data.minUm, decimals: 3))–\(NumberDisplay.format(data.maxUm, decimals: 3)) µm"
    }

    // MARK: Indices

    private var indicesSection: some View {
        section("Indices", subtitle: environmentText) {
            Grid(alignment: .trailing, horizontalSpacing: Space.m, verticalSpacing: Space.xs) {
                GridRow {
                    Text("λ (µm)").gridColumnAlignment(.leading)
                    Text("n").help("Index relative to air at the analysis temperature and pressure")
                    Text("n cat.").help("Index at the catalog's reference conditions")
                    Text("α (1/mm)").help("Internal absorption coefficient")
                }
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                Divider()
                ForEach(data.indices) { r in
                    let primary = primaryUm.map { abs($0 - r.um) < 1e-9 } ?? false
                    GridRow {
                        HStack(spacing: Space.s) {
                            Circle().fill(Palette.wavelengthColor(r.um)).frame(width: 8, height: 8)
                            Text(NumberDisplay.format(r.um, decimals: 4)).fontWeight(primary ? .semibold : .regular)
                            if !r.inRange {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                                    .help("Outside the glass's data range: the index is extrapolated")
                            }
                        }
                        Text(NumberDisplay.format(r.n, decimals: 6))
                        Text(NumberDisplay.format(r.nCatalog, decimals: 6)).foregroundStyle(.secondary)
                        Text(r.absorptionPerMm.isFinite ? Fmt.short(r.absorptionPerMm) : "—").foregroundStyle(.secondary)
                    }
                    .numeric()
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(rowLabel(r, primary: primary))
                }
            }
            .font(.callout)
            // Numbers never wrap mid-digit in a narrow pane; they shrink a
            // little instead.
            .lineLimit(1)
            .minimumScaleFactor(0.75)
            .textSelection(.enabled)
        }
    }

    private func rowLabel(_ r: GlassInfoData.Index, primary: Bool) -> String {
        var s = "\(NumberDisplay.format(r.um, decimals: 4)) micrometers\(primary ? ", primary" : ""): n \(NumberDisplay.format(r.n, decimals: 6))"
        s += ", catalog \(NumberDisplay.format(r.nCatalog, decimals: 6))"
        if !r.inRange { s += ", outside the data range" }
        return s
    }

    private var environmentText: String {
        var parts: [String] = []
        if let t = data.temperatureC, t.isFinite { parts.append("\(NumberDisplay.format(t, decimals: 1)) °C") }
        if let p = data.pressureAtm, p.isFinite { parts.append("\(NumberDisplay.format(p, decimals: 3)) atm") }
        return parts.joined(separator: ", ")
    }

    // MARK: Properties

    private var propertiesSection: some View {
        section("Properties") {
            KeyValueGrid(rows: propertyRows)
            if let c = data.comment {
                Text(c).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }
    }

    private var propertyRows: [(String, String)] {
        var rows: [(String, String)] = []
        if let f = data.formula { rows.append(("Dispersion formula", f)) }
        if data.minUm.isFinite, data.maxUm.isFinite {
            rows.append(("Valid range", "\(NumberDisplay.format(data.minUm, decimals: 3))–\(NumberDisplay.format(data.maxUm, decimals: 3)) µm"))
        }
        rows.append(("Thermo-optic data (dn/dT)", data.hasThermalData ? "Yes" : "None"))
        rows.append(("Thermal expansion", data.cte.map { "\(NumberDisplay.format($0 * 1e6, decimals: 2)) × 10⁻⁶/K" } ?? "—"))
        if let d = data.density { rows.append(("Density", "\(NumberDisplay.format(d, decimals: 3)) g/cm³")) }
        if let c = data.relativeCost { rows.append(("Relative cost", NumberDisplay.format(c, decimals: 2))) }
        return rows
    }

    private func section<C: View>(_ title: String, subtitle: String = "", @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(alignment: .firstTextBaseline, spacing: Space.s) {
                Text(title).font(.subheadline.weight(.semibold))
                if !subtitle.isEmpty { Text(subtitle).font(.caption).foregroundStyle(.secondary) }
            }
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            content()
        }
    }
}

/// n against λ over the glass's range, with the system wavelengths as
/// coloured points on the curve (triangles when outside the data range).
struct DispersionChart: View {
    let curve: [(Double, Double)]
    let indices: [GlassInfoData.Index]

    var body: some View {
        let pts = curve.filter { $0.0.isFinite && $0.1.isFinite }
        let marks = indices.filter { $0.um.isFinite && $0.n.isFinite }
        Chart {
            ForEach(Array(pts.enumerated()), id: \.offset) { _, p in
                LineMark(x: .value("Wavelength (µm)", p.0), y: .value("n", p.1))
                    .foregroundStyle(Color.primary.opacity(0.75))
                    .lineStyle(StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
                    .interpolationMethod(.monotone)
            }
            ForEach(marks) { r in
                RuleMark(x: .value("λ", r.um))
                    .foregroundStyle(Palette.wavelengthColor(r.um).opacity(0.35))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [2, 3]))
                PointMark(x: .value("Wavelength (µm)", r.um), y: .value("n", r.n))
                    .symbol(r.inRange ? BasicChartSymbolShape.circle : .triangle)
                    .symbolSize(55)
                    .foregroundStyle(Palette.wavelengthColor(r.um))
                    .accessibilityLabel("\(NumberDisplay.format(r.um, decimals: 4)) µm")
                    .accessibilityValue("n \(NumberDisplay.format(r.n, decimals: 6))")
            }
        }
        .chartXAxisLabel("Wavelength (µm)", alignment: .center)
        .chartYAxisLabel("n")
        .chartXScale(domain: xDomain(pts, marks))
        .chartYScale(domain: yDomain(pts, marks))
        .chartXAxis { AxisMarks(preset: .aligned, values: .automatic(desiredCount: 6)) }
        .chartYAxis { AxisMarks(values: .automatic(desiredCount: 4)) }
        .environment(\.locale, NumberDisplay.locale)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Refractive index versus wavelength")
    }

    private func xDomain(_ pts: [(Double, Double)], _ marks: [GlassInfoData.Index]) -> ClosedRange<Double> {
        let v = pts.map(\.0) + marks.map(\.um)
        guard let lo = v.min(), let hi = v.max(), lo < hi else { return 0.3...1.0 }
        return lo...hi
    }

    private func yDomain(_ pts: [(Double, Double)], _ marks: [GlassInfoData.Index]) -> ClosedRange<Double> {
        let v = pts.map(\.1) + marks.map(\.n)
        guard var lo = v.min(), var hi = v.max() else { return 1...2 }
        if lo == hi { lo -= 0.01; hi += 0.01 }
        let pad = (hi - lo) * 0.05
        return (lo - pad)...(hi + pad)
    }
}
