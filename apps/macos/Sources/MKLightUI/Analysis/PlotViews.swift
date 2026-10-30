import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

/// Plot views. They only display engine results; units are converted for
/// display (mm → µm) but nothing optical is computed here.

private func wavelengthLabel(_ um: Double) -> String { String(format: "%.4f µm", um) }

private func fieldLabel(_ i: Int, _ v: Point2, _ unit: String) -> String {
    let fmt: (Double) -> String = { String(format: abs($0) < 100 ? "%.2f" : "%.1f", $0) }
    return v.x == 0 ? "F\(i + 1): \(fmt(v.y))\(unit)" : "F\(i + 1): (\(fmt(v.x)), \(fmt(v.y)))\(unit)"
}

private struct WavelengthScale {
    let labels: [String]
    let colors: [Color]
    init(_ wavelengthsUm: [Double]) {
        labels = wavelengthsUm.map(wavelengthLabel)
        colors = wavelengthsUm.map(Palette.wavelengthColor)
    }
}

// MARK: - Spot diagram

struct SpotDiagramView: View {
    let result: SpotDiagramResult
    let fieldUnit: String

    var body: some View {
        let scale = WavelengthScale(result.wavelengthsUm)
        // Common scale across fields (µm), at least the Airy disk.
        let extent = max(result.fields.map(\.geoRadius).max() ?? 0, result.airyRadius) * 1000 * 1.1
        let m = extent > 0 ? extent : 1
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 12)], spacing: 12) {
                ForEach(result.fields, id: \.field) { f in
                    VStack(spacing: 4) {
                        Chart {
                            ForEach(Array(f.points.enumerated()), id: \.offset) { _, p in
                                PointMark(x: .value("x (µm)", p.x * 1000), y: .value("y (µm)", p.y * 1000))
                                    .symbolSize(5)
                                    .foregroundStyle(by: .value("Wavelength", scale.labels[p.wavelength]))
                            }
                            if result.airyRadius > 0 {
                                ForEach(0...64, id: \.self) { k in
                                    let a = Double(k) / 64 * 2 * .pi
                                    LineMark(
                                        x: .value("x (µm)", result.airyRadius * 1000 * cos(a)),
                                        y: .value("y (µm)", result.airyRadius * 1000 * sin(a)),
                                        series: .value("series", "Airy"))
                                        .foregroundStyle(.secondary)
                                        .lineStyle(StrokeStyle(lineWidth: 0.8, dash: [3, 2]))
                                }
                            }
                        }
                        .chartForegroundStyleScale(domain: scale.labels, range: scale.colors)
                        .chartXScale(domain: -m...m)
                        .chartYScale(domain: -m...m)
                        .chartLegend(.hidden)
                        .aspectRatio(1, contentMode: .fit)
                        Text(fieldLabel(f.field, f.fieldValue, fieldUnit)).font(.caption.bold())
                        Text(String(format: "RMS %.2f µm   GEO %.2f µm", f.rmsRadius * 1000, f.geoRadius * 1000))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(10)
            legend(scale)
            Text(String(format: "Scale ±%.1f µm · dashed circle: Airy radius %.2f µm", m, result.airyRadius * 1000))
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.bottom, 8)
        }
    }
}

private func legend(_ scale: WavelengthScale) -> some View {
    HStack(spacing: 12) {
        ForEach(scale.labels.indices, id: \.self) { i in
            HStack(spacing: 4) {
                Circle().fill(scale.colors[i]).frame(width: 8, height: 8)
                Text(scale.labels[i]).font(.caption)
            }
        }
    }
}

// MARK: - Ray / OPD fans

struct FanView: View {
    let result: FanResult
    let fieldUnit: String

    private struct Sample: Identifiable {
        let id: Int
        let pupil: Double
        let value: Double
        let series: String
    }

    var body: some View {
        let scale = WavelengthScale(result.wavelengthsUm)
        let isOPD = result.quantity == .opd
        let factor = isOPD ? 1 : 1000.0  // mm → µm for ray fans
        let unit = isOPD ? "waves" : "µm"
        let all = result.fields.flatMap { $0.tangential + $0.sagittal }.flatMap(\.values).compactMap { $0 }.map { abs($0 * factor) }
        let m = max(all.max() ?? 0, isOPD ? 0.05 : 0.5) * 1.1
        ScrollView {
            VStack(spacing: 12) {
                ForEach(result.fields, id: \.field) { f in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(fieldLabel(f.field, f.fieldValue, fieldUnit)).font(.caption.bold())
                        HStack(spacing: 16) {
                            fanChart(f.tangential, title: isOPD ? "Tangential W(Py)" : "Tangential εy(Py)", factor: factor, unit: unit, m: m, scale: scale)
                            fanChart(f.sagittal, title: isOPD ? "Sagittal W(Px)" : "Sagittal εx(Px)", factor: factor, unit: unit, m: m, scale: scale)
                        }
                    }
                }
                legend(scale)
            }
            .padding(10)
        }
    }

    private func fanChart(_ curves: [FanResult.Curve], title: String, factor: Double, unit: String, m: Double, scale: WavelengthScale) -> some View {
        var samples: [Sample] = []
        for c in curves {
            for (k, v) in c.values.enumerated() {
                if let v { samples.append(Sample(id: samples.count, pupil: c.pupil[k], value: v * factor, series: scale.labels[c.wavelength])) }
            }
        }
        return VStack(spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Chart(samples) { s in
                LineMark(x: .value("Pupil", s.pupil), y: .value(unit, s.value), series: .value("λ", s.series))
                    .foregroundStyle(by: .value("λ", s.series))
                    .lineStyle(StrokeStyle(lineWidth: 1.4))
            }
            .chartForegroundStyleScale(domain: scale.labels, range: scale.colors)
            .chartXScale(domain: -1...1)
            .chartYScale(domain: -m...m)
            .chartYAxisLabel(unit)
            .chartLegend(.hidden)
            .frame(minHeight: 150)
        }
    }
}

// MARK: - Field curvature and distortion

struct FieldCurvatureView: View {
    let result: FieldCurvatureResult

    private struct Sample: Identifiable {
        let id: Int
        let field: Double
        let value: Double
        let series: String
        let wavelength: String
        let kind: String
    }

    var body: some View {
        let scale = WavelengthScale(result.wavelengthsUm)
        var curv: [Sample] = [], dist: [Sample] = []
        for c in result.curves {
            let wl = scale.labels[c.wavelength]
            for (k, f) in result.fieldValues.enumerated() {
                if let t = c.tangential[k] { curv.append(Sample(id: curv.count, field: f, value: t, series: wl + " T", wavelength: wl, kind: "Tangential")) }
                if let s = c.sagittal[k] { curv.append(Sample(id: curv.count, field: f, value: s, series: wl + " S", wavelength: wl, kind: "Sagittal")) }
                if let d = c.distortionPercent[k] { dist.append(Sample(id: dist.count, field: f, value: d, series: wl, wavelength: wl, kind: "Distortion")) }
            }
        }
        return VStack(spacing: 6) {
            HStack(spacing: 16) {
                VStack(spacing: 2) {
                    Text("Field curvature (focus shift, mm)").font(.caption).foregroundStyle(.secondary)
                    Chart(curv) { s in
                        LineMark(x: .value("Focus (mm)", s.value), y: .value("Field", s.field), series: .value("Series", s.series))
                            .foregroundStyle(by: .value("λ", s.wavelength))
                            .lineStyle(by: .value("Orientation", s.kind))
                    }
                    .chartForegroundStyleScale(domain: scale.labels, range: scale.colors)
                    .chartYAxisLabel("Field (\(result.fieldUnit))")
                }
                VStack(spacing: 2) {
                    Text("Distortion (%)").font(.caption).foregroundStyle(.secondary)
                    Chart(dist) { s in
                        LineMark(x: .value("Distortion (%)", s.value), y: .value("Field", s.field), series: .value("Series", s.series))
                            .foregroundStyle(by: .value("λ", s.wavelength))
                    }
                    .chartForegroundStyleScale(domain: scale.labels, range: scale.colors)
                    .chartLegend(.hidden)
                }
            }
            Text("Solid: tangential · dashed: sagittal").font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
    }
}

// MARK: - MTF

struct MTFView: View {
    let result: MTFResult
    let fieldUnit: String

    private struct Sample: Identifiable {
        let id: Int
        let frequency: Double
        let modulation: Double
        let series: String
        let field: String
        let orientation: String
    }

    var body: some View {
        var samples: [Sample] = []
        let fieldNames = result.curves.map(\.field).reduce(into: [Int]()) { if !$0.contains($1) { $0.append($1) } }
        let labels = fieldNames.enumerated().map { k, i in
            fieldLabel(i, k < result.fieldValues.count ? result.fieldValues[k] : Point2(0, 0), fieldUnit)
        }
        for c in result.curves {
            let name = labels[fieldNames.firstIndex(of: c.field) ?? 0]
            for (k, f) in result.frequencies.enumerated() {
                samples.append(Sample(id: samples.count, frequency: f, modulation: c.modulation[k], series: "\(name) \(c.orientation.rawValue)", field: name, orientation: c.orientation.rawValue.capitalized))
            }
        }
        if let dl = result.diffractionLimit {
            for (k, f) in result.frequencies.enumerated() {
                samples.append(Sample(id: samples.count, frequency: f, modulation: dl[k], series: "Diffraction limit", field: "Diffraction limit", orientation: "Tangential"))
            }
        }
        var domain = labels, colors = fieldNames.map(Palette.fieldColor)
        if result.diffractionLimit != nil {
            domain.append("Diffraction limit")
            colors.append(.primary)
        }
        return VStack(spacing: 4) {
            Chart(samples) { s in
                LineMark(x: .value("Frequency (cycles/mm)", s.frequency), y: .value("Modulation", s.modulation), series: .value("Series", s.series))
                    .foregroundStyle(by: .value("Field", s.field))
                    .lineStyle(by: .value("Orientation", s.orientation))
            }
            .chartForegroundStyleScale(domain: domain, range: colors)
            .chartYScale(domain: 0...1)
            .chartXAxisLabel("Spatial frequency (cycles/mm)")
            .chartYAxisLabel("Modulation")
            Text(String(format: "Polychromatic · cutoff %.0f cycles/mm · solid: tangential, dashed: sagittal", result.cutoffFrequency))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(10)
    }
}

// MARK: - First-order data

struct FirstOrderView: View {
    let result: FirstOrderResult

    var body: some View {
        Table(result.entries) {
            TableColumn("Quantity", value: \.label)
            TableColumn("Value") { e in
                Text(NumberDisplay.format(e.value, decimals: 6)).monospacedDigit().frame(maxWidth: .infinity, alignment: .trailing)
            }
            TableColumn("Unit", value: \.unit).width(60)
        }
    }
}
