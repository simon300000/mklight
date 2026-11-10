import Charts
import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

/// Resolves a `spectrum` setting to the weighted wavelengths the engine
/// will use (for example by `SpectrumSetting.resolve(_:engine:snapshot:)`).
public typealias SpectrumResolver = @Sendable (JSONValue) async throws -> [SpectrumSample]

/// The run-local spectrum of an analysis (or a design job): the design's
/// wavelengths, or a photopic, scotopic, blackbody, explicit or sensor
/// spectrum with its sampling, and a small chart of the weights the run will
/// use. Writes the engine's JSON (`SpectrumSetting.json`), `nil` for the
/// design's wavelengths.
///
/// With a `resolve` function the chart shows the wavelengths the engine
/// resolves; otherwise (or until it answers) a local preview, labelled as
/// approximate.
public struct SpectrumControl: View {
    @Binding var value: JSONValue?
    let systemWavelengths: [SpectrumSample]
    /// The design's primary wavelength (index into `systemWavelengths`).
    var systemPrimary: Int?
    var resolve: SpectrumResolver?

    @State private var importing = false
    @State private var report: SpectrumCSV?
    @State private var resolved: (key: JSONValue, samples: [SpectrumSample])?

    /// - Parameters:
    ///   - value: the `spectrum` setting; `nil` is the design's wavelengths.
    ///   - systemWavelengths: the design's wavelengths and weights (shown
    ///     for "Design wavelengths", the seed of an explicit list).
    ///   - systemPrimary: the design's primary wavelength.
    ///   - resolve: asks the engine what a spectrum resolves to.
    public init(
        value: Binding<JSONValue?>, systemWavelengths: [SpectrumSample], systemPrimary: Int? = nil, resolve: SpectrumResolver? = nil
    ) {
        _value = value
        self.systemWavelengths = systemWavelengths
        self.systemPrimary = systemPrimary
        self.resolve = resolve
    }

    private var setting: SpectrumSetting? { SpectrumSetting(json: value) }

    public var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.s) {
                Text("Spectrum").foregroundStyle(.secondary)
                kindMenu
            }
            if let setting {
                editor(setting)
                preview(setting)
                ForEach(setting.issues, id: \.self) { issue in
                    Label(issue, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                Text("A spectrum this control cannot edit: \(JSONText.text(value))")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
                    .lineLimit(3)
            }
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.commaSeparatedText, .tabSeparatedText, .plainText, .text]) { result in
            guard case .success(let url) = result else { return }
            importCSV(url)
        }
        .task(id: value) { await resolveCurrent() }
    }

    // MARK: Kind

    private var kindMenu: some View {
        Menu {
            Button {
                write(.design)
            } label: {
                Label(SpectrumSetting.Kind.design.title, systemImage: SpectrumSetting.Kind.design.systemImage)
            }
            Divider()
            ForEach([SpectrumSetting.Kind.photopic, .scotopic, .blackbody, .explicit], id: \.self) { k in
                Button {
                    choose(k)
                } label: {
                    Label(k.title + (k == .blackbody ? "…" : ""), systemImage: k.systemImage)
                }
            }
            Button {
                importing = true
            } label: {
                Label("Sensor Response from CSV…", systemImage: SpectrumSetting.Kind.sensor.systemImage)
            }
        } label: {
            Text(setting?.kind.title ?? "Custom")
        }
        .fixedSize()
        .help("The wavelengths this analysis runs with: the design's, or a spectrum sampled for this run only.")
        .accessibilityLabel("Spectrum")
        .accessibilityValue(setting?.kind.title ?? "Custom")
    }

    private func choose(_ k: SpectrumSetting.Kind) {
        let s = setting?.sampling ?? .default
        switch k {
        case .design: write(.design)
        case .photopic: write(.photopic(s.clamped(to: SpectrumSetting.photopicRangeUm)))
        case .scotopic: write(.scotopic(s.clamped(to: SpectrumSetting.scotopicRangeUm)))
        case .blackbody:
            if case .blackbody(let t, _) = setting { write(.blackbody(temperatureK: t, sampling: s)) }
            else { write(.blackbody(temperatureK: 5500, sampling: s)) }
        case .explicit:
            if case .explicit = setting { return }
            let seed = currentSamples ?? systemWavelengths
            write(.explicit(seed.isEmpty ? [SpectrumSample(um: 0.5876, weight: 1)] : seed))
        case .sensor: importing = true
        }
    }

    private func write(_ s: SpectrumSetting) {
        let j = s.json
        if j != value { value = j }
    }

    // MARK: Editors

    @ViewBuilder private func editor(_ s: SpectrumSetting) -> some View {
        switch s {
        case .design:
            EmptyView()
        case .blackbody(_, let sampling):
            samplingEditor(s, sampling)
        case .photopic(let sampling), .scotopic(let sampling):
            samplingEditor(s, sampling)
        case .explicit(let list):
            explicitEditor(list)
        case .sensor(let table, let sampling):
            VStack(alignment: .leading, spacing: Space.s) {
                HStack(spacing: Space.s) {
                    Text(Self.tableSummary(table)).monospacedDigit().foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                    Spacer(minLength: Space.s)
                    Button("Import CSV…") { importing = true }
                        .controlSize(.small)
                        .help("Two columns, wavelength (µm or nm) and response; a header line is optional.")
                }
                if let report { importReport(report) }
                samplingEditor(s, sampling)
            }
        }
    }

    /// Sampling (and a blackbody's temperature, in the same grid so that the
    /// fields line up).
    private func samplingEditor(_ s: SpectrumSetting, _ sampling: SpectralSampling) -> some View {
        Grid(alignment: .leading, horizontalSpacing: Space.s, verticalSpacing: Space.s) {
            if case .blackbody(let t, _) = s {
                GridRow {
                    Text("Temperature").foregroundStyle(.secondary)
                    HStack(spacing: Space.xs) {
                        SpectrumNumberField(value: t, width: 52, label: "Temperature in kelvin") { v in
                            write(.blackbody(temperatureK: v, sampling: sampling))
                        }
                        Text("K").foregroundStyle(.secondary)
                        Spacer(minLength: Space.s)
                        ForEach(SpectrumSetting.blackbodyPresets, id: \.self) { p in
                            Button(SpectrumSetting.kelvin(p)) { write(.blackbody(temperatureK: p, sampling: sampling)) }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                                .monospacedDigit()
                                .disabled(p == t)
                                .help(p == 5500 ? "Daylight, 5500 K" : "Tungsten halogen, 3200 K")
                        }
                    }
                }
            }
            GridRow {
                Text("Samples").foregroundStyle(.secondary)
                HStack(spacing: Space.xs) {
                    SpectrumNumberField(value: Double(sampling.samples), width: 44, integer: true, label: "Number of samples") { v in
                        var n = sampling
                        n.samples = min(max(Int(saturating: v), SpectralSampling.sampleLimits.lowerBound), SpectralSampling.sampleLimits.upperBound)
                        write(s.with(sampling: n))
                    }
                    Picker("Spacing", selection: Binding(get: { sampling.spacing }, set: { sp in
                        var n = sampling
                        n.spacing = sp
                        write(s.with(sampling: n))
                    })) {
                        ForEach(SpectralSampling.Spacing.allCases, id: \.self) { Text($0.title).tag($0) }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .fixedSize()
                    .help("Equal energy: bins of equal energy, each sample at its centroid. Uniform: evenly spaced samples weighted by their energy.")
                }
            }
            GridRow {
                Text("Range").foregroundStyle(.secondary)
                HStack(spacing: Space.xs) {
                    SpectrumNumberField(value: sampling.lowerUm, width: 52, label: "Shortest wavelength in micrometres") { v in
                        var n = sampling
                        n.lowerUm = v
                        write(s.with(sampling: n))
                    }
                    Text("–").foregroundStyle(.secondary)
                    SpectrumNumberField(value: sampling.upperUm, width: 52, label: "Longest wavelength in micrometres") { v in
                        var n = sampling
                        n.upperUm = v
                        write(s.with(sampling: n))
                    }
                    Text("µm").foregroundStyle(.secondary)
                }
            }
        }
    }

    private func explicitEditor(_ list: [SpectrumSample]) -> some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Grid(alignment: .trailing, horizontalSpacing: Space.s, verticalSpacing: Space.xs) {
                GridRow {
                    Text("λ (µm)")
                    Text("Weight")
                    Color.clear.frame(width: 1, height: 1)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(Array(list.enumerated()), id: \.offset) { k, w in
                    GridRow {
                        HStack(spacing: Space.xs) {
                            Circle().fill(Palette.wavelengthColor(w.um)).frame(width: 7, height: 7)
                                .accessibilityHidden(true)
                            SpectrumNumberField(value: w.um, width: 64, label: "Wavelength \(k + 1) in micrometres") { v in
                                var l = list
                                l[k].um = v
                                write(.explicit(l))
                            }
                        }
                        SpectrumNumberField(value: w.weight, width: 52, label: "Weight of wavelength \(k + 1)") { v in
                            var l = list
                            l[k].weight = v
                            write(.explicit(l))
                        }
                        Button {
                            var l = list
                            l.remove(at: k)
                            write(.explicit(l))
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .disabled(list.count <= 1)
                        .help("Remove this wavelength")
                        .accessibilityLabel("Remove wavelength \(k + 1)")
                    }
                }
            }
            HStack(spacing: Space.s) {
                Button("Add", systemImage: "plus") {
                    let last = list.last?.um ?? 0.55
                    write(.explicit(list + [SpectrumSample(um: ((last + 0.05) * 10_000).rounded() / 10_000, weight: 1)]))
                }
                Button("Design Wavelengths") { write(.explicit(systemWavelengths)) }
                    .disabled(systemWavelengths.isEmpty || systemWavelengths == list)
                    .help("Start again from the design's wavelengths and weights")
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
        }
    }

    private func importReport(_ r: SpectrumCSV) -> some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            Text("Imported \(r.table.count) rows in \(r.unit.rawValue)" + (r.header == nil ? "" : ", header skipped"))
                .font(.caption).foregroundStyle(.secondary)
            if !r.badRows.isEmpty {
                DisclosureGroup {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(r.badRows.prefix(20), id: \.line) { b in
                            Text("Line \(b.line): \(b.reason) — \(b.text)").lineLimit(1).truncationMode(.tail)
                        }
                        if r.badRows.count > 20 { Text("… and \(r.badRows.count - 20) more") }
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                } label: {
                    Label("\(r.badRows.count) row\(r.badRows.count == 1 ? "" : "s") skipped", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
        }
    }

    private func importCSV(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let data = try? Data(contentsOf: url) else {
            report = SpectrumCSV(table: [], unit: .micrometre, header: nil, badRows: [.init(line: 0, text: url.lastPathComponent, reason: "cannot be read")])
            return
        }
        let text = String(decoding: data, as: UTF8.self)
        let r = SpectrumCSV.parse(text)
        report = r
        guard r.isUsable, let range = r.rangeUm else { return }
        var sampling = setting?.sampling ?? .default
        sampling.lowerUm = range.lowerBound
        sampling.upperUm = range.upperBound
        write(.sensor(table: r.table, sampling: sampling))
    }

    static func tableSummary(_ t: [SpectrumSample]) -> String {
        guard let a = t.first?.um, let b = t.last?.um else { return "No table" }
        return String(format: "%d rows · %.3f–%.3f µm", t.count, a, b)
    }

    // MARK: Preview

    /// The weights of the current setting from the engine when it has
    /// answered for this very value.
    private var engineSamples: [SpectrumSample]? {
        guard let resolved, let value, resolved.key == value else { return nil }
        return resolved.samples
    }

    private var currentSamples: [SpectrumSample]? {
        guard let setting else { return nil }
        if case .design = setting { return systemWavelengths }
        return engineSamples ?? setting.previewSamples()
    }

    @ViewBuilder private func preview(_ s: SpectrumSetting) -> some View {
        let source: SpectrumPreview.Source =
            switch s {
            case .design: .design
            case .explicit: .exact
            default: engineSamples != nil ? .engine : .approximate
            }
        if let samples = currentSamples, !samples.isEmpty {
            SpectrumPreview(
                samples: samples, curve: s.previewCurve(), source: source,
                primary: source == .design ? systemPrimary.flatMap { samples.indices.contains($0) ? $0 : nil } : nil)
        }
    }

    private func resolveCurrent() async {
        guard let resolve, let value, let s = SpectrumSetting(json: value), s.issues.isEmpty else { return }
        if case .explicit = s { return }
        try? await Task.sleep(for: .milliseconds(250))
        guard !Task.isCancelled else { return }
        do {
            let samples = try await resolve(value)
            guard !Task.isCancelled else { return }
            resolved = (value, samples)
        } catch {
            // The local preview stays (labelled approximate).
        }
    }
}

extension SpectralSampling {
    /// The range moved inside `r` (for the CIE tables).
    func clamped(to r: ClosedRange<Double>) -> SpectralSampling {
        var s = self
        s.lowerUm = min(max(lowerUm, r.lowerBound), r.upperBound)
        s.upperUm = min(max(upperUm, r.lowerBound), r.upperBound)
        if s.upperUm <= s.lowerUm {
            s.lowerUm = r.lowerBound
            s.upperUm = r.upperBound
        }
        return s
    }
}

/// The weights a run uses against wavelength: one bar per wavelength in its
/// spectral colour (tallest = 1), over the continuous response it samples
/// (light area), with the primary marked.
struct SpectrumPreview: View {
    enum Source {
        /// The design's own wavelengths.
        case design
        /// Explicit wavelengths: exactly what the run uses.
        case exact
        /// Resolved by the engine for this spectrum.
        case engine
        /// Computed locally from approximations.
        case approximate

        var caption: String {
            switch self {
            case .design: "Design"
            case .exact: "Explicit"
            case .engine: "Engine-resolved"
            case .approximate: "Approximate preview"
            }
        }
    }

    let samples: [SpectrumSample]
    let curve: [SpectrumSample]?
    let source: Source
    /// The primary's index; by default the engine's rule for spectra
    /// (nearest the energy-weighted mean).
    var primary: Int?

    var body: some View {
        let peak = max(samples.map(\.weight).max() ?? 1, .leastNormalMagnitude)
        let primary = primary ?? SpectrumSetting.primaryIndex(samples)
        let domain = xDomain
        VStack(alignment: .leading, spacing: Space.xxs) {
            Chart {
                if let curve {
                    ForEach(Array(curve.enumerated()), id: \.offset) { _, p in
                        AreaMark(x: .value("λ", p.um), y: .value("Response", p.weight))
                            .foregroundStyle(Color.primary.opacity(0.07))
                            .interpolationMethod(.monotone)
                    }
                }
                ForEach(Array(samples.enumerated()), id: \.offset) { k, w in
                    RuleMark(x: .value("λ", w.um), yStart: .value("Weight", 0), yEnd: .value("Weight", w.weight / peak))
                        .foregroundStyle(Palette.wavelengthColor(w.um))
                        .lineStyle(StrokeStyle(lineWidth: samples.count > 40 ? 1.5 : 3, lineCap: .butt))
                    if k == primary {
                        PointMark(x: .value("λ", w.um), y: .value("Weight", w.weight / peak))
                            .symbolSize(22)
                            .foregroundStyle(Palette.wavelengthColor(w.um))
                    }
                }
            }
            .chartXScale(domain: domain)
            .chartYScale(domain: 0...1.08)
            .chartYAxis(.hidden)
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 4)) { v in
                    AxisGridLine().foregroundStyle(.quaternary)
                    AxisTick()
                    AxisValueLabel { if let d = v.as(Double.self) { Text(String(format: "%.2f", d)).monospacedDigit() } }
                }
            }
            .frame(height: 64)
            HStack(spacing: Space.xs) {
                Text(source.caption)
                Text("·")
                Text("\(samples.count) λ")
                if let primary {
                    Text("·")
                    Text(String(format: "primary %.4f µm", samples[primary].um))
                }
            }
            .font(.caption2)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Spectrum preview, \(source.caption.lowercased())")
        .accessibilityValue(accessibilityValue(primary))
    }

    private var xDomain: ClosedRange<Double> {
        let xs = samples.map(\.um) + (curve?.map(\.um) ?? [])
        guard let lo = xs.min(), let hi = xs.max() else { return 0.4...0.7 }
        let pad = max((hi - lo) * 0.08, 0.012)
        return (lo - pad)...(hi + pad)
    }

    private func accessibilityValue(_ primary: Int?) -> String {
        guard let lo = samples.map(\.um).min(), let hi = samples.map(\.um).max() else { return "" }
        var s = String(format: "%d wavelengths from %.3f to %.3f micrometres", samples.count, lo, hi)
        if let primary { s += String(format: ", primary %.4f", samples[primary].um) }
        return s
    }
}

/// A compact number field that commits on Return or when it loses focus,
/// if the text was edited; unreadable text reverts.
struct SpectrumNumberField: View {
    let value: Double
    var width: CGFloat = 60
    var integer = false
    let label: String
    let commit: (Double) -> Void

    var body: some View {
        CommittingField(value: value, format: entry, commit: commit) { text, _ in
            TextField(label, text: text)
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
                .frame(width: width)
        }
        .accessibilityLabel(label)
    }

    /// A finite number, whole where the value is.
    private var entry: FieldEdit<Double>.Format {
        FieldEdit.Format(
            exact: { [integer] in integer ? String(Int(saturating: $0)) : FieldText.exact($0) },
            display: { [integer] in integer ? String(Int(saturating: $0)) : Fmt.short($0) },
            parse: { [integer] text in
                guard let d = LensValueParser.number(text.trimmingCharacters(in: .whitespaces)), d.isFinite else { return nil }
                return integer ? d.rounded() : d
            })
    }
}
