import Foundation
import MKLightEngine
import MKLightModel

/// One weighted wavelength (µm). In a sensor table `weight` is the
/// response at that wavelength.
public struct SpectrumSample: Equatable, Hashable, Sendable {
    public var um: Double
    public var weight: Double

    public init(um: Double, weight: Double) {
        self.um = um
        self.weight = weight
    }
}

/// How the engine samples a continuous spectrum (`SpectralSampling`):
/// `samples` wavelengths over `range_um`, either in bins of equal energy
/// (each at its bin's centroid, equal weights) or evenly spaced with the
/// energy of their cells as weights.
public struct SpectralSampling: Equatable, Hashable, Sendable {
    public enum Spacing: String, CaseIterable, Sendable {
        case equalWeight = "equal_weight"
        case uniform

        public var title: String { self == .equalWeight ? "Equal energy" : "Uniform" }
    }

    public var samples: Int
    public var lowerUm: Double
    public var upperUm: Double
    public var spacing: Spacing

    /// The engine's defaults: 9 samples over 0.38–0.78 µm, equal energy.
    public static let `default` = SpectralSampling()
    /// The engine's limits on `samples`.
    public static let sampleLimits = 1...4096

    public init(samples: Int = 9, lowerUm: Double = 0.38, upperUm: Double = 0.78, spacing: Spacing = .equalWeight) {
        self.samples = samples
        self.lowerUm = lowerUm
        self.upperUm = upperUm
        self.spacing = spacing
    }

    /// Reads `{samples, range_um, spacing}`; missing keys take the engine
    /// defaults, unknown keys or wrong types give `nil` (the engine rejects
    /// them too).
    public init?(json: JSONValue?) {
        self.init()
        guard let json, json != .null else { return }
        guard let o = json.object, Set(o.keys).isSubset(of: ["samples", "range_um", "spacing"]) else { return nil }
        if let v = o["samples"] {
            guard let n = v.int else { return nil }
            samples = n
        }
        if let v = o["range_um"] {
            guard let r = v.array?.compactMap(\.doubleValue), r.count == 2, v.array?.count == 2 else { return nil }
            lowerUm = r[0]
            upperUm = r[1]
        }
        if let v = o["spacing"] {
            guard let s = v.string.flatMap(Spacing.init(rawValue:)) else { return nil }
            spacing = s
        }
    }

    /// Only what differs from the defaults (`nil` when nothing does), so
    /// that the run's settings stay short.
    public var json: JSONValue? {
        var o: [String: JSONValue] = [:]
        let d = Self.default
        if samples != d.samples { o["samples"] = .integer(Int64(samples)) }
        if lowerUm != d.lowerUm || upperUm != d.upperUm { o["range_um"] = .array([.number(lowerUm), .number(upperUm)]) }
        if spacing != d.spacing { o["spacing"] = .string(spacing.rawValue) }
        return o.isEmpty ? nil : .object(o)
    }

    /// The engine's checks (`validate_sampling`).
    public var issues: [String] {
        var out: [String] = []
        if !Self.sampleLimits.contains(samples) { out.append("Samples must be between 1 and 4096.") }
        if !(lowerUm > 0 && upperUm > lowerUm && upperUm.isFinite) { out.append("The range must be positive and increasing.") }
        return out
    }
}

/// The `spectrum` setting of every wavelength-using analysis: absent (the
/// design's wavelengths) or a run-local spectrum in the engine's format
/// (`mklight_analysis::spectrum::Spectrum`, tagged by `kind`):
///
/// ```json
/// {"kind": "explicit", "wavelengths": [{"um": 0.55, "weight": 1}]}
/// {"kind": "photopic" | "scotopic", "sampling": {...}}
/// {"kind": "blackbody", "temperature_k": 5500, "sampling": {...}}
/// {"kind": "sensor", "table": [[0.4, 0.1], ...], "sampling": {...}}
/// ```
///
/// `sampling` is `{samples, range_um: [min, max], spacing: "equal_weight" |
/// "uniform"}`, every key optional. The engine places the primary at the
/// sample nearest the energy-weighted mean wavelength.
public enum SpectrumSetting: Equatable, Hashable, Sendable {
    case design
    case photopic(SpectralSampling)
    case scotopic(SpectralSampling)
    case blackbody(temperatureK: Double, sampling: SpectralSampling)
    case explicit([SpectrumSample])
    case sensor(table: [SpectrumSample], sampling: SpectralSampling)

    public enum Kind: String, CaseIterable, Identifiable, Sendable {
        case design, photopic, scotopic, blackbody, explicit, sensor
        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .design: "Design wavelengths"
            case .photopic: "Photopic V(λ)"
            case .scotopic: "Scotopic V′(λ)"
            case .blackbody: "Blackbody"
            case .explicit: "Explicit wavelengths"
            case .sensor: "Sensor response"
            }
        }

        public var systemImage: String {
            switch self {
            case .design: "list.bullet"
            case .photopic: "sun.max"
            case .scotopic: "moon"
            case .blackbody: "flame"
            case .explicit: "slider.horizontal.3"
            case .sensor: "camera.aperture"
            }
        }
    }

    /// The CIE tables' ranges in the engine (CIE 018:2019, 1 nm steps):
    /// sampling ranges must lie inside them.
    public static let photopicRangeUm = 0.36...0.83
    public static let scotopicRangeUm = 0.38...0.78
    /// Quick picks for the blackbody temperature (K): daylight, tungsten
    /// halogen, a D65-like sky, a warm incandescent.
    public static let blackbodyPresets: [Double] = [5500, 3200, 6500, 2856]

    public var kind: Kind {
        switch self {
        case .design: .design
        case .photopic: .photopic
        case .scotopic: .scotopic
        case .blackbody: .blackbody
        case .explicit: .explicit
        case .sensor: .sensor
        }
    }

    public var sampling: SpectralSampling? {
        switch self {
        case .photopic(let s), .scotopic(let s), .blackbody(_, let s), .sensor(_, let s): s
        case .design, .explicit: nil
        }
    }

    /// The same spectrum with other sampling (no change for design and
    /// explicit wavelengths).
    public func with(sampling s: SpectralSampling) -> SpectrumSetting {
        switch self {
        case .photopic: .photopic(s)
        case .scotopic: .scotopic(s)
        case .blackbody(let t, _): .blackbody(temperatureK: t, sampling: s)
        case .sensor(let table, _): .sensor(table: table, sampling: s)
        case .design, .explicit: self
        }
    }

    // MARK: JSON

    /// Reads the setting; `nil` / `null` is the design. Returns `nil` for
    /// anything the engine would reject structurally (unknown kind or key,
    /// wrong types).
    public init?(json: JSONValue?) {
        guard let json, json != .null else {
            self = .design
            return
        }
        guard let o = json.object, let kind = o["kind"]?.string else { return nil }
        func keys(_ allowed: Set<String>) -> Bool { Set(o.keys).isSubset(of: allowed.union(["kind"])) }
        switch kind {
        case "photopic", "scotopic":
            guard keys(["sampling"]), let s = SpectralSampling(json: o["sampling"]) else { return nil }
            self = kind == "photopic" ? .photopic(s) : .scotopic(s)
        case "blackbody":
            guard keys(["temperature_k", "sampling"]), let t = o["temperature_k"]?.doubleValue,
                let s = SpectralSampling(json: o["sampling"])
            else { return nil }
            self = .blackbody(temperatureK: t, sampling: s)
        case "explicit":
            guard keys(["wavelengths"]), let list = o["wavelengths"]?.array else { return nil }
            var out: [SpectrumSample] = []
            for w in list {
                guard let wo = w.object, Set(wo.keys).isSubset(of: ["um", "weight"]), let um = wo["um"]?.doubleValue else { return nil }
                let weight: Double
                if let v = wo["weight"] {
                    guard let d = v.doubleValue else { return nil }
                    weight = d
                } else {
                    weight = 1  // the engine's default
                }
                out.append(SpectrumSample(um: um, weight: weight))
            }
            self = .explicit(out)
        case "sensor":
            guard keys(["table", "sampling"]), let rows = o["table"]?.array, let s = SpectralSampling(json: o["sampling"]) else { return nil }
            var table: [SpectrumSample] = []
            for r in rows {
                guard let p = r.array, p.count == 2, let um = p[0].doubleValue, let v = p[1].doubleValue else { return nil }
                table.append(SpectrumSample(um: um, weight: v))
            }
            self = .sensor(table: table, sampling: s)
        default:
            return nil
        }
    }

    /// The engine's JSON; `nil` for the design (the setting is left out).
    public var json: JSONValue? {
        func with(_ kind: String, _ extra: [String: JSONValue], _ s: SpectralSampling) -> JSONValue {
            var o = extra
            o["kind"] = .string(kind)
            if let sj = s.json { o["sampling"] = sj }
            return .object(o)
        }
        switch self {
        case .design:
            return nil
        case .photopic(let s):
            return with("photopic", [:], s)
        case .scotopic(let s):
            return with("scotopic", [:], s)
        case .blackbody(let t, let s):
            return with("blackbody", ["temperature_k": .number(t)], s)
        case .explicit(let list):
            return .object([
                "kind": .string("explicit"),
                "wavelengths": .array(list.map { .object(["um": .number($0.um), "weight": .number($0.weight)]) }),
            ])
        case .sensor(let table, let s):
            return with("sensor", ["table": .array(table.map { .array([.number($0.um), .number($0.weight)]) })], s)
        }
    }

    // MARK: Validation

    /// What the engine would refuse, in words, before a run. Empty when the
    /// spectrum is valid (energy-in-range is checked by the engine).
    public var issues: [String] {
        switch self {
        case .design:
            return []
        case .photopic(let s):
            return s.issues + Self.rangeIssue(s, inside: Self.photopicRangeUm, what: "the photopic table")
        case .scotopic(let s):
            return s.issues + Self.rangeIssue(s, inside: Self.scotopicRangeUm, what: "the scotopic table")
        case .blackbody(let t, let s):
            return (t > 0 && t.isFinite ? [] : ["The temperature must be positive."]) + s.issues
        case .explicit(let list):
            var out: [String] = []
            if list.isEmpty { out.append("Add at least one wavelength.") }
            if list.contains(where: { !($0.um > 0 && $0.um.isFinite) }) { out.append("Wavelengths must be positive.") }
            if list.contains(where: { !($0.weight >= 0 && $0.weight.isFinite) }) { out.append("Weights must not be negative.") }
            return out
        case .sensor(let table, let s):
            var out = s.issues
            if let problem = Self.tableIssue(table) {
                out.append(problem)
            } else if let lo = table.first?.um, let hi = table.last?.um {
                out += Self.rangeIssue(s, inside: lo...hi, what: "the response table")
            }
            return out
        }
    }

    static func tableIssue(_ table: [SpectrumSample]) -> String? {
        if table.count < 2 { return "The response table needs at least two rows." }
        if table.contains(where: { !($0.um > 0 && $0.um.isFinite && $0.weight >= 0 && $0.weight.isFinite) }) {
            return "Responses must not be negative."
        }
        if zip(table, table.dropFirst()).contains(where: { $0.um >= $1.um }) { return "Wavelengths must increase." }
        return nil
    }

    static func rangeIssue(_ s: SpectralSampling, inside r: ClosedRange<Double>, what: String) -> [String] {
        guard s.lowerUm > 0, s.upperUm > s.lowerUm else { return [] }
        if s.lowerUm < r.lowerBound - 1e-12 || s.upperUm > r.upperBound + 1e-12 {
            return [String(format: "The range must lie inside %@ (%.3f–%.3f µm).", what, r.lowerBound, r.upperBound)]
        }
        return []
    }

    // MARK: Text

    /// A short description for settings summaries: `photopic`,
    /// `blackbody 5500 K`, `3 wavelengths`, `sensor (42 rows)`.
    public var summary: String {
        switch self {
        case .design: "design"
        case .photopic(let s): "photopic" + Self.samplesSuffix(s)
        case .scotopic(let s): "scotopic" + Self.samplesSuffix(s)
        case .blackbody(let t, let s): "blackbody \(Self.kelvin(t))" + Self.samplesSuffix(s)
        case .explicit(let l): l.count == 1 ? String(format: "%.4g µm", l[0].um) : "\(l.count) wavelengths"
        case .sensor(let t, let s): "sensor (\(t.count) rows)" + Self.samplesSuffix(s)
        }
    }

    static func samplesSuffix(_ s: SpectralSampling) -> String {
        s.samples == SpectralSampling.default.samples ? "" : " ×\(s.samples)"
    }

    static func kelvin(_ t: Double) -> String {
        t.rounded() == t ? "\(Int(t)) K" : String(format: "%.1f K", t)
    }

    // MARK: Preview

    /// The continuous response the spectrum samples, on its own range, for
    /// a preview drawing (peak 1). Photopic and scotopic use the Gaussian
    /// fits V(λ) ≈ 1.019·exp(−285.4(λ − 0.559)²) and
    /// V′(λ) ≈ 0.992·exp(−321.9(λ − 0.503)²) (λ in µm) instead of the CIE
    /// tables the engine samples: a display approximation. The blackbody is
    /// Planck's law. `nil` for design and explicit wavelengths.
    public func previewCurve(points: Int = 161) -> [SpectrumSample]? {
        guard let s = sampling, s.lowerUm > 0, s.upperUm > s.lowerUm, s.upperUm.isFinite, points > 1 else { return nil }
        let f: (Double) -> Double
        switch self {
        case .photopic: f = { 1.019 * exp(-285.4 * ($0 - 0.559) * ($0 - 0.559)) }
        case .scotopic: f = { 0.992 * exp(-321.9 * ($0 - 0.503) * ($0 - 0.503)) }
        case .blackbody(let t, _):
            guard t > 0, t.isFinite else { return nil }
            // Planck: B_λ ∝ λ⁻⁵ / (exp(c₂/λT) − 1), c₂ = 14387.77 µm·K.
            f = { l in exp(-5 * log(l) - Self.logExpm1(14387.768775039337 / (l * t))) }
        case .sensor(let table, _):
            guard Self.tableIssue(table) == nil else { return nil }
            f = { Self.interpolate(table, $0) }
        case .design, .explicit:
            return nil
        }
        let xs = (0..<points).map { s.lowerUm + (s.upperUm - s.lowerUm) * Double($0) / Double(points - 1) }
        let ys = xs.map(f)
        let peak = ys.filter(\.isFinite).max() ?? 0
        guard peak > 0 else { return nil }
        return zip(xs, ys).map { SpectrumSample(um: $0, weight: $1.isFinite ? $1 / peak : 0) }
    }

    /// The weighted wavelengths a run would use, approximated locally for a
    /// preview (the engine's sampling on `previewCurve`). Explicit
    /// wavelengths are exact; the design returns `nil` (the caller knows
    /// the design's wavelengths).
    public func previewSamples() -> [SpectrumSample]? {
        switch self {
        case .design: return nil
        case .explicit(let l): return l
        default:
            guard let s = sampling, s.issues.isEmpty, let curve = previewCurve(points: 801) else { return nil }
            return Self.sample(curve, s)
        }
    }

    /// The index the engine makes primary: nearest the energy-weighted mean
    /// wavelength, the first on ties.
    public static func primaryIndex(_ samples: [SpectrumSample]) -> Int? {
        let total = samples.reduce(0) { $0 + $1.weight }
        guard !samples.isEmpty else { return nil }
        let mean = total > 0 ? samples.reduce(0) { $0 + $1.um * $1.weight } / total : samples[0].um
        var best = 0
        for (i, w) in samples.enumerated() where abs(w.um - mean) < abs(samples[best].um - mean) { best = i }
        return best
    }

    static func logExpm1(_ x: Double) -> Double { x > 50 ? x : log(expm1(x)) }

    static func interpolate(_ t: [SpectrumSample], _ x: Double) -> Double {
        guard let first = t.first, let last = t.last, x >= first.um, x <= last.um else { return 0 }
        var hi = t.firstIndex { $0.um >= x } ?? t.count - 1
        hi = min(max(hi, 1), t.count - 1)
        let a = t[hi - 1], b = t[hi]
        return a.weight + (b.weight - a.weight) * (x - a.um) / (b.um - a.um)
    }

    /// The engine's two spacings on a finely tabulated curve (trapezoids).
    static func sample(_ curve: [SpectrumSample], _ s: SpectralSampling) -> [SpectrumSample]? {
        guard curve.count > 1 else { return nil }
        // Cumulative energy and first moment at every node.
        var energy = [0.0], moment = [0.0]
        for (p, q) in zip(curve, curve.dropFirst()) {
            let e = (q.um - p.um) * (p.weight + q.weight) / 2
            energy.append(energy.last! + e)
            moment.append(moment.last! + e * (p.um + q.um) / 2)
        }
        let total = energy.last!
        guard total > 0 else { return nil }
        /// Energy and moment from the start to `x` (linear inside a step).
        func cumulative(_ x: Double) -> (Double, Double) {
            let i = min(max((curve.firstIndex { $0.um >= x } ?? curve.count - 1), 1), curve.count - 1)
            let a = curve[i - 1], b = curve[i]
            let f = (x - a.um) / (b.um - a.um)
            let vx = a.weight + (b.weight - a.weight) * f
            let e = (x - a.um) * (a.weight + vx) / 2
            return (energy[i - 1] + e, moment[i - 1] + e * (a.um + x) / 2)
        }
        let n = s.samples
        var out: [SpectrumSample] = []
        switch s.spacing {
        case .uniform:
            let d = (s.upperUm - s.lowerUm) / Double(max(n, 2) - 1)
            for i in 0..<n {
                let um = n == 1 ? (s.lowerUm + s.upperUm) / 2 : s.lowerUm + d * Double(i)
                let half = n == 1 ? (s.upperUm - s.lowerUm) / 2 : d / 2
                let e = cumulative(min(um + half, s.upperUm)).0 - cumulative(max(um - half, s.lowerUm)).0
                out.append(SpectrumSample(um: um, weight: e))
            }
        case .equalWeight:
            var left = s.lowerUm
            for i in 0..<n {
                let target = total * Double(i + 1) / Double(n)
                // Bisection on the cumulative energy.
                var lo = left, hi = s.upperUm
                if i + 1 < n {
                    for _ in 0..<60 {
                        let mid = (lo + hi) / 2
                        if cumulative(mid).0 < target { lo = mid } else { hi = mid }
                    }
                }
                let right = i + 1 == n ? s.upperUm : hi
                let (e0, m0) = cumulative(left), (e1, m1) = cumulative(right)
                let e = e1 - e0
                out.append(SpectrumSample(um: e > 0 ? (m1 - m0) / e : (left + right) / 2, weight: e))
                left = right
            }
        }
        return out
    }
}

/// Reads a sensor response table from CSV text: two columns, wavelength
/// and response; a header line is optional; separators are commas,
/// semicolons or tabs (with semicolons and tabs a decimal comma is
/// accepted); `#` starts a comment line. Wavelengths in nanometres are
/// detected by magnitude (or a header naming `nm`) and converted to µm.
/// Rows that cannot be read are reported with their line numbers and left
/// out; the table is sorted by wavelength.
public struct SpectrumCSV: Equatable, Sendable {
    public enum Unit: String, Sendable { case micrometre = "µm", nanometre = "nm" }

    public struct BadRow: Equatable, Sendable {
        /// 1-based line number in the file.
        public var line: Int
        public var text: String
        public var reason: String
    }

    public var table: [SpectrumSample]
    public var unit: Unit
    public var header: String?
    public var badRows: [BadRow]

    /// Readable table: at least two rows left.
    public var isUsable: Bool { table.count >= 2 }

    /// The sampling range that covers the table.
    public var rangeUm: ClosedRange<Double>? {
        guard let a = table.first?.um, let b = table.last?.um, b > a else { return nil }
        return a...b
    }

    public static func parse(_ text: String) -> SpectrumCSV {
        var rows: [(line: Int, text: String, um: Double, value: Double)] = []
        var bad: [BadRow] = []
        var header: String?
        let lines = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
        for (k, raw) in lines.enumerated() {
            let line = raw.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\u{feff}"))
            if line.isEmpty || line.hasPrefix("#") { continue }
            let separator: Character = line.contains("\t") ? "\t" : line.contains(";") ? ";" : ","
            let decimalComma = separator != ","
            let cells = line.split(separator: separator, omittingEmptySubsequences: false).map {
                $0.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            }
            func number(_ s: String) -> Double? {
                Double(decimalComma ? s.replacingOccurrences(of: ",", with: ".") : s)
            }
            let a = cells.count > 0 ? number(cells[0]) : nil
            let b = cells.count > 1 ? number(cells[1]) : nil
            guard let a, let b else {
                if rows.isEmpty && bad.isEmpty && header == nil && cells.contains(where: { number($0) == nil }) {
                    header = line
                } else {
                    bad.append(BadRow(line: k + 1, text: line, reason: cells.count < 2 ? "needs two columns" : "not a number"))
                }
                continue
            }
            if !(a.isFinite && a > 0) {
                bad.append(BadRow(line: k + 1, text: line, reason: "wavelength must be positive"))
            } else if !(b.isFinite && b >= 0) {
                bad.append(BadRow(line: k + 1, text: line, reason: "response must not be negative"))
            } else {
                rows.append((k + 1, line, a, b))
            }
        }
        // Units: a header that names them wins; otherwise wavelengths above
        // 50 are nanometres (optical µm values stay far below).
        let h = header?.lowercased() ?? ""
        let unit: Unit =
            if h.contains("nm") || h.contains("nanomet") { .nanometre }
            else if h.contains("µm") || h.contains("um") || h.contains("micro") { .micrometre }
            else { (rows.map(\.um).max() ?? 0) > 50 ? .nanometre : .micrometre }
        let scale = unit == .nanometre ? 0.001 : 1
        var table: [SpectrumSample] = []
        for r in rows.sorted(by: { $0.um < $1.um }) {
            let um = r.um * scale
            if let last = table.last, abs(last.um - um) < 1e-12 {
                bad.append(BadRow(line: r.line, text: r.text, reason: "repeats a wavelength"))
                continue
            }
            table.append(SpectrumSample(um: um, weight: r.value))
        }
        return SpectrumCSV(table: table, unit: unit, header: header, badRows: bad.sorted { $0.line < $1.line })
    }
}

extension SpectrumSetting {
    /// The run's weighted wavelengths as the engine resolves them, read from
    /// a `system_summary` result (`wavelengths_um`, `wavelength_weights`).
    public static func resolvedSamples(from json: JSONValue?) -> [SpectrumSample]? {
        guard let um = json?["wavelengths_um"]?.nfArray, !um.isEmpty else { return nil }
        let w = json?["wavelength_weights"]?.nfArray ?? Array(repeating: 1, count: um.count)
        guard w.count == um.count else { return nil }
        return zip(um, w).map { SpectrumSample(um: $0, weight: $1) }
    }

    /// Asks the engine for the wavelengths a spectrum resolves to: a
    /// `system_summary` run with the spectrum on `snapshot`.
    public static func resolve(_ spectrum: JSONValue, engine: any EngineClient, snapshot: ModelSnapshot) async throws -> [SpectrumSample] {
        let out = try await engine.submit(EngineAnalysisRequest(analysis: "system_summary", settings: ["spectrum": spectrum]), on: snapshot).result()
        guard let samples = resolvedSamples(from: out.json) else { throw EngineError.unsupported("system_summary has no wavelengths") }
        return samples
    }
}
