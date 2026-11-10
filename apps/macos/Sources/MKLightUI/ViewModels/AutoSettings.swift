import Foundation
import MKLightEngine
import MKLightModel

/// Settings the engine chooses from the system when they are left at 0,
/// `null` or empty (docs/WAVE.md, "Automatic sampling and ranges"): the
/// pupil grid, image grids and spacings, the MTF frequency limit, the
/// fixed MTF frequencies and the through-focus range. The app shows them
/// as "Auto" with the value the last run used, read from its result
/// ("Auto (282)"), and never as a bare 0.
public enum AutoSettings {
    /// What makes a setting automatic.
    public enum Sentinel: Sendable, Equatable {
        /// `0` (grids, PSF sizes).
        case zero
        /// `null` (limits, ranges, spacings).
        case null
        /// An empty list (fixed MTF frequencies).
        case empty
    }

    /// The automatic settings of `analysis`, by key path into its settings
    /// (`pupil.grid` is the `grid` of the `pupil` object).
    public static func sentinel(_ key: String, analysis: String) -> Sentinel? {
        switch key {
        case "pupil.grid", "optics.pupil.grid", "optics.psf_size": return .zero
        case "grid_size" where ["psf", "diffraction_encircled_energy", "line_spread"].contains(analysis): return .zero
        case "max_frequency" where analysis == "mtf": return .null
        case "focus_range_mm" where analysis == "through_focus_mtf": return .null
        case "spacing_mm" where analysis == "psf": return .null
        case "optics.psf_spacing_mm": return .null
        case "through_focus.min" where analysis == "spot_diagram", "through_focus.max" where analysis == "spot_diagram": return .null
        case "frequencies" where ["mtf_vs_field", "through_focus_mtf"].contains(analysis): return .empty
        default: return nil
        }
    }

    /// Whether `value` asks the engine to choose.
    public static func isAuto(_ value: JSONValue?, _ sentinel: Sentinel) -> Bool {
        switch sentinel {
        case .zero: return value == nil || value?.doubleValue == 0
        case .null: return value == nil || value == .null
        case .empty: return value == nil || value == .null || value?.array?.isEmpty == true
        }
    }

    /// The display of an automatic setting: "Auto (282)", or "Auto" before
    /// a run has said what it used.
    public static func display(used: String?) -> String {
        guard let used, !used.isEmpty else { return "Auto" }
        return "Auto (\(used))"
    }

    /// The value the engine used for the automatic setting `key`, read from
    /// the run's result, as compact text (a range when fields differ:
    /// "254–371").
    public static func used(_ key: String, analysis: String, output: AnalysisOutput?) -> String? {
        guard let output else { return nil }
        let j = output.json
        switch (analysis, key) {
        case ("psf", "grid_size"):
            return range((j["psfs"]?.array ?? []).compactMap { $0[path: "psf", "size"]?.doubleValue })
        case ("psf", "spacing_mm"):
            let s = (j["psfs"]?.array ?? []).compactMap { $0[path: "psf", "spacing_mm"]?.doubleValue }
            return s.min().map { micrometres($0, upper: s.max()) }
        case ("line_spread", "grid_size"):
            return range((j["fields"]?.array ?? []).compactMap { f in output.buffer(ref: f["positions_mm"]).map { Double($0.count) } })
        case ("mtf", "max_frequency"):
            guard let f = output.buffer(key: "frequencies"), f.count > 0 else { return nil }
            return String(format: "%.3g", f[f.count - 1])
        case ("mtf_vs_field", "frequencies"), ("through_focus_mtf", "frequencies"):
            guard let f = output.buffer(key: "frequencies"), f.count > 0 else { return nil }
            return f.doubles.map { String(format: "%.3g", $0) }.joined(separator: ", ")
        case ("through_focus_mtf", "focus_range_mm"):
            guard let f = output.buffer(key: "focus_mm"), f.count > 0 else { return nil }
            let half = f.doubles.map(abs).max() ?? .nan
            return half.isFinite ? "±" + String(format: "%.3g", half) : nil
        case ("spot_diagram", "through_focus"):
            // The planes the run sampled (`fields[].through_focus[].defocus`).
            let planes = (j["fields"]?.array?.first?["through_focus"]?.array ?? []).compactMap { $0["defocus"]?.doubleValue }
            guard let lo = planes.min(), let hi = planes.max(), lo.isFinite, hi.isFinite else { return nil }
            if abs(lo + hi) <= 1e-9 * max(1, abs(hi)) { return "±" + String(format: "%.3g", hi) }
            return String(format: "%.3g to %.3g", lo, hi)
        case (_, "pupil.grid"):
            // Wavefront maps report their grids; other analyses do not.
            return range((j["maps"]?.array ?? []).compactMap { $0["grid"]?.doubleValue })
        case ("image_simulation", "optics.psf_size"):
            return j["psf_size"]?.doubleValue.flatMap(whole).map { String($0) }
        case ("image_simulation", "optics.psf_spacing_mm"):
            return j["psf_spacing_mm"]?.doubleValue.map { micrometres($0, upper: nil) }
        default:
            return nil
        }
    }

    /// "282", or "254–371" when the values differ; `nil` without values or
    /// when one has no whole part.
    static func range(_ v: [Double]) -> String? {
        guard let lo = v.min(), let hi = v.max(), let a = whole(lo), let b = whole(hi) else { return nil }
        return lo == hi ? String(a) : "\(a)–\(b)"
    }

    /// The whole part of `v`; `nil` when `v` is not finite or beyond `Int`.
    static func whole(_ v: Double) -> Int? { Int(exactly: v.rounded(.towardZero)) }

    static func micrometres(_ mm: Double, upper: Double?) -> String {
        let a = String(format: "%.3g", mm * 1000)
        guard let upper, upper > mm * (1 + 1e-6) else { return a + " µm" }
        return a + "–" + String(format: "%.3g", upper * 1000) + " µm"
    }
}
