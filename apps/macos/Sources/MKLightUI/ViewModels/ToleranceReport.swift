import Foundation
import MKLightEngine
import MKLightModel

// MARK: - Unit conversion

extension ToleranceRow {
    /// The row with its limits written in `unit` (the same engine limits):
    /// fringes ↔ ΔR ↔ % of R for radii, arc minutes ↔ degrees, % of V_d ↔
    /// ΔV_d. `nil` when the conversion needs what the design lacks (a
    /// radius of a flat surface, a clear aperture).
    public func converted(to unit: ToleranceUnit, in c: ToleranceContext) -> ToleranceRow? {
        guard unit != self.unit, kind.units.contains(unit) else { return self }
        var out = self
        out.unit = unit
        switch kind {
        case .tiltX, .tiltY:
            let k = unit == .arcmin ? 60.0 : 1.0 / 60
            out.minus = Self.tidy(minus * k)
            out.plus = Self.tidy(plus * k)
            return out
        case .abbe:
            guard let i = c.system.surfaceIndex(of: surface), case .model(_, let vd, _) = c.system.surfaces[i].material, vd > 0 else { return nil }
            let k = unit == .percent ? 100 / vd : vd / 100
            out.minus = Self.tidy(minus * k)
            out.plus = Self.tidy(plus * k)
            return out
        case .radius:
            guard case .success(let (m, p)) = engineLimits(in: c), let curv = c.system.value(at: .curvature(surface: surface)) else { return nil }
            switch unit {
            case .fringes:
                guard let h = c.semiDiameter(surface), h > 0 else { return nil }
                let k = (h * h) / (c.testWavelengthUm * 1e-3)
                out.minus = m * k
                out.plus = p * k
            case .radiusMM, .percent:
                guard curv != 0, curv + p != 0, curv - m != 0 else { return nil }
                let r = 1 / curv
                // c in [c − m, c + p]: R in [1/(c + p), 1/(c − m)] (same sign as R).
                let r1 = 1 / (curv + p), r2 = 1 / (curv - m)
                guard (r1 > 0) == (r > 0), (r2 > 0) == (r > 0) else { return nil }
                let dm = r - r1, dp = r2 - r
                let k = unit == .percent ? 100 / abs(r) : 1
                out.minus = max(dm, 0) * k
                out.plus = max(dp, 0) * k
            default: return nil
            }
            out.minus = Self.tidy(out.minus)
            out.plus = Self.tidy(out.plus)
            return out
        default:
            return out
        }
    }

    /// Rounds off conversion noise (9 significant digits).
    static func tidy(_ v: Double) -> Double {
        guard v != 0, v.isFinite else { return v }
        let q = pow(10, (log10(abs(v))).rounded(.down) - 8)
        return (v / q).rounded() * q
    }
}

// MARK: - Export

/// The result as text and CSV: a summary to paste into a report, the
/// sensitivity table and the Monte-Carlo trials.
public enum ToleranceReport {
    /// A plain-text summary: criterion, nominal, prediction, percentiles,
    /// yield, the largest contributors and the compensators.
    public static func summary(_ r: ToleranceResult, rows: [ToleranceRow], system: OpticalSystem, title: String) -> String {
        let d = r.display
        var lines = ["Tolerance analysis: \(title)", "Criterion: \(d.title)\(d.unit.isEmpty ? "" : " (\(d.unit))")"]
        if r.stopped { lines.append("Stopped early: statistics of the completed evaluations.") }
        if r.hasSensitivity {
            lines.append("Nominal: \(d.format(d.value(r.nominal)))")
            lines.append("Predicted (nominal + RSS): \(d.format(d.value(r.predicted)))")
        }
        if r.hasMonteCarlo {
            lines.append("Monte Carlo: \(r.trials) trials, seed \(r.seed)\(r.failures > 0 ? ", \(r.failures) failed" : "")")
            for p in ToleranceResult.levels {
                lines.append("  \(Fmt.short(p)) % of builds \(d.higherIsBetter ? "at least" : "at most"): \(d.format(d.value(r.percentile(p))))")
            }
            if let y = r.yield { lines.append("  Yield: \(Fmt.percent(y))") }
        }
        if r.hasSensitivity {
            lines.append("Largest contributors:")
            for s in r.sensitivities.prefix(5) {
                lines.append("  \(label(s, rows: rows, system: system)) \(limits(s, rows: rows)): \(Fmt.percent(r.share(s), digits: 0)) of the RSS")
            }
        }
        for (j, c) in r.compensators.enumerated() {
            let t = ToleranceCompensator(c.param, range: c.range).title(in: system)
            var s = "Compensator \(t)"
            if let (lo, hi) = r.compensatorRange(j) { s += ": moved \(Fmt.short(lo)) to +\(Fmt.short(hi)) \(c.param.unit(in: system))" }
            if let range = c.range { s += " of ±\(Fmt.short(range))" }
            if r.compensatorStd.indices.contains(j) { s += "; Monte Carlo σ \(Fmt.short(r.compensatorStd[j]))" }
            lines.append(s)
        }
        return lines.joined(separator: "\n")
    }

    /// One line per sensitivity operand, worst first.
    public static func sensitivityCSV(_ r: ToleranceResult, rows: [ToleranceRow], system: OpticalSystem) -> String {
        let d = r.display
        let n = name(d)
        var out = ["rank,tolerance,minus,plus,unit,parameter,criterion_minus,criterion_plus,\(n)_minus,\(n)_plus,change_minus,change_plus,share"]
        for (k, s) in r.sensitivities.enumerated() {
            let row = rows.indices.contains(s.id) ? rows[s.id] : nil
            let fields: [String] = [
                "\(k + 1)", csv(label(s, rows: rows, system: system)), row.map { num($0.minus) } ?? num(s.minus), row.map { num($0.plus) } ?? num(s.plus),
                csv(row?.unit.label ?? ""), csv(s.param.map { $0.label(in: system) } ?? s.label), num(s.criterionMinus), num(s.criterionPlus),
                num(d.value(s.criterionMinus)), num(d.value(s.criterionPlus)), num(d.value(s.criterionMinus) - d.value(r.nominal)),
                num(d.value(s.criterionPlus) - d.value(r.nominal)), num(r.share(s)),
            ]
            out.append(fields.joined(separator: ","))
        }
        return out.joined(separator: "\n") + "\n"
    }

    /// One line per Monte-Carlo trial (completed ones, in order).
    public static func trialsCSV(_ r: ToleranceResult) -> String {
        let d = r.display
        var out = ["trial,criterion,\(name(d))"]
        for (k, c) in r.criteria.enumerated() { out.append("\(k + 1),\(num(c)),\(num(d.value(c)))") }
        return out.joined(separator: "\n") + "\n"
    }

    static func name(_ d: CriterionDisplay) -> String {
        switch d {
        case .spot: "rms_spot_um"
        case .wavefront: "rms_wavefront_waves"
        case .mtf: "mtf"
        case .merit: "merit"
        }
    }

    static func label(_ s: ToleranceResult.Sensitivity, rows: [ToleranceRow], system: OpticalSystem) -> String {
        rows.indices.contains(s.id) ? rows[s.id].label(in: system) : s.label
    }

    static func limits(_ s: ToleranceResult.Sensitivity, rows: [ToleranceRow]) -> String {
        rows.indices.contains(s.id) ? rows[s.id].limitsText : "±" + Fmt.short(max(s.minus, s.plus))
    }

    static func num(_ v: Double) -> String { v.isFinite ? String(format: "%.10g", v) : "" }

    static func csv(_ s: String) -> String {
        s.contains(",") || s.contains("\"") ? "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\"" : s
    }
}
