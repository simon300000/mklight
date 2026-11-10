import Foundation
import MKLightEngine
import MKLightModel

/// What the engine reports about where a run was computed, in words for
/// the run details: how an `auto` request was routed (the envelope's
/// `backend.cost_decision`, docs/RESULTS.md §5), the tier of a run that
/// was not design tier, and the backend attempts that did not compute the
/// result (`run_info.failed_attempts`).
public enum RunRoute {
    /// The route of an `auto` request in one line: the decision and, for a
    /// decision of the cost model, both predicted times. `nil` for an
    /// explicit request, for a run without a decision and for a decision
    /// this app does not know.
    public static func line(requested: String, decision: JSONValue?) -> String? {
        let asked = requested.hasPrefix("default:") ? String(requested.dropFirst("default:".count)) : requested
        guard asked == "auto", let d = decision?.object, let chosen = d["decision"]?.string else { return nil }
        let rule = d["rule"]?.string
        let line: String
        switch (chosen, rule) {
        case ("qualified_metal", "cost model"?):
            line = "GPU by predicted time" + predicted(d)
        case ("cpu_reference", "cost model"?):
            // The GPU is chosen only for a clear predicted advantage.
            if let (gpu, cpu) = times(d), gpu < cpu {
                line = "CPU: the GPU's predicted gain is too small" + predicted(d)
            } else {
                line = "CPU by predicted time" + predicted(d)
            }
        case ("cpu_reference", "accuracy exclusion"?):
            line = "CPU: this analysis is not accurate enough on the GPU"
        case ("cpu_reference", "no device share"?):
            line = "CPU: the GPU takes no part of a run of this size and shape"
        case ("cpu_reference", "known failure"?):
            line = "CPU: this request already failed on the GPU"
        case ("qualified_metal", nil):
            switch d["reason"]?.string {
            case "no cost calibration for this device": line = "GPU: no timing data for this device, so no prediction"
            case "no calibration for this analysis": line = "GPU: no timing data for this analysis, so no prediction"
            default: line = "GPU, without a prediction"
            }
        default:
            return nil
        }
        return d["repeated"]?.bool == true ? line + "; kept from the first identical request" : line
    }

    /// The tier of a run that was not design tier; `nil` for design tier.
    public static func tier(_ tier: Tier) -> String? {
        switch tier {
        case .design: nil
        case .preview: "Preview: cheaper sampling, not for final decisions"
        }
    }

    /// A backend attempt that did not compute the result: what happened,
    /// and the engine's reason.
    public struct Attempt: Equatable, Sendable, Identifiable {
        public var id: Int
        public var title: String
        public var reason: String
    }

    /// One entry per attempt, in the engine's order.
    public static func attempts(_ list: [BackendAttempt]) -> [Attempt] {
        list.enumerated().map { k, a in
            let name = backendName(a.backend)
            let took = a.elapsedMs > 0 ? duration(a.elapsedMs) : nil
            let title: String =
                switch a.outcome {
                case "failed": "\(name) failed" + (took.map { " after \($0)" } ?? "") + "; completed on the CPU"
                case "skipped": "\(name) not tried: the same input already failed on it"
                case "reference_completion": "\(name) path completed the result on the CPU" + (took.map { " (\($0))" } ?? "")
                default: "\(name) not used (\(a.outcome.replacingOccurrences(of: "_", with: " ")))"
                }
            return Attempt(id: k, title: title, reason: a.reason)
        }
    }

    /// `gpu:metal` → `GPU`; other ids as the engine names them.
    static func backendName(_ id: String) -> String {
        if id.hasPrefix("gpu:") { return "GPU" }
        return id == "cpu_reference" ? "CPU reference" : id
    }

    /// Both predicted times (GPU, CPU) in ms, when the decision has them.
    private static func times(_ d: [String: JSONValue]) -> (gpu: Double, cpu: Double)? {
        guard let p = d["predicted_ms"], let gpu = p["metal"]?.nf, let cpu = p["cpu"]?.nf,
            gpu.isFinite, cpu.isFinite, gpu >= 0, cpu >= 0
        else { return nil }
        return (gpu, cpu)
    }

    /// ` (GPU predicted 12 ms, CPU 31 ms)`, or nothing without both times.
    private static func predicted(_ d: [String: JSONValue]) -> String {
        guard let (gpu, cpu) = times(d) else { return "" }
        return " (GPU predicted \(duration(gpu)), CPU \(duration(cpu)))"
    }

    /// `Fmt.duration`, with a time that rounds to nothing as `<1 ms`.
    private static func duration(_ ms: Double) -> String {
        ms < 0.5 ? "<1 ms" : Fmt.duration(ms: ms)
    }
}
