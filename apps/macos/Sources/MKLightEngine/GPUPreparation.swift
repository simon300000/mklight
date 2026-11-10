import Foundation

/// The engine's preparation of the GPU's compute kernels at launch
/// (`mk_engine_gpu_preparation_json`, docs/FFI.md "GPU kernel preparation"):
/// Metal compiles every kernel once, which takes seconds on a new install or
/// update; prepared ahead, no analysis waits for it.
public enum GPUPreparation: Equatable, Sendable, Decodable {
    /// Not started (or an engine without a GPU path, such as the mock).
    case idle
    /// Creating kernels: `done` of `total`.
    case preparing(device: String, done: Int, total: Int)
    /// Every kernel is ready; `fromCache` were loaded instead of compiled.
    /// `warning`: the saved kernels could not be read or written, so the
    /// next launch compiles again.
    case ready(device: String, kernels: Int, fromCache: Int, seconds: Double, warning: String? = nil)
    /// No GPU: analyses use the CPU reference.
    case unavailable(reason: String)
    /// Preparation failed; kernels compile when first used.
    case failed(device: String, message: String)

    /// Still running.
    public var isPreparing: Bool {
        if case .preparing = self { return true }
        return false
    }

    /// Fraction done while preparing (0 before the total is known).
    public var fraction: Double {
        guard case .preparing(_, let done, let total) = self, total > 0 else { return 0 }
        return Double(done) / Double(total)
    }

    private enum Keys: String, CodingKey {
        case state, device, done, total, kernels, from_cache, seconds, warning, reason, message
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        let device = { (try? c.decode(String.self, forKey: .device)) ?? "" }
        switch try c.decode(String.self, forKey: .state) {
        case "preparing":
            self = .preparing(device: device(), done: try c.decode(Int.self, forKey: .done),
                              total: try c.decode(Int.self, forKey: .total))
        case "ready":
            self = .ready(device: device(), kernels: try c.decode(Int.self, forKey: .kernels),
                          fromCache: try c.decode(Int.self, forKey: .from_cache),
                          seconds: try c.decode(Double.self, forKey: .seconds),
                          warning: try c.decodeIfPresent(String.self, forKey: .warning))
        case "unavailable":
            self = .unavailable(reason: try c.decode(String.self, forKey: .reason))
        case "failed":
            self = .failed(device: device(), message: try c.decode(String.self, forKey: .message))
        default:
            self = .idle
        }
    }
}
