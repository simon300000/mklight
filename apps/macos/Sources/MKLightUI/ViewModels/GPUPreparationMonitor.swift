import Foundation
import MKLightEngine
import Observation

/// Prepares the GPU's compute kernels when the app opens and follows the
/// engine's progress for the window status. Metal compiles each kernel once
/// (about 30 s for all of them on an M2 Pro after an install or update);
/// done at launch, analyses rarely wait for it. The compiled kernels are kept in the user's
/// Caches folder, so later launches load them in milliseconds.
@MainActor
@Observable
final class GPUPreparationMonitor {
    static let shared = GPUPreparationMonitor()

    /// What the window status shows.
    enum Display: Equatable {
        case hidden
        /// Preparing for longer than a glance: shown with its progress.
        case preparing
        /// Finished after being shown: a short confirmation.
        case finished
    }

    private(set) var state: GPUPreparation = .idle
    private(set) var display: Display = .hidden
    private var started = false

    /// Where compiled kernels are kept: the app's Caches folder (the system
    /// may purge it; the kernels are then compiled again).
    static var cacheDirectory: URL? {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appending(path: Bundle.main.bundleIdentifier ?? "MKLight", directoryHint: .isDirectory)
            .appending(path: "GPU Kernels", directoryHint: .isDirectory)
    }

    /// Starts the preparation once per app run and follows it to the end.
    func start(engine: any EngineClient) {
        guard !started else { return }
        started = true
        // Compiling is work the user is waiting for: no App Nap while it
        // runs, even when the app is behind other windows.
        let activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated], reason: "Preparing the GPU's compute kernels")
        engine.prepareGPU(cacheDirectory: Self.cacheDirectory)
        Task {
            await follow(engine)
            ProcessInfo.processInfo.endActivity(activity)
        }
    }

    /// How long a preparation is followed. One that takes longer keeps
    /// running in the engine, but the window status no longer shows it and
    /// the app may nap again.
    static let followLimit: Duration = .seconds(600)

    func follow(_ engine: any EngineClient, limit: Duration = followLimit, interval: Duration = .milliseconds(200)) async {
        let begin = ContinuousClock.now
        while true {
            state = engine.gpuPreparation()
            guard state.isPreparing, begin.duration(to: .now) < limit else { break }
            // Loading from the cache takes milliseconds: no flash.
            if display == .hidden, begin.duration(to: .now) > .milliseconds(400) {
                display = .preparing
            }
            try? await Task.sleep(for: interval)
        }
        guard display == .preparing else { return }
        if case .ready = state {
            display = .finished
            try? await Task.sleep(for: .seconds(3))
        }
        display = .hidden
    }

    /// One line for the status details.
    var summary: String? {
        switch state {
        case .idle:
            nil
        case .preparing(_, let done, let total):
            total > 0 ? "Preparing (\(done) of \(total))" : "Preparing"
        case .ready(_, let kernels, let fromCache, let seconds, _):
            fromCache == kernels
                ? "Ready (\(kernels), loaded from cache)"
                : "Ready (\(kernels), prepared in \(String(format: "%.1f", seconds)) s)"
        case .unavailable(let reason):
            "No GPU: \(reason)"
        case .failed(_, let message):
            "Compiled on first use (\(message))"
        }
    }

    /// A second line for the status details: the kernels are ready, but
    /// the saved ones could not be read or written.
    var note: String? {
        guard case .ready(_, _, _, _, let warning?) = state else { return nil }
        return "Kernel cache unavailable: the next launch compiles again (\(warning))"
    }
}
