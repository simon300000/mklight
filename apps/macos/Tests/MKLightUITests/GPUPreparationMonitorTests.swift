import Foundation
import Synchronization
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// An engine whose GPU preparation reads as preparing a number of times and
/// then as `last`.
private final class PreparingEngine: EngineClient, Sendable {
    private let mock = MockEngine(simulatedLatency: .zero)
    private let preparingReads: Mutex<Int>
    private let last: GPUPreparation

    init(preparingReads: Int, then last: GPUPreparation) {
        self.preparingReads = Mutex(preparingReads)
        self.last = last
    }

    var info: EngineInfo { mock.info }
    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot { try await mock.openSnapshot(system) }
    func analysisDescriptors() async -> [AnalysisDescriptor] { await mock.analysisDescriptors() }
    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob { mock.submit(request, on: snapshot) }
    func materialNames() async -> [String] { [] }

    func gpuPreparation() -> GPUPreparation {
        preparingReads.withLock { n in
            guard n > 0 else { return last }
            n -= 1
            return .preparing(device: "GPU", done: 1, total: 2)
        }
    }
}

@Suite @MainActor struct GPUPreparationMonitorTests {
    /// A preparation that never ends is followed up to the limit only.
    @Test func aStuckPreparationIsFollowedUpToTheLimit() async {
        let monitor = GPUPreparationMonitor()
        let clock = ContinuousClock(), begin = clock.now
        await monitor.follow(
            PreparingEngine(preparingReads: .max, then: .idle), limit: .milliseconds(60), interval: .milliseconds(5))
        #expect(clock.now - begin < .seconds(5))
        #expect(monitor.state.isPreparing && monitor.display == .hidden)
    }

    /// The end of a preparation is what the status shows, a failure too.
    @Test func theFinalStateIsShown() async {
        let failed = GPUPreparationMonitor()
        await failed.follow(
            PreparingEngine(preparingReads: 2, then: .failed(device: "", message: "no cache folder")), interval: .milliseconds(5))
        #expect(failed.state == .failed(device: "", message: "no cache folder"))
        #expect(failed.summary == "Compiled on first use (no cache folder)" && failed.display == .hidden)
        let ready = GPUPreparationMonitor()
        await ready.follow(
            PreparingEngine(preparingReads: 0, then: .ready(device: "GPU", kernels: 3, fromCache: 3, seconds: 0.01)),
            interval: .milliseconds(5))
        #expect(ready.summary == "Ready (3, loaded from cache)" && ready.display == .hidden)
        #expect(ready.note == nil && failed.note == nil)
    }

    /// Kernels that are ready but could not be saved or loaded: the status
    /// says ready, and a second line says what the next launch does.
    @Test func aKernelCacheWarningIsASecondLine() async {
        let monitor = GPUPreparationMonitor()
        #expect(monitor.note == nil)
        await monitor.follow(
            PreparingEngine(
                preparingReads: 0,
                then: .ready(device: "GPU", kernels: 205, fromCache: 0, seconds: 31.24, warning: "kernel archive: permission denied")),
            interval: .milliseconds(5))
        #expect(monitor.summary == "Ready (205, prepared in 31.2 s)")
        #expect(monitor.note == "Kernel cache unavailable: the next launch compiles again (kernel archive: permission denied)")
    }
}
