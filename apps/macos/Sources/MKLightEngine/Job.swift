import Foundation
import MKLightModel
import Synchronization

/// Progress of a running job: a monotone fraction in 0…1 and a stage label
/// (mirror of the engine's `Progress`).
public struct JobProgress: Sendable, Hashable {
    public var fraction: Double
    public var stage: String
    public init(fraction: Double, stage: String) {
        self.fraction = fraction
        self.stage = stage
    }
}

/// Handle to one asynchronous analysis job.
///
/// The job runs independently of whoever awaits it: cancelling the awaiting
/// task does not cancel the job, only `cancel()` does. That lets a view drop
/// interest in a result without killing a job another view still needs.
public final class AnalysisJob: Sendable, Identifiable {
    public let id = UUID()
    /// The typed request, for jobs started with `submit(_: AnalysisRequest, …)`.
    public let request: AnalysisRequest?
    /// The request by name, for jobs started with `submit(_: EngineAnalysisRequest, …)`.
    public let engineRequest: EngineAnalysisRequest?
    public let snapshotHash: SnapshotHash
    /// Progress updates. The stream finishes when the job ends. It has a single
    /// consumer (iterate it from one task).
    public let progress: AsyncStream<JobProgress>
    private let task: Task<AnalysisOutput, any Error>
    private let onCancel: @Sendable () -> Void
    private let onStop: (@Sendable () -> Bool)?

    public init(
        request: AnalysisRequest?, engineRequest: EngineAnalysisRequest? = nil, snapshotHash: SnapshotHash,
        progress: AsyncStream<JobProgress>, task: Task<AnalysisOutput, any Error>,
        onCancel: @escaping @Sendable () -> Void = {}, onStop: (@Sendable () -> Bool)? = nil
    ) {
        self.request = request
        self.engineRequest = engineRequest
        self.snapshotHash = snapshotHash
        self.progress = progress
        self.task = task
        self.onCancel = onCancel
        self.onStop = onStop
    }

    /// Whether the job can be **stopped**: ended early while keeping its
    /// best result so far (optimisation and tolerancing jobs on engines
    /// that support it). Distinct from `cancel()`, which discards it.
    public var canStop: Bool { onStop != nil }

    /// Asks the job to stop and return its best result so far. Returns
    /// whether the request reached a running job; the job then ends
    /// normally and `result()` returns the stopped result.
    @discardableResult
    public func stop() -> Bool { onStop?() ?? false }

    /// The analysis name (typed kind or engine name).
    public var analysis: String { request?.kind.rawValue ?? engineRequest?.analysis ?? "" }

    /// Waits for the result. Throws `EngineError.cancelled` if the job was
    /// cancelled.
    public func result() async throws -> AnalysisOutput {
        do {
            return try await task.value
        } catch is CancellationError {
            throw EngineError.cancelled
        }
    }

    /// Requests cooperative cancellation. Idempotent.
    public func cancel() {
        onCancel()
        task.cancel()
    }

    /// Runs `operation` as a job on the cooperative pool. `operation` reports
    /// progress through the reporter and must check for cancellation
    /// (`try reporter.checkCancelled()`) at bounded intervals.
    public static func run(
        request: AnalysisRequest, snapshot: ModelSnapshot, priority: TaskPriority = .userInitiated,
        onCancel: @escaping @Sendable () -> Void = {},
        operation: @escaping @Sendable (JobReporter) async throws -> AnalysisOutput
    ) -> AnalysisJob {
        make(
            request: request, engineRequest: nil, snapshot: snapshot, priority: priority, onCancel: onCancel, onStop: nil,
            operation)
    }

    /// As `run(request:…)`, for a request by name.
    public static func run(
        engineRequest: EngineAnalysisRequest, snapshot: ModelSnapshot, priority: TaskPriority = .userInitiated,
        onCancel: @escaping @Sendable () -> Void = {}, onStop: (@Sendable () -> Bool)? = nil,
        operation: @escaping @Sendable (JobReporter) async throws -> AnalysisOutput
    ) -> AnalysisJob {
        make(
            request: nil, engineRequest: engineRequest, snapshot: snapshot, priority: priority, onCancel: onCancel,
            onStop: onStop, operation)
    }

    private static func make(
        request: AnalysisRequest?, engineRequest: EngineAnalysisRequest?, snapshot: ModelSnapshot,
        priority: TaskPriority, onCancel: @escaping @Sendable () -> Void, onStop: (@Sendable () -> Bool)?,
        _ operation: @escaping @Sendable (JobReporter) async throws -> AnalysisOutput
    ) -> AnalysisJob {
        let (stream, continuation) = AsyncStream.makeStream(of: JobProgress.self, bufferingPolicy: .bufferingNewest(8))
        let reporter = JobReporter(continuation: continuation)
        let task = Task.detached(priority: priority) {
            defer { reporter.finish() }
            reporter.report(0, stage: "Queued")
            let out = try await operation(reporter)
            try Task.checkCancellation()
            reporter.report(1, stage: "Done")
            return out
        }
        return AnalysisJob(
            request: request, engineRequest: engineRequest, snapshotHash: snapshot.hash, progress: stream, task: task,
            onCancel: onCancel, onStop: onStop)
    }
}

/// Written by a running job; forwards monotone progress to the job's stream.
public final class JobReporter: Sendable {
    private let state = Mutex(JobProgress(fraction: 0, stage: ""))
    private let continuation: AsyncStream<JobProgress>.Continuation

    public init(continuation: AsyncStream<JobProgress>.Continuation) {
        self.continuation = continuation
    }

    /// Reports `fraction` (clamped to 0…1, never decreasing) and an optional
    /// new stage label. Safe to call from concurrent child tasks: the update
    /// is yielded while the lock is held, so the stream sees the same order
    /// as the monotone state (yielding to a buffering stream never blocks).
    public func report(_ fraction: Double, stage: String? = nil) {
        state.withLock { p in
            let f = fraction.isFinite ? min(max(fraction, 0), 1) : 0
            p.fraction = max(p.fraction, f)
            if let stage, !stage.isEmpty { p.stage = stage }
            continuation.yield(p)
        }
    }

    public var current: JobProgress { state.withLock { $0 } }

    /// Throws `CancellationError` once the job has been cancelled.
    public func checkCancelled() throws { try Task.checkCancellation() }

    public func finish() { continuation.finish() }
}
