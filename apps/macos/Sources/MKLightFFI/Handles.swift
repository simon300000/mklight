import Foundation
import MKLightEngine
import MKLightModel
import Synchronization

/// One job event from the engine's callback thread.
struct JobEvent: Sendable {
    var state: JobState
    var progress: Double
    var stage: String
}

/// Routes job events from the engine's single `mklight-events` thread to the
/// Swift job that owns the id.
///
/// Registration happens under the same lock that `deliver` takes, and
/// `submit` registers while still holding it, so an event can never arrive
/// for a job before its stream exists. Events of ids nobody registered
/// (synchronous helper jobs) are dropped. Streams buffer without limit: a
/// job produces at most one event per progress interval.
final class JobEventHub: Sendable {
    private let streams = Mutex<[UInt64: AsyncStream<JobEvent>.Continuation]>([:])

    /// Submits under the registration lock and returns the job id with its
    /// event stream. The stream finishes after the terminal event.
    func submit(_ body: () throws -> UInt64) throws -> (UInt64, AsyncStream<JobEvent>) {
        try streams.withLock { map in
            let id = try body()
            let (stream, continuation) = AsyncStream.makeStream(of: JobEvent.self, bufferingPolicy: .unbounded)
            map[id] = continuation
            return (id, stream)
        }
    }

    func deliver(_ id: UInt64, _ event: JobEvent) {
        let continuation = streams.withLock { map in
            event.state.isTerminal ? map.removeValue(forKey: id) : map[id]
        }
        continuation?.yield(event)
        if event.state.isTerminal { continuation?.finish() }
    }

    /// Forgets a job whose consumer went away.
    func remove(_ id: UInt64) {
        streams.withLock { $0.removeValue(forKey: id) }?.finish()
    }
}

/// Owns an `MKEngine*`: the engine handle, its job-event hub and the cached
/// analysis list. Thread-safe (the C engine is). `deinit` clears the job
/// callback (waiting for a call in progress) and destroys the engine; the
/// engine's workers stop once every session is gone too. Sessions and jobs
/// keep their core alive.
final class FFIEngineCore: @unchecked Sendable {
    let pointer: OpaquePointer
    let hub = JobEventHub()
    let version: String
    /// `mk_engine_list_analyses`, read once (the registry is fixed after
    /// creation).
    let descriptors: [AnalysisDescriptor]
    /// Why `mk_engine_prepare_gpu` refused to start, if it did.
    let gpuPreparationError = Mutex<String?>(nil)

    init(configuration: String?) throws {
        guard CABI.isLinked else { throw CABI.notLinked }
        let runtimeMajor = CABI.abiVersion >> 16
        guard runtimeMajor == CABI.compiledABIMajor else {
            throw EngineError.libraryUnavailable(
                "engine ABI major version \(runtimeMajor), app built for \(CABI.compiledABIMajor)")
        }
        pointer = try CABI.engineCreate(config: configuration)
        version = CABI.engineVersion
        do {
            try CABI.setJobCallback(pointer, hub: hub)
            let list = try CABI.listAnalyses(pointer)
            descriptors = try decodeJSON([AnalysisDescriptor].self, list)
        } catch {
            CABI.engineDestroy(pointer)
            throw error
        }
    }

    deinit {
        // mk_engine_destroy clears the callback first, so `hub` (released
        // after this) is never used by a late event.
        CABI.engineDestroy(pointer)
    }

    func descriptor(_ name: String) -> AnalysisDescriptor? { descriptors.first { $0.name == name } }
}

/// Owns an `MKSession*`. Calls on a session are serialised by the engine;
/// `locked` additionally makes a *sequence* of calls atomic (set the
/// snapshot's system, then submit against it), so concurrent submissions
/// from several tasks can never run a job on the wrong system.
final class FFISessionHandle: @unchecked Sendable {
    let pointer: OpaquePointer
    let core: FFIEngineCore
    private let lock = Mutex(())

    init(core: FFIEngineCore, pointer: OpaquePointer) {
        self.core = core
        self.pointer = pointer
    }

    convenience init(core: FFIEngineCore) throws {
        self.init(core: core, pointer: try CABI.sessionNew(core.pointer))
    }

    deinit { CABI.sessionDestroy(pointer) }

    func locked<T>(_ body: (OpaquePointer) throws -> T) throws -> T {
        try lock.withLock { _ in try body(pointer) }
    }
}

/// Runs blocking C calls on a global dispatch queue, off the main thread and
/// off Swift's cooperative pool, and resumes the caller with the result.
func offMain<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { (c: CheckedContinuation<T, any Error>) in
        DispatchQueue.global(qos: .userInitiated).async {
            c.resume(with: Result { try body() })
        }
    }
}
