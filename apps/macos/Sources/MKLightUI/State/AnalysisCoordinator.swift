import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// Runs the visible analyses of one window against immutable snapshots.
///
/// Every model change is debounced, frozen into a snapshot by the engine,
/// and the visible analyses are re-run asynchronously. A newer request for an
/// analysis cancels the older job. Results stay on screen while a new one is
/// computed and are marked stale when they no longer match the current
/// snapshot and settings.
@MainActor
@Observable
public final class AnalysisCoordinator {
    public enum Phase: Equatable, Sendable {
        case idle
        case running(JobProgress)
        case finished
        case failed(String)
        case cancelled
    }

    public struct Slot: Sendable {
        public var output: AnalysisOutput?
        public var phase = Phase.idle
        var job: AnalysisJob?
        var request: AnalysisRequest?
    }

    public private(set) var engine: any EngineClient
    public private(set) var snapshot: ModelSnapshot?
    /// Errors from the latest validation; while non-empty, analyses are not
    /// run and every shown result is stale.
    public private(set) var validationErrors: [String] = []
    public private(set) var slots: [AnalysisKind: Slot] = [:]
    public var debounce: Duration = .milliseconds(150)

    private var latestSystem: OpticalSystem?
    private var snapshotTask: Task<Void, Never>?
    private var visible: Set<AnalysisKind> = []
    private var settings = AnalysisSettingsSet()

    public init(engine: any EngineClient) {
        self.engine = engine
    }

    public func attach(engine: any EngineClient) {
        guard engine.info != self.engine.info else { return }
        cancelAll()
        self.engine = engine
        slots = [:]
        snapshot = nil
        if let s = latestSystem { update(system: s, immediately: true) }
    }

    public func slot(_ kind: AnalysisKind) -> Slot { slots[kind] ?? Slot() }

    public func output(_ kind: AnalysisKind) -> AnalysisOutput? { slots[kind]?.output }

    /// Whether the shown result of `kind` is out of date.
    public func isStale(_ kind: AnalysisKind) -> Bool {
        guard let out = slots[kind]?.output else { return false }
        return !validationErrors.isEmpty || !out.runInfo.isCurrent(for: snapshot, request: settings.request(for: kind))
    }

    public var isBusy: Bool {
        slots.values.contains { if case .running = $0.phase { true } else { false } }
    }

    // MARK: Inputs

    /// Call on every model change. Snapshots are taken after `debounce`.
    public func update(system: OpticalSystem, immediately: Bool = false) {
        latestSystem = system
        snapshotTask?.cancel()
        let delay = immediately ? .zero : debounce
        snapshotTask = Task { [weak self] in
            if delay > .zero {
                try? await Task.sleep(for: delay)
                if Task.isCancelled { return }
            }
            await self?.openSnapshot(system)
        }
    }

    public func update(settings: AnalysisSettingsSet) {
        self.settings = settings
        refreshVisible()
    }

    public func setVisible(_ kinds: Set<AnalysisKind>) {
        visible = kinds
        refreshVisible()
    }

    /// Runs `kind` again even if its result is current.
    public func rerun(_ kind: AnalysisKind) {
        guard let snapshot else { return }
        start(kind, request: settings.request(for: kind), on: snapshot)
    }

    public func rerunVisible() {
        for k in visible { rerun(k) }
    }

    public func cancel(_ kind: AnalysisKind) {
        slots[kind]?.job?.cancel()
    }

    public func cancelAll() {
        for k in slots.keys { cancel(k) }
    }

    // MARK: Internals

    private func openSnapshot(_ system: OpticalSystem) async {
        do {
            let snap = try await engine.openSnapshot(system)
            guard system == latestSystem else { return }  // superseded by a newer edit
            snapshot = snap
            validationErrors = []
            refreshVisible()
        } catch let EngineError.invalidModel(issues) {
            guard system == latestSystem else { return }
            validationErrors = issues
        } catch {
            guard system == latestSystem else { return }
            validationErrors = [error.localizedDescription]
        }
    }

    private func refreshVisible() {
        guard let snapshot, validationErrors.isEmpty else { return }
        for kind in visible {
            let request = settings.request(for: kind)
            let slot = self.slot(kind)
            if let out = slot.output, out.runInfo.isCurrent(for: snapshot, request: request) { continue }
            if let job = slot.job, case .running = slot.phase, job.snapshotHash == snapshot.hash, slot.request == request {
                continue
            }
            start(kind, request: request, on: snapshot)
        }
    }

    private func start(_ kind: AnalysisKind, request: AnalysisRequest, on snapshot: ModelSnapshot) {
        slots[kind, default: Slot()].job?.cancel()
        let job = engine.submit(request, on: snapshot)
        slots[kind, default: Slot()].job = job
        slots[kind, default: Slot()].request = request
        slots[kind, default: Slot()].phase = .running(JobProgress(fraction: 0, stage: "Queued"))
        let id = job.id
        Task { [weak self] in
            for await p in job.progress {
                guard let self, self.slots[kind]?.job?.id == id else { return }
                self.slots[kind]?.phase = .running(p)
            }
        }
        Task { [weak self] in
            do {
                let out = try await job.result()
                guard let self, self.slots[kind]?.job?.id == id else { return }
                self.slots[kind]?.output = out
                self.slots[kind]?.phase = .finished
                self.slots[kind]?.job = nil
            } catch {
                guard let self, self.slots[kind]?.job?.id == id else { return }
                self.slots[kind]?.phase = (error as? EngineError) == .cancelled ? .cancelled : .failed(error.localizedDescription)
                self.slots[kind]?.job = nil
            }
        }
    }
}
