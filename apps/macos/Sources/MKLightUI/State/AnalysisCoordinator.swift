import Foundation
import MKLightEngine
import MKLightModel
import Observation
import Synchronization

/// Runs the visible analyses of one window against immutable snapshots.
///
/// Every model change is debounced, frozen into a snapshot by the engine,
/// and the visible analyses are re-run asynchronously. A newer request for an
/// analysis cancels the older job. Results stay on screen while a new one is
/// computed and are marked stale when they no longer match the current
/// snapshot (system **and** glass library) and settings.
///
/// The coordinator talks to a per-document engine
/// (`EngineClient.makeDocumentEngine()`), so user glass catalogs loaded in
/// one window do not affect another. Its catalogs follow the project's list
/// (`open`, `syncCatalogs`).
///
/// Jobs do not outlive their window: `cancelAll()` runs when the window
/// closes, and a coordinator that goes away cancels its jobs (engine jobs
/// survive their session otherwise and would occupy shared workers).
@MainActor
@Observable
public final class AnalysisCoordinator {
    public enum Phase: Equatable, Sendable {
        case idle
        case running(JobProgress)
        case finished
        case failed(String)
        /// The engine build does not offer the analysis (with its reason).
        case unavailable(String)
        case cancelled
    }

    public struct Slot: Sendable {
        public var output: AnalysisOutput?
        public var phase = Phase.idle
        var job: AnalysisJob?
        var request: AnalysisRequest?
        var engineRequest: EngineAnalysisRequest?
    }

    /// The engine every window shares (`attach`).
    public private(set) var engine: any EngineClient
    /// This window's session of it.
    public private(set) var documentEngine: any EngineClient
    public private(set) var snapshot: ModelSnapshot?
    /// Errors from the latest validation; while non-empty, analyses are not
    /// run and every shown result is stale.
    public private(set) var validationErrors: [String] = []
    public private(set) var slots: [AnalysisKind: Slot] = [:]
    /// The analysis requested by name (the "Engine" pane), if visible.
    public private(set) var engineSlot = Slot()
    /// Every analysis the engine offers, with settings schemas.
    public private(set) var descriptors: [AnalysisDescriptor] = []
    /// User glass catalogs of this document's session.
    public private(set) var catalogs: [CatalogInfo] = []
    /// Problems loading the project's catalogs (missing files, parse errors).
    public private(set) var catalogWarnings: [String] = []
    public var debounce: Duration = .milliseconds(150)

    private var latestSystem: OpticalSystem?
    private var snapshotTask: Task<Void, Never>?
    private var visible: Set<AnalysisKind> = []
    private var settings = AnalysisSettingsSet()
    private var engineRequest: EngineAnalysisRequest?
    /// While positive, model changes are recorded but not snapshotted (the
    /// session's catalogs are being loaded).
    private var snapshotsHeld = 0
    /// The current job of every slot, reachable from `deinit`.
    private let liveJobs = LiveJobs()

    public init(engine: any EngineClient) {
        self.engine = engine
        documentEngine = engine.makeDocumentEngine()
        Task { await self.loadEngineMetadata() }
    }

    deinit { liveJobs.cancelAll() }

    public func attach(engine: any EngineClient) {
        guard engine.info != self.engine.info else { return }
        cancelAll()
        self.engine = engine
        documentEngine = engine.makeDocumentEngine()
        slots = [:]
        engineSlot = Slot()
        snapshot = nil
        descriptors = []
        Task { await loadEngineMetadata() }
        if let s = latestSystem { update(system: s, immediately: true) }
    }

    public func slot(_ kind: AnalysisKind) -> Slot { slots[kind] ?? Slot() }

    public func output(_ kind: AnalysisKind) -> AnalysisOutput? { slots[kind]?.output }

    /// Whether the shown result of `kind` is out of date.
    public func isStale(_ kind: AnalysisKind) -> Bool {
        guard let out = slots[kind]?.output else { return false }
        return !validationErrors.isEmpty || !out.isCurrent(for: snapshot, request: settings.request(for: kind))
    }

    /// Whether the shown result of the request by name is out of date.
    public var isEngineResultStale: Bool {
        guard let out = engineSlot.output else { return false }
        guard let engineRequest else { return true }
        return !validationErrors.isEmpty || !out.isCurrent(for: snapshot, request: engineRequest)
    }

    public var isBusy: Bool {
        (slots.values.map(\.phase) + [engineSlot.phase]).contains { if case .running = $0 { true } else { false } }
    }

    public func descriptor(_ name: String) -> AnalysisDescriptor? { descriptors.first { $0.name == name } }

    // MARK: Inputs

    /// Call on every model change. Snapshots are taken after `debounce`.
    public func update(system: OpticalSystem, immediately: Bool = false) {
        latestSystem = system
        snapshotTask?.cancel()
        guard snapshotsHeld == 0 else { return }  // taken once the catalogs are loaded
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

    /// The analysis by name to keep current (`nil`: none is visible).
    public func setEngineRequest(_ request: EngineAnalysisRequest?) {
        guard request != engineRequest else { return }
        engineRequest = request
        if request == nil { engineSlot.job?.cancel() }
        refreshVisible()
    }

    /// Runs `kind` again even if its result is current.
    public func rerun(_ kind: AnalysisKind) {
        guard let snapshot else { return }
        start(kind, request: settings.request(for: kind), on: snapshot)
    }

    public func rerunEngineAnalysis() {
        guard let snapshot, let engineRequest else { return }
        startEngine(engineRequest, on: snapshot)
    }

    public func rerunVisible() {
        for k in visible { rerun(k) }
        rerunEngineAnalysis()
    }

    public func cancel(_ kind: AnalysisKind) {
        slots[kind]?.job?.cancel()
    }

    public func cancelEngineAnalysis() { engineSlot.job?.cancel() }

    public func cancelAll() {
        for k in slots.keys { cancel(k) }
        cancelEngineAnalysis()
    }

    // MARK: Glass catalogs

    /// Loads a user glass catalog into this document's session. The library
    /// hash changes, so every result becomes stale and visible ones re-run.
    public func loadCatalog(at url: URL) async throws -> CatalogInfo {
        let info = try await documentEngine.loadCatalog(at: url)
        await catalogsChanged()
        return info
    }

    /// The first snapshot of a document: the project's catalogs are loaded
    /// into the session **before** `system` is snapshotted, so no analysis
    /// runs against a library that lacks them.
    public func open(system: OpticalSystem, catalogs refs: [CatalogReference], projectDirectory: URL?) async {
        latestSystem = system
        if await !syncCatalogs(refs, projectDirectory: projectDirectory) {
            update(system: latestSystem ?? system, immediately: true)
        }
    }

    /// Makes the session's user catalogs match the project's list `refs`.
    /// Referenced catalogs the session lacks are loaded (trying each
    /// reference's relative path from `projectDirectory`, then its absolute
    /// path, as the engine does on open; missing ones become
    /// `catalogWarnings`). When the session holds a catalog the project no
    /// longer references (undo of "Load Glass Catalog"), the session is
    /// rebuilt, since the engine cannot unload a catalog, and `refs` loaded
    /// into it. Model changes wait meanwhile; the system is snapshotted
    /// again afterwards. Returns whether the library was touched.
    @discardableResult
    public func syncCatalogs(_ refs: [CatalogReference], projectDirectory: URL?) async -> Bool {
        let loaded = Set(await documentEngine.catalogs().filter { $0.source == "user" }.map(\.name))
        var toLoad = refs.filter { !loaded.contains($0.name) }
        if !loaded.isSubset(of: Set(refs.map(\.name))) {
            cancelAll()
            documentEngine = engine.makeDocumentEngine()
            toLoad = refs
        } else if toLoad.isEmpty {
            catalogWarnings = []
            return false
        }
        snapshotsHeld += 1
        catalogWarnings = await load(toLoad, projectDirectory: projectDirectory)
        catalogs = await documentEngine.catalogs()
        snapshotsHeld -= 1
        if snapshotsHeld == 0, let s = latestSystem { update(system: s, immediately: true) }
        return true
    }

    /// Loads `refs`; returns a warning for each one that could not be loaded.
    private func load(_ refs: [CatalogReference], projectDirectory: URL?) async -> [String] {
        var warnings: [String] = []
        for ref in refs {
            var loaded = false
            var lastError: String?
            for url in ref.candidates(projectDirectory: projectDirectory) where FileManager.default.fileExists(atPath: url.path) {
                do {
                    _ = try await documentEngine.loadCatalog(at: url)
                    loaded = true
                    break
                } catch {
                    lastError = error.localizedDescription
                }
            }
            if !loaded {
                warnings.append("glass catalog '\(ref.name)' could not be loaded (\(lastError ?? "file not found"))")
            }
        }
        return warnings
    }

    private func catalogsChanged() async {
        catalogs = await documentEngine.catalogs()
        if snapshotsHeld == 0, let s = latestSystem { await openSnapshot(s) }
    }

    // MARK: Internals

    /// Reads the analysis list and catalogs of the document engine.
    public func loadEngineMetadata() async {
        let engine = documentEngine
        descriptors = await engine.analysisDescriptors()
        catalogs = await engine.catalogs()
    }

    private func openSnapshot(_ system: OpticalSystem) async {
        do {
            let snap = try await documentEngine.openSnapshot(system)
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
            if let out = slot.output, out.isCurrent(for: snapshot, request: request) { continue }
            if slot.request == request, slot.job?.snapshotHash == snapshot.hash, case .running = slot.phase { continue }
            start(kind, request: request, on: snapshot)
        }
        if let engineRequest {
            if let out = engineSlot.output, out.isCurrent(for: snapshot, request: engineRequest) { return }
            if engineSlot.engineRequest == engineRequest, engineSlot.job?.snapshotHash == snapshot.hash,
                case .running = engineSlot.phase
            {
                return
            }
            startEngine(engineRequest, on: snapshot)
        }
    }

    private func start(_ kind: AnalysisKind, request: AnalysisRequest, on snapshot: ModelSnapshot) {
        slots[kind, default: Slot()].job?.cancel()
        let job = documentEngine.submit(request, on: snapshot)
        liveJobs.set(kind.rawValue, job)
        slots[kind, default: Slot()].job = job
        slots[kind, default: Slot()].request = request
        slots[kind, default: Slot()].phase = .running(JobProgress(fraction: 0, stage: "Queued"))
        observe(job) { [weak self] update in self?.slots[kind]?.apply(update, job: job.id) }
    }

    private func startEngine(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) {
        engineSlot.job?.cancel()
        let job = documentEngine.submit(request, on: snapshot)
        liveJobs.set(" engine", job)
        engineSlot.job = job
        engineSlot.engineRequest = request
        engineSlot.phase = .running(JobProgress(fraction: 0, stage: "Queued"))
        observe(job) { [weak self] update in self?.engineSlot.apply(update, job: job.id) }
    }

    /// Forwards a job's progress and outcome to the main actor.
    private func observe(_ job: AnalysisJob, _ apply: @escaping @MainActor (Slot.Update) -> Void) {
        Task { @MainActor in
            for await p in job.progress { apply(.progress(p)) }
        }
        Task { @MainActor in
            do {
                apply(.finished(try await job.result()))
            } catch let error as EngineError {
                switch error {
                case .cancelled: apply(.phase(.cancelled))
                case .unsupported(let why): apply(.phase(.unavailable(why)))
                default: apply(.phase(.failed(error.localizedDescription)))
                }
            } catch {
                apply(.phase(.failed(error.localizedDescription)))
            }
        }
    }
}

/// The latest job of each slot, so a coordinator's `deinit` (which cannot
/// touch main-actor state) can cancel them.
final class LiveJobs: Sendable {
    private let jobs = Mutex<[String: AnalysisJob]>([:])

    func set(_ slot: String, _ job: AnalysisJob) { jobs.withLock { $0[slot] = job } }

    func cancelAll() {
        for job in jobs.withLock({ Array($0.values) }) { job.cancel() }
    }
}

extension AnalysisCoordinator.Slot {
    enum Update {
        case progress(JobProgress)
        case finished(AnalysisOutput)
        case phase(AnalysisCoordinator.Phase)
    }

    /// Applies an update from job `id`; updates of a superseded job are
    /// ignored.
    mutating func apply(_ update: Update, job id: UUID) {
        guard job?.id == id else { return }
        switch update {
        case .progress(let p):
            if case .running = phase { phase = .running(p) }
        case .finished(let out):
            output = out
            phase = .finished
            job = nil
        case .phase(let p):
            phase = p
            job = nil
        }
    }
}
