import Foundation
import MKLightEngine
import MKLightModel
import Observation
import Synchronization

/// A problem the engine (or the model mirror) found in the design.
public struct ValidationIssue: Identifiable, Hashable, Sendable {
    public enum Severity: String, Sendable, Comparable {
        case error, warning, info
        public static func < (a: Severity, b: Severity) -> Bool { a.rank < b.rank }
        var rank: Int { self == .error ? 0 : self == .warning ? 1 : 2 }
    }
    public var id: Int
    public var severity: Severity
    /// The engine's stable code (`docs/RESULTS.md` §6), empty if none.
    public var code: String
    public var message: String
    /// Surface index the issue concerns.
    public var surface: Int?
    /// Field index the issue concerns (read from the message for the
    /// field and chief-ray codes).
    public var field: Int?

    /// Issues of a `validate` result.
    static func from(validate json: JSONValue) -> [ValidationIssue] {
        (json["issues"]?.array ?? []).enumerated().map { k, i in
            let message = i["message"]?.string ?? ""
            return ValidationIssue(
                id: k, severity: Severity(rawValue: i["severity"]?.string ?? "") ?? .warning, code: i["code"]?.string ?? "",
                message: message, surface: i["surface"]?.int, field: fieldIndex(in: message))
        }
    }

    /// `field 2` / `field index 2` in a message → 2 (as written: system field index).
    static func fieldIndex(in message: String) -> Int? {
        let words = message.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        for (k, w) in words.enumerated() where w == "field" {
            for next in words.dropFirst(k + 1).prefix(2) { if let i = Int(next) { return i } }
        }
        return nil
    }
}

/// Runs a window's analyses against immutable engine snapshots.
///
/// Every design change is debounced and frozen into an engine snapshot
/// (system + glass library hash). Panes with automatic update re-run; the
/// others keep their result and show it as stale. A newer request cancels
/// the older job of the same pane. Results never run against the live
/// document: each job runs against the snapshot it was submitted with.
///
/// Besides the visible panes there are three internal ones: the layout
/// (viewport), first-order data (status bar) and validation (issues).
/// The workspace talks to a per-document engine session, so user glass
/// catalogs of one window do not affect another; jobs end with the window.
///
/// Platform-neutral: Foundation and Observation only.
@MainActor
@Observable
public final class Workspace {
    public private(set) var engine: any EngineClient
    public private(set) var documentEngine: any EngineClient
    public private(set) var descriptors: [AnalysisDescriptor] = []
    public private(set) var catalogs: [CatalogInfo] = []
    public private(set) var catalogWarnings: [String] = []
    /// Glass names for completion in the material column.
    public private(set) var materialNames: [String] = []
    /// n_d and V_d of the design's catalogue glasses with their offsets
    /// (the engine's `glass_info`), filled on request by `glass(_:)`.
    public private(set) var glassNumbers: [MaterialRef: GlassNumbers] = [:]
    @ObservationIgnored private var glassRequests: Set<MaterialRef> = []
    public private(set) var snapshot: ModelSnapshot?
    /// Why the engine refused the latest design (no snapshot, analyses wait).
    public private(set) var modelErrors: [String] = []
    /// The engine's validation of the current snapshot.
    public private(set) var issues: [ValidationIssue] = []
    /// Backend preference of every analysis of the window.
    public var backend = BackendPreference.auto {
        didSet { if backend != oldValue { refresh(forced: true) } }
    }

    public private(set) var panes: [AnalysisPane] = []
    /// The viewport's scene (`layout`).
    public let layout = AnalysisPane(id: "layout", analysis: "layout")
    /// First-order data for the status bar.
    public let summary = AnalysisPane(id: "summary", analysis: "first_order")
    let validation = AnalysisPane(id: "validation", analysis: "validate")
    /// Beam footprints on every surface for the 3-D viewport; runs only
    /// while `showFootprints` is on.
    public let footprints = AnalysisPane(
        id: "footprints", analysis: "footprint",
        settings: ["pattern": .string("hexapolar"), "rings": .integer(6), "wavelengths": .object(["type": .string("primary")])])
    public var showFootprints = false {
        didSet {
            guard showFootprints != oldValue else { return }
            if showFootprints { refreshPane(footprints, force: true) } else { footprints.job?.cancel() }
        }
    }
    /// The pane that analysis commands act on.
    public var focusedPaneID: String?

    public var debounce: Duration = .milliseconds(150)

    /// The system the next snapshot is taken of: the design, or the
    /// previewed candidate.
    private var latestSystem: OpticalSystem?
    /// The document's design (never a preview).
    private var designSystem: OpticalSystem?
    /// The latest snapshot of the document's design; design jobs run on it
    /// even while a candidate is previewed.
    public private(set) var designSnapshot: ModelSnapshot?
    /// A candidate design (an optimisation or glass substitution result)
    /// that the layout and the analyses show instead of the document's,
    /// until it is applied or the preview ends. Never edited.
    public private(set) var previewSystem: OpticalSystem?
    /// What is previewed, e.g. "Candidate 2".
    public private(set) var previewTitle: String?
    public var isPreviewing: Bool { previewSystem != nil }
    /// Results of the design just before a preview started, for before /
    /// after comparisons: the layout and every pane's output by pane id.
    public private(set) var baseline: [String: AnalysisOutput] = [:]
    private var snapshotTask: Task<Void, Never>?
    private var snapshotsHeld = 0
    @ObservationIgnored private let liveJobs = LiveJobs()

    public init(engine: any EngineClient, panes: [AnalysisPane] = []) {
        self.engine = engine
        documentEngine = engine.makeDocumentEngine()
        for p in [layout, summary, validation, footprints] + panes { p.workspace = self }
        self.panes = panes
        focusedPaneID = panes.first?.id
    }

    deinit { liveJobs.cancelAll() }

    /// Every pane the workspace keeps current.
    var allPanes: [AnalysisPane] { [validation, summary, layout, footprints] + panes }

    public func descriptor(_ name: String) -> AnalysisDescriptor? { descriptors.first { $0.name == name } }

    public var focusedPane: AnalysisPane? { panes.first { $0.id == focusedPaneID } ?? panes.first }

    /// Analyses a pane can show: everything except design jobs and
    /// internal ones, grouped by category in the engine's order.
    public var paneAnalyses: [AnalysisDescriptor] {
        descriptors.filter { !["optimization", "diagnostic"].contains($0.category) && $0.name != "layout" }
    }

    public var isBusy: Bool { allPanes.contains(where: \.isRunning) }

    /// The validation errors that block analyses: the engine's refusal, or
    /// error-severity issues.
    public var blockingErrors: [String] { modelErrors }

    // MARK: Engine

    /// Reads the analysis list, catalogs and glass names of the session.
    public func loadEngineMetadata() async {
        let engine = documentEngine
        descriptors = await engine.analysisDescriptors()
        catalogs = await engine.catalogs()
        materialNames = await engine.materialNames()
        migrateSavedPanes()
        for p in allPanes where p.output == nil { refreshPane(p) }
    }

    /// Brings panes saved by an older version up to the current starting
    /// settings where the user never changed them: an image simulation
    /// saved with the engine's defaults (which show no blur) takes the
    /// starting patch, PSF and pupil sampling, and waits for Update.
    private func migrateSavedPanes() {
        for p in panes where p.analysis == "image_simulation" && p.output == nil && !p.isRunning {
            if let m = ImageSimulationSettings.migrated(p.settings) {
                p.setSettings(m)
                continue
            }
            guard ImageSimulationSettings.isEngineDefault(p.settings, defaults: descriptor(p.analysis)?.defaultSettings) else { continue }
            p.autoUpdate = Self.autoUpdateDefault(p.analysis)
            p.setSettings(startingSettings(p.analysis))
        }
    }

    // MARK: Panes

    @discardableResult
    public func addPane(_ analysis: String, settings: [String: JSONValue] = [:], after: AnalysisPane? = nil) -> AnalysisPane {
        let pane = AnalysisPane(
            analysis: analysis, settings: settings.isEmpty ? startingSettings(analysis) : settings, autoUpdate: Self.autoUpdateDefault(analysis))
        pane.workspace = self
        if let after, let i = panes.firstIndex(where: { $0 === after }) {
            panes.insert(pane, at: i + 1)
        } else {
            panes.append(pane)
        }
        focusedPaneID = pane.id
        refreshPane(pane)
        return pane
    }

    public func removePane(_ pane: AnalysisPane) {
        pane.job?.cancel()
        panes.removeAll { $0 === pane }
        if focusedPaneID == pane.id { focusedPaneID = panes.first?.id }
    }

    public func movePane(_ pane: AnalysisPane, by offset: Int) {
        guard let i = panes.firstIndex(where: { $0 === pane }) else { return }
        let j = min(max(i + offset, 0), panes.count - 1)
        guard i != j else { return }
        panes.remove(at: i)
        panes.insert(pane, at: j)
    }

    /// Shows `analysis` in the focused pane (or a new one).
    public func open(_ analysis: String, inNewPane: Bool = false) {
        if !inNewPane, let p = panes.first(where: { $0.analysis == analysis }) {
            focusedPaneID = p.id
        } else if !inNewPane, let p = focusedPane {
            p.show(analysis, settings: startingSettings(analysis))
            p.autoUpdate = Self.autoUpdateDefault(analysis)
        } else {
            addPane(analysis, after: focusedPane)
        }
    }

    /// The settings a new pane of `analysis` starts with: the engine's
    /// defaults, except where they show nothing useful (image simulation).
    public func startingSettings(_ analysis: String) -> [String: JSONValue] {
        switch analysis {
        case "image_simulation": ImageSimulationSettings.starting(defaults: descriptor(analysis)?.defaultSettings)
        case "spot_diagram": Self.spotStartingSettings
        default: [:]
        }
    }

    /// A new spot diagram shows each field before, on and after focus: three
    /// planes over the engine's automatic range (the depth of focus or the
    /// geometric blur), so the bokeh is visible without any setting.
    nonisolated public static let spotStartingSettings: [String: JSONValue] = [
        "through_focus": .object(["min": .null, "max": .null, "steps": .integer(3)])
    ]

    /// Long-running analyses wait for an explicit re-run by default.
    static func autoUpdateDefault(_ analysis: String) -> Bool {
        !["stray_light", "ghost_irradiance", "tolerance", "optimize", "through_focus_mtf", "mtf_vs_field", "image_simulation"].contains(analysis)
    }

    func paneChanged(_ pane: AnalysisPane) { refreshPane(pane, force: true) }
    func settingsChanged(_ pane: AnalysisPane) { refreshPane(pane, force: true) }

    // MARK: Design changes

    /// Call on every design change. Snapshots are taken after `debounce`.
    public func update(system: OpticalSystem, immediately: Bool = false) {
        designSystem = system
        if previewSystem != nil {
            // A design change ends the preview: the panes follow the design again.
            previewSystem = nil
            previewTitle = nil
        }
        latestSystem = system
        snapshotTask?.cancel()
        guard snapshotsHeld == 0 else { return }
        let delay = immediately ? .zero : debounce
        snapshotTask = Task { [weak self] in
            if delay > .zero {
                try? await Task.sleep(for: delay)
                if Task.isCancelled { return }
            }
            await self?.openSnapshot(system)
        }
    }

    /// Shows `candidate` in the layout and the analyses instead of the
    /// design (`nil` ends the preview). The document does not change.
    public func preview(_ candidate: OpticalSystem?, title: String? = nil) {
        guard candidate != previewSystem || title != previewTitle else { return }
        if candidate != nil && previewSystem == nil {
            var b: [String: AnalysisOutput] = [:]
            for p in allPanes { if let o = p.output, !isStale(p) { b[p.id] = o } }
            baseline = b
        }
        previewSystem = candidate
        previewTitle = candidate == nil ? nil : title
        guard let target = candidate ?? designSystem else { return }
        latestSystem = target
        snapshotTask?.cancel()
        guard snapshotsHeld == 0 else { return }
        snapshotTask = Task { [weak self] in await self?.openSnapshot(target) }
    }

    /// Waits for the pending snapshot (tests, sheets that need it current).
    public func settle() async {
        await snapshotTask?.value
    }

    private func openSnapshot(_ system: OpticalSystem) async {
        do {
            let snap = try await documentEngine.openSnapshot(system)
            if system == designSystem { designSnapshot = snap }
            guard system == latestSystem else { return }
            snapshot = snap
            modelErrors = []
            refresh(forced: false)
        } catch let EngineError.invalidModel(messages) {
            guard system == latestSystem else { return }
            modelErrors = messages
            issues = messages.enumerated().map {
                ValidationIssue(id: $0.offset, severity: .error, code: "", message: $0.element, surface: nil, field: nil)
            }
        } catch {
            guard system == latestSystem else { return }
            modelErrors = [error.localizedDescription]
        }
    }

    // MARK: Running

    func request(for pane: AnalysisPane) -> EngineAnalysisRequest {
        var settings = ImageSourceImport.engineSettings(pane.analysis, pane.settings, defaults: descriptor(pane.analysis)?.defaultSettings)
        if let d = descriptor(pane.analysis), !d.settingsFields.isEmpty { settings = d.acceptedSettings(settings) }
        return EngineAnalysisRequest(analysis: pane.analysis, settings: settings, execution: ExecutionSettings(backend: backend))
    }

    /// Whether the pane's output no longer matches the design, the glass
    /// library or the pane's settings.
    public func isStale(_ pane: AnalysisPane) -> Bool {
        guard let out = pane.output else { return false }
        if !modelErrors.isEmpty { return true }
        return !out.isCurrent(for: snapshot, request: request(for: pane))
    }

    private func refresh(forced: Bool) {
        for p in allPanes { refreshPane(p, force: forced) }
    }

    /// Runs `pane` if it should be current and is not: automatic panes on
    /// every change, manual ones only when asked or never run.
    private func refreshPane(_ pane: AnalysisPane, force: Bool = false) {
        guard pane.workspace === self, snapshot != nil, modelErrors.isEmpty else { return }
        if pane === footprints && !showFootprints { return }
        let manualAndShown = !pane.autoUpdate && pane.output != nil && !force
        if manualAndShown { return }
        run(pane, force: false)
    }

    /// Submits the pane's request against the current snapshot, unless its
    /// output is current (or the same job is already running) and `force`
    /// is false.
    public func run(_ pane: AnalysisPane, force: Bool) {
        guard let snapshot, modelErrors.isEmpty else { return }
        let request = request(for: pane)
        if !force {
            if let out = pane.output, out.isCurrent(for: snapshot, request: request) { return }
            if pane.isRunning, pane.lastRequest == request, pane.job?.snapshotHash == snapshot.hash { return }
        }
        if let d = descriptor(pane.analysis), !d.available {
            pane.job?.cancel()
            pane.phase = .unavailable(d.unavailableReason ?? "not available in this engine build")
            return
        }
        if !descriptors.isEmpty, descriptor(pane.analysis) == nil {
            pane.phase = .unavailable("This engine has no analysis named “\(pane.analysis)”.")
            return
        }
        pane.job?.cancel()
        let job = documentEngine.submit(request, on: snapshot)
        liveJobs.set(pane.id, job)
        pane.job = job
        pane.lastRequest = request
        pane.phase = .running(JobProgress(fraction: 0, stage: "Queued"))
        observe(job, pane)
    }

    private func observe(_ job: AnalysisJob, _ pane: AnalysisPane) {
        Task { @MainActor [weak pane] in
            for await p in job.progress {
                guard let pane, pane.job === job, pane.isRunning else { continue }
                pane.phase = .running(p)
            }
        }
        Task { @MainActor [weak self, weak pane] in
            let outcome: Result<AnalysisOutput, any Error>
            do { outcome = .success(try await job.result()) } catch { outcome = .failure(error) }
            guard let pane, pane.job === job else { return }
            pane.job = nil
            switch outcome {
            case .success(let out):
                pane.output = out
                pane.phase = .finished
                if pane === self?.validation { self?.issues = ValidationIssue.from(validate: out.json) }
            case .failure(EngineError.cancelled):
                pane.phase = .cancelled
            case .failure(EngineError.unsupported(let why)):
                pane.phase = .unavailable(why)
            case .failure(let error):
                pane.phase = .failed(error.localizedDescription)
            }
        }
    }

    public func rerunVisible() {
        for p in allPanes where p !== footprints || showFootprints { run(p, force: true) }
    }

    public func cancelAll() {
        for p in allPanes { p.job?.cancel() }
    }

    // MARK: Glass catalogs

    /// The numbers of a catalogue glass with its offset, when the engine
    /// has given them; asks for them otherwise (the dictionary fills in
    /// later). `nil` for other materials.
    public func glass(_ m: MaterialRef) -> GlassNumbers? {
        guard case .catalog(let cat, let name, let o) = m else { return nil }
        if let g = glassNumbers[m] { return g }
        guard !glassRequests.contains(m), designSnapshot ?? snapshot != nil else { return nil }
        glassRequests.insert(m)
        var s: [String: JSONValue] = ["name": .string(name), "catalog": cat.map(JSONValue.string) ?? .null]
        if !o.isZero { s["offset"] = .object(["dn_d": .number(o.dnD), "dv_d": .number(o.dvD)]) }
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let job = try self.submitJob("glass_info", settings: s, key: "glass_info:\(name)")
                let out = try await job.result()
                if let g = GlassNumbers(json: out.json) { self.glassNumbers[m] = g }
            } catch {
                // Unknown to the engine: asked once.
            }
        }
        return nil
    }

    /// Forgets the glass numbers (the catalogs changed).
    private func resetGlassNumbers() {
        glassNumbers = [:]
        glassRequests = []
    }

    /// Loads a user glass catalog into this document's session.
    public func loadCatalog(at url: URL) async throws -> CatalogInfo {
        let info = try await documentEngine.loadCatalog(at: url)
        resetGlassNumbers()
        catalogs = await documentEngine.catalogs()
        materialNames = await documentEngine.materialNames()
        if snapshotsHeld == 0, let s = latestSystem { await openSnapshot(s) }
        return info
    }

    /// Imports a lens prescription file with this document's catalogs
    /// (after loading one that has the glasses a first import missed).
    public func importPrescription(at url: URL) async throws -> PrescriptionImport {
        try await documentEngine.importPrescription(at: url)
    }

    /// The first snapshot of a document: the project's catalogs are loaded
    /// before the system is snapshotted.
    public func open(system: OpticalSystem, catalogs refs: [CatalogReference], projectDirectory: URL?) async {
        latestSystem = system
        designSystem = system
        if await !syncCatalogs(refs, projectDirectory: projectDirectory) {
            update(system: latestSystem ?? system, immediately: true)
        }
    }

    /// Makes the session's user catalogs match the project's list. A
    /// catalog the project no longer references (undo of Load Glass
    /// Catalog) rebuilds the session, since the engine cannot unload one.
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
        var warnings: [String] = []
        for ref in toLoad {
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
            if !loaded { warnings.append("Glass catalog “\(ref.name)” could not be loaded (\(lastError ?? "file not found")).") }
        }
        catalogWarnings = warnings
        resetGlassNumbers()
        catalogs = await documentEngine.catalogs()
        materialNames = await documentEngine.materialNames()
        snapshotsHeld -= 1
        if snapshotsHeld == 0, let s = latestSystem { update(system: s, immediately: true) }
        return true
    }

    // MARK: Design jobs

    /// Starts a design job (`optimize`, `tolerance`, `solves`) on the
    /// current snapshot. It does not belong to a pane.
    public func submitJob(_ analysis: String, settings: [String: JSONValue], key: String? = nil) throws -> AnalysisJob {
        // Jobs run on the design, also while a candidate is previewed.
        guard let snapshot = designSnapshot ?? snapshot, modelErrors.isEmpty else {
            throw EngineError.invalidModel(modelErrors.isEmpty ? ["The design has not been checked by the engine yet."] : modelErrors)
        }
        if let d = descriptor(analysis), !d.available {
            throw EngineError.unsupported(d.unavailableReason ?? "\(analysis) is not available")
        }
        let job = documentEngine.submit(
            EngineAnalysisRequest(analysis: analysis, settings: settings, execution: ExecutionSettings(backend: backend)),
            on: snapshot)
        liveJobs.set("job:" + (key ?? analysis), job)
        return job
    }

    /// Starts a design job on `system`, a variant of the design (the
    /// tolerance run's surfaces get placements that return to the axis),
    /// on a snapshot of its own; the design's snapshot when they are equal.
    public func submitJob(_ analysis: String, settings: [String: JSONValue], on system: OpticalSystem, key: String? = nil) async throws -> AnalysisJob {
        guard let design = designSnapshot ?? snapshot, modelErrors.isEmpty else {
            throw EngineError.invalidModel(modelErrors.isEmpty ? ["The design has not been checked by the engine yet."] : modelErrors)
        }
        guard system != design.system else { return try submitJob(analysis, settings: settings, key: key) }
        if let d = descriptor(analysis), !d.available {
            throw EngineError.unsupported(d.unavailableReason ?? "\(analysis) is not available")
        }
        let snap = try await documentEngine.openSnapshot(system)
        let job = documentEngine.submit(
            EngineAnalysisRequest(analysis: analysis, settings: settings, execution: ExecutionSettings(backend: backend)), on: snap)
        liveJobs.set("job:" + (key ?? analysis), job)
        return job
    }

    /// Solves the current design in the engine (`applySolves`).
    public func applySolves(settings: [String: JSONValue] = [:]) async throws -> SolveApplication {
        guard let snapshot = designSnapshot ?? snapshot else { throw EngineError.invalidModel(modelErrors) }
        return try await documentEngine.applySolves(settings: settings, to: snapshot)
    }
}

/// The latest job of each pane, so `deinit` can cancel them.
final class LiveJobs: Sendable {
    private let jobs = Mutex<[String: AnalysisJob]>([:])
    func set(_ key: String, _ job: AnalysisJob) { jobs.withLock { $0[key] = job } }
    func cancelAll() {
        for job in jobs.withLock({ Array($0.values) }) { job.cancel() }
    }
}
