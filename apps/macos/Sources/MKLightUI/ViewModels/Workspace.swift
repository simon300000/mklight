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
/// catalogs of one design do not affect another; jobs end with the design.
///
/// Platform-neutral: Foundation and Observation only.
@MainActor
@Observable
public final class Workspace {
    public private(set) var engine: any EngineClient
    public private(set) var documentEngine: any EngineClient
    public private(set) var descriptors: [AnalysisDescriptor] = [] {
        didSet { requestEpoch += 1 }
    }
    public private(set) var catalogs: [CatalogInfo] = []
    public private(set) var catalogWarnings: [String] = []
    /// Glass names for completion in the material column.
    public private(set) var materialNames: [String] = []
    /// n_d and V_d of the design's catalogue glasses with their offsets
    /// (the engine's `glass_info`), filled on request by `glass(_:)`.
    public private(set) var glassNumbers: [MaterialRef: GlassNumbers] = [:]
    /// The glasses asked for, each with the number of its request. A
    /// catalog change forgets them, and a request it forgot no longer
    /// counts when it answers.
    @ObservationIgnored private var glassRequests: [MaterialRef: Int] = [:]
    @ObservationIgnored private var glassRequestCount = 0
    public private(set) var snapshot: ModelSnapshot?
    /// Why the engine refused the latest design (no snapshot, analyses wait).
    public private(set) var modelErrors: [String] = []
    /// The problems of the latest design: the engine's refusal while it
    /// lasts, else the engine's validation of the current snapshot.
    public var issues: [ValidationIssue] { refusalIssues.isEmpty ? validationIssues : refusalIssues }
    /// The messages of the engine's refusal of the latest design, as issues.
    private var refusalIssues: [ValidationIssue] = []
    /// The issues of the latest `validate` result.
    private var validationIssues: [ValidationIssue] = []
    /// Backend preference of every analysis of the window.
    public var backend = BackendPreference.auto {
        didSet {
            guard backend != oldValue else { return }
            requestEpoch += 1
            refresh(forced: true)
        }
    }
    /// Counts the changes of what every pane's request is built from besides
    /// the pane itself (the descriptors and the backend), so a pane's stored
    /// request is rebuilt after one.
    @ObservationIgnored private var requestEpoch = 0

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
    /// Where each new pane goes: beside the pane it was added after (or
    /// the focused one), by pane id. The window arrangement reads it once.
    @ObservationIgnored public var placementHints: [String: String] = [:]

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
    @ObservationIgnored let liveJobs = LiveJobs()
    /// Counts the job requests of the workspace, in the order they were made.
    @ObservationIgnored private var jobRequests = 0

    public init(engine: any EngineClient, panes: [AnalysisPane] = []) {
        self.engine = engine
        documentEngine = engine.makeDocumentEngine()
        for p in [layout, summary, validation, footprints] + panes { p.workspace = self }
        self.panes = panes
        focusedPaneID = panes.first?.id
    }

    deinit { liveJobs.cancelAll() }

    /// The number of a new job request.
    private func nextJobRequest() -> Int {
        jobRequests += 1
        return jobRequests
    }

    /// Every pane the workspace keeps current.
    var allPanes: [AnalysisPane] { [validation, summary, layout, footprints] + panes }

    public func descriptor(_ name: String) -> AnalysisDescriptor? { descriptors.first { $0.name == name } }

    public var focusedPane: AnalysisPane? { panes.first { $0.id == focusedPaneID } ?? panes.first }

    /// Analyses a pane can show: everything except design jobs and
    /// internal ones, grouped by category in the engine's order.
    public var paneAnalyses: [AnalysisDescriptor] {
        // `ray_grid` is raw ray data for scripts (no plot): the spot diagram shows it;
        // `convert_spec` serves the System inspector's type pickers.
        descriptors.filter {
            !["optimization", "diagnostic"].contains($0.category) && !["layout", "ray_grid", "convert_spec"].contains($0.name)
        }
    }

    public var isBusy: Bool { allPanes.contains(where: \.isRunning) }

    // MARK: Engine

    /// Reads the analysis list, catalogs and glass names of the session.
    public func loadEngineMetadata() async {
        let engine = documentEngine
        descriptors = await engine.analysisDescriptors()
        catalogs = await engine.catalogs()
        materialNames = await engine.materialNames()
        // A Glass Data pane without a glass (saved while the design had
        // none) would only fail: it takes the design's first catalogue glass.
        for p in panes where p.analysis == "glass_info" && p.settings["name"]?.string == nil {
            p.setSettings(p.settings.merging(startingSettings(p.analysis)) { a, _ in a })
        }
        for p in allPanes where p.output == nil { refreshPane(p) }
    }

    // MARK: Panes

    @discardableResult
    public func addPane(_ analysis: String, settings: [String: JSONValue] = [:], after: AnalysisPane? = nil) -> AnalysisPane {
        let pane = AnalysisPane(
            analysis: analysis, settings: settings.isEmpty ? startingSettings(analysis) : settings, autoUpdate: Self.autoUpdateDefault(analysis))
        pane.workspace = self
        if let hint = after?.id ?? focusedPane?.id { placementHints[pane.id] = hint }
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
        liveJobs.remove(pane.id)
        panes.removeAll { $0 === pane }
        if focusedPaneID == pane.id { focusedPaneID = panes.first?.id }
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
    /// defaults, except where they show nothing useful (image simulation,
    /// a glass without its dispersion curve).
    public func startingSettings(_ analysis: String) -> [String: JSONValue] {
        baseStartingSettings(analysis).merging(Self.extraStartingSettings[analysis] ?? [:]) { a, _ in a }
    }

    /// Starting settings that only change engine defaults. `glass_info`
    /// samples its dispersion curve (the engine's default is none).
    nonisolated static let extraStartingSettings: [String: [String: JSONValue]] = [
        "glass_info": ["curve_samples": .integer(200)]
    ]

    private func baseStartingSettings(_ analysis: String) -> [String: JSONValue] {
        switch analysis {
        case "image_simulation": ImageSimulationSettings.starting(defaults: descriptor(analysis)?.defaultSettings)
        case "spot_diagram": Self.spotStartingSettings
        // The engine needs a glass: the design's first catalogue glass.
        case "glass_info": firstGlass.map { ["name": .string($0)] } ?? [:]
        default: [:]
        }
    }

    /// A new spot diagram shows each field before, on and after focus: three
    /// planes over the engine's automatic range (the depth of focus or the
    /// geometric blur), so the bokeh is visible without any setting.
    nonisolated public static let spotStartingSettings: [String: JSONValue] = [
        "through_focus": .object(["min": .null, "max": .null, "steps": .integer(3)])
    ]

    /// The first catalogue glass of the design, in surface order.
    private var firstGlass: String? {
        designSystem?.surfaces.lazy.compactMap { s -> String? in
            if case .catalog(_, let name, _) = s.material { return name }
            return nil
        }.first ?? materialNames.first
    }

    /// Long-running analyses wait for an explicit re-run by default.
    static func autoUpdateDefault(_ analysis: String) -> Bool {
        !["stray_light", "ghost_irradiance", "through_focus_mtf", "mtf_vs_field", "image_simulation"].contains(analysis)
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

    private func openSnapshot(_ system: OpticalSystem) async {
        do {
            let snap = try await documentEngine.openSnapshot(system)
            if system == designSystem { designSnapshot = snap }
            guard system == latestSystem else { return }
            snapshot = snap
            modelErrors = []
            refusalIssues = []
            refresh(forced: false)
        } catch let EngineError.invalidModel(messages) {
            guard system == latestSystem else { return }
            modelErrors = messages
            refusalIssues = messages.enumerated().map {
                ValidationIssue(id: $0.offset, severity: .error, code: "", message: $0.element, surface: nil, field: nil)
            }
        } catch {
            guard system == latestSystem else { return }
            modelErrors = [error.localizedDescription]
        }
    }

    // MARK: Running

    /// The pane's request: its settings as the engine takes them, with the
    /// window's backend. Built once per pane analysis, pane settings,
    /// descriptors and backend, and kept on the pane (views read
    /// `isStale` many times per update).
    func request(for pane: AnalysisPane) -> EngineAnalysisRequest {
        // Read on every call, so whoever observes the result follows them.
        let analysis = pane.analysis, paneSettings = pane.settings, backend = self.backend, descriptors = self.descriptors
        if let stored = pane.storedRequest, stored.epoch == requestEpoch { return stored.request }
        let d = descriptors.first { $0.name == analysis }
        var settings = paneSettings
        if let d, !d.settingsFields.isEmpty { settings = d.acceptedSettings(settings) }
        let request = EngineAnalysisRequest(analysis: analysis, settings: settings, execution: ExecutionSettings(backend: backend))
        pane.storedRequest = (requestEpoch, request)
        return request
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
        liveJobs.set(pane.id, job, request: nextJobRequest())
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
                if pane === self?.validation { self?.validationIssues = ValidationIssue.from(validate: out.json) }
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
        // A refused design runs no job: its glasses are asked for once the
        // engine accepts it.
        guard glassRequests[m] == nil, designSnapshot ?? snapshot != nil, modelErrors.isEmpty else { return nil }
        glassRequestCount += 1
        let request = glassRequestCount
        glassRequests[m] = request
        var s: [String: JSONValue] = ["name": .string(name), "catalog": cat.map(JSONValue.string) ?? .null]
        if !o.isZero { s["offset"] = .object(["dn_d": .number(o.dnD), "dv_d": .number(o.dvD)]) }
        // One job key per glass, catalog and offset: only a newer request
        // for the same glass replaces a running one.
        let key = "glass_info:\(cat ?? ""):\(name):\(o.dnD):\(o.dvD)"
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let job = try self.submitJob("glass_info", settings: s, key: key)
                let out = try await job.result()
                if self.glassRequests[m] == request, let g = GlassNumbers(json: out.json) { self.glassNumbers[m] = g }
            } catch EngineError.cancelled, EngineError.invalidModel {
                // Not answered (the job was cancelled, or the design was
                // refused before it started): asked again.
                if self.glassRequests[m] == request { self.glassRequests[m] = nil }
            } catch {
                // Unknown to the engine: asked once.
            }
        }
        return nil
    }

    /// Forgets the glass numbers (the catalogs changed).
    private func resetGlassNumbers() {
        glassNumbers = [:]
        glassRequests = [:]
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
        if snapshotsHeld == 0 { retakeSnapshots() }
        return true
    }

    /// Takes new snapshots after the session's glass library changed: of
    /// the design and, while a candidate is previewed, of the candidate too
    /// (the design stays the document's and the preview stays on).
    private func retakeSnapshots() {
        guard let design = designSystem else { return }
        let candidate = previewSystem
        snapshotTask?.cancel()
        snapshotTask = Task { [weak self] in
            await self?.openSnapshot(design)
            if let candidate, !Task.isCancelled { await self?.openSnapshot(candidate) }
        }
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
        liveJobs.set("job:" + (key ?? analysis), job, request: nextJobRequest())
        return job
    }

    /// Starts a design job on `system`, which may differ from the design (a
    /// tolerance preview rebuilds a build of the system its run used), on a
    /// snapshot of its own; the design's snapshot when they are equal.
    public func submitJob(_ analysis: String, settings: [String: JSONValue], on system: OpticalSystem, key: String? = nil) async throws -> AnalysisJob {
        // The request counts from here: its snapshot is taken first, and a
        // request made later may reach the engine before this one.
        let request = nextJobRequest()
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
        liveJobs.set("job:" + (key ?? analysis), job, request: request)
        return job
    }

    /// `system`'s fields and aperture in another field kind or aperture
    /// type (the engine's `convert_spec`); `nil` when the engine cannot
    /// answer (a refused design, an engine without the analysis).
    public func convertSpec(of system: OpticalSystem, fieldKind: FieldKind? = nil, aperture: ApertureSpec.Kind? = nil) async -> SpecConversion? {
        let settings: [String: JSONValue] = [
            "field_kind": fieldKind.map { .string($0.rawValue) } ?? .null,
            "aperture": aperture.map { .string($0.rawValue) } ?? .null,
        ]
        guard let job = try? await submitJob("convert_spec", settings: settings, on: system, key: "convert_spec"),
            let out = try? await job.result()
        else { return nil }
        return SpecConversion(json: out.json)
    }

    /// Solves the current design in the engine (`applySolves`).
    public func applySolves(settings: [String: JSONValue] = [:]) async throws -> SolveApplication {
        guard let snapshot = designSnapshot ?? snapshot else { throw EngineError.invalidModel(modelErrors) }
        return try await documentEngine.applySolves(settings: settings, to: snapshot)
    }
}

/// The running jobs of a workspace by key (a pane's id, or `job:` and a
/// design job's key), so `deinit` can cancel them. A job leaves when it ends
/// or its pane is removed, and with it its result and engine buffers. Each
/// key has one job, that of the latest request: the job of an earlier
/// request is cancelled, whichever of the two is set first.
final class LiveJobs: Sendable {
    private let jobs = Mutex<[String: (request: Int, job: AnalysisJob)]>([:])

    /// Tracks `job`, started for the workspace's `request`-th job request.
    func set(_ key: String, _ job: AnalysisJob, request: Int) {
        let earlier: AnalysisJob? = jobs.withLock { jobs in
            if let current = jobs[key], current.request > request { return job }
            return jobs.updateValue((request, job), forKey: key)?.job
        }
        earlier?.cancel()
        guard earlier !== job else { return }
        Task { [weak self] in
            _ = try? await job.result()
            self?.jobs.withLock { if $0[key]?.job === job { $0[key] = nil } }
        }
    }

    /// Cancels and forgets the job of `key`.
    func remove(_ key: String) {
        jobs.withLock { $0.removeValue(forKey: key) }?.job.cancel()
    }

    /// The keys with a running job.
    var keys: Set<String> { jobs.withLock { Set($0.keys) } }

    func cancelAll() {
        for job in jobs.withLock({ $0.values.map(\.job) }) { job.cancel() }
    }
}
