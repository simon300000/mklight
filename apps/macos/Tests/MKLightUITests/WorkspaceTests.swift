import Foundation
import Synchronization
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// Polls `cond` on the main actor until it holds or the time runs out.
@MainActor
func waitUntil(_ timeout: Duration = .seconds(5), _ cond: () -> Bool) async throws {
    let clock = ContinuousClock()
    let end = clock.now + timeout
    while !cond() {
        guard clock.now < end else {
            Issue.record("condition not met within \(timeout)")
            return
        }
        try await Task.sleep(for: .milliseconds(5))
    }
}

/// An engine whose document sessions each have their own glass library, as
/// the real one does. Analyses by name finish at once with the session's
/// library hash, except `forever`, which runs until cancelled. It logs
/// catalog loads and snapshots so tests can check their order.
final class LibraryEngine: EngineClient, @unchecked Sendable {
    final class Shared: Sendable {
        let log = Mutex<[String]>([])
        let sessions = Mutex(0)
        let jobs = Mutex<[AnalysisJob]>([])
        /// Every submission: analysis name and whether it asked for a fresh run.
        let submissions = Mutex<[(analysis: String, fresh: Bool)]>([])
        func append(_ s: String) { log.withLock { $0.append(s) } }
    }

    let shared: Shared
    let session: Int
    let loadDelay: Duration
    private let loaded = Mutex<[String]>([])
    private let mock = MockEngine(simulatedLatency: .zero)

    init(shared: Shared = Shared(), session: Int = 0, loadDelay: Duration = .zero) {
        self.shared = shared
        self.session = session
        self.loadDelay = loadDelay
    }

    var info: EngineInfo { EngineInfo(name: "library engine", version: "1", isMock: false) }

    func makeDocumentEngine() -> any EngineClient {
        let n = shared.sessions.withLock { n in
            n += 1
            return n
        }
        return LibraryEngine(shared: shared, session: n, loadDelay: loadDelay)
    }

    var libraryHash: String { "lib:" + loaded.withLock { $0.joined(separator: ",") } }

    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot {
        let s = try await mock.openSnapshot(system)
        shared.append("snapshot \(session) \(libraryHash)")
        return ModelSnapshot(system: s.system, hash: s.hash, libraryHash: libraryHash)
    }

    func materialNames() async -> [String] { ["N-BK7", "N-SK16"] }

    func analysisDescriptors() async -> [AnalysisDescriptor] {
        ["first_order", "layout", "validate", "forever"].map {
            AnalysisDescriptor(name: $0, title: $0, category: "geometric", available: true)
        }
    }

    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        shared.submissions.withLock { $0.append((request.analysis, request.fresh)) }
        let lib = snapshot.libraryHash
        let job = AnalysisJob.run(engineRequest: request, snapshot: snapshot) { reporter in
            while request.analysis == "forever" {
                try reporter.checkCancelled()
                try await Task.sleep(for: .milliseconds(5))
            }
            let info = RunInfo(
                engineVersion: "1", snapshotHash: snapshot.hash, analysis: request.analysis, settings: .object(request.settings),
                backend: "cpu_reference", precision: .f64Reference, seed: nil, startedUnixMs: 0, durationMs: 0)
            return AnalysisOutput(runInfo: info, json: .object([:]), libraryHash: lib, engineRequest: request)
        }
        if request.analysis == "forever" { shared.jobs.withLock { $0.append(job) } }
        return job
    }

    func catalogs() async -> [CatalogInfo] {
        [CatalogInfo(name: "BUILTIN", source: "builtin", path: nil, glassCount: 13)]
            + loaded.withLock { $0 }.map { CatalogInfo(name: $0, source: "user", path: nil, glassCount: 1, inProject: true) }
    }

    func loadCatalog(at url: URL) async throws -> CatalogInfo {
        let name = url.deletingPathExtension().lastPathComponent.uppercased()
        shared.append("load \(session) \(name)")
        if loadDelay > .zero { try await Task.sleep(for: loadDelay) }
        loaded.withLock { list in if !list.contains(name) { list.append(name) } }
        return CatalogInfo(name: name, source: "user", path: url.path, glassCount: 1, inProject: true)
    }
}

@Suite @MainActor struct WorkspaceLifecycleTests {
    private func catalogDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-catalogs-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in ["A", "B"] { try Data().write(to: dir.appendingPathComponent("\(name).agf")) }
        return dir
    }

    private func ref(_ name: String) -> CatalogReference { CatalogReference(name: name, path: nil, relativePath: "\(name).agf") }

    /// On open, the project's catalogs reach the session before any
    /// snapshot, even when the design changes while they load.
    @Test func projectCatalogsLoadBeforeTheFirstSnapshot() async throws {
        let dir = try catalogDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engine = LibraryEngine(loadDelay: .milliseconds(100))
        let w = Workspace(engine: engine)
        await w.loadEngineMetadata()
        var edited = SampleSystems.singlet()
        let opening = Task { await w.open(system: SampleSystems.singlet(), catalogs: [ref("A")], projectDirectory: dir) }
        try await waitUntil { engine.shared.log.withLock { !$0.isEmpty } }
        edited.surfaces[1].thickness += 1
        w.update(system: edited, immediately: true)
        await opening.value
        try await waitUntil { w.snapshot?.system == edited && !w.isBusy && w.summary.output != nil }
        let log = engine.shared.log.withLock { $0 }
        #expect(log.first == "load 1 A")
        #expect(log.dropFirst().allSatisfy { $0 == "snapshot 1 lib:A" }, "\(log)")
        #expect(w.summary.output?.libraryHash == "lib:A")
    }

    /// Undoing Load Glass Catalog removes the reference; the session is
    /// rebuilt without it and results move to the new library.
    /// A run the user asks for (Try Again, Run, ⌘R, a job's Try Again)
    /// asks the engine to recompute; automatic refreshes may reuse results.
    @Test func manualRetriesAskForAFreshRun() async throws {
        let dir = try catalogDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engine = LibraryEngine()
        let w = Workspace(engine: engine)
        await w.loadEngineMetadata()
        await w.open(system: SampleSystems.singlet(), catalogs: [], projectDirectory: dir)
        try await waitUntil { w.summary.output != nil && !w.isBusy }
        let last = { engine.shared.submissions.withLock { $0.last } }
        #expect(engine.shared.submissions.withLock { $0.allSatisfy { !$0.fresh } }, "automatic runs reuse results")
        w.summary.rerun(fresh: true)
        #expect(last()?.analysis == w.summary.analysis && last()?.fresh == true)
        w.summary.rerun()
        #expect(last()?.fresh == false)
        _ = try w.submitJob("forever", settings: [:], fresh: true)
        #expect(last()?.analysis == "forever" && last()?.fresh == true)
        engine.shared.jobs.withLock { $0.forEach { $0.cancel() } }
    }

    @Test func catalogListChangesRebuildTheSession() async throws {
        let dir = try catalogDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engine = LibraryEngine()
        let w = Workspace(engine: engine)
        await w.loadEngineMetadata()
        await w.open(system: SampleSystems.singlet(), catalogs: [ref("A")], projectDirectory: dir)
        try await waitUntil { w.summary.output?.libraryHash == "lib:A" }
        _ = try await w.loadCatalog(at: dir.appendingPathComponent("B.agf"))
        #expect(await w.syncCatalogs([ref("A"), ref("B")], projectDirectory: dir) == false)
        try await waitUntil { w.summary.output?.libraryHash == "lib:A,B" }
        #expect(await w.syncCatalogs([ref("A")], projectDirectory: dir))
        #expect(engine.shared.sessions.withLock { $0 } == 2, "a new session")
        try await waitUntil { w.summary.output?.libraryHash == "lib:A" && !w.isBusy }
        #expect(!w.summary.isStale)
        #expect(await w.syncCatalogs([ref("A"), ref("C")], projectDirectory: dir))
        #expect(w.catalogWarnings.count == 1 && w.catalogWarnings[0].contains("“C”"))
    }

    /// A catalog change while a candidate is previewed takes new snapshots
    /// of both: the design stays the document's and the preview stays on.
    @Test func catalogChangeDuringAPreviewKeepsTheDesign() async throws {
        let dir = try catalogDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let w = Workspace(engine: LibraryEngine())
        await w.loadEngineMetadata()
        let design = SampleSystems.singlet()
        await w.open(system: design, catalogs: [ref("A")], projectDirectory: dir)
        try await waitUntil { w.summary.output?.libraryHash == "lib:A" }
        var candidate = design
        candidate.surfaces[1].thickness += 1
        w.preview(candidate, title: "Candidate 1")
        try await waitUntil { w.snapshot?.system == candidate }
        #expect(await w.syncCatalogs([ref("A"), ref("B")], projectDirectory: dir))
        try await waitUntil { w.snapshot?.libraryHash == "lib:A,B" && w.designSnapshot?.libraryHash == "lib:A,B" }
        #expect(w.isPreviewing && w.previewTitle == "Candidate 1")
        #expect(w.snapshot?.system == candidate)
        #expect(w.designSnapshot?.system == design)
        w.preview(nil)
        try await waitUntil { w.snapshot?.system == design }
    }

    /// A window's jobs end with it.
    @Test func droppingTheWorkspaceCancelsItsJobs() async throws {
        let engine = LibraryEngine()
        var w: Workspace? = Workspace(engine: engine, panes: [AnalysisPane(analysis: "forever")])
        await w?.loadEngineMetadata()
        w?.update(system: SampleSystems.singlet(), immediately: true)
        try await waitUntil { engine.shared.jobs.withLock { $0.count } == 1 }
        let job = try #require(engine.shared.jobs.withLock { $0.first })
        let outcome = Mutex<(any Error)?>(nil)
        let watcher = Task.detached {
            do { _ = try await job.result() } catch { outcome.withLock { $0 = error } }
        }
        weak let weakWorkspace = w
        w = nil
        try await waitUntil { weakWorkspace == nil }
        try await waitUntil { outcome.withLock { $0 != nil } }
        #expect(outcome.withLock { $0 as? EngineError } == .cancelled)
        job.cancel()
        _ = await watcher.result
    }

    /// The workspace tracks running jobs only: a finished job and a removed
    /// pane's job leave (with their results), and a job submitted under a
    /// key that still has one cancels it.
    @Test func liveJobsHoldOnlyRunningJobs() async throws {
        let done = AnalysisPane(analysis: "first_order")
        let endless = AnalysisPane(analysis: "forever")
        let w = Workspace(engine: LibraryEngine(), panes: [done, endless])
        await w.loadEngineMetadata()
        w.update(system: SampleSystems.singlet(), immediately: true)
        try await waitUntil { done.output != nil && endless.isRunning && !w.summary.isRunning && !w.layout.isRunning && !w.validation.isRunning }
        try await waitUntil { w.liveJobs.keys == [endless.id] }
        let running = try #require(endless.job)
        w.removePane(endless)
        #expect(w.liveJobs.keys.isEmpty)
        await #expect(throws: EngineError.cancelled) { _ = try await running.result() }

        let first = try w.submitJob("forever", settings: [:], key: "k")
        let second = try w.submitJob("forever", settings: [:], key: "k")
        await #expect(throws: EngineError.cancelled) { _ = try await first.result() }
        #expect(w.liveJobs.keys == ["job:k"])
        second.cancel()
        try await waitUntil { w.liveJobs.keys.isEmpty }

        // A job of an earlier request that is set late does not replace the
        // job of a later one.
        let engine = LibraryEngine()
        let snapshot = try await engine.openSnapshot(SampleSystems.singlet())
        let later = engine.submit(EngineAnalysisRequest(analysis: "forever"), on: snapshot)
        let earlier = engine.submit(EngineAnalysisRequest(analysis: "forever"), on: snapshot)
        let jobs = LiveJobs()
        jobs.set("k", later, request: 2)
        jobs.set("k", earlier, request: 1)
        await #expect(throws: EngineError.cancelled) { _ = try await earlier.result() }
        #expect(jobs.keys == ["k"])
        jobs.remove("k")
        await #expect(throws: EngineError.cancelled) { _ = try await later.result() }
        #expect(jobs.keys.isEmpty)
    }

    @Test func cancelAllStopsRunningPanes() async throws {
        let w = Workspace(engine: LibraryEngine(), panes: [AnalysisPane(analysis: "forever")])
        await w.loadEngineMetadata()
        w.update(system: SampleSystems.singlet(), immediately: true)
        try await waitUntil { w.panes[0].isRunning }
        w.cancelAll()
        try await waitUntil { w.panes[0].phase == .cancelled }
        #expect(!w.isBusy)
    }
}

@Suite @MainActor struct WorkspaceTests {
    private func make(_ panes: [AnalysisPane], latency: Duration = .zero) async -> Workspace {
        let w = Workspace(engine: MockEngine(simulatedLatency: latency), panes: panes)
        w.debounce = .zero
        await w.loadEngineMetadata()
        return w
    }

    /// Panes run against the snapshot, read the engine JSON, and go stale
    /// when the design changes; automatic panes re-run, manual ones wait.
    @Test func panesRunAndTrackStaleness() async throws {
        let auto = AnalysisPane(analysis: "spot_diagram")
        let manual = AnalysisPane(analysis: "ray_fan", autoUpdate: false)
        let w = await make([auto, manual])
        var s = SampleSystems.cookeTriplet()
        w.update(system: s, immediately: true)
        try await waitUntil { auto.output != nil && manual.output != nil && !w.isBusy }
        #expect(!auto.isStale && !manual.isStale)
        let spot = try #require(SpotDiagramView(output: auto.output!, context: AnalysisContext(system: s)))
        #expect(spot.fields.count == 3 && spot.fields.allSatisfy { !$0.points.isEmpty })
        #expect(FanView(output: manual.output!, context: AnalysisContext(system: s)) != nil)
        s.surfaces[2].thickness += 0.5
        w.update(system: s, immediately: true)
        try await waitUntil { w.snapshot?.system == s && !w.isBusy }
        #expect(!auto.isStale, "automatic panes follow the design")
        #expect(manual.isStale, "manual panes keep their result, stale")
        manual.rerun()
        try await waitUntil { !manual.isStale && !manual.isRunning }
    }

    /// A settings change re-runs the pane; the engine's run info records it.
    @Test func settingsChangeReruns() async throws {
        let pane = AnalysisPane(analysis: "spot_diagram")
        let w = await make([pane])
        w.update(system: SampleSystems.cookeTriplet(), immediately: true)
        try await waitUntil { pane.output != nil && !pane.isRunning }
        pane.set("rings", .integer(3))
        #expect(pane.isStale)
        try await waitUntil { !pane.isRunning && !pane.isStale }
        #expect(pane.output?.runInfo.settings["rings"] == .integer(3))
    }

    /// A pane's request follows the pane's analysis and settings, the
    /// engine's descriptors and the window's backend.
    @Test func paneRequestFollowsItsInputs() async throws {
        let pane = AnalysisPane(analysis: "spot_diagram", settings: ["rings": .integer(4), "not_a_setting": .bool(true)])
        let w = Workspace(engine: MockEngine(simulatedLatency: .zero), panes: [pane])
        #expect(w.request(for: pane).settings["not_a_setting"] == .bool(true), "no schema yet: sent as it is")
        await w.loadEngineMetadata()
        #expect(w.request(for: pane).settings == ["rings": .integer(4)], "reduced to what the schema declares")
        #expect(w.request(for: pane) == w.request(for: pane))
        pane.set("rings", .integer(3))
        #expect(w.request(for: pane).settings == ["rings": .integer(3)])
        w.backend = .cpuReference
        #expect(w.request(for: pane).execution.backend == .cpuReference)
        pane.show("ray_fan", settings: ["samples": .integer(11)])
        #expect(w.request(for: pane) == EngineAnalysisRequest(
            analysis: "ray_fan", settings: ["samples": .integer(11)], execution: ExecutionSettings(backend: .cpuReference)))
    }

    @Test func unavailableAndUnknownAnalyses() async throws {
        let seidel = AnalysisPane(analysis: "seidel")
        let unknown = AnalysisPane(analysis: "no_such_analysis")
        let w = await make([seidel, unknown])
        w.update(system: SampleSystems.cookeTriplet(), immediately: true)
        try await waitUntil { w.snapshot != nil }
        if case .unavailable = seidel.phase {} else { Issue.record("seidel: \(seidel.phase)") }
        if case .unavailable(let why) = unknown.phase { #expect(why.contains("no_such_analysis")) } else { Issue.record("\(unknown.phase)") }
    }

    /// A design the engine refuses blocks every analysis and is reported.
    @Test func invalidDesignBlocksAnalyses() async throws {
        let pane = AnalysisPane(analysis: "first_order")
        let w = await make([pane])
        var s = SampleSystems.singlet()
        w.update(system: s, immediately: true)
        try await waitUntil { pane.output != nil }
        s.wavelengths.wavelengths = []
        w.update(system: s, immediately: true)
        try await waitUntil { !w.modelErrors.isEmpty }
        #expect(pane.isStale && w.issues.contains { $0.severity == .error })
    }

    /// Returning to a design the engine accepts (undo of the edit it
    /// refused) drops the refusal's errors, also when that design's
    /// validation is still current and does not run again.
    @Test func returningToAValidDesignDropsTheRefusal() async throws {
        let w = await make([])
        let valid = SampleSystems.singlet()
        w.update(system: valid, immediately: true)
        try await waitUntil { w.validation.output != nil && !w.isBusy }
        let before = w.issues
        #expect(!before.contains { $0.severity == .error })
        var refused = valid
        refused.wavelengths.wavelengths = []
        w.update(system: refused, immediately: true)
        try await waitUntil { !w.modelErrors.isEmpty }
        #expect(w.issues.contains { $0.severity == .error })
        w.update(system: valid, immediately: true)
        try await waitUntil { w.modelErrors.isEmpty }
        #expect(w.issues == before)
    }

    /// Glass numbers asked for while the design is refused are asked for
    /// again once the engine accepts it.
    @Test func glassNumbersAreAskedAgainAfterARefusedDesign() async throws {
        let w = await make([])
        let valid = SampleSystems.singlet()
        w.update(system: valid, immediately: true)
        try await waitUntil { w.snapshot != nil }
        var refused = valid
        refused.wavelengths.wavelengths = []
        w.update(system: refused, immediately: true)
        try await waitUntil { !w.modelErrors.isEmpty }
        let glass = MaterialRef.glass("N-BK7")
        #expect(w.glass(glass) == nil)
        w.update(system: valid, immediately: true)
        try await waitUntil { w.modelErrors.isEmpty }
        try await waitUntil { w.glass(glass) != nil }
        #expect(w.glass(glass)?.nd == 1.5168)
    }

    /// A newer request cancels the older job of the same pane.
    @Test func newerRequestCancelsOlderJob() async throws {
        let pane = AnalysisPane(analysis: "spot_diagram")
        let w = await make([pane], latency: .milliseconds(300))
        w.update(system: SampleSystems.cookeTriplet(), immediately: true)
        try await waitUntil { pane.isRunning }
        let first = try #require(pane.job)
        pane.set("rings", .integer(2))
        let second = try #require(pane.job)
        #expect(first !== second)
        await #expect(throws: EngineError.cancelled) { _ = try await first.result() }
        try await waitUntil(.seconds(10)) { pane.output != nil && !pane.isRunning }
        #expect(pane.output?.runInfo.settings["rings"] == .integer(2))
    }

    /// Validation issues come from the `validate` analysis, with the
    /// surface and field they concern.
    @Test func validationIssuesSelectTheirSubject() {
        let json = JSONValue.object([
            "issues": .array([
                .object(["severity": .string("warning"), "code": .string("chief_ray"), "surface": .null, "message": .string("chief ray of field 2 misses the image")]),
                .object(["severity": .string("error"), "code": .string("material"), "surface": .integer(3), "message": .string("unknown glass 'XX'")]),
            ])
        ])
        let issues = ValidationIssue.from(validate: json)
        #expect(issues.map(\.severity) == [.warning, .error])
        #expect(issues[0].field == 2 && issues[0].surface == nil)
        #expect(issues[1].surface == 3 && issues[1].code == "material")
    }

    /// The pane layout round-trips through the project's `views` section.
    @Test func layoutPersistsInTheProject() async throws {
        let w = await make([AnalysisPane(id: "p1", analysis: "spot_diagram", settings: ["rings": .integer(4)], autoUpdate: false)])
        let window = WindowState()
        window.viewportMode = .orbit3D
        let arrangement = PanelArrangement.standard(panes: ["p1"])
        arrangement.move(.lensData, to: .windowEdge(window: arrangement.main.id, edge: .bottom))
        let layout = WorkspaceLayout(workspace: w, window: window, arrangement: arrangement)
        let doc = DesignDocument(system: SampleSystems.singlet())
        doc.setViews(layout.views)
        let file = try ProjectFile.decode(doc.savedProject.encoded())
        let back = WorkspaceLayout(views: file.sections["views"] ?? .null)
        #expect(back == layout)
        #expect(back.panes.first?.settings["rings"] == .integer(4) && back.panes.first?.autoUpdate == false)
        let restored = WindowState()
        back.apply(to: restored)
        #expect(restored.viewportMode == .orbit3D)
        #expect(back.makeArrangement().json == arrangement.json)
        #expect(WorkspaceLayout.isSaved(file.sections["views"]!) && !WorkspaceLayout.isSaved(.array([])))
    }

    /// The engine's layout buffers become the viewport scene.
    @Test func layoutBuffersBecomeAScene() async throws {
        let w = await make([])
        w.update(system: SampleSystems.cookeTriplet(), immediately: true)
        try await waitUntil { w.layout.output != nil }
        let scene = try #require(EngineLayout.scene(from: w.layout.output))
        #expect(scene.polylines.contains { $0.kind == .ray } && scene.polylines.contains { $0.kind == .surfaceProfile })
        #expect(!scene.elements.isEmpty && scene.elements.allSatisfy { $0.front.count == $0.back.count })
        #expect(EngineLayout.autoSemiDiameters(scene).count >= 6)
    }

    /// Every analysis the mock offers shows in a dedicated or generic view
    /// without failing, and unknown JSON falls back to the generic view.
    @Test func resultViewsReadEngineJSON() {
        let out = AnalysisOutput(
            runInfo: RunInfo(
                engineVersion: "1", snapshotHash: SnapshotHash(hex: "00"), analysis: "spot_diagram", settings: .object([:]),
                backend: "cpu_reference", precision: .f64Reference, seed: nil, startedUnixMs: 0, durationMs: 0),
            json: .object(["unexpected": .bool(true)]))
        #expect(SpotDiagramView(output: out, context: AnalysisContext()) == nil)
        #expect(MTFView(output: out, context: AnalysisContext()) == nil)
        #expect(WavefrontView(output: out, context: AnalysisContext()) == nil)
        #expect(GenericResultView.scalars(out.json).map(\.0) == ["unexpected"])
    }
}
