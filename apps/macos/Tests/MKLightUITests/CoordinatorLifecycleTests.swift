import Foundation
import Synchronization
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// An engine whose document sessions each have their own glass library, as
/// the real one does, and whose analyses by name run until cancelled. It
/// logs catalog loads and snapshots so tests can check their order.
final class LibraryEngine: EngineClient, @unchecked Sendable {
    final class Shared: Sendable {
        let log = Mutex<[String]>([])
        let sessions = Mutex(0)
        let jobs = Mutex<[AnalysisJob]>([])
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

    var info: EngineInfo { EngineInfo(name: "library engine", version: "1", backends: ["cpu_reference"], isMock: false) }

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

    func validate(_ system: OpticalSystem) async -> [ModelIssue] { [] }
    func materialNames() async -> [String] { [] }
    /// The mock's result, stamped with the snapshot's library.
    func submit(_ request: AnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let inner = mock.submit(request, on: snapshot)
        let library = snapshot.libraryHash
        return AnalysisJob.run(request: request, snapshot: snapshot, onCancel: { inner.cancel() }) { _ in
            var out = try await inner.result()
            out.libraryHash = library
            return out
        }
    }

    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let job = AnalysisJob.run(engineRequest: request, snapshot: snapshot) { reporter in
            while true {
                try reporter.checkCancelled()
                try await Task.sleep(for: .milliseconds(5))
            }
        }
        shared.jobs.withLock { $0.append(job) }
        return job
    }

    func catalogs() async -> [CatalogInfo] {
        [CatalogInfo(name: "BUILTIN", source: "builtin", path: nil, glassCount: 13)]
            + loaded.withLock { $0 }.map { CatalogInfo(name: $0, source: "user", path: nil, glassCount: 1) }
    }

    func loadCatalog(at url: URL) async throws -> CatalogInfo {
        let name = url.deletingPathExtension().lastPathComponent.uppercased()
        shared.append("load \(session) \(name)")
        if loadDelay > .zero { try await Task.sleep(for: loadDelay) }
        loaded.withLock { list in if !list.contains(name) { list.append(name) } }
        return CatalogInfo(name: name, source: "user", path: url.path, glassCount: 1)
    }
}

@Suite @MainActor struct CoordinatorLifecycleTests {
    /// A directory holding empty `A.agf` and `B.agf`.
    private func catalogDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-catalogs-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in ["A", "B"] { try Data().write(to: dir.appendingPathComponent("\(name).agf")) }
        return dir
    }

    private func ref(_ name: String) -> CatalogReference {
        CatalogReference(name: name, path: nil, relativePath: "\(name).agf")
    }

    /// On open, the project's catalogs reach the session before any
    /// snapshot, even when the model changes while they load: no analysis
    /// ever runs against a library without them.
    @Test func projectCatalogsLoadBeforeTheFirstSnapshot() async throws {
        let dir = try catalogDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engine = LibraryEngine(loadDelay: .milliseconds(100))
        let c = AnalysisCoordinator(engine: engine)
        c.setVisible([.firstOrder])
        var edited = SampleSystems.singlet()
        let opening = Task { await c.open(system: SampleSystems.singlet(), catalogs: [ref("A")], projectDirectory: dir) }
        try await waitUntil { engine.shared.log.withLock { !$0.isEmpty } }
        edited.surfaces[1].thickness += 1
        c.update(system: edited, immediately: true)  // an edit while the catalog loads
        await opening.value
        try await waitUntil { c.snapshot?.system == edited && !c.isBusy }
        let log = engine.shared.log.withLock { $0 }
        #expect(log.first == "load 1 A")
        #expect(log.dropFirst().allSatisfy { $0 == "snapshot 1 lib:A" }, "\(log)")
        #expect(c.snapshot?.libraryHash == "lib:A" && c.catalogWarnings.isEmpty)
        #expect(c.output(.firstOrder)?.libraryHash == "lib:A")
    }

    /// Undoing "Load Glass Catalog" removes the reference; the engine cannot
    /// unload a catalog, so the session is rebuilt with the remaining ones
    /// and results move to the new library. Redo loads it again.
    @Test func catalogListChangesRebuildTheSession() async throws {
        let dir = try catalogDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let engine = LibraryEngine()
        let c = AnalysisCoordinator(engine: engine)
        c.setVisible([.firstOrder])
        await c.open(system: SampleSystems.singlet(), catalogs: [ref("A")], projectDirectory: dir)
        try await waitUntil { c.output(.firstOrder)?.libraryHash == "lib:A" }
        // The user loads B; the project then records it: nothing to do.
        _ = try await c.loadCatalog(at: dir.appendingPathComponent("B.agf"))
        #expect(await c.syncCatalogs([ref("A"), ref("B")], projectDirectory: dir) == false)
        try await waitUntil { c.output(.firstOrder)?.libraryHash == "lib:A,B" }
        #expect(engine.shared.sessions.withLock { $0 } == 1)
        // Undo: B is no longer referenced.
        #expect(await c.syncCatalogs([ref("A")], projectDirectory: dir))
        #expect(engine.shared.sessions.withLock { $0 } == 2, "a new session")
        #expect(c.catalogs.map(\.name) == ["BUILTIN", "A"])
        try await waitUntil { c.output(.firstOrder)?.libraryHash == "lib:A" && !c.isBusy }
        #expect(c.snapshot?.libraryHash == "lib:A" && !c.isStale(.firstOrder))
        // Redo: B is loaded into the new session.
        #expect(await c.syncCatalogs([ref("A"), ref("B")], projectDirectory: dir))
        #expect(engine.shared.sessions.withLock { $0 } == 2)
        try await waitUntil { c.output(.firstOrder)?.libraryHash == "lib:A,B" }
        // A reference whose file is gone is a warning, retried on the next change.
        #expect(await c.syncCatalogs([ref("A"), ref("B"), ref("C")], projectDirectory: dir))
        #expect(c.catalogWarnings.count == 1 && c.catalogWarnings[0].contains("'C'"))
        #expect(await c.syncCatalogs([ref("A"), ref("B")], projectDirectory: dir) == false)
        #expect(c.catalogWarnings.isEmpty)
    }

    /// A window's jobs end with it: engine jobs outlive their session, so
    /// a coordinator that goes away cancels them.
    @Test func droppingTheCoordinatorCancelsItsJobs() async throws {
        let engine = LibraryEngine()
        var c: AnalysisCoordinator? = AnalysisCoordinator(engine: engine)
        c?.debounce = .zero
        c?.setEngineRequest(EngineAnalysisRequest(analysis: "forever"))
        c?.update(system: SampleSystems.singlet(), immediately: true)
        try await waitUntil { engine.shared.jobs.withLock { $0.count } == 1 && c?.isBusy == true }
        let job = try #require(engine.shared.jobs.withLock { $0.first })
        let outcome = Mutex<(any Error)?>(nil)
        let watcher = Task.detached {
            do { _ = try await job.result() } catch { outcome.withLock { $0 = error } }
        }
        weak let weakCoordinator = c
        c = nil
        try await waitUntil { weakCoordinator == nil }
        try await waitUntil { outcome.withLock { $0 != nil } }
        #expect(outcome.withLock { $0 as? EngineError } == .cancelled)
        job.cancel()  // clean up should the test fail
        _ = await watcher.result
    }

    /// `cancelAll()` (run when a window closes) stops every running job.
    @Test func cancelAllStopsEngineJobs() async throws {
        let engine = LibraryEngine()
        let c = AnalysisCoordinator(engine: engine)
        c.setEngineRequest(EngineAnalysisRequest(analysis: "forever"))
        c.update(system: SampleSystems.singlet(), immediately: true)
        try await waitUntil { c.isBusy }
        c.cancelAll()
        try await waitUntil { c.engineSlot.phase == .cancelled }
        #expect(!c.isBusy)
    }

    private func waitUntil(_ timeout: Duration = .seconds(5), _ cond: () -> Bool) async throws {
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
}
