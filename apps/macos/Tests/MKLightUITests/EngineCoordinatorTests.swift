import Foundation
import Synchronization
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// An engine with the capabilities of the real one that the mock lacks: a
/// per-document session with a glass library hash, analyses by name, and a
/// typed analysis (PSF) that this "build" does not offer.
final class SessionEngine: EngineClient, @unchecked Sendable {
    let mock = MockEngine(simulatedLatency: .zero)
    let library = Mutex("L0")
    let sessionsMade = Mutex(0)

    var info: EngineInfo { EngineInfo(name: "session engine", version: "1", backends: ["cpu_reference"], isMock: false) }

    func makeDocumentEngine() -> any EngineClient {
        sessionsMade.withLock { $0 += 1 }
        return self
    }

    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot {
        let s = try await mock.openSnapshot(system)
        return ModelSnapshot(system: s.system, hash: s.hash, libraryHash: library.withLock { $0 })
    }

    func validate(_ system: OpticalSystem) async -> [ModelIssue] { await mock.validate(system) }
    func materialNames() async -> [String] { [] }

    func submit(_ request: AnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        if request.kind == .psf {
            return AnalysisJob.run(request: request, snapshot: snapshot) { _ in throw EngineError.unsupported("PSF: not in this build") }
        }
        return mock.submit(request, on: snapshot)
    }

    func analysisDescriptors() async -> [AnalysisDescriptor] {
        [AnalysisDescriptor(name: "system_summary", title: "System summary", category: "system", available: true)]
    }

    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let lib = snapshot.libraryHash
        return AnalysisJob.run(engineRequest: request, snapshot: snapshot) { _ in
            let info = RunInfo(
                engineVersion: "1", snapshotHash: snapshot.hash, analysis: request.analysis,
                settings: .object(request.settings.merging(["extra_default": .integer(3)]) { a, _ in a }),
                backend: "cpu_reference", precision: .f64Reference, seed: nil, startedUnixMs: 0, durationMs: 0)
            return AnalysisOutput(
                runInfo: info,
                result: .generic(GenericResult(analysis: request.analysis, json: .object(["n": .integer(Int64(snapshot.system.surfaces.count))]))),
                libraryHash: lib, engineRequest: request)
        }
    }

    func catalogs() async -> [CatalogInfo] { [CatalogInfo(name: "BUILTIN", source: "builtin", path: nil, glassCount: 13)] }

    func loadCatalog(at url: URL) async throws -> CatalogInfo {
        library.withLock { $0 = "L1" }
        return CatalogInfo(name: "MYCAT", source: "user", path: url.path, glassCount: 1)
    }
}

@Suite @MainActor struct EngineCoordinatorTests {
    @Test func unavailableAnalysesAndEnginePane() async throws {
        let engine = SessionEngine()
        let c = AnalysisCoordinator(engine: MockEngine(simulatedLatency: .zero))
        c.attach(engine: engine)
        #expect(engine.sessionsMade.withLock { $0 } == 1, "one session per document")
        await c.loadEngineMetadata()
        #expect(c.descriptor("system_summary") != nil)
        c.debounce = .milliseconds(1)
        c.setVisible([.psf, .firstOrder])
        let request = EngineAnalysisRequest(analysis: "system_summary", settings: ["verbose": .bool(true)])
        c.setEngineRequest(request)
        var s = SampleSystems.singlet()
        c.update(system: s, immediately: true)
        try await waitUntil { c.engineSlot.output != nil && c.slot(.psf).phase != .idle && !c.isBusy }
        #expect(c.slot(.psf).phase == .unavailable("PSF: not in this build"))
        #expect(c.output(.firstOrder) != nil)
        #expect(!c.isEngineResultStale, "engine-added defaults do not make it stale")
        // A model edit makes the result stale until the re-run arrives.
        s.surfaces[2].thickness += 1
        c.update(system: s)
        try await waitUntil { c.snapshot?.system == s && !c.isBusy }
        #expect(!c.isEngineResultStale)
        let hashAfterEdit = c.engineSlot.output?.runInfo.snapshotHash
        #expect(hashAfterEdit == c.snapshot?.hash)
        // Loading a catalog changes the library hash: same system, new results.
        let before = c.engineSlot.output
        _ = try await c.loadCatalog(at: URL(fileURLWithPath: "/tmp/MYCAT.agf"))
        #expect(c.snapshot?.libraryHash == "L1")
        try await waitUntil { c.engineSlot.output?.libraryHash == "L1" && !c.isBusy }
        #expect(before?.isCurrent(for: c.snapshot, request: request) == false, "old library → stale")
        #expect(!c.isStale(.firstOrder))
        // Hiding the pane stops tracking it.
        c.setEngineRequest(nil)
        #expect(c.isEngineResultStale)
    }

    @Test func projectCatalogsReportMissingFiles() async throws {
        let c = AnalysisCoordinator(engine: SessionEngine())
        c.update(system: SampleSystems.singlet(), immediately: true)
        await c.loadProjectCatalogs(
            [CatalogReference(name: "GONE", path: "/nonexistent/gone.agf", relativePath: "glass/gone.agf")],
            projectDirectory: URL(fileURLWithPath: "/nonexistent"))
        #expect(c.catalogWarnings.count == 1 && c.catalogWarnings[0].contains("GONE"))
    }

    @Test func viewStateEnginePane() {
        let v = WindowViewState()
        v.selectedAnalysis = .mtf
        #expect(v.visibleAnalyses.contains(.mtf) && v.engineRequest == nil)
        v.analysisPane = .engine
        v.engineSettings["system_summary"] = ["x": .integer(1)]
        #expect(!v.visibleAnalyses.contains(.mtf) && v.visibleAnalyses.contains(.firstOrder))
        #expect(v.engineRequest?.analysis == "system_summary" && v.engineRequest?.settings["x"] == .integer(1))
        v.analysisPane = .typed(.psf)
        #expect(v.selectedAnalysis == .psf && !v.showsEnginePane && v.engineRequest == nil)
        v.analysisSettings.configuration = 2
        #expect(v.analysisSettings.request(for: .mtf).configuration == 2)
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
