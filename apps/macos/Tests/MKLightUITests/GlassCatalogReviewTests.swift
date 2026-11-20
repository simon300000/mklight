import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// Fixes from the hands-on review of glass catalog management: statuses
/// that follow renames, Undo of Remove from Library, the user-glass editor's
/// numbers and messages, statuses for the lens table, recorded library
/// catalogs the library still provides, and pane failures that no longer
/// apply.
@Suite @MainActor struct GlassCatalogReviewTests {
    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-glassreview-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    @Test func statusesFollowGlassAndCatalogRenames() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let lib = GlassLibrary(directory: dir)
        try lib.createUserCatalog(named: "Mine")
        try lib.putGlass(.object(["name": .string("G1")]), in: "Mine")
        lib.setStatus(.obsolete, catalog: "Mine", name: "G1")
        lib.setStatus(.preferred, catalog: "BUILTIN", name: "N-BK7")

        // Renaming the glass in the editor keeps its status.
        try lib.putGlass(.object(["name": .string("G2")]), in: "Mine", replacing: "G1")
        #expect(lib.override(catalog: "Mine", name: "G1") == nil)
        #expect(lib.override(catalog: "Mine", name: "G2")?.status == .obsolete)
        // Saving under the same name changes nothing.
        try lib.putGlass(.object(["name": .string("G2"), "nd": .number(1.6)]), in: "Mine", replacing: "G2")
        #expect(lib.override(catalog: "Mine", name: "G2")?.status == .obsolete)

        // Renaming the catalog carries its glasses' statuses; others stay.
        try lib.renameUserCatalog("Mine", to: "Ours")
        #expect(lib.override(catalog: "Ours", name: "G2")?.status == .obsolete)
        #expect(lib.override(catalog: "Mine", name: "G2") == nil)
        #expect(lib.override(catalog: "BUILTIN", name: "N-BK7")?.status == .preferred)
        // And they are saved.
        #expect(GlassLibrary(directory: dir).override(catalog: "Ours", name: "G2")?.status == .obsolete)
    }

    @Test func removeFromLibraryCanBeUndone() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let files = ["A.agf", "B.agf", "C.agf"].map { dir.appendingPathComponent($0) }
        for f in files { try Data("x".utf8).write(to: f) }
        let lib = GlassLibrary(directory: dir.appendingPathComponent("support"))
        for f in files { lib.add(f) }
        let model = GlassBrowserModel(library: lib)

        let b = CatalogInfo(name: "B", source: "user", path: files[1].path, glassCount: 2, library: true)
        model.removeFromLibrary(b)
        #expect(lib.entries.map(\.title) == ["A.agf", "C.agf"])
        #expect(model.removal != nil)
        model.undoRemoval()
        // Back in its place, and the offer is gone.
        #expect(lib.entries.map(\.title) == ["A.agf", "B.agf", "C.agf"])
        #expect(model.removal == nil)
        #expect(!lib.restore(lib.entries[0], at: 0))

        // Another note ends the offer.
        model.removeFromLibrary(b)
        model.message = "Something else."
        #expect(model.removal == nil)
    }

    @Test func editorNumbersAndMessages() throws {
        #expect(GlassEditorModel.clean(2.0539125955e-16) == 0)
        #expect(GlassEditorModel.clean(-3e-14) == 0)
        #expect(GlassEditorModel.clean(1.51680000000004) == 1.5168)
        #expect(GlassEditorModel.clean(64.17) == 64.17)
        #expect(GlassEditorModel.clean(-0.0009) == -0.0009)

        let refusal = EngineError.invalidModel(["invalid model: the dispersion gives n_d = 1, V_d = NaN"])
        #expect(GlassEditorModel.describe(refusal) == "The dispersion gives n_d = 1, V_d = NaN")
        #expect(GlassEditorModel.describe(EngineError.engine(.invalidModel, "invalid model: bad")) == "Bad")

        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let lib = GlassLibrary(directory: dir)
        try lib.createUserCatalog(named: "Mine")
        let e = GlassEditorModel(newIn: "Mine", library: lib)
        #expect(!e.renames)
    }

    @Test func statusIndexResolvesBareNamesInLoadOrder() throws {
        func row(_ cat: String, _ name: String, _ status: String) -> GlassRow {
            GlassRow(json: .object(["catalog": .string(cat), "name": .string(name), "status": .string(status)]))!
        }
        // Rows in any order; the engine resolves a bare name to the first
        // catalog in load order (built-in, library, project).
        let rows = [row("VENDOR", "N-BK7", "obsolete"), row("BUILTIN", "N-BK7", "standard"), row("VENDOR", "F5", "preferred")]
        let index = Workspace.statusIndex(rows, catalogOrder: ["BUILTIN", "VENDOR"])
        #expect(index["N-BK7"] == .standard)
        #expect(index["VENDOR:N-BK7"] == .obsolete)
        #expect(index["F5"] == .preferred)

        let ws = Workspace(engine: MockEngine(simulatedLatency: .zero))
        #expect(ws.glassStatus(.air) == nil)
    }

    /// An infrared glass whose range leaves out the d line has no n_d, V_d
    /// (the engine reports 1 and 0); the table shows a dash, not 1.00000.
    @Test func glassesWithoutTheDLineShowNoNumbers() throws {
        let ir = try #require(GlassRow(json: .object([
            "catalog": .string("IR"), "name": .string("IRG"), "nd": .number(1), "vd": .number(0),
            "min_um": .number(2), "max_um": .number(14),
        ])))
        #expect(!ir.hasDLine && ir.ndText == "—" && ir.vdText == "—")
        let crown = try #require(GlassRow(json: .object(["catalog": .string("C"), "name": .string("K"), "nd": .number(1.5168), "vd": .number(64.17)])))
        #expect(crown.hasDLine && crown.ndText == "1.51680" && crown.vdText == "64.17")
        #expect(glassCountText(1) == "1 glass" && glassCountText(0) == "0 glasses")
    }

    /// A project records the library catalogs it uses by path. Opened where
    /// the library has a catalog of that name elsewhere, it is not reported
    /// missing; a catalog nobody provides is, with what to do.
    @Test func recordedLibraryCatalogsTheLibraryProvidesAreNotMissing() async throws {
        let ws = Workspace(engine: LibraryCatalogEngine())
        await ws.loadEngineMetadata()
        let refs = [
            CatalogReference(name: "VENDOR", path: "/nonexistent/old/VENDOR.agf"),
            CatalogReference(name: "GONE", path: "/nonexistent/GONE.agf"),
        ]
        await ws.open(system: SampleSystems.singlet(), catalogs: refs, projectDirectory: nil)
        #expect(ws.catalogWarnings.count == 1)
        let w = try #require(ws.catalogWarnings.first)
        #expect(w.contains("“GONE”") && w.contains("not found") && w.contains("library"))
    }

    /// A pane whose result is current again (the glass that broke the
    /// design came back unchanged) drops the failure of the snapshot between.
    @Test func paneFailureClearsWhenItsResultIsCurrentAgain() async throws {
        let pane = AnalysisPane(analysis: "first_order")
        let ws = Workspace(engine: FailingEngine(), panes: [pane])
        await ws.loadEngineMetadata()
        let good = SampleSystems.singlet()
        ws.update(system: good, immediately: true)
        try await waitUntil { pane.output != nil && pane.phase == .finished }
        var bad = good
        bad.name = "bad"
        ws.update(system: bad, immediately: true)
        try await waitUntil { if case .failed = pane.phase { true } else { false } }
        ws.update(system: good, immediately: true)
        try await waitUntil { pane.phase == .finished }
        #expect(pane.output != nil)
    }
}

/// An engine whose library provides catalog `VENDOR` (from another path
/// than projects record).
final class LibraryCatalogEngine: EngineClient, @unchecked Sendable {
    private let mock = MockEngine(simulatedLatency: .zero)
    var info: EngineInfo { EngineInfo(name: "library catalogs", version: "1", isMock: false) }
    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot { try await mock.openSnapshot(system) }
    func materialNames() async -> [String] { [] }
    func analysisDescriptors() async -> [AnalysisDescriptor] { await mock.analysisDescriptors() }
    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob { mock.submit(request, on: snapshot) }
    func catalogs() async -> [CatalogInfo] {
        [
            CatalogInfo(name: "BUILTIN", source: "builtin", path: nil, glassCount: 13),
            CatalogInfo(name: "VENDOR", source: "user", path: "/new/VENDOR.agf", glassCount: 3, library: true),
        ]
    }
}

/// An engine whose analyses fail on a system named "bad".
final class FailingEngine: EngineClient, @unchecked Sendable {
    private let mock = MockEngine(simulatedLatency: .zero)
    var info: EngineInfo { EngineInfo(name: "failing", version: "1", isMock: false) }
    func openSnapshot(_ system: OpticalSystem) async throws -> ModelSnapshot { try await mock.openSnapshot(system) }
    func materialNames() async -> [String] { [] }
    func analysisDescriptors() async -> [AnalysisDescriptor] {
        ["first_order", "layout", "validate"].map { AnalysisDescriptor(name: $0, title: $0, category: "geometric", available: true) }
    }
    func submit(_ request: EngineAnalysisRequest, on snapshot: ModelSnapshot) -> AnalysisJob {
        let bad = snapshot.system.name == "bad"
        return AnalysisJob.run(engineRequest: request, snapshot: snapshot) { _ in
            if bad { throw EngineError.engine(.invalidModel, "unknown material 'Mine:G1'") }
            let info = RunInfo(
                engineVersion: "1", snapshotHash: snapshot.hash, analysis: request.analysis, settings: .object(request.settings),
                backend: "cpu_reference", precision: .f64Reference, seed: nil, startedUnixMs: 0, durationMs: 0)
            return AnalysisOutput(runInfo: info, json: .object([:]), libraryHash: snapshot.libraryHash, engineRequest: request)
        }
    }
}
