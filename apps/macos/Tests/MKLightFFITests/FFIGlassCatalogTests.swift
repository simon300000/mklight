import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// Glass catalog management on the real engine: the app-wide library in a
/// design's session, overrides, reload, removal and undo, the Glass
/// Catalogs window's model, user glasses and AGF export. Synthetic catalogs
/// only.
@Suite(.serialized, engineLinked) @MainActor struct FFIGlassCatalogTests {
    /// Two made-up glasses.
    static let agf = """
        NM SYNCROWN 2 517642 1.5168 64.17 0 1 0\r
        CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0\r
        LD 0.3 2.5\r
        NM SYNFLINT 2 620364 1.62004 36.37 0 0 0\r
        ED 8.2 - 3.6 0.0025 0\r
        CD 1.34533359 0.00997743871 0.209073176 0.0470450767 0.937357162 111.886764 0 0 0 0\r
        LD 0.32 2.5\r

        """

    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-glass-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    private func wait(_ timeout: Duration = .seconds(30), _ cond: () -> Bool) async throws {
        let clock = ContinuousClock()
        let end = clock.now + timeout
        while !cond() {
            guard clock.now < end else {
                Issue.record("condition not met within \(timeout)")
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    @Test func libraryOverridesReloadAndRemoval() async throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("SYN.agf")
        try Self.agf.write(to: file, atomically: true, encoding: .utf8)
        let lib = GlassLibrary(directory: dir.appendingPathComponent("support"))
        lib.add(file)

        let w = Workspace(engine: try FFIEngine())
        w.debounce = .zero
        w.glassLibrary = lib
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        await w.open(system: system, catalogs: [], projectDirectory: nil)
        let syn = try #require(w.catalogs.first { $0.name == "SYN" })
        #expect(syn.library && !syn.inProject && syn.glassCount == 2)

        // Overrides reach the glass rows.
        lib.setStatus(.obsolete, catalog: "SYN", name: "SYNCROWN")
        lib.setExcluded(true, catalog: "SYN", name: "SYNFLINT")
        await w.syncLibrary()
        let rows = try await w.glassRows(in: "SYN")
        let crown = try #require(rows.first { $0.name == "SYNCROWN" })
        #expect(crown.status == .obsolete && crown.overridden)
        let flint = try #require(rows.first { $0.name == "SYNFLINT" })
        #expect(flint.excludeSubstitution && flint.density == 3.6 && flint.tce == 8.2)
        let d = try await w.glassDetails(catalog: "SYN", name: "SYNCROWN")
        #expect(d.catalogStatus == .preferred && d.formula == 2 && d.indexCurve.count > 10)

        // Reload: a changed file reports what changed.
        try Self.agf.replacingOccurrences(of: "1.62004 36.37", with: "1.62010 36.37").write(to: file, atomically: true, encoding: .utf8)
        let changes = try await w.reloadCatalog(named: "SYN")
        #expect(changes.changed.map(\.name) == ["SYNFLINT"])
        #expect(changes.changed.first?.ndAfter == 1.6201)

        // Used by the design: the save records the library catalog.
        let doc = DesignDocument(project: ProjectFile(system: system))
        var s = system
        s.surfaces[1].material = .catalog(catalog: "SYN", name: "SYNCROWN")
        w.update(system: s, immediately: true)
        try await wait { w.snapshot?.system == s }
        doc.setLibraryCatalogsInUse(await w.libraryCatalogsInUse())
        #expect(doc.savedProject.catalogs.map(\.name) == ["SYN"])
        #expect(doc.catalogs.isEmpty)

        // A project catalog: add, remove (undoable), undo.
        let other = dir.appendingPathComponent("OTHER.agf")
        try Self.agf.replacingOccurrences(of: "SYN", with: "OTH").write(to: other, atomically: true, encoding: .utf8)
        let undo = UndoManager()
        undo.groupsByEvent = false
        let info = try await w.loadCatalog(at: other)
        undo.beginUndoGrouping()
        doc.addCatalog(CatalogReference(name: info.name, path: other.path), undoManager: undo)
        undo.endUndoGrouping()
        #expect(w.catalogs.contains { $0.name == "OTHER" && $0.inProject })
        undo.beginUndoGrouping()
        doc.removeCatalog(named: "OTHER", undoManager: undo)
        undo.endUndoGrouping()
        #expect(undo.undoActionName == "Remove Glass Catalog")
        await w.syncCatalogs(doc.catalogs, projectDirectory: nil)
        #expect(!w.catalogs.contains { $0.name == "OTHER" })
        #expect(w.catalogs.contains { $0.name == "SYN" })
        undo.undo()
        await w.syncCatalogs(doc.catalogs, projectDirectory: nil)
        #expect(w.catalogs.contains { $0.name == "OTHER" && $0.inProject })
    }

    @Test func browserModelUserGlassesAndExport() async throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("SYN.agf")
        try Self.agf.write(to: file, atomically: true, encoding: .utf8)
        let lib = GlassLibrary(directory: dir.appendingPathComponent("support"))
        lib.add(file)
        let engine = try FFIEngine()
        let m = GlassBrowserModel(library: lib)
        m.engine = engine
        await m.refresh()
        #expect(m.libraryCatalogs.map(\.name) == ["SYN"])
        #expect(m.builtinCatalogs.count == 1)
        m.search = "syn"
        #expect(m.visibleRows.map(\.name) == ["SYNCROWN", "SYNFLINT"])
        m.filter = .preferred
        #expect(m.visibleRows.map(\.name) == ["SYNCROWN"])
        m.filter = .all
        m.sortOrder = [KeyPathComparator(\.vd)]
        #expect(m.visibleRows.map(\.name) == ["SYNFLINT", "SYNCROWN"])
        #expect(m.assignment.index == nil && m.assignment.why != nil)

        // A user glass: duplicate a catalog glass, change it, save.
        let flint = try #require(m.rows.first { $0.name == "SYNFLINT" })
        await m.duplicate(flint)
        let e = try #require(m.editor)
        #expect(e.catalog == "User Glasses" && e.name == "SYNFLINT" && e.kind == .formula && e.formula == 2)
        e.name = "MYFLINT"
        #expect(m.saveEditor())
        let saved = try #require(lib.userCatalog("User Glasses")?.glasses["MYFLINT"])
        #expect(abs((saved["nd"]?.doubleValue ?? 0) - 1.62004) < 1e-4)
        // The base glass's data stay (density from the ED record).
        #expect(saved["density"]?.doubleValue == 3.6)

        // A model glass.
        m.newGlass()
        let mg = try #require(m.editor)
        mg.name = "MODEL1"
        mg.nd = 1.7
        mg.vd = 35
        mg.dpgf = 0.002
        #expect(m.saveEditor())
        await m.refresh()
        #expect(m.userCatalogs.first?.glassCount == 2)
        m.scope = .catalog("User Glasses")
        await m.loadRows()
        #expect(Set(m.rows.map(\.name)) == ["MYFLINT", "MODEL1"])

        // A bad glass does not save.
        m.newGlass()
        let bad = try #require(m.editor)
        bad.nd = 0.9
        #expect(!m.saveEditor())
        #expect(bad.problem != nil)
        m.editor = nil

        // AGF export of the user catalog reads back.
        let export = try #require(m.agfExport(of: "User Glasses"))
        #expect(export.warnings.isEmpty)
        #expect(export.text.contains("NM MODEL1 1 "))
        #expect(export.text.contains("NM MYFLINT 2 "))
    }
}
