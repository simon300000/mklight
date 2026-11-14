import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The app-wide glass library (files, folders, user catalogs, overrides)
/// and the user-glass editor, without an engine.
@Suite @MainActor struct GlassLibraryTests {
    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-glasslib-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    @Test func entriesAndFoldersPersistAndExpand() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let folder = dir.appendingPathComponent("catalogs")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for f in ["B.agf", "a.AGF", "notes.txt", "C.json"] { try Data("x".utf8).write(to: folder.appendingPathComponent(f)) }
        let single = dir.appendingPathComponent("single.agf")
        try Data("x".utf8).write(to: single)

        let lib = GlassLibrary(directory: dir.appendingPathComponent("support"))
        let r0 = lib.revision
        #expect(lib.add(single))
        #expect(!lib.add(single))
        #expect(lib.add(folder))
        #expect(lib.revision > r0)
        #expect(lib.entries.map(\.isFolder) == [false, true])
        #expect(lib.catalogFiles.map(\.lastPathComponent) == ["single.agf", "a.AGF", "B.agf", "C.json"])
        #expect(lib.entry(containing: folder.appendingPathComponent("B.agf").path)?.isFolder == true)

        lib.setStatus(.preferred, catalog: "CAT", name: "G1")
        lib.setExcluded(true, catalog: "cat", name: "g1")
        #expect(lib.overrides == [GlassOverride(catalog: "CAT", name: "G1", status: .preferred, excludeSubstitution: true)])

        // A second instance reads the same settings.
        let again = GlassLibrary(directory: dir.appendingPathComponent("support"))
        #expect(again.entries == lib.entries)
        #expect(again.overrides == lib.overrides)

        lib.setStatus(nil, catalog: "CAT", name: "G1")
        #expect(lib.override(catalog: "CAT", name: "G1")?.status == nil)
        lib.clearOverride(catalog: "CAT", name: "G1")
        #expect(lib.overrides.isEmpty)
        lib.remove(lib.entries[0])
        #expect(lib.entries.count == 1)
    }

    @Test func userCatalogsAndGlasses() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let lib = GlassLibrary(directory: dir)
        #expect(throws: GlassLibrary.LibraryError.self) { try lib.createUserCatalog(named: "BUILTIN") }
        #expect(throws: GlassLibrary.LibraryError.self) { try lib.createUserCatalog(named: "  ") }
        let c = try lib.createUserCatalog(named: "Mine")
        #expect(FileManager.default.fileExists(atPath: c.url.path))
        #expect(throws: GlassLibrary.LibraryError.duplicateName("mine")) { try lib.createUserCatalog(named: "mine") }
        #expect(lib.catalogFiles == [c.url])

        let g: JSONValue = .object(["name": .string("G1"), "nd": .number(1.6)])
        try lib.putGlass(g, in: "Mine")
        #expect(throws: GlassLibrary.LibraryError.duplicateName("g1")) {
            try lib.putGlass(.object(["name": .string("g1")]), in: "Mine")
        }
        #expect(throws: GlassLibrary.LibraryError.self) { try lib.putGlass(.object(["name": .string("two words")]), in: "Mine") }
        #expect(lib.freeGlassName("G1", in: "Mine") == "G1_2")
        #expect(lib.freeGlassName("NEW GLASS", in: "Mine") == "NEW_GLASS")
        // Renaming a glass in the editor replaces it.
        try lib.putGlass(.object(["name": .string("G2"), "nd": .number(1.7)]), in: "Mine", replacing: "G1")
        #expect(lib.userCatalog("Mine")?.glasses.keys.sorted() == ["G2"])

        // On disk: the engine's GlassCatalog JSON.
        let file = try JSONValue.parse(Data(contentsOf: c.url))
        #expect(file["name"]?.string == "Mine")
        #expect(file["glasses"]?["G2"]?["nd"]?.doubleValue == 1.7)

        try lib.renameUserCatalog("Mine", to: "Ours")
        #expect(!FileManager.default.fileExists(atPath: c.url.path))
        #expect(lib.userCatalog("Ours")?.glasses.count == 1)
        let reread = GlassLibrary(directory: dir)
        #expect(reread.userCatalogs.map(\.name) == ["Ours"])

        lib.setStatus(.obsolete, catalog: "Ours", name: "G2")
        try lib.deleteGlass("G2", from: "Ours")
        #expect(lib.overrides.isEmpty)
        try lib.deleteUserCatalog("Ours")
        #expect(lib.userCatalogs.isEmpty)
    }

    @Test func editorBuildsTheEngineSpec() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let lib = GlassLibrary(directory: dir)
        try lib.createUserCatalog(named: "Mine")
        let e = GlassEditorModel(newIn: "Mine", library: lib)
        #expect(e.name == "NEWGLASS")
        e.nd = 1.7
        e.vd = 30
        e.density = 3.5
        let spec = e.spec
        #expect(spec["model"]?["nd"]?.doubleValue == 1.7)
        #expect(spec["formula"] == nil)
        #expect(spec["density"]?.doubleValue == 3.5)
        #expect(spec["status"] == .string("standard"))
        #expect(spec["min_um"] == nil)

        e.kind = .formula
        e.formula = 6
        #expect(e.coefficients.count == 8)
        #expect(GlassEditorModel.coefficientNames(6) == ["K1", "L1", "K2", "L2", "K3", "L3", "K4", "L4"])
        e.formula = 5
        #expect(e.coefficients.count == 3)
        #expect(e.spec["formula"]?.int == 5)
        #expect(e.spec["coefficients"]?.array?.count == 3)
        for f in 1...13 { #expect(!GlassEditorModel.coefficientNames(f).isEmpty) }
        e.name = "has space"
        // Without an engine the name check still runs first.
        #expect(GlassLibrary.nameProblem(e.name, glass: true) != nil)
    }

    @Test func statusJSONRoundTrips() throws {
        for s in GlassStatus.choices + [.other(7)] {
            #expect(GlassStatus(json: s.json) == s)
        }
        let o = GlassOverride(catalog: "C", name: "N", status: .other(9))
        let back = try MKLightJSON.decode(GlassOverride.self, from: MKLightJSON.encode(o))
        #expect(back == o)
        let text = String(decoding: try MKLightJSON.encode(GlassOverride(catalog: "C", name: "N", excludeSubstitution: true)), as: UTF8.self)
        #expect(text == #"{"catalog":"C","exclude_substitution":true,"name":"N"}"#)
    }
}
