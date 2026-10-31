import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// Index and Abbe offsets of catalogue glasses: the table's suffix, the
/// inspector's validation and undo, keeping or dropping an offset when the
/// glass is retyped, variables, the glass map's offset point and the
/// substitution note.
@Suite @MainActor struct GlassOffsetTests {
    private let cooke = SampleSystems.cookeTriplet()
    private func id(_ i: Int) -> SurfaceID { cooke.surfaces[i].id }

    @Test func compactTextInTheTable() {
        #expect(MaterialOffset(dnD: 0.0012, dvD: -0.8).compactText == "+0.0012/\u{2212}0.8")
        #expect(MaterialOffset(dnD: -0.0005).compactText == "\u{2212}0.0005/0" && MaterialOffset.zero.compactText.isEmpty)
        #expect(MaterialOffset(dnD: 0.00123, dvD: 0.25).longText == "Δn_d +0.00123, ΔV_d +0.25")
        var s = cooke
        s.surfaces[1].material = .catalog(catalog: nil, name: "N-SK16", offset: MaterialOffset(dnD: 0.0012, dvD: -0.8))
        let rows = LensRow.rows(for: s, variables: [.materialDvD(surface: id(1))])
        #expect(rows[1].cells[.material] == "N-SK16" && rows[1].notes[.material] == "+0.0012/\u{2212}0.8")
        #expect(rows[1].markers[.material]?.isVariable == true, "an offset variable marks the glass")
        #expect(rows[3].notes[.material] == nil)
    }

    @Test func offsetEditingValidationAndUndo() throws {
        // The engine's bounds: |Δn_d| ≤ 0.05, |ΔV_d| ≤ 10.
        #expect(try LensValueParser.offsetDnD("0.0012").get() == 0.0012)
        #expect(try LensValueParser.offsetDnD("\u{2212}0,002").get() == -0.002)
        if case .failure(let f) = LensValueParser.offsetDnD("0.06") { #expect(f.description.contains("±0.05")) } else { Issue.record("Δn_d too large") }
        if case .failure = LensValueParser.offsetDvD("-10.5") {} else { Issue.record("ΔV_d too large") }
        if case .failure = LensValueParser.offsetDvD("x") {} else { Issue.record("not a number") }
        #expect(try LensValueParser.offsetDvD("-10").get() == -10)
        // An edit is one undoable step on the document.
        let doc = DesignDocument(system: cooke)
        let undo = UndoManager()
        undo.groupsByEvent = false
        undo.beginUndoGrouping()
        _ = doc.edit("Change Δn_d", undoManager: undo) { s in
            if let m = s.surfaces[1].material.withOffset(MaterialOffset(dnD: 0.0012)) { s.surfaces[1].material = m }
        }
        undo.endUndoGrouping()
        #expect(doc.system.surfaces[1].material.catalogOffset == MaterialOffset(dnD: 0.0012))
        #expect(doc.system.value(at: .materialDnD(surface: id(1))) == 0.0012 && undo.undoActionName == "Change Δn_d")
        undo.undo()
        #expect(doc.system.surfaces[1].material.catalogOffset == .zero && doc.system == cooke)
        undo.redo()
        #expect(doc.system.surfaces[1].material.catalogOffset?.dnD == 0.0012)
        // Air and model glasses have no offset.
        #expect(MaterialRef.air.withOffset(MaterialOffset(dnD: 0.001)) == nil && MaterialRef.air.catalogOffset == nil)
        // Offsets can be variables, bounded by the engine's limits.
        #expect(DesignVariable.defaultBounds(for: .materialDnD(surface: id(1)), in: cooke) == (-0.05, 0.05))
        #expect(DesignVariable.defaultBounds(for: .materialDvD(surface: id(1)), in: cooke) == (-10, 10))
    }

    /// Retyping the same glass keeps its melt offset; another glass starts
    /// without one.
    @Test func retypingKeepsOrDropsTheOffset() {
        let melt = MaterialRef.catalog(catalog: "SCHOTT", name: "N-BK7", offset: MaterialOffset(dnD: 0.001, dvD: -0.5))
        #expect(melt.replaced(by: .glass("N-BK7")) == .catalog(catalog: "SCHOTT", name: "N-BK7", offset: MaterialOffset(dnD: 0.001, dvD: -0.5)))
        #expect(melt.replaced(by: .catalog(catalog: "SCHOTT", name: "n-bk7")).catalogOffset?.dnD == 0.001)
        #expect(melt.replaced(by: .glass("N-SK16")) == .glass("N-SK16"))
        #expect(melt.replaced(by: .catalog(catalog: "OHARA", name: "N-BK7")).catalogOffset == .zero, "another maker's melt")
        #expect(melt.replaced(by: .air) == .air)
        var s = cooke
        s.surfaces[1].material = .catalog(catalog: nil, name: "N-SK16", offset: MaterialOffset(dnD: 0.002))
        LensEdit.setMaterial(id(1), LensValueParser.MaterialEdit(material: .glass("N-SK16"), interaction: .refract)).apply(to: &s)
        #expect(s.surfaces[1].material.catalogOffset?.dnD == 0.002)
        LensEdit.setMaterial(id(1), LensValueParser.MaterialEdit(material: .glass("F2"), interaction: .refract)).apply(to: &s)
        #expect(s.surfaces[1].material == .glass("F2"))
    }

    /// The glass map's `offset` points: their own id, the design's surface,
    /// the offset, and the catalogue point they shift from.
    @Test func glassMapOffsetPoint() throws {
        let json: JSONValue = .object([
            "glasses": .array([
                .object([
                    "catalog": .string("SCHOTT"), "name": .string("N-BK7"), "kind": .string("catalog"), "nd": .number(1.5168), "vd": .number(64.17),
                    "pgf": .number(0.5349), "status": .string("standard"), "surfaces": .array([]),
                ]),
                .object([
                    "catalog": .string("SCHOTT"), "name": .string("N-BK7"), "kind": .string("offset"), "nd": .number(1.5180), "vd": .number(63.37),
                    "pgf": .number(0.5349), "status": .string("standard"), "surfaces": .array([.integer(1)]),
                    "offset": .object(["dn_d": .number(0.0012), "dv_d": .number(-0.8)]),
                ]),
            ])
        ])
        let d = try #require(GlassMapData(output: MapReaderTests.output("glass_map", json: json, buffers: [])))
        #expect(d.glasses.count == 2 && d.glasses[0].id != d.glasses[1].id)
        let o = d.glasses[1]
        #expect(o.kind == .offset && o.inDesign && !d.glasses[0].inDesign && o.nominalID == d.glasses[0].id)
        #expect(o.offset == MaterialOffset(dnD: 0.0012, dvD: -0.8) && o.displayName == "N-BK7 +0.0012/\u{2212}0.8")
        #expect(d.designHighlights == [GlassHighlight(id: o.id, label: "S1")])
        #expect(GlassNumbers(json: .object(["nd": .number(1.518), "vd": .number(63.37), "catalog_nd": .number(1.5168), "catalog_vd": .number(64.17)]))?.catalogVd == 64.17)
    }

    /// Substitution says that a replacement drops the melt offset, and
    /// leaves the offset variables of the glasses it replaces out.
    @Test func substitutionNote() throws {
        var s = cooke
        s.surfaces[3].material = .catalog(catalog: nil, name: "F2", offset: MaterialOffset(dnD: 0.0012, dvD: -0.8))
        let session = GlassSubstitutionSession()
        session.configure(for: s)
        let note = try #require(session.offsetNote(system: s))
        #expect(note.contains("S3 F2 +0.0012/\u{2212}0.8 carries a melt offset") && note.contains("without it, even the same glass"))
        session.selected.remove(id(3))
        #expect(session.offsetNote(system: s) == nil, "only glasses that may change")
        session.selected.insert(id(3))
        #expect(GlassSlot.all(in: s).first { $0.index == 3 }?.offset?.dnD == 0.0012)
        let vars = [DesignVariable(.materialDnD(surface: id(3))), DesignVariable(.curvature(surface: id(1)))]
        let settings = session.settings(system: s, merit: MeritSettings(), variables: vars)
        let sent = try #require(settings["variables"]?.array)
        #expect(sent.count == 1 && sent[0]["param"]?["param"]?.string == "curvature")
    }
}
