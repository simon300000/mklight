import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The lens table and inspector edits, through the view models, with undo.
@Suite @MainActor struct EditingTests {
    private func editor() -> (DesignDocument, LensEditor, UndoManager) {
        let doc = DesignDocument(system: SampleSystems.cookeTriplet())
        let undo = UndoManager()
        undo.groupsByEvent = false
        return (doc, LensEditor(document: doc, undoManager: undo), undo)
    }

    private func step(_ undo: UndoManager, _ f: () -> Void) {
        undo.beginUndoGrouping()
        f()
        undo.endUndoGrouping()
    }

    private func row(_ doc: DesignDocument, _ i: Int) -> LensRow { LensRow.rows(for: doc.system)[i] }

    @Test func rowsShowTheCommonColumns() {
        let s = SampleSystems.cookeTriplet()
        let rows = LensRow.rows(for: s, autoSemiDiameters: [s.surfaces[1].id: 9.63])
        #expect(rows.map(\.surfaceText).first == "OBJ" && rows.last?.surfaceText == "IMA")
        #expect(rows[s.stop].role == .stop && rows[s.stop].surfaceText == "STO")
        #expect(rows[1].text(.radius) == "22.0136" && rows[1].text(.material) == "N-SK16")
        #expect(rows[1].text(.semiDiameter) == "9.630" && rows[1].semiDiameterIsAuto)
        #expect(rows[0].text(.thickness) == "∞" && !rows[0].editable.contains(.radius), "object at infinity")
        #expect(!rows.last!.editable.contains(.thickness) && !rows.last!.editable.contains(.material))
        #expect(rows[2].text(.material).isEmpty, "air is shown blank")
    }

    @Test func typedValuesBecomeUndoableEdits() {
        let (doc, ed, undo) = editor()
        let id = doc.system.surfaces[1].id
        step(undo) { #expect(ed.commit(row(doc, 1), .radius, text: "∞") == nil) }
        #expect(doc.system.surfaces[1].shape.curvature == 0)
        step(undo) { #expect(ed.commit(row(doc, 1), .thickness, text: "4.5") == nil) }
        step(undo) { #expect(ed.commit(row(doc, 1), .material, text: "n-bk7") == nil) }
        #expect(doc.system.surfaces[1].material == .glass("N-BK7"))
        step(undo) { #expect(ed.commit(row(doc, 1), .semiDiameter, text: "8") == nil) }
        #expect(doc.system.surfaces[1].semiDiameter == .fixed(8))
        // Rejected text changes nothing.
        let before = doc.system
        #expect(ed.commit(row(doc, 1), .thickness, text: "abc") != nil)
        #expect(ed.commit(row(doc, 2), .semiDiameter, text: "-1") != nil)
        #expect(doc.system == before)
        // Unchanged text is not an edit.
        #expect(ed.commit(row(doc, 1), .thickness, text: "4.5") == nil && !undo.canRedo)
        undo.undo()
        undo.undo()
        #expect(doc.system.surfaces[1].material == .glass("N-SK16"))
        undo.undo()
        undo.undo()
        #expect(doc.system == SampleSystems.cookeTriplet())
        undo.redo()
        #expect(doc.system.surfaces[1].shape.curvature == 0 && doc.system.surfaces[1].id == id)
    }

    @Test func insertDeleteAndStopAreUndoable() {
        let (doc, ed, undo) = editor()
        let start = doc.system
        var inserted: SurfaceID?
        step(undo) { inserted = ed.insert(at: 3) }
        #expect(doc.system.surfaces.count == start.surfaces.count + 1 && doc.system.surfaces[3].id == inserted)
        #expect(doc.system.stop == start.stop + 1, "the stop keeps its surface")
        var next: SurfaceID?
        step(undo) { next = ed.delete(doc.system.surfaces[1].id) }
        #expect(next == doc.system.surfaces[1].id)
        step(undo) { ed.makeStop(doc.system.surfaces[2].id) }
        #expect(doc.system.stop == 2)
        #expect(ed.delete(doc.system.surfaces[0].id) == nil, "the object cannot be deleted")
        undo.undo()
        undo.undo()
        undo.undo()
        #expect(doc.system == start)
        #expect(doc.revision > 0)
    }

    @Test func variablesAreUndoableAndFollowTheDesign() {
        let (doc, ed, undo) = editor()
        let r2 = ParamPath.curvature(surface: doc.system.surfaces[2].id)
        step(undo) { ed.toggleVariable(r2) }
        #expect(doc.isVariable(r2) && doc.variables.count == 1)
        #expect(LensRow.rows(for: doc.system, variables: [r2])[2].markers[.radius]?.isVariable == true)
        undo.undo()
        #expect(!doc.isVariable(r2))
        undo.redo()
        #expect(doc.isVariable(r2))
        // Deleting the surface removes its variable with it, in one step.
        step(undo) { ed.delete(doc.system.surfaces[2].id) }
        #expect(doc.variables.isEmpty)
        undo.undo()
        #expect(doc.isVariable(r2))
        // The variables are saved in the project.
        let saved = doc.savedProject.sections["variables"]
        #expect(saved?.array?.count == 1)
    }

    @Test func solveMarkersShowInTheTable() {
        var s = SampleSystems.cookeTriplet()
        s.solves = [Solve(target: .thickness(surface: s.surfaces[6].id), kind: .marginalRayHeight(height: 0))]
        let rows = LensRow.rows(for: s)
        #expect(rows[6].markers[.thickness]?.solve?.letter == "M")
        #expect(rows[6].markers[.radius] == nil)
    }

    /// The inspector's form is data: a parameter is read and written by its
    /// key in the shape's JSON, and a coefficient list grows and shrinks.
    @Test func dataDrivenSurfaceParameters() throws {
        let asphere = SurfaceTypes.descriptor(for: SurfaceShape.Kind.evenAsphere)
        #expect(asphere.sections.map(\.title) == ["Base Conic", "Polynomial"])
        let shape = SurfaceShape.evenAsphere(curvature: 0.02, conic: -1, coefficients: [1e-5])
        let radius = asphere.sections[0].params[0]
        #expect(SurfaceTypes.displayValue(radius, in: shape) == 50)
        let stored = try SurfaceTypes.storedValue(radius, text: "25").get()
        let changed = try #require(SurfaceTypes.shape(shape, setting: radius.key, to: stored))
        #expect(changed.curvature == 0.04)
        let list = try #require(asphere.sections[1].coefficients)
        #expect(list.term(0) == "A4" && list.term(2) == "A8" && list.unit(0) == "mm⁻³")
        let longer = try #require(SurfaceTypes.shape(shape, setting: list.key, to: .array([.number(1e-5), .number(2e-7)])))
        #expect(longer.asphereCoefficients == [1e-5, 2e-7])
        #expect(SurfaceTypes.shape(shape, setting: "no_such_key", to: .number(1)) == nil, "the model rejects unknown keys")
        #expect(SurfaceTypes.descriptor(for: SurfaceShape.Kind.coordinateBreak).placementFirst)
        // The extended shapes get their descriptors with the inspector redesign.
        #expect(
            Set(SurfaceTypes.all.map(\.kind)) == Set(SurfaceShape.Kind.allCases.filter(\.isLegacy)),
            "every legacy model type has a descriptor")
        #expect(SurfaceTypes.all.map(\.id) == SurfaceTypes.all.map(\.kind.typeName), "descriptor ids are the JSON tags")
        // Sag preview of a paraboloid.
        let pts = try #require(SurfaceTypes.profile(.conic(curvature: 0.01, conic: -1), semiDiameter: 10))
        #expect(abs(pts.last!.1 - 0.5) < 1e-12)
    }

    @Test func parameterPathsReadAndWriteTheModel() {
        var s = SampleSystems.cookeTriplet()
        let id = s.surfaces[2].id
        #expect(s.value(at: .thickness(surface: id)) == s.surfaces[2].thickness)
        let a = s.setValue(12, at: .thickness(surface: id))
        let b = s.setValue(0.5, at: .tiltX(surface: id))
        let c = s.setValue(1, at: .asphere(surface: id, index: 0))
        #expect(a && s.surfaces[2].thickness == 12)
        #expect(b && s.surfaces[2].placement?.tiltX == 0.5)
        #expect(!c, "not an asphere")
        #expect(s.value(at: .semiDiameter(surface: id)) == nil, "automatic")
        let d = s.setValue(8, at: .fieldY(index: 1))
        #expect(d && s.fields.fields[1].y == 8)
        #expect(ParamPath.curvature(surface: id).label(in: s) == "Radius 2")
    }
}
