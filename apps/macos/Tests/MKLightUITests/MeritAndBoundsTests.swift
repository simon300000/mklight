import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The merit function editor and variable bounds through their view models.
@Suite @MainActor struct MeritAndBoundsTests {
    private func d(_ type: String) throws -> OperandDescriptor { try #require(OperandCatalog.descriptor(type)) }

    private func undoManager() -> UndoManager {
        let u = UndoManager()
        u.groupsByEvent = false
        return u
    }

    private func step(_ undo: UndoManager, _ f: () -> Void) {
        undo.beginUndoGrouping()
        f()
        undo.endUndoGrouping()
    }

    /// Operands add after the selection, reorder, duplicate and delete;
    /// `combine` references follow their operands.
    @Test func operandsAddReorderDuplicateAndDelete() throws {
        var m = MeritSettings()
        m.mode = .custom
        let efl = m.add(try d("efl"))
        let spot = m.add(try d("rms_spot"))
        let comb = m.add(try d("combine"))
        let i = try #require(m.operands.firstIndex { $0.id == comb })
        m.operands[i].kind["args"] = .array([.integer(0), .integer(1)])
        #expect(m.operands.map(\.type) == ["efl", "rms_spot", "combine"])
        // Insert after the first: references shift.
        _ = m.add(try d("bfl"), after: 0)
        #expect(m.operands.map(\.type) == ["efl", "bfl", "rms_spot", "combine"])
        #expect(m.operands[3].kind["args"]?.intArray == [0, 2])
        // Move the spot to the top: the combine still names efl and spot.
        m.move(fromOffsets: IndexSet(integer: 2), toOffset: 0)
        #expect(m.operands.map(\.type) == ["rms_spot", "efl", "bfl", "combine"])
        #expect(m.operands[3].kind["args"]?.intArray == [1, 0])
        // Duplicate keeps the parameters under a new identity.
        let copies = m.duplicate([spot])
        #expect(copies.count == 1 && m.operands[1].type == "rms_spot" && m.operands[1].id != spot)
        #expect(m.operands[4].kind["args"]?.intArray == [2, 0])
        // Deleting a referenced operand drops the reference.
        m.delete([efl])
        #expect(m.operands.map(\.type) == ["rms_spot", "rms_spot", "bfl", "combine"])
        #expect(m.operands[3].kind["args"]?.intArray == [0])
        // The spec is the engine's MeritFunctionSpec and reads back.
        let back = MeritSettings(section: m.section)
        #expect(back.section.differences(from: m.section).isEmpty, "\(back.section.differences(from: m.section))")
        #expect(back == m && back.operands.count == 4)
        #expect(m.jobSettings["merit"]?["operands"]?.array?.count == 4 && m.jobSettings["default_merit"] == nil)
    }

    /// Operands keep the engine's JSON form, including kinds the app does
    /// not edit, and new ones get the descriptor's defaults.
    @Test func operandJSONRoundTripsAndChecksItself() throws {
        let json = try JSONValue.parse(
            Data(
                #"{"kind":{"type":"ray_aberration","axis":"y","px":0,"py":1,"reference":"centroid","group":{"sampling":{"rings":3,"arms":6},"scope":"monochromatic"}},"target":0,"weight":0.25,"comparison":"equal","role":"residual","field":{"field":"index","index":1},"wavelength":{"wavelength":"primary"},"config":0,"enabled":true}"#
                    .utf8))
        let op = try #require(MeritOperand(json: json))
        #expect(op.json == json.withLabelRemoved)
        #expect(op.descriptor?.singleField == true && op.localIssues(fieldCount: 3, wavelengthCount: 3).isEmpty)
        var bad = op
        bad.field = .object(["field": .string("all")])
        #expect(bad.localIssues(fieldCount: 3, wavelengthCount: 3) == ["Needs one field, not all fields."])
        let mtf = MeritOperand(kind: try d("mtf").newKind)
        #expect(mtf.kind["frequency"]?.doubleValue == 30 && mtf.kind["orientation"]?.string == "average")
        let thick = MeritOperand(kind: try d("centre_thickness").newKind)
        #expect(thick.localIssues(fieldCount: 1, wavelengthCount: 1) == ["Surface is required."])
        #expect(MeritOperand(kind: ["type": .string("mtf")]).residual(0.3) == 0.3 - 0)
        var lower = mtf
        lower.comparison = .greaterThan
        lower.target = 0.5
        #expect(lower.residual(0.6) == 0 && lower.residual(0.3) < 0)
    }

    /// The live evaluation's values, contributions and shares come from the
    /// optimiser's start columns.
    @Test func contributionsComeFromTheStartValues() throws {
        let json = try JSONValue.parse(
            Data(
                #"{"start":{"merit":0.05,"violation":0},"merit_spec":{"operands":[]},"operands":[{"index":0,"kind":"efl","role":"residual","comparison":"equal","target":50,"weight":0.01,"enabled":true,"start":51,"error":null},{"index":1,"kind":"rms_spot","role":"residual","comparison":"less_than","target":0.01,"weight":100,"enabled":true,"start":0.03,"error":null},{"index":2,"kind":"mtf","role":"residual","comparison":"greater_than","target":0.5,"weight":1,"enabled":true,"start":0.7},{"index":3,"kind":"efl","role":"constraint","comparison":"equal","target":50,"weight":1,"enabled":true,"start":51},{"index":4,"kind":"centre_thickness","role":"residual","comparison":"equal","target":1,"weight":1,"enabled":true,"start":null,"error":"InvalidSurface"}]}"#
                    .utf8))
        let e = try #require(MeritEvaluation(json: json))
        #expect(e.merit == 0.05)
        #expect(abs(e.rows[0].contribution - 0.01) < 1e-15)
        #expect(abs(e.rows[1].contribution - 0.04) < 1e-12)
        #expect(e.rows[2].contribution == 0, "a satisfied bound contributes nothing")
        #expect(e.rows[3].contribution.isNaN && e.rows[3].isConstraint)
        #expect(abs(e.rows[0].fraction - 0.2) < 1e-9 && abs(e.rows[1].fraction - 0.8) < 1e-9)
        #expect(e.rows[4].error == "InvalidSurface")
        #expect(ContributionBar.percent(0.8) == "80%" && ContributionBar.percent(0.0004) == "<0.1%")
    }

    /// Merit edits are undoable document changes saved in the `merit`
    /// section; the editor's working copy follows undo.
    @Test func meritEditsAreUndoableAndSaved() throws {
        let doc = DesignDocument(system: SampleSystems.cookeTriplet())
        let undo = undoManager()
        let session = OptimizeSession()
        session.load(doc)
        #expect(session.merit.mode == .default && session.merit.holdEFL)
        step(undo) { session.editMerit(doc, "Change Criterion", undoManager: undo) { $0.defaults.criterion = .mtf; $0.defaults.worstField = true } }
        #expect(doc.merit.defaults.criterion == .mtf)
        let crit = session.settings(variables: [])["default_merit"]?["criterion"]
        #expect(crit?["type"]?.string == "mtf" && crit?["worst_field"]?.bool == true && crit?["frequencies"]?.nfArray == [10, 30])
        step(undo) {
            session.editMerit(doc, "Use Custom Merit", undoManager: undo) {
                $0.mode = .custom
                _ = $0.add(OperandCatalog.descriptor("efl")!)
            }
        }
        #expect(doc.merit.mode == .custom && doc.merit.operands.count == 1)
        #expect(session.settings(variables: [])["merit"] != nil)
        #expect(undo.undoActionName == "Use Custom Merit")
        // The section is in the saved project.
        #expect(doc.savedProject.sections["merit"]?["mode"]?.string == "custom")
        undo.undo()
        session.load(doc)
        #expect(session.merit.mode == .default && session.merit.operands.isEmpty)
        undo.undo()
        session.load(doc)
        #expect(session.merit.defaults.criterion == .rmsSpot)
        undo.redo()
        session.load(doc)
        #expect(session.merit.defaults.criterion == .mtf)
    }

    /// New variables get sensible bounds; bounds edit and pin undoably;
    /// pinned variables are held, not sent.
    @Test func variablesGetBoundsAndPin() throws {
        let doc = DesignDocument(system: SampleSystems.cookeTriplet())
        let undo = undoManager()
        let s = doc.system
        let glass = ParamPath.thickness(surface: s.surfaces[1].id)  // N-SK16
        let air = ParamPath.thickness(surface: s.surfaces[2].id)
        let back = ParamPath.thickness(surface: s.surfaces[s.imageIndex - 1].id)
        let r1 = ParamPath.curvature(surface: s.surfaces[1].id)
        for p in [glass, air, back, r1] { step(undo) { doc.setVariable(p, true, undoManager: undo) } }
        #expect(doc.variable(glass)?.lower == DesignVariable.glassMinimum && doc.variable(glass)?.upper == nil)
        #expect(doc.variable(air)?.lower == 0 && doc.variable(back)?.lower == 0)
        #expect(doc.variable(r1)?.isBounded == false)
        step(undo) { doc.setBounds(r1, lower: -0.1, upper: 0.1, undoManager: undo) }
        #expect(undo.undoActionName == "Change Bounds" && doc.variable(r1)?.lower == -0.1)
        let value = try #require(doc.system.value(at: air))
        step(undo) { doc.setBounds(air, lower: value, upper: value, undoManager: undo) }
        #expect(undo.undoActionName == "Pin Variable" && doc.variable(air)?.isPinned == true)
        let session = OptimizeSession()
        let sent = session.settings(variables: doc.variables)["variables"]?.array ?? []
        #expect(sent.count == 3, "the pinned variable is held, not varied")
        #expect(sent.contains { $0["param"]?["param"]?.string == "curvature" && $0["upper"]?.doubleValue == 0.1 })
        // The lens table marks bounded and pinned variables.
        let rows = LensRow.rows(
            for: doc.system, variables: Set(doc.variables.map(\.param)),
            bounds: Dictionary(uniqueKeysWithValues: doc.variables.map { ($0.param, $0.isPinned ? .pinned : $0.isBounded ? .bounded : .free) }))
        #expect(rows[2].markers[.thickness]?.bounds == .pinned)
        #expect(rows[1].markers[.radius]?.bounds == .bounded)
        undo.undo()
        #expect(doc.variable(air)?.isPinned == false)
        undo.undo()
        #expect(doc.variable(r1)?.isBounded == false)
        // Pinning every variable blocks the run instead of varying the automatic set.
        let only = [DesignVariable(r1, lower: 0.02, upper: 0.02)]
        #expect(session.variableProblem(only) != nil && session.settings(variables: only)["variables"] == nil)
        #expect(DesignVariable(r1, lower: 0, upper: 1).outOfBounds(2) && !DesignVariable(r1, lower: 0, upper: 1).outOfBounds(1))
    }

    /// The default merit settings write the engine's `DefaultMeritSpec`.
    @Test func defaultMeritEncodesTheGenerator() {
        var d = DefaultMeritSettings()
        d.fields = [0, 2]
        d.thickness = false
        d.squeezeTarget = 2
        let j = d.json
        #expect(j["fields"]?.intArray == [0, 2] && j["wavelengths"] == .null && j["thickness"] == .null)
        #expect(j["squeeze_target"]?.doubleValue == 2 && j[path: "sampling", "rings"]?.int == 3)
        #expect(DefaultMeritSettings(json: j) == d)
        #expect(d.summary.contains("squeeze 2"))
    }
}

extension JSONValue {
    /// The operand JSON as `MeritOperand.json` writes it (no empty label).
    fileprivate var withLabelRemoved: JSONValue {
        guard var o = object else { return self }
        if o["label"]?.string == "" { o["label"] = nil }
        return .object(o)
    }
}
