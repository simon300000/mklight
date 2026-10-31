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

    /// The live evaluation reads `merit_evaluation`: the merit, status,
    /// each operand's value, residual, contribution and share as the engine
    /// reports them, failed operands with their error, and the expanded spec.
    @Test func evaluationReadsTheEngineReport() throws {
        let json = try JSONValue.parse(
            Data(
                #"{"merit":0.05,"violation":0,"status":{"status":"degraded","failures":1},"backend":{"reported_on":"cpu-f64-reference"},"merit_spec":{"schema_version":1,"name":"default","on_failure":{"mode":"penalty","value":1000},"operands":[{"kind":{"type":"efl"},"target":50,"weight":0.01},{"kind":{"type":"efl"},"target":51,"weight":1,"role":"constraint","label":"hold EFL"}]},"operands":[{"index":0,"kind":"efl","role":"residual","comparison":"equal","target":50,"weight":0.01,"enabled":true,"start":51,"value":51,"residual":1,"contribution":0.01,"contribution_fraction":0.2,"error":null},{"index":1,"kind":"rms_spot","role":"residual","comparison":"less_than","target":0.01,"weight":100,"enabled":true,"value":0.03,"residual":0.02,"contribution":0.04,"contribution_fraction":0.8},{"index":2,"kind":"mtf","role":"residual","comparison":"greater_than","target":0.5,"weight":1,"enabled":true,"value":0.7,"residual":0,"contribution":0,"contribution_fraction":0},{"index":3,"kind":"efl","role":"constraint","comparison":"equal","target":50,"weight":1,"enabled":true,"value":51,"residual":null,"contribution":null,"contribution_fraction":null},{"index":4,"kind":"centre_thickness","role":"residual","comparison":"equal","target":1,"weight":1,"enabled":true,"value":null,"residual":null,"contribution":null,"contribution_fraction":null,"error":"InvalidSurface"}]}"#
                    .utf8))
        let e = try #require(MeritEvaluation(json: json))
        #expect(e.merit == 0.05 && e.violation == 0 && e.status == .degraded(failures: 1) && e.reportedOn == "cpu-f64-reference")
        #expect(e.rows[0].value == 51 && e.rows[0].residual == 1 && abs(e.rows[0].contribution - 0.01) < 1e-15)
        #expect(abs(e.rows[1].contribution - 0.04) < 1e-12 && e.rows[1].fraction == 0.8)
        #expect(e.rows[2].contribution == 0, "a satisfied bound contributes nothing")
        #expect(e.rows[3].contribution.isNaN && e.rows[3].fraction.isNaN && e.rows[3].isConstraint)
        #expect(e.rows[4].error == "InvalidSurface" && e.rows[4].value.isNaN)
        #expect(e.row(1, kind: "rms_spot") != nil && e.row(1, kind: "efl") == nil, "a stale row of another operand is not shown")
        #expect(e.specOperands.count == 2 && e.specOptions["name"]?.string == "default" && e.specOptions["on_failure"] != nil)
        #expect(OptimizeSession.isEFLHold(e.specOperands[1]) && !OptimizeSession.isEFLHold(e.specOperands[0]))
        #expect(ContributionBar.percent(0.8) == "80%" && ContributionBar.percent(0.0004) == "<0.1%")
        // The spec's failure policy survives the saved section.
        var m = MeritSettings()
        m.mode = .custom
        m.operands = e.specOperands
        m.specOptions = e.specOptions
        let back = MeritSettings(section: m.section)
        #expect(back == m && back.spec["on_failure"]?["value"]?.doubleValue == 1000 && back.spec["schema_version"]?.int == 1)
    }

    /// The live evaluation runs `merit_evaluation` with the optimiser's
    /// merit, variables and algorithm; the mock engine answers it.
    @Test func liveEvaluationUsesMeritEvaluation() async throws {
        let system = SampleSystems.cookeTriplet()
        let doc = DesignDocument(system: system)
        let w = Workspace(engine: MockEngine(simulatedLatency: .zero))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        let clock = ContinuousClock()
        let end = clock.now + .seconds(20)
        while w.snapshot?.system != system && clock.now < end { try await Task.sleep(for: .milliseconds(5)) }
        let session = OptimizeSession()
        session.load(doc)
        session.algorithm = "de"
        let s = session.evaluationSettings(variables: [], system: system)
        #expect(s["algorithm"]?.string == "de" && s["max_iterations"] == nil && s["default_merit"] != nil)
        session.scheduleEvaluation(in: w, variables: [], system: system, delay: .zero)
        while session.evaluation == nil && session.evaluationError == nil && clock.now < end { try await Task.sleep(for: .milliseconds(5)) }
        let e = try #require(session.evaluation, "\(session.evaluationError ?? "")")
        #expect(e.merit > 0 && e.rows.contains { $0.isConstraint && $0.label == "hold EFL" })
        // Converting keeps the generated operands but leaves the EFL hold to
        // the switch, and evaluates to the same merit.
        await session.convertToCustom(in: w, document: doc, variables: [], undoManager: nil)
        #expect(doc.merit.mode == .custom && !doc.merit.operands.contains(where: OptimizeSession.isEFLHold))
        #expect(doc.merit.operands.count == e.rows.count - 1 && doc.merit.specOptions["name"]?.string == "default")
        session.scheduleEvaluation(in: w, variables: [], system: system, delay: .zero)
        while (session.evaluation == e || session.evaluation == nil) && clock.now < end { try await Task.sleep(for: .milliseconds(5)) }
        let custom = try #require(session.evaluation)
        #expect(abs(custom.merit - e.merit) <= 1e-12 * max(e.merit, 1))
    }

    /// A parameter a solve sets cannot become a variable; a variable left
    /// on one is flagged, left out of the run, and blocks it when it is the
    /// only one. `curvature` and `shape_param curvature_y` name the same
    /// parameter of a toroid.
    @Test func solvedParametersAreNotVariables() throws {
        var system = SampleSystems.cookeTriplet()
        let r2 = ParamPath.curvature(surface: system.surfaces[2].id)
        let r1 = ParamPath.curvature(surface: system.surfaces[1].id)
        let doc = DesignDocument(system: system)
        let undo = undoManager()
        step(undo) { doc.setVariable(r2, true, undoManager: undo) }
        step(undo) { doc.setVariable(r1, true, undoManager: undo) }
        system.solves = [Solve(target: r2, kind: .marginalRayAngle(slope: 0))]
        step(undo) { doc.replace(system: system, actionName: "Add Solve", undoManager: undo) }
        #expect(doc.variableRefusal(r2) != nil && doc.variableRefusal(r1) == nil)
        #expect(doc.isVariable(r2), "an existing variable stays, flagged")
        let session = OptimizeSession()
        let sent = session.settings(variables: doc.variables, system: doc.system)["variables"]?.array ?? []
        #expect(sent.count == 1 && sent.first?["param"]?["surface"]?.int == Int(system.surfaces[1].id.rawValue))
        #expect(session.variableProblem(doc.variables, system: doc.system) == nil)
        let onlySolved = doc.variables.filter { $0.param == r2 }
        #expect(session.variableProblem(onlySolved, system: doc.system)?.contains("solve") == true)
        // Removing it works; making it a variable again is refused.
        step(undo) { doc.setVariable(r2, false, undoManager: undo) }
        step(undo) { doc.setVariable(r2, true, undoManager: undo) }
        #expect(!doc.isVariable(r2))
        // The table marks a solved variable through the canonical path.
        let rows = LensRow.rows(for: doc.system, variables: Set(doc.variables.map { doc.system.canonical($0.param) }))
        #expect(rows[2].markers[.radius]?.solve != nil && rows[2].markers[.radius]?.isVariable == false)
        // Toroid: the radius column is the `curvature` path (the y section);
        // an older `curvature_y` variable is the same parameter.
        var t = SampleSystems.cookeTriplet()
        let id = t.surfaces[1].id
        t.surfaces[1].shape = .toroid(curvatureX: 0.01, curvatureY: 0.02, conicY: 0, yAsphere: [])
        let ty = ParamPath.shapeParam(surface: id, parameter: .curvatureY)
        #expect(LensColumn.radius.param(for: t.surfaces[1]) == .curvature(surface: id))
        #expect(t.canonical(ty) == .curvature(surface: id) && t.canonical(.shapeParam(surface: id, parameter: .curvatureX)) != .curvature(surface: id))
        let tdoc = DesignDocument(system: t)
        tdoc.setVariable(ty, true, undoManager: nil)
        #expect(tdoc.isVariable(.curvature(surface: id)))
        tdoc.setVariable(.curvature(surface: id), true, undoManager: nil)
        #expect(tdoc.variables.count == 1, "no second variable for the same parameter")
        t.solves = [Solve(target: .curvature(surface: id), kind: .fNumber(value: 4))]
        #expect(t.isSolved(ty) && ParamPath.curvature(surface: id).label(in: t) == "Radius y 1")
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
