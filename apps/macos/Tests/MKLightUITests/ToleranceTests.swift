import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The tolerance editor's model: presets and their expansion over the
/// elements, rows to engine operands (fringes, ΔR, %, arc minutes), edits
/// with undo, the document round trip, compensators, the criterion's
/// display and the result's statistics, previews and export.
@Suite @MainActor struct ToleranceTests {
    private let cooke = SampleSystems.cookeTriplet()

    /// The Cooke triplet with fixed clear apertures (fringes need them).
    private var context: ToleranceContext {
        var sd: [SurfaceID: Double] = [:]
        for s in cooke.surfaces { sd[s.id] = 8 }
        return ToleranceContext(system: cooke, semiDiameters: sd)
    }

    private func id(_ i: Int) -> SurfaceID { cooke.surfaces[i].id }

    // MARK: Presets and expansion

    @Test func presetsSetEveryValue() {
        var g = ToleranceGenerator()
        #expect(g.preset == .precision && g.radiusFringes == 3 && g.thicknessMM == 0.05)
        g.apply(.commercial)
        #expect(g.radiusFringes == 5 && g.thicknessMM == 0.2 && g.decentreMM == 0.1 && g.tiltArcmin == 3 && g.index == 0.001 && g.abbePercent == 1)
        g.apply(.highPrecision)
        #expect(g.radiusFringes == 1 && g.thicknessMM == 0.025 && g.decentreMM == 0.01 && g.tiltArcmin == 0.5 && g.index == 0.0002)
        #expect(ToleranceGenerator(json: g.json) == g)
    }

    /// Every lens surface's power, decentre and tilt in x and y; every gap
    /// inside the lens but not the back focus; index and Abbe only for model
    /// glasses.
    @Test func generatorExpandsOverTheElements() {
        let rows = ToleranceGenerator().rows(for: cooke)
        func count(_ k: ToleranceKind) -> Int { rows.filter { $0.kind == k }.count }
        #expect(count(.radius) == 6 && count(.decenterX) == 6 && count(.decenterY) == 6 && count(.tiltX) == 6 && count(.tiltY) == 6)
        #expect(count(.thickness) == 6, "S1–S6 gaps (the stop's included); the back focus S7 is the compensator")
        #expect(!rows.contains { $0.kind == .thickness && $0.surface == id(7) })
        #expect(count(.index) == 0 && count(.abbe) == 0, "the Cooke's glasses are catalogue glasses")
        #expect(ToleranceGenerator.catalogueGlasses(cooke) == ["N-SK16", "F2"])
        var model = cooke
        model.surfaces[1].material = .model(nd: 1.62, vd: 60.3, dpgf: 0)
        let m = ToleranceGenerator().rows(for: model)
        #expect(m.filter { $0.kind == .index }.map(\.surface) == [id(1)] && m.filter { $0.kind == .abbe }.count == 1)
        // Grouped by element: L1 is S1–S2, the air after it its own group.
        let groups = LensElements.groups(cooke)
        #expect(groups.map(\.title) == ["L1", "L2", "L3"] && groups[0].surfaces == [1, 2] && groups[2].surfaces == [6, 7] && groups[0].glasses == ["N-SK16"])
        #expect(LensElements.groupTitle(of: ToleranceRow(.thickness, surface: id(5), minus: 0.05), in: cooke, groups: groups) == "Stop S5")
        let t2 = ToleranceRow(.thickness, surface: id(2), minus: 0.05)
        #expect(LensElements.groupTitle(of: t2, in: cooke, groups: groups) == "Air after L1")
        #expect(LensElements.groupTitle(of: ToleranceRow(.radius, surface: id(2), minus: 3), in: cooke, groups: groups) == "L1")
    }

    @Test func rowsBecomeEngineOperands() throws {
        let c = context
        let curv = try #require(cooke.value(at: .curvature(surface: id(1))))
        // 3 fringes at 632.8 nm over h = 8 mm: Δc = 3 λ / h².
        let fr = try ToleranceRow(.radius, surface: id(1), minus: 3).engineLimits(in: c).get()
        #expect(abs(fr.0 - 3 * 0.6328e-3 / 64) < 1e-15 && fr.0 == fr.1)
        // ΔR = 0.1 mm: c in [1/(R + 0.1), 1/(R − 0.1)].
        let r = 1 / curv
        let mm = try ToleranceRow(.radius, surface: id(1), minus: 0.1, unit: .radiusMM).engineLimits(in: c).get()
        #expect(abs(mm.0 - (curv - 1 / (r + 0.1))) < 1e-15 && abs(mm.1 - (1 / (r - 0.1) - curv)) < 1e-15)
        let pc = try ToleranceRow(.radius, surface: id(1), minus: 0.5, unit: .percent).engineLimits(in: c).get()
        #expect(abs(pc.1 - (1 / (r - abs(r) * 0.005) - curv)) < 1e-15)
        // A negative radius keeps its sign.
        let c4 = try #require(cooke.value(at: .curvature(surface: id(4))))
        let neg = try ToleranceRow(.radius, surface: id(4), minus: 0.2, unit: .radiusMM).engineLimits(in: c).get()
        #expect(neg.0 > 0 && neg.1 > 0 && abs(neg.1 - (1 / (1 / c4 - 0.2) - c4)) < 1e-15)
        // Tilts are sent in degrees, decentres in mm.
        #expect(try ToleranceRow(.tiltX, surface: id(1), minus: 1.5).engineLimits(in: c).get() == (0.025, 0.025))
        let op = try ToleranceRow(.decenterY, surface: id(3), minus: 0.02, plus: 0.03, distribution: .normal(sigmas: 2)).operand(in: c).get()
        #expect(op["param"] == .object(["param": .string("decenter_y"), "surface": .integer(Int64(id(3).rawValue))]))
        #expect(op["minus"]?.nf == 0.02 && op["plus"]?.nf == 0.03 && op["label"]?.string == "Decentre y S3")
        #expect(op["distribution"] == .object(["type": .string("truncated_normal"), "sigmas": .number(2)]))
        // Problems stay with the row.
        if case .failure(let e) = ToleranceRow(.index, surface: id(1), minus: 0.001).engineLimits(in: c) { #expect(e.isSkipped) } else { Issue.record("catalogue glass") }
        if case .failure(let e) = ToleranceRow(.radius, surface: id(1), minus: -1).engineLimits(in: c) { #expect(!e.isSkipped) } else { Issue.record("negative") }
        if case .failure = ToleranceRow(.radius, surface: id(1), minus: abs(r) + 1, unit: .radiusMM).engineLimits(in: c) {} else { Issue.record("ΔR ≥ |R|") }
        var noAperture = c
        noAperture.semiDiameters = [:]
        if case .failure = ToleranceRow(.radius, surface: id(1), minus: 3).engineLimits(in: noAperture) {} else { Issue.record("no aperture") }
        // Abbe in % of V_d on a model glass.
        var model = cooke
        model.surfaces[1].material = .model(nd: 1.62, vd: 60, dpgf: 0)
        let ab = try ToleranceRow(.abbe, surface: id(1), minus: 0.5).engineLimits(in: ToleranceContext(system: model)).get()
        #expect(abs(ab.0 - 0.3) < 1e-12)
    }

    @Test func unitsConvertKeepingTheEngineLimits() throws {
        let c = context
        let row = ToleranceRow(.radius, surface: id(1), minus: 3)
        let limits = try row.engineLimits(in: c).get()
        for u in [ToleranceUnit.radiusMM, .percent] {
            let r = try #require(row.converted(to: u, in: c))
            let l = try r.engineLimits(in: c).get()
            #expect(abs(l.0 - limits.0) < 1e-12 * max(1, limits.0) && abs(l.1 - limits.1) < 1e-12 * max(1, limits.1), "\(u)")
            let back = try #require(r.converted(to: .fringes, in: c))
            #expect(abs(back.minus - 3) < 1e-6 && abs(back.plus - 3) < 1e-6)
        }
        let tilt = try #require(ToleranceRow(.tiltY, surface: id(2), minus: 1).converted(to: .degrees, in: c))
        #expect(abs(tilt.minus - 1.0 / 60) < 1e-9 && tilt.unit == .degrees)
        // A flat surface has no ΔR.
        var flat = cooke
        flat.surfaces[1].shape = .plane
        #expect(ToleranceRow(.radius, surface: id(1), minus: 3).converted(to: .radiusMM, in: ToleranceContext(system: flat, semiDiameters: c.semiDiameters)) == nil)
    }

    /// Decentred and tilted surfaces get a placement that returns to the
    /// axis after them, so the rest of the system stays in place.
    @Test func placementsReturnAfterTheirSurface() {
        var t = ToleranceSettings()
        t.rows = [ToleranceRow(.decenterY, surface: id(3), minus: 0.02), ToleranceRow(.radius, surface: id(1), minus: 3)]
        let p = t.preparedSystem(cooke)
        #expect(p.surfaces[3].placement == Placement(restoreAfter: true))
        #expect(p.surfaces[1].placement == nil)
        var tilted = cooke
        tilted.surfaces[3].placement = Placement(tiltX: 2)
        #expect(t.preparedSystem(tilted).surfaces[3].placement == Placement(tiltX: 2), "a placement of the design is left alone")
    }

    // MARK: Editing, undo and the document

    @Test func editsAreUndoableAndSaved() throws {
        let doc = DesignDocument(system: cooke)
        let undo = UndoManager()
        undo.groupsByEvent = false
        let session = ToleranceSession()
        session.load(doc)
        func step(_ name: String, _ f: (inout ToleranceSettings) -> Void) {
            undo.beginUndoGrouping()
            session.edit(doc, name, context: context, undoManager: undo, f)
            undo.endUndoGrouping()
        }
        #expect(!session.settings.rows.isEmpty && session.settings.compensators.count == 1, "a new design starts from the default set")
        let n = session.settings.rows.count
        let first = session.settings.rows[0].id, second = session.settings.rows[1].id
        step("Delete Tolerance") { $0.delete([first]) }
        #expect(session.settings.rows.count == n - 1 && doc.tolerances.rows.count == n - 1)
        #expect(undo.undoActionName == "Delete Tolerance")
        var dup: [UUID] = []
        step("Duplicate Tolerance") { dup = $0.duplicate([$0.rows[0].id]) }
        #expect(session.settings.rows.count == n && dup.count == 1 && session.settings.rows[1].id == dup[0])
        step("Change Tolerance") { $0.rows[0].plus = 7 }
        #expect(doc.tolerances.rows[0].plus == 7)
        undo.undo()
        session.load(doc)
        #expect(session.settings.rows[0].plus != 7)
        undo.undo()
        undo.undo()
        session.load(doc)
        #expect(session.settings.rows.count == n && session.settings.rows[1].id == second, "ids of rows still there survive the undo")
        undo.redo()
        session.load(doc)
        #expect(session.settings.rows.count == n - 1)
    }

    @Test func theSetRoundTripsThroughTheProject() throws {
        var t = ToleranceSettings.defaults(for: cooke)
        t.rows.append(ToleranceRow(.other, surface: id(2), minus: 0.01, param: .conic(surface: id(2))))
        t.rows[0].distribution = .endPoint
        t.compensators.append(ToleranceCompensator(.decenterX(surface: id(3)), range: nil))
        t.criterion.kind = .mtf
        t.criterion.defaults.mtfFrequencies = [10, 30]
        t.criterion.threshold = 0.4
        t.mode = .monteCarlo
        t.trials = 500
        t.seed = 3
        let doc = DesignDocument(system: cooke)
        doc.setTolerances(t, context: context, actionName: "Edit", undoManager: nil)
        let data = try doc.savedProject.encoded()
        let back = DesignDocument(project: try ProjectFile.decode(data))
        #expect(back.tolerances == t)
        // The section holds the engine's spec as expanded.
        let section = try #require(back.project.sections["tolerances"])
        #expect(section["schema_version"]?.int == 1)
        let ops = try #require(section["operands"]?.array)
        let first0 = try t.rows[0].engineLimits(in: context).get().0
        #expect(ops.count == t.rows.count && ops[0]["minus"]?.nf == first0)
        #expect(section[path: "monte_carlo", "trials"]?.int == 500 && section[path: "monte_carlo", "seed"]?.int == 3)
        #expect(abs((section[path: "monte_carlo", "threshold"]?.nf ?? 0) - 0.36) < 1e-12, "MTF ≥ 0.4 is a shortfall² ≤ 0.36")
        #expect(section["compensators"]?.array?.count == 2)
        // A spec written elsewhere (no app rows) reads its operands as rows.
        var foreign = section.object!
        foreign["app"] = nil
        let f = ToleranceSettings(section: .object(foreign), system: cooke)
        #expect(f.rows.count == t.rows.count && f.rows.contains { $0.kind == .radius && $0.unit == .radiusMM })
        let r0 = try #require(f.rows.first { $0.kind == .radius && $0.surface == t.rows[0].surface })
        let a = try r0.engineLimits(in: context).get(), b = try t.rows[0].engineLimits(in: context).get()
        #expect(abs(a.0 - b.0) < 1e-12 && abs(a.1 - b.1) < 1e-12)
    }

    // MARK: Compensators

    @Test func backFocusIsTheDefaultCompensator() throws {
        let t = ToleranceSettings.defaults(for: cooke)
        let bf = try #require(t.compensators.first)
        #expect(bf.param == .thickness(surface: id(7)) && bf.range == 0.5 && bf.title(in: cooke) == "Back focus")
        let cand = ToleranceCompensator.candidates(cooke, variables: [.curvature(surface: id(2)), .thickness(surface: id(7))])
        #expect(cand.map(\.group) == ["Back focus", "Air spaces", "Design variables", "Surfaces"])
        #expect(cand[0].params == [bf.param])
        #expect(cand[1].params == [.thickness(surface: id(2)), .thickness(surface: id(4)), .thickness(surface: id(5))])
        #expect(cand[2].params == [.curvature(surface: id(2))], "the back focus is listed once")
        #expect(cand[3].params.contains(.decenterX(surface: id(3))) && !cand[3].params.contains(.curvature(surface: id(2))))
        let (s, _) = t.jobSettings(in: context, merit: MeritSettings())
        #expect(s["compensators"] == .array([.object(["param": .object(["param": .string("thickness"), "surface": .integer(Int64(id(7).rawValue))]), "range": .number(0.5)])]))
        #expect(ToleranceCompensator(json: ToleranceCompensator(.tiltX(surface: id(1)), range: nil).json)?.range == nil)
    }

    @Test func criterionSettingsAndDisplay() {
        var c = ToleranceCriterion()
        var s = c.jobSettings(merit: MeritSettings())
        #expect(s["default_merit"]?[path: "criterion", "type"]?.string == "rms_spot" && s["default_merit"]?["thickness"] == .null)
        c.kind = .mtf
        s = c.jobSettings(merit: MeritSettings())
        #expect(s["default_merit"]?[path: "criterion", "target"]?.nf == 1, "the MTF shortfall from 1")
        var custom = MeritSettings()
        custom.mode = .custom
        c.kind = .designMerit
        #expect(c.jobSettings(merit: custom)["criterion"] == custom.spec && c.display(merit: custom) == .merit)
        // Spot: √merit in µm; MTF: 1 − √merit.
        #expect(abs(CriterionDisplay.spot.value(4e-4) - 20) < 1e-9 && abs(CriterionDisplay.spot.criterion(20) - 4e-4) < 1e-15)
        #expect(abs(CriterionDisplay.mtf.value(0.09) - 0.7) < 1e-12 && CriterionDisplay.mtf.higherIsBetter)
        #expect(CriterionDisplay.spot.formatChange(-0.001) == "\u{2007}0.00" && CriterionDisplay.spot.formatChange(1.234) == "+1.23")
    }

    // MARK: Results

    private func result(compensated: Bool = true) -> ToleranceResult {
        func op(_ k: Int, _ p: ParamPath, _ cm: Double, _ cp: Double) -> JSONValue {
            .object([
                "index": .integer(Int64(k)), "label": .string(""), "param": (try? JSONValue.from(p)) ?? .null, "minus": .number(1e-4), "plus": .number(1e-4),
                "criterion_minus": .number(cm), "criterion_plus": .number(cp), "change_minus": .number(cm - 4e-4), "change_plus": .number(cp - 4e-4),
                "worst": .number(max(cm, cp) - 4e-4), "ok": .bool(true),
                "compensators_minus": .array(compensated ? [.number(42.1 - 0.1 * Double(k))] : []),
                "compensators_plus": .array(compensated ? [.number(42.1 + 0.2)] : []),
            ])
        }
        let json: JSONValue = .object([
            "status": .string("complete"), "stopped": .bool(false),
            "compensators": .array(compensated ? [ToleranceCompensator(.thickness(surface: id(7)), range: 0.5).json] : []),
            "sensitivity": .object([
                "nominal": .object(["criterion": .number(4e-4), "compensators": .array([.number(42.1)])]),
                "operands": .array([
                    op(0, .curvature(surface: id(1)), 9e-4, 5e-4), op(1, .thickness(surface: id(1)), 4.1e-4, 4.2e-4),
                    op(2, .decenterY(surface: id(3)), 4e-4, 6e-4),
                ]),
                "ranking": .array([.integer(0), .integer(2), .integer(1)]), "rss_change": .number(5.4e-4), "predicted_criterion": .number(9.4e-4),
            ]),
            "monte_carlo": .object([
                "trials": .integer(5), "seed": .integer(3), "criteria": .array([4e-4, 5e-4, 6e-4, 8e-4, 1e-3].map(JSONValue.number)),
                "mean": .number(6.6e-4), "std_dev": .number(2e-4), "min": .number(4e-4), "max": .number(1e-3),
                "compensator_mean": .array([.number(42.15)]), "compensator_std": .array([.number(0.05)]),
            ]),
        ])
        return ToleranceResult(json: json, system: cooke, display: .spot)
    }

    @Test func resultsRankShareAndPercentiles() {
        let r = result()
        #expect(r.sensitivities.map(\.id) == [0, 2, 1])
        #expect(r.sensitivities[0].label == "Radius 1", "labels from the parameter when the engine has none")
        let shares = r.sensitivities.map { r.share($0) }
        #expect(abs(shares.reduce(0, +) - 1) < 1e-12 && shares[0] > 0.8)
        #expect(r.worstOffenders == [0])
        #expect(r.percentile(50) == 6e-4 && abs(r.percentile(90) - 9.2e-4) < 1e-15 && r.percentile(100) == 1e-3)
        #expect(r.cumulative(6e-4) == 0.6)
        let range = r.compensatorRange(0)
        #expect(abs((range?.0 ?? 0) + 0.2) < 1e-9 && abs((range?.1 ?? 0) - 0.2) < 1e-9)
        #expect(ToleranceResult.levels == [50, 90, 97.7])
    }

    /// A preview is the build the sensitivity evaluated: the tolerance at
    /// its limit and the compensator at its re-optimised value.
    @Test func previewsRebuildTheEvaluatedSystem() async throws {
        let doc = DesignDocument(system: cooke)
        let w = Workspace(engine: MockEngine(simulatedLatency: .zero))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: cooke, immediately: true)
        try await waitUntil { w.snapshot != nil }
        let session = ToleranceSession()
        session.load(doc)
        session.edit(doc, "Set", context: context, undoManager: nil) {
            $0.rows = [ToleranceRow(.radius, surface: id(1), minus: 0.1, unit: .radiusMM), ToleranceRow(.decenterY, surface: id(3), minus: 0.02)]
            $0.mode = .both
            $0.trials = 20
        }
        session.start(in: w, context: context, merit: doc.merit)
        try await waitUntil { session.state == .finished }
        let r = try #require(session.result)
        #expect(r.sensitivities.count == 2 && session.resultRows.count == 2)
        let k = try #require(r.sensitivities.first { $0.param == .decenterY(surface: id(3)) })
        let s = try #require(session.trialSystem(k.id, plus: true))
        #expect(s.surfaces[3].placement?.decenterY == 0.02 && s.surfaces[3].placement?.restoreAfter == true)
        #expect(s.value(at: .thickness(surface: id(7))) == k.compensatorsPlus.first)
        session.preview(k.id, plus: true, in: w)
        #expect(w.isPreviewing && w.previewTitle?.hasPrefix("Tolerance: Decentre y S3 at +0.02") == true)
        #expect(doc.system == cooke, "a preview never changes the design")
        session.preview(nil, in: w)
        #expect(!w.isPreviewing)
        // Stop keeps what completed; Cancel discards.
        session.start(in: w, context: context, merit: doc.merit)
        session.cancel()
        #expect(session.state == .setup)
    }

    @Test func summaryAndCSVExport() {
        let r = result()
        let rows = [
            ToleranceRow(.radius, surface: id(1), minus: 3), ToleranceRow(.thickness, surface: id(1), minus: 0.05),
            ToleranceRow(.decenterY, surface: id(3), minus: 0.02),
        ]
        let text = ToleranceReport.summary(r, rows: rows, system: cooke, title: "Cooke")
        #expect(text.contains("Criterion: RMS spot (µm)") && text.contains("Nominal: 20.0") && text.contains("Radius S1 ±3 fr"))
        #expect(text.contains("50 % of builds at most: 24.5") && text.contains("Compensator Back focus: moved -0.2 to +0.2 mm of ±0.5"))
        let csv = ToleranceReport.sensitivityCSV(r, rows: rows, system: cooke).split(separator: "\n")
        #expect(csv.count == 4 && csv[0].hasPrefix("rank,tolerance,minus,plus,unit") && csv[1].hasPrefix("1,Radius S1,3,3,fr,Radius 1,"))
        #expect(csv[2].hasPrefix("2,Decentre y S3,0.02,0.02,mm"))
        let trials = ToleranceReport.trialsCSV(r).split(separator: "\n")
        #expect(trials.count == 6 && trials[0] == "trial,criterion,rms_spot_um" && trials[1].hasPrefix("1,0.0004,20"))
    }

    @Test func runIsBlockedByBrokenRowsOnly() {
        let doc = DesignDocument(system: cooke)
        let session = ToleranceSession()
        session.load(doc)
        session.edit(doc, "Set", context: context, undoManager: nil) {
            $0.rows = [ToleranceRow(.radius, surface: id(1), minus: 3), ToleranceRow(.index, surface: id(1), minus: 0.001)]
        }
        #expect(session.runProblem(context: context) == nil, "a catalogue glass's index is skipped, not an error")
        let (s, ids) = session.settings.jobSettings(in: context, merit: MeritSettings())
        #expect(s["tolerances"]?.array?.count == 1 && ids == [session.settings.rows[0].id])
        session.edit(doc, "Break", context: context, undoManager: nil) { $0.rows[0].minus = -1 }
        #expect(session.runProblem(context: context) == "Fix the tolerance marked in red first.")
    }
}
