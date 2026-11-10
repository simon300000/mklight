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
        // The engine's V_d of N-SK16 and F2 (Abbe tolerances on catalogue glasses are in % of it).
        return ToleranceContext(system: cooke, semiDiameters: sd, abbeNumbers: [id(1): 60.32, id(3): 36.37, id(6): 60.32])
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
        #expect(count(.irregularityAstigmatism) == 6 && count(.irregularitySpherical) == 6, "every powered lens surface")
        #expect(count(.thickness) == 6, "S1–S6 gaps (the stop's included); the back focus S7 is the compensator")
        #expect(!rows.contains { $0.kind == .thickness && $0.surface == id(7) })
        // Index and Abbe of every glass: catalogue glasses through their melt offset.
        #expect(rows.filter { $0.kind == .index }.map(\.surface) == [id(1), id(3), id(6)] && count(.abbe) == 3)
        #expect(rows.first { $0.kind == .index }?.path(in: cooke) == .materialDnD(surface: id(1)))
        #expect(rows.first { $0.kind == .abbe }?.path(in: cooke) == .materialDvD(surface: id(1)))
        // Each element moves as one: decentre and tilt in x and y.
        for k in ToleranceKind.addableElement { #expect(count(k) == 3, "\(k)") }
        let l2 = rows.filter { $0.kind.isElement && $0.surface == id(3) }
        #expect(l2.count == 4 && l2.allSatisfy { $0.last == id(4) && $0.pivot == nil })
        var model = cooke
        model.surfaces[1].material = .model(nd: 1.62, vd: 60.3, dpgf: 0)
        let m = ToleranceGenerator().rows(for: model)
        #expect(m.filter { $0.kind == .index }.map(\.surface) == [id(1), id(3), id(6)])
        #expect(m.first { $0.kind == .index }?.path(in: model) == .modelNd(surface: id(1)))
        // A value of 0 leaves its kind out.
        var g = ToleranceGenerator()
        g.irregularityFringes = 0
        g.elementTiltArcmin = 0
        let fewer = g.rows(for: cooke)
        #expect(!fewer.contains { $0.kind.isElement && $0.kind.component == .tiltX } && !fewer.contains { $0.kind == .irregularitySpherical })
        #expect(fewer.contains { $0.kind == .elementDecenterX })
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
        #expect(op["minus"]?.nf == 0.02 && op["plus"]?.nf == 0.03 && op["label"]?.string == "Surface decentre y S3")
        #expect(op["distribution"] == .object(["type": .string("truncated_normal"), "sigmas": .number(2)]))
        // A catalogue glass's index moves its melt offset; its Abbe in % needs the engine's V_d.
        let nd = try ToleranceRow(.index, surface: id(1), minus: 0.001).operand(in: c).get()
        #expect(nd["param"] == .object(["param": .string("material_dn_d"), "surface": .integer(Int64(id(1).rawValue))]) && nd["minus"]?.nf == 0.001)
        var noVd = c
        noVd.abbeNumbers = [:]
        if case .failure = ToleranceRow(.abbe, surface: id(1), minus: 0.5).engineLimits(in: noVd) {} else { Issue.record("V_d unknown") }
        var withVd = c
        withVd.abbeNumbers = [id(1): 60]
        let vd = try ToleranceRow(.abbe, surface: id(1), minus: 0.5).engineLimits(in: withVd).get()
        #expect(abs(vd.0 - 0.3) < 1e-12 && abs(vd.1 - 0.3) < 1e-12)
        #expect(try ToleranceRow(.abbe, surface: id(1), minus: 0.4, unit: .abbe).engineLimits(in: c).get() == (0.4, 0.4))
        // Air is no glass; a saved melt offset plus the limit stays within ±0.05.
        if case .failure = ToleranceRow(.index, surface: id(2), minus: 0.001).engineLimits(in: c) {} else { Issue.record("air") }
        var melt = c
        melt.system.surfaces[1].material = .catalog(catalog: nil, name: "N-SK16", offset: MaterialOffset(dnD: 0.0495))
        if case .failure = ToleranceRow(.index, surface: id(1), minus: 0.001).engineLimits(in: melt) {} else { Issue.record("offset cap") }
        // Problems stay with the row.
        if case .failure = ToleranceRow(.radius, surface: id(1), minus: -1).engineLimits(in: c) {} else { Issue.record("negative") }
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

    /// Element rows move a surface range rigidly about a pivot: the
    /// engine's `element_placement` path, labelled by element, the moving
    /// surfaces shown; rows starting on one surface share range and pivot.
    @Test func elementGroupsAndPivots() throws {
        let c = context
        var t = ToleranceSettings()
        let dec = ToleranceRow(.elementDecenterX, surface: id(3), minus: 0.03, last: id(4))
        let tilt = ToleranceRow(.elementTiltY, surface: id(3), minus: 1, last: id(4))
        t.rows = [dec, tilt]
        #expect(dec.label(in: cooke) == "Element decentre x L2" && dec.surfaceLabel(in: cooke) == "S3–4")
        #expect(dec.movingSurfaces(in: cooke) == [3, 4] && dec.pivotIndex(in: cooke) == 3, "the pivot defaults to the first vertex")
        let op = try tilt.operand(in: c).get()
        #expect(op["param"]?["param"]?.string == "element_placement" && op["param"]?["last"]?.int == Int(id(4).rawValue))
        #expect(op["param"]?["component"]?.string == "tilt_y" && (op["param"]?["pivot"] ?? .null) == .null)
        #expect(abs((op["plus"]?.nf ?? 0) - 1.0 / 60) < 1e-15, "tilts are sent in degrees")
        // Moving the pivot of one row moves it for the whole group.
        t.setElement(dec.id, first: id(3), last: id(4), pivot: id(4))
        #expect(t.rows.allSatisfy { $0.pivot == id(4) && $0.last == id(4) })
        #expect(t.rows[1].pivotIndex(in: cooke) == 4)
        let pivoted = try t.rows[1].operand(in: c).get()
        #expect(pivoted["param"]?["pivot"]?.int == Int(id(4).rawValue))
        // A cemented range: the doublet-like range S3–S5 moves three surfaces.
        t.setElement(dec.id, first: id(3), last: id(5), pivot: nil)
        #expect(t.rows.allSatisfy { $0.last == id(5) && $0.pivot == nil } && t.rows[0].movingSurfaces(in: cooke) == [3, 4, 5])
        #expect(t.rows[0].label(in: cooke) == "Element decentre x S3–5", "no element spans S3–5")
        #expect(t.issues(in: c).isEmpty)
        // Rows of one first surface that disagree are flagged; a row added there joins the group.
        t.rows[1].last = id(4)
        #expect(t.issues(in: c).count == 2)
        t.rows[1].last = id(5)
        t.add([ToleranceRow(.elementTiltX, surface: id(3), minus: 1, last: id(4), pivot: id(4))])
        #expect(t.rows[2].last == id(5) && t.rows[2].pivot == nil && t.issues(in: c).isEmpty)
        // The range is interior, and the design's own element placement wins.
        if case .failure = ToleranceRow(.elementDecenterX, surface: id(3), minus: 0.03, last: id(8)).engineLimits(in: c) {} else {
            Issue.record("range through the image")
        }
        var placed = c
        placed.system.surfaces[3].elementPlacement = ElementPlacement(last: id(4))
        if case .failure(let e) = t.rows[0].engineLimits(in: placed) { #expect(e.text.contains("S3–4")) } else { Issue.record("design placement") }
        // The saved form and an operand written elsewhere keep range and pivot.
        let back = try #require(ToleranceRow(json: t.rows[0].json))
        #expect(back == t.rows[0] && back.last == id(5))
        let foreign = try #require(ToleranceRow(operand: pivoted, system: cooke))
        #expect(foreign.kind == .elementTiltY && foreign.last == id(4) && foreign.pivot == id(4) && abs(foreign.minus - 1) < 1e-12)
        // No return placements are added any more: the run sends the design as it is.
        #expect(t.jobSettings(in: c, merit: MeritSettings()).rowIDs.count == 3)
    }

    /// Irregularity rows: test fringes (astigmatic and spherical) on the
    /// engine's `irregularity` path, the test wavelength sent once.
    @Test func irregularityRows() throws {
        let c = context
        let astig = ToleranceRow(.irregularityAstigmatism, surface: id(2), minus: 0.5)
        let sph = ToleranceRow(.irregularitySpherical, surface: id(2), minus: 0.25, plus: 0.5)
        #expect(astig.unit == .fringes && astig.limitsText == "±0.5 fr" && astig.label(in: cooke) == "Astig. irregularity S2")
        let a = try astig.operand(in: c).get()
        #expect(a["param"] == .object(["param": .string("irregularity"), "surface": .integer(Int64(id(2).rawValue)), "mode": .string("astigmatism")]))
        #expect(a["minus"]?.nf == 0.5, "fringes go to the engine as they are")
        let s = try sph.operand(in: c).get()
        #expect(s["param"]?["mode"]?.string == "spherical" && s["minus"]?.nf == 0.25 && s["plus"]?.nf == 0.5)
        if case .failure = ToleranceRow(.irregularityAstigmatism, surface: id(8), minus: 0.5).engineLimits(in: c) {} else { Issue.record("image") }
        // The test wavelength is one setting: fringes of power and irregularity use it.
        var t = ToleranceSettings()
        t.rows = [astig, ToleranceRow(.radius, surface: id(1), minus: 3)]
        t.testWavelengthUm = 0.5461
        let (job, _) = t.jobSettings(in: c, merit: MeritSettings())
        #expect(job["irregularity_wavelength_um"]?.nf == 0.5461)
        let power = try #require(job["tolerances"]?.array?[1]["minus"]?.nf)
        #expect(abs(power - 3 * 0.5461e-3 / 64) < 1e-15)
        let back = ToleranceSettings(section: t.section(context: c), system: cooke)
        #expect(back.testWavelengthUm == 0.5461 && back == t)
        // A foreign spec's irregularity operand reads back as a row.
        #expect(ToleranceRow(operand: a, system: cooke)?.kind == .irregularityAstigmatism)
    }

    /// The Monte-Carlo levels and recorded trials: sent, saved, validated.
    @Test func percentilesSetting() throws {
        var t = ToleranceSettings()
        #expect(t.percentiles == [50, 90, 97.7] && t.recordTrials == 200)
        t.rows = [ToleranceRow(.radius, surface: id(1), minus: 3)]
        t.percentiles = [10, 50, 99]
        t.trials = 120
        let (job, _) = t.jobSettings(in: context, merit: MeritSettings())
        #expect(job["percentiles"] == .array([.number(10), .number(50), .number(99)]))
        #expect(job["record_trials"]?.int == 120, "at most every trial")
        t.mode = .sensitivity
        #expect(t.jobSettings(in: context, merit: MeritSettings()).settings["record_trials"]?.int == 0)
        let section = t.section(context: context)
        #expect(section[path: "monte_carlo", "percentiles"]?.nfArray == [10, 50, 99] && section[path: "monte_carlo", "record_trials"]?.int == 120)
        #expect(ToleranceSettings(section: section, system: cooke).percentiles == [10, 50, 99])
        #expect(ToleranceSettings.validPercentiles([97.7, 50, 50, 120, -1, .nan]) == [50, 97.7])
        #expect(ToleranceSettings.validPercentiles([200]) == nil)
        // Undoable through the session like any edit.
        let doc = DesignDocument(system: cooke)
        let undo = UndoManager()
        undo.groupsByEvent = false
        let session = ToleranceSession()
        session.load(doc)
        undo.beginUndoGrouping()
        session.edit(doc, "Change Percentiles", context: context, undoManager: undo) { $0.percentiles = [25, 75] }
        undo.endUndoGrouping()
        #expect(doc.tolerances.percentiles == [25, 75])
        undo.undo()
        session.load(doc)
        #expect(session.settings.percentiles == [50, 90, 97.7])
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
            "status": .string("finished"), "stopped": .bool(false),
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
                "nominal": .object(["criterion": .number(4e-4), "ok": .bool(true), "compensators": .array([.number(42.1)])]),
                "seconds_per_trial": .number(0.021),
                "percentiles": .array([
                    .object(["percentile": .number(50), "value": .number(6e-4)]), .object(["percentile": .number(90), "value": .number(9.2e-4)]),
                ]),
                "records": .array([
                    .object([
                        "trial": .integer(0), "criterion": .number(4e-4), "ok": .bool(true), "compensators": .array([.number(42.12)]),
                        "perturbations": .array([.number(5e-5), .number(-0.01), .number(0)]),
                    ])
                ]),
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
        // The engine's percentiles, nominal, timing and trial records.
        #expect(r.percentiles.map(\.percentile) == [50, 90] && r.percentiles.map(\.value) == [6e-4, 9.2e-4])
        #expect(r.nominal == 4e-4 && r.secondsPerTrial == 0.021 && r.records.count == 1 && r.record(0)?.compensators == [42.12])
        #expect(r.cumulative(6e-4) == 0.6)
        #expect(r.trialsWorstFirst == [4, 3, 2, 1, 0])
        #expect(r.trials(shownIn: 22...25) == [2, 1], "√5e-4 = 22.4 µm and √6e-4 = 24.5 µm")
        let range = r.compensatorRange(0)
        #expect(abs((range?.0 ?? 0) + 0.2) < 1e-9 && abs((range?.1 ?? 0) - 0.2) < 1e-9)
        // A Monte Carlo alone still has the nominal.
        let mcOnly = ToleranceResult(json: .object(["monte_carlo": .object(["trials": .integer(2), "nominal": .object(["criterion": .number(3e-4)])])]), system: cooke)
        #expect(mcOnly.hasNominal && mcOnly.nominal == 3e-4 && !mcOnly.hasSensitivity && mcOnly.trials == 2)
    }

    /// Timing: trials done and time left from the progress and the wall
    /// time per trial.
    @Test func timingEstimates() {
        // Monte Carlo half done after 10 s of its own: 0.1 s per trial, 10 s left.
        let mc = ToleranceSession.timing(
            elapsed: 12, monteCarloElapsed: 10, fraction: 0.2 + 0.8 * 0.5, mode: .both, trials: 200, operands: 10, lastSecondsPerTrial: nil)
        #expect(mc.trialsDone == 100 && abs((mc.secondsPerTrial ?? 0) - 0.1) < 1e-12 && abs((mc.secondsLeft ?? 0) - 10) < 1e-9)
        // In the sensitivity: its own rest, then the trials at the last run's rate.
        let s = ToleranceSession.timing(
            elapsed: 2, monteCarloElapsed: nil, fraction: 0.1, mode: .both, trials: 100, operands: 10, lastSecondsPerTrial: 0.05)
        #expect(s.trialsDone == nil && s.secondsPerTrial == 0.05 && abs((s.secondsLeft ?? 0) - (2 + 5)) < 1e-9)
        // Without a last run, a trial costs about one sensitivity evaluation.
        let first = ToleranceSession.timing(
            elapsed: 2.1, monteCarloElapsed: nil, fraction: 0.1, mode: .both, trials: 100, operands: 10, lastSecondsPerTrial: nil)
        #expect(abs((first.secondsPerTrial ?? 0) - 0.2) < 1e-9)
        #expect(ToleranceProgress.seconds(0.0213) == "21.3 ms" && ToleranceProgress.seconds(1.234) == "1.23 s")
        #expect(ToleranceProgress.duration(4.2) == "4 s" && ToleranceProgress.duration(600) == "10 min")
        #expect(AnalysisSection.estimate(secondsPerTrial: 0.02, trials: 200) == "Last run: 20 ms per trial, so 200 trials take about 4 s.")
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
        #expect(session.runSystem == cooke, "the design is sent as it is")
        let k = try #require(r.sensitivities.first { $0.param == .decenterY(surface: id(3)) })
        // A sensitivity case, rebuilt by the engine.
        session.preview(.sensitivity(operand: k.id, plus: true), in: w)
        #expect(session.previewLoading)
        try await waitUntil { session.preview != nil }
        let p = try #require(session.preview)
        #expect(p.system.surfaces[3].placement?.decenterY == 0.02 && p.system.surfaces[3].placement?.restoreAfter == true)
        #expect(w.isPreviewing && w.previewTitle?.hasPrefix("Tolerance: Surface decentre y S3 at +0.02") == true)
        let change = try #require(session.changes(p.perturbations).first)
        #expect(change.text == "+0.02 mm" && change.fraction == 1 && session.changes(p.perturbations).count == 1)
        #expect(doc.system == cooke, "a preview never changes the design")
        // A Monte-Carlo trial: the same build each time, read-only.
        session.preview(.trial(3), in: w)
        try await waitUntil { session.preview?.previewCase == .trial(3) }
        let t3 = try #require(session.preview)
        #expect(w.previewTitle == "Tolerance: trial 4, compensated" && t3.perturbations.count == 2)
        #expect(t3.criterion == r.trialCriteria[3], "the rebuild is the trial the run evaluated")
        session.endPreview(in: w)
        #expect(!w.isPreviewing && session.previewed == nil)
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
        #expect(text.contains("90 % of builds at most: 30.3") && text.contains("0.021 s per trial"))
        let csv = ToleranceReport.sensitivityCSV(r, rows: rows, system: cooke).split(separator: "\n")
        #expect(csv.count == 4 && csv[0].hasPrefix("rank,tolerance,minus,plus,unit") && csv[1].hasPrefix("1,Radius S1,3,3,fr,Radius 1,"))
        #expect(csv[2].hasPrefix("2,Surface decentre y S3,0.02,0.02,mm"))
        let trials = ToleranceReport.trialsCSV(r, rows: rows, system: cooke).split(separator: "\n", omittingEmptySubsequences: false)
        #expect(trials[0] == "trial,criterion,rms_spot_um,compensator_1,Radius S1,Thickness S1–2,Surface decentre y S3")
        #expect(trials[1] == "1,0.0004,20,42.12,5e-05,-0.01,0" && trials[2].hasPrefix("2,0.0005,22.36") && trials[2].hasSuffix(",,,,"))
    }

    /// A toleranced build's layout: nothing lies exactly in the section, so
    /// profiles and meridional rays near it are drawn; rims and skew rays
    /// are not. A layout with in-section lines is left alone.
    @Test func previewLayoutKeepsItsSection() {
        func line(_ k: LayoutScene.PolylineKind, x: Double, _ inSection: Bool = false) -> LayoutScene.Polyline {
            LayoutScene.Polyline(kind: k, points: [Point3(x, -10, 0), Point3(x, 10, 5)], inSection: inSection)
        }
        var lines = [line(.surfaceProfile, x: 0.03), line(.ray, x: 0.1), line(.ray, x: 2.1), line(.surfaceRim, x: 9), line(.imagePlane, x: 0, true)]
        EngineLayout.markNearSection(&lines, section: "yz")
        #expect(lines.map(\.inSection) == [true, true, false, false, true])
        var nominal = [line(.surfaceProfile, x: 0, true), line(.ray, x: 0.1)]
        EngineLayout.markNearSection(&nominal, section: "yz")
        #expect(nominal.map(\.inSection) == [true, false])
    }

    @Test func runIsBlockedByBrokenRowsOnly() {
        let doc = DesignDocument(system: cooke)
        let session = ToleranceSession()
        session.load(doc)
        session.edit(doc, "Set", context: context, undoManager: nil) {
            $0.rows = [ToleranceRow(.radius, surface: id(1), minus: 3), ToleranceRow(.index, surface: id(1), minus: 0.001)]
        }
        #expect(session.runProblem(context: context) == nil, "a catalogue glass's index runs on its melt offset")
        let (s, ids) = session.settings.jobSettings(in: context, merit: MeritSettings())
        #expect(s["tolerances"]?.array?.count == 2 && ids == session.settings.rows.map(\.id))
        session.edit(doc, "Break", context: context, undoManager: nil) { $0.rows[0].minus = -1 }
        #expect(session.runProblem(context: context) == "Fix the tolerance marked in red first.")
    }
}
