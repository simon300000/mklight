import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// Glass substitution settings, progress and results (the engine run is in
/// FFIGlassSubstitutionTests).
@Suite @MainActor struct GlassSubstitutionTests {
    @Test func candidatesEncodeTheEngineCandidateSet() {
        var c = GlassCandidates()
        #expect(c.json["nearest"]?.int == 10 && c.json["window"] == .null)
        #expect(c.json["statuses"]?.array?.compactMap(\.string) == ["preferred", "standard", "melt"])
        #expect(c.json["min_transmission"] == nil)
        c.rule = .window
        c.windowNd = 0.02
        c.windowVd = 3
        c.include = GlassCandidates.names("N-BK7, SCHOTT:F2 ;N-SF6")
        c.minTransmission = 0.9
        c.catalogs = ["SCHOTT"]
        let j = c.json
        #expect(j["nearest"] == .null && j[path: "window", "nd"]?.doubleValue == 0.02 && j[path: "window", "vd"]?.doubleValue == 3)
        #expect(j["include"]?.array?.compactMap(\.string) == ["N-BK7", "SCHOTT:F2", "N-SF6"])
        #expect(j["min_transmission"]?.doubleValue == 0.9 && j["catalogs"]?.array?.count == 1)
        #expect(c.summary.contains("n_d ±0.02") && c.summary.contains("SCHOTT") && c.summary.contains("limits"))
    }

    @Test func settingsCarryTheSelectionOverridesMeritAndBudget() throws {
        let system = SampleSystems.cookeTriplet()
        let doc = DesignDocument(system: system)
        let s = GlassSubstitutionSession()
        s.configure(for: system)
        let slots = GlassSlot.all(in: system)
        #expect(slots.count == 3 && s.selected.count == 3)
        s.selected.remove(slots[1].id)
        var own = GlassCandidates()
        own.nearest = 3
        s.overrides[slots[2].id] = own
        s.algorithm = .hammer
        s.maxSeconds = 30
        let j = s.settings(system: system, merit: doc.merit, variables: doc.variables)
        let mv = try #require(j["material_variables"]?.array)
        #expect(mv.map { $0["surface"]?.int } == [Int(slots[0].id.rawValue), Int(slots[2].id.rawValue)])
        #expect(mv[1][path: "candidates", "nearest"]?.int == 3 && mv[0][path: "candidates", "nearest"]?.int == 10)
        #expect(j["algorithm"]?.string == "hammer" && j["max_seconds"]?.doubleValue == 30 && j["top_n"]?.int == 10)
        #expect(j["default_merit"] != nil && j["hold_efl"]?.bool == true)
        // A design change that keeps the glass gaps keeps the choice.
        s.configure(for: system)
        #expect(s.selected.count == 2)
    }

    @Test func progressAndResultsAreRead() throws {
        let p = try #require(GlassProgressInfo.parse(
            #"glass_substitution: combinations 12; best merit 3.250000e-4; glasses [{"surface":1,"catalog":"BUILTIN","name":"N-BK7","nd":1.5168,"vd":64.17,"dpgf":0.0}]"#))
        #expect(p.combinations == 12 && p.bestMerit == 3.25e-4 && p.glasses == ["N-BK7"])
        #expect(GlassProgressInfo.parse("ranking glass combinations on the CPU reference") == nil)
        let json = try JSONValue.parse(
            Data(
                #"{"status":"stopped","stopped":true,"algorithm":"tabu","termination":"cancelled","starting_merit":0.002,"statistics":{"combinations":5,"failed":1},"history":[{"best_merit":0.003},{"best_merit":0.001}],"candidate_sets":[{"surface":1,"glasses":[{},{}]}],"results":[{"rank":1,"merit":0.001,"violation":0,"feasible":true,"local_termination":"cancelled","glasses":[{"surface":1,"catalog":"BUILTIN","name":"N-BK7","nd":1.5168,"vd":64.17,"dpgf":0}],"variables":[{"param":{"param":"curvature","surface":1},"start":0.02,"value":0.021,"change":0.001}]}]}"#
                    .utf8))
        let r = try #require(GlassSubstitutionResult(json: json))
        #expect(r.stopped && r.combinations == 5 && r.failed == 1 && r.candidateCounts[SurfaceID(1)] == 2)
        #expect(r.ranked.count == 1 && r.ranked[0].cutShort && r.ranked[0].movedChanges.count == 1)
        #expect(r.ranked[0].glasses[0].mapID == "BUILTIN/N-BK7")
        #expect(GlassSubstitutionSession.runningMinimum([3, 1, 2]) == [3, 1, 1])
    }

    /// Surface ids that are no ids (negative, not whole, beyond `Int`) are
    /// left out of a result rather than read.
    @Test func resultsWithImpossibleSurfaceIDsAreRead() throws {
        let json = try JSONValue.parse(
            Data(
                #"{"results":[{"glasses":[{"surface":-4,"name":"X"},{"surface":1e30,"name":"Y"}]}],"candidate_sets":[{"surface":-1,"glasses":[{}]},{"surface":1e30,"glasses":[{}]},{"surface":2.5,"glasses":[{}]},{"surface":3,"glasses":[{}]}]}"#
                    .utf8))
        let r = try #require(GlassSubstitutionResult(json: json))
        #expect(r.candidateCounts == [SurfaceID(3): 1])
        #expect(r.ranked[0].glasses.map(\.surface) == [SurfaceID(0), SurfaceID(0)])
    }
}
