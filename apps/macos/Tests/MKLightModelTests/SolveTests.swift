import Foundation
import Testing

@testable import MKLightModel

/// Solves and solve reports against JSON written by the Rust side
/// (`dump_fixtures` in mklight-core, `dump_solve_fixtures` in mklight-trace).
@Suite struct SolveFixtureTests {
    @Test func everySolveKindRoundTrips() throws {
        let solves = try Fixtures.roundTrip([Solve].self, "solve_kinds")
        #expect(Set(solves.map(\.kind.kind)) == Set(SolveKind.Kind.allCases))
        #expect(solves[1].kind == .marginalRayAngle(slope: -0.1))
        #expect(solves[4].kind == .pickup(source: .conic(surface: SurfaceID(2)), scale: 2, offset: -1))
        #expect(solves[7].kind == .edgeThickness(value: 2, radius: 8))
        #expect(solves[8].kind == .edgeThickness(value: 1.5, radius: nil))
    }

    @Test func pickupDefaultsDecode() throws {
        let json = #"{"target":{"param":"conic","surface":1},"kind":{"type":"pickup","source":{"param":"conic","surface":2}}}"#
        let s = try MKLightJSON.decode(Solve.self, from: Data(json.utf8))
        #expect(s.kind == .pickup(source: .conic(surface: SurfaceID(2)), scale: 1, offset: 0))
    }

    @Test func solvedSystemRoundTripsAndValidates() throws {
        let s = try Fixtures.roundTrip(OpticalSystem.self, "solved_triplet")
        #expect(s.solves.count == 6)
        #expect(s.configurations.map(\.name) == ["infinity", "2 m"])
        #expect(s.structuralIssues().isEmpty, "\(s.structuralIssues())")
        #expect(s.solvedParams.contains(.thickness(surface: SurfaceID(7))))
        #expect(s.solve(for: .curvature(surface: SurfaceID(3)))?.kind == .chiefRayAngle(slope: 0.25))
    }

    /// Without solves the key is omitted, so such systems hash as before.
    @Test func emptySolvesAreOmitted() throws {
        let tree = try JSONValue.parse(MKLightJSON.encode(SampleSystems.cookeTriplet()))
        #expect(tree["solves"] == nil)
    }

    @Test func solveReportsRoundTrip() throws {
        struct Entry: Codable, Equatable {
            var system: OpticalSystem
            var report: SolveReport
        }
        let entries = try Fixtures.roundTrip([Entry].self, "solve_reports")
        #expect(entries.map(\.report.configuration) == [nil, 0, 1])
        let given = entries[0]
        #expect(given.system.configurations.count == 2, "solving as given keeps the configurations")
        #expect(given.report.outcomes.count == 2)
        #expect(given.report.outcomes.allSatisfy { abs($0.residual) < 1e-6 })
        #expect(given.report.runInfo["engine_version"] != nil)
        // Per-configuration results are flattened, and the override of a
        // solved thickness is reported.
        #expect(entries[1].system.configurations.isEmpty && entries[2].system.configurations.isEmpty)
        #expect(entries[2].report.warnings.first?.contains("solve wins") == true)
        #expect(entries[2].report.runInfo["settings"]?["configuration"] == .integer(1))
        #expect(Set(entries[1].report.solvedParams) == [.thickness(surface: SurfaceID(2)), .curvature(surface: SurfaceID(2))])
    }

    @Test func slopeIsLabelledAsParaxialAngle() {
        #expect(SolveKind.Kind.marginalRayAngle.valueLabel?.label == "paraxial angle u")
        #expect(SolveKind.Kind.chiefRayAngle.valueLabel?.unit == "")
        var k = SolveKind.chiefRayAngle(slope: 0.1)
        k.value = 0.2
        #expect(k == .chiefRayAngle(slope: 0.2), "the slope passes through unconverted")
    }
}

@Suite struct SolveEditingTests {
    private func solved() -> OpticalSystem {
        var s = SampleSystems.cookeTriplet()
        s.solves = [
            Solve(target: .thickness(surface: SurfaceID(7)), kind: .marginalRayHeight(height: 0)),
            Solve(target: .curvature(surface: SurfaceID(6)), kind: .pickup(source: .curvature(surface: SurfaceID(1)), scale: -1, offset: 0)),
            Solve(target: .fieldY(index: 2), kind: .pickup(source: .fieldY(index: 1), scale: 1.5, offset: 0)),
        ]
        return s
    }

    @Test func deletingASurfaceDropsItsSolves() {
        var s = solved()
        let deleted = s.deleteSurface(at: 1)
        #expect(deleted)
        #expect(s.solves.map(\.target) == [.thickness(surface: SurfaceID(7)), .fieldY(index: 2)])
    }

    @Test func removingAFieldRenumbersSolves() {
        var s = solved()
        let first = s.removeField(at: 0)
        #expect(first)
        #expect(s.solves[2].target == .fieldY(index: 1))
        #expect(s.solves[2].kind == .pickup(source: .fieldY(index: 0), scale: 1.5, offset: 0))
        let second = s.removeField(at: 0)
        #expect(second)
        #expect(s.solves.count == 2, "a pickup of a removed field goes")
    }

    @Test func invalidatedSolveIsDropped() {
        let before = solved()
        var s = before
        s.surfaces[6].shape = .paraxial(focalLength: 40)
        s.dropReferencesInvalidated(since: before)
        #expect(s.solves.count == 2)
        #expect(s.structuralIssues().isEmpty)
    }

    @Test func structuralChecks() {
        var s = solved()
        s.solves.append(Solve(target: .thickness(surface: SurfaceID(7)), kind: .chiefRayHeight(height: 1)))
        s.solves.append(Solve(target: .conic(surface: SurfaceID(5)), kind: .fNumber(value: -1)))
        s.solves.append(Solve(target: .thickness(surface: SurfaceID(3)), kind: .edgeThickness(value: .infinity, radius: nil)))
        let messages = s.structuralIssues().map(\.message)
        #expect(messages.contains { $0.contains("already has a solve") })
        #expect(messages.contains { $0.contains("surface has no conic") })
        #expect(messages.contains { $0.contains("F/# must be > 0") })
        #expect(messages.contains { $0.contains("finite") })
        // An automatic semi-diameter may be a target (the solve fixes it).
        var t = SampleSystems.cookeTriplet()
        t.solves = [Solve(target: .semiDiameter(surface: SurfaceID(1)), kind: .pickup(source: .thickness(surface: SurfaceID(1)), scale: 3, offset: 0))]
        #expect(t.structuralIssues().isEmpty)
    }
}
