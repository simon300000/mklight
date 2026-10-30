import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// Optimize, tolerance and apply-solves flows through their view models.
@Suite @MainActor struct DesignJobTests {
    private func workspace(_ system: OpticalSystem, latency: Duration = .zero) async throws -> Workspace {
        let w = Workspace(engine: MockEngine(simulatedLatency: latency))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await waitUntil { w.snapshot?.system == system }
        return w
    }

    @Test func meritProgressIsParsed() {
        #expect(MeritPoint.parse("lm 12: merit 3.4e-4") == MeritPoint(iteration: 12, merit: 3.4e-4))
        #expect(MeritPoint.parse("de 3: merit 0.5") == MeritPoint(iteration: 3, merit: 0.5))
        #expect(MeritPoint.parse("tolerance sensitivity") == nil)
    }

    /// The optimised design is applied as one undoable step.
    @Test func optimizeAppliesAsOneUndoableStep() async throws {
        let doc = DesignDocument(system: SampleSystems.cookeTriplet())
        let undo = UndoManager()
        let w = try await workspace(doc.system)
        let r1 = ParamPath.curvature(surface: doc.system.surfaces[1].id)
        doc.setVariable(r1, true, undoManager: nil)
        let session = OptimizeSession()
        session.maxIterations = 6
        #expect(session.settings(variables: doc.variables)["variables"]?.array?.count == 1)
        session.start(in: w, variables: doc.variables)
        #expect(session.isRunning)
        try await waitUntil { session.state == .finished }
        let result = try #require(session.result)
        #expect(result.improved && result.finalMerit < result.startMerit && result.iterations == 6)
        #expect(result.variables.count == 1 && result.variables[0].param == r1)
        #expect(session.trace.count == 6, "merit history for the sparkline")
        let before = doc.system
        session.apply(to: doc, undoManager: undo)
        #expect(doc.system == result.system && doc.system != before)
        #expect(doc.system.surfaces[1].shape.curvature != before.surfaces[1].shape.curvature)
        #expect(doc.system.surfaces[2] == before.surfaces[2], "only the variable changed")
        #expect(undo.undoActionName == "Optimize")
        undo.undo()
        #expect(doc.system == before)
        undo.redo()
        #expect(doc.system == result.system)
    }

    /// Stop ends the run early and keeps the best design so far; Cancel
    /// discards it.
    @Test func stopKeepsTheResultAndCancelDiscardsIt() async throws {
        let doc = DesignDocument(system: SampleSystems.cookeTriplet())
        let w = try await workspace(doc.system, latency: .milliseconds(400))
        let session = OptimizeSession()
        session.maxIterations = 1000
        session.start(in: w, variables: [])
        try await waitUntil { session.trace.count >= 2 }
        #expect(session.canStop)
        session.stop()
        try await waitUntil { session.state == .finished }
        #expect(session.result?.stopped == true && session.result?.system != nil)
        session.reset()
        session.start(in: w, variables: [])
        try await waitUntil { session.trace.count >= 1 }
        session.cancel()
        #expect(session.state == .setup && session.result == nil)
    }

    @Test func toleranceReportsSensitivityAndMonteCarlo() async throws {
        let w = try await workspace(SampleSystems.cookeTriplet())
        let session = ToleranceSession()
        session.trials = 50
        session.seed = 7
        session.start(in: w)
        try await waitUntil { session.state == .finished }
        let r = try #require(session.result)
        #expect(!r.sensitivities.isEmpty && r.hasMonteCarlo && r.criteria.count == 50)
        #expect(zip(r.sensitivities, r.sensitivities.dropFirst()).allSatisfy { $0.worst >= $1.worst }, "ranked worst first")
        #expect(Histogram.bins(r.criteria, count: 10).map(\.1).reduce(0, +) == 50)
    }

    /// Apply Solves: a dry run that changes nothing, then one undoable commit.
    @Test func applySolvesIsADryRunThenAnUndoableCommit() async throws {
        var s = SampleSystems.cookeTriplet()
        let r1 = ParamPath.curvature(surface: s.surfaces[1].id), r6 = ParamPath.curvature(surface: s.surfaces[6].id)
        s.solves = [Solve(target: r6, kind: .pickup(source: r1, scale: -1, offset: 0))]
        let doc = DesignDocument(system: s)
        let undo = UndoManager()
        let w = try await workspace(s)
        let session = SolvesSession()
        await session.dryRun(in: w)
        #expect(session.state == .finished && session.outcomes.count == 1 && session.changes == 1)
        #expect(doc.system == s, "the dry run changes nothing")
        await session.commit(in: w, document: doc, undoManager: undo)
        #expect(session.committed)
        #expect(doc.system.value(at: r6) == -(s.value(at: r1) ?? 0))
        #expect(undo.undoActionName == "Apply Solves")
        undo.undo()
        #expect(doc.system == s)
    }

    @Test func jobsNeedAValidSnapshot() async throws {
        let w = Workspace(engine: MockEngine(simulatedLatency: .zero))
        let session = OptimizeSession()
        session.start(in: w, variables: [])
        if case .failed = session.state {} else { Issue.record("expected a failure without a snapshot") }
    }
}

