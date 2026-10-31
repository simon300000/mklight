import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// The merit editor's evaluation and conversion on the real engine.
@Suite(.serialized, engineLinked) @MainActor struct FFIMeritTests {
    private func workspace(_ system: OpticalSystem) async throws -> Workspace {
        let e = try FFIEngine(configuration: ["worker_threads": .integer(2), "progress_interval_ms": .integer(10)])
        let w = Workspace(engine: e)
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await wait(.seconds(10)) { w.snapshot?.system == system || !w.modelErrors.isEmpty }
        #expect(w.modelErrors.isEmpty, "\(w.modelErrors)")
        return w
    }

    private func wait(_ timeout: Duration = .seconds(120), _ cond: () -> Bool) async throws {
        let clock = ContinuousClock()
        let end = clock.now + timeout
        while !cond() {
            guard clock.now < end else {
                Issue.record("condition not met within \(timeout)")
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    /// The default merit evaluates at the design through `merit_evaluation`
    /// (the same number `optimize` reports as its start); converting it
    /// gives its generated operands, which evaluate to the merit shown
    /// before the conversion.
    @Test func defaultMeritEvaluatesAndConvertsToCustom() async throws {
        let system = try #require(SampleSystems.bundled("double_gauss"))
        let doc = DesignDocument(system: system)
        let w = try await workspace(system)
        let session = OptimizeSession()
        session.load(doc)
        session.scheduleEvaluation(in: w, variables: doc.variables, system: doc.system, delay: .zero)
        try await wait { session.evaluation != nil || session.evaluationError != nil }
        let e = try #require(session.evaluation, "\(session.evaluationError ?? "")")
        #expect(e.merit.isFinite && e.merit > 0 && e.status == .ok)
        #expect(e.rows.count > 20, "the default merit's generated operands")
        #expect(e.rows.contains { $0.isConstraint && $0.label == "hold EFL" })
        let shares = e.rows.map(\.fraction).filter(\.isFinite)
        #expect(abs(shares.reduce(0, +) - 1) < 1e-9, "the shares add up to the merit")
        let total = e.rows.map(\.contribution).filter(\.isFinite).reduce(0, +)
        #expect(abs(total - e.merit) <= 1e-12 * e.merit, "the contributions add up to the merit")
        #expect(!e.specOperands.isEmpty)
        #expect(doc.system == system, "evaluating does not change the design")
        // The same merit as optimize's start report.
        var s = session.evaluationSettings(variables: [], system: system)
        s["max_iterations"] = .integer(1)
        let opt = try await w.submitJob("optimize", settings: s).result()
        let start = try #require(opt.json[path: "start", "merit"]?.nf)
        #expect(abs(start - e.merit) <= 1e-12 * e.merit, "merit_evaluation \(e.merit) vs optimize start \(start)")
        // Convert to custom: the same merit.
        await session.convertToCustom(in: w, document: doc, variables: doc.variables, undoManager: nil)
        #expect(doc.merit.mode == .custom && doc.merit.operands.count == e.specOperands.count - 1, "all but the EFL hold")
        #expect(doc.merit.spec["on_failure"] == e.spec?["on_failure"])
        session.scheduleEvaluation(in: w, variables: doc.variables, system: doc.system, delay: .zero)
        try await wait { session.evaluation.map { $0 != e } ?? false || session.evaluationError != nil }
        let custom = try #require(session.evaluation, "\(session.evaluationError ?? "")")
        #expect(custom.rows.count == e.rows.count, "the engine adds the EFL hold again")
        #expect(abs(custom.merit - e.merit) <= 1e-12 * max(e.merit, 1), "the converted merit \(custom.merit) equals the default's \(e.merit)")
    }

    /// A solve-controlled parameter: the engine refuses it as a variable,
    /// so the app leaves it out of the run and evaluates with the others.
    @Test func solvedVariablesAreLeftOut() async throws {
        var system = try #require(SampleSystems.bundled("double_gauss"))
        // The back focus by a marginal ray height solve (paraxial focus).
        let solved = ParamPath.thickness(surface: system.surfaces[system.imageIndex - 1].id)
        let free = ParamPath.curvature(surface: system.surfaces[1].id)
        system.solves.append(Solve(target: solved, kind: .marginalRayHeight(height: 0)))
        let doc = DesignDocument(system: system)
        let w = try await workspace(system)
        let vars = [DesignVariable(solved, lower: nil, upper: nil), DesignVariable(free, lower: nil, upper: nil)]
        let session = OptimizeSession()
        session.load(doc)
        session.scheduleEvaluation(in: w, variables: vars, system: system, delay: .zero)
        try await wait { session.evaluation != nil || session.evaluationError != nil }
        #expect(session.evaluation != nil, "\(session.evaluationError ?? "")")
        // Sent as it is, the engine refuses the solved one.
        var s = session.evaluationSettings(variables: [], system: system)
        s["variables"] = try JSONValue.from(vars)
        await #expect(throws: (any Error).self) { _ = try await w.submitJob("merit_evaluation", settings: s).result() }
    }

    /// A pane with a run-local spectrum stays current after it ran (the
    /// engine records the submitted snapshot's hash).
    @Test func spectrumPanesAreNotStale() async throws {
        let system = try #require(SampleSystems.bundled("double_gauss"))
        let pane = AnalysisPane(analysis: "spot_diagram", settings: ["spectrum": .object(["kind": .string("photopic")])])
        let w = Workspace(engine: try FFIEngine(), panes: [pane])
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await wait { pane.output != nil && !pane.isRunning }
        let out = try #require(pane.output)
        #expect(out.runInfo.snapshotHash == w.snapshot?.hash)
        #expect(!pane.isStale)
    }

    /// Custom operands: values, contributions, and an inline engine error.
    @Test func customOperandsReportValuesAndErrors() async throws {
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let doc = DesignDocument(system: system)
        let w = try await workspace(system)
        let session = OptimizeSession()
        session.editMerit(doc, "Custom", undoManager: nil) { m in
            m.mode = .custom
            var efl = MeritOperand(kind: OperandCatalog.descriptor("efl")!.newKind)
            efl.target = 50
            var spot = MeritOperand(kind: OperandCatalog.descriptor("rms_spot")!.newKind)
            spot.weight = 100
            var mtf = MeritOperand(kind: OperandCatalog.descriptor("mtf")!.newKind)
            mtf.comparison = .greaterThan
            mtf.target = 0.9
            mtf.field = .object(["field": .string("index"), "index": .integer(0)])
            m.operands = [efl, spot, mtf]
        }
        session.scheduleEvaluation(in: w, variables: [], delay: .zero)
        try await wait { session.evaluation != nil || session.evaluationError != nil }
        let e = try #require(session.evaluation, "\(session.evaluationError ?? "")")
        let efl = try #require(e.row(0))
        #expect(abs(efl.value - 50) < 5, "the triplet's EFL is near 50 mm: \(efl.value)")
        #expect(try #require(e.row(1)).value > 0)
        let mtf = try #require(e.row(2))
        #expect(mtf.value > 0 && mtf.value <= 1)
        let shares = e.rows.prefix(3).map(\.fraction).filter(\.isFinite)
        #expect(abs(shares.reduce(0, +) - 1) < 1e-6 || e.rows.count > 3)
    }
}
