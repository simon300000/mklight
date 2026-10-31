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
        try await wait { w.snapshot?.system == system }
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

    /// The default merit evaluates at the design; converting it gives its
    /// generated operands, which evaluate to the same merit.
    @Test func defaultMeritEvaluatesAndConvertsToCustom() async throws {
        let system = try #require(SampleSystems.bundled("double_gauss"))
        let doc = DesignDocument(system: system)
        let w = try await workspace(system)
        let session = OptimizeSession()
        session.load(doc)
        session.scheduleEvaluation(in: w, variables: doc.variables, delay: .zero)
        try await wait { session.evaluation != nil || session.evaluationError != nil }
        let e = try #require(session.evaluation, "\(session.evaluationError ?? "")")
        #expect(e.merit.isFinite && e.merit > 0)
        #expect(e.rows.count > 20, "the default merit's generated operands")
        #expect(!e.specOperands.isEmpty)
        #expect(doc.system == system, "evaluating does not change the design")
        await session.convertToCustom(in: w, document: doc, variables: doc.variables, undoManager: nil)
        #expect(doc.merit.mode == .custom && doc.merit.operands.count == e.specOperands.count)
        session.scheduleEvaluation(in: w, variables: doc.variables, delay: .zero)
        try await wait { session.evaluation.map { $0.rows.count >= doc.merit.operands.count && $0 != e } ?? false || session.evaluationError != nil }
        let custom = try #require(session.evaluation)
        #expect(abs(custom.merit - e.merit) <= 1e-9 * max(e.merit, 1), "the converted merit equals the default's")
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
