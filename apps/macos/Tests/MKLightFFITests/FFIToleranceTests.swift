import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// Tolerancing on the real engine: the Cooke triplet's default set
/// (fringes over the layout's clear apertures, thicknesses, surface
/// decentres and tilts with return placements) and the back-focus
/// compensator, sensitivity and Monte Carlo, reproducible for a seed, and
/// a preview of a toleranced build.
@Suite(.serialized, engineLinked) @MainActor struct FFIToleranceTests {
    private func wait(_ timeout: Duration = .seconds(180), _ cond: () -> Bool) async throws {
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

    @Test func sensitivityAndMonteCarloOnTheCooke() async throws {
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let doc = DesignDocument(system: system)
        let w = Workspace(engine: try FFIEngine(configuration: ["worker_threads": .integer(2), "progress_interval_ms": .integer(10)]))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await wait { w.snapshot?.system == system && w.layout.output != nil }
        let context = ToleranceContext(system: system, semiDiameters: EngineLayout.autoSemiDiameters(EngineLayout.scene(from: w.layout.output)))

        let session = ToleranceSession()
        session.load(doc)
        session.edit(doc, "Set", context: context, undoManager: nil) {
            $0.mode = .both
            $0.trials = 40
            $0.seed = 5
            $0.criterion.threshold = 30
        }
        #expect(session.runProblem(context: context) == nil, "every default tolerance expands: \(session.settings.issues(in: context))")
        let rows = session.settings.rows.count
        session.start(in: w, context: context, merit: doc.merit)
        try await wait { session.state == .finished || { if case .failed = session.state { true } else { false } }() }
        let r = try #require(session.result, "\(session.state)")
        #expect(r.display == .spot)
        // Sensitivity: one operand per row, ranked, nominal with the back focus re-optimised.
        #expect(r.sensitivities.count == rows && session.resultRows.count == rows)
        #expect(zip(r.sensitivities, r.sensitivities.dropFirst()).allSatisfy { $0.worst >= $1.worst })
        let nominalUm = r.display.value(r.nominal)
        #expect(nominalUm > 5 && nominalUm < 60, "the Cooke's RMS spot, µm: \(nominalUm)")
        #expect(r.nominalCompensators.count == 1 && abs(r.nominalCompensators[0] - system.surfaces[system.imageIndex - 1].thickness) < 0.5)
        #expect(r.predicted > r.nominal && abs(r.sensitivities.map { r.share($0) }.reduce(0, +) - 1) < 1e-9)
        // Decentres and tilts move their surface only, and they matter.
        let dec = r.sensitivities.filter { if case .decenterY = $0.param { true } else { false } }
        #expect(!dec.isEmpty && dec.allSatisfy { $0.ok && $0.worst > 0 })
        // Monte Carlo: the trials, percentiles, yield and the compensator's spread.
        #expect(r.trials == 40 && r.criteria.count + r.failures == 40 && r.seed == 5)
        #expect(r.percentile(50) <= r.percentile(90) && r.percentile(90) <= r.percentile(97.7))
        #expect(r.yield != nil && r.compensatorStd.count == 1 && r.compensatorRange(0) != nil)
        // The same seed gives the same trials.
        let first = r.criteria
        session.start(in: w, context: context, merit: doc.merit)
        try await wait { session.state == .finished }
        #expect(session.result?.criteria == first)
        // A build previews in the window, not applied.
        let worst = try #require(r.sensitivities.first)
        let build = try #require(session.trialSystem(worst.id, plus: true))
        #expect(build != system)
        session.preview(worst.id, plus: true, in: w)
        #expect(w.isPreviewing && doc.system == system)
        session.preview(nil, in: w)
        let csv = ToleranceReport.sensitivityCSV(r, rows: session.resultRows, system: system)
        #expect(csv.split(separator: "\n").count == rows + 1)
    }

    /// Stop keeps the completed trials.
    @Test func stopKeepsPartialResults() async throws {
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let doc = DesignDocument(system: system)
        let w = Workspace(engine: try FFIEngine(configuration: ["worker_threads": .integer(2), "progress_interval_ms": .integer(10)]))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await wait { w.snapshot?.system == system && w.layout.output != nil }
        let context = ToleranceContext(system: system, semiDiameters: EngineLayout.autoSemiDiameters(EngineLayout.scene(from: w.layout.output)))
        let session = ToleranceSession()
        session.load(doc)
        session.edit(doc, "Set", context: context, undoManager: nil) {
            $0.mode = .monteCarlo
            $0.trials = 20000
        }
        session.start(in: w, context: context, merit: doc.merit)
        try await wait { session.canStop }
        try await Task.sleep(for: .milliseconds(600))
        session.stop()
        try await wait { session.state == .finished }
        let r = try #require(session.result)
        #expect(r.stopped && r.trials > 0 && r.trials < 20000 && r.criteria.count + r.failures == r.trials)
    }
}
