import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// Tolerancing on the real engine: the Cooke triplet's default set (power
/// and irregularity in test fringes, thicknesses, surface and element
/// decentres and tilts, the index and Abbe number of its catalogue glasses
/// through their melt offsets) with the back-focus compensator,
/// sensitivity and Monte Carlo, reproducible for a seed, the engine's
/// nominal, percentiles, timing and trial records, and the rebuild of a
/// trial and a sensitivity case for the preview.
@Suite(.serialized, engineLinked) @MainActor struct FFIToleranceTests {
    private func wait(_ timeout: Duration = .seconds(300), _ cond: () -> Bool) async throws {
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

    /// A workspace on the Cooke with its layout and the Abbe numbers of
    /// its glasses, and the tolerance context they give.
    private func cooke() async throws -> (OpticalSystem, Workspace, ToleranceContext) {
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let w = Workspace(engine: try FFIEngine(configuration: ["worker_threads": .integer(2), "progress_interval_ms": .integer(10)]))
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        try await wait { w.snapshot?.system == system && w.layout.output != nil }
        let glasses = system.surfaces.filter { if case .catalog = $0.material { true } else { false } }.count
        _ = w.abbeNumbers(of: system)
        try await wait { w.abbeNumbers(of: system).count == glasses }
        let context = ToleranceContext(
            system: system, semiDiameters: EngineLayout.autoSemiDiameters(EngineLayout.scene(from: w.layout.output)),
            abbeNumbers: w.abbeNumbers(of: system))
        return (system, w, context)
    }

    private func finished(_ s: ToleranceSession) -> Bool {
        if case .failed = s.state { return true }
        return s.state == .finished
    }

    @Test func sensitivityAndMonteCarloOnTheCooke() async throws {
        let (system, w, context) = try await cooke()
        let doc = DesignDocument(system: system)
        // The engine's V_d of N-SK16 (catalogue 60.32).
        let vd = try #require(context.abbeNumbers[system.surfaces[1].id])
        #expect(abs(vd - 60.3) < 0.2)

        let session = ToleranceSession()
        session.load(doc)
        session.edit(doc, "Set", context: context, undoManager: nil) {
            $0.mode = .both
            $0.trials = 40
            $0.seed = 5
            $0.recordTrials = 10
            $0.percentiles = [50, 84.1, 97.7]
            $0.criterion.threshold = 30
        }
        #expect(session.runProblem(context: context) == nil, "every default tolerance runs: \(session.settings.issues(in: context))")
        let kinds = Set(session.settings.rows.map(\.kind))
        #expect(kinds.isSuperset(of: [.irregularityAstigmatism, .irregularitySpherical, .elementDecenterX, .elementTiltY, .index, .abbe]))
        let rows = session.settings.rows.count
        session.start(in: w, context: context, merit: doc.merit)
        try await wait { finished(session) }
        let r = try #require(session.result, "\(session.state)")
        #expect(r.display == .spot)
        // Sensitivity: one operand per row, ranked, nominal with the back focus re-optimised.
        #expect(r.sensitivities.count == rows && session.resultRows.count == rows)
        #expect(zip(r.sensitivities, r.sensitivities.dropFirst()).allSatisfy { $0.worst >= $1.worst })
        let nominalUm = r.display.value(r.nominal)
        #expect(nominalUm > 5 && nominalUm < 60, "the Cooke's RMS spot, µm: \(nominalUm)")
        #expect(r.nominalCompensators.count == 1 && abs(r.nominalCompensators[0] - system.surfaces[system.imageIndex - 1].thickness) < 0.5)
        #expect(r.predicted > r.nominal && abs(r.sensitivities.map { r.share($0) }.reduce(0, +) - 1) < 1e-9)
        // Every kind moves the criterion: surface and element motions, irregularity and the melt.
        func moved(_ k: ToleranceKind) -> Bool {
            let ss = r.sensitivities.filter { session.resultRows[$0.id].kind == k }
            return !ss.isEmpty && ss.allSatisfy { $0.ok } && ss.contains { $0.worst > 0 }
        }
        for k in [ToleranceKind.decenterY, .elementDecenterY, .elementTiltX, .irregularityAstigmatism, .irregularitySpherical, .index, .abbe] {
            #expect(moved(k), "\(k)")
        }
        // Monte Carlo: the trials, the engine's nominal, levels, yield, timing and records.
        #expect(r.trials == 40 && r.criteria.count + r.failures == 40 && r.seed == 5 && r.trialCriteria.count == 40)
        #expect(r.percentiles.map(\.percentile) == [50, 84.1, 97.7])
        #expect(r.percentiles[0].value <= r.percentiles[1].value && r.percentiles[1].value <= r.percentiles[2].value)
        #expect(r.yield != nil && r.compensatorStd.count == 1 && r.compensatorRange(0) != nil)
        #expect((r.secondsPerTrial ?? 0) > 0 && session.lastSecondsPerTrial == r.secondsPerTrial)
        #expect(r.records.count == 10 && r.records.allSatisfy { $0.perturbations.count == rows && $0.compensators.count == 1 })
        #expect(r.records.allSatisfy { $0.criterion == r.trialCriteria[$0.trial] })
        // The same seed gives the same trials.
        let first = r.criteria
        session.start(in: w, context: context, merit: doc.merit)
        try await wait { finished(session) }
        #expect(session.result?.criteria == first)
        let csv = ToleranceReport.sensitivityCSV(r, rows: session.resultRows, system: system)
        #expect(csv.split(separator: "\n").count == rows + 1)
        let trials = ToleranceReport.trialsCSV(r, rows: session.resultRows, system: system).split(separator: "\n")
        #expect(trials.count == 41 && trials[0].split(separator: ",").count == 3 + 1 + rows)
    }

    /// A trial and a sensitivity case rebuilt by the engine for the
    /// preview (fixed seed): the same criterion, perturbations and
    /// compensators as the run, the moved element and the irregular
    /// surfaces in the rebuilt system, read-only.
    @Test func trialPreviewOnTheCooke() async throws {
        let (system, w, context) = try await cooke()
        let doc = DesignDocument(system: system)
        let session = ToleranceSession()
        session.load(doc)
        session.edit(doc, "Set", context: context, undoManager: nil) {
            $0.mode = .both
            $0.trials = 12
            $0.seed = 11
            $0.recordTrials = 12
        }
        session.start(in: w, context: context, merit: doc.merit)
        try await wait { finished(session) }
        let r = try #require(session.result, "\(session.state)")
        let k = try #require(r.trialsWorstFirst.first)
        let record = try #require(r.record(k))

        session.preview(.trial(k), in: w)
        try await wait { session.preview != nil || session.previewError != nil }
        let p = try #require(session.preview, "\(session.previewError ?? "")")
        #expect(abs(p.criterion - r.trialCriteria[k]) <= 1e-9 * abs(r.trialCriteria[k]), "the rebuild is the trial the run evaluated")
        #expect(p.perturbations == record.perturbations)
        #expect(zip(p.compensators, record.compensators).allSatisfy { abs($0 - $1) < 1e-9 })
        #expect(w.isPreviewing && w.previewTitle == "Tolerance: trial \(k + 1), compensated" && doc.system == system)
        // The rebuilt system: element L1 moved as one, surfaces made irregular, the back focus compensated.
        let l1 = try #require(p.system.surfaces[1].elementPlacement)
        #expect(l1.last == system.surfaces[2].id && !l1.motion.isIdentity)
        #expect(p.system.surfaces.contains { if case .sagOverlay = $0.shape { true } else { false } })
        #expect(p.system.surfaces[system.imageIndex - 1].thickness == p.compensators[0])
        let changes = session.changes(p.perturbations)
        #expect(changes.count == session.resultRows.count && changes.allSatisfy { abs($0.fraction) <= 1 + 1e-9 })

        // A sensitivity case: one perturbation at its limit.
        let s = try #require(r.sensitivities.first { session.resultRows[$0.id].kind == .elementTiltX })
        session.preview(.sensitivity(operand: s.id, plus: true), in: w)
        try await wait { session.preview?.previewCase == .sensitivity(operand: s.id, plus: true) || session.previewError != nil }
        let q = try #require(session.preview)
        #expect(abs(q.criterion - s.criterionPlus) <= 1e-9 * abs(s.criterionPlus))
        #expect(q.perturbations.enumerated().allSatisfy { $0.offset == s.id ? $0.element == s.plus : $0.element == 0 })
        let only = try #require(session.changes(q.perturbations).first)
        #expect(session.changes(q.perturbations).count == 1 && abs(only.fraction - 1) < 1e-12 && only.text.hasSuffix("′"))
        #expect(w.previewTitle?.hasPrefix("Tolerance: Element tilt x L") == true)
        session.endPreview(in: w)
        #expect(!w.isPreviewing && doc.system == system)
    }

    /// Stop keeps the completed trials.
    @Test func stopKeepsPartialResults() async throws {
        let (system, w, context) = try await cooke()
        let doc = DesignDocument(system: system)
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
        #expect(r.hasNominal, "the Monte Carlo reports its own nominal")
    }
}
