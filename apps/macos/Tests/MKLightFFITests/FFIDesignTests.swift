import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// The app's flows on the real engine: every dedicated result view reads
/// the engine's JSON, Stop keeps an optimisation's result (ABI 1.2), solves
/// commit through `mk_session_apply_solves`.
@Suite(.serialized, engineLinked) @MainActor struct FFIDesignTests {
    private func engine() throws -> FFIEngine {
        try FFIEngine(configuration: ["worker_threads": .integer(2), "progress_interval_ms": .integer(10)])
    }

    private func wait(_ timeout: Duration = .seconds(60), _ cond: () -> Bool) async throws {
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

    /// The dedicated views understand the engine's results (docs/RESULTS.md).
    @Test func dedicatedViewsReadEngineResults() async throws {
        let analyses = [
            "first_order", "spot_diagram", "ray_fan", "opd_fan", "field_curvature_distortion", "grid_distortion", "seidel",
            "longitudinal_aberration", "chromatic_focal_shift", "lateral_colour", "relative_illumination", "footprint",
            "encircled_energy", "geometric_mtf", "wavefront_map", "psf", "mtf", "mtf_vs_field", "through_focus_mtf",
            "diffraction_encircled_energy", "line_spread", "ghost_paths", "system_transmission", "polarization_pupil_map",
            "best_focus",
        ]
        let panes = analyses.map { AnalysisPane(analysis: $0) }
        let w = Workspace(engine: try engine(), panes: panes)
        w.debounce = .zero
        await w.loadEngineMetadata()
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        w.update(system: system, immediately: true)
        try await wait(.seconds(120)) { panes.allSatisfy { $0.output != nil || $0.phase != .idle && !$0.isRunning } && !w.isBusy }
        let c = AnalysisContext(system: system)
        for p in panes {
            guard let out = p.output else {
                Issue.record("\(p.analysis): \(p.phase)")
                continue
            }
            let dedicated: Bool =
                switch p.analysis {
                case "first_order": FirstOrderView(output: out) != nil
                case "spot_diagram": SpotDiagramView(output: out, context: c) != nil
                case "ray_fan", "opd_fan": FanView(output: out, context: c) != nil
                case "field_curvature_distortion": FieldCurvatureView(output: out, context: c) != nil
                case "grid_distortion": GridDistortionView(output: out) != nil
                case "seidel": SeidelView(output: out, context: c) != nil
                case "longitudinal_aberration": LongitudinalView(output: out, context: c) != nil
                case "chromatic_focal_shift": ChromaticFocalShiftView(output: out) != nil
                case "lateral_colour": LateralColourView(output: out, context: c) != nil
                case "relative_illumination": RelativeIlluminationView(output: out, context: c) != nil
                case "footprint": FootprintView(output: out, context: c) != nil
                case "encircled_energy", "diffraction_encircled_energy": EncircledEnergyView(output: out, context: c) != nil
                case "geometric_mtf", "mtf": MTFView(output: out, context: c) != nil
                case "mtf_vs_field": MTFVsFieldView(output: out, context: c) != nil
                case "through_focus_mtf": ThroughFocusMTFView(output: out, context: c) != nil
                case "wavefront_map": WavefrontView(output: out, context: c) != nil
                case "psf": PSFView(output: out, context: c) != nil
                case "line_spread": LineSpreadView(output: out, context: c) != nil
                case "ghost_paths": GhostPathsView(output: out, context: c) != nil
                case "system_transmission": TransmissionView(output: out, context: c) != nil
                case "polarization_pupil_map": PolarizationView(output: out) != nil
                case "best_focus": BestFocusView(output: out, context: c) != nil
                default: false
                }
            #expect(dedicated, "\(p.analysis): the dedicated view cannot read the engine's result")
            #expect(out.backend != nil, "\(p.analysis): the envelope's backend report")
        }
        #expect(EngineLayout.scene(from: w.layout.output) != nil, "layout buffers")
        #expect(w.issues.allSatisfy { $0.severity != .error })
        #expect(FirstOrderData(w.summary.output?.json)?.entry(["efl"]) != nil)
    }

    /// Stop ends an optimisation early and keeps its best design; the
    /// result applies as one undoable edit.
    @Test func stopKeepsTheOptimizedDesign() async throws {
        let w = Workspace(engine: try engine())
        w.debounce = .zero
        await w.loadEngineMetadata()
        let doc = DesignDocument(system: try #require(SampleSystems.bundled("double_gauss")))
        w.update(system: doc.system, immediately: true)
        try await wait { w.snapshot != nil }
        let session = OptimizeSession()
        session.algorithm = "de"
        session.maxIterations = 100_000
        session.start(in: w, variables: [])
        try await wait { session.trace.count >= 2 }
        #expect(session.canStop)
        session.stop()
        try await wait { session.state == .finished || { if case .failed = session.state { true } else { false } }() }
        let r = try #require(session.result, "\(session.state)")
        #expect(r.stopped && r.status == "stopped")
        let undo = UndoManager()
        let before = doc.system
        if r.improved {
            session.apply(to: doc, undoManager: undo)
            #expect(doc.system == r.system)
            undo.undo()
            #expect(doc.system == before)
        }
    }

    /// Apply Solves commits the engine's solved system.
    @Test func solvesCommitThroughTheEngine() async throws {
        var s = try #require(SampleSystems.bundled("cooke_triplet"))
        let back = ParamPath.thickness(surface: s.surfaces[s.imageIndex - 1].id)
        s.solves = [Solve(target: back, kind: .marginalRayHeight(height: 0))]
        let w = Workspace(engine: try engine())
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: s, immediately: true)
        try await wait { w.snapshot != nil }
        let doc = DesignDocument(system: s)
        let session = SolvesSession()
        await session.dryRun(in: w)
        #expect(session.state == .finished && session.outcomes.count == 1)
        let solved = try #require(session.outcomes.first?.value)
        await session.commit(in: w, document: doc, undoManager: nil)
        #expect(session.committed)
        #expect(abs((doc.system.value(at: back) ?? 0) - solved) < 1e-9)
    }
}
