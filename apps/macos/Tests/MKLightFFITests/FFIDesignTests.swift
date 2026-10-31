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
        let scene = try #require(EngineLayout.scene(from: w.layout.output), "layout buffers")
        let rays = scene.polylines.filter { $0.kind == .ray }
        #expect(!rays.isEmpty && rays.allSatisfy { $0.field != nil }, "rays are the polylines with a field")
        #expect(scene.polylines.filter { $0.kind == .stop }.allSatisfy { $0.surface == system.stop })
        #expect(!scene.elements.isEmpty)
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

extension FFIDesignTests {
    /// The field maps, the glass map and the image simulation read the
    /// real engine's results, and their key numbers are sane.
    @Test func mapViewsReadEngineResults() async throws {
        let e = try engine()
        func run(_ sample: String, _ analysis: String, _ settings: [String: JSONValue] = [:]) async throws -> AnalysisOutput {
            let snap = try await e.openSnapshot(try #require(SampleSystems.bundled(sample)))
            return try await e.submit(EngineAnalysisRequest(analysis: analysis, settings: settings), on: snap).result()
        }
        func keys(_ analysis: String, _ out: AnalysisOutput) -> [String: String] {
            Dictionary(
                AnalysisResultView.keyNumbers(analysis: analysis, output: out, context: AnalysisContext()).map { ($0.label, $0.value) },
                uniquingKeysWith: { a, _ in a })
        }

        // Pupil aberration: a 5 × 5 map on three samples, and a y sweep.
        for sample in ["cooke_triplet", "double_gauss", "anamorphic"] {
            let out = try await run(sample, "pupil_aberration")
            #expect(PupilAberrationView(output: out) != nil, "\(sample)")
            let d = try #require(FieldMapData(output: out))
            #expect(d.layout == .map && d.nx == 5 && d.ny == 5)
            let x = try #require(d.columnIndex("stop_error_x"))
            #expect(abs(d.value(12, x)) < 1e-9, "no stop error on axis")
            #expect(keys("pupil_aberration", out)["Max stop error"] != nil)
        }
        let sweep = try #require(
            FieldMapData(output: try await run("cooke_triplet", "pupil_aberration", ["mode": .string("y"), "grid": .array([.integer(5), .integer(9)])])))
        #expect(sweep.layout == .sweepY && sweep.count == 9)
        #expect(sweep.sweep(0).map(\.0) == [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

        // Relative illumination: 1 on axis, falling off, rotationally symmetric.
        let ri = try await run("double_gauss", "relative_illumination_map", ["grid": .array([.integer(7), .integer(7)])])
        #expect(RelativeIlluminationMapView(output: ri) != nil)
        let rd = try #require(FieldMapData(output: ri))
        let c = try #require(rd.columnIndex("relative_illumination"))
        let range = try #require(rd.range(c))
        #expect(abs(range.upperBound - 1) < 1e-6 && range.lowerBound > 0.3 && range.lowerBound < 1)
        #expect(rd.isRadial(c, tolerance: 1e-2))
        #expect(!Contours.illuminationLevels(in: range).isEmpty)
        #expect(Contours.illuminationLevels(in: range).allSatisfy { !Contours.mapSegments(rd.grid(c), level: $0).isEmpty })
        // Corners outside the field carry no data (not 0), drawn as such.
        #expect(rd.outsideFieldIsNoData && rd.grid(c).noData == .unitDisk)
        #expect(keys("relative_illumination_map", ri)["Min RI"] == String(format: "%.3f", range.lowerBound))

        // Chief ray angle against a sensor limit the corner exceeds.
        let limit: JSONValue = .array([.array([.number(0), .number(5)]), .array([.number(20), .number(24)])])
        let cra = try await run("cooke_triplet", "chief_ray_angle", ["sensor_limit": limit])
        #expect(ChiefRayAngleView(output: cra) != nil)
        let cd = try #require(FieldMapData(output: cra))
        let inc = try #require(cd.columnIndex("incidence")), margin = try #require(cd.columnIndex("margin"))
        #expect(abs(cd.value(12, inc)) < 1e-9 && abs(cd.value(12, margin) - 5) < 1e-9)
        let worst = try #require(cd.argmax(inc))
        #expect(cd.value(worst, inc) > 10 && cd.value(worst, inc) < 45)
        #expect(abs(cd.value(worst, margin) - (FieldMapData.limit(FieldMapData.sensorLimit(limit), at: cd.imageRadius[worst]) - cd.value(worst, inc))) < 1e-9)
        #expect(cd.isRadial(inc))
        let curve = cd.radial(inc)
        #expect(ChiefRayAngleView.bands(curve: curve, limit: FieldMapData.sensorLimit(limit)).contains { $0.violated })
        let ck = keys("chief_ray_angle", cra)
        #expect(ck["Max CRA"] != nil && ck["Min margin"]?.hasPrefix("-") == true)
        // The anamorphic sample is not rotationally symmetric: shown as a map.
        let ana = try #require(FieldMapData(output: try await run("anamorphic", "chief_ray_angle")))
        #expect(!ana.isRadial(try #require(ana.columnIndex("incidence"))))

        // Glass map of the built-in catalogue with the design's glasses.
        let gm = try await run("double_gauss", "glass_map")
        #expect(GlassMapView(output: gm, context: AnalysisContext()) != nil)
        let g = try #require(GlassMapData(output: gm))
        #expect(g.glasses.count >= 10)
        #expect(g.glasses.allSatisfy { $0.nd > 1.3 && $0.nd < 2.2 && $0.vd > 15 && $0.vd < 100 })
        let sk16 = try #require(g.glasses.first { $0.name == "N-SK16" })
        let f2 = try #require(g.glasses.first { $0.name == "F2" })
        #expect(sk16.surfaces == [1, 3, 8, 10] && f2.surfaces == [4, 7])
        #expect(Set(g.designHighlights.map(\.id)) == [sk16.id, f2.id])
        #expect(abs(f2.deltaPgF) < 0.005, "F2 defines the normal line")
        #expect(keys("glass_map", gm)["In design"] == "2")

        // Image simulation of the default grid pattern.
        let sim = try await run("cooke_triplet", "image_simulation")
        #expect(ImageSimulationView(output: sim) != nil)
        let s = try #require(ImageSimulationData(output: sim))
        #expect(s.image.width == 64 && s.image.height == 64 && s.image.channels == 1)
        #expect(s.nodes.count == s.fieldGrid * s.fieldGrid && s.pixelSize.x > 0)
        #expect(s.source == .pattern("grid") && s.sourceImage?.width == 64)
        #expect(keys("image_simulation", sim).count == 3)
        // An imported buffer, bright in its upper half (row 0 is the low-y
        // edge): the image keeps that orientation, and the run's settings
        // carry the samples the source view shows.
        let w = 32, h = 32
        let samples: [Double] = (0..<(w * h)).map { k -> Double in k / w >= h / 2 ? 1 : 0.05 }
        let src = LinearImage(width: w, height: h, channels: 1, data: samples)
        let size: JSONValue = .array([.integer(32), .integer(32)])
        let imported = try await run("cooke_triplet", "image_simulation", ["source": src.sourceJSON, "size": size])
        let id = try #require(ImageSimulationData(output: imported))
        #expect(id.source == ImageSimulationData.Source.buffer(src))
        let out = id.image.data
        func mean(_ rows: Range<Int>) -> Double {
            var total = 0.0
            for y in rows { for x in 0..<w { total += out[y * w + x] } }
            return total / Double(rows.count * w)
        }
        #expect(mean(20..<30) > 3 * mean(2..<12), "high rows are the bright upper half")
    }
}
