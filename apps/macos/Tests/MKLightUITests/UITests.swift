import Foundation
import Metal
import Testing
import simd

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

@Suite struct ParserTests {
    @Test func numbers() {
        #expect(LensValueParser.number(" 12.5 ") == 12.5)
        #expect(LensValueParser.number("\u{2212}3") == -3)
        #expect(LensValueParser.number("∞") == .infinity)
        #expect(LensValueParser.number("1e-3") == 0.001)
        #expect(LensValueParser.number("nan") == nil)
        #expect(LensValueParser.number("") == nil)
        #expect(LensValueParser.number("12mm") == nil)
    }

    @Test func radiusToCurvature() {
        #expect(LensValueParser.curvature(fromRadiusText: "50") == .success(0.02))
        #expect(LensValueParser.curvature(fromRadiusText: "inf") == .success(0))
        #expect(LensValueParser.curvature(fromRadiusText: "0") == .success(0))
        #expect(LensValueParser.curvature(fromRadiusText: "x") == .failure(.notANumber("x")))
    }

    @Test func thicknessAndSemiDiameter() {
        #expect(LensValueParser.thickness("inf", allowInfinite: true) == .success(.infinity))
        #expect(LensValueParser.thickness("inf", allowInfinite: false) == .failure(.mustBeFinite))
        #expect(LensValueParser.thickness("-12", allowInfinite: false) == .success(-12))
        #expect(LensValueParser.semiDiameter("auto") == .success(.auto))
        #expect(LensValueParser.semiDiameter("") == .success(.auto))
        #expect(LensValueParser.semiDiameter("12.000 A") == .failure(.notANumber("12.000 A")))
        #expect(LensValueParser.semiDiameter("7") == .success(.fixed(7)))
        #expect(LensValueParser.semiDiameter("-1") == .failure(.mustBePositive))
    }

    @Test func materials() {
        typealias E = LensValueParser.MaterialEdit
        #expect(LensValueParser.material("air") == .success(E(material: .air, interaction: .refract)))
        #expect(LensValueParser.material("n-bk7") == .success(E(material: .glass("N-BK7"), interaction: .refract)))
        #expect(LensValueParser.material("mirror") == .success(E(material: nil, interaction: .reflect)))
        #expect(LensValueParser.material("1.5168/64.17") == .success(E(material: .model(nd: 1.5168, vd: 64.17, dpgf: 0), interaction: .refract)))
        #expect(LensValueParser.material("n=1.6") == .success(E(material: .constant(n: 1.6), interaction: .refract)))
        #expect(LensValueParser.material("1.45") == .success(E(material: .constant(n: 1.45), interaction: .refract)))
        #expect(LensValueParser.material("schott:n-sf6") == .success(E(material: .catalog(catalog: "SCHOTT", name: "N-SF6"), interaction: .refract)))
        #expect(LensValueParser.material("0.9/-3") == .failure(.invalidMaterial("0.9/-3")))
        #expect(LensValueParser.material("bad glass!") == .failure(.invalidMaterial("bad glass!")))
    }
}

@Suite struct LensEditTests {
    @Test func editsAddressSurfacesById() {
        var s = SampleSystems.cookeTriplet()
        let id = s.surfaces[3].id
        LensEdit.insert(at: 1).apply(to: &s)
        LensEdit.setThickness(id, 1.5).apply(to: &s)
        #expect(s.surfaces[4].id == id && s.surfaces[4].thickness == 1.5)
        LensEdit.setMaterial(id, .init(material: nil, interaction: .reflect)).apply(to: &s)
        #expect(s.surfaces[4].interaction == .reflect && s.surfaces[4].material == .glass("F2"))
        LensEdit.setCurvature(s.surfaces[1].id, 0.01).apply(to: &s)
        #expect(s.surfaces[1].shape == .conic(curvature: 0.01, conic: 0), "inserted plane becomes a sphere")
    }

    @Test func rowsReflectModelAndParseEdits() throws {
        let s = SampleSystems.cookeTriplet()
        let rows = LensRow.rows(for: s, layout: nil)
        #expect(rows.map(\.indexText) == ["OBJ", "1", "2", "3", "4", "STO", "6", "7", "IMA"])
        #expect(rows[0].thickness == "Infinity")
        #expect(rows[1].radius == "22.0136")
        #expect(rows[1].material == "N-SK16")
        #expect(rows[5].radius == "—" || rows[5].radius == "Infinity")
        #expect(!rows[8].isEditable(.thickness) && !rows[0].isEditable(.type))
        let edit = try #require(rows[2].edit(.radius, text: "-400"))
        #expect(edit == .success(.setCurvature(s.surfaces[2].id, -1 / 400.0)))
        #expect(rows[2].edit(.thickness, text: "inf").map { (try? $0.get()) == nil } == true, "only the object may be infinite")
        #expect(rows[0].edit(.thickness, text: "inf").map { (try? $0.get()) != nil } == true)
    }

    @Test func autoSemiDiameterFromLayout() {
        let s = SampleSystems.cookeTriplet()
        let scene = MockOptics(system: s).layout(LayoutSettings())
        let rows = LensRow.rows(for: s, layout: scene)
        #expect(rows[1].semiDiameter.hasSuffix(" A"))
        #expect(rows[1].semiDiameterExact == "auto")
    }
}

@Suite struct ViewportMathTests {
    @Test func sectionViewMapsAxisRightAndYUp() {
        var cam = ViewportCamera()
        cam.pointsPerMM = 2
        let m = ViewportMath.viewProjection(cam, mode: .section2D, origin: .zero, size: CGSize(width: 200, height: 100), depthRadius: 100)
        let p = m * SIMD4<Float>(0, 10, 25, 1)  // y = 10 mm, z = 25 mm
        #expect(abs(p.x - 25 * 2 * 2 / 200) < 1e-6)
        #expect(abs(p.y - 10 * 2 * 2 / 100) < 1e-6)
        #expect(p.z >= 0 && p.z <= 1)
    }

    @Test func originShiftIsExact() {
        // Large world coordinates: the scene-centred input stays small and the
        // double-precision translation lands the point exactly.
        var cam = ViewportCamera()
        cam.center = SIMD3(0, 0, 100_000.25)
        cam.pointsPerMM = 1
        let origin = SIMD3<Double>(0, 0, 100_000)
        let m = ViewportMath.viewProjection(cam, mode: .section2D, origin: origin, size: CGSize(width: 2, height: 2), depthRadius: 10)
        let p = m * SIMD4<Float>(0, 0, 0.25, 1)  // world z = 100000.25 → centre
        #expect(abs(p.x) < 1e-6)
    }

    @Test func zoomKeepsCursorPointFixed() {
        var cam = ViewportCamera()
        cam.pointsPerMM = 3
        cam.center = SIMD3(0, 1, 5)
        let cursor = CGPoint(x: 30, y: -12)
        let r = ViewportMath.rotation(cam, mode: .section2D)
        let before = cam.center + r.transpose * SIMD3(30 / 3, -12 / 3, 0)
        let z = ViewportMath.zoom(cam, mode: .section2D, factor: 2.5, about: cursor)
        let after = z.center + r.transpose * SIMD3(30 / z.pointsPerMM, -12 / z.pointsPerMM, 0)
        #expect(simd_length(before - after) < 1e-12)
    }

    @Test func fitFramesPoints() {
        let pts = [SIMD3<Double>(0, -10, 0), SIMD3(0, 10, 100)]
        let c = ViewportMath.fit(ViewportCamera(), mode: .section2D, points: pts, size: CGSize(width: 560, height: 400))
        #expect(abs(c.center.z - 50) < 1e-12 && abs(c.center.y) < 1e-12)
        #expect(abs(c.pointsPerMM - 560 / (100 * 1.12)) < 1e-9)
        #expect(c.isFitted)
    }

    @Test func orbitRotationIsOrthonormal() {
        var cam = ViewportCamera()
        cam.yaw = 0.7
        cam.pitch = -0.4
        let r = ViewportMath.rotation(cam, mode: .orbit3D)
        let i = r.transpose * r
        for c in 0..<3 {
            for k in 0..<3 { #expect(abs(i[c][k] - (c == k ? 1 : 0)) < 1e-12) }
        }
        #expect(abs(simd_determinant(r) - 1) < 1e-12)
    }

    @Test func niceLengths() {
        #expect(ViewportMath.niceLength(1.3) == 1)
        #expect(ViewportMath.niceLength(2.9) == 2)
        #expect(ViewportMath.niceLength(4) == 5)
        #expect(ViewportMath.niceLength(0.08) == 0.1)
    }

    @Test func sceneGeometryCentresAndFilters() {
        let scene = MockOptics(system: SampleSystems.cookeTriplet()).layout(LayoutSettings())
        let style = ViewportStyle(mode: .section2D, coloring: .field, selectedSurface: nil, stale: false, dark: false, scale: 2)
        let (segs2D, fills) = SceneGeometry.build(scene, style: style)
        var s3 = style
        s3.mode = .orbit3D
        let (segs3D, fills3D) = SceneGeometry.build(scene, style: s3)
        #expect(segs3D.count > segs2D.count, "3-D adds sagittal profiles, rims and skew rays")
        #expect(fills.count == 3 * 40 * 6 && fills3D.isEmpty)
        let maxAbs = segs2D.flatMap { [$0.p0, $0.p1] }.map { max(abs($0.x), abs($0.y), abs($0.z)) }.max() ?? 0
        let b = scene.bounds
        #expect(Double(maxAbs) <= max(b.max.z - b.min.z, b.max.y - b.min.y) / 2 + 1e-3)
        #expect(MemoryLayout<SegmentInstance>.stride == 64 && MemoryLayout<FillVertex>.stride == 32)
        #expect(MemoryLayout<Uniforms>.stride == 80)
    }
}

@Suite struct ShaderTests {
    /// The viewport shader source compiles with safe math on this machine's
    /// GPU, and both pipelines can be built.
    @Test @MainActor func shadersCompile() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let r = try ViewportRenderer(device: device, pixelFormat: .bgra8Unorm)
        #expect(r.sampleCount >= 1)
    }
}

@Suite @MainActor struct CoordinatorTests {
    @Test func runsVisibleAnalysesAndTracksStaleness() async throws {
        let c = AnalysisCoordinator(engine: MockEngine(simulatedLatency: .milliseconds(10)))
        c.debounce = .milliseconds(5)
        c.setVisible([.layout, .firstOrder])
        var s = SampleSystems.cookeTriplet()
        c.update(system: s, immediately: true)
        try await waitUntil { c.output(.layout) != nil && c.output(.firstOrder) != nil }
        #expect(!c.isStale(.layout))
        #expect(c.output(.spotDiagram) == nil, "hidden analyses do not run")
        let firstHash = c.snapshot?.hash

        s.surfaces[2].thickness += 1
        c.update(system: s)
        try await waitUntil { c.snapshot?.hash != firstHash }
        #expect(c.isStale(.layout) || c.output(.layout)?.runInfo.snapshotHash == c.snapshot?.hash)
        try await waitUntil { !c.isStale(.layout) && !c.isBusy }
        #expect(c.output(.layout)?.runInfo.snapshotHash == c.snapshot?.hash)
    }

    @Test func invalidModelBlocksAnalyses() async throws {
        let c = AnalysisCoordinator(engine: MockEngine(simulatedLatency: .zero))
        c.setVisible([.firstOrder])
        c.update(system: SampleSystems.singlet(), immediately: true)
        try await waitUntil { c.output(.firstOrder) != nil }
        var bad = SampleSystems.singlet()
        bad.stop = 0
        c.update(system: bad, immediately: true)
        try await waitUntil { !c.validationErrors.isEmpty }
        #expect(c.isStale(.firstOrder))
    }

    @Test func newerRequestCancelsOlderJob() async throws {
        let c = AnalysisCoordinator(engine: MockEngine(simulatedLatency: .seconds(20)))
        c.setVisible([.mtf])
        c.update(system: SampleSystems.cookeTriplet(), immediately: true)
        try await waitUntil { if case .running = c.slot(.mtf).phase { true } else { false } }
        let first = c.slot(.mtf).job
        var settings = AnalysisSettingsSet()
        settings.mtf.maxFrequency = 50
        c.update(settings: settings)
        let second = c.slot(.mtf).job
        #expect(first != nil && second != nil && first !== second)
        await #expect(throws: EngineError.cancelled) { try await first?.result() }
        c.cancelAll()
        try await waitUntil { c.slot(.mtf).phase == .cancelled }
    }

    private func waitUntil(_ timeout: Duration = .seconds(5), _ cond: () -> Bool) async throws {
        let clock = ContinuousClock()
        let end = clock.now + timeout
        while !cond() {
            guard clock.now < end else {
                Issue.record("condition not met within \(timeout)")
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
}

@Suite @MainActor struct DocumentTests {
    @Test func editsAreUndoableValueSnapshots() {
        let doc = ProjectDocument(system: SampleSystems.cookeTriplet())
        let undo = UndoManager()
        undo.groupsByEvent = false
        let original = doc.system
        undo.beginUndoGrouping()
        let changed = doc.edit("Change Thickness", undoManager: undo) { $0.surfaces[2].thickness = 7 }
        undo.endUndoGrouping()
        #expect(changed)
        let unchanged = doc.edit("No-op", undoManager: undo) { $0.surfaces[2].thickness = 7 }
        #expect(!unchanged, "no-op edits register nothing")
        #expect(undo.undoActionName == "Change Thickness")
        undo.undo()
        #expect(doc.system == original)
        undo.redo()
        #expect(doc.system.surfaces[2].thickness == 7)
    }

    @Test func snapshotForSavingMatchesModel() throws {
        let doc = ProjectDocument(system: SampleSystems.singlet())
        doc.edit("Rename", undoManager: nil) { $0.name = "Renamed" }
        let file = try doc.snapshot(contentType: .mklightProject)
        #expect(file.system.name == "Renamed")
        #expect(try ProjectFile.decode(file.encoded()) == file)
    }
}
