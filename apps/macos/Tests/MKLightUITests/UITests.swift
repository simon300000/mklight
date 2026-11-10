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

    @Test func lengthsAcceptUnits() {
        #expect(LensValueParser.length("25.4") == 25.4)
        #expect(LensValueParser.length("1 in") == 25.4 && LensValueParser.length("2cm") == 20)
        #expect(LensValueParser.length("500 µm") == 0.5 && LensValueParser.length("0.1 m") == 100)
        #expect(LensValueParser.length("∞") == .infinity && LensValueParser.length("12 furlongs") == nil)
        #expect(LensValueParser.curvature(fromRadiusText: "100 mm") == .success(0.01))
        #expect(LensValueParser.thickness("3 mm", allowInfinite: false) == .success(3))
        #expect(LensValueParser.semiDiameter("5 mm") == .success(.fixed(5)))
    }

    @Test func radiusToCurvature() {
        #expect(LensValueParser.curvature(fromRadiusText: "50") == .success(0.02))
        #expect(LensValueParser.curvature(fromRadiusText: "1e-320") == .failure(.mustBeFinite), "subnormal radius → c = ∞")
        #expect(LensValueParser.radius("1e-320") == .failure(.mustBeFinite))
        #expect(LensValueParser.radius("-40") == .success(-40) && LensValueParser.radius("inf") == .success(.infinity))
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

    /// Projects `p` (world) to screen points in double precision.
    private func exactScreen(_ p: SIMD3<Double>, _ cam: ViewportCamera, _ mode: ViewportMode) -> SIMD2<Double> {
        let v = ViewportMath.rotation(cam, mode: mode) * (p - cam.center)
        return SIMD2(v.x, v.y) * cam.pointsPerMM
    }

    /// Largest distance (points) of the float-projected segment endpoints from
    /// the exact projected lines they belong to.
    private func maxLineError(_ scene: LayoutScene, _ cam: ViewportCamera, _ mode: ViewportMode, _ frame: GeometryFrame, _ size: CGSize) -> Double {
        let style = ViewportStyle(mode: mode, coloring: .field, selectedSurface: nil, stale: false, dark: false, scale: 1)
        let (segs, _) = SceneGeometry.build(scene, style: style, frame: frame)
        let m = ViewportMath.viewProjection(cam, mode: mode, frame: frame, size: size, depthRadius: 200)
        let lines = scene.polylines.filter { $0.points.count == 2 }
        var worst = 0.0
        for (seg, line) in zip(segs, lines) {
            let a = exactScreen(SIMD3(line.points[0].x, line.points[0].y, line.points[0].z), cam, mode)
            let b = exactScreen(SIMD3(line.points[1].x, line.points[1].y, line.points[1].z), cam, mode)
            let dir = simd_normalize(b - a)
            for p in [seg.p0, seg.p1] {
                let c = m * p
                let q = SIMD2(Double(c.x / c.w) * size.width / 2, Double(c.y / c.w) * size.height / 2)
                let d = q - a
                worst = max(worst, abs(d.x * dir.y - d.y * dir.x))
            }
        }
        return worst
    }

    /// At 10⁶ points/mm looking at the image plane, ~60 mm from the scene
    /// centre, a float32 scene-centred upload jitters by points; the
    /// camera-relative clipped frame stays far below a pixel.
    @Test(arguments: [ViewportMode.section2D, .orbit3D])
    func deepZoomIsJitterFree(mode: ViewportMode) {
        let focus = SIMD3<Double>(0, 0.0123456, 61.3304)
        let rays = (0..<8).map { k -> LayoutScene.Polyline in
            let a = Double(k) * 0.37
            // Steep rays, so errors along z show up across the line on screen.
            let start = Point3(focus.x + 20 * cos(a), focus.y + 30 * sin(a + 0.4), focus.z - 30)
            let end = Point3(focus.x + 1e-4 * sin(a), focus.y + 2e-4 * cos(a), focus.z + 1e-4)
            return .init(kind: .ray, points: [start, end], inSection: true)
        }
        let scene = LayoutScene(
            polylines: rays, elements: [], surfaces: [],
            bounds: Bounds3(min: Point3(-10, -10, -60), max: Point3(10, 10, 62)), wavelengthsUm: [0.55], fieldCount: 1)
        var cam = ViewportCamera()
        cam.yaw = 0.3
        cam.pitch = -0.2
        cam.center = focus
        cam.pointsPerMM = 1e6
        let size = CGSize(width: 1000, height: 800)
        let deep = SceneGeometry.frame(for: scene, camera: cam, mode: mode, size: size)
        #expect(deep.rotation != nil && deep.clipHalfExtent != nil)
        #expect(maxLineError(scene, cam, mode, deep, size) < 0.05)
        let centred = GeometryFrame(origin: SceneGeometry.origin(of: scene), rotation: nil, clipHalfExtent: nil)
        let coarse = maxLineError(scene, cam, mode, centred, size)
        #expect(coarse > 0.5, "the scene-centred frame really is too coarse here (\(coarse) pt)")
        // At ordinary zoom the cheap scene-centred frame is kept.
        cam.pointsPerMM = 10
        #expect(SceneGeometry.frame(for: scene, camera: cam, mode: mode, size: size) == centred)
    }

    @Test func clippingKeepsInsideAndCutsAtWindow() {
        let h = SIMD2<Double>(1, 1)
        #expect(SceneGeometry.clip(SIMD3(-3, 0, 0), SIMD3(3, 0, 6), to: h).map { [$0.0, $0.1] } == [SIMD3(-1, 0, 2), SIMD3(1, 0, 4)])
        #expect(SceneGeometry.clip(SIMD3(2, 2, 0), SIMD3(3, 5, 0), to: h) == nil)
        let tri = SceneGeometry.clip(polygon: [SIMD3(0, 0, 0), SIMD3(4, 0, 0), SIMD3(0, 4, 0)], to: h)
        #expect(tri.count == 4 && tri.allSatisfy { abs($0.x) <= 1 + 1e-12 && abs($0.y) <= 1 + 1e-12 })
    }

    @Test func sceneGeometryCentresAndFilters() throws {
        let scene = try MockOptics(system: SampleSystems.cookeTriplet()).layout()
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
    /// GPU, and both pipelines can be built. Skipped (not passed) without a
    /// Metal device.
    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "needs a Metal device"))
    @MainActor func shadersCompile() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let r = try ViewportRenderer(device: device, pixelFormat: .bgra8Unorm)
        #expect(r.sampleCount >= 1)
    }
}

