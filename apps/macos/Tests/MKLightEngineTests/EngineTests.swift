import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel

private let fixtureDir = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("MKLightModelTests/Fixtures")

private let fast = MockEngine(simulatedLatency: .zero)

@Suite struct RunInfoTests {
    @Test func decodesEngineRunInfo() throws {
        let data = try Data(contentsOf: fixtureDir.appendingPathComponent("run_info.json"))
        let info = try MKLightJSON.decode(RunInfo.self, from: data)
        #expect(info.precision == .f32GPU)
        #expect(info.seed == 42)
        #expect(info.snapshotHash.hex.count == 64)
        #expect(info.settings["rings"] == .number(6))
        let back = try JSONValue.parse(MKLightJSON.encode(info))
        #expect(try JSONValue.parse(data).differences(from: back).isEmpty)
    }
}

@Suite struct SnapshotTests {
    @Test func hashIsStableAndSensitive() async throws {
        let a = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        let b = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        #expect(a.hash == b.hash && a == b)
        var s = SampleSystems.cookeTriplet()
        s.surfaces[2].thickness += 1e-9
        let c = try await fast.openSnapshot(s)
        #expect(c.hash != a.hash)
        #expect(a.hash.short.count == 12)
    }

    @Test func invalidModelIsRejected() async {
        var s = OpticalSystem.newEmpty()
        s.stop = 0
        await #expect(throws: EngineError.self) { try await fast.openSnapshot(s) }
        let issues = await fast.validate(s)
        #expect(!issues.isEmpty)
    }
}

@Suite struct JobTests {
    @Test func progressIsMonotoneAndResultCarriesRunInfo() async throws {
        let engine = MockEngine(simulatedLatency: .milliseconds(40))
        let snap = try await engine.openSnapshot(SampleSystems.cookeTriplet())
        var settings = SpotDiagramSettings()
        settings.rings = 3
        let request = AnalysisRequest.spotDiagram(settings)
        let job = engine.submit(request, on: snap)
        var fractions: [Double] = []
        for await p in job.progress { fractions.append(p.fraction) }
        #expect(fractions.count >= 3)
        #expect(zip(fractions, fractions.dropFirst()).allSatisfy { $0 <= $1 })
        #expect(fractions.last == 1)
        let out = try await job.result()
        #expect(out.runInfo.snapshotHash == snap.hash)
        #expect(out.runInfo.analysis == "spot_diagram")
        #expect(out.runInfo.backend == "mock")
        #expect(out.runInfo.warnings.contains(MockEngine.mockWarning))
        #expect(out.runInfo.isCurrent(for: snap, request: request))
        settings.rings = 4
        #expect(!out.runInfo.isCurrent(for: snap, request: .spotDiagram(settings)), "different settings → stale")
    }

    @Test func cancellationStopsJob() async throws {
        let engine = MockEngine(simulatedLatency: .seconds(30))
        let snap = try await engine.openSnapshot(SampleSystems.cookeTriplet())
        let job = engine.submit(.mtf(MTFSettings()), on: snap)
        let clock = ContinuousClock()
        let t0 = clock.now
        try await Task.sleep(for: .milliseconds(20))
        job.cancel()
        await #expect(throws: EngineError.cancelled) { try await job.result() }
        #expect(clock.now - t0 < .seconds(5))
        // The progress stream finishes too.
        for await _ in job.progress {}
    }

    @Test func awaiterCancellationDoesNotCancelJob() async throws {
        let engine = MockEngine(simulatedLatency: .milliseconds(50))
        let snap = try await engine.openSnapshot(SampleSystems.singlet())
        let job = engine.submit(.firstOrder(FirstOrderSettings()), on: snap)
        let waiter = Task { try await job.result() }
        waiter.cancel()
        _ = try? await waiter.value
        let out = try await job.result()
        #expect(out.result.kind == .firstOrder)
    }

    @Test func randomSpotIsReproducibleFromSeed() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        var s = SpotDiagramSettings()
        s.pattern = .random
        s.seed = 1234
        func points(_ s: SpotDiagramSettings) async throws -> [SpotDiagramResult.Point] {
            guard case .spotDiagram(let r) = try await fast.submit(.spotDiagram(s), on: snap).result().result else { return [] }
            return r.fields.flatMap(\.points)
        }
        let a = try await points(s), b = try await points(s)
        #expect(!a.isEmpty && a == b)
        s.seed = 4321
        #expect(try await points(s) != a)
        let out = try await fast.submit(.spotDiagram(s), on: snap).result()
        #expect(out.runInfo.seed == 4321)
    }

    @Test func requestAndOutputCodableRoundTrip() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        for kind in AnalysisKind.allCases {
            let req = AnalysisSettingsSet().request(for: kind)
            #expect(try MKLightJSON.decode(AnalysisRequest.self, from: MKLightJSON.encode(req)) == req)
            let out = try await fast.submit(req, on: snap).result()
            #expect(out.result.kind == kind)
            let back = try MKLightJSON.decode(AnalysisOutput.self, from: MKLightJSON.encode(out))
            #expect(back == out, "\(kind) output round trip")
        }
    }
}

/// Known-answer checks of the mock's paraxial and real-ray code, so the
/// synthetic data the UI is developed against is at least self-consistent.
@Suite struct MockOpticsTests {
    @Test func thickLensFocalLengths() {
        let s = SampleSystems.singlet()
        let o = MockOptics(system: s)
        let n = 1.51680, r1 = 103.4, r2 = -103.4, t = 5.0
        // Lensmaker's equation for a thick lens; the mock's Cauchy fit passes
        // through n_d exactly.
        let power = (n - 1) * (1 / r1 - 1 / r2 + (n - 1) * t / (n * r1 * r2))
        let f = 1 / power
        let fo = o.firstOrder(FirstOrderSettings())
        #expect(abs(fo["efl"]! - f) < 1e-9 * f)
        let bfl = f * (1 - (n - 1) * t / (n * r1))
        #expect(abs(fo["bfl"]! - bfl) < 1e-9 * f)
        #expect(abs(fo["epd"]! - 20) < 1e-12)
        #expect(abs(fo["image_f_number"]! - f / 20) < 1e-9)
    }

    @Test func tripletFirstOrderMatchesIndependentTrace() {
        // Reference values from an independent y-nu trace at n_d.
        let fo = MockOptics(system: SampleSystems.cookeTriplet()).firstOrder(FirstOrderSettings())
        #expect(abs(fo["efl"]! - 47.10087534020714) < 1e-9)
        #expect(abs(fo["bfl"]! - 41.36144145724218) < 1e-9)
    }

    @Test func paraboloidFocusesOnAxisPerfectly() {
        // Concave paraboloid (R = −200, k = −1): every axial ray meets the
        // focus at z = −100 exactly.
        let s = OpticalSystem(
            name: "paraboloid", aperture: .entrancePupilDiameter(80),
            surfaces: [
                Surface(id: SurfaceID(0), thickness: .infinity),
                Surface(id: SurfaceID(1), label: "STOP", shape: .conic(curvature: -1 / 200.0, conic: -1), thickness: -100, interaction: .reflect),
                Surface(id: SurfaceID(2), thickness: 0),
            ], stop: 1)
        let o = MockOptics(system: s)
        let r = o.spot(SpotDiagramSettings())
        #expect(r.fields[0].points.count == 127 * 3)
        #expect(r.fields[0].rmsRadius < 1e-12)
        #expect(r.fields[0].geoRadius < 1e-12)
        #expect(abs(o.firstOrder(FirstOrderSettings())["efl"]! - 100) < 1e-12)
    }

    @Test func sphericalMirrorHasThirdOrderAberration() {
        // A spherical mirror at the paraxial focus: transverse aberration
        // grows as Py³ for small apertures.
        let s = OpticalSystem(
            name: "sphere", aperture: .entrancePupilDiameter(10),
            surfaces: [
                Surface(id: SurfaceID(0), thickness: .infinity),
                Surface(id: SurfaceID(1), label: "STOP", shape: .conic(curvature: -1 / 200.0, conic: 0), thickness: -100, interaction: .reflect),
                Surface(id: SurfaceID(2), thickness: 0),
            ], stop: 1)
        let o = MockOptics(system: s)
        var fan = FanSettings()
        fan.samples = 5
        fan.wavelengths = .primary
        let curve = o.rayFan(fan).fields[0].tangential[0]
        let e1 = curve.values[4]!, eHalf = curve.values[3]!
        #expect(e1 != 0)
        #expect(abs(eHalf / e1 - 0.125) < 2e-3)
        // Third order: longitudinal aberration h²/(4R) times the ray slope
        // 2h/R gives |ε(1)| = h³/(2R²) at the paraxial focus.
        let h = 5.0, rr = 200.0
        let expected = h * h * h / (2 * rr * rr)
        #expect(abs(abs(e1) - expected) < 0.01 * expected)
    }

    @Test func diffractionMTFKnownValues() {
        #expect(MockOptics.diffractionMTF(0, cutoff: 100) == 1)
        let half = MockOptics.diffractionMTF(50, cutoff: 100)
        #expect(abs(half - 2 / Double.pi * (acos(0.5) - 0.5 * 0.75.squareRoot())) < 1e-15)
        #expect(MockOptics.diffractionMTF(100, cutoff: 100) == 0)
    }

    @Test func layoutSceneOfTriplet() {
        let o = MockOptics(system: SampleSystems.cookeTriplet())
        var settings = LayoutSettings()
        settings.include3D = false
        let scene = o.layout(settings)
        #expect(scene.elements.count == 3)
        #expect(scene.surfaces.count == 9)
        let rays = scene.polylines.filter { $0.kind == .ray }
        #expect(rays.count == 3 * 7)
        #expect(rays.allSatisfy { $0.inSection && $0.wavelength == 1 })
        #expect(rays.filter { !$0.vignetted }.allSatisfy { $0.points.count == 9 })
        #expect(scene.polylines.contains { $0.kind == .stop && $0.surface == 5 })
        #expect(abs(scene.surfaces[8].vertex.z - 61.33041) < 1e-9)
        #expect(scene.bounds.min.z < 0 && scene.bounds.max.z > 60)
    }

    @Test func distortionIsZeroOnAxisAndFieldCurvatureFinite() {
        let o = MockOptics(system: SampleSystems.cookeTriplet())
        let r = o.fieldCurvature(FieldCurvatureSettings())
        #expect(r.fieldValues.first == 0 && r.fieldValues.last == 20)
        for c in r.curves {
            #expect(c.distortionPercent[0] == 0)
            #expect(c.tangential.allSatisfy { $0.map { abs($0) < 5 } ?? false })
        }
    }

    @Test func psfPeakIsStrehlBounded() {
        var s = PSFSettings()
        s.imageSampling = 32
        let r = MockOptics(system: SampleSystems.cookeTriplet()).psf(s)
        #expect(r.values.count == 32 * 32)
        #expect(r.strehl > 0 && r.strehl <= 1)
        #expect(r.values.allSatisfy { $0 >= 0 && $0.isFinite })
    }
}
