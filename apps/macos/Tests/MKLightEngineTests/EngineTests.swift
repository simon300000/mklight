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
        #expect(info.settings["rings"] == .integer(6) && info.settings["rings"] == .number(6))
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
        s.execution.seed = 1234
        func points(_ s: SpotDiagramSettings) async throws -> [SpotDiagramResult.Point] {
            guard case .spotDiagram(let r) = try await fast.submit(.spotDiagram(s), on: snap).result().result else { return [] }
            return r.fields.flatMap(\.points)
        }
        let a = try await points(s), b = try await points(s)
        #expect(!a.isEmpty && a == b)
        s.execution.seed = 4321
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
    /// Seeds are full 64-bit values. Two seeds that round to the same
    /// `Double` must still give a stale result, and the recorded settings must
    /// carry the seed exactly (an engine could not reproduce a rounded one).
    @Test func seedsAbove2To53AreExact() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        var s = SpotDiagramSettings()
        s.pattern = .random
        s.rings = 2
        s.execution.seed = 1 << 60
        let out = try await fast.submit(.spotDiagram(s), on: snap).result()
        #expect(out.runInfo.isCurrent(for: snap, request: .spotDiagram(s)))
        var other = s
        other.execution.seed = (1 << 60) + 1
        #expect(Double(other.execution.seed) == Double(s.execution.seed), "the two seeds collide as Double")
        #expect(!out.runInfo.isCurrent(for: snap, request: .spotDiagram(other)))
        s.execution.seed = 12_345_678_901_234_567_891
        let json = String(decoding: try MKLightJSON.encode(AnalysisRequest.spotDiagram(s).settingsJSON()), as: UTF8.self)
        #expect(json.contains("\"seed\":12345678901234567891"))
        let info = try await fast.submit(.spotDiagram(s), on: snap).result().runInfo
        let decoded = try MKLightJSON.decode(RunInfo.self, from: MKLightJSON.encode(info))
        #expect(decoded.settings["execution"]?["seed"] == .unsigned(12_345_678_901_234_567_891))
        #expect(decoded.seed == 12_345_678_901_234_567_891)
    }

    /// Backend and precision are part of a job's identity: changing either
    /// makes an existing result stale, and the run info records the request.
    @Test func executionSettingsTakePartInStaleness() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        var set = AnalysisSettingsSet()
        let out = try await fast.submit(set.request(for: .mtf), on: snap).result()
        #expect(out.runInfo.isCurrent(for: snap, request: set.request(for: .mtf)))
        #expect(out.runInfo.settings["execution"]?["precision"] == .string("f64_reference"))
        set.execution.precision = .f32GPU
        #expect(!out.runInfo.isCurrent(for: snap, request: set.request(for: .mtf)))
        set.execution.precision = .f64Reference
        set.execution.backend = .gpu
        #expect(!out.runInfo.isCurrent(for: snap, request: set.request(for: .mtf)))
        let gpu = try await fast.submit(set.request(for: .mtf), on: snap).result()
        #expect(gpu.runInfo.precision == .f64Reference && gpu.runInfo.warnings.contains { $0.contains("f64 only") },
            "the mock reports what it actually used")
        // A deterministic analysis does not depend on the seed.
        set.execution.seed = 99
        #expect(set.request(for: .mtf).seed == nil)
    }

    @Test func invalidSettingsFailFast() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        var mtf = MTFSettings()
        mtf.maxFrequency = -50
        var psf = PSFSettings()
        psf.field = 7
        var spot = SpotDiagramSettings()
        spot.rings = 100_000
        spot.wavelengths = .index(5)
        var layout = LayoutSettings()
        layout.configuration = 3
        for (req, n) in [(AnalysisRequest.mtf(mtf), 1), (.psf(psf), 1), (.spotDiagram(spot), 2), (.layout(layout), 1)] {
            #expect(req.problems(for: snap.system).count == n, "\(req.kind): \(req.problems(for: snap.system))")
            await #expect(throws: EngineError.invalidSettings(req.problems(for: snap.system))) {
                try await MockEngine(simulatedLatency: .seconds(30)).submit(req, on: snap).result()
            }
        }
        for kind in AnalysisKind.allCases {
            #expect(AnalysisSettingsSet().request(for: kind).problems(for: snap.system).isEmpty)
        }
    }

    /// Progress reported from concurrent child tasks (how engine jobs fan out
    /// over fields and wavelengths) still arrives in non-decreasing order.
    @Test func concurrentProgressStaysMonotone() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.singlet())
        for _ in 0..<50 {
            let job = AnalysisJob.run(request: .firstOrder(FirstOrderSettings()), snapshot: snap) { r in
                await withTaskGroup(of: Void.self) { g in
                    for child in 0..<8 {
                        g.addTask {
                            for k in 0..<200 { r.report(Double(k * 8 + child) / 1600) }
                        }
                    }
                }
                return try await fast.submit(.firstOrder(FirstOrderSettings()), on: snap).result()
            }
            var last = -1.0, decreases = 0
            for await p in job.progress {
                if p.fraction < last { decreases += 1 }
                last = p.fraction
            }
            #expect(decreases == 0)
            _ = try await job.result()
        }
    }
}

/// serde_json writes non-finite `f64` as `null`; the engine writes infinite
/// lengths as `"inf"`. Results must decode both and round-trip them.
@Suite struct NonFiniteResultTests {
    @Test func decodesNullAndInfinityStrings() throws {
        let json = #"""
            {"entries":[{"key":"efl","label":"EFL","value":"inf","unit":"mm"},
                        {"key":"bfl","label":"BFL","value":null,"unit":"mm"},
                        {"key":"epd","label":"EPD","value":10,"unit":"mm"}],
             "wavelength_um":0.5875618}
            """#
        let r = try MKLightJSON.decode(FirstOrderResult.self, from: Data(json.utf8))
        #expect(r["efl"] == .infinity && r["bfl"]!.isNaN && r["epd"] == 10)
        let back = try MKLightJSON.decode(FirstOrderResult.self, from: MKLightJSON.encode(r))
        #expect(back == r)
        let tree = try JSONValue.from(r)
        #expect(tree["entries"] == .array([
            .object(["key": .string("efl"), "label": .string("EFL"), "value": .string("inf"), "unit": .string("mm")]),
            .object(["key": .string("bfl"), "label": .string("BFL"), "value": .null, "unit": .string("mm")]),
            .object(["key": .string("epd"), "label": .string("EPD"), "value": .integer(10), "unit": .string("mm")]),
        ]))
        let mtf = #"{"frequencies":[0,50],"curves":[{"field":0,"orientation":"sagittal","modulation":[1,null]}],"diffraction_limit":null,"cutoff_frequency":"inf","field_values":[[0,null]]}"#
        let m = try MKLightJSON.decode(MTFResult.self, from: Data(mtf.utf8))
        #expect(m.curves[0].modulation[1].isNaN && m.cutoffFrequency == .infinity && m.fieldValues[0].y.isNaN)
        #expect(try MKLightJSON.decode(MTFResult.self, from: MKLightJSON.encode(m)) == m)
    }

    /// An afocal (Kepler) telescope has an infinite EFL; the mock reports it
    /// as such and the output still serialises.
    @Test func afocalOutputSerialises() async throws {
        let s = OpticalSystem(
            name: "Kepler", aperture: .entrancePupilDiameter(10),
            surfaces: [
                Surface(id: SurfaceID(0), thickness: .infinity),
                Surface(id: SurfaceID(1), label: "STOP", shape: .paraxial(focalLength: 100), thickness: 120),
                Surface(id: SurfaceID(2), shape: .paraxial(focalLength: 20), thickness: 20),
                Surface(id: SurfaceID(3), thickness: 0),
            ], stop: 1)
        let snap = try await fast.openSnapshot(s)
        for kind in AnalysisKind.allCases {
            let out = try await fast.submit(AnalysisSettingsSet().request(for: kind), on: snap).result()
            let back = try MKLightJSON.decode(AnalysisOutput.self, from: MKLightJSON.encode(out))
            #expect(back == out, "\(kind)")
            if case .firstOrder(let fo) = out.result { #expect(fo["efl"] == .infinity) }
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

    @Test func paraboloidFocusesOnAxisPerfectly() throws {
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
        let r = try o.spot(SpotDiagramSettings())
        #expect(r.fields[0].points.count == 127 * 3)
        #expect(r.fields[0].rmsRadius < 1e-12)
        #expect(r.fields[0].geoRadius < 1e-12)
        #expect(abs(o.firstOrder(FirstOrderSettings())["efl"]! - 100) < 1e-12)
    }

    @Test func sphericalMirrorHasThirdOrderAberration() throws {
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
        let curve = try o.rayFan(fan).fields[0].tangential[0]
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
        #expect(MockOptics.diffractionMTF(-50, cutoff: 100) == MockOptics.diffractionMTF(50, cutoff: 100))
        #expect(MockOptics.diffractionMTF(-50, cutoff: 100) <= 1)
    }

    @Test func layoutSceneOfTriplet() throws {
        let o = MockOptics(system: SampleSystems.cookeTriplet())
        var settings = LayoutSettings()
        settings.include3D = false
        let scene = try o.layout(settings)
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

    @Test func distortionIsZeroOnAxisAndFieldCurvatureFinite() throws {
        let o = MockOptics(system: SampleSystems.cookeTriplet())
        let r = try o.fieldCurvature(FieldCurvatureSettings())
        #expect(r.fieldValues.first == 0 && r.fieldValues.last == 20)
        for c in r.curves {
            #expect(c.distortionPercent[0] == 0)
            #expect(c.tangential.allSatisfy { $0.map { abs($0) < 5 } ?? false })
        }
    }

    /// A finite object on the entrance pupil (object distance 0, stop at
    /// surface 1) with an object-space NA: the pupil slopes degenerate. The
    /// mock must warn and stay finite rather than emit NaN.
    @Test func objectOnEntrancePupilStaysFinite() async throws {
        var s = SampleSystems.singlet()
        s.surfaces[0].thickness = 0
        s.aperture = .objectSpaceNA(0.1)
        s.fields.kind = .objectHeight
        s.fields.fields = [Field(), Field(x: 0, y: 1)]
        let snap = try await fast.openSnapshot(s)
        let o = MockOptics(system: s)
        #expect(o.warnings.contains { $0.contains("entrance pupil") })
        for kind in AnalysisKind.allCases {
            let out = try await fast.submit(AnalysisSettingsSet().request(for: kind), on: snap).result()
            _ = try MKLightJSON.encode(out)
            if case .opdFan(let fan) = out.result {
                #expect(fan.fields.allSatisfy { $0.tangential.allSatisfy { $0.values.allSatisfy { $0?.isFinite ?? true } } })
            }
        }
        // Paraxially, the marginal ray then leaves the axial object point with
        // slope NA/n, whatever the pupil radius.
        let m = o.marginalStart()
        #expect(abs(m.u0 - 0.1) < 1e-12 && m.y1 == 0)
    }

    @Test func computeHonoursCancellation() async throws {
        let o = MockOptics(system: SampleSystems.cookeTriplet())
        var s = PSFSettings()
        s.imageSampling = 512
        let t = Task { () throws -> PSFResult in
            withUnsafeCurrentTask { $0?.cancel() }
            return try o.psf(s)
        }
        await #expect(throws: CancellationError.self) { try await t.value }
    }

    @Test func psfPeakIsStrehlBounded() throws {
        var s = PSFSettings()
        s.imageSampling = 32
        let r = try MockOptics(system: SampleSystems.cookeTriplet()).psf(s)
        #expect(r.values.count == 32 * 32)
        #expect(r.strehl > 0 && r.strehl <= 1)
        #expect(r.values.allSatisfy { $0 >= 0 && $0.isFinite })
    }
}

@Suite struct PrescriptionImportDecodingTests {
    @Test func decodesTheEngineResult() throws {
        let json = """
            {"format": "seq", "encoding": "utf8",
             "system": \(String(decoding: try MKLightJSON.encode(OpticalSystem.newEmpty()), as: UTF8.self)),
             "warnings": [
               {"kind": "unknown_glass", "line": 4, "surface": 1, "message": "glass X …"},
               {"kind": "something_new", "line": null, "surface": null, "message": "?"}
             ]}
            """
        let r = try MKLightJSON.decode(PrescriptionImport.self, from: Data(json.utf8))
        #expect(r.format == .seq && r.encoding == "utf8")
        #expect(r.warnings.map(\.kind) == [.unknownGlass, .other])
        #expect(r.warnings[0].line == 4 && r.warnings[0].surface == 1 && r.warnings[1].line == nil)
        #expect(PrescriptionFormat(fileExtension: "ZMX") == .zmx && PrescriptionFormat(fileExtension: "txt") == nil)
    }

    @Test func mockEngineDoesNotImport() async {
        #expect(throws: EngineError.self) { try fast.importPrescription(Data(), format: .zmx) }
        await #expect(throws: EngineError.self) {
            try await fast.importPrescription(at: URL(fileURLWithPath: "/tmp/lens.txt"))
        }
    }
}
