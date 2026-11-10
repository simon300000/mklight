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
        #expect(info.tier == .design && info.aimStop == .requested && info.failedAttempts.isEmpty)
        let back = try JSONValue.parse(MKLightJSON.encode(info))
        #expect(try JSONValue.parse(data).differences(from: back).isEmpty)
    }

    /// The tier, the aim stop and the failed backend attempts (a key the
    /// engine writes only when there are some) round-trip; a result
    /// without them reads as the engine's defaults, and a tier this app
    /// does not know reads as not design grade, with a warning.
    @Test func decodesTierAimStopAndFailedAttempts() throws {
        let data = try Data(contentsOf: fixtureDir.appendingPathComponent("run_info.json"))
        guard case .object(var o) = try JSONValue.parse(data) else {
            Issue.record("run_info.json is not an object")
            return
        }
        func decode() throws -> RunInfo { try MKLightJSON.decode(RunInfo.self, from: MKLightJSON.encode(JSONValue.object(o))) }
        o["tier"] = .string("preview")
        o["aim_stop"] = .string("conditioned")
        o["failed_attempts"] = .array([
            .object(["backend": .string("gpu:metal"), "outcome": .string("failed"), "reason": .string("device lost"), "elapsed_ms": .number(3.5)])
        ])
        let info = try decode()
        #expect(info.tier == .preview && info.aimStop == .conditioned)
        #expect(info.failedAttempts == [BackendAttempt(backend: "gpu:metal", outcome: "failed", reason: "device lost", elapsedMs: 3.5)])
        #expect(try JSONValue.object(o).differences(from: JSONValue.parse(MKLightJSON.encode(info))).isEmpty)
        o["tier"] = .string("draft")
        let unknown = try decode()
        #expect(unknown.tier == .preview && unknown.warnings.contains("unknown tier 'draft'"))
        o["tier"] = nil
        o["aim_stop"] = nil
        o["failed_attempts"] = nil
        let bare = try decode()
        #expect(bare.tier == .design && bare.aimStop == .requested && bare.failedAttempts.isEmpty)
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
        do {
            _ = try await fast.openSnapshot(s)
            Issue.record("an invalid model opened")
        } catch EngineError.invalidModel(let issues) {
            #expect(!issues.isEmpty)
        } catch {
            Issue.record("\(error)")
        }
    }
}

/// The analyses the mock runs from a system alone (no required settings).
private let plainAnalyses = ["layout", "spot_diagram", "ray_fan", "opd_fan", "field_curvature_distortion", "first_order"]

private func request(_ analysis: String, _ settings: [String: JSONValue] = [:]) -> EngineAnalysisRequest {
    EngineAnalysisRequest(analysis: analysis, settings: settings)
}

@Suite struct JobTests {
    @Test func progressIsMonotoneAndResultCarriesRunInfo() async throws {
        let engine = MockEngine(simulatedLatency: .milliseconds(40))
        let snap = try await engine.openSnapshot(SampleSystems.cookeTriplet())
        let spot = request("spot_diagram", ["rings": .integer(3)])
        let job = engine.submit(spot, on: snap)
        #expect(job.analysis == "spot_diagram" && job.engineRequest == spot && job.snapshotHash == snap.hash)
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
        #expect(out.isCurrent(for: snap, request: spot))
        #expect(!out.isCurrent(for: snap, request: request("spot_diagram", ["rings": .integer(4)])), "different settings → stale")
        #expect(!out.isCurrent(for: nil, request: spot), "no snapshot → stale")
    }

    @Test func cancellationStopsJob() async throws {
        let engine = MockEngine(simulatedLatency: .seconds(30))
        let snap = try await engine.openSnapshot(SampleSystems.cookeTriplet())
        let job = engine.submit(request("spot_diagram"), on: snap)
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
        let job = engine.submit(request("first_order"), on: snap)
        let waiter = Task { try await job.result() }
        waiter.cancel()
        _ = try? await waiter.value
        let out = try await job.result()
        #expect(out.runInfo.analysis == "first_order" && out.json["entries"]?.array?.isEmpty == false)
    }

    @Test func randomSpotIsReproducibleFromSeed() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        func run(seed: Int64) async throws -> AnalysisOutput {
            try await fast.submit(request("spot_diagram", ["pattern": .string("random"), "seed": .integer(seed)]), on: snap).result()
        }
        let a = try #require(try await run(seed: 1234).buffer(named: "points"))
        let b = try #require(try await run(seed: 1234).buffer(named: "points"))
        #expect(a.count > 0 && a == b)
        let other = try await run(seed: 4321)
        #expect(other.buffer(named: "points") != a)
        #expect(other.runInfo.seed == 4321)
    }

    /// Every analysis the mock runs from a system alone finishes under its
    /// own name with a result object, on a design with several fields and
    /// wavelengths.
    @Test func plainAnalysesRunByName() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        for name in plainAnalyses {
            let out = try await fast.submit(request(name), on: snap).result()
            #expect(out.runInfo.analysis == name && out.json.object?.isEmpty == false, "\(name)")
            #expect(out.engineRequest == request(name) && out.backend?.used == "mock", "\(name)")
        }
    }

    /// Seeds are full 64-bit values. Two seeds that round to the same
    /// `Double` must still give a stale result, with or without the request
    /// that produced it, and the recorded run carries the seed exactly (an
    /// engine could not reproduce a rounded one).
    @Test func seedsAbove2To53AreExact() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        let seed: Int64 = 1 << 60
        func spot(_ seed: Int64) -> EngineAnalysisRequest {
            request("spot_diagram", ["pattern": .string("random"), "rings": .integer(2), "seed": .integer(seed)])
        }
        #expect(Double(seed + 1) == Double(seed), "the two seeds collide as Double")
        let out = try await fast.submit(spot(seed), on: snap).result()
        #expect(out.runInfo.seed == 1 << 60)
        #expect(out.isCurrent(for: snap, request: spot(seed)) && !out.isCurrent(for: snap, request: spot(seed + 1)))
        var recorded = out
        recorded.engineRequest = nil
        #expect(recorded.isCurrent(for: snap, request: spot(seed)) && !recorded.isCurrent(for: snap, request: spot(seed + 1)))
        // Above `Int64.max` too: the run info keeps every digit.
        var info = out.runInfo
        info.seed = 12_345_678_901_234_567_891
        info.settings = .object(["seed": .unsigned(12_345_678_901_234_567_891)])
        let data = try MKLightJSON.encode(info)
        #expect(String(decoding: data, as: UTF8.self).contains("\"seed\":12345678901234567891"))
        let decoded = try MKLightJSON.decode(RunInfo.self, from: data)
        #expect(decoded.settings["seed"] == .unsigned(12_345_678_901_234_567_891))
        #expect(decoded.seed == 12_345_678_901_234_567_891)
    }

    /// The backend preference is part of a job's identity: changing it
    /// makes an existing result stale. The mock reports what it actually
    /// used.
    @Test func executionSettingsTakePartInStaleness() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.cookeTriplet())
        var r = request("first_order")
        let out = try await fast.submit(r, on: snap).result()
        #expect(out.isCurrent(for: snap, request: r))
        r.execution.backend = .gpu
        #expect(!out.isCurrent(for: snap, request: r))
        let gpu = try await fast.submit(r, on: snap).result()
        #expect(gpu.isCurrent(for: snap, request: r))
        #expect(gpu.runInfo.precision == .f64Reference && gpu.runInfo.backend == "mock" && gpu.backend?.used == "mock")
    }

    /// Progress reported from concurrent child tasks (how engine jobs fan out
    /// over fields and wavelengths) still arrives in non-decreasing order.
    @Test func concurrentProgressStaysMonotone() async throws {
        let snap = try await fast.openSnapshot(SampleSystems.singlet())
        for _ in 0..<50 {
            let job = AnalysisJob.run(engineRequest: request("first_order"), snapshot: snap) { r in
                await withTaskGroup(of: Void.self) { g in
                    for child in 0..<8 {
                        g.addTask {
                            for k in 0..<200 { r.report(Double(k * 8 + child) / 1600) }
                        }
                    }
                }
                return try await fast.submit(request("first_order"), on: snap).result()
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
/// lengths as `"inf"`. Readers must take both.
@Suite struct NonFiniteResultTests {
    /// `EngineDouble` fields (here a glass's numbers) decode a number,
    /// `null` (NaN) and the infinity strings, and round-trip them.
    @Test func decodesNullAndInfinityStrings() throws {
        let json = #"{"catalog":"X","name":"G","nd":1.5,"vd":null,"min_um":"-inf","max_um":"inf"}"#
        let g = try MKLightJSON.decode(GlassInfo.self, from: Data(json.utf8))
        #expect(g.nd == 1.5 && g.vd.isNaN && g.minUm == -.infinity && g.maxUm == .infinity)
        let back = try MKLightJSON.decode(GlassInfo.self, from: MKLightJSON.encode(g))
        #expect(back == g, "NaN compares equal to NaN")
        #expect(try JSONValue.from(g) == JSONValue.parse(Data(json.utf8)))
        #expect(throws: DecodingError.self) {
            try MKLightJSON.decode(GlassInfo.self, from: Data(json.replacingOccurrences(of: "\"inf\"", with: "\"wide\"").utf8))
        }
        // The tolerant accessors read the same forms from a result's JSON.
        let tree = try JSONValue.parse(Data(json.utf8))
        #expect(tree["vd"]?.nf?.isNaN == true && tree["min_um"]?.nf == -.infinity && tree["name"]?.nf == nil)
        // Points compare the same way, so a scene with an undefined point
        // still equals itself.
        #expect(Point3(0, .nan, 1) == Point3(0, .nan, 1) && Point3(0, .nan, 1) != Point3(0, 2, 1))
    }

    /// An afocal (Kepler) telescope has an infinite EFL; the mock reports it
    /// in the engine's form (`"inf"`) and every analysis still runs.
    @Test func afocalSystemRunsByName() async throws {
        let s = OpticalSystem(
            name: "Kepler", aperture: .entrancePupilDiameter(10),
            surfaces: [
                Surface(id: SurfaceID(0), thickness: .infinity),
                Surface(id: SurfaceID(1), label: "STOP", shape: .paraxial(focalLength: 100), thickness: 120),
                Surface(id: SurfaceID(2), shape: .paraxial(focalLength: 20), thickness: 20),
                Surface(id: SurfaceID(3), thickness: 0),
            ], stop: 1)
        let snap = try await fast.openSnapshot(s)
        for name in plainAnalyses {
            let out = try await fast.submit(request(name), on: snap).result()
            #expect(out.runInfo.analysis == name, "\(name)")
            // The result is valid JSON text: no bare non-finite number.
            #expect(throws: Never.self, "\(name)") { try JSONValue.parse(MKLightJSON.encode(out.json)) }
        }
        let fo = try await fast.submit(request("first_order"), on: snap).result()
        let efl = try #require(fo.json["entries"]?.array?.first { $0["key"]?.string == "efl" })
        #expect(efl["value"] == .string("inf") && efl["value"]?.nf == .infinity)
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
        let fo = o.firstOrder()
        #expect(abs(fo["efl"]! - f) < 1e-9 * f)
        let bfl = f * (1 - (n - 1) * t / (n * r1))
        #expect(abs(fo["bfl"]! - bfl) < 1e-9 * f)
        #expect(abs(fo["entrance_pupil_diameter"]! - 20) < 1e-12)
        #expect(abs(fo["image_f_number"]! - f / 20) < 1e-9)
    }

    @Test func tripletFirstOrderMatchesIndependentTrace() {
        // Reference values from an independent y-nu trace at n_d.
        let fo = MockOptics(system: SampleSystems.cookeTriplet()).firstOrder()
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
        let r = try o.spot(MockOptics.SpotDiagramSettings())
        #expect(r.fields[0].points.count == 127 * 3)
        #expect(r.fields[0].rmsRadius < 1e-12)
        #expect(r.fields[0].geoRadius < 1e-12)
        #expect(abs(o.firstOrder()["efl"]! - 100) < 1e-12)
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
        var fan = MockOptics.FanSettings()
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

    @Test func layoutSceneOfTriplet() throws {
        let o = MockOptics(system: SampleSystems.cookeTriplet())
        var settings = MockOptics.LayoutSettings()
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
        let r = try o.fieldCurvature(MockOptics.FieldCurvatureSettings())
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
        for name in plainAnalyses {
            let out = try await fast.submit(request(name), on: snap).result()
            #expect(throws: Never.self, "\(name)") { try JSONValue.parse(MKLightJSON.encode(out.json)) }
        }
        // Vignetted samples are `nil`; every other one is a finite number.
        let fan = try o.opdFan(MockOptics.FanSettings())
        #expect(fan.fields.allSatisfy { $0.tangential.allSatisfy { $0.values.allSatisfy { $0?.isFinite ?? true } } })
        // Paraxially, the marginal ray then leaves the axial object point with
        // slope NA/n, whatever the pupil radius.
        let m = o.marginalStart()
        #expect(abs(m.u0 - 0.1) < 1e-12 && m.y1 == 0)
    }

    @Test func computeHonoursCancellation() async throws {
        let o = MockOptics(system: SampleSystems.cookeTriplet())
        let t = Task { () throws -> Int in
            withUnsafeCurrentTask { $0?.cancel() }
            return try o.spot(MockOptics.SpotDiagramSettings()).fields.count
        }
        await #expect(throws: CancellationError.self) { try await t.value }
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
