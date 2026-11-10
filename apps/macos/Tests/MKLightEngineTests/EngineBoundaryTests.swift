import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel

/// The engine boundary pieces an FFI client relies on, tested without the
/// engine library: schema handling, the settings a schema accepts, buffers,
/// error mapping and staleness.
@Suite struct EngineBoundaryTests {
    /// A descriptor in the exact JSON form of `mk_engine_list_analyses`,
    /// plus a key a newer engine might add.
    static let descriptorJSON = """
        {"name":"spot_diagram","title":"Spot diagram","category":"geometric","description":"d",
         "available":true,"unavailable_reason":null,"future_field":{"x":1},
         "settings_schema":{"type":"object","required":["rings"],"properties":{
            "rings":{"type":"integer","minimum":1,"default":6,"title":"Rings"},
            "seed":{"type":"integer","default":1},
            "fields":{"type":["integer","null"]},
            "wavelengths":{"type":"integer"},
            "pattern":{"type":"string","enum":["hexapolar","square"]},
            "max_frequency":{"type":"number","unit":"cycles/mm"}}},
         "default_settings":null}
        """

    func descriptor() throws -> AnalysisDescriptor {
        try MKLightJSON.decode(AnalysisDescriptor.self, from: Data(Self.descriptorJSON.utf8))
    }

    @Test func descriptorAndSchemaDecode() throws {
        let d = try descriptor()
        #expect(d.available && d.unavailableReason == nil && d.defaultSettings == .null)
        let fields = d.settingsFields
        #expect(fields.first?.name == "rings" && fields.first?.required == true, "required first")
        #expect(fields.first { $0.name == "max_frequency" }?.unit == "cycles/mm")
        #expect(fields.first { $0.name == "fields" }?.types == ["integer", "null"])
        let rings = try #require(fields.first { $0.name == "rings" })
        #expect(rings.accepts(.integer(3)) && rings.accepts(.number(3)) && !rings.accepts(.number(3.5)))
        #expect(!rings.accepts(.string("3")))
        let pattern = try #require(fields.first { $0.name == "pattern" })
        #expect(pattern.accepts(.string("square")) && !pattern.accepts(.string("random")))
        // Minimal descriptors (older engines) decode with defaults.
        let minimal = try MKLightJSON.decode(AnalysisDescriptor.self, from: Data(#"{"name":"x"}"#.utf8))
        #expect(minimal.title == "x" && minimal.available && minimal.settingsFields.isEmpty)
        // Stop support is what the engine lists; absent means none.
        #expect(!d.stoppable && !minimal.stoppable)
        let solver = try MKLightJSON.decode(AnalysisDescriptor.self, from: Data(#"{"name":"optimize","stoppable":true}"#.utf8))
        #expect(solver.stoppable)
        let again = try MKLightJSON.decode(AnalysisDescriptor.self, from: try MKLightJSON.encode(solver))
        #expect(again.stoppable && again == solver)
    }

    /// A mock job can be stopped exactly when its descriptor says so.
    @Test func mockJobsStopAsTheirDescriptorsSay() async throws {
        let mock = MockEngine(simulatedLatency: .zero)
        let snap = try await mock.openSnapshot(SampleSystems.singlet())
        let descriptors = await mock.analysisDescriptors()
        #expect(Set(descriptors.filter(\.stoppable).map(\.name)) == ["optimize", "tolerance"])
        for d in descriptors where d.available {
            let job = mock.submit(EngineAnalysisRequest(analysis: d.name), on: snap)
            #expect(job.canStop == d.stoppable, "\(d.name)")
            job.cancel()
            _ = try? await job.result()
        }
    }

    /// Settings are reduced to what the schema declares: unknown keys,
    /// values of another type and values outside an enum are dropped, so
    /// the engine's defaults apply.
    @Test func acceptedSettingsFollowTheSchema() throws {
        let out = try descriptor().acceptedSettings([
            "rings": .integer(9), "seed": .integer(42), "fields": .integer(2), "wavelengths": .string("all"),
            "pattern": .string("random"), "reference": .string("chief"), "max_frequency": .number(62.5),
        ])
        #expect(out["rings"] == .integer(9) && out["seed"] == .integer(42) && out["max_frequency"] == .number(62.5))
        #expect(out["fields"] == .integer(2))
        #expect(out["wavelengths"] == nil, "a string where the schema wants an integer")
        #expect(out["pattern"] == nil, "a value outside the enum is dropped")
        #expect(out["reference"] == nil, "unknown to this schema")
        #expect(try descriptor().acceptedSettings(["fields": .null, "seed": .unsigned(.max)]) == ["fields": .null, "seed": .unsigned(.max)])
    }

    @Test func buffersIndexRowMajorAndCompareByContent() throws {
        let b = ResultBuffer(name: "n_after", doubles: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        #expect(b[[1, 0]] == 4 && b[[0, 2]] == 3 && b[5] == 6)
        #expect(b.withUnsafeBufferPointer(of: Double.self) { Array($0) } == [1, 2, 3, 4, 5, 6])
        #expect(b.withUnsafeBufferPointer(of: Int64.self) { _ in 0 } == nil, "dtype mismatch")
        #expect(b == ResultBuffer(name: "n_after", doubles: [1, 2, 3, 4, 5, 6], shape: [2, 3]))
        #expect(b != ResultBuffer(name: "n_after", doubles: [1, 2, 3, 4, 5, 7], shape: [2, 3]))
        #expect(b.description == "n_after: f64[2×3]")
        // Other dtypes read through the raw memory without copying.
        var words: [UInt32] = [7, 8, 9]
        let owner = NSData(bytes: &words, length: 12)
        let u = ResultBuffer(name: "w", dtype: .u32, shape: [3], count: 3, base: owner.bytes, owner: owner)
        #expect(u.doubles == [7, 8, 9])
        #expect(u[2] == 9 && u.row(1) == [8])
        // Every dtype converts, element by element and as a whole.
        func buffer<T>(_ values: [T], _ dtype: BufferDType) -> ResultBuffer {
            let data = values.withUnsafeBytes { NSData(bytes: $0.baseAddress, length: $0.count) }
            return ResultBuffer(name: "b", dtype: dtype, shape: [values.count], count: values.count, base: data.bytes, owner: data)
        }
        let cases: [(ResultBuffer, [Double])] = [
            (buffer([1.5, -2.25] as [Double], .f64), [1.5, -2.25]), (buffer([1.5, -2.25] as [Float], .f32), [1.5, -2.25]),
            (buffer([-7, 8] as [Int32], .i32), [-7, 8]), (buffer([7, .max] as [UInt32], .u32), [7, 4_294_967_295]),
            (buffer([-7, 1 << 40] as [Int64], .i64), [-7, 1_099_511_627_776]), (buffer([7, 1 << 40] as [UInt64], .u64), [7, 1_099_511_627_776]),
            (buffer([0, 255] as [UInt8], .u8), [0, 255]),
        ]
        for (b, expected) in cases {
            #expect(b.doubles == expected, "\(b.dtype)")
            #expect(expected.indices.map { b[$0] } == expected, "\(b.dtype)")
        }
        #expect(ResultBuffer(name: "empty", doubles: []).doubles.isEmpty)
        #expect(BufferDType(name: "f32") == .f32 && BufferDType.u8.byteWidth == 1)
    }

    /// `int` reads whole numbers an `Int` holds, and nothing else.
    @Test func intReadsOnlyWholeNumbersInRange() {
        #expect(JSONValue.number(-3).int == -3 && JSONValue.number(0).int == 0 && JSONValue.integer(7).int == 7)
        #expect(JSONValue.number(2.5).int == nil)
        for bad in [1e30, -1e30, 9.3e18, Double.infinity, -.infinity, .nan] {
            #expect(JSONValue.number(bad).int == nil, "\(bad)")
        }
        #expect(JSONValue.unsigned(.max).int == nil && JSONValue.string("3").int == nil)
        #expect(JSONValue.array([.integer(1), .number(1e30)]).intArray == nil)
    }

    /// The mock takes seeds and counts that no run could have (negative
    /// ones) without failing.
    @Test func mockTakesNegativeSeedsAndCounts() async throws {
        let mock = MockEngine(simulatedLatency: .zero)
        let snap = try await mock.openSnapshot(SampleSystems.singlet())
        func run(_ analysis: String, _ settings: [String: JSONValue]) async throws -> AnalysisOutput {
            try await mock.submit(EngineAnalysisRequest(analysis: analysis, settings: settings), on: snap).result()
        }
        let spot = try await run("spot_diagram", ["pattern": .string("random"), "seed": .integer(-1)])
        #expect(spot.runInfo.seed == nil && spot.buffer(named: "points") != nil)
        let optimize = try await run("optimize", ["max_iterations": .integer(-3), "seed": .integer(-1)])
        #expect(optimize.json["iterations"]?.int == 0)
        let tolerance = try await run("tolerance", ["trials": .integer(-5), "seed": .integer(-9)])
        #expect(tolerance.json[path: "monte_carlo", "trials"]?.int == 0)
        let preview = try await run("tolerance", ["preview_case": .object(["type": .string("trial"), "trial": .integer(-2)])])
        #expect(preview.json["preview"]?.object != nil)
    }

    @Test func errorCodesMapToCases() {
        #expect(EngineError.from(code: .cancelled, message: "x") == .cancelled)
        #expect(EngineError.from(code: .invalidModel, message: "bad stop") == .invalidModel(["bad stop"]))
        #expect(EngineError.from(code: .unsupported, message: "no gpu") == .unsupported("no gpu"))
        #expect(EngineError.from(code: .jobFailed, message: "boom") == .failed("boom"))
        #expect(EngineError.from(code: .io, message: "disk") == .engine(.io, "disk"))
        // What a failed job reports: the code of its own error.
        #expect(EngineError.from(code: .notFound, message: "unknown material 'X'") == .engine(.notFound, "unknown material 'X'"))
        #expect(EngineError.from(code: .compute, message: "diverged").errorDescription == "diverged")
        for code in EngineErrorCode.allCases {
            #expect(EngineError.from(code: code, message: "m").code == code, "\(code) round trips")
        }
        #expect(EngineErrorCode(rawValue: 12) == .panic && EngineErrorCode(rawValue: 0) == nil)
    }

    @Test func runInfoToleratesUnknownPrecision() throws {
        let json = """
            {"engine_version":"9","snapshot_hash":"ab","analysis":"x","settings":{},"backend":"vulkan",
             "precision":"f16_tensor","seed":null,"started_unix_ms":1,"duration_ms":2.5}
            """
        let info = try MKLightJSON.decode(RunInfo.self, from: Data(json.utf8))
        #expect(info.precision == .f32GPU && info.warnings == ["unknown precision 'f16_tensor'"])
    }

    /// The envelope's `backend` object keeps the engine's `cost_decision`
    /// as sent; a missing or null one is none.
    @Test func backendReportCarriesTheCostDecision() throws {
        let json = """
            {"requested":"default:auto","used":"gpu:metal","precision":"f32_gpu","device":"Apple M2 Pro","fallback_reason":null,
             "execution":[],"cost_decision":{"decision":"qualified_metal","rule":"cost model","predicted_ms":{"cpu":556.6,"metal":312.3}}}
            """
        let report = try #require(BackendReport(json: try MKLightJSON.decode(JSONValue.self, from: Data(json.utf8))))
        #expect(report.requested == "default:auto" && report.used == "gpu:metal" && report.fallbackReason == nil)
        #expect(report.costDecision?["decision"]?.string == "qualified_metal")
        #expect(report.costDecision?["predicted_ms"]?["metal"]?.nf == 312.3)
        let explicit = try #require(BackendReport(json: .object(["requested": .string("cpu_reference"), "cost_decision": .null])))
        #expect(explicit.requested == "cpu_reference" && explicit.costDecision == nil)
        #expect(try #require(BackendReport(json: .object(["requested": .string("auto")]))).costDecision == nil)
        #expect(BackendReport(requested: "auto", used: "mock", precision: "f64_reference").costDecision == nil)
    }

    /// Staleness compares the request that produced a result and both
    /// hashes (system and glass library).
    @Test func outputStalenessUsesRequestAndLibraryHash() {
        let snap = ModelSnapshot(system: SampleSystems.singlet(), hash: SnapshotHash(hex: "aa"), libraryHash: "L1")
        let request = EngineAnalysisRequest(analysis: "mtf", settings: ["max_frequency": .integer(100)])
        let info = RunInfo(
            engineVersion: "1", snapshotHash: snap.hash, analysis: "mtf", settings: .object(["normalised": .bool(true)]),
            backend: "cpu_reference", precision: .f64Reference, seed: nil, startedUnixMs: 0, durationMs: 1)
        let out = AnalysisOutput(runInfo: info, json: .null, libraryHash: "L1", engineRequest: request)
        #expect(out.isCurrent(for: snap, request: request), "engine-normalised settings do not matter")
        #expect(!out.isCurrent(for: snap, request: EngineAnalysisRequest(analysis: "mtf", settings: ["max_frequency": .integer(50)])))
        #expect(!out.isCurrent(for: snap, request: EngineAnalysisRequest(analysis: "psf", settings: request.settings)))
        let newLibrary = ModelSnapshot(system: snap.system, hash: snap.hash, libraryHash: "L2")
        #expect(newLibrary != snap)
        #expect(!out.isCurrent(for: newLibrary, request: request), "a catalog change makes it stale")
        let edited = ModelSnapshot(system: snap.system, hash: SnapshotHash(hex: "bb"), libraryHash: "L1")
        #expect(!out.isCurrent(for: edited, request: request), "an edit makes it stale")
        // Without its request: the engine's recorded settings must contain
        // the requested ones.
        let recorded = EngineAnalysisRequest(analysis: "mtf", settings: ["normalised": .bool(true)])
        let bare = AnalysisOutput(runInfo: info, json: .null, libraryHash: "L1")
        #expect(bare.isCurrent(for: snap, request: recorded))
        #expect(!bare.isCurrent(for: snap, request: EngineAnalysisRequest(analysis: "mtf", settings: ["normalised": .bool(false)])))
        #expect(!bare.isCurrent(for: newLibrary, request: recorded))
    }

    /// The mock runs a few analyses by name in the engine's result format
    /// and fails others cleanly.
    @Test func mockRejectsRequestsByName() async throws {
        let mock = MockEngine(simulatedLatency: .zero)
        let snap = try await mock.openSnapshot(SampleSystems.singlet())
        let job = mock.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap)
        await #expect(throws: EngineError.self) { try await job.result() }
        #expect(job.analysis == "system_summary" && job.engineRequest.settings.isEmpty)
        let names = await mock.analysisDescriptors().filter(\.available).map(\.name)
        #expect(["layout", "spot_diagram", "first_order", "optimize"].allSatisfy(names.contains))
        #expect(await mock.analysisDescriptors().contains { $0.name == "psf" && !$0.available })
        let spot = try await mock.submit(EngineAnalysisRequest(analysis: "spot_diagram"), on: snap).result()
        #expect(spot.json["fields"]?.array?.isEmpty == false && spot.buffer(named: "points") != nil)
        await #expect(throws: EngineError.self) { try await mock.loadCatalog(at: URL(fileURLWithPath: "/x.agf")) }
    }
}
