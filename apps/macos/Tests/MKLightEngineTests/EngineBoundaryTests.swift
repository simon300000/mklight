import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel

/// The engine boundary pieces an FFI client relies on, tested without the
/// engine library: schema handling, request translation, tolerant result
/// decoding, buffers, error mapping and staleness.
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
    }

    /// Swift settings become engine settings: execution → seed, selections →
    /// indices (or the engine default), and only keys the schema accepts.
    @Test func typedRequestTranslation() throws {
        var s = SpotDiagramSettings()
        s.rings = 9
        s.pattern = .random
        s.fields = .index(2)
        s.wavelengths = .all
        s.execution.seed = 42
        let out = AnalysisRequest.spotDiagram(s).engineSettings(for: try descriptor())
        #expect(out["rings"] == .integer(9))
        #expect(out["seed"] == .integer(42))
        #expect(out["fields"] == .integer(2))
        #expect(out["wavelengths"] == nil, "'all' leaves the engine default")
        #expect(out["pattern"] == nil, "a value outside the enum is dropped")
        #expect(out["reference"] == nil && out["configuration"] == nil, "unknown to this schema")
        var big = s
        big.execution.seed = UInt64.max
        #expect(AnalysisRequest.spotDiagram(big).engineSettings(for: try descriptor())["seed"] == .unsigned(.max))
        #expect(AnalysisKind.firstOrder.engineName == "first_order" && AnalysisKind.mtf.engineName == "mtf")
    }

    /// First-order data runs as `first_order`, or as the `paraxial`
    /// placeholder on engines that predate it; an available entry wins.
    @Test func engineNameFollowsTheDescriptors() {
        func d(_ name: String, _ available: Bool) -> AnalysisDescriptor {
            AnalysisDescriptor(name: name, title: name, category: "geometric", available: available)
        }
        let fo = AnalysisKind.firstOrder
        #expect(fo.engineName(in: [d("first_order", true), d("paraxial", false)]) == "first_order")
        #expect(fo.engineName(in: [d("paraxial", false)]) == "paraxial", "older engine: show its reason")
        #expect(fo.engineName(in: [d("paraxial", true)]) == "paraxial")
        #expect(fo.engineName(in: [d("first_order", false), d("paraxial", true)]) == "paraxial")
        #expect(fo.engineName(in: [d("first_order", false), d("paraxial", false)]) == "first_order")
        #expect(fo.engineName(in: []) == "first_order")
        #expect(AnalysisKind.mtf.engineName(in: [d("paraxial", true)]) == "mtf")
    }

    /// A result whose JSON does not match the typed mirror never fails: it
    /// is shown generically; first-order data is read from any numbers.
    @Test func tolerantResultDecoding() throws {
        let odd = JSONValue.object(["rings": .array([.integer(1)]), "new_thing": .string("x")])
        let buffers = [ResultBuffer(name: "pts", doubles: [1, 2, 3, 4], shape: [2, 2])]
        guard case .generic(let g) = AnalysisResult.fromEngine(kind: .spotDiagram, json: odd, buffers: buffers) else {
            Issue.record("expected generic")
            return
        }
        #expect(g.analysis == "spot_diagram" && g.note != nil && g.buffer(named: "pts")?.count == 4)
        let paraxial = JSONValue.object([
            "wavelength_um": .number(0.5875618),
            "first_order": .object(["efl": .number(100.0009), "bfl": .number(95.085), "afocal": .bool(false)]),
            "image_f_number": .number(5), "magnification": .string("-inf"), "label": .string("ignored"),
        ])
        guard case .firstOrder(let fo) = AnalysisResult.fromEngine(kind: .firstOrder, json: paraxial, buffers: []) else {
            Issue.record("expected first-order data")
            return
        }
        #expect(fo["efl"] == 100.0009 && fo["bfl"] == 95.085 && fo["image_f_number"] == 5)
        #expect(fo["magnification"] == -.infinity && fo["label"] == nil && fo["afocal"] == nil)
        #expect(fo.wavelengthUm == 0.5875618 && fo["wavelength_um"] == nil)
        #expect(fo.entries.first { $0.key == "efl" }?.label == "EFL")
        // A typed first-order result decodes as such.
        let typed = try JSONValue.from(FirstOrderResult(entries: [.init(key: "efl", label: "EFL", value: 50, unit: "mm")], wavelengthUm: 0.55))
        guard case .firstOrder(let t) = AnalysisResult.fromEngine(kind: .firstOrder, json: typed, buffers: []) else {
            Issue.record("typed first order")
            return
        }
        #expect(t["efl"] == 50 && t.entries[0].unit == "mm")
        // Unknown analyses in the JSON envelope decode as generic.
        let env = #"{"analysis":"ghost_paths","data":{"paths":3}}"#
        let r = try MKLightJSON.decode(AnalysisResult.self, from: Data(env.utf8))
        #expect(r.kind == nil && r.analysisName == "ghost_paths")
        #expect(try JSONValue.parse(MKLightJSON.encode(r)) == JSONValue.parse(Data(env.utf8)))
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
        #expect(BufferDType(name: "f32") == .f32 && BufferDType.u8.byteWidth == 1)
    }

    @Test func errorCodesMapToCases() {
        #expect(EngineError.from(code: .cancelled, message: "x") == .cancelled)
        #expect(EngineError.from(code: .invalidModel, message: "bad stop") == .invalidModel(["bad stop"]))
        #expect(EngineError.from(code: .unsupported, message: "no gpu") == .unsupported("no gpu"))
        #expect(EngineError.from(code: .jobFailed, message: "boom") == .failed("boom"))
        #expect(EngineError.from(code: .io, message: "disk") == .engine(.io, "disk"))
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

    /// Staleness compares the front-end request that produced a result and
    /// both hashes (system and glass library).
    @Test func outputStalenessUsesRequestAndLibraryHash() {
        let snap = ModelSnapshot(system: SampleSystems.singlet(), hash: SnapshotHash(hex: "aa"), libraryHash: "L1")
        let request = AnalysisRequest.mtf(MTFSettings())
        let info = RunInfo(
            engineVersion: "1", snapshotHash: snap.hash, analysis: "mtf", settings: .object(["normalised": .bool(true)]),
            backend: "cpu_reference", precision: .f64Reference, seed: nil, startedUnixMs: 0, durationMs: 1)
        let out = AnalysisOutput(runInfo: info, result: .generic(GenericResult(analysis: "mtf", json: .null)), libraryHash: "L1", request: request)
        #expect(out.isCurrent(for: snap, request: request), "engine-normalised settings do not matter")
        var other = MTFSettings()
        other.maxFrequency = 50
        #expect(!out.isCurrent(for: snap, request: .mtf(other)))
        let newLibrary = ModelSnapshot(system: snap.system, hash: snap.hash, libraryHash: "L2")
        #expect(newLibrary != snap)
        #expect(!out.isCurrent(for: newLibrary, request: request), "a catalog change makes it stale")
        // By name: the engine's recorded settings must contain the request's.
        let byName = EngineAnalysisRequest(analysis: "mtf", settings: ["normalised": .bool(true)])
        let generic = AnalysisOutput(runInfo: info, result: .generic(GenericResult(analysis: "mtf", json: .null)), libraryHash: "L1")
        #expect(generic.isCurrent(for: snap, request: byName))
        #expect(!generic.isCurrent(for: snap, request: EngineAnalysisRequest(analysis: "mtf", settings: ["normalised": .bool(false)])))
        #expect(!generic.isCurrent(for: newLibrary, request: byName))
    }

    /// Engines without by-name analyses fail such jobs cleanly.
    @Test func mockRejectsRequestsByName() async throws {
        let mock = MockEngine(simulatedLatency: .zero)
        let snap = try await mock.openSnapshot(SampleSystems.singlet())
        let job = mock.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap)
        await #expect(throws: EngineError.self) { try await job.result() }
        #expect(job.analysis == "system_summary" && job.request == nil)
        #expect(await mock.analysisDescriptors().count == AnalysisKind.allCases.count)
        await #expect(throws: EngineError.self) { try await mock.loadCatalog(at: URL(fileURLWithPath: "/x.agf")) }
    }
}
