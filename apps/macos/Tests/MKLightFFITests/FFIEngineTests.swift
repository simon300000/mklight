import Foundation
import Synchronization
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

#if canImport(CMKLight)
import CMKLight
#endif

/// Integration tests against the real Rust engine (`libmklight_ffi.a`).
/// They are skipped, with the reason below, when the library was not
/// linked; build it with `scripts/build-macos.sh --engine-only`.
let engineLinked: ConditionTrait = .enabled(
    if: FFIEngine.isLinked,
    "libmklight_ffi.a is not linked: run scripts/build-macos.sh --engine-only, then swift test")

private let fixtureDir = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("MKLightModelTests/Fixtures")

/// One engine for the suite (two workers, fast progress polling).
private let shared: FFIEngine? = try? FFIEngine(configuration: [
    "worker_threads": .integer(2), "progress_interval_ms": .integer(10),
])

private func engine() throws -> FFIEngine { try #require(shared) }

private func sample(_ id: String) throws -> OpticalSystem { try #require(SampleSystems.bundled(id)) }

/// Paraxial EFL by a y-nu trace over engine-supplied numbers: curvatures and
/// thicknesses from `system_summary`, indices from its `n_after` buffer
/// (read without copying). Test-only arithmetic that cross-checks the data
/// path; the app itself never computes optics.
private func paraxialEFL(summary: AnalysisOutput, wavelength w: Int) throws -> (efl: Double, imageHeight: Double) {
    let n = try #require(summary.buffer(named: "n_after"))
    #expect(n.dtype == .f64 && n.shape.count == 2)
    guard case .array(let rows)? = summary.json["surfaces"] else { throw EngineError.failed("no surface table") }
    var y = 1.0, nu = 0.0  // unit-height ray parallel to the axis
    var nBefore = n[[0, w]]
    for (i, row) in rows.enumerated() where i > 0 {
        let c = row["curvature"]?.doubleValue ?? 0
        let nAfter = n[[i, w]]
        nu -= y * c * (nAfter - nBefore)
        nBefore = nAfter
        if i < rows.count - 1 { y += (row["thickness"]?.doubleValue ?? 0) * nu / nAfter }
    }
    return (-1 / nu, y)
}

@Suite(.serialized, engineLinked) struct FFIEngineTests {
    @Test func libraryAndAnalysisList() async throws {
        let e = try engine()
        #expect(!e.info.isMock && !e.info.version.isEmpty)
        let names = await e.analysisDescriptors().map(\.name)
        for n in ["validate", "system_summary", "glass_info", "rng_stream", "first_order"] { #expect(names.contains(n)) }
        let glass = try #require(await e.analysisDescriptors().first { $0.name == "glass_info" })
        #expect(glass.settingsFields.first?.name == "name" && glass.settingsFields.first?.required == true)
        #expect(glass.defaultSettings == .null, "a required setting means no defaults")
    }

    /// The binding reproduces the engine's BLAKE3 content hashes of the
    /// fixtures (the mock engine cannot: it hashes with SHA-256).
    @Test func snapshotHashesMatchEngineFixtures() async throws {
        let e = try engine()
        let hashes = try MKLightJSON.decode(
            [String: String].self, from: Data(contentsOf: fixtureDir.appendingPathComponent("snapshot_hashes.json")))
        #expect(hashes.count >= 3)
        for (file, hex) in hashes {
            let system = try MKLightJSON.decode(
                OpticalSystem.self, from: Data(contentsOf: fixtureDir.appendingPathComponent(file)))
            let snap = try await e.openSnapshot(system)
            #expect(snap.hash.hex == hex, "\(file)")
            #expect(snap.libraryHash?.count == 64)
        }
    }

    @Test(arguments: SampleSystems.bundled.map(\.id))
    func everySampleOpensValidatesAndSummarises(id: String) async throws {
        let e = try engine().newSession()
        let snap = try await e.openSnapshot(sample(id))
        let validation = try await e.submit(EngineAnalysisRequest(analysis: "validate"), on: snap).result()
        #expect(validation.json["ok"] == .bool(true) && validation.json["errors"]?.int == 0, "\(id): \(validation.json)")
        let out = try await e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap).result()
        #expect(out.runInfo.snapshotHash == snap.hash && out.runInfo.analysis == "system_summary")
        #expect(out.libraryHash == snap.libraryHash)
        let n = try #require(out.buffer(named: "n_after"))
        #expect(n.shape == [snap.system.surfaces.count, snap.system.wavelengths.wavelengths.count])
        #expect((0..<n.count).allSatisfy { n[$0] > 0.9 && n[$0] < 2.1 }, "indices are relative to air")
    }

    @Test func achromatEFLIs100mm() async throws {
        let e = try engine().makeDocumentEngine()
        let achromat = try sample("achromat")
        let snap = try await e.openSnapshot(achromat)
        let summary = try await e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap).result()
        let (efl, imageHeight) = try paraxialEFL(summary: summary, wavelength: achromat.wavelengths.primary)
        #expect(abs(efl - 100) < 0.01, "EFL \(efl)")
        #expect(abs(imageHeight) < 1e-6, "the sample's image is at the paraxial focus (y = \(imageHeight))")
        // The engine's own first-order analysis must agree: the pane's
        // reader has to find a numeric EFL in its report.
        let fo = try await e.submit(EngineAnalysisRequest(analysis: "first_order"), on: snap).result()
        let report = try #require(FirstOrderData(fo.json), "first_order did not read as first-order data: \(fo.json)")
        let engineEFL = try #require(report.entry(["efl"])?.value, "first_order reports no 'efl' entry: \(report.entries.map(\.key))")
        #expect(abs(engineEFL - efl) < 1e-6 * efl, "engine EFL \(engineEFL), y-nu trace \(efl)")
    }

    /// Submitting never blocks the caller (the main actor in the app):
    /// while another thread holds the session lock, as a catalog load does,
    /// `submit` returns at once and the job runs once the lock is free,
    /// against its own snapshot. A job cancelled while waiting is never run.
    @Test func submitDoesNotBlockTheCaller() async throws {
        let e = try engine().newSession()
        let achromat = try await e.openSnapshot(try sample("achromat"))
        let singlet = try await e.openSnapshot(SampleSystems.singlet())  // the session holds the singlet now
        await withCheckedContinuation { (held: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                try? e.session.locked { _ in
                    held.resume()
                    usleep(400_000)
                }
            }
        }
        let clock = ContinuousClock(), t0 = clock.now
        let job = e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: achromat)
        let doomed = e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: singlet)
        let elapsed = clock.now - t0
        doomed.cancel()
        #expect(elapsed < .milliseconds(100), "submit waited \(elapsed) for the session lock")
        let out = try await job.result()
        #expect(out.runInfo.snapshotHash == achromat.hash)
        guard case .array(let rows)? = out.json["surfaces"] else {
            Issue.record("no surface table")
            return
        }
        #expect(rows.count == achromat.system.surfaces.count)
        await #expect(throws: EngineError.cancelled) { try await doomed.result() }
        #expect(clock.now - t0 >= .milliseconds(350), "the job waited for the lock")
    }

    @Test func analysisReportsProgressAndIsReproducible() async throws {
        let e = try engine()
        let snap = try await e.openSnapshot(SampleSystems.singlet())
        func run(seed: Int64) async throws -> (AnalysisOutput, [JobProgress]) {
            let r = EngineAnalysisRequest(
                analysis: "rng_stream",
                settings: ["count": .integer(4000), "chunk": .integer(500), "delay_ms": .integer(5), "seed": .integer(seed)])
            let job = e.submit(r, on: snap)
            var updates: [JobProgress] = []
            for await p in job.progress { updates.append(p) }
            return (try await job.result(), updates)
        }
        let (a, updates) = try await run(seed: 7)
        #expect(updates.count >= 2)
        #expect(zip(updates, updates.dropFirst()).allSatisfy { $0.fraction <= $1.fraction }, "monotone")
        #expect(updates.last?.fraction == 1)
        #expect(a.runInfo.seed == 7 && a.runInfo.settings["seed"] == .integer(7))
        let u = try #require(a.buffer(named: "uniform"))
        #expect(u.count == 4000 && u.dtype == .f64)
        let zeroCopy = u.withUnsafeBufferPointer(of: Double.self) { $0.reduce(0, +) / Double($0.count) }
        #expect(abs(try #require(zeroCopy) - 0.5) < 0.03)
        #expect(u.withUnsafeBufferPointer(of: Float.self) { _ in 0 } == nil, "typed access checks the dtype")
        let (b, _) = try await run(seed: 7)
        let (c, _) = try await run(seed: 8)
        #expect(b.buffer(named: "uniform") == u, "same seed, same stream")
        #expect(c.buffer(named: "uniform") != u)
        #expect(a.isCurrent(for: snap, request: try #require(a.engineRequest)))
        #expect(!a.isCurrent(for: snap, request: try #require(c.engineRequest)), "another seed is another request")
    }

    /// A request by name runs with exactly its settings: a seed it leaves
    /// out is the engine's default (the one the settings form shows), not
    /// one the binding adds, and a seed it sets is the one used.
    @Test func seedComesFromTheSettingsOnly() async throws {
        let e = try engine()
        let snap = try await e.openSnapshot(SampleSystems.singlet())
        let descriptor = try #require(await e.analysisDescriptors().first { $0.name == "rng_stream" })
        let shown = try #require(descriptor.settingsFields.first { $0.name == "seed" }?.defaultValue)
        var r = EngineAnalysisRequest(analysis: "rng_stream", settings: ["count": .integer(16)])
        let byDefault = try await e.submit(r, on: snap).result()
        #expect(byDefault.runInfo.settings["seed"] == shown)
        #expect(byDefault.runInfo.seed == shown.int.flatMap { UInt64(exactly: $0) })
        r.settings["seed"] = .integer(5)
        let explicit = try await e.submit(r, on: snap).result()
        #expect(explicit.runInfo.settings["seed"] == .integer(5) && explicit.runInfo.seed == 5)
        #expect(explicit.buffer(named: "uniform") != byDefault.buffer(named: "uniform"))
    }

    @Test func cancellingAJobStopsIt() async throws {
        let e = try engine()
        let snap = try await e.openSnapshot(SampleSystems.singlet())
        let r = EngineAnalysisRequest(
            analysis: "rng_stream", settings: ["count": .integer(1_000_000), "chunk": .integer(1000), "delay_ms": .integer(20)])
        let job = e.submit(r, on: snap)
        let clock = ContinuousClock(), t0 = clock.now
        var sawRunning = false
        for await p in job.progress where p.stage != "Queued" {
            sawRunning = true
            job.cancel()
            break
        }
        #expect(sawRunning)
        await #expect(throws: EngineError.cancelled) { try await job.result() }
        #expect(clock.now - t0 < .seconds(5), "the job stopped long before its 20 s of work")
        // The engine is still usable afterwards.
        let ok = try await e.submit(EngineAnalysisRequest(analysis: "validate"), on: snap).result()
        #expect(ok.runInfo.analysis == "validate")
    }

    /// Extended surface shapes open, validate and now run through every
    /// solver, including the layout (which draws their real profiles, marks
    /// DOE surfaces and reports automatic semi-diameters) and Seidel.
    @Test(arguments: ["q_con", "hybrid_doe", "anamorphic", "zernike_sag"])
    func extendedShapesRunThroughLayoutAndSeidel(id: String) async throws {
        let e = try engine()
        let system = try sample(id)
        let extended = system.surfaces.contains { !$0.shape.isLegacy }
        #expect(extended)
        let snap = try await e.openSnapshot(system)
        for analysis in ["seidel", "layout"] {
            let out = try await e.submit(EngineAnalysisRequest(analysis: analysis), on: snap).result()
            #expect(out.runInfo.analysis == analysis, "\(id) \(analysis)")
        }
        let layout = try await e.submit(EngineAnalysisRequest(analysis: "layout"), on: snap).result()
        let surfaces = layout.json["surfaces"]?.array ?? []
        #expect(!surfaces.isEmpty)
        let doe = surfaces.contains { $0["is_diffractive"]?.bool == true }
        #expect(doe == system.surfaces.contains { if case .diffractive = $0.shape { true } else { false } }, "\(id)")
        #expect(surfaces.allSatisfy { ($0["semi_diameter"]?.nf ?? .nan).isFinite || $0["index"]?.int == 0 }, "\(id)")
        // The x–z section and the 3-D surface mesh, as the viewport asks for them.
        let xz = try await e.submit(
            EngineAnalysisRequest(analysis: "layout", settings: ["section": .string("xz"), "mesh_rings": .integer(6)]), on: snap
        ).result()
        let scene = try #require(EngineLayout.scene(from: xz))
        #expect(scene.section == "xz")
        #expect(!scene.mesh.isEmpty, "\(id): no surface mesh")
        #expect(scene.mesh.allSatisfy { $0.surface >= 1 && $0.a.z.isFinite })
        #expect(!scene.elements.isEmpty || id == "zernike_sag", "\(id): no glass fills in the x–z section")
        if doe { #expect(scene.polylines.contains { $0.doe }, "\(id): DOE flag not read") }
    }

    /// Footprint overlays become closed beam outlines on every surface.
    @Test func footprintOverlaysReadAsOutlines() async throws {
        let e = try engine()
        let snap = try await e.openSnapshot(try sample("double_gauss"))
        let out = try await e.submit(EngineAnalysisRequest(analysis: "footprint", settings: ["rings": .integer(4)]), on: snap).result()
        let overlay = try #require(FootprintOverlay(output: out))
        #expect(overlay.outlines.count >= 10)
        #expect(overlay.outlines.allSatisfy { $0.points.count >= 3 })
        #expect(Set(overlay.outlines.map(\.surface)).count >= 10)
    }

    /// The fixture with every extended shape: the engine accepts the mirror's
    /// encoding and reports no validation issues.
    @Test func extendedShapesFixtureValidatesInTheEngine() async throws {
        let e = try engine()
        let system = try MKLightJSON.decode(
            OpticalSystem.self, from: Data(contentsOf: fixtureDir.appendingPathComponent("extended_shapes.json")))
        let snap = try await e.openSnapshot(system)
        let out = try await e.submit(EngineAnalysisRequest(analysis: "validate"), on: snap).result()
        #expect(out.json["issues"]?.array?.isEmpty == true, "\(out.json)")
        #expect(system.structuralIssues().isEmpty)
    }

    @Test func errorsMapToCodes() async throws {
        let e = try engine()
        var broken = SampleSystems.singlet()
        broken.stop = 0
        await #expect(throws: EngineError.self) { try await e.openSnapshot(broken) }
        do {
            _ = try await e.openSnapshot(broken)
        } catch let error as EngineError {
            #expect(error.code == .invalidModel)
            if case .invalidModel(let issues) = error { #expect(issues.first?.contains("stop") == true) }
        }
        let snap = try await e.openSnapshot(SampleSystems.singlet())
        func failure(_ r: EngineAnalysisRequest) async -> EngineError? {
            do {
                _ = try await e.submit(r, on: snap).result()
                return nil
            } catch { return error as? EngineError }
        }
        #expect(await failure(EngineAnalysisRequest(analysis: "no_such_analysis"))?.code == .notFound)
        #expect(await failure(EngineAnalysisRequest(analysis: "rng_stream", settings: ["bogus": .integer(1)]))?.code == .invalidArgument)
        #expect(await failure(EngineAnalysisRequest(analysis: "glass_info"))?.code == .invalidArgument, "name is required")
        #expect(await failure(EngineAnalysisRequest(analysis: "glass_info", settings: ["name": .string("NO-SUCH-GLASS")]))?.code == .notFound)
        if let unavailable = await e.analysisDescriptors().first(where: { !$0.available }) {
            #expect(await failure(EngineAnalysisRequest(analysis: unavailable.name))?.code == .unsupported)
        }
        // Statuses of the session calls keep their codes too.
        do {
            _ = try await e.loadCatalog(at: URL(fileURLWithPath: "/nonexistent/x.agf"))
            Issue.record("a missing catalog loaded")
        } catch let error as EngineError {
            #expect(error.code == .io)
        }
        let junk = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-junk-\(UUID()).json")
        try Data("not json".utf8).write(to: junk)
        defer { try? FileManager.default.removeItem(at: junk) }
        do {
            _ = try await e.loadCatalog(at: junk)
            Issue.record("junk loaded as a catalog")
        } catch let error as EngineError {
            #expect(error.code == .parse)
        }
    }

    /// A job that fails while running throws the error of its own status
    /// code and message, not a generic "job failed".
    @Test func failedJobsKeepTheirErrorCode() async throws {
        let e = try engine().newSession()
        var system = SampleSystems.singlet()
        system.surfaces[1].material = .catalog(catalog: nil, name: "NOPE-42")
        let snap = try await e.openSnapshot(system)
        do {
            _ = try await e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap).result()
            Issue.record("a system with an unknown glass was summarised")
        } catch let error as EngineError {
            #expect(error.code == .notFound, "\(error)")
            #expect(error.errorDescription?.contains("NOPE-42") == true, "\(error)")
            guard case .engine(.notFound, _) = error else {
                Issue.record("expected .engine(.notFound, …), got \(error)")
                return
            }
        }
    }

    /// Which jobs can be stopped is what the engine's analysis list says.
    @Test func stopSupportComesFromTheAnalysisList() async throws {
        let e = try engine()
        let stoppable = Set(await e.analysisDescriptors().filter(\.stoppable).map(\.name))
        #expect(stoppable == ["optimize", "tolerance", "glass_substitution"])
        let snap = try await e.openSnapshot(SampleSystems.singlet())
        for name in ["optimize", "tolerance", "glass_substitution", "rng_stream", "validate"] {
            let job = e.submit(EngineAnalysisRequest(analysis: name), on: snap)
            #expect(job.canStop == stoppable.contains(name), "\(name)")
            job.cancel()
            _ = try? await job.result()
        }
    }

    /// The app reads and writes `.mklight` files itself (`ProjectFile`) and
    /// has no binding for the engine's project calls, so this test makes
    /// them directly: a file the engine saves reads in the mirror and
    /// writes back losslessly, and the engine opens what the mirror wrote.
    @Test func projectsRoundTripThroughEngineAndMirror() async throws {
        #if canImport(CMKLight)
        func check(_ status: MKStatus) throws {
            guard status != MK_OK else { return }
            let message = String(cString: mk_last_error())
            throw EngineError.from(code: EngineErrorCode(rawValue: status) ?? .internal, message: message)
        }
        func json(_ call: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> MKStatus) throws -> JSONValue {
            var out: UnsafeMutablePointer<CChar>?
            let status = call(&out)
            defer { mk_string_free(out) }
            try check(status)
            return try JSONValue.parse(Data(String(cString: try #require(out)).utf8))
        }
        /// `mk_session_open`: the opened project's system and session info.
        func open(_ url: URL, on e: FFIEngine) throws -> (system: OpticalSystem, info: JSONValue) {
            var session: OpaquePointer?
            try check(mk_session_open(e.core.pointer, url.path, &session))
            defer { mk_session_destroy(session) }
            let system = try json { mk_session_get_system_json(session, $0) }.decode(OpticalSystem.self)
            return (system, try json { mk_session_info_json(session, $0) })
        }
        let e = try engine()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-ffi-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("achromat.mklight")
        let doc = try e.newSession()
        _ = try await doc.openSnapshot(sample("achromat"))
        let views = #"[{"key":"w1","analysis":"system_summary"}]"#
        let saved = try doc.session.locked { s in
            try check(mk_session_set_project_section_json(s, "views", views))
            try check(mk_session_save(s, url.path))
            return try json { mk_session_info_json(s, $0) }
        }
        #expect(saved["dirty"] == .bool(false))
        // The Swift mirror reads the engine's file and writes it back losslessly.
        var file = try ProjectFile.decode(Data(contentsOf: url))
        #expect(file.system == (try sample("achromat")))
        #expect(file.sections["views"]?.differences(from: .array([.object(["key": .string("w1"), "analysis": .string("system_summary"), "settings": .null, "ui": .null])])).isEmpty == true)
        file.system.name = "Edited in Swift"
        try file.encoded().write(to: url)
        let reopened = try open(url, on: e)
        #expect(reopened.system.name == "Edited in Swift" && reopened.system == file.system)
        #expect(reopened.info["open_warnings"]?.array?.isEmpty == true)
        let snap = try await e.newSession().openSnapshot(file.system)
        #expect(reopened.info["snapshot_hash"] == .string(snap.hash.hex), "the engine read the system the mirror wrote")
        // A document without a schema version is refused by the engine too.
        let early = try JSONValue.object([
            "format": .string("mklight-project"), "version": .integer(1), "system": JSONValue.from(sample("singlet")),
        ])
        let earlyURL = dir.appendingPathComponent("early.mklight")
        try MKLightJSON.encode(early).write(to: earlyURL)
        #expect(throws: EngineError.self) { try open(earlyURL, on: e) }
        #else
        Issue.record("the engine library is linked, but the tests cannot import its C module")
        #endif
    }

    @Test func catalogsChangeTheLibraryHash() async throws {
        let e = try engine().newSession()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-cat-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let agf = dir.appendingPathComponent("MYCAT.agf")
        try Data(
            """
            CC Test catalog
            NM MY-BK7 2 517642 1.5168 64.17 0 1 0
            CD 1.03961212E+000 6.00069867E-003 2.31792344E-001 2.00179144E-002 1.01046945E+000 1.03560653E+002 0 0 0 0
            LD 3.0E-001 2.5E+000

            """.utf8
        ).write(to: agf)
        var system = SampleSystems.singlet()
        system.surfaces[1].material = .catalog(catalog: "MYCAT", name: "MY-BK7")
        func validationErrors(_ snapshot: ModelSnapshot) async throws -> Int? {
            try await e.submit(EngineAnalysisRequest(analysis: "validate"), on: snapshot).result().json["errors"]?.int
        }
        let before = try await e.openSnapshot(system)
        #expect(try await validationErrors(before) ?? 0 > 0, "unknown glass before loading")
        let info = try await e.loadCatalog(at: agf)
        #expect(info.name == "MYCAT" && info.glassCount == 1 && info.source == "user")
        #expect(await e.catalogs().map(\.name) == ["BUILTIN", "MYCAT"])
        let after = try await e.openSnapshot(system)
        #expect(after.hash == before.hash, "same system")
        #expect(after.libraryHash != before.libraryHash, "different glass data")
        #expect(try await validationErrors(after) == 0)
        let glasses = try await e.glasses(in: "MYCAT")
        #expect(glasses.map(\.name) == ["MY-BK7"] && abs(glasses[0].nd - 1.5168) < 1e-3)
        let out = try await e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: after).result()
        #expect(out.isCurrent(for: after, request: EngineAnalysisRequest(analysis: "system_summary")))
        #expect(!out.isCurrent(for: before, request: EngineAnalysisRequest(analysis: "system_summary")), "stale against the old library")
        await #expect(throws: EngineError.self) { try await e.loadCatalog(at: dir.appendingPathComponent("missing.agf")) }
    }

    /// Jobs submitted concurrently on different snapshots each run on their
    /// own snapshot (set + submit is one locked step).
    @Test func concurrentSubmissionsKeepTheirSnapshots() async throws {
        let e = try engine().newSession()
        var systems: [OpticalSystem] = []
        for k in 0..<8 {
            var s = SampleSystems.singlet()
            s.surfaces[2].thickness = 90 + Double(k)
            systems.append(s)
        }
        let snaps = try await withThrowingTaskGroup(of: ModelSnapshot.self) { g in
            for s in systems { g.addTask { try await e.openSnapshot(s) } }
            return try await g.reduce(into: []) { $0.append($1) }
        }
        try await withThrowingTaskGroup(of: Void.self) { g in
            for snap in snaps {
                g.addTask {
                    let out = try await e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap).result()
                    #expect(out.runInfo.snapshotHash == snap.hash)
                    let rows = try #require(out.json["surfaces"]?.array)
                    #expect(rows[2]["thickness"]?.doubleValue == snap.system.surfaces[2].thickness)
                }
            }
            try await g.waitForAll()
        }
    }

    /// The route the engine records for automatic runs reads as a line of
    /// the run details: every decision it sends is one the app knows.
    /// Analyses without a GPU path, and an engine without a GPU, send none.
    @Test func routeDecisionsOfAutomaticRunsReadAsALine() async throws {
        let e = try engine()
        let snap = try await e.openSnapshot(try #require(SampleSystems.bundled("double_gauss")))
        for (analysis, settings) in [
            ("spot_diagram", ["rings": JSONValue.integer(6)]), ("spot_diagram", ["rings": .integer(200)]),
            ("encircled_energy", [:]), ("wavefront_map", [:]), ("system_summary", [:]),
        ] {
            let out = try await e.submit(EngineAnalysisRequest(analysis: analysis, settings: settings), on: snap).result()
            let backend = try #require(out.backend, "\(analysis)")
            #expect(backend.requested == "default:auto", "\(analysis)")
            let line = RunRoute.line(requested: backend.requested, decision: backend.costDecision)
            #expect((line == nil) == (backend.costDecision == nil), "\(analysis): \(String(describing: backend.costDecision))")
            if let line {
                // The line names the backend the decision chose.
                let chosen = backend.costDecision?["decision"]?.string
                #expect(line.hasPrefix(chosen == "cpu_reference" ? "CPU" : "GPU"), "\(analysis): \(line)")
                if backend.costDecision?["rule"]?.string == "cost model" { #expect(line.contains("GPU predicted "), "\(line)") }
            }
            #expect(RunRoute.attempts(out.runInfo.failedAttempts).count == out.runInfo.failedAttempts.count)
        }
    }

    /// A preparation the engine refused to start reads as failed, with the
    /// engine's message, rather than as not started.
    @Test func aRefusedGPUPreparationReadsAsFailed() throws {
        let e = try FFIEngine()
        #expect(e.gpuPreparation() == .idle, "not started")
        e.core.gpuPreparationError.withLock { $0 = "cache_dir is not valid UTF-8" }
        #expect(e.gpuPreparation() == .failed(device: "", message: "cache_dir is not valid UTF-8"))
    }

    /// Engine and session handles are released when the last reference goes.
    @Test func handlesAreFreed() async throws {
        weak var core: FFIEngineCore?
        weak var session: FFISessionHandle?
        do {
            let e = try FFIEngine()
            core = e.core
            session = e.session
            let snap = try await e.openSnapshot(SampleSystems.singlet())
            _ = try await e.submit(EngineAnalysisRequest(analysis: "validate"), on: snap).result()
            #expect(core != nil)
        }
        // Jobs release their references when their tasks finish.
        for _ in 0..<50 where core != nil || session != nil { try await Task.sleep(for: .milliseconds(20)) }
        #expect(session == nil && core == nil)
    }
}

@Suite struct FFIAvailabilityTests {
    /// Without the library, construction fails with a clear error instead of
    /// crashing; with it, it succeeds.
    @Test func constructionReflectsLinkage() {
        if FFIEngine.isLinked {
            #expect(throws: Never.self) { _ = try FFIEngine() }
        } else {
            #expect(throws: EngineError.self) { _ = try FFIEngine() }
        }
    }
}
