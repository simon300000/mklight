import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel

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
private func paraxialEFL(summary: GenericResult, wavelength w: Int) throws -> (efl: Double, imageHeight: Double) {
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
        #expect(e.info.backends.contains("cpu_reference"))
        let names = await e.analysisDescriptors().map(\.name)
        for n in ["validate", "system_summary", "glass_info", "rng_stream"] { #expect(names.contains(n)) }
        // First-order data: `first_order`, or the `paraxial` placeholder of
        // engines that predate it.
        let descriptors = await e.analysisDescriptors()
        #expect(names.contains(AnalysisKind.firstOrder.engineName(in: descriptors)))
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
        let e = try engine().newSession(system: sample(id))
        let snap = try await e.openSnapshot(sample(id))
        #expect(try e.currentSnapshotHash() == snap.hash)
        let issues = await e.validate(try sample(id))
        #expect(issues.filter { $0.severity == .error }.isEmpty, "\(id): \(issues)")
        let out = try await e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap).result()
        guard case .generic(let g) = out.result else { Issue.record("summary is generic"); return }
        #expect(out.runInfo.snapshotHash == snap.hash && out.runInfo.analysis == "system_summary")
        #expect(out.libraryHash == snap.libraryHash)
        let n = try #require(g.buffer(named: "n_after"))
        #expect(n.shape == [snap.system.surfaces.count, snap.system.wavelengths.wavelengths.count])
        #expect((0..<n.count).allSatisfy { n[$0] > 0.9 && n[$0] < 2.1 }, "indices are relative to air")
    }

    @Test func achromatEFLIs100mm() async throws {
        let e = try engine().makeDocumentEngine()
        let achromat = try sample("achromat")
        let snap = try await e.openSnapshot(achromat)
        let out = try await e.submit(EngineAnalysisRequest(analysis: "system_summary"), on: snap).result()
        guard case .generic(let summary) = out.result else { Issue.record("no summary"); return }
        let (efl, imageHeight) = try paraxialEFL(summary: summary, wavelength: achromat.wavelengths.primary)
        #expect(abs(efl - 100) < 0.01, "EFL \(efl)")
        #expect(abs(imageHeight) < 1e-6, "the sample's image is at the paraxial focus (y = \(imageHeight))")
        // The engine's own first-order analysis, once it is available, must
        // agree: it has to deliver a typed result with a numeric EFL.
        let descriptors = await e.analysisDescriptors()
        let name = AnalysisKind.firstOrder.engineName(in: descriptors)
        if descriptors.first(where: { $0.name == name })?.available == true {
            let fo = try await e.submit(.firstOrder(FirstOrderSettings()), on: snap).result()
            guard case .firstOrder(let r) = fo.result else {
                Issue.record("\(name) did not decode as first-order data: \(fo.result)")
                return
            }
            let engineEFL = try #require(r["efl"], "\(name) reports no 'efl' entry: \(r.entries.map(\.key))")
            #expect(abs(engineEFL - efl) < 1e-6 * efl, "engine EFL \(engineEFL), y-nu trace \(efl)")
        } else {
            let job = e.submit(.firstOrder(FirstOrderSettings()), on: snap)
            await #expect(throws: EngineError.self) { try await job.result() }
        }
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
        guard case .generic(let g) = out.result, case .array(let rows)? = g.json["surfaces"] else {
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
        func run(seed: UInt64) async throws -> (AnalysisOutput, [JobProgress]) {
            var r = EngineAnalysisRequest(
                analysis: "rng_stream", settings: ["count": .integer(4000), "chunk": .integer(500), "delay_ms": .integer(5)])
            r.execution.seed = seed
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
        guard case .generic(let ga) = a.result else { Issue.record("generic"); return }
        let u = try #require(ga.buffer(named: "uniform"))
        #expect(u.count == 4000 && u.dtype == .f64)
        let zeroCopy = u.withUnsafeBufferPointer(of: Double.self) { $0.reduce(0, +) / Double($0.count) }
        #expect(abs(try #require(zeroCopy) - 0.5) < 0.03)
        #expect(u.withUnsafeBufferPointer(of: Float.self) { _ in 0 } == nil, "typed access checks the dtype")
        let (b, _) = try await run(seed: 7)
        let (c, _) = try await run(seed: 8)
        guard case .generic(let gb) = b.result, case .generic(let gc) = c.result else { return }
        #expect(gb.buffer(named: "uniform") == u, "same seed, same stream")
        #expect(gc.buffer(named: "uniform") != u)
        #expect(a.isCurrent(for: snap, request: try #require(a.engineRequest)))
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
        #expect(ok.result.analysisName == "validate")
    }

    /// Extended surface shapes open and validate (every sample above), but
    /// the engine's legacy-only solvers (layout, Seidel, field curvature,
    /// ghosts, merit evaluation) refuse them with a clear error.
    @Test(arguments: ["q_con", "hybrid_doe", "anamorphic", "zernike_sag"])
    func legacyOnlyAnalysesRefuseExtendedShapes(id: String) async throws {
        let e = try engine()
        let system = try sample(id)
        let extended = system.surfaces.contains { !$0.shape.isLegacy }
        #expect(extended)
        let snap = try await e.openSnapshot(system)
        for analysis in ["seidel", "layout"] {
            do {
                _ = try await e.submit(EngineAnalysisRequest(analysis: analysis), on: snap).result()
                Issue.record("\(id): \(analysis) accepted extended shapes")
            } catch let error as EngineError {
                let text = error.errorDescription ?? "\(error)"
                #expect(text.contains("extended surface shapes"), "\(id) \(analysis): \(text)")
            }
        }
    }

    /// The fixture with every extended shape: the engine accepts the mirror's
    /// encoding and reports no validation issues.
    @Test func extendedShapesFixtureValidatesInTheEngine() async throws {
        let e = try engine()
        let system = try MKLightJSON.decode(
            OpticalSystem.self, from: Data(contentsOf: fixtureDir.appendingPathComponent("extended_shapes.json")))
        let issues = await e.validate(system)
        #expect(issues.isEmpty, "\(issues)")
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
        #expect(await failure(EngineAnalysisRequest(analysis: "glass_info", settings: ["name": .string("NO-SUCH-GLASS")]))?.code != nil)
        if let unavailable = await e.analysisDescriptors().first(where: { !$0.available }) {
            #expect(await failure(EngineAnalysisRequest(analysis: unavailable.name))?.code == .unsupported)
        }
        #expect(throws: EngineError.self) { try e.openProject(at: URL(fileURLWithPath: "/nonexistent/x.mklight")) }
        do {
            _ = try e.openProject(at: URL(fileURLWithPath: "/nonexistent/x.mklight"))
        } catch let error as EngineError {
            #expect(error.code == .io)
        }
        let junk = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-junk-\(UUID()).mklight")
        try Data("not json".utf8).write(to: junk)
        defer { try? FileManager.default.removeItem(at: junk) }
        do {
            _ = try e.openProject(at: junk)
            Issue.record("junk opened")
        } catch let error as EngineError {
            #expect(error.code == .parse)
        }
    }

    @Test func typedSettingsFailFastAndUnavailableAnalysesExplainWhy() async throws {
        let e = try engine()
        let snap = try await e.openSnapshot(SampleSystems.singlet())
        var psf = PSFSettings()
        psf.field = 9
        await #expect(throws: EngineError.self) { try await e.submit(.psf(psf), on: snap).result() }
        let layout = await e.analysisDescriptors().first { $0.name == "layout" }
        if layout?.available == false {
            do {
                _ = try await e.submit(.layout(LayoutSettings()), on: snap).result()
            } catch let EngineError.unsupported(why) {
                #expect(why.contains("Layout"))
            }
        }
    }

    @Test func projectsRoundTripThroughEngineAndMirror() async throws {
        let e = try engine()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mklight-ffi-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("achromat.mklight")
        let doc = try e.newSession(system: sample("achromat"))
        try doc.setProjectSection("views", .array([.object(["key": .string("w1"), "analysis": .string("system_summary")])]))
        try doc.save(to: url)
        #expect(try doc.sessionInfo().dirty == false)
        // The Swift mirror reads the engine's file and writes it back losslessly.
        var file = try ProjectFile.decode(Data(contentsOf: url))
        #expect(file.system == (try sample("achromat")))
        #expect(file.sections["views"]?.differences(from: .array([.object(["key": .string("w1"), "analysis": .string("system_summary"), "settings": .null, "ui": .null])])).isEmpty == true)
        file.system.name = "Edited in Swift"
        try file.encoded().write(to: url)
        let reopened = try e.openProject(at: url)
        #expect(try reopened.system().name == "Edited in Swift")
        #expect(try reopened.sessionInfo().migration?.notes.isEmpty ?? true)
        let snapA = try await reopened.openSnapshot(file.system)
        #expect(try reopened.currentSnapshotHash() == snapA.hash)
        // An early app envelope migrates.
        let early = try JSONValue.object([
            "format": .string("mklight-project"), "version": .integer(1), "system": JSONValue.from(sample("singlet")),
        ])
        let earlyURL = dir.appendingPathComponent("early.mklight")
        try MKLightJSON.encode(early).write(to: earlyURL)
        let migrated = try e.openProject(at: earlyURL)
        #expect(try migrated.sessionInfo().migration?.fromVersion == 0)
        #expect(try migrated.project().system == (try sample("singlet")))
    }

    @Test func paramEditsAreAtomicAndProduceNewSnapshots() async throws {
        let e = try engine().newSession()
        let snap = try await e.openSnapshot(SampleSystems.singlet())
        let id = snap.system.surfaces[2].id
        let edited = try await e.applyParamEdits(
            [ParamEdit(.thickness(surface: id), 90), ParamEdit(.wavelengthUm(index: 0), 0.5)], to: snap)
        #expect(edited.hash != snap.hash)
        #expect(edited.system.surfaces[2].thickness == 90 && edited.system.wavelengths.wavelengths[0].um == 0.5)
        #expect(snap.system.surfaces[2].thickness != 90, "snapshots are immutable")
        await #expect(throws: EngineError.self) {
            try await e.applyParamEdits([ParamEdit(.thickness(surface: id), 80), ParamEdit(.conic(surface: SurfaceID(99)), 1)], to: edited)
        }
        #expect(try e.currentSnapshotHash() == edited.hash, "a failed batch changes nothing")
        let inf = try await e.applyParamEdits([ParamEdit(.thickness(surface: snap.system.surfaces[0].id), 1000)], to: edited)
        #expect(inf.system.surfaces[0].thickness == 1000)
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
        let before = try await e.openSnapshot(system)
        #expect(await e.validate(system).contains { $0.severity == .error }, "unknown glass before loading")
        let info = try await e.loadCatalog(at: agf)
        #expect(info.name == "MYCAT" && info.glassCount == 1 && info.source == "user")
        #expect(await e.catalogs().map(\.name) == ["BUILTIN", "MYCAT"])
        let after = try await e.openSnapshot(system)
        #expect(after.hash == before.hash, "same system")
        #expect(after.libraryHash != before.libraryHash, "different glass data")
        #expect(await e.validate(system).filter { $0.severity == .error }.isEmpty)
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
                    guard case .generic(let r) = out.result, case .array(let rows)? = r.json["surfaces"] else { return }
                    #expect(rows[2]["thickness"]?.doubleValue == snap.system.surfaces[2].thickness)
                }
            }
            try await g.waitForAll()
        }
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
