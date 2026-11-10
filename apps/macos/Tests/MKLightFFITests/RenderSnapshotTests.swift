#if canImport(AppKit)
import AppKit
import Foundation
import SwiftUI
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// Renders the spectrum control and the x / y first-order views on the real
/// engine to PNGs (light and dark, scale 2) for a visual check. Skipped
/// unless `MKLIGHT_RENDER_DIR` names an output directory:
///
///     MKLIGHT_RENDER_DIR=/tmp/shots swift test --filter RenderSnapshotTests
@Suite(.serialized, engineLinked, .enabled(if: ProcessInfo.processInfo.environment["MKLIGHT_RENDER_DIR"] != nil))
@MainActor struct RenderSnapshotTests {
    let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MKLIGHT_RENDER_DIR"] ?? NSTemporaryDirectory())

    /// Draws `view` through AppKit (so that menus, fields and pickers
    /// render) after letting its tasks run.
    func render(_ name: String, width: CGFloat, settle: Duration = .milliseconds(600), _ view: some View) async throws {
        for dark in [false, true] {
            let host = NSHostingView(rootView: view.frame(width: width).fixedSize(horizontal: false, vertical: true)
                .background(Color(nsColor: .windowBackgroundColor)))
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: 100), styleMask: [.borderless], backing: .buffered, defer: false)
            window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
            window.contentView = host
            try await Task.sleep(for: settle)
            let size = host.fittingSize
            window.setContentSize(size)
            host.frame = NSRect(origin: .zero, size: size)
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(100))
            let rep = try #require(NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2), pixelsHigh: Int(size.height * 2), bitsPerSample: 8,
                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
            rep.size = size
            NSAppearance(named: dark ? .darkAqua : .aqua)!.performAsCurrentDrawingAppearance {
                host.cacheDisplay(in: host.bounds, to: rep)
            }
            let data = try #require(rep.representation(using: .png, properties: [:]))
            try data.write(to: dir.appendingPathComponent("\(name)-\(dark ? "dark" : "light").png"))
            window.contentView = nil
        }
    }

    @Test func spectrumControl() async throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let e = try FFIEngine()
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let snap = try await e.openSnapshot(system)
        let resolve: SpectrumResolver = { try await SpectrumSetting.resolve($0, engine: e, snapshot: snap) }
        let design = system.wavelengths.wavelengths.map { SpectrumSample(um: $0.um, weight: $0.weight) }
        let cases: [(String, SpectrumSetting, Bool)] = [
            ("design", .design, true),
            ("photopic-engine", .photopic(.default), true),
            ("photopic-approx", .scotopic(SpectralSampling(samples: 15, spacing: .uniform)), false),
            ("blackbody", .blackbody(temperatureK: 3200, sampling: SpectralSampling(samples: 12, lowerUm: 0.4, upperUm: 0.75, spacing: .uniform)), true),
            ("explicit", .explicit(design + [SpectrumSample(um: 0.4358, weight: 0.5)]), true),
            ("sensor", .sensor(table: SpectrumCSV.parse("nm,QE\n400,0.1\n450,0.6\n550,0.8\n650,0.5\n700,0.05").table, sampling: SpectralSampling(samples: 7, lowerUm: 0.4, upperUm: 0.7)), true),
            ("invalid", .photopic(SpectralSampling(lowerUm: 0.3, upperUm: 0.9)), false),
        ]
        for (name, s, engine) in cases {
            try await render("spectrum-\(name)", width: 320,
                SpectrumControl(value: .constant(s.json), systemWavelengths: design, resolve: engine ? resolve : nil)
                    .controlSize(.small).padding(Space.l))
        }
    }

    @Test func settingsPopoverAndBar() async throws {
        let pane = AnalysisPane(analysis: "spot_diagram", settings: ["spectrum": SpectrumSetting.blackbody(temperatureK: 5500, sampling: .default).json!])
        let w = Workspace(engine: try FFIEngine(), panes: [pane])
        w.debounce = .zero
        await w.loadEngineMetadata()
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        w.update(system: system, immediately: true)
        try await Task.sleep(for: .seconds(1))
        let d = try #require(w.descriptor("spot_diagram"))
        try await render("settings-panel", width: 340, SettingsPanel(pane: pane, descriptor: d, system: system))
        try await render("settings-bar", width: 560, SettingsBar(pane: pane, descriptor: d, system: system))
        #expect(SettingsBar.summary(pane, d).contains("Spectrum blackbody 5500 K"))
        let text = SettingsBar.summary(pane, d)
        try text.write(to: dir.appendingPathComponent("settings-summary.txt"), atomically: true, encoding: .utf8)
    }

    @Test func firstOrderViews() async throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let e = try FFIEngine()
        for id in ["anamorphic", "hybrid_doe", "double_gauss", "toroid"] {
            let system = try #require(SampleSystems.bundled(id))
            let snap = try await e.openSnapshot(system)
            let out = try await e.submit(EngineAnalysisRequest(analysis: "first_order", settings: ["wavelength": .object(["type": .string("all")])]), on: snap).result()
            let view = try #require(FirstOrderView(output: out, system: system))
            try await render("first-order-\(id)", width: 420, view.frame(height: 640))
            if let json = try? MKLightJSON.encode(out.json, pretty: true) {
                try json.write(to: dir.appendingPathComponent("first-order-\(id).json"))
            }
        }
        // The toolbar's status, its popover and the inspector's System tab.
        let system = try #require(SampleSystems.bundled("anamorphic"))
        let doc = DesignDocument(system: system)
        let w = Workspace(engine: try FFIEngine())
        w.debounce = .zero
        await w.loadEngineMetadata()
        w.update(system: system, immediately: true)
        let clock = ContinuousClock(), end = clock.now + .seconds(30)
        while w.summary.output == nil && clock.now < end { try await Task.sleep(for: .milliseconds(20)) }
        let window = WindowState()
        try await render("status-capsule", width: 480, KeyNumbers(workspace: w, system: system).padding(Space.m))
        try await render("status-popover", width: 340, StatusDetails(workspace: w, window: window, engine: w.engine, system: system))
        try await render("inspector-system", width: 300, SystemInspector(design: doc, workspace: w, window: window).frame(height: 900))
    }

    /// The tolerance editor and its results on the Double Gauss and the Cooke.
    @Test func tolerancePanel() async throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for sample in ["double_gauss", "cooke_triplet"] {
            let system = try #require(SampleSystems.bundled(sample))
            let doc = DesignDocument(system: system)
            let w = Workspace(engine: try FFIEngine())
            w.debounce = .zero
            await w.loadEngineMetadata()
            w.update(system: system, immediately: true)
            while w.layout.output == nil { try await Task.sleep(for: .milliseconds(20)) }
            let glasses = system.surfaces.filter { if case .catalog = $0.material { true } else { false } }.map(\.material)
            while Set(glasses).count > w.glassNumbers.count { _ = w.abbeNumbers(of: system); try await Task.sleep(for: .milliseconds(20)) }
            let window = WindowState()
            let session = ToleranceSession()
            session.load(doc)
            let context = ToleranceContext(
                system: system, semiDiameters: session.semiDiameters(from: w.layout.output), abbeNumbers: w.abbeNumbers(of: system))
            try await render("tol-editor-\(sample)", width: 340, ToleranceEditor(design: doc, workspace: w, session: session, context: context, edit: { name, change in
                session.edit(doc, name, context: context, undoManager: nil, change)
            }))
            session.edit(doc, "Set", context: context, undoManager: nil) {
                $0.trials = 500
                $0.criterion.threshold = sample == "double_gauss" ? 20.5 : 40
            }
            session.start(in: w, context: context, merit: doc.merit)
            while session.isRunning || session.result == nil { try await Task.sleep(for: .milliseconds(20)) }
            let r = try #require(session.result)
            try await render("tol-result-\(sample)", width: 340, ToleranceResultView(design: doc, workspace: w, window: window, session: session, result: r))
            // The worst trial rebuilt and previewed, its bar's trials listed.
            if let k = r.trialsWorstFirst.first {
                session.preview(.trial(k), in: w)
                while session.previewLoading { try await Task.sleep(for: .milliseconds(20)) }
                try await render("tol-trial-\(sample)", width: 340, ToleranceResultView(design: doc, workspace: w, window: window, session: session, result: r))
                session.endPreview(in: w)
            }
            session.edit(doc, "MTF", context: context, undoManager: nil) {
                $0.criterion.kind = .mtf
                $0.criterion.threshold = nil
                $0.mode = .monteCarlo
                $0.trials = 40
            }
            session.start(in: w, context: context, merit: doc.merit)
            while session.isRunning || session.result == nil { try await Task.sleep(for: .milliseconds(20)) }
            try await render("tol-result-mtf-\(sample)", width: 340, ToleranceResultView(design: doc, workspace: w, window: window, session: session, result: try #require(session.result)))
        }
    }

    /// Whole analysis panes at fixed sizes through AppKit (menus, fields,
    /// scroll views and glass draw): the spot diagram's field × focus grid
    /// at narrow, medium and wide widths, and the settings forms generated
    /// from the schemas (a name ending in "+settings" opens them).
    @Test func analysisPanes() async throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let cases: [(name: String, sample: String, analysis: String, settings: [String: JSONValue], size: CGSize)] = [
            ("pane-spot-narrow", "double_gauss", "spot_diagram", Workspace.spotStartingSettings, CGSize(width: 300, height: 560)),
            ("pane-spot", "double_gauss", "spot_diagram", Workspace.spotStartingSettings, CGSize(width: 520, height: 520)),
            ("pane-spot-wide", "double_gauss", "spot_diagram", Workspace.spotStartingSettings, CGSize(width: 960, height: 560)),
            ("pane-spot-5-dense", "double_gauss", "spot_diagram", ["through_focus": .object(["min": .number(-0.15), "max": .number(0.15), "steps": .integer(5)]), "rings": .integer(24)], CGSize(width: 820, height: 520)),
            ("pane-spot-single", "double_gauss", "spot_diagram", ["through_focus": .null], CGSize(width: 520, height: 420)),
            ("pane-spot+settings", "double_gauss", "spot_diagram", Workspace.spotStartingSettings, CGSize(width: 520, height: 760)),
            ("pane-mtf+settings", "double_gauss", "mtf", ["detector": .object(["pitch_mm": .array([.number(0.004), .number(0.004)]), "fill_factor": .number(0.9), "aperture_mm": .null, "diffusion_sigma_mm": .number(0)])], CGSize(width: 560, height: 820)),
            ("pane-transmission+settings", "cooke_triplet", "system_transmission", ["pupil": .object(["type": .string("random"), "count": .integer(512), "seed": .integer(3)])], CGSize(width: 560, height: 640)),
            ("pane-tfmtf+settings", "cooke_triplet", "through_focus_mtf", [:], CGSize(width: 480, height: 720)),
            ("pane-ee+settings", "cooke_triplet", "diffraction_encircled_energy", ["center": .object(["type": .string("point"), "x": .number(0.001), "y": .number(0)])], CGSize(width: 520, height: 720)),
        ]
        let engine = try FFIEngine(configuration: ["worker_threads": .integer(4)])
        for c in cases {
            let pane = AnalysisPane(analysis: c.analysis, settings: c.settings)
            pane.showsSettings = c.name.hasSuffix("+settings")
            let w = Workspace(engine: engine, panes: [pane])
            w.debounce = .zero
            await w.loadEngineMetadata()
            let system = try #require(SampleSystems.bundled(c.sample))
            w.update(system: system, immediately: true)
            let clock = ContinuousClock()
            let end = clock.now + .seconds(120)
            while (pane.output == nil || pane.isRunning || w.isBusy) && clock.now < end {
                if case .failed(let why) = pane.phase { Issue.record("\(c.name): \(why)"); break }
                try await Task.sleep(for: .milliseconds(20))
            }
            guard pane.output != nil else { Issue.record("\(c.name): no output, \(pane.phase)"); continue }
            let window = WindowState()
            for dark in [false, true] {
                let view = AnalysisPaneView(pane: pane, workspace: w, window: window, system: system)
                    .frame(width: c.size.width, height: c.size.height)
                    .background(Color(nsColor: .windowBackgroundColor))
                let host = NSHostingView(rootView: view)
                let win = NSWindow(contentRect: NSRect(origin: .zero, size: c.size), styleMask: [.borderless], backing: .buffered, defer: false)
                win.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
                win.contentView = host
                host.frame = NSRect(origin: .zero, size: c.size)
                try await Task.sleep(for: .milliseconds(700))
                host.layoutSubtreeIfNeeded()
                try await Task.sleep(for: .milliseconds(100))
                let rep = try #require(NSBitmapImageRep(
                    bitmapDataPlanes: nil, pixelsWide: Int(c.size.width * 2), pixelsHigh: Int(c.size.height * 2), bitsPerSample: 8,
                    samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
                rep.size = c.size
                NSAppearance(named: dark ? .darkAqua : .aqua)!.performAsCurrentDrawingAppearance {
                    host.cacheDisplay(in: host.bounds, to: rep)
                }
                let data = try #require(rep.representation(using: .png, properties: [:]))
                try data.write(to: dir.appendingPathComponent("\(c.name)-\(dark ? "dark" : "light").png"))
                win.contentView = nil
            }
        }
    }
}
#endif
