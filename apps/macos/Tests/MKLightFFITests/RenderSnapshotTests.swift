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
        try await render("settings-popover", width: 340, SettingsPopover(pane: pane, descriptor: d, system: system))
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
}
#endif
