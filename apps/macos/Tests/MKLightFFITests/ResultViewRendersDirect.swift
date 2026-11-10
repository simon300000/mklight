import AppKit
import Foundation
import SwiftUI
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

extension ResultViewRenders {
    private func write<V: View>(_ view: V, size: CGSize, to path: String) throws {
        for dark in [false, true] {
            let v = view.frame(width: size.width, height: size.height)
                .background(Color(nsColor: .windowBackgroundColor))
                .environment(\.colorScheme, dark ? .dark : .light)
                .environment(\.locale, Locale(identifier: "en_US_POSIX"))
            var png: Data?
            NSAppearance(named: dark ? .darkAqua : .aqua)!.performAsCurrentDrawingAppearance {
                let r = ImageRenderer(content: v)
                r.scale = 2
                if let img = r.nsImage, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) {
                    png = rep.representation(using: .png, properties: [:])
                }
            }
            try #require(png).write(to: URL(fileURLWithPath: "\(path)-\(dark ? "dark" : "light").png"))
        }
    }

    /// Draws `view` through an AppKit window, which (unlike `ImageRenderer`)
    /// draws the content of scroll views.
    private func host<V: View>(_ view: V, size: CGSize, to path: String) async throws {
        for dark in [false, true] {
            let host = NSHostingView(rootView: view.frame(width: size.width, height: size.height)
                .background(Color(nsColor: .windowBackgroundColor)))
            let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
            window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
            window.contentView = host
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(400))
            let rep = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            NSAppearance(named: dark ? .darkAqua : .aqua)!.performAsCurrentDrawingAppearance {
                host.cacheDisplay(in: host.bounds, to: rep)
            }
            try #require(rep.representation(using: .png, properties: [:])).write(to: URL(fileURLWithPath: "\(path)-\(dark ? "dark" : "light").png"))
            window.contentView = nil
        }
    }

    /// The Glass Data view of a design glass, wide, and narrow with an
    /// offset and a wavelength outside the glass's data range.
    @Test func renderGlassData() async throws {
        guard let dir = ProcessInfo.processInfo.environment["MKLIGHT_RENDER_DIR"] else { return }
        let engine = try FFIEngine()
        let system = try #require(SampleSystems.bundled("double_gauss"))
        let snap = try await engine.openSnapshot(system)
        let context = AnalysisContext(system: system)
        let plain = try await engine.submit(
            EngineAnalysisRequest(analysis: "glass_info", settings: ["name": .string("N-SK16"), "curve_samples": .integer(200)]), on: snap
        ).result()
        try await host(try #require(GlassDataView(output: plain, context: context)), size: CGSize(width: 560, height: 820), to: "\(dir)/glass-data")
        let offset = try await engine.submit(
            EngineAnalysisRequest(
                analysis: "glass_info",
                settings: [
                    "name": .string("N-SK16"), "curve_samples": .integer(200),
                    "offset": .object(["dn_d": .number(0.001), "dv_d": .number(-0.5)]),
                    "wavelengths_um": .array([.number(0.4861327), .number(0.5875618), .number(0.6562725), .number(3.0)]),
                ]), on: snap
        ).result()
        try await host(try #require(GlassDataView(output: offset, context: context)), size: CGSize(width: 380, height: 900), to: "\(dir)/glass-data-narrow")
    }

    /// A large synthetic catalogue (every status), a selected glass, and
    /// image simulations of an imported picture and the text chart.
    @Test func renderDirect() async throws {
        guard let dir = ProcessInfo.processInfo.environment["MKLIGHT_RENDER_DIR"] else { return }
        var rng = SystemRandomNumberGenerator()
        let statuses: [GlassStatus] = [.preferred, .standard, .obsolete, .special, .melt, .unknown]
        var glasses: [GlassMapPoint] = (0..<1500).map { k in
            let vd = Double.random(in: 20...95, using: &rng)
            let nd = 1.45 + (95 - vd) / 75 * 0.35 + Double.random(in: -0.08...0.12, using: &rng)
            let pgf = GlassMapPoint.normalLine(vd) + Double.random(in: -0.01...0.03, using: &rng)
            return GlassMapPoint(
                id: "CAT/G\(k)", catalog: "CAT", name: "G\(k)", kind: .catalog, nd: nd, vd: vd, pgf: pgf,
                status: statuses[k % statuses.count], surfaces: [], excludeSubstitution: nil)
        }
        glasses[10].surfaces = [1, 4]
        glasses[11].surfaces = [2]
        glasses.append(
            GlassMapPoint(
                id: "model:M:6", catalog: nil, name: "1.70/40", kind: .model, nd: 1.70, vd: 40, pgf: GlassMapPoint.normalLine(40),
                status: .unknown, surfaces: [6], excludeSubstitution: nil))
        let data = GlassMapData(glasses: glasses)
        try write(
            GlassMapChart(glasses: data.glasses, highlighted: data.designHighlights, selection: .constant("CAT/G10"), plane: .ndVd).padding(),
            size: CGSize(width: 760, height: 480), to: "\(dir)/glass-chart-1500")
        try write(
            GlassMapChart(glasses: data.glasses, highlighted: data.designHighlights, selection: .constant(nil), plane: .pgfVd).padding(),
            size: CGSize(width: 760, height: 480), to: "\(dir)/glass-chart-1500-pgf")

        let engine = try FFIEngine()
        let snap = try await engine.openSnapshot(try #require(SampleSystems.bundled("double_gauss")))
        let gm = try await engine.submit(EngineAnalysisRequest(analysis: "glass_map"), on: snap).result()
        let view = try #require(GlassMapView(output: gm, context: AnalysisContext(system: nil), selection: "BUILTIN/N-SK16"))
        try write(view, size: CGSize(width: 760, height: 460), to: "\(dir)/glass-view-selected")
        try write(view, size: CGSize(width: 420, height: 620), to: "\(dir)/glass-view-narrow")

        // An imported picture: colour bands and a bright disc at the upper left.
        let w = 160, h = 120
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        for (k, c) in [(1.0, 0.2, 0.2), (0.2, 0.8, 0.3), (0.2, 0.4, 1.0)].enumerated() {
            ctx.setFillColor(red: c.0, green: c.1, blue: c.2, alpha: 1)
            ctx.fill(CGRect(x: 0, y: k * h / 3, width: w, height: h / 3))
        }
        ctx.setFillColor(red: 1, green: 1, blue: 0.9, alpha: 1)
        ctx.fillEllipse(in: CGRect(x: 20, y: 70, width: 40, height: 40))  // CG's y is up
        let image = try #require(ctx.makeImage())
        let lin = try #require(ImageSourceImport.linear(image, width: 96, height: 72))
        let cooke = try await engine.openSnapshot(try #require(SampleSystems.bundled("cooke_triplet")))
        let sim = try await engine.submit(
            EngineAnalysisRequest(
                analysis: "image_simulation",
                settings: ["source": lin.sourceJSON, "size": .array([.integer(96), .integer(72)]), "monochrome": .bool(false)]),
            on: cooke
        ).result()
        try write(try #require(ImageSimulationView(output: sim)), size: CGSize(width: 760, height: 380), to: "\(dir)/image-sim-imported")
        let text = try await engine.submit(
            EngineAnalysisRequest(
                analysis: "image_simulation",
                settings: [
                    "source": .object(["kind": .string("pattern"), "pattern": .string("text_chart")]),
                    "size": .array([.integer(128), .integer(128)]),
                ]),
            on: cooke
        ).result()
        try write(try #require(ImageSimulationView(output: text)), size: CGSize(width: 760, height: 420), to: "\(dir)/image-sim-text")
    }
}
