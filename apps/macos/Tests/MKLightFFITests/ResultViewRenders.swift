import AppKit
import Foundation
import SwiftUI
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// Development aid: renders analysis panes from real engine results to
/// PNGs (light and dark) for visual review. Runs only when
/// `MKLIGHT_RENDER_DIR` names an output directory.
@Suite(.serialized, engineLinked) @MainActor struct ResultViewRenders {
    @Test func renderPanes() async throws {
        guard let dir = ProcessInfo.processInfo.environment["MKLIGHT_RENDER_DIR"] else { return }
        var cases: [(name: String, sample: String, analysis: String, settings: [String: JSONValue], size: CGSize)] = [
            ("pupil", "cooke_triplet", "pupil_aberration", ["grid": .array([.integer(9), .integer(9)])], CGSize(width: 560, height: 460)),
            ("pupil-sweep", "cooke_triplet", "pupil_aberration", ["mode": .string("y"), "grid": .array([.integer(11), .integer(11)])], CGSize(width: 520, height: 380)),
            ("ri-map", "double_gauss", "relative_illumination_map", ["grid": .array([.integer(15), .integer(15)])], CGSize(width: 820, height: 460)),
            ("ri-map-narrow", "double_gauss", "relative_illumination_map", [:], CGSize(width: 420, height: 640)),
            (
                "cra", "cooke_triplet", "chief_ray_angle",
                ["grid": .array([.integer(9), .integer(9)]), "sensor_limit": .array([.array([.number(0), .number(5)]), .array([.number(20), .number(24)])])],
                CGSize(width: 620, height: 420)
            ),
            ("cra-sweep", "double_gauss", "chief_ray_angle", ["mode": .string("y"), "grid": .array([.integer(21), .integer(21)]), "sensor_limit": .array([.array([.number(0), .number(12)])])], CGSize(width: 620, height: 420)),
            ("cra-anamorphic", "anamorphic", "chief_ray_angle", ["grid": .array([.integer(9), .integer(9)])], CGSize(width: 560, height: 460)),
            ("glass-map", "double_gauss", "glass_map", [:], CGSize(width: 820, height: 480)),
            ("image-sim", "cooke_triplet", "image_simulation", ["source": .object(["kind": .string("pattern"), "pattern": .string("siemens_star")]), "size": .array([.integer(96), .integer(96)])], CGSize(width: 760, height: 460)),
        ]
        // The star on a small patch (finer pixels than the blur), as the
        // pane starts (10 %) and at 2 %, at 256 px.
        for (name, extent) in [("image-sim-star-10", 0.1), ("image-sim-star-2", 0.02)] {
            var settings = ImageSimulationSettings.starting(defaults: nil)
            settings["optics"] = .object(["field_extent": .number(extent)])
            cases.append((name, "cooke_triplet", "image_simulation", settings, CGSize(width: 900, height: 560)))
        }
        // A user image, as the source menu or a drop imports it.
        let picture = ProcessInfo.processInfo.environment["MKLIGHT_RENDER_IMAGE"] ?? "/Library/User Pictures/Animals/Owl.heic"
        if let image = ImageSourceImport.decode(URL(fileURLWithPath: picture)),
            let lin = ImageSourceImport.linear(image, width: 256, height: 256)
        {
            var settings = ImageSimulationSettings.starting(defaults: nil)
            settings["source"] = lin.sourceJSON
            settings["monochrome"] = .bool(false)
            settings[ImageSourceImport.fileKey] = .string(picture)
            cases.append(("image-sim-imported-pane", "cooke_triplet", "image_simulation", settings, CGSize(width: 900, height: 560)))
        }
        let engine = try FFIEngine(configuration: ["worker_threads": .integer(4)])
        for c in cases {
            let pane = AnalysisPane(analysis: c.analysis, settings: c.settings)
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
                    .environment(\.colorScheme, dark ? .dark : .light)
                let appearance = NSAppearance(named: dark ? .darkAqua : .aqua)!
                var png: Data?
                appearance.performAsCurrentDrawingAppearance {
                    let r = ImageRenderer(content: view)
                    r.scale = 2
                    if let img = r.nsImage, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) {
                        png = rep.representation(using: .png, properties: [:])
                    }
                }
                try #require(png).write(to: URL(fileURLWithPath: "\(dir)/\(c.name)-\(dark ? "dark" : "light").png"))
            }
        }
    }
}
