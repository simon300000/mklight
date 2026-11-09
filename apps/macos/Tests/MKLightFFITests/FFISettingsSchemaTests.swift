import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// The settings schemas of the real engine, through the control mapping.
@Suite(engineLinked) @MainActor struct FFISettingsSchemaTests {
    /// Every setting of every analysis the engine offers gets a control or
    /// is deliberately hidden; none falls back to JSON. Runs on the real
    /// engine's schemas.
    @Test func everyEngineSettingHasAControl() async throws {
        let w = Workspace(engine: try FFIEngine(), panes: [])
        await w.loadEngineMetadata()
        try #require(!w.paneAnalyses.isEmpty, "the engine lists analyses")
        var hidden: [String] = []
        for d in w.paneAnalyses where d.available {
            let bar = SettingsBar(pane: AnalysisPane(analysis: d.name), descriptor: d, system: SampleSystems.cookeTriplet())
            for f in d.settingsFields {
                let c = bar.control(of: f)
                if c == .hidden { hidden.append("\(d.name).\(f.name)") }
                // Groups and variants have fields to show.
                if case .group = c { #expect(!f.properties.isEmpty, "\(d.name).\(f.name)") }
                if case .variants = c { #expect(!f.variants.isEmpty, "\(d.name).\(f.name)") }
            }
        }
        // The only hidden settings are those with no sensible control or no
        // meaning in the app: programmatic hooks, availability overrides and
        // explicit path lists (automatic by default).
        let allowed: Set<String> = [
            "glass_map.metadata", "ghost_paths.paths", "ghost_irradiance.paths", "ghost_irradiance.points", "stray_light.ghost_partition",
        ]
        let unexpected = hidden.filter { h in !allowed.contains(h) && !["configuration", "seed", "delay_ms", "chunk"].contains(h.split(separator: ".").last.map(String.init) ?? "") }
        #expect(unexpected.isEmpty, "hidden without reason: \(unexpected)")
        // The structured settings of the audit have their controls.
        func control(_ a: String, _ k: String) -> SettingControl? {
            guard let d = w.descriptor(a), let f = d.settingsFields.first(where: { $0.name == k }) else { return nil }
            return SettingsBar(pane: AnalysisPane(analysis: a), descriptor: d, system: nil).control(of: f)
        }
        #expect(control("spot_diagram", "through_focus") == .throughFocus)
        #expect(control("mtf", "detector") == .group(nullable: true))
        #expect(control("mtf", "huygens") == .group(nullable: false))
        #expect(control("best_focus", "range") == .pair(integer: false, nullable: true))
        #expect(control("system_transmission", "pupil") == .variants(nullable: false))
        #expect(control("system_transmission", "model") == .variants(nullable: false))
        #expect(control("system_transmission", "wavelengths") == .indexList(.wavelengths))
        #expect(control("polarization_pupil_map", "state") == .variants(nullable: false))
        #expect(control("diffraction_encircled_energy", "center") == .variants(nullable: false))
        #expect(control("stray_light", "source") == .variants(nullable: false))
        #expect(control("ghost_paths", "clip") == .variants(nullable: false))
        #expect(control("glass_info", "offset") == .group(nullable: false))
        if case .table = control("system_transmission", "fields") {} else { Issue.record("transmission field points are a table") }
    }
}
