import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightFFI
@testable import MKLightModel
@testable import MKLightUI

/// Run-local spectra and the x / y first order on the real engine.
@Suite(.serialized, engineLinked) struct FFISpectrumFirstOrderTests {
    private func engine() throws -> FFIEngine {
        try FFIEngine(configuration: ["worker_threads": .integer(2)])
    }

    private func run(_ e: FFIEngine, _ analysis: String, _ settings: [String: JSONValue], on system: OpticalSystem) async throws -> AnalysisOutput {
        let snap = try await e.openSnapshot(system)
        return try await e.submit(EngineAnalysisRequest(analysis: analysis, settings: settings), on: snap).result()
    }

    /// `mtf` and `spot_diagram` accept a photopic and a blackbody spectrum;
    /// `run_info.settings` records it as sent (so the pane's request still
    /// matches the result and it does not look stale).
    @Test func analysesRunWithARunLocalSpectrum() async throws {
        let e = try engine()
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let spectra: [SpectrumSetting] = [
            .photopic(SpectralSampling(samples: 5)),
            .blackbody(temperatureK: 3200, sampling: SpectralSampling(samples: 4, lowerUm: 0.45, upperUm: 0.7, spacing: .uniform)),
        ]
        for analysis in ["mtf", "spot_diagram"] {
            for s in spectra {
                let json = try #require(s.json)
                let out = try await run(e, analysis, ["spectrum": json], on: system)
                let info = out.runInfo
                #expect(info.settings["spectrum"] == json, "\(analysis): run_info records the spectrum")
                #expect(EngineAnalysisRequest(analysis: analysis, settings: ["spectrum": json]).matches(info))
                #expect(info.warnings.allSatisfy { !$0.contains("error") })
            }
        }
        // The design snapshot is untouched: without a spectrum the run has
        // the design's three wavelengths.
        let plain = try await run(e, "system_summary", [:], on: system)
        #expect(plain.json["wavelengths_um"]?.nfArray?.count == system.wavelengths.wavelengths.count)
    }

    /// `system_summary` resolves a spectrum to the wavelengths the run
    /// uses (the control's preview), close to the local approximation.
    @Test func systemSummaryResolvesSpectra() async throws {
        let e = try engine()
        let system = try #require(SampleSystems.bundled("cooke_triplet"))
        let snap = try await e.openSnapshot(system)
        let photopic = SpectrumSetting.photopic(SpectralSampling(samples: 7))
        let resolved = try await SpectrumSetting.resolve(try #require(photopic.json), engine: e, snapshot: snap)
        #expect(resolved.count == 7)
        let local = try #require(photopic.previewSamples())
        for (a, b) in zip(resolved, local) { #expect(abs(a.um - b.um) < 0.01, "local preview near the engine's samples") }
        let bb = SpectrumSetting.blackbody(temperatureK: 5500, sampling: SpectralSampling(samples: 5, spacing: .uniform))
        let r2 = try await SpectrumSetting.resolve(try #require(bb.json), engine: e, snapshot: snap)
        let l2 = try #require(bb.previewSamples())
        #expect(r2.map(\.um) == l2.map(\.um) || zip(r2, l2).allSatisfy { abs($0.um - $1.um) < 1e-9 })
        let total = r2.reduce(0) { $0 + $1.weight }, ltotal = l2.reduce(0) { $0 + $1.weight }
        for (a, b) in zip(r2, l2) { #expect(abs(a.weight / total - b.weight / ltotal) < 2e-3) }
        let sensor = SpectrumSetting.sensor(
            table: [SpectrumSample(um: 0.4, weight: 0), SpectrumSample(um: 0.5, weight: 1), SpectrumSample(um: 0.6, weight: 0)],
            sampling: SpectralSampling(samples: 1, lowerUm: 0.4, upperUm: 0.6))
        let r3 = try await SpectrumSetting.resolve(try #require(sensor.json), engine: e, snapshot: snap)
        #expect(r3.count == 1 && abs(r3[0].um - 0.5) < 1e-9)
    }

    /// The anamorphic sample reports both sections and its squeeze; the
    /// Double Gauss is rotational.
    @Test func firstOrderReportsXYData() async throws {
        let e = try engine()
        let anamorphic = try #require(SampleSystems.bundled("anamorphic"))
        for analysis in ["first_order", "system_summary"] {
            let out = try await run(e, analysis, [:], on: anamorphic)
            let xy = try #require(ParaxialXYData(out.json), "\(analysis) has paraxial_xy")
            #expect(xy.symmetry != nil && xy.symmetry != .rotational, "\(analysis): \(String(describing: xy.symmetry))")
            #expect(xy.isAnamorphic == true)
            #expect(ParaxialXYData.isAnamorphic(out.json, system: nil))
            let p = try #require(xy.primary)
            #expect(p.eflX.isFinite && p.eflY.isFinite && p.squeeze.isFinite)
            #expect(abs(p.squeeze - p.eflY / p.eflX) < 1e-9 * abs(p.squeeze))
            #expect(abs(p.squeeze) != 1, "an anamorphic squeeze")
            #expect(xy.rows.count == anamorphic.wavelengths.wavelengths.count)
            #expect(p.wavelength == anamorphic.wavelengths.primary)
        }
        let dg = try #require(SampleSystems.bundled("double_gauss"))
        let out = try await run(e, "first_order", [:], on: dg)
        let xy = try #require(ParaxialXYData(out.json))
        #expect(xy.symmetry == .rotational && xy.isAnamorphic == false)
        let p = try #require(xy.primary)
        #expect(abs(p.eflX - p.eflY) < 1e-9 * abs(p.eflY) && abs(p.squeeze - 1) < 1e-9)
        #expect(!ParaxialXYData.isAnamorphic(out.json, system: dg))
        let efl = try #require(FirstOrderData(out.json)?.entry(["efl"]))
        #expect(abs(efl.value - p.eflY) < 1e-9 * abs(p.eflY), "EFL y is the legacy EFL")
    }
}
