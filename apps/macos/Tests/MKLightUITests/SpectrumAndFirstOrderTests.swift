import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The `spectrum` setting in the engine's format, the sensor CSV reader and
/// the x / y first-order reader.
@Suite struct SpectrumSettingTests {
    private func json(_ s: String) throws -> JSONValue { try JSONValue.parse(Data(s.utf8)) }

    /// Every kind writes exactly the engine's JSON
    /// (`mklight_analysis::spectrum::Spectrum`) and reads it back.
    @Test func everyKindRoundTripsTheEngineFormat() throws {
        let cases: [(SpectrumSetting, String)] = [
            (.photopic(.default), #"{"kind":"photopic"}"#),
            (.scotopic(SpectralSampling(samples: 15)), #"{"kind":"scotopic","sampling":{"samples":15}}"#),
            (
                .photopic(SpectralSampling(samples: 5, lowerUm: 0.4, upperUm: 0.7, spacing: .uniform)),
                #"{"kind":"photopic","sampling":{"samples":5,"range_um":[0.4,0.7],"spacing":"uniform"}}"#
            ),
            (.blackbody(temperatureK: 5500, sampling: .default), #"{"kind":"blackbody","temperature_k":5500}"#),
            (
                .blackbody(temperatureK: 3200.5, sampling: SpectralSampling(spacing: .uniform)),
                #"{"kind":"blackbody","temperature_k":3200.5,"sampling":{"spacing":"uniform"}}"#
            ),
            (
                .explicit([SpectrumSample(um: 0.55, weight: 1), SpectrumSample(um: 0.65, weight: 0.5)]),
                #"{"kind":"explicit","wavelengths":[{"um":0.55,"weight":1},{"um":0.65,"weight":0.5}]}"#
            ),
            (
                .sensor(table: [SpectrumSample(um: 0.4, weight: 0), SpectrumSample(um: 0.5, weight: 1)], sampling: SpectralSampling(samples: 3, lowerUm: 0.4, upperUm: 0.5)),
                #"{"kind":"sensor","table":[[0.4,0],[0.5,1]],"sampling":{"samples":3,"range_um":[0.4,0.5]}}"#
            ),
        ]
        for (setting, expected) in cases {
            let written = try #require(setting.json)
            #expect(written == (try json(expected)), "\(setting)")
            #expect(SpectrumSetting(json: written) == setting)
            #expect(SpectrumSetting(json: try json(expected)) == setting)
            #expect(setting.issues.isEmpty, "\(setting): \(setting.issues)")
        }
        #expect(SpectrumSetting.design.json == nil)
        #expect(SpectrumSetting(json: nil) == .design && SpectrumSetting(json: .null) == .design)
    }

    /// Defaults the engine fills in, and what it rejects structurally.
    @Test func readingFollowsTheEngine() throws {
        // A full default sampling reads as the default and is written short.
        let full = try json(#"{"kind":"photopic","sampling":{"samples":9,"range_um":[0.38,0.78],"spacing":"equal_weight"}}"#)
        #expect(SpectrumSetting(json: full) == .photopic(.default))
        #expect(SpectrumSetting(json: full)?.json == (try json(#"{"kind":"photopic"}"#)))
        // Explicit weights default to 1; integers read as numbers.
        #expect(SpectrumSetting(json: try json(#"{"kind":"explicit","wavelengths":[{"um":1}]}"#)) == .explicit([SpectrumSample(um: 1, weight: 1)]))
        #expect(SpectrumSetting(json: try json(#"{"kind":"blackbody","temperature_k":6500}"#)) == .blackbody(temperatureK: 6500, sampling: .default))
        // deny_unknown_fields, missing required keys, unknown kinds.
        for bad in [
            #"{"kind":"photopic","samples":5}"#, #"{"kind":"photopic","sampling":{"count":5}}"#, #"{"kind":"blackbody"}"#,
            #"{"kind":"explicit","wavelengths":[{"um":0.5,"w":1}]}"#, #"{"kind":"sensor","table":[[0.4,1,2]]}"#, #"{"kind":"laser"}"#,
            #"{"wavelengths":[]}"#, #"{"kind":"photopic","sampling":{"spacing":"log"}}"#, #""photopic""#,
        ] {
            #expect(SpectrumSetting(json: try json(bad)) == nil, "\(bad)")
        }
    }

    /// The engine's validation limits, in words, before a run.
    @Test func validationMirrorsTheEngine() {
        #expect(!SpectrumSetting.photopic(SpectralSampling(samples: 0)).issues.isEmpty)
        #expect(!SpectrumSetting.photopic(SpectralSampling(samples: 4097)).issues.isEmpty)
        #expect(SpectrumSetting.photopic(SpectralSampling(samples: 4096)).issues.isEmpty)
        #expect(!SpectrumSetting.photopic(SpectralSampling(lowerUm: 0.7, upperUm: 0.5)).issues.isEmpty)
        #expect(!SpectrumSetting.photopic(SpectralSampling(lowerUm: 0.3, upperUm: 0.7)).issues.isEmpty, "outside the CIE table")
        #expect(SpectrumSetting.photopic(SpectralSampling(lowerUm: 0.36, upperUm: 0.83)).issues.isEmpty)
        #expect(!SpectrumSetting.scotopic(SpectralSampling(lowerUm: 0.36, upperUm: 0.78)).issues.isEmpty)
        #expect(!SpectrumSetting.blackbody(temperatureK: 0, sampling: .default).issues.isEmpty)
        #expect(SpectrumSetting.blackbody(temperatureK: 300, sampling: SpectralSampling(lowerUm: 0.3, upperUm: 2)).issues.isEmpty)
        #expect(!SpectrumSetting.explicit([]).issues.isEmpty)
        #expect(!SpectrumSetting.explicit([SpectrumSample(um: -1, weight: 1)]).issues.isEmpty)
        #expect(!SpectrumSetting.explicit([SpectrumSample(um: 0.5, weight: -1)]).issues.isEmpty)
        let table = [SpectrumSample(um: 0.4, weight: 0.2), SpectrumSample(um: 0.6, weight: 1)]
        #expect(!SpectrumSetting.sensor(table: table, sampling: .default).issues.isEmpty, "default range outside the table")
        #expect(SpectrumSetting.sensor(table: table, sampling: SpectralSampling(lowerUm: 0.4, upperUm: 0.6)).issues.isEmpty)
        #expect(!SpectrumSetting.sensor(table: table.reversed(), sampling: SpectralSampling(lowerUm: 0.4, upperUm: 0.6)).issues.isEmpty)
    }

    @Test func summaries() {
        #expect(SpectrumSetting.photopic(.default).summary == "photopic")
        #expect(SpectrumSetting.scotopic(SpectralSampling(samples: 21)).summary == "scotopic ×21")
        #expect(SpectrumSetting.blackbody(temperatureK: 5500, sampling: .default).summary == "blackbody 5500 K")
        #expect(SpectrumSetting.explicit([SpectrumSample(um: 0.55, weight: 1)]).summary == "0.55 µm")
        #expect(SpectrumSetting.explicit([SpectrumSample(um: 0.5, weight: 1), SpectrumSample(um: 0.6, weight: 1)]).summary == "2 wavelengths")
    }

    /// The local preview follows the engine's sampling rules: equal energy
    /// gives equal weights, uniform spacing includes the end points, the
    /// blackbody peaks by Wien's law, the primary is nearest the mean.
    @Test func localPreviewSampling() throws {
        let eq = try #require(SpectrumSetting.photopic(.default).previewSamples())
        #expect(eq.count == 9)
        let w = eq.map(\.weight)
        #expect(w.max()! - w.min()! < 1e-6 * w.max()!)
        #expect(zip(eq, eq.dropFirst()).allSatisfy { $0.um < $1.um })
        let peakIndex = try #require(SpectrumSetting.primaryIndex(eq))
        #expect(abs(eq[peakIndex].um - 0.56) < 0.02)
        let uni = try #require(SpectrumSetting.scotopic(SpectralSampling(samples: 5, lowerUm: 0.4, upperUm: 0.6, spacing: .uniform)).previewSamples())
        #expect(uni.map(\.um) == [0.4, 0.45, 0.5, 0.55, 0.6].map { $0 })
        let bb = try #require(SpectrumSetting.blackbody(temperatureK: 5000, sampling: SpectralSampling(samples: 201, lowerUm: 0.3, upperUm: 1.2, spacing: .uniform)).previewSamples())
        let peak = bb.dropFirst().dropLast().max { $0.weight < $1.weight }!
        #expect(abs(peak.um - 2897.77 / 5000) < 0.006)
        // A triangular sensor response has its centroid at the apex.
        let tri = try #require(SpectrumSetting.sensor(
            table: [SpectrumSample(um: 0.4, weight: 0), SpectrumSample(um: 0.5, weight: 1), SpectrumSample(um: 0.6, weight: 0)],
            sampling: SpectralSampling(samples: 1, lowerUm: 0.4, upperUm: 0.6)).previewSamples())
        #expect(abs(tri[0].um - 0.5) < 1e-6)
        #expect(SpectrumSetting.primaryIndex([SpectrumSample(um: 0.4, weight: 1), SpectrumSample(um: 0.6, weight: 1)]) == 0, "first on ties")
    }

    /// `system_summary`'s wavelengths become the preview's samples.
    @Test func resolvedSamplesFromSystemSummary() throws {
        let r = try json(#"{"wavelengths_um":[0.5,0.6],"wavelength_weights":[0.25,0.75],"primary_wavelength":1}"#)
        #expect(SpectrumSetting.resolvedSamples(from: r) == [SpectrumSample(um: 0.5, weight: 0.25), SpectrumSample(um: 0.6, weight: 0.75)])
        #expect(SpectrumSetting.resolvedSamples(from: try json(#"{"wavelengths_um":[]}"#)) == nil)
    }
}

@Suite struct SpectrumCSVTests {
    @Test func nanometresWithHeaderAndCommas() {
        let r = SpectrumCSV.parse("wavelength,response\n400,0.1\n500,0.9\n600,0.4\n")
        #expect(r.unit == .nanometre && r.header == "wavelength,response")
        #expect(r.table == [SpectrumSample(um: 0.4, weight: 0.1), SpectrumSample(um: 0.5, weight: 0.9), SpectrumSample(um: 0.6, weight: 0.4)])
        #expect(r.badRows.isEmpty && r.isUsable && r.rangeUm == 0.4...0.6)
    }

    @Test func micrometresWithoutHeader() {
        let r = SpectrumCSV.parse("0.45, 0.2\r\n0.55, 1\r\n0.65, 0.3")
        #expect(r.unit == .micrometre && r.header == nil)
        #expect(r.table.map(\.um) == [0.45, 0.55, 0.65])
    }

    @Test func semicolonsAndTabsWithDecimalCommas() {
        let s = SpectrumCSV.parse("λ (nm);QE\n450;0,25\n550;0,75\n")
        #expect(s.unit == .nanometre && s.table == [SpectrumSample(um: 0.45, weight: 0.25), SpectrumSample(um: 0.55, weight: 0.75)])
        let t = SpectrumCSV.parse("0.4\t0.5\n0.5\t1.0\n0.6\t0,5")
        #expect(t.unit == .micrometre && t.table.map(\.weight) == [0.5, 1, 0.5])
    }

    /// A header that names the unit wins over the magnitude; words that
    /// merely contain the letters do not count.
    @Test func headerNamesTheUnit() {
        #expect(SpectrumCSV.parse("lambda_um,r\n1.0,1\n1.5,2").unit == .micrometre)
        #expect(SpectrumCSV.parse("wavelength [nm],r\n40,1\n45,2").table.map(\.um) == [0.04, 0.045])
        #expect(SpectrumCSV.parse("Spectrum,Response\n400,1\n500,2").unit == .nanometre, "'Spectrum' is not µm")
        #expect(SpectrumCSV.parse("Wavelength (µm),QE\n400,1\n500,2").unit == .micrometre)
    }

    /// Unreadable rows are reported with their line numbers and skipped;
    /// the table is sorted; repeated wavelengths are reported.
    @Test func badRowsAreReported() {
        let text = """
            # sensor QE
            nm,qe
            600,0.5
            500,abc
            400,0.1
            450
            700,-0.2
            -5,1
            400,0.3
            550,0.9
            """
        let r = SpectrumCSV.parse(text)
        #expect(r.unit == .nanometre)
        #expect(r.table.map(\.um) == [0.4, 0.55, 0.6])
        #expect(r.badRows.map(\.line) == [4, 6, 7, 8, 9])
        #expect(r.badRows.map(\.reason) == ["not a number", "needs two columns", "response must not be negative", "wavelength must be positive", "repeats a wavelength"])
    }

    @Test func emptyOrTooShort() {
        #expect(!SpectrumCSV.parse("").isUsable)
        #expect(!SpectrumCSV.parse("x,y\n500,1").isUsable)
        #expect(SpectrumCSV.parse("x,y\n500,1").rangeUm == nil)
    }
}

@Suite @MainActor struct SpectrumSettingsBarTests {
    private let descriptor = AnalysisDescriptor(
        name: "spot_diagram", title: "Spot", category: "geometric", available: true,
        settingsSchema: .object([
            "properties": .object([
                "spectrum": .object(["type": .array([.string("object"), .string("null")]), "description": .string("Optional run-local spectrum")]),
                "rings": .object(["type": .string("integer"), "default": .integer(6)]),
            ])
        ]))

    @Test func spectrumHasItsOwnControlAndSummary() throws {
        let pane = AnalysisPane(analysis: "spot_diagram")
        let bar = SettingsBar(pane: pane, descriptor: descriptor, system: SampleSystems.cookeTriplet())
        let field = try #require(descriptor.settingsFields.first { $0.name == "spectrum" })
        #expect(bar.control(of: field) == .spectrum)
        #expect(SettingsBar.summary(pane, descriptor) == "")
        pane.set("spectrum", SpectrumSetting.photopic(.default).json)
        #expect(SettingsBar.summary(pane, descriptor) == "Spectrum photopic")
        pane.set("spectrum", SpectrumSetting.blackbody(temperatureK: 3200, sampling: .default).json)
        #expect(SettingsBar.summary(pane, descriptor) == "Spectrum blackbody 3200 K")
        pane.set("spectrum", .object(["kind": .string("laser")]))
        #expect(SettingsBar.summary(pane, descriptor) == "Spectrum custom")
        // The control writes nil for the design's wavelengths.
        let binding = bar.spectrumBinding(field)
        binding.wrappedValue = nil
        #expect(pane.settings["spectrum"] == nil && SettingsBar.summary(pane, descriptor) == "")
        #expect(bar.systemSpectrum.count == 3)
    }
}

@Suite struct ParaxialXYReaderTests {
    private let anamorphic = """
        {"entries":[{"key":"efl","label":"Effective focal length","value":99.5,"unit":"mm"}],
         "paraxial_xy":{"primary_wavelength":1,"squeeze_definition":"signed EFL_y / EFL_x","symmetry":"meridional",
           "warnings":["mixed xy power"],
           "wavelengths":[
             {"wavelength":0,"wavelength_um":0.4861,"efl_x":49.2,"efl_y":98.4,"bfl_x":95.6,"bfl_y":95.6,"squeeze":2,"doe_power":[]},
             {"wavelength":1,"wavelength_um":0.5876,"efl_x":49.75,"efl_y":99.5,"bfl_x":96.7,"bfl_y":96.6,"squeeze":2,"doe_power":[{"surface":2,"power":0.00057}]},
             {"wavelength":2,"wavelength_um":0.6563,"efl_x":"inf","efl_y":100,"bfl_x":null,"bfl_y":97,"squeeze":0,"doe_power":[]}]}}
        """

    @Test func readsEveryWavelengthPrimaryFirst() throws {
        let xy = try #require(ParaxialXYData(try JSONValue.parse(Data(anamorphic.utf8))))
        #expect(xy.symmetry == .meridional && xy.isAnamorphic == true)
        #expect(xy.squeezeDefinition == "signed EFL_y / EFL_x" && xy.warnings == ["mixed xy power"])
        let p = try #require(xy.primary)
        #expect(p.wavelength == 1 && p.eflX == 49.75 && p.eflY == 99.5 && p.bflX == 96.7 && p.bflY == 96.6 && p.squeeze == 2)
        #expect(p.doePower == [ParaxialXYData.DOEPower(surface: 2, power: 0.00057)] && xy.hasDOEs)
        #expect(xy.others.map(\.wavelength) == [0, 2])
        let last = xy.rows[2]
        #expect(last.eflX == .infinity && last.bflX.isNaN, "nf values")
        #expect(ParaxialXYData.squeezeText(2) == "2.0000×" && ParaxialXYData.squeezeText(-1.5, digits: 2) == "-1.50×")
    }

    /// The engine's symmetry decides; the surface kinds only when it does
    /// not report one.
    @Test func symmetryDecidesAnamorphic() throws {
        var toroid = SampleSystems.singlet()
        toroid.setShapeKind(.toroid, at: 1)
        let rotational: JSONValue = .object(["paraxial_xy": .object(["symmetry": .string("rotational"), "wavelengths": .array([])])])
        #expect(!ParaxialXYData.isAnamorphic(rotational, system: toroid), "the engine says rotational")
        let none: JSONValue = .object(["paraxial_xy": .object(["symmetry": .string("none")])])
        #expect(ParaxialXYData.isAnamorphic(none, system: SampleSystems.singlet()))
        // system_summary has the class at the top level too.
        #expect(ParaxialXYData.isAnamorphic(.object(["symmetry": .string("meridional")]), system: nil))
        // An engine without paraxial_xy: the old heuristic.
        let old: JSONValue = .object(["entries": .array([])])
        #expect(ParaxialXYData(old) == nil)
        #expect(ParaxialXYData.isAnamorphic(old, system: toroid))
        #expect(!ParaxialXYData.isAnamorphic(old, system: SampleSystems.singlet()))
        #expect(!ParaxialXYData.isAnamorphic(nil, system: nil))
    }
}
