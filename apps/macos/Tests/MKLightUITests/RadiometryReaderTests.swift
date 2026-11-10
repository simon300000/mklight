import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The ghost and stray-light summaries on results recorded from the engine
/// (Double Gauss sample, CPU reference): a ratio or a standard error the
/// engine reports as `null` is undefined and reads "—", never a number.
@Suite struct RadiometryReaderTests {
    static func json(_ text: String) throws -> JSONValue { try JSONValue.parse(Data(text.utf8)) }

    /// `ghost_irradiance` of a source at 1.5 field heights (`points`): its
    /// image misses the map. The first four and the last two of 45 ghosts.
    static let ghostsOfAnOutOfFieldSource = """
        {"map": {"bins": [16, 16], "half_size": [12.312675462838424, 12.312675462838424]},
         "total_flux_ratio": null, "total_peak_ratio": null,
         "ghosts": [
          {"label": "R5·R3", "flux": 0.0006546073014556638, "flux_ratio": null, "peak_ratio": null},
          {"label": "R2·R1", "flux": 0.0001380118554636203, "flux_ratio": null, "peak_ratio": null},
          {"label": "R3·R2", "flux": 0.00013079453868060554, "flux_ratio": null, "peak_ratio": null},
          {"label": "R3·R1", "flux": 0.00023928810812790617, "flux_ratio": null, "peak_ratio": null},
          {"label": "R11·R8", "flux": 0.0, "flux_ratio": 0.0, "peak_ratio": 0.0},
          {"label": "R11·R10", "flux": 0.0, "flux_ratio": 0.0, "peak_ratio": 0.0}]}
        """

    /// The same source together with the axial field: the first three
    /// ghosts with an undefined peak ratio and the first three with one.
    static let ghostsOfAFieldAndAnOutOfFieldSource = """
        {"map": {"bins": [16, 16], "half_size": [12.312675462838424, 12.312675462838424]},
         "total_flux_ratio": 0.020016196269044377, "total_peak_ratio": 0.0014953683585082508,
         "ghosts": [
          {"label": "R11·R9", "flux": 0.0008163000028750996, "flux_ratio": 0.0025205033257534606, "peak_ratio": null},
          {"label": "R5·R3", "flux": 0.0007849933627651981, "flux_ratio": 0.0024238372835664592, "peak_ratio": null},
          {"label": "R9·R7", "flux": 0.0007113220007965862, "flux_ratio": 0.002196360973140567, "peak_ratio": null},
          {"label": "R10·R2", "flux": 0.00022255112355024878, "flux_ratio": 0.0006871748684097454, "peak_ratio": 0.00015343803533521096},
          {"label": "R11·R10", "flux": 0.00018206880893173027, "flux_ratio": 0.0005621769408453789, "peak_ratio": 6.100827826981511e-05},
          {"label": "R7·R5", "flux": 0.0006801400648613002, "flux_ratio": 0.002100079982142769, "peak_ratio": 6.092662159094099e-05}]}
        """

    /// `stray_light` with one source ray: no standard errors.
    static let strayLightOfOneRay = """
        {"detector": {"half_x": 12.928309235980345, "half_y": 12.928309235980345, "nx": 4, "ny": 4},
         "stray_flux": 0.0, "stray_std_error": null, "stray_fraction": 0.0,
         "totals": [
          {"class": "primary", "flux": 617.8788124708939, "std_error": null, "hits": 1,
           "moments": {"centroid": [-0.011605258751726577, -0.017321448861464805], "centroid_std_error": [null, null],
                       "flux": 617.8788124708939, "rms_radius": 0.0, "rms_radius_std_error": null}},
          {"class": "ghost", "flux": 0.0, "std_error": null, "hits": 0},
          {"class": "scatter", "flux": 0.0, "std_error": null, "hits": 0},
          {"class": "mechanical", "flux": 0.0, "std_error": null, "hits": 0},
          {"class": "sneak", "flux": 0.0, "std_error": null, "hits": 0}]}
        """

    /// `stray_light` with 2000 source rays.
    static let strayLight = """
        {"detector": {"half_x": 12.928309235980345, "half_y": 12.928309235980345, "nx": 4, "ny": 4},
         "stray_flux": 3.5305640525407402, "stray_std_error": 1.0380818453363327, "stray_fraction": 0.007193265606646787,
         "totals": [
          {"class": "primary", "flux": 312.6464601657003, "std_error": 6.9093204407015785, "hits": 1012},
          {"class": "ghost", "flux": 3.4487560512934268, "std_error": 1.0375864100887917, "hits": 11},
          {"class": "scatter", "flux": 0.08180800124731326, "std_error": 0.03620270975734429, "hits": 6},
          {"class": "mechanical", "flux": 0.0, "std_error": 0.0, "hits": 0},
          {"class": "sneak", "flux": 0.0, "std_error": 0.0, "hits": 0}]}
        """

    /// No row shows a non-number as a number.
    static func expectNoNonNumbers(_ rows: [(String, String)]) {
        for (label, value) in rows {
            let v = value.lowercased()
            #expect(!v.contains("nan") && !v.contains("inf") && !v.contains("∞") && !v.contains("null"), "\(label): \(value)")
        }
    }

    @Test func undefinedGhostRatiosReadAsDashes() throws {
        let s = GhostIrradianceSummary(try Self.json(Self.ghostsOfAnOutOfFieldSource))
        #expect(s.totalFluxRatio == nil && s.totalPeakRatio == nil)
        #expect(s.ghosts.map(\.peakRatio) == [nil, nil, nil, nil, 0, 0])
        let rows = s.rows()
        #expect(rows.map(\.0) == ["Ghost / primary flux", "Ghost / primary peak", "R5·R3", "R2·R1", "R3·R2", "R3·R1", "R11·R8", "R11·R10"])
        // Undefined is "—"; a ghost without light has the ratio 0.
        #expect(rows.map(\.1) == ["—", "—", "—", "—", "—", "—", "0", "0"])
        Self.expectNoNonNumbers(rows)
    }

    @Test func ghostsKeepTheEngineRankWhenSomeRatiosAreUndefined() throws {
        let s = GhostIrradianceSummary(try Self.json(Self.ghostsOfAFieldAndAnOutOfFieldSource))
        #expect(s.totalFluxRatio == 0.020016196269044377)
        let rows = s.rows(limit: 5)
        #expect(rows.count == 7)
        #expect(rows.map(\.0) == ["Ghost / primary flux", "Ghost / primary peak", "R11·R9", "R5·R3", "R9·R7", "R10·R2", "R11·R10"])
        #expect(rows.map(\.1) == ["0.0200162", "0.00149537", "—", "—", "—", "1.534e-04", "6.101e-05"])
        Self.expectNoNonNumbers(rows)
    }

    @Test func undefinedStrayErrorsReadAsDashes() throws {
        let s = StrayLightSummary(try Self.json(Self.strayLightOfOneRay))
        #expect(s.flux == 0 && s.stdError == nil && s.fraction == 0)
        #expect(s.totals.map(\.stdError) == [nil, nil, nil, nil, nil])
        let rows = s.rows
        #expect(rows.map(\.0) == ["Stray flux", "± standard error", "Stray fraction", "Primary", "Ghost", "Scatter", "Mechanical", "Sneak"])
        // The undefined error has no unit either; zero flux stays a number.
        #expect(rows.map(\.1) == ["0 W", "—", "0", "617.879 W", "0 W", "0 W", "0 W", "0 W"])
        Self.expectNoNonNumbers(rows)
    }

    @Test func definedStrayErrorsReadAsNumbers() throws {
        let s = StrayLightSummary(try Self.json(Self.strayLight))
        #expect(s.stdError == 1.0380818453363327)
        #expect(s.totals.map(\.name) == ["primary", "ghost", "scatter", "mechanical", "sneak"])
        #expect(s.totals[1].stdError == 1.0375864100887917)
        #expect(s.rows.map(\.1) == ["3.53056 W", "1.03808 W", "0.00719327", "312.646 W", "3.44876 W", "0.081808 W", "0 W", "0 W"])
    }

    /// A summary without the keys (a failed or foreign result) is all
    /// dashes, and a missing reference flux leaves the fraction out.
    @Test func missingValuesAreUndefined() throws {
        let ghost = GhostIrradianceSummary(.object([:]))
        #expect(ghost.rows().map(\.1) == ["—", "—"])
        let stray = StrayLightSummary(try Self.json(#"{"stray_flux": 2.5, "stray_std_error": "inf", "stray_fraction": null}"#))
        #expect(stray.rows.map(\.0) == ["Stray flux", "± standard error"])
        #expect(stray.rows.map(\.1) == ["2.5 W", "—"])
    }

    /// The views show the readers' rows beside their maps.
    @Test @MainActor func viewsShowTheSummaries() throws {
        let map = ResultBuffer(name: "total_ghost_map", doubles: Array(repeating: 0.5, count: 256), shape: [16, 16])
        let ghost = MapReaderTests.output("ghost_irradiance", json: try Self.json(Self.ghostsOfAnOutOfFieldSource), buffers: [map])
        let ghostRows = try #require(GhostIrradianceView(output: ghost)).map.rows
        #expect(ghostRows.map(\.1) == ["—", "—", "—", "—", "—", "—", "0", "0"])

        let total = ResultBuffer(name: "total_irradiance", doubles: Array(repeating: 0, count: 16), shape: [4, 4])
        let stray = MapReaderTests.output("stray_light", json: try Self.json(Self.strayLightOfOneRay), buffers: [total])
        let strayRows = try #require(StrayLightView(output: stray)).map.rows
        #expect(strayRows.map(\.1) == ["0 W", "—", "0", "617.879 W", "0 W", "0 W", "0 W", "0 W"])
    }
}
