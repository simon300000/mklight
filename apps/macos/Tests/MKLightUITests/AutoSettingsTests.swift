import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The engine's system-aware defaults in the app: automatic settings shown
/// as "Auto (n)" with the value the run used, warning fixes that go back to
/// automatic sampling, the image simulation's patch centre, and maps with
/// no data outside the field.
@Suite @MainActor struct AutoSettingsTests {
    private func output(_ analysis: String, _ json: JSONValue, _ buffers: [ResultBuffer] = []) -> AnalysisOutput {
        MapReaderTests.output(analysis, json: json, buffers: buffers)
    }

    @Test func automaticSettingsAreKnownBySentinel() {
        #expect(AutoSettings.sentinel("grid_size", analysis: "psf") == .zero)
        #expect(AutoSettings.sentinel("grid_size", analysis: "grid_distortion") == nil, "a grid of field points is never automatic")
        #expect(AutoSettings.sentinel("pupil.grid", analysis: "mtf") == .zero)
        #expect(AutoSettings.sentinel("max_frequency", analysis: "mtf") == .null)
        #expect(AutoSettings.sentinel("frequencies", analysis: "through_focus_mtf") == .empty)
        #expect(AutoSettings.sentinel("optics.psf_size", analysis: "image_simulation") == .zero)
        #expect(AutoSettings.isAuto(.integer(0), .zero) && AutoSettings.isAuto(nil, .zero) && !AutoSettings.isAuto(.integer(128), .zero))
        #expect(AutoSettings.isAuto(.null, .null) && !AutoSettings.isAuto(.number(50), .null))
        #expect(AutoSettings.isAuto(.array([]), .empty) && !AutoSettings.isAuto(.array([.number(10)]), .empty))
    }

    /// "Auto (n)" with the value the engine used, read from each result;
    /// never a bare 0.
    @Test func autoShowsTheValueTheRunUsed() {
        #expect(AutoSettings.display(used: nil) == "Auto" && AutoSettings.display(used: "128") == "Auto (128)")
        let psf = output("psf", .object(["psfs": .array([
            .object(["psf": .object(["size": .integer(282), "spacing_mm": .number(0.000387)])]),
            .object(["psf": .object(["size": .integer(371), "spacing_mm": .number(0.000393)])]),
        ])]))
        #expect(AutoSettings.used("grid_size", analysis: "psf", output: psf) == "282–371")
        #expect(AutoSettings.used("spacing_mm", analysis: "psf", output: psf) == "0.387–0.393 µm")
        let mtf = output("mtf", .object(["frequencies": .string("frequencies")]), [ResultBuffer(name: "frequencies", doubles: [0, 26, 52])])
        #expect(AutoSettings.used("max_frequency", analysis: "mtf", output: mtf) == "52")
        let tf = output(
            "through_focus_mtf", .object(["frequencies": .string("frequencies"), "focus_mm": .string("focus_mm")]),
            [ResultBuffer(name: "frequencies", doubles: [2, 5, 10]), ResultBuffer(name: "focus_mm", doubles: [-0.18, 0, 0.18])])
        #expect(AutoSettings.used("frequencies", analysis: "through_focus_mtf", output: tf) == "2, 5, 10")
        #expect(AutoSettings.used("focus_range_mm", analysis: "through_focus_mtf", output: tf) == "±0.18")
        let wf = output("wavefront_map", .object(["maps": .array([.object(["grid": .integer(128)]), .object(["grid": .integer(64)])])]))
        #expect(AutoSettings.used("pupil.grid", analysis: "wavefront_map", output: wf) == "64–128")
        let sim = output("image_simulation", .object(["psf_size": .integer(455), "psf_spacing_mm": .number(0.000386)]))
        #expect(AutoSettings.used("optics.psf_size", analysis: "image_simulation", output: sim) == "455")
        #expect(AutoSettings.used("optics.psf_spacing_mm", analysis: "image_simulation", output: sim) == "0.386 µm")
        #expect(AutoSettings.used("grid_size", analysis: "psf", output: nil) == nil)
        // The number field shows no digits for an automatic value.
        #expect(SettingNumberField.display(.integer(0), isInteger: true, auto: .zero) == "")
        #expect(SettingNumberField.display(.integer(0), isInteger: true, auto: nil) == "0")
        #expect(SettingNumberField.display(.integer(256), isInteger: true, auto: .zero) == "256")
        #expect(FrequencyListSetting.parse("10, 20 5") == [5, 10, 20] && FrequencyListSetting.parse("10, x") == nil)
    }

    /// Warning fixes send explicit coarse settings back to automatic; an
    /// automatic setting that still warns gets no button.
    @Test func warningFixesGoBackToAutomatic() {
        func settings(_ s: [String: JSONValue]) -> (String) -> JSONValue? { { s[$0] } }
        let pupil = RunDetails.remedy(.pupilSampling, settings: settings(["pupil": .object(["grid": .integer(64), "rays": .integer(65)])]))
        #expect(pupil?.key == "pupil" && pupil?.value["grid"]?.int == 0 && pupil?.value["rays"]?.int == 65)
        #expect(pupil?.title == "Sample the Pupil Automatically (grid 64 → auto)")
        #expect(RunDetails.remedy(.pupilSampling, settings: settings(["pupil": .object(["grid": .integer(0)])])) == nil)
        let grid = RunDetails.remedy(.imageGrid, settings: settings(["grid_size": .integer(128)]))
        #expect(grid?.key == "grid_size" && grid?.value == .integer(0))
        #expect(RunDetails.remedy(.imageGrid, settings: settings(["grid_size": .integer(0)])) == nil)
        let sim = RunDetails.remedy(.imageGrid, settings: settings(["optics": .object(["psf_size": .integer(33), "field_extent": .number(0.1)])]))
        #expect(sim?.key == "optics" && sim?.value["psf_size"]?.int == 0 && sim?.value["field_extent"]?.doubleValue == 0.1)
    }

    /// The patch centre is `optics.field_center: [hx, hy]`; the design's
    /// field points are offered normalised by the largest radial field.
    @Test func fieldCentreSettingsEncoding() throws {
        #expect(ImageSimulationSettings.centerJSON(SIMD2(0.5, -0.70710678)) == .array([.number(0.5), .number(-0.7071)]))
        #expect(ImageSimulationSettings.centerJSON(SIMD2(-0.00001, 0)) == .array([.number(0), .number(0)]))
        let dg = try #require(SampleSystems.bundled("double_gauss"))
        let pts = ImageSimulationSettings.fieldPoints(dg)
        #expect(pts.first?.field == .zero && pts.first?.title == "On axis")
        #expect(pts.last?.field == SIMD2(0, 1) && pts.last?.title.hasPrefix("Field \(dg.fields.fields.count)") == true)
        #expect(ImageSimulationSettings.centerTitle(SIMD2(0, 1), points: pts) == pts.last?.title)
        #expect(ImageSimulationSettings.centerTitle(SIMD2(0.3, -0.45), points: pts) == "(+0.30, −0.45)")
        #expect(ImageSimulationSettings.snap(SIMD2(0.02, 0.99), to: pts.map(\.field), within: 0.06) == SIMD2(0, 1))
        #expect(ImageSimulationSettings.snap(SIMD2(0.3, 0.3), to: pts.map(\.field), within: 0.06) == SIMD2(0.3, 0.3))
        #expect(ImageSimulationSettings.outsideField(center: SIMD2(0, 1), extent: 0.1))
        #expect(!ImageSimulationSettings.outsideField(center: SIMD2(0.5, 0.5), extent: 0.1))
        // The pane writes it inside `optics`, keeping the other keys.
        let w = Workspace(engine: MockEngine(simulatedLatency: .zero))
        let pane = w.addPane("image_simulation", settings: ["optics": .object(["field_extent": .number(0.1), "psf_size": .integer(0)])])
        ImageSimulationSettings.setFieldCenter(SIMD2(0.5, 0.5), in: pane)
        #expect(pane.settings["optics"]?["field_center"] == .array([.number(0.5), .number(0.5)]))
        #expect(pane.settings["optics"]?["field_extent"]?.doubleValue == 0.1 && ImageSimulationSettings.fieldCenter(pane) == SIMD2(0.5, 0.5))
        // A run's patch: the middle and half-span of its mapping nodes.
        let nodes: [JSONValue] = [[0.4, 0.4], [0.6, 0.4], [0.4, 0.6], [0.6, 0.6]].map { f in
            .object(["field": .array(f.map(JSONValue.number)), "ideal_mm": .array([.number(1), .number(1)]), "real_mm": .array([.number(1), .number(1)])])
        }
        let data = try #require(ImageSimulationData(output: output(
            "image_simulation", .object(["image": .string("image"), "field_mapping": .array(nodes), "psf_size": .integer(455)]),
            [ResultBuffer(name: "image", doubles: [Double](repeating: 0.5, count: 4), shape: [2, 2, 1])])))
        let c = try #require(ImageSimulationSettings.fieldCenter(fromRun: data))
        #expect(abs(c.x - 0.5) < 1e-12 && abs(c.y - 0.5) < 1e-12)
        #expect(abs((ImageSimulationSettings.fieldExtent(fromRun: data) ?? 0) - 0.1) < 1e-12 && data.psfSize == 455)
    }

    /// New panes start on a visible patch with the engine's automatic PSF
    /// and pupil.
    @Test func imageSimulationStartsWithAutomaticSampling() {
        let defaults: JSONValue = .object(["optics": .object([
            "field_extent": .number(0.7071), "psf_size": .integer(0), "field_center": .array([.number(0), .number(0)]),
            "pupil": .object(["grid": .integer(0), "rays": .integer(65), "amplitude": .string("radiometric")]),
        ])])
        let s = ImageSimulationSettings.starting(defaults: defaults)
        #expect(s["size"] == .array([.integer(256), .integer(256)]))
        #expect(s["optics"]?["field_extent"]?.doubleValue == 0.1 && s["optics"]?["psf_size"]?.int == 0)
        #expect(s["optics"]?["pupil"]?["grid"]?.int == 0)
        #expect(ImageSimulationSettings.extentTitle(ImageSimulationSettings.fullExtent) == "full field")
        #expect(ImageSimulationSettings.extentTitle(1) == "full square" && ImageSimulationSettings.extentTitle(0.1) == "10 %")
    }

    /// Relative illumination cells outside the field are null: they stay
    /// no data (not 0), and the map is clipped at the field's edge.
    @Test func outOfFieldCellsAreNoData() {
        let n = 5
        var field: [Double] = [], values: [Double] = [], image: [Double] = []
        for j in 0..<n {
            for i in 0..<n {
                let x = -1 + 2 * Double(i) / Double(n - 1), y = -1 + 2 * Double(j) / Double(n - 1)
                field += [x, y]
                let outside = (x * x + y * y).squareRoot() > 1
                values.append(outside ? .nan : 1 - 0.1 * (x * x + y * y))
                image += outside ? [.nan, .nan] : [10 * x, 10 * y]
            }
        }
        let d = FieldMapData(
            nx: n, ny: n, columns: [.init(name: "relative_illumination", unit: "1", definition: "")],
            field: (0..<(n * n)).map { SIMD2(field[2 * $0], field[2 * $0 + 1]) },
            image: (0..<(n * n)).map { SIMD2(image[2 * $0], image[2 * $0 + 1]) }, values: values)
        #expect(d.outsideFieldIsNoData)
        let g = d.grid(0)
        #expect(g.noData == .unitDisk)
        #expect(d.range(0) == 0.9...1, "the range ignores the cells without data")
        let up = g.upsampled(to: 41)
        for row in 0..<up.ny {
            for col in 0..<up.nx {
                let x = -1 + 2 * Double(col) / Double(up.nx - 1), y = -1 + 2 * Double(row) / Double(up.ny - 1)
                let v = up.values[row * up.nx + col]
                let r = (x * x + y * y).squareRoot()
                if r > 1 + 1e-9 { #expect(v.isNaN) } else if r < 0.95 { #expect(v.isFinite && v >= 0.9 - 1e-12) }
            }
        }
        // A map with every sample inside the field is unchanged.
        var full = d
        full.values = values.map { $0.isNaN ? 0.5 : $0 }
        #expect(!full.outsideFieldIsNoData && full.grid(0).noData == nil)
    }
}
