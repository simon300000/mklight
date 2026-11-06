import CoreGraphics
import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The readers and pure helpers of the field maps, the glass map and the
/// image simulation: marching squares, the sRGB transfer, image row order,
/// the field-map reader, glass statuses, the CRA margin bands, the compact
/// array settings and the image source import.
@Suite struct MapReaderTests {
    static func output(_ analysis: String, json: JSONValue, buffers: [ResultBuffer], settings: JSONValue = .object([:])) -> AnalysisOutput {
        AnalysisOutput(
            runInfo: RunInfo(
                engineVersion: "1", snapshotHash: SnapshotHash(hex: "00"), analysis: analysis, settings: settings,
                backend: "cpu_reference", precision: .f64Reference, seed: nil, startedUnixMs: 0, durationMs: 0),
            result: .generic(GenericResult(analysis: analysis, json: json, buffers: buffers)))
    }

    /// A 3 × 2 field map with two columns: `a = Hx + Hy`, `b = 1 - Hx²`.
    static func fieldMap() -> AnalysisOutput {
        let hx = [-1.0, 0, 1], hy = [-1.0, 1]
        var field: [Double] = [], image: [Double] = [], values: [Double] = []
        for y in hy {
            for x in hx {
                field += [x, y]
                image += [10 * x, 10 * y]
                values += [x + y, 1 - x * x]
            }
        }
        let json: JSONValue = .object([
            "columns": .array([
                .object(["name": .string("stop_error_x"), "unit": .string("mm"), "definition": .string("A")]),
                .object(["name": .string("relative_illumination"), "unit": .string("1"), "definition": .string("B")]),
            ]),
            "field_coordinates": .string("field_coordinates"), "image_mm": .string("image_mm"), "values": .string("values"),
            "shape": .array([.integer(3), .integer(2)]),
        ])
        return output(
            "pupil_aberration", json: json,
            buffers: [
                ResultBuffer(name: "field_coordinates", doubles: field, shape: [2, 3, 2]),
                ResultBuffer(name: "image_mm", doubles: image, shape: [2, 3, 2]),
                ResultBuffer(name: "values", doubles: values, shape: [2, 3, 2]),
            ])
    }

    @Test func fieldMapReader() throws {
        let d = try #require(FieldMapData(output: Self.fieldMap()))
        #expect(d.nx == 3 && d.ny == 2 && d.layout == .map)
        #expect(d.columns.map(\.name) == ["stop_error_x", "relative_illumination"])
        #expect(d.columns[0].titleWithUnit == "Stop error x (mm)")
        #expect(d.columns[1].unitLabel == "")
        #expect(d.value(5, 0) == 2)  // Hx 1, Hy 1
        #expect(d.fieldRangeX == -1...1 && d.fieldRangeY == -1...1)
        let g = d.grid(0)
        #expect(g.values == [-2, -1, 0, 0, 1, 2])
        #expect(d.isSigned(0) && !d.isSigned(1))
        #expect(abs(d.imageRadius[5] - 10 * 2.0.squareRoot()) < 1e-12)
        #expect(d.radial(1).first?.0 == 10)  // (0, ±1) is nearest the axis
        #expect(d.argmax(0) == 5 && d.argmin(0) == 0)
        // A sweep: one row.
        let sweep = FieldMapData(
            nx: 3, ny: 1, columns: d.columns, field: [SIMD2(-1, 0), SIMD2(0, 0), SIMD2(1, 0)],
            image: [SIMD2(-1, 0), SIMD2(0, 0), SIMD2(1, 0)], values: [1, 0, 2, 0, 3, 0])
        #expect(sweep.layout == .sweepX)
        #expect(sweep.sweep(0).map(\.1) == [1, 2, 3])
        // Unknown shapes are not read.
        #expect(FieldMapData(output: Self.output("pupil_aberration", json: .object([:]), buffers: [])) == nil)
        // Radial symmetry: a = Hx + Hy differs between (−1, −1) and (1, 1).
        #expect(!d.isRadial(0) && d.isRadial(1))
    }

    @Test func sensorLimitInterpolation() {
        let t = FieldMapData.sensorLimit(.array([.array([.number(0), .number(10)]), .array([.integer(20), .number(30)])]))
        #expect(t.count == 2)
        #expect(FieldMapData.limit(t, at: -1) == 10)
        #expect(FieldMapData.limit(t, at: 10) == 20)
        #expect(FieldMapData.limit(t, at: 50) == 30)
        #expect(FieldMapData.limit([], at: 1).isNaN)
    }

    @Test @MainActor func marginBandsSplitAtTheCrossing() {
        // CRA rises 0…20° over 0…10 mm; the limit is 15°: violated beyond 7.5 mm.
        let bands = ChiefRayAngleView.bands(curve: [(0, 0), (10, 20)], limit: [(0, 15)])
        #expect(bands.count == 2)
        #expect(bands[0].violated == false && bands[1].violated == true)
        let crossing = bands[0].points.last!.x
        #expect(abs(crossing - 7.5) < 1e-9)
        #expect(bands[1].points.first!.x == crossing)
        #expect(ChiefRayAngleView.bands(curve: [(0, 0), (10, 20)], limit: []).isEmpty)
    }

    @Test func marchingSquaresTracesACircle() {
        // f = x² + y² on a 41 × 41 grid; the 0.25 iso-line is the circle r = 0.5.
        let n = 41
        var v: [Double] = []
        for j in 0..<n {
            for i in 0..<n {
                let x = -1 + 2 * Double(i) / Double(n - 1), y = -1 + 2 * Double(j) / Double(n - 1)
                v.append(x * x + y * y)
            }
        }
        let g = Grid2D(nx: n, ny: n, values: v)
        let segs = Contours.segments(g, level: 0.25)
        #expect(segs.count > 30)
        for s in segs {
            for p in [s.a, s.b] {
                #expect(abs((p.x * p.x + p.y * p.y).squareRoot() - 0.5) < 0.01)
            }
        }
        #expect(Contours.segments(g, level: 5).isEmpty)
        // A NaN corner removes its cells only.
        var holes = v
        holes[20 * n + 30] = .nan
        #expect(Contours.segments(Grid2D(nx: n, ny: n, values: holes, xRange: -1...1, yRange: -1...1), level: 0.25).count <= segs.count)
    }

    @Test func marchingSquaresSaddle() {
        // Corners 1 0 / 0 1 with a high centre: the lines cut off the low corners.
        let g = Grid2D(nx: 2, ny: 2, values: [1, 0, 0.2, 1], xRange: 0...1, yRange: 0...1)
        let segs = Contours.segments(g, level: 0.5)
        #expect(segs.count == 2)
        // Each segment separates one low corner: (1, 0) or (0, 1).
        let mids = segs.map { ($0.a + $0.b) / 2 }
        #expect(mids.contains { $0.x > 0.5 && $0.y < 0.5 })
        #expect(mids.contains { $0.x < 0.5 && $0.y > 0.5 })
    }

    @Test func contourLevels() {
        // About five round levels inside the data's range.
        #expect(Contours.illuminationLevels(in: 0.42...1) == [0.5, 0.6, 0.7, 0.8, 0.9])
        #expect(Contours.illuminationLevels(in: 0.87...1) == [0.88, 0.9, 0.92, 0.94, 0.96, 0.98])
        #expect(Contours.levels(in: 0...1) == [0.2, 0.4, 0.6, 0.8])
    }

    @Test func upsamplingKeepsTheNodes() {
        let g = Grid2D(nx: 2, ny: 2, values: [0, 1, 2, 3])
        let u = g.upsampled(to: 5)
        #expect(u.nx == 5 && u.ny == 5)
        #expect(u.values[0] == 0 && u.values[4] == 1 && u.values[20] == 2 && u.values[24] == 3)
        #expect(abs(u.values[12] - 1.5) < 1e-12)
        #expect(g.node(3) == SIMD2(1, 1))
    }

    @Test func srgbTransfer() {
        #expect(SRGB.encode(0) == 0 && SRGB.encode(1) == 1)
        #expect(abs(SRGB.encode(0.5) - 0.735_356_983) < 1e-6)
        #expect(abs(SRGB.encode(0.002) - 0.02584) < 1e-5)  // linear segment
        #expect(SRGB.encode(-0.3) == 0 && SRGB.encode(7) == 1 && SRGB.encode(.nan) == 0)
        for x in stride(from: 0.0, through: 1, by: 0.05) {
            #expect(abs(SRGB.decode(SRGB.encode(x)) - x) < 1e-12)
        }
        #expect(SRGB.byte(SRGB.decode(128.0 / 255)) == 128)
    }

    @Test func imageRowsAreFlippedForDisplay() throws {
        // Row 0 (low y) dark, row 1 bright; noise below zero clamps.
        let img = LinearImage(width: 2, height: 2, channels: 1, data: [-0.1, 0, 1, 1])
        let px = img.rgba8()
        #expect(Array(px[0..<4]) == [255, 255, 255, 255])  // top row of the display is row 1
        #expect(Array(px[8..<12]) == [0, 0, 0, 255])
        #expect(img.rgba8(gain: 0.25)[0] == SRGB.byte(0.25))
        #expect(img.clippedFraction(gain: 2) == 0.5)
        let rgb = LinearImage(width: 1, height: 1, channels: 3, data: [1, 0, 0.5])
        #expect(Array(rgb.rgba8()[0..<3]) == [255, 0, SRGB.byte(0.5)])
        #expect(try #require(rgb.cgImage()).width == 1)
        #expect(LinearImage(source: rgb.sourceJSON) == rgb)
        #expect(LinearImage(source: .object(["kind": .string("buffer"), "width": .integer(2), "height": .integer(2), "channels": .integer(1), "data": .array([])])) == nil)
    }

    @Test func patternsMatchTheEngineDefinitions() throws {
        let grid = try #require(LinearImage.pattern("grid", width: 64, height: 64))
        // Grid lines every 8 px at u·8 fract < 0.1: columns 0, 8, 16… dark.
        #expect(grid.data[3 * 64 + 0] == 0 && grid.data[3 * 64 + 3] == 1 && grid.data[3 * 64 + 8] == 0)
        let edge = try #require(LinearImage.pattern("slanted_edge", width: 64, height: 64))
        #expect(edge.data[32 * 64 + 10] == 0 && edge.data[32 * 64 + 50] == 1)
        let star = try #require(LinearImage.pattern("siemens_star", width: 32, height: 32))
        #expect(Set(star.data) == [0, 1])
        let text = try #require(LinearImage.pattern("text_chart", width: 30, height: 30))
        // The first glyph (M) at scale 4: its row 0 is 10001.
        #expect(text.data[0] == 0 && text.data[4] == 1 && text.data[16] == 0)
        #expect(LinearImage.pattern("unknown", width: 4, height: 4) == nil)
    }

    @Test func imageSimulationReader() throws {
        let json: JSONValue = .object([
            "channels": .integer(1), "field_grid": .integer(2), "image": .string("image"), "size": .array([.integer(2), .integer(1)]),
            "pixel_size_mm": .array([.number(0.01), .number(0.02)]), "origin_mm": .array([.number(-0.01), .number(-0.01)]),
            "field_mapping": .array([
                .object([
                    "field": .array([.number(1), .number(0)]), "ideal_mm": .array([.number(10), .number(0)]),
                    "real_mm": .array([.number(9.9), .number(0)]), "relative_illumination": .number(0.8),
                ]),
                .object([
                    "field": .array([.number(0), .number(0)]), "ideal_mm": .array([.number(0), .number(0)]),
                    "real_mm": .array([.number(0), .number(0)]), "relative_illumination": .number(1),
                ]),
            ]),
        ])
        let settings: JSONValue = .object(["source": .object(["kind": .string("pattern"), "pattern": .string("siemens_star")])])
        let out = Self.output("image_simulation", json: json, buffers: [ResultBuffer(name: "image", doubles: [0.2, 0.4], shape: [1, 2, 1])], settings: settings)
        let d = try #require(ImageSimulationData(output: out))
        #expect(d.image.width == 2 && d.image.height == 1)
        #expect(d.source == .pattern("siemens_star"))
        #expect(d.sourceTitle == "Siemens star pattern")
        #expect(d.sourceImage?.width == 2)
        #expect(abs(d.nodes[0].distortionPercent + 1) < 1e-9)
        #expect(d.nodes[1].distortionPercent.isNaN)
        #expect(ImageSimulationData.maxDistortionPercent(d.nodes) == d.nodes[0].distortionPercent)
        #expect(ImageSimulationData.minRelativeIllumination(d.nodes) == 0.8)
        let keys = AnalysisResultView.keyNumbers(analysis: "image_simulation", output: out, context: AnalysisContext())
        #expect(keys.map(\.label) == ["Pixel", "Max distortion", "Min RI"])
    }

    @Test func glassStatuses() {
        #expect(GlassStatus(json: .string("preferred")) == .preferred)
        #expect(GlassStatus(json: .string("Obsolete")) == .obsolete)
        #expect(GlassStatus(json: .string("melt")) == .melt)
        #expect(GlassStatus(json: .object(["other": .integer(7)])) == .other(7))
        #expect(GlassStatus(json: .null) == .unknown)
        #expect(GlassStatus(json: nil) == .unknown)
        #expect(GlassStatus.other(7).group == .unknown && GlassStatus.other(7).label == "Status 7")
        #expect(GlassStatus.obsolete.hollow && !GlassStatus.preferred.hollow)
        // Every legend status has its own colour.
        let colours = GlassStatus.allCases.map { $0.rgb(dark: false) }
        #expect(Set(colours.map { "\($0)" }).count == colours.count)
    }

    @Test @MainActor func glassMapReader() throws {
        let json: JSONValue = .object([
            "coordinates": .string("coordinates"),
            "glasses": .array([
                .object([
                    "catalog": .string("C"), "name": .string("A"), "kind": .string("catalog"), "nd": .number(1.5), "vd": .number(64),
                    "pgf": .number(0.53), "status": .string("standard"), "surfaces": .array([]), "exclude_substitution": .bool(true),
                ]),
                .object([
                    "catalog": .string("C"), "name": .string("B"), "kind": .string("catalog"), "nd": .number(1.6), "vd": .number(36),
                    "pgf": .number(0.58), "status": .null, "surfaces": .array([.integer(2), .integer(5)]), "exclude_substitution": .null,
                ]),
                .object([
                    "catalog": .null, "name": .string("1.7/40"), "kind": .string("model"), "nd": .number(1.7), "vd": .number(40),
                    "pgf": .null, "status": .null, "surfaces": .array([.integer(3)]),
                ]),
            ]),
        ])
        // The buffer wins over the JSON (it is the documented coordinate source).
        let coords = ResultBuffer(name: "coordinates", doubles: [1.51, 64, 0.53, 1.6, 36, 0.58, 1.7, 40, .nan], shape: [3, 3])
        let d = try #require(GlassMapData(output: Self.output("glass_map", json: json, buffers: [coords])))
        #expect(d.glasses.count == 3)
        #expect(d.glasses[0].nd == 1.51 && d.glasses[0].status == .standard && d.glasses[0].excludeSubstitution == true)
        #expect(d.glasses[1].id == "C/B" && d.glasses[1].surfaceLabel == "S2, S5")
        #expect(d.glasses[2].kind == .model && d.glasses[2].inDesign && d.glasses[2].pgf.isNaN)
        #expect(d.designHighlights == [GlassHighlight(id: "C/B", label: "S2, S5"), GlassHighlight(id: d.glasses[2].id, label: "S3")])
        #expect(d.catalogs == ["C"])
        #expect(abs(GlassMapPoint.normalLine(36.37) - 0.5826) < 1e-4)  // F2 lies on the normal line
        let chart = GlassMapChart(glasses: d.glasses, selection: .constant(nil))
        #expect(chart.xDomain.upperBound < -30 && chart.xDomain.lowerBound < -64)  // V_d decreases to the right
        #expect(GlassMapData(output: Self.output("glass_map", json: .object([:]), buffers: [])) == nil)
    }

    @Test @MainActor func arraySettings() {
        let grid = SettingsField(name: "grid", schema: .object(["type": .string("array"), "default": .array([.integer(5), .integer(5)]), "description": .string("[nx,ny], 1..101 samples")]), required: false)
        let extent = SettingsField(name: "extent", schema: .object(["type": .string("array"), "default": .array([.number(1), .number(1)])]), required: false)
        #expect(SettingPairField(field: grid, label: "Grid", value: nil, set: { _ in }).isInteger)
        #expect(SettingPairField(field: grid, label: "Grid", value: nil, set: { _ in }).bounds == 1...101)
        #expect(!SettingPairField(field: extent, label: "Extent", value: nil, set: { _ in }).isInteger)
        #expect(SettingsBar.arraySummary(grid, .array([.integer(9), .integer(7)])) == "9 × 7")
        let limit = SettingsField(name: "sensor_limit", schema: .object(["type": .string("array"), "default": .array([])]), required: false)
        #expect(SettingsBar.arraySummary(limit, .array([.array([.number(0), .number(5)])])) == "1 point")
        let source = SettingsField(name: "source", schema: .object(["type": .string("object"), "default": .object(["kind": .string("pattern"), "pattern": .string("grid")])]), required: false)
        #expect(SettingsBar.arraySummary(source, .object(["kind": .string("pattern"), "pattern": .string("text_chart")])) == "Text Chart")
        let d = AnalysisDescriptor(
            name: "chief_ray_angle", title: "", category: "", description: "", available: true, unavailableReason: nil,
            settingsSchema: .object(["properties": .object([:])]), defaultSettings: .object([:]))
        let bar = SettingsBar(pane: AnalysisPane(analysis: "chief_ray_angle"), descriptor: d, system: nil)
        #expect(bar.control(of: grid) == .pair(integer: true, nullable: false))
        #expect(bar.control(of: limit) == .sensorLimit)
        #expect(bar.control(of: source) == .imageSource)
    }

    @Test func importedImagesAreNotSaved() {
        let buffer = LinearImage(width: 1, height: 1, channels: 1, data: [0.5]).sourceJSON
        let s: [String: JSONValue] = ["source": buffer, "source_file": .string("/tmp/a.jpg"), "source_field_aspect": .number(1), "size": .array([.integer(8), .integer(8)])]
        let saved = ImageSourceImport.persistable(s)
        #expect(saved["source"] == nil && saved["source_field_aspect"] == nil)
        #expect(saved["source_file"] == .string("/tmp/a.jpg") && saved["size"] != nil)
        // A buffer typed by hand (no file) is kept.
        #expect(ImageSourceImport.persistable(["source": buffer]) == ["source": buffer])
        #expect(ImageSourceImport.targetSize(.array([.integer(64), .integer(48)])) == (64, 48))
        #expect(ImageSourceImport.targetSize(.array([.integer(4096), .integer(1024)])) == (512, 128))
        #expect(ImageSourceImport.targetSize(nil) == (64, 64))
    }

    @Test func importResamplesInLinearLightBottomRowFirst() throws {
        // 4 × 4 sRGB: top half white, bottom half mid-grey (code 128).
        let ctx = try #require(
            CGContext(
                data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        ctx.setFillColor(red: 128.0 / 255, green: 128.0 / 255, blue: 128.0 / 255, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: 4, height: 2))  // CG's y is up: the bottom half
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 2, width: 4, height: 2))
        let image = try #require(ctx.makeImage())
        let lin = try #require(ImageSourceImport.linear(image, width: 4, height: 4))
        #expect(lin.channels == 3 && lin.isValid)
        // Row 0 is the bottom (grey, linear ≈ 0.216); the last row is white.
        #expect(abs(lin.data[0] - SRGB.decode(128.0 / 255)) < 0.01)
        #expect(abs(lin.data[(3 * 4) * 3] - 1) < 0.01)
        // Aspect: a 2:1 crop of the 4 × 4 image keeps the middle rows only.
        let wide = try #require(ImageSourceImport.linear(image, width: 4, height: 2, aspect: 2))
        #expect(wide.height == 2)
        #expect(abs(wide.data[0] - SRGB.decode(128.0 / 255)) < 0.02 && abs(wide.data[4 * 3] - 1) < 0.02)
    }
}
