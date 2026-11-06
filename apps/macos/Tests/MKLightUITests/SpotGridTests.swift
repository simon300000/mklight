import CoreGraphics
import Foundation
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The spot diagram's field × focus grid: points moved to each plane, the
/// single-plane fallback, the marker style, the layout and the box scale.
@Suite @MainActor struct SpotGridTests {
    static let base = 0.02
    static let planes = [-0.08, 0.0, 0.08]

    /// Two fields of three points each (one wavelength), with slopes.
    static func output(slopes: Bool = true, reference: String = "chief") -> AnalysisOutput {
        let points: [Double] = [0.001, 0, 0, 0.002, -0.001, -0.001, 0.003, 0.001, 0, 0, -0.002, 0.004]
        let slope: [Double] = [0.1, 0, 0, -0.2, 0.05, 0.05, -0.1, 0.1, 0.2, 0, 0, 0.3]
        func field(_ i: Int) -> JSONValue {
            var f: [String: JSONValue] = [
                "field": .integer(Int64(i)), "field_value": .array([.number(0), .number(Double(10 * i))]),
                "point_range": .array([.integer(Int64(3 * i)), .integer(3)]),
                "rms_radius": .number(0.0015 + Double(i) * 0.001), "geo_radius": .number(0.0025 + Double(i) * 0.002),
                "reference": .array([.number(0), .number(0)]),
            ]
            if slopes {
                f["through_focus"] = .array(
                    planes.enumerated().map { k, d in
                        .object([
                            "defocus": .number(d), "rms_radius_chief": .number(0.01 * Double(k + 1) + Double(i)),
                            "rms_radius_centroid": .number(0.02 * Double(k + 1) + Double(i)),
                        ])
                    })
            }
            return .object(f)
        }
        let json: JSONValue = .object([
            "fields": .array([field(0), field(1)]), "defocus": .number(base), "airy_radius": .number(0.0005),
            "wavelengths_um": .array([.number(0.55)]), "wavelengths": .array([.integer(0)]),
        ])
        var buffers = [
            ResultBuffer(name: "points", doubles: points, shape: [6, 2]),
            ResultBuffer(name: "point_wavelength", doubles: Array(repeating: 0, count: 6)),
        ]
        if slopes { buffers.append(ResultBuffer(name: "point_slope", doubles: slope, shape: [6, 2])) }
        return MapReaderTests.output(
            "spot_diagram", json: json, buffers: buffers, settings: .object(["reference": .string(reference)]))
    }

    @Test func fieldByFocusGrid() throws {
        let o = Self.output()
        let g = try #require(SpotGrid(output: o, context: AnalysisContext()))
        #expect(g.rows.count == 2 && g.rows.allSatisfy { $0.cells.count == 3 })
        #expect(g.planes == Self.planes)
        #expect(g.baseColumn == nil, "the base defocus 0.02 is not one of the planes")
        let p = o.buffer(named: "points")!.doubles, s = o.buffer(named: "point_slope")!.doubles
        for (r, row) in g.rows.enumerated() {
            for (c, cell) in row.cells.enumerated() {
                #expect(cell.defocus == Self.planes[c])
                let got = try #require(cell.groups.first).points
                #expect(got.count == 3 && cell.pointCount == 3)
                for k in 0..<3 {
                    let i = 3 * r + k, d = Self.planes[c] - Self.base
                    #expect(abs(got[k].x - (p[2 * i] + d * s[2 * i])) < 1e-15)
                    #expect(abs(got[k].y - (p[2 * i + 1] + d * s[2 * i + 1])) < 1e-15)
                }
                #expect(cell.rms == 0.01 * Double(c + 1) + Double(r), "chief reference")
            }
        }
        #expect(g.rows[1].label.hasPrefix("Field 2"))
        let centroid = try #require(SpotGrid(output: Self.output(reference: "centroid"), context: AnalysisContext()))
        #expect(centroid.rows[0].cells[2].rms == 0.06)
        #expect(SpotDiagramView(output: o, context: AnalysisContext()) != nil)
    }

    @Test func singlePlaneWithoutSlopes() throws {
        let g = try #require(SpotGrid(output: Self.output(slopes: false), context: AnalysisContext()))
        #expect(g.planes == [Self.base] && g.baseColumn == 0)
        #expect(g.rows.allSatisfy { $0.cells.count == 1 && $0.points.count == 3 })
        #expect(g.rows[0].cells[0].rms == 0.0015 && abs(g.rows[1].cells[0].geo - 0.0045) < 1e-15)
        #expect(g.rows[0].cells[0].groups[0].points[1] == SIMD2(0, 0.002))
    }

    @Test func notASpotDiagram() {
        let o = MapReaderTests.output("spot_diagram", json: .object(["unexpected": .bool(true)]), buffers: [])
        #expect(SpotGrid(output: o, context: AnalysisContext()) == nil)
        #expect(SpotDiagramView(output: o, context: AnalysisContext()) == nil)
    }

    @Test func markerStyle() {
        let counts = [1, 10, 100, 500, 1000, 2000, 10_000, 100_000]
        for cell in [48.0, 120, 400] as [CGFloat] {
            let styles = counts.map { SpotMarker.style(points: $0, cell: cell) }
            for s in styles {
                #expect(s.radius >= 0.5 && s.radius <= 1.25)
                #expect(s.opacity >= 0.5 && s.opacity <= 0.85)
            }
            for (a, b) in zip(styles, styles.dropFirst()) {
                #expect(b.radius <= a.radius && b.opacity <= a.opacity)
            }
        }
        #expect(SpotMarker.style(points: 2000, cell: 120).radius < SpotMarker.style(points: 100, cell: 120).radius)
        #expect(SpotMarker.style(points: 2000, cell: 120).opacity < SpotMarker.style(points: 100, cell: 120).opacity)
        #expect(SpotMarker.style(points: 2000, cell: 160).radius > SpotMarker.style(points: 2000, cell: 80).radius)
        #expect(SpotMarker.style(points: 50, cell: 400).opacity == 0.85)
        #expect(SpotMarker.style(points: 5000, cell: 400).opacity == 0.5)
        let sizes = stride(from: 40.0, through: 500, by: 20).map { SpotMarker.style(points: 1000, cell: CGFloat($0)).radius }
        #expect(zip(sizes, sizes.dropFirst()).allSatisfy { $0 <= $1 })
    }

    @Test func layout() {
        let narrow = SpotGridLayout.metrics(columns: 3, rows: 3, in: CGSize(width: 260, height: 600))
        #expect(narrow.cell >= SpotGridLayout.minCell && !narrow.scrollsHorizontally)
        #expect(narrow.rowHeader + 3 * (narrow.cell + SpotGridLayout.spacing) <= 260)
        // Wide and short: cells grow, capped to fit the height.
        let wide = SpotGridLayout.metrics(columns: 3, rows: 2, in: CGSize(width: 2400, height: 700))
        let height = SpotGridLayout.columnHeader + SpotGridLayout.spacing
            + 2 * (wide.cell + SpotGridLayout.caption + 2 + SpotGridLayout.spacing)
        #expect(wide.cell > 200 && height <= 700)
        #expect(wide.cell < (2400 - wide.rowHeader) / 3 - SpotGridLayout.spacing)
        // Huge: capped.
        #expect(SpotGridLayout.metrics(columns: 1, rows: 1, in: CGSize(width: 5000, height: 5000)).cell == SpotGridLayout.maxCell)
        // Many rows: rows scroll at a comfortable size instead of shrinking.
        let many = SpotGridLayout.metrics(columns: 3, rows: 12, in: CGSize(width: 900, height: 500))
        #expect(many.cell >= SpotGridLayout.comfortableCell)
        // Too narrow: horizontal scrolling as a last resort.
        let tiny = SpotGridLayout.metrics(columns: 9, rows: 2, in: CGSize(width: 200, height: 400))
        #expect(tiny.scrollsHorizontally && tiny.cell == SpotGridLayout.minCell)
    }

    @Test func scale() throws {
        let g = try #require(SpotGrid(output: Self.output(), context: AnalysisContext()))
        let common = g.halfWidths(.grid), perRow = g.halfWidths(.field)
        let extents = g.rows.map { $0.cells.map(\.extent) }
        let all = extents.flatMap { $0 }.max()!
        #expect(common.count == 2 && common.allSatisfy { abs($0 - all * 1.15) < 1e-15 })
        for (r, e) in extents.enumerated() {
            #expect(abs(perRow[r] - e.max()! * 1.15) < 1e-15)
        }
        #expect(perRow[0] != perRow[1] && perRow.max() == common[0])
        // The Airy radius is a floor.
        let h = SpotGrid.halfWidths(extents: [[0.001], [0.002]], airy: 0.0015, scale: .field)
        #expect(abs(h[0] - 0.0015 * 1.15) < 1e-15 && abs(h[1] - 0.002 * 1.15) < 1e-15)
    }

    @Test func planeLabelsAndThinning() {
        #expect(SpotGrid.planeLabel(0) == "0")
        #expect(SpotGrid.planeLabel(0.08) == "+0.08 mm")
        #expect(SpotGrid.planeLabel(-0.08) == "\u{2212}0.08 mm")
        #expect(SpotGrid.planeLabel(0.0005) == "+0.5 µm")
        #expect(SpotGrid.thin(3, to: 9, keeping: 1) == [0, 1, 2])
        let t = SpotGrid.thin(101, to: 9, keeping: 51)
        #expect(t.count == 9 && t.first == 0 && t.last == 100 && t.contains(51))
    }
}
