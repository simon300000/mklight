import CoreGraphics
import Foundation
import MKLightEngine
import MKLightModel

// Readers for the field-map analyses (`pupil_aberration`,
// `relative_illumination_map`, `chief_ray_angle`), `glass_map` and
// `image_simulation` (docs/RESULTS.md), and the pure helpers their views
// use: iso-lines by marching squares and the sRGB transfer.

// MARK: - Field maps

/// The shared result of the signed Cartesian field maps: `field_coordinates
/// [ny, nx, 2]`, `image_mm [ny, nx, 2]` and `values [ny, nx, columns]`,
/// with the `columns` described in the JSON. Row 0 is the lowest Hy, x
/// fastest (as `Grid2D`).
struct FieldMapData: Sendable {
    struct Column: Sendable, Hashable, Identifiable {
        var name: String
        var unit: String
        var definition: String
        var id: String { name }

        /// "Stop error x (mm)".
        var title: String {
            let t = name.replacingOccurrences(of: "_", with: " ")
            return t.prefix(1).uppercased() + t.dropFirst()
        }

        var unitLabel: String {
            switch unit {
            case "", "1": ""
            case "deg": "°"
            default: unit
            }
        }

        var titleWithUnit: String { unitLabel.isEmpty ? title : "\(title) (\(unitLabel))" }
    }

    /// How the samples are laid out.
    enum Layout: Equatable, Sendable {
        /// An nx × ny map (both > 1).
        case map
        /// A sweep along Hx (ny == 1) or Hy (nx == 1).
        case sweepX, sweepY
    }

    var nx: Int
    var ny: Int
    var columns: [Column]
    /// Normalised [Hx, Hy] per sample, `j * nx + i`.
    var field: [SIMD2<Double>]
    /// Chief-ray image point (mm) per sample; NaN for unusable fields.
    var image: [SIMD2<Double>]
    /// `values[(j * nx + i) * columns.count + c]`.
    var values: [Double]

    init(nx: Int, ny: Int, columns: [Column], field: [SIMD2<Double>], image: [SIMD2<Double>], values: [Double]) {
        self.nx = nx
        self.ny = ny
        self.columns = columns
        self.field = field
        self.image = image
        self.values = values
    }

    init?(output: AnalysisOutput) {
        let j = output.json
        guard let cols = j["columns"]?.array, !cols.isEmpty,
            let f = output.buffer(ref: j["field_coordinates"]) ?? output.buffer(named: "field_coordinates"),
            let v = output.buffer(ref: j["values"]) ?? output.buffer(named: "values"),
            f.shape.count == 3, f.shape[2] == 2, v.shape.count == 3
        else { return nil }
        let ny = f.shape[0], nx = f.shape[1], nc = v.shape[2]
        guard nx > 0, ny > 0, v.shape[0] == ny, v.shape[1] == nx, nc == cols.count else { return nil }
        let img = output.buffer(ref: j["image_mm"]) ?? output.buffer(named: "image_mm")
        let n = nx * ny
        self.init(
            nx: nx, ny: ny,
            columns: cols.map {
                Column(name: $0["name"]?.string ?? "", unit: $0["unit"]?.string ?? "", definition: $0["definition"]?.string ?? "")
            },
            field: (0..<n).map { SIMD2(f[2 * $0], f[2 * $0 + 1]) },
            image: (0..<n).map { k in
                guard let img, img.count == 2 * n else { return SIMD2(.nan, .nan) }
                return SIMD2(img[2 * k], img[2 * k + 1])
            },
            values: v.doubles)
    }

    var layout: Layout {
        if nx > 1 && ny > 1 { return .map }
        return ny == 1 && nx > 1 ? .sweepX : .sweepY
    }

    var count: Int { nx * ny }

    func columnIndex(_ name: String) -> Int? { columns.firstIndex { $0.name == name } }

    func value(_ sample: Int, _ column: Int) -> Double { values[sample * columns.count + column] }

    /// One column at every sample.
    func column(_ c: Int) -> [Double] { (0..<count).map { value($0, c) } }

    /// The range of the normalised field covered by the samples.
    var fieldRangeX: ClosedRange<Double> { Self.range(field.map(\.x)) }
    var fieldRangeY: ClosedRange<Double> { Self.range(field.map(\.y)) }

    /// A column as a grid over the normalised field (map layout).
    func grid(_ c: Int) -> Grid2D {
        Grid2D(nx: nx, ny: ny, values: column(c), xRange: fieldRangeX, yRange: fieldRangeY)
    }

    /// A column along the swept field axis, `(H, value)` in sample order.
    func sweep(_ c: Int) -> [(Double, Double)] {
        (0..<count).map { k in (layout == .sweepX ? field[k].x : field[k].y, value(k, c)) }
    }

    /// The chief ray's image radius (mm) at each sample.
    var imageRadius: [Double] { image.map { ($0.x * $0.x + $0.y * $0.y).squareRoot() } }

    /// `(image radius, value)` at every usable sample, by radius.
    func radial(_ c: Int) -> [(Double, Double)] {
        zip(imageRadius, column(c)).filter { $0.0.isFinite && $0.1.isFinite }.sorted { $0.0 < $1.0 }
    }

    /// The finite range of a column.
    func range(_ c: Int) -> ClosedRange<Double>? {
        let f = column(c).filter(\.isFinite)
        guard let lo = f.min(), let hi = f.max() else { return nil }
        return lo...hi
    }

    /// Whether a column is best shown on a signed (diverging) scale: it
    /// takes both signs, or it is a difference that is zero on axis.
    func isSigned(_ c: Int) -> Bool {
        guard let r = range(c) else { return false }
        if r.lowerBound < 0 && r.upperBound > 0 { return true }
        let name = columns[c].name
        return r.lowerBound < 0 || name.contains("error") || name.contains("variation") || name.hasPrefix("angle_") || name == "margin"
    }

    /// Whether a column depends on the field radius only (within `tolerance`
    /// of the column's range): the system is rotationally symmetric for it.
    func isRadial(_ c: Int, tolerance: Double = 1e-3) -> Bool {
        guard let r = range(c) else { return true }
        let scale = max(r.upperBound - r.lowerBound, abs(r.upperBound), 1e-12)
        var byRadius: [Int: Double] = [:]
        let rx = max(abs(fieldRangeX.lowerBound), abs(fieldRangeX.upperBound), 1e-12)
        let ry = max(abs(fieldRangeY.lowerBound), abs(fieldRangeY.upperBound), 1e-12)
        guard abs(rx - ry) < 1e-9 else { return false }
        for k in 0..<count {
            let v = value(k, c)
            guard v.isFinite else { continue }
            let key = Int((field[k].x * field[k].x + field[k].y * field[k].y).squareRoot() * 1e6)
            if let w = byRadius[key], abs(w - v) > tolerance * scale { return false }
            byRadius[key] = v
        }
        return true
    }

    /// The sample with the largest finite value of a column.
    func argmax(_ c: Int) -> Int? {
        (0..<count).filter { value($0, c).isFinite }.max { value($0, c) < value($1, c) }
    }

    func argmin(_ c: Int) -> Int? {
        (0..<count).filter { value($0, c).isFinite }.min { value($0, c) < value($1, c) }
    }

    static func range(_ v: [Double]) -> ClosedRange<Double> {
        let f = v.filter(\.isFinite)
        guard let lo = f.min(), let hi = f.max() else { return -1...1 }
        return lo < hi ? lo...hi : (lo - 1)...(hi + 1)
    }

    /// `[[radius mm, max deg], …]` from `sensor_limit`.
    static func sensorLimit(_ v: JSONValue?) -> [(Double, Double)] {
        (v?.array ?? []).compactMap { row in
            guard let r = row.array, r.count == 2, let a = r[0].doubleValue, let b = r[1].doubleValue else { return nil }
            return (a, b)
        }
    }

    /// The sensor limit at a radius: linear, held at the end points (as the engine).
    static func limit(_ table: [(Double, Double)], at r: Double) -> Double {
        guard let first = table.first, let last = table.last else { return .nan }
        if r <= first.0 { return first.1 }
        if r >= last.0 { return last.1 }
        for k in 1..<table.count where r <= table[k].0 {
            let (r0, a0) = table[k - 1], (r1, a1) = table[k]
            return r1 > r0 ? a0 + (a1 - a0) * (r - r0) / (r1 - r0) : a1
        }
        return last.1
    }
}

// MARK: - Iso-lines

/// Iso-lines of a sampled grid by marching squares, with linear
/// interpolation along the cell edges and the cell mean to resolve saddles.
enum Contours {
    struct Segment: Equatable, Sendable {
        var a: SIMD2<Double>
        var b: SIMD2<Double>
    }

    /// The segments of the `level` iso-line of `grid`, in the grid's x / y
    /// units (samples at the nodes, spanning `xRange` × `yRange`). Cells
    /// with a non-finite corner are skipped.
    static func segments(_ grid: Grid2D, level: Double) -> [Segment] {
        let nx = grid.nx, ny = grid.ny
        guard nx >= 2, ny >= 2, level.isFinite else { return [] }
        let dx = (grid.xRange.upperBound - grid.xRange.lowerBound) / Double(nx - 1)
        let dy = (grid.yRange.upperBound - grid.yRange.lowerBound) / Double(ny - 1)
        func p(_ i: Double, _ j: Double) -> SIMD2<Double> {
            SIMD2(grid.xRange.lowerBound + i * dx, grid.yRange.lowerBound + j * dy)
        }
        func t(_ a: Double, _ b: Double) -> Double { a == b ? 0.5 : min(max((level - a) / (b - a), 0), 1) }
        var out: [Segment] = []
        for j in 0..<(ny - 1) {
            for i in 0..<(nx - 1) {
                // Corners counter-clockwise from the lower left.
                let v0 = grid.values[j * nx + i], v1 = grid.values[j * nx + i + 1]
                let v2 = grid.values[(j + 1) * nx + i + 1], v3 = grid.values[(j + 1) * nx + i]
                guard v0.isFinite, v1.isFinite, v2.isFinite, v3.isFinite else { continue }
                let code = (v0 >= level ? 1 : 0) | (v1 >= level ? 2 : 0) | (v2 >= level ? 4 : 0) | (v3 >= level ? 8 : 0)
                if code == 0 || code == 15 { continue }
                let x = Double(i), y = Double(j)
                // Edge crossings: bottom, right, top, left.
                let bottom = p(x + t(v0, v1), y)
                let right = p(x + 1, y + t(v1, v2))
                let top = p(x + t(v3, v2), y + 1)
                let left = p(x, y + t(v0, v3))
                switch code {
                case 1, 14: out.append(Segment(a: left, b: bottom))
                case 2, 13: out.append(Segment(a: bottom, b: right))
                case 3, 12: out.append(Segment(a: left, b: right))
                case 4, 11: out.append(Segment(a: right, b: top))
                case 6, 9: out.append(Segment(a: bottom, b: top))
                case 7, 8: out.append(Segment(a: left, b: top))
                case 5, 10:
                    // Saddle: the centre decides which corners connect.
                    let centreHigh = (v0 + v1 + v2 + v3) / 4 >= level
                    if (code == 5) == centreHigh {
                        out.append(Segment(a: left, b: top))
                        out.append(Segment(a: bottom, b: right))
                    } else {
                        out.append(Segment(a: left, b: bottom))
                        out.append(Segment(a: right, b: top))
                    }
                default: break
                }
            }
        }
        return out
    }

    /// Round contour levels within a range: `0.5, 0.6, … 0.9, 0.95` for
    /// relative illumination, else about `count` nice steps.
    static func levels(in r: ClosedRange<Double>, count: Int = 6) -> [Double] {
        guard r.upperBound > r.lowerBound, r.lowerBound.isFinite, r.upperBound.isFinite else { return [] }
        let step = Nice.length((r.upperBound - r.lowerBound) / Double(count))
        var out: [Double] = []
        var v = (r.lowerBound / step).rounded(.up) * step
        while v < r.upperBound - step * 1e-9 {
            if v > r.lowerBound + step * 1e-9 { out.append(v) }
            v += step
        }
        return out
    }

    /// Illumination levels 0.5…0.9 by 0.1 and 0.95, inside the data range.
    static func illuminationLevels(in r: ClosedRange<Double>) -> [Double] {
        [0.5, 0.6, 0.7, 0.8, 0.9, 0.95].filter { $0 > r.lowerBound && $0 < r.upperBound }
    }
}

/// Bilinear upsampling of a node grid (samples at the nodes) for display,
/// so the heat map and its linear contours agree. NaN where a corner is.
extension Grid2D {
    func upsampled(to n: Int) -> Grid2D {
        guard nx >= 2, ny >= 2, n > max(nx, ny) else { return self }
        let w = n, h = max(2, Int((Double(n) * Double(ny - 1) / Double(max(nx - 1, 1))).rounded()))
        var out = [Double](repeating: .nan, count: w * h)
        for row in 0..<h {
            let y = Double(row) / Double(h - 1) * Double(ny - 1)
            let j = min(Int(y), ny - 2), b = y - Double(j)
            for col in 0..<w {
                let x = Double(col) / Double(w - 1) * Double(nx - 1)
                let i = min(Int(x), nx - 2), a = x - Double(i)
                let v00 = values[j * nx + i], v10 = values[j * nx + i + 1]
                let v01 = values[(j + 1) * nx + i], v11 = values[(j + 1) * nx + i + 1]
                out[row * w + col] = (1 - b) * ((1 - a) * v00 + a * v10) + b * ((1 - a) * v01 + a * v11)
            }
        }
        return Grid2D(nx: w, ny: h, values: out, xRange: xRange, yRange: yRange)
    }
}

// MARK: - Glass map

/// A glass's retained catalogue status (AGF), or unknown.
enum GlassStatus: Hashable, Sendable, CaseIterable {
    case preferred, standard, obsolete, special, melt, other(Int), unknown

    static let allCases: [GlassStatus] = [.preferred, .standard, .special, .melt, .obsolete, .unknown]

    init(json: JSONValue?) {
        switch json {
        case .string(let s):
            self = switch s.lowercased() {
            case "preferred": .preferred
            case "standard": .standard
            case "obsolete": .obsolete
            case "special": .special
            case "melt": .melt
            default: .unknown
            }
        case .object(let o):
            self = o["other"]?.int.map { .other($0) } ?? .unknown
        default:
            self = .unknown
        }
    }

    var label: String {
        switch self {
        case .preferred: "Preferred"
        case .standard: "Standard"
        case .obsolete: "Obsolete"
        case .special: "Special"
        case .melt: "Melt"
        case .other(let n): "Status \(n)"
        case .unknown: "Unknown"
        }
    }

    /// The legend group (unrecognised codes join "Unknown").
    var group: GlassStatus {
        if case .other = self { return .unknown }
        return self
    }

    /// The categorical palette slot.
    var slot: Int {
        switch group {
        case .preferred: 5  // green
        case .standard: 0  // blue
        case .special: 6  // violet
        case .melt: 3  // yellow
        case .obsolete: 7  // red
        default: -1  // neutral grey
        }
    }

    /// Drawn hollow, so status does not rely on colour alone.
    var hollow: Bool { group == .obsolete }
}

/// One glass of a glass map.
struct GlassMapPoint: Identifiable, Hashable, Sendable {
    enum Kind: String, Sendable { case catalog, model }
    /// `catalog/name` for catalogue glasses, `model:name` for model glasses
    /// (with their first surface, as model names need not be unique).
    var id: String
    var catalog: String?
    var name: String
    var kind: Kind
    var nd: Double
    var vd: Double
    var pgf: Double
    var status: GlassStatus
    /// Surface indices whose following medium is this glass.
    var surfaces: [Int]
    var excludeSubstitution: Bool?

    /// Whether the design uses the glass.
    var inDesign: Bool { kind == .model || !surfaces.isEmpty }

    /// "S1, S3".
    var surfaceLabel: String { surfaces.map { "S\($0)" }.joined(separator: ", ") }

    var displayName: String { kind == .model ? "\(name) (model)" : name }

    static func id(catalog: String?, name: String, kind: Kind, surfaces: [Int]) -> String {
        kind == .model ? "model:\(name):\(surfaces.first ?? -1)" : "\(catalog ?? "")/\(name)"
    }

    /// The deviation of P_g,F from the normal line through K7 and F2
    /// (P_g,F = 0.6438 − 0.001682 V_d).
    var deltaPgF: Double { pgf - GlassMapPoint.normalLine(vd) }

    static func normalLine(_ vd: Double) -> Double { 0.6438 - 0.001682 * vd }
}

/// A glass to mark on the map, with its label ("S1, S3").
struct GlassHighlight: Hashable, Sendable {
    var id: String
    var label: String
}

/// The `glass_map` result.
struct GlassMapData: Sendable {
    var glasses: [GlassMapPoint]

    init(glasses: [GlassMapPoint]) { self.glasses = glasses }

    init?(output: AnalysisOutput) {
        guard let list = output.json["glasses"]?.array else { return nil }
        let coords = output.buffer(ref: output.json["coordinates"]) ?? output.buffer(named: "coordinates")
        let fromBuffer = coords.map { $0.shape.count == 2 && $0.shape[0] == list.count && $0.shape[1] == 3 } ?? false
        glasses = list.enumerated().map { k, g in
            let kind = GlassMapPoint.Kind(rawValue: g["kind"]?.string ?? "") ?? .catalog
            let name = g["name"]?.string ?? "?"
            let surfaces = g["surfaces"]?.intArray ?? []
            func coord(_ c: Int, _ key: String) -> Double {
                if fromBuffer, let coords { return coords[k * 3 + c] }
                return g[key]?.nf ?? .nan
            }
            return GlassMapPoint(
                id: GlassMapPoint.id(catalog: g["catalog"]?.string, name: name, kind: kind, surfaces: surfaces),
                catalog: g["catalog"]?.string, name: name, kind: kind,
                nd: coord(0, "nd"), vd: coord(1, "vd"), pgf: coord(2, "pgf"),
                status: GlassStatus(json: g["status"]), surfaces: surfaces,
                excludeSubstitution: g["exclude_substitution"]?.bool)
        }
    }

    /// The design's glasses, labelled with their surfaces.
    var designHighlights: [GlassHighlight] {
        glasses.filter(\.inDesign).map { GlassHighlight(id: $0.id, label: $0.surfaceLabel) }
    }

    var catalogs: [String] {
        var seen = Set<String>()
        return glasses.compactMap(\.catalog).filter { seen.insert($0).inserted }
    }
}

// MARK: - Image simulation

/// The sRGB transfer function (IEC 61966-2-1).
enum SRGB {
    static func encode(_ linear: Double) -> Double {
        guard linear.isFinite else { return 0 }
        let x = min(max(linear, 0), 1)
        return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055
    }

    static func decode(_ code: Double) -> Double {
        let x = min(max(code, 0), 1)
        return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4)
    }

    /// 8-bit code value of a linear sample (clamped, rounded).
    static func byte(_ linear: Double) -> UInt8 { UInt8((encode(linear) * 255).rounded()) }
}

/// A linear-light raster with row 0 at the low-y edge (the engine's image
/// and buffer-source convention), 1 or 3 interleaved channels.
struct LinearImage: Sendable, Equatable {
    var width: Int
    var height: Int
    var channels: Int
    var data: [Double]

    var isValid: Bool { width > 0 && height > 0 && [1, 3].contains(channels) && data.count == width * height * channels }

    /// RGBA8 sRGB bytes with row 0 at the top (flipped), after scaling by
    /// `gain` (2^exposure).
    func rgba8(gain: Double = 1) -> [UInt8] {
        var px = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            let row = height - 1 - y
            for x in 0..<width {
                let s = (y * width + x) * channels, o = (row * width + x) * 4
                let r = data[s] * gain
                let g = channels == 3 ? data[s + 1] * gain : r
                let b = channels == 3 ? data[s + 2] * gain : r
                px[o] = SRGB.byte(r)
                px[o + 1] = SRGB.byte(g)
                px[o + 2] = SRGB.byte(b)
            }
        }
        return px
    }

    func cgImage(gain: Double = 1) -> CGImage? {
        guard isValid else { return nil }
        let data = Data(rgba8(gain: gain)) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// The fraction of samples above 1 after `gain` (clipped highlights).
    func clippedFraction(gain: Double = 1) -> Double {
        guard !data.isEmpty else { return 0 }
        return Double(data.lazy.filter { $0 * gain > 1 }.count) / Double(data.count)
    }

    /// The image source as the engine's settings expect it.
    var sourceJSON: JSONValue {
        .object([
            "kind": .string("buffer"), "width": .integer(Int64(width)), "height": .integer(Int64(height)),
            "channels": .integer(Int64(channels)), "data": .array(data.map { .number($0) }),
        ])
    }

    init(width: Int, height: Int, channels: Int, data: [Double]) {
        self.width = width
        self.height = height
        self.channels = channels
        self.data = data
    }

    /// A `{kind: buffer}` source from settings.
    init?(source: JSONValue?) {
        guard let s = source, s["kind"]?.string == "buffer", let w = s["width"]?.int, let h = s["height"]?.int,
            let c = s["channels"]?.int, let d = s["data"]?.array
        else { return nil }
        self.init(width: w, height: h, channels: c, data: d.map { $0.doubleValue ?? 0 })
        if !isValid { return nil }
    }

    /// One of the engine's built-in test patterns at `width × height`,
    /// sample for sample as the engine draws it (`pattern_value` in
    /// crates/mklight-wave/src/image_simulation.rs): display only, so the
    /// source can be compared with the simulated image.
    static func pattern(_ name: String, width w: Int, height h: Int) -> LinearImage? {
        guard w > 0, h > 0, ImageSimulationData.patterns.contains(where: { $0.id == name }) else { return nil }
        let font: [[Int]] = [[17, 27, 21, 17, 17, 17, 17], [17, 18, 20, 24, 20, 18, 17], [16, 16, 16, 16, 16, 16, 31]]
        var data = [Double](repeating: 1, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let u = (Double(x) + 0.5) / Double(w), v = (Double(y) + 0.5) / Double(h)
                let value: Double
                switch name {
                case "grid":
                    let fu = (u * 8).truncatingRemainder(dividingBy: 1), fv = (v * 8).truncatingRemainder(dividingBy: 1)
                    value = fu < 0.1 || fv < 0.1 ? 0 : 1
                case "siemens_star":
                    value = cos(atan2(v - 0.5, u - 0.5) * 24) > 0 ? 1 : 0
                case "slanted_edge":
                    value = u > 0.5 + 0.1 * (v - 0.5) ? 1 : 0
                default:  // text_chart
                    let band = min(Int(v * 3), 2)
                    let scale = [4, 2, 1][band]
                    let gx = x / scale, gy = y / scale
                    let glyph = (gx / 6) % 3
                    value = gy % 9 < 7 && gx % 6 < 5 && font[glyph][gy % 9] & (1 << (4 - gx % 6)) != 0 ? 0 : 1
                }
                data[y * w + x] = value
            }
        }
        return LinearImage(width: w, height: h, channels: 1, data: data)
    }
}

/// The `image_simulation` result and the source it was run with.
struct ImageSimulationData: Sendable {
    struct Node: Sendable, Hashable {
        var field: SIMD2<Double>
        var ideal: SIMD2<Double>
        var real: SIMD2<Double>
        var relativeIllumination: Double

        /// Distortion of the node in percent of its ideal radius.
        var distortionPercent: Double {
            let ri = (ideal.x * ideal.x + ideal.y * ideal.y).squareRoot()
            guard ri > 1e-12 else { return .nan }
            let rr = (real.x * real.x + real.y * real.y).squareRoot()
            return (rr - ri) / ri * 100
        }
    }

    enum Source: Sendable, Equatable {
        case pattern(String)
        case buffer(LinearImage)
        case path(String)
    }

    static let patterns: [(id: String, title: String, symbol: String)] = [
        ("grid", "Grid", "grid"), ("siemens_star", "Siemens star", "sun.max"),
        ("slanted_edge", "Slanted edge", "righttriangle"), ("text_chart", "Text chart", "textformat"),
    ]

    var image: LinearImage
    var pixelSize: SIMD2<Double>
    var origin: SIMD2<Double>
    var nodes: [Node]
    var fieldGrid: Int
    var source: Source

    init?(output: AnalysisOutput) {
        let j = output.json
        guard let b = output.buffer(ref: j["image"]) ?? output.buffer(named: "image"), b.shape.count == 3 else { return nil }
        let h = b.shape[0], w = b.shape[1], c = b.shape[2]
        image = LinearImage(width: w, height: h, channels: c, data: b.doubles)
        guard image.isValid else { return nil }
        let px = j["pixel_size_mm"]?.nfArray ?? [.nan, .nan]
        let o = j["origin_mm"]?.nfArray ?? [.nan, .nan]
        pixelSize = px.count == 2 ? SIMD2(px[0], px[1]) : SIMD2(.nan, .nan)
        origin = o.count == 2 ? SIMD2(o[0], o[1]) : SIMD2(.nan, .nan)
        fieldGrid = j["field_grid"]?.int ?? 0
        nodes = (j["field_mapping"]?.array ?? []).compactMap { n in
            guard let f = n["field"]?.nfArray, let i = n["ideal_mm"]?.nfArray, let r = n["real_mm"]?.nfArray,
                f.count == 2, i.count == 2, r.count == 2
            else { return nil }
            return Node(
                field: SIMD2(f[0], f[1]), ideal: SIMD2(i[0], i[1]), real: SIMD2(r[0], r[1]),
                relativeIllumination: n["relative_illumination"]?.nf ?? .nan)
        }
        source = Self.source(output.runInfo.settings["source"])
    }

    static func source(_ s: JSONValue?) -> Source {
        switch s?["kind"]?.string {
        case "buffer": LinearImage(source: s).map { .buffer($0) } ?? .pattern("grid")
        case "path": .path(s?["path"]?.string ?? "")
        default: .pattern(s?["pattern"]?.string ?? "grid")
        }
    }

    /// The largest node distortion (signed, percent, by magnitude).
    var maxDistortionPercent: Double? {
        nodes.map(\.distortionPercent).filter(\.isFinite).max { abs($0) < abs($1) }
    }

    var minRelativeIllumination: Double? { nodes.map(\.relativeIllumination).filter(\.isFinite).min() }

    /// The source raster at the output size, as the engine samples it:
    /// patterns are drawn, buffers are the run's own.
    var sourceImage: LinearImage? {
        switch source {
        case .pattern(let p): LinearImage.pattern(p, width: image.width, height: image.height)
        case .buffer(let b): b
        case .path: nil
        }
    }

    var sourceTitle: String {
        switch source {
        case .pattern(let p): (Self.patterns.first { $0.id == p }?.title ?? p) + " pattern"
        case .buffer: "Imported image"
        case .path(let p): (p as NSString).lastPathComponent
        }
    }
}
