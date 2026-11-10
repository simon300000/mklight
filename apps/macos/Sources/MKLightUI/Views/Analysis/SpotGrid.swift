import MKLightEngine
import MKLightModel
import SwiftUI

// The spot diagram as a field × focus grid: the data read once from the
// engine result, the marker style, the grid layout and the cell view.

// MARK: - Data

/// The spot diagram result as a grid of cells: one row per field, one
/// column per focus plane. Every cell's points are computed once here.
struct SpotGrid: Sendable {
    /// The points of one wavelength in one cell, relative to the cell's
    /// reference (mm).
    struct Group: Sendable {
        var wavelength: Int
        var points: [SIMD2<Double>]
    }

    /// One (field, plane) cell.
    struct Cell: Sendable, Identifiable {
        var id: Int
        var column: Int
        /// Absolute defocus of the plane from the image surface (mm).
        var defocus: Double
        var groups: [Group]
        var pointCount: Int
        /// RMS radius about the reference at this plane (mm).
        var rms: Double
        /// Largest distance from the reference at this plane (mm).
        var geo: Double
        /// Largest finite point radius (mm); `geo` when there are no points.
        var extent: Double
    }

    /// One field.
    struct Row: Sendable, Identifiable {
        var id: Int
        /// "Field 1".
        var title: String
        /// The field value, e.g. "14°".
        var value: String
        var cells: [Cell]

        /// "Field 1 · 14°".
        var label: String { value.isEmpty ? title : title + " · " + value }
        /// The points at the base defocus, all wavelengths.
        var points: [SIMD2<Double>] { cells.first { $0.column == baseColumn }?.groups.flatMap(\.points) ?? [] }
        fileprivate var baseColumn: Int
    }

    /// Defocus of each column (mm, absolute from the image surface).
    var planes: [Double]
    /// Planes the engine sampled (before thinning to `planes`).
    var sampledPlanes: Int
    /// The column at the base defocus, if one is.
    var baseColumn: Int?
    var rows: [Row]
    var airy: Double
    var wavelengths: [Int]
    var wavelengthsUm: [Double]
    /// "chief" or "centroid".
    var reference: String

    /// Columns drawn at most; more sampled planes are thinned evenly.
    static let maxColumns = 9

    /// Reads a spot diagram result; `nil` when `fields` is missing.
    init?(output: AnalysisOutput, context: AnalysisContext) {
        let j = output.json
        guard let list = j["fields"]?.array, !list.isEmpty else { return nil }
        let pts = output.buffer(named: "points"), pw = output.buffer(named: "point_wavelength")
        let slope = output.buffer(named: "point_slope")
        // `points` is `[n, 2]`.
        guard pts.map({ $0.count == $0.rows * 2 }) ?? true else { return nil }
        wavelengthsUm = context.wavelengthsUm(of: output)
        wavelengths = j["wavelengths"]?.intArray ?? Array(wavelengthsUm.indices)
        airy = j["airy_radius"]?.nf ?? .nan
        let base = j["defocus"]?.nf ?? 0
        reference = output.runInfo.settings["reference"]?.string ?? "chief"
        let rmsKey = reference == "centroid" ? "rms_radius_centroid" : "rms_radius_chief"

        // Planes from the first field's through-focus list; the base plane
        // alone without one (or without slopes to move the points).
        let focus = list[0]["through_focus"]?.array ?? []
        let sampled = focus.compactMap { $0["defocus"]?.nf }
        let hasFocus = slope != nil && !sampled.isEmpty
        let all = hasFocus ? sampled : [base]
        let picked = Self.thin(all.count, to: Self.maxColumns, keeping: Self.nearest(all, to: base))
        planes = picked.map { all[$0] }
        sampledPlanes = all.count
        let baseIndex = planes.firstIndex { abs($0 - base) <= 1e-12 * max(1, abs(base)) }
        baseColumn = baseIndex

        let planeList = planes
        rows = list.enumerated().map { k, f in
            let index = f["field"]?.int ?? k
            let entries = f["through_focus"]?.array ?? []
            var range = 0..<0
            if let pts, let r = f["point_range"]?.intArray, r.count == 2, case (let end, false) = r[0].addingReportingOverflow(r[1]) {
                let lo = max(r[0], 0), hi = min(end, pts.rows)
                if lo < hi { range = lo..<hi }
            }
            let cells = planeList.enumerated().map { c, d in
                let delta = d - base
                var byWavelength: [Int: [SIMD2<Double>]] = [:]
                var order: [Int] = []
                var extent = 0.0
                if let pts {
                    for i in range {
                        var p = SIMD2(pts[i * 2], pts[i * 2 + 1])
                        if delta != 0, let slope, i * 2 + 1 < slope.count {
                            p += delta * SIMD2(slope[i * 2], slope[i * 2 + 1])
                        }
                        let w = pw.flatMap { i < $0.count ? Int(exactly: $0[i]) : nil } ?? 0
                        if byWavelength[w] == nil { order.append(w) }
                        byWavelength[w, default: []].append(p)
                        let r = (p * p).sum().squareRoot()
                        if r.isFinite { extent = max(extent, r) }
                    }
                }
                let groups = order.map { Group(wavelength: $0, points: byWavelength[$0] ?? []) }
                let count = groups.reduce(0) { $0 + $1.points.count }
                // Statistics: the plane's through-focus entry, else (base plane
                // without through-focus) the field's own values.
                let entry = hasFocus ? entries[safe: picked[c]] : nil
                let isBase = c == baseIndex
                let rms = entry?[rmsKey]?.nf ?? (isBase ? f["rms_radius"]?.nf : nil) ?? .nan
                let engineGeo = f["geo_radius"]?.nf ?? .nan
                let geo = isBase && engineGeo.isFinite ? engineGeo : (count > 0 ? extent : .nan)
                return Cell(
                    id: k * planeList.count + c, column: c, defocus: d, groups: groups, pointCount: count,
                    rms: rms, geo: geo, extent: count > 0 ? extent : (geo.isFinite ? geo : 0))
            }
            return Row(
                id: index, title: "Field \(index + 1)", value: context.fieldLabel(f["field_value"]?.nfArray),
                cells: cells, baseColumn: baseIndex ?? 0)
        }
    }

    /// Box half-width of each row (mm): the largest cell extent (or the Airy
    /// radius) × 1.15, over the whole grid or per row.
    static func halfWidths(extents: [[Double]], airy: Double, scale: SpotScale) -> [Double] {
        let a = airy.isFinite ? airy : 0
        func hw(_ e: [Double]) -> Double { max(e.filter(\.isFinite).max() ?? 0, a, 1e-6) * 1.15 }
        switch scale {
        case .grid:
            let w = hw(extents.flatMap { $0 })
            return extents.map { _ in w }
        case .field:
            return extents.map(hw)
        }
    }

    func halfWidths(_ scale: SpotScale) -> [Double] {
        Self.halfWidths(extents: rows.map { $0.cells.map(\.extent) }, airy: airy, scale: scale)
    }

    /// Indices of `count` items thinned evenly to at most `limit`, always
    /// keeping the first, the last and `keeping`.
    static func thin(_ count: Int, to limit: Int, keeping: Int?) -> [Int] {
        guard count > limit, limit >= 2 else { return Array(0..<count) }
        var set = Set((0..<limit).map { Int((Double($0) * Double(count - 1) / Double(limit - 1)).rounded()) })
        if let keeping, (0..<count).contains(keeping), !set.contains(keeping) {
            // Swap the nearest interior pick for the kept index.
            if let near = set.filter({ $0 != 0 && $0 != count - 1 }).min(by: { abs($0 - keeping) < abs($1 - keeping) }) {
                set.remove(near)
            }
            set.insert(keeping)
        }
        return set.sorted()
    }

    private static func nearest(_ values: [Double], to x: Double) -> Int? {
        values.indices.min { abs(values[$0] - x) < abs(values[$1] - x) }
    }

    /// A plane's column label: "0", "+0.08 mm", "−0.0125 mm", "+5 µm".
    static func planeLabel(_ d: Double) -> String {
        guard d.isFinite else { return Fmt.short(d) }
        let a = abs(d)
        if a < 1e-9 { return "0" }
        let sign = d < 0 ? "\u{2212}" : "+"
        if a < 1e-3 { return sign + String(format: "%.3g µm", a * 1000) }
        return sign + String(format: "%.3g mm", a)
    }
}

extension Array {
    fileprivate subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}

/// The box scale of the spot grid.
enum SpotScale: String, CaseIterable, Identifiable, Sendable {
    /// One half-width for every cell.
    case grid
    /// One half-width per field (row).
    case field

    var id: String { rawValue }
    var title: String {
        switch self {
        case .grid: "Common"
        case .field: "Per field"
        }
    }
}

// MARK: - Marker

/// Point marker size and opacity of a spot cell.
enum SpotMarker {
    /// Radius (pt) and opacity of the markers of a cell of side `cell` pt
    /// holding `points` points: radius 0.3 · cell / √n, shrinking with density
    /// and growing with the cell, within 0.5…1.25 pt; the opacity falls from
    /// 0.85 (≤ 200 points) to 0.5 (≥ 1500 points), so dense spots keep
    /// their structure.
    static func style(points: Int, cell: CGFloat) -> (radius: CGFloat, opacity: Double) {
        let n = Double(max(points, 1))
        let radius = min(max(0.3 * cell / CGFloat(n.squareRoot()), 0.5), 1.25)
        let t = min(max((log(n) - log(200)) / (log(1500) - log(200)), 0), 1)
        return (radius, 0.85 - 0.35 * t)
    }
}

// MARK: - Layout

/// Cell size of the spot grid: all columns side by side, rows scrolling.
enum SpotGridLayout {
    static let minCell: CGFloat = 48
    /// Rows scroll rather than shrink cells below this to fit the height.
    static let comfortableCell: CGFloat = 120
    static let maxCell: CGFloat = 420
    static let spacing: CGFloat = Space.s
    static let columnHeader: CGFloat = 18
    /// The caption line under each plot.
    static let caption: CGFloat = 16

    struct Metrics: Equatable {
        var cell: CGFloat
        var rowHeader: CGFloat
        /// The columns do not fit at `minCell`: the grid scrolls sideways.
        var scrollsHorizontally: Bool
    }

    /// Width of the row-header column for a pane `width` wide.
    static func rowHeaderWidth(for width: CGFloat) -> CGFloat { width < 420 ? 60 : 84 }

    /// The cell side for `columns` × `rows` cells in `size`: as wide as the
    /// width allows, capped so the grid fits the height when that leaves
    /// cells of at least `comfortableCell`, never below `minCell` nor above
    /// `maxCell`.
    static func metrics(columns: Int, rows: Int, in size: CGSize) -> Metrics {
        let cols = CGFloat(max(columns, 1)), rws = CGFloat(max(rows, 1))
        let header = rowHeaderWidth(for: size.width)
        let byWidth = (size.width - header - cols * spacing) / cols
        let byHeight = (size.height - columnHeader - spacing - rws * (caption + 2 + spacing)) / rws
        var cell = min(byWidth, max(byHeight, comfortableCell), maxCell)
        let scrolls = cell < minCell
        cell = max(cell, minCell).rounded(.down)
        return Metrics(cell: cell, rowHeader: header, scrollsHorizontally: scrolls)
    }
}

// MARK: - Views

/// The field × focus grid with its headers.
struct SpotGridView: View {
    let grid: SpotGrid
    let scale: SpotScale

    var body: some View {
        let halfWidths = grid.halfWidths(scale)
        GeometryReader { geo in
            let m = SpotGridLayout.metrics(columns: grid.planes.count, rows: grid.rows.count, in: geo.size)
            ScrollView(m.scrollsHorizontally ? [.horizontal, .vertical] : .vertical) {
                Grid(alignment: .top, horizontalSpacing: SpotGridLayout.spacing, verticalSpacing: SpotGridLayout.spacing) {
                    GridRow {
                        Text(grid.planes.count > 1 ? "Defocus" : "")
                            .font(.caption2).foregroundStyle(.tertiary)
                            .frame(width: m.rowHeader, alignment: .leading)
                        ForEach(Array(grid.planes.enumerated()), id: \.offset) { c, d in
                            columnHeader(c, d).frame(width: m.cell, height: SpotGridLayout.columnHeader)
                        }
                    }
                    ForEach(Array(grid.rows.enumerated()), id: \.element.id) { r, row in
                        GridRow {
                            rowHeader(row, halfWidth: halfWidths[r]).frame(width: m.rowHeader, alignment: .leading)
                            ForEach(row.cells) { cell in
                                SpotCellView(
                                    cell: cell, label: row.label, halfWidth: halfWidths[r], airy: grid.airy,
                                    wavelengthsUm: grid.wavelengthsUm, size: m.cell,
                                    showsGeo: cell.column == grid.baseColumn)
                            }
                        }
                    }
                }
                .padding(.trailing, Space.xs)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func columnHeader(_ c: Int, _ d: Double) -> some View {
        let isBase = c == grid.baseColumn && grid.planes.count > 1
        return Text(SpotGrid.planeLabel(d))
            .font(.caption.monospacedDigit().weight(isBase ? .semibold : .regular))
            .foregroundStyle(isBase ? .primary : .secondary)
            .lineLimit(1).minimumScaleFactor(0.7)
            .help(isBase ? "Defocus the points were computed at" : "Defocus from the image surface")
    }

    private func rowHeader(_ row: SpotGrid.Row, halfWidth: Double) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(row.title).font(.caption.weight(.semibold)).lineLimit(1)
            Text(row.value).font(.caption2).foregroundStyle(.secondary).lineLimit(2).minimumScaleFactor(0.8)
            if scale == .field {
                Text("±\(Fmt.length(halfWidth))").font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                    .lineLimit(1).minimumScaleFactor(0.7)
            }
        }
        .padding(.top, Space.l)
    }
}

/// One spot plot with its RMS caption.
struct SpotCellView: View {
    let cell: SpotGrid.Cell
    let label: String
    let halfWidth: Double
    let airy: Double
    let wavelengthsUm: [Double]
    let size: CGFloat
    var showsGeo = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(spacing: 2) {
            Canvas { ctx, sz in draw(&ctx, sz) }
                .frame(width: size, height: size)
                .accessibilityLabel(
                    "\(label), defocus \(SpotGrid.planeLabel(cell.defocus)), spot RMS \(Fmt.length(cell.rms)), GEO \(Fmt.length(cell.geo))")
            ViewThatFits(in: .horizontal) {
                if showsGeo { Text("RMS \(Fmt.length(cell.rms)) · GEO \(Fmt.length(cell.geo))").fixedSize() }
                Text("RMS \(Fmt.length(cell.rms))").fixedSize()
                Text(Fmt.length(cell.rms)).fixedSize()
                Text(Fmt.length(cell.rms)).minimumScaleFactor(0.6)
            }
            .lineLimit(1)
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
            .frame(width: size, height: SpotGridLayout.caption)
        }
    }

    private func draw(_ ctx: inout GraphicsContext, _ sz: CGSize) {
        let s = min(sz.width, sz.height) / 2 / CGFloat(halfWidth)
        let c = CGPoint(x: sz.width / 2, y: sz.height / 2)
        ctx.stroke(Path(CGRect(origin: .zero, size: sz)), with: .color(.secondary.opacity(0.25)), lineWidth: 0.5)
        var cross = Path()
        cross.move(to: CGPoint(x: c.x, y: 0))
        cross.addLine(to: CGPoint(x: c.x, y: sz.height))
        cross.move(to: CGPoint(x: 0, y: c.y))
        cross.addLine(to: CGPoint(x: sz.width, y: c.y))
        ctx.stroke(cross, with: .color(.secondary.opacity(0.2)), lineWidth: 0.5)
        if airy.isFinite {
            let r = CGFloat(airy) * s
            ctx.stroke(
                Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                with: .color(.secondary), style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
        }
        let marker = SpotMarker.style(points: cell.pointCount, cell: min(sz.width, sz.height))
        let r = marker.radius, d = 2 * marker.radius
        let dark = scheme == .dark
        // One path per wavelength, filled once.
        for g in cell.groups {
            var path = Path()
            for p in g.points where p.x.isFinite && p.y.isFinite {
                path.addEllipse(in: CGRect(x: c.x + CGFloat(p.x) * s - r, y: c.y - CGFloat(p.y) * s - r, width: d, height: d))
            }
            let um = wavelengthsUm.indices.contains(g.wavelength) ? wavelengthsUm[g.wavelength] : 0.55
            let rgb = Palette.wavelengthRGB(um, dark: dark)
            ctx.fill(path, with: .color(Color(.sRGB, red: rgb.x, green: rgb.y, blue: rgb.z, opacity: marker.opacity)))
        }
    }
}
