import Foundation
import MKLightEngine
import MKLightModel

/// Reads the engine's `layout` result (docs/RESULTS.md, docs/ANALYSIS.md
/// §2.5: polylines in typed buffers, global frame) into the `LayoutScene`
/// the Metal viewport draws. Glass fills are built from each element's
/// front and back profiles in the meridional section.
enum EngineLayout {
    static let byName: [String: LayoutScene.PolylineKind] = [
        "surface_profile": .surfaceProfile, "surface_rim": .surfaceRim, "element_edge": .elementEdge, "stop": .stop,
        "image_plane": .imagePlane, "ray": .ray, "axis": .axis, "element_outline": .elementEdge,
    ]

    /// `nil` when the output is not a layout the reader understands.
    static func scene(from output: AnalysisOutput?) -> LayoutScene? {
        guard let output else { return nil }
        if case .layout(let s) = output.result { return s }
        let json = output.json
        let precise = output.buffer(named: "vertices_f64")
        let rough = output.buffer(named: "vertices")
        guard let offsets = output.buffer(named: "offsets"), let kinds = output.buffer(named: "kinds"),
            let verts = precise ?? rough, verts.shape.last == 3
        else { return nil }
        let origin = json["origin"]?.nfArray ?? [0, 0, 0]
        let shift = precise == nil && origin.count == 3 ? origin : [0, 0, 0]
        let surf = output.buffer(named: "surfaces"), fld = output.buffer(named: "fields")
        let wav = output.buffer(named: "wavelengths"), flags = output.buffer(named: "flags")
        let n = kinds.count
        guard offsets.count == n + 1 else { return nil }
        let names = json["kind_names"]?.array?.compactMap(\.string) ?? []
        var polylines: [LayoutScene.Polyline] = []
        polylines.reserveCapacity(n)
        for i in 0..<n {
            let a = Int(offsets[i]), b = Int(offsets[i + 1])
            guard a >= 0, b <= verts.rows, a <= b else { continue }
            let pts = (a..<b).map { k in
                Point3(verts[k * 3] + shift[0], verts[k * 3 + 1] + shift[1], verts[k * 3 + 2] + shift[2])
            }
            let f = flags.map { Int($0[i]) } ?? 1
            func idx(_ buf: ResultBuffer?) -> Int? {
                guard let buf, i < buf.count else { return nil }
                let v = Int(buf[i])
                return v >= 0 ? v : nil
            }
            polylines.append(
                LayoutScene.Polyline(
                    kind: names.indices.contains(Int(kinds[i])) ? byName[names[Int(kinds[i])]] ?? .elementEdge : .elementEdge, points: pts, inSection: f & 1 != 0, surface: idx(surf),
                    field: idx(fld), wavelength: idx(wav), vignetted: f & 2 != 0))
        }
        let surfaces: [LayoutScene.SurfaceInfo] = (json["surfaces"]?.array ?? []).compactMap { s in
            guard let index = s["index"]?.int, let id = s["id"]?.int, let v = s["vertex"]?.nfArray, v.count == 3 else { return nil }
            return LayoutScene.SurfaceInfo(
                index: index, id: SurfaceID(UInt64(id)), vertex: Point3(v[0], v[1], v[2]),
                semiDiameter: s["semi_diameter"]?.nf ?? .nan, semiDiameterIsAuto: s["semi_diameter_is_auto"]?.bool ?? false)
        }
        // Glass fills: the in-section profiles of each element's first and
        // last surface, ordered by height so their samples pair up.
        func profile(_ surface: Int) -> [Point3]? {
            polylines.first { $0.kind == .surfaceProfile && $0.inSection && $0.surface == surface }?.points
                .sorted { $0.y < $1.y }
        }
        let elements: [LayoutScene.Element] = (json["elements"]?.array ?? []).compactMap { e in
            guard let first = e["first_surface"]?.int, let last = e["last_surface"]?.int,
                let front = profile(first), let back = profile(last)
            else { return nil }
            return LayoutScene.Element(firstSurface: first, lastSurface: last, front: front, back: resample(back, like: front))
        }
        let lo = json[path: "bounds", "min"]?.nfArray ?? [], hi = json[path: "bounds", "max"]?.nfArray ?? []
        let bounds: Bounds3
        if lo.count == 3, hi.count == 3, (lo + hi).allSatisfy(\.isFinite) {
            bounds = Bounds3(min: Point3(lo[0], lo[1], lo[2]), max: Point3(hi[0], hi[1], hi[2]))
        } else {
            let all = polylines.flatMap(\.points)
            guard !all.isEmpty else { return nil }
            bounds = Bounds3(
                min: Point3(all.map(\.x).min()!, all.map(\.y).min()!, all.map(\.z).min()!),
                max: Point3(all.map(\.x).max()!, all.map(\.y).max()!, all.map(\.z).max()!))
        }
        return LayoutScene(
            polylines: polylines, elements: elements, surfaces: surfaces, bounds: bounds,
            wavelengthsUm: json["wavelengths_um"]?.nfArray ?? [], fieldCount: json["field_count"]?.int ?? 1)
    }

    /// `points` resampled to the heights of `like` (linear in y), so a fill
    /// can pair them; heights outside its range clamp to the ends.
    static func resample(_ points: [Point3], like: [Point3]) -> [Point3] {
        guard points.count >= 2 else { return like.map { _ in points.first ?? Point3(0, 0, 0) } }
        let lo = points.first!.y, hi = points.last!.y
        return like.map { q in
            let y = min(max(q.y, lo), hi)
            var k = 0
            while k < points.count - 2 && points[k + 1].y < y { k += 1 }
            let a = points[k], b = points[k + 1]
            let t = b.y == a.y ? 0 : (y - a.y) / (b.y - a.y)
            return Point3(a.x + (b.x - a.x) * t, q.y, a.z + (b.z - a.z) * t)
        }
    }

    /// Automatic semi-diameters the engine drew, by surface id.
    static func autoSemiDiameters(_ scene: LayoutScene?) -> [SurfaceID: Double] {
        Dictionary(
            (scene?.surfaces ?? []).filter { $0.semiDiameter.isFinite }.map { ($0.id, $0.semiDiameter) },
            uniquingKeysWith: { a, _ in a })
    }
}
