import CoreGraphics
import MKLightEngine
import simd

/// GPU instance data for one polyline segment (matches `Segment` in
/// Viewport.metal: four float4, 64 bytes).
struct SegmentInstance {
    var p0: SIMD4<Float>
    var p1: SIMD4<Float>
    var color: SIMD4<Float>
    /// x: width (pixels), y: dash period (pixels, 0 = solid).
    var params: SIMD4<Float>
}

/// Matches `FillVertex` in Viewport.metal.
struct FillVertex {
    var position: SIMD4<Float>
    var color: SIMD4<Float>
}

/// How the scene is drawn: colours, widths and which items are visible.
struct ViewportStyle: Equatable {
    var mode: ViewportMode
    var coloring: RayColoring
    var selectedSurface: Int?
    var stale: Bool
    var dark: Bool
    /// Device pixels per point.
    var scale: Float
    /// Selection highlight (the system accent colour), sRGB.
    var accent: SIMD3<Float> = SIMD3(0.16, 0.47, 0.84)
}

/// The coordinate frame vertex positions are uploaded in. Positions are
/// transformed in double precision and only then rounded to float.
///
/// float32 has a 24-bit significand, so a vertex at distance `D` (mm) from
/// the frame origin is off by up to about `D·2⁻²⁴` mm, which the view
/// magnifies to `D·2⁻²⁴·pointsPerMM` points; the GPU matrix adds errors of
/// the same order. Near the scene that is negligible, but at high zoom far
/// from the frame origin it becomes visible jitter. So at such zooms the frame
/// is moved to the camera ("relative to eye"), positions are rotated into
/// view axes on the CPU, and segments and fills are clipped in double to a
/// window a little larger than the view. Every uploaded coordinate is then
/// within about one viewport of the origin, and the error stays a tiny
/// fraction of a pixel at any zoom.
struct GeometryFrame: Equatable {
    /// World point (mm) subtracted from every position.
    var origin: SIMD3<Double>
    /// World→view rotation applied on the CPU, or `nil` to keep world axes
    /// (the GPU matrix then rotates).
    var rotation: simd_double3x3?
    /// Half extents (mm) of the view-space x/y window geometry is clipped to.
    var clipHalfExtent: SIMD2<Double>?

    func apply(_ p: Point3) -> SIMD3<Double> {
        let q = SIMD3(p.x, p.y, p.z) - origin
        return rotation.map { $0 * q } ?? q
    }
}

/// Converts an engine `LayoutScene` into GPU buffers in a `GeometryFrame`.
enum SceneGeometry {
    static func origin(of scene: LayoutScene) -> SIMD3<Double> {
        let c = scene.bounds.center
        return SIMD3(c.x, c.y, c.z)
    }

    /// Largest projected float error (points) accepted before switching to a
    /// camera-relative, clipped frame.
    static let maxFloatErrorPoints = 0.02

    /// The frame to build `scene` in for this camera: scene-centred (camera
    /// moves only change the GPU matrix) while the float error is negligible,
    /// else camera-relative, view-aligned and clipped (rebuilt whenever the
    /// camera changes, which is cheap for layout scenes).
    static func frame(for scene: LayoutScene, camera: ViewportCamera, mode: ViewportMode, size: CGSize) -> GeometryFrame {
        let o = origin(of: scene)
        let reach = sceneReach(scene, from: o) + simd_length(camera.center - o)
        if reach * camera.pointsPerMM * 0x1p-24 <= maxFloatErrorPoints {
            return GeometryFrame(origin: o, rotation: nil, clipHalfExtent: nil)
        }
        let half = SIMD2(Double(max(size.width, 1)), Double(max(size.height, 1))) / camera.pointsPerMM
        return GeometryFrame(origin: camera.center, rotation: ViewportMath.rotation(camera, mode: mode), clipHalfExtent: half)
    }

    /// Distance from `o` to the farthest corner of the scene bounds.
    static func sceneReach(_ scene: LayoutScene, from o: SIMD3<Double>) -> Double {
        let b = scene.bounds
        let lo = SIMD3(b.min.x, b.min.y, b.min.z) - o, hi = SIMD3(b.max.x, b.max.y, b.max.z) - o
        return simd_length(simd_max(simd_abs(lo), simd_abs(hi)))
    }

    /// Clips segment `a`–`b` to `|x| ≤ h.x, |y| ≤ h.y` (Liang–Barsky); depth
    /// is not clipped.
    static func clip(_ a: SIMD3<Double>, _ b: SIMD3<Double>, to h: SIMD2<Double>) -> (SIMD3<Double>, SIMD3<Double>)? {
        let d = b - a
        var t0 = 0.0, t1 = 1.0
        for (p, q) in [(-d.x, a.x + h.x), (d.x, h.x - a.x), (-d.y, a.y + h.y), (d.y, h.y - a.y)] {
            if p == 0 {
                if q < 0 { return nil }
                continue
            }
            let r = q / p
            if p < 0 {
                if r > t1 { return nil }
                t0 = max(t0, r)
            } else {
                if r < t0 { return nil }
                t1 = min(t1, r)
            }
        }
        return (t0 == 0 ? a : a + t0 * d, t1 == 1 ? b : a + t1 * d)
    }

    /// Clips a convex polygon to the same window (Sutherland–Hodgman).
    static func clip(polygon: [SIMD3<Double>], to h: SIMD2<Double>) -> [SIMD3<Double>] {
        var poly = polygon
        let planes: [(SIMD3<Double>) -> Double] = [{ h.x - $0.x }, { h.x + $0.x }, { h.y - $0.y }, { h.y + $0.y }]
        for inside in planes {
            guard let last = poly.last else { break }
            var out: [SIMD3<Double>] = []
            var prev = last, dPrev = inside(last)
            for cur in poly {
                let dCur = inside(cur)
                if (dCur >= 0) != (dPrev >= 0) { out.append(prev + (dPrev / (dPrev - dCur)) * (cur - prev)) }
                if dCur >= 0 { out.append(cur) }
                prev = cur
                dPrev = dCur
            }
            poly = out
        }
        return poly
    }

    /// Scene-centred build (world axes, no clipping).
    static func build(_ scene: LayoutScene, style: ViewportStyle) -> (segments: [SegmentInstance], fills: [FillVertex]) {
        build(scene, style: style, frame: GeometryFrame(origin: origin(of: scene), rotation: nil, clipHalfExtent: nil))
    }

    static func build(_ scene: LayoutScene, style: ViewportStyle, frame: GeometryFrame) -> (segments: [SegmentInstance], fills: [FillVertex]) {
        func f4(_ q: SIMD3<Double>) -> SIMD4<Float> { SIMD4(Float(q.x), Float(q.y), Float(q.z), 1) }
        let ink: SIMD3<Float> = style.dark ? SIMD3(0.88, 0.89, 0.91) : SIMD3(0.13, 0.14, 0.16)
        let faint: SIMD3<Float> = style.dark ? SIMD3(0.55, 0.56, 0.60) : SIMD3(0.55, 0.56, 0.58)
        let accent = style.accent
        let s = style.scale

        var segs: [SegmentInstance] = []
        segs.reserveCapacity(scene.polylines.reduce(0) { $0 + max($1.points.count - 1, 0) })
        func add(_ line: LayoutScene.Polyline, color: SIMD4<Float>, width: Float, dash: Float = 0) {
            guard line.points.count >= 2 else { return }
            let params = SIMD4(width * s, dash * s, 0, 0)
            var a = frame.apply(line.points[0])
            for k in 1..<line.points.count {
                let b = frame.apply(line.points[k])
                defer { a = b }
                if let h = frame.clipHalfExtent {
                    guard let (c0, c1) = clip(a, b, to: h) else { continue }
                    segs.append(SegmentInstance(p0: f4(c0), p1: f4(c1), color: color, params: params))
                } else {
                    segs.append(SegmentInstance(p0: f4(a), p1: f4(b), color: color, params: params))
                }
            }
        }

        // Rays first so surfaces draw on top of them.
        for line in scene.polylines where line.kind == .ray && (style.mode == .orbit3D || line.inSection) {
            let rgb: SIMD3<Double> =
                style.coloring == .field
                ? Palette.fieldRGB(line.field ?? 0, dark: style.dark)
                : Palette.wavelengthRGB(
                    scene.wavelengthsUm.indices.contains(line.wavelength ?? -1) ? scene.wavelengthsUm[line.wavelength!] : 0.55,
                    dark: style.dark)
            var a: Float = line.vignetted ? 0.35 : 0.9
            if style.stale { a *= 0.35 }
            add(line, color: SIMD4(SIMD3<Float>(rgb), a), width: 1.1)
        }
        for line in scene.polylines where line.kind != .ray && (style.mode == .orbit3D || line.inSection) {
            let selected = line.surface != nil && line.surface == style.selectedSurface
            switch line.kind {
            case .axis: add(line, color: SIMD4(faint, 0.8), width: 0.8, dash: 10)
            case .surfaceProfile, .elementEdge:
                add(line, color: SIMD4(selected ? accent : ink, 1), width: selected ? 2.4 : 1.4)
            case .surfaceRim: add(line, color: SIMD4(selected ? accent : faint, 0.8), width: selected ? 1.8 : 1)
            case .stop: add(line, color: SIMD4(selected ? accent : ink, 1), width: selected ? 2.8 : 2.2)
            case .imagePlane: add(line, color: SIMD4(selected ? accent : ink, 1), width: selected ? 2.4 : 1.8)
            case .ray: break
            }
        }

        // Glass fill between matched front/back profiles (section view only).
        var fills: [FillVertex] = []
        if style.mode == .section2D {
            let glass: SIMD4<Float> = style.dark ? SIMD4(0.35, 0.60, 0.95, 0.22) : SIMD4(0.45, 0.68, 0.95, 0.25)
            for e in scene.elements where e.front.count == e.back.count && e.front.count >= 2 {
                for k in 0..<(e.front.count - 1) {
                    let a = frame.apply(e.front[k]), b = frame.apply(e.back[k])
                    let c = frame.apply(e.front[k + 1]), d = frame.apply(e.back[k + 1])
                    for tri in [[a, b, c], [c, b, d]] {
                        let poly = frame.clipHalfExtent.map { clip(polygon: tri, to: $0) } ?? tri
                        guard poly.count >= 3 else { continue }
                        for j in 1..<(poly.count - 1) {
                            fills += [poly[0], poly[j], poly[j + 1]].map { FillVertex(position: f4($0), color: glass) }
                        }
                    }
                }
            }
        }
        return (segs, fills)
    }

    /// Points used to fit the camera (only what the mode shows).
    static func fitPoints(_ scene: LayoutScene, mode: ViewportMode) -> [SIMD3<Double>] {
        scene.polylines.filter { mode == .orbit3D || $0.inSection }.flatMap(\.points).map { SIMD3($0.x, $0.y, $0.z) }
    }
}
