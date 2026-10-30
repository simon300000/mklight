import AppKit
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
}

/// Converts an engine `LayoutScene` into GPU buffers. Positions are made
/// relative to the scene centre in double precision before rounding to
/// float, so float32 keeps full resolution near the lens.
enum SceneGeometry {
    static func origin(of scene: LayoutScene) -> SIMD3<Double> {
        let c = scene.bounds.center
        return SIMD3(c.x, c.y, c.z)
    }

    static func build(_ scene: LayoutScene, style: ViewportStyle) -> (segments: [SegmentInstance], fills: [FillVertex]) {
        let o = origin(of: scene)
        func local(_ p: Point3) -> SIMD4<Float> {
            SIMD4(Float(p.x - o.x), Float(p.y - o.y), Float(p.z - o.z), 1)
        }
        let ink: SIMD3<Float> = style.dark ? SIMD3(0.88, 0.89, 0.91) : SIMD3(0.13, 0.14, 0.16)
        let faint: SIMD3<Float> = style.dark ? SIMD3(0.55, 0.56, 0.60) : SIMD3(0.55, 0.56, 0.58)
        let accent: SIMD3<Float> = {
            let c = NSColor.controlAccentColor.usingColorSpace(.sRGB) ?? .systemBlue
            return SIMD3(Float(c.redComponent), Float(c.greenComponent), Float(c.blueComponent))
        }()
        let s = style.scale

        var segs: [SegmentInstance] = []
        segs.reserveCapacity(scene.polylines.reduce(0) { $0 + max($1.points.count - 1, 0) })
        func add(_ line: LayoutScene.Polyline, color: SIMD4<Float>, width: Float, dash: Float = 0) {
            guard line.points.count >= 2 else { return }
            for k in 0..<(line.points.count - 1) {
                segs.append(SegmentInstance(p0: local(line.points[k]), p1: local(line.points[k + 1]), color: color, params: SIMD4(width * s, dash * s, 0, 0)))
            }
        }

        // Rays first so surfaces draw on top of them.
        for line in scene.polylines where line.kind == .ray && (style.mode == .orbit3D || line.inSection) {
            let rgb: SIMD3<Double> =
                style.coloring == .field
                ? Palette.fieldRGBValue(line.field ?? 0)
                : Palette.wavelengthRGB(scene.wavelengthsUm.indices.contains(line.wavelength ?? -1) ? scene.wavelengthsUm[line.wavelength!] : 0.55)
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
                    let a = local(e.front[k]), b = local(e.back[k]), c = local(e.front[k + 1]), d = local(e.back[k + 1])
                    fills += [a, b, c, c, b, d].map { FillVertex(position: $0, color: glass) }
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
