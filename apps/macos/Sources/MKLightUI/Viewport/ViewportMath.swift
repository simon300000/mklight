import Foundation
import MKLightEngine
import simd

/// View transforms for the layout viewport. Orthographic only: optical
/// layouts are read with true proportions.
///
/// World → view: the basis puts world +z (the optical axis) to the right and
/// world +y up, so the 2-D view is the meridional Y-Z section. The 3-D view
/// orbits that basis by `yaw` (about view up) and `pitch` (about view right).
enum ViewportMath {
    /// Rotation from world to view axes for `mode`.
    static func rotation(_ camera: ViewportCamera, mode: ViewportMode) -> simd_double3x3 {
        // Columns are the images of world x, y, z: x → −view z, y → view y, z → view x.
        let basis = simd_double3x3(columns: (SIMD3(0, 0, -1), SIMD3(0, 1, 0), SIMD3(1, 0, 0)))
        guard mode == .orbit3D else { return basis }
        let (cy, sy) = (cos(camera.yaw), sin(camera.yaw))
        let (cp, sp) = (cos(camera.pitch), sin(camera.pitch))
        let yawM = simd_double3x3(columns: (SIMD3(cy, 0, -sy), SIMD3(0, 1, 0), SIMD3(sy, 0, cy)))
        let pitchM = simd_double3x3(columns: (SIMD3(1, 0, 0), SIMD3(0, cp, sp), SIMD3(0, -sp, cp)))
        return pitchM * yawM * basis
    }

    /// Matrix taking scene-centred positions (world − `origin`) to clip space
    /// for a viewport of `size` points. Built in double precision.
    static func viewProjection(
        _ camera: ViewportCamera, mode: ViewportMode, origin: SIMD3<Double>, size: CGSize, depthRadius: Double
    ) -> simd_float4x4 {
        viewProjection(
            camera, mode: mode, frame: GeometryFrame(origin: origin, rotation: nil, clipHalfExtent: nil), size: size,
            depthRadius: depthRadius)
    }

    /// Matrix taking positions in `frame` to clip space. With a
    /// camera-relative, view-aligned frame (deep zoom) it is a pure scale.
    static func viewProjection(
        _ camera: ViewportCamera, mode: ViewportMode, frame: GeometryFrame, size: CGSize, depthRadius: Double
    ) -> simd_float4x4 {
        let rc = rotation(camera, mode: mode)
        let r = frame.rotation.map { rc * $0.transpose } ?? rc  // frame axes → view axes
        let t = rc * (frame.origin - camera.center)  // translation in view space
        let sx = 2 * camera.pointsPerMM / max(Double(size.width), 1)
        let sy = 2 * camera.pointsPerMM / max(Double(size.height), 1)
        let sz = 0.5 / max(depthRadius, 1e-9)
        // clip = S · (R·q + t), z mapped from [−D, D] to [1, 0].
        let m = simd_double4x4(columns: (
            SIMD4(sx * r.columns.0.x, sy * r.columns.0.y, -sz * r.columns.0.z, 0),
            SIMD4(sx * r.columns.1.x, sy * r.columns.1.y, -sz * r.columns.1.z, 0),
            SIMD4(sx * r.columns.2.x, sy * r.columns.2.y, -sz * r.columns.2.z, 0),
            SIMD4(sx * t.x, sy * t.y, 0.5 - sz * t.z, 1)
        ))
        return simd_float4x4(
            SIMD4<Float>(m.columns.0), SIMD4<Float>(m.columns.1), SIMD4<Float>(m.columns.2), SIMD4<Float>(m.columns.3))
    }

    /// A camera that frames `points` in a viewport of `size` points with a
    /// margin, keeping the orbit angles.
    static func fit(_ camera: ViewportCamera, mode: ViewportMode, points: [SIMD3<Double>], size: CGSize) -> ViewportCamera {
        guard !points.isEmpty, size.width > 1, size.height > 1 else { return camera }
        let r = rotation(camera, mode: mode)
        var lo = SIMD3<Double>(repeating: .infinity), hi = SIMD3<Double>(repeating: -.infinity)
        for p in points {
            let v = r * p
            lo = simd_min(lo, v)
            hi = simd_max(hi, v)
        }
        var c = camera
        let mid = (lo + hi) / 2
        c.center = r.transpose * mid
        let w = max(hi.x - lo.x, 1e-6), h = max(hi.y - lo.y, 1e-6)
        c.pointsPerMM = min(Double(size.width) / (w * 1.12), Double(size.height) / (h * 1.12))
        c.isFitted = true
        return c
    }

    /// Pans by a screen delta in points.
    static func pan(_ camera: ViewportCamera, mode: ViewportMode, by delta: CGSize) -> ViewportCamera {
        let r = rotation(camera, mode: mode)
        var c = camera
        c.center -= r.transpose * SIMD3(Double(delta.width) / c.pointsPerMM, Double(delta.height) / c.pointsPerMM, 0)
        return c
    }

    /// Zooms by `factor` keeping the point under `cursor` (points from the
    /// viewport centre, y up) fixed.
    static func zoom(_ camera: ViewportCamera, mode: ViewportMode, factor: Double, about cursor: CGPoint) -> ViewportCamera {
        guard factor.isFinite, factor > 0 else { return camera }
        var c = camera
        let newZoom = min(max(c.pointsPerMM * factor, 1e-4), 1e6)
        let o = SIMD3(Double(cursor.x), Double(cursor.y), 0)
        let shift = o / c.pointsPerMM - o / newZoom
        c.center += rotation(camera, mode: mode).transpose * shift
        c.pointsPerMM = newZoom
        return c
    }

    static func orbit(_ camera: ViewportCamera, by delta: CGSize) -> ViewportCamera {
        var c = camera
        c.yaw += Double(delta.width) * 0.01
        c.pitch = min(max(c.pitch - Double(delta.height) * 0.01, -1.55), 1.55)
        return c
    }

    /// A "nice" scale-bar length (1-2-5 series) close to `target` mm.
    static func niceLength(_ target: Double) -> Double {
        guard target > 0, target.isFinite else { return 1 }
        let e = pow(10, floor(log10(target)))
        let m = target / e
        return (m < 1.5 ? 1 : (m < 3.5 ? 2 : (m < 7.5 ? 5 : 10))) * e
    }
}
