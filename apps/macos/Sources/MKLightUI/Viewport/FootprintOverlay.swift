import Foundation
import MKLightEngine

/// Beam footprints on every surface for the 3-D viewport, read from the
/// engine's `footprint` overlays (docs/RESULTS.md): one convex outline per
/// (surface, field, wavelength), as global points, plus the global hit
/// points of the rays that pass each surface.
struct FootprintOverlay: Equatable, Sendable {
    struct Outline: Equatable, Sendable {
        var surface: Int
        var field: Int
        var wavelength: Int
        /// Counterclockwise, unclosed, global (mm).
        var points: [Point3]
    }
    var outlines: [Outline]
    /// Hit points per surface (global), for counts and fitting.
    var pointCount: Int

    /// `nil` when the output has no overlays.
    init?(output: AnalysisOutput?) {
        guard let output, let overlays = output.json["overlays"]?.array, !overlays.isEmpty,
            let buf = output.buffer(named: "global_outlines"), buf.shape.last == 3
        else { return nil }
        var list: [Outline] = []
        for o in overlays {
            guard let range = o["outline"]?.intArray, range.count == 2, range[1] >= 3 else { continue }
            let start = range[0], n = range[1]
            guard start >= 0, start + n <= buf.rows else { continue }
            let pts = (start..<(start + n)).map { k in Point3(buf[k * 3], buf[k * 3 + 1], buf[k * 3 + 2]) }
            guard pts.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else { continue }
            list.append(Outline(surface: o["surface"]?.int ?? 0, field: o["field"]?.int ?? 0, wavelength: o["wavelength"]?.int ?? 0, points: pts))
        }
        guard !list.isEmpty else { return nil }
        outlines = list
        pointCount = output.buffer(named: "global_points")?.rows ?? 0
    }
}
