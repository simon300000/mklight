import Foundation
import MKLightModel

/// A point in global coordinates (mm). Points compare with NaN equal to NaN,
/// so scenes stay `Equatable`.
public struct Point3: Sendable, Hashable {
    public var x, y, z: Double
    public init(_ x: Double, _ y: Double, _ z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }
    public static func == (a: Point3, b: Point3) -> Bool { a.engineValues == b.engineValues }
    public func hash(into h: inout Hasher) { h.combine(engineValues) }
    private var engineValues: [EngineDouble] { [x, y, z].map(EngineDouble.init(wrappedValue:)) }
}

public struct Bounds3: Sendable, Hashable {
    public var min: Point3
    public var max: Point3
    public init(min: Point3, max: Point3) {
        self.min = min
        self.max = max
    }
    public static let empty = Bounds3(min: Point3(0, 0, 0), max: Point3(0, 0, 0))
    public var center: Point3 { Point3((min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2) }

    /// Smallest box containing `points` (`.empty` when there are none).
    public static func enclosing<S: Sequence>(_ points: S) -> Bounds3 where S.Element == Point3 {
        var lo = Point3(.infinity, .infinity, .infinity), hi = Point3(-.infinity, -.infinity, -.infinity)
        for p in points {
            lo = Point3(Swift.min(lo.x, p.x), Swift.min(lo.y, p.y), Swift.min(lo.z, p.z))
            hi = Point3(Swift.max(hi.x, p.x), Swift.max(hi.y, p.y), Swift.max(hi.z, p.z))
        }
        return lo.x.isFinite ? Bounds3(min: lo, max: hi) : .empty
    }
}

/// Geometry for the layout viewport, entirely computed by the engine: the UI
/// only projects and draws these polylines.
public struct LayoutScene: Sendable, Hashable {
    public enum PolylineKind: Sendable, Hashable {
        /// A surface's profile curve.
        case surfaceProfile
        /// The clear-aperture rim of a surface (3-D only).
        case surfaceRim
        /// Edge/flange lines closing a lens element.
        case elementEdge
        case stop
        case imagePlane
        case ray
        case axis
    }

    public struct Polyline: Sendable, Hashable {
        public var kind: PolylineKind
        public var points: [Point3]
        /// Lies in the meridional (y-z) section: drawn in the 2-D section view.
        public var inSection: Bool
        public var surface: Int?
        public var field: Int?
        public var wavelength: Int?
        /// For rays: whether the ray was clipped before the image.
        public var vignetted: Bool
        /// Belongs to a diffractive (DOE) surface (layout flag bit 2).
        public var doe = false

        public init(
            kind: PolylineKind, points: [Point3], inSection: Bool, surface: Int? = nil, field: Int? = nil,
            wavelength: Int? = nil, vignetted: Bool = false, doe: Bool = false
        ) {
            self.doe = doe
            self.kind = kind
            self.points = points
            self.inSection = inSection
            self.surface = surface
            self.field = field
            self.wavelength = wavelength
            self.vignetted = vignetted
        }
    }

    /// A lens element's meridional section as two profiles sampled at matching
    /// heights (front[i] and back[i] share a height), for filling.
    public struct Element: Sendable, Hashable {
        public var firstSurface: Int
        public var lastSurface: Int
        public var front: [Point3]
        public var back: [Point3]
        public init(firstSurface: Int, lastSurface: Int, front: [Point3], back: [Point3]) {
            self.firstSurface = firstSurface
            self.lastSurface = lastSurface
            self.front = front
            self.back = back
        }
    }

    public struct SurfaceInfo: Sendable, Hashable {
        public var index: Int
        public var id: SurfaceID
        /// Global vertex position.
        public var vertex: Point3
        /// Clear semi-diameter used for drawing (fixed, or computed from rays).
        public var semiDiameter: Double
        public var semiDiameterIsAuto: Bool
        /// The surface carries a DOE phase.
        public var isDiffractive = false
        public init(
            index: Int, id: SurfaceID, vertex: Point3, semiDiameter: Double, semiDiameterIsAuto: Bool, isDiffractive: Bool = false
        ) {
            self.isDiffractive = isDiffractive
            self.index = index
            self.id = id
            self.vertex = vertex
            self.semiDiameter = semiDiameter
            self.semiDiameterIsAuto = semiDiameterIsAuto
        }
    }

    public var polylines: [Polyline]
    public var elements: [Element]
    public var surfaces: [SurfaceInfo]
    public var bounds: Bounds3
    public var wavelengthsUm: [Double]
    public var fieldCount: Int
    /// The 2-D section the in-section polylines lie in: `yz` or `xz`.
    public var section = "yz"
    /// The optional triangulated surfaces (`layout` with `mesh_rings`).
    public var mesh: [MeshTriangle] = []

    /// One triangle of the surface mesh, in global coordinates.
    public struct MeshTriangle: Sendable, Hashable {
        public var a, b, c: Point3
        public var surface: Int
        public init(a: Point3, b: Point3, c: Point3, surface: Int) {
            self.a = a
            self.b = b
            self.c = c
            self.surface = surface
        }
    }

    public init(
        polylines: [Polyline], elements: [Element], surfaces: [SurfaceInfo], bounds: Bounds3, wavelengthsUm: [Double],
        fieldCount: Int
    ) {
        self.polylines = polylines
        self.elements = elements
        self.surfaces = surfaces
        self.bounds = bounds
        self.wavelengthsUm = wavelengthsUm
        self.fieldCount = fieldCount
    }
}
