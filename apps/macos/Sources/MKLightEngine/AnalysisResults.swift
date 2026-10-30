import Foundation
import MKLightModel

/// A point in global coordinates (mm), serialised as `[x, y, z]`.
public struct Point3: Codable, Sendable, Hashable {
    public var x, y, z: Double
    public init(_ x: Double, _ y: Double, _ z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }
    public init(from decoder: any Decoder) throws {
        var c = try decoder.unkeyedContainer()
        x = try c.decode(Double.self)
        y = try c.decode(Double.self)
        z = try c.decode(Double.self)
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(x)
        try c.encode(y)
        try c.encode(z)
    }
}

/// A 2-D point (mm or normalised units, as documented per use), `[x, y]`.
public struct Point2: Codable, Sendable, Hashable {
    public var x, y: Double
    public init(_ x: Double, _ y: Double) {
        self.x = x
        self.y = y
    }
    public init(from decoder: any Decoder) throws {
        var c = try decoder.unkeyedContainer()
        x = try c.decode(Double.self)
        y = try c.decode(Double.self)
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(x)
        try c.encode(y)
    }
}

public struct Bounds3: Codable, Sendable, Hashable {
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

// MARK: - Layout

/// Geometry for the layout viewport, entirely computed by the engine: the UI
/// only projects and draws these polylines.
public struct LayoutScene: Codable, Sendable, Hashable {
    public enum PolylineKind: String, Codable, Sendable, Hashable {
        /// A surface's profile curve.
        case surfaceProfile = "surface_profile"
        /// The clear-aperture rim of a surface (3-D only).
        case surfaceRim = "surface_rim"
        /// Edge/flange lines closing a lens element.
        case elementEdge = "element_edge"
        case stop
        case imagePlane = "image_plane"
        case ray
        case axis
    }

    public struct Polyline: Codable, Sendable, Hashable {
        public var kind: PolylineKind
        public var points: [Point3]
        /// Lies in the meridional (y-z) section: drawn in the 2-D section view.
        public var inSection: Bool
        public var surface: Int?
        public var field: Int?
        public var wavelength: Int?
        /// For rays: whether the ray was clipped before the image.
        public var vignetted: Bool

        public init(
            kind: PolylineKind, points: [Point3], inSection: Bool, surface: Int? = nil, field: Int? = nil,
            wavelength: Int? = nil, vignetted: Bool = false
        ) {
            self.kind = kind
            self.points = points
            self.inSection = inSection
            self.surface = surface
            self.field = field
            self.wavelength = wavelength
            self.vignetted = vignetted
        }

        private enum CodingKeys: String, CodingKey {
            case kind, points, inSection = "in_section", surface, field, wavelength, vignetted
        }
    }

    /// A lens element's meridional section as two profiles sampled at matching
    /// heights (front[i] and back[i] share a height), for filling.
    public struct Element: Codable, Sendable, Hashable {
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
        private enum CodingKeys: String, CodingKey {
            case firstSurface = "first_surface", lastSurface = "last_surface", front, back
        }
    }

    public struct SurfaceInfo: Codable, Sendable, Hashable {
        public var index: Int
        public var id: SurfaceID
        /// Global vertex position.
        public var vertex: Point3
        /// Clear semi-diameter used for drawing (fixed, or computed from rays).
        public var semiDiameter: Double
        public var semiDiameterIsAuto: Bool
        public init(index: Int, id: SurfaceID, vertex: Point3, semiDiameter: Double, semiDiameterIsAuto: Bool) {
            self.index = index
            self.id = id
            self.vertex = vertex
            self.semiDiameter = semiDiameter
            self.semiDiameterIsAuto = semiDiameterIsAuto
        }
        private enum CodingKeys: String, CodingKey {
            case index, id, vertex, semiDiameter = "semi_diameter", semiDiameterIsAuto = "semi_diameter_is_auto"
        }
    }

    public var polylines: [Polyline]
    public var elements: [Element]
    public var surfaces: [SurfaceInfo]
    public var bounds: Bounds3
    public var wavelengthsUm: [Double]
    public var fieldCount: Int

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

    private enum CodingKeys: String, CodingKey {
        case polylines, elements, surfaces, bounds, wavelengthsUm = "wavelengths_um", fieldCount = "field_count"
    }
}

// MARK: - Spot diagram

public struct SpotDiagramResult: Codable, Sendable, Hashable {
    public struct Point: Codable, Sendable, Hashable {
        /// Position relative to the reference point (mm).
        public var x, y: Double
        public var wavelength: Int
        public init(x: Double, y: Double, wavelength: Int) {
            self.x = x
            self.y = y
            self.wavelength = wavelength
        }
    }

    public struct FieldSpot: Codable, Sendable, Hashable {
        public var field: Int
        /// Field value as specified (degrees or mm).
        public var fieldValue: Point2
        public var points: [Point]
        /// Reference point on the image surface (global mm).
        public var reference: Point2
        public var centroid: Point2
        /// RMS radius about the reference (mm).
        public var rmsRadius: Double
        /// Largest distance from the reference (mm).
        public var geoRadius: Double
        public init(
            field: Int, fieldValue: Point2, points: [Point], reference: Point2, centroid: Point2, rmsRadius: Double,
            geoRadius: Double
        ) {
            self.field = field
            self.fieldValue = fieldValue
            self.points = points
            self.reference = reference
            self.centroid = centroid
            self.rmsRadius = rmsRadius
            self.geoRadius = geoRadius
        }
        private enum CodingKeys: String, CodingKey {
            case field, fieldValue = "field_value", points, reference, centroid
            case rmsRadius = "rms_radius", geoRadius = "geo_radius"
        }
    }

    public var fields: [FieldSpot]
    public var wavelengthsUm: [Double]
    /// Airy disk radius `1.22 λ N_w` at the primary wavelength (mm).
    public var airyRadius: Double

    public init(fields: [FieldSpot], wavelengthsUm: [Double], airyRadius: Double) {
        self.fields = fields
        self.wavelengthsUm = wavelengthsUm
        self.airyRadius = airyRadius
    }
    private enum CodingKeys: String, CodingKey {
        case fields, wavelengthsUm = "wavelengths_um", airyRadius = "airy_radius"
    }
}

// MARK: - Fans

/// Transverse ray aberration (mm) or OPD (waves) against normalised pupil
/// coordinate, per field and wavelength (CONVENTIONS §4).
public struct FanResult: Codable, Sendable, Hashable {
    public enum Quantity: String, Codable, Sendable, Hashable {
        /// `εy` (tangential) / `εx` (sagittal) in mm.
        case transverseAberration = "transverse_aberration"
        /// OPD in waves at each ray's wavelength.
        case opd
    }

    public struct Curve: Codable, Sendable, Hashable {
        public var wavelength: Int
        /// Normalised pupil coordinate `Py` (tangential) or `Px` (sagittal).
        public var pupil: [Double]
        /// Value per pupil sample; `nil` where the ray was vignetted.
        public var values: [Double?]
        public init(wavelength: Int, pupil: [Double], values: [Double?]) {
            self.wavelength = wavelength
            self.pupil = pupil
            self.values = values
        }
    }

    public struct FieldFan: Codable, Sendable, Hashable {
        public var field: Int
        public var fieldValue: Point2
        public var tangential: [Curve]
        public var sagittal: [Curve]
        public init(field: Int, fieldValue: Point2, tangential: [Curve], sagittal: [Curve]) {
            self.field = field
            self.fieldValue = fieldValue
            self.tangential = tangential
            self.sagittal = sagittal
        }
        private enum CodingKeys: String, CodingKey { case field, fieldValue = "field_value", tangential, sagittal }
    }

    public var quantity: Quantity
    public var fields: [FieldFan]
    public var wavelengthsUm: [Double]
    public init(quantity: Quantity, fields: [FieldFan], wavelengthsUm: [Double]) {
        self.quantity = quantity
        self.fields = fields
        self.wavelengthsUm = wavelengthsUm
    }
    private enum CodingKeys: String, CodingKey { case quantity, fields, wavelengthsUm = "wavelengths_um" }
}

// MARK: - Field curvature and distortion

public struct FieldCurvatureResult: Codable, Sendable, Hashable {
    public struct Curve: Codable, Sendable, Hashable {
        public var wavelength: Int
        /// Tangential and sagittal focus shift from the image surface (mm).
        public var tangential: [Double?]
        public var sagittal: [Double?]
        /// Distortion (%) relative to the paraxial chief-ray height.
        public var distortionPercent: [Double?]
        public init(wavelength: Int, tangential: [Double?], sagittal: [Double?], distortionPercent: [Double?]) {
            self.wavelength = wavelength
            self.tangential = tangential
            self.sagittal = sagittal
            self.distortionPercent = distortionPercent
        }
        private enum CodingKeys: String, CodingKey {
            case wavelength, tangential, sagittal, distortionPercent = "distortion_percent"
        }
    }
    /// Field values (in `fieldUnit`), from axis to the maximum field.
    public var fieldValues: [Double]
    public var fieldUnit: String
    public var curves: [Curve]
    public var wavelengthsUm: [Double]
    public init(fieldValues: [Double], fieldUnit: String, curves: [Curve], wavelengthsUm: [Double]) {
        self.fieldValues = fieldValues
        self.fieldUnit = fieldUnit
        self.curves = curves
        self.wavelengthsUm = wavelengthsUm
    }
    private enum CodingKeys: String, CodingKey {
        case fieldValues = "field_values", fieldUnit = "field_unit", curves, wavelengthsUm = "wavelengths_um"
    }
}

// MARK: - MTF

public struct MTFResult: Codable, Sendable, Hashable {
    public enum Orientation: String, Codable, Sendable, Hashable { case tangential, sagittal }
    public struct Curve: Codable, Sendable, Hashable {
        public var field: Int
        public var orientation: Orientation
        public var modulation: [Double]
        public init(field: Int, orientation: Orientation, modulation: [Double]) {
            self.field = field
            self.orientation = orientation
            self.modulation = modulation
        }
    }
    /// Spatial frequencies (cycles/mm).
    public var frequencies: [Double]
    public var curves: [Curve]
    /// Diffraction-limited MTF on axis, if computed.
    public var diffractionLimit: [Double]?
    /// Incoherent cutoff `1/(λ N_w)` (cycles/mm).
    public var cutoffFrequency: Double
    public var fieldValues: [Point2]
    public init(
        frequencies: [Double], curves: [Curve], diffractionLimit: [Double]?, cutoffFrequency: Double,
        fieldValues: [Point2]
    ) {
        self.frequencies = frequencies
        self.curves = curves
        self.diffractionLimit = diffractionLimit
        self.cutoffFrequency = cutoffFrequency
        self.fieldValues = fieldValues
    }
    private enum CodingKeys: String, CodingKey {
        case frequencies, curves, diffractionLimit = "diffraction_limit", cutoffFrequency = "cutoff_frequency"
        case fieldValues = "field_values"
    }
}

// MARK: - PSF

public struct PSFResult: Codable, Sendable, Hashable {
    public var field: Int
    public var width: Int
    public var height: Int
    /// Pixel pitch on the image surface (mm).
    public var pixelPitch: Double
    /// Row-major intensities (row 0 at −y), normalised so the diffraction-
    /// limited peak is 1; the peak value is therefore the Strehl ratio.
    public var values: [Float]
    public var strehl: Double
    public init(field: Int, width: Int, height: Int, pixelPitch: Double, values: [Float], strehl: Double) {
        self.field = field
        self.width = width
        self.height = height
        self.pixelPitch = pixelPitch
        self.values = values
        self.strehl = strehl
    }
    private enum CodingKeys: String, CodingKey {
        case field, width, height, pixelPitch = "pixel_pitch", values, strehl
    }
}

// MARK: - First-order data

public struct FirstOrderResult: Codable, Sendable, Hashable {
    public struct Entry: Codable, Sendable, Hashable, Identifiable {
        /// Stable key, e.g. `efl`.
        public var key: String
        public var label: String
        public var value: Double
        public var unit: String
        public var id: String { key }
        public init(key: String, label: String, value: Double, unit: String) {
            self.key = key
            self.label = label
            self.value = value
            self.unit = unit
        }
    }
    public var entries: [Entry]
    public var wavelengthUm: Double
    public init(entries: [Entry], wavelengthUm: Double) {
        self.entries = entries
        self.wavelengthUm = wavelengthUm
    }
    public subscript(key: String) -> Double? { entries.first { $0.key == key }?.value }
    private enum CodingKeys: String, CodingKey { case entries, wavelengthUm = "wavelength_um" }
}

// MARK: - Envelope

public enum AnalysisResult: Codable, Sendable, Hashable {
    case layout(LayoutScene)
    case spotDiagram(SpotDiagramResult)
    case rayFan(FanResult)
    case opdFan(FanResult)
    case fieldCurvature(FieldCurvatureResult)
    case mtf(MTFResult)
    case psf(PSFResult)
    case firstOrder(FirstOrderResult)

    public var kind: AnalysisKind {
        switch self {
        case .layout: .layout
        case .spotDiagram: .spotDiagram
        case .rayFan: .rayFan
        case .opdFan: .opdFan
        case .fieldCurvature: .fieldCurvature
        case .mtf: .mtf
        case .psf: .psf
        case .firstOrder: .firstOrder
        }
    }

    private enum CodingKeys: String, CodingKey { case analysis, data }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(AnalysisKind.self, forKey: .analysis) {
        case .layout: self = .layout(try c.decode(LayoutScene.self, forKey: .data))
        case .spotDiagram: self = .spotDiagram(try c.decode(SpotDiagramResult.self, forKey: .data))
        case .rayFan: self = .rayFan(try c.decode(FanResult.self, forKey: .data))
        case .opdFan: self = .opdFan(try c.decode(FanResult.self, forKey: .data))
        case .fieldCurvature: self = .fieldCurvature(try c.decode(FieldCurvatureResult.self, forKey: .data))
        case .mtf: self = .mtf(try c.decode(MTFResult.self, forKey: .data))
        case .psf: self = .psf(try c.decode(PSFResult.self, forKey: .data))
        case .firstOrder: self = .firstOrder(try c.decode(FirstOrderResult.self, forKey: .data))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(kind, forKey: .analysis)
        switch self {
        case .layout(let v): try c.encode(v, forKey: .data)
        case .spotDiagram(let v): try c.encode(v, forKey: .data)
        case .rayFan(let v), .opdFan(let v): try c.encode(v, forKey: .data)
        case .fieldCurvature(let v): try c.encode(v, forKey: .data)
        case .mtf(let v): try c.encode(v, forKey: .data)
        case .psf(let v): try c.encode(v, forKey: .data)
        case .firstOrder(let v): try c.encode(v, forKey: .data)
        }
    }
}

/// A finished analysis: its data plus the run metadata.
public struct AnalysisOutput: Codable, Sendable, Hashable {
    public var runInfo: RunInfo
    public var result: AnalysisResult
    public init(runInfo: RunInfo, result: AnalysisResult) {
        self.runInfo = runInfo
        self.result = result
    }
    private enum CodingKeys: String, CodingKey { case runInfo = "run_info", result }
}
