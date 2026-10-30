import Foundation
import MKLightModel

/// The analyses a front end can request.
public enum AnalysisKind: String, Codable, Sendable, Hashable, CaseIterable, Identifiable {
    case layout
    case spotDiagram = "spot_diagram"
    case rayFan = "ray_fan"
    case opdFan = "opd_fan"
    case fieldCurvature = "field_curvature_distortion"
    case mtf
    case psf
    case firstOrder = "first_order"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .layout: "Layout"
        case .spotDiagram: "Spot Diagram"
        case .rayFan: "Ray Fan"
        case .opdFan: "OPD Fan"
        case .fieldCurvature: "Field Curvature / Distortion"
        case .mtf: "MTF"
        case .psf: "PSF"
        case .firstOrder: "First-Order Data"
        }
    }

    public var shortTitle: String {
        switch self {
        case .fieldCurvature: "Field Curv."
        case .firstOrder: "First Order"
        case .spotDiagram: "Spot"
        default: title
        }
    }
}

/// Which fields an analysis covers.
public enum FieldSelection: Codable, Sendable, Hashable {
    case all
    case index(Int)

    public func indices(count: Int) -> [Int] {
        switch self {
        case .all: Array(0..<count)
        case .index(let i): (0..<count).contains(i) ? [i] : []
        }
    }

    private enum CodingKeys: String, CodingKey { case type, index }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self = try c.decode(String.self, forKey: .type) == "all" ? .all : .index(try c.decode(Int.self, forKey: .index))
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .all: try c.encode("all", forKey: .type)
        case .index(let i):
            try c.encode("index", forKey: .type)
            try c.encode(i, forKey: .index)
        }
    }
}

/// Which wavelengths an analysis covers.
public enum WavelengthSelection: Codable, Sendable, Hashable {
    case all
    case primary
    case index(Int)

    public func indices(in set: WavelengthSet) -> [Int] {
        switch self {
        case .all: Array(set.wavelengths.indices)
        case .primary: set.wavelengths.indices.contains(set.primary) ? [set.primary] : []
        case .index(let i): set.wavelengths.indices.contains(i) ? [i] : []
        }
    }

    private enum CodingKeys: String, CodingKey { case type, index }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(String.self, forKey: .type) {
        case "all": self = .all
        case "primary": self = .primary
        default: self = .index(try c.decode(Int.self, forKey: .index))
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .all: try c.encode("all", forKey: .type)
        case .primary: try c.encode("primary", forKey: .type)
        case .index(let i):
            try c.encode("index", forKey: .type)
            try c.encode(i, forKey: .index)
        }
    }
}

/// Settings structs are plain serialisable values. Together with the snapshot
/// hash and the seed they fully determine a result, and they are recorded in
/// `RunInfo.settings`.

public struct LayoutSettings: Codable, Sendable, Hashable {
    /// Meridional rays per field, evenly spaced across the pupil.
    public var raysPerField = 7
    public var fields = FieldSelection.all
    public var wavelengths = WavelengthSelection.primary
    /// Points per surface profile.
    public var profileSamples = 41
    /// Also produce out-of-section geometry (sagittal profiles, rims, skew rays)
    /// for the 3-D view.
    public var include3D = true
    /// Configuration index (0 = base system).
    public var configuration = 0
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case raysPerField = "rays_per_field", fields, wavelengths
        case profileSamples = "profile_samples", include3D = "include_3d", configuration
    }
}

public enum PupilPattern: String, Codable, Sendable, Hashable, CaseIterable {
    case hexapolar, square
    /// Uniform random over the pupil; uses the settings' `seed`.
    case random
}

public enum SpotReference: String, Codable, Sendable, Hashable, CaseIterable {
    case chief, centroid
}

public struct SpotDiagramSettings: Codable, Sendable, Hashable {
    public var pattern = PupilPattern.hexapolar
    /// Rings (hexapolar), grid half-width (square) or √samples (random).
    public var rings = 6
    public var fields = FieldSelection.all
    public var wavelengths = WavelengthSelection.all
    public var reference = SpotReference.chief
    /// Philox seed for stochastic sampling (CONVENTIONS §6).
    public var seed: UInt64 = 1
    public var configuration = 0
    public init() {}
}

/// Settings for transverse ray fans and OPD fans.
public struct FanSettings: Codable, Sendable, Hashable {
    /// Pupil samples across each fan (−1…1).
    public var samples = 41
    public var fields = FieldSelection.all
    public var wavelengths = WavelengthSelection.all
    public var configuration = 0
    public init() {}
}

public struct FieldCurvatureSettings: Codable, Sendable, Hashable {
    /// Field samples from axis to the maximum field.
    public var fieldSamples = 21
    public var wavelengths = WavelengthSelection.all
    public var configuration = 0
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case fieldSamples = "field_samples", wavelengths, configuration
    }
}

public struct MTFSettings: Codable, Sendable, Hashable {
    /// Highest spatial frequency plotted (cycles/mm).
    public var maxFrequency = 100.0
    public var frequencySamples = 41
    /// Pupil grid size for the wavefront.
    public var pupilSampling = 64
    public var fields = FieldSelection.all
    /// Polychromatic over these wavelengths, weighted by the system weights.
    public var wavelengths = WavelengthSelection.all
    public var configuration = 0
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case maxFrequency = "max_frequency", frequencySamples = "frequency_samples"
        case pupilSampling = "pupil_sampling", fields, wavelengths, configuration
    }
}

public struct PSFSettings: Codable, Sendable, Hashable {
    public var field = 0
    public var wavelengths = WavelengthSelection.all
    public var pupilSampling = 64
    /// Image grid size (pixels per side).
    public var imageSampling = 64
    public var configuration = 0
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case field, wavelengths, pupilSampling = "pupil_sampling", imageSampling = "image_sampling", configuration
    }
}

public struct FirstOrderSettings: Codable, Sendable, Hashable {
    public var wavelength = WavelengthSelection.primary
    public var configuration = 0
    public init() {}
}

/// One analysis request: which analysis and its settings. Encoded as
/// `{"analysis": "<kind>", "settings": {...}}` for the engine.
public enum AnalysisRequest: Codable, Sendable, Hashable {
    case layout(LayoutSettings)
    case spotDiagram(SpotDiagramSettings)
    case rayFan(FanSettings)
    case opdFan(FanSettings)
    case fieldCurvature(FieldCurvatureSettings)
    case mtf(MTFSettings)
    case psf(PSFSettings)
    case firstOrder(FirstOrderSettings)

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

    /// Random seed, for stochastic analyses.
    public var seed: UInt64? {
        if case .spotDiagram(let s) = self, s.pattern == .random { return s.seed }
        return nil
    }

    private var settingsValue: any Encodable & Sendable {
        switch self {
        case .layout(let s): s
        case .spotDiagram(let s): s
        case .rayFan(let s), .opdFan(let s): s
        case .fieldCurvature(let s): s
        case .mtf(let s): s
        case .psf(let s): s
        case .firstOrder(let s): s
        }
    }

    /// Settings as JSON, as recorded in `RunInfo.settings`.
    public func settingsJSON() throws -> JSONValue { try JSONValue.from(settingsValue) }

    private enum CodingKeys: String, CodingKey { case analysis, settings }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try c.decode(AnalysisKind.self, forKey: .analysis)
        switch kind {
        case .layout: self = .layout(try c.decode(LayoutSettings.self, forKey: .settings))
        case .spotDiagram: self = .spotDiagram(try c.decode(SpotDiagramSettings.self, forKey: .settings))
        case .rayFan: self = .rayFan(try c.decode(FanSettings.self, forKey: .settings))
        case .opdFan: self = .opdFan(try c.decode(FanSettings.self, forKey: .settings))
        case .fieldCurvature: self = .fieldCurvature(try c.decode(FieldCurvatureSettings.self, forKey: .settings))
        case .mtf: self = .mtf(try c.decode(MTFSettings.self, forKey: .settings))
        case .psf: self = .psf(try c.decode(PSFSettings.self, forKey: .settings))
        case .firstOrder: self = .firstOrder(try c.decode(FirstOrderSettings.self, forKey: .settings))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(kind, forKey: .analysis)
        try c.encode(settingsValue, forKey: .settings)
    }
}

/// The settings of every analysis, as a front end keeps them per window.
public struct AnalysisSettingsSet: Codable, Sendable, Hashable {
    public var layout = LayoutSettings()
    public var spot = SpotDiagramSettings()
    public var rayFan = FanSettings()
    public var opdFan = FanSettings()
    public var fieldCurvature = FieldCurvatureSettings()
    public var mtf = MTFSettings()
    public var psf = PSFSettings()
    public var firstOrder = FirstOrderSettings()
    public init() {}

    public func request(for kind: AnalysisKind) -> AnalysisRequest {
        switch kind {
        case .layout: .layout(layout)
        case .spotDiagram: .spotDiagram(spot)
        case .rayFan: .rayFan(rayFan)
        case .opdFan: .opdFan(opdFan)
        case .fieldCurvature: .fieldCurvature(fieldCurvature)
        case .mtf: .mtf(mtf)
        case .psf: .psf(psf)
        case .firstOrder: .firstOrder(firstOrder)
        }
    }
}
