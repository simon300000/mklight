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

/// Which compute backend a job should use. A preference: the engine falls back
/// to the CPU reference (and says so in `RunInfo.warnings`) when a backend is
/// unavailable or cannot meet the error budget for a design (CONVENTIONS §7).
public enum BackendPreference: String, Codable, Sendable, Hashable, CaseIterable {
    /// The engine chooses.
    case auto
    /// The CPU f64 reference tracer.
    case cpuReference = "cpu_reference"
    /// This platform's GPU backend (Metal on macOS).
    case gpu

    public var title: String {
        switch self {
        case .auto: "Automatic"
        case .cpuReference: "CPU reference"
        case .gpu: "GPU"
        }
    }
}

/// How a job is executed: what a front end chooses of a job's identity
/// `(snapshot, settings, backend, precision, seed)` (ARCHITECTURE §State
/// model) besides the snapshot and the analysis settings. The engine chooses
/// the precision with the backend; `RunInfo.precision` records it. It is
/// part of every typed settings struct, so it is recorded in
/// `RunInfo.settings` and takes part in the staleness check: another backend
/// preference makes the old result stale.
public struct ExecutionSettings: Codable, Sendable, Hashable {
    public var backend = BackendPreference.auto
    /// Seed of a typed request's stochastic parts (random pupil sampling;
    /// CONVENTIONS §6). A request by name carries its seed in its settings.
    public var seed: UInt64 = 1
    public init(backend: BackendPreference = .auto, seed: UInt64 = 1) {
        self.backend = backend
        self.seed = seed
    }
}

/// Settings structs are plain serialisable values. Together with the snapshot
/// hash they fully determine a result, and they are recorded in
/// `RunInfo.settings`. `AnalysisRequest.problems(for:)` checks them against a
/// system before a job starts.

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
    public var execution = ExecutionSettings()
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case raysPerField = "rays_per_field", fields, wavelengths
        case profileSamples = "profile_samples", include3D = "include_3d", configuration, execution
    }
}

public enum PupilPattern: String, Codable, Sendable, Hashable, CaseIterable {
    case hexapolar, square
    /// Uniform random over the pupil; uses `execution.seed`.
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
    public var configuration = 0
    public var execution = ExecutionSettings()
    public init() {}
}

/// Settings for transverse ray fans and OPD fans.
public struct FanSettings: Codable, Sendable, Hashable {
    /// Pupil samples across each fan (−1…1).
    public var samples = 41
    public var fields = FieldSelection.all
    public var wavelengths = WavelengthSelection.all
    public var configuration = 0
    public var execution = ExecutionSettings()
    public init() {}
}

public struct FieldCurvatureSettings: Codable, Sendable, Hashable {
    /// Field samples from axis to the maximum field.
    public var fieldSamples = 21
    public var wavelengths = WavelengthSelection.all
    public var configuration = 0
    public var execution = ExecutionSettings()
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case fieldSamples = "field_samples", wavelengths, configuration, execution
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
    public var execution = ExecutionSettings()
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case maxFrequency = "max_frequency", frequencySamples = "frequency_samples"
        case pupilSampling = "pupil_sampling", fields, wavelengths, configuration, execution
    }
}

public struct PSFSettings: Codable, Sendable, Hashable {
    public var field = 0
    public var wavelengths = WavelengthSelection.all
    public var pupilSampling = 64
    /// Image grid size (pixels per side).
    public var imageSampling = 64
    public var configuration = 0
    public var execution = ExecutionSettings()
    public init() {}

    private enum CodingKeys: String, CodingKey {
        case field, wavelengths, pupilSampling = "pupil_sampling", imageSampling = "image_sampling", configuration
        case execution
    }
}

public struct FirstOrderSettings: Codable, Sendable, Hashable {
    public var wavelength = WavelengthSelection.primary
    public var configuration = 0
    public var execution = ExecutionSettings()
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

    /// How the job is executed.
    public var execution: ExecutionSettings {
        get {
            switch self {
            case .layout(let s): s.execution
            case .spotDiagram(let s): s.execution
            case .rayFan(let s), .opdFan(let s): s.execution
            case .fieldCurvature(let s): s.execution
            case .mtf(let s): s.execution
            case .psf(let s): s.execution
            case .firstOrder(let s): s.execution
            }
        }
        set {
            switch self {
            case .layout(var s): s.execution = newValue; self = .layout(s)
            case .spotDiagram(var s): s.execution = newValue; self = .spotDiagram(s)
            case .rayFan(var s): s.execution = newValue; self = .rayFan(s)
            case .opdFan(var s): s.execution = newValue; self = .opdFan(s)
            case .fieldCurvature(var s): s.execution = newValue; self = .fieldCurvature(s)
            case .mtf(var s): s.execution = newValue; self = .mtf(s)
            case .psf(var s): s.execution = newValue; self = .psf(s)
            case .firstOrder(var s): s.execution = newValue; self = .firstOrder(s)
            }
        }
    }

    public var configuration: Int {
        switch self {
        case .layout(let s): s.configuration
        case .spotDiagram(let s): s.configuration
        case .rayFan(let s), .opdFan(let s): s.configuration
        case .fieldCurvature(let s): s.configuration
        case .mtf(let s): s.configuration
        case .psf(let s): s.configuration
        case .firstOrder(let s): s.configuration
        }
    }

    /// The seed the result depends on (`RunInfo.seed`): `execution.seed` for
    /// analyses that sample randomly with these settings, else `nil`.
    public var seed: UInt64? {
        if case .spotDiagram(let s) = self, s.pattern == .random { return s.execution.seed }
        return nil
    }

    /// Problems that make the settings unusable for `system` (empty when they
    /// are valid). Engines refuse such requests with
    /// `EngineError.invalidSettings` before starting any work.
    public func problems(for system: OpticalSystem) -> [String] {
        var out: [String] = []
        func range(_ name: String, _ v: Int, _ r: ClosedRange<Int>) {
            if !r.contains(v) { out.append("\(name) must be in \(r.lowerBound)…\(r.upperBound) (is \(v))") }
        }
        func fields(_ f: FieldSelection) {
            if case .index(let i) = f, !system.fields.fields.indices.contains(i) { out.append("field \(i + 1) does not exist") }
        }
        func wavelengths(_ w: WavelengthSelection) {
            if case .index(let i) = w, !system.wavelengths.wavelengths.indices.contains(i) {
                out.append("wavelength \(i + 1) does not exist")
            }
        }
        let k = configuration
        if k != 0 && !system.configurations.indices.contains(k) { out.append("configuration \(k) does not exist") }
        switch self {
        case .layout(let s):
            range("rays per field", s.raysPerField, 1...1001)
            range("profile samples", s.profileSamples, 3...2001)
            fields(s.fields)
            wavelengths(s.wavelengths)
        case .spotDiagram(let s):
            range("rings", s.rings, 1...256)
            fields(s.fields)
            wavelengths(s.wavelengths)
        case .rayFan(let s), .opdFan(let s):
            range("fan samples", s.samples, 3...10001)
            fields(s.fields)
            wavelengths(s.wavelengths)
        case .fieldCurvature(let s):
            range("field samples", s.fieldSamples, 2...10001)
            wavelengths(s.wavelengths)
        case .mtf(let s):
            if !(s.maxFrequency.isFinite && s.maxFrequency > 0) {
                out.append("maximum frequency must be a positive number of cycles/mm")
            }
            range("frequency samples", s.frequencySamples, 2...10001)
            range("pupil sampling", s.pupilSampling, 8...4096)
            fields(s.fields)
            wavelengths(s.wavelengths)
        case .psf(let s):
            fields(.index(s.field))
            range("pupil sampling", s.pupilSampling, 8...4096)
            range("image sampling", s.imageSampling, 16...4096)
            wavelengths(s.wavelengths)
        case .firstOrder(let s):
            wavelengths(s.wavelength)
        }
        return out
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
    /// Backend and seed for every analysis of the window.
    public var execution = ExecutionSettings()
    public init() {}

    /// The configuration every analysis of the window uses (reads the layout's;
    /// setting it sets all).
    public var configuration: Int {
        get { layout.configuration }
        set {
            layout.configuration = newValue
            spot.configuration = newValue
            rayFan.configuration = newValue
            opdFan.configuration = newValue
            fieldCurvature.configuration = newValue
            mtf.configuration = newValue
            psf.configuration = newValue
            firstOrder.configuration = newValue
        }
    }

    /// The request for `kind`, carrying the window's `execution` settings.
    public func request(for kind: AnalysisKind) -> AnalysisRequest {
        var r: AnalysisRequest =
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
        r.execution = execution
        return r
    }
}
