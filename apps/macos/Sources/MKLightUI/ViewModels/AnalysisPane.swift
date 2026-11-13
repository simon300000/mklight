import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// One analysis shown in a window: which engine analysis, its settings, its
/// latest output and the state of its job. Settings are plain JSON in the
/// analysis' own schema, so they are reproducible (the engine records the
/// complete settings, including the seed, in `RunInfo`).
@MainActor
@Observable
public final class AnalysisPane: Identifiable {
    public enum Phase: Equatable, Sendable {
        case idle
        case running(JobProgress)
        case finished
        case failed(String)
        /// The engine does not offer the analysis (with its reason).
        case unavailable(String)
        case cancelled
    }

    /// Stable key (the project's `views[].key`).
    public let id: String
    public private(set) var analysis: String {
        didSet { storedRequest = nil }
    }
    public private(set) var settings: [String: JSONValue] {
        didSet { storedRequest = nil }
    }
    /// Re-run automatically when the design changes; otherwise the result
    /// goes stale and waits for Re-run.
    public var autoUpdate: Bool
    /// Whether the settings form is open in the pane (saved with the layout;
    /// closed by default, the plot with a one-line summary).
    public var showsSettings = false
    public internal(set) var output: AnalysisOutput? {
        didSet { derived = nil }
    }
    public internal(set) var phase = Phase.idle
    /// The request the current job (or the last finished one) ran.
    public internal(set) var lastRequest: EngineAnalysisRequest?

    @ObservationIgnored var job: AnalysisJob?
    /// Values read from `output` once per output rather than in every view
    /// update (the layout scene is thousands of points).
    @ObservationIgnored private var derived: (scene: LayoutScene?, autoSemiDiameters: [SurfaceID: Double])?
    @ObservationIgnored weak var workspace: Workspace? {
        didSet { storedRequest = nil }
    }
    /// The request `Workspace.request(for:)` built from the analysis and
    /// settings, with the workspace's epoch it was built at.
    @ObservationIgnored var storedRequest: (epoch: Int, request: EngineAnalysisRequest)?

    public init(id: String = "pane-" + UUID().uuidString.prefix(8).lowercased(), analysis: String,
                settings: [String: JSONValue] = [:], autoUpdate: Bool = true) {
        self.id = id
        self.analysis = analysis
        self.settings = settings
        self.autoUpdate = autoUpdate
    }

    /// The `layout` output as a scene, parsed once per output.
    var layoutScene: LayoutScene? { derivedValues.scene }

    /// The automatic semi-diameters in the `layout` output, by surface id.
    var autoSemiDiameters: [SurfaceID: Double] { derivedValues.autoSemiDiameters }

    private var derivedValues: (scene: LayoutScene?, autoSemiDiameters: [SurfaceID: Double]) {
        let o = output  // tracked: views update when the output changes
        if let derived { return derived }
        let scene = EngineLayout.scene(from: o)
        let d = (scene: scene, autoSemiDiameters: EngineLayout.autoSemiDiameters(scene))
        derived = d
        return d
    }

    public var isRunning: Bool { if case .running = phase { true } else { false } }

    public var progress: JobProgress? { if case .running(let p) = phase { p } else { nil } }

    /// Whether the shown output is out of date (design, glass library or
    /// settings changed since it was computed).
    public var isStale: Bool { workspace?.isStale(self) ?? false }

    /// Shows another analysis; its settings start from the defaults.
    public func show(_ name: String, settings: [String: JSONValue] = [:]) {
        guard name != analysis || settings != self.settings else { return }
        job?.cancel()
        job = nil
        analysis = name
        self.settings = settings
        output = nil
        phase = .idle
        workspace?.paneChanged(self)
    }

    /// Changes one setting (`nil` restores the engine default) and re-runs.
    public func set(_ key: String, _ value: JSONValue?) {
        var s = settings
        s[key] = value
        guard s != settings else { return }
        settings = s
        workspace?.settingsChanged(self)
    }

    /// Replaces all settings and re-runs.
    public func setSettings(_ s: [String: JSONValue]) {
        guard s != settings else { return }
        settings = s
        workspace?.settingsChanged(self)
    }

    /// Runs the pane again. `fresh` (a run the user asked for: Run, Try
    /// Again) skips the engine's stored results and recomputes.
    public func rerun(fresh: Bool = false) { workspace?.run(self, force: true, fresh: fresh) }

    /// The settings a new pane of this analysis starts with (Defaults).
    public var startingSettings: [String: JSONValue] { workspace?.startingSettings(analysis) ?? [:] }
    public func cancel() { job?.cancel() }

    /// The value of a setting: the pane's own, else the engine default.
    public func value(_ key: String) -> JSONValue? {
        settings[key] ?? workspace?.descriptor(analysis)?.defaultSettings[key]
            ?? workspace?.descriptor(analysis)?.settingsFields.first { $0.name == key }?.defaultValue
    }
}
