import Foundation
import MKLightEngine
import MKLightModel
import Observation

public enum ViewportMode: String, CaseIterable, Identifiable, Sendable {
    case section2D = "2D"
    case orbit3D = "3D"
    public var id: String { rawValue }
    public var title: String { self == .section2D ? "Y-Z Section" : "3D" }
}

public enum RayColoring: String, CaseIterable, Identifiable, Sendable {
    case field, wavelength
    public var id: String { rawValue }
    public var title: String { self == .field ? "Color by Field" : "Color by Wavelength" }
}

public enum InspectorTab: String, CaseIterable, Identifiable, Sendable {
    case system = "System"
    case surface = "Surface"
    public var id: String { rawValue }
}

/// Orthographic viewport camera. Pure view state: it never enters the model.
public struct ViewportCamera: Equatable, Sendable {
    /// World point (mm) at the centre of the view.
    public var center = SIMD3<Double>(0, 0, 0)
    /// Zoom in points per millimetre.
    public var pointsPerMM = 4.0
    /// Orbit angles (radians) for the 3-D view.
    public var yaw = -0.55
    public var pitch = 0.35
    /// False until the camera has been fitted to a scene.
    public var isFitted = false

    public init() {}
}

/// Per-window UI state, kept apart from the document's simulation state.
@MainActor
@Observable
public final class WindowViewState {
    /// Selected surface, by stable id so it survives inserts and deletes.
    public var selection: SurfaceID?
    public var showInspector = true
    public var inspectorTab = InspectorTab.system
    public var showLayout = true
    public var showAnalyses = true
    public var selectedAnalysis = AnalysisKind.spotDiagram
    /// The analysis area shows the "Engine" pane (any engine analysis by
    /// name) instead of `selectedAnalysis`.
    public var showsEnginePane = false
    /// Engine analysis selected in the Engine pane.
    public var engineAnalysis = "system_summary"
    /// Settings per engine analysis (partial; the engine fills defaults).
    public var engineSettings: [String: [String: JSONValue]] = [:]
    public var viewportMode = ViewportMode.section2D
    public var camera = ViewportCamera()
    public var rayColoring = RayColoring.field
    /// Incremented to ask the viewport to fit the scene.
    public var fitRequest = 0
    public var psfLogScale = false
    /// Settings of every analysis; together with the snapshot they determine
    /// each result.
    public var analysisSettings = AnalysisSettingsSet()
    /// Last input error from the lens editor, shown in the status bar.
    public var editorMessage: String?

    public init() {}

    /// Analyses whose results are on screen.
    public var visibleAnalyses: Set<AnalysisKind> {
        var s: Set<AnalysisKind> = [.firstOrder]
        if showLayout { s.insert(.layout) }
        if showAnalyses && !showsEnginePane { s.insert(selectedAnalysis) }
        return s
    }

    /// The analysis area's selection: a typed analysis or the Engine pane.
    public var analysisPane: AnalysisPane {
        get { showsEnginePane ? .engine : .typed(selectedAnalysis) }
        set {
            switch newValue {
            case .engine: showsEnginePane = true
            case .typed(let k):
                showsEnginePane = false
                selectedAnalysis = k
            }
        }
    }

    /// The request of the Engine pane while it is shown.
    public var engineRequest: EngineAnalysisRequest? {
        guard showAnalyses && showsEnginePane else { return nil }
        return EngineAnalysisRequest(
            analysis: engineAnalysis, settings: engineSettings[engineAnalysis] ?? [:],
            execution: analysisSettings.execution)
    }
}

/// What the analysis area shows.
public enum AnalysisPane: Hashable, Sendable {
    case typed(AnalysisKind)
    /// Any engine analysis by name, shown generically.
    case engine
}
