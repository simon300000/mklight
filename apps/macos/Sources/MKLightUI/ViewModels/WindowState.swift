import Foundation
import MKLightEngine
import MKLightModel
import Observation

public enum ViewportMode: String, CaseIterable, Identifiable, Sendable, Codable {
    /// The meridional (y–z) section.
    case section2D = "2D"
    /// The sagittal (x–z) section, for designs without rotational symmetry.
    case sectionXZ = "2D-xz"
    case orbit3D = "3D"
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .section2D: "Section"
        case .sectionXZ: "X–Z"
        case .orbit3D: "3D"
        }
    }
    /// A 2-D section (y–z or x–z).
    public var isSection: Bool { self != .orbit3D }
    /// The engine `layout` setting `section` for this mode.
    public var engineSection: String { self == .sectionXZ ? "xz" : "yz" }
}

public enum RayColoring: String, CaseIterable, Identifiable, Sendable, Codable {
    case field, wavelength
    public var id: String { rawValue }
    public var title: String { self == .field ? "Color by Field" : "Color by Wavelength" }
}

public enum InspectorTab: String, CaseIterable, Identifiable, Sendable {
    case surface = "Surface"
    case system = "System"
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
    /// Vertical exaggeration of the section views (1 = true proportions):
    /// slim designs are drawn `stretch` times taller than long. The 3-D
    /// view always keeps true proportions.
    public var stretch = 1.0
    /// The stretch was chosen (in the menu, or saved with the project), so
    /// the first fit keeps it instead of choosing one for a slim design.
    public var stretchChosen = false

    public init() {}

    /// The vertical scale factor in `mode`.
    public func verticalScale(_ mode: ViewportMode) -> Double { mode == .orbit3D ? 1 : max(stretch, 1e-3) }

    /// The stretches offered in the viewport controls.
    public static let stretches: [Double] = [1, 2, 5, 10, 20, 50]

    /// The stretch that makes a section of `height` points (at true
    /// proportions) readable in a view `viewHeight` points tall: 1 unless
    /// it fills less than an eighth of the view, else the smallest offered
    /// stretch that brings it to a third (at most ×50).
    public static func automaticStretch(height: Double, viewHeight: Double) -> Double {
        guard height > 0, viewHeight > 0, height < viewHeight / 8 else { return 1 }
        let want = viewHeight / 3 / height
        return stretches.first { $0 >= want } ?? stretches.last!
    }
}

/// Which design job panel is open in the inspector.
public enum DesignJobKind: String, Identifiable, Sendable, CaseIterable {
    case optimize, glass, tolerance, solves
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .optimize: "Optimize"
        case .glass: "Glass Substitution"
        case .tolerance: "Tolerance"
        case .solves: "Apply Solves"
        }
    }
    public var symbol: String {
        switch self {
        case .optimize: "wand.and.sparkles"
        case .glass: "swatchpalette"
        case .tolerance: "plusminus.circle"
        case .solves: "function"
        }
    }
}

/// Per-window UI state, kept apart from the design (`DesignDocument`) and
/// the analysis state (`Workspace`). The layout parts are saved in the
/// project's `views` section (`WorkspaceLayout`).
@MainActor
@Observable
public final class WindowState {
    /// Selected surface, by stable id so it survives inserts and deletes.
    public var selection: SurfaceID?
    /// A field to highlight in the system inspector (from an issue).
    public var highlightedField: Int?
    public var showInspector = true
    public var inspectorTab = InspectorTab.system
    public var showViewport = true
    public var showIssues = false
    public var viewportMode = ViewportMode.section2D
    public var camera = ViewportCamera()
    public var rayColoring = RayColoring.field
    /// Incremented to ask the viewport to fit the scene.
    public var fitRequest = 0
    /// Fit frames the lens only (surfaces), not the rays to the image.
    public var fitLens = false
    /// Height fraction of the lens data editor.
    public var editorFraction = 0.36
    /// Width fraction of the viewport in the lower half.
    public var viewportFraction = 0.42
    /// Width fractions of the analysis panes (normalised when used).
    public var paneWidths: [String: Double] = [:]
    /// A transient message (an edit the editor rejected).
    public var message: String?
    /// The design job shown in the inspector (`nil`: surface or system).
    public var activeJob: DesignJobKind?
    /// The window's design jobs. They live as long as the window, so a job
    /// keeps running while its panel is closed.
    public let optimize = OptimizeSession()
    public let tolerance = ToleranceSession()
    public let glass = GlassSubstitutionSession()
    public let solves = SolvesSession()
    /// The lens table's focused cell, for menu commands.
    public var focusedColumn: LensColumn?

    public init() {}

    /// Opens a design job in the inspector.
    public func open(_ job: DesignJobKind) {
        activeJob = job
        showInspector = true
    }

    /// The running design job, if any.
    public var runningJob: DesignJobKind? {
        optimize.isRunning ? .optimize : glass.isRunning ? .glass : tolerance.isRunning ? .tolerance : nil
    }

    /// Shows a transient message; it clears itself after a few seconds.
    public func flash(_ text: String?) {
        message = text
        guard let text else { return }
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(6))
            if self?.message == text { self?.message = nil }
        }
    }
}

/// The saved layout of a window, stored as the project's `views` section
/// (`[{key, analysis, settings, ui}]`, docs/FFI.md): one entry per analysis
/// pane plus a `layout` entry for the viewport and split positions.
public struct WorkspaceLayout: Equatable, Sendable {
    public struct Pane: Equatable, Sendable {
        public var key: String
        public var analysis: String
        public var settings: [String: JSONValue]
        public var autoUpdate: Bool
        public var showsSettings: Bool
        public var width: Double?
    }
    public var panes: [Pane] = []
    public var editorFraction = 0.36
    public var viewportFraction = 0.42
    public var showViewport = true
    public var viewportMode = ViewportMode.section2D
    public var rayColoring = RayColoring.field
    /// The section views' vertical stretch, when one was chosen or applied
    /// (`nil`: the first fit picks one for a slim design).
    public var stretch: Double?
    public var layoutSettings: [String: JSONValue] = [:]

    public init() {}

    /// The default for a new design: spot diagram and ray fan.
    public static var standard: WorkspaceLayout {
        var l = WorkspaceLayout()
        l.panes = [
            Pane(key: "pane-spot", analysis: "spot_diagram", settings: Workspace.spotStartingSettings, autoUpdate: true, showsSettings: false, width: nil),
            Pane(key: "pane-fan", analysis: "ray_fan", settings: [:], autoUpdate: true, showsSettings: false, width: nil),
        ]
        return l
    }

    public init(views: JSONValue) {
        for v in views.array ?? [] {
            guard let key = v["key"]?.string else { continue }
            let ui = v["ui"]?.object ?? [:]
            let settings = v["settings"]?.object ?? [:]
            if key == "layout" {
                editorFraction = ui["editor_fraction"]?.doubleValue ?? editorFraction
                viewportFraction = ui["viewport_fraction"]?.doubleValue ?? viewportFraction
                showViewport = ui["show_viewport"]?.bool ?? true
                viewportMode = ui["mode"]?.string.flatMap(ViewportMode.init(rawValue:)) ?? .section2D
                rayColoring = ui["ray_coloring"]?.string.flatMap(RayColoring.init(rawValue:)) ?? .field
                stretch = ui["stretch"]?.doubleValue.flatMap { $0.isFinite && $0 >= 1 ? $0 : nil }
                layoutSettings = settings
            } else if let analysis = v["analysis"]?.string {
                panes.append(
                    Pane(
                        key: key, analysis: analysis, settings: settings, autoUpdate: ui["auto_update"]?.bool ?? true,
                        showsSettings: ui["settings_visible"]?.bool ?? false, width: ui["width"]?.doubleValue))
            }
        }
    }

    public var views: JSONValue {
        var layoutUI: [String: JSONValue] = [
            "editor_fraction": .number(editorFraction), "viewport_fraction": .number(viewportFraction),
            "show_viewport": .bool(showViewport), "mode": .string(viewportMode.rawValue),
            "ray_coloring": .string(rayColoring.rawValue),
        ]
        if let stretch { layoutUI["stretch"] = .number(stretch) }
        var out: [JSONValue] = [
            .object([
                "key": .string("layout"), "analysis": .string("layout"), "settings": .object(layoutSettings),
                "ui": .object(layoutUI),
            ])
        ]
        for p in panes {
            var ui: [String: JSONValue] = ["auto_update": .bool(p.autoUpdate), "settings_visible": .bool(p.showsSettings)]
            if let w = p.width { ui["width"] = .number(w) }
            out.append(
                .object([
                    "key": .string(p.key), "analysis": .string(p.analysis), "settings": .object(p.settings), "ui": .object(ui),
                ]))
        }
        return .array(out)
    }

    /// Whether `views` holds a saved layout at all.
    public static func isSaved(_ views: JSONValue) -> Bool {
        (views.array ?? []).contains { $0["key"]?.string == "layout" }
    }
}

extension WorkspaceLayout {
    /// The layout of a window now.
    @MainActor
    public init(workspace: Workspace, window: WindowState) {
        panes = workspace.panes.map {
            Pane(
                key: $0.id, analysis: $0.analysis, settings: ImageSourceImport.persistable($0.settings), autoUpdate: $0.autoUpdate,
                showsSettings: $0.showsSettings, width: window.paneWidths[$0.id])
        }
        editorFraction = window.editorFraction
        viewportFraction = window.viewportFraction
        showViewport = window.showViewport
        viewportMode = window.viewportMode
        rayColoring = window.rayColoring
        let c = window.camera
        stretch = c.stretchChosen || c.stretch != 1 ? c.stretch : nil
        layoutSettings = workspace.layout.settings
    }

    /// Panes for a new workspace.
    @MainActor
    public func makePanes() -> [AnalysisPane] {
        panes.map {
            let p = AnalysisPane(id: $0.key, analysis: $0.analysis, settings: $0.settings, autoUpdate: $0.autoUpdate)
            p.showsSettings = $0.showsSettings
            return p
        }
    }

    @MainActor
    public func apply(to window: WindowState) {
        window.editorFraction = editorFraction
        window.viewportFraction = viewportFraction
        window.showViewport = showViewport
        window.viewportMode = viewportMode
        window.rayColoring = rayColoring
        if let stretch {
            window.camera.stretch = stretch
            window.camera.stretchChosen = true
        }
        window.paneWidths = Dictionary(panes.compactMap { p in p.width.map { (p.key, $0) } }, uniquingKeysWith: { a, _ in a })
    }
}
