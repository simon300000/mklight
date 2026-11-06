import Foundation
import MKLightEngine
import MKLightModel
import Observation
import Synchronization

/// A variable of the design: a parameter the optimiser may change, with
/// optional bounds (the engine's `VariableSpec`, stored in the project's
/// `variables` section).
public struct DesignVariable: Codable, Sendable, Hashable {
    public var param: ParamPath
    public var lower: Double?
    public var upper: Double?

    public init(_ param: ParamPath, lower: Double? = nil, upper: Double? = nil) {
        self.param = param
        self.lower = lower
        self.upper = upper
    }

    /// Held at one value (equal bounds): kept as a variable but not varied.
    public var isPinned: Bool { lower != nil && lower == upper }
    public var isBounded: Bool { lower != nil || upper != nil }

    /// Whether `value` violates the bounds.
    public func outOfBounds(_ value: Double) -> Bool {
        (lower.map { value < $0 - 1e-12 } ?? false) || (upper.map { value > $0 + 1e-12 } ?? false)
    }

    /// Sensible bounds for a new variable: air gaps stay non-negative,
    /// glass keeps a minimum centre thickness, model glasses stay in the
    /// range of real glasses, semi-diameters and normalisation radii stay
    /// positive. Curvatures, conics and coefficients are left free.
    public static func defaultBounds(for param: ParamPath, in system: OpticalSystem) -> (lower: Double?, upper: Double?) {
        switch param {
        case .thickness(let id):
            guard let i = system.surfaceIndex(of: id), i > 0, i < system.imageIndex else { return (nil, nil) }
            let s = system.surfaces[i]
            if s.isCoordinateBreak { return (nil, nil) }
            if i == system.imageIndex - 1 { return (0, nil) }
            return s.material.isAir ? (0, nil) : (glassMinimum, nil)
        case .semiDiameter: return (0, nil)
        case .modelNd: return (1.4, 2.1)
        case .modelVd: return (15, 100)
        // A melt offset stays within the engine's validation.
        case .materialDnD: return (-MaterialOffset.maxDnD, MaterialOffset.maxDnD)
        case .materialDvD: return (-MaterialOffset.maxDvD, MaterialOffset.maxDvD)
        case .shapeParam(_, .normRadius): return (1e-3, nil)
        default: return (nil, nil)
        }
    }

    /// The smallest glass centre thickness a new glass-thickness variable gets.
    public static let glassMinimum = 1.0

    private enum CodingKeys: String, CodingKey { case param, lower, upper }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        param = try c.decode(ParamPath.self, forKey: .param)
        lower = try c.decodeIfPresent(Double.self, forKey: .lower)
        upper = try c.decodeIfPresent(Double.self, forKey: .upper)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(param, forKey: .param)
        try c.encodeIfPresent(lower, forKey: .lower)
        try c.encodeIfPresent(upper, forKey: .upper)
    }
}

/// The simulation state of one design: the optical system plus the other
/// project sections (catalogs, variables, views, metadata, unknown keys).
/// It knows nothing about windows, selection or panes.
///
/// Every edit replaces the system with a new immutable value and registers
/// the previous value with the undo manager, so undo and redo restore whole
/// snapshots; the window's `Workspace` freezes each value into an engine
/// snapshot (content-hashed) and runs analyses against it. Changes that are
/// not design changes (the saved pane layout) are written without undo.
///
/// Platform-neutral: Foundation and Observation only.
@MainActor
@Observable
public final class DesignDocument {
    public private(set) var system: OpticalSystem
    /// The project sections other than the system.
    public private(set) var project: ProjectFile
    /// Incremented on every change of `system` (including undo and redo).
    public private(set) var revision = 0

    /// The project as last changed, readable from any thread (the document
    /// machinery may save off the main thread).
    @ObservationIgnored nonisolated let persisted: Mutex<ProjectFile>

    /// What importing the lens file this design came from left out, until
    /// its window takes it (`takeImportReport`).
    @ObservationIgnored private var pendingImportReport: ImportReport?

    public nonisolated init(project: ProjectFile, importReport: ImportReport? = nil) {
        _system = project.system
        _project = project
        persisted = Mutex(project)
        pendingImportReport = importReport
    }

    public nonisolated convenience init(system: OpticalSystem = .newEmpty()) {
        self.init(project: ProjectFile(system: system))
    }

    /// A copy of the project for saving (any thread).
    public nonisolated var savedProject: ProjectFile { persisted.withLock { $0 } }

    /// The import report of a design opened from a lens prescription file,
    /// once: later calls return `nil`.
    public func takeImportReport() -> ImportReport? {
        defer { pendingImportReport = nil }
        return pendingImportReport
    }

    // MARK: Editing

    /// Applies `change` to a copy of the system. If the result differs, it
    /// becomes the current value, undoable as `actionName`. Configuration
    /// overrides, solves and mechanics the change invalidates are dropped in
    /// the same step; variables of removed parameters too. Returns whether
    /// anything changed.
    @discardableResult
    public func edit(_ actionName: String, undoManager: UndoManager?, _ change: (inout OpticalSystem) -> Void) -> Bool {
        var next = system
        change(&next)
        next.dropReferencesInvalidated(since: system)
        guard next != system else { return false }
        replace(system: next, actionName: actionName, undoManager: undoManager)
        return true
    }

    /// Replaces the whole system (an optimisation or solve result) as one
    /// undoable step.
    public func replace(system value: OpticalSystem, actionName: String, undoManager: UndoManager?) {
        guard value != system else { return }
        let previousSystem = system
        let previousVariables = project.sections["variables"]
        system = value
        project.system = value
        // Variables of parameters the new system no longer has go with it.
        let kept = variables.filter { $0.param.resolves(in: value) }
        if kept.count != variables.count { setSection("variables", Self.encode(kept)) }
        revision += 1
        persisted.withLock { $0.system = value; $0.sections["variables"] = project.sections["variables"] }
        undoManager?.registerUndo(withTarget: self) { doc in
            MainActor.assumeIsolated {
                doc.replace(system: previousSystem, actionName: actionName, undoManager: undoManager)
                if let previousVariables { doc.setSection("variables", previousVariables) }
            }
        }
        undoManager?.setActionName(actionName)
    }

    // MARK: Variables

    /// The design's variables (`variables` section), in order.
    public var variables: [DesignVariable] {
        guard let v = project.sections["variables"], case .array = v,
            let list = try? MKLightJSON.decode([DesignVariable].self, from: MKLightJSON.encode(v))
        else { return [] }
        return list
    }

    /// Whether `path` (or another path to the same parameter) is a variable.
    public func isVariable(_ path: ParamPath) -> Bool { variable(path) != nil }

    /// Why `path` cannot become a variable: a solve sets it, and `optimize`
    /// applies solves first and refuses solved parameters as variables.
    public func variableRefusal(_ path: ParamPath) -> String? {
        guard let s = system.solve(for: path) else { return nil }
        return "Set by a solve (\(s.kind.kind.title)), so the optimizer cannot vary it. Remove the solve to make it a variable."
    }

    /// Makes `path` a variable or a fixed parameter (undoable). A parameter
    /// a solve sets cannot become one (`variableRefusal`).
    public func setVariable(_ path: ParamPath, _ on: Bool, undoManager: UndoManager?) {
        var list = variables
        let p = system.canonical(path)
        if on {
            guard variableRefusal(path) == nil, !list.contains(where: { system.canonical($0.param) == p }) else { return }
            let b = DesignVariable.defaultBounds(for: path, in: system)
            list.append(DesignVariable(path, lower: b.lower, upper: b.upper))
        } else {
            list.removeAll { system.canonical($0.param) == p }
        }
        setSectionUndoable("variables", Self.encode(list), actionName: on ? "Make Variable" : "Remove Variable", undoManager: undoManager)
    }

    /// Sets a variable's bounds (`nil`: unbounded on that side), undoable.
    public func setBounds(_ path: ParamPath, lower: Double?, upper: Double?, undoManager: UndoManager?) {
        var list = variables
        guard let i = list.firstIndex(where: { $0.param == path }) else { return }
        guard list[i].lower != lower || list[i].upper != upper else { return }
        let action = lower != nil && lower == upper ? "Pin Variable" : list[i].isPinned ? "Unpin Variable" : "Change Bounds"
        list[i].lower = lower
        list[i].upper = upper
        setSectionUndoable("variables", Self.encode(list), actionName: action, undoManager: undoManager)
    }

    /// The variable of `path`, if it is one.
    public func variable(_ path: ParamPath) -> DesignVariable? {
        let p = system.canonical(path)
        return variables.first { system.canonical($0.param) == p }
    }

    // MARK: Merit function

    /// The design's merit function (`merit` section): the default
    /// generator's settings or custom operands.
    public var merit: MeritSettings { MeritSettings(section: project.sections["merit"]) }

    /// Replaces the merit function (undoable).
    public func setMerit(_ value: MeritSettings, actionName: String, undoManager: UndoManager?) {
        guard value != merit else { return }
        setSectionUndoable("merit", value.section, actionName: actionName, undoManager: undoManager)
    }

    // MARK: Tolerances

    /// The design's tolerancing (`tolerances` section). A design that never
    /// had one starts from the generator's default set with the back focus
    /// as compensator (not saved until edited).
    public var tolerances: ToleranceSettings {
        let t = ToleranceSettings(section: project.sections["tolerances"], system: system)
        if project.sections["tolerances"] == nil || project.sections["tolerances"] == .null { return .defaults(for: system) }
        return t
    }

    /// Replaces the tolerancing (undoable). `context` expands the rows to
    /// the engine's operands kept in the section.
    public func setTolerances(_ value: ToleranceSettings, context: ToleranceContext?, actionName: String, undoManager: UndoManager?) {
        let section = value.section(context: context, merit: merit)
        guard section != project.sections["tolerances"] else { return }
        setSectionUndoable("tolerances", section, actionName: actionName, undoManager: undoManager)
    }

    /// Removes every variable (undoable).
    public func clearVariables(undoManager: UndoManager?) {
        guard !variables.isEmpty else { return }
        setSectionUndoable("variables", .array([]), actionName: "Remove All Variables", undoManager: undoManager)
    }

    static func encode(_ list: [DesignVariable]) -> JSONValue {
        (try? JSONValue.from(list)) ?? .array([])
    }

    // MARK: Other sections

    /// User glass catalogs referenced by the project.
    public var catalogs: [CatalogReference] { project.catalogs }

    /// Records a user glass catalog (undoable).
    public func addCatalog(_ ref: CatalogReference, undoManager: UndoManager?) {
        var next = project
        next.addCatalog(ref)
        guard next.sections["catalogs"] != project.sections["catalogs"] else { return }
        setSectionUndoable("catalogs", next.sections["catalogs"] ?? .array([]), actionName: "Add Glass Catalog", undoManager: undoManager)
    }

    /// The saved window layout (`views` section).
    public var views: JSONValue { project.sections["views"] ?? .array([]) }

    /// Stores the window layout. Not a design change: no undo, and it is
    /// written with the next save.
    public func setViews(_ value: JSONValue) {
        guard project.sections["views"] != value else { return }
        setSection("views", value)
    }

    private func setSectionUndoable(_ key: String, _ value: JSONValue, actionName: String, undoManager: UndoManager?) {
        let previous = project.sections[key] ?? .null
        guard previous != value else { return }
        setSection(key, value)
        undoManager?.registerUndo(withTarget: self) { doc in
            MainActor.assumeIsolated {
                doc.setSectionUndoable(key, previous, actionName: actionName, undoManager: undoManager)
            }
        }
        undoManager?.setActionName(actionName)
    }

    private func setSection(_ key: String, _ value: JSONValue) {
        project.sections[key] = value
        persisted.withLock { $0.sections[key] = value }
    }
}
