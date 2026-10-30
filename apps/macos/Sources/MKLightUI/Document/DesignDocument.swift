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

    public nonisolated init(project: ProjectFile) {
        _system = project.system
        _project = project
        persisted = Mutex(project)
    }

    public nonisolated convenience init(system: OpticalSystem = .newEmpty()) {
        self.init(project: ProjectFile(system: system))
    }

    /// A copy of the project for saving (any thread).
    public nonisolated var savedProject: ProjectFile { persisted.withLock { $0 } }

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

    public func isVariable(_ path: ParamPath) -> Bool { variables.contains { $0.param == path } }

    /// Makes `path` a variable or a fixed parameter (undoable).
    public func setVariable(_ path: ParamPath, _ on: Bool, undoManager: UndoManager?) {
        var list = variables
        if on {
            guard !list.contains(where: { $0.param == path }) else { return }
            list.append(DesignVariable(path))
        } else {
            list.removeAll { $0.param == path }
        }
        setSectionUndoable("variables", Self.encode(list), actionName: on ? "Make Variable" : "Remove Variable", undoManager: undoManager)
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
