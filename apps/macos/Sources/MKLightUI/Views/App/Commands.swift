import MKLightEngine
import MKLightModel
import SwiftUI
#if os(macOS)
import AppKit
#endif

/// What the focused design window can do; published to the menus with
/// `focusedSceneValue`.
struct WindowActions {
    var insertBefore: () -> Void
    var insertAfter: () -> Void
    var deleteSurface: () -> Void
    var makeStop: () -> Void
    var toggleVariable: () -> Void
    var canToggleVariable: Bool
    var hasSurfaceSelection: Bool
    var loadCatalog: () -> Void
    var designJob: (DesignJobKind) -> Void
    var showIssues: () -> Void
    var addPane: (String) -> Void
    var paneAnalyses: [AnalysisDescriptor]
    var rerun: () -> Void
    var cancel: () -> Void
    var fit: () -> Void
    var setMode: (ViewportMode) -> Void
    var toggleViewport: () -> Void
    var toggleLensData: () -> Void
    var showsLayout: Bool
    var showsLensData: Bool
    /// The panel the panel commands act on, if it is open.
    var focusedPanel: PanelID?
    var focusedPanelDetached: Bool
    var focusedPanelAlone: Bool
    var moveToNewWindow: (PanelID) -> Void
    var moveToMainWindow: (PanelID) -> Void
    var closePanel: (PanelID) -> Void
    var resetLayout: () -> Void
    var setColoring: (RayColoring) -> Void
    var backend: Binding<BackendPreference>
    var replaceWithSample: (OpticalSystem) -> Void
}

extension FocusedValues {
    @Entry var windowActions: WindowActions?
    /// The design's undo stack, from whichever of its windows is key.
    @Entry var documentUndo: DocumentUndo?
}

/// A design's undo stack, and a count of its changes so the menu updates.
struct DocumentUndo {
    let manager: UndoManager
    let revision: Int
}

/// Counts changes to every design's undo stack. The menu commands read
/// it, so Undo and Redo update as the stack changes, whichever window is
/// key (focused values from a separate window reach the menus only when
/// the focus moves).
@MainActor
@Observable
final class UndoMenuState {
    static let shared = UndoMenuState()
    var revision = 0
}

struct MKLightCommands: Commands {
    @FocusedValue(\.windowActions) private var actions
    @FocusedValue(\.documentUndo) private var undo
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        // Undo and Redo act on the design's document from any of its windows,
        // the separate panel windows included; text being edited undoes its
        // typing first.
        CommandGroup(replacing: .undoRedo) {
            let _ = UndoMenuState.shared.revision
            let u = undo?.manager
            Button(u.flatMap { $0.canUndo ? $0.undoMenuItemTitle : nil } ?? "Undo") { Self.perform(redo: false, u) }
                .keyboardShortcut("z", modifiers: .command)
                .disabled(!(u?.canUndo ?? false) && !Self.editingText)
            Button(u.flatMap { $0.canRedo ? $0.redoMenuItemTitle : nil } ?? "Redo") { Self.perform(redo: true, u) }
                .keyboardShortcut("z", modifiers: [.command, .shift])
                .disabled(!(u?.canRedo ?? false) && !Self.editingText)
        }
        CommandGroup(after: .newItem) {
            SampleMenu()
            Button("Show Welcome Window") { openWindow(id: WelcomeView.windowID) }
                .keyboardShortcut("1", modifiers: [.command, .shift])
        }
        SidebarCommands()
        InspectorCommands()
        CommandMenu("Lens") {
            Button("Insert Surface Before") { actions?.insertBefore() }
                .keyboardShortcut("i", modifiers: [.command, .shift])
                .disabled(actions == nil)
            Button("Insert Surface After") { actions?.insertAfter() }
                .keyboardShortcut("i", modifiers: [.command, .option])
                .disabled(actions == nil)
            Button("Delete Surface") { actions?.deleteSurface() }
                .keyboardShortcut(.delete, modifiers: [.command])
                .disabled(actions?.hasSurfaceSelection != true)
            Divider()
            Button("Make Aperture Stop") { actions?.makeStop() }
                .keyboardShortcut("t", modifiers: [.command, .shift])
                .disabled(actions?.hasSurfaceSelection != true)
            Button("Toggle Variable") { actions?.toggleVariable() }
                .keyboardShortcut("v", modifiers: [.command, .control])
                .disabled(actions?.canToggleVariable != true)
            Divider()
            Menu("Replace with Sample") {
                ForEach(SampleSystems.bundled) { e in
                    Button(e.system.name) { actions?.replaceWithSample(e.system) }
                }
            }
            .disabled(actions == nil)
            Divider()
            Button("Glass Catalogs…") { openWindow(id: GlassBrowserView.windowID) }
                .keyboardShortcut("g", modifiers: [.command, .option])
            Button("Add Glass Catalog to Design…") { actions?.loadCatalog() }
                .disabled(actions == nil)
        }
        CommandMenu("Design") {
            Group {
            Button("Optimize…") { actions?.designJob(.optimize) }
                .keyboardShortcut("o", modifiers: [.command, .control])
            Button("Glass Substitution…") { actions?.designJob(.glass) }
                .keyboardShortcut("g", modifiers: [.command, .control])
            Button("Tolerance…") { actions?.designJob(.tolerance) }
                .keyboardShortcut("t", modifiers: [.command, .control])
            Button("Apply Solves…") { actions?.designJob(.solves) }
                .keyboardShortcut("l", modifiers: [.command, .control])
            Divider()
            Button("Show Issues") { actions?.showIssues() }
                .keyboardShortcut("e", modifiers: [.command, .control])
            }
            .disabled(actions == nil)
        }
        CommandMenu("Analysis") {
            Group {
            Menu("New Analysis Pane") {
                ForEach(actions?.paneAnalyses ?? []) { d in
                    Button(AnalysisCatalog.title(of: d)) { actions?.addPane(d.name) }.disabled(!d.available)
                }
            }
            Divider()
            Button("Run Analyses") { actions?.rerun() }
                .keyboardShortcut("r", modifiers: [.command])
            Button("Cancel Running Analyses") { actions?.cancel() }
                .keyboardShortcut(".", modifiers: [.command])
            Divider()
            if let backend = actions?.backend {
                Picker("Compute On", selection: backend) {
                    Text("Automatic").tag(BackendPreference.auto)
                    Text("CPU Reference (f64)").tag(BackendPreference.cpuReference)
                    Text("GPU").tag(BackendPreference.gpu)
                }
            }
            }
            .disabled(actions == nil)
        }
        CommandGroup(before: .sidebar) {
            Button(actions?.showsLensData == true ? "Hide Lens Data" : "Show Lens Data") { actions?.toggleLensData() }
                .keyboardShortcut("e", modifiers: [.command, .option])
                .disabled(actions == nil)
            Button(actions?.showsLayout == false ? "Show Layout" : "Hide Layout") { actions?.toggleViewport() }
                .keyboardShortcut("l", modifiers: [.command, .option])
                .disabled(actions == nil)
            Menu("Panel") {
                let panel = actions?.focusedPanel
                Button("Move to New Window") { if let panel { actions?.moveToNewWindow(panel) } }
                    .disabled(panel == nil || actions?.focusedPanelAlone == true)
                Button("Move to Main Window") { if let panel { actions?.moveToMainWindow(panel) } }
                    .disabled(panel == nil || actions?.focusedPanelDetached != true)
                Button(panel == .lensData ? "Hide Lens Data" : panel == .layout ? "Hide Layout" : "Close Pane") { if let panel { actions?.closePanel(panel) } }
                    .disabled(panel == nil)
                Divider()
                Button("Reset Panel Layout") { actions?.resetLayout() }
                    .disabled(actions == nil)
            }
            Divider()
            Button("Section View") { actions?.setMode(.section2D) }
                .keyboardShortcut("2", modifiers: [.command])
            Button("X–Z Section") { actions?.setMode(.sectionXZ) }
                .keyboardShortcut("2", modifiers: [.command, .option])
            Button("3D View") { actions?.setMode(.orbit3D) }
                .keyboardShortcut("3", modifiers: [.command])
            Button("Fit Layout") { actions?.fit() }
                .keyboardShortcut("0", modifiers: [.command])
            Menu("Ray Colors") {
                ForEach(RayColoring.allCases) { c in Button(c.title) { actions?.setColoring(c) } }
            }
            Divider()
        }
    }
}

extension MKLightCommands {
    /// A text field is being edited (its field editor has the focus).
    @MainActor static var editingText: Bool {
        #if os(macOS)
        NSApp.keyWindow?.firstResponder is NSTextView
        #else
        false
        #endif
    }

    /// Undo or redo: through the field editor while text is edited (it
    /// closes its typing group first), else on the document's stack.
    @MainActor static func perform(redo: Bool, _ manager: UndoManager?) {
        #if os(macOS)
        if editingText {
            NSApp.sendAction(Selector(redo ? "redo:" : "undo:"), to: nil, from: nil)
            return
        }
        #endif
        guard let manager, manager.groupingLevel == 0 else { return }
        if redo { manager.redo() } else { manager.undo() }
    }
}

/// File ▸ New from Sample: the engine's built-in samples.
struct SampleMenu: View {
    @Environment(\.newDocument) private var newDocument

    var body: some View {
        Menu("New from Sample") {
            ForEach(SampleSystems.bundled) { e in
                Button(e.system.name) { newDocument { ProjectDocument(system: e.system) } }
            }
        }
    }
}
