import MKLightEngine
import MKLightModel
import SwiftUI

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
    var issueCount: Int
    var addPane: (String) -> Void
    var paneAnalyses: [AnalysisDescriptor]
    var rerun: () -> Void
    var cancel: () -> Void
    var fit: () -> Void
    var setMode: (ViewportMode) -> Void
    var toggleViewport: () -> Void
    var setColoring: (RayColoring) -> Void
    var backend: Binding<BackendPreference>
    var replaceWithSample: (OpticalSystem) -> Void
}

extension FocusedValues {
    @Entry var windowActions: WindowActions?
}

struct MKLightCommands: Commands {
    @FocusedValue(\.windowActions) private var actions
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
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
            Button("Load Glass Catalog…") { actions?.loadCatalog() }
                .disabled(actions == nil)
        }
        CommandMenu("Design") {
            Group {
            Button("Optimize…") { actions?.designJob(.optimize) }
                .keyboardShortcut("o", modifiers: [.command, .control])
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
            Button("Show Layout") { actions?.toggleViewport() }
                .keyboardShortcut("l", modifiers: [.command, .option])
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
