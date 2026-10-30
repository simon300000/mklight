import MKLightEngine
import MKLightModel
import SwiftUI

/// The app's scenes: a document group of `.mklight` projects plus menus.
/// The executable passes the engine; UI code only sees `EngineClient`.
public struct MKLightScene: Scene {
    let engine: any EngineClient

    public init(engine: any EngineClient) {
        self.engine = engine
    }

    public var body: some Scene {
        DocumentGroup(newDocument: { ProjectDocument() }) { config in
            DocumentWindow(document: config.document)
                .environment(\.mklightEngine, engine)
                .frame(minWidth: 900, minHeight: 600)
        }
        .commands { MKLightCommands() }
    }
}

struct MKLightCommands: Commands {
    @FocusedValue(\.documentActions) private var actions

    var body: some Commands {
        CommandGroup(after: .newItem) {
            SampleMenu()
        }
        CommandMenu("Lens") {
            Button("Insert Surface Before") { actions?.insertBefore() }
                .keyboardShortcut("i", modifiers: [.command, .shift])
                .disabled(actions == nil)
            Button("Insert Surface After") { actions?.insertAfter() }
                .keyboardShortcut("i", modifiers: [.command, .option])
                .disabled(actions == nil)
            Button("Delete Surface") { actions?.deleteSurface() }
                .keyboardShortcut(.delete, modifiers: [.command])
                .disabled(actions?.hasSelection != true)
            Divider()
            Button("Make Stop") { actions?.makeStop() }
                .keyboardShortcut("t", modifiers: [.command, .shift])
                .disabled(actions?.hasSelection != true)
            Divider()
            Menu("Replace with Sample") {
                Button("Cooke Triplet") { actions?.loadSample(SampleSystems.cookeTriplet()) }
                Button("Singlet") { actions?.loadSample(SampleSystems.singlet()) }
                Button("Empty System") { actions?.loadSample(.newEmpty()) }
            }
            .disabled(actions == nil)
        }
        CommandMenu("Analysis") {
            Button("Re-run Visible Analyses") { actions?.rerunAnalyses() }
                .keyboardShortcut("r", modifiers: [.command])
                .disabled(actions == nil)
            Button("Cancel Running Analyses") { actions?.cancelAnalyses() }
                .keyboardShortcut(".", modifiers: [.command])
                .disabled(actions == nil)
            Divider()
            Button("Fit Layout") { actions?.fitView() }
                .keyboardShortcut("0", modifiers: [.command])
                .disabled(actions == nil)
            Button("Toggle 2D / 3D Layout") { actions?.toggleViewportMode() }
                .keyboardShortcut("3", modifiers: [.command])
                .disabled(actions == nil)
        }
    }
}

/// File ▸ New from Sample.
struct SampleMenu: View {
    @Environment(\.newDocument) private var newDocument

    var body: some View {
        Menu("New from Sample") {
            Button("Cooke Triplet") { newDocument { ProjectDocument(system: SampleSystems.cookeTriplet()) } }
            Button("Singlet") { newDocument { ProjectDocument(system: SampleSystems.singlet()) } }
        }
    }
}
