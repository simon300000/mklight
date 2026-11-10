import MKLightEngine
import MKLightModel
import SwiftUI

/// The app's scenes: `.mklight` documents, a welcome window with the sample
/// designs, and the menus. The executable passes the engine; views only see
/// `EngineClient`.
public struct MKLightScene: Scene {
    let engine: any EngineClient

    public init(engine: any EngineClient) {
        self.engine = engine
        ProjectDocument.setImportEngine(engine)
        GPUPreparationMonitor.shared.start(engine: engine)
    }

    public var body: some Scene {
        DocumentGroup(newDocument: { ProjectDocument() }) { config in
            DesignWindow(document: config.document, fileURL: config.fileURL, engine: engine)
                .frame(minWidth: 960, minHeight: 620)
                // One number format app-wide (NumberDisplay): a decimal point, no grouping.
                .environment(\.locale, NumberDisplay.locale)
        }
        .defaultSize(width: 1520, height: 940)
        .commands { MKLightCommands() }

        Window("Welcome to MKLight", id: WelcomeView.windowID) {
            WelcomeView(engine: engine)
                .environment(\.locale, NumberDisplay.locale)
        }
        .windowResizability(.contentSize)
        .defaultPosition(.center)
        #if os(macOS)
        .windowStyle(.hiddenTitleBar)
        .defaultLaunchBehavior(ProcessInfo.processInfo.arguments.contains("-MKLightNoWelcome") ? .suppressed : .presented)
        #endif
    }
}
