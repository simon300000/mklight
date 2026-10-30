import AppKit
import MKLightEngine
import MKLightFFI
import MKLightUI
import SwiftUI

/// MKLight for macOS.
///
/// The engine is chosen here and nowhere else: the Rust engine over its C
/// ABI (`FFIEngine`). Debug builds without the engine library (see
/// scripts/build-macos.sh) fall back to `MockEngine`, which labels all of its
/// results as synthetic; release builds refuse to start without it.
@MainActor
enum EngineChoice {
    static let engine: any EngineClient = {
        do {
            return try FFIEngine()
        } catch {
            #if DEBUG
            FileHandle.standardError.write(
                Data("MKLight: \(error.localizedDescription). Using the mock engine (synthetic data).\n".utf8))
            return MockEngine()
            #else
            let alert = NSAlert()
            alert.messageText = "The MKLight engine could not be started"
            alert.informativeText = error.localizedDescription
            alert.alertStyle = .critical
            alert.runModal()
            exit(1)
            #endif
        }
    }()
}

struct MKLightApp: App {
    var body: some Scene {
        MKLightScene(engine: EngineChoice.engine)
    }
}

#if DEBUG
// Screenshots: `-MKLightAppearance light|dark` overrides the system appearance.
switch UserDefaults.standard.string(forKey: "MKLightAppearance") {
case "light": NSApplication.shared.appearance = NSAppearance(named: .aqua)
case "dark": NSApplication.shared.appearance = NSAppearance(named: .darkAqua)
default: break
}
#endif

MKLightApp.main()
