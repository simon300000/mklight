import MKLightEngine
import MKLightUI
import SwiftUI

/// MKLight for macOS.
///
/// The engine is chosen here and nowhere else. Until the Rust engine's C ABI
/// (crates/mklight-ffi) is linked, the app runs on `MockEngine`, which labels
/// all of its results as synthetic.
struct MKLightApp: App {
    var body: some Scene {
        MKLightScene(engine: MockEngine())
    }
}

MKLightApp.main()
