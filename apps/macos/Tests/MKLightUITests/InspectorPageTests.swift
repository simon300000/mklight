import Testing

@testable import MKLightUI

/// Inspector pages: showing a surface (lens table "Show in Inspector", an
/// issue) leaves an open job instead of being hidden behind it.
@Suite @MainActor struct InspectorPageTests {
    @Test func showingASurfaceLeavesTheJob() {
        let w = WindowState()
        w.page = .job(.solves)
        w.page = .surface
        #expect(w.activeJob == nil && w.page == .surface)
        w.open(.tolerance)
        w.page = .system
        #expect(w.activeJob == nil && w.page == .system)
    }
}
