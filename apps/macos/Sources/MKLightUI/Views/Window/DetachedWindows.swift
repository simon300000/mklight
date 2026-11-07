#if os(macOS)
import AppKit
import MKLightModel
import SwiftUI

/// A window that panels were moved out to. It shows a part of the design's
/// session, so it updates with the main window: the same selection, the
/// same analyses, edits on the document's undo stack. Closing it closes
/// its panels; Move to Main Window puts them back. It closes when the main
/// window closes and opens again with the project.
@MainActor
final class DetachedPanelWindowController: NSWindowController, NSWindowDelegate {
    private weak var session: DesignSession?
    let model: PanelWindowModel
    private var quiet = false

    init(session: DesignSession, model: PanelWindowModel) {
        self.session = session
        self.model = model
        let window = DocumentPanelWindow(
            contentRect: CGRect(x: 0, y: 0, width: 640, height: 480),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        window.documentUndoManager = session.undoManager
        window.isReleasedWhenClosed = false
        window.isRestorable = false
        window.tabbingMode = .disallowed
        window.toolbarStyle = .unifiedCompact
        window.minSize = CGSize(width: 360, height: 260)
        let host = NSHostingController(
            rootView: AnyView(DetachedPanelsView(session: session, model: model).environment(\.locale, NumberDisplay.locale)))
        host.sceneBridgingOptions = [.toolbars, .title]
        host.sizingOptions = []
        window.contentViewController = host
        super.init(window: window)
        window.delegate = self
        let frame = model.frame.flatMap(Self.visible) ?? Self.cascade(from: session.mainWindow)
        window.setFrame(frame, display: false)
        model.frame = window.frame
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("not used") }

    /// Closes the window without closing its panels (they moved, or the
    /// main window is closing).
    func closeQuietly() {
        quiet = true
        window?.close()
    }

    /// Document commands (Save, Save As, Revert, Duplicate, Print…) go to
    /// the design's document, as from the main window.
    override func supplementalTarget(forAction action: Selector, sender: Any?) -> Any? {
        if let doc = session?.mainWindow?.windowController?.document as? NSDocument, doc.responds(to: action) { return doc }
        return super.supplementalTarget(forAction: action, sender: sender)
    }

    func windowWillClose(_ notification: Notification) {
        if !quiet { session?.windowClosed(model) }
    }

    func windowDidMove(_ notification: Notification) { recordFrame() }
    func windowDidEndLiveResize(_ notification: Notification) { recordFrame() }

    private func recordFrame() {
        guard let f = window?.frame, model.frame != f else { return }
        model.frame = f
    }

    /// A saved frame, if it is still on one of the screens.
    private static func visible(_ f: CGRect) -> CGRect? {
        NSScreen.screens.contains { $0.visibleFrame.intersection(f).width > 80 && $0.visibleFrame.intersection(f).height > 80 } ? f : nil
    }

    private static func cascade(from main: NSWindow?) -> CGRect {
        let m = main?.frame ?? NSScreen.main?.visibleFrame ?? CGRect(x: 100, y: 100, width: 1200, height: 800)
        return CGRect(x: m.midX - 280, y: m.midY - 240, width: 560, height: 440)
    }
}

/// Edits made in a separate window go on the document's undo stack, like
/// the main window's (Edit ▸ Undo, and the document's edited state).
final class DocumentPanelWindow: NSWindow {
    weak var documentUndoManager: UndoManager?
    override var undoManager: UndoManager? { documentUndoManager ?? super.undoManager }
}

/// A separate window's content: its panels, and a toolbar to add an
/// analysis here or move everything back into the main window.
struct DetachedPanelsView: View {
    let session: DesignSession
    let model: PanelWindowModel

    var body: some View {
        PanelWindowContent(session: session, model: model)
            .navigationTitle(title)
            .navigationSubtitle(session.design.system.name)
            .toolbar {
                ToolbarItemGroup(placement: .primaryAction) {
                    Menu {
                        ForEach(AnalysisCatalog.grouped(session.workspace.paneAnalyses), id: \.category) { group in
                            Section(AnalysisCatalog.categoryTitle(group.category)) {
                                ForEach(group.items) { d in
                                    Button {
                                        session.addPane(d.name, inWindow: model.id)
                                    } label: {
                                        Label(AnalysisCatalog.title(of: d), systemImage: AnalysisCatalog.symbol(for: d.name))
                                    }
                                    .disabled(!d.available)
                                }
                            }
                        }
                    } label: {
                        Label("Add Analysis", systemImage: "plus.rectangle.on.rectangle")
                    }
                    .help("Add an analysis pane to this window")
                    Button("Move to Main Window", systemImage: "pip.enter") { session.dock(model) }
                        .help("Move these panels back into the main window")
                }
            }
            .background { DetachedObservers(session: session) }
            .frame(minWidth: 360, minHeight: 240)
    }

    /// The panels in front, e.g. "Spot Diagram, Ray Fan".
    private var title: String {
        let shown = model.groups.compactMap(\.current).map(session.title(of:))
        guard !shown.isEmpty else { return "Panels" }
        return shown.count > 3 ? shown.prefix(3).joined(separator: ", ") + " +\(shown.count - 3)" : shown.joined(separator: ", ")
    }
}

/// Publishes the design's menu actions while a separate window is key.
private struct DetachedObservers: View {
    let session: DesignSession

    var body: some View {
        Color.clear
            .focusedSceneValue(\.windowActions, session.actions())
            .focusedSceneValue(\.documentUndo, session.undoState)
            .accessibilityHidden(true)
    }
}
#endif
