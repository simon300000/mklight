import MKLightEngine
import MKLightModel
import SwiftUI
#if os(macOS)
import AppKit
#endif

/// Everything one open design shares across its windows: the document,
/// the workspace (snapshots, panes, jobs), the window state (selection,
/// camera, design jobs) and where the panels are. The main window owns it;
/// the separate panel windows show parts of it, so a selection, an edit or
/// an analysis update reaches every window at once.
@MainActor
@Observable
final class DesignSession {
    let design: DesignDocument
    let workspace: Workspace
    let window: WindowState
    let arrangement: PanelArrangement
    let drag = PanelDragController()
    /// The panel the panel commands act on (Move to New Window, Close).
    var focusedPanel: PanelID?
    /// The document's undo manager (the main window's): edits made in a
    /// separate window are the document's edits.
    @ObservationIgnored var undoManager: UndoManager? {
        didSet { if undoManager !== oldValue { observeUndo() } }
    }
    /// Counts changes to the undo stack, so the Edit menu's Undo and Redo
    /// (titles, enabled state) follow it in every window of the design.
    private(set) var undoRevision = 0
    @ObservationIgnored private var undoObservers: [any NSObjectProtocol] = []
    /// Asks the main window for a glass catalog file.
    @ObservationIgnored var requestCatalog: () -> Void = {}
    /// New panes to place in a given separate window, by pane id.
    @ObservationIgnored private var pendingWindow: [String: UUID] = [:]

    init(design: DesignDocument, workspace: Workspace, window: WindowState, arrangement: PanelArrangement) {
        self.design = design
        self.workspace = workspace
        self.window = window
        self.arrangement = arrangement
        drag.session = self
    }

    private func observeUndo() {
        for o in undoObservers { NotificationCenter.default.removeObserver(o) }
        undoObservers = []
        guard let undoManager else { return }
        let names: [Notification.Name] = [.NSUndoManagerDidCloseUndoGroup, .NSUndoManagerDidUndoChange, .NSUndoManagerDidRedoChange]
        undoObservers = names.map { name in
            NotificationCenter.default.addObserver(forName: name, object: undoManager, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.undoRevision += 1
                    UndoMenuState.shared.revision += 1
                }
            }
        }
    }

    /// The undo stack for the Edit menu.
    var undoState: DocumentUndo? { undoManager.map { DocumentUndo(manager: $0, revision: undoRevision) } }

    // MARK: Panels

    func title(of panel: PanelID) -> String {
        switch panel {
        case .lensData: "Lens Data"
        case .layout: "Layout"
        case .analysis(let id): pane(id).map { AnalysisCatalog.title(of: $0.analysis) } ?? "Analysis"
        }
    }

    func symbol(of panel: PanelID) -> String {
        switch panel {
        case .lensData: "tablecells"
        case .layout: "rectangle.lefthalf.inset.filled"
        case .analysis(let id): pane(id).map { AnalysisCatalog.symbol(for: $0.analysis) } ?? "chart.xyaxis.line"
        }
    }

    func pane(_ id: String) -> AnalysisPane? { workspace.panes.first { $0.id == id } }

    /// Whether the panel is in a separate window.
    func isDetached(_ panel: PanelID) -> Bool {
        arrangement.location(of: panel).map { $0.window !== arrangement.main } ?? false
    }

    /// Whether the panel is alone in a separate window (moving it to a new
    /// window would change nothing).
    func isAloneInWindow(_ panel: PanelID) -> Bool {
        arrangement.location(of: panel).map { $0.window !== arrangement.main && $0.window.panels == [panel] } ?? false
    }

    func focus(_ panel: PanelID) {
        if focusedPanel != panel { focusedPanel = panel }
        if let id = panel.paneID, workspace.focusedPaneID != id { workspace.focusedPaneID = id }
    }

    /// Moves a panel (a drop, or a menu command). A new window opens at
    /// `frame`; the only panel of a separate window moves that window.
    func move(_ panel: PanelID, to target: PanelDropTarget, frame: CGRect? = nil) {
        #if os(macOS)
        if target == .newWindow, case let (w, _)? = arrangement.location(of: panel), w !== arrangement.main, w.panels == [panel] {
            if let frame, let nsWindow = controllers[w.id]?.window {
                nsWindow.setFrame(frame, display: true, animate: false)
                nsWindow.makeKeyAndOrderFront(nil)
            }
            return
        }
        #endif
        guard arrangement.canMove(panel, to: target) else { return }
        arrangement.move(panel, to: target, frame: frame)
        arrangement.select(panel)
        focus(panel)
        syncWindows()
        #if os(macOS)
        if case let (w, _)? = arrangement.location(of: panel) {
            (w === arrangement.main ? mainWindow : controllers[w.id]?.window)?.makeKeyAndOrderFront(nil)
        }
        #endif
    }

    func moveToNewWindow(_ panel: PanelID) {
        #if os(macOS)
        move(panel, to: .newWindow, frame: drag.defaultFrame(for: panel))
        #endif
    }

    func moveToMainWindow(_ panel: PanelID) {
        guard isDetached(panel) else { return }
        arrangement.returnToMain(panel)
        focus(panel)
        syncWindows()
        #if os(macOS)
        mainWindow?.makeKeyAndOrderFront(nil)
        #endif
    }

    /// Moves a separate window's panels back into the main window.
    func dock(_ model: PanelWindowModel) {
        arrangement.dock(model)
        syncWindows()
        #if os(macOS)
        mainWindow?.makeKeyAndOrderFront(nil)
        #endif
    }

    /// Closes a panel: an analysis pane is removed, the lens data and the
    /// layout are hidden until shown again.
    func close(_ panel: PanelID) {
        if let id = panel.paneID, let pane = pane(id) { workspace.removePane(pane) }
        arrangement.remove(panel)
        if focusedPanel == panel { focusedPanel = nil }
        syncWindows()
    }

    /// Shows or hides the lens data or the layout.
    func toggle(_ panel: PanelID) {
        if arrangement.contains(panel) {
            close(panel)
        } else {
            arrangement.show(panel)
            reveal(panel)
        }
    }

    /// Shows a panel: brings its tab to the front and its window forward.
    func reveal(_ panel: PanelID) {
        arrangement.select(panel)
        #if os(macOS)
        if case let (w, _)? = arrangement.location(of: panel), w !== arrangement.main {
            controllers[w.id]?.window?.orderFront(nil)
        }
        #endif
    }

    /// Brings the arrangement in line with the workspace's panes: closed
    /// panes leave it; new ones go beside the pane they were added after.
    func reconcile() {
        let ids = Set(workspace.panes.map(\.id))
        for p in arrangement.allPanels {
            if let id = p.paneID, !ids.contains(id) { arrangement.remove(p) }
        }
        for pane in workspace.panes where !arrangement.contains(.analysis(pane.id)) {
            let hint = workspace.placementHints.removeValue(forKey: pane.id)
            if let w = pendingWindow.removeValue(forKey: pane.id), arrangement.window(w) != nil {
                arrangement.place(.analysis(pane.id), inWindow: w)
            } else {
                arrangement.place(.analysis(pane.id), near: hint.map(PanelID.analysis))
            }
        }
        syncWindows()
    }

    /// Adds an analysis pane to a given window (a separate window's Add menu).
    func addPane(_ analysis: String, inWindow id: UUID) {
        let pane = workspace.addPane(analysis)
        pendingWindow[pane.id] = id
        reconcile()
    }

    /// Every panel back in the main window, in the standard arrangement.
    func resetLayout() {
        arrangement.reset(to: .standard(panes: workspace.panes.map(\.id)))
        syncWindows()
        #if os(macOS)
        mainWindow?.makeKeyAndOrderFront(nil)
        #endif
    }

    // MARK: Menu actions

    private var editor: LensEditor { LensEditor(document: design, undoManager: undoManager) }
    private var system: OpticalSystem { design.system }
    private var selectedIndex: Int? { window.selection.flatMap { system.surfaceIndex(of: $0) } }

    /// What the menus do in any of the design's windows.
    func actions() -> WindowActions {
        let column = window.focusedColumn
        let path = selectedIndex.flatMap { i in column?.param(for: system.surfaces[i]) }
        let panel = focusedPanel ?? workspace.focusedPaneID.map(PanelID.analysis)
        return WindowActions(
            insertBefore: { [self] in insert(at: selectedIndex ?? system.imageIndex) },
            insertAfter: { [self] in insert(at: (selectedIndex ?? system.imageIndex - 1) + 1) },
            deleteSurface: { [self] in deleteSelected() },
            makeStop: { [self] in if let id = window.selection { editor.makeStop(id) } },
            toggleVariable: { [self] in
                guard let path else { return }
                if !design.isVariable(path), let why = design.variableRefusal(path) { window.flash(why) } else { editor.toggleVariable(path) }
            },
            canToggleVariable: path.map { $0.resolves(in: system) } ?? false,
            hasSurfaceSelection: selectedIndex.map { $0 > 0 && $0 < system.imageIndex } ?? false,
            loadCatalog: { [self] in
                showMainWindow()
                requestCatalog()
            },
            designJob: { [self] job in
                window.open(job)
                showMainWindow()
            },
            showIssues: { [self] in
                window.showIssues = true
                showMainWindow()
            },
            addPane: { [self] in workspace.addPane($0) },
            paneAnalyses: workspace.paneAnalyses,
            rerun: { [self] in workspace.rerunVisible() },
            cancel: { [self] in workspace.cancelAll() },
            fit: { [self] in window.fitRequest += 1 },
            setMode: { [self] in window.viewportMode = $0 },
            toggleViewport: { [self] in toggle(.layout) },
            toggleLensData: { [self] in toggle(.lensData) },
            showsLayout: arrangement.contains(.layout),
            showsLensData: arrangement.contains(.lensData),
            focusedPanel: panel.flatMap { arrangement.contains($0) ? $0 : nil },
            focusedPanelDetached: panel.map(isDetached) ?? false,
            focusedPanelAlone: panel.map(isAloneInWindow) ?? false,
            moveToNewWindow: { [self] in moveToNewWindow($0) },
            moveToMainWindow: { [self] in moveToMainWindow($0) },
            closePanel: { [self] in close($0) },
            resetLayout: { [self] in resetLayout() },
            setColoring: { [self] in window.rayColoring = $0 },
            backend: Binding(get: { [self] in workspace.backend }, set: { [self] in workspace.backend = $0 }),
            replaceWithSample: { [self] s in
                design.replace(system: s, actionName: "Replace with Sample", undoManager: undoManager)
                window.selection = nil
                window.fitRequest += 1
            })
    }

    private func insert(at index: Int) {
        if let id = editor.insert(at: index) { window.selection = id }
    }

    private func deleteSelected() {
        // A surface that cannot be deleted (the last one between object and image) stays selected.
        guard let id = window.selection, let i = selectedIndex, i > 0, i < system.imageIndex, let next = editor.delete(id) else { return }
        window.selection = next
    }

    private func showMainWindow() {
        #if os(macOS)
        mainWindow?.makeKeyAndOrderFront(nil)
        #endif
    }

    // MARK: Windows

    #if os(macOS)
    @ObservationIgnored private var controllers: [UUID: DetachedPanelWindowController] = [:]
    @ObservationIgnored private(set) weak var mainWindow: NSWindow?
    @ObservationIgnored private var closeObserver: (any NSObjectProtocol)?
    @ObservationIgnored private var tearingDown = false

    /// The main window is on screen: separate windows saved with the
    /// project open beside it, and close with it.
    func attach(mainWindow w: NSWindow) {
        guard mainWindow !== w else { return }
        mainWindow = w
        if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
        closeObserver = NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: w, queue: .main) {
            [weak self] _ in
            MainActor.assumeIsolated { self?.tearDown() }
        }
        syncWindows()
    }

    /// Opens a window for each separate panel window and closes the ones
    /// whose panels have left.
    func syncWindows() {
        let live = Set(arrangement.detached.map(\.id))
        for (id, c) in controllers where !live.contains(id) {
            controllers[id] = nil
            c.closeQuietly()
        }
        guard mainWindow != nil, !tearingDown else { return }
        for w in arrangement.detached where controllers[w.id] == nil {
            let c = DetachedPanelWindowController(session: self, model: w)
            controllers[w.id] = c
            c.showWindow(nil)
        }
    }

    /// The main window closes: the separate windows close with it. They
    /// stay in the saved arrangement and open again with the project.
    func tearDown() {
        guard !tearingDown else { return }
        tearingDown = true
        drag.cancel()
        for c in controllers.values { c.closeQuietly() }
        controllers = [:]
        if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
        closeObserver = nil
    }

    /// The user closed a separate window: its panels close with it.
    func windowClosed(_ model: PanelWindowModel) {
        guard !tearingDown, controllers.removeValue(forKey: model.id) != nil else { return }
        let panels = arrangement.discard(model)
        for p in panels {
            if let id = p.paneID, let pane = pane(id) { workspace.removePane(pane) }
        }
        if let f = focusedPanel, panels.contains(f) { focusedPanel = nil }
    }
    #else
    func syncWindows() {}
    #endif
}

extension EnvironmentValues {
    /// The design session of the window (main or separate).
    @Entry var designSession: DesignSession?
    /// The panel shares its place with others, as one of several tabs: its
    /// header leaves the title to the tab.
    @Entry var panelTabbed = false
}
