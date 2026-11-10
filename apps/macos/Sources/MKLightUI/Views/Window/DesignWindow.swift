import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

/// A design's main window.
///
/// ```
/// ┌ sidebar ──┬ panels ───────────────────────────────────┬ inspector ┐
/// │ analyses  │ lens data                                 │ surface / │
/// │ (search)  ├ layout ──────┬ analysis │ analysis …      │ system /  │
/// │           │              │          │                 │ job panel │
/// └───────────┴──────────────┴──────────┴─────────────────┴───────────┘
/// ```
///
/// The panels (lens data, layout, analyses) are tab groups in a tree of
/// splits (`PanelArrangement`) that the user rearranges by dragging; a panel
/// dragged out of the window opens in a window of its own. The document
/// (`DesignDocument`) holds the design; the window owns the design's
/// `DesignSession` (window state, workspace, arrangement), which the
/// separate windows share. The layout is saved in the project's `views`.
struct DesignWindow: View {
    let document: ProjectDocument
    var fileURL: URL?
    let engine: any EngineClient
    @State private var session: DesignSession?

    var body: some View {
        Group {
            if let session {
                DesignWindowContent(
                    design: document.design, session: session, workspace: session.workspace, window: session.window, fileURL: fileURL,
                    engine: engine)
            } else {
                Color.clear
            }
        }
        .onAppear {
            guard session == nil else { return }
            let views = document.design.views
            let layout = WorkspaceLayout.isSaved(views) ? WorkspaceLayout(views: views) : .standard
            let window = WindowState()
            layout.apply(to: window)
            let ws = Workspace(engine: engine, panes: layout.makePanes())
            ws.layout.setSettings(layout.layoutSettings)
            session = DesignSession(design: document.design, workspace: ws, window: window, arrangement: layout.makeArrangement())
        }
    }
}

private struct DesignWindowContent: View {
    let design: DesignDocument
    let session: DesignSession
    let workspace: Workspace
    @Bindable var window: WindowState
    let fileURL: URL?
    let engine: any EngineClient
    @Environment(\.undoManager) private var undoManager
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var columns = NavigationSplitViewVisibility.all
    @State private var importingCatalog = false
    @Environment(\.openDocument) private var openDocument
    /// The import report on screen (a design opened from a lens file).
    @State private var importReport: ImportReport?
    /// The report sheet asked for a catalog: pick one when it has closed.
    @State private var catalogAfterReport = false
    /// Import the lens file again once the picked catalog has loaded.
    @State private var reimportAfterCatalog = false

    private var system: OpticalSystem { design.system }
    private var projectDirectory: URL? { fileURL?.deletingLastPathComponent() }

    var body: some View {
        NavigationSplitView(columnVisibility: $columns) {
            AnalysisSidebar(workspace: workspace, window: window)
                .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 320)
        } detail: {
            PanelWindowContent(session: session, model: session.arrangement.main, onWindow: { session.attach(mainWindow: $0) })
            .overlay(alignment: .bottom) { toast }
            .inspector(isPresented: $window.showInspector) {
                // Built only while shown: a hidden inspector stays in the
                // view tree and would rebuild its form on every selection.
                Group {
                    if window.showInspector { InspectorView(design: design, workspace: workspace, window: window) }
                }
                .inspectorColumnWidth(min: 300, ideal: 360, max: 560)
            }
        }
        .environment(\.designSession, session)
        .navigationSubtitle(system.name)
        .toolbar { toolbar }
        // The menu actions and the saved layout read the selection and the
        // camera; a child view keeps those reads out of this body, so a
        // click or a pan does not rebuild the window.
        .background {
            WindowObservers(
                actions: { session.actions() }, undo: { session.undoState },
                layout: { WorkspaceLayout(workspace: workspace, window: window, arrangement: session.arrangement) },
                save: { design.setViews($0.views) })
        }
        .fileImporter(isPresented: $importingCatalog, allowedContentTypes: [.json, .plainText, .data]) { result in
            if case .success(let url) = result {
                loadCatalog(url)
            } else {
                reimportAfterCatalog = false
            }
        }
        .sheet(item: $importReport, onDismiss: {
            guard catalogAfterReport else { return }
            catalogAfterReport = false
            reimportAfterCatalog = true
            importingCatalog = true
        }) { report in
            ImportReportSheet(report: report, loadCatalog: prescriptionURL == nil ? nil : { catalogAfterReport = true })
        }
        .dropDestination(for: URL.self) { urls, _ in openDropped(urls) }
        .onAppear {
            session.undoManager = undoManager
            session.requestCatalog = { importingCatalog = true }
        }
        // Panes added or closed anywhere find their place in the arrangement;
        // a pane chosen in the sidebar comes to the front.
        .onChange(of: workspace.panes.map(\.id), initial: true) { _, _ in session.reconcile() }
        .onChange(of: workspace.focusedPaneID) { _, id in if let id { session.reveal(.analysis(id)) } }
        .task {
            await workspace.loadEngineMetadata()
            await workspace.open(system: system, catalogs: design.catalogs, projectDirectory: projectDirectory)
            if let report = design.takeImportReport() { present(report) }
            applyLaunchOptions()
        }
        .onDisappear {
            #if os(macOS)
            session.tearDown()
            #endif
            window.optimize.cancel()
            window.tolerance.cancel()
            window.glass.cancel()
            workspace.cancelAll()
        }
        .onChange(of: design.system) { _, s in workspace.update(system: s) }
        .onChange(of: design.catalogs) { _, refs in
            Task { await workspace.syncCatalogs(refs, projectDirectory: projectDirectory) }
        }
    }

    /// A transient message (a rejected edit) floating over the content.
    @ViewBuilder private var toast: some View {
        if let m = window.message {
            Label(m, systemImage: "info.circle")
                .font(.callout)
                .lineLimit(2)
                .padding(.horizontal, Space.l)
                .padding(.vertical, Space.m)
                .glassEffect(.regular, in: .capsule)
                .padding(.bottom, Space.l)
                .transition(reduceMotion ? .opacity : .move(edge: .bottom).combined(with: .opacity))
                .accessibilityAddTraits(.isStaticText)
                .onTapGesture { window.message = nil }
        }
    }

    // MARK: Toolbar

    /// Three groups, by function: design jobs; the window's content
    /// (analyses, layout); the inspector at the trailing edge. The design's
    /// status is in the centre.
    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            DesignStatus(workspace: workspace, window: window, engine: engine, system: system)
        }
        ToolbarItemGroup(placement: .primaryAction) {
            // Toggles, like the Inspector button: highlighted while the
            // job's panel is open; a second click closes it.
            ForEach(DesignJobKind.allCases) { job in
                Toggle(isOn: Binding(get: { window.isShowing(job) }, set: { _ in window.toggle(job) })) {
                    Label(job.title, systemImage: job.symbol)
                }
                .help(help(job))
            }
        }
        ToolbarSpacer(.fixed, placement: .primaryAction)
        ToolbarItemGroup(placement: .primaryAction) {
            Menu {
                AddPaneMenuItems(workspace: workspace)
            } label: {
                Label("Add Analysis", systemImage: "plus.rectangle.on.rectangle")
            }
            .help("Add an analysis pane")
            Toggle(isOn: shows(.lensData)) { Label("Lens Data", systemImage: "tablecells") }
                .help("Show or hide the lens data (⌥⌘E)")
            Toggle(isOn: shows(.layout)) { Label("Layout", systemImage: "rectangle.lefthalf.inset.filled") }
                .help("Show or hide the layout viewport (⌥⌘L)")
        }
        ToolbarSpacer(.fixed, placement: .primaryAction)
        ToolbarItem(placement: .primaryAction) {
            Button("Inspector", systemImage: "sidebar.right") { window.showInspector.toggle() }
                .help("Show or hide the inspector (⌃⌘I)")
        }
    }

    /// The toolbar's panel toggles: on while the panel is open in any window.
    private func shows(_ panel: PanelID) -> Binding<Bool> {
        Binding(get: { session.arrangement.contains(panel) }, set: { _ in session.toggle(panel) })
    }

    private func help(_ job: DesignJobKind) -> String {
        switch job {
        case .optimize: "Optimize the design (⌃⌘O)"
        case .glass: "Replace glasses with catalogue glasses, re-optimising each combination (⌃⌘G)"
        case .tolerance: "Tolerance analysis: sensitivities and Monte Carlo (⌃⌘T)"
        case .solves: "Preview and apply the design's solves (⌃⌘L)"
        }
    }

    /// Debug builds: `-MKLightSelectSurface 3`, `-MKLightInspector system`,
    /// `-MKLightSheet optimize`, `-MKLightSidebar hidden`, `-MKLightViewport 3D`,
    /// `-MKLightFootprints YES`, `-MKLightFit lens`, `-MKLightYaw 0.6`,
    /// `-MKLightShowIssues YES`, `-MKLightToast text` (screenshots, UI checks).
    private func applyLaunchOptions() {
        #if DEBUG
        let d = UserDefaults.standard
        if let i = d.object(forKey: "MKLightSelectSurface") as? Int ?? Int(d.string(forKey: "MKLightSelectSurface") ?? ""),
            system.surfaces.indices.contains(i)
        {
            window.selection = system.surfaces[i].id
        }
        if let t = d.string(forKey: "MKLightInspector") { window.inspectorTab = t == "system" ? .system : .surface }
        if d.string(forKey: "MKLightSidebar") == "hidden" { columns = .detailOnly }
        if d.string(forKey: "MKLightViewport") == "3D" { window.viewportMode = .orbit3D }
        if d.bool(forKey: "MKLightFootprints") { workspace.showFootprints = true }
        if d.string(forKey: "MKLightFit") == "lens" { window.fitLens = true }
        if let yaw = d.object(forKey: "MKLightYaw") as? Double ?? Double(d.string(forKey: "MKLightYaw") ?? "") { window.camera.yaw = yaw }
        if d.bool(forKey: "MKLightShowIssues") {
            Task {
                try? await Task.sleep(for: .seconds(3))
                window.showIssues = true
            }
        }
        if let text = d.string(forKey: "MKLightToast") {
            Task {
                try? await Task.sleep(for: .seconds(2))
                window.flash(text)
            }
        }
        if let job = d.string(forKey: "MKLightSheet").flatMap(DesignJobKind.init(rawValue:)) {
            Task {
                try? await Task.sleep(for: .seconds(2))
                window.open(job)
            }
        }
        #endif
    }

    private func loadCatalog(_ url: URL) {
        let reimport = reimportAfterCatalog
        reimportAfterCatalog = false
        Task {
            do {
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let info = try await workspace.loadCatalog(at: url)
                design.addCatalog(
                    CatalogReference(name: info.name, path: url.path, format: url.pathExtension.lowercased() == "json" ? "json" : "agf"),
                    undoManager: undoManager)
                window.flash("Loaded glass catalog \(info.name) (\(info.glassCount) glasses).")
                if reimport, let source = prescriptionURL { await importAgain(source) }
            } catch {
                window.flash("Glass catalog: \(error.localizedDescription)")
            }
        }
    }

    // MARK: Lens prescription files

    /// The lens file this design was opened from, while it has not been
    /// saved as a project.
    private var prescriptionURL: URL? {
        fileURL.flatMap { PrescriptionFormat(fileExtension: $0.pathExtension) != nil ? $0 : nil }
    }

    /// Shows what an import left out: the report sheet, or a short note when
    /// everything mapped.
    private func present(_ report: ImportReport) {
        if report.warnings.isEmpty {
            window.flash("Imported \(report.fileName): \(report.summary).")
        } else {
            importReport = report
        }
    }

    /// Imports the lens file again with the catalogs now loaded (undoable).
    private func importAgain(_ url: URL) async {
        do {
            let imported = try await workspace.importPrescription(at: url)
            var system = imported.system
            if system.name.isEmpty { system.name = url.deletingPathExtension().lastPathComponent }
            design.edit("Import Lens File Again", undoManager: undoManager) { $0 = system }
            present(ImportReport(imported, fileName: url.lastPathComponent))
        } catch {
            window.flash("Import: \(error.localizedDescription)")
        }
    }

    /// Opens dropped projects and lens files, each in its own window.
    private func openDropped(_ urls: [URL]) -> Bool {
        let files = urls.filter(ProjectDocument.canOpen)
        guard !files.isEmpty else { return false }
        Task {
            for url in files {
                do {
                    try await openDocument(at: url)
                } catch {
                    window.flash("\(url.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }
        return true
    }
}

/// Publishes the window's menu actions and saves its layout. Its body
/// alone depends on the selection, the focused column and the camera.
private struct WindowObservers: View {
    let actions: () -> WindowActions
    let undo: () -> DocumentUndo?
    let layout: () -> WorkspaceLayout
    let save: (WorkspaceLayout) -> Void

    var body: some View {
        Color.clear
            .focusedSceneValue(\.windowActions, actions())
            .focusedSceneValue(\.documentUndo, undo())
            .onChange(of: layout()) { _, l in save(l) }
            .accessibilityHidden(true)
    }
}

/// Menu items that add an analysis pane, grouped by category.
struct AddPaneMenuItems: View {
    let workspace: Workspace

    var body: some View {
        ForEach(AnalysisCatalog.grouped(workspace.paneAnalyses), id: \.category) { group in
            Section(AnalysisCatalog.categoryTitle(group.category)) {
                ForEach(group.items) { d in
                    Button {
                        workspace.addPane(d.name)
                    } label: {
                        Label(AnalysisCatalog.title(of: d), systemImage: AnalysisCatalog.symbol(for: d.name))
                    }
                    .disabled(!d.available)
                }
            }
        }
    }
}
