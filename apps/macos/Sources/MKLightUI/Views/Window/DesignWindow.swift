import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

/// One design window.
///
/// ```
/// ┌ sidebar ──┬ lens data table ─────────────────────────────┬ inspector ┐
/// │ analyses  ├ layout viewport │ analysis panes …           │ surface / │
/// │ (search)  │                 │                            │ system    │
/// └───────────┴ status bar ──────────────────────────────────┴───────────┘
/// ```
///
/// The document (`DesignDocument`) holds the design; this view owns the
/// window's `WindowState` (selection, layout) and `Workspace` (snapshots,
/// panes, jobs). The pane layout is saved in the project's `views`.
struct DesignWindow: View {
    let document: ProjectDocument
    var fileURL: URL?
    let engine: any EngineClient
    @State private var window = WindowState()
    @State private var workspace: Workspace?

    var body: some View {
        Group {
            if let workspace {
                DesignWindowContent(
                    design: document.design, workspace: workspace, window: window, fileURL: fileURL, engine: engine)
            } else {
                Color.clear
            }
        }
        .onAppear {
            guard workspace == nil else { return }
            let views = document.design.views
            let layout = WorkspaceLayout.isSaved(views) ? WorkspaceLayout(views: views) : .standard
            layout.apply(to: window)
            let ws = Workspace(engine: engine, panes: layout.makePanes())
            ws.layout.setSettings(layout.layoutSettings)
            workspace = ws
        }
    }
}

private struct DesignWindowContent: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    let fileURL: URL?
    let engine: any EngineClient
    @Environment(\.undoManager) private var undoManager
    @State private var columns = NavigationSplitViewVisibility.all
    @State private var importingCatalog = false

    private var system: OpticalSystem { design.system }
    private var editor: LensEditor { LensEditor(document: design, undoManager: undoManager) }
    private var selectedIndex: Int? { window.selection.flatMap { system.surfaceIndex(of: $0) } }
    private var projectDirectory: URL? { fileURL?.deletingLastPathComponent() }

    var body: some View {
        NavigationSplitView(columnVisibility: $columns) {
            AnalysisSidebar(workspace: workspace, window: window)
                .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 300)
        } detail: {
            VStack(spacing: 0) {
                SplitLayout(axis: .vertical, fraction: $window.editorFraction, minFirst: 110, minSecond: 220) {
                    LensTableView(design: design, workspace: workspace, window: window)
                } second: {
                    SplitLayout(
                        axis: .horizontal, fraction: $window.viewportFraction, showsFirst: window.showViewport, minFirst: 220,
                        minSecond: 280
                    ) {
                        LayoutPaneView(workspace: workspace, window: window, system: system)
                    } second: {
                        AnalysisArea(workspace: workspace, window: window, system: system)
                    }
                }
                StatusBar(workspace: workspace, window: window, engine: engine)
            }
            .inspector(isPresented: $window.showInspector) {
                InspectorView(design: design, workspace: workspace, window: window)
                    .inspectorColumnWidth(min: 270, ideal: 300, max: 440)
            }
        }
        .navigationSubtitle(system.name)
        .toolbar { toolbar }
        .focusedSceneValue(\.windowActions, actions)
        .sheet(item: $window.activeJob) { kind in
            switch kind {
            case .optimize: OptimizeSheet(design: design, workspace: workspace)
            case .tolerance: ToleranceSheet(workspace: workspace)
            case .solves: SolvesSheet(design: design, workspace: workspace)
            }
        }
        .fileImporter(isPresented: $importingCatalog, allowedContentTypes: [.json, .plainText, .data]) { result in
            if case .success(let url) = result { loadCatalog(url) }
        }
        .task {
            await workspace.loadEngineMetadata()
            await workspace.open(system: system, catalogs: design.catalogs, projectDirectory: projectDirectory)
        }
        .onDisappear { workspace.cancelAll() }
        .onChange(of: design.system) { _, s in workspace.update(system: s) }
        .onChange(of: design.catalogs) { _, refs in
            Task { await workspace.syncCatalogs(refs, projectDirectory: projectDirectory) }
        }
        .onChange(of: WorkspaceLayout(workspace: workspace, window: window)) { _, layout in
            design.setViews(layout.views)
        }
        .onChange(of: window.selection) { _, s in if s != nil { window.inspectorTab = .surface } }
    }

    // MARK: Toolbar

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            ControlGroup {
                Button("Optimize", systemImage: "wand.and.rays") { window.activeJob = .optimize }
                    .help("Optimize the design (⌃⌘O)")
                Button("Tolerance", systemImage: "plusminus") { window.activeJob = .tolerance }
                    .help("Tolerance analysis: sensitivities and Monte Carlo (⌃⌘T)")
                Button("Solves", systemImage: "function") { window.activeJob = .solves }
                    .help("Preview and apply the design's solves (⌃⌘S)")
            } label: {
                Label("Design", systemImage: "wand.and.rays")
            }
            IssuesButton(workspace: workspace, window: window)
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Menu {
                AddPaneMenuItems(workspace: workspace)
            } label: {
                Label("Add Analysis", systemImage: "plus.rectangle.on.rectangle")
            }
            .help("Add an analysis pane")
            Toggle(isOn: $window.showViewport) { Label("Layout", systemImage: "rectangle.lefthalf.inset.filled") }
                .help("Show or hide the layout viewport (⌥⌘L)")
            Button("Inspector", systemImage: "sidebar.right") { window.showInspector.toggle() }
                .help("Show or hide the inspector (⌃⌘I)")
        }
    }

    // MARK: Actions

    private var actions: WindowActions {
        let column = window.focusedColumn
        let path = window.selection.flatMap { id in column?.param(id) }
        return WindowActions(
            insertBefore: { insert(at: selectedIndex ?? system.imageIndex) },
            insertAfter: { insert(at: (selectedIndex ?? system.imageIndex - 1) + 1) },
            deleteSurface: { deleteSelected() },
            makeStop: { if let id = window.selection { editor.makeStop(id) } },
            toggleVariable: { if let path { editor.toggleVariable(path) } },
            canToggleVariable: path.map { $0.resolves(in: system) } ?? false,
            hasSurfaceSelection: selectedIndex.map { $0 > 0 && $0 < system.imageIndex } ?? false,
            loadCatalog: { importingCatalog = true },
            designJob: { window.activeJob = $0 },
            showIssues: { window.showIssues = true },
            issueCount: workspace.issues.count,
            addPane: { workspace.addPane($0) },
            paneAnalyses: workspace.paneAnalyses,
            rerun: { workspace.rerunVisible() },
            cancel: { workspace.cancelAll() },
            fit: { window.fitRequest += 1 },
            setMode: { window.viewportMode = $0 },
            toggleViewport: { window.showViewport.toggle() },
            setColoring: { window.rayColoring = $0 },
            backend: Binding(get: { workspace.backend }, set: { workspace.backend = $0 }),
            replaceWithSample: { s in
                design.replace(system: s, actionName: "Replace with Sample", undoManager: undoManager)
                window.selection = nil
                window.fitRequest += 1
            })
    }

    private func insert(at index: Int) {
        if let id = editor.insert(at: index) { window.selection = id }
    }

    private func deleteSelected() {
        guard let id = window.selection, let i = selectedIndex, i > 0, i < system.imageIndex else { return }
        window.selection = editor.delete(id)
    }

    private func loadCatalog(_ url: URL) {
        Task {
            do {
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let info = try await workspace.loadCatalog(at: url)
                design.addCatalog(
                    CatalogReference(name: info.name, path: url.path, format: url.pathExtension.lowercased() == "json" ? "json" : "agf"),
                    undoManager: undoManager)
                window.flash("Loaded glass catalog \(info.name) (\(info.glassCount) glasses).")
            } catch {
                window.flash("Glass catalog: \(error.localizedDescription)")
            }
        }
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
