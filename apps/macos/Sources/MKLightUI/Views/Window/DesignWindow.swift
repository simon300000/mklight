import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

/// One design window.
///
/// ```
/// ┌ sidebar ──┬ lens data table ─────────────────────────────┬ inspector ┐
/// │ analyses  ├ layout viewport │ analysis panes …           │ surface / │
/// │ (search)  │                 │                            │ system /  │
/// │           │                 │                            │ job panel │
/// └───────────┴──────────────────────────────────────────────┴───────────┘
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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var columns = NavigationSplitViewVisibility.all
    @State private var importingCatalog = false

    private var system: OpticalSystem { design.system }
    private var editor: LensEditor { LensEditor(document: design, undoManager: undoManager) }
    private var selectedIndex: Int? { window.selection.flatMap { system.surfaceIndex(of: $0) } }
    private var projectDirectory: URL? { fileURL?.deletingLastPathComponent() }

    var body: some View {
        NavigationSplitView(columnVisibility: $columns) {
            AnalysisSidebar(workspace: workspace, window: window)
                .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 320)
        } detail: {
            SplitLayout(axis: .vertical, fraction: $window.editorFraction, minFirst: 120, minSecond: 240) {
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
            .overlay(alignment: .bottom) { toast }
            .inspector(isPresented: $window.showInspector) {
                InspectorView(design: design, workspace: workspace, window: window)
                    .inspectorColumnWidth(min: 300, ideal: 360, max: 560)
            }
        }
        .navigationSubtitle(system.name)
        .toolbar { toolbar }
        .focusedSceneValue(\.windowActions, actions)
        .fileImporter(isPresented: $importingCatalog, allowedContentTypes: [.json, .plainText, .data]) { result in
            if case .success(let url) = result { loadCatalog(url) }
        }
        .task {
            await workspace.loadEngineMetadata()
            await workspace.open(system: system, catalogs: design.catalogs, projectDirectory: projectDirectory)
            applyLaunchOptions()
        }
        .onDisappear {
            window.optimize.cancel()
            window.tolerance.cancel()
            window.glass.cancel()
            workspace.cancelAll()
        }
        .onChange(of: design.system) { _, s in workspace.update(system: s) }
        .onChange(of: design.catalogs) { _, refs in
            Task { await workspace.syncCatalogs(refs, projectDirectory: projectDirectory) }
        }
        .onChange(of: WorkspaceLayout(workspace: workspace, window: window)) { _, layout in
            design.setViews(layout.views)
        }
        .onChange(of: window.selection) { _, s in
            if s != nil && window.activeJob == nil { window.inspectorTab = .surface }
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
            ForEach(DesignJobKind.allCases) { job in
                Button(job.title, systemImage: job.symbol) { toggle(job) }
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
            Toggle(isOn: $window.showViewport) { Label("Layout", systemImage: "rectangle.lefthalf.inset.filled") }
                .help("Show or hide the layout viewport (⌥⌘L)")
        }
        ToolbarSpacer(.fixed, placement: .primaryAction)
        ToolbarItem(placement: .primaryAction) {
            Button("Inspector", systemImage: "sidebar.right") { window.showInspector.toggle() }
                .help("Show or hide the inspector (⌃⌘I)")
        }
    }

    private func toggle(_ job: DesignJobKind) {
        if window.activeJob == job && window.showInspector { window.activeJob = nil } else { window.open(job) }
    }

    private func help(_ job: DesignJobKind) -> String {
        switch job {
        case .optimize: "Optimize the design (⌃⌘O)"
        case .glass: "Replace glasses with catalogue glasses, re-optimising each combination (⌃⌘G)"
        case .tolerance: "Tolerance analysis: sensitivities and Monte Carlo (⌃⌘T)"
        case .solves: "Preview and apply the design's solves (⌃⌘L)"
        }
    }

    // MARK: Actions

    private var actions: WindowActions {
        let column = window.focusedColumn
        let path = selectedIndex.flatMap { i in column?.param(for: system.surfaces[i]) }
        return WindowActions(
            insertBefore: { insert(at: selectedIndex ?? system.imageIndex) },
            insertAfter: { insert(at: (selectedIndex ?? system.imageIndex - 1) + 1) },
            deleteSurface: { deleteSelected() },
            makeStop: { if let id = window.selection { editor.makeStop(id) } },
            toggleVariable: { if let path { editor.toggleVariable(path) } },
            canToggleVariable: path.map { $0.resolves(in: system) } ?? false,
            hasSurfaceSelection: selectedIndex.map { $0 > 0 && $0 < system.imageIndex } ?? false,
            loadCatalog: { importingCatalog = true },
            designJob: { window.open($0) },
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

    /// Debug builds: `-MKLightSelectSurface 3`, `-MKLightInspector system`,
    /// `-MKLightSheet optimize`, `-MKLightSidebar hidden`, `-MKLightViewport 3D`,
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
