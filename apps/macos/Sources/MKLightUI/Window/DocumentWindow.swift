import MKLightEngine
import MKLightModel
import SwiftUI

/// One document window: lens data editor on top, layout viewport and
/// analyses below, system/surface inspector on the right, status bar at the
/// bottom.
///
/// The document (`ProjectDocument`) holds only the model; this view owns the
/// window's `WindowViewState` and `AnalysisCoordinator`.
struct DocumentWindow: View {
    @ObservedObject var document: ProjectDocument
    @State private var viewState = WindowViewState()
    @State private var analyses = AnalysisCoordinator(engine: MockEngine())
    @Environment(\.undoManager) private var undoManager
    @Environment(\.mklightEngine) private var engine

    private var system: OpticalSystem { document.system }

    var body: some View {
        VSplitView {
            LensEditorPanel(
                system: system, layout: layoutScene, selection: $viewState.selection,
                onEdit: { apply($0) }, onMessage: { viewState.editorMessage = $0 })
                .frame(minHeight: 120, idealHeight: 230)
            HSplitView {
                if viewState.showLayout {
                    LayoutPanel(viewState: viewState, analyses: analyses, system: system, isMockEngine: engine.info.isMock)
                        .frame(minWidth: 280, idealWidth: 520)
                }
                if viewState.showAnalyses {
                    AnalysisArea(viewState: viewState, analyses: analyses, system: system)
                        .frame(minWidth: 320, idealWidth: 480)
                }
            }
            .frame(minHeight: 280)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            StatusBar(analyses: analyses, viewState: viewState, engineInfo: engine.info)
        }
        .inspector(isPresented: $viewState.showInspector) {
            InspectorView(system: system, viewState: viewState, edit: { name, change in
                document.edit(name, undoManager: undoManager, change)
            })
            .inspectorColumnWidth(min: 260, ideal: 310, max: 460)
        }
        .toolbar { toolbar }
        .navigationSubtitle(system.name)
        .focusedSceneValue(\.documentActions, actions)
        .task {
            analyses.attach(engine: engine)
            analyses.update(settings: viewState.analysisSettings)
            analyses.setVisible(viewState.visibleAnalyses)
            analyses.update(system: system, immediately: true)
        }
        .onChange(of: document.system) { _, s in analyses.update(system: s) }
        .onChange(of: viewState.analysisSettings) { _, s in analyses.update(settings: s) }
        .onChange(of: viewState.visibleAnalyses) { _, v in analyses.setVisible(v) }
        .onChange(of: viewState.selection) { _, s in if s != nil { viewState.inspectorTab = .surface } }
    }

    private var layoutScene: LayoutScene? {
        if case .layout(let s) = analyses.output(.layout)?.result { return s }
        return nil
    }

    @discardableResult
    private func apply(_ e: LensEdit) -> Bool {
        let changed = document.edit(e.actionName, undoManager: undoManager) { e.apply(to: &$0) }
        if changed, case .insert(let i) = e, document.system.surfaces.indices.contains(i) {
            viewState.selection = document.system.surfaces[i].id
        }
        return changed
    }

    private var selectedIndex: Int? { viewState.selection.flatMap { system.surfaceIndex(of: $0) } }

    var actions: DocumentActions {
        DocumentActions(
            insertBefore: { apply(.insert(at: max(selectedIndex ?? system.imageIndex, 1))) },
            insertAfter: { apply(.insert(at: (selectedIndex ?? system.imageIndex - 1) + 1)) },
            deleteSurface: { if let i = selectedIndex { apply(.delete(at: i)) } },
            makeStop: { if let i = selectedIndex { apply(.makeStop(at: i)) } },
            rerunAnalyses: { analyses.rerunVisible() },
            cancelAnalyses: { analyses.cancelAll() },
            fitView: { viewState.fitRequest += 1 },
            toggleViewportMode: { viewState.viewportMode = viewState.viewportMode == .section2D ? .orbit3D : .section2D },
            loadSample: { sample in
                document.edit("Load Sample", undoManager: undoManager) { $0 = sample }
                viewState.selection = nil
                viewState.fitRequest += 1
            },
            hasSelection: selectedIndex.map { $0 > 0 && $0 < system.imageIndex } ?? false)
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .navigation) {
            Button("Insert Surface", systemImage: "plus.rectangle") { actions.insertBefore() }
                .help("Insert a surface before the selection (Insert)")
            Button("Delete Surface", systemImage: "minus.rectangle") { actions.deleteSurface() }
                .help("Delete the selected surface (Delete)")
                .disabled(!actions.hasSelection)
        }
        ToolbarItemGroup {
            Picker("View", selection: $viewState.viewportMode) {
                ForEach(ViewportMode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .help("Y-Z section or 3-D orbit view")
            Button("Fit", systemImage: "arrow.up.left.and.arrow.down.right") { viewState.fitRequest += 1 }
                .help("Fit the layout in the viewport")
            Picker("Ray Colors", selection: $viewState.rayColoring) {
                ForEach(RayColoring.allCases) { Text($0.title).tag($0) }
            }
            .help("Color rays by field or by wavelength")
            Toggle(isOn: $viewState.showLayout) { Label("Layout", systemImage: "rectangle.split.2x1") }
                .help("Show the layout viewport")
            Toggle(isOn: $viewState.showAnalyses) { Label("Analyses", systemImage: "chart.xyaxis.line") }
                .help("Show the analysis area")
            Button("Inspector", systemImage: "sidebar.right") { viewState.showInspector.toggle() }
                .help("Show or hide the inspector")
        }
    }
}

/// Header with structure buttons above the lens table.
struct LensEditorPanel: View {
    let system: OpticalSystem
    let layout: LayoutScene?
    @Binding var selection: SurfaceID?
    let onEdit: (LensEdit) -> Bool
    let onMessage: (String?) -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Lens Data").font(.headline)
                Text("\(system.surfaces.count) surfaces · stop \(system.stop)").foregroundStyle(.secondary).font(.caption)
                Spacer()
                Text("Radius/Thickness/Semi-Dia in mm · A = automatic semi-diameter").font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            LensDataEditor(system: system, layout: layout, selection: $selection, onEdit: onEdit, onMessage: onMessage)
        }
    }
}

/// Bottom status bar: engine, snapshot, validation, key first-order data,
/// and editor messages.
struct StatusBar: View {
    let analyses: AnalysisCoordinator
    let viewState: WindowViewState
    let engineInfo: EngineInfo
    @State private var showIssues = false

    var body: some View {
        HStack(spacing: 10) {
            if engineInfo.isMock {
                Label("Mock engine", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .help("Results are synthetic placeholders until the Rust engine is linked.")
            } else {
                Text("\(engineInfo.name) \(engineInfo.version)")
            }
            if let snap = analyses.snapshot {
                Text("snapshot \(snap.hash.short)").monospaced().foregroundStyle(.secondary)
            }
            if !analyses.validationErrors.isEmpty {
                Button {
                    showIssues.toggle()
                } label: {
                    Label("\(analyses.validationErrors.count) model error(s)", systemImage: "xmark.octagon.fill").foregroundStyle(.red)
                }
                .buttonStyle(.borderless)
                .popover(isPresented: $showIssues) {
                    VStack(alignment: .leading) {
                        ForEach(analyses.validationErrors, id: \.self) { Text($0) }
                    }
                    .padding()
                }
            }
            if let msg = viewState.editorMessage {
                Text(msg).foregroundStyle(.red).lineLimit(1)
            }
            Spacer()
            if case .firstOrder(let fo) = analyses.output(.firstOrder)?.result {
                Group {
                    if let efl = fo["efl"] { Text(String(format: "EFL %.3f mm", efl)) }
                    if let f = fo["image_f_number"] ?? fo["working_f_number"] { Text(String(format: "F/%.2f", f)) }
                    if let tt = fo["total_track"] { Text(String(format: "TT %.2f mm", tt)) }
                }
                .monospacedDigit()
                .foregroundStyle(analyses.isStale(.firstOrder) ? .tertiary : .secondary)
            }
            if analyses.isBusy {
                ProgressView().controlSize(.mini)
            }
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(.bar)
    }
}

/// Window commands, published to the menu bar through `focusedSceneValue`.
struct DocumentActions {
    var insertBefore: () -> Void
    var insertAfter: () -> Void
    var deleteSurface: () -> Void
    var makeStop: () -> Void
    var rerunAnalyses: () -> Void
    var cancelAnalyses: () -> Void
    var fitView: () -> Void
    var toggleViewportMode: () -> Void
    var loadSample: (OpticalSystem) -> Void
    var hasSelection: Bool
}

extension FocusedValues {
    @Entry var documentActions: DocumentActions?
}

extension EnvironmentValues {
    /// The engine every window talks to.
    @Entry public var mklightEngine: any EngineClient = MockEngine()
}
