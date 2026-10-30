import MKLightEngine
import MKLightModel
import SwiftUI

/// What a result view needs besides the output.
struct AnalysisContext {
    var system: OpticalSystem?
    var selectSurface: (Int) -> Void = { _ in }

    var fieldUnit: String { FieldLabel.unit(system) }
    var wavelengthsUm: [Double] { system?.wavelengths.wavelengths.map(\.um) ?? [] }
    var primaryWavelength: Int { system?.wavelengths.primary ?? 0 }

    func fieldLabel(_ value: [Double]?) -> String { FieldLabel.text(value, unit: fieldUnit) }
}

/// The analysis panes, side by side with resizable widths.
struct AnalysisArea: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let system: OpticalSystem

    var body: some View {
        if workspace.panes.isEmpty {
            ContentUnavailableView {
                Label("No Analyses", systemImage: "chart.xyaxis.line")
            } description: {
                Text("Choose an analysis in the sidebar, or add a pane.")
            } actions: {
                Menu("Add Analysis") { AddPaneMenuItems(workspace: workspace) }.fixedSize()
            }
        } else {
            PaneRow(items: workspace.panes, weights: $window.paneWidths) { pane in
                AnalysisPaneView(pane: pane, workspace: workspace, window: window, system: system)
            }
        }
    }
}

/// One analysis pane: header (analysis, state, actions), settings bar,
/// result, footer (key numbers and the backend actually used).
struct AnalysisPaneView: View {
    let pane: AnalysisPane
    let workspace: Workspace
    @Bindable var window: WindowState
    let system: OpticalSystem

    private var descriptor: AnalysisDescriptor? { workspace.descriptor(pane.analysis) }
    private var isFocused: Bool { workspace.focusedPane === pane && workspace.panes.count > 1 }
    private var context: AnalysisContext {
        AnalysisContext(
            system: pane.output.flatMap { _ in workspace.snapshot?.system } ?? system,
            selectSurface: { i in
                if system.surfaces.indices.contains(i) { window.selection = system.surfaces[i].id }
            })
    }

    var body: some View {
        VStack(spacing: 0) {
            PaneHeader(pane: pane, workspace: workspace, descriptor: descriptor, isFocused: isFocused)
            if pane.showsSettings, let descriptor, descriptor.available, !descriptor.settingsFields.isEmpty {
                SettingsBar(pane: pane, descriptor: descriptor, system: system)
                Divider()
            }
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(alignment: .top) {
                    if let p = pane.progress, pane.output != nil {
                        ProgressView(value: p.fraction).progressViewStyle(.linear).controlSize(.mini).padding(.horizontal, 1)
                    }
                }
            if let out = pane.output {
                Divider()
                PaneFooter(pane: pane, output: out, context: context)
            }
        }
        .background(.background)
        .contentShape(Rectangle())
        .simultaneousGesture(TapGesture().onEnded { workspace.focusedPaneID = pane.id })
        .accessibilityElement(children: .contain)
        .accessibilityLabel(AnalysisCatalog.title(of: pane.analysis))
    }

    @ViewBuilder private var content: some View {
        if let out = pane.output {
            ZStack {
                AnalysisResultView(analysis: pane.analysis, output: out, context: context)
                    .opacity(pane.isStale ? 0.4 : 1)
                    .saturation(pane.isStale ? 0.2 : 1)
                    .allowsHitTesting(!pane.isStale || pane.isRunning)
                if pane.isStale && !pane.isRunning {
                    StaleBanner(pane: pane, blocked: !workspace.modelErrors.isEmpty)
                }
                if case .failed(let why) = pane.phase {
                    VStack {
                        Spacer()
                        Label(why, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .padding(Space.s)
                            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                            .padding(Space.m)
                    }
                }
            }
            .animation(.easeOut(duration: 0.2), value: pane.isStale)
        } else {
            placeholder
        }
    }

    @ViewBuilder private var placeholder: some View {
        switch pane.phase {
        case .running(let p):
            VStack(spacing: Space.m) {
                ProgressView(value: p.fraction > 0 ? p.fraction : nil).frame(width: 160)
                Text(p.stage.isEmpty ? "Computing…" : p.stage).font(.caption).foregroundStyle(.secondary)
                Button("Cancel") { pane.cancel() }.controlSize(.small)
            }
        case .unavailable(let why):
            ContentUnavailableView(
                "Not Available", systemImage: "nosign",
                description: Text(why.isEmpty ? "This engine build does not provide this analysis." : why))
        case .failed(let why):
            ContentUnavailableView {
                Label("Analysis Failed", systemImage: "exclamationmark.triangle")
            } description: {
                Text(why)
            } actions: {
                Button("Try Again") { pane.rerun() }
            }
        default:
            if !workspace.modelErrors.isEmpty {
                ContentUnavailableView {
                    Label("Design Not Ready", systemImage: "xmark.octagon")
                } description: {
                    Text(workspace.modelErrors.first ?? "")
                } actions: {
                    Button("Show Issues") { window.showIssues = true }
                }
            } else if pane.phase == .cancelled {
                ContentUnavailableView {
                    Label("Cancelled", systemImage: "stop.circle")
                } actions: {
                    Button("Run") { pane.rerun() }
                }
            } else {
                ProgressView().controlSize(.small)
            }
        }
    }
}

/// Shown over a stale result: the design changed since it was computed.
struct StaleBanner: View {
    let pane: AnalysisPane
    let blocked: Bool

    var body: some View {
        VStack(spacing: Space.s) {
            Text(blocked ? "The design has errors" : "Design changed").font(.callout.weight(.medium))
            if !blocked {
                Button {
                    pane.rerun()
                } label: {
                    Label("Update", systemImage: "arrow.clockwise")
                }
                .keyboardShortcut("r", modifiers: .command)
                .controlSize(.regular)
            }
        }
        .padding(Space.l)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
    }
}

struct PaneHeader: View {
    let pane: AnalysisPane
    let workspace: Workspace
    let descriptor: AnalysisDescriptor?
    let isFocused: Bool

    var body: some View {
        HStack(spacing: Space.s) {
            Menu {
                AnalysisPickerItems(workspace: workspace) { pane.show($0) }
            } label: {
                Label(AnalysisCatalog.title(of: pane.analysis), systemImage: AnalysisCatalog.symbol(for: pane.analysis))
                    .font(.headline)
                    .foregroundStyle(isFocused ? Color.accentColor : .primary)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help(descriptor?.description ?? "")
            Spacer(minLength: Space.s)
            if let p = pane.progress {
                Text(p.stage.isEmpty ? "Computing" : p.stage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Button("Cancel", systemImage: "xmark.circle.fill") { pane.cancel() }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Cancel this analysis (⌘.)")
            } else if pane.isStale {
                Text("Out of date").font(.caption).foregroundStyle(.secondary)
            }
            Group {
                Toggle(isOn: Binding(get: { pane.autoUpdate }, set: { pane.autoUpdate = $0; if $0 { pane.rerun() } })) {
                    Label("Update Automatically", systemImage: "arrow.triangle.2.circlepath")
                }
                .toggleStyle(.button)
                .help(pane.autoUpdate ? "Updates when the design changes. Click to update on request only." : "Updates on request only. Click to update automatically.")
                Button("Run", systemImage: "arrow.clockwise") { pane.rerun() }
                    .help("Run again (⌘R)")
                Toggle(isOn: Binding(get: { pane.showsSettings }, set: { pane.showsSettings = $0 })) {
                    Label("Settings", systemImage: "slider.horizontal.3")
                }
                .toggleStyle(.button)
                .help("Show or hide the settings")
                Menu {
                    Button("Duplicate Pane") { workspace.addPane(pane.analysis, settings: pane.settings, after: pane) }
                    Button("Move Left") { workspace.movePane(pane, by: -1) }.disabled(workspace.panes.first === pane)
                    Button("Move Right") { workspace.movePane(pane, by: 1) }.disabled(workspace.panes.last === pane)
                    Divider()
                    Button("Copy Result as JSON") { copyJSON() }.disabled(pane.output == nil)
                    Divider()
                    Button("Close Pane", role: .destructive) { workspace.removePane(pane) }
                } label: {
                    Label("More", systemImage: "ellipsis.circle")
                }
                .menuIndicator(.hidden)
                .fixedSize()
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .controlSize(.small)
        }
        .padding(.horizontal, Space.m)
        .frame(height: 30)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
    }

    private func copyJSON() {
        guard let out = pane.output, let data = try? MKLightJSON.encode(out.json, pretty: true) else { return }
        Clipboard.copy(String(decoding: data, as: UTF8.self))
    }
}

/// Analyses grouped by category, for menus.
struct AnalysisPickerItems: View {
    let workspace: Workspace
    let choose: (String) -> Void

    var body: some View {
        ForEach(AnalysisCatalog.grouped(workspace.paneAnalyses), id: \.category) { g in
            Section(AnalysisCatalog.categoryTitle(g.category)) {
                ForEach(g.items) { d in
                    Button {
                        choose(d.name)
                    } label: {
                        Label(AnalysisCatalog.title(of: d), systemImage: AnalysisCatalog.symbol(for: d.name))
                    }
                    .disabled(!d.available)
                }
            }
        }
    }
}

/// Key numbers of the result, and how it was computed.
struct PaneFooter: View {
    let pane: AnalysisPane
    let output: AnalysisOutput
    let context: AnalysisContext
    @State private var showDetails = false

    var body: some View {
        HStack(spacing: Space.l) {
            let numbers = AnalysisResultView.keyNumbers(analysis: pane.analysis, output: output, context: context)
            ForEach(numbers.prefix(4), id: \.label) { n in
                HStack(spacing: 3) {
                    Text(n.label).foregroundStyle(.secondary)
                    Text(n.value).numeric()
                }
                .lineLimit(1)
            }
            Spacer(minLength: Space.s)
            Button {
                showDetails.toggle()
            } label: {
                HStack(spacing: Space.xs) {
                    if output.backend?.fallbackReason != nil || !output.runInfo.warnings.isEmpty {
                        Image(systemName: output.runInfo.warnings.isEmpty ? "info.circle" : "exclamationmark.triangle")
                            .foregroundStyle(output.runInfo.warnings.isEmpty ? Color.secondary : Color.orange)
                    }
                    Text(backend)
                    Text(duration).foregroundStyle(.tertiary)
                }
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("How this result was computed")
            .popover(isPresented: $showDetails) { RunDetails(output: output).frame(width: 380) }
        }
        .font(.caption)
        .padding(.horizontal, Space.m)
        .frame(height: 22)
        .opacity(pane.isStale ? 0.5 : 1)
    }

    private var backend: String {
        output.backend?.summary ?? output.runInfo.backend
    }

    private var duration: String {
        let ms = output.runInfo.durationMs
        return ms < 1000 ? String(format: "%.0f ms", ms) : String(format: "%.1f s", ms / 1000)
    }
}

/// Run information: backend, fallback, seed, snapshot, warnings.
struct RunDetails: View {
    let output: AnalysisOutput

    var body: some View {
        let info = output.runInfo
        VStack(alignment: .leading, spacing: Space.m) {
            Text("Run Information").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: Space.xs) {
                row("Computed on", output.backend?.summary ?? info.backend)
                if let r = output.backend?.requested, !r.isEmpty { row("Requested", r) }
                if let why = output.backend?.fallbackReason { row("Why", why) }
                row("Precision", info.precision == .f64Reference ? "f64 reference" : "f32 (GPU, qualified)")
                if let seed = info.seed { row("Seed", "\(seed)") }
                row("Snapshot", info.snapshotHash.short)
                row("Engine", info.engineVersion)
                row("Time", String(format: "%.1f ms", info.durationMs))
            }
            .font(.callout)
            if !info.warnings.isEmpty {
                Divider()
                Text("Warnings").font(.subheadline.weight(.medium))
                ForEach(info.warnings, id: \.self) { w in
                    Label(w, systemImage: "exclamationmark.triangle").font(.caption).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(Space.l)
        .textSelection(.enabled)
    }

    private func row(_ k: String, _ v: String) -> some View {
        GridRow {
            Text(k).foregroundStyle(.secondary)
            Text(v).fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Copies text to the pasteboard.
enum Clipboard {
    static func copy(_ s: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
        #elseif canImport(UIKit)
        UIPasteboard.general.string = s
        #endif
    }
}
