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

/// One analysis pane: a header (the analysis, its state, and its controls
/// in a glass cluster), the plot as content, and a footer with the key
/// numbers and the backend that computed them. Settings open in a popover
/// from the header, or stay as a compact strip when pinned.
struct AnalysisPaneView: View {
    let pane: AnalysisPane
    let workspace: Workspace
    @Bindable var window: WindowState
    let system: OpticalSystem
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

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
            PaneHeader(pane: pane, workspace: workspace, descriptor: descriptor, system: system, isFocused: isFocused)
            if pane.showsSettings, let descriptor, descriptor.available, !descriptor.settingsFields.isEmpty {
                SettingsBar(pane: pane, descriptor: descriptor, system: system)
                    .padding(.vertical, Space.xxs)
                    .background(.fill.quaternary, in: .rect(cornerRadius: 10))
                    .padding(.horizontal, Space.m)
                    .padding(.bottom, Space.xs)
            }
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(alignment: .top) {
                    if let p = pane.progress, pane.output != nil {
                        ProgressView(value: p.fraction > 0 ? p.fraction : nil)
                            .progressViewStyle(.linear)
                            .controlSize(.mini)
                            .padding(.horizontal, Space.m)
                            .transition(.opacity)
                    }
                }
            if let out = pane.output {
                PaneFooter(pane: pane, output: out, context: context)
            }
        }
        .background(.background)
        .overlay(alignment: .top) {
            // The focused pane, when there are several: a hairline in the accent colour.
            if isFocused { Rectangle().fill(.tint).frame(height: 2).accessibilityHidden(true) }
        }
        .contentShape(Rectangle())
        .simultaneousGesture(TapGesture().onEnded { workspace.focusedPaneID = pane.id })
        .accessibilityElement(children: .contain)
        .accessibilityLabel(AnalysisCatalog.title(of: pane.analysis))
    }

    @ViewBuilder private var content: some View {
        if let out = pane.output {
            ZStack {
                AnalysisResultView(analysis: pane.analysis, output: out, context: context)
                    .opacity(pane.isStale ? 0.35 : 1)
                    .saturation(pane.isStale ? 0.1 : 1)
                    .blur(radius: pane.isStale && !reduceMotion ? 1.5 : 0)
                    .allowsHitTesting(!pane.isStale || pane.isRunning)
                    .modifier(UpdatingPulse(active: pane.isRunning && !reduceMotion))
                if pane.isStale && !pane.isRunning {
                    StaleBanner(pane: pane, blocked: !workspace.modelErrors.isEmpty)
                        .transition(reduceMotion ? .opacity : .scale(scale: 0.9).combined(with: .opacity))
                }
                if case .failed(let why) = pane.phase {
                    VStack {
                        Spacer()
                        Label(why, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .padding(.horizontal, Space.l)
                            .padding(.vertical, Space.s)
                            .glassEffect(.regular, in: .capsule)
                            .padding(Space.m)
                    }
                }
            }
            .animation(reduceMotion ? nil : .easeOut(duration: 0.25), value: pane.isStale)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: pane.isRunning)
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
                Button("Cancel") { pane.cancel() }.buttonStyle(.glass).controlSize(.small)
            }
        case .unavailable(let why):
            ContentUnavailableView(
                "Not Available", systemImage: "nosign",
                description: Text(why.isEmpty ? "This engine build does not provide this analysis." : why))
        case .failed(let why):
            if workspace.issues.contains(where: { $0.severity == .error }) {
                ContentUnavailableView {
                    Label("Design Has Errors", systemImage: "xmark.octagon")
                } description: {
                    Text(why)
                } actions: {
                    Button("Show Issues") { window.showIssues = true }.buttonStyle(.glass)
                }
            } else {
                ContentUnavailableView {
                    Label("Analysis Failed", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(why)
                } actions: {
                    Button("Try Again") { pane.rerun() }.buttonStyle(.glass)
                }
            }
        default:
            if !workspace.modelErrors.isEmpty {
                ContentUnavailableView {
                    Label("Design Not Ready", systemImage: "xmark.octagon")
                } description: {
                    Text(workspace.modelErrors.first ?? "")
                } actions: {
                    Button("Show Issues") { window.showIssues = true }.buttonStyle(.glass)
                }
            } else if pane.phase == .cancelled {
                ContentUnavailableView {
                    Label("Cancelled", systemImage: "stop.circle")
                } actions: {
                    Button("Run") { pane.rerun() }.buttonStyle(.glass)
                }
            } else {
                ProgressView().controlSize(.small)
            }
        }
    }
}

/// While a stale result updates, it breathes gently (off with Reduce Motion).
struct UpdatingPulse: ViewModifier {
    let active: Bool
    @State private var low = false

    func body(content: Content) -> some View {
        content
            .opacity(active && low ? 0.75 : 1)
            .onChange(of: active, initial: true) { _, a in
                if a {
                    withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) { low = true }
                } else {
                    withAnimation(.easeOut(duration: 0.2)) { low = false }
                }
            }
    }
}

/// Shown over a stale result: the design changed since it was computed.
/// A glass card (control layer) over the dimmed plot, with Update as the
/// one prominent action.
struct StaleBanner: View {
    let pane: AnalysisPane
    let blocked: Bool

    var body: some View {
        VStack(spacing: Space.m) {
            Label(blocked ? "The design has errors" : "Design changed", systemImage: blocked ? "xmark.octagon" : "clock.arrow.circlepath")
                .font(.callout.weight(.medium))
            if !blocked {
                Button {
                    pane.rerun()
                } label: {
                    Label("Update", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.glassProminent)
                .keyboardShortcut("r", modifiers: .command)
            }
        }
        .padding(.horizontal, Space.xl)
        .padding(.vertical, Space.l)
        .glassEffect(.regular, in: .rect(cornerRadius: 18))
        .accessibilityElement(children: .contain)
    }
}

struct PaneHeader: View {
    let pane: AnalysisPane
    let workspace: Workspace
    let descriptor: AnalysisDescriptor?
    let system: OpticalSystem
    let isFocused: Bool
    @State private var showSettings = false

    var body: some View {
        HStack(alignment: .center, spacing: Space.s) {
            VStack(alignment: .leading, spacing: 0) {
                Menu {
                    AnalysisPickerItems(workspace: workspace) { pane.show($0) }
                } label: {
                    Label(AnalysisCatalog.title(of: pane.analysis), systemImage: AnalysisCatalog.symbol(for: pane.analysis))
                        .font(.headline)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help(descriptor?.description ?? "")
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .padding(.leading, 2)
                    .contentTransition(.opacity)
            }
            Spacer(minLength: Space.s)
            GlassEffectContainer(spacing: Space.s) {
                HStack(spacing: Space.xs) {
                    if pane.isRunning {
                        Button("Cancel", systemImage: "xmark") { pane.cancel() }
                            .help("Cancel this analysis (⌘.)")
                    } else {
                        Button("Run", systemImage: "arrow.clockwise") { pane.rerun() }
                            .help("Run again (⌘R)")
                    }
                    Toggle(isOn: Binding(get: { pane.autoUpdate }, set: { pane.autoUpdate = $0; if $0 { pane.rerun() } })) {
                        Label("Update Automatically", systemImage: pane.autoUpdate ? "bolt.fill" : "bolt.slash")
                    }
                    .toggleStyle(.button)
                    .help(pane.autoUpdate ? "Updates when the design changes. Click to update on request only." : "Updates on request only. Click to update automatically.")
                    if let descriptor, descriptor.available, !descriptor.settingsFields.isEmpty {
                        Button("Settings", systemImage: "slider.horizontal.3") { showSettings.toggle() }
                            .help("Settings of this analysis")
                            .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                                SettingsPopover(pane: pane, descriptor: descriptor, system: system)
                            }
                    }
                    Menu {
                        Button("Duplicate Pane") { workspace.addPane(pane.analysis, settings: pane.settings, after: pane) }
                        Button("Move Left") { workspace.movePane(pane, by: -1) }.disabled(workspace.panes.first === pane)
                        Button("Move Right") { workspace.movePane(pane, by: 1) }.disabled(workspace.panes.last === pane)
                        Divider()
                        Toggle("Pin Settings in Pane", isOn: Binding(get: { pane.showsSettings }, set: { pane.showsSettings = $0 }))
                        Button("Copy Result as JSON") { copyJSON() }.disabled(pane.output == nil)
                        Divider()
                        Button("Close Pane", role: .destructive) { workspace.removePane(pane) }
                    } label: {
                        Label("More", systemImage: "ellipsis")
                    }
                    .menuIndicator(.hidden)
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .controlSize(.small)
            }
        }
        .padding(.horizontal, Space.m)
        .padding(.top, Space.s)
        .padding(.bottom, Space.xs)
    }

    /// State, then the settings that differ from the defaults.
    private var subtitle: String {
        if let p = pane.progress { return p.stage.isEmpty ? "Computing…" : p.stage }
        let s = SettingsBar.summary(pane, descriptor)
        if pane.isStale { return s.isEmpty ? "Out of date" : "Out of date · " + s }
        if !s.isEmpty { return s }
        return "Default settings"
    }

    private func copyJSON() {
        guard let out = pane.output, let data = try? MKLightJSON.encode(out.json, pretty: true) else { return }
        Clipboard.copy(String(decoding: data, as: UTF8.self))
    }
}

/// The schema-driven settings of a pane, in a popover.
struct SettingsPopover: View {
    let pane: AnalysisPane
    let descriptor: AnalysisDescriptor
    let system: OpticalSystem

    var body: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            HStack {
                Text("\(AnalysisCatalog.title(of: pane.analysis)) Settings").font(.headline)
                Spacer()
                if !pane.settings.isEmpty {
                    Button("Defaults") { pane.setSettings([:]) }
                        .help("Restore the engine's default settings")
                }
            }
            ScrollView {
                SettingsBar(pane: pane, descriptor: descriptor, system: system, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 460)
            .fixedSize(horizontal: false, vertical: true)
            Toggle("Pin in pane", isOn: Binding(get: { pane.showsSettings }, set: { pane.showsSettings = $0 }))
                .toggleStyle(.switch)
                .controlSize(.mini)
                .font(.caption)
        }
        .padding(Space.l)
        .frame(width: 340)
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

/// Key numbers of the result, and a badge for the backend that computed it.
struct PaneFooter: View {
    let pane: AnalysisPane
    let output: AnalysisOutput
    let context: AnalysisContext
    @State private var showDetails = false

    var body: some View {
        HStack(spacing: Space.l) {
            let numbers = AnalysisResultView.keyNumbers(analysis: pane.analysis, output: output, context: context)
            ViewThatFits(in: .horizontal) {
                numbersRow(numbers.prefix(4))
                numbersRow(numbers.prefix(3))
                numbersRow(numbers.prefix(2))
                numbersRow(numbers.prefix(1))
                Color.clear.frame(width: 0, height: 0)
            }
            .layoutPriority(1)
            Spacer(minLength: Space.s)
            Button {
                showDetails.toggle()
            } label: {
                BackendBadge(output: output)
            }
            .buttonStyle(.plain)
            .help("How this result was computed")
            .popover(isPresented: $showDetails) { RunDetails(output: output).frame(width: 380) }
        }
        .font(.caption)
        .padding(.horizontal, Space.m)
        .frame(height: 28)
        .opacity(pane.isStale ? 0.5 : 1)
    }

    private func numbersRow(_ numbers: ArraySlice<KeyNumber>) -> some View {
        HStack(spacing: Space.l) {
            ForEach(numbers, id: \.label) { n in
                HStack(spacing: 3) {
                    Text(n.label).foregroundStyle(.secondary)
                    Text(n.value).numeric().fontWeight(.medium)
                }
                .fixedSize()
                .accessibilityElement(children: .combine)
            }
        }
    }
}

/// The backend that computed a result, as a small tinted capsule: CPU f64
/// reference in a neutral fill, GPU f32 in the accent's; a warning or a
/// fallback adds an orange mark. Content, not glass.
struct BackendBadge: View {
    let output: AnalysisOutput

    var body: some View {
        let gpu = output.runInfo.precision != .f64Reference
        let warned = !output.runInfo.warnings.isEmpty || output.backend?.fallbackReason != nil
        HStack(spacing: Space.xs) {
            if warned {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            }
            Image(systemName: gpu ? "cpu.fill" : "cpu")
            Text(gpu ? "GPU · f32" : "CPU · f64")
            Text(duration).foregroundStyle(.secondary)
        }
        .font(.caption2.weight(.medium).monospacedDigit())
        .lineLimit(1)
        .fixedSize()
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(gpu ? AnyShapeStyle(.tint.opacity(0.16)) : AnyShapeStyle(.fill.tertiary), in: Capsule())
        .foregroundStyle(gpu ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Computed on \(output.backend?.summary ?? output.runInfo.backend) in \(duration)\(warned ? ", with warnings" : "")")
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
