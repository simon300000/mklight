import MKLightEngine
import MKLightModel
import SwiftUI

/// What a result view needs besides the output.
struct AnalysisContext {
    var system: OpticalSystem?
    var selectSurface: (Int) -> Void = { _ in }
    /// The pane showing the result (views with in-place setting controls).
    var pane: AnalysisPane?

    var fieldUnit: String { FieldLabel.unit(system) }
    var wavelengthsUm: [Double] { system?.wavelengths.wavelengths.map(\.um) ?? [] }
    var primaryWavelength: Int { system?.wavelengths.primary ?? 0 }

    func fieldLabel(_ value: [Double]?) -> String { FieldLabel.text(value, unit: fieldUnit) }

    /// Whether the run of `output` had its own spectrum (`spectrum` in its
    /// recorded settings): the wavelength indices of its result then refer
    /// to that spectrum's samples, not to the design's wavelengths.
    static func hasRunSpectrum(_ output: AnalysisOutput) -> Bool {
        output.runInfo.settings["spectrum"].map { $0 != .null } ?? false
    }

    /// The wavelengths (µm) that the wavelength indices of `output` refer
    /// to, from the result where it states them: its `wavelengths_um`, or
    /// the `wavelength` and `wavelength_um` of each of `entries` (curves).
    /// Otherwise the run's explicit spectrum, or the design's wavelengths
    /// when the run had no spectrum of its own. A sampled spectrum the
    /// result does not state leaves the indices without a wavelength.
    func wavelengthsUm(of output: AnalysisOutput, entries: [JSONValue] = []) -> [Double] {
        if let um = output.json["wavelengths_um"]?.nfArray { return um }
        var um = wavelengthsUm
        if Self.hasRunSpectrum(output) {
            if case .explicit(let samples)? = SpectrumSetting(json: output.runInfo.settings["spectrum"]) {
                um = samples.map(\.um)
            } else {
                um = []
            }
        }
        for e in entries {
            guard let w = e["wavelength"]?.int, let v = e["wavelength_um"]?.nf, (0..<SpectralSampling.sampleLimits.upperBound).contains(w)
            else { continue }
            if w >= um.count { um += Array(repeating: .nan, count: w + 1 - um.count) }
            um[w] = v
        }
        return um
    }
}

/// One analysis pane: a header (the analysis, its state and its controls
/// in a glass cluster), the plot as content, and a footer with the key
/// numbers and the backend that computed them. The settings stay closed
/// behind the header's one-line summary; the sliders button opens them as a
/// form above the plot, and the choice is saved with the layout.
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
            }, pane: pane)
    }

    var body: some View {
        VStack(spacing: 0) {
            // The header and settings stay above the result, which is
            // clipped to its own area for drawing and hit testing alike.
            PaneHeader(pane: pane, workspace: workspace, descriptor: descriptor, system: system, isFocused: isFocused)
                .zIndex(2)
            if pane.showsSettings, let descriptor, descriptor.available, !descriptor.settingsFields.isEmpty {
                SettingsPanel(pane: pane, descriptor: descriptor, system: system)
                    .transition(reduceMotion ? .opacity : .move(edge: .top).combined(with: .opacity))
                    .zIndex(1)
            }
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipped()
                .contentShape(Rectangle())
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
        .animation(reduceMotion ? nil : .snappy(duration: 0.25), value: pane.showsSettings)
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
                    Button("Try Again") { pane.rerun(fresh: true) }.buttonStyle(.glass)
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
                    Button("Run") { pane.rerun(fresh: true) }.buttonStyle(.glass)
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
    @Environment(\.panelTabbed) private var tabbed
    @Environment(\.designSession) private var session

    var body: some View {
        let full = AnalysisCatalog.title(of: pane.analysis), short = AnalysisCatalog.shortTitle(of: pane.analysis)
        // As the pane narrows: the cluster folds into Run and More first,
        // then the title takes a smaller type step, then its abbreviation
        // (the full title stays in the help and the VoiceOver label).
        ViewThatFits(in: .horizontal) {
            row(full, .headline, compact: false)
            row(full, .headline, compact: true)
            row(full, .subheadline.weight(.semibold), compact: true)
            row(short, .headline, compact: true)
            row(short, .subheadline.weight(.semibold), compact: true)
        }
        .padding(.horizontal, Space.m)
        .padding(.top, Space.s)
        .padding(.bottom, Space.xs)
    }

    private func row(_ title: String, _ font: Font, compact: Bool) -> some View {
        HStack(alignment: .center, spacing: Space.s) {
            VStack(alignment: .leading, spacing: 0) {
                // The title is the pane's handle (drag it to move the pane);
                // the chevron chooses another analysis. As one of several
                // tabs, the tab carries the title.
                HStack(spacing: 2) {
                    if !tabbed {
                        Label(title, systemImage: AnalysisCatalog.symbol(for: pane.analysis))
                            .font(font)
                            .lineLimit(1)
                            .fixedSize()
                            .help(titleHelp + "\nDrag to move the pane, or out of the window to open it in a window of its own.")
                            .accessibilityAddTraits(.isHeader)
                            .panelHandle(.analysis(pane.id))
                    }
                    analysisMenu
                    if tabbed { stateLine }
                }
                if !tabbed { stateLine }
            }
            Spacer(minLength: Space.xs)
            cluster(compact: compact)
        }
    }

    /// The state line truncates; it never decides the layout.
    private var stateLine: some View {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .padding(.leading, 2)
                    .contentTransition(.opacity)
                    .frame(minWidth: 0, idealWidth: 0, maxWidth: .infinity, alignment: .leading)
                    .help(subtitle)
    }

    private var hasSettings: Bool { descriptor.map { $0.available && !$0.settingsFields.isEmpty } ?? false }

    private var analysisMenu: some View {
        Menu {
            AnalysisPickerItems(workspace: workspace) { pane.show($0) }
        } label: {
            Image(systemName: "chevron.down")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Show another analysis in this pane")
        .accessibilityLabel("Change Analysis")
        .accessibilityValue(AnalysisCatalog.title(of: pane.analysis))
    }

    private var titleHelp: String {
        let t = AnalysisCatalog.title(of: pane.analysis)
        guard let d = descriptor?.description, !d.isEmpty else { return t }
        return t + ": " + d
    }

    private func cluster(compact: Bool) -> some View {
        GlassEffectContainer(spacing: Space.s) {
            HStack(spacing: Space.xs) {
                if pane.isRunning {
                    Button("Cancel", systemImage: "xmark") { pane.cancel() }
                        .help("Cancel this analysis (⌘.)")
                } else {
                    Button("Run", systemImage: "arrow.clockwise") { pane.rerun(fresh: true) }
                        .help("Run again (⌘R)")
                }
                if !compact {
                    Toggle(isOn: autoUpdate) {
                        Label("Update Automatically", systemImage: pane.autoUpdate ? "bolt.fill" : "bolt.slash")
                    }
                    .toggleStyle(.button)
                    .help(pane.autoUpdate ? "Updates when the design changes. Click to update on request only." : "Updates on request only. Click to update automatically.")
                    if hasSettings {
                        Toggle(isOn: showSettings) {
                            Label("Settings", systemImage: "slider.horizontal.3")
                        }
                        .toggleStyle(.button)
                        .help(pane.showsSettings ? "Hide the settings" : "Show the settings of this analysis")
                    }
                }
                Menu {
                    if compact {
                        Toggle("Update Automatically", isOn: autoUpdate)
                        if hasSettings { Toggle("Show Settings", isOn: showSettings) }
                        Divider()
                    }
                    Button("Duplicate Pane") { workspace.addPane(pane.analysis, settings: pane.settings, after: pane) }
                    Button("Copy Result as JSON") { copyJSON() }.disabled(pane.output == nil)
                    Divider()
                    if let session {
                        PanelMenuItems(panel: .analysis(pane.id), session: session)
                    } else {
                        Button("Close Pane", role: .destructive) { workspace.removePane(pane) }
                    }
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
        .fixedSize()
    }

    private var autoUpdate: Binding<Bool> {
        Binding(get: { pane.autoUpdate }, set: { pane.autoUpdate = $0; if $0 { pane.rerun() } })
    }

    private var showSettings: Binding<Bool> {
        Binding(get: { pane.showsSettings }, set: { pane.showsSettings = $0 })
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

/// The schema-driven settings of a pane, as a form between the header and
/// the plot: a grouped card in the content layer (no glass), scrolling
/// when long so the plot keeps most of the pane, with Defaults when the
/// settings differ from a new pane's.
struct SettingsPanel: View {
    let pane: AnalysisPane
    let descriptor: AnalysisDescriptor
    let system: OpticalSystem?

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            ScrollView {
                SettingsBar(pane: pane, descriptor: descriptor, system: system)
                    .environment(\.analysisPane, pane)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, Space.m)
                    .padding(.top, Space.m)
            }
            .frame(maxHeight: 300)
            .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                if pane.settings != pane.startingSettings {
                    Button("Defaults") { pane.setSettings(pane.startingSettings) }
                        .help("Restore the default settings")
                }
                Button("Done") { pane.showsSettings = false }
                    .help("Hide the settings")
            }
            .controlSize(.small)
            .buttonStyle(.borderless)
            .padding(.horizontal, Space.m)
            .padding(.bottom, Space.s)
        }
        .background(.fill.quaternary, in: .rect(cornerRadius: 10))
        .padding(.horizontal, Space.m)
        .padding(.bottom, Space.xs)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(AnalysisCatalog.title(of: pane.analysis)) settings")
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
                ViewThatFits(in: .horizontal) {
                    BackendBadge(output: output)
                    BackendBadge(output: output, compact: true)
                }
            }
            .buttonStyle(.plain)
            .layoutPriority(2)
            .help(badgeHelp)
            .accessibilityHint("Shows how this result was computed" + (output.runInfo.warnings.isEmpty ? "" : ", and its warnings"))
            .popover(isPresented: $showDetails) { RunDetails(output: output, pane: pane).frame(width: 400) }
            #if DEBUG
            // `-MKLightRunDetails <analysis>` opens the run details (screenshots).
            .task {
                guard UserDefaults.standard.string(forKey: "MKLightRunDetails") == pane.analysis, !pane.isStale else { return }
                try? await Task.sleep(for: .seconds(1))
                showDetails = true
            }
            #endif
        }
        .font(.caption)
        .padding(.horizontal, Space.m)
        .frame(height: 28)
        .opacity(pane.isStale ? 0.5 : 1)
    }

    /// The warnings themselves on hover; else what the badge is.
    private var badgeHelp: String {
        var parts: [String] = []
        if let why = output.backend?.fallbackReason { parts.append("Ran on the CPU: " + why) }
        let groups = RunWarnings.group(output.runInfo.warnings)
        parts += groups.prefix(3).map(\.line)
        if groups.count > 3 { parts.append("… \(groups.count - 3) more") }
        return parts.isEmpty ? "How this result was computed" : parts.joined(separator: "\n") + "\nClick for details."
    }

    private func numbersRow(_ numbers: ArraySlice<KeyNumber>) -> some View {
        HStack(spacing: Space.l) {
            ForEach(numbers, id: \.label) { n in
                HStack(spacing: 3) {
                    Subscripted.text(n.label).foregroundStyle(.secondary)
                    Text(n.value).numeric().fontWeight(.medium)
                }
                .fixedSize()
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(Subscripted.spoken(n.label))
                .accessibilityValue(n.value)
            }
        }
    }
}

/// The backend that computed a result, as a small tinted capsule: CPU f64
/// reference in a neutral fill, GPU f32 in the accent's; a warning adds an
/// orange mark. A run the router placed on the CPU is not one: the engine
/// warns only when a backend that was asked for or tried was not used.
/// Content, not glass.
struct BackendBadge: View {
    let output: AnalysisOutput
    /// Only the processor symbol (narrow panes).
    var compact = false

    var body: some View {
        let gpu = output.runInfo.precision != .f64Reference
        let groups = RunWarnings.group(output.runInfo.warnings)
        HStack(spacing: Space.xs) {
            if !groups.isEmpty {
                HStack(spacing: 2) {
                    Image(systemName: "exclamationmark.triangle.fill")
                    if groups.count > 1 { Text("\(groups.count)") }
                }
                .foregroundStyle(.orange)
            }
            Image(systemName: gpu ? "cpu.fill" : "cpu")
            if !compact {
                Text(gpu ? "GPU · f32" : "CPU · f64")
                Text(duration).foregroundStyle(.secondary)
            }
        }
        .font(.caption2.weight(.medium).monospacedDigit())
        .lineLimit(1)
        .fixedSize()
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(gpu ? AnyShapeStyle(.tint.opacity(0.16)) : AnyShapeStyle(.fill.tertiary), in: Capsule())
        .foregroundStyle(gpu ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityText(groups))
    }

    private func accessibilityText(_ groups: [WarningGroup]) -> String {
        var t = "Computed on \(output.backend?.summary ?? output.runInfo.backend) in \(duration)"
        if let why = output.backend?.fallbackReason { t += ". Ran on the CPU: \(why)" }
        if !groups.isEmpty {
            t += ". \(groups.count) warning\(groups.count == 1 ? "" : "s"): " + groups.map(\.line).joined(separator: ". ")
        }
        return t
    }

    private var duration: String { Fmt.duration(ms: output.runInfo.durationMs) }
}

/// Run information: backend, how an automatic request was routed, why the
/// run was on the CPU, the tier when it is not design tier, seed,
/// snapshot, the backend attempts that did not compute the result, and the
/// warnings, grouped (one line per kind, with the fields it concerns),
/// with a one-click remedy where the analysis has the setting for it.
struct RunDetails: View {
    let output: AnalysisOutput
    var pane: AnalysisPane?
    @State private var showAll = false

    var body: some View {
        let info = output.runInfo
        VStack(alignment: .leading, spacing: Space.m) {
            Text("Run Information").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: Space.xs) {
                row("Computed on", output.backend?.summary ?? info.backend)
                if let r = output.backend?.requested, !r.isEmpty { row("Requested", r) }
                if let b = output.backend, let route = RunRoute.line(requested: b.requested, decision: b.costDecision) {
                    row("Route", route)
                }
                if let why = output.backend?.fallbackReason { row("Ran on the CPU", why) }
                row("Precision", info.precision == .f64Reference ? "f64 reference" : "f32 (GPU, qualified)")
                if let tier = RunRoute.tier(info.tier) { row("Tier", tier) }
                if let seed = info.seed { row("Seed", "\(seed)") }
                row("Snapshot", info.snapshotHash.short)
                row("Engine", info.engineVersion)
                row("Time", String(format: "%.1f ms", info.durationMs))
            }
            .font(.callout)
            let attempts = RunRoute.attempts(info.failedAttempts)
            if !attempts.isEmpty {
                Divider()
                Text(attempts.count == 1 ? "Attempt Not Used" : "\(attempts.count) Attempts Not Used").font(.subheadline.weight(.medium))
                ForEach(attempts) { a in
                    VStack(alignment: .leading, spacing: 3) {
                        Label {
                            Text(a.title).fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                        }
                        .font(.callout)
                        Text(a.reason).font(.caption).foregroundStyle(.secondary)
                            .padding(.leading, 22)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
            if !info.warnings.isEmpty {
                Divider()
                let groups = RunWarnings.group(info.warnings)
                Text(groups.count == 1 ? "Warning" : "\(groups.count) Warnings").font(.subheadline.weight(.medium))
                ForEach(groups) { g in
                    VStack(alignment: .leading, spacing: 3) {
                        Label {
                            Text(g.message).fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                        }
                        .font(.callout)
                        if !g.scopes.isEmpty {
                            Text(g.scopes.joined(separator: ", ")).font(.caption).foregroundStyle(.secondary)
                                .padding(.leading, 22)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        if let fix = remedy(g) {
                            Button(fix.title, action: fix.apply)
                                .controlSize(.small)
                                .buttonStyle(.bordered)
                                .padding(.leading, 22)
                                .help(fix.help)
                        }
                    }
                    .accessibilityElement(children: .combine)
                }
                if info.warnings.count > groups.count {
                    DisclosureGroup("All \(info.warnings.count) messages from the engine", isExpanded: $showAll) {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(info.warnings.enumerated()), id: \.offset) { _, w in
                                Text(w).font(.caption.monospaced()).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                    .font(.caption)
                }
            }
        }
        .padding(Space.l)
        .textSelection(.enabled)
    }

    struct Fix {
        var title: String
        var help: String
        var apply: () -> Void
    }

    /// A setting change that addresses the warning, when the analysis has it.
    private func remedy(_ g: WarningGroup) -> Fix? {
        guard let pane, let kind = g.remedy, let r = RunDetails.remedy(kind, settings: { pane.value($0) }) else { return nil }
        return Fix(title: r.title, help: r.help) { pane.set(r.key, r.value) }
    }

    /// The setting change for a warning, given the pane's settings: an
    /// explicit coarse pupil grid, image grid or PSF window goes back to the
    /// engine's automatic choice, which sizes it from the system. Automatic
    /// settings that still warn reached the engine's cap: nothing to offer.
    struct Remedy: Equatable {
        var title: String
        var help: String
        var key: String
        var value: JSONValue
    }

    static func remedy(_ kind: WarningGroup.Remedy, settings value: (String) -> JSONValue?) -> Remedy? {
        // Image simulation keeps its pupil and PSF window in `optics`.
        let optics = value("pupil") == nil && value("grid_size") == nil ? value("optics")?.object : nil
        switch kind {
        case .pupilSampling:
            guard var pupil = (optics?["pupil"] ?? value("pupil"))?.object, let grid = pupil["grid"]?.int, grid > 0 else { return nil }
            pupil["grid"] = .integer(0)
            let title = "Sample the Pupil Automatically (grid \(grid) → auto)"
            let help = "The pupil grid is set to \(grid). Automatic sampling picks a grid fine enough for the wavefront's slope (64 to 1024), then runs again."
            if var o = optics {
                o["pupil"] = .object(pupil)
                return Remedy(title: title, help: help, key: "optics", value: .object(o))
            }
            return Remedy(title: title, help: help, key: "pupil", value: .object(pupil))
        case .imageGrid:
            if var o = optics {
                guard let size = o["psf_size"]?.int, size > 0 else { return nil }
                o["psf_size"] = .integer(0)
                return Remedy(
                    title: "Size the PSF Window Automatically (\(size) → auto)",
                    help: "The PSF window is set to \(size) samples. The automatic window holds the blur and its diffraction wings, then runs again.",
                    key: "optics", value: .object(o))
            }
            guard let size = value("grid_size")?.int, size > 0 else { return nil }
            return Remedy(
                title: "Size the Image Grid Automatically (\(size) → auto)",
                help: "The image grid is set to \(size). The automatic grid holds the blur and its diffraction wings, then runs again.",
                key: "grid_size", value: .integer(0))
        }
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
