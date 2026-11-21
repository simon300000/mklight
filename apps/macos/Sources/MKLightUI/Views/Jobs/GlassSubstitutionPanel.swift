import MKLightEngine
import MKLightModel
import SwiftUI

/// Glass substitution as a design job: choose the lenses whose glass may
/// change and the candidates, run a quick or thorough search that
/// re-optimises the shape for every glass combination, then compare the
/// ranked combinations (each previewed in the window's layout and
/// analyses, and on the glass map) and apply one as an undoable step.
struct GlassSubstitutionPanel: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    @Bindable var session: GlassSubstitutionSession
    @Namespace private var glass
    @Environment(\.undoManager) private var undoManager
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var map: GlassMapData?
    @State private var mapSelection: String?
    @State private var plane = GlassMapPlane.ndVd

    private var slots: [GlassSlot] { GlassSlot.all(in: design.system) }

    var body: some View {
        JobPanelFrame(kind: .glass, subtitle: subtitle) {
            switch session.state {
            case .setup: setup
            case .running: running
            case .finished: results
            case .failed(let why): JobFailure(title: String(localized: "Glass Substitution Failed"), reason: why)
            }
        } actions: {
            actions
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.35), value: phase)
        .onAppear {
            session.configure(for: design.system)
            autoRun()
        }
        .onChange(of: design.system) { _, s in session.configure(for: s) }
        .task(id: workspace.designSnapshot?.libraryHash ?? "") { await loadMap() }
        .onDisappear {
            // Closing the panel shows the design again; the ranking is kept.
            if !session.isRunning { workspace.preview(nil) }
        }
    }

    private var subtitle: String {
        let n = session.selected.count
        return n == 0 ? String(localized: "Snap model glasses to real glasses") : n == 1 ? String(localized: "1 glass may change") : String(localized: "\(n) glasses may change", locale: NumberDisplay.locale)
    }

    private var phase: Int {
        switch session.state {
        case .setup, .failed: 0
        case .running: 1
        case .finished: session.applied ? 3 : 2
        }
    }

    // MARK: Setup

    private var setup: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            GroupBox {
                VStack(alignment: .leading, spacing: Space.s) {
                    if slots.isEmpty {
                        Text("The design has no glass to replace.").font(.callout).foregroundStyle(.secondary)
                    }
                    ForEach(slots) { slot in slotRow(slot) }
                    HStack {
                        Button("All") { session.selected = Set(slots.map(\.id)) }
                        Button("None") { session.selected = [] }
                        Spacer()
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                    if session.selected.isEmpty {
                        Text("With no glass ticked, every model glass (n_d, V_d) snaps to its nearest real glasses.")
                            .hint()
                    }
                    if let note = session.offsetNote(system: design.system) {
                        Label(note, systemImage: "drop.halffull")
                            .hint()
                    }
                }
                .padding(Space.xs)
            } label: {
                Text("Glasses That May Change")
            }
            GroupBox {
                CandidateForm(candidates: $session.candidates, catalogs: workspace.catalogs)
                    .padding(Space.xs)
            } label: {
                Text("Candidates for Every Lens")
            }
            GroupBox {
                VStack(alignment: .leading, spacing: Space.s) {
                    Picker("Search", selection: $session.algorithm) {
                        ForEach(GlassSubstitutionSession.Algorithm.allCases) { Text($0.title).tag($0) }
                    }
                    Text(session.algorithm.detail).hint()
                    LabeledContent("Combinations to try") {
                        TextField("Combinations", value: $session.maxCombinations.atLeast(1), format: .number.grouping(.never))
                            .labelsHidden().multilineTextAlignment(.trailing).frame(width: 80)
                    }
                    LabeledContent("Time limit") {
                        HStack(spacing: 4) {
                            OptionalNumberField(title: String(localized: "Time limit"), value: session.maxSeconds, prompt: String(localized: "none"), width: 80) { v in
                                session.maxSeconds = v.flatMap { $0 > 0 ? $0 : nil }
                            }
                            Text("s").foregroundStyle(.secondary)
                        }
                    }
                    LabeledContent("Shape iterations each") {
                        TextField("Iterations", value: $session.localIterations.atLeast(1), format: .number.grouping(.never))
                            .labelsHidden().multilineTextAlignment(.trailing).frame(width: 80)
                    }
                    .help("Damped least-squares iterations that re-optimize the shape for each glass combination")
                    LabeledContent("Keep the best") {
                        TextField("Keep", value: $session.topN.atLeast(1), format: .number.grouping(.never))
                            .labelsHidden().multilineTextAlignment(.trailing).frame(width: 80)
                    }
                }
                .padding(Space.xs)
            } label: {
                Text("Search")
            }
            GroupBox {
                VStack(alignment: .leading, spacing: Space.xs) {
                    Text(meritSummary).font(.callout).fixedSize(horizontal: false, vertical: true)
                    if design.variables.isEmpty {
                        Text("Each combination re-optimizes every curvature and the back focus.")
                            .hint(.callout)
                    } else {
                        Picker("Shape", selection: $session.useDesignVariables) {
                            Text(String(localized: "Re-optimize the design variables (\(design.variables.filter { !$0.isPinned }.count))", locale: NumberDisplay.locale)).tag(true)
                            Text("Re-optimize every curvature and the back focus").tag(false)
                        }
                        .pickerStyle(.inline)
                        .labelsHidden()
                    }
                    Button("Edit Merit and Variables…") { window.open(.optimize) }
                        .buttonStyle(.link)
                        .controlSize(.small)
                }
                .padding(Space.xs)
            } label: {
                Text("Merit")
            }
            if !workspace.modelErrors.isEmpty {
                Label("Fix the design's errors first.", systemImage: "xmark.octagon").font(.callout).foregroundStyle(.red)
            }
        }
        .onAppear { if design.variables.isEmpty { session.useDesignVariables = false } }
    }

    private var meritSummary: String {
        let m = design.merit
        let n = m.operands.count, summary = m.defaults.summary
        switch (m.mode == .custom, m.holdEFL) {
        case (true, true): return String(localized: "Custom merit, \(n) operands, focal length held.", locale: NumberDisplay.locale)
        case (true, false): return String(localized: "Custom merit, \(n) operands.", locale: NumberDisplay.locale)
        case (false, true): return String(localized: "Default merit: \(summary), focal length held.", locale: NumberDisplay.locale)
        case (false, false): return String(localized: "Default merit: \(summary).", locale: NumberDisplay.locale)
        }
    }

    private func slotRow(_ slot: GlassSlot) -> some View {
        HStack(spacing: Space.s) {
            Toggle(isOn: Binding(get: { session.selected.contains(slot.id) }, set: { on in
                if on { session.selected.insert(slot.id) } else { session.selected.remove(slot.id) }
            })) {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(spacing: Space.xs) {
                        Text(String(localized: "S\(slot.index)", locale: NumberDisplay.locale)).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                        Text(slot.material).font(.callout.weight(.medium))
                        if slot.isModel { Text("model").font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 4).background(.quaternary, in: Capsule()) }
                        if let o = slot.offset, !o.isZero {
                            Text(o.compactText).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                                .help("Melt offset Δn_d / ΔV_d: a replacement glass is installed without it")
                        }
                    }
                    if let o = session.overrides[slot.id] {
                        Text(o.summary).font(.caption).foregroundStyle(.tint).lineLimit(1)
                    }
                }
            }
            .toggleStyle(.checkbox)
            Spacer()
            OverrideButton(slot: slot, session: session, catalogs: workspace.catalogs)
        }
        .accessibilityElement(children: .contain)
    }

    // MARK: Running

    private var running: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            HStack(alignment: .firstTextBaseline, spacing: Space.xl) {
                BigNumber(label: String(localized: "Combinations"), value: session.progress.map { "\($0.combinations) / \(session.maxCombinations)" } ?? "—")
                BigNumber(label: String(localized: "Best merit"), value: session.progress.map { Fmt.short($0.bestMerit) } ?? "—")
            }
            if let g = session.progress?.glasses, !g.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Best glasses so far").font(.caption).foregroundStyle(.secondary)
                    FlowLayout(spacing: Space.xs, lineSpacing: Space.xs) {
                        ForEach(Array(zip(selectedSlots, g)), id: \.0.id) { slot, name in
                            Text(String(localized: "S\(slot.index) \(name)", locale: NumberDisplay.locale)).font(.caption.monospacedDigit())
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(.tint.opacity(0.12), in: Capsule())
                        }
                    }
                }
                .accessibilityElement(children: .combine)
            }
            Sparkline(values: session.trace).frame(height: 110).padding(.vertical, Space.s)
            Text("Stop keeps the combinations ranked so far. Cancel discards the run. You can close this panel; the search continues.")
                .hint()
        }
    }

    private var selectedSlots: [GlassSlot] {
        let s = slots.filter { session.selected.contains($0.id) }
        return s.isEmpty ? slots.filter(\.isModel) : s
    }

    // MARK: Results

    @ViewBuilder private var results: some View {
        if let r = session.result {
            VStack(alignment: .leading, spacing: Space.l) {
                if session.applied { AppliedNotice() }
                Text(statusText(r)).hint(.callout)
                if r.ranked.isEmpty {
                    ContentUnavailableView("No Combinations", systemImage: "questionmark.circle", description: Text("No combination could be evaluated."))
                } else {
                    rankingList(r)
                    if let c = session.selectedCombination { comparison(c, start: r.startingMerit) }
                    glassMap
                }
            }
        }
    }

    private func rankingList(_ r: GlassSubstitutionResult) -> some View {
        List(selection: Binding(get: { session.selectedRank }, set: { session.select($0, in: workspace) })) {
            ForEach(r.ranked) { c in
                HStack(alignment: .firstTextBaseline, spacing: Space.s) {
                    Text(String(localized: "\(c.rank)", locale: NumberDisplay.locale)).font(.callout.monospacedDigit().weight(.semibold)).frame(width: 20, alignment: .trailing)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(c.glasses.map(\.name).joined(separator: " · ")).font(.callout).lineLimit(1)
                        Text(changeSummary(c)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer(minLength: Space.xs)
                    VStack(alignment: .trailing, spacing: 1) {
                        Text(Fmt.short(c.merit)).font(.callout.monospacedDigit())
                        if !c.feasible {
                            Text("constraints not met").font(.caption2).foregroundStyle(.orange)
                        } else if r.startingMerit > 0, c.merit.isFinite {
                            Text(Fmt.percent(1 - c.merit / r.startingMerit, digits: 0)).font(.caption2.monospacedDigit())
                                .foregroundStyle(c.merit < r.startingMerit ? AnyShapeStyle(.green) : AnyShapeStyle(.secondary))
                        }
                    }
                }
                .tag(Optional(c.rank))
                .accessibilityElement(children: .combine)
                .accessibilityLabel(String(localized: "Rank \(c.rank), merit \(Fmt.short(c.merit)), glasses \(c.glasses.map(\.name).joined(separator: ", "))", locale: NumberDisplay.locale))
            }
        }
        .listStyle(.bordered)
        .alternatingRowBackgrounds()
        .frame(height: min(CGFloat(r.ranked.count) * 40 + 8, 280))
        .disabled(session.applied)
    }

    private func changeSummary(_ c: GlassCombination) -> String {
        let moved = c.movedChanges.count
        var s = moved == 0 ? String(localized: "shape unchanged") : moved == 1 ? String(localized: "1 variable re-optimized") : String(localized: "\(moved) variables re-optimized", locale: NumberDisplay.locale)
        if c.cutShort { s += " · " + String(localized: "cut short") }
        return s
    }

    /// Before / after: glasses per surface, merit, variable changes and the
    /// two layouts side by side.
    private func comparison(_ c: GlassCombination, start: Double) -> some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack {
                Text(String(localized: "Candidate \(c.rank)", locale: NumberDisplay.locale)).font(.headline)
                Spacer()
                if !session.applied { Text("Previewed in the window").font(.caption).foregroundStyle(.tint) }
            }
            HStack(alignment: .firstTextBaseline, spacing: Space.xl) {
                BigNumber(label: String(localized: "Merit now"), value: Fmt.short(start))
                BigNumber(label: String(localized: "Candidate"), value: Fmt.short(c.merit))
            }
            Grid(alignment: .leading, horizontalSpacing: Space.s, verticalSpacing: 3) {
                ForEach(c.glasses, id: \.surface) { g in
                    GridRow {
                        Text(String(localized: "S\(design.system.surfaceIndex(of: g.surface).map(String.init) ?? "?")", locale: NumberDisplay.locale)).foregroundStyle(.secondary).monospacedDigit()
                        Text(session.startGlasses[g.surface] ?? "—").lineLimit(1)
                        Image(systemName: "arrow.right").font(.caption2).foregroundStyle(.secondary).accessibilityLabel("to")
                        Text(g.name).fontWeight(.semibold).lineLimit(1)
                        Text(String(format: "%.4f / %.1f", g.nd, g.vd)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
            .font(.callout)
            if !c.movedChanges.isEmpty {
                DisclosureGroup(String(localized: "Shape changes (\(c.movedChanges.count))", locale: NumberDisplay.locale)) {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(c.movedChanges.enumerated()), id: \.offset) { _, ch in
                            HStack {
                                Text(ch.param.map { $0.label(in: design.system) } ?? "?").lineLimit(1)
                                Spacer()
                                Text(String(localized: "\(Fmt.short(display(ch.start, ch.param))) → \(Fmt.short(display(ch.value, ch.param)))", locale: NumberDisplay.locale)).monospacedDigit()
                            }
                            .font(.caption)
                        }
                    }
                    .padding(.top, 2)
                }
                .font(.callout)
            }
            // While previewed, the candidate's layout; once applied, the design's.
            LayoutComparison(
                before: EngineLayout.scene(from: workspace.baseline["layout"]),
                after: workspace.isPreviewing || session.applied ? workspace.layout.layoutScene : nil)
        }
        .padding(Space.m)
        .background(.fill.quinary, in: .rect(cornerRadius: 10))
        .textSelection(.enabled)
    }

    private func display(_ v: Double, _ p: ParamPath?) -> Double {
        switch p {
        case .curvature?, .shapeParam(_, .curvatureX)?, .shapeParam(_, .curvatureY)?: RadiusDisplay.radius(fromCurvature: v)
        default: v
        }
    }

    @ViewBuilder private var glassMap: some View {
        if let map {
            VStack(alignment: .leading, spacing: Space.s) {
                HStack {
                    Text("Glass Map").font(.headline)
                    Spacer()
                    Picker("Plane", selection: $plane) {
                        Text("n_d").tag(GlassMapPlane.ndVd)
                        Text("P_g,F").tag(GlassMapPlane.pgfVd)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                }
                GlassMapChart(glasses: map.glasses, highlighted: highlights, selection: $mapSelection, plane: plane)
                    .frame(height: 260)
                Text("Ringed: the candidate's glasses, labeled with their surfaces.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    /// The previewed combination's glasses on the map.
    private var highlights: [GlassHighlight] {
        guard let c = session.selectedCombination else { return [] }
        var byGlass: [String: [String]] = [:]
        for g in c.glasses {
            byGlass[g.mapID, default: []].append("S\(design.system.surfaceIndex(of: g.surface).map(String.init) ?? "?")")
        }
        return byGlass.map { GlassHighlight(id: $0.key, label: $0.value.joined(separator: ", ")) }.sorted { $0.id < $1.id }
    }

    private func loadMap() async {
        guard let job = try? workspace.submitJob("glass_map", settings: [:], key: "glass-map"), let out = try? await job.result() else { return }
        map = GlassMapData(output: out)
    }

    private func statusText(_ r: GlassSubstitutionResult) -> String {
        let n = r.combinations
        var s: String
        switch (n == 1, r.failed > 0) {
        case (true, true): s = String(localized: "1 combination re-optimized, \(r.failed) failed.", locale: NumberDisplay.locale)
        case (true, false): s = String(localized: "1 combination re-optimized.")
        case (false, true): s = String(localized: "\(n) combinations re-optimized, \(r.failed) failed.", locale: NumberDisplay.locale)
        case (false, false): s = String(localized: "\(n) combinations re-optimized.", locale: NumberDisplay.locale)
        }
        s += " "
        if r.stopped { s += String(localized: "Stopped early: the ranking so far.") + " " }
        if r.termination == "time_limit" { s += String(localized: "The time limit ended the search.") + " " }
        if !session.applied { s += String(localized: "Select a combination to preview it in the window.") }
        return s
    }

    // MARK: Actions

    @ViewBuilder private var actions: some View {
        let bar = JobActions(namespace: glass)
        let cannotRun = workspace.snapshot == nil || !workspace.modelErrors.isEmpty || (session.selected.isEmpty && !slots.contains(where: \.isModel))
        switch session.state {
        case .setup:
            bar.start(String(localized: "Find Glasses"), disabled: cannotRun) { session.start(in: workspace, document: design) }
        case .failed:
            bar.failed(back: { session.reset(workspace) }, retryDisabled: cannotRun) {
                session.start(in: workspace, document: design, fresh: true)
            }
        case .running(let p):
            bar.running(
                p, detail: session.progress.map { String(localized: "\($0.combinations) combinations", locale: NumberDisplay.locale) } ?? String(localized: "Preparing"), canStop: session.canStop,
                stopHelp: String(localized: "End now and keep the combinations ranked so far"), cancel: session.cancel, stop: session.stop)
        case .finished:
            if session.applied {
                bar.done(help: String(localized: "Back to the glasses and the search settings")) { session.reset(workspace) }
            } else {
                bar.review(discard: { session.reset(workspace) }, applyDisabled: session.selectedCombination?.system == nil) {
                    session.apply(to: design, workspace: workspace, undoManager: undoManager)
                }
            }
        }
    }

    /// Debug builds: `-MKLightAutoRun glass` starts at once.
    private func autoRun() {
        #if DEBUG
        guard UserDefaults.standard.string(forKey: "MKLightAutoRun") == "glass", case .setup = session.state else { return }
        let applyAfter = UserDefaults.standard.double(forKey: "MKLightAutoApply")
        let pick = UserDefaults.standard.integer(forKey: "MKLightAutoPick")
        Task {
            try? await Task.sleep(for: .seconds(2))
            session.start(in: workspace, document: design)
            while session.isRunning { try? await Task.sleep(for: .milliseconds(100)) }
            if pick > 1 { session.select(pick, in: workspace) }
            if applyAfter > 0 {
                try? await Task.sleep(for: .seconds(applyAfter))
                session.apply(to: design, workspace: workspace, undoManager: undoManager)
            }
        }
        #endif
    }
}

/// Candidate settings for one lens, in a popover.
private struct OverrideButton: View {
    let slot: GlassSlot
    @Bindable var session: GlassSubstitutionSession
    let catalogs: [CatalogInfo]
    @State private var show = false

    var body: some View {
        Button {
            show = true
        } label: {
            Image(systemName: session.overrides[slot.id] == nil ? "slider.horizontal.3" : "slider.horizontal.below.square.filled.and.square")
        }
        .buttonStyle(.borderless)
        .help("Candidates for this lens only")
        .accessibilityLabel(String(localized: "Candidates for surface \(slot.index)", locale: NumberDisplay.locale))
        .popover(isPresented: $show) {
            VStack(alignment: .leading, spacing: Space.m) {
                Text(String(localized: "Candidates for S\(slot.index) · \(slot.material)", locale: NumberDisplay.locale)).font(.headline)
                Toggle("Own candidates for this lens", isOn: Binding(
                    get: { session.overrides[slot.id] != nil },
                    set: { session.overrides[slot.id] = $0 ? session.candidates : nil }))
                if session.overrides[slot.id] != nil {
                    CandidateForm(
                        candidates: Binding(get: { session.overrides[slot.id] ?? session.candidates }, set: { session.overrides[slot.id] = $0 }),
                        catalogs: catalogs)
                } else {
                    Text("Uses the candidates for every lens.").font(.callout).foregroundStyle(.secondary)
                }
            }
            .padding(Space.l)
            .frame(width: 340)
        }
    }
}

/// Which catalogue glasses are candidates, in plain terms.
struct CandidateForm: View {
    @Binding var candidates: GlassCandidates
    let catalogs: [CatalogInfo]
    @State private var showLimits = false

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            LabeledContent("Catalogs") {
                Menu(candidates.catalogs.isEmpty ? String(localized: "All loaded") : candidates.catalogs.joined(separator: ", ")) {
                    Button("All loaded") { candidates.catalogs = [] }
                    Divider()
                    ForEach(catalogs) { c in
                        Toggle(String(localized: "\(c.name) (\(c.glassCount))", locale: NumberDisplay.locale), isOn: Binding(
                            get: { candidates.catalogs.contains(c.name) },
                            set: { on in
                                if on { candidates.catalogs.append(c.name) } else { candidates.catalogs.removeAll { $0 == c.name } }
                            }))
                    }
                }
                .fixedSize()
            }
            LabeledContent("Status") {
                Menu(statusTitle) {
                    ForEach(GlassCandidates.allStatuses, id: \.self) { s in
                        Toggle(GlassCandidates.statusTitle(s), isOn: Binding(
                            get: { candidates.statuses.contains(s) },
                            set: { on in
                                if on { candidates.statuses.insert(s) } else if candidates.statuses.count > 1 { candidates.statuses.remove(s) }
                            }))
                    }
                }
                .fixedSize()
            }
            Picker("Choose", selection: $candidates.rule) {
                Text("The nearest glasses").tag(GlassCandidates.Rule.nearest)
                Text("Glasses within a window").tag(GlassCandidates.Rule.window)
            }
            switch candidates.rule {
            case .nearest:
                Stepper(String(localized: "\(candidates.nearest) nearest to the current glass", locale: NumberDisplay.locale), value: $candidates.nearest, in: 1...200)
                    .font(.callout.monospacedDigit())
            case .window:
                HStack {
                    Text("n_d ±").foregroundStyle(.secondary)
                    OptionalNumberField(title: String(localized: "n_d window"), value: candidates.windowNd, width: 64) { candidates.windowNd = max($0 ?? 0.05, 1e-4) }
                    Text("V_d ±").foregroundStyle(.secondary)
                    OptionalNumberField(title: String(localized: "V_d window"), value: candidates.windowVd, width: 56) { candidates.windowVd = max($0 ?? 5, 0.1) }
                }
                .font(.callout)
            }
            NameListField(title: String(localized: "Always include"), names: $candidates.include, prompt: "N-BK7, SCHOTT:F2")
            NameListField(title: String(localized: "Never use"), names: $candidates.exclude, prompt: String(localized: "names"))
            DisclosureGroup("Limits", isExpanded: $showLimits) {
                Grid(alignment: .leading, horizontalSpacing: Space.s, verticalSpacing: Space.xs) {
                    limit(String(localized: "Min. transmission"), \.minTransmission, help: String(localized: "Internal transmission at the shortest wavelength through the lens's center thickness (0…1)"))
                    limit(String(localized: "Max. relative cost"), \.maxRelativeCost, help: String(localized: "Catalog relative cost"))
                    limit(String(localized: "Max. climate class"), \.maxClimateClass, help: String(localized: "Climate resistance class (CR)"))
                    limit(String(localized: "Max. stain class"), \.maxStainClass, help: String(localized: "Stain resistance class (FR)"))
                }
                .padding(.top, Space.xs)
            }
            .font(.callout)
        }
    }

    private var statusTitle: String {
        GlassCandidates.allStatuses.filter(candidates.statuses.contains).map(GlassCandidates.statusTitle).joined(separator: ", ")
    }

    private func limit(_ title: String, _ key: WritableKeyPath<GlassCandidates, Double?>, help: String) -> some View {
        GridRow {
            Text(title).help(help)
            OptionalNumberField(title: title, value: candidates[keyPath: key], width: 70) { candidates[keyPath: key] = $0 }
        }
    }
}

/// Comma-separated glass names. Commits on Return or when focus leaves, if
/// the text was edited.
private struct NameListField: View {
    let title: String
    @Binding var names: [String]
    let prompt: String

    var body: some View {
        LabeledContent(title) {
            CommittingField(value: names, format: Self.entry, commit: { names = $0 }) { text, _ in
                TextField(title, text: text, prompt: Text(prompt))
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 150)
            }
        }
    }

    static let entry: FieldEdit<[String]>.Format = {
        let text: ([String]) -> String = { $0.joined(separator: ", ") }
        return FieldEdit.Format(exact: text, display: text, parse: GlassCandidates.names)
    }()
}

/// The design's and the candidate's layouts, one above the other.
struct LayoutComparison: View {
    let before: LayoutScene?
    let after: LayoutScene?

    var body: some View {
        HStack(spacing: Space.s) {
            figure(String(localized: "Before"), before)
            figure(String(localized: "Candidate"), after)
        }
        .frame(height: 96)
    }

    private func figure(_ title: String, _ scene: LayoutScene?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous).fill(.background)
                if let scene { LayoutThumbnail(scene: scene).padding(Space.xs) } else { ProgressView().controlSize(.small) }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(String(localized: "\(title) layout", locale: NumberDisplay.locale))
    }
}
