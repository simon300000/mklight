import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

/// A design job (Optimize, Tolerance, Apply Solves) as an inspector panel
/// rather than a modal sheet: the design, the layout and the analyses stay
/// visible and usable while it runs, and a job keeps running when its
/// panel closes (the toolbar status shows it).
struct JobPanel: View {
    let kind: DesignJobKind
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState

    var body: some View {
        switch kind {
        case .optimize: OptimizePanel(design: design, workspace: workspace, window: window, session: window.optimize)
        case .glass: GlassSubstitutionPanel(design: design, workspace: workspace, window: window, session: window.glass)
        case .tolerance: TolerancePanel(design: design, workspace: workspace, window: window, session: window.tolerance)
        case .solves: SolvesPanel(design: design, workspace: workspace, window: window, session: window.solves)
        }
    }
}

/// Frame of a job panel: a header floating over the scrolling content, and
/// the actions in a glass bar at the bottom whose buttons morph as the job
/// moves from setup to running to its result.
struct JobPanelFrame<Content: View, Actions: View>: View {
    let kind: DesignJobKind
    let subtitle: String
    @Bindable var window: WindowState
    @ViewBuilder var content: Content
    @ViewBuilder var actions: Actions

    var body: some View {
        ScrollView {
            content
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, Space.l)
                .padding(.vertical, Space.m)
        }
        .scrollEdgeEffectStyle(.soft, for: [.top, .bottom])
        .safeAreaBar(edge: .top) { header }
        .safeAreaBar(edge: .bottom) {
            GlassEffectContainer(spacing: Space.m) {
                HStack(spacing: Space.s) { actions }
            }
            .padding(Space.l)
        }
    }

    private var header: some View {
        HStack(spacing: Space.m) {
            Image(systemName: kind.symbol)
                .font(.title2)
                .foregroundStyle(.tint)
                .frame(width: 32, height: 32)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(kind.title).font(.title3.weight(.semibold))
                Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            Button("Close", systemImage: "xmark") { window.activeJob = nil }
                .labelStyle(.iconOnly)
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .help("Close the panel. A running job continues; the toolbar shows its progress.")
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, Space.l)
        .padding(.vertical, Space.m)
    }
}

/// One button of a job's action bar: a glass capsule that morphs with its
/// neighbours (`glassEffectID`). Prominent is the single primary action of
/// the flow, tinted with the accent colour.
struct GlassAction: View {
    let title: String
    var systemImage: String?
    var prominent = false
    var role: ButtonRole?
    var id: String
    let namespace: Namespace.ID
    let action: () -> Void
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(role: role, action: action) {
            Group {
                if let systemImage { Label(title, systemImage: systemImage) } else { Text(title) }
            }
            .font(.body.weight(prominent ? .semibold : .regular))
            .foregroundStyle(prominent && isEnabled ? AnyShapeStyle(.white) : role == .destructive ? AnyShapeStyle(.red) : AnyShapeStyle(.primary))
            .lineLimit(1)
            .padding(.horizontal, Space.xl)
            .padding(.vertical, Space.m)
            .frame(maxWidth: prominent ? .infinity : nil)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .glassEffect(prominent && isEnabled ? .regular.tint(.accentColor).interactive() : .regular.interactive(), in: .capsule)
        .opacity(isEnabled ? 1 : 0.55)
        .glassEffectID(id, in: namespace)
    }
}

/// A glass capsule with live progress, between the actions while a job runs.
struct GlassProgress: View {
    let progress: JobProgress
    var detail: String = ""
    let namespace: Namespace.ID

    var body: some View {
        HStack(spacing: Space.s) {
            ProgressView(value: progress.fraction > 0 ? progress.fraction : nil)
                .progressViewStyle(.circular)
                .controlSize(.small)
            VStack(alignment: .leading, spacing: 0) {
                Text(detail.isEmpty ? (progress.stage.isEmpty ? "Running" : progress.stage) : detail)
                    .font(.callout.monospacedDigit())
                    .lineLimit(1)
                if progress.fraction >= 0.01 {
                    Text(Fmt.percent(progress.fraction, digits: 0)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, Space.l)
        .padding(.vertical, Space.s)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: .capsule)
        .glassEffectID("status", in: namespace)
        .accessibilityElement(children: .combine)
    }
}

/// A large number with its label (job panels).
struct BigNumber: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.medium).monospacedDigit()).contentTransition(.numericText())
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Optimize

struct OptimizePanel: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    @Bindable var session: OptimizeSession
    @Namespace private var glass
    @Environment(\.undoManager) private var undoManager
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        JobPanelFrame(kind: .optimize, subtitle: design.system.name, window: window) {
            switch session.state {
            case .setup: setup
            case .running(let p): running(p)
            case .finished: result
            case .failed(let why):
                ContentUnavailableView {
                    Label("Optimization Failed", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(why)
                }
            }
        } actions: {
            actions
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.35), value: phase)
        .onAppear {
            session.load(design)
            autoRun()
        }
        .onChange(of: design.revision) { _, _ in session.load(design) }
        .onChange(of: phase) { _, p in syncPreview(p) }
        .onDisappear { if !session.isRunning { workspace.preview(nil) } }
        .onChange(of: design.merit) { _, _ in session.load(design) }
        .task(id: evaluationKey) {
            session.load(design)
            if case .setup = session.state { session.scheduleEvaluation(in: workspace, variables: design.variables, system: design.system) }
        }
    }

    /// A finished, unapplied result is previewed in the window (layout and
    /// analyses) until it is applied or discarded.
    private func syncPreview(_ p: Int) {
        if p == 2, let s = session.result?.system, session.result?.improved == true {
            workspace.preview(s, title: session.result?.stopped == true ? "Optimized design (stopped)" : "Optimized design")
        } else if p != 1 {
            workspace.preview(nil)
        }
    }

    /// Re-evaluates the merit when the design, its merit, the variables or
    /// the engine snapshot change.
    private var evaluationKey: [AnyHashable] {
        [design.revision, design.merit, design.variables, workspace.designSnapshot?.hash.hex ?? "", session.useDesignVariables, phase]
    }

    /// A key for the morphing animation.
    private var phase: Int {
        switch session.state {
        case .setup, .failed: 0
        case .running: 1
        case .finished: session.applied ? 3 : 2
        }
    }

    private var stochastic: Bool { ["de", "cma_es", "hammer"].contains(session.algorithm) }
    private var blocked: Bool { workspace.snapshot == nil || !workspace.modelErrors.isEmpty }

    private var setup: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            MeritFunctionSection(design: design, workspace: workspace, session: session)
            GroupBox {
                VStack(alignment: .leading, spacing: Space.s) {
                    Picker("Vary", selection: $session.useDesignVariables) {
                        Text("Design variables (\(design.variables.count))").tag(true)
                        Text("Every curvature and the back focus").tag(false)
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                    .disabled(design.variables.isEmpty)
                    if session.useDesignVariables && !design.variables.isEmpty {
                        VariablesBoundsList(design: design, workspace: workspace)
                    } else if design.variables.isEmpty {
                        Text("Mark values with V in the lens data or the inspector to choose them; each gets sensible bounds you can edit here.")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    if let why = session.variableProblem(design.variables, system: design.system) {
                        Label(why, systemImage: "pin.fill").font(.caption).foregroundStyle(.orange)
                    }
                }
                .padding(Space.xs)
            } label: {
                Text("Variables")
            }
            GroupBox("Algorithm") {
                VStack(alignment: .leading, spacing: Space.s) {
                    Picker("Method", selection: $session.algorithm) {
                        ForEach(OptimizeSession.algorithms, id: \.id) { Text($0.title).tag($0.id) }
                    }
                    LabeledContent("Iterations") {
                        TextField("Iterations", value: $session.maxIterations, format: .number, prompt: Text("default"))
                            .labelsHidden().multilineTextAlignment(.trailing).frame(width: 90)
                    }
                    if stochastic {
                        LabeledContent("Seed") {
                            TextField("Seed", value: $session.seed, format: .number)
                                .labelsHidden().multilineTextAlignment(.trailing).frame(width: 90)
                        }
                    }
                }
                .padding(Space.xs)
            }
            if !workspace.modelErrors.isEmpty {
                Label("Fix the design's errors first.", systemImage: "xmark.octagon").font(.callout).foregroundStyle(.red)
            }
        }
        .onAppear { if design.variables.isEmpty { session.useDesignVariables = false } }
    }

    private func running(_ p: JobProgress) -> some View {
        VStack(alignment: .leading, spacing: Space.l) {
            HStack(alignment: .firstTextBaseline, spacing: Space.xl) {
                BigNumber(label: "Iteration", value: session.trace.last.map { "\($0.iteration)" } ?? "—")
                BigNumber(label: "Search merit", value: session.trace.last.map { Fmt.short($0.merit) } ?? "—")
                    .help(["de", "cma_es", "nelder_mead"].contains(session.algorithm)
                        ? "The search's merit, including the penalty that holds the focal length; the result reports the merit itself."
                        : "The best merit so far, as the search evaluates it.")
            }
            if let first = session.trace.first, let last = session.trace.last, first.merit > 0 {
                BigNumber(label: "Improvement", value: Fmt.percent(1 - last.merit / first.merit))
            }
            Sparkline(values: session.trace.map(\.merit))
                .frame(height: 120)
                .padding(.vertical, Space.s)
            Text("Stop keeps the best design so far. Cancel discards the run. You can close this panel; the job continues.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var result: some View {
        if let r = session.result {
            VStack(alignment: .leading, spacing: Space.l) {
                HStack(alignment: .firstTextBaseline, spacing: Space.xl) {
                    BigNumber(label: "Merit before", value: Fmt.short(r.startMerit))
                    BigNumber(label: "After", value: Fmt.short(r.finalMerit))
                }
                BigNumber(label: "Improvement", value: Fmt.percent(r.improvement))
                if !session.applied, r.system != nil, r.improved {
                    Toggle("Preview in the window", isOn: Binding(get: { workspace.isPreviewing }, set: { on in
                        workspace.preview(on ? r.system : nil, title: "Optimized design")
                    }))
                    .help("Show the optimized design in the layout and analyses before applying it")
                    if workspace.isPreviewing {
                        LayoutComparison(
                            before: EngineLayout.scene(from: workspace.baseline["layout"]),
                            after: EngineLayout.scene(from: workspace.layout.output))
                    }
                }
                Text(statusText(r)).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if session.trace.count > 1 {
                    Sparkline(values: session.trace.map(\.merit)).frame(height: 64)
                }
                Text("Variables").font(.headline)
                VStack(alignment: .leading, spacing: Space.s) {
                    ForEach(r.variables) { v in
                        VStack(alignment: .leading, spacing: 1) {
                            HStack(spacing: Space.xs) {
                                Text(v.param.map { $0.label(in: design.system) } ?? v.label)
                                if v.atBound { Text("at bound").font(.caption2).foregroundStyle(.orange) }
                            }
                            .font(.callout)
                            HStack(spacing: Space.xs) {
                                Text(Fmt.short(displayed(v.start, v.param)))
                                Image(systemName: "arrow.right").font(.caption2).accessibilityLabel("to")
                                Text(Fmt.short(displayed(v.value, v.param))).fontWeight(.semibold)
                                Spacer()
                                Text(delta(v)).foregroundStyle(.secondary)
                            }
                            .font(.callout.monospacedDigit())
                        }
                        .accessibilityElement(children: .combine)
                    }
                }
                .textSelection(.enabled)
                if !r.evaluatedOn.isEmpty {
                    Text(
                        r.evaluatedOn.hasPrefix("cpu")
                            ? "Evaluated on the CPU f64 reference."
                            : "Searched on \(r.evaluatedOn); merits reported on the CPU f64 reference."
                    )
                    .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func delta(_ v: OptimizeResult.Variable) -> String {
        let d = displayed(v.value, v.param) - displayed(v.start, v.param)
        let unit = v.param.map { $0.isCurvature ? "mm" : $0.unit(in: design.system) } ?? ""
        return (d >= 0 ? "+" : "") + Fmt.short(d) + (unit.isEmpty ? "" : " " + unit)
    }

    /// Curvatures are shown as radii.
    private func displayed(_ v: Double, _ p: ParamPath?) -> Double {
        p?.isCurvature == true ? RadiusDisplay.radius(fromCurvature: v) : v
    }

    private func statusText(_ r: OptimizeResult) -> String {
        var s = "\(r.iterations) iterations, \(r.evaluations) evaluations. "
        if r.stopped { s += "Stopped early: this is the best design found so far. " }
        if !r.improved { s += "No better design was found; the result equals the start. " }
        if !r.feasible { s += "Some constraints are not met. " }
        if session.applied { s += "Applied to the design (Undo with ⌘Z)." }
        return s
    }

    @ViewBuilder private var actions: some View {
        switch session.state {
        case .setup, .failed:
            if case .failed = session.state {
                GlassAction(title: "Back", id: "secondary", namespace: glass) { session.reset() }
            }
            GlassAction(title: "Optimize", systemImage: "play.fill", prominent: true, id: "primary", namespace: glass) {
                session.start(in: workspace, variables: design.variables)
            }
            .keyboardShortcut(.defaultAction)
            .disabled(blocked || session.variableProblem(design.variables, system: design.system) != nil)
        case .running(let p):
            GlassProgress(progress: p, detail: session.trace.last.map { "Iteration \($0.iteration)" } ?? "", namespace: glass)
            GlassAction(title: "Cancel", role: .destructive, id: "secondary", namespace: glass) { session.cancel() }
                .help("End the run and discard it")
            GlassAction(title: "Stop", systemImage: "stop.fill", id: "primary", namespace: glass) { session.stop() }
                .disabled(!session.canStop)
                .help("End now and keep the best design so far")
        case .finished:
            if session.applied {
                GlassAction(title: "Run Again", id: "secondary", namespace: glass) { session.reset() }
                GlassAction(title: "Done", prominent: true, id: "primary", namespace: glass) {
                    session.reset()
                    window.activeJob = nil
                }
                .keyboardShortcut(.defaultAction)
            } else {
                GlassAction(title: "Discard", id: "secondary", namespace: glass) { session.reset() }
                GlassAction(title: "Apply to Design", prominent: true, id: "primary", namespace: glass) {
                    session.apply(to: design, undoManager: undoManager)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(session.result?.system == nil || session.result?.improved == false)
            }
        }
    }

    /// Debug builds: `-MKLightAutoRun lm` (or `de`) starts at once.
    private func autoRun() {
        #if DEBUG
        let d = UserDefaults.standard
        autoMerit(d)
        guard let a = d.string(forKey: "MKLightAutoRun"), ["lm", "dogleg", "de", "cma_es", "nelder_mead", "hammer"].contains(a),
            case .setup = session.state
        else { return }
        session.algorithm = a
        if a == "de" { session.maxIterations = 100_000 }
        // `-MKLightAutoStop s` presses Stop after s seconds; `-MKLightAutoApply s`
        // applies the result s seconds after it arrives (screenshots).
        let stopAfter = d.double(forKey: "MKLightAutoStop"), applyAfter = d.double(forKey: "MKLightAutoApply")
        Task {
            try? await Task.sleep(for: .seconds(1))
            session.start(in: workspace, variables: design.variables)
            if stopAfter > 0 {
                try? await Task.sleep(for: .seconds(stopAfter))
                session.stop()
            }
            if applyAfter > 0 {
                while session.isRunning { try? await Task.sleep(for: .milliseconds(100)) }
                try? await Task.sleep(for: .seconds(applyAfter))
                session.apply(to: design, undoManager: undoManager)
            }
        }
        #endif
    }
}

#if DEBUG
extension OptimizePanel {
    /// Debug builds, for screenshots and checks of the live merit:
    /// `-MKLightAutoEdit s` edits the merit s seconds after the panel opens
    /// and restores it a second later (default: the MTF target up 0.1, or
    /// one more ring; custom: the first operand's weight doubled), and `-MKLightAutoConvert s`
    /// presses Convert to Custom s seconds after it opens.
    fileprivate func autoMerit(_ d: UserDefaults) {
        let editAfter = d.double(forKey: "MKLightAutoEdit"), convertAfter = d.double(forKey: "MKLightAutoConvert")
        guard editAfter > 0 || convertAfter > 0 else { return }
        Task {
            let start = ContinuousClock.now
            if editAfter > 0 {
                try? await Task.sleep(for: .seconds(editAfter))
                for k in 0..<2 {
                    session.editMerit(design, "Auto Edit", undoManager: undoManager) { m in
                        if m.mode == .default && m.defaults.criterion == .mtf {
                            m.defaults.mtfTarget += k == 0 ? 0.1 : -0.1
                        } else if m.mode == .default {
                            m.defaults.rings += k == 0 ? 1 : -1
                        } else if !m.operands.isEmpty {
                            m.operands[0].weight *= k == 0 ? 2 : 0.5
                        }
                    }
                    try? await Task.sleep(for: .seconds(1))
                }
            }
            if convertAfter > 0 {
                try? await Task.sleep(until: start + .seconds(convertAfter))
                await session.convertToCustom(in: workspace, document: design, variables: design.variables, undoManager: undoManager)
            }
        }
    }
}
#endif

// MARK: - Apply solves

struct SolvesPanel: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    let session: SolvesSession
    @Namespace private var glass
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        let n = design.system.solves.count
        JobPanelFrame(kind: .solves, subtitle: "\(n) solve\(n == 1 ? "" : "s") in \(design.system.name)", window: window) {
            if design.system.solves.isEmpty {
                ContentUnavailableView(
                    "No Solves", systemImage: "function",
                    description: Text("Add solves in the surface inspector: marginal or chief ray height on a thickness, angle or F-number on a curvature, pickups."))
            } else {
                switch session.state {
                case .setup, .running:
                    ProgressView("Solving…").frame(maxWidth: .infinity).padding(Space.xxl)
                case .failed(let why):
                    ContentUnavailableView("Solving Failed", systemImage: "exclamationmark.triangle", description: Text(why))
                case .finished:
                    report
                }
            }
        } actions: {
            if session.committed {
                Text("Applied. Undo with ⌘Z.").font(.callout).foregroundStyle(.secondary).frame(maxWidth: .infinity)
                GlassAction(title: "Done", prominent: true, id: "primary", namespace: glass) { window.activeJob = nil }
                    .keyboardShortcut(.defaultAction)
            } else {
                GlassAction(title: "Apply to Design", prominent: true, id: "primary", namespace: glass) {
                    Task { await session.commit(in: workspace, document: design, undoManager: undoManager) }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(session.state != .finished || session.changes == 0)
            }
        }
        .task(id: design.system) {
            if !design.system.solves.isEmpty && !session.committed { await session.dryRun(in: workspace) }
        }
        .onAppear {
            // A panel reopened after a commit starts over with a new dry run.
            if session.committed && !design.system.solves.isEmpty { Task { await session.dryRun(in: workspace) } }
        }
    }

    private var report: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            Text(session.committed ? "The design now holds these solved values." : "Dry run: the values the solves would set. Nothing has changed yet.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(session.outcomes) { o in
                VStack(alignment: .leading, spacing: 1) {
                    HStack {
                        Text(o.target).font(.callout)
                        Spacer()
                        Text(o.kind).font(.caption).foregroundStyle(.secondary)
                    }
                    HStack(spacing: Space.xs) {
                        Text(o.previous.map(Fmt.short) ?? "—")
                        Image(systemName: "arrow.right").font(.caption2).accessibilityLabel("to")
                        Text(Fmt.short(o.value)).fontWeight(o.previous != o.value ? .semibold : .regular)
                        Spacer()
                        Text("residual \(Fmt.short(o.residual))").font(.caption).foregroundStyle(.secondary)
                    }
                    .font(.callout.monospacedDigit())
                }
                .accessibilityElement(children: .combine)
            }
            .textSelection(.enabled)
            ForEach(session.warnings, id: \.self) { w in
                Label(w, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
            }
        }
    }
}
