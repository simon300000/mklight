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
        case .tolerance: TolerancePanel(workspace: workspace, window: window, session: window.tolerance)
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

    var body: some View {
        Button(role: role, action: action) {
            Group {
                if let systemImage { Label(title, systemImage: systemImage) } else { Text(title) }
            }
            .font(.body.weight(prominent ? .semibold : .regular))
            .foregroundStyle(prominent ? AnyShapeStyle(.white) : role == .destructive ? AnyShapeStyle(.red) : AnyShapeStyle(.primary))
            .lineLimit(1)
            .padding(.horizontal, Space.xl)
            .padding(.vertical, Space.m)
            .frame(maxWidth: prominent ? .infinity : nil)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .glassEffect(prominent ? .regular.tint(.accentColor).interactive() : .regular.interactive(), in: .capsule)
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
                if progress.fraction > 0 {
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
        .onAppear(perform: autoRun)
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
            GroupBox("Merit Function") {
                VStack(alignment: .leading, spacing: Space.s) {
                    Picker("Criterion", selection: $session.criterion) {
                        ForEach(OptimizeSession.criteria, id: \.id) { Text($0.title).tag($0.id) }
                    }
                    Stepper("Pupil rings: \(session.rings)", value: $session.rings, in: 1...12)
                    Stepper("Pupil arms: \(session.arms)", value: $session.arms, in: 3...24)
                    Toggle("Hold the focal length", isOn: $session.holdEFL)
                    Text(
                        "The default merit: \(session.criterion == "rms_spot" ? "RMS spot radius" : "RMS wavefront") over every field and wavelength by their weights, with glass and air thickness boundaries."
                    )
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                .padding(Space.xs)
            }
            GroupBox("Variables") {
                VStack(alignment: .leading, spacing: Space.s) {
                    Picker("Vary", selection: $session.useDesignVariables) {
                        Text("Design variables (\(design.variables.count))").tag(true)
                        Text("Every curvature and the back focus").tag(false)
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                    .disabled(design.variables.isEmpty)
                    if session.useDesignVariables && !design.variables.isEmpty {
                        FlowLayout(spacing: Space.xs, lineSpacing: Space.xs) {
                            ForEach(design.variables.map { $0.param.label(in: design.system) }, id: \.self) { label in
                                Text(label).font(.caption)
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(.tint.opacity(0.12), in: Capsule())
                            }
                        }
                    } else if design.variables.isEmpty {
                        Text("Mark values with V in the lens data or the inspector to choose them.")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(Space.xs)
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
                BigNumber(label: "Merit", value: session.trace.last.map { Fmt.short($0.merit) } ?? "—")
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
        return (d >= 0 ? "+" : "") + Fmt.short(d)
    }

    /// Curvatures are shown as radii.
    private func displayed(_ v: Double, _ p: ParamPath?) -> Double {
        switch p {
        case .curvature?, .shapeParam(_, .curvatureX)?, .shapeParam(_, .curvatureY)?: RadiusDisplay.radius(fromCurvature: v)
        default: v
        }
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
            .disabled(blocked)
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
        guard let a = UserDefaults.standard.string(forKey: "MKLightAutoRun"), a != "tolerance", case .setup = session.state else { return }
        session.algorithm = a
        if a == "de" { session.maxIterations = 100_000 }
        Task {
            try? await Task.sleep(for: .seconds(1))
            session.start(in: workspace, variables: design.variables)
        }
        #endif
    }
}

// MARK: - Tolerance

struct TolerancePanel: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    @Bindable var session: ToleranceSession
    @Namespace private var glass
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        JobPanelFrame(kind: .tolerance, subtitle: workspace.snapshot?.system.name ?? "", window: window) {
            switch session.state {
            case .setup:
                setup
            case .running:
                VStack(alignment: .leading, spacing: Space.m) {
                    Text("Sensitivities first, then the Monte-Carlo trials.").font(.callout).foregroundStyle(.secondary)
                    Text("Stop keeps the trials completed so far.").font(.caption).foregroundStyle(.secondary)
                }
            case .finished:
                if let r = session.result { ToleranceResultView(result: r) }
            case .failed(let why):
                ContentUnavailableView("Tolerancing Failed", systemImage: "exclamationmark.triangle", description: Text(why))
            }
        } actions: {
            switch session.state {
            case .running(let p):
                GlassProgress(progress: p, namespace: glass)
                GlassAction(title: "Cancel", role: .destructive, id: "secondary", namespace: glass) { session.cancel() }
                GlassAction(title: "Stop", systemImage: "stop.fill", id: "primary", namespace: glass) { session.stop() }
                    .disabled(!session.canStop)
            default:
                GlassAction(
                    title: session.state == .finished ? "Run Again" : "Run", systemImage: "play.fill", prominent: true, id: "primary",
                    namespace: glass
                ) { session.start(in: workspace) }
                .keyboardShortcut(.defaultAction)
                .disabled(workspace.snapshot == nil || !workspace.modelErrors.isEmpty)
            }
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.35), value: session.isRunning)
        .onAppear(perform: autoRun)
    }

    private var setup: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Space.s) {
                Picker("Analysis", selection: $session.mode) {
                    Text("Sensitivity and Monte Carlo").tag("both")
                    Text("Sensitivity only").tag("sensitivity")
                    Text("Monte Carlo only").tag("monte_carlo")
                }
                if session.mode != "sensitivity" {
                    LabeledContent("Trials") {
                        TextField("Trials", value: $session.trials, format: .number).labelsHidden().multilineTextAlignment(.trailing).frame(width: 90)
                    }
                    LabeledContent("Seed") {
                        TextField("Seed", value: $session.seed, format: .number).labelsHidden().multilineTextAlignment(.trailing).frame(width: 90)
                    }
                }
                Text(
                    "A basic tolerance set: ±0.05 mm on every thickness except the back focus and ±0.5 % on every curvature, judged by the default RMS spot criterion."
                )
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .padding(Space.xs)
        }
    }

    /// Debug builds: `-MKLightAutoRun tolerance` starts at once.
    private func autoRun() {
        #if DEBUG
        guard UserDefaults.standard.string(forKey: "MKLightAutoRun") == "tolerance", case .setup = session.state else { return }
        Task {
            try? await Task.sleep(for: .seconds(1))
            session.start(in: workspace)
        }
        #endif
    }
}

struct ToleranceResultView: View {
    let result: ToleranceResult

    static func sci(_ v: Double) -> String { v.isFinite ? String(format: "%+.2e", v) : "—" }
    static func e(_ v: Double) -> String { v.isFinite ? String(format: "%.3e", v) : "—" }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xl) {
            if !result.sensitivities.isEmpty {
                VStack(alignment: .leading, spacing: Space.s) {
                    Text("Sensitivity").font(.headline)
                    KeyValueGrid(rows: [
                        ("Nominal", Self.e(result.nominal)), ("RSS change", Self.e(result.rssChange)), ("Predicted", Self.e(result.predicted)),
                    ])
                    let worst = result.sensitivities.map(\.worst).filter(\.isFinite).max() ?? 1
                    ForEach(result.sensitivities) { s in
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text(s.label.prefix(1).uppercased() + s.label.dropFirst()).font(.callout).lineLimit(1)
                                Spacer()
                                Text("±\(Fmt.short(max(abs(s.minus), abs(s.plus))))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                            GeometryReader { g in
                                Capsule().fill(Palette.fieldColor(0))
                                    .frame(width: max(3, g.size.width * CGFloat(s.worst / max(worst, 1e-300))), height: 5)
                            }
                            .frame(height: 5)
                            .accessibilityHidden(true)
                            Text("Δ \(Self.sci(s.changeMinus)) / \(Self.sci(s.changePlus))")
                                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        }
                        .accessibilityElement(children: .combine)
                    }
                }
                .textSelection(.enabled)
            }
            if result.hasMonteCarlo {
                VStack(alignment: .leading, spacing: Space.s) {
                    Text("Monte Carlo · \(result.trials) trials").font(.headline)
                    Histogram(values: result.criteria).frame(height: 150)
                    KeyValueGrid(
                        rows: [("Mean", Self.e(result.mean)), ("Std. dev.", Self.e(result.stdDev)), ("Min", Self.e(result.min)), ("Max", Self.e(result.max))]
                            + result.percentiles.map { (String(format: "%.0f %% below", $0.0), Self.e($0.1)) }
                            + (result.yield.map { [("Yield", Fmt.percent($0))] } ?? []))
                }
            }
        }
    }
}

/// A histogram of Monte-Carlo criteria.
struct Histogram: View {
    let values: [Double]

    var body: some View {
        let bins = Self.bins(values, count: 20)
        let w = bins.count > 1 ? (bins[1].0 - bins[0].0) : 1
        Chart(bins, id: \.0) { b in
            RectangleMark(
                xStart: .value("Criterion", b.0 - 0.45 * w), xEnd: .value("Criterion", b.0 + 0.45 * w),
                yStart: .value("Trials", 0), yEnd: .value("Trials", b.1)
            )
            .foregroundStyle(Palette.fieldColor(0))
            .cornerRadius(2)
        }
        .environment(\.locale, Locale(identifier: "en_US_POSIX"))
        .chartXAxisLabel("Criterion", alignment: .center)
        .chartYAxisLabel("Trials")
        .font(.caption2)
        .accessibilityLabel("Histogram of \(values.count) trial criteria")
    }

    static func bins(_ v: [Double], count: Int) -> [(Double, Int)] {
        guard let lo = v.min(), let hi = v.max(), hi > lo else { return v.isEmpty ? [] : [(v[0], v.count)] }
        let w = (hi - lo) / Double(count)
        var c = Array(repeating: 0, count: count)
        for x in v { c[min(Int((x - lo) / w), count - 1)] += 1 }
        return c.enumerated().map { (lo + (Double($0.offset) + 0.5) * w, $0.element) }
    }
}

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
