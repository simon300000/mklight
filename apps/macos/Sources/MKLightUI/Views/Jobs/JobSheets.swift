import Charts
import MKLightEngine
import MKLightModel
import SwiftUI

/// Common frame of the design-job sheets: title, content, buttons.
struct JobSheetFrame<Content: View, Buttons: View>: View {
    let title: String
    let subtitle: String
    let symbol: String
    @ViewBuilder var content: Content
    @ViewBuilder var buttons: Buttons

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: Space.m) {
                Image(systemName: symbol).font(.title2).foregroundStyle(.tint).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.title3.weight(.semibold))
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(Space.xl)
            Divider()
            content.frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            HStack(spacing: Space.m) { buttons }
                .padding(Space.l)
        }
        .frame(minWidth: 640, idealWidth: 700, minHeight: 600, idealHeight: 680)
    }
}

// MARK: - Optimize

struct OptimizeSheet: View {
    let design: DesignDocument
    let workspace: Workspace
    @State private var session = OptimizeSession()
    @Environment(\.dismiss) private var dismiss
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        JobSheetFrame(title: "Optimize", subtitle: design.system.name, symbol: "wand.and.rays") {
            switch session.state {
            case .setup: setup
            case .running(let p): running(p)
            case .finished: result
            case .failed(let why):
                ContentUnavailableView {
                    Label("Optimization Failed", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(why)
                } actions: {
                    Button("Back to Settings") { session.reset() }
                }
            }
        } buttons: {
            buttons
        }
    }

    private var stochastic: Bool { ["de", "cma_es", "hammer"].contains(session.algorithm) }

    private var setup: some View {
        Form {
            Section("Merit Function") {
                Picker("Criterion", selection: $session.criterion) {
                    ForEach(OptimizeSession.criteria, id: \.id) { Text($0.title).tag($0.id) }
                }
                Stepper("Pupil rings: \(session.rings)", value: $session.rings, in: 1...12)
                Stepper("Pupil arms: \(session.arms)", value: $session.arms, in: 3...24)
                Toggle("Hold the focal length", isOn: $session.holdEFL)
                Text("The default merit: \(session.criterion == "rms_spot" ? "RMS spot radius" : "RMS wavefront") over every field and wavelength by their weights, with glass and air thickness boundaries.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Variables") {
                Picker("Vary", selection: $session.useDesignVariables) {
                    Text("Design variables (\(design.variables.count))").tag(true)
                    Text("Every curvature and the back focus").tag(false)
                }
                .pickerStyle(.radioGroupOrInline)
                .disabled(design.variables.isEmpty)
                if session.useDesignVariables && !design.variables.isEmpty {
                    Text(design.variables.map { $0.param.label(in: design.system) }.joined(separator: ", "))
                        .font(.caption).foregroundStyle(.secondary)
                } else if design.variables.isEmpty {
                    Text("Mark values with V in the lens data or the inspector to choose them.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("Algorithm") {
                Picker("Method", selection: $session.algorithm) {
                    ForEach(OptimizeSession.algorithms, id: \.id) { Text($0.title).tag($0.id) }
                }
                LabeledContent("Iterations") {
                    TextField(
                        "Iterations", value: $session.maxIterations, format: .number, prompt: Text("default")
                    )
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                }
                if stochastic {
                    LabeledContent("Seed") {
                        TextField("Seed", value: $session.seed, format: .number).labelsHidden().multilineTextAlignment(.trailing).frame(width: 90)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .onAppear {
            if design.variables.isEmpty { session.useDesignVariables = false }
            #if DEBUG
            // Screenshots and UI checks: `-MKLightAutoRun lm` starts at once.
            if let a = UserDefaults.standard.string(forKey: "MKLightAutoRun") {
                session.algorithm = a
                if a == "de" { session.maxIterations = 100_000 }
                Task {
                    try? await Task.sleep(for: .seconds(1))
                    session.start(in: workspace, variables: design.variables)
                }
            }
            #endif
        }
    }

    private func running(_ p: JobProgress) -> some View {
        VStack(alignment: .leading, spacing: Space.l) {
            HStack(alignment: .firstTextBaseline, spacing: Space.xxl) {
                BigNumber(label: "Iteration", value: session.trace.last.map { "\($0.iteration)" } ?? "—")
                BigNumber(label: "Merit", value: session.trace.last.map { Fmt.short($0.merit) } ?? "—")
                if let first = session.trace.first, let last = session.trace.last, first.merit > 0 {
                    BigNumber(label: "Improvement", value: Fmt.percent(1 - last.merit / first.merit))
                }
                Spacer()
            }
            Sparkline(values: session.trace.map(\.merit))
                .frame(height: 140)
                .padding(.vertical, Space.m)
            ProgressView(value: p.fraction) { Text(p.stage).font(.caption).foregroundStyle(.secondary) }
            Text("Stop keeps the best design found so far; Cancel discards the run.").font(.caption).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(Space.xl)
    }

    @ViewBuilder private var result: some View {
        if let r = session.result {
            ScrollView {
                VStack(alignment: .leading, spacing: Space.l) {
                    HStack(alignment: .firstTextBaseline, spacing: Space.xxl) {
                        BigNumber(label: "Merit before", value: Fmt.short(r.startMerit))
                        BigNumber(label: "Merit after", value: Fmt.short(r.finalMerit))
                        BigNumber(label: "Improvement", value: Fmt.percent(r.improvement))
                        Spacer()
                    }
                    Text(statusText(r)).font(.callout).foregroundStyle(.secondary)
                    if session.trace.count > 1 {
                        Sparkline(values: session.trace.map(\.merit)).frame(height: 70)
                    }
                    Grid(alignment: .trailing, horizontalSpacing: Space.xl, verticalSpacing: 3) {
                        GridRow {
                            Text("Parameter").gridColumnAlignment(.leading)
                            Text("Before")
                            Text("After")
                            Text("Change")
                        }
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        Divider().gridCellUnsizedAxes(.horizontal)
                        ForEach(r.variables) { v in
                            GridRow {
                                HStack(spacing: Space.xs) {
                                    Text(v.param.map { label($0) } ?? v.label)
                                    if v.atBound { Text("at bound").font(.caption2).foregroundStyle(.orange) }
                                }
                                Text(Fmt.short(displayed(v.start, v.param)))
                                Text(Fmt.short(displayed(v.value, v.param)))
                                Text(Fmt.short(displayed(v.value, v.param) - displayed(v.start, v.param))).foregroundStyle(.secondary)
                            }
                            .font(.callout.monospacedDigit())
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
                .padding(Space.xl)
            }
        }
    }

    /// Curvatures are shown as radii.
    private func displayed(_ v: Double, _ p: ParamPath?) -> Double {
        if case .curvature? = p { return RadiusDisplay.radius(fromCurvature: v) }
        return v
    }

    private func label(_ p: ParamPath) -> String { p.label(in: design.system) }

    private func statusText(_ r: OptimizeResult) -> String {
        var s = "\(r.iterations) iterations, \(r.evaluations) evaluations. "
        if r.stopped { s += "Stopped early: this is the best design found so far. " }
        if !r.improved { s += "No better design was found; the result equals the start. " }
        if !r.feasible { s += "Some constraints are not met. " }
        if session.applied { s += "Applied to the design (Undo with ⌘Z)." }
        return s
    }

    @ViewBuilder private var buttons: some View {
        switch session.state {
        case .setup, .failed:
            Spacer()
            Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
            Button("Optimize") { session.start(in: workspace, variables: design.variables) }
                .keyboardShortcut(.defaultAction)
                .disabled(workspace.snapshot == nil || !workspace.modelErrors.isEmpty)
        case .running:
            Spacer()
            Button("Cancel", role: .cancel) { session.cancel() }.keyboardShortcut(.cancelAction)
            Button("Stop") { session.stop() }
                .keyboardShortcut(.defaultAction)
                .disabled(!session.canStop)
                .help("End now and keep the best design so far")
        case .finished:
            Button("Run Again") { session.reset() }
            Spacer()
            if session.applied {
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            } else {
                Button("Discard") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Apply to Design") {
                    session.apply(to: design, undoManager: undoManager)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(session.result?.system == nil || session.result?.improved == false)
            }
        }
    }
}

struct BigNumber: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title2.weight(.medium).monospacedDigit())
        }
        .accessibilityElement(children: .combine)
    }
}

extension PickerStyle where Self == InlinePickerStyle {
    static var radioGroupOrInline: InlinePickerStyle { .inline }
}

// MARK: - Tolerance

struct ToleranceSheet: View {
    let workspace: Workspace
    @State private var session = ToleranceSession()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        JobSheetFrame(title: "Tolerance", subtitle: workspace.snapshot?.system.name ?? "", symbol: "plusminus") {
            switch session.state {
            case .setup:
                Form {
                    Section {
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
                    } footer: {
                        Text("A basic tolerance set: ±0.05 mm on every thickness except the back focus and ±0.5 % on every curvature, judged by the default RMS spot criterion.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .formStyle(.grouped)
            case .running(let p):
                VStack(spacing: Space.l) {
                    ProgressView(value: p.fraction) { Text(p.stage).font(.caption).foregroundStyle(.secondary) }
                    Text("Stop keeps the trials completed so far.").font(.caption).foregroundStyle(.secondary)
                }
                .padding(Space.xxl)
            case .finished:
                if let r = session.result { ToleranceResultView(result: r) }
            case .failed(let why):
                ContentUnavailableView("Tolerancing Failed", systemImage: "exclamationmark.triangle", description: Text(why))
            }
        } buttons: {
            Spacer()
            switch session.state {
            case .running:
                Button("Cancel", role: .cancel) { session.cancel() }.keyboardShortcut(.cancelAction)
                Button("Stop") { session.stop() }.keyboardShortcut(.defaultAction).disabled(!session.canStop)
            default:
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(session.state == .finished ? "Run Again" : "Run") { session.start(in: workspace) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(workspace.snapshot == nil || !workspace.modelErrors.isEmpty)
            }
        }
    }
}

struct ToleranceResultView: View {
    let result: ToleranceResult

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.xl) {
                if !result.sensitivities.isEmpty {
                    VStack(alignment: .leading, spacing: Space.s) {
                        Text("Sensitivity").font(.headline)
                        Text("Nominal criterion \(Fmt.short(result.nominal)) · RSS change \(Fmt.short(result.rssChange)) · predicted \(Fmt.short(result.predicted))")
                            .font(.caption).foregroundStyle(.secondary)
                        Grid(alignment: .trailing, horizontalSpacing: Space.l, verticalSpacing: 3) {
                            GridRow {
                                Text("Tolerance").gridColumnAlignment(.leading)
                                Text("−")
                                Text("+")
                                Text("Δ at −")
                                Text("Δ at +")
                                Text("")
                            }
                            .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                            let worst = result.sensitivities.map(\.worst).filter(\.isFinite).max() ?? 1
                            ForEach(result.sensitivities) { s in
                                GridRow {
                                    Text(s.label).gridColumnAlignment(.leading)
                                    Text(Fmt.short(s.minus))
                                    Text(Fmt.short(s.plus))
                                    Text(Fmt.short(s.changeMinus))
                                    Text(Fmt.short(s.changePlus))
                                    Capsule().fill(Palette.fieldColor(0)).frame(width: max(2, 80 * CGFloat(s.worst / max(worst, 1e-300))), height: 6)
                                        .frame(width: 80, alignment: .leading)
                                        .accessibilityHidden(true)
                                }
                                .font(.callout.monospacedDigit())
                            }
                        }
                        .textSelection(.enabled)
                    }
                }
                if result.hasMonteCarlo {
                    VStack(alignment: .leading, spacing: Space.s) {
                        Text("Monte Carlo · \(result.trials) trials").font(.headline)
                        HStack(alignment: .top, spacing: Space.xl) {
                            KeyValueGrid(
                                rows: [("Mean", Fmt.short(result.mean)), ("Std. dev.", Fmt.short(result.stdDev)), ("Min", Fmt.short(result.min)), ("Max", Fmt.short(result.max))]
                                    + result.percentiles.map { (String(format: "%.0f %%", $0.0), Fmt.short($0.1)) }
                                    + (result.yield.map { [("Yield", Fmt.percent($0))] } ?? []))
                            Histogram(values: result.criteria).frame(height: 180)
                        }
                    }
                }
            }
            .padding(Space.xl)
        }
    }
}

/// A histogram of Monte-Carlo criteria.
struct Histogram: View {
    let values: [Double]

    var body: some View {
        let bins = Self.bins(values, count: 20)
        Chart(bins, id: \.0) { b in
            BarMark(x: .value("Criterion", b.0), y: .value("Trials", b.1), width: .ratio(0.9))
                .foregroundStyle(Palette.fieldColor(0))
                .cornerRadius(2)
        }
        .chartXAxisLabel("Criterion", alignment: .center)
        .chartYAxisLabel("Trials")
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

struct SolvesSheet: View {
    let design: DesignDocument
    let workspace: Workspace
    @State private var session = SolvesSession()
    @Environment(\.dismiss) private var dismiss
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        JobSheetFrame(title: "Apply Solves", subtitle: "\(design.system.solves.count) solve\(design.system.solves.count == 1 ? "" : "s") in \(design.system.name)", symbol: "function") {
            if design.system.solves.isEmpty {
                ContentUnavailableView(
                    "No Solves", systemImage: "function",
                    description: Text("Add solves in the surface inspector: marginal or chief ray height on a thickness, angle or F-number on a curvature, pickups."))
            } else {
                switch session.state {
                case .setup, .running:
                    ProgressView("Solving…").padding(Space.xxl)
                case .failed(let why):
                    ContentUnavailableView("Solving Failed", systemImage: "exclamationmark.triangle", description: Text(why))
                case .finished:
                    report
                }
            }
        } buttons: {
            if session.committed {
                Text("Applied. Undo with ⌘Z.").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            } else {
                Spacer()
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Apply to Design") {
                    Task { await session.commit(in: workspace, document: design, undoManager: undoManager) }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(session.state != .finished || session.changes == 0)
            }
        }
        .task { if !design.system.solves.isEmpty { await session.dryRun(in: workspace) } }
    }

    private var report: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.l) {
                Text(session.committed ? "The design now holds these solved values." : "Dry run: the values the solves would set. Nothing has changed yet.")
                    .font(.callout).foregroundStyle(.secondary)
                Grid(alignment: .trailing, horizontalSpacing: Space.l, verticalSpacing: 3) {
                    GridRow {
                        Text("Parameter").gridColumnAlignment(.leading)
                        Text("Solve").gridColumnAlignment(.leading)
                        Text("Now")
                        Text("Solved")
                        Text("Residual")
                    }
                    .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    ForEach(session.outcomes) { o in
                        GridRow {
                            Text(o.target)
                            Text(o.kind).foregroundStyle(.secondary)
                            Text(o.previous.map(Fmt.short) ?? "—")
                            Text(Fmt.short(o.value)).fontWeight(o.previous != o.value ? .semibold : .regular)
                            Text(Fmt.short(o.residual)).foregroundStyle(.secondary)
                        }
                        .font(.callout.monospacedDigit())
                    }
                }
                .textSelection(.enabled)
                ForEach(session.warnings, id: \.self) { w in
                    Label(w, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                }
            }
            .padding(Space.xl)
        }
    }
}
