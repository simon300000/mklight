import Charts
import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Panel

/// The Tolerance panel: the tolerance set (generator, table grouped by
/// element), compensators, criterion and analysis; the run with live
/// progress; and the result (sensitivities, Monte-Carlo distribution,
/// compensator statistics, export, previews of toleranced designs).
struct TolerancePanel: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    @Bindable var session: ToleranceSession
    @Namespace private var glass
    @Environment(\.undoManager) private var undoManager
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The design with the clear apertures the layout resolved (fringes
    /// are counted over them), the Abbe numbers of its catalogue glasses
    /// and the set's test wavelength.
    private var context: ToleranceContext {
        session.settings.context(
            ToleranceContext(
                system: design.system, semiDiameters: session.semiDiameters(from: workspace.layout.output),
                abbeNumbers: workspace.abbeNumbers(of: design.system)))
    }

    private var subtitle: String {
        let t = session.settings
        let n = t.rows.count, c = t.compensators.count
        let comp = c == 0 ? "no compensator" : c == 1 ? t.compensators[0].title(in: design.system).lowercased() : "\(c) compensators"
        return "\(design.system.name) · \(n) tolerance\(n == 1 ? "" : "s") · \(comp)"
    }

    var body: some View {
        JobPanelFrame(kind: .tolerance, subtitle: subtitle, window: window) {
            switch session.state {
            case .setup:
                ToleranceEditor(design: design, workspace: workspace, session: session, context: context, edit: edit)
            case .running(let p):
                ToleranceProgress(session: session, progress: p)
            case .finished:
                if let r = session.result {
                    ToleranceResultView(design: design, workspace: workspace, window: window, session: session, result: r)
                }
            case .failed(let why):
                VStack(alignment: .leading, spacing: Space.l) {
                    Label {
                        Text("Tolerancing failed: \(why)").fixedSize(horizontal: false, vertical: true)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    }
                    .font(.callout)
                    ToleranceEditor(design: design, workspace: workspace, session: session, context: context, edit: edit)
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
        .onChange(of: design.project.sections["tolerances"]) { _, _ in session.load(design) }
        .onChange(of: phase) { _, p in if p != 2 { session.endPreview(in: workspace) } }
        .onDisappear { if !session.isRunning, session.previewed != nil { session.endPreview(in: workspace) } }
    }

    private var phase: Int {
        switch session.state {
        case .setup, .failed: 0
        case .running: 1
        case .finished: 2
        }
    }

    private var problem: String? {
        if workspace.snapshot == nil { return "The design has not been checked by the engine yet." }
        if !workspace.modelErrors.isEmpty { return "Fix the design's errors first." }
        return session.runProblem(context: context)
    }

    private func edit(_ name: String, _ change: (inout ToleranceSettings) -> Void) {
        session.edit(design, name, context: context, undoManager: undoManager, change)
    }

    private func run() {
        session.load(design)
        session.start(in: workspace, context: context, merit: design.merit)
    }

    @ViewBuilder private var actions: some View {
        switch session.state {
        case .setup, .failed:
            if session.result != nil {
                GlassAction(title: "Show Results", id: "secondary", namespace: glass) { session.showResult() }
                    .help("Show the last run's results")
            }
            GlassAction(title: "Run", systemImage: "play.fill", prominent: true, id: "primary", namespace: glass, action: run)
                .keyboardShortcut(.defaultAction)
                .disabled(problem != nil)
                .help(problem ?? "Run the sensitivity analysis and the Monte-Carlo trials")
        case .running(let p):
            GlassProgress(
                progress: p,
                detail: ToleranceProgress.stageText(p.stage, trialsDone: session.timing(at: Date(), progress: p).trialsDone, of: session.settings.trials),
                namespace: glass)
            GlassAction(title: "Cancel", role: .destructive, id: "secondary", namespace: glass) { session.cancel() }
                .help("End the run and discard it")
            GlassAction(title: "Stop", systemImage: "stop.fill", id: "primary", namespace: glass) { session.stop() }
                .disabled(!session.canStop)
                .help("End now and keep the sensitivities and trials completed so far")
        case .finished:
            GlassAction(title: "Edit", id: "secondary", namespace: glass) { session.edit() }
                .help("Back to the tolerances, compensators and criterion")
            GlassAction(title: "Run Again", systemImage: "play.fill", prominent: true, id: "primary", namespace: glass, action: run)
                .keyboardShortcut(.defaultAction)
                .disabled(problem != nil)
        }
    }

    /// Debug builds: `-MKLightAutoRun tolerance` starts at once;
    /// `-MKLightAutoStop s` presses Stop after s seconds;
    /// `-MKLightTolerancePreview k` previews the k-th largest sensitivity,
    /// `trial` the worst trial (and selects its histogram bar).
    private func autoRun() {
        #if DEBUG
        let d = UserDefaults.standard
        guard d.string(forKey: "MKLightAutoRun") == "tolerance", case .setup = session.state else { return }
        if let t = d.string(forKey: "MKLightToleranceTrials").flatMap(Int.init) {
            edit("Trials") { $0.trials = t }
        }
        let stopAfter = d.double(forKey: "MKLightAutoStop")
        Task {
            // The layout's clear apertures and the glasses' Abbe numbers first.
            try? await Task.sleep(for: .seconds(1.5))
            for _ in 0..<50 where problem != nil { try? await Task.sleep(for: .milliseconds(100)) }
            run()
            if stopAfter > 0 {
                try? await Task.sleep(for: .seconds(stopAfter))
                session.stop()
            }
            if let p = d.string(forKey: "MKLightTolerancePreview") {
                while session.isRunning || session.result == nil { try? await Task.sleep(for: .milliseconds(100)) }
                try? await Task.sleep(for: .seconds(0.5))
                if p == "trial", let k = session.result?.trialsWorstFirst.first {
                    session.preview(.trial(k), in: workspace)
                } else if let k = Int(p), let s = session.result?.sensitivities.dropFirst(k).first {
                    session.preview(.sensitivity(operand: s.id, plus: s.worsePlus), in: workspace)
                }
            }
        }
        #endif
    }
}

// MARK: - Editor

/// The tolerance set and its run settings.
struct ToleranceEditor: View {
    let design: DesignDocument
    let workspace: Workspace
    let session: ToleranceSession
    let context: ToleranceContext
    let edit: (String, (inout ToleranceSettings) -> Void) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            if session.result != nil, case .setup = session.state {
                Button {
                    session.showResult()
                } label: {
                    Label("Show the last results", systemImage: "chart.bar.doc.horizontal")
                }
                .buttonStyle(.link)
                .font(.callout)
            }
            ToleranceTableSection(design: design, session: session, context: context, edit: edit)
            CompensatorSection(design: design, session: session, edit: edit)
            CriterionSection(design: design, session: session, edit: edit)
            AnalysisSection(session: session, edit: edit)
        }
    }
}

/// The tolerance table: the default-tolerance generator, add, duplicate
/// and delete, and the rows grouped by element.
struct ToleranceTableSection: View {
    let design: DesignDocument
    let session: ToleranceSession
    let context: ToleranceContext
    let edit: (String, (inout ToleranceSettings) -> Void) -> Void
    @State private var selection: Set<UUID> = []
    @State private var showGenerator = false
    @State private var collapsed: Set<String> = []

    private var system: OpticalSystem { design.system }
    private var rows: [ToleranceRow] { session.settings.rows }

    var body: some View {
        let issues = session.settings.issues(in: context)
        GroupBox {
            VStack(alignment: .leading, spacing: Space.s) {
                toolbar
                TestWavelengthField(wavelength: session.settings.testWavelengthUm) { v in
                    edit("Change Test Wavelength") { $0.testWavelengthUm = v }
                }
                if rows.isEmpty {
                    Text("No tolerances. Use Default Tolerances, or add them one by one.")
                        .font(.callout).foregroundStyle(.secondary).padding(.vertical, Space.m)
                } else {
                    ToleranceColumnsHeader()
                    let leading = leadingElementRows
                    ForEach(groups, id: \.key) { g in
                        groupHeader(g)
                        ForEach(collapsed.contains(g.key) ? [] : g.rows) { r in
                            ToleranceRowView(
                                row: r, system: system, context: context, issue: issues[r.id], showsElementLine: leading.contains(r.id),
                                selected: selection.contains(r.id),
                                select: { additive in select(r.id, additive: additive) }, edit: edit,
                                duplicate: { duplicate([r.id]) }, delete: { delete([r.id]) })
                        }
                    }
                }
            }
            .padding(Space.xs)
        } label: {
            HStack {
                Text("Tolerances")
                Spacer()
                Text("\(rows.count)").foregroundStyle(.secondary).monospacedDigit()
            }
        }
        .onDeleteCommand { delete(selection) }
        .onChange(of: rows.map(\.id)) { _, ids in selection.formIntersection(ids) }
    }

    private var toolbar: some View {
        HStack(spacing: Space.s) {
            Button {
                showGenerator = true
            } label: {
                Label("Default Tolerances…", systemImage: "wand.and.stars")
            }
            .help("Generate a tolerance set for every element from a class of manufacture")
            .popover(isPresented: $showGenerator, arrowEdge: .leading) {
                ToleranceGeneratorView(design: design, session: session, edit: edit) { showGenerator = false }
                    .padding(Space.l)
                    .frame(width: 330)
            }
            Spacer(minLength: 0)
            Menu {
                Section("Surface") {
                    ForEach(ToleranceKind.addable) { k in
                        Menu(k.title) {
                            ForEach(surfaces(for: k), id: \.self) { i in
                                Button(surfaceTitle(i)) { add(k, at: i) }
                            }
                        }
                    }
                }
                Section("Element") {
                    ForEach(ToleranceKind.addableElement) { k in
                        Menu(k.title) {
                            ForEach(LensElements.groups(system)) { g in
                                Button(elementTitle(g)) { addElement(k, g) }
                            }
                        }
                    }
                }
            } label: {
                Label("Add", systemImage: "plus")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Add a tolerance on a surface")
            Button {
                duplicate(selection)
            } label: {
                Label("Duplicate", systemImage: "plus.square.on.square")
            }
            .labelStyle(.iconOnly)
            .disabled(selection.isEmpty)
            .help("Duplicate the selected tolerances (⌘D)")
            .keyboardShortcut("d", modifiers: .command)
            Button {
                delete(selection)
            } label: {
                Label("Delete", systemImage: "minus")
            }
            .labelStyle(.iconOnly)
            .disabled(selection.isEmpty)
            .help("Delete the selected tolerances (⌫)")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
    }

    /// An element's title (with its glasses), a disclosure chevron, and
    /// when collapsed its count and the problems in it.
    private func groupHeader(_ g: (key: String, title: String, rows: [ToleranceRow])) -> some View {
        let closed = collapsed.contains(g.key)
        return Button {
            if closed { collapsed.remove(g.key) } else { collapsed.insert(g.key) }
        } label: {
            HStack(spacing: Space.xs) {
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.semibold))
                    .rotationEffect(.degrees(closed ? 0 : 90))
                    .frame(width: 10)
                Text(g.title).font(.caption.weight(.semibold))
                Spacer(minLength: Space.xs)
                if closed { Text("\(g.rows.count)").font(.caption.monospacedDigit()) }
            }
            .foregroundStyle(.secondary)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.top, Space.xs)
        .accessibilityAddTraits(.isHeader)
        .accessibilityLabel(g.title)
        .accessibilityValue(closed ? "Collapsed, \(g.rows.count) tolerances" : "Expanded")
        .help(closed ? "Show the \(g.rows.count) tolerances of \(g.key)" : "Hide the tolerances of \(g.key)")
    }

    /// The first element row of each group (rows starting on one surface
    /// share range and pivot, so one line shows them).
    private var leadingElementRows: Set<UUID> {
        var seen = Set<SurfaceID>(), out = Set<UUID>()
        for r in rows where r.kind.isElement && seen.insert(r.surface).inserted { out.insert(r.id) }
        return out
    }

    /// Rows by element (`L1`, `Air after L1`, …) in surface order.
    private var groups: [(key: String, title: String, rows: [ToleranceRow])] {
        let lens = LensElements.groups(system)
        var order: [String] = []
        var byTitle: [String: [ToleranceRow]] = [:]
        var key: [String: (Int, Int)] = [:]
        for r in rows {
            let t = LensElements.groupTitle(of: r, in: system, groups: lens)
            if byTitle[t] == nil { order.append(t) }
            byTitle[t, default: []].append(r)
            let i = system.surfaceIndex(of: r.surface) ?? .max
            let k = (i, t.hasPrefix("Air") ? 1 : 0)
            key[t] = key[t].map { $0 < k ? $0 : k } ?? k
        }
        order.sort { (key[$0] ?? (0, 0)) < (key[$1] ?? (0, 0)) }
        return order.map { t in
            let lensTitle = lens.first { $0.title == t }
            let glasses = lensTitle.map { $0.glasses.isEmpty ? "" : " · " + $0.glasses.joined(separator: ", ") } ?? ""
            return (t, t + glasses, byTitle[t] ?? [])
        }
    }

    private func surfaces(for k: ToleranceKind) -> [Int] {
        guard system.imageIndex > 1 else { return [] }
        let all = Array(1..<system.imageIndex).filter { !system.surfaces[$0].isCoordinateBreak }
        switch k {
        case .index, .abbe: return all.filter { ToleranceRow.hasGlass(system.surfaces[$0].material) }
        case .radius, .irregularityAstigmatism, .irregularitySpherical:
            return all.filter { system.value(at: .curvature(surface: system.surfaces[$0].id)) != nil }
        default: return all
        }
    }

    /// `L2 · S3–5 · N-SK16, F2`.
    private func elementTitle(_ g: LensElements.Group) -> String {
        let range = g.surfaces.count > 1 ? "S\(g.surfaces[0])–\(g.surfaces[g.surfaces.count - 1])" : "S\(g.surfaces[0])"
        return ([g.title, range] + (g.glasses.isEmpty ? [] : [g.glasses.joined(separator: ", ")])).joined(separator: " · ")
    }

    private func addElement(_ k: ToleranceKind, _ g: LensElements.Group) {
        guard let f = g.surfaces.first, let l = g.surfaces.last else { return }
        let row = ToleranceRow(
            k, surface: system.surfaces[f].id, minus: value(for: k), last: system.surfaces[l].id, distribution: session.settings.generator.distribution)
        let after = rows.last { (system.surfaceIndex(of: $0.surface) ?? 0) <= l }?.id
        edit("Add Tolerance") { $0.add([row], after: after) }
        selection = [row.id]
    }

    private func surfaceTitle(_ i: Int) -> String {
        let s = system.surfaces[i]
        var t = "S\(i)"
        if i == system.stop { t += " (stop)" }
        if !s.label.isEmpty { t += " " + s.label }
        if LensElements.isSolid(system, i) { t += " · " + s.material.displayName }
        return t
    }

    private func value(for k: ToleranceKind) -> Double {
        let g = session.settings.generator
        // A kind the class leaves out (0) is added with the precision class's value.
        let p = ToleranceGenerator(preset: .precision)
        func or(_ v: Double, _ d: Double) -> Double { v > 0 ? v : d }
        switch k {
        case .radius: return or(g.radiusFringes, p.radiusFringes)
        case .irregularityAstigmatism, .irregularitySpherical: return or(g.irregularityFringes, p.irregularityFringes)
        case .thickness: return or(g.thicknessMM, p.thicknessMM)
        case .decenterX, .decenterY: return or(g.decentreMM, p.decentreMM)
        case .tiltX, .tiltY: return or(g.tiltArcmin, p.tiltArcmin)
        case .elementDecenterX, .elementDecenterY: return or(g.elementDecentreMM, p.elementDecentreMM)
        case .elementTiltX, .elementTiltY: return or(g.elementTiltArcmin, p.elementTiltArcmin)
        case .index: return or(g.index, p.index)
        case .abbe: return or(g.abbePercent, p.abbePercent)
        case .other: return 0
        }
    }

    private func add(_ k: ToleranceKind, at i: Int) {
        let row = ToleranceRow(k, surface: system.surfaces[i].id, minus: value(for: k), distribution: session.settings.generator.distribution)
        // After the last row on this surface or before it.
        let after = rows.last { (system.surfaceIndex(of: $0.surface) ?? 0) <= i }?.id
        edit("Add Tolerance") { $0.add([row], after: after) }
        selection = [row.id]
    }

    private func duplicate(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        var new: [UUID] = []
        edit(ids.count == 1 ? "Duplicate Tolerance" : "Duplicate Tolerances") { new = $0.duplicate(ids) }
        selection = Set(new)
    }

    private func delete(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        edit(ids.count == 1 ? "Delete Tolerance" : "Delete Tolerances") { $0.delete(ids) }
        selection.subtract(ids)
    }

    private func select(_ id: UUID, additive: Bool) {
        if additive {
            if selection.contains(id) { selection.remove(id) } else { selection.insert(id) }
        } else {
            selection = [id]
        }
    }
}

/// Column titles of the tolerance table.
private struct ToleranceColumnsHeader: View {
    var body: some View {
        HStack(spacing: Space.s) {
            Text("Kind").frame(maxWidth: .infinity, alignment: .leading)
            Text("Surface").frame(width: ToleranceRowView.surfaceWidth, alignment: .leading)
            Text("− / + limits").frame(width: ToleranceRowView.limitsWidth, alignment: .trailing)
            Text("Unit").frame(width: ToleranceRowView.unitWidth, alignment: .leading)
        }
        .font(.caption2)
        .foregroundStyle(.tertiary)
        .accessibilityHidden(true)
    }
}

/// One tolerance: kind, surface, limits (one ± field when symmetric, −
/// and + otherwise) and unit, with its problem inline. Click selects,
/// ⌘-click adds to the selection; the context menu has the rest.
struct ToleranceRowView: View {
    let row: ToleranceRow
    let system: OpticalSystem
    let context: ToleranceContext
    let issue: ToleranceIssue?
    /// Whether this row shows its element's moving surfaces and pivot (the
    /// first element row of a group: the group shares them).
    var showsElementLine = true
    let selected: Bool
    let select: (Bool) -> Void
    let edit: (String, (inout ToleranceSettings) -> Void) -> Void
    let duplicate: () -> Void
    let delete: () -> Void

    static let surfaceWidth: CGFloat = 46
    static let limitsWidth: CGFloat = 124
    static let unitWidth: CGFloat = 26

    /// Limits as typed: `0.0005`, not `5.000e-04`.
    static let limitFormat: (Double) -> String = { v in
        guard v.isFinite else { return Fmt.short(v) }
        return abs(v) >= 1e-6 && abs(v) < 1e6 ? String(format: "%.6g", v) : Fmt.short(v)
    }

    private var symmetric: Bool { row.minus == row.plus }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: Space.s) {
                HStack(spacing: Space.xs) {
                    Image(systemName: row.kind.symbol).font(.caption).foregroundStyle(.secondary).frame(width: 14)
                    Text(row.kind == .other ? row.path(in: system).label(in: system) : row.kind.shortTitle)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(row.label(in: system))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .onTapGesture { select(NSEventModifier.commandOrShift) }
                surfaceMenu.frame(width: Self.surfaceWidth, alignment: .leading)
                limits.frame(width: Self.limitsWidth, alignment: .trailing)
                unitMenu.frame(width: Self.unitWidth, alignment: .leading)
            }
            if row.kind.isElement && showsElementLine { elementLine }
            if let issue {
                Text(issue.text)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 18)
            }
        }
        .font(.callout)
        .padding(.vertical, 1)
        .padding(.horizontal, Space.xs)
        .background(selected ? AnyShapeStyle(.tint.opacity(0.16)) : AnyShapeStyle(.clear), in: .rect(cornerRadius: 5))
        .contextMenu { menu }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(row.label(in: system)), \(row.limitsText)")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }

    @ViewBuilder private var limits: some View {
        HStack(spacing: 3) {
            if symmetric {
                Text("±").foregroundStyle(.secondary)
                OptionalNumberField(title: "Limit", value: row.minus, prompt: "0", width: 64, format: ToleranceRowView.limitFormat, compact: true) { v in
                    set("Change Tolerance") { $0.minus = v ?? $0.minus; $0.plus = v ?? $0.plus }
                }
            } else {
                Text("−").foregroundStyle(.secondary)
                OptionalNumberField(title: "Minus limit", value: row.minus, prompt: "0", width: 50, format: ToleranceRowView.limitFormat, compact: true) { v in
                    set("Change Tolerance") { $0.minus = v ?? $0.minus }
                }
                Text("+").foregroundStyle(.secondary)
                OptionalNumberField(title: "Plus limit", value: row.plus, prompt: "0", width: 50, format: ToleranceRowView.limitFormat, compact: true) { v in
                    set("Change Tolerance") { $0.plus = v ?? $0.plus }
                }
            }
        }
        .controlSize(.small)
    }

    private var surfaceMenu: some View {
        Menu {
            if row.kind.isElement {
                Section("Element") {
                    ForEach(LensElements.groups(system)) { g in
                        if let f = g.surfaces.first, let l = g.surfaces.last {
                            Button("\(g.title) · S\(f)–\(l)") { setElement(first: f, last: l, pivot: nil) }
                        }
                    }
                }
                let moving = row.movingSurfaces(in: system)
                Menu("First Surface") {
                    ForEach(Array(1..<max(system.imageIndex, 2)), id: \.self) { i in
                        Button("S\(i)") { setElement(first: i, last: max(i, moving.last ?? i), pivot: pivotChoice) }
                            .disabled(i == moving.first)
                    }
                }
                Menu("Last Surface") {
                    ForEach(Array((moving.first ?? 1)..<max(system.imageIndex, 2)), id: \.self) { i in
                        Button("S\(i)") { setElement(first: moving.first ?? i, last: i, pivot: pivotChoice) }
                            .disabled(i == moving.last)
                    }
                }
            } else {
                ForEach(Array(1..<max(system.imageIndex, 2)), id: \.self) { i in
                    Button("S\(i)" + (i == system.stop ? " (stop)" : "")) {
                        set("Change Tolerance Surface") { $0.surface = system.surfaces[i].id }
                    }
                }
            }
        } label: {
            Text(row.kind.isElement ? (row.elementTitle(in: system) ?? row.surfaceLabel(in: system)) : row.surfaceLabel(in: system))
                .monospacedDigit()
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(row.kind == .other)
        .help(row.kind.isElement ? "The element: the surfaces that move together" : "The surface (a thickness is the gap after it)")
    }

    // MARK: Elements

    /// Where the pivot sits relative to the range, kept when the range moves.
    private enum PivotChoice { case first, last, surface(Int) }

    private var pivotChoice: PivotChoice {
        let moving = row.movingSurfaces(in: system)
        guard let p = row.pivotIndex(in: system), p != moving.first else { return .first }
        return p == moving.last ? .last : .surface(p)
    }

    /// "S3 S4 S5 move together · pivot S3, first vertex", the pivot a menu.
    private var elementLine: some View {
        let moving = row.movingSurfaces(in: system)
        let pivot = row.pivotIndex(in: system)
        return HStack(spacing: Space.xs) {
            Image(systemName: "link").font(.caption2).foregroundStyle(.tertiary)
            Text(moving.map { "S\($0)" }.joined(separator: " ") + (moving.count > 1 ? " move together" : " moves"))
                .lineLimit(1)
            Text("·").foregroundStyle(.tertiary)
            Menu {
                if let f = moving.first { Button("First Vertex (S\(f))") { setPivot(.first) } }
                if let l = moving.last, l != moving.first { Button("Last Vertex (S\(l))") { setPivot(.last) } }
                Menu("Surface") {
                    ForEach(Array(1..<max(system.imageIndex, 2)), id: \.self) { i in
                        Button("S\(i)" + (i == system.stop ? " (stop)" : "")) { setPivot(.surface(i)) }
                    }
                }
            } label: {
                Text("pivot " + pivotText(pivot, moving: moving)).font(.caption2).foregroundStyle(.tint)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("The vertex the element tilts about; every element tolerance starting on S\(moving.first ?? 0) uses it")
            Spacer(minLength: 0)
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .padding(.leading, 18)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Moves \(moving.map { "surface \($0)" }.joined(separator: ", ")), pivot \(pivotText(pivot, moving: moving))")
    }

    private func pivotText(_ p: Int?, moving: [Int]) -> String {
        guard let p else { return "?" }
        if p == moving.first { return "S\(p), first vertex" }
        if p == moving.last { return "S\(p), last vertex" }
        return "S\(p)"
    }

    private func setPivot(_ c: PivotChoice) {
        let moving = row.movingSurfaces(in: system)
        guard let f = moving.first, let l = moving.last else { return }
        setElement(first: f, last: l, pivot: c, name: "Change Element Pivot")
    }

    private func setElement(first f: Int, last l: Int, pivot c: PivotChoice?, name: String = "Change Element") {
        guard system.surfaces.indices.contains(f), system.surfaces.indices.contains(l) else { return }
        let p: SurfaceID? =
            switch c ?? .first {
            case .first: nil
            case .last: system.surfaces[l].id
            case .surface(let i): system.surfaces.indices.contains(i) ? system.surfaces[i].id : nil
            }
        let id = row.id
        edit(name) { $0.setElement(id, first: system.surfaces[f].id, last: system.surfaces[l].id, pivot: p) }
    }

    @ViewBuilder private var unitMenu: some View {
        if row.kind.units.count < 2 {
            Text(row.unit.label.isEmpty ? row.path(in: system).unit(in: system) : row.unit.label)
                .foregroundStyle(.secondary)
                .help(row.unit.title)
        } else {
            unitChoice
        }
    }

    private var unitChoice: some View {
        Menu {
            ForEach(row.kind.units, id: \.self) { u in
                Button {
                    convert(to: u)
                } label: {
                    if u == row.unit { Label(u.title, systemImage: "checkmark") } else { Text(u.title) }
                }
                .disabled(row.converted(to: u, in: context) == nil)
            }
        } label: {
            Text(row.unit.label.isEmpty ? row.path(in: system).unit(in: system) : row.unit.label).foregroundStyle(.secondary)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Unit: " + row.unit.title + ". Changing it converts the limits.")
    }

    @ViewBuilder private var menu: some View {
        if row.kind.units.count > 1 {
            Menu("Unit") {
                ForEach(row.kind.units, id: \.self) { u in
                    Button(u.title) { convert(to: u) }.disabled(row.converted(to: u, in: context) == nil)
                }
            }
        }
        Menu("Distribution") {
            ForEach(ToleranceDistribution.choices, id: \.self) { d in
                Button {
                    set("Change Distribution") { $0.distribution = d }
                } label: {
                    if d == row.distribution { Label(d.title, systemImage: "checkmark") } else { Text(d.title) }
                }
            }
        }
        Button(symmetric ? "Make Asymmetric" : "Make Symmetric") {
            set(symmetric ? "Make Tolerance Asymmetric" : "Make Tolerance Symmetric") { r in
                if r.minus == r.plus { r.plus = r.minus * 2 } else { r.plus = r.minus }
            }
        }
        Divider()
        Button("Duplicate", action: duplicate)
        Button("Delete", role: .destructive, action: delete)
    }

    private func convert(to u: ToleranceUnit) {
        guard let c = row.converted(to: u, in: context) else { return }
        set("Change Tolerance Unit") { $0 = c }
    }

    private func set(_ name: String, _ change: @escaping (inout ToleranceRow) -> Void) {
        let id = row.id
        edit(name) { t in
            guard let i = t.rows.firstIndex(where: { $0.id == id }) else { return }
            change(&t.rows[i])
        }
    }
}

/// Whether the current click is additive (⌘ or ⇧).
enum NSEventModifier {
    @MainActor static var commandOrShift: Bool {
        #if os(macOS)
        let f = NSEvent.modifierFlags
        return f.contains(.command) || f.contains(.shift)
        #else
        return false
        #endif
    }
}

/// The "Default tolerances" generator: a class of manufacture, its values
/// per kind, the distribution, and Replace.
struct ToleranceGeneratorView: View {
    let design: DesignDocument
    let session: ToleranceSession
    let edit: (String, (inout ToleranceSettings) -> Void) -> Void
    let done: () -> Void

    private var g: ToleranceGenerator { session.settings.generator }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            Text("Default Tolerances").font(.headline)
            Picker("Class", selection: Binding(get: { g.preset }, set: { p in if let p { set("Choose Tolerance Class") { $0.apply(p) } } })) {
                ForEach(ToleranceGenerator.Preset.allCases) { Text($0.title).tag(Optional($0)) }
                if g.preset == nil { Text("Custom").tag(ToleranceGenerator.Preset?.none) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            let nm = Fmt.short(session.settings.testWavelengthUm * 1000)
            Grid(alignment: .leading, horizontalSpacing: Space.m, verticalSpacing: Space.xs) {
                header("Surfaces")
                row("Radius (power)", \.radiusFringes, "fringes", help: "Power error in test fringes at \(nm) nm over the clear aperture: Δc = N λ / h².")
                row("Irregularity", \.irregularityFringes, "fringes", help: "Astigmatic and spherical irregularity of every lens surface, each this many test fringes at \(nm) nm (peak to valley over the clear aperture).")
                row("Thickness", \.thicknessMM, "mm", help: "Every glass and air thickness inside the lens; the back focus is the compensator.")
                row("Decentre", \.decentreMM, "mm", help: "Every lens surface alone, in x and in y; the rest of the system stays on the axis.")
                row("Tilt (wedge)", \.tiltArcmin, "′", help: "Every lens surface alone about x and about y, in arc minutes.")
                header("Elements")
                row("Decentre", \.elementDecentreMM, "mm", help: "Every element as a rigid body (a cemented group together), in x and in y.")
                row("Tilt", \.elementTiltArcmin, "′", help: "Every element as a rigid body about its first vertex, about x and about y; the pivot can be changed per element.")
                header("Glasses")
                row("Index", \.index, "Δn_d", help: "n_d of every glass: a model glass's n_d, a catalogue glass's melt offset Δn_d.")
                row("Abbe number", \.abbePercent, "% of V_d", help: "V_d of every glass: a model glass's V_d, a catalogue glass's melt offset ΔV_d.")
            }
            Picker("Distribution", selection: Binding(get: { g.distribution }, set: { d in set("Change Distribution") { $0.distribution = d } })) {
                ForEach(ToleranceDistribution.choices, id: \.self) { Text($0.title).tag($0) }
            }
            .help("How Monte-Carlo trials draw each perturbation within its limits")
            Text("A value of 0 leaves that kind out.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("Cancel", action: done).keyboardShortcut(.cancelAction)
                Button("Replace Tolerances") {
                    edit("Default Tolerances") { $0.generate(for: design.system) }
                    done()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .help("Replace every tolerance by this class's set for each element")
            }
        }
        .controlSize(.small)
    }

    private func header(_ title: String) -> some View {
        GridRow {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.top, Space.xs)
                .gridCellColumns(3)
                .accessibilityAddTraits(.isHeader)
        }
    }

    private func row(_ title: String, _ key: WritableKeyPath<ToleranceGenerator, Double>, _ unit: String, help: String) -> some View {
        GridRow {
            Text(title).font(.callout)
            OptionalNumberField(title: title, value: g[keyPath: key], prompt: "0", width: 72) { v in
                guard let v, v >= 0 else { return }
                set("Change Default Tolerance") {
                    $0[keyPath: key] = v
                    $0.preset = nil
                }
            }
            Text(unit).font(.callout).foregroundStyle(.secondary)
        }
        .help(help)
    }

    private func set(_ name: String, _ change: @escaping (inout ToleranceGenerator) -> Void) {
        edit(name) { change(&$0.generator) }
    }
}

/// The test wavelength fringes are counted at (power and irregularity),
/// once for the whole set: "Test fringes at 632.8 nm".
struct TestWavelengthField: View {
    let wavelength: Double
    let commit: (Double) -> Void

    var body: some View {
        HStack(spacing: Space.xs) {
            Image(systemName: "water.waves").font(.caption).foregroundStyle(.secondary)
            Text("Test fringes at")
            OptionalNumberField(title: "Test wavelength", value: wavelength * 1000, prompt: "632.8", width: 58, compact: true) { v in
                guard let v, v >= 200, v <= 20_000 else { return }
                commit(v / 1000)
            }
            Text("nm").foregroundStyle(.secondary)
            if abs(wavelength - ToleranceSettings.defaultTestWavelengthUm) > 1e-9 {
                Button("He–Ne") { commit(ToleranceSettings.defaultTestWavelengthUm) }
                    .buttonStyle(.link)
                    .help("Back to 632.8 nm")
            }
            Spacer(minLength: 0)
        }
        .font(.caption)
        .controlSize(.small)
        .help("The interferometer wavelength: one test fringe is half a wave of sag at it. Power and irregularity tolerances are counted in these fringes.")
    }
}

/// The compensators: back focus by default, any parameter with a range.
struct CompensatorSection: View {
    let design: DesignDocument
    let session: ToleranceSession
    let edit: (String, (inout ToleranceSettings) -> Void) -> Void

    private var system: OpticalSystem { design.system }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Space.s) {
                if session.settings.compensators.isEmpty {
                    Text("None: every trial is judged as built, without refocusing.").font(.callout).foregroundStyle(.secondary)
                }
                ForEach(session.settings.compensators) { c in
                    HStack(spacing: Space.s) {
                        Text(c.title(in: system)).lineLimit(1)
                        if let v = system.value(at: c.param) {
                            Text(Fmt.short(v)).foregroundStyle(.secondary).monospacedDigit().lineLimit(1)
                        }
                        Spacer(minLength: Space.xs)
                        Text("±").foregroundStyle(.secondary)
                        OptionalNumberField(title: "Range", value: c.range, prompt: "free", width: 60) { v in
                            set(c.id, "Change Compensator Range") { $0.range = v.map(abs) }
                        }
                        Text(unit(c.param)).foregroundStyle(.secondary).frame(minWidth: 22, alignment: .leading)
                        Button {
                            edit("Remove Compensator") { $0.compensators.removeAll { $0.id == c.id } }
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .help("Remove this compensator")
                        .accessibilityLabel("Remove \(c.title(in: system))")
                    }
                    .font(.callout)
                    .controlSize(.small)
                    .accessibilityElement(children: .contain)
                }
                Menu {
                    ForEach(ToleranceCompensator.candidates(system, variables: design.variables.map(\.param)), id: \.group) { g in
                        Section(g.group) {
                            ForEach(g.params.prefix(g.group == "Surfaces" ? 0 : 40), id: \.self) { p in button(p) }
                            if g.group == "Surfaces" {
                                ForEach(surfaceGroups(g.params), id: \.0) { title, ps in
                                    Menu(title) { ForEach(ps, id: \.self) { p in button(p) } }
                                }
                            }
                        }
                    }
                } label: {
                    Label("Add Compensator", systemImage: "plus")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .controlSize(.small)
                Text("Re-optimised for every trial within ± the range around its nominal value; empty is free.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .padding(Space.xs)
        } label: {
            Text("Compensators")
        }
    }

    private func button(_ p: ParamPath) -> some View {
        let used = session.settings.compensators.contains { system.canonical($0.param) == system.canonical(p) }
        return Button(ToleranceCompensator(p).title(in: system)) {
            edit("Add Compensator") { $0.compensators.append(ToleranceCompensator(p, range: Self.defaultRange(p))) }
        }
        .disabled(used)
    }

    /// Surface parameters by surface, for a submenu each.
    private func surfaceGroups(_ ps: [ParamPath]) -> [(String, [ParamPath])] {
        var order: [String] = []
        var by: [String: [ParamPath]] = [:]
        for p in ps {
            let t = p.surface.flatMap(system.surfaceIndex(of:)).map { "Surface \($0)" } ?? "Other"
            if by[t] == nil { order.append(t) }
            by[t, default: []].append(p)
        }
        return order.map { ($0, by[$0] ?? []) }
    }

    static func defaultRange(_ p: ParamPath) -> Double? {
        switch p {
        case .thickness: 0.5
        case .decenterX, .decenterY: 0.1
        case .tiltX, .tiltY: 0.1
        default: nil
        }
    }

    private func unit(_ p: ParamPath) -> String { p.unit(in: system) }

    private func set(_ id: UUID, _ name: String, _ change: @escaping (inout ToleranceCompensator) -> Void) {
        edit(name) { t in
            guard let i = t.compensators.firstIndex(where: { $0.id == id }) else { return }
            change(&t.compensators[i])
        }
    }
}

/// What a trial is judged by, and the pass limit for the yield.
struct CriterionSection: View {
    let design: DesignDocument
    let session: ToleranceSession
    let edit: (String, (inout ToleranceSettings) -> Void) -> Void

    private var c: ToleranceCriterion { session.settings.criterion }
    private var d: DefaultMeritSettings { c.defaults }
    private var system: OpticalSystem { design.system }
    private var display: CriterionDisplay { c.display(merit: design.merit) }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Space.s) {
                Picker("Criterion", selection: Binding(get: { c.kind }, set: { k in set("Change Criterion") { $0.kind = k; $0.threshold = nil } })) {
                    ForEach(ToleranceCriterion.Kind.allCases) { Text($0.title).tag($0) }
                }
                switch c.kind {
                case .rmsSpot:
                    Picker("Reference", selection: bindD(\.reference, "Change Reference")) {
                        Text("Centroid").tag("centroid")
                        Text("Chief ray").tag("chief")
                    }
                case .wavefront:
                    Picker("Reference", selection: bindD(\.reference, "Change Reference")) {
                        Text("Centroid (piston and tilt removed)").tag("centroid")
                        Text("Chief ray (piston removed)").tag("chief")
                    }
                case .mtf:
                    LabeledContent("Frequencies") {
                        FrequencyListField(values: d.mtfFrequencies) { v in setD("Change MTF Frequencies") { $0.mtfFrequencies = v } }
                    }
                    Picker("Method", selection: bindD(\.mtfMethod, "Change MTF Method")) {
                        Text("Diffraction").tag("diffraction")
                        Text("Geometric").tag("geometric")
                    }
                    Toggle("Weakest field", isOn: bindD(\.worstField, "Change MTF Field Rule"))
                        .help("Judge by the weakest field and direction at each frequency instead of the weighted mean.")
                case .designMerit:
                    Text(design.merit.mode == .custom ? "The Optimize panel's custom merit (\(design.merit.operands.count) operands)." : "The Optimize panel's merit: " + design.merit.defaults.summary + ".")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                if c.kind != .designMerit {
                    HStack {
                        Stepper("Rings \(d.rings)", value: bindD(\.rings, "Change Pupil Rings"), in: 1...12)
                        Spacer()
                        Stepper("Arms \(d.arms)", value: bindD(\.arms, "Change Pupil Arms"), in: 3...24)
                    }
                    .font(.callout.monospacedDigit())
                    HStack {
                        IndexChecklist(
                            title: "Fields",
                            all: system.fields.fields.indices.map { k in
                                "\(k + 1): " + FieldLabel.text([system.fields.fields[k].x, system.fields.fields[k].y], unit: system.fields.kind.unit)
                            }, selected: d.fields) { v in setD("Change Criterion Fields") { $0.fields = v } }
                        IndexChecklist(
                            title: "Wavelengths", all: system.wavelengths.wavelengths.map { String(format: "%.4f µm", $0.um) },
                            selected: d.wavelengths) { v in setD("Change Criterion Wavelengths") { $0.wavelengths = v } }
                        Spacer(minLength: 0)
                    }
                }
                LabeledContent(display.higherIsBetter ? "Pass at or above" : "Pass at or below") {
                    HStack(spacing: Space.xs) {
                        OptionalNumberField(title: "Pass limit", value: c.threshold, prompt: "none", width: 70) { v in
                            set("Change Pass Limit") { $0.threshold = v }
                        }
                        if !display.unit.isEmpty { Text(display.unit).foregroundStyle(.secondary) }
                    }
                }
                .help("The yield is the share of Monte-Carlo trials that pass.")
                Text(caption).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .padding(Space.xs)
        } label: {
            Text("Criterion")
        }
    }

    private var caption: String {
        switch display {
        case .spot: "RMS spot radius over the chosen fields and wavelengths (root of the weighted mean square), after the compensators."
        case .wavefront: "RMS wavefront error in waves over the chosen fields and wavelengths, after the compensators."
        case .mtf: "Polychromatic tangential and sagittal MTF, the root-mean-square shortfall from 1 shown as an MTF. Each trial costs as much as an MTF analysis per compensator step: start with a few dozen trials."
        case .merit: "The merit value itself (lower is better)."
        }
    }

    private func set(_ name: String, _ change: @escaping (inout ToleranceCriterion) -> Void) {
        edit(name) { change(&$0.criterion) }
    }

    private func setD(_ name: String, _ change: @escaping (inout DefaultMeritSettings) -> Void) {
        edit(name) { change(&$0.criterion.defaults) }
    }

    private func bindD<T: Equatable>(_ key: WritableKeyPath<DefaultMeritSettings, T>, _ name: String) -> Binding<T> {
        Binding(get: { d[keyPath: key] }, set: { v in setD(name) { $0[keyPath: key] = v } })
    }
}

/// Sensitivity, Monte Carlo or both; trials and seed.
struct AnalysisSection: View {
    let session: ToleranceSession
    let edit: (String, (inout ToleranceSettings) -> Void) -> Void

    private var t: ToleranceSettings { session.settings }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Space.s) {
                Picker("Run", selection: Binding(get: { t.mode }, set: { m in edit("Change Tolerance Analysis") { $0.mode = m } })) {
                    ForEach(ToleranceSettings.Mode.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                if t.mode != .monteCarlo {
                    Text("Sensitivity: each tolerance at its − and + limit alone (\(2 * t.rows.count + 1) evaluations), ranked, with the root-sum-square prediction.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                if t.mode != .sensitivity {
                    HStack(spacing: Space.l) {
                        LabeledContent("Trials") {
                            OptionalNumberField(title: "Trials", value: Double(t.trials), prompt: "200", width: 64, format: { String(Int($0)) }) { v in
                                if let v, v >= 1 { edit("Change Trials") { $0.trials = min(Int(v.rounded()), 100_000) } }
                            }
                        }
                        LabeledContent("Seed") {
                            OptionalNumberField(title: "Seed", value: Double(t.seed), prompt: "0", width: 64, format: { String(Int($0)) }) { v in
                                if let v, v >= 0 { edit("Change Seed") { $0.seed = Int(v.rounded()) } }
                            }
                        }
                    }
                    .fixedSize()
                    LabeledContent("Levels") {
                        HStack(spacing: Space.xs) {
                            PercentileListField(values: t.percentiles) { v in edit("Change Percentiles") { $0.percentiles = v } }
                            Text("% of builds").foregroundStyle(.secondary)
                        }
                    }
                    .help("The percentiles of the builds the results report, e.g. 50, 90, 97.7")
                    LabeledContent("Keep") {
                        HStack(spacing: Space.xs) {
                            OptionalNumberField(title: "Recorded trials", value: Double(t.recordTrials), prompt: "0", width: 64, format: { String(Int($0)) }) { v in
                                if let v, v >= 0 { edit("Change Recorded Trials") { $0.recordTrials = min(Int(v.rounded()), ToleranceSettings.maxRecordTrials) } }
                            }
                            Text("trials’ perturbations").foregroundStyle(.secondary)
                        }
                    }
                    .help("The first trials keep their perturbations and compensator values for the trial list and the CSV export (at most 10 000). Any trial can be previewed.")
                    Text("Monte Carlo: every tolerance drawn at random per trial; the same seed gives the same trials.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    if let spt = session.lastSecondsPerTrial {
                        Label(Self.estimate(secondsPerTrial: spt, trials: t.trials), systemImage: "clock")
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(Space.xs)
        } label: {
            Text("Analysis")
        }
    }

    /// "Last run: 0.021 s per trial, so 200 trials take about 4 s."
    static func estimate(secondsPerTrial spt: Double, trials: Int) -> String {
        "Last run: \(ToleranceProgress.seconds(spt)) per trial, so \(trials) trials take about \(ToleranceProgress.duration(spt * Double(trials)))."
    }
}

/// Percent levels typed as a list: "50, 90, 97.7".
struct PercentileListField: View {
    let values: [Double]
    let commit: ([Double]) -> Void
    @State private var text = ""
    @State private var invalid = false
    @FocusState private var focused: Bool

    private var formatted: String { values.map(Fmt.short).joined(separator: ", ") }

    var body: some View {
        TextField("Percentiles", text: $text, prompt: Text("50, 90, 97.7"))
            .labelsHidden()
            .multilineTextAlignment(.trailing)
            .font(.body.monospacedDigit())
            .frame(width: 110)
            .focused($focused)
            .onSubmit(finish)
            .onChange(of: focused) { _, f in if !f { finish() } }
            .onAppear { text = formatted }
            .onChange(of: values) { _, _ in if !focused { text = formatted } }
            .overlay(alignment: .bottomTrailing) {
                if invalid { Text("0 to 100").font(.caption2).foregroundStyle(.red).offset(y: 12) }
            }
    }

    private func finish() {
        let parts = text.split(whereSeparator: { $0 == "," || $0 == ";" || $0 == " " }).map(String.init)
        let nums = parts.compactMap { LensValueParser.number($0.replacingOccurrences(of: "%", with: "")) }
        guard nums.count == parts.count, let v = ToleranceSettings.validPercentiles(nums), nums.allSatisfy({ $0 >= 0 && $0 <= 100 }) else {
            invalid = !text.trimmingCharacters(in: .whitespaces).isEmpty
            text = formatted
            return
        }
        invalid = false
        if v != values { commit(v) }
        text = v.map(Fmt.short).joined(separator: ", ")
    }
}

// MARK: - Running

struct ToleranceProgress: View {
    let session: ToleranceSession
    let progress: JobProgress

    /// "Monte Carlo 40 / 200", "Sensitivity", from the engine's stage text
    /// (or the trials counted from the progress).
    static func stageText(_ stage: String, trialsDone: Int? = nil, of trials: Int? = nil) -> String {
        let s = stage.lowercased()
        if s.contains("monte carlo") || s.contains("trial") {
            let nums = stage.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            if nums.count >= 2 { return "Monte Carlo \(nums[0]) / \(nums[1])" }
            if let d = trialsDone, let n = trials { return "Monte Carlo \(d) / \(n)" }
            return "Monte Carlo"
        }
        if s.contains("qualifying") { return "Preparing" }
        if s.contains("sensitiv") { return "Sensitivity" }
        if s.isEmpty || s == "queued" { return "Starting" }
        return stage.prefix(1).uppercased() + stage.dropFirst()
    }

    /// "0.021 s", "12 ms", "1.4 s".
    static func seconds(_ s: Double) -> String {
        guard s.isFinite else { return "—" }
        if s < 0.1 { return "\(Fmt.short((s * 1000 * 10).rounded() / 10)) ms" }
        return "\(Fmt.short((s * 100).rounded() / 100)) s"
    }

    /// "4 s", "2 min", "1 h 5 min".
    static func duration(_ s: Double) -> String {
        guard s.isFinite else { return "—" }
        if s < 1 { return "under a second" }
        if s < 90 { return "\(Int(s.rounded())) s" }
        let m = Int((s / 60).rounded())
        return m < 90 ? "\(m) min" : "\(m / 60) h \(m % 60) min"
    }

    /// "0:42 elapsed · 0.021 s per trial · about 2 min left".
    static func timing(elapsed: TimeInterval, _ t: ToleranceSession.Timing) -> String {
        func clock(_ s: TimeInterval) -> String { String(format: "%d:%02d", Int(s) / 60, Int(s) % 60) }
        var parts = [clock(elapsed) + " elapsed"]
        if let spt = t.secondsPerTrial { parts.append(seconds(spt) + " per trial") }
        if let left = t.secondsLeft { parts.append(left < 1 ? "almost done" : "about \(duration(left)) left") }
        return parts.joined(separator: " · ")
    }

    var body: some View {
        let t = session.settings
        VStack(alignment: .leading, spacing: Space.l) {
            TimelineView(.periodic(from: .now, by: 1)) { tl in
                let timing = session.timing(at: tl.date, progress: progress)
                VStack(alignment: .leading, spacing: Space.l) {
                    HStack(alignment: .firstTextBaseline, spacing: Space.xl) {
                        BigNumber(label: "Progress", value: Fmt.percent(progress.fraction, digits: 0))
                        BigNumber(label: "Stage", value: Self.stageText(progress.stage, trialsDone: timing.trialsDone, of: t.trials))
                    }
                    ProgressView(value: min(max(progress.fraction, 0), 1))
                    if let start = session.startedAt {
                        Text(Self.timing(elapsed: tl.date.timeIntervalSince(start), timing))
                            .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
            }
            Text("\(t.rows.count) tolerances · \(t.compensators.count) compensator\(t.compensators.count == 1 ? "" : "s")"
                + (t.mode == .sensitivity ? "" : " · \(t.trials) trials, seed \(t.seed)"))
                .font(.callout).foregroundStyle(.secondary)
            Text("Stop keeps the sensitivities and trials completed so far. Cancel discards the run. You can close this panel; the job continues.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Results

struct ToleranceResultView: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    let session: ToleranceSession
    let result: ToleranceResult
    @State private var showAll = false
    @State private var exporting: CSVFile?
    @State private var copied = false
    /// The histogram bar whose trials the list shows (`nil`: the worst trials).
    @State private var bin: ClosedRange<Double>?

    private var d: CriterionDisplay { result.display }
    private var system: OpticalSystem { design.system }
    private var rows: [ToleranceRow] { session.resultRows }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xl) {
            summary
            if session.previewed != nil {
                TolerancePreviewCard(session: session, result: result, workspace: workspace)
            }
            if result.hasSensitivity { sensitivity }
            if result.hasMonteCarlo { monteCarlo }
            if !result.compensators.isEmpty { compensators }
        }
        .fileExporter(
            isPresented: Binding(get: { exporting != nil }, set: { if !$0 { exporting = nil } }), document: exporting,
            contentType: .commaSeparatedText, defaultFilename: exporting?.name ?? "tolerance.csv"
        ) { _ in exporting = nil }
        .onAppear(perform: selectPreviewedBin)
        .onChange(of: session.previewed) { _, _ in selectPreviewedBin() }
    }

    /// A previewed trial shows its bar's trials.
    private func selectPreviewedBin() {
        guard case .trial(let k)? = session.previewed, result.trialCriteria.indices.contains(k) else { return }
        let v = d.value(result.trialCriteria[k])
        if let b = bin, b.contains(v) { return }
        bin = MonteCarloChart.bins(result).first { $0.range.contains(v) }?.range
    }

    // MARK: Summary

    private var summary: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            HStack(alignment: .firstTextBaseline) {
                Text(d.title + (d.unit.isEmpty ? "" : " (\(d.unit))")).font(.headline)
                Spacer()
                exportMenu
            }
            Grid(alignment: .leading, horizontalSpacing: Space.xl, verticalSpacing: Space.m) {
                GridRow {
                    if result.hasNominal {
                        BigNumber(label: "Nominal", value: d.format(d.value(result.nominal)))
                            .help(result.compensators.isEmpty ? "The design as it is" : "The design as it is, with the compensators re-optimised")
                    }
                    if result.hasSensitivity {
                        BigNumber(label: "Predicted (RSS)", value: d.format(d.value(result.predicted)))
                            .help("Nominal plus the root-sum-square of every tolerance's worst change")
                    }
                }
                if result.hasMonteCarlo {
                    ForEach(Array(stride(from: 0, to: result.percentiles.count, by: 3)), id: \.self) { start in
                        GridRow {
                            ForEach(Array(result.percentiles[start..<min(start + 3, result.percentiles.count)].enumerated()), id: \.offset) { _, p in
                                BigNumber(label: "\(Fmt.short(p.percentile)) % of builds", value: d.format(d.value(p.value)))
                                    .help("\(Fmt.short(p.percentile)) % of the Monte-Carlo trials reach \(d.higherIsBetter ? "at least" : "at most") this value")
                            }
                        }
                    }
                }
            }
            if let y = result.yield {
                Label("Yield \(Fmt.percent(y)) at \(d.higherIsBetter ? "≥" : "≤") \(d.format(d.value(result.threshold ?? .nan)))", systemImage: "checkmark.seal")
                    .font(.callout)
            }
            if let note = statusNote {
                Label(note, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var statusNote: String? {
        var parts: [String] = []
        if result.stopped {
            parts.append("Stopped early: " + (result.hasMonteCarlo ? "statistics of the \(result.trials) completed trials" : "the sensitivities completed so far") + ".")
        }
        if result.failures > 0 { parts.append("\(result.failures) trials failed (no image) and are left out.") }
        if result.hasMonteCarlo && !result.hasSensitivity { parts.append("Run the sensitivity too to see the largest contributors.") }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }

    private var exportMenu: some View {
        Menu {
            Button(copied ? "Copied" : "Copy Summary") {
                Clipboard.copy(ToleranceReport.summary(result, rows: rows, system: system, title: system.name))
                copied = true
            }
            Divider()
            if result.hasSensitivity {
                Button("Save Sensitivity Table as CSV…") {
                    exporting = CSVFile(name: "\(system.name) sensitivity.csv", text: ToleranceReport.sensitivityCSV(result, rows: rows, system: system))
                }
            }
            if result.hasMonteCarlo {
                Button("Save Monte-Carlo Trials as CSV…") {
                    exporting = CSVFile(
                        name: "\(system.name) trials.csv", text: ToleranceReport.trialsCSV(result, rows: rows, system: session.runSystem ?? system))
                }
            }
        } label: {
            Label("Export", systemImage: "square.and.arrow.up")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .controlSize(.small)
        .help("Copy a summary as text, or save the tables as CSV")
    }

    // MARK: Sensitivity

    private var sensitivity: some View {
        let list = showAll ? result.sensitivities : Array(result.sensitivities.prefix(12))
        let worst = result.worstOffenders
        return VStack(alignment: .leading, spacing: Space.xs) {
            HStack(alignment: .firstTextBaseline) {
                Text("Sensitivity").font(.headline)
                Spacer()
                Text("Δ \(d.unit.isEmpty ? d.title : d.unit) at − / +").font(.caption).foregroundStyle(.secondary)
            }
            Text("Each tolerance alone at its limits, largest share of the RSS first. Click a row to select its surface; the eye previews that build.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, Space.xs)
            ForEach(Array(list.enumerated()), id: \.element.id) { k, s in
                SensitivityRow(
                    rank: k + 1, s: s, row: rows.indices.contains(s.id) ? rows[s.id] : nil, result: result, system: system,
                    isWorst: worst.contains(s.id), selected: isPreviewed(s),
                    select: { selectSurface(s) },
                    preview: { togglePreview(s) })
            }
            if result.sensitivities.count > 12 {
                Button(showAll ? "Show the 12 largest" : "Show all \(result.sensitivities.count)") { showAll.toggle() }
                    .buttonStyle(.link).font(.callout)
            }
            if result.sensitivityCancelled {
                Text("Stopped: tolerances not evaluated are missing.").font(.caption).foregroundStyle(.orange)
            }
        }
    }

    private func isPreviewed(_ s: ToleranceResult.Sensitivity) -> Bool {
        if case .sensitivity(let k, _)? = session.previewed { return k == s.id }
        return false
    }

    private func selectSurface(_ s: ToleranceResult.Sensitivity) {
        guard let id = s.param?.surface else { return }
        window.selection = id
    }

    private func togglePreview(_ s: ToleranceResult.Sensitivity) {
        if isPreviewed(s) {
            session.endPreview(in: workspace)
        } else {
            session.preview(.sensitivity(operand: s.id, plus: s.worsePlus), in: workspace)
        }
    }

    // MARK: Monte Carlo

    private var monteCarlo: some View {
        let previewedTrial: Int? = if case .trial(let k)? = session.previewed { k } else { nil }
        return VStack(alignment: .leading, spacing: Space.s) {
            Text("Monte Carlo").font(.headline)
            MonteCarloChart(
                result: result, selection: $bin,
                marked: previewedTrial.flatMap { result.trialCriteria.indices.contains($0) ? d.value(result.trialCriteria[$0]) : nil }
            )
            .frame(height: 190)
            FlowLayout(spacing: Space.m, lineSpacing: Space.xxs) {
                if result.hasNominal { LegendItem(color: .accentColor, label: "Nominal " + d.format(d.value(result.nominal))) }
                if !result.percentiles.isEmpty {
                    LegendItem(color: .secondary, label: result.percentiles.map { Fmt.short($0.percentile) }.joined(separator: ", ") + " % of builds", dashed: true)
                }
                LegendItem(color: .primary.opacity(0.75), label: "Cumulative share (right axis)")
            }
            .font(.caption)
            TrialList(result: result, session: session, workspace: workspace, bin: $bin)
            KeyValueGrid(
                rows: [
                    ("Trials", "\(result.trials), seed \(result.seed)"),
                    ("Mean", d.formatWithUnit(d.value(result.mean))),
                    ("Best", d.formatWithUnit(d.value(result.min))), ("Worst", d.formatWithUnit(d.value(result.max))),
                ] + (result.secondsPerTrial.map { [("Time per trial", ToleranceProgress.seconds($0))] } ?? [])
                    + (result.failures > 0 ? [("Failed trials", "\(result.failures)")] : []))
        }
    }

    // MARK: Compensators

    private var compensators: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            Text("Compensators").font(.headline)
            ForEach(Array(result.compensators.enumerated()), id: \.offset) { j, c in
                CompensatorStats(j: j, compensator: c, result: result, system: system)
            }
        }
    }
}

/// The Monte-Carlo trials of the selected histogram bar (or the worst
/// ones), each previewable: trial number, its value and, when recorded,
/// its first compensator's move.
struct TrialList: View {
    let result: ToleranceResult
    let session: ToleranceSession
    let workspace: Workspace
    @Binding var bin: ClosedRange<Double>?
    @State private var limit = 6

    private var d: CriterionDisplay { result.display }

    var body: some View {
        let trials = bin.map { result.trials(shownIn: $0) } ?? result.trialsWorstFirst
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline) {
                Text(title(trials.count)).font(.callout.weight(.medium))
                Spacer()
                if bin != nil {
                    Button("Worst Trials") { bin = nil }
                        .buttonStyle(.link).font(.caption)
                        .help("List the worst trials instead of the selected bar's")
                }
            }
            if trials.isEmpty {
                Text("No trial in this bar.").font(.caption).foregroundStyle(.secondary)
            }
            ForEach(trials.prefix(limit), id: \.self) { k in row(k) }
            if trials.count > limit {
                Button("Show \(min(trials.count - limit, 20)) more") { limit += 20 }
                    .buttonStyle(.link).font(.caption)
            }
            Text(bin == nil ? "Click a bar of the histogram for its trials; the eye rebuilds a trial in the window." : "The eye rebuilds a trial in the window, read-only.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .onChange(of: bin) { _, _ in limit = 6 }
    }

    private func title(_ n: Int) -> String {
        guard let b = bin else { return "Worst trials" }
        return "\(n) trial\(n == 1 ? "" : "s") at \(d.format(b.lowerBound))–\(d.format(b.upperBound))\(d.unit.isEmpty ? "" : " " + d.unit)"
    }

    private func row(_ k: Int) -> some View {
        let selected = session.previewed == .trial(k)
        let rec = result.record(k)
        let comp: String? = rec.flatMap { r in
            guard let c = r.compensators.first, let n = result.nominalCompensators.first, let p = result.compensators.first,
                let system = session.runSystem
            else { return nil }
            return ToleranceCompensator(p.param).title(in: system) + " " + Fmt.signed(c - n, decimals: 3).replacingOccurrences(of: "-", with: "\u{2212}")
        }
        return HStack(spacing: Space.s) {
            Text("Trial \(k + 1)").monospacedDigit().frame(width: 72, alignment: .leading)
            Text(d.formatWithUnit(d.value(result.trialCriteria[k]))).monospacedDigit()
            if let comp { Text(comp).font(.caption.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1) }
            Spacer(minLength: Space.xs)
            Image(systemName: selected ? "eye.fill" : "eye").foregroundStyle(selected ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
        }
        .font(.callout)
        .padding(.vertical, 2)
        .padding(.horizontal, Space.xs)
        .background(selected ? AnyShapeStyle(.tint.opacity(0.16)) : AnyShapeStyle(.clear), in: .rect(cornerRadius: 5))
        .contentShape(Rectangle())
        .onTapGesture {
            if selected { session.endPreview(in: workspace) } else { session.preview(.trial(k), in: workspace) }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityLabel("Trial \(k + 1), \(d.formatWithUnit(d.value(result.trialCriteria[k])))")
        .accessibilityHint(selected ? "Ends the preview" : "Previews this build in the window")
        .help(selected ? "End the preview" : "Preview trial \(k + 1) in the window: its perturbations\(result.compensators.isEmpty ? "" : " and re-optimised compensators"); not applied")
    }
}

/// The previewed build: what it is, its criterion, its compensators and
/// its perturbations (largest share of their limit first), read-only.
struct TolerancePreviewCard: View {
    let session: ToleranceSession
    let result: ToleranceResult
    let workspace: Workspace
    @State private var showAll = false

    private var d: CriterionDisplay { result.display }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(alignment: .firstTextBaseline, spacing: Space.s) {
                Image(systemName: "eye").foregroundStyle(.tint)
                Text(title).font(.callout.weight(.semibold)).lineLimit(2)
                Spacer(minLength: 0)
                Button("End Preview") { session.endPreview(in: workspace) }.controlSize(.small)
            }
            if let error = session.previewError {
                Label(error, systemImage: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let p = session.preview {
                Text("\(d.title) \(d.formatWithUnit(d.value(p.criterion))) · shown in the layout and the analyses, not applied")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                if !p.compensators.isEmpty { compensators(p) }
                perturbations(p)
            } else {
                HStack(spacing: Space.s) {
                    ProgressView().controlSize(.small)
                    Text("Rebuilding the build in the engine…").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .padding(Space.m)
        .background(.tint.opacity(0.08), in: .rect(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.tint.opacity(0.25)))
        .onChange(of: session.previewed) { _, _ in showAll = false }
    }

    private var title: String {
        guard let c = session.previewed else { return "" }
        return String(session.title(of: c).dropFirst("Tolerance: ".count)).capitalizedFirst
    }

    private func compensators(_ p: TolerancePreview) -> some View {
        let system = session.runSystem ?? p.system
        return VStack(alignment: .leading, spacing: 1) {
            ForEach(Array(zip(result.compensators, p.compensators).enumerated()), id: \.offset) { j, pair in
                let (c, v) = pair
                let unit = c.param.unit(in: system)
                let n = result.nominalCompensators.indices.contains(j) ? result.nominalCompensators[j] : nil
                HStack(spacing: Space.s) {
                    Text(ToleranceCompensator(c.param).title(in: system))
                    Spacer(minLength: Space.xs)
                    Text(Fmt.value(v, unit: unit)).monospacedDigit()
                    if let n { Text("(" + Fmt.signed(v - n, decimals: 4).trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "-", with: "\u{2212}") + ")").monospacedDigit().foregroundStyle(.secondary) }
                }
                .font(.caption)
                .help("Re-optimised for this build\(n.map { "; nominal " + Fmt.value($0, unit: unit) } ?? "")")
            }
        }
    }

    @ViewBuilder private func perturbations(_ p: TolerancePreview) -> some View {
        let changes = session.changes(p.perturbations)
        if changes.isEmpty {
            Text("No perturbation.").font(.caption).foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 2) {
                Text(changes.count == 1 ? "Perturbation" : "Perturbations, largest share of their limit first")
                    .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                ForEach(showAll ? changes : Array(changes.prefix(6))) { c in
                    HStack(spacing: Space.s) {
                        Text(c.label).lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: Space.xs)
                        Text(c.text).monospacedDigit()
                        LimitBar(fraction: c.fraction).frame(width: 44, height: 6)
                    }
                    .font(.caption)
                    .help("\(c.label): \(c.text) of \(c.limits)")
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("\(c.label), \(c.text) of \(c.limits)")
                }
                if changes.count > 6 {
                    Button(showAll ? "Show the 6 largest" : "Show all \(changes.count)") { showAll.toggle() }
                        .buttonStyle(.link).font(.caption)
                }
            }
        }
    }
}

/// A perturbation within its limits: a bar from the centre, to the right
/// for +, to the left for −, full at the limit.
struct LimitBar: View {
    let fraction: Double

    var body: some View {
        GeometryReader { g in
            let w = g.size.width, h = g.size.height
            let f = CGFloat(min(max(fraction, -1), 1))
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Rectangle().fill(.tint).frame(width: max(abs(f) * w / 2, 1), height: h).offset(x: f >= 0 ? w / 2 : w / 2 + f * w / 2)
                Rectangle().fill(.primary.opacity(0.5)).frame(width: 1, height: h).offset(x: w / 2)
            }
        }
        .accessibilityHidden(true)
    }
}

extension String {
    /// The first letter in upper case.
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}

/// One sensitivity: rank, tolerance, share bar and the change at each
/// limit; the worst offenders tinted.
struct SensitivityRow: View {
    let rank: Int
    let s: ToleranceResult.Sensitivity
    let row: ToleranceRow?
    let result: ToleranceResult
    let system: OpticalSystem
    let isWorst: Bool
    let selected: Bool
    let select: () -> Void
    let preview: () -> Void
    @State private var hovering = false

    private var d: CriterionDisplay { result.display }

    var body: some View {
        let share = result.share(s)
        let nominal = d.value(result.nominal)
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: Space.s) {
                Text("\(rank)").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 18, alignment: .trailing)
                Text(row?.label(in: system) ?? s.label).lineLimit(1)
                    .fontWeight(isWorst ? .semibold : .regular)
                Text(row?.limitsText ?? "").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer(minLength: Space.xs)
                Button(action: preview) {
                    Image(systemName: selected ? "eye.fill" : "eye")
                }
                .buttonStyle(.borderless)
                .opacity(hovering || selected ? 1 : 0.35)
                .help(selected ? "End the preview" : "Preview this build in the window: the tolerance at its worse limit\(result.compensators.isEmpty ? "" : ", compensated"); not applied")
                .accessibilityLabel(selected ? "End preview" : "Preview")
            }
            HStack(spacing: Space.s) {
                Color.clear.frame(width: 18, height: 1)
                ShareBar(fraction: share, highlighted: isWorst)
                Spacer(minLength: Space.xs)
                Text(d.formatChange(d.value(s.criterionMinus) - nominal) + " / " + d.formatChange(d.value(s.criterionPlus) - nominal))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .font(.callout)
        .padding(.vertical, 3)
        .padding(.horizontal, Space.xs)
        .background(background, in: .rect(cornerRadius: 6))
        .contentShape(Rectangle())
        .onTapGesture(perform: select)
        .onHover { hovering = $0 }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(rank). \(row?.label(in: system) ?? s.label)\(isWorst ? ", a worst offender" : "")")
        .accessibilityValue("\(ContributionBar.percent(share)) of the RSS")
        .accessibilityAddTraits(.isButton)
        .help("\(row?.label(in: system) ?? s.label): \(ContributionBar.percent(share)) of the RSS. Click to select the surface.")
    }

    private var background: AnyShapeStyle {
        if selected { return AnyShapeStyle(.tint.opacity(0.16)) }
        if isWorst { return AnyShapeStyle(Color.orange.opacity(0.12)) }
        return AnyShapeStyle(.clear)
    }
}

/// A share of the RSS as a bar with its percentage.
struct ShareBar: View {
    let fraction: Double
    var highlighted = false

    var body: some View {
        HStack(spacing: Space.xs) {
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(highlighted ? AnyShapeStyle(Color.orange) : AnyShapeStyle(.tint))
                        .frame(width: max(fraction > 0 ? 2 : 0, g.size.width * min(max(fraction, 0), 1)))
                }
            }
            .frame(width: 90, height: 5)
            Text(ContributionBar.percent(fraction)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 34, alignment: .trailing)
        }
        .accessibilityHidden(true)
    }
}

/// The trials' distribution: a histogram, the cumulative share of builds
/// (right axis), the nominal design and the engine's percentile levels. A
/// click on a bar selects it (its trials are listed); the previewed trial
/// is marked under the axis.
struct MonteCarloChart: View {
    let result: ToleranceResult
    var selection: Binding<ClosedRange<Double>?> = .constant(nil)
    /// The shown value of the previewed trial.
    var marked: Double? = nil

    private var d: CriterionDisplay { result.display }

    /// A bar: its centre, the trials in it and the values it spans.
    struct Bin: Identifiable {
        var center: Double
        var count: Int
        var range: ClosedRange<Double>
        var id: Double { center }
    }

    /// The histogram of the shown values (10, 16 or 24 bars by trial count).
    static func bins(_ result: ToleranceResult) -> [Bin] {
        let shown = result.criteria.map(result.display.value).filter(\.isFinite)
        let b = Histogram.bins(shown, count: shown.count >= 200 ? 24 : shown.count >= 60 ? 16 : 10)
        let w = b.count > 1 ? b[1].0 - b[0].0 : 0
        return b.map { Bin(center: $0.0, count: $0.1, range: ($0.0 - w / 2)...($0.0 + w / 2)) }
    }

    var body: some View {
        let shown = result.criteria.map(d.value).filter(\.isFinite)
        let bins = Self.bins(result)
        let width = bins.count > 1 ? bins[1].center - bins[0].center : max(abs(bins.first?.center ?? 1) * 0.1, 1e-9)
        let peak = Double(bins.map(\.count).max() ?? 1)
        let curve = cumulative(shown)
        let nominal = result.hasNominal ? d.value(result.nominal) : nil
        let dom = domain(shown, nominal: nominal)
        let levels = Self.labelled(result.percentiles.map { (p: $0.percentile, v: d.value($0.value)) }, span: dom.upperBound - dom.lowerBound)
        let selected = selection.wrappedValue
        Chart {
            ForEach(bins) { b in
                let isSelected = selected.map { abs($0.lowerBound - b.range.lowerBound) < width * 1e-6 } ?? false
                RectangleMark(
                    xStart: .value(d.title, b.center - 0.46 * width), xEnd: .value(d.title, b.center + 0.46 * width),
                    yStart: .value("Trials", 0), yEnd: .value("Trials", b.count)
                )
                .foregroundStyle(
                    isSelected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(Palette.fieldColor(0).opacity(selected == nil ? 0.75 : 0.4)))
                .cornerRadius(1.5)
            }
            ForEach(Array(curve.enumerated()), id: \.offset) { _, p in
                LineMark(x: .value(d.title, p.0), y: .value("Trials", p.1 * peak), series: .value("Series", "cumulative"))
                    .foregroundStyle(Color.primary.opacity(0.75))
                    .lineStyle(StrokeStyle(lineWidth: 1.4))
                    .interpolationMethod(.stepEnd)
            }
            ForEach(levels, id: \.p) { l in
                if l.v.isFinite {
                    RuleMark(x: .value(d.title, l.v))
                        .foregroundStyle(Color.secondary)
                        .lineStyle(StrokeStyle(lineWidth: 0.8, dash: [3, 3]))
                        .annotation(position: .top, alignment: .center, spacing: 2) {
                            Text(l.label).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        }
                }
            }
            if let nominal, nominal.isFinite {
                RuleMark(x: .value(d.title, nominal))
                    .foregroundStyle(Color.accentColor)
                    .lineStyle(StrokeStyle(lineWidth: 1.6))
            }
            if let marked, marked.isFinite {
                PointMark(x: .value(d.title, marked), y: .value("Trials", 0))
                    .symbol(.triangle)
                    .symbolSize(60)
                    .foregroundStyle(Color.accentColor)
            }
        }
        .environment(\.locale, NumberDisplay.locale)
        .chartXAxisLabel(d.title + (d.unit.isEmpty ? "" : " (\(d.unit))"), alignment: .center)
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { _ in
                AxisGridLine().foregroundStyle(.quaternary)
                AxisValueLabel()
            }
            AxisMarks(position: .trailing, values: [0, 0.5, 1].map { $0 * peak }) { v in
                AxisValueLabel {
                    if let y = v.as(Double.self) { Text(String(format: "%.0f%%", y / max(peak, 1e-12) * 100)) }
                }
            }
        }
        .chartYAxisLabel("Trials", position: .leading)
        .chartXScale(domain: dom)
        .chartOverlay { proxy in
            GeometryReader { geo in
                if let f = proxy.plotFrame {
                    let frame = geo[f]
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .frame(width: frame.width, height: frame.height)
                        .offset(x: frame.minX, y: frame.minY)
                        .onTapGesture { p in
                            guard let x: Double = proxy.value(atX: p.x) else { return }
                            let hit = bins.first { $0.count > 0 && $0.range.contains(x) }?.range
                            selection.wrappedValue = hit == selected ? nil : hit
                        }
                        .help("Click a bar to list its trials")
                }
            }
        }
        .padding(.top, Space.l)
        .font(.caption2)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Distribution of \(shown.count) Monte-Carlo trials of the \(d.title)")
        .accessibilityValue(
            levels.map { "\(Fmt.short($0.p)) percent of builds \(d.higherIsBetter ? "at least" : "at most") \(d.formatWithUnit($0.v))" }.joined(separator: ", "))
        .accessibilityHint("Select a bar to list its trials")
    }

    /// Percent levels with their labels; a level too close to the one
    /// before it (under 8 % of the axis) shares that label ("90, 97.7").
    static func labelled(_ levels: [(p: Double, v: Double)], span: Double) -> [(p: Double, v: Double, label: String)] {
        var out: [(p: Double, v: Double, label: String)] = []
        for l in levels {
            if let last = out.last, last.v.isFinite, l.v.isFinite, abs(l.v - last.v) < 0.08 * span, !last.label.isEmpty {
                out[out.count - 1].label += ", " + Fmt.short(l.p)
                out.append((l.p, l.v, ""))
            } else {
                out.append((l.p, l.v, Fmt.short(l.p)))
            }
        }
        return out
    }

    /// The cumulative share of trials as good as or better than each value,
    /// rising from the best to the worst.
    private func cumulative(_ v: [Double]) -> [(Double, Double)] {
        let sorted = d.higherIsBetter ? v.sorted(by: >) : v.sorted()
        guard !sorted.isEmpty else { return [] }
        return sorted.enumerated().map { ($0.element, Double($0.offset + 1) / Double(sorted.count)) }
    }

    private func domain(_ v: [Double], nominal: Double?) -> ClosedRange<Double> {
        let all = v + (nominal.map { [$0] } ?? [])
        guard let lo = all.min(), let hi = all.max() else { return 0...1 }
        let pad = max((hi - lo) * 0.06, abs(hi) * 1e-6, 1e-12)
        return (lo - pad)...(hi + pad)
    }
}

/// A compensator's nominal value, the range the sensitivity evaluations
/// used against the allowed range, and its Monte-Carlo spread.
struct CompensatorStats: View {
    let j: Int
    let compensator: ToleranceResult.Compensator
    let result: ToleranceResult
    let system: OpticalSystem

    var body: some View {
        let title = ToleranceCompensator(compensator.param, range: compensator.range).title(in: system)
        let unit = compensator.param.unit(in: system)
        let used = result.compensatorRange(j)
        let nominal = result.nominalCompensators.indices.contains(j) ? result.nominalCompensators[j] : nil
        VStack(alignment: .leading, spacing: Space.xs) {
            HStack(alignment: .firstTextBaseline) {
                Text(title).font(.callout.weight(.medium))
                Spacer()
                if let r = compensator.range { Text("allowed ±\(Fmt.short(r)) \(unit)").font(.caption).foregroundStyle(.secondary) }
            }
            if let used, let r = compensator.range, r > 0 {
                RangeUsageBar(lo: used.0, hi: used.1, range: r)
                    .frame(height: 10)
                    .help("The sensitivity evaluations moved it from \(Fmt.short(used.0)) to +\(Fmt.short(used.1)) \(unit) of ±\(Fmt.short(r)) allowed")
            }
            KeyValueGrid(rows: rowsText(nominal: nominal, used: used, unit: unit))
            if let used, let r = compensator.range, max(-used.0, used.1) >= r * 0.99 {
                Label("It reached its limit: some builds need more adjustment than allowed.", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(Space.s)
        .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 8))
    }

    private func rowsText(nominal: Double?, used: (Double, Double)?, unit: String) -> [(String, String)] {
        var rows: [(String, String)] = []
        if let nominal { rows.append(("Nominal (re-optimised)", Fmt.value(nominal, unit: unit))) }
        if let used { rows.append(("Sensitivity range used", Fmt.signed(used.0, decimals: 4).trimmingCharacters(in: .whitespaces) + " to " + Fmt.signed(used.1, decimals: 4).trimmingCharacters(in: .whitespaces) + " " + unit)) }
        if result.compensatorMean.indices.contains(j) {
            let mean = result.compensatorMean[j], sd = result.compensatorStd.indices.contains(j) ? result.compensatorStd[j] : .nan
            let shift = nominal.map { mean - $0 }
            rows.append(("Monte Carlo mean", Fmt.value(mean, unit: unit) + (shift.map { " (" + Fmt.signed($0, decimals: 4).trimmingCharacters(in: .whitespaces) + ")" } ?? "")))
            rows.append(("Monte Carlo σ (±3σ)", String(format: "%.4g (±%.4g) ", sd, 3 * sd) + unit))
        }
        return rows
    }
}

/// The used part of a compensator's ±range, centred on nominal.
struct RangeUsageBar: View {
    let lo: Double
    let hi: Double
    let range: Double

    var body: some View {
        GeometryReader { g in
            let w = g.size.width, h = g.size.height
            let x0 = w / 2 + CGFloat(max(lo / range, -1)) * w / 2, x1 = w / 2 + CGFloat(min(hi / range, 1)) * w / 2
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary).frame(height: h * 0.5)
                Capsule().fill(.tint).frame(width: max(x1 - x0, 2), height: h * 0.5).offset(x: x0)
                Rectangle().fill(.primary.opacity(0.6)).frame(width: 1, height: h).offset(x: w / 2)
            }
            .frame(height: h)
        }
        .accessibilityHidden(true)
    }
}

/// A histogram's bins: centres and counts.
enum Histogram {
    static func bins(_ v: [Double], count: Int) -> [(Double, Int)] {
        guard let lo = v.min(), let hi = v.max(), hi > lo else { return v.isEmpty ? [] : [(v[0], v.count)] }
        let w = (hi - lo) / Double(count)
        var c = Array(repeating: 0, count: count)
        for x in v { c[min(Int((x - lo) / w), count - 1)] += 1 }
        return c.enumerated().map { (lo + (Double($0.offset) + 0.5) * w, $0.element) }
    }
}

/// A CSV file for the exporter.
struct CSVFile: FileDocument {
    static var readableContentTypes: [UTType] { [.commaSeparatedText] }
    var name: String
    var text: String

    init(name: String, text: String) {
        self.name = name
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        name = "tolerance.csv"
        text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}
