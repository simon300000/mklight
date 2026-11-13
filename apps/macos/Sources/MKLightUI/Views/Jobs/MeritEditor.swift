import MKLightEngine
import MKLightModel
import SwiftUI

/// A compact optional number: empty text is "none". Shows the formatted
/// value, edits the exact one, and commits on Return or when focus leaves,
/// if the text was edited; rejected text reverts.
struct OptionalNumberField: View {
    let title: String
    let value: Double?
    var prompt = "none"
    var width: CGFloat = 76
    var format: (Double) -> String = Fmt.short
    /// Accept `∞` / `inf` (radius bounds).
    var allowsInfinite = false
    /// A borderless cell on a light fill (dense tables) instead of a
    /// bordered field.
    var compact = false
    let commit: (Double?) -> Void

    var body: some View {
        CommittingField(value: value, format: entry, commit: commit) { text, focused in
            TextField(title, text: text, prompt: Text(prompt))
                .labelsHidden()
                .textFieldStyle(compact ? AnyTextFieldStyle(.plain) : AnyTextFieldStyle(.roundedBorder))
                .multilineTextAlignment(.trailing)
                .font(.callout.monospacedDigit())
                .padding(.horizontal, compact ? 4 : 0)
                .padding(.vertical, compact ? 1 : 0)
                .background {
                    if compact {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(focused ? AnyShapeStyle(.background) : AnyShapeStyle(.quaternary.opacity(0.6)))
                            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(focused ? AnyShapeStyle(.tint) : AnyShapeStyle(.clear)))
                    }
                }
                .frame(width: width)
        }
        .accessibilityLabel(title)
    }

    /// Empty text is "none"; a number must be finite unless `∞` is allowed.
    private var entry: FieldEdit<Double?>.Format {
        FieldEdit.Format(
            exact: { $0.map(NumberDisplay.exact) ?? "" }, display: { $0.map(format) ?? "" },
            parse: { [allowsInfinite] text in
                let t = text.trimmingCharacters(in: .whitespaces)
                if t.isEmpty { return .some(nil) }
                guard let v = LensValueParser.number(t), v.isFinite || allowsInfinite else { return nil }
                return v
            })
    }
}

// MARK: - Merit function

/// The Optimize panel's merit function: the default generator (criterion,
/// pupil sampling, fields and wavelengths, thickness limits, squeeze) or a
/// custom operand table, with the merit evaluated live at the design.
struct MeritFunctionSection: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var session: OptimizeSession
    @Environment(\.undoManager) private var undoManager
    @State private var selection: Set<UUID> = []

    private var merit: MeritSettings { session.merit }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Space.m) {
                HStack(spacing: Space.m) {
                    Picker("Mode", selection: Binding(get: { merit.mode }, set: setMode)) {
                        Text("Default").tag(MeritSettings.Mode.default)
                        Text("Custom").tag(MeritSettings.Mode.custom)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                    .help("Default: the engine generates the merit from a few choices. Custom: edit every operand.")
                    Spacer(minLength: 0)
                    MeritReadout(session: session)
                }
                switch merit.mode {
                case .default:
                    DefaultMeritForm(design: design, session: session)
                    Button {
                        Task { await session.convertToCustom(in: workspace, document: design, variables: design.variables, undoManager: undoManager) }
                    } label: {
                        Label("Convert to Custom", systemImage: "list.bullet.rectangle")
                    }
                    .disabled(session.evaluating || workspace.snapshot == nil)
                    .help("Start a custom merit from the operands the default generates for this design")
                case .custom:
                    OperandTable(design: design, workspace: workspace, session: session, selection: $selection)
                }
                if let e = session.evaluationError, !session.evaluating {
                    Label("The engine could not evaluate this merit: \(e)", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange).lineLimit(4).fixedSize(horizontal: false, vertical: true)
                }
                Toggle("Hold the focal length", isOn: Binding(get: { merit.holdEFL }, set: { v in edit("Hold Focal Length") { $0.holdEFL = v } }))
                    .help("Keep the EFL at its start value as a constraint, unless the merit has an EFL operand.")
            }
            .padding(Space.xs)
        } label: {
            Text("Merit Function")
        }
    }

    private func setMode(_ m: MeritSettings.Mode) {
        edit(m == .custom ? "Use Custom Merit" : "Use Default Merit") { $0.mode = m }
    }

    private func edit(_ name: String, _ change: (inout MeritSettings) -> Void) {
        session.editMerit(design, name, undoManager: undoManager, change)
    }
}

/// The merit at the current design, evaluated by the engine.
struct MeritReadout: View {
    let session: OptimizeSession

    var body: some View {
        HStack(spacing: Space.xs) {
            if session.evaluating { ProgressView().controlSize(.mini) }
            if let e = session.evaluationError, !session.evaluating {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange).help(e)
                    .accessibilityLabel("Evaluation failed: \(e)")
            }
            VStack(alignment: .trailing, spacing: 0) {
                Text("Merit").font(.caption2).foregroundStyle(.secondary)
                Text(session.evaluation.map { Fmt.short($0.merit) } ?? "—")
                    .font(.callout.weight(.medium).monospacedDigit())
                    .contentTransition(.numericText())
            }
        }
        .opacity(session.evaluating ? 0.6 : 1)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Current merit")
        .accessibilityValue(session.evaluation.map { Fmt.short($0.merit) } ?? "not evaluated")
        .help(helpText)
    }

    private var helpText: String {
        guard let e = session.evaluation else { return "The merit at the current design, before optimizing." }
        let on = e.reportedOn.hasPrefix("cpu") || e.reportedOn.isEmpty ? "CPU f64 reference" : e.reportedOn
        var s = "Merit \(Fmt.short(e.merit)) at the current design (\(on))."
        if e.violation.isFinite && e.violation > 1e-9 { s += " Constraint violation \(Fmt.short(e.violation))." }
        switch e.status {
        case .ok: break
        case .degraded(let n): s += " \(n) residual\(n == 1 ? "" : "s") carry penalty values (failed rays or operands)."
        case .failed: s += " The evaluation failed: no meaningful merit."
        }
        return s
    }
}

/// The default merit generator's settings.
struct DefaultMeritForm: View {
    let design: DesignDocument
    let session: OptimizeSession
    @Environment(\.undoManager) private var undoManager

    private var d: DefaultMeritSettings { session.merit.defaults }
    private var system: OpticalSystem { design.system }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            Picker("Criterion", selection: bind(\.criterion, "Change Criterion")) {
                ForEach(DefaultMeritSettings.Criterion.allCases) { Text($0.title).tag($0) }
            }
            switch d.criterion {
            case .rmsSpot:
                Picker("Reference", selection: bind(\.reference, "Change Reference")) {
                    Text("Centroid").tag("centroid")
                    Text("Chief ray").tag("chief")
                }
                if d.reference == "centroid" {
                    Toggle("White-light centroid", isOn: Binding(get: { d.scope == "polychromatic" }, set: { v in set("Change Centroid") { $0.scope = v ? "polychromatic" : "monochromatic" } }))
                        .help("One centroid for all wavelengths, so lateral color counts; otherwise one per wavelength.")
                }
            case .wavefront:
                Picker("Reference", selection: bind(\.reference, "Change Reference")) {
                    Text("Centroid (piston and tilt removed)").tag("centroid")
                    Text("Chief ray (piston removed)").tag("chief")
                }
            case .mtf:
                LabeledContent("Frequencies") {
                    FrequencyListField(values: d.mtfFrequencies) { v in set("Change MTF Frequencies") { $0.mtfFrequencies = v } }
                }
                LabeledContent("Target") {
                    OptionalNumberField(title: "MTF target", value: d.mtfTarget, prompt: "0.5") { v in
                        set("Change MTF Target") { $0.mtfTarget = min(max(v ?? 0.5, 0), 1) }
                    }
                }
                Picker("Method", selection: bind(\.mtfMethod, "Change MTF Method")) {
                    Text("Diffraction").tag("diffraction")
                    Text("Geometric").tag("geometric")
                }
                Toggle("Weakest field", isOn: bind(\.worstField, "Change MTF Field Rule"))
                    .help("Maximise the weakest field and direction at each frequency (minimax), instead of the field-weighted bounds.")
            }
            HStack {
                Stepper("Rings \(d.rings)", value: bind(\.rings, "Change Pupil Rings"), in: 1...12)
                Spacer()
                Stepper("Arms \(d.arms)", value: bind(\.arms, "Change Pupil Arms"), in: 3...24)
            }
            .font(.callout.monospacedDigit())
            HStack {
                IndexChecklist(
                    title: "Fields", all: system.fields.fields.indices.map { k in "\(k + 1): " + FieldLabel.text([system.fields.fields[k].x, system.fields.fields[k].y], unit: system.fields.kind.unit) },
                    selected: d.fields) { v in set("Change Merit Fields") { $0.fields = v } }
                IndexChecklist(
                    title: "Wavelengths", all: system.wavelengths.wavelengths.map { String(format: "%.4f µm", $0.um) },
                    selected: d.wavelengths) { v in set("Change Merit Wavelengths") { $0.wavelengths = v } }
                Spacer(minLength: 0)
            }
            ThicknessLimits(d: d) { name, change in set(name, change) }
            if ParaxialXYData.isAnamorphic(nil, system: system) || d.squeezeTarget != nil {
                LabeledContent("Squeeze target") {
                    OptionalNumberField(title: "Squeeze target", value: d.squeezeTarget) { v in set("Change Squeeze Target") { $0.squeezeTarget = v } }
                }
                .help("Hold the signed EFL_y / EFL_x at this value (anamorphic designs).")
            }
            Text(caption).hint()
        }
    }

    private var caption: String {
        switch d.criterion {
        case .rmsSpot: "The mean square RMS spot radius over the chosen fields and wavelengths, by their weights."
        case .wavefront: "The mean square RMS wavefront error over the chosen fields and wavelengths, by their weights."
        case .mtf: "Polychromatic tangential and sagittal MTF lower bounds at each frequency."
        }
    }

    private func set(_ name: String, _ change: (inout DefaultMeritSettings) -> Void) {
        session.editMerit(design, name, undoManager: undoManager) { change(&$0.defaults) }
    }

    private func bind<T: Equatable>(_ key: WritableKeyPath<DefaultMeritSettings, T>, _ name: String) -> Binding<T> {
        Binding(get: { d[keyPath: key] }, set: { v in set(name) { $0[keyPath: key] = v } })
    }
}

/// Comma-separated frequencies (cycles/mm): positive, kept sorted. Commits
/// on Return or when focus leaves, if the text was edited; text that is not
/// such a list reverts.
struct FrequencyListField: View {
    let values: [Double]
    let commit: ([Double]) -> Void

    var body: some View {
        HStack(spacing: Space.xs) {
            CommittingField(value: values, format: Self.entry, commit: commit) { text, _ in
                TextField("Frequencies", text: text, prompt: Text("10, 30"))
                    .labelsHidden().textFieldStyle(.roundedBorder).font(.callout.monospacedDigit()).frame(width: 110)
            }
            Text("cyc/mm").foregroundStyle(.secondary).font(.callout)
        }
    }

    static let entry = FieldEdit<[Double]>.Format(
        exact: { FrequencyListSetting.text($0, exact: true) }, display: { FrequencyListSetting.text($0) },
        parse: { FrequencyListSetting.parse($0).flatMap { $0.isEmpty ? nil : $0 } })
}

/// A menu of checkboxes over indices; every one selected is `nil` (all).
struct IndexChecklist: View {
    let title: String
    let all: [String]
    let selected: [Int]?
    let commit: ([Int]?) -> Void

    var body: some View {
        Menu {
            Button("All") { commit(nil) }
            Divider()
            ForEach(all.indices, id: \.self) { k in
                Toggle(all[k], isOn: Binding(get: { isOn(k) }, set: { toggle(k, $0) }))
            }
        } label: {
            Text("\(title): \(summary)")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    private var summary: String {
        guard let s = selected, s.count < all.count else { return "All" }
        return s.count == 1 ? "\(s[0] + 1)" : "\(s.count) of \(all.count)"
    }

    private func isOn(_ k: Int) -> Bool { selected?.contains(k) ?? true }

    private func toggle(_ k: Int, _ on: Bool) {
        var s = Set(selected ?? Array(all.indices))
        if on { s.insert(k) } else { s.remove(k) }
        guard !s.isEmpty else { return }
        commit(s.count == all.count ? nil : s.sorted())
    }
}

/// The default merit's glass and air thickness boundaries.
struct ThicknessLimits: View {
    let d: DefaultMeritSettings
    let set: (String, (inout DefaultMeritSettings) -> Void) -> Void
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            Grid(alignment: .leading, horizontalSpacing: Space.s, verticalSpacing: Space.xs) {
                GridRow {
                    Text("")
                    Text("Glass").font(.caption).foregroundStyle(.secondary)
                    Text("Air").font(.caption).foregroundStyle(.secondary)
                }
                row("Min center", \.glassMinCentre, \.airMinCentre)
                row("Max center", \.glassMaxCentre, \.airMaxCentre)
                row("Min edge", \.glassMinEdge, \.airMinEdge)
            }
            .disabled(!d.thickness)
            .padding(.top, Space.xs)
        } label: {
            Toggle("Thickness limits (mm)", isOn: Binding(get: { d.thickness }, set: { v in set("Change Thickness Limits") { $0.thickness = v } }))
        }
    }

    private func row(_ title: String, _ glass: WritableKeyPath<DefaultMeritSettings, Double?>, _ air: WritableKeyPath<DefaultMeritSettings, Double?>) -> some View {
        GridRow {
            Text(title).font(.callout)
            OptionalNumberField(title: "Glass \(title.lowercased())", value: d[keyPath: glass], width: 64) { v in set("Change Thickness Limits") { $0[keyPath: glass] = v } }
            OptionalNumberField(title: "Air \(title.lowercased())", value: d[keyPath: air], width: 64) { v in set("Change Thickness Limits") { $0[keyPath: air] = v } }
        }
    }
}

// MARK: - Custom operands

/// The custom merit's operands: a list (drag to reorder, ⌫ deletes) with
/// each operand's current value and its share of the merit, and the
/// selected operand's parameters below.
struct OperandTable: View {
    let design: DesignDocument
    let workspace: Workspace
    let session: OptimizeSession
    @Binding var selection: Set<UUID>
    @Environment(\.undoManager) private var undoManager
    @State private var showPicker = false

    private var operands: [MeritOperand] { session.merit.operands }

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            toolbar
            if operands.isEmpty {
                Text("No operands. Add one with +, or convert the default merit.")
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 60)
            } else {
                List(selection: $selection) {
                    ForEach(Array(operands.enumerated()), id: \.element.id) { k, op in
                        OperandRow(index: k, operand: op, evaluation: session.evaluation?.row(k, kind: op.type), issues: issues(op))
                            .tag(op.id)
                            .contextMenu {
                                Button("Duplicate") { duplicate([op.id]) }
                                Button(op.enabled ? "Disable" : "Enable") { update(op.id, op.enabled ? "Disable Operand" : "Enable Operand") { $0.enabled.toggle() } }
                                Divider()
                                Button("Delete", role: .destructive) { delete([op.id]) }
                            }
                    }
                    .onMove { from, to in edit("Move Operand") { $0.move(fromOffsets: from, toOffset: to) } }
                }
                .listStyle(.bordered)
                .alternatingRowBackgrounds()
                .frame(height: min(max(CGFloat(operands.count) * 40 + 8, 96), 300))
                .onDeleteCommand { delete(selection) }
                .accessibilityLabel("Merit operands")
            }
            if selection.count == 1, let id = selection.first, let i = operands.firstIndex(where: { $0.id == id }) {
                OperandDetail(index: i, operand: operands[i], system: design.system, evaluation: session.evaluation?.row(i, kind: operands[i].type)) { name, change in
                    update(id, name, change)
                }
                .id(id)
            }
        }
    }

    private var toolbar: some View {
        HStack(spacing: Space.xs) {
            Button {
                showPicker = true
            } label: {
                Label("Add Operand", systemImage: "plus")
            }
            .popover(isPresented: $showPicker, arrowEdge: .bottom) {
                OperandPicker { d in
                    showPicker = false
                    add(d)
                }
            }
            .help("Add an operand after the selection")
            Button {
                delete(selection)
            } label: {
                Label("Delete", systemImage: "minus")
            }
            .disabled(selection.isEmpty)
            .help("Delete the selected operands (⌫)")
            Button {
                duplicate(selection)
            } label: {
                Label("Duplicate", systemImage: "plus.square.on.square")
            }
            .disabled(selection.isEmpty)
            .help("Duplicate the selected operands")
            Button {
                moveSelection(by: -1)
            } label: {
                Label("Move Up", systemImage: "arrow.up")
            }
            .disabled(selection.count != 1 || firstSelected == 0)
            .keyboardShortcut(.upArrow, modifiers: [.command, .option])
            .help("Move up (⌥⌘↑)")
            Button {
                moveSelection(by: 1)
            } label: {
                Label("Move Down", systemImage: "arrow.down")
            }
            .disabled(selection.count != 1 || firstSelected.map { $0 >= operands.count - 1 } ?? true)
            .keyboardShortcut(.downArrow, modifiers: [.command, .option])
            .help("Move down (⌥⌘↓)")
            Spacer()
            Text("\(operands.count) operand\(operands.count == 1 ? "" : "s")").font(.caption).foregroundStyle(.secondary)
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
        .controlSize(.small)
    }

    private var firstSelected: Int? { selection.first.flatMap { id in operands.firstIndex { $0.id == id } } }

    private func issues(_ op: MeritOperand) -> [String] {
        op.localIssues(fieldCount: design.system.fields.fields.count, wavelengthCount: design.system.wavelengths.wavelengths.count)
    }

    private func edit(_ name: String, _ change: (inout MeritSettings) -> Void) {
        session.editMerit(design, name, undoManager: undoManager, change)
    }

    private func update(_ id: UUID, _ name: String, _ change: (inout MeritOperand) -> Void) {
        edit(name) { m in
            if let i = m.operands.firstIndex(where: { $0.id == id }) { change(&m.operands[i]) }
        }
    }

    private func add(_ d: OperandDescriptor) {
        var new: UUID?
        edit("Add Operand") { new = $0.add(d, after: firstSelected ?? (operands.isEmpty ? nil : operands.count - 1)) }
        if let new { selection = [new] }
    }

    private func delete(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        edit(ids.count == 1 ? "Delete Operand" : "Delete Operands") { $0.delete(ids) }
        selection = []
    }

    private func duplicate(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        var copies: [UUID] = []
        edit("Duplicate Operand") { copies = $0.duplicate(ids) }
        selection = Set(copies)
    }

    private func moveSelection(by step: Int) {
        guard let i = firstSelected else { return }
        let j = i + step
        guard j >= 0, j < operands.count else { return }
        edit("Move Operand") { $0.move(fromOffsets: IndexSet(integer: i), toOffset: step > 0 ? j + 1 : j) }
    }
}

/// One operand: number, kind and parameters, target, current value and its
/// contribution to the merit as a bar.
struct OperandRow: View {
    let index: Int
    let operand: MeritOperand
    let evaluation: MeritEvaluation.Row?
    let issues: [String]

    var body: some View {
        HStack(alignment: .center, spacing: Space.s) {
            Text("\(index + 1)").font(.caption.monospacedDigit()).foregroundStyle(.tertiary).frame(width: 24, alignment: .trailing)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(OperandCatalog.title(operand.type)).font(.callout).lineLimit(1).layoutPriority(1)
                    if operand.isConstraint {
                        Text("constraint").font(.caption2).foregroundStyle(.secondary)
                            .padding(.horizontal, 4).background(.quaternary, in: Capsule())
                    }
                    if let problem { Image(systemName: "exclamationmark.triangle.fill").font(.caption2).foregroundStyle(.orange).help(problem) }
                }
                Text(detail).font(.caption.monospacedDigit()).foregroundStyle(problem == nil ? AnyShapeStyle(.secondary) : AnyShapeStyle(.orange)).lineLimit(1)
            }
            Spacer(minLength: Space.xs)
            VStack(alignment: .trailing, spacing: 2) {
                Text(evaluation.map { Fmt.short($0.value) } ?? "—").font(.callout.monospacedDigit()).lineLimit(1)
                ContributionBar(fraction: evaluation?.fraction ?? .nan)
            }
            .frame(width: 78, alignment: .trailing)
        }
        .opacity(operand.enabled ? 1 : 0.45)
        .padding(.vertical, 1)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Operand \(index + 1), \(OperandCatalog.title(operand.type)), \(detail)")
        .accessibilityValue(accessibilityValue)
    }

    private var problem: String? { evaluation?.error.map { "The engine could not evaluate it: \($0)" } ?? issues.first }

    private var detail: String {
        if let problem { return problem }
        var parts: [String] = []
        if let s = OperandSummary.parameters(operand), !s.isEmpty { parts.append(s) }
        parts.append("\(operand.comparison.symbol) \(Fmt.short(operand.target))")
        if !operand.isConstraint { parts.append("w \(Fmt.short(operand.weight))") }
        return parts.joined(separator: " · ")
    }

    private var accessibilityValue: String {
        guard let e = evaluation else { return "not evaluated" }
        var s = "value \(Fmt.short(e.value))"
        if e.fraction.isFinite { s += ", \(Fmt.percent(e.fraction)) of the merit" }
        return s
    }
}

/// A small horizontal bar with the share of the merit.
struct ContributionBar: View {
    let fraction: Double

    var body: some View {
        HStack(spacing: 4) {
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    if fraction.isFinite {
                        Capsule().fill(.tint).frame(width: max(fraction > 0 ? 2 : 0, g.size.width * min(fraction, 1)))
                    }
                }
            }
            .frame(width: 34, height: 4)
            Text(fraction.isFinite ? Self.percent(fraction) : "")
                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 34, alignment: .trailing)
        }
        .accessibilityHidden(true)
    }

    static func percent(_ f: Double) -> String {
        f >= 0.995 ? "100%" : f >= 0.1 ? String(format: "%.0f%%", f * 100) : f >= 0.001 ? String(format: "%.1f%%", f * 100) : f > 0 ? "<0.1%" : "0%"
    }
}

/// Short texts for an operand's parameters and selectors.
enum OperandSummary {
    static func parameters(_ op: MeritOperand) -> String? {
        var parts: [String] = []
        for p in op.descriptor?.params ?? [] {
            guard let v = op.kind[p.key], v != .null else { continue }
            switch p.kind {
            case .surface: parts.append("\(p.title.lowercased()) S\(v.int.map(String.init) ?? "?")")
            case .wavelength: parts.append("\(p.title) λ\((v.int ?? 0) + 1)")
            case .choice: parts.append((v.string ?? v.displayText).replacingOccurrences(of: "_", with: " "))
            case .flag: if v.bool == true { parts.append(p.title.lowercased()) }
            case .sampling: parts.append("\(v["rings"]?.int ?? 3)×\(v["arms"]?.int ?? 6)")
            case .operandList: parts.append("of " + (v.intArray ?? []).map { "#\($0 + 1)" }.joined(separator: ", "))
            case .number, .integer: parts.append("\(p.title.lowercased()) \(v.doubleValue.map(Fmt.short) ?? v.displayText)")
            }
        }
        // Field and wavelength first: generated operands differ in them,
        // and a long row truncates its end.
        var selectors: [String] = []
        switch op.field["field"]?.string {
        case "index": selectors.append("F\((op.field["index"]?.int ?? 0) + 1)")
        case "normalized": selectors.append("H(\(Fmt.short(op.field["hx"]?.nf ?? 0)), \(Fmt.short(op.field["hy"]?.nf ?? 0)))")
        default: break
        }
        switch op.wavelength["wavelength"]?.string {
        case "all": selectors.append("poly")
        case "index": selectors.append("λ\((op.wavelength["index"]?.int ?? 0) + 1)")
        default: break
        }
        parts = selectors + parts
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }
}

/// The selected operand's parameters, selectors, target, comparison and
/// weight, with its value and errors.
struct OperandDetail: View {
    let index: Int
    let operand: MeritOperand
    let system: OpticalSystem
    let evaluation: MeritEvaluation.Row?
    let update: (String, (inout MeritOperand) -> Void) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(alignment: .firstTextBaseline) {
                Text("\(index + 1). \(OperandCatalog.title(operand.type))").font(.headline)
                Spacer()
                if let d = operand.descriptor, !d.unit.isEmpty { Text(d.unit).font(.caption).foregroundStyle(.secondary) }
            }
            if let s = operand.descriptor?.summary, !s.isEmpty { Text(s).font(.caption).foregroundStyle(.secondary) }
            ForEach(operand.descriptor?.params ?? []) { p in param(p) }
            if operand.descriptor == nil {
                Text("Parameters: " + JSONText.text(.object(operand.kind))).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(3)
            }
            fieldMenu
            wavelengthMenu
            HStack {
                Picker("Comparison", selection: Binding(get: { operand.comparison }, set: { v in update("Change Comparison") { $0.comparison = v } })) {
                    ForEach(MeritOperand.Comparison.allCases) { Text($0.symbol).tag($0) }
                }
                .pickerStyle(.segmented)
                .fixedSize()
                .help("= target, ≤ an upper bound, ≥ a lower bound")
                OptionalNumberField(title: "Target", value: operand.target, prompt: "0") { v in update("Change Target") { $0.target = v ?? 0 } }
                Spacer()
            }
            HStack {
                Toggle("Constraint", isOn: Binding(get: { operand.isConstraint }, set: { v in update(v ? "Make Constraint" : "Make Residual") { $0.isConstraint = v } }))
                    .help("Hold it exactly (or as a bound) instead of weighting it in the merit")
                Spacer()
                if !operand.isConstraint {
                    Text("Weight").foregroundStyle(.secondary)
                    OptionalNumberField(title: "Weight", value: operand.weight, prompt: "1", width: 70) { v in
                        update("Change Weight") { $0.weight = max(v ?? 1, 0) }
                    }
                }
            }
            HStack {
                Toggle("Enabled", isOn: Binding(get: { operand.enabled }, set: { v in update(v ? "Enable Operand" : "Disable Operand") { $0.enabled = v } }))
                Spacer()
                if let e = evaluation {
                    Text("Value \(Fmt.short(e.value))").font(.callout.monospacedDigit())
                    if e.fraction.isFinite { Text("· \(ContributionBar.percent(e.fraction))").font(.callout.monospacedDigit()).foregroundStyle(.secondary) }
                }
            }
            if let error = evaluation?.error {
                WarningNote(error)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(operand.localIssues(fieldCount: system.fields.fields.count, wavelengthCount: system.wavelengths.wavelengths.count), id: \.self) {
                WarningNote($0)
            }
        }
        .controlSize(.small)
        .padding(Space.s)
        .background(.fill.quinary, in: .rect(cornerRadius: 8))
    }

    @ViewBuilder private func param(_ p: OperandParam) -> some View {
        let v = operand.kind[p.key]
        let name = "Change \(p.title)"
        switch p.kind {
        case .number(let unit):
            LabeledContent(p.title) {
                HStack(spacing: 4) {
                    OptionalNumberField(title: p.title, value: v?.doubleValue, prompt: p.required ? "0" : "auto") { x in
                        update(name) { $0.kind[p.key] = x.map(JSONValue.number) ?? (p.required ? .number(0) : nil) }
                    }
                    if !unit.isEmpty { Text(unit).font(.caption).foregroundStyle(.secondary) }
                }
            }
        case .integer(let lo):
            LabeledContent(p.title) {
                OptionalNumberField(title: p.title, value: v?.doubleValue, prompt: "\(lo)") { x in
                    update(name) { $0.kind[p.key] = .integer(Int64(max(Int(saturating: x ?? Double(lo)), lo))) }
                }
            }
        case .surface:
            Picker(p.title, selection: Binding(get: { v?.int ?? -1 }, set: { x in update(name) { $0.kind[p.key] = x < 0 ? nil : .integer(Int64(x)) } })) {
                if !p.required { Text(p.key == "to" ? "Next" : p.key == "from" ? "First" : "Image").tag(-1) }
                ForEach(1..<system.surfaces.count, id: \.self) { i in
                    Text(i == system.imageIndex ? "Image" : "Surface \(i)" + (system.surfaces[i].label.isEmpty ? "" : " · \(system.surfaces[i].label)"))
                        .tag(Int(system.surfaces[i].id.rawValue))
                }
            }
        case .wavelength:
            Picker(p.title, selection: Binding(get: { v?.int ?? 0 }, set: { x in update(name) { $0.kind[p.key] = .integer(Int64(x)) } })) {
                ForEach(system.wavelengths.wavelengths.indices, id: \.self) { k in
                    Text(String(format: "%.4f µm", system.wavelengths.wavelengths[k].um)).tag(k)
                }
            }
        case .choice(let options):
            Picker(p.title, selection: Binding(get: { v?.string ?? options.first ?? "" }, set: { x in update(name) { $0.kind[p.key] = .string(x) } })) {
                ForEach(options, id: \.self) { Text(SettingsBar.title(.string($0))).tag($0) }
            }
        case .flag:
            Toggle(p.title, isOn: Binding(get: { v?.bool ?? false }, set: { x in update(name) { $0.kind[p.key] = .bool(x) } }))
        case .sampling:
            HStack {
                Stepper("Rings \(v?["rings"]?.int ?? 3)", value: Binding(get: { v?["rings"]?.int ?? 3 }, set: { x in update(name) { $0.kind[p.key] = sampling(rings: x, arms: v?["arms"]?.int ?? 6, old: v) } }), in: 1...12)
                Stepper("Arms \(v?["arms"]?.int ?? 6)", value: Binding(get: { v?["arms"]?.int ?? 6 }, set: { x in update(name) { $0.kind[p.key] = sampling(rings: v?["rings"]?.int ?? 3, arms: x, old: v) } }), in: 3...24)
            }
            .font(.callout.monospacedDigit())
        case .operandList:
            LabeledContent(p.title) {
                OperandIndexField(indices: v?.intArray ?? []) { x in update(name) { $0.kind[p.key] = .array(x.map { .integer(Int64($0)) }) } }
            }
            .help("Numbers of earlier operands, e.g. 1, 3")
        }
    }

    private func sampling(rings: Int, arms: Int, old: JSONValue?) -> JSONValue {
        var o = old?.object ?? [:]
        o["rings"] = .integer(Int64(rings))
        o["arms"] = .integer(Int64(arms))
        return .object(o)
    }

    private var fieldMenu: some View {
        let single = operand.descriptor?.singleField ?? false
        let current = operand.field["field"]?.string == "index" ? (operand.field["index"]?.int ?? 0) : -1
        return Picker("Field", selection: Binding(get: { current }, set: { k in
            update("Change Operand Field") { $0.field = k < 0 ? .object(["field": .string("all")]) : .object(["field": .string("index"), "index": .integer(Int64(k))]) }
        })) {
            if !single || current < 0 { Text("All fields (weighted)").tag(-1) }
            ForEach(system.fields.fields.indices, id: \.self) { k in
                Text("\(k + 1): " + FieldLabel.text([system.fields.fields[k].x, system.fields.fields[k].y], unit: system.fields.kind.unit)).tag(k)
            }
        }
    }

    private var wavelengthMenu: some View {
        let w = operand.wavelength
        let current = w["wavelength"]?.string == "index" ? (w["index"]?.int ?? 0) : w["wavelength"]?.string == "all" ? -2 : -1
        return Picker("Wavelength", selection: Binding(get: { current }, set: { k in
            update("Change Operand Wavelength") {
                $0.wavelength = k == -1 ? .object(["wavelength": .string("primary")])
                    : k == -2 ? .object(["wavelength": .string("all")]) : .object(["wavelength": .string("index"), "index": .integer(Int64(k))])
            }
        })) {
            Text("Primary").tag(-1)
            Text("All (polychromatic)").tag(-2)
            ForEach(system.wavelengths.wavelengths.indices, id: \.self) { k in
                Text(String(format: "%.4f µm", system.wavelengths.wavelengths[k].um)).tag(k)
            }
        }
    }
}

/// Operand numbers (1-based in the text, 0-based stored); empty is none.
/// Commits on Return or when focus leaves, if the text was edited; text
/// that is not a list of operand numbers reverts.
struct OperandIndexField: View {
    let indices: [Int]
    let commit: ([Int]) -> Void

    var body: some View {
        CommittingField(value: indices, format: Self.entry, commit: commit) { text, _ in
            TextField("Operands", text: text, prompt: Text("1, 2"))
                .labelsHidden().textFieldStyle(.roundedBorder).frame(width: 110)
        }
    }

    static let entry: FieldEdit<[Int]>.Format = {
        let text: ([Int]) -> String = { $0.map { "\($0 + 1)" }.joined(separator: ", ") }
        return FieldEdit.Format(
            exact: text, display: text,
            parse: { t in
                let parts = t.split(whereSeparator: { $0 == "," || $0 == ";" || $0 == " " })
                let v = parts.compactMap { Int($0) }.filter { $0 >= 1 }.map { $0 - 1 }
                return v.count == parts.count ? v : nil
            })
    }()
}

/// The searchable operand picker, grouped by category.
struct OperandPicker: View {
    let pick: (OperandDescriptor) -> Void
    @State private var query = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            TextField("Search operands", text: $query, prompt: Text("Search operands"))
                .textFieldStyle(.roundedBorder)
                .focused($focused)
                .padding(Space.m)
                .onSubmit { if let first = OperandCatalog.search(query).first?.items.first { pick(first) } }
            List {
                ForEach(OperandCatalog.search(query), id: \.category) { group in
                    Section(group.category) {
                        ForEach(group.items) { d in
                            Button {
                                pick(d)
                            } label: {
                                VStack(alignment: .leading, spacing: 1) {
                                    HStack {
                                        Text(d.title)
                                        Spacer()
                                        if !d.unit.isEmpty { Text(d.unit).font(.caption).foregroundStyle(.secondary) }
                                    }
                                    if !d.summary.isEmpty { Text(d.summary).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
        }
        .frame(width: 320, height: 380)
        .onAppear { focused = true }
    }
}

// MARK: - Variables and bounds

/// The design's variables with their bounds, edited inline: min and max
/// (empty is unbounded) in the parameter's unit, pin (equal bounds: held at
/// its value), remove. Curvatures are bounded as a radius window in mm by
/// default (`RadiusBounds`), or as curvatures.
struct VariablesBoundsList: View {
    let design: DesignDocument
    let workspace: Workspace
    @AppStorage("MKLightCurvatureBoundsUnit") private var curvatureUnit = CurvatureBoundsUnit.radius
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        let vars = design.variables
        VStack(alignment: .leading, spacing: 0) {
            if vars.contains(where: { $0.param.isCurvature }) {
                HStack(spacing: Space.s) {
                    Text("Curvatures as").font(.caption).foregroundStyle(.secondary)
                    Picker("Bound curvatures as", selection: $curvatureUnit) {
                        Text("Radius mm").tag(CurvatureBoundsUnit.radius)
                        Text("Curvature 1/mm").tag(CurvatureBoundsUnit.curvature)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)
                    .fixedSize()
                    .help("Enter and read the bounds of radii as a radius window in mm, or as the curvature c = 1/R the optimizer uses")
                    Spacer(minLength: 0)
                    Image(systemName: "questionmark.circle")
                        .font(.caption).foregroundStyle(.secondary)
                        .help(RadiusBounds.help)
                        .accessibilityLabel("About radius bounds")
                        .accessibilityValue(RadiusBounds.help)
                }
                .padding(.bottom, Space.s)
            }
            HStack(spacing: Space.s) {
                Text("Variable").frame(maxWidth: .infinity, alignment: .leading)
                Text("Min").frame(width: VariableBoundsRow.fieldWidth, alignment: .trailing)
                Text("Max").frame(width: VariableBoundsRow.fieldWidth, alignment: .trailing)
                Text("Unit").frame(width: VariableBoundsRow.unitWidth, alignment: .leading)
                Color.clear.frame(width: 40)
            }
            .font(.caption).foregroundStyle(.secondary)
            .padding(.bottom, Space.xs)
            ForEach(Array(vars.enumerated()), id: \.element.param) { k, v in
                if k > 0 { Divider() }
                VariableBoundsRow(design: design, workspace: workspace, variable: v, curvatureUnit: curvatureUnit)
                    .padding(.vertical, 3)
            }
        }
    }
}

struct VariableBoundsRow: View {
    let design: DesignDocument
    let workspace: Workspace
    let variable: DesignVariable
    var curvatureUnit = CurvatureBoundsUnit.radius
    @Environment(\.undoManager) private var undoManager
    @State private var rejection: String?

    static let fieldWidth: CGFloat = 70
    static let unitWidth: CGFloat = 40

    private var path: ParamPath { variable.param }
    private var value: Double? { design.system.value(at: path) }
    private var asRadius: Bool { path.isCurvature && curvatureUnit == .radius }
    private var label: String { path.label(in: design.system) }
    private var unit: String { asRadius ? "mm" : path.unit(in: design.system) }

    /// The min and max fields: a radius window for curvatures shown as radii.
    private var shown: (min: Double?, max: Double?) {
        asRadius ? RadiusBounds.radius(lower: variable.lower, upper: variable.upper) : (variable.lower, variable.upper)
    }

    var body: some View {
        let shown = self.shown
        HStack(spacing: Space.s) {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(label).font(.callout).lineLimit(1)
                    if variable.isPinned { Image(systemName: "pin.fill").font(.caption2).foregroundStyle(.tint).accessibilityLabel("pinned") }
                }
                Text(caption).font(.caption.monospacedDigit()).foregroundStyle(problem == nil ? AnyShapeStyle(.secondary) : AnyShapeStyle(.orange))
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .help(solvedHelp ?? problem ?? help)
            OptionalNumberField(
                title: "Minimum of \(label)" + (unit.isEmpty ? "" : " in \(unit)"), value: shown.min, prompt: "−∞", width: Self.fieldWidth,
                format: BoundsFormat.compact, allowsInfinite: asRadius
            ) { v in setShown(min: v, max: shown.max) }
            OptionalNumberField(
                title: "Maximum of \(label)" + (unit.isEmpty ? "" : " in \(unit)"), value: shown.max, prompt: asRadius ? "∞" : "+∞", width: Self.fieldWidth,
                format: BoundsFormat.compact, allowsInfinite: asRadius
            ) { v in setShown(min: shown.min, max: v) }
            Text(unit.isEmpty ? "–" : unit)
                .font(.caption).foregroundStyle(.secondary)
                .lineLimit(1).minimumScaleFactor(0.8)
                .frame(width: Self.unitWidth, alignment: .leading)
                .help(unit.isEmpty ? "Dimensionless" : "The bounds are in \(unit)")
                .accessibilityHidden(true)
            HStack(spacing: 2) {
                Button {
                    if variable.isPinned {
                        let d = DesignVariable.defaultBounds(for: path, in: design.system)
                        design.setBounds(path, lower: d.lower, upper: d.upper, undoManager: undoManager)
                    } else if let value {
                        design.setBounds(path, lower: value, upper: value, undoManager: undoManager)
                    }
                } label: {
                    Image(systemName: variable.isPinned ? "pin.slash" : "pin")
                }
                .help(variable.isPinned ? "Unpin: let the optimizer vary it again" : "Pin at its value (equal bounds): keep it as a variable but hold it")
                .accessibilityLabel(variable.isPinned ? "Unpin" : "Pin")
                .disabled(value == nil)
                Button {
                    design.setVariable(path, false, undoManager: undoManager)
                } label: {
                    Image(systemName: "xmark.circle")
                }
                .help("Fix this parameter (remove the variable)")
                .accessibilityLabel("Remove variable")
            }
            .buttonStyle(.borderless)
            .frame(width: 40)
        }
        .controlSize(.small)
    }

    private var help: String {
        let other = BoundsFormat.counterpart(lower: variable.lower, upper: variable.upper, shownAs: curvatureUnit).map { "Bounds as \($0). " } ?? ""
        if asRadius { return other + RadiusBounds.help }
        if path.isCurvature { return other + "Bounds on the curvature c = 1/R (1/mm)." }
        return unit.isEmpty ? "Bounds of a dimensionless parameter." : "Bounds in \(unit)."
    }

    /// Stores typed min / max in the shown unit (a radius window converts
    /// to curvature bounds; a window through R = 0 is refused).
    private func setShown(min: Double?, max: Double?) {
        guard asRadius else {
            rejection = nil
            design.setBounds(path, lower: min, upper: max, undoManager: undoManager)
            return
        }
        switch RadiusBounds.curvature(minRadius: min, maxRadius: max) {
        case .success(let c):
            rejection = nil
            design.setBounds(path, lower: c.lower, upper: c.upper, undoManager: undoManager)
        case .failure(let p):
            rejection = p.description
        }
    }

    /// A variable on a parameter a solve sets, in full.
    private var solvedHelp: String? {
        design.system.solve(for: path).map {
            "A \($0.kind.kind.title.lowercased()) solve sets this value, so the optimizer leaves the variable out of the run. Remove the variable, or remove the solve to vary it."
        }
    }

    /// Why the variable cannot be optimised as it is.
    private var problem: String? {
        if let rejection { return rejection }
        if design.system.isSolved(path) { return "Set by a solve: not varied" }
        if case .semiDiameter = path, design.system.surfaceIndex(of: path.surface ?? SurfaceID(0)).map({ design.system.surfaces[$0].semiDiameter == .auto }) ?? false {
            return "The semi-diameter is automatic: set a value in the lens data first."
        }
        guard let value else { return "This parameter does not exist on the surface any more." }
        if let lo = variable.lower, let hi = variable.upper, lo > hi { return "The minimum is above the maximum." }
        if variable.outOfBounds(value) { return "Now \(nowText(value)): outside its bounds" }
        return nil
    }

    private func nowText(_ value: Double) -> String {
        asRadius ? "R " + Fmt.short(RadiusDisplay.radius(fromCurvature: value)) + " mm" : Fmt.short(value) + (unit.isEmpty ? "" : " " + unit)
    }

    private var caption: String {
        if let problem { return problem }
        guard let value else { return "" }
        // The value now, in the row's unit (the unit column names it); the
        // bounds in the other unit are in the help.
        if path.isCurvature {
            let r = RadiusDisplay.radius(fromCurvature: value)
            guard asRadius else { return "now c \(BoundsFormat.compact(value))" }
            let s = shown
            return "now R \(Fmt.short(r))" + (RadiusBounds.passesThroughFlat(minRadius: s.min, maxRadius: s.max) ? " · through ∞" : "")
        }
        return "now \(Fmt.short(value))"
    }
}

/// A type-erased text field style (bordered or plain by a flag).
struct AnyTextFieldStyle: TextFieldStyle {
    private let make: (TextField<_Label>) -> AnyView

    init<S: TextFieldStyle>(_ style: S) {
        make = { AnyView($0.textFieldStyle(style)) }
    }

    func _body(configuration: TextField<_Label>) -> some View { make(configuration) }
}
