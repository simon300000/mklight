import MKLightEngine
import MKLightModel
import SwiftUI

/// A cell of the lens table.
struct CellID: Hashable {
    var row: SurfaceID
    var column: LensColumn
}

/// The lens data editor: one row per surface with the parameters every
/// type has. Type-specific parameters are in the surface inspector.
///
/// Keyboard: ↑/↓ select rows; Return (or a double click) edits the row;
/// in a cell, Tab / ⇧Tab move across, ↑/↓ and Return move between rows
/// (committing), Esc reverts and leaves the cell. ⇧⌘I / ⌥⌘I insert, ⌘⌫
/// deletes, ⇧⌘T makes the stop, ⌃⌘V toggles the focused value's variable.
struct LensTableView: View {
    let design: DesignDocument
    let workspace: Workspace
    @Bindable var window: WindowState
    @Environment(\.undoManager) private var undoManager
    @FocusState private var focus: CellID?

    private var system: OpticalSystem { design.system }
    private var editor: LensEditor { LensEditor(document: design, undoManager: undoManager) }

    /// Issues by surface index.
    private var issues: [Int: [ValidationIssue]] {
        Dictionary(grouping: workspace.issues.filter { $0.surface != nil && $0.severity != .info }, by: { $0.surface! })
    }

    private var rows: [LensRow] {
        LensRow.rows(
            for: system, autoSemiDiameters: EngineLayout.autoSemiDiameters(EngineLayout.scene(from: workspace.layout.output)),
            variables: Set(design.variables.map(\.param)),
            bounds: Dictionary(design.variables.map { ($0.param, $0.isPinned ? .pinned : $0.isBounded ? .bounded : .free) }, uniquingKeysWith: { a, _ in a }))
    }

    var body: some View {
        let rows = self.rows
        VStack(spacing: 0) {
            header
            Table(rows, selection: $window.selection) {
                TableColumn("Surf") { row in SurfaceLabel(row: row, issues: issues[row.index] ?? []) }
                    .width(min: 40, ideal: 46, max: 64)
                TableColumn("Type") { row in
                    TypeCell(row: row, shape: system.surfaces[row.index].shape, editor: editor)
                }
                    .width(min: 90, ideal: 112, max: 150)
                column(.comment, rows: rows).width(min: 70, ideal: 120)
                column(.radius, rows: rows).width(min: 80, ideal: 104, max: 160)
                column(.thickness, rows: rows).width(min: 80, ideal: 104, max: 160)
                column(.material, rows: rows).width(min: 80, ideal: 104, max: 160)
                column(.semiDiameter, rows: rows).width(min: 80, ideal: 104, max: 160)
                column(.conic, rows: rows).width(min: 64, ideal: 88, max: 140)
            }
            .tableStyle(.inset)
            .scrollEdgeEffectStyle(.hard, for: .top)
            .contextMenu(forSelectionType: SurfaceID.self) { ids in
                contextMenu(ids.first)
            } primaryAction: { ids in
                if let id = ids.first { focus = CellID(row: id, column: .radius) }
            }
            .onKeyPress(.return) {
                guard focus == nil, let id = window.selection else { return .ignored }
                focus = CellID(row: id, column: rows.first { $0.id == id }?.editable.contains(.radius) == true ? .radius : .comment)
                return .handled
            }
            .onChange(of: focus) { _, f in
                if let f {
                    window.selection = f.row
                    window.focusedColumn = f.column
                }
            }
            .accessibilityLabel("Lens data")
        }
    }

    private var header: some View {
        HStack(spacing: Space.m) {
            Text("Lens Data").font(.headline)
            Text("\(system.surfaces.count - 2) surfaces · stop on \(system.stop) · mm")
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
            if !design.variables.isEmpty {
                Label("\(design.variables.count) variable\(design.variables.count == 1 ? "" : "s")", systemImage: "v.square")
                    .font(.caption).foregroundStyle(.tint)
            }
            GlassEffectContainer(spacing: Space.s) {
                HStack(spacing: Space.xs) {
                    Button("Insert Surface", systemImage: "plus") {
                        let i = window.selection.flatMap { system.surfaceIndex(of: $0) } ?? system.imageIndex
                        if let id = editor.insert(at: max(i, 1)) { window.selection = id }
                    }
                    .help("Insert a surface before the selection (⇧⌘I)")
                    Button("Delete Surface", systemImage: "minus") {
                        if let id = window.selection { window.selection = editor.delete(id) }
                    }
                    .help("Delete the selected surface (⌘⌫)")
                    .disabled(!canDelete)
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .controlSize(.small)
            }
        }
        .padding(.horizontal, Space.l)
        .padding(.vertical, Space.xs)
        .frame(minHeight: 36)
    }

    private var canDelete: Bool {
        guard let i = window.selection.flatMap({ system.surfaceIndex(of: $0) }) else { return false }
        return i > 0 && i < system.imageIndex && system.surfaces.count > 3
    }

    private func column(_ c: LensColumn, rows: [LensRow]) -> TableColumn<LensRow, Never, some View, Text> {
        TableColumn(c.title) { row in
            LensCell(
                row: row, column: c, focus: $focus,
                completions: c == .material ? workspace.materialNames : [],
                commit: { text in commit(row, c, text) },
                move: { dir in move(from: CellID(row: row.id, column: c), dir, rows: rows) })
        }
    }

    private func commit(_ row: LensRow, _ c: LensColumn, _ text: String) -> Bool {
        if let error = editor.commit(row, c, text: text) {
            window.flash(error)
            return false
        }
        return true
    }

    enum Move { case up, down, left, right }

    private func move(from cell: CellID, _ dir: Move, rows: [LensRow]) {
        guard let r = rows.firstIndex(where: { $0.id == cell.row }) else { return }
        let cols = LensColumn.editable
        switch dir {
        case .up, .down:
            var k = r + (dir == .up ? -1 : 1)
            while rows.indices.contains(k), !rows[k].editable.contains(cell.column) { k += dir == .up ? -1 : 1 }
            if rows.indices.contains(k) { focus = CellID(row: rows[k].id, column: cell.column) } else { focus = nil }
        case .left, .right:
            guard var c = cols.firstIndex(of: cell.column) else { return }
            repeat { c += dir == .left ? -1 : 1 } while cols.indices.contains(c) && !rows[r].editable.contains(cols[c])
            if cols.indices.contains(c) { focus = CellID(row: cell.row, column: cols[c]) }
        }
    }

    @ViewBuilder private func contextMenu(_ id: SurfaceID?) -> some View {
        if let id, let i = system.surfaceIndex(of: id) {
            Button("Insert Surface Before") { window.selection = editor.insert(at: max(i, 1)) }
            Button("Insert Surface After") { window.selection = editor.insert(at: min(i + 1, system.imageIndex)) }
            Button("Delete Surface") { window.selection = editor.delete(id) }
                .disabled(i == 0 || i == system.imageIndex || system.surfaces.count <= 3)
            Divider()
            Button("Make Aperture Stop") { editor.makeStop(id) }.disabled(i == 0 || i == system.imageIndex || i == system.stop)
            Menu("Variables") {
                ForEach([LensColumn.radius, .thickness, .conic], id: \.self) { c in
                    if let p = c.param(id), p.resolves(in: system) {
                        Toggle(c.title, isOn: Binding(get: { design.isVariable(p) }, set: { design.setVariable(p, $0, undoManager: undoManager) }))
                    }
                }
            }
            Divider()
            Button("Show in Inspector") {
                window.selection = id
                window.inspectorTab = .surface
                window.showInspector = true
            }
        }
    }
}

/// `OBJ`, `1`, `STO`, …, `IMA`; the stop gets the aperture symbol.
private struct SurfaceLabel: View {
    let row: LensRow
    let issues: [ValidationIssue]

    var body: some View {
        HStack(spacing: 3) {
            if let worst = issues.map(\.severity).min() {
                Image(systemName: worst == .error ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(worst == .error ? .red : .orange)
                    .help(issues.map(\.message).joined(separator: "\n"))
                    .accessibilityLabel(issues.map(\.message).joined(separator: ". "))
            }
            Text(row.surfaceText)
                .font(row.role == .surface ? .body.monospacedDigit() : .callout.weight(.semibold))
                .foregroundStyle(row.role == .object || row.role == .image ? .secondary : .primary)
            if row.role == .stop {
                Image(systemName: "camera.aperture").font(.caption).foregroundStyle(.tint)
                    .accessibilityLabel("aperture stop")
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(row.role == .stop ? "Surface \(row.index), aperture stop" : row.role == .object ? "Object" : row.role == .image ? "Image" : "Surface \(row.index)")
    }
}

/// The surface type, changeable from the grouped type menu.
private struct TypeCell: View {
    let row: LensRow
    let shape: SurfaceShape
    let editor: LensEditor

    var body: some View {
        if row.role == .object || row.role == .image {
            Text(row.typeTitle).foregroundStyle(.secondary)
        } else {
            TypeMenu(shape: shape, compact: true) { editor.setType($0, row.id) }
                .menuStyle(.borderlessButton)
                .menuIndicator(.visible)
                .labelStyle(.titleAndIcon)
        }
    }
}

/// An editable cell: shows the formatted value, edits the exact one.
private struct LensCell: View {
    let row: LensRow
    let column: LensColumn
    var focus: FocusState<CellID?>.Binding
    let completions: [String]
    let commit: (String) -> Bool
    let move: (LensTableView.Move) -> Void
    @State private var text = ""
    @State private var editing = false

    private var id: CellID { CellID(row: row.id, column: column) }
    private var display: String { row.text(column) }

    var body: some View {
        HStack(spacing: 3) {
            if let hint = row.hints[column] {
                Text(hint).font(.caption2.weight(.medium)).foregroundStyle(.tertiary)
                    .help("The \(hint) section's value (the surface is anamorphic; the other section is in the inspector)")
                    .accessibilityLabel("\(hint) section")
            }
            if row.editable.contains(column) {
                field
            } else {
                Text(display)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: column.isNumeric ? .trailing : .leading)
            }
            if let m = row.markers[column] { MarkerBadge(marker: m) }
        }
        .onAppear { text = display }
        .onChange(of: display) { _, d in if !editing { text = d } }
    }

    private var field: some View {
        TextField(column.title, text: $text, prompt: Text(""))
            .labelsHidden()
            .textFieldStyle(.plain)
            .multilineTextAlignment(column.isNumeric ? .trailing : .leading)
            .font(column.isNumeric ? .body.monospacedDigit() : .body)
            .foregroundStyle(dimmed ? .secondary : .primary)
            .focused(focus, equals: id)
            .onChange(of: focus.wrappedValue == id) { _, isFocused in
                if isFocused {
                    editing = true
                    text = row.exact[column] ?? display
                } else if editing {
                    editing = false
                    if !finish() { text = display }
                }
            }
            .onSubmit {
                if finish() { move(.down) }
            }
            .onKeyPress(.upArrow) { step(.up) }
            .onKeyPress(.downArrow) { step(.down) }
            .onKeyPress(.escape) {
                editing = false
                text = display
                focus.wrappedValue = nil
                return .handled
            }
            .materialSuggestions(column == .material && editing ? suggestions : [])
            .accessibilityLabel(column.title)
            .accessibilityValue(display)
    }

    /// Automatic semi-diameters (computed by the engine) are shown dimmed.
    private var dimmed: Bool { column == .semiDiameter && row.semiDiameterIsAuto && !editing }

    private var suggestions: [String] {
        let q = text.trimmingCharacters(in: .whitespaces).uppercased()
        guard q.count >= 1 else { return [] }
        return Array(completions.filter { $0.uppercased().hasPrefix(q) && $0.uppercased() != q }.prefix(12))
    }

    private func step(_ dir: LensTableView.Move) -> KeyPress.Result {
        guard finish() else { return .handled }
        move(dir)
        return .handled
    }

    /// Commits the text; false when it was rejected.
    private func finish() -> Bool {
        let exact = row.exact[column] ?? display
        guard text != exact, text != display else { return true }
        let ok = commit(text)
        if !ok { text = exact }
        return ok
    }
}

extension View {
    /// Glass-name completion for the material column.
    @ViewBuilder
    func materialSuggestions(_ names: [String]) -> some View {
        #if os(macOS)
        self.textInputSuggestions(names, id: \.self) { name in
            Text(name).textInputCompletion(name)
        }
        #else
        self
        #endif
    }
}

/// `V` (variable) or a solve letter next to a value.
struct MarkerBadge: View {
    let marker: CellMarker

    var body: some View {
        HStack(spacing: 2) {
            if let s = marker.solve {
                Text(s.letter)
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .foregroundStyle(.secondary)
                    .frame(width: 13, height: 13)
                    .background(RoundedRectangle(cornerRadius: 3).fill(.quaternary))
                    .help("Solve: \(s.title)")
                    .accessibilityLabel("controlled by a \(s.title) solve")
            }
            if marker.isVariable {
                Text("V")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .foregroundStyle(marker.bounds == .pinned ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tint))
                    .frame(width: 13, height: 13)
                    .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(.tint.opacity(0.6), lineWidth: 1))
                    // A bounded variable: a small bar under the V; pinned: a dot.
                    .overlay(alignment: .bottom) {
                        if marker.bounds == .bounded {
                            Capsule().fill(.tint).frame(width: 7, height: 1.5).offset(y: -1.5)
                        }
                    }
                    .overlay(alignment: .topTrailing) {
                        if marker.bounds == .pinned { Circle().fill(.tint).frame(width: 4, height: 4).offset(x: 1.5, y: -1.5) }
                    }
                    .help(marker.bounds == .pinned ? "Variable, pinned (equal bounds)" : marker.bounds == .bounded ? "Variable with bounds" : "Variable")
                    .accessibilityLabel(marker.bounds == .pinned ? "pinned variable" : marker.bounds == .bounded ? "bounded variable" : "variable")
            }
        }
    }
}
