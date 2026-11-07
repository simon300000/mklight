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
    /// The cell being edited: the only one shown as a text field, the
    /// others are text (fewer AppKit views to lay out and to expose).
    @State private var active: CellID?
    /// The last mouse-down on a row (not observed: it drives no view).
    @State private var press = RowPress()

    private var system: OpticalSystem { design.system }
    private var editor: LensEditor { LensEditor(document: design, undoManager: undoManager) }

    /// Issues by surface index.
    private var issues: [Int: [ValidationIssue]] {
        Dictionary(grouping: workspace.issues.filter { $0.surface != nil && $0.severity != .info }, by: { $0.surface! })
    }

    private var rows: [LensRow] {
        LensRow.rows(
            for: system, autoSemiDiameters: workspace.layout.autoSemiDiameters,
            variables: Set(design.variables.map { system.canonical($0.param) }),
            bounds: Dictionary(
                design.variables.map { (system.canonical($0.param), $0.isPinned ? .pinned : $0.isBounded ? .bounded : .free) }, uniquingKeysWith: { a, _ in a }))
    }

    /// The grid's own keyboard focus (row selection with ↑/↓, Return to edit).
    @FocusState private var gridFocused: Bool
    /// Column widths the user dragged in the header, by column.
    @State private var dragged: [Int: CGFloat] = [:]

    /// The columns: the surface label, the type, then the shared parameters.
    /// Anamorphic surfaces show both sections in the radius and conic cells.
    private func columns(hasOffsets: Bool) -> [LensGridColumn] {
        let base = [
            LensGridColumn(title: "Surf", min: 40, ideal: 46, max: 64),
            LensGridColumn(title: "Type", min: 90, ideal: 112, max: 150),
            LensGridColumn(title: LensColumn.comment.title, min: 70, ideal: 120),
            LensGridColumn(title: LensColumn.radius.title, min: 80, ideal: 116, max: 220),
            LensGridColumn(title: LensColumn.thickness.title, min: 80, ideal: 104, max: 160),
            LensGridColumn(title: LensColumn.material.title, min: hasOffsets ? 150 : 80, ideal: hasOffsets ? 168 : 104, max: 240),
            LensGridColumn(title: LensColumn.semiDiameter.title, min: 80, ideal: 104, max: 160),
            LensGridColumn(title: LensColumn.conic.title, min: 64, ideal: 88, max: 180),
        ]
        return base.enumerated().map { k, c in dragged[k].map { c.fixed(at: $0) } ?? c }
    }

    // The grid is plain SwiftUI rather than `Table`: a table hosts every
    // cell in its own AppKit view, re-tiled on each frame of a window or
    // sidebar resize, and builds cells for off-screen rows whenever an
    // accessibility client reads them. Here a resize is one layout pass.
    // The body does not read the selection: a click repaints only the
    // rows' highlight (`SelectedRow`), not the grid.
    var body: some View {
        let rows = self.rows
        let issues = self.issues
        let emphasized = gridFocused || focus != nil
        let columns = columns(hasOffsets: rows.contains { $0.notes[.material] != nil })
        VStack(spacing: 0) {
            LensTableHeader(design: design, window: window, editor: editor)
            ScrollView([.vertical, .horizontal]) {
                LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                    Section {
                        ForEach(Array(rows.enumerated()), id: \.element.id) { k, row in
                            gridRow(row, alternate: k % 2 == 1, columns: columns, rows: rows, issues: issues[row.index] ?? [], emphasized: emphasized)
                        }
                    } header: {
                        LensGridHeader(columns: columns, dragged: $dragged)
                    }
                }
                .padding(.horizontal, Space.m)
                .padding(.bottom, Space.s)
                .containerRelativeFrame(.horizontal) { length, _ in
                    max(length, LensGridLayout.minimumWidth(columns) + 2 * Space.m)
                }
            }
            .scrollEdgeEffectStyle(.hard, for: .top)
            .background(Color(nsColor: .controlBackgroundColor))
            .focusable()
            .focusEffectDisabled()
            .focused($gridFocused)
            .onKeyPress(.upArrow) { selectAdjacent(-1, rows: rows) }
            .onKeyPress(.downArrow) { selectAdjacent(1, rows: rows) }
            .onKeyPress(.return) {
                guard active == nil, let id = window.selection else { return .ignored }
                edit(id, rows: rows)
                return .handled
            }
            .onChange(of: focus) { _, f in
                if let f {
                    window.selection = f.row
                    window.focusedColumn = f.column
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Lens data")
        }
    }

    private func gridRow(
        _ row: LensRow, alternate: Bool, columns: [LensGridColumn], rows: [LensRow], issues: [ValidationIssue], emphasized: Bool
    ) -> some View {
        LensGridLayout(columns: columns) {
            SurfaceLabel(row: row, issues: issues).gridCell()
            TypeCell(row: row, shape: system.surfaces[row.index].shape, editor: editor).gridCell()
            column(.comment, row: row, rows: rows)
            column(.radius, row: row, rows: rows)
            column(.thickness, row: row, rows: rows)
            column(.material, row: row, rows: rows)
            column(.semiDiameter, row: row, rows: rows)
            column(.conic, row: row, rows: rows)
        }
        .frame(minHeight: 24)
        .modifier(SelectedRow(window: window, id: row.id, alternate: alternate, emphasized: emphasized))
        .contentShape(Rectangle())
        // Select on mouse-down, like a table (a tap fires on mouse-up).
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !press.isDown else { return }
                    press.down(row.id, wasSelected: window.selection == row.id)
                    select(row.id)
                }
                .onEnded { _ in press.isDown = false })
        // A double click edits the radius, unless it landed on a cell (which then edits).
        .simultaneousGesture(TapGesture(count: 2).onEnded { if active?.row != row.id { edit(row.id, rows: rows) } })
        .contextMenu { contextMenu(row.id) }
        .accessibilityElement(children: .contain)
    }

    private func select(_ id: SurfaceID) {
        if window.selection != id { window.selection = id }
        if active == nil && !gridFocused { gridFocused = true }
    }

    private func begin(_ cell: CellID?) {
        active = cell
        focus = cell
        if cell == nil { gridFocused = true }
    }

    /// Edits the row's radius, or its comment where it has none.
    private func edit(_ id: SurfaceID, rows: [LensRow]) {
        begin(CellID(row: id, column: rows.first { $0.id == id }?.editable.contains(.radius) == true ? .radius : .comment))
    }

    private func selectAdjacent(_ step: Int, rows: [LensRow]) -> KeyPress.Result {
        guard active == nil, !rows.isEmpty else { return .ignored }
        let i = window.selection.flatMap { id in rows.firstIndex { $0.id == id } }
        let next = i.map { min(max($0 + step, 0), rows.count - 1) } ?? (step > 0 ? 0 : rows.count - 1)
        window.selection = rows[next].id
        return .handled
    }

    @ViewBuilder private func column(_ c: LensColumn, row: LensRow, rows: [LensRow]) -> some View {
        let shown = row.shown(c)
        Group {
            if shown.count > 1 {
                // y and x sections side by side, each editable in place.
                HStack(spacing: Space.s) {
                    cell(row, shown[0], rows: rows)
                    cell(row, shown[1], rows: rows)
                }
            } else {
                cell(row, shown[0], rows: rows)
            }
        }
        .gridCell()
    }

    private func cell(_ row: LensRow, _ c: LensColumn, rows: [LensRow]) -> some View {
        let id = CellID(row: row.id, column: c)
        return LensCell(
            row: row, column: c, focus: $focus, isActive: active == id,
            completions: c == .material ? workspace.materialNames : [],
            // Like a table: the first click selects the row, the next edits.
            click: { if press.wasSelected(row.id) { begin(id) } },
            begin: { begin(id) },
            end: { if active == id { active = nil } },
            cancel: { begin(nil) },
            commit: { text in commit(row, c, text) },
            move: { dir in move(from: CellID(row: row.id, column: c), dir, rows: rows) })
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
            // An x section continues in the next row's x section, or its column.
            func target(_ row: LensRow) -> LensColumn? {
                row.editable.contains(cell.column) ? cell.column : row.editable.contains(cell.column.base) ? cell.column.base : nil
            }
            var k = r + (dir == .up ? -1 : 1)
            while rows.indices.contains(k), target(rows[k]) == nil { k += dir == .up ? -1 : 1 }
            begin(rows.indices.contains(k) ? target(rows[k]).map { CellID(row: rows[k].id, column: $0) } : nil)
        case .left, .right:
            guard var c = cols.firstIndex(of: cell.column) else { return }
            repeat { c += dir == .left ? -1 : 1 } while cols.indices.contains(c) && !rows[r].editable.contains(cols[c])
            if cols.indices.contains(c) { begin(CellID(row: cell.row, column: cols[c])) }
        }
    }

    /// "Radius", or "Radius y" / "Radius x" on anamorphic surfaces.
    private func variableTitle(_ c: LensColumn, _ s: Surface) -> String {
        guard s.shape.isAnamorphic else { return c.title }
        switch c {
        case .radius: return "Radius y"
        case .conic: return "Conic y"
        default: return c.title
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
                ForEach([LensColumn.radius, .radiusX, .thickness, .conic, .conicX], id: \.self) { c in
                    if let p = c.param(for: system.surfaces[i]), p.resolves(in: system) {
                        let refusal = design.isVariable(p) ? nil : design.variableRefusal(p)
                        Toggle(variableTitle(c, system.surfaces[i]), isOn: Binding(get: { design.isVariable(p) }, set: { design.setVariable(p, $0, undoManager: undoManager) }))
                            .disabled(refusal != nil)
                            .help(refusal ?? "")
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

/// The last mouse-down on a lens row: whether that row was already
/// selected, so the click's mouse-up edits the cell (second click) or not
/// (first click). A plain object: it is read in gestures, never in a body.
final class RowPress {
    private var row: SurfaceID?
    private var rowWasSelected = false
    var isDown = false

    func down(_ id: SurfaceID, wasSelected: Bool) {
        row = id
        rowWasSelected = wasSelected
        isDown = true
    }

    func wasSelected(_ id: SurfaceID) -> Bool { row == id && rowWasSelected }
}

/// A lens row's selection highlight. The modifier alone reads the
/// selection, so a click updates the two highlights, not the rows.
private struct SelectedRow: ViewModifier {
    let window: WindowState
    let id: SurfaceID
    let alternate: Bool
    let emphasized: Bool

    func body(content: Content) -> some View {
        let selected = window.selection == id
        content
            .background {
                if selected {
                    RoundedRectangle(cornerRadius: 5)
                        .fill(emphasized ? Color(nsColor: .selectedContentBackgroundColor) : Color(nsColor: .unemphasizedSelectedContentBackgroundColor))
                } else if alternate {
                    RoundedRectangle(cornerRadius: 5).fill(Color(nsColor: NSColor.alternatingContentBackgroundColors[1]))
                }
            }
            .environment(\.backgroundProminence, selected && emphasized ? .increased : .standard)
            .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// The lens table's title bar: counts and the insert / delete buttons.
private struct LensTableHeader: View {
    let design: DesignDocument
    let window: WindowState
    let editor: LensEditor
    @Environment(\.panelTabbed) private var tabbed

    private var system: OpticalSystem { design.system }

    var body: some View {
        HStack(spacing: Space.m) {
            if !tabbed {
                Text("Lens Data")
                    .font(.headline)
                    .help("Drag to move the lens data, or out of the window to open it in a window of its own")
                    .accessibilityAddTraits(.isHeader)
                    .panelHandle(.lensData)
            }
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

/// An editable cell: shows the formatted value, edits the exact one. Only
/// the cell being edited is a text field.
private struct LensCell: View {
    let row: LensRow
    let column: LensColumn
    var focus: FocusState<CellID?>.Binding
    let isActive: Bool
    let completions: [String]
    /// A click on the value (edits only on the row's second click).
    let click: () -> Void
    /// Edits the cell.
    let begin: () -> Void
    let end: () -> Void
    /// Leaves the cell, back to row selection.
    let cancel: () -> Void
    let commit: (String) -> Bool
    let move: (LensTableView.Move) -> Void
    @State private var text = ""
    /// The text last committed, so leaving the field does not commit it again.
    @State private var committed: String?

    private var id: CellID { CellID(row: row.id, column: column) }
    private var display: String { row.text(column) }

    var body: some View {
        HStack(spacing: 3) {
            if let hint = row.hints[column] {
                // A section of an anamorphic surface: the axis right before its value.
                Spacer(minLength: 0)
                Text(hint).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                    .help(hintHelp(hint))
                    .accessibilityHidden(true)
            }
            if row.editable.contains(column) {
                if row.notes[column] != nil && !isActive {
                    // The glass name in full; the melt offset after it.
                    value.fixedSize(horizontal: true, vertical: false)
                } else if row.hints[column] != nil {
                    value.fixedSize(horizontal: true, vertical: false).frame(minWidth: 14, alignment: .trailing)
                } else {
                    value
                }
            } else {
                Text(display)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: column.isNumeric ? .trailing : .leading)
            }
            if let note = row.notes[column], !isActive {
                Text(note)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .help("Melt offset Δn_d / ΔV_d on the catalogue glass; edit it in the inspector")
                    .accessibilityLabel("offset \(note)")
            }
            if let m = row.markers[column] { MarkerBadge(marker: m) }
        }
    }

    @ViewBuilder private var value: some View {
        if isActive {
            field
        } else {
            Text(display)
                .font(column.isNumeric ? .body.monospacedDigit() : .body)
                .foregroundStyle(dimmed ? .secondary : .primary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: column.isNumeric ? .trailing : .leading)
                .contentShape(Rectangle())
                .onTapGesture(perform: click)
                .accessibilityLabel(accessibilityTitle)
                .accessibilityValue(display)
                .accessibilityAddTraits(.isButton)
                // Pressed by an assistive app (no mouse-down on the row): edit.
                .accessibilityAction(.default, begin)
                .accessibilityAction(named: "Edit", begin)
        }
    }

    private var field: some View {
        TextField(column.title, text: $text, prompt: Text(""))
            .labelsHidden()
            .textFieldStyle(.plain)
            .multilineTextAlignment(column.isNumeric ? .trailing : .leading)
            .font(column.isNumeric ? .body.monospacedDigit() : .body)
            .focused(focus, equals: id)
            .onAppear {
                text = row.exact[column] ?? display
                committed = nil
                // The field replacing another one can lose the focus to the
                // grid as the old field goes away: claim it once more.
                focus.wrappedValue = id
                DispatchQueue.main.async { if focus.wrappedValue != id { focus.wrappedValue = id } }
            }
            // Leaving the field (a click elsewhere) commits it.
            .onChange(of: focus.wrappedValue == id) { _, isFocused in
                if !isFocused { end() }
            }
            .onDisappear { _ = finish() }
            .onSubmit {
                if finish() { move(.down) }
            }
            .onKeyPress(.upArrow) { step(.up) }
            .onKeyPress(.downArrow) { step(.down) }
            // ⇧Tab arrives as a back-tab character.
            .onKeyPress(keys: [.tab, KeyEquivalent("\u{19}")]) { press in
                step(press.modifiers.contains(.shift) || press.key != .tab ? .left : .right)
            }
            .onKeyPress(.escape) {
                text = row.exact[column] ?? display
                cancel()
                return .handled
            }
            .materialSuggestions(column == .material ? suggestions : [])
            .accessibilityLabel(accessibilityTitle)
            .accessibilityValue(display)
    }

    /// Automatic semi-diameters (computed by the engine) and a cylinder's
    /// flat section are shown dimmed.
    private var dimmed: Bool { (column == .semiDiameter && row.semiDiameterIsAuto) || row.isFlatSection(column) }

    private func hintHelp(_ hint: String) -> String {
        let what = column.base == .radius ? "radius" : "conic"
        if row.isFlatSection(column) { return "The \(hint) section's \(what): flat (this cylinder has power in the other section)" }
        return "The \(hint) section's \(what) (the surface is anamorphic: y and x sections)"
    }

    /// The cell's name for VoiceOver: "Radius, x section".
    private var accessibilityTitle: String {
        guard let hint = row.hints[column] else { return column.title }
        return "\(column.base.title), \(hint) section"
    }

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
        guard text != exact, text != display, text != committed else { return true }
        let ok = commit(text)
        if ok { committed = text } else { text = exact }
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
            if marker.isVariable && marker.solve != nil {
                // A solve sets it: the optimizer leaves the variable out.
                Text("V")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .foregroundStyle(.orange)
                    .frame(width: 13, height: 13)
                    .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(.orange.opacity(0.6), lineWidth: 1))
                    .help("Variable on a solved value: the optimizer leaves it out. Remove the variable or the solve.")
                    .accessibilityLabel("variable left out: a solve sets it")
            } else if marker.isVariable {
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
