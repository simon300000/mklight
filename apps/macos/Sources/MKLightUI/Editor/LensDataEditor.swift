import AppKit
import MKLightEngine
import MKLightModel
import SwiftUI

/// Spreadsheet-like lens data editor, backed by `NSTableView` for proper
/// keyboard navigation:
///
/// * ↑/↓ select a surface; ←/→ (or Tab/⇧Tab) move the focused cell.
/// * Return starts editing the focused cell (or opens its menu); typing a
///   character starts editing with that character.
/// * While editing, Return commits and moves down, Tab/⇧Tab commit and move
///   across, Escape cancels.
/// * Delete removes the selected surface, Insert or `+` inserts one before it.
///
/// Invalid input is rejected with a beep and a message; the model is only
/// changed through `onEdit`.
struct LensDataEditor: NSViewRepresentable {
    let system: OpticalSystem
    let layout: LayoutScene?
    @Binding var selection: SurfaceID?
    /// Applies an edit; returns whether the model changed.
    let onEdit: (LensEdit) -> Bool
    let onMessage: (String?) -> Void

    func makeCoordinator() -> LensTableController { LensTableController() }

    func makeNSView(context: Context) -> NSScrollView {
        let c = context.coordinator
        let table = LensTableView()
        table.controller = c
        table.style = .fullWidth
        table.usesAlternatingRowBackgroundColors = true
        table.gridStyleMask = [.solidVerticalGridLineMask]
        table.rowHeight = 22
        table.intercellSpacing = NSSize(width: 0, height: 0)
        table.allowsMultipleSelection = false
        table.allowsEmptySelection = true
        table.columnAutoresizingStyle = .lastColumnOnlyAutoresizingStyle
        for col in LensColumn.allCases {
            let tc = NSTableColumn(identifier: NSUserInterfaceItemIdentifier(col.rawValue))
            tc.title = col.title
            tc.width = col.width
            tc.minWidth = 36
            tc.headerCell.alignment = col.isNumeric ? .right : .left
            table.addTableColumn(tc)
        }
        table.dataSource = c
        table.delegate = c
        table.menu = c.makeContextMenu()
        c.table = table

        let scroll = NSScrollView()
        scroll.documentView = table
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = true
        scroll.autohidesScrollers = true
        return scroll
    }

    func updateNSView(_ view: NSScrollView, context: Context) {
        let c = context.coordinator
        c.onEdit = onEdit
        c.onMessage = onMessage
        c.onSelect = { selection = $0 }
        c.coatingNames = system.coatings.keys.sorted()
        c.update(rows: LensRow.rows(for: system, layout: layout), selection: selection)
    }
}

/// Text field that shows formatted values and switches to the full-precision
/// value while it is being edited.
final class LensTextField: NSTextField {
    var row = -1
    var column = LensColumn.label
    var exactText: String?
    var displayText = ""

    override func becomeFirstResponder() -> Bool {
        let ok = super.becomeFirstResponder()
        if ok, isEditable, let exact = exactText {
            stringValue = exact
            currentEditor()?.selectAll(nil)
        }
        return ok
    }
}

final class LensCellView: NSTableCellView {
    var isFocusCell = false {
        didSet { needsDisplay = true }
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard isFocusCell else { return }
        NSColor.keyboardFocusIndicatorColor.setStroke()
        let path = NSBezierPath(roundedRect: bounds.insetBy(dx: 1.5, dy: 1.5), xRadius: 3, yRadius: 3)
        path.lineWidth = 2
        path.stroke()
    }
}

final class LensTableView: NSTableView {
    weak var controller: LensTableController?

    override func keyDown(with event: NSEvent) {
        if controller?.handleKey(event) == true { return }
        super.keyDown(with: event)
    }

    override func becomeFirstResponder() -> Bool {
        let ok = super.becomeFirstResponder()
        controller?.refreshFocus()
        return ok
    }

    override func resignFirstResponder() -> Bool {
        let ok = super.resignFirstResponder()
        DispatchQueue.main.async { [weak self] in self?.controller?.refreshFocus() }
        return ok
    }

    override func mouseDown(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        let col = column(at: p)
        super.mouseDown(with: event)
        if col >= 0, let c = LensColumn(rawValue: tableColumns[col].identifier.rawValue) {
            controller?.focus(column: c)
        }
    }
}

@MainActor
final class LensTableController: NSObject, NSTableViewDataSource, NSTableViewDelegate, NSTextFieldDelegate {
    weak var table: LensTableView?
    var rows: [LensRow] = []
    var onEdit: (LensEdit) -> Bool = { _ in false }
    var onMessage: (String?) -> Void = { _ in }
    var onSelect: (SurfaceID?) -> Void = { _ in }
    var coatingNames: [String] = []
    private(set) var focusedColumn = LensColumn.radius
    private var updatingSelection = false
    private var pendingEdit: (row: Int, column: LensColumn)?
    private var suppressEndEditing = false

    private static let editable: [LensColumn] = LensColumn.allCases.filter { $0 != .index }

    // MARK: Model → table

    func update(rows newRows: [LensRow], selection: SurfaceID?) {
        guard let table else { return }
        if newRows.count != rows.count {
            rows = newRows
            table.reloadData()
        } else if newRows != rows {
            let changed = IndexSet(newRows.indices.filter { newRows[$0] != rows[$0] })
            rows = newRows
            table.reloadData(forRowIndexes: changed, columnIndexes: IndexSet(integersIn: 0..<table.numberOfColumns))
        }
        let wanted = selection.flatMap { id in rows.firstIndex { $0.id == id } } ?? -1
        if wanted != table.selectedRow {
            updatingSelection = true
            if wanted >= 0 {
                table.selectRowIndexes(IndexSet(integer: wanted), byExtendingSelection: false)
                table.scrollRowToVisible(wanted)
            } else {
                table.deselectAll(nil)
            }
            updatingSelection = false
        }
        refreshFocus()
        if let p = pendingEdit {
            pendingEdit = nil
            DispatchQueue.main.async { [weak self] in self?.beginEditing(row: p.row, column: p.column) }
        }
    }

    func numberOfRows(in tableView: NSTableView) -> Int { rows.count }

    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        guard let id = tableColumn?.identifier, let col = LensColumn(rawValue: id.rawValue), rows.indices.contains(row) else {
            return nil
        }
        let r = rows[row]
        let cell = (tableView.makeView(withIdentifier: id, owner: nil) as? LensCellView) ?? makeCell(col, id: id)
        cell.isFocusCell = row == tableView.selectedRow && col == focusedColumn && tableView.window?.firstResponder === tableView
        if col.isPopup, let popup = cell.subviews.first as? NSPopUpButton {
            configure(popup, row: r, column: col)
        } else if let field = cell.textField as? LensTextField {
            let (display, exact) = r.text(col)
            field.row = row
            field.column = col
            field.displayText = display
            field.exactText = exact
            field.stringValue = display
            field.isEditable = r.isEditable(col)
            field.textColor =
                (col == .semiDiameter && r.semiDiameterIsAuto) || !r.isEditable(col) && col != .index
                ? .secondaryLabelColor : .labelColor
            field.font =
                col == .index
                ? .monospacedDigitSystemFont(ofSize: NSFont.smallSystemFontSize + 1, weight: r.role == .stop ? .bold : .regular)
                : (col.isNumeric ? .monospacedDigitSystemFont(ofSize: NSFont.systemFontSize - 1, weight: .regular) : .systemFont(ofSize: NSFont.systemFontSize - 1))
        }
        return cell
    }

    private func makeCell(_ col: LensColumn, id: NSUserInterfaceItemIdentifier) -> LensCellView {
        let cell = LensCellView()
        cell.identifier = id
        let content: NSView
        if col.isPopup {
            let popup = NSPopUpButton(frame: .zero, pullsDown: false)
            popup.isBordered = false
            popup.controlSize = .small
            popup.font = .systemFont(ofSize: NSFont.systemFontSize - 1)
            popup.target = self
            popup.action = #selector(popupChanged(_:))
            content = popup
        } else {
            let field = LensTextField(string: "")
            field.isBordered = false
            field.drawsBackground = false
            field.lineBreakMode = .byTruncatingTail
            field.alignment = col.isNumeric ? .right : .left
            field.delegate = self
            field.cell?.sendsActionOnEndEditing = false
            cell.textField = field
            content = field
        }
        content.translatesAutoresizingMaskIntoConstraints = false
        cell.addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 4),
            content.trailingAnchor.constraint(equalTo: cell.trailingAnchor, constant: -4),
            content.centerYAnchor.constraint(equalTo: cell.centerYAnchor),
        ])
        return cell
    }

    private func configure(_ popup: NSPopUpButton, row r: LensRow, column: LensColumn) {
        popup.removeAllItems()
        popup.tag = r.index
        popup.identifier = NSUserInterfaceItemIdentifier(column.rawValue)
        switch column {
        case .type:
            for k in SurfaceShape.Kind.allCases {
                popup.addItem(withTitle: k.title)
                popup.lastItem?.representedObject = k.rawValue
            }
            popup.selectItem(withTitle: r.shapeKind.title)
        default:
            popup.addItem(withTitle: "None")
            for n in coatingNames { popup.addItem(withTitle: n) }
            popup.selectItem(withTitle: r.coating ?? "None")
        }
        popup.isEnabled = r.isEditable(column)
    }

    @objc private func popupChanged(_ sender: NSPopUpButton) {
        guard rows.indices.contains(sender.tag), let id = sender.identifier, let col = LensColumn(rawValue: id.rawValue) else { return }
        let r = rows[sender.tag]
        switch col {
        case .type:
            if let raw = sender.selectedItem?.representedObject as? String, let k = SurfaceShape.Kind(rawValue: raw) {
                _ = onEdit(.setShapeKind(r.id, k))
            }
        default:
            let title = sender.titleOfSelectedItem
            _ = onEdit(.setCoating(r.id, title == "None" ? nil : title))
        }
        table?.window?.makeFirstResponder(table)
    }

    // MARK: Selection and focus

    func tableViewSelectionDidChange(_ notification: Notification) {
        guard !updatingSelection, let table else { return }
        let row = table.selectedRow
        onSelect(rows.indices.contains(row) ? rows[row].id : nil)
        refreshFocus()
    }

    func focus(column: LensColumn) {
        guard column != .index else { return }
        focusedColumn = column
        refreshFocus()
    }

    func refreshFocus() {
        guard let table else { return }
        let active = table.window?.firstResponder === table
        for row in 0..<table.numberOfRows {
            for (ci, tc) in table.tableColumns.enumerated() {
                guard let cell = table.view(atColumn: ci, row: row, makeIfNecessary: false) as? LensCellView else { continue }
                let focus = active && row == table.selectedRow && tc.identifier.rawValue == focusedColumn.rawValue
                if cell.isFocusCell != focus { cell.isFocusCell = focus }
            }
        }
    }

    private func moveFocus(by delta: Int) {
        let cols = Self.editable
        guard let i = cols.firstIndex(of: focusedColumn) else { return }
        focusedColumn = cols[(i + delta + cols.count) % cols.count]
        refreshFocus()
        if let table, let ci = columnIndex(focusedColumn) { table.scrollColumnToVisible(ci) }
    }

    private func columnIndex(_ c: LensColumn) -> Int? {
        table?.tableColumns.firstIndex { $0.identifier.rawValue == c.rawValue }
    }

    // MARK: Keyboard

    func handleKey(_ event: NSEvent) -> Bool {
        guard let table else { return false }
        let row = table.selectedRow
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        guard let chars = event.charactersIgnoringModifiers, let scalar = chars.unicodeScalars.first else { return false }
        let key = Int(scalar.value)
        switch key {
        case NSLeftArrowFunctionKey:
            moveFocus(by: -1)
            return true
        case NSRightArrowFunctionKey:
            moveFocus(by: 1)
            return true
        case 0x09:  // Tab
            moveFocus(by: 1)
            return true
        case 0x19:  // Shift-Tab
            moveFocus(by: -1)
            return true
        case 0x0D, 0x03:  // Return, Enter
            if row >= 0 { beginEditing(row: row, column: focusedColumn) }
            return true
        case 0x7F, NSDeleteFunctionKey:
            // Structural deletes need the explicit command (⌘⌫, also in the
            // menu). A plain Backspace/Delete clears the focused cell into
            // an edit, as in a spreadsheet; Escape restores it.
            if row >= 0, flags.contains(.command), flags.isDisjoint(with: [.option, .control]) {
                deleteSurface(at: row)
                return true
            }
            if row >= 0, flags.isDisjoint(with: [.command, .option, .control]) {
                if !focusedColumn.isPopup, beginEditing(row: row, column: focusedColumn), let editor = table.currentEditor() {
                    editor.selectAll(nil)
                    editor.delete(nil)
                }
                return true
            }
            return false
        case NSInsertFunctionKey:
            insertSurface(at: max(row, 1))
            return true
        default:
            break
        }
        // Type-to-edit: a printable character (including `+`, `-` and `.`,
        // which start numbers) starts editing the focused cell.
        if flags.isDisjoint(with: [.command, .control, .option]), row >= 0, let ch = event.characters,
            let first = ch.unicodeScalars.first, first.value >= 0x20, first.value < 0xF700,
            !focusedColumn.isPopup
        {
            if beginEditing(row: row, column: focusedColumn) {
                table.currentEditor()?.insertText(ch)
            }
            return true
        }
        return false
    }

    @discardableResult
    func beginEditing(row: Int, column: LensColumn) -> Bool {
        guard let table, rows.indices.contains(row), let ci = columnIndex(column) else { return false }
        focusedColumn = column
        if table.selectedRow != row { table.selectRowIndexes(IndexSet(integer: row), byExtendingSelection: false) }
        table.scrollRowToVisible(row)
        table.scrollColumnToVisible(ci)
        guard rows[row].isEditable(column) else {
            NSSound.beep()
            return false
        }
        if column.isPopup {
            (table.view(atColumn: ci, row: row, makeIfNecessary: true)?.subviews.first as? NSPopUpButton)?.performClick(nil)
            return false
        }
        table.editColumn(ci, row: row, with: nil, select: true)
        return table.currentEditor() != nil
    }

    // MARK: Editing

    func control(_ control: NSControl, textView: NSTextView, doCommandBy selector: Selector) -> Bool {
        guard let field = control as? LensTextField else { return false }
        if selector == #selector(NSResponder.cancelOperation(_:)) {
            suppressEndEditing = true
            field.stringValue = field.displayText
            table?.window?.makeFirstResponder(table)
            suppressEndEditing = false
            return true
        }
        return false
    }

    func controlTextDidEndEditing(_ notification: Notification) {
        guard !suppressEndEditing, let field = notification.object as? LensTextField, rows.indices.contains(field.row) else { return }
        let row = rows[field.row], column = field.column
        let movement = (notification.userInfo?["NSTextMovement"] as? Int).flatMap(NSTextMovement.init(rawValue:)) ?? .other
        let text = field.stringValue
        var changed = false
        if text != field.exactText, let parsed = row.edit(column, text: text) {
            switch parsed {
            case .success(let edit):
                onMessage(nil)
                changed = onEdit(edit)
            case .failure(let f):
                NSSound.beep()
                onMessage("Surface \(row.index) \(column.title): \(f.description)")
                field.stringValue = field.displayText
                pendingEdit = (field.row, column)
                DispatchQueue.main.async { [weak self] in self?.beginPendingEdit() }
                return
            }
        } else {
            field.stringValue = field.displayText
        }
        table?.window?.makeFirstResponder(table)

        var next: (Int, LensColumn)?
        switch movement {
        case .tab, .backtab:
            let cols = Self.editable
            if let i = cols.firstIndex(of: column) {
                next = (field.row, cols[(i + (movement == .tab ? 1 : -1) + cols.count) % cols.count])
            }
        case .return:
            if field.row + 1 < rows.count {
                focusedColumn = column
                table?.selectRowIndexes(IndexSet(integer: field.row + 1), byExtendingSelection: false)
            }
        default:
            break
        }
        if let (r, c) = next {
            focusedColumn = c
            pendingEdit = (r, c)
            // When the model changed, `update` begins the edit after the
            // reload; otherwise begin it now.
            if !changed { DispatchQueue.main.async { [weak self] in self?.beginPendingEdit() } }
        }
        refreshFocus()
    }

    private func beginPendingEdit() {
        guard let p = pendingEdit else { return }
        pendingEdit = nil
        beginEditing(row: p.row, column: p.column)
    }

    // MARK: Structure

    func insertSurface(at index: Int) {
        guard !rows.isEmpty else { return }
        let i = min(max(index, 1), rows.count - 1)
        if onEdit(.insert(at: i)) {
            DispatchQueue.main.async { [weak self] in
                guard let self, let table = self.table, self.rows.indices.contains(i) else { return }
                table.selectRowIndexes(IndexSet(integer: i), byExtendingSelection: false)
            }
        }
    }

    func deleteSurface(at index: Int) {
        guard rows.indices.contains(index) else { return }
        if rows[index].role == .object || rows[index].role == .image || !onEdit(.delete(at: index)) {
            NSSound.beep()
        }
    }

    func makeContextMenu() -> NSMenu {
        let m = NSMenu()
        m.addItem(withTitle: "Insert Surface Before", action: #selector(menuInsertBefore), keyEquivalent: "").target = self
        m.addItem(withTitle: "Insert Surface After", action: #selector(menuInsertAfter), keyEquivalent: "").target = self
        m.addItem(withTitle: "Delete Surface", action: #selector(menuDelete), keyEquivalent: "").target = self
        m.addItem(.separator())
        m.addItem(withTitle: "Make Stop", action: #selector(menuMakeStop), keyEquivalent: "").target = self
        return m
    }

    private var clickedRow: Int { table.map { $0.clickedRow >= 0 ? $0.clickedRow : $0.selectedRow } ?? -1 }

    @objc private func menuInsertBefore() { insertSurface(at: max(clickedRow, 1)) }
    @objc private func menuInsertAfter() { insertSurface(at: clickedRow + 1) }
    @objc private func menuDelete() { deleteSurface(at: clickedRow) }
    @objc private func menuMakeStop() {
        if !onEdit(.makeStop(at: clickedRow)) { NSSound.beep() }
    }
}
