import AppKit
import Foundation
import SwiftUI
import Testing

@testable import MKLightEngine
@testable import MKLightModel
@testable import MKLightUI

/// The one commit rule of the app's text-entry fields (`FieldEdit`): only
/// edited text is committed, never a value equal to the current one, text
/// that was not edited follows the value, and edited text is replaced only
/// by its own commit or an explicit revert. Each test is a scenario the
/// rule exists for.
@Suite @MainActor struct FieldEditTests {
    /// A number: compact at rest, every digit with the focus.
    private let number = FieldEdit<Double>.Format(exact: FieldText.exact, display: Fmt.short, parse: { LensValueParser.number($0) })

    /// A list of numbers in the order typed.
    private let list = FieldEdit<[Double]>.Format(
        exact: { FrequencyListSetting.text($0, exact: true) }, display: { FrequencyListSetting.text($0) },
        parse: { FrequencyListSetting.numbers($0) })

    private func setting(_ name: String, _ schema: [String: JSONValue]) -> SettingsField {
        SettingsField(name: name, schema: .object(schema), required: false)
    }

    // MARK: The rule

    @Test func textAtRestIsCompactAndFocusedTextReadsBackExactly() {
        let third = 100.0 / 3
        var e = FieldEdit(third, number)
        #expect(e.text == "33.3333" && !e.isEdited)
        e.beginEditing(number)
        #expect(e.text == "33.333333333333336" && !e.isEdited)
        #expect(LensValueParser.number(e.text) == third)
    }

    /// After Return the focused field shows every digit, as before it:
    /// rounded text there would count as an edit and be committed when the
    /// focus leaves, though nothing was typed.
    @Test func returnThenBlurDoesNotRound() {
        let third = 100.0 / 3
        var e = FieldEdit(third, number)
        e.beginEditing(number)
        #expect(e.submit(number) == .unchanged)
        #expect(e.text == "33.333333333333336", "still every digit after Return")
        #expect(e.endEditing(number) == .unchanged)
        #expect(e.text == "33.3333" && e.value == third)

        // The same after a real edit: Return commits once, the blur nothing.
        e.beginEditing(number)
        e.text = "12.3456789012"
        #expect(e.submit(number) == .value(12.3456789012))
        #expect(e.text == "12.3456789012", "the focused field shows what was typed, not 12.3457")
        #expect(e.endEditing(number) == .unchanged)
        #expect(e.text == "12.3457" && e.value == 12.3456789012)
    }

    /// A visit writes nothing: a setting that shows its default is not
    /// stored as that default, rounded to six digits, by tabbing through.
    @Test func tabThroughAnUnsetSettingStoresNothing() {
        let field = setting("spacing", ["type": .string("number"), "default": .number(0.1234567891)])
        let unset = SettingNumberField(field: field, label: "", value: field.defaultValue, set: { _ in }).entry
        var e = FieldEdit(field.defaultValue, unset)
        #expect(e.text == "0.123457")
        e.beginEditing(unset)
        #expect(e.endEditing(unset) == .unchanged)
        #expect(e.text == "0.123457" && e.value == .number(0.1234567891))

        // Clearing a setting that is at its default changes nothing either.
        e.beginEditing(unset)
        e.text = ""
        #expect(e.endEditing(unset) == .unchanged)
        #expect(e.text == "0.123457")

        // Away from the default, clearing restores it (`nil`).
        let stored = SettingNumberField(field: field, label: "", value: .number(2), set: { _ in }).entry
        var s = FieldEdit(JSONValue?.some(.number(2)), stored)
        s.beginEditing(stored)
        s.text = " "
        #expect(s.endEditing(stored) == .value(nil))
    }

    @Test func aValueEqualToTheCurrentOneIsNotCommitted() {
        var e = FieldEdit(10, number)
        e.beginEditing(number)
        e.text = "1e1"
        #expect(e.isEdited)
        #expect(e.endEditing(number) == .unchanged)
        #expect(e.text == "10", "the text is the value's again")
    }

    /// A change from outside (undo, a preset, an optimiser result) while the
    /// field has the focus: text that was not edited follows, so that a
    /// later blur cannot write the old value back; the user's text stays.
    @Test func aValueChangedFromOutsideWhileFocused() {
        var e = FieldEdit(0.5875618, number)
        e.beginEditing(number)
        e.valueChanged(0.5460740, number)
        #expect(e.text == "0.546074" && !e.isEdited)
        #expect(e.endEditing(number) == .unchanged)

        e.beginEditing(number)
        e.text = "0.6"
        e.valueChanged(0.4861327, number)
        #expect(e.text == "0.6" && e.isEdited, "an edit in progress is not overwritten")
        #expect(e.endEditing(number) == .value(0.6))

        // Typed text is compared with the value as it is now.
        e.beginEditing(number)
        e.text = "0.7"
        e.valueChanged(0.7, number)
        #expect(e.endEditing(number) == .unchanged)
        #expect(e.text == "0.7" && e.value == 0.7)
    }

    /// The owner stores something else than was typed (a clamp), or nothing
    /// (a refused value): the field follows what the owner reports.
    @Test func theFieldFollowsWhatTheOwnerStored() {
        var e = FieldEdit(200, number)
        e.beginEditing(number)
        e.text = "500000"
        #expect(e.endEditing(number) == .value(500_000))
        e.valueChanged(100_000, number)
        #expect(e.text == "100000")

        e.beginEditing(number)
        e.text = "0"
        #expect(e.submit(number) == .value(0))
        e.valueChanged(100_000, number)
        #expect(e.text == "100000", "the refused 0 does not stay in the field")
        #expect(e.endEditing(number) == .unchanged)
    }

    @Test func blurWithUnparsableText() {
        var e = FieldEdit(5, number)
        e.beginEditing(number)
        e.text = "abc"
        #expect(e.endEditing(number) == .invalid("abc"))
        #expect(e.text == "abc" && e.value == 5, "the rule drops nothing by itself")

        // Policy "revert": the value shows again.
        var reverted = e
        reverted.revert(number)
        #expect(reverted.text == "5" && !reverted.isEdited)

        // Policy "keep": the text waits; a change of the value does not
        // replace it, and correcting it commits.
        e.valueChanged(6, number)
        #expect(e.text == "abc")
        e.beginEditing(number)
        #expect(e.text == "abc")
        #expect(e.submit(number) == .invalid("abc"))
        e.text = "7"
        #expect(e.endEditing(number) == .value(7))
        #expect(e.text == "7")
    }

    /// A list of numbers is stored as typed; sorting and removing repeats
    /// is for frequencies only.
    @Test func aListKeepsTheOrderTyped() {
        var e = FieldEdit([3.0, 1, 2], list)
        #expect(e.text == "3, 1, 2")
        e.beginEditing(list)
        #expect(e.endEditing(list) == .unchanged)
        e.beginEditing(list)
        e.text = "30, 10 20; 10"
        #expect(e.endEditing(list) == .value([30, 10, 20, 10]))
        #expect(e.text == "30, 10, 20, 10")
        e.beginEditing(list)
        e.text = "30, x"
        #expect(e.endEditing(list) == .invalid("30, x"))
    }

    /// The inspector keeps a field in place while it shows another surface:
    /// text typed for one surface must not be written to the next.
    @Test func textTypedForOneOwnerIsCommittedToThatOwner() {
        var e = FieldEdit(10, owner: "surface 1", number)
        e.beginEditing(number)
        e.text = "12"
        // The field now shows surface 2: the edit ends, for surface 1.
        #expect(e.valueChanged(20, owner: "surface 2", number) == .value(12))
        #expect(e.text == "20" && e.value == 20 && e.owner == AnyHashable("surface 2"))
        #expect(e.endEditing(number) == .unchanged, "nothing is left to write to surface 2")

        // Text that was not edited simply follows the new owner.
        e.beginEditing(number)
        #expect(e.valueChanged(10, owner: "surface 1", number) == .unchanged)
        #expect(e.text == "10")

        // Text that is no value is not carried over to the new owner.
        e.text = "abc"
        #expect(e.valueChanged(20, owner: "surface 2", number) == .invalid("abc"))
        #expect(e.text == "20" && !e.isEdited)

        // The same owner with a new value keeps the edit.
        e.text = "25"
        #expect(e.valueChanged(21, owner: "surface 2", number) == .unchanged)
        #expect(e.text == "25")
        #expect(e.endEditing(number) == .value(25))
    }

    // MARK: List and table fields

    /// The focus leaving commits what was typed, without Return; text that
    /// is no list is rejected, not read as an empty list.
    @Test func listFieldsCommitOnBlurAndRejectGarbage() {
        let operands = OperandIndexField.entry
        var o = FieldEdit([0, 2], operands)
        #expect(o.text == "1, 3")
        o.beginEditing(operands)
        o.text = "2, 4"
        #expect(o.endEditing(operands) == .value([1, 3]))
        o.beginEditing(operands)
        o.text = "x"
        #expect(o.endEditing(operands) == .invalid("x"), "not an empty list")
        o.text = ""
        #expect(o.endEditing(operands) == .value([]))

        let frequencies = FrequencyListField.entry
        var f = FieldEdit([10.0, 30], frequencies)
        f.beginEditing(frequencies)
        f.text = "40, 20"
        f.valueChanged([10, 30], frequencies)
        #expect(f.text == "40, 20", "a refresh of the values does not overwrite the text being typed")
        #expect(f.endEditing(frequencies) == .value([20, 40]))
        for bad in ["", "20, inf", "20, x", "-5"] {
            f.beginEditing(frequencies)
            f.text = bad
            #expect(f.endEditing(frequencies) == .invalid(bad))
            f.revert(frequencies)
        }
        #expect(f.text == "20, 40")

        let percentiles = PercentileListField.entry
        var p = FieldEdit([50, 97.72498680518208], percentiles)
        #expect(p.text == "50, 97.725")
        p.beginEditing(percentiles)
        #expect(p.endEditing(percentiles) == .unchanged)
        #expect(p.value == [50, 97.72498680518208], "a level is not rounded by a visit")
        p.beginEditing(percentiles)
        p.text = "90%, 50"
        #expect(p.endEditing(percentiles) == .value([50, 90]))
        p.beginEditing(percentiles)
        p.text = "50, 120"
        #expect(p.endEditing(percentiles) == .invalid("50, 120"))
    }

    /// A cell of a table setting commits by itself; the cells that were not
    /// edited keep their stored values, not what their text rounds to.
    @Test func tableCellsKeepStoredValues() {
        let columns = [setting("hx", ["type": .string("number")]), setting("n", ["type": .string("integer")])]
        let stored = JSONValue.array([.array([.number(0.1234567891), .integer(3)])])
        var rows = TableSettingEditor.rows(of: stored, columns: columns, rowsAreArrays: true)
        #expect(TableSettingEditor.value(of: rows, columns: columns, rowsAreArrays: true) == stored)

        let count = TableSettingEditor.entry(columns[1])
        var e = FieldEdit(rows[0].cells[1], count)
        e.beginEditing(count)
        e.text = "4.6"
        #expect(e.endEditing(count) == .value(.integer(5)))
        rows[0].cells[1] = e.value
        #expect(TableSettingEditor.value(of: rows, columns: columns, rowsAreArrays: true) == .array([.array([.number(0.1234567891), .integer(5)])]))

        // An emptied number leaves the row out until it is filled again.
        e.beginEditing(count)
        e.text = ""
        #expect(e.endEditing(count) == .value(nil))
        rows[0].cells[1] = e.value
        #expect(TableSettingEditor.value(of: rows, columns: columns, rowsAreArrays: true) == .array([]))
        e.beginEditing(count)
        e.text = "many"
        #expect(e.endEditing(count) == .invalid("many"))
    }

    // MARK: Special cases on top of the rule

    /// 0 asks the engine only where 0 is the automatic value; a through-
    /// focus bound (automatic when null) can be 0.
    @Test func zeroIsAutomaticOnlyForTheZeroSentinel() {
        let bound = setting("max", ["type": .array([.string("number"), .string("null")])])
        let nullAuto = SettingNumberField(field: bound, label: "", value: .number(0.1), auto: .null, set: { _ in }).entry
        var e = FieldEdit(JSONValue?.some(.number(0.1)), nullAuto)
        e.beginEditing(nullAuto)
        e.text = "0"
        #expect(e.endEditing(nullAuto) == .value(.number(0)))
        e.beginEditing(nullAuto)
        e.text = "auto"
        #expect(e.endEditing(nullAuto) == .value(.null))
        #expect(e.text == "", "an automatic value shows the placeholder")

        var grid = setting("grid", ["type": .string("integer")])
        grid.minimum = 0
        let zeroAuto = SettingNumberField(field: grid, label: "", value: .integer(128), auto: .zero, set: { _ in }).entry
        var g = FieldEdit(JSONValue?.some(.integer(128)), zeroAuto)
        g.beginEditing(zeroAuto)
        g.text = "0"
        #expect(g.endEditing(zeroAuto) == .value(.integer(0)))
        #expect(g.text == "")
        g.beginEditing(zeroAuto)
        g.text = "-7.4"
        #expect(g.endEditing(zeroAuto) == .unchanged, "held to the minimum, which is the automatic value it has")
    }

    /// One number of a pair, as `SettingPairField` reads it.
    private let box = PairEdit.Format(
        exact: { $0.map(FieldText.exact) ?? "" }, display: { $0.map(Fmt.short) ?? "" },
        parse: { LensValueParser.number($0).flatMap { $0.isFinite ? .some($0) : nil } })

    /// After a commit both boxes show the committed pair, so that leaving
    /// the second box has nothing to commit.
    @Test func aPairShowsWhatWasCommitted() {
        var p = PairEdit()
        p.valueChanged([1.0000001, 2], box)
        #expect(p.boxes.map(\.text) == ["1", "2"])
        #expect(p.focusChanged(to: 0, box) == .unchanged)
        #expect(p.boxes.map(\.text) == ["1.0000001", "2"], "only the focused box shows every digit")
        p.boxes[0].text = "3"
        #expect(p.focusChanged(to: 1, box) == .value([3, 2]))
        #expect(p.boxes.map(\.text) == ["3", "2"])
        #expect(p.focusChanged(to: nil, box) == .unchanged, "the second box has nothing to commit")
        #expect(p.boxes.map(\.text) == ["3", "2"])

        // A box that was not edited keeps its stored number.
        p.valueChanged([1.0000001, 2], box)
        _ = p.focusChanged(to: 1, box)
        p.boxes[1].text = "5"
        #expect(p.submit(box) == .value([1.0000001, 5]))
        #expect(p.focusChanged(to: nil, box) == .unchanged)

        // Text that is no number gives the edit up.
        _ = p.focusChanged(to: 0, box)
        p.boxes[0].text = "x"
        #expect(p.focusChanged(to: 1, box) == .unchanged)
        #expect(p.boxes[0].text == "1")
    }

    /// An automatic pair is entered one number at a time: the first waits
    /// for the second while the focus stays in the pair.
    @Test func halfAPairWaitsWhileThePairHasTheFocus() {
        var p = PairEdit(nullable: true)
        p.valueChanged([nil, nil], box)
        _ = p.focusChanged(to: 0, box)
        p.boxes[0].text = "0.25"
        #expect(p.submit(box) == .unchanged)
        #expect(p.focusChanged(to: 1, box) == .unchanged)
        #expect(p.boxes.map(\.text) == ["0.25", ""], "not reverted on the way to the other box")
        p.boxes[1].text = "-0.5"
        #expect(p.focusChanged(to: nil, box) == .value([0.25, -0.5]))
        #expect(p.boxes.map(\.text) == ["0.25", "-0.5"])

        // Both boxes emptied is the automatic state again.
        p.valueChanged([0.25, -0.5], box)
        _ = p.focusChanged(to: 0, box)
        p.boxes[0].text = ""
        _ = p.focusChanged(to: 1, box)
        p.boxes[1].text = " "
        #expect(p.focusChanged(to: nil, box) == .value(nil))
        #expect(p.boxes.map(\.text) == ["", ""])

        // Half a pair left behind is given up.
        p.valueChanged([nil, nil], box)
        _ = p.focusChanged(to: 0, box)
        p.boxes[0].text = "1"
        #expect(p.focusChanged(to: nil, box) == .unchanged)
        #expect(p.boxes.map(\.text) == ["", ""])
    }
}

/// `CommittingField` in a window: the focus, Return and the removal of a
/// field reach the rule, and its commits reach the owner. The fields are
/// driven through AppKit (first responder, field editor); the window is
/// never shown.
@Suite(.serialized) @MainActor struct CommittingFieldTests {
    @Observable @MainActor final class Model {
        var bound: Double? = 100.0 / 3
        var operands = [0, 2]
        var surface = SurfaceID(1)
        var thickness: [SurfaceID: Double] = [SurfaceID(1): 10, SurfaceID(2): 20]
        var kept = 0.1234567891
        var rejected: String?
        var pair: JSONValue? = .array([.number(1.0000001), .number(2)])
        var table: JSONValue? = .array([.array([.number(0.1234567891), .number(3)])])
        var showsTable = true
        var commits: [String] = []
    }

    static let number = FieldEdit<Double>.Format(exact: FieldText.exact, display: Fmt.short, parse: { LensValueParser.number($0) })

    struct Fields: View {
        let model: Model

        var body: some View {
            VStack {
                OptionalNumberField(title: "Bound", value: model.bound) { v in
                    model.commits.append("bound \(v.map(NumberDisplay.exact) ?? "none")")
                    model.bound = v
                }
                OperandIndexField(indices: model.operands) { v in
                    model.commits.append("operands \(v)")
                    model.operands = v
                }
                let surface = model.surface
                NumberField(title: "Thickness", value: model.thickness[surface], owner: surface) { v in
                    model.commits.append("thickness of \(surface.rawValue): \(NumberDisplay.exact(v))")
                    model.thickness[surface] = v
                }
                CommittingField(
                    value: model.kept, format: CommittingFieldTests.number, invalid: .keep, rejected: { model.rejected = $0 },
                    commit: { v in
                        model.commits.append("kept \(NumberDisplay.exact(v))")
                        model.kept = v
                    }
                ) { text, _ in
                    TextField("Kept", text: text)
                }
                let column = SettingsField(name: "x", schema: .object(["type": .string("number")]), required: false)
                SettingPairField(field: SettingsField(name: "extent", schema: .object(["type": .string("array")]), required: false), label: "Extent", value: model.pair) { v in
                    model.commits.append("pair \(JSONText.text(v))")
                    model.pair = v
                }
                if model.showsTable {
                    TableSettingEditor(title: "Table", help: "", columns: [column, column], rowsAreArrays: true, value: model.table) { v in
                        model.commits.append("table \(JSONText.text(v))")
                        model.table = v
                    }
                }
            }
            .frame(width: 400)
        }
    }

    /// The fields of `Fields` in a window.
    @MainActor final class Stage {
        let model = Model()
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 400, height: 400), styleMask: [.titled], backing: .buffered, defer: false)
        private let host: NSHostingView<Fields>

        init() {
            _ = NSApplication.shared
            host = NSHostingView(rootView: Fields(model: model))
            window.contentView = host
        }

        /// The texts of the editable fields, top to bottom.
        var texts: [String] { fields(host).map(\.stringValue) }

        private func fields(_ view: NSView) -> [NSTextField] {
            ((view as? NSTextField).flatMap { $0.isEditable ? [$0] : nil } ?? []) + view.subviews.flatMap(fields)
        }

        /// Waits for the views to reach `condition`.
        func until(_ what: Comment, _ condition: () -> Bool) async throws {
            for _ in 0..<500 where !condition() { try await Task.sleep(for: .milliseconds(20)) }
            #expect(condition(), what)
        }

        /// Moves the focus to field `index` (`nil`: out of the fields).
        func focus(_ index: Int?) {
            window.makeFirstResponder(index.map { fields(host)[$0] })
        }

        /// Replaces the focused field's text, as typing does.
        func type(_ text: String) throws {
            let editor = try #require(window.firstResponder as? NSTextView)
            editor.selectAll(nil)
            editor.insertText(text, replacementRange: editor.selectedRange())
        }

        func pressReturn() throws {
            try #require(window.firstResponder as? NSTextView).insertNewline(nil)
        }
    }

    @Test func commitsOnReturnAndWhenTheFocusLeaves() async throws {
        let stage = Stage(), model = stage.model
        try await stage.until("the fields show their values") { stage.texts == ["33.3333", "1, 3", "10", "0.123457", "1", "2", "0.123457", "3"] }

        // The focus and Return alone commit nothing and round nothing.
        stage.focus(0)
        try await stage.until("every digit with the focus") { stage.texts[0] == "33.333333333333336" }
        try stage.pressReturn()
        stage.focus(1)
        try await stage.until("compact again at rest") { stage.texts[0] == "33.3333" }
        #expect(model.commits.isEmpty && model.bound == 100.0 / 3)

        // Return commits once; leaving the field afterwards commits nothing more.
        stage.focus(0)
        try await stage.until("every digit with the focus") { stage.texts[0] == "33.333333333333336" }
        try stage.type("12.3456789012")
        try stage.pressReturn()
        try await stage.until("Return commits") { model.commits == ["bound 12.3456789012"] }
        #expect(stage.texts[0] == "12.3456789012")
        stage.focus(1)
        try await stage.until("compact at rest") { stage.texts[0] == "12.3457" }
        #expect(model.commits == ["bound 12.3456789012"])

        // Leaving a field commits what was typed, without Return.
        try stage.type("2, 4")
        stage.focus(0)
        try await stage.until("the focus leaving commits") { model.commits.last == "operands [1, 3]" }
        #expect(stage.texts[1] == "2, 4" && model.commits.count == 2)

        // Text that is no value reverts and commits nothing.
        try stage.type("abc")
        stage.focus(1)
        try await stage.until("reverted") { stage.texts[0] == "12.3457" }
        #expect(model.commits.count == 2)
    }

    @Test func textTypedForOneSurfaceIsCommittedToThatSurface() async throws {
        let stage = Stage(), model = stage.model
        try await stage.until("the fields show their values") { stage.texts.count == 8 && stage.texts[2] == "10" }
        stage.focus(2)
        try await stage.until("the field has the focus") { stage.texts[2] == "10.0" }
        try stage.type("12")
        try await stage.until("typed") { stage.texts[2] == "12" }
        // The inspector shows the next surface while the field keeps the focus.
        model.surface = SurfaceID(2)
        try await stage.until("the field shows the new surface") { stage.texts[2] == "20.0" }
        #expect(model.commits == ["thickness of 1: 12.0"])
        stage.focus(nil)
        try await stage.until("at rest") { stage.texts[2] == "20" }
        #expect(model.commits.count == 1 && model.thickness == [SurfaceID(1): 12, SurfaceID(2): 20])
    }

    /// The policy "keep": text that is no value stays to be corrected, and
    /// the field's error presentation is told.
    @Test func keptTextWaitsToBeCorrected() async throws {
        let stage = Stage(), model = stage.model
        try await stage.until("the fields show their values") { stage.texts.count == 8 && stage.texts[3] == "0.123457" }
        stage.focus(3)
        try await stage.until("every digit with the focus") { stage.texts[3] == "0.1234567891" }
        try stage.type("abc")
        stage.focus(0)
        try await stage.until("the field is told") { model.rejected == "abc" }
        #expect(stage.texts[3] == "abc" && model.commits.isEmpty)
        // New values arrive: the focused, unedited field follows its own.
        model.kept = 6
        model.bound = 2
        try await stage.until("the values arrived") { stage.texts[0] == "2.0" }
        #expect(stage.texts[3] == "abc" && model.rejected == "abc", "a new value does not replace the text")
        stage.focus(3)
        try await stage.until("the other field is at rest") { stage.texts[0] == "2" }
        try stage.type("7")
        stage.focus(0)
        try await stage.until("the corrected text commits") { model.commits == ["kept 7.0"] }
        #expect(model.rejected == nil && stage.texts[3] == "7")
    }

    /// The two boxes of a pair are one value: the first box commits when the
    /// focus moves to the second, and shows what was committed.
    @Test func aPairCommitsWhenTheFocusLeavesABox() async throws {
        let stage = Stage(), model = stage.model
        try await stage.until("the fields show their values") { stage.texts.count == 8 && stage.texts[4] == "1" }
        stage.focus(4)
        try await stage.until("every digit with the focus") { stage.texts[4] == "1.0000001" }
        try stage.type("3")
        stage.focus(5)
        try await stage.until("the pair is committed") { model.commits == ["pair [3,2]"] }
        stage.focus(nil)
        try await stage.until("at rest") { stage.texts[5] == "2" }
        #expect(stage.texts[4] == "3" && model.commits.count == 1)
    }

    @Test func aTableCellCommitsWhenItsEditorGoesAway() async throws {
        let stage = Stage(), model = stage.model
        try await stage.until("the fields show their values") { stage.texts.count == 8 }
        stage.focus(7)
        try stage.type("4")
        try await stage.until("typed") { stage.texts[7] == "4" }
        // A popover that closes takes its fields away without moving the focus.
        model.showsTable = false
        try await stage.until("the cell is committed") { !model.commits.isEmpty }
        #expect(model.commits == ["table [[0.1234567891,4]]"], "the cell that was not edited keeps every digit")
    }
}
