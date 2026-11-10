import Foundation

/// The state of one text field that edits a value, and the one rule by
/// which typed text becomes a value:
///
/// * only text the user edited is committed: text left as the field showed
///   it stands for the value it was shown for;
/// * a value equal to the current one is not committed;
/// * text that was not edited follows the value;
/// * edited text is replaced only by its own commit, or by `revert`.
///
/// The field shows `text`. The owner reports the value with `valueChanged`,
/// the focus with `beginEditing` and `endEditing`, Return with `submit`,
/// and writes what those return. No UI: `CommittingField` is the view.
struct FieldEdit<Value: Equatable> {
    /// How a value reads as text and text as a value.
    struct Format {
        /// The text while the field has the focus: it parses back to
        /// exactly the value.
        var exact: (Value) -> String
        /// The text at rest.
        var display: (Value) -> String
        /// The value of typed text; `nil` when the text is not one.
        var parse: (String) -> Value?
    }

    /// What the end of an edit asks of the owner.
    enum Commit: Equatable {
        /// Nothing to write: the text was not edited, or it is the current
        /// value.
        case unchanged
        /// Write this value: typed, and not the current one.
        case value(Value)
        /// The edited text is not a value. The field still shows it: the
        /// owner reverts it (`revert`) or leaves it to be corrected.
        case invalid(String)
    }

    /// The text the field shows. The user's typing sets it.
    var text: String
    /// The text of `value` as the field shows it: other text is an edit.
    private var base: String
    /// The current value: the reported one, or the one just committed.
    private(set) var value: Value
    /// Whose value it is, where one field shows several objects in turn.
    private(set) var owner: AnyHashable?
    private(set) var isFocused = false

    /// A field at rest that shows `value`.
    init(_ value: Value, owner: AnyHashable? = nil, _ format: Format) {
        self.value = value
        self.owner = owner
        text = format.display(value)
        base = text
    }

    /// Whether the text is the user's and not the value's.
    var isEdited: Bool { text != base }

    /// The field took the focus: text that was not edited shows every
    /// digit of the value.
    mutating func beginEditing(_ format: Format) {
        isFocused = true
        rebase(format)
    }

    /// The owner's value is `value` (of `owner`). Text that was not edited
    /// follows it; edited text stays, and is compared with the new value
    /// when it is committed. With another owner the edit ends here: the
    /// result is the commit of the text typed for the earlier owner (an
    /// invalid text of an earlier owner is given up), and the field shows
    /// the new one.
    @discardableResult
    mutating func valueChanged(_ value: Value, owner: AnyHashable? = nil, _ format: Format) -> Commit {
        var earlier = Commit.unchanged
        if owner != self.owner {
            earlier = submit(format)
            text = base
            self.owner = owner
        }
        self.value = value
        rebase(format)
        return earlier
    }

    /// Return: commits edited text; the field keeps the focus and shows
    /// every digit of what was typed.
    mutating func submit(_ format: Format) -> Commit {
        guard isEdited else { return .unchanged }
        guard let typed = format.parse(text) else { return .invalid(text) }
        let changed = typed != value
        value = typed
        base = shown(typed, format)
        text = base
        return changed ? .value(typed) : .unchanged
    }

    /// The focus left: commits edited text and shows the value at rest.
    /// Text that is not a value stays as typed.
    mutating func endEditing(_ format: Format) -> Commit {
        let commit = submit(format)
        isFocused = false
        rebase(format)
        return commit
    }

    /// Gives the edit up: the field shows the value again.
    mutating func revert(_ format: Format) {
        base = shown(value, format)
        text = base
    }

    private func shown(_ v: Value, _ format: Format) -> String {
        isFocused ? format.exact(v) : format.display(v)
    }

    /// Takes the value's text for the present focus as the base; text that
    /// was not edited becomes it.
    private mutating func rebase(_ format: Format) {
        let edited = isEdited
        base = shown(value, format)
        if !edited { text = base }
    }
}
