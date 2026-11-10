import SwiftUI

/// What a field does with typed text that is not a value.
enum InvalidText {
    /// The field shows the value again.
    case revert
    /// The text stays in the field, to be corrected.
    case keep
}

/// A text field that edits a value by the rule of `FieldEdit`: it shows
/// the value's display text at rest and its exact text with the focus,
/// commits edited text on Return and when the focus leaves, and follows
/// the value while the text is not edited.
///
/// `field` builds the text field itself (title, prompt, style, size) from
/// the text and whether it has the focus; the value, its format and the
/// commit are the owner's.
struct CommittingField<Value: Equatable, Field: View>: View {
    private let value: Value
    private let owner: AnyHashable?
    private let format: FieldEdit<Value>.Format
    private let invalid: InvalidText
    private let endsOnDisappear: Bool
    private let rejected: (String?) -> Void
    private let commit: (Value) -> Void
    private let field: (Binding<String>, Bool) -> Field
    @State private var edit: FieldEdit<Value>
    /// The commit of the owner the text is being typed for.
    @State private var typed: ((Value) -> Void)?
    @FocusState private var focused: Bool

    /// - Parameters:
    ///   - owner: The object the value belongs to, where the field stays in
    ///     place while it shows another object: text typed for one object
    ///     is committed to that object.
    ///   - invalid: What happens to text that is not a value.
    ///   - endsOnDisappear: Also commit when the field goes away with an
    ///     edit in it (a popover that closes). Only where `commit` is safe
    ///     once the edited object is gone.
    ///   - rejected: The field's error presentation: called with the text
    ///     that is not a value, and with `nil` when the field is in order
    ///     again.
    init(
        value: Value, owner: AnyHashable? = nil, format: FieldEdit<Value>.Format, invalid: InvalidText = .revert,
        endsOnDisappear: Bool = false, rejected: @escaping (String?) -> Void = { _ in }, commit: @escaping (Value) -> Void,
        @ViewBuilder field: @escaping (_ text: Binding<String>, _ focused: Bool) -> Field
    ) {
        self.value = value
        self.owner = owner
        self.format = format
        self.invalid = invalid
        self.endsOnDisappear = endsOnDisappear
        self.rejected = rejected
        self.commit = commit
        self.field = field
        _edit = State(initialValue: FieldEdit(value, owner: owner, format))
    }

    var body: some View {
        field(text, focused)
            .focused($focused)
            .onSubmit {
                sync()
                deliver(edit.submit(format))
            }
            .onChange(of: focused) { _, isFocused in
                sync()
                if isFocused { edit.beginEditing(format) } else { deliver(edit.endEditing(format)) }
            }
            .onAppear(perform: sync)
            .onDisappear { if endsOnDisappear { deliver(edit.endEditing(format)) } }
            .onChange(of: value) { _, _ in
                sync()
                // A field at rest that shows the new value is in order.
                if !focused, !edit.isEdited { rejected(nil) }
            }
            .onChange(of: owner) { _, _ in sync() }
    }

    /// The field's text. Typing notes the commit of the owner the field
    /// shows at that moment.
    private var text: Binding<String> {
        Binding(
            get: { edit.text },
            set: { typed in
                edit.text = typed
                if owner != nil { self.typed = commit }
            })
    }

    /// Reports the value the field is given now. Text typed for the owner
    /// the field showed before goes to that owner.
    private func sync() {
        if case .value(let v) = edit.valueChanged(value, owner: owner, format) { typed?(v) }
    }

    private func deliver(_ result: FieldEdit<Value>.Commit) {
        switch result {
        case .unchanged:
            rejected(nil)
        case .value(let v):
            rejected(nil)
            commit(v)
        case .invalid(let text):
            if invalid == .revert { edit.revert(format) }
            rejected(text)
        }
    }
}
