import MKLightEngine
import MKLightModel
import SwiftUI

/// A labelled number in an inspector form: shows the formatted value, edits
/// the exact one, accepts `∞` where the parser does, and commits on Return
/// or when focus leaves. An optional `V` toggle makes it a variable.
struct NumberField: View {
    let title: String
    /// What the value means, shown after the title (e.g. `r⁴` after `A₄`).
    var detail: String = ""
    let value: Double?
    var unit: String = ""
    var placeholder: String = ""
    var help: String = ""
    var parse: (String) -> Result<Double, LensValueParser.Failure> = LensValueParser.finite
    var variable: FieldVariable?
    /// The title of the solve that sets the value (an `S` badge).
    var solve: String?
    let commit: (Double) -> Void
    @State private var text = ""
    @State private var error: String?
    @FocusState private var focused: Bool

    var body: some View {
        LabeledContent {
            HStack(spacing: Space.xs) {
                TextField(title, text: $text, prompt: Text(placeholder))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .font(.body.monospacedDigit())
                    .focused($focused)
                    .onSubmit(finish)
                    .onChange(of: focused) { _, f in
                        if f { text = value.map(NumberDisplay.exact) ?? "" } else { finish() }
                    }
                if !unit.isEmpty {
                    Text(unit).foregroundStyle(.secondary).frame(minWidth: 22, alignment: .leading)
                }
                if let solve {
                    Text("S").font(.system(size: 9, weight: .bold, design: .rounded)).foregroundStyle(.secondary)
                        .frame(width: 13, height: 13).background(RoundedRectangle(cornerRadius: 3).fill(.quaternary))
                        .help("Controlled by a \(solve) solve")
                }
                if let variable {
                    VariableToggle(isOn: variable.isOn, solve: variable.solve)
                }
            }
        } label: {
            HStack(spacing: Space.xs) {
                Text(title)
                if !detail.isEmpty { Text(detail).foregroundStyle(.secondary).font(.callout) }
            }
            .help(help)
        }
        .onAppear { text = formatted }
        .onChange(of: value) { _, _ in
            if !focused {
                text = formatted
                error = nil
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if let error { Text(error).font(.caption2).foregroundStyle(.red).offset(y: 12) }
        }
    }

    private var formatted: String { value.map(Fmt.short) ?? "" }

    private func finish() {
        let exact = value.map(NumberDisplay.exact) ?? ""
        guard text != exact, text != formatted else {
            error = nil
            return
        }
        switch parse(text) {
        case .success(let v):
            error = nil
            commit(v)
        case .failure(let f):
            error = f.description
            text = exact
        }
    }
}

/// Whether a value is a variable, and the solve that sets it, if any.
struct FieldVariable {
    var isOn: Binding<Bool>
    var solve: String?
}

/// The `V` toggle beside a value. A parameter a solve sets cannot become a
/// variable (the optimizer applies solves first and refuses it): the toggle
/// is struck through and says why; a variable left on such a parameter
/// shows in orange and a click removes it.
struct VariableToggle: View {
    @Binding var isOn: Bool
    /// The solve's title when a solve sets the parameter.
    var solve: String?

    private var refused: Bool { solve != nil && !isOn }

    var body: some View {
        Button {
            if !refused { isOn.toggle() }
        } label: {
            Text("V")
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .frame(width: 14, height: 14)
                .foregroundStyle(style)
                .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(border, lineWidth: 1))
                .overlay {
                    if refused {
                        Rectangle().fill(.tertiary).frame(width: 16, height: 1).rotationEffect(.degrees(-45))
                    }
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
        .accessibilityLabel("Variable")
        .accessibilityValue(isOn ? "on" : "off")
        .accessibilityHint(help)
    }

    private var style: AnyShapeStyle {
        if isOn { return solve == nil ? AnyShapeStyle(.tint) : AnyShapeStyle(.orange) }
        return AnyShapeStyle(.tertiary)
    }

    private var border: AnyShapeStyle {
        guard isOn else { return AnyShapeStyle(.quaternary) }
        return solve == nil ? AnyShapeStyle(.tint.opacity(0.7)) : AnyShapeStyle(.orange.opacity(0.7))
    }

    private var help: String {
        if let solve {
            return isOn
                ? "Set by a solve (\(solve)): the optimizer leaves this variable out. Click to remove it."
                : "Set by a solve (\(solve)), so it cannot be a variable: the optimizer applies solves first. Remove the solve to vary it."
        }
        return isOn ? "Variable: the optimizer may change it. Click to fix it." : "Fixed. Click to make it a variable."
    }
}

/// A labelled text value committed on Return or focus loss.
struct TextValueField: View {
    let title: String
    let value: String
    var prompt: String = ""
    var suggestions: [String] = []
    let commit: (String) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        LabeledContent(title) {
            TextField(title, text: $text, prompt: Text(prompt))
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .focused($focused)
                .onSubmit { if text != value { commit(text) } }
                .onChange(of: focused) { _, f in if !f && text != value { commit(text) } }
                .materialSuggestions(focused ? matches : [])
        }
        .onAppear { text = value }
        .onChange(of: value) { _, v in if !focused { text = v } }
    }

    private var matches: [String] {
        let q = text.uppercased()
        guard !q.isEmpty, !suggestions.isEmpty else { return [] }
        return Array(suggestions.filter { $0.uppercased().hasPrefix(q) && $0.uppercased() != q }.prefix(12))
    }
}
