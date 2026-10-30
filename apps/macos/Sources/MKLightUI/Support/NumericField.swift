import MKLightModel
import SwiftUI

/// A text field for one number. It commits on Return or focus loss, rejects
/// invalid input (beep, red outline, reverts), and shows the model value
/// whenever it is not being edited.
struct NumericField: View {
    let title: String
    let value: Double
    var unit: String = ""
    var parse: (String) -> Result<Double, LensValueParser.Failure> = LensValueParser.finite
    let commit: (Double) -> Void

    @State private var text = ""
    @State private var error: String?
    @FocusState private var focused: Bool

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: 4) {
                TextField(title, text: $text)
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .monospacedDigit()
                    .focused($focused)
                    .onSubmit(apply)
                    .overlay(
                        RoundedRectangle(cornerRadius: 4).stroke(Color.red, lineWidth: error == nil ? 0 : 1.5)
                    )
                    .help(error ?? "")
                if !unit.isEmpty {
                    Text(unit).foregroundStyle(.secondary).frame(minWidth: 18, alignment: .leading)
                }
            }
        }
        .onAppear { text = formatted }
        .onChange(of: value) { if !focused { text = formatted } }
        .onChange(of: focused) { _, now in if !now { apply() } }
    }

    private var formatted: String {
        if value.isInfinite { return value > 0 ? "Infinity" : "-Infinity" }
        if value == value.rounded() && abs(value) < 1e15 { return String(Int64(value)) }
        return "\(value)"  // shortest text that round-trips
    }

    private func apply() {
        switch parse(text) {
        case .success(let v):
            error = nil
            if v != value { commit(v) }
            text = v.isInfinite ? (v > 0 ? "Infinity" : "-Infinity") : text
        case .failure(let f):
            NSSound.beep()
            error = f.description
            text = formatted
        }
    }
}
