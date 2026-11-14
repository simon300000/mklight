import MKLightEngine
import MKLightModel
import SwiftUI

/// Edits one user glass: name and catalog, a model glass or a dispersion
/// formula, range, physical data and status. The engine checks every
/// change and shows the n_d and V_d it computes.
struct GlassEditorSheet: View {
    @Bindable var editor: GlassEditorModel
    let engine: (any EngineClient)?
    let save: () -> Bool
    let cancel: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section {
                    TextField("Name", text: $editor.name)
                    Picker("Catalog", selection: $editor.catalog) {
                        ForEach(editor.library.userCatalogs) { Text($0.name).tag($0.name) }
                    }
                    .disabled(editor.originalName != nil)
                }
                Section {
                    Picker("Dispersion", selection: $editor.kind) {
                        ForEach(GlassEditorModel.Kind.allCases) { Text($0.title).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    switch editor.kind {
                    case .model:
                        number("nd", value: $editor.nd)
                        number("Vd", value: $editor.vd)
                        number("ΔPg,F", value: $editor.dpgf)
                    case .formula:
                        Picker("Formula", selection: $editor.formula) {
                            ForEach(GlassEditorModel.formulaNames, id: \.0) { f in Text("\(f.0): \(f.1)").tag(f.0) }
                        }
                        let names = GlassEditorModel.coefficientNames(editor.formula)
                        ForEach(names.indices, id: \.self) { i in
                            number(names[i], value: Binding(
                                get: { i < editor.coefficients.count ? editor.coefficients[i] : 0 },
                                set: { if i < editor.coefficients.count { editor.coefficients[i] = $0 } }))
                        }
                    }
                } header: {
                    Text("Dispersion")
                } footer: {
                    Text(editor.kind == .model
                        ? "A Cauchy fit through nd, the F–C dispersion and ΔPg,F from the normal line (valid 0.35–1.1 µm unless set below)."
                        : "AGF formula coefficients, λ in µm.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Range and Data") {
                    optionalNumber("Shortest λ (µm)", value: $editor.minUm)
                    optionalNumber("Longest λ (µm)", value: $editor.maxUm)
                    optionalNumber("Density (g/cm³)", value: $editor.density)
                    optionalNumber("TCE (10⁻⁶/K)", value: $editor.tce)
                    optionalNumber("Relative cost", value: $editor.relativeCost)
                }
                Section("Availability") {
                    Picker("Status", selection: $editor.status) {
                        ForEach(GlassStatus.choices) { Text($0.title).tag($0) }
                    }
                    Toggle("Exclude from glass substitution", isOn: $editor.excludeSubstitution)
                    TextField("Comment", text: $editor.comment, axis: .vertical).lineLimit(1...3)
                }
            }
            .formStyle(.grouped)
            Divider()
            HStack(spacing: Space.l) {
                status
                Spacer()
                Button("Cancel", role: .cancel, action: cancel).keyboardShortcut(.cancelAction)
                Button("Save") { _ = save() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.glassProminent)
                    .disabled(editor.preview == nil)
            }
            .padding(Space.xl)
        }
        .frame(width: 480, height: 640)
        .navigationTitle(editor.title)
        .task(id: editor.spec) {
            // Checked after a short pause, not on every keystroke.
            try? await Task.sleep(for: .milliseconds(150))
            if let engine { editor.check(engine: engine) }
        }
    }

    @ViewBuilder private var status: some View {
        if let p = editor.problem {
            Label(p, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .font(.callout)
                .lineLimit(2)
        } else if let n = editor.previewNumbers {
            Text(n.nd > 0 ? String(format: "nd %.5f   Vd %.2f", n.nd, n.vd) : "No nd, Vd (the d line is outside the range)")
                .font(.callout)
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
    }

    /// Up to 12 significant digits, with the app's number format (a decimal point).
    private static let format = FloatingPointFormatStyle<Double>.number.precision(.significantDigits(1...12)).grouping(.never)
        .locale(NumberDisplay.locale)

    private func number(_ title: String, value: Binding<Double>) -> some View {
        TextField(title, value: value, format: Self.format)
            .monospacedDigit()
            .multilineTextAlignment(.trailing)
    }

    private func optionalNumber(_ title: String, value: Binding<Double?>) -> some View {
        TextField(title, value: value, format: Self.format, prompt: Text("none"))
            .monospacedDigit()
            .multilineTextAlignment(.trailing)
    }
}
