import MKLightModel
import SwiftUI

/// A diffractive surface's order efficiency model (SURFACES §3): ideal
/// (all light in the trace order), a continuous kinoform blaze, or an
/// N-level staircase, with a readout of the trace order's efficiency at the
/// design wavelength and, estimated without the relief's dispersion, at the
/// system wavelengths. The engine traces with its own model; these figures
/// are for orientation.
struct DiffractionEfficiencySection: View {
    let model: DiffractionEfficiency
    let phase: DiffractivePhase
    let wavelengths: [Double]
    let commit: (DiffractionEfficiency) -> Void

    private enum Kind: String, CaseIterable { case ideal, kinoform, multilevel }

    private var kind: Kind {
        switch model {
        case .ideal: .ideal
        case .kinoform: .kinoform
        case .multilevel: .multilevel
        }
    }

    private var levels: Int { if case .multilevel(let n) = model { n } else { 8 } }

    var body: some View {
        Section {
            Picker("Model", selection: Binding(get: { kind }, set: set)) {
                Text("Ideal").tag(Kind.ideal)
                Text("Kinoform").tag(Kind.kinoform)
                Text("Multilevel").tag(Kind.multilevel)
            }
            .help("Ideal: every photon in the trace order. Kinoform: a continuous blazed relief. Multilevel: a staircase of N etch levels.")
            if case .multilevel(let n) = model {
                Stepper(value: Binding(get: { n }, set: { commit(.multilevel(levels: max(1, min($0, 256)))) }), in: 1...256) {
                    LabeledContent("Levels", value: "\(n)")
                }
                .help("Number of etch levels of the staircase (2 binary, 4, 8, 16 …).")
            }
            LabeledContent("Trace order at λ₀") {
                Text(Fmt.percent(efficiency(at: phase.designWavelengthUm), digits: 1)).monospacedDigit()
            }
            .help("Efficiency of the trace order (\(phase.traceOrder)) at the design wavelength \(Fmt.short(phase.designWavelengthUm)) µm.")
            if model != .ideal && !wavelengths.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(wavelengths.enumerated()), id: \.offset) { _, um in
                        let e = efficiency(at: um)
                        HStack(spacing: Space.s) {
                            Text("\(Fmt.short(um)) µm").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                .frame(width: 62, alignment: .leading)
                            GeometryReader { g in
                                ZStack(alignment: .leading) {
                                    Capsule().fill(.quaternary)
                                    Capsule().fill(Palette.wavelengthColor(um)).frame(width: max(2, g.size.width * e))
                                }
                            }
                            .frame(height: 5)
                            Text(Fmt.percent(e, digits: 1)).font(.caption.monospacedDigit()).frame(width: 46, alignment: .trailing)
                        }
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("\(Fmt.short(um)) micrometres")
                        .accessibilityValue(Fmt.percent(e, digits: 1))
                    }
                }
                .padding(.vertical, 2)
            }
        } header: {
            Text("Diffraction Efficiency")
        } footer: {
            if model != .ideal {
                Text("System wavelengths estimated with blaze m₀λ₀/λ, ignoring the relief material's dispersion. The engine traces the exact model.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    /// The trace order's efficiency at `um`, the relief blazed for the
    /// design order at λ₀ (non-dispersive estimate away from λ₀).
    func efficiency(at um: Double) -> Double {
        guard um > 0 else { return .nan }
        let blaze = Double(phase.designOrder) * phase.designWavelengthUm / um
        return model.efficiency(order: phase.traceOrder, selected: phase.traceOrder, blaze: blaze)
    }

    private func set(_ k: Kind) {
        switch k {
        case .ideal: commit(.ideal)
        case .kinoform: commit(.kinoform)
        case .multilevel: commit(.multilevel(levels: levels))
        }
    }
}
