import MKLightEngine
import MKLightModel
import SwiftUI

/// The window's bottom bar: engine and compute backend, the snapshot,
/// transient messages, and the design's key first-order numbers.
struct StatusBar: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let engine: any EngineClient

    var body: some View {
        HStack(spacing: Space.l) {
            if engine.info.isMock {
                Label("Mock engine", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .help("Results are synthetic: the engine library is not linked into this build.")
            }
            Menu {
                Picker("Compute On", selection: Binding(get: { workspace.backend }, set: { workspace.backend = $0 })) {
                    Text("Automatic").tag(BackendPreference.auto)
                    Text("CPU Reference (f64)").tag(BackendPreference.cpuReference)
                    Text("GPU").tag(BackendPreference.gpu)
                }
                .pickerStyle(.inline)
            } label: {
                Label(backendTitle, systemImage: workspace.backend == .gpu ? "cpu.fill" : "cpu")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Where analyses compute. Automatic uses the GPU where the engine has a path and it pays off.")
            if let snap = workspace.snapshot {
                Text(snap.hash.short)
                    .monospaced()
                    .foregroundStyle(.tertiary)
                    .help("Engine snapshot of the design (content hash \(snap.hash.hex))")
            }
            if !workspace.catalogWarnings.isEmpty {
                Label("\(workspace.catalogWarnings.count) catalog warning\(workspace.catalogWarnings.count == 1 ? "" : "s")", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .help(workspace.catalogWarnings.joined(separator: "\n"))
            }
            if let m = window.message {
                Text(m).foregroundStyle(.secondary).lineLimit(1).transition(.opacity)
            } else if let first = workspace.modelErrors.first {
                Label(first, systemImage: "xmark.octagon").foregroundStyle(.red).lineLimit(1)
            }
            Spacer(minLength: Space.l)
            KeyNumbers(workspace: workspace)
            if workspace.isBusy {
                ProgressView().controlSize(.mini).help("Computing")
            }
        }
        .font(.caption)
        .padding(.horizontal, Space.l)
        .frame(height: 24)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
        .animation(.default, value: window.message)
    }

    private var backendTitle: String {
        switch workspace.backend {
        case .auto: "Automatic"
        case .cpuReference: "CPU reference"
        case .gpu: "GPU"
        }
    }
}

/// EFL, working F/#, total track and image height from the engine's
/// first-order data.
struct KeyNumbers: View {
    let workspace: Workspace

    var body: some View {
        let entries = FirstOrderData(workspace.summary.output?.json)
        HStack(spacing: Space.l) {
            ForEach(Self.keys, id: \.key) { k in
                if let e = entries?.entry(k.candidates) {
                    HStack(spacing: 3) {
                        Text(k.label).foregroundStyle(.secondary)
                        Text(e.formatted).numeric()
                    }
                    .help(e.label)
                }
            }
        }
        .opacity(workspace.summary.isStale ? 0.45 : 1)
        .accessibilityElement(children: .combine)
    }

    static let keys: [(key: String, label: String, candidates: [String])] = [
        ("efl", "EFL", ["efl"]),
        ("fno", "F/#", ["working_f_number", "image_f_number", "f_number", "paraxial_working_f_number"]),
        ("tt", "TT", ["total_track", "track"]),
        ("img", "Image", ["image_height", "paraxial_image_height", "max_image_height"]),
    ]
}
