import MKLightEngine
import MKLightModel
import SwiftUI

/// The window's status in the toolbar's centre, one glass capsule: a
/// running design job or analysis activity, the design's key first-order
/// numbers (with the engine, backend and snapshot one click away) and its
/// validation state. It replaces a bottom status bar: macOS keeps critical
/// information out of the bottom of the window.
struct DesignStatus: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let engine: any EngineClient
    let system: OpticalSystem
    @State private var showDetails = false

    var body: some View {
        HStack(spacing: Space.m) {
            if let job = window.runningJob {
                Button {
                    window.open(job)
                } label: {
                    HStack(spacing: Space.xs) {
                        ProgressView().controlSize(.mini)
                        Text(jobStatus(job)).monospacedDigit()
                    }
                }
                .help("Show the running \(job.title.lowercased()) job")
            } else if workspace.isBusy {
                ProgressView().controlSize(.mini).help("Computing").accessibilityLabel("Computing")
            }
            Button {
                showDetails.toggle()
            } label: {
                KeyNumbers(workspace: workspace, system: system, isMock: engine.info.isMock)
            }
            .help("First-order data, engine and compute backend")
            .popover(isPresented: $showDetails, arrowEdge: .bottom) {
                StatusDetails(workspace: workspace, window: window, engine: engine, system: system).frame(width: 340)
            }
            IssuesButton(workspace: workspace, window: window)
        }
        .buttonStyle(.borderless)
        .font(.callout)
        .padding(.horizontal, Space.s)
    }

    private func jobStatus(_ job: DesignJobKind) -> String {
        switch job {
        case .optimize: window.optimize.trace.last.map { "Optimizing · \($0.iteration)" } ?? "Optimizing"
        case .tolerance: "Tolerancing"
        case .solves: "Solving"
        }
    }
}

/// EFL, working F/# and total track from the engine's first-order data.
/// An anamorphic design shows the y section's EFL, labelled as such.
struct KeyNumbers: View {
    let workspace: Workspace
    let system: OpticalSystem
    var isMock = false

    var body: some View {
        let entries = FirstOrderData(workspace.summary.output?.json)
        let xy = AnamorphicFirstOrder(workspace.summary.output?.json)
        HStack(spacing: Space.l) {
            if isMock {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    .accessibilityLabel("Mock engine: synthetic results")
            }
            if let xy, let x = xy.eflX, let y = xy.eflY {
                number("EFL x/y", Fmt.short(x) + " / " + Fmt.short(y) + " mm")
            } else if let e = entries?.entry(["efl"]) {
                number(system.hasNonRotationalSurfaces ? "EFL y" : "EFL", e.formatted)
            }
            if let e = entries?.entry(["working_f_number", "image_f_number", "f_number", "paraxial_working_f_number"]) {
                number("F/#", e.formatted)
            }
            if let e = entries?.entry(["total_track", "track"]) { number("TT", e.formatted) }
            if entries == nil && !workspace.modelErrors.isEmpty {
                Text("Design not ready").foregroundStyle(.secondary)
            } else if entries == nil {
                Text(system.name.isEmpty ? "Untitled" : system.name).foregroundStyle(.secondary)
            }
        }
        .opacity(workspace.summary.isStale ? 0.5 : 1)
        .accessibilityElement(children: .combine)
    }

    private func number(_ label: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(label).foregroundStyle(.secondary)
            Text(value).monospacedDigit().foregroundStyle(.primary)
        }
        .fixedSize()
    }
}

/// The status popover: first-order data, anamorphic data (or what the
/// engine does not report yet), engine and backend, snapshot, catalogs.
struct StatusDetails: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let engine: any EngineClient
    let system: OpticalSystem

    var body: some View {
        let entries = FirstOrderData(workspace.summary.output?.json)
        VStack(alignment: .leading, spacing: Space.l) {
            VStack(alignment: .leading, spacing: Space.s) {
                Text("First Order").font(.headline)
                if let entries {
                    KeyValueGrid(rows: Self.keys.compactMap { k in entries.entry(k.1).map { (k.0, $0.formatted) } })
                } else {
                    Text("No first-order data yet.").foregroundStyle(.secondary)
                }
                if system.hasNonRotationalSurfaces {
                    AnamorphicSummary(json: workspace.summary.output?.json)
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: Space.s) {
                Text("Engine").font(.headline)
                if engine.info.isMock {
                    Label("Mock engine: results are synthetic. The engine library is not linked into this build.", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                KeyValueGrid(rows: [("Engine", "\(engine.info.name) \(engine.info.version)")] + (workspace.snapshot.map { [("Snapshot", $0.hash.short)] } ?? []))
                Picker("Compute on", selection: Binding(get: { workspace.backend }, set: { workspace.backend = $0 })) {
                    Text("Automatic").tag(BackendPreference.auto)
                    Text("CPU reference (f64)").tag(BackendPreference.cpuReference)
                    Text("GPU").tag(BackendPreference.gpu)
                }
                .help("Where analyses compute. Automatic uses the GPU where the engine has a path and it pays off.")
            }
            if !workspace.catalogWarnings.isEmpty {
                Divider()
                ForEach(workspace.catalogWarnings, id: \.self) { w in
                    Label(w, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                }
            }
        }
        .padding(Space.l)
        .textSelection(.enabled)
    }

    static let keys: [(String, [String])] = [
        ("Effective focal length", ["efl"]), ("Back focal length", ["bfl"]),
        ("Working F/#", ["working_f_number", "image_f_number", "f_number", "paraxial_working_f_number"]),
        ("Entrance pupil diameter", ["epd"]), ("Total track", ["total_track", "track"]),
        ("Image height", ["image_height", "paraxial_image_height", "max_image_height"]),
    ]
}

/// EFL in x and y and the squeeze of an anamorphic design when the engine
/// reports them; otherwise a note that its first order covers the y
/// section only.
struct AnamorphicSummary: View {
    let json: JSONValue?

    var body: some View {
        if let xy = AnamorphicFirstOrder(json) {
            KeyValueGrid(rows: [
                ("EFL x", xy.eflX.map { Fmt.value($0, unit: "mm") } ?? "—"),
                ("EFL y", xy.eflY.map { Fmt.value($0, unit: "mm") } ?? "—"),
                ("Squeeze EFL y / x", xy.squeeze.map { String(format: "%.4f×", $0) } ?? "—"),
            ])
        } else {
            Label {
                Text("This design is not rotationally symmetric. The engine's first-order data covers the y section; EFL x and the squeeze are not reported yet.")
            } icon: {
                Image(systemName: "info.circle")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }
}
