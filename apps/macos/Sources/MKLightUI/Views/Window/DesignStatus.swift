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
            GPUPreparationBadge()
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
        case .glass: window.glass.progress.map { "Glasses · \($0.combinations)" } ?? "Substituting glasses"
        case .tolerance: "Tolerancing"
        case .solves: "Solving"
        }
    }
}

/// The GPU's kernels being prepared at launch, with their progress, then a
/// short confirmation. Hidden when they load from the cache at once.
struct GPUPreparationBadge: View {
    private var monitor = GPUPreparationMonitor.shared

    var body: some View {
        switch monitor.display {
        case .hidden:
            EmptyView()
        case .preparing:
            let percent = Int((monitor.state.fraction * 100).rounded())
            HStack(spacing: Space.xs) {
                ProgressView().controlSize(.mini)
                Text("Preparing GPU \(percent)%").monospacedDigit().foregroundStyle(.secondary)
            }
            .fixedSize()
            .help("Compiling the GPU's compute kernels, once after installing or updating MKLight or macOS. A GPU analysis started now waits only for the kernels it needs.")
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Preparing GPU")
            .accessibilityValue("\(percent) percent")
        case .finished:
            // Text, not a Label: in the toolbar a Label shows only its icon,
            // which would read as the issues button's checkmark.
            Text("GPU ready")
                .foregroundStyle(.secondary)
                .fixedSize()
        }
    }
}

/// EFL, working F/# and total track from the engine's first-order data.
/// A design that is not rotationally symmetric (the engine's `symmetry`)
/// shows the y section's EFL, labelled as such, and its squeeze.
struct KeyNumbers: View {
    let workspace: Workspace
    let system: OpticalSystem
    var isMock = false

    var body: some View {
        let json = workspace.summary.output?.json
        let entries = FirstOrderData(json)
        let anamorphic = ParaxialXYData.isAnamorphic(json, system: system)
        HStack(spacing: Space.l) {
            if isMock {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    .accessibilityLabel("Mock engine: synthetic results")
            }
            if anamorphic, let p = ParaxialXYData(json)?.primary {
                number("EFL y", Fmt.value(p.eflY, unit: "mm"))
                number("Squeeze", ParaxialXYData.squeezeText(p.squeeze, digits: 2))
            } else if let e = entries?.entry(["efl"]) {
                number(anamorphic ? "EFL y" : "EFL", e.formatted)
            }
            if let e = entries?.entry(["working_f_number", "image_f_number"]) { number("F/#", e.formatted) }
            if let e = entries?.entry(["total_track"]) { number("TT", e.formatted) }
            if entries == nil && (!workspace.modelErrors.isEmpty || workspace.issues.contains { $0.severity == .error }) {
                Text("Design has errors").foregroundStyle(.secondary)
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

/// The status popover: first-order data, the x / y data and symmetry of an
/// anamorphic design, engine and backend, snapshot, catalogs.
struct StatusDetails: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let engine: any EngineClient
    let system: OpticalSystem

    var body: some View {
        let json = workspace.summary.output?.json
        let entries = FirstOrderData(json)
        let anamorphic = ParaxialXYData.isAnamorphic(json, system: system)
        VStack(alignment: .leading, spacing: Space.l) {
            VStack(alignment: .leading, spacing: Space.s) {
                Text("First Order").font(.headline)
                if let entries {
                    // An anamorphic design's report is the y section's.
                    KeyValueGrid(rows: Self.keys.compactMap { k in
                        entries.entry(k.1).map { (anamorphic && Self.ySection.contains(k.0) ? k.0 + " (y)" : k.0, $0.formatted) }
                    })
                } else {
                    Text("No first-order data yet.").foregroundStyle(.secondary)
                }
                if anamorphic {
                    Divider()
                    AnamorphicSummary(json: json)
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: Space.s) {
                Text("Engine").font(.headline)
                if engine.info.isMock {
                    Label("Mock engine: results are synthetic. The engine library is not linked into this build.", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                KeyValueGrid(rows: [("Engine", "\(engine.info.name) \(engine.info.version)")]
                    + (workspace.snapshot.map { [("Snapshot", $0.hash.short)] } ?? [])
                    + (GPUPreparationMonitor.shared.summary.map { [("GPU kernels", $0)] } ?? []))
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

    /// The rows and the engine's `first_order` keys they show (the first one present).
    static let keys: [(String, [String])] = [
        ("Effective focal length", ["efl"]), ("Back focal length", ["bfl"]), ("Working F/#", ["working_f_number", "image_f_number"]),
        ("Entrance pupil diameter", ["entrance_pupil_diameter"]), ("Total track", ["total_track"]),
        ("Image height", ["paraxial_image_height"]),
    ]

    /// The rows that are the y section's for an anamorphic design.
    static let ySection: Set<String> = ["Effective focal length", "Back focal length", "Working F/#", "Image height"]
}

/// The x and y first order of a design that is not rotationally symmetric,
/// from the engine's `paraxial_xy`: EFL and BFL per section, the signed
/// squeeze (with its definition as help) and the symmetry class at the
/// primary wavelength. `detailed` adds DOE powers, the engine's warnings
/// and the other wavelengths (in a disclosure).
struct AnamorphicSummary: View {
    let xy: ParaxialXYData?
    var detailed = false
    @State private var showOthers = false

    init(json: JSONValue?, detailed: Bool = false) {
        xy = ParaxialXYData(json)
        self.detailed = detailed
    }

    var body: some View {
        if let xy, let p = xy.primary {
            VStack(alignment: .leading, spacing: Space.s) {
                table(p, xy)
                if detailed {
                    Text(String(format: "At λ = %.4f µm (primary)", p.wavelengthUm))
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    if !p.doePower.isEmpty { DOEPowerGrid(row: p) }
                    ForEach(xy.warnings, id: \.self) { w in
                        Label(w, systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if !xy.others.isEmpty {
                        DisclosureGroup(isExpanded: $showOthers) {
                            XYWavelengthTable(rows: xy.others).padding(.top, Space.xs)
                        } label: {
                            Text("Other wavelengths (\(xy.others.count))").font(.callout)
                        }
                    }
                }
            }
        } else {
            Label {
                Text("This design is not rotationally symmetric. The engine's first-order data covers the y section.")
            } icon: {
                Image(systemName: "info.circle")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func table(_ p: ParaxialXYData.Row, _ xy: ParaxialXYData) -> some View {
        Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: Space.xs) {
            GridRow {
                Color.clear.gridCellUnsizedAxes([.horizontal, .vertical])
                Text("x (mm)").gridColumnAlignment(.trailing)
                Text("y (mm)").gridColumnAlignment(.trailing)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .accessibilityHidden(true)
            xyRow("EFL", "Effective focal length", p.eflX, p.eflY)
            xyRow("BFL", "Back focal length", p.bflX, p.bflY)
            GridRow {
                Text("Squeeze").foregroundStyle(.secondary)
                Text(ParaxialXYData.squeezeText(p.squeeze)).numeric().gridCellColumns(2).gridColumnAlignment(.trailing)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
            .help("Squeeze: " + (xy.squeezeDefinition ?? "signed EFL y / EFL x"))
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Squeeze, EFL y over EFL x")
            .accessibilityValue(ParaxialXYData.squeezeText(p.squeeze))
            if let s = xy.symmetry {
                GridRow {
                    Text("Symmetry").foregroundStyle(.secondary)
                    Text(s.title).gridCellColumns(2).frame(maxWidth: .infinity, alignment: .trailing)
                }
                .accessibilityElement(children: .combine)
            }
        }
        .font(.callout)
        .textSelection(.enabled)
    }

    private func xyRow(_ label: String, _ spoken: String, _ x: Double, _ y: Double) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(Fmt.short(x)).numeric().lineLimit(1).fixedSize()
            Text(Fmt.short(y)).numeric().lineLimit(1).fixedSize()
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(spoken)
        .accessibilityValue("x \(Fmt.value(x, unit: "mm")), y \(Fmt.value(y, unit: "mm"))")
    }
}

/// The phase power of each DOE at one wavelength.
struct DOEPowerGrid: View {
    let row: ParaxialXYData.Row

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: Space.xs) {
            ForEach(row.doePower, id: \.surface) { d in
                GridRow {
                    Text("DOE power, surface \(d.surface)").foregroundStyle(.secondary)
                    Text(Fmt.value(d.power, unit: "1/mm")).numeric().gridColumnAlignment(.trailing)
                }
                .accessibilityElement(children: .combine)
            }
        }
        .font(.callout)
        .textSelection(.enabled)
        .help("Phase power of the trace order (base refractive power excluded)")
    }
}

/// EFL, BFL and squeeze in x and y per wavelength.
struct XYWavelengthTable: View {
    let rows: [ParaxialXYData.Row]

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: Space.m, verticalSpacing: Space.xs) {
            GridRow {
                Text("λ µm")
                Text("EFL x")
                Text("EFL y")
                Text("BFL x")
                Text("BFL y")
                Text("Squeeze")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            ForEach(rows) { r in
                GridRow {
                    HStack(spacing: Space.xs) {
                        Circle().fill(Palette.wavelengthColor(r.wavelengthUm)).frame(width: 6, height: 6)
                        Text(String(format: "%.4f", r.wavelengthUm))
                    }
                    Text(Fmt.short(r.eflX))
                    Text(Fmt.short(r.eflY))
                    Text(Fmt.short(r.bflX))
                    Text(Fmt.short(r.bflY))
                    Text(ParaxialXYData.squeezeText(r.squeeze))
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(String(format: "%.4f micrometres", r.wavelengthUm))
                .accessibilityValue(
                    "EFL x \(Fmt.short(r.eflX)), y \(Fmt.short(r.eflY)) millimetres; BFL x \(Fmt.short(r.bflX)), y \(Fmt.short(r.bflY)); squeeze \(ParaxialXYData.squeezeText(r.squeeze))"
                )
            }
        }
        .font(.caption)
        .monospacedDigit()
        .textSelection(.enabled)
    }
}
