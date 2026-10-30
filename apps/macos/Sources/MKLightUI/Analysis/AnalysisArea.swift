import MKLightEngine
import MKLightModel
import SwiftUI

/// Tabbed analysis area. Shows the selected analysis with its settings, its
/// job status (progress, cancel, staleness) and its run metadata. The
/// "Engine" pane runs any analysis the engine offers, by name, with a
/// settings form built from its schema.
struct AnalysisArea: View {
    @Bindable var viewState: WindowViewState
    let analyses: AnalysisCoordinator
    let system: OpticalSystem

    static let tabs: [AnalysisKind] = [.spotDiagram, .rayFan, .opdFan, .fieldCurvature, .mtf, .psf, .firstOrder]

    var body: some View {
        VStack(spacing: 0) {
            ViewThatFits(in: .horizontal) {
                Picker("Analysis", selection: $viewState.analysisPane) {
                    ForEach(Self.tabs) { Text($0.shortTitle).tag(AnalysisPane.typed($0)) }
                    Text("Engine").tag(AnalysisPane.engine)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                Picker("Analysis", selection: $viewState.analysisPane) {
                    ForEach(Self.tabs) { Text($0.title).tag(AnalysisPane.typed($0)) }
                    Divider()
                    Text("Engine analyses").tag(AnalysisPane.engine)
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            if viewState.showsEnginePane {
                EnginePane(viewState: viewState, analyses: analyses)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    AnalysisSettingsBar(kind: viewState.selectedAnalysis, viewState: viewState, system: system)
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 4)
                Divider()
                ZStack {
                    content(viewState.selectedAnalysis)
                        .opacity(analyses.isStale(viewState.selectedAnalysis) ? 0.45 : 1)
                        .animation(.easeOut(duration: 0.15), value: analyses.isStale(viewState.selectedAnalysis))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                AnalysisStatusBar(
                    slot: analyses.slot(viewState.selectedAnalysis), stale: analyses.isStale(viewState.selectedAnalysis),
                    cancel: { analyses.cancel(viewState.selectedAnalysis) },
                    rerun: { analyses.rerun(viewState.selectedAnalysis) })
            }
        }
    }

    @ViewBuilder private func content(_ kind: AnalysisKind) -> some View {
        switch analyses.output(kind)?.result {
        case .spotDiagram(let r): SpotDiagramView(result: r, fieldUnit: system.fields.kind.unit)
        case .rayFan(let r), .opdFan(let r): FanView(result: r, fieldUnit: system.fields.kind.unit)
        case .fieldCurvature(let r): FieldCurvatureView(result: r)
        case .mtf(let r): MTFView(result: r, fieldUnit: system.fields.kind.unit)
        case .psf(let r): PSFView(result: r, logScale: viewState.psfLogScale)
        case .firstOrder(let r): FirstOrderView(result: r)
        case .generic(let g): GenericResultView(result: g)
        case .layout, nil:
            PhasePlaceholder(phase: analyses.slot(kind).phase, validationErrors: analyses.validationErrors)
        }
    }
}

/// What to show while an analysis has no result.
struct PhasePlaceholder: View {
    let phase: AnalysisCoordinator.Phase
    let validationErrors: [String]

    var body: some View {
        if case .unavailable(let why) = phase {
            ContentUnavailableView("Not Available", systemImage: "puzzlepiece.extension", description: Text(why))
        } else if case .failed(let msg) = phase {
            ContentUnavailableView("Analysis Failed", systemImage: "exclamationmark.triangle", description: Text(msg))
        } else if !validationErrors.isEmpty {
            ContentUnavailableView(
                "Invalid Model", systemImage: "exclamationmark.triangle",
                description: Text(validationErrors.joined(separator: "\n")))
        } else if case .cancelled = phase {
            ContentUnavailableView("Cancelled", systemImage: "xmark.circle")
        } else {
            ProgressView().controlSize(.small)
        }
    }
}

/// Any engine analysis by name: a picker over the engine's list (analyses
/// of later phases are shown disabled with their reason), a settings form
/// generated from the analysis's schema, and the result shown generically.
struct EnginePane: View {
    @Bindable var viewState: WindowViewState
    let analyses: AnalysisCoordinator

    var body: some View {
        let descriptor = analyses.descriptor(viewState.engineAnalysis)
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Picker("Analysis", selection: $viewState.engineAnalysis) {
                    ForEach(analyses.descriptors) { d in
                        Text(d.available ? d.title : "\(d.title) (unavailable)").tag(d.name)
                    }
                    if descriptor == nil { Text(viewState.engineAnalysis).tag(viewState.engineAnalysis) }
                }
                .fixedSize()
                if let d = descriptor, !d.description.isEmpty {
                    Text(d.description).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                Spacer(minLength: 0)
            }
            .controlSize(.small)
            .padding(.horizontal, 8)
            .padding(.bottom, 4)
            if let d = descriptor, !d.settingsFields.isEmpty {
                SchemaSettingsForm(fields: d.settingsFields, values: settingsBinding(for: d.name))
                    .padding(.horizontal, 8)
                    .padding(.bottom, 4)
            }
            Divider()
            ZStack {
                if let d = descriptor, !d.available {
                    ContentUnavailableView(
                        "Not Available", systemImage: "puzzlepiece.extension",
                        description: Text(d.unavailableReason ?? "This engine build does not offer \(d.title)."))
                } else if case .generic(let g)? = analyses.engineSlot.output?.result {
                    GenericResultView(result: g).opacity(analyses.isEngineResultStale ? 0.45 : 1)
                } else if let out = analyses.engineSlot.output {
                    Text("Result of \(out.result.analysisName)").foregroundStyle(.secondary)
                } else {
                    PhasePlaceholder(phase: analyses.engineSlot.phase, validationErrors: analyses.validationErrors)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            AnalysisStatusBar(
                slot: analyses.engineSlot, stale: analyses.isEngineResultStale, cancel: { analyses.cancelEngineAnalysis() },
                rerun: { analyses.rerunEngineAnalysis() })
        }
    }

    private func settingsBinding(for name: String) -> Binding<[String: JSONValue]> {
        Binding(
            get: { viewState.engineSettings[name] ?? [:] },
            set: { viewState.engineSettings[name] = $0 })
    }
}

/// Controls for a settings schema: numbers and integers as fields with
/// their unit, booleans as toggles, enums as pickers, strings as text
/// fields and arrays/objects as JSON text. Empty means "engine default".
struct SchemaSettingsForm: View {
    let fields: [SettingsField]
    @Binding var values: [String: JSONValue]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(fields) { f in control(f).help(f.description.isEmpty ? f.title : f.description) }
                Button("Defaults") { values = [:] }.disabled(values.isEmpty)
            }
            .controlSize(.small)
        }
    }

    @ViewBuilder private func control(_ f: SettingsField) -> some View {
        if !f.enumValues.isEmpty {
            Picker(f.title, selection: binding(f, default: f.defaultValue ?? f.enumValues[0])) {
                ForEach(f.enumValues, id: \.self) { Text($0.displayText).tag($0) }
            }
            .fixedSize()
        } else if f.types.contains("boolean") {
            Toggle(f.title, isOn: Binding(
                get: { values[f.name] == .bool(true) || (values[f.name] == nil && f.defaultValue == .bool(true)) },
                set: { values[f.name] = .bool($0) }))
        } else {
            LabeledContent(f.title + (f.required ? " *" : "")) {
                TextField(placeholder(f), text: textBinding(f))
                    .frame(width: f.types.contains("string") ? 120 : 80)
                if !f.unit.isEmpty { Text(f.unit).foregroundStyle(.secondary) }
            }
            .fixedSize()
        }
    }

    private func placeholder(_ f: SettingsField) -> String {
        f.defaultValue.map { $0 == .null ? "default" : $0.displayText } ?? (f.required ? "required" : "default")
    }

    private func binding(_ f: SettingsField, default d: JSONValue) -> Binding<JSONValue> {
        Binding(get: { values[f.name] ?? d }, set: { values[f.name] = $0 })
    }

    /// Text editing: empty removes the key; numbers parse by the schema's
    /// type (integers stay exact); anything else is tried as JSON, then as a
    /// string.
    private func textBinding(_ f: SettingsField) -> Binding<String> {
        Binding(
            get: {
                guard let v = values[f.name] else { return "" }
                if case .string(let s) = v { return s }
                return (try? String(decoding: MKLightJSON.encode(v), as: UTF8.self)) ?? ""
            },
            set: { text in
                let t = text.trimmingCharacters(in: .whitespaces)
                if t.isEmpty {
                    values.removeValue(forKey: f.name)
                } else if f.types.contains("integer"), let i = Int64(t) {
                    values[f.name] = .integer(i)
                } else if f.types.contains("number"), let d = Double(t) {
                    values[f.name] = .number(d)
                } else if !f.types.contains("string") || f.types.count > 1, let v = try? JSONValue.parse(Data(t.utf8)) {
                    values[f.name] = v
                } else {
                    values[f.name] = .string(t)
                }
            })
    }
}

/// Settings controls for the selected analysis. They edit the window's
/// `AnalysisSettingsSet`; changing them re-runs the analysis.
struct AnalysisSettingsBar: View {
    let kind: AnalysisKind
    @Bindable var viewState: WindowViewState
    let system: OpticalSystem

    var body: some View {
        HStack(spacing: 12) {
            switch kind {
            case .spotDiagram:
                Picker("Pattern", selection: $viewState.analysisSettings.spot.pattern) {
                    ForEach(PupilPattern.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
                }
                .fixedSize()
                Stepper("Rings \(viewState.analysisSettings.spot.rings)", value: $viewState.analysisSettings.spot.rings, in: 1...32)
                    .fixedSize()
                Picker("Reference", selection: $viewState.analysisSettings.spot.reference) {
                    ForEach(SpotReference.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
                }
                .fixedSize()
                if viewState.analysisSettings.spot.pattern == .random {
                    TextField("Seed", value: $viewState.analysisSettings.execution.seed, format: .number.grouping(.never))
                        .frame(width: 80)
                        .help("Philox seed of the random pupil samples")
                }
                wavelengthPicker($viewState.analysisSettings.spot.wavelengths)
            case .rayFan:
                Stepper("Samples \(viewState.analysisSettings.rayFan.samples)", value: $viewState.analysisSettings.rayFan.samples, in: 3...401, step: 2)
                    .fixedSize()
                wavelengthPicker($viewState.analysisSettings.rayFan.wavelengths)
            case .opdFan:
                Stepper("Samples \(viewState.analysisSettings.opdFan.samples)", value: $viewState.analysisSettings.opdFan.samples, in: 3...401, step: 2)
                    .fixedSize()
                wavelengthPicker($viewState.analysisSettings.opdFan.wavelengths)
            case .fieldCurvature:
                Stepper("Field samples \(viewState.analysisSettings.fieldCurvature.fieldSamples)", value: $viewState.analysisSettings.fieldCurvature.fieldSamples, in: 2...201)
                    .fixedSize()
                wavelengthPicker($viewState.analysisSettings.fieldCurvature.wavelengths)
            case .mtf:
                LabeledContent("Max frequency") {
                    TextField("Max frequency", value: $viewState.analysisSettings.mtf.maxFrequency, format: .number)
                        .frame(width: 60)
                    Text("cycles/mm").foregroundStyle(.secondary)
                }
                .fixedSize()
                wavelengthPicker($viewState.analysisSettings.mtf.wavelengths)
            case .psf:
                Picker("Field", selection: $viewState.analysisSettings.psf.field) {
                    ForEach(system.fields.fields.indices, id: \.self) { Text("\($0 + 1)").tag($0) }
                }
                .fixedSize()
                Picker("Grid", selection: $viewState.analysisSettings.psf.imageSampling) {
                    ForEach([32, 64, 128, 256], id: \.self) { Text("\($0)²").tag($0) }
                }
                .fixedSize()
                Toggle("Log scale", isOn: $viewState.psfLogScale)
                wavelengthPicker($viewState.analysisSettings.psf.wavelengths)
            case .firstOrder, .layout:
                Text("Paraxial quantities at the primary wavelength").foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            executionMenu
        }
        .controlSize(.small)
    }

    /// Backend and precision for every analysis of the window. Part of each
    /// request, so changing them marks results stale and re-runs them.
    private var executionMenu: some View {
        Menu {
            Picker("Backend", selection: $viewState.analysisSettings.execution.backend) {
                ForEach(BackendPreference.allCases, id: \.self) { Text($0.title).tag($0) }
            }
            Picker("Precision", selection: $viewState.analysisSettings.execution.precision) {
                Text("f64 reference").tag(Precision.f64Reference)
                Text("f32 GPU (error budget)").tag(Precision.f32GPU)
            }
            if !system.configurations.isEmpty {
                Picker("Configuration", selection: $viewState.analysisSettings.configuration) {
                    ForEach(system.configurations.indices, id: \.self) { k in
                        let name = system.configurations[k].name
                        Text(name.isEmpty ? "Configuration \(k + 1)" : "\(k + 1): \(name)").tag(k)
                    }
                }
            }
        } label: {
            Label("Execution", systemImage: "cpu")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Compute backend, precision and configuration for this window's analyses")
    }

    private func wavelengthPicker(_ sel: Binding<WavelengthSelection>) -> some View {
        Picker("λ", selection: sel) {
            Text("All λ").tag(WavelengthSelection.all)
            Text("Primary").tag(WavelengthSelection.primary)
            ForEach(system.wavelengths.wavelengths.indices, id: \.self) { i in
                Text(String(format: "%.4f µm", system.wavelengths.wavelengths[i].um)).tag(WavelengthSelection.index(i))
            }
        }
        .fixedSize()
    }
}

/// Progress, cancel, staleness and reproducibility metadata of one analysis.
struct AnalysisStatusBar: View {
    let slot: AnalysisCoordinator.Slot
    let stale: Bool
    let cancel: () -> Void
    let rerun: () -> Void
    @State private var showInfo = false

    var body: some View {
        HStack(spacing: 8) {
            switch slot.phase {
            case .running(let p):
                ProgressView(value: p.fraction).frame(width: 120)
                Text(p.stage).foregroundStyle(.secondary).lineLimit(1)
                Button("Cancel", systemImage: "xmark.circle.fill") { cancel() }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .help("Cancel this analysis")
            case .failed(let msg):
                Label(msg, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red).lineLimit(1)
            case .unavailable(let why):
                Label(why, systemImage: "puzzlepiece.extension").foregroundStyle(.secondary).lineLimit(1)
            case .cancelled:
                Text("Cancelled").foregroundStyle(.secondary)
            case .idle, .finished:
                if let info = slot.output?.runInfo {
                    let seed = info.seed.map { " · seed \($0)" } ?? ""
                    Text(
                        "\(info.backend) · \(info.precision.rawValue) · \(String(format: "%.0f", info.durationMs)) ms\(seed) · snapshot \(info.snapshotHash.short)"
                    )
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    if !info.warnings.isEmpty {
                        Image(systemName: "exclamationmark.bubble").foregroundStyle(.orange)
                            .help(info.warnings.joined(separator: "\n"))
                    }
                }
            }
            if stale {
                Text("STALE")
                    .font(.caption2.bold())
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(Capsule().fill(Color.orange.opacity(0.85)))
                    .foregroundStyle(.white)
                    .help("This result was computed from an older model or different settings.")
            }
            Spacer(minLength: 0)
            Button("Re-run", systemImage: "arrow.clockwise") { rerun() }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .help("Run again")
            Button("Run info", systemImage: "info.circle") { showInfo.toggle() }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .disabled(slot.output == nil)
                .help("Settings, snapshot and backend of this result")
                .popover(isPresented: $showInfo) { RunInfoView(info: slot.output?.runInfo) }
        }
        .font(.caption)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }
}

struct RunInfoView: View {
    let info: RunInfo?

    var body: some View {
        let json = info.flatMap { try? String(decoding: MKLightJSON.encode($0, pretty: true), as: UTF8.self) } ?? ""
        VStack(alignment: .leading, spacing: 8) {
            Text("Run Info").font(.headline)
            Text("Everything needed to reproduce this result: snapshot hash, settings, backend, precision and seed.")
                .font(.caption)
                .foregroundStyle(.secondary)
            ScrollView {
                Text(json).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(width: 420, height: 280)
            Button("Copy JSON") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(json, forType: .string)
            }
        }
        .padding()
    }
}
