import MKLightEngine
import MKLightModel
import SwiftUI

/// Tabbed analysis area. Shows the selected analysis with its settings, its
/// job status (progress, cancel, staleness) and its run metadata.
struct AnalysisArea: View {
    @Bindable var viewState: WindowViewState
    let analyses: AnalysisCoordinator
    let system: OpticalSystem

    static let tabs: [AnalysisKind] = [.spotDiagram, .rayFan, .opdFan, .fieldCurvature, .mtf, .psf, .firstOrder]

    var body: some View {
        VStack(spacing: 0) {
            ViewThatFits(in: .horizontal) {
                Picker("Analysis", selection: $viewState.selectedAnalysis) {
                    ForEach(Self.tabs) { Text($0.shortTitle).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                Picker("Analysis", selection: $viewState.selectedAnalysis) {
                    ForEach(Self.tabs) { Text($0.title).tag($0) }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
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
            AnalysisStatusBar(kind: viewState.selectedAnalysis, analyses: analyses)
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
        case .layout, nil:
            if case .failed(let msg) = analyses.slot(kind).phase {
                ContentUnavailableView("Analysis Failed", systemImage: "exclamationmark.triangle", description: Text(msg))
            } else if !analyses.validationErrors.isEmpty {
                ContentUnavailableView("Invalid Model", systemImage: "exclamationmark.triangle", description: Text(analyses.validationErrors.joined(separator: "\n")))
            } else {
                ProgressView().controlSize(.small)
            }
        }
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
                    TextField("Seed", value: $viewState.analysisSettings.spot.seed, format: .number.grouping(.never))
                        .frame(width: 80)
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
        }
        .controlSize(.small)
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
    let kind: AnalysisKind
    let analyses: AnalysisCoordinator
    @State private var showInfo = false

    var body: some View {
        let slot = analyses.slot(kind)
        HStack(spacing: 8) {
            switch slot.phase {
            case .running(let p):
                ProgressView(value: p.fraction).frame(width: 120)
                Text(p.stage).foregroundStyle(.secondary).lineLimit(1)
                Button("Cancel", systemImage: "xmark.circle.fill") { analyses.cancel(kind) }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .help("Cancel this analysis")
            case .failed(let msg):
                Label(msg, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red).lineLimit(1)
            case .cancelled:
                Text("Cancelled").foregroundStyle(.secondary)
            case .idle, .finished:
                if let info = slot.output?.runInfo {
                    Text("\(info.backend) · \(String(format: "%.0f", info.durationMs)) ms · snapshot \(info.snapshotHash.short)")
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            if analyses.isStale(kind) {
                Text("STALE")
                    .font(.caption2.bold())
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(Capsule().fill(Color.orange.opacity(0.85)))
                    .foregroundStyle(.white)
                    .help("This result was computed from an older model or different settings.")
            }
            Spacer(minLength: 0)
            Button("Re-run", systemImage: "arrow.clockwise") { analyses.rerun(kind) }
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
