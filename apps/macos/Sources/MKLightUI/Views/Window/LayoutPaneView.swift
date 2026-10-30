import MKLightEngine
import MKLightModel
import SwiftUI

/// The layout viewport: the engine's `layout` scene drawn by Metal, with
/// its controls in a slim header and a scale bar.
struct LayoutPaneView: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let system: OpticalSystem
    @State private var error: String?
    @Environment(\.self) private var environment

    private var pane: AnalysisPane { workspace.layout }

    var body: some View {
        let scene = EngineLayout.scene(from: pane.output)
        VStack(spacing: 0) {
            header
            ZStack {
                ViewportView(
                    content: ViewportContent(
                        scene: scene, mode: window.viewportMode, coloring: window.rayColoring,
                        selectedSurface: window.selection.flatMap { system.surfaceIndex(of: $0) }, stale: pane.isStale,
                        fitRequest: window.fitRequest, accent: accent),
                    camera: $window.camera, error: $error)
                if let error {
                    ContentUnavailableView("Viewport Unavailable", systemImage: "exclamationmark.triangle", description: Text(error))
                        .background(.background)
                } else if scene == nil {
                    placeholder
                }
            }
            .overlay(alignment: .bottomLeading) {
                if scene != nil { ScaleBar(pointsPerMM: window.camera.pointsPerMM).padding(Space.m) }
            }
            .overlay(alignment: .bottomTrailing) {
                if pane.isStale && !pane.isRunning, scene != nil {
                    Text("Design changed").font(.caption).foregroundStyle(.secondary).padding(Space.m)
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: Space.s) {
            Label("Layout", systemImage: "camera.aperture").font(.headline).labelStyle(.titleOnly)
            Spacer(minLength: Space.s)
            if pane.isRunning { ProgressView().controlSize(.mini) }
            Picker("View", selection: $window.viewportMode) {
                ForEach(ViewportMode.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .help("Meridional section or 3D view (⌘2, ⌘3)")
            Menu {
                Picker("Ray Colors", selection: $window.rayColoring) {
                    ForEach(RayColoring.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.inline)
                Divider()
                Menu("Rays per Field") {
                    ForEach([3, 5, 7, 11, 15], id: \.self) { n in
                        Button("\(n)") { pane.set("rays_per_field", .integer(Int64(n))) }
                    }
                }
                Menu("Wavelengths") {
                    Button("Primary") { pane.set("wavelengths", .object(["type": .string("primary")])) }
                    Button("All") { pane.set("wavelengths", .object(["type": .string("all")])) }
                }
            } label: {
                Label("Display", systemImage: "eye")
            }
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Ray colors and sampling")
            Button("Fit", systemImage: "arrow.up.left.and.arrow.down.right") { window.fitRequest += 1 }
                .help("Fit the layout (⌘0, or double-click)")
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
        .controlSize(.small)
        .padding(.horizontal, Space.m)
        .frame(height: 30)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
    }

    @ViewBuilder private var placeholder: some View {
        switch pane.phase {
        case .unavailable(let why):
            ContentUnavailableView("Layout Not Available", systemImage: "nosign", description: Text(why)).background(.background)
        case .failed(let why):
            ContentUnavailableView("No Layout", systemImage: "exclamationmark.triangle", description: Text(why)).background(.background)
        default:
            if let e = workspace.modelErrors.first {
                ContentUnavailableView("Design Not Ready", systemImage: "xmark.octagon", description: Text(e)).background(.background)
            } else {
                ProgressView().controlSize(.small)
            }
        }
    }

    private var accent: SIMD3<Float> {
        let r = Color.accentColor.resolve(in: environment)
        return SIMD3(r.red, r.green, r.blue)
    }
}

/// A scale bar of a round length.
struct ScaleBar: View {
    let pointsPerMM: Double

    var body: some View {
        let mm = Nice.length(90 / max(pointsPerMM, 1e-9))
        VStack(alignment: .leading, spacing: 2) {
            Rectangle().frame(width: max(mm * pointsPerMM, 1), height: 1.5)
            Text(Nice.lengthLabel(mm)).font(.caption2.monospacedDigit())
        }
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Scale bar \(Nice.lengthLabel(mm))")
    }
}
