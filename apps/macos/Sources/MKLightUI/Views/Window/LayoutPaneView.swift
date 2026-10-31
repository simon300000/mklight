import MKLightEngine
import MKLightModel
import SwiftUI

/// The layout viewport: the engine's `layout` scene drawn by Metal, edge to
/// edge, with its controls floating over it on Liquid Glass (view mode,
/// display options, fit) and a scale bar. The scene extends under the
/// sidebar's glass (background extension).
struct LayoutPaneView: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let system: OpticalSystem
    @State private var error: String?
    @Environment(\.self) private var environment
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Namespace private var glass

    private var pane: AnalysisPane { workspace.layout }

    var body: some View {
        let scene = EngineLayout.scene(from: pane.output)
        let footprints = workspace.showFootprints && window.viewportMode == .orbit3D ? FootprintOverlay(output: workspace.footprints.output) : nil
        ZStack {
            ViewportView(
                content: ViewportContent(
                    scene: scene, mode: window.viewportMode, coloring: window.rayColoring,
                    selectedSurface: window.selection.flatMap { system.surfaceIndex(of: $0) }, stale: pane.isStale,
                    fitRequest: window.fitRequest, accent: accent, footprints: footprints),
                camera: $window.camera, error: $error
            )
            .accessibilityLabel(accessibilityText(scene))
            if let error {
                ContentUnavailableView("Viewport Unavailable", systemImage: "exclamationmark.triangle", description: Text(error))
                    .background(.background)
            } else if scene == nil {
                placeholder
            }
        }
        .overlay(alignment: .top) { controls }
        .overlay(alignment: .bottomLeading) {
            if scene != nil { ScaleBar(pointsPerMM: window.camera.pointsPerMM).padding(Space.l) }
        }
        .overlay(alignment: .bottomTrailing) {
            if pane.isStale && !pane.isRunning, scene != nil {
                Button {
                    pane.rerun()
                } label: {
                    Label("Design changed", systemImage: "clock.arrow.circlepath")
                }
                .buttonStyle(.glass)
                .controlSize(.small)
                .help("The layout is out of date. Click to update it.")
                .padding(Space.l)
                .transition(.opacity)
            }
        }
        .overlay(alignment: .bottom) {
            if let title = workspace.previewTitle {
                Label("Previewing \(title) · not applied", systemImage: "eye")
                    .font(.callout.weight(.medium))
                    .padding(.horizontal, Space.l)
                    .padding(.vertical, Space.s)
                    .glassEffect(.regular.tint(.accentColor.opacity(0.25)), in: .capsule)
                    .padding(.bottom, Space.l)
                    .accessibilityLabel("Previewing \(title). The design has not changed.")
                    .transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.3), value: window.viewportMode)
        .animation(reduceMotion ? nil : .smooth(duration: 0.25), value: workspace.previewTitle)
        .clipped()
        .onAppear(perform: syncSection)
        .onChange(of: window.viewportMode) { _, _ in syncSection() }
        .onChange(of: anamorphic) { _, a in if !a && window.viewportMode == .sectionXZ { window.viewportMode = .section2D } }
    }

    /// The design has no rotational symmetry: both sections matter.
    private var anamorphic: Bool { ParaxialXYData.isAnamorphic(workspace.summary.output?.json, system: system) }

    private var modes: [ViewportMode] { anamorphic || window.viewportMode == .sectionXZ ? [.section2D, .sectionXZ, .orbit3D] : [.section2D, .orbit3D] }

    private func title(_ m: ViewportMode) -> String { m == .section2D && modes.count == 3 ? "Y–Z" : m.title }

    /// The engine draws the section the view shows (`section`), and the
    /// surface mesh only when the 3-D view asks for it.
    private func syncSection() {
        let wanted: JSONValue? = window.viewportMode == .sectionXZ ? .string("xz") : nil
        if pane.settings["section"] != wanted { pane.set("section", wanted) }
    }

    private var meshOn: Bool { (pane.settings["mesh_rings"]?.int ?? 0) > 0 }

    /// Title on the leading side; mode, display options and fit on the
    /// trailing side, each cluster one glass group whose selection morphs.
    private var controls: some View {
        GlassEffectContainer(spacing: Space.m) {
            HStack(spacing: Space.s) {
                HStack(spacing: Space.s) {
                    Text("Layout").font(.headline)
                    if pane.isRunning { ProgressView().controlSize(.mini) }
                }
                .padding(.horizontal, Space.l)
                .padding(.vertical, Space.s)
                .glassEffect(.regular, in: .capsule)
                Spacer(minLength: Space.s)
                HStack(spacing: 0) {
                    ForEach(modes) { mode in
                        Button {
                            window.viewportMode = mode
                        } label: {
                            Text(title(mode))
                                .font(.callout.weight(window.viewportMode == mode ? .semibold : .regular))
                                .padding(.horizontal, Space.m)
                                .padding(.vertical, Space.xs + 1)
                                .contentShape(Capsule())
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(window.viewportMode == mode ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
                        .background {
                            if window.viewportMode == mode {
                                Capsule().fill(.tint).matchedGeometryEffect(id: "mode", in: glass)
                            }
                        }
                        .accessibilityAddTraits(window.viewportMode == mode ? .isSelected : [])
                    }
                }
                .padding(3)
                .glassEffect(.regular.interactive(), in: .capsule)
                .help(modes.count == 3 ? "Y–Z section, X–Z section or 3D view (⌘2, ⌥⌘2, ⌘3)" : "Meridional section or 3D view (⌘2, ⌘3)")
                HStack(spacing: Space.xs) {
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
                        Divider()
                        Toggle("Surface Mesh in 3D", isOn: Binding(get: { meshOn }, set: { pane.set("mesh_rings", $0 ? .integer(16) : nil) }))
                        Toggle("Beam Footprints in 3D", isOn: Binding(get: { workspace.showFootprints }, set: { workspace.showFootprints = $0 }))
                    } label: {
                        Label("Display", systemImage: "eye")
                    }
                    .menuIndicator(.hidden)
                    .help("Ray colors and sampling")
                    if window.viewportMode == .orbit3D {
                        Toggle(isOn: Binding(get: { workspace.showFootprints }, set: { workspace.showFootprints = $0 })) {
                            Label("Footprints", systemImage: workspace.footprints.isRunning ? "circle.dotted" : "circle.dashed")
                        }
                        .toggleStyle(.button)
                        .tint(.accentColor)
                        .help(workspace.showFootprints ? "Hide the beam footprints on the surfaces" : "Show the beam footprint of every field on every surface")
                        .accessibilityLabel("Beam footprints")
                        .accessibilityValue(workspace.showFootprints ? "on" : "off")
                    }
                    Button("Fit", systemImage: "arrow.up.left.and.arrow.down.right") { window.fitRequest += 1 }
                        .help("Fit the layout (⌘0, or double-click)")
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
            }
        }
        .controlSize(.small)
        .padding(Space.m)
    }

    @ViewBuilder private var placeholder: some View {
        switch pane.phase {
        case .unavailable(let why):
            ContentUnavailableView("Layout Not Available", systemImage: "nosign", description: Text(why))
                .frame(maxWidth: .infinity, maxHeight: .infinity).background(.background)
        case .failed(let why) where workspace.issues.contains { $0.severity == .error }:
            ContentUnavailableView("Design Has Errors", systemImage: "xmark.octagon", description: Text(why))
                .frame(maxWidth: .infinity, maxHeight: .infinity).background(.background)
        case .failed(let why):
            ContentUnavailableView("No Layout", systemImage: "exclamationmark.triangle", description: Text(why))
                .frame(maxWidth: .infinity, maxHeight: .infinity).background(.background)
        default:
            if let e = workspace.modelErrors.first {
                ContentUnavailableView("Design Not Ready", systemImage: "xmark.octagon", description: Text(e))
                    .frame(maxWidth: .infinity, maxHeight: .infinity).background(.background)
            } else {
                ProgressView().controlSize(.small)
            }
        }
    }

    private func accessibilityText(_ scene: LayoutScene?) -> String {
        guard let scene else { return "Layout, not available" }
        let rays = scene.polylines.filter { $0.kind == .ray }.count
        let length = scene.bounds.max.z - scene.bounds.min.z
        return "Layout, \(window.viewportMode == .orbit3D ? "3D view" : window.viewportMode == .sectionXZ ? "x–z section" : "meridional section"), \(system.surfaces.count - 2) surfaces, \(rays) rays, \(Fmt.short(length)) millimetres long"
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
