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

    private var pane: AnalysisPane { workspace.layout }

    var body: some View {
        let scene = pane.layoutScene
        let footprints = workspace.showFootprints && window.viewportMode == .orbit3D ? FootprintOverlay(output: workspace.footprints.output) : nil
        ZStack {
            LayoutViewport(
                window: window, system: system, scene: scene, stale: pane.isStale, accent: accent, footprints: footprints,
                error: $error
            )
            .accessibilityLabel(LayoutViewport.accessibilityText(scene, mode: window.viewportMode, system: system))
            if let error {
                ContentUnavailableView("Viewport Unavailable", systemImage: "exclamationmark.triangle", description: Text(error))
                    .background(.background)
            } else if scene == nil {
                placeholder
            }
        }
        .overlay(alignment: .top) { LayoutControls(workspace: workspace, window: window, modes: modes) }
        .overlay(alignment: .bottomLeading) {
            if scene != nil { CameraScaleBar(window: window).padding(Space.l) }
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
        .overlay(alignment: .top) {
            if let title = workspace.previewTitle {
                Label("Previewing \(title) · not applied", systemImage: "eye")
                    .font(.callout.weight(.medium))
                    .padding(.horizontal, Space.l)
                    .padding(.vertical, Space.s)
                    .glassEffect(.regular.tint(.accentColor.opacity(0.25)), in: .capsule)
                    .padding(.top, 56)
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


    /// The engine draws the section the view shows (`section`).
    private func syncSection() {
        let wanted: JSONValue? = window.viewportMode == .sectionXZ ? .string("xz") : nil
        if pane.settings["section"] != wanted { pane.set("section", wanted) }
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

    private var accent: SIMD3<Float> {
        let r = Color.accentColor.resolve(in: environment)
        return SIMD3(r.red, r.green, r.blue)
    }
}

/// The Metal viewport. Its body alone reads the selection, so selecting a
/// surface updates the viewport, not the pane and its controls.
private struct LayoutViewport: View {
    let window: WindowState
    let system: OpticalSystem
    let scene: LayoutScene?
    let stale: Bool
    let accent: SIMD3<Float>
    let footprints: FootprintOverlay?
    @Binding var error: String?

    var body: some View {
        @Bindable var window = window
        ViewportView(
            content: ViewportContent(
                scene: scene, mode: window.viewportMode, coloring: window.rayColoring,
                selectedSurface: window.selection.flatMap { system.surfaceIndex(of: $0) }, stale: stale,
                fitRequest: window.fitRequest, accent: accent, footprints: footprints, fitLens: window.fitLens),
            accessibilityText: Self.accessibilityText(scene, mode: window.viewportMode, system: system), camera: $window.camera,
            error: $error)
    }

    /// The layout in words: the view, the surfaces between object and
    /// image, the rays drawn and the scene's length.
    static func accessibilityText(_ scene: LayoutScene?, mode: ViewportMode, system: OpticalSystem) -> String {
        guard let scene else { return "Layout, not available" }
        let rays = scene.polylines.filter { $0.kind == .ray }.count
        let length = scene.bounds.max.z - scene.bounds.min.z
        return "Layout, \(mode == .orbit3D ? "3D view" : mode == .sectionXZ ? "x–z section" : "meridional section"), \(system.surfaces.count - 2) surfaces, \(rays) rays, \(Fmt.short(length)) millimeters long"
    }
}

/// The scale bar, following the camera (its body alone reads it, so a pan
/// or zoom does not update the pane).
private struct CameraScaleBar: View {
    let window: WindowState

    var body: some View {
        ScaleBar(pointsPerMM: window.camera.pointsPerMM, stretch: window.camera.verticalScale(window.viewportMode))
    }
}

/// The viewport's floating controls: title, view mode, display options, fit.
private struct LayoutControls: View {
    let workspace: Workspace
    @Bindable var window: WindowState
    let modes: [ViewportMode]
    @Namespace private var glass
    @Environment(\.panelTabbed) private var tabbed

    private var pane: AnalysisPane { workspace.layout }
    private var meshOn: Bool { (pane.settings["mesh_rings"]?.int ?? 0) > 0 }
    private func title(_ m: ViewportMode) -> String { m == .section2D && modes.count == 3 ? "Y–Z" : m.title }

    var body: some View { controls }

    /// The vertical stretch of the section views; a change refits the view.
    private var stretch: Binding<Double> {
        Binding(
            get: { window.camera.stretch },
            set: { s in
                window.camera.stretch = s
                window.camera.stretchChosen = true
                window.fitRequest += 1
            })
    }

    /// Title on the leading side; mode, display options and fit on the
    /// trailing side, each cluster one glass group whose selection morphs.
    private var controls: some View {
        GlassEffectContainer(spacing: Space.m) {
            HStack(spacing: Space.s) {
                // The title chip steps aside when the viewport is narrow.
                // The title chip is the panel's handle (drag to move it).
                ViewThatFits(in: .horizontal) {
                    if !tabbed {
                        HStack(spacing: Space.s) {
                            Label("Layout", systemImage: "rectangle.lefthalf.inset.filled").font(.headline)
                            if pane.isRunning { ProgressView().controlSize(.mini) }
                        }
                        .fixedSize()
                        .padding(.horizontal, Space.l)
                        .padding(.vertical, Space.s)
                        .glassEffect(.regular, in: .capsule)
                        .help("Drag to move the layout, or out of the window to open it in a window of its own")
                        .panelHandle(.layout)
                    }
                    Color.clear.frame(width: 0, height: 0)
                }
                .layoutPriority(-1)
                Spacer(minLength: Space.s)
                HStack(spacing: 0) {
                    ForEach(modes) { mode in
                        Button {
                            window.viewportMode = mode
                        } label: {
                            Text(title(mode))
                                .font(.callout.weight(window.viewportMode == mode ? .semibold : .regular))
                                .fixedSize()
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
                        Button("Fit All") {
                            window.fitLens = false
                            window.fitRequest += 1
                        }
                        Button("Fit Lens") {
                            window.fitLens = true
                            window.fitRequest += 1
                        }
                        Divider()
                        Picker("Vertical Stretch", selection: stretch) {
                            ForEach(ViewportCamera.stretches, id: \.self) { s in
                                Text(s == 1 ? "Off (true proportions)" : "×\(Int(s))").tag(s)
                            }
                        }
                        .disabled(window.viewportMode == .orbit3D)
                        Divider()
                        Toggle("Surface Mesh in 3D", isOn: Binding(get: { meshOn }, set: { pane.set("mesh_rings", $0 ? .integer(16) : nil) }))
                        Toggle("Beam Footprints in 3D", isOn: Binding(get: { workspace.showFootprints }, set: { workspace.showFootprints = $0 }))
                    } label: {
                        Label("Display", systemImage: "eye")
                    }
                    .menuIndicator(.hidden)
                    .help("Ray colors and sampling")
                    if window.viewportMode != .orbit3D && window.camera.stretch != 1 {
                        // A stretched section says so in the controls; the menu sets it.
                        Menu {
                            Picker("Vertical Stretch", selection: stretch) {
                                ForEach(ViewportCamera.stretches, id: \.self) { s in
                                    Text(s == 1 ? "Off (true proportions)" : "×\(Int(s))").tag(s)
                                }
                            }
                            .pickerStyle(.inline)
                        } label: {
                            Label("Stretch ×\(Int(window.camera.stretch))", systemImage: "arrow.up.and.down")
                                .labelStyle(.titleAndIcon)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.tint)
                        }
                        .menuIndicator(.hidden)
                        .buttonBorderShape(.capsule)
                        .fixedSize()
                        .help("The section is drawn \(Int(window.camera.stretch)) times taller than true proportions. Click to change or turn off.")
                        .accessibilityLabel("Vertical stretch")
                        .accessibilityValue("\(Int(window.camera.stretch)) times")
                    }
                    if window.viewportMode == .orbit3D {
                        Button {
                            workspace.showFootprints.toggle()
                        } label: {
                            Label("Footprints", systemImage: workspace.showFootprints ? "circle.dashed.inset.filled" : "circle.dashed")
                                .foregroundStyle(workspace.showFootprints ? AnyShapeStyle(.tint) : AnyShapeStyle(.primary))
                        }
                        .help(workspace.showFootprints ? "Hide the beam footprints on the surfaces" : "Show the beam footprint of every field on every surface")
                        .accessibilityLabel("Beam footprints")
                        .accessibilityValue(workspace.showFootprints ? "on" : "off")
                    }
                    Button("Fit", systemImage: "arrow.up.left.and.arrow.down.right") {
                        window.fitLens = false
                        window.fitRequest += 1
                    }
                        .help("Fit the layout (⌘0, or double-click)")
                    PanelMoreMenu(panel: .layout)
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
            }
        }
        .controlSize(.small)
        .padding(Space.m)
    }

}

/// A scale bar of a round length; with a vertical stretch, an upright bar
/// of the same length drawn stretched, and the factor.
struct ScaleBar: View {
    let pointsPerMM: Double
    var stretch = 1.0

    var body: some View {
        let mm = Nice.length(90 / max(pointsPerMM, 1e-9))
        let len = max(mm * pointsPerMM, 1)
        // Upright: a round length as drawn (stretched), about 60 pt tall.
        let up = Nice.length(60 / max(pointsPerMM * stretch, 1e-9))
        HStack(alignment: .bottom, spacing: Space.s) {
            if stretch != 1 {
                HStack(alignment: .center, spacing: 3) {
                    Rectangle().frame(width: 1.5, height: max(up * pointsPerMM * stretch, 1))
                    Text(Nice.lengthLabel(up)).font(.caption2.monospacedDigit())
                }
                .foregroundStyle(.tint)
            }
            VStack(alignment: .leading, spacing: 2) {
                Rectangle().frame(width: len, height: 1.5)
                HStack(spacing: Space.s) {
                    Text(Nice.lengthLabel(mm))
                    if stretch != 1 {
                        Text("vertical ×\(Int(stretch))").foregroundStyle(.tint).fontWeight(.semibold)
                    }
                }
                .font(.caption2.monospacedDigit())
            }
        }
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            "Scale bar \(Nice.lengthLabel(mm))"
                + (stretch != 1 ? ", vertical stretch \(Int(stretch)) times, upright bar \(Nice.lengthLabel(Nice.length(60 / max(pointsPerMM * stretch, 1e-9))))" : ""))
    }
}
