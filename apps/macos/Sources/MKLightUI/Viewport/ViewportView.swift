import AppKit
import MetalKit
import MKLightEngine
import MKLightModel
import SwiftUI

/// MTKView that turns mouse, trackpad and scroll input into camera changes.
///
/// 2-D: drag pans. 3-D: drag orbits, ⇧-drag or right-drag pans. Scroll wheel
/// zooms about the cursor; trackpad scrolling pans and pinching zooms.
/// Double-click fits the scene.
final class ViewportMTKView: MTKView {
    var renderer: ViewportRenderer?
    var onCamera: (ViewportCamera) -> Void = { _ in }
    var onFit: () -> Void = {}
    var onAppearance: () -> Void = {}

    private var camera: ViewportCamera { renderer?.camera ?? ViewportCamera() }
    private var mode: ViewportMode { renderer?.mode ?? .section2D }

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    private func set(_ c: ViewportCamera) {
        renderer?.camera = c
        needsDisplay = true
        onCamera(c)
    }

    /// Cursor position in points from the view centre, y up.
    private func centred(_ event: NSEvent) -> CGPoint {
        let p = convert(event.locationInWindow, from: nil)
        return CGPoint(x: p.x - bounds.midX, y: p.y - bounds.midY)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        if event.clickCount == 2 { onFit() }
    }

    override func mouseDragged(with event: NSEvent) {
        let d = CGSize(width: event.deltaX, height: -event.deltaY)
        if mode == .orbit3D && !event.modifierFlags.contains(.shift) {
            set(ViewportMath.orbit(camera, by: d))
        } else {
            set(ViewportMath.pan(camera, mode: mode, by: d))
        }
    }

    override func rightMouseDragged(with event: NSEvent) {
        set(ViewportMath.pan(camera, mode: mode, by: CGSize(width: event.deltaX, height: -event.deltaY)))
    }

    override func otherMouseDragged(with event: NSEvent) { rightMouseDragged(with: event) }

    override func scrollWheel(with event: NSEvent) {
        if event.hasPreciseScrollingDeltas && !event.modifierFlags.contains(.command) {
            set(ViewportMath.pan(camera, mode: mode, by: CGSize(width: event.scrollingDeltaX, height: -event.scrollingDeltaY)))
        } else {
            let dy = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY * 0.1 : event.scrollingDeltaY
            set(ViewportMath.zoom(camera, mode: mode, factor: pow(1.1, Double(dy)), about: centred(event)))
        }
    }

    override func magnify(with event: NSEvent) {
        set(ViewportMath.zoom(camera, mode: mode, factor: 1 + Double(event.magnification), about: centred(event)))
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        onAppearance()
    }
}

/// SwiftUI wrapper of the Metal viewport.
struct ViewportView: NSViewRepresentable {
    let scene: LayoutScene?
    @Binding var camera: ViewportCamera
    let mode: ViewportMode
    let coloring: RayColoring
    let selectedSurface: Int?
    let stale: Bool
    let fitRequest: Int
    @Binding var error: String?

    final class Coordinator {
        var lastFit = -1
        var lastMode: ViewportMode?
        var lastSceneID: [Int]?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> ViewportMTKView {
        let view = ViewportMTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.colorPixelFormat = .bgra8Unorm
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.autoResizeDrawable = true
        if let device = view.device {
            do {
                let r = try ViewportRenderer(device: device, pixelFormat: view.colorPixelFormat)
                view.sampleCount = r.sampleCount
                view.renderer = r
                view.delegate = r
            } catch {
                DispatchQueue.main.async { self.error = "Metal viewport unavailable: \(error.localizedDescription)" }
            }
        } else {
            DispatchQueue.main.async { self.error = "No Metal device" }
        }
        return view
    }

    func updateNSView(_ view: ViewportMTKView, context: Context) {
        guard let r = view.renderer else { return }
        let dark = view.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        view.clearColor = dark ? MTLClearColor(red: 0.11, green: 0.11, blue: 0.12, alpha: 1) : MTLClearColor(red: 0.985, green: 0.985, blue: 0.98, alpha: 1)
        view.onCamera = { c in DispatchQueue.main.async { camera = c } }
        view.onFit = { fit(view, r) }
        view.onAppearance = { view.needsDisplay = true }
        let style = ViewportStyle(
            mode: mode, coloring: coloring, selectedSurface: selectedSurface, stale: stale, dark: dark,
            scale: Float(view.window?.backingScaleFactor ?? 2))
        r.mode = mode
        r.setScene(scene, style: style)
        if r.camera != camera { r.camera = camera }

        let c = context.coordinator
        let needsFit = scene != nil && (!camera.isFitted || c.lastFit != fitRequest || c.lastMode != mode)
        if needsFit, view.bounds.width > 1 {
            c.lastFit = fitRequest
            c.lastMode = mode
            fit(view, r)
        }
        view.needsDisplay = true
    }

    private func fit(_ view: ViewportMTKView, _ r: ViewportRenderer) {
        guard let scene = r.scene else { return }
        let c = ViewportMath.fit(r.camera, mode: r.mode, points: SceneGeometry.fitPoints(scene, mode: r.mode), size: view.bounds.size)
        r.camera = c
        view.needsDisplay = true
        DispatchQueue.main.async { camera = c }
    }
}

/// The layout panel: viewport plus overlays (mode, scale bar, status).
struct LayoutPanel: View {
    @Bindable var viewState: WindowViewState
    let analyses: AnalysisCoordinator
    let system: MKLightModelSystem
    let isMockEngine: Bool
    @State private var error: String?

    var body: some View {
        let output = analyses.output(.layout)
        let scene: LayoutScene? = if case .layout(let s) = output?.result { s } else { nil }
        let selectedIndex = viewState.selection.flatMap { system.surfaceIndex(of: $0) }
        ZStack {
            ViewportView(
                scene: scene, camera: $viewState.camera, mode: viewState.viewportMode, coloring: viewState.rayColoring,
                selectedSurface: selectedIndex, stale: analyses.isStale(.layout), fitRequest: viewState.fitRequest, error: $error)
            if let error {
                ContentUnavailableView("Viewport Unavailable", systemImage: "exclamationmark.triangle", description: Text(error))
            } else if scene == nil {
                PhasePlaceholder(phase: analyses.slot(.layout).phase, validationErrors: analyses.validationErrors)
            }
        }
        .overlay(alignment: .topLeading) {
            HStack(spacing: 6) {
                Text(viewState.viewportMode.title).font(.caption.bold())
                if analyses.isStale(.layout) {
                    Text("updating…").font(.caption).foregroundStyle(.secondary)
                }
                if case .running(let p) = analyses.slot(.layout).phase {
                    ProgressView(value: p.fraction).frame(width: 60).controlSize(.mini)
                }
            }
            .padding(6)
        }
        .overlay(alignment: .bottomLeading) { ScaleBar(pointsPerMM: viewState.camera.pointsPerMM).padding(8) }
        .overlay(alignment: .topTrailing) {
            if isMockEngine {
                Text("MOCK ENGINE · synthetic data")
                    .font(.caption2.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.orange.opacity(0.85)))
                    .foregroundStyle(.white)
                    .padding(6)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            Text(viewState.viewportMode == .section2D ? "drag: pan · scroll/pinch: zoom · double-click: fit" : "drag: orbit · ⇧drag: pan · scroll/pinch: zoom")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(6)
        }
    }
}

typealias MKLightModelSystem = MKLightModel.OpticalSystem

struct ScaleBar: View {
    let pointsPerMM: Double

    var body: some View {
        let mm = ViewportMath.niceLength(100 / max(pointsPerMM, 1e-9))
        let width = mm * pointsPerMM
        VStack(alignment: .leading, spacing: 2) {
            Rectangle().frame(width: max(width, 1), height: 2)
            Text(mm >= 1 ? String(format: "%g mm", mm) : String(format: "%g µm", mm * 1000)).font(.caption2.monospacedDigit())
        }
        .foregroundStyle(.secondary)
    }
}
