import MetalKit
import MKLightEngine
import SwiftUI

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

/// What the Metal viewport draws and how. Shared by the macOS and iPadOS
/// wrappers; only input handling differs per platform.
struct ViewportContent: Equatable {
    var scene: LayoutScene?
    var mode: ViewportMode
    var coloring: RayColoring
    var selectedSurface: Int?
    var stale: Bool
    var fitRequest: Int
    var accent: SIMD3<Float>
    /// Beam footprints on every surface (3-D only), from `footprint`.
    var footprints: FootprintOverlay? = nil
}

/// Camera gestures common to both platforms (in view points, y up).
@MainActor
final class ViewportInput {
    var renderer: ViewportRenderer?
    var onCamera: (ViewportCamera) -> Void = { _ in }
    var redraw: () -> Void = {}

    var camera: ViewportCamera { renderer?.camera ?? ViewportCamera() }
    var mode: ViewportMode { renderer?.mode ?? .section2D }

    func set(_ c: ViewportCamera) {
        renderer?.camera = c
        redraw()
        onCamera(c)
    }

    func drag(_ d: CGSize, pan: Bool) {
        set(mode == .orbit3D && !pan ? ViewportMath.orbit(camera, by: d) : ViewportMath.pan(camera, mode: mode, by: d))
    }

    func zoom(_ factor: Double, about p: CGPoint) {
        set(ViewportMath.zoom(camera, mode: mode, factor: factor, about: p))
    }

    func fit(size: CGSize) {
        guard let r = renderer, let scene = r.scene else { return }
        set(ViewportMath.fit(r.camera, mode: r.mode, points: SceneGeometry.fitPoints(scene, mode: r.mode), size: size))
    }
}

/// Creates and updates the `MTKView`; the same code serves both
/// representables.
@MainActor
enum ViewportBridge {
    static func make(input: ViewportInput, error: Binding<String?>) -> ViewportMTKView {
        let view = ViewportMTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.input = input
        view.colorPixelFormat = .bgra8Unorm
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.autoResizeDrawable = true
        input.redraw = { [weak view] in view?.redraw() }
        guard let device = view.device else {
            DispatchQueue.main.async { error.wrappedValue = "This Mac has no Metal device." }
            return view
        }
        do {
            let r = try ViewportRenderer(device: device, pixelFormat: view.colorPixelFormat)
            view.sampleCount = r.sampleCount
            input.renderer = r
            view.delegate = r
        } catch let e {
            DispatchQueue.main.async { error.wrappedValue = "The Metal viewport could not start: \(e.localizedDescription)" }
        }
        return view
    }

    static func update(
        _ view: ViewportMTKView, content: ViewportContent, camera: Binding<ViewportCamera>, coordinator: ViewportCoordinator,
        dark: Bool, scale: CGFloat
    ) {
        guard let input = view.input, let r = input.renderer else { return }
        view.clearColor =
            dark
            ? MTLClearColor(red: 0.118, green: 0.118, blue: 0.125, alpha: 1)
            : MTLClearColor(red: 0.988, green: 0.988, blue: 0.984, alpha: 1)
        input.onCamera = { c in DispatchQueue.main.async { camera.wrappedValue = c } }
        let style = ViewportStyle(
            mode: content.mode, coloring: content.coloring, selectedSurface: content.selectedSurface, stale: content.stale,
            dark: dark, scale: Float(scale), accent: content.accent, footprints: content.footprints)
        r.mode = content.mode
        r.setScene(content.scene, style: style)
        if r.camera != camera.wrappedValue { r.camera = camera.wrappedValue }
        let c = coordinator
        let needsFit =
            content.scene != nil && (!camera.wrappedValue.isFitted || c.lastFit != content.fitRequest || c.lastMode != content.mode)
        if needsFit, view.bounds.width > 1 {
            c.lastFit = content.fitRequest
            c.lastMode = content.mode
            input.fit(size: view.bounds.size)
        }
        view.redraw()
    }
}

final class ViewportCoordinator {
    var lastFit = -1
    var lastMode: ViewportMode?
}

#if os(macOS)

/// MTKView with mouse and trackpad camera control: drag pans (2-D) or
/// orbits (3-D; ⇧-drag or right-drag pans), the scroll wheel and pinch zoom
/// about the cursor, trackpad scrolling pans, double-click fits.
final class ViewportMTKView: MTKView {
    var input: ViewportInput?

    func redraw() { needsDisplay = true }

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    private func centred(_ event: NSEvent) -> CGPoint {
        let p = convert(event.locationInWindow, from: nil)
        return CGPoint(x: p.x - bounds.midX, y: p.y - bounds.midY)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        if event.clickCount == 2 { input?.fit(size: bounds.size) }
    }

    override func mouseDragged(with event: NSEvent) {
        input?.drag(CGSize(width: event.deltaX, height: -event.deltaY), pan: event.modifierFlags.contains(.shift))
    }

    override func rightMouseDragged(with event: NSEvent) {
        input?.drag(CGSize(width: event.deltaX, height: -event.deltaY), pan: true)
    }

    override func otherMouseDragged(with event: NSEvent) { rightMouseDragged(with: event) }

    override func scrollWheel(with event: NSEvent) {
        if event.hasPreciseScrollingDeltas && !event.modifierFlags.contains(.command) {
            input?.drag(CGSize(width: event.scrollingDeltaX, height: -event.scrollingDeltaY), pan: true)
        } else {
            let dy = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY * 0.1 : event.scrollingDeltaY
            input?.zoom(pow(1.1, Double(dy)), about: centred(event))
        }
    }

    override func magnify(with event: NSEvent) {
        input?.zoom(1 + Double(event.magnification), about: centred(event))
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        redraw()
    }
}

struct ViewportView: NSViewRepresentable {
    let content: ViewportContent
    @Binding var camera: ViewportCamera
    @Binding var error: String?
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.displayScale) private var displayScale

    func makeCoordinator() -> ViewportCoordinator { ViewportCoordinator() }

    func makeNSView(context: Context) -> ViewportMTKView {
        let view = ViewportBridge.make(input: ViewportInput(), error: $error)
        view.setAccessibilityElement(true)
        view.setAccessibilityRole(.image)
        return view
    }

    func updateNSView(_ view: ViewportMTKView, context: Context) {
        view.setAccessibilityLabel(accessibilityText)
        ViewportBridge.update(
            view, content: content, camera: $camera, coordinator: context.coordinator, dark: colorScheme == .dark,
            scale: displayScale)
    }

    private var accessibilityText: String {
        guard let s = content.scene else { return "Lens layout, not computed yet" }
        let rays = s.polylines.filter { $0.kind == .ray }.count
        return "Lens layout, \(content.mode == .orbit3D ? "3D view" : content.mode == .sectionXZ ? "x–z section" : "meridional section"), \(s.surfaces.count) surfaces, \(rays) rays"
    }
}

#elseif canImport(UIKit)

/// MTKView with touch camera control: one-finger drag pans (2-D) or orbits
/// (3-D), two-finger drag pans, pinch zooms, double tap fits.
final class ViewportMTKView: MTKView {
    var input: ViewportInput?
    private var lastPan = CGPoint.zero

    func redraw() { setNeedsDisplay() }

    override init(frame: CGRect, device: (any MTLDevice)?) {
        super.init(frame: frame, device: device)
        let pan = UIPanGestureRecognizer(target: self, action: #selector(onPan(_:)))
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(onPinch(_:)))
        let tap = UITapGestureRecognizer(target: self, action: #selector(onDoubleTap))
        tap.numberOfTapsRequired = 2
        [pan, pinch, tap].forEach(addGestureRecognizer)
        isAccessibilityElement = true
        accessibilityTraits = .image
    }

    required init(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    @objc private func onPan(_ g: UIPanGestureRecognizer) {
        let t = g.translation(in: self)
        if g.state == .began { lastPan = .zero }
        input?.drag(CGSize(width: t.x - lastPan.x, height: -(t.y - lastPan.y)), pan: g.numberOfTouches > 1)
        lastPan = t
    }

    @objc private func onPinch(_ g: UIPinchGestureRecognizer) {
        let p = g.location(in: self)
        input?.zoom(Double(g.scale), about: CGPoint(x: p.x - bounds.midX, y: bounds.midY - p.y))
        g.scale = 1
    }

    @objc private func onDoubleTap() { input?.fit(size: bounds.size) }
}

struct ViewportView: UIViewRepresentable {
    let content: ViewportContent
    @Binding var camera: ViewportCamera
    @Binding var error: String?
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.displayScale) private var displayScale

    func makeCoordinator() -> ViewportCoordinator { ViewportCoordinator() }

    func makeUIView(context: Context) -> ViewportMTKView { ViewportBridge.make(input: ViewportInput(), error: $error) }

    func updateUIView(_ view: ViewportMTKView, context: Context) {
        view.accessibilityLabel = content.scene == nil ? "Lens layout, not computed yet" : "Lens layout"
        ViewportBridge.update(
            view, content: content, camera: $camera, coordinator: context.coordinator, dark: colorScheme == .dark,
            scale: displayScale)
    }
}

#endif
