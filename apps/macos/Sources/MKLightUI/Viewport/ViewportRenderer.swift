import AppKit
import Metal
import MetalKit
import MKLightEngine
import simd

/// Loads the viewport shaders.
///
/// Order: a precompiled `MKLightViewport.metallib` in the app bundle (built by
/// scripts/build-macos.sh when the Metal toolchain is installed), else the
/// `Viewport.metal` source from the package resource bundle, compiled at run
/// time with safe (IEEE) math. SwiftPM only compiles `.metal` files when the
/// optional Metal toolchain component is installed, so the source path is the
/// one that always works.
@MainActor
enum ShaderLibrary {
    private static var cache: [UInt64: any MTLLibrary] = [:]
    static let resourceBundleName = "MKLight_MKLightUI.bundle"

    static func library(for device: any MTLDevice) throws -> any MTLLibrary {
        if let lib = cache[device.registryID] { return lib }
        let lib = try load(device)
        cache[device.registryID] = lib
        return lib
    }

    private static func load(_ device: any MTLDevice) throws -> any MTLLibrary {
        if let url = Bundle.main.url(forResource: "MKLightViewport", withExtension: "metallib"),
            let lib = try? device.makeLibrary(URL: url)
        {
            return lib
        }
        guard let url = sourceURL() else {
            throw NSError(domain: "MKLight", code: 1, userInfo: [NSLocalizedDescriptionKey: "Viewport.metal not found in \(resourceBundleName)"])
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version3_1
        return try device.makeLibrary(source: source, options: options)
    }

    /// The resource bundle is looked up where SwiftPM's accessor would look,
    /// without its `fatalError` when missing.
    private static func sourceURL() -> URL? {
        let exe = Bundle.main.executableURL?.deletingLastPathComponent()
        let code = Bundle(for: BundleToken.self)
        let candidates = [
            Bundle.main.resourceURL, Bundle.main.bundleURL, exe, code.resourceURL, code.bundleURL.deletingLastPathComponent(),
        ].compactMap { $0 }
        for dir in candidates {
            let b = dir.appendingPathComponent(resourceBundleName)
            if let bundle = Bundle(url: b), let url = bundle.url(forResource: "Viewport", withExtension: "metal") {
                return url
            }
        }
        return nil
    }
}

private final class BundleToken {}

/// Draws an engine `LayoutScene` as anti-aliased thick polylines (instanced
/// screen-space quads, 4× MSAA) plus translucent glass fills. Draws on demand
/// only (the view is paused and redrawn when the scene or camera changes).
@MainActor
final class ViewportRenderer: NSObject, MTKViewDelegate {
    let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let linePipeline: any MTLRenderPipelineState
    private let fillPipeline: any MTLRenderPipelineState
    let sampleCount: Int

    private var segmentBuffer: (any MTLBuffer)?
    private var segmentCount = 0
    private var fillBuffer: (any MTLBuffer)?
    private var fillCount = 0

    private(set) var scene: LayoutScene?
    private var style: ViewportStyle?
    /// Frame of the uploaded geometry; `nil` when it must be rebuilt.
    private(set) var frame: GeometryFrame?
    private var depthRadius = 1.0
    var camera = ViewportCamera()
    var mode = ViewportMode.section2D

    init(device: any MTLDevice, pixelFormat: MTLPixelFormat) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw NSError(domain: "MKLight", code: 2) }
        queue = q
        let samples = device.supportsTextureSampleCount(4) ? 4 : 1
        sampleCount = samples
        let lib = try ShaderLibrary.library(for: device)

        func pipeline(_ vertex: String, _ fragment: String) throws -> any MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: vertex)
            d.fragmentFunction = lib.makeFunction(name: fragment)
            d.rasterSampleCount = samples
            let a = d.colorAttachments[0]!
            a.pixelFormat = pixelFormat
            a.isBlendingEnabled = true
            // Premultiplied-alpha "over".
            a.sourceRGBBlendFactor = .one
            a.sourceAlphaBlendFactor = .one
            a.destinationRGBBlendFactor = .oneMinusSourceAlpha
            a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            return try device.makeRenderPipelineState(descriptor: d)
        }
        linePipeline = try pipeline("line_vertex", "line_fragment")
        fillPipeline = try pipeline("fill_vertex", "fill_fragment")
        super.init()
    }

    /// Replaces the drawn geometry if the scene or style changed.
    func setScene(_ scene: LayoutScene?, style: ViewportStyle) {
        guard scene != self.scene || style != self.style else { return }
        self.scene = scene
        self.style = style
        frame = nil
        if scene == nil {
            segmentCount = 0
            fillCount = 0
        }
    }

    /// (Re)builds the GPU buffers when the scene, style or geometry frame
    /// changed. The frame follows the camera only at deep zoom.
    private func prepareGeometry(size: CGSize) {
        guard let scene, let style else { return }
        let wanted = SceneGeometry.frame(for: scene, camera: camera, mode: mode, size: size)
        guard wanted != frame else { return }
        frame = wanted
        let b = scene.bounds
        let diagonal = simd_length(SIMD3(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z))
        depthRadius = max(diagonal, SceneGeometry.sceneReach(scene, from: camera.center), 1)
        let (segs, fills) = SceneGeometry.build(scene, style: style, frame: wanted)
        segmentCount = segs.count
        fillCount = fills.count
        segmentBuffer = segs.isEmpty ? nil : segs.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        fillBuffer = fills.isEmpty ? nil : fills.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        view.needsDisplay = true
    }

    func draw(in view: MTKView) {
        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
            let cmd = queue.makeCommandBuffer(), let enc = cmd.makeRenderCommandEncoder(descriptor: pass)
        else { return }
        let drawableSize = view.drawableSize
        prepareGeometry(size: view.bounds.size)
        guard let frame else {
            enc.endEncoding()
            cmd.present(drawable)
            cmd.commit()
            return
        }
        var uniforms = Uniforms(
            viewProjection: ViewportMath.viewProjection(camera, mode: mode, frame: frame, size: view.bounds.size, depthRadius: depthRadius),
            viewportSize: SIMD2(Float(drawableSize.width), Float(drawableSize.height)),
            feather: 0.75, pad: 0)
        if fillCount > 0, let fb = fillBuffer {
            enc.setRenderPipelineState(fillPipeline)
            enc.setVertexBuffer(fb, offset: 0, index: 0)
            enc.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: fillCount)
        }
        if segmentCount > 0, let sb = segmentBuffer {
            enc.setRenderPipelineState(linePipeline)
            enc.setVertexBuffer(sb, offset: 0, index: 0)
            enc.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: segmentCount)
        }
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}

/// Matches `Uniforms` in Viewport.metal.
struct Uniforms {
    var viewProjection: simd_float4x4
    var viewportSize: SIMD2<Float>
    var feather: Float
    var pad: Float
}
