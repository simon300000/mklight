import CoreGraphics
import ImageIO
import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

// `image_simulation` (docs/RESULTS.md): the simulated sensor image next to
// its source, side by side, split or blinking, with exposure, zoom and pan;
// and the import of a user image as the run's source.

/// `image_simulation`: the engine's linear-light image (converted to sRGB
/// for display, row 0 at the low-y edge) and the source it was run with:
/// a built-in pattern drawn at the output size exactly as the engine
/// defines it, or the imported image's samples from the run's settings.
struct ImageSimulationView: View {
    enum Compare: String, CaseIterable { case sideBySide = "Side by Side", split = "Split", blink = "Blink" }

    let data: ImageSimulationData
    let source: LinearImage?
    /// Identifies the result (its start time and snapshot).
    let run: String
    @State private var compare = Compare.sideBySide
    @State private var exposure = 0.0
    @State private var zoom: CGFloat = 1
    @State private var zoomBase: CGFloat = 1
    @State private var pan = CGSize.zero
    @State private var panBase = CGSize.zero
    @State private var split: CGFloat = 0.5
    @State private var blinkSource = false
    @State private var showMapping = false
    @State private var cache = Rendered()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The display images, made once per exposure (not on every gesture frame).
    final class Rendered {
        var gain = Double.nan
        var simulated: CGImage?
        var source: CGImage?
        var clipped = 0.0
        var sourceMade = false
        var run = ""

        func update(_ data: ImageSimulationData, source: LinearImage?, gain g: Double, run r: String) {
            if r != run {
                // Another result in the same view: start again.
                run = r
                gain = .nan
                sourceMade = false
            }
            if g != gain {
                gain = g
                simulated = data.image.cgImage(gain: g)
                clipped = data.image.clippedFraction(gain: g)
            }
            if !sourceMade {
                sourceMade = true
                self.source = source?.cgImage()
            }
        }
    }

    init?(output: AnalysisOutput) {
        guard let d = ImageSimulationData(output: output) else { return nil }
        data = d
        source = d.sourceImage
        run = "\(output.runInfo.snapshotHash)-\(output.runInfo.startedUnixMs)-\(output.runInfo.durationMs)"
    }

    private var gain: Double { pow(2, exposure) }

    /// Width / height of the image in mm (pixels may be rectangular).
    private var aspect: CGFloat {
        let px = data.pixelSize
        let w = Double(data.image.width) * (px.x.isFinite && px.x > 0 ? px.x : 1)
        let h = Double(data.image.height) * (px.y.isFinite && px.y > 0 ? px.y : 1)
        return CGFloat(w / max(h, 1e-12))
    }

    var body: some View {
        // The source is shown at unit exposure: it is the scene, not a sensor reading.
        let _ = cache.update(data, source: source, gain: gain, run: run)
        let simulated = cache.simulated, src = cache.source
        VStack(alignment: .leading, spacing: Space.s) {
            controls
            GeometryReader { geo in
                content(simulated: simulated, source: src, size: geo.size)
                    .frame(width: geo.size.width, height: geo.size.height)
            }
            caption
        }
        .padding(Space.m)
    }

    // MARK: Controls

    private var controls: some View {
        FlowLayout(spacing: Space.m, lineSpacing: Space.xs) {
            Picker("Compare", selection: $compare) {
                ForEach(Compare.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .disabled(source == nil)
            HStack(spacing: Space.xs) {
                Text("Exposure").foregroundStyle(.secondary)
                Slider(value: $exposure, in: -4...4, step: 0.5).frame(width: 90).accessibilityLabel("Exposure")
                    .accessibilityValue(String(format: "%+.1f stops", exposure))
                Text(String(format: "%+.1f EV", exposure)).numeric().frame(width: 52, alignment: .leading)
            }
            HStack(spacing: Space.xs) {
                Button { setZoom(zoom / 2) } label: { Image(systemName: "minus.magnifyingglass") }
                    .help("Zoom out").accessibilityLabel("Zoom out").disabled(zoom <= 1)
                Text(String(format: "%.0f%%", zoom * 100)).numeric().frame(minWidth: 40)
                Button { setZoom(zoom * 2) } label: { Image(systemName: "plus.magnifyingglass") }
                    .help("Zoom in").accessibilityLabel("Zoom in").disabled(zoom >= 32)
                Button("Fit") { setZoom(1) }.help("Fit the image and reset the pan").disabled(zoom == 1 && pan == .zero)
            }
            .buttonStyle(.borderless)
            Button {
                showMapping = true
            } label: {
                Label("Field Mapping", systemImage: "tablecells")
            }
            .buttonStyle(.borderless)
            .disabled(data.nodes.isEmpty)
            .popover(isPresented: $showMapping) { FieldMappingTable(nodes: data.nodes).padding(Space.l) }
        }
        .font(.callout)
        .controlSize(.small)
    }

    private func setZoom(_ z: CGFloat) {
        zoom = min(max(z, 1), 32)
        zoomBase = zoom
        if zoom == 1 {
            pan = .zero
            panBase = .zero
        }
    }

    // MARK: Content

    @ViewBuilder
    private func content(simulated: CGImage?, source src: CGImage?, size: CGSize) -> some View {
        let mode = src == nil ? Compare.sideBySide : compare
        switch mode {
        case .sideBySide:
            let wide = size.width / max(size.height, 1) > aspect * 1.2
            let layout = wide ? AnyLayout(HStackLayout(spacing: Space.m)) : AnyLayout(VStackLayout(spacing: Space.m))
            layout {
                if let src { frame(src, title: data.sourceTitle, isSource: true) }
                frame(simulated, title: "Simulated", isSource: false)
            }
        case .split:
            splitView(simulated: simulated, source: src)
        case .blink:
            blinkView(simulated: simulated, source: src)
        }
    }

    /// One image in a fitted frame with its title, zoomed and panned.
    private func frame(_ image: CGImage?, title: String, isSource: Bool) -> some View {
        ZoomedImage(image: image, aspect: aspect, zoom: zoom, pan: pan, title: title)
            .gesture(zoomGesture)
            .simultaneousGesture(panGesture)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(isSource ? "Source: \(data.sourceTitle)" : "Simulated image, \(data.image.width) by \(data.image.height) pixels")
            .accessibilityAddTraits(.isImage)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func splitView(simulated: CGImage?, source: CGImage?) -> some View {
        GeometryReader { geo in
            let fit = Self.fitted(aspect: aspect, in: geo.size)
            ZStack(alignment: .topLeading) {
                ZoomedImage(image: simulated, aspect: aspect, zoom: zoom, pan: pan)
                ZoomedImage(image: source, aspect: aspect, zoom: zoom, pan: pan)
                    .mask(alignment: .leading) {
                        Rectangle().frame(width: max(fit.minX + fit.width * split, 0)).frame(maxWidth: .infinity, alignment: .leading)
                    }
                ImageTag(text: data.sourceTitle).offset(x: fit.minX, y: fit.minY)
                ImageTag(text: "Simulated").frame(width: fit.width, alignment: .trailing).offset(x: fit.minX, y: fit.minY)
                // The divider and its handle.
                let x = fit.minX + fit.width * split
                Rectangle().fill(.white).frame(width: 1.5, height: fit.height).shadow(radius: 1).offset(x: x - 0.75, y: fit.minY)
                Circle()
                    .fill(.white)
                    .frame(width: 18, height: 18)
                    .overlay(Image(systemName: "arrow.left.and.right").font(.system(size: 9, weight: .bold)).foregroundStyle(.black))
                    .shadow(radius: 1.5)
                    .offset(x: x - 9, y: fit.midY - 9)
                    .highPriorityGesture(
                        DragGesture(minimumDistance: 0).onChanged { v in
                            split = min(max((v.location.x - fit.minX) / max(fit.width, 1), 0), 1)
                        }
                    )
                    .accessibilityElement()
                    .accessibilityLabel("Split position")
                    .accessibilityValue(String(format: "%.0f percent source", split * 100))
                    .accessibilityAdjustableAction { d in
                        split = min(max(split + (d == .increment ? 0.1 : -0.1), 0), 1)
                    }
            }
            .gesture(zoomGesture)
            .simultaneousGesture(panGesture)
        }
    }

    @ViewBuilder
    private func blinkView(simulated: CGImage?, source: CGImage?) -> some View {
        if reduceMotion {
            // No flashing: show one at a time, switched by a click.
            frame(blinkSource ? source : simulated, title: blinkSource ? data.sourceTitle : "Simulated", isSource: blinkSource)
                .onTapGesture { blinkSource.toggle() }
                .help("Click to switch between the source and the simulated image")
        } else {
            TimelineView(.periodic(from: .now, by: 0.8)) { t in
                let showSource = Int(t.date.timeIntervalSinceReferenceDate / 0.8) % 2 == 0
                frame(showSource ? source : simulated, title: showSource ? data.sourceTitle : "Simulated", isSource: showSource)
            }
        }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { v in zoom = min(max(zoomBase * v.magnification, 1), 32) }
            .onEnded { _ in
                zoomBase = zoom
                if zoom == 1 { pan = .zero; panBase = .zero }
            }
    }

    private var panGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { v in
                guard zoom > 1 else { return }
                pan = CGSize(width: panBase.width + v.translation.width, height: panBase.height + v.translation.height)
            }
            .onEnded { _ in panBase = pan }
    }

    /// The rectangle of an image of `aspect` fitted in `size`, centred.
    static func fitted(aspect: CGFloat, in size: CGSize) -> CGRect {
        let w = min(size.width, size.height * aspect), h = w / max(aspect, 1e-6)
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    private var caption: some View {
        let clipped = cache.clipped
        return HStack(spacing: Space.l) {
            Text("\(data.image.width) × \(data.image.height) px" + (data.image.channels == 3 ? ", RGB" : ", monochrome"))
            if data.pixelSize.x.isFinite, abs(data.pixelSize.x - data.pixelSize.y) > 1e-9 * data.pixelSize.x {
                Text("Pixel \(Fmt.length(data.pixelSize.x)) × \(Fmt.length(data.pixelSize.y))")
            }
            if clipped > 0.001 { Text(String(format: "%.1f%% clipped", clipped * 100)).foregroundStyle(.orange) }
            Spacer(minLength: 0)
            if case .pattern = data.source {
                Text("Source drawn as the engine defines it").help(
                    "The engine does not return the source; built-in patterns are drawn here at the output size with the engine's definition, for comparison.")
            }
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
        .lineLimit(1)
    }
}

/// A label on an image: content-layer, small and legible on any picture.
private struct ImageTag: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(.black.opacity(0.55), in: .rect(cornerRadius: 4))
            .padding(Space.xs)
            .accessibilityHidden(true)
    }
}

/// An image fitted in its frame, magnified about the centre and panned,
/// pixels kept sharp.
struct ZoomedImage: View {
    let image: CGImage?
    let aspect: CGFloat
    let zoom: CGFloat
    let pan: CGSize
    var title: String?

    var body: some View {
        GeometryReader { geo in
            let fit = ImageSimulationView.fitted(aspect: aspect, in: geo.size)
            ZStack(alignment: .topLeading) {
                Rectangle().fill(.black)
                if let image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .interpolation(.none)
                        .frame(width: fit.width, height: fit.height)
                        .scaleEffect(zoom)
                        .offset(pan)
                } else {
                    Text("No preview").font(.caption).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                if let title { ImageTag(text: title) }
            }
            .frame(width: fit.width, height: fit.height)
            .clipped()
            .overlay(Rectangle().strokeBorder(.separator))
            .offset(x: fit.minX, y: fit.minY)
        }
    }
}

/// The field nodes of an image simulation: ideal and real image points,
/// distortion and relative illumination.
struct FieldMappingTable: View {
    let nodes: [ImageSimulationData.Node]

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            Text("Field Mapping").font(.headline)
            Text("PSF grid nodes; the image is mapped bilinearly between them.").font(.caption).foregroundStyle(.secondary)
            ScrollView {
                Grid(alignment: .trailing, horizontalSpacing: Space.l, verticalSpacing: 3) {
                    GridRow {
                        Text("Field (Hx, Hy)").gridColumnAlignment(.leading)
                        Text("Ideal (mm)")
                        Text("Real (mm)")
                        Text("Distortion")
                        Text("Rel. illum.")
                    }
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    Divider().gridCellUnsizedAxes(.horizontal)
                    ForEach(Array(nodes.enumerated()), id: \.offset) { _, n in
                        GridRow {
                            Text(String(format: "%+.2f, %+.2f", n.field.x, n.field.y))
                            Text(String(format: "%.4f, %.4f", n.ideal.x, n.ideal.y))
                            Text(String(format: "%.4f, %.4f", n.real.x, n.real.y))
                            Text(n.distortionPercent.isFinite ? String(format: "%+.3f %%", n.distortionPercent) : "—")
                            Text(n.relativeIllumination.isFinite ? String(format: "%.3f", n.relativeIllumination) : "—")
                        }
                        .font(.caption.monospacedDigit())
                    }
                }
                .textSelection(.enabled)
            }
            .frame(maxHeight: 320)
        }
        .fixedSize(horizontal: true, vertical: false)
    }
}

// MARK: - Importing a source image

/// A user image as the simulation's source. Any format ImageIO reads is
/// decoded (EXIF orientation applied), cropped to fill the output's aspect,
/// resampled in linear light to the output size (at most 512 px) and sent
/// as a `{kind: buffer}` source with row 0 at the low-y edge, the engine's
/// convention for buffers and its output. The samples live in the pane's
/// settings for the session only: the project keeps the file's path
/// (`source_file`, which the engine never sees) and the image is decoded
/// again when the project opens.
@MainActor
enum ImageSourceImport {
    /// The app-only setting holding the imported file's path.
    nonisolated static let fileKey = "source_file"
    nonisolated static let maxSide = 512
    /// The app-only setting recording the field aspect the samples were cropped to.
    nonisolated static let aspectKey = "source_field_aspect"

    private static var decoded: [String: CGImage] = [:]

    /// The settings to save in the project: an imported image's samples are
    /// dropped (its path stays), so it is decoded again on open.
    nonisolated static func persistable(_ settings: [String: JSONValue]) -> [String: JSONValue] {
        guard settings[fileKey] != nil, settings["source"]?["kind"]?.string == "buffer" else { return settings }
        var s = settings
        s["source"] = nil
        s[aspectKey] = nil
        return s
    }

    /// The output size the source is resampled to: the run's `size`, scaled
    /// down to at most `maxSide` on its longer side.
    nonisolated static func targetSize(_ size: JSONValue?) -> (Int, Int) {
        let a = size?.array?.compactMap(\.doubleValue) ?? []
        let w = a.count == 2 ? max(Int(a[0]), 2) : 64, h = a.count == 2 ? max(Int(a[1]), 2) : 64
        let s = min(1, Double(maxSide) / Double(max(w, h)))
        return (max(Int((Double(w) * s).rounded()), 2), max(Int((Double(h) * s).rounded()), 2))
    }

    /// Decodes an image file (any format ImageIO reads), at most 2048 px.
    static func decode(_ url: URL) -> CGImage? {
        if let c = decoded[url.path] { return c }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 2048, kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary) else { return nil }
        decoded[url.path] = image
        return image
    }

    /// The image cropped to the field's aspect (`aspect`, width / height in
    /// mm; default the pixel grid's) and resampled in linear light to
    /// `width × height` (Core Graphics colour-manages it to linear sRGB),
    /// row 0 at the bottom; one channel for grey images, else three. The
    /// engine stretches the source over the image field, so cropping to the
    /// field's aspect keeps the picture's proportions.
    nonisolated static func linear(_ image: CGImage, width: Int, height: Int, aspect: Double? = nil) -> LinearImage? {
        guard width > 0, height > 0, let space = CGColorSpace(name: CGColorSpace.linearSRGB) else { return nil }
        let grey = image.colorSpace?.model == .monochrome
        var buf = [Float](repeating: 0, count: width * height * 4)
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.floatComponents.rawValue)
            .union(.byteOrder32Little)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 32, bytesPerRow: width * 16,
                space: space, bitmapInfo: info.rawValue)
            else { return false }
            ctx.interpolationQuality = .high
            // Aspect fill: crop the image's longer side to the field's aspect,
            // then stretch the crop over the pixel grid.
            let iw = CGFloat(image.width), ih = CGFloat(image.height)
            let a = CGFloat(aspect.flatMap { $0.isFinite && $0 > 0 ? $0 : nil } ?? Double(width) / Double(height))
            let cw = min(iw, ih * a), ch = cw / a
            let sx = CGFloat(width) / cw, sy = CGFloat(height) / ch
            ctx.draw(image, in: CGRect(x: -(iw - cw) / 2 * sx, y: -(ih - ch) / 2 * sy, width: iw * sx, height: ih * sy))
            return true
        }
        guard ok else { return nil }
        let c = grey ? 1 : 3
        var data = [Double](repeating: 0, count: width * height * c)
        for y in 0..<height {
            // Bitmap memory starts at the top row; the engine's row 0 is the bottom.
            let srcRow = height - 1 - y
            for x in 0..<width {
                let s = (srcRow * width + x) * 4, o = (y * width + x) * c
                if grey {
                    data[o] = max(Double(buf[s]), 0)
                } else {
                    for k in 0..<3 { data[o + k] = max(Double(buf[s + k]), 0) }
                }
            }
        }
        return LinearImage(width: width, height: height, channels: c, data: data)
    }

    /// The field's aspect (width / height in mm) of the pane's last result.
    static func fieldAspect(_ pane: AnalysisPane) -> Double? {
        guard pane.analysis == "image_simulation", let j = pane.output?.json, let s = j["size"]?.nfArray,
            let p = j["pixel_size_mm"]?.nfArray, s.count == 2, p.count == 2
        else { return nil }
        let a = s[0] * p[0] / (s[1] * p[1])
        return a.isFinite && a > 0 ? a : nil
    }

    /// Uses an image file as the pane's source.
    static func use(_ url: URL, in pane: AnalysisPane) -> Bool {
        guard let image = decode(url) else { return false }
        let (w, h) = targetSize(pane.value("size"))
        guard let lin = linear(image, width: w, height: h, aspect: fieldAspect(pane)) else { return false }
        var s = pane.settings
        s[fileKey] = .string(url.path)
        s["source"] = lin.sourceJSON
        s[aspectKey] = fieldAspect(pane).map { .number($0) }
        pane.setSettings(s)
        return true
    }

    /// Uses a built-in pattern (and forgets an imported file).
    static func use(pattern: String, in pane: AnalysisPane) {
        var s = pane.settings
        s[fileKey] = nil
        s[aspectKey] = nil
        s["source"] = .object(["kind": .string("pattern"), "pattern": .string(pattern)])
        pane.setSettings(s)
    }

    /// Makes sure an imported file's samples are in the settings at the
    /// current output size (after opening a project, or a size change).
    /// Returns false when the file cannot be read; the run then keeps the
    /// default grid pattern.
    @discardableResult
    static func refresh(_ pane: AnalysisPane) -> Bool {
        guard let path = pane.settings[fileKey]?.string else { return true }
        let (w, h) = targetSize(pane.value("size"))
        let aspect = fieldAspect(pane)
        if let cur = LinearImage(source: pane.settings["source"]), cur.width == w, cur.height == h,
            pane.settings[aspectKey]?.doubleValue == aspect
        {
            return true
        }
        guard let image = decode(URL(fileURLWithPath: path)), let lin = linear(image, width: w, height: h, aspect: aspect) else {
            if pane.settings["source"]?["kind"]?.string == "buffer" { pane.set("source", nil) }
            return false
        }
        var s = pane.settings
        s["source"] = lin.sourceJSON
        s[aspectKey] = aspect.map { .number($0) }
        pane.setSettings(s)
        return true
    }
}
