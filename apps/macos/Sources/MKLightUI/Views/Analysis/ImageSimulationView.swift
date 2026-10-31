import CoreGraphics
import ImageIO
import MKLightEngine
import MKLightModel
import SwiftUI
import simd
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
    /// The pane, for the field of view and resolution controls.
    let pane: AnalysisPane?
    /// The built-in pattern the run's (antialiased) buffer was drawn from.
    let pattern: String?
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

    init?(output: AnalysisOutput, pane: AnalysisPane? = nil) {
        guard let d = ImageSimulationData(output: output) else { return nil }
        data = d
        source = d.sourceImage
        self.pane = pane
        // A buffer drawn from a built-in pattern (not an imported file) keeps the pattern's name.
        if case .buffer = d.source, let p = pane, p.settings[ImageSourceImport.fileKey] == nil,
            let name = p.value("source")?["pattern"]?.string
        {
            pattern = name
        } else {
            pattern = nil
        }
        run = "\(output.runInfo.snapshotHash)-\(output.runInfo.startedUnixMs)-\(output.runInfo.durationMs)"
    }

    private var sourceTitle: String {
        guard let pattern else { return data.sourceTitle }
        return (ImageSimulationData.patterns.first { $0.id == pattern }?.title ?? pattern) + " pattern"
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
            if let pane { ImageViewControls(pane: pane, data: data) }
            SegmentedOrMenu(title: "Compare", options: Compare.allCases.map { ($0, $0.rawValue) }, selection: $compare)
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
                if let src { frame(src, title: sourceTitle, isSource: true) }
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
            .accessibilityLabel(isSource ? "Source: \(sourceTitle)" : "Simulated image, \(data.image.width) by \(data.image.height) pixels")
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
                ImageTag(text: sourceTitle).offset(x: fit.minX, y: fit.minY)
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
            frame(blinkSource ? source : simulated, title: blinkSource ? sourceTitle : "Simulated", isSource: blinkSource)
                .onTapGesture { blinkSource.toggle() }
                .help("Click to switch between the source and the simulated image")
        } else {
            TimelineView(.periodic(from: .now, by: 0.8)) { t in
                let showSource = Int(t.date.timeIntervalSinceReferenceDate / 0.8) % 2 == 0
                frame(showSource ? source : simulated, title: showSource ? sourceTitle : "Simulated", isSource: showSource)
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

    /// The patch's field position and where it lands on the sensor.
    private func patchHelp(_ c: SIMD2<Double>) -> String {
        var t = "Patch centred at normalised field " + ImageSimulationSettings.coordinates(c)
        if let n = data.nodes.min(by: { simd_distance($0.field, c) < simd_distance($1.field, c) }), n.real.x.isFinite {
            t += ", image point (" + String(format: "%.3f, %.3f", n.real.x, n.real.y) + ") mm"
        }
        return t + "."
    }

    private var psfAuto: Bool { (pane?.value("optics")?["psf_size"]?.int ?? 0) == 0 }

    private var caption: some View {
        let clipped = cache.clipped
        return HStack(spacing: Space.l) {
            Text("\(data.image.width) × \(data.image.height) px" + (data.image.channels == 3 ? ", RGB" : ", monochrome"))
            if data.pixelSize.x.isFinite, abs(data.pixelSize.x - data.pixelSize.y) > 1e-9 * data.pixelSize.x {
                Text("Pixel \(Fmt.length(data.pixelSize.x)) × \(Fmt.length(data.pixelSize.y))")
            }
            if let c = ImageSimulationSettings.fieldCenter(fromRun: data) {
                Text("At " + ImageSimulationSettings.centerTitle(c, points: ImageSimulationSettings.fieldPoints(pane?.workspace?.snapshot?.system)))
                    .help(patchHelp(c))
            }
            if let n = data.psfSize, let d = data.psfSpacing, d.isFinite {
                Text("PSF \(n) × \(Fmt.length(d))")
                    .help("The PSF window and sample spacing this run used\(psfAuto ? ", chosen by the engine from the blur and diffraction" : "").")
            }
            if clipped > 0.001 { Text(String(format: "%.1f%% clipped", clipped * 100)).foregroundStyle(.orange) }
            Spacer(minLength: 0)
            if pattern != nil {
                Text("Pattern area-sampled (antialiased)").help(
                    "The built-in pattern is drawn by the app with the engine's definition, averaged over each pixel (4 × 4 samples), and sent to the engine as the source image.")
            } else if case .pattern = data.source {
                Text("Source drawn as the engine defines it").help(
                    "The engine does not return the source; built-in patterns are drawn here at the output size with the engine's definition, for comparison.")
            }
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
        .lineLimit(1)
    }
}

/// The simulated patch of the field (a square of the normalised field
/// around a chosen centre: on axis, a design field point, or a point picked
/// on a small field diagram) and the resolution, with what they mean on the
/// sensor: the patch's width and the pixel size, so blur and aberrations
/// can be made visible.
struct ImageViewControls: View {
    let pane: AnalysisPane
    let data: ImageSimulationData
    @State private var picking = false

    private var extent: Double { ImageSimulationSettings.fieldExtent(pane) }
    private var size: Int { ImageSimulationSettings.size(pane) }
    private var center: SIMD2<Double> { ImageSimulationSettings.fieldCenter(pane) }
    private var system: OpticalSystem? { pane.workspace?.designSnapshot?.system ?? pane.workspace?.snapshot?.system }
    private var points: [ImageSimulationSettings.FieldPoint] { ImageSimulationSettings.fieldPoints(system) }

    /// Image-plane half-width (mm) per unit of field extent, from the result.
    private var mmPerExtent: Double? {
        let w = data.pixelSize.x * Double(data.image.width)
        let e = ImageSimulationSettings.fieldExtent(fromRun: data)
        guard w.isFinite, w > 0, let e, e > 0 else { return nil }
        return w / 2 / e
    }

    var body: some View {
        HStack(spacing: Space.s) {
            Menu {
                Picker("Field of view", selection: Binding(get: { extent }, set: { ImageSimulationSettings.setFieldExtent($0, in: pane) })) {
                    ForEach(ImageSimulationSettings.extents, id: \.self) { e in
                        Text(label(e)).tag(e)
                    }
                }
                .pickerStyle(.inline)
            } label: {
                Text("View " + ImageSimulationSettings.extentTitle(extent))
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("The simulated part of the field: a square around the patch centre, as a fraction of the field. A smaller patch has finer pixels, so the blur shows.")
            .accessibilityLabel("Field of view")
            .accessibilityValue(ImageSimulationSettings.extentTitle(extent))
            Menu {
                Picker("Patch centre", selection: Binding(get: { center }, set: { ImageSimulationSettings.setFieldCenter($0, in: pane) })) {
                    ForEach(points) { p in Text(p.title).tag(p.field) }
                    if !points.contains(where: { $0.field == center }) {
                        Text(ImageSimulationSettings.centerTitle(center, points: points)).tag(center)
                    }
                }
                .pickerStyle(.inline)
                Divider()
                Button("Pick on the Field…") { picking = true }
            } label: {
                Text("At " + ImageSimulationSettings.centerTitle(center, points: points))
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Where the patch is centred: on the axis, at a design field point, or anywhere you pick on the field.")
            .accessibilityLabel("Patch centre")
            .accessibilityValue(ImageSimulationSettings.centerTitle(center, points: points))
            .popover(isPresented: $picking) {
                FieldCenterPicker(center: center, extent: extent, points: points) { ImageSimulationSettings.setFieldCenter($0, in: pane) }
                    .padding(Space.l)
            }
            Menu {
                Picker("Resolution", selection: Binding(get: { size }, set: { ImageSimulationSettings.setSize($0, in: pane) })) {
                    ForEach(ImageSimulationSettings.sizes, id: \.self) { n in Text("\(n) × \(n) px").tag(n) }
                }
                .pickerStyle(.inline)
            } label: {
                Text("\(size) px")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Output pixels per side")
            .accessibilityLabel("Resolution")
            .accessibilityValue("\(size) pixels")
        }
    }

    private func label(_ e: Double) -> String {
        var t = ImageSimulationSettings.extentTitle(e)
        if let k = mmPerExtent {
            let width = 2 * k * e
            t += " · " + Fmt.length(width) + " wide, pixel " + Fmt.length(width / Double(size))
        }
        return t
    }
}

/// A small diagram of the normalised field: the field disk, the design's
/// field points and the simulated patch. A click (or a drag) moves the
/// patch centre; it snaps to a field point nearby and to the axis.
struct FieldCenterPicker: View {
    let center: SIMD2<Double>
    let extent: Double
    let points: [ImageSimulationSettings.FieldPoint]
    let set: (SIMD2<Double>) -> Void
    @State private var dragged: SIMD2<Double>?

    private let side: CGFloat = 200
    /// Normalised field units shown on each side of the axis.
    private let span = 1.15

    var body: some View {
        let c = dragged ?? center
        VStack(alignment: .leading, spacing: Space.s) {
            Text("Patch Centre").font(.headline)
            Canvas { ctx, size in
                let k = size.width / 2 / span
                func pt(_ f: SIMD2<Double>) -> CGPoint { CGPoint(x: size.width / 2 + f.x * k, y: size.height / 2 - f.y * k) }
                let disk = Path(ellipseIn: CGRect(x: size.width / 2 - k, y: size.height / 2 - k, width: 2 * k, height: 2 * k))
                ctx.fill(disk, with: .color(.secondary.opacity(0.12)))
                ctx.stroke(disk, with: .color(.secondary.opacity(0.6)), lineWidth: 1)
                var axes = Path()
                axes.move(to: CGPoint(x: 0, y: size.height / 2))
                axes.addLine(to: CGPoint(x: size.width, y: size.height / 2))
                axes.move(to: CGPoint(x: size.width / 2, y: 0))
                axes.addLine(to: CGPoint(x: size.width / 2, y: size.height))
                ctx.stroke(axes, with: .color(.secondary.opacity(0.35)), style: StrokeStyle(lineWidth: 0.5, dash: [3, 3]))
                for p in points {
                    let q = pt(p.field)
                    ctx.fill(Path(ellipseIn: CGRect(x: q.x - 3, y: q.y - 3, width: 6, height: 6)), with: .color(.secondary))
                }
                let lo = pt(c - SIMD2(extent, extent)), hi = pt(c + SIMD2(extent, extent))
                let patch = CGRect(x: lo.x, y: hi.y, width: max(hi.x - lo.x, 3), height: max(lo.y - hi.y, 3))
                ctx.fill(Path(patch), with: .color(.accentColor.opacity(0.25)))
                ctx.stroke(Path(patch), with: .color(.accentColor), lineWidth: 1.5)
                let m = pt(c)
                ctx.fill(Path(ellipseIn: CGRect(x: m.x - 2.5, y: m.y - 2.5, width: 5, height: 5)), with: .color(.accentColor))
            }
            .frame(width: side, height: side)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { v in dragged = field(at: v.location) }
                    .onEnded { v in
                        dragged = nil
                        set(field(at: v.location))
                    }
            )
            .accessibilityElement()
            .accessibilityLabel("Field diagram")
            .accessibilityValue("Patch centre " + ImageSimulationSettings.centerTitle(c, points: points))
            Text(ImageSimulationSettings.centerTitle(c, points: points)).font(.callout.monospacedDigit())
            if ImageSimulationSettings.outsideField(center: c, extent: extent) {
                Label("Part of the patch lies outside the design's field.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            Text("Click to centre the patch; it snaps to the axis and the field points.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .frame(width: side)
    }

    private func field(at p: CGPoint) -> SIMD2<Double> {
        let k = Double(side) / 2 / span
        let f = SIMD2((Double(p.x) - Double(side) / 2) / k, (Double(side) / 2 - Double(p.y)) / k)
        return ImageSimulationSettings.snap(f, to: points.map(\.field), within: 0.06)
    }
}

/// The app's settings of `image_simulation` panes: the starting settings
/// (a patch of 10 % of the field at 256 px, with the engine's automatic PSF
/// window and pupil sampling), the field-of-view, patch-centre and
/// resolution choices, and the antialiased built-in patterns sent as the
/// source.
@MainActor
enum ImageSimulationSettings {
    /// Field extents offered (half-width of the square, in units of the
    /// normalised field); 1/√2 inscribes the square in the field disk.
    nonisolated static let fullExtent = 0.5.squareRoot()
    static let extents: [Double] = [fullExtent, 0.5, 0.25, 0.1, 0.05, 0.02]
    static let sizes: [Int] = [128, 256, 512, 1024]

    /// Starting settings: a patch of 10 % of the field at 256 px on the
    /// axis. The engine sizes the PSF window and the pupil grid itself;
    /// its default view (the inscribed square at 64 px) has pixels larger
    /// than the spot, so the blur would not show.
    nonisolated static func starting(defaults: JSONValue?) -> [String: JSONValue] {
        var optics = defaults?["optics"]?.object ?? [:]
        optics["field_extent"] = .number(0.1)
        return [
            "size": .array([.integer(256), .integer(256)]),
            "optics": .object(optics),
            "source": .object(["kind": .string("pattern"), "pattern": .string("siemens_star")]),
        ]
    }

    /// Whether `settings` are the engine's defaults, as projects saved
    /// before the starting settings existed hold them: nothing set, or only
    /// values equal to the defaults.
    nonisolated static func isEngineDefault(_ settings: [String: JSONValue], defaults: JSONValue?) -> Bool {
        settings.allSatisfy { k, v in defaults?[k] == v }
    }

    /// Settings saved with the earlier starting PSF window and pupil (161
    /// samples, a 96 grid of 97 rays, chosen when the engine had no
    /// automatic sampling) go back to automatic; anything else is kept.
    nonisolated static func migrated(_ settings: [String: JSONValue]) -> [String: JSONValue]? {
        guard var optics = settings["optics"]?.object, optics["psf_size"]?.int == 161,
            var pupil = optics["pupil"]?.object, pupil["grid"]?.int == 96, pupil["rays"]?.int == 97
        else { return nil }
        optics["psf_size"] = .integer(0)
        pupil["grid"] = .integer(0)
        pupil["rays"] = .integer(65)
        optics["pupil"] = .object(pupil)
        var s = settings
        s["optics"] = .object(optics)
        return s
    }

    static func fieldExtent(_ pane: AnalysisPane) -> Double { pane.value("optics")?["field_extent"]?.doubleValue ?? fullExtent }

    /// The run's field extent: half the span of its field mapping nodes.
    nonisolated static func fieldExtent(fromRun data: ImageSimulationData) -> Double? {
        let xs = data.nodes.map(\.field.x)
        guard let lo = xs.min(), let hi = xs.max(), hi > lo else { return nil }
        return (hi - lo) / 2
    }

    /// The run's patch centre: the middle of its field mapping nodes.
    nonisolated static func fieldCenter(fromRun data: ImageSimulationData) -> SIMD2<Double>? {
        let xs = data.nodes.map(\.field.x), ys = data.nodes.map(\.field.y)
        guard let x0 = xs.min(), let x1 = xs.max(), let y0 = ys.min(), let y1 = ys.max() else { return nil }
        return SIMD2((x0 + x1) / 2, (y0 + y1) / 2)
    }

    static func size(_ pane: AnalysisPane) -> Int {
        Int(pane.value("size")?.array?.first?.doubleValue ?? 64)
    }

    static func fieldCenter(_ pane: AnalysisPane) -> SIMD2<Double> {
        let a = pane.value("optics")?["field_center"]?.array?.compactMap(\.doubleValue) ?? []
        return a.count == 2 ? SIMD2(a[0], a[1]) : .zero
    }

    static func setFieldExtent(_ e: Double, in pane: AnalysisPane) {
        var optics = pane.value("optics")?.object ?? [:]
        optics["field_extent"] = .number(e)
        pane.set("optics", .object(optics))
    }

    static func setFieldCenter(_ c: SIMD2<Double>, in pane: AnalysisPane) {
        var optics = pane.value("optics")?.object ?? [:]
        optics["field_center"] = centerJSON(c)
        pane.set("optics", .object(optics))
    }

    /// `[hx, hy]`, rounded to 1e-4 so picked points read cleanly.
    nonisolated static func centerJSON(_ c: SIMD2<Double>) -> JSONValue {
        func r(_ v: Double) -> Double { let x = (v * 1e4).rounded() / 1e4; return x == 0 ? 0 : x }
        return .array([.number(r(c.x)), .number(r(c.y))])
    }

    static func setSize(_ n: Int, in pane: AnalysisPane) {
        pane.set("size", .array([.integer(Int64(n)), .integer(Int64(n))]))
    }

    nonisolated static func extentTitle(_ e: Double) -> String {
        if e >= 1 { return "full square" }
        if abs(e - fullExtent) < 1e-6 { return "full field" }
        return String(format: "%g %%", e * 100)
    }

    /// A choice of patch centre: the axis or a design field point, in
    /// normalised field coordinates.
    struct FieldPoint: Identifiable, Hashable {
        var title: String
        var field: SIMD2<Double>
        var id: String { title }
    }

    /// The axis and the design's field points (normalised by the largest
    /// radial field, CONVENTIONS §3), each listed once.
    nonisolated static func fieldPoints(_ system: OpticalSystem?) -> [FieldPoint] {
        var out = [FieldPoint(title: "On axis", field: .zero)]
        guard let s = system else { return out }
        let f = s.fields.fields
        let r = f.map { ($0.x * $0.x + $0.y * $0.y).squareRoot() }.max() ?? 0
        guard r > 0 else { return out }
        for (k, p) in f.enumerated() {
            let h = SIMD2(p.x / r, p.y / r)
            guard !out.contains(where: { simd_distance($0.field, h) < 1e-9 }) else { continue }
            out.append(FieldPoint(title: "Field \(k + 1) " + coordinates(h), field: h))
        }
        return out
    }

    nonisolated static func coordinates(_ c: SIMD2<Double>) -> String {
        func f(_ v: Double) -> String {
            Fmt.signed(v, decimals: 2).trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "-", with: "−")
        }
        return "(" + f(c.x) + ", " + f(c.y) + ")"
    }

    /// "on axis", "Field 3 (0.00, 1.00)" or "(0.30, −0.45)".
    nonisolated static func centerTitle(_ c: SIMD2<Double>, points: [FieldPoint]) -> String {
        if c == .zero { return "on axis" }
        if let p = points.first(where: { simd_distance($0.field, c) < 1e-4 }) { return p.title }
        return coordinates(c)
    }

    /// The nearest of `targets` (and the axis) within `within`, else `p`.
    nonisolated static func snap(_ p: SIMD2<Double>, to targets: [SIMD2<Double>], within d: Double) -> SIMD2<Double> {
        let near = ([SIMD2<Double>.zero] + targets).min { simd_distance($0, p) < simd_distance($1, p) }
        if let near, simd_distance(near, p) < d { return near }
        return p
    }

    /// Whether a corner of the patch lies outside the unit field disk.
    nonisolated static func outsideField(center c: SIMD2<Double>, extent e: Double) -> Bool {
        let far = SIMD2(abs(c.x) + e, abs(c.y) + e)
        return (far.x * far.x + far.y * far.y).squareRoot() > 1 + 1e-9
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
                            Text(Fmt.signed(n.field.x, decimals: 2) + ", " + Fmt.signed(n.field.y, decimals: 2))
                            Text(String(format: "%.4f, %.4f", n.ideal.x, n.ideal.y))
                            Text(String(format: "%.4f, %.4f", n.real.x, n.real.y))
                            Text(n.distortionPercent.isFinite ? Fmt.signed(n.distortionPercent, decimals: 3) + " %" : "—")
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

    private static var patternCache: [String: JSONValue] = [:]

    /// The settings sent to the engine: a built-in pattern becomes its
    /// antialiased (area-sampled) image at the run's size, as a buffer
    /// source; the engine itself samples patterns at pixel centres.
    static func engineSettings(_ analysis: String, _ settings: [String: JSONValue], defaults: JSONValue?) -> [String: JSONValue] {
        guard analysis == "image_simulation" else { return settings }
        let source = settings["source"] ?? defaults?["source"]
        guard source?["kind"]?.string == "pattern", let name = source?["pattern"]?.string else { return settings }
        let a = (settings["size"] ?? defaults?["size"])?.array?.compactMap(\.doubleValue) ?? []
        let w = a.count == 2 ? Int(a[0]) : 64, h = a.count == 2 ? Int(a[1]) : 64
        guard (2...maxPatternSide).contains(w), (2...maxPatternSide).contains(h) else { return settings }
        let key = "\(name)-\(w)x\(h)"
        var s = settings
        if let cached = patternCache[key] {
            s["source"] = cached
        } else if let image = LinearImage.pattern(name, width: w, height: h, supersample: 4) {
            let json = image.sourceJSON
            patternCache[key] = json
            s["source"] = json
        }
        return s
    }

    /// The largest pattern drawn by the app (the engine accepts buffers up to 8192).
    nonisolated static let maxPatternSide = 4096

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
