import MKLightEngine
import MKLightModel
import Observation
import SwiftUI
import UniformTypeIdentifiers

/// The welcome window: start a new design, open one (a project or a `.zmx`
/// or `.seq` lens file, also by dropping it on the window), or start from
/// one of the engine's sample designs, each shown with a thumbnail of its
/// layout as the engine draws it.
struct WelcomeView: View {
    static let windowID = "welcome"
    let engine: any EngineClient
    @Environment(\.newDocument) private var newDocument
    @Environment(\.openDocument) private var openDocument
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var importing = false
    @State private var error: String?
    @State private var thumbnails = SampleThumbnails()
    @State private var dropTargeted = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var classic: [SampleSystems.Entry] { SampleSystems.bundled.filter { $0.system.surfaces.allSatisfy(\.shape.isLegacy) } }
    private var extended: [SampleSystems.Entry] { SampleSystems.bundled.filter { !$0.system.surfaces.allSatisfy(\.shape.isLegacy) } }

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            ScrollView {
                VStack(alignment: .leading, spacing: Space.xl) {
                    section("Sample Designs", classic)
                    if !extended.isEmpty { section("Surface Types", extended) }
                }
                .padding(Space.xl)
            }
            .scrollEdgeEffectStyle(.soft, for: .top)
            .background(.background)
        }
        .frame(width: 960, height: 620)
        .task { await thumbnails.load(SampleSystems.bundled, engine: engine.makeDocumentEngine()) }
        .fileImporter(isPresented: $importing, allowedContentTypes: UTType.mklightReadable) { result in
            switch result {
            case .success(let url): open([url])
            case .failure(let e): error = e.localizedDescription
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            let files = urls.filter(ProjectDocument.canOpen)
            guard !files.isEmpty else { return false }
            open(files)
            return true
        } isTargeted: { dropTargeted = $0 }
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(.tint, lineWidth: 3)
                    .padding(Space.s)
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .smooth(duration: 0.15), value: dropTargeted)
    }

    /// Opens projects and lens files, each in its own window, then closes
    /// the welcome window.
    private func open(_ urls: [URL]) {
        Task {
            var opened = false
            for url in urls {
                do {
                    try await openDocument(at: url)
                    opened = true
                } catch {
                    self.error = "\(url.lastPathComponent): \(error.localizedDescription)"
                }
            }
            if opened { dismissWindow(id: Self.windowID) }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            Spacer().frame(height: Space.xl)
            Image(systemName: "camera.aperture")
                .font(.system(size: 56, weight: .regular))
                .foregroundStyle(.tint)
                .symbolRenderingMode(.hierarchical)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: Space.xs) {
                Text("MKLight").font(.largeTitle.weight(.bold))
                Text("Sequential optical design").font(.title3).foregroundStyle(.secondary)
            }
            Spacer().frame(height: Space.m)
            GlassEffectContainer(spacing: Space.m) {
                VStack(alignment: .leading, spacing: Space.m) {
                    Button {
                        newDocument { ProjectDocument() }
                        dismissWindow(id: Self.windowID)
                    } label: {
                        Label("New Design", systemImage: "plus").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glassProminent)
                    .keyboardShortcut(.defaultAction)
                    Button {
                        importing = true
                    } label: {
                        Label("Open…", systemImage: "folder").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.glass)
                    .keyboardShortcut("o")
                    .help("Opens an MKLight project, or imports a .zmx or .seq lens file")
                }
                .controlSize(.extraLarge)
            }
            Spacer()
            Label("Drop a project or a .zmx or .seq lens file here to open it.", systemImage: "arrow.down.document")
                .font(.caption)
                .foregroundStyle(.secondary)
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            Label(engineLine, systemImage: engine.info.isMock ? "exclamationmark.triangle.fill" : "cpu")
                .font(.caption)
                .foregroundStyle(engine.info.isMock ? AnyShapeStyle(.orange) : AnyShapeStyle(.secondary))
        }
        .padding(Space.xxl)
        .frame(width: 280)
        .frame(maxHeight: .infinity)
        .background(.background.secondary)
    }

    private func section(_ title: String, _ entries: [SampleSystems.Entry]) -> some View {
        VStack(alignment: .leading, spacing: Space.m) {
            Text(title).font(.title3.weight(.semibold))
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 140, maximum: 220), spacing: Space.l)], spacing: Space.l) {
                ForEach(entries) { e in
                    Button {
                        newDocument { ProjectDocument(system: e.system) }
                        dismissWindow(id: Self.windowID)
                    } label: {
                        SampleCard(entry: e, thumbnail: thumbnails.items[e.id])
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var engineLine: String {
        engine.info.isMock ? "Mock engine — synthetic data" : "\(engine.info.name) \(engine.info.version)"
    }
}

/// A sample design: thumbnail, name and a line of facts.
private struct SampleCard: View {
    let entry: SampleSystems.Entry
    let thumbnail: SampleThumbnails.Item?
    @State private var hover = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let s = entry.system
        VStack(alignment: .leading, spacing: Space.s) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.fill.quinary)
                switch thumbnail {
                case .scene(let scene)?:
                    LayoutThumbnail(scene: scene).padding(Space.m)
                case .unavailable?:
                    Image(systemName: "eye.slash").font(.title2).foregroundStyle(.tertiary)
                        .accessibilityLabel("No layout")
                case nil:
                    ProgressView().controlSize(.small)
                }
            }
            .aspectRatio(16 / 9, contentMode: .fit)
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(hover ? AnyShapeStyle(.tint) : AnyShapeStyle(.separator), lineWidth: hover ? 2 : 0.5))
            VStack(alignment: .leading, spacing: 1) {
                Text(s.name).font(.body.weight(.semibold)).lineLimit(1)
                Text(detail(s)).font(.caption).foregroundStyle(.secondary).lineLimit(2, reservesSpace: true)
            }
            .padding(.horizontal, 2)
        }
        .contentShape(Rectangle())
        .scaleEffect(hover && !reduceMotion ? 1.02 : 1)
        .animation(reduceMotion ? nil : .smooth(duration: 0.2), value: hover)
        .onHover { hover = $0 }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(s.name). \(detail(s))")
        .accessibilityAddTraits(.isButton)
        .accessibilityHint("Creates a new design from this sample")
    }

    private func detail(_ s: OpticalSystem) -> String {
        let n = s.surfaces.count - 2
        let kinds = Set(s.surfaces.map(\.shape.kind)).filter { !$0.isLegacy }.map { SurfaceTypes.descriptor(for: $0).title }.sorted()
        let notes = !kinds.isEmpty ? kinds.joined(separator: ", ") : s.notes
        return "\(n) surfaces" + (notes.isEmpty ? "" : " · " + notes)
    }
}

/// The samples' layouts, computed once by the engine on its own session
/// (every surface type, extended shapes included).
@MainActor
@Observable
final class SampleThumbnails {
    enum Item {
        case scene(LayoutScene)
        case unavailable
    }
    var items: [String: Item] = [:]

    func load(_ entries: [SampleSystems.Entry], engine: any EngineClient) async {
        for e in entries where items[e.id] == nil {
            items[e.id] = await Self.scene(e.system, engine: engine).map(Item.scene) ?? .unavailable
        }
    }

    nonisolated static func scene(_ system: OpticalSystem, engine: any EngineClient) async -> LayoutScene? {
        guard let snapshot = try? await engine.openSnapshot(system) else { return nil }
        let job = engine.submit(EngineAnalysisRequest(analysis: "layout", settings: ["rays_per_field": .integer(5)]), on: snapshot)
        guard let output = try? await job.result() else { return nil }
        return EngineLayout.scene(from: output)
    }
}

/// The meridional section of an engine layout scene, fitted: glass fills,
/// surface profiles and edges, rays coloured by field.
struct LayoutThumbnail: View {
    let scene: LayoutScene

    /// The lens's profiles plus a margin along the axis, as frame points.
    static func lensWindow(_ lines: [LayoutScene.Polyline]) -> [Point3] {
        let lens = lines.filter { ![.ray, .axis, .imagePlane].contains($0.kind) }.flatMap(\.points)
        guard let z0 = lens.map(\.z).min(), let z1 = lens.map(\.z).max(), let h = lens.map({ abs($0.y) }).max() else {
            return lines.flatMap(\.points)
        }
        let m = max(z1 - z0, 4 * h)
        return [Point3(0, -1.6 * h, z0 - m), Point3(0, 1.6 * h, z1 + m)]
    }

    var body: some View {
        Canvas { ctx, size in
            let lines = scene.polylines.filter(\.inSection)
            let slim = SceneGeometry.isSlim(scene, mode: .section2D)
            // A slim scene: the lens with a margin of rays, not the long focus.
            let pts = slim ? Self.lensWindow(lines) : lines.flatMap(\.points)
            guard let zMin = pts.map(\.z).min(), let zMax = pts.map(\.z).max(),
                let yMin = pts.map(\.y).min(), let yMax = pts.map(\.y).max()
            else { return }
            let scale = min(size.width / max(zMax - zMin, 1e-9), size.height / max(yMax - yMin, 1e-9))
            let ox = (size.width - (zMax - zMin) * scale) / 2, oy = (size.height - (yMax - yMin) * scale) / 2
            func p(_ q: Point3) -> CGPoint { CGPoint(x: ox + (q.z - zMin) * scale, y: size.height - oy - (q.y - yMin) * scale) }
            for e in scene.elements where e.front.count > 1 {
                var path = Path()
                path.move(to: p(e.front[0]))
                for q in e.front.dropFirst() { path.addLine(to: p(q)) }
                for q in e.back.reversed() { path.addLine(to: p(q)) }
                path.closeSubpath()
                ctx.fill(path, with: .color(.accentColor.opacity(0.14)))
            }
            for l in lines where l.points.count > 1 {
                var path = Path()
                path.move(to: p(l.points[0]))
                for q in l.points.dropFirst() { path.addLine(to: p(q)) }
                if l.doe && l.kind == .surfaceProfile {
                    ctx.stroke(path, with: .color(Palette.slotColor(6).opacity(0.8)), style: StrokeStyle(lineWidth: 2.6, dash: [1.5, 1.5]))
                }
                switch l.kind {
                case .ray:
                    ctx.stroke(path, with: .color(Palette.fieldColor(l.field ?? 0).opacity(0.75)), lineWidth: 0.6)
                case .axis:
                    ctx.stroke(path, with: .color(.secondary.opacity(0.4)), style: StrokeStyle(lineWidth: 0.5, dash: [3, 3]))
                case .stop, .imagePlane:
                    ctx.stroke(path, with: .color(.secondary), lineWidth: 1)
                default:
                    ctx.stroke(path, with: .color(.primary.opacity(0.85)), lineWidth: 1)
                }
            }
        }
        .clipped()
        .accessibilityHidden(true)
    }
}
