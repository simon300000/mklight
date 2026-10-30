import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

/// The welcome window: start a new design, open one, or start from one of
/// the engine's sample designs.
struct WelcomeView: View {
    static let windowID = "welcome"
    let engine: any EngineClient
    @Environment(\.newDocument) private var newDocument
    @Environment(\.openDocument) private var openDocument
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var importing = false
    @State private var error: String?

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: Space.l) {
                Image(systemName: "camera.aperture")
                    .font(.system(size: 52, weight: .light))
                    .foregroundStyle(.tint)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: Space.xs) {
                    Text("MKLight").font(.largeTitle.weight(.semibold))
                    Text("Sequential optical design").font(.title3).foregroundStyle(.secondary)
                }
                Spacer().frame(height: Space.l)
                Button {
                    newDocument { ProjectDocument() }
                    dismissWindow(id: Self.windowID)
                } label: {
                    Label("New Design", systemImage: "doc.badge.plus").frame(maxWidth: .infinity, alignment: .leading)
                }
                Button {
                    importing = true
                } label: {
                    Label("Open…", systemImage: "folder").frame(maxWidth: .infinity, alignment: .leading)
                }
                Spacer()
                if let error { Text(error).font(.caption).foregroundStyle(.red) }
                Text(engineLine).font(.caption).foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .controlSize(.large)
            .padding(Space.xxl)
            .frame(width: 260)

            Divider()

            VStack(alignment: .leading, spacing: 0) {
                Text("Sample Designs").font(.headline).padding([.horizontal, .top], Space.xl).padding(.bottom, Space.m)
                List(SampleSystems.bundled) { e in
                    Button {
                        newDocument { ProjectDocument(system: e.system) }
                        dismissWindow(id: Self.windowID)
                    } label: {
                        SampleRow(entry: e)
                    }
                    .buttonStyle(.plain)
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
            .frame(width: 400)
            .background(.background.secondary)
        }
        .frame(height: 440)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.mklightProject, .json]) { result in
            switch result {
            case .success(let url):
                Task {
                    do {
                        try await openDocument(at: url)
                        dismissWindow(id: Self.windowID)
                    } catch {
                        self.error = error.localizedDescription
                    }
                }
            case .failure(let e): error = e.localizedDescription
            }
        }
    }

    private var engineLine: String {
        engine.info.isMock ? "Mock engine — synthetic data" : "\(engine.info.name) \(engine.info.version)"
    }
}

private struct SampleRow: View {
    let entry: SampleSystems.Entry

    var body: some View {
        let s = entry.system
        HStack(spacing: Space.l) {
            Image(systemName: symbol)
                .font(.title2)
                .foregroundStyle(.secondary)
                .frame(width: 32)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(s.name).font(.body.weight(.medium))
                Text(detail(s)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
        }
        .padding(.vertical, Space.xs)
        .contentShape(Rectangle())
    }

    private var symbol: String {
        switch entry.id {
        case "newtonian": "scope"
        case "relay": "arrow.left.and.right.circle"
        case "singlet": "circle"
        default: "camera.aperture"
        }
    }

    private func detail(_ s: OpticalSystem) -> String {
        let n = s.surfaces.count - 2
        let notes = s.notes.isEmpty ? "" : " · " + s.notes
        return "\(n) surfaces\(notes)"
    }
}
