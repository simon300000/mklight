import Foundation
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    /// `.mklight` project files (JSON). Declared as an exported type in the
    /// app's Info.plist (see scripts/build-macos.sh).
    public static let mklightProject = UTType(exportedAs: "com.mklight.project", conformingTo: .json)
}

/// The file adapter of a design: reads and writes the engine's project
/// format and hands the content to `DesignDocument`, which holds the
/// simulation state. SwiftUI's document machinery provides autosave,
/// versions and the dirty state (changes register with the undo manager).
///
/// `ReferenceFileDocument` requires `ObservableObject`; the state that views
/// observe lives in the `@Observable` `DesignDocument`.
public final class ProjectDocument: ReferenceFileDocument, Sendable {
    public typealias Snapshot = ProjectFile

    public static var readableContentTypes: [UTType] { [.mklightProject, .json] }
    public static var writableContentTypes: [UTType] { [.mklightProject] }

    public let design: DesignDocument

    public init(system: OpticalSystem = .newEmpty()) {
        design = DesignDocument(system: system)
    }

    public init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        design = DesignDocument(project: try ProjectFile.decode(data))
    }

    public func snapshot(contentType: UTType) throws -> ProjectFile {
        var file = design.savedProject
        let app = Bundle.main.infoDictionary
        file.stampMetadata(
            appName: "MKLight", appVersion: app?["CFBundleShortVersionString"] as? String ?? "dev", engineVersion: nil)
        return file
    }

    public func fileWrapper(snapshot: ProjectFile, configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: try snapshot.encoded())
    }
}
