import Foundation
import MKLightEngine
import MKLightModel
import Synchronization
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    /// `.mklight` project files (JSON). Declared as an exported type in the
    /// app's Info.plist (see scripts/build-macos.sh).
    public static let mklightProject = UTType(exportedAs: "com.mklight.project", conformingTo: .json)
    /// `.zmx` lens prescription files, imported (never written). Declared as
    /// an imported type in the app's Info.plist.
    public static let zmxPrescription = UTType(importedAs: "com.mklight.zmx-prescription", conformingTo: .data)
    /// `.seq` lens prescription files (command scripts), imported.
    public static let seqPrescription = UTType(importedAs: "com.mklight.seq-prescription", conformingTo: .data)
    /// The lens prescription files MKLight imports.
    public static var lensPrescriptions: [UTType] { [.zmxPrescription, .seqPrescription] }
    /// Everything the app opens: projects, system JSON and lens files.
    public static var mklightReadable: [UTType] { [.mklightProject, .json] + lensPrescriptions }
}

extension PrescriptionFormat {
    /// The format of a document content type, falling back to the file name.
    init?(contentType: UTType, fileName: String?) {
        if contentType.conforms(to: .zmxPrescription) {
            self = .zmx
        } else if contentType.conforms(to: .seqPrescription) {
            self = .seq
        } else if let fileName, let f = PrescriptionFormat(fileExtension: (fileName as NSString).pathExtension) {
            self = f
        } else {
            return nil
        }
    }
}

/// The file adapter of a design: reads and writes the engine's project
/// format and hands the content to `DesignDocument`, which holds the
/// simulation state. SwiftUI's document machinery provides autosave,
/// versions and the dirty state (changes register with the undo manager).
///
/// Lens prescription files (`.zmx`, `.seq`) open as designs too: the engine
/// imports them, and the design keeps an `ImportReport` that its window
/// shows. They are never written; saving asks for a new `.mklight` file.
///
/// `ReferenceFileDocument` requires `ObservableObject`; the state that views
/// observe lives in the `@Observable` `DesignDocument`.
public final class ProjectDocument: ReferenceFileDocument, Sendable {
    public typealias Snapshot = ProjectFile

    public static var readableContentTypes: [UTType] { UTType.mklightReadable }
    public static var writableContentTypes: [UTType] { [.mklightProject] }

    /// The engine that imports lens prescription files (set by `MKLightScene`).
    private static let importEngine = Mutex<(any EngineClient)?>(nil)

    public static func setImportEngine(_ engine: any EngineClient) {
        importEngine.withLock { $0 = engine }
    }

    public let design: DesignDocument

    public init(system: OpticalSystem = .newEmpty()) {
        design = DesignDocument(system: system)
    }

    public init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let fileName = configuration.file.filename
        if let format = PrescriptionFormat(contentType: configuration.contentType, fileName: fileName) {
            design = try Self.importDesign(data, format: format, fileName: fileName ?? "Lens.\(format.fileExtension)")
        } else {
            design = DesignDocument(project: try ProjectFile.decode(data))
        }
    }

    /// A design from a lens prescription file, with its import report.
    static func importDesign(_ data: Data, format: PrescriptionFormat, fileName: String) throws -> DesignDocument {
        guard let engine = importEngine.withLock({ $0 }) else {
            throw EngineError.unsupported("No engine is available to import lens files")
        }
        let imported = try engine.importPrescription(data, format: format)
        var system = imported.system
        if system.name.isEmpty { system.name = (fileName as NSString).deletingPathExtension }
        return DesignDocument(project: ProjectFile(system: system), importReport: ImportReport(imported, fileName: fileName))
    }

    /// Whether the app opens `url` as a design (by its extension), for
    /// drag and drop.
    public static func canOpen(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return ext == "mklight" || ext == "json" || PrescriptionFormat(fileExtension: ext) != nil
    }

    public func snapshot(contentType: UTType) throws -> ProjectFile {
        var file = design.savedProject
        let app = Bundle.main.infoDictionary
        file.stampMetadata(
            appName: "MKLight", appVersion: app?["CFBundleShortVersionString"] as? String ?? "dev", engineVersion: nil)
        return file
    }

    public func fileWrapper(snapshot: ProjectFile, configuration: WriteConfiguration) throws -> FileWrapper {
        // Never write project JSON into a lens file.
        if UTType.lensPrescriptions.contains(where: { configuration.contentType.conforms(to: $0) }) {
            throw CocoaError(.fileWriteNoPermission)
        }
        return FileWrapper(regularFileWithContents: try snapshot.encoded())
    }
}
