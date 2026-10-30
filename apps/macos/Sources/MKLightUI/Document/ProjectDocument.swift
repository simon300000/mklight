import Foundation
import MKLightModel
import Synchronization
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    /// `.mklight` project files (JSON). Declared as an exported type in the
    /// app's Info.plist (see scripts/build-macos.sh).
    public static let mklightProject = UTType(exportedAs: "com.mklight.project", conformingTo: .json)
}

/// The document model: the optical system value plus undo. It holds no view
/// state (selection, panes, camera and analysis settings live in
/// `WindowViewState`), and it never computes optics.
///
/// Every edit replaces `system` with a new value and registers the previous
/// value with the undo manager, so undo/redo restores whole value snapshots.
///
/// Concurrency: `system` is only mutated by the main-actor `edit` method (and
/// by the initialisers); the document machinery's possibly off-main reads go
/// through the lock-protected `persisted` copy. That is why the class is
/// `@unchecked Sendable` rather than main-actor isolated: SwiftUI may create
/// documents from a file off the main actor.
public final class ProjectDocument: ReferenceFileDocument, @unchecked Sendable {
    public typealias Snapshot = ProjectFile

    public static var readableContentTypes: [UTType] { [.mklightProject, .json] }
    public static var writableContentTypes: [UTType] { [.mklightProject] }

    @Published public private(set) var system: OpticalSystem

    /// Copy of `system` readable from any thread, for `snapshot(contentType:)`
    /// which the document machinery may call off the main actor.
    private let persisted: Mutex<OpticalSystem>

    public init(system: OpticalSystem = .newEmpty()) {
        self.system = system
        persisted = Mutex(system)
    }

    public init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let file = try ProjectFile.decode(data)
        _system = Published(initialValue: file.system)
        persisted = Mutex(file.system)
    }

    public func snapshot(contentType: UTType) throws -> ProjectFile {
        ProjectFile(system: persisted.withLock { $0 })
    }

    public func fileWrapper(snapshot: ProjectFile, configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: try snapshot.encoded())
    }

    /// Applies `change` to a copy of the system. If the result differs, it
    /// becomes the current value and the change is undoable as `actionName`.
    /// Configuration overrides and mechanical parts that the change
    /// invalidates (a conic override on a surface made plane, a
    /// semi-diameter override on a surface switched to automatic, …) are
    /// removed as part of the same undoable change. Returns whether anything
    /// changed.
    @MainActor @discardableResult
    public func edit(_ actionName: String, undoManager: UndoManager?, _ change: (inout OpticalSystem) -> Void) -> Bool {
        var next = system
        change(&next)
        next.dropReferencesInvalidated(since: system)
        guard next != system else { return false }
        replace(with: next, actionName: actionName, undoManager: undoManager)
        return true
    }

    @MainActor private func replace(with value: OpticalSystem, actionName: String, undoManager: UndoManager?) {
        let previous = system
        system = value
        persisted.withLock { $0 = value }
        undoManager?.registerUndo(withTarget: self) { doc in
            doc.replace(with: previous, actionName: actionName, undoManager: undoManager)
        }
        undoManager?.setActionName(actionName)
    }
}
