import Foundation
import MKLightEngine
import MKLightModel

/// What importing a lens prescription file (`.zmx`, `.seq`) left out or
/// changed, grouped by what it means for the design. The window shows it in
/// `ImportReportSheet` once, when the imported design opens.
public struct ImportReport: Identifiable, Sendable, Hashable {
    /// What a group of warnings is about, most serious first.
    public enum Category: Int, CaseIterable, Sendable, Comparable {
        case problems, glasses, surfaces, dropped, assumed

        public static func < (a: Category, b: Category) -> Bool { a.rawValue < b.rawValue }

        init(_ kind: ImportWarning.Kind) {
            switch kind {
            case .malformed, .validation, .other: self = .problems
            case .unknownGlass: self = .glasses
            case .unsupportedSurface, .nonSequential: self = .surfaces
            case .unsupportedRecord, .solveDropped: self = .dropped
            case .approximated, .defaulted: self = .assumed
            }
        }

        public var title: String {
            switch self {
            case .problems: "Problems"
            case .glasses: "Glasses"
            case .surfaces: "Surfaces"
            case .dropped: "Not Imported"
            case .assumed: "Assumed or Approximated"
            }
        }

        public var symbol: String {
            switch self {
            case .problems: "exclamationmark.triangle"
            case .glasses: "drop"
            case .surfaces: "camera.aperture"
            case .dropped: "minus.circle"
            case .assumed: "slider.horizontal.3"
            }
        }
    }

    /// A warning with its position in the report, unique across groups (a
    /// list needs row identities that do not repeat between its sections).
    public struct Item: Identifiable, Sendable, Hashable {
        public var id: Int
        public var warning: ImportWarning
    }

    public struct Group: Identifiable, Sendable, Hashable {
        public var category: Category
        public var items: [Item]
        public var id: Category { category }
        public var warnings: [ImportWarning] { items.map(\.warning) }
    }

    public let id: UUID
    public var fileName: String
    public var format: PrescriptionFormat
    public var encoding: String
    /// Surfaces between object and image.
    public var lensSurfaces: Int
    public var warnings: [ImportWarning]

    public init(_ imported: PrescriptionImport, fileName: String) {
        id = UUID()
        self.fileName = fileName
        format = imported.format
        encoding = imported.encoding
        lensSurfaces = max(imported.system.surfaces.count - 2, 0)
        warnings = imported.warnings
    }

    /// The warnings by category, in order of seriousness; each group keeps
    /// the file's order.
    public var groups: [Group] {
        Dictionary(grouping: warnings.enumerated().map { Item(id: $0.offset, warning: $0.element) }, by: { Category($0.warning.kind) })
            .map { Group(category: $0.key, items: $0.value.sorted { $0.id < $1.id }) }
            .sorted { $0.category < $1.category }
    }

    /// Glasses that are in none of the loaded catalogs: loading the right
    /// catalog and importing again resolves them.
    public var hasUnknownGlasses: Bool { warnings.contains { $0.kind == .unknownGlass } }

    /// "6 surfaces · .zmx · UTF-16"
    public var summary: String {
        let surfaces = lensSurfaces == 1 ? "1 surface" : "\(lensSurfaces) surfaces"
        return [surfaces, "." + format.fileExtension, Self.encodingName(encoding)].joined(separator: " · ")
    }

    /// The sheet's explanation under its title.
    public var headline: String {
        let n = warnings.count
        return n == 1
            ? "1 item could not be imported exactly. Review it before relying on the design."
            : "\(n) items could not be imported exactly. Review them before relying on the design."
    }

    static func encodingName(_ e: String) -> String {
        switch e {
        case "utf8", "utf8_bom": "UTF-8"
        case "utf16_le", "utf16_be": "UTF-16"
        case "latin1": "Latin-1"
        default: e
        }
    }

    /// The engine's message as a sentence (it starts in lower case).
    public static func sentence(_ w: ImportWarning) -> String {
        guard let first = w.message.first else { return w.message }
        return first.uppercased() + w.message.dropFirst()
    }

    /// "Surface 3 · Line 42", or nil for warnings about the whole file.
    public static func location(_ w: ImportWarning) -> String? {
        let parts = [w.surface.map { "Surface \($0)" }, w.line.map { "Line \($0)" }].compactMap { $0 }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// The report as plain text, for the pasteboard.
    public var plainText: String {
        var lines = ["Imported \(fileName) (\(summary))"]
        for g in groups {
            lines.append("")
            lines.append("\(g.category.title):")
            for w in g.warnings {
                lines.append("- " + Self.sentence(w) + (Self.location(w).map { " [\($0)]" } ?? ""))
            }
        }
        return lines.joined(separator: "\n")
    }
}
