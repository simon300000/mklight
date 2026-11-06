import Foundation
import MKLightModel

/// Lens prescription formats the engine imports (docs/IMPORT.md).
public enum PrescriptionFormat: String, Codable, Sendable, CaseIterable {
    /// `.zmx` keyword records.
    case zmx
    /// `.seq` command scripts.
    case seq

    /// The format of a file extension (case-insensitive, without the dot).
    public init?(fileExtension: String) {
        self.init(rawValue: fileExtension.lowercased())
    }

    public var fileExtension: String { rawValue }
}

/// Something the importer dropped or changed: the engine's `ImportWarning`.
public struct ImportWarning: Codable, Sendable, Hashable {
    public enum Kind: String, Codable, Sendable, CaseIterable {
        case unsupportedRecord = "unsupported_record"
        case unsupportedSurface = "unsupported_surface"
        case unknownGlass = "unknown_glass"
        case solveDropped = "solve_dropped"
        case nonSequential = "non_sequential"
        case approximated
        case malformed
        case defaulted
        case validation
        /// A kind this app does not know yet (a newer engine).
        case other

        public init(from decoder: any Decoder) throws {
            self = Kind(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .other
        }
    }

    public var kind: Kind
    /// 1-based line in the file.
    public var line: Int?
    /// Index of the surface in the imported system.
    public var surface: Int?
    public var message: String

    public init(kind: Kind, line: Int? = nil, surface: Int? = nil, message: String) {
        self.kind = kind
        self.line = line
        self.surface = surface
        self.message = message
    }
}

/// An imported prescription: the system and everything that did not map.
public struct PrescriptionImport: Codable, Sendable {
    public var format: PrescriptionFormat
    /// Text encoding the file was read in (`utf8`, `utf16_le`, …).
    public var encoding: String
    public var system: OpticalSystem
    public var warnings: [ImportWarning]

    public init(format: PrescriptionFormat, encoding: String, system: OpticalSystem, warnings: [ImportWarning]) {
        self.format = format
        self.encoding = encoding
        self.system = system
        self.warnings = warnings
    }
}
