import Foundation

/// A user glass catalog the project depends on (engine `CatalogRef`). Vendor
/// catalogs are referenced, never embedded.
public struct CatalogReference: Codable, Sendable, Hashable {
    /// Catalog name, as in `MaterialRef.catalog(catalog:name:)`.
    public var name: String
    /// Absolute path where the catalog was last found.
    public var path: String?
    /// Path relative to the project file's directory, tried first on open.
    public var relativePath: String?
    /// `agf` or `json`.
    public var format: String

    public init(name: String, path: String?, relativePath: String? = nil, format: String = "agf") {
        self.name = name
        self.path = path
        self.relativePath = relativePath
        self.format = format
    }

    /// File locations to try, in order (as the engine does): the relative
    /// path from the project's directory, then the absolute path.
    public func candidates(projectDirectory: URL?) -> [URL] {
        var out: [URL] = []
        if let dir = projectDirectory, let rel = relativePath { out.append(dir.appendingPathComponent(rel)) }
        if let path { out.append(URL(fileURLWithPath: path)) }
        return out
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: AnyKey.self)
        name = try c.value("name")
        path = try c.optional("path")
        relativePath = try c.optional("relative_path")
        format = try c.value("format", default: "agf")
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(name, "name")
        try c.putOptional(path, "path")
        try c.putOptional(relativePath, "relative_path")
        try c.put(format, "format")
    }
}

/// An on-disk `.mklight` project in the engine's format (docs/FFI.md,
/// "Project file"): `{"format", "schema_version", "metadata", "system",
/// "catalogs", "views", "merit", "variables", "tolerances", …}`.
///
/// The app edits only `system` (and the catalog list); every other section,
/// including keys a newer application added, is kept verbatim in `sections`
/// so a round trip loses nothing. The system itself is decoded strictly, as
/// the engine does: an unknown model field is an error, never dropped. Like
/// the engine, only `currentVersion` is read: a document without a
/// `schema_version` or at another version is refused.
public struct ProjectFile: Codable, Sendable, Hashable {
    public static let formatName = "mklight-project"
    /// The engine's `PROJECT_SCHEMA_VERSION` this app reads and writes.
    public static let currentVersion = 1

    public var format: String
    public var schemaVersion: Int
    public var system: OpticalSystem
    /// Every other top-level key, verbatim.
    public var sections: [String: JSONValue]

    /// A new project around `system`, with the engine's default sections.
    public init(system: OpticalSystem) {
        format = Self.formatName
        schemaVersion = Self.currentVersion
        self.system = system
        sections = [
            "metadata": .object([:]), "catalogs": .array([]), "views": .array([]), "merit": .null,
            "variables": .null, "tolerances": .null,
        ]
    }

    public enum Error: Swift.Error, LocalizedError, Equatable {
        case unsupportedVersion(Int)
        case notAProject(String)
        public var errorDescription: String? {
            switch self {
            case .unsupportedVersion(let v) where v > ProjectFile.currentVersion:
                "This project was written by a newer MKLight (schema version \(v)); please update MKLight."
            case .unsupportedVersion(let v):
                "This project uses schema version \(v), which this MKLight does not read (it reads version \(ProjectFile.currentVersion))."
            case .notAProject(let why): "Not an MKLight project: \(why)"
            }
        }
    }

    /// User glass catalogs referenced by the project.
    public var catalogs: [CatalogReference] {
        get {
            guard let v = sections["catalogs"], let data = try? MKLightJSON.encode(v) else { return [] }
            return (try? MKLightJSON.decode([CatalogReference].self, from: data)) ?? []
        }
        set { sections["catalogs"] = (try? JSONValue.from(newValue)) ?? .array([]) }
    }

    /// Adds or replaces (by name) a catalog reference.
    public mutating func addCatalog(_ ref: CatalogReference) {
        var list = catalogs.filter { $0.name != ref.name }
        list.append(ref)
        catalogs = list
    }

    /// Stamps `metadata` the way the engine does on save.
    public mutating func stampMetadata(appName: String, appVersion: String, engineVersion: String?, now: Date = Date()) {
        var meta: [String: JSONValue] = if case .object(let o) = sections["metadata"] { o } else { [:] }
        let stamp = ISO8601DateFormatter().string(from: now)
        if case .string(let s)? = meta["created"], !s.isEmpty {} else { meta["created"] = .string(stamp) }
        meta["modified"] = .string(stamp)
        meta["app_name"] = .string(appName)
        meta["app_version"] = .string(appVersion)
        if let engineVersion { meta["engine_version"] = .string(engineVersion) }
        if meta["author"] == nil { meta["author"] = .string("") }
        sections["metadata"] = .object(meta)
    }

    public static func decode(_ data: Data) throws -> ProjectFile {
        guard case .object(let doc) = try JSONValue.parse(data) else {
            throw Error.notAProject("the document is not a JSON object")
        }
        switch doc["schema_version"] {
        case nil, .null?:
            throw Error.notAProject("it has no schema_version")
        case .integer(let v)? where v >= 0:
            guard v == Int64(currentVersion) else { throw Error.unsupportedVersion(Int(clamping: v)) }
        case .unsigned?:
            throw Error.unsupportedVersion(Int.max)
        default:
            throw Error.notAProject("schema_version must be a non-negative integer")
        }
        guard let systemValue = doc["system"] else { throw Error.notAProject("it has no 'system'") }
        let format: String
        switch doc["format"] {
        case nil, .null?: format = formatName
        case .string(let f)?: format = f
        default: throw Error.notAProject("'format' must be a string")
        }
        guard format == formatName else { throw Error.notAProject("format '\(format)'") }
        let system = try MKLightJSON.decode(OpticalSystem.self, from: MKLightJSON.encode(systemValue))
        var file = ProjectFile(system: system)
        for (k, v) in doc where !["format", "schema_version", "system"].contains(k) {
            file.sections[k] = v
        }
        return file
    }

    public func encoded() throws -> Data {
        try MKLightJSON.encode(self, pretty: true)
    }

    public init(from decoder: any Decoder) throws {
        let doc = try JSONValue(from: decoder)
        self = try Self.decode(MKLightJSON.encode(doc))
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: AnyKey.self)
        try c.put(format, "format")
        try c.put(schemaVersion, "schema_version")
        try c.put(system, "system")
        for (k, v) in sections { try c.put(v, k) }
    }
}
