import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// The app-wide glass library: the catalog files and folders the user
/// registers once, the user's own glass catalogs, and the availability
/// overrides (preferred, standard, obsolete, excluded from substitution).
///
/// It lives in Application Support (`MKLight/Glass Library.json` and the
/// `MKLight/User Glasses` folder), never in projects. Every design's engine
/// session loads `catalogFiles` and applies `overrides` (`Workspace.syncLibrary`);
/// a design records in its project only the library catalogs it uses, when
/// it is saved, so the project opens where the library differs (docs/APP.md,
/// "Glass catalogs").
///
/// User catalogs are the engine's `GlassCatalog` JSON, one file per catalog.
/// Their glasses come from the engine (`EngineClient.glass(from:)`), which
/// computes n_d and V_d; this type only stores them. Vendor catalog files
/// are never written.
///
/// Platform-neutral: Foundation and Observation only.
@MainActor
@Observable
public final class GlassLibrary {
    /// A registered catalog file, or a folder whose `.agf` and `.json` files
    /// are all loaded.
    public struct Entry: Codable, Hashable, Identifiable, Sendable {
        public var path: String
        public var isFolder: Bool
        public var id: String { path }
        public var url: URL { URL(fileURLWithPath: path) }
        public var title: String { url.lastPathComponent }

        public init(path: String, isFolder: Bool) {
            self.path = path
            self.isFolder = isFolder
        }

        private enum CodingKeys: String, CodingKey { case path, isFolder = "folder" }
    }

    /// One of the user's own catalogs (`User Glasses/<name>.json`).
    public struct UserCatalog: Hashable, Identifiable, Sendable {
        public var name: String
        public var url: URL
        /// The glasses in the engine's JSON form, by name.
        public var glasses: [String: JSONValue]
        public var id: String { name }

        /// The catalog in the engine's `GlassCatalog` JSON form.
        public var json: JSONValue { .object(["name": .string(name), "glasses": .object(glasses)]) }
    }

    public enum LibraryError: LocalizedError, Equatable {
        case invalidName(String)
        case duplicateName(String)
        case unknownCatalog(String)
        case unknownGlass(String)

        public var errorDescription: String? {
            switch self {
            case .invalidName(let why): why
            case .duplicateName(let n): "There is already a catalog or glass named “\(n)”."
            case .unknownCatalog(let n): "There is no user catalog named “\(n)”."
            case .unknownGlass(let n): "There is no glass named “\(n)”."
            }
        }
    }

    /// The library every window uses.
    public static let shared = GlassLibrary(directory: defaultDirectory)

    /// `~/Library/Application Support/MKLight` (debug builds:
    /// `-MKLightSupportDirectory <path>` for UI checks with a scratch library).
    public static var defaultDirectory: URL {
        #if DEBUG
        if let p = UserDefaults.standard.string(forKey: "MKLightSupportDirectory") { return URL(fileURLWithPath: p, isDirectory: true) }
        #endif
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("MKLight", isDirectory: true)
    }

    public let directory: URL
    public private(set) var entries: [Entry] = []
    public private(set) var overrides: [GlassOverride] = []
    public private(set) var userCatalogs: [UserCatalog] = []
    /// Counts every change, so open designs load the library again.
    public private(set) var revision = 0
    /// Files of the user-catalog folder that could not be read.
    public private(set) var problems: [String] = []

    public var settingsURL: URL { directory.appendingPathComponent("Glass Library.json") }
    public var userGlassesDirectory: URL { directory.appendingPathComponent("User Glasses", isDirectory: true) }

    public init(directory: URL) {
        self.directory = directory
        load()
    }

    // MARK: Files

    /// Every catalog file the designs load, in order: registered files,
    /// the `.agf` and `.json` files of registered folders (by name), then
    /// the user catalogs.
    public var catalogFiles: [URL] {
        var out: [URL] = []
        for e in entries {
            if e.isFolder {
                let files = (try? FileManager.default.contentsOfDirectory(at: e.url, includingPropertiesForKeys: nil)) ?? []
                out += files.filter { Self.isCatalogFile($0) }.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            } else {
                out.append(e.url)
            }
        }
        return out + userCatalogs.map(\.url)
    }

    /// Whether the app reads `url` as a glass catalog (`.agf` or `.json`).
    public static func isCatalogFile(_ url: URL) -> Bool {
        ["agf", "json"].contains(url.pathExtension.lowercased())
    }

    /// Whether `name` is one of the user's catalogs.
    public func isUserCatalog(_ name: String) -> Bool { userCatalog(name) != nil }

    public func userCatalog(_ name: String) -> UserCatalog? {
        userCatalogs.first { $0.name.caseInsensitiveCompare(name) == .orderedSame }
    }

    /// The entry a catalog loaded from `path` belongs to (the file itself or its folder).
    public func entry(containing path: String) -> Entry? {
        let p = URL(fileURLWithPath: path).standardizedFileURL.path
        return entries.first { e in
            let ep = e.url.standardizedFileURL.path
            return e.isFolder ? URL(fileURLWithPath: p).deletingLastPathComponent().path == ep : p == ep
        }
    }

    // MARK: Library entries

    /// Registers a catalog file or a folder of catalogs. Returns whether it
    /// was new.
    @discardableResult
    public func add(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
        let e = Entry(path: url.standardizedFileURL.path, isFolder: exists && isDir.boolValue)
        guard !entries.contains(where: { $0.path == e.path }) else { return false }
        entries.append(e)
        changed()
        return true
    }

    public func remove(_ entry: Entry) {
        entries.removeAll { $0.path == entry.path }
        changed()
    }

    /// Tells the open designs to load the library again (a catalog file
    /// changed on disk and was reloaded).
    public func touch() { revision += 1 }

    // MARK: Availability overrides

    public func override(catalog: String, name: String) -> GlassOverride? {
        overrides.first { $0.matches(catalog: catalog, name: name) }
    }

    /// Sets a glass's status; `nil` returns it to the catalog's.
    public func setStatus(_ status: GlassStatus?, catalog: String, name: String) {
        updateOverride(catalog: catalog, name: name) { $0.status = status }
    }

    /// Excludes a glass from glass substitution or allows it; `nil` returns
    /// it to the catalog's setting.
    public func setExcluded(_ excluded: Bool?, catalog: String, name: String) {
        updateOverride(catalog: catalog, name: name) { $0.excludeSubstitution = excluded }
    }

    /// Returns a glass to the catalog's status and exclusion.
    public func clearOverride(catalog: String, name: String) {
        updateOverride(catalog: catalog, name: name) { $0 = GlassOverride(catalog: $0.catalog, name: $0.name) }
    }

    private func updateOverride(catalog: String, name: String, _ change: (inout GlassOverride) -> Void) {
        var o = override(catalog: catalog, name: name) ?? GlassOverride(catalog: catalog, name: name)
        change(&o)
        let before = overrides
        overrides.removeAll { $0.matches(catalog: catalog, name: name) }
        if !o.isEmpty { overrides.append(o) }
        guard overrides != before else { return }
        changed()
    }

    // MARK: User catalogs

    /// Why `name` cannot name a new catalog or glass, if it cannot.
    public static func nameProblem(_ name: String, glass: Bool = false) -> String? {
        let n = name.trimmingCharacters(in: .whitespaces)
        if n.isEmpty { return "Enter a name." }
        if glass, n.contains(where: \.isWhitespace) { return "Glass names cannot contain spaces (AGF names are single words)." }
        if n.contains(where: { "/:\\".contains($0) }) { return "Names cannot contain “/”, “:” or “\\”." }
        if !glass, n.caseInsensitiveCompare("BUILTIN") == .orderedSame { return "“BUILTIN” is the built-in catalog's name." }
        return nil
    }

    /// A new, empty user catalog.
    @discardableResult
    public func createUserCatalog(named raw: String) throws -> UserCatalog {
        let name = raw.trimmingCharacters(in: .whitespaces)
        if let why = Self.nameProblem(name) { throw LibraryError.invalidName(why) }
        guard userCatalog(name) == nil else { throw LibraryError.duplicateName(name) }
        let c = UserCatalog(name: name, url: userGlassesDirectory.appendingPathComponent(name + ".json"), glasses: [:])
        try write(c)
        userCatalogs.append(c)
        userCatalogs.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        changed()
        return c
    }

    /// Renames a user catalog (its file too). Designs that name the catalog
    /// in a glass reference (`CATALOG:GLASS`) keep the old name.
    public func renameUserCatalog(_ old: String, to raw: String) throws {
        let name = raw.trimmingCharacters(in: .whitespaces)
        guard var c = userCatalog(old) else { throw LibraryError.unknownCatalog(old) }
        guard name != c.name else { return }
        if let why = Self.nameProblem(name) { throw LibraryError.invalidName(why) }
        if let other = userCatalog(name), other.url != c.url { throw LibraryError.duplicateName(name) }
        let oldURL = c.url
        c.name = name
        c.url = userGlassesDirectory.appendingPathComponent(name + ".json")
        try write(c)
        if oldURL != c.url { try? FileManager.default.removeItem(at: oldURL) }
        userCatalogs.removeAll { $0.url == oldURL }
        userCatalogs.append(c)
        userCatalogs.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        changed()
    }

    /// Moves a user catalog's file to the Trash.
    public func deleteUserCatalog(_ name: String) throws {
        guard let c = userCatalog(name) else { throw LibraryError.unknownCatalog(name) }
        do {
            try FileManager.default.trashItem(at: c.url, resultingItemURL: nil)
        } catch {
            try FileManager.default.removeItem(at: c.url)
        }
        userCatalogs.removeAll { $0.url == c.url }
        changed()
    }

    /// Adds or replaces a glass (engine JSON with a `name`) in a user
    /// catalog. `replacing` is the glass's previous name when it was
    /// renamed in the editor.
    public func putGlass(_ glass: JSONValue, in catalog: String, replacing old: String? = nil) throws {
        guard let name = glass["name"]?.string else { throw LibraryError.invalidName("The glass has no name.") }
        if let why = Self.nameProblem(name, glass: true) { throw LibraryError.invalidName(why) }
        guard var c = userCatalog(catalog) else { throw LibraryError.unknownCatalog(catalog) }
        let clash = c.glasses.keys.first { $0.caseInsensitiveCompare(name) == .orderedSame }
        if let clash, clash.caseInsensitiveCompare(old ?? "") != .orderedSame { throw LibraryError.duplicateName(name) }
        if let old { c.glasses[old] = nil }
        if let clash { c.glasses[clash] = nil }
        c.glasses[name] = glass
        try replace(c)
    }

    public func deleteGlass(_ name: String, from catalog: String) throws {
        guard var c = userCatalog(catalog) else { throw LibraryError.unknownCatalog(catalog) }
        guard c.glasses.removeValue(forKey: name) != nil else { throw LibraryError.unknownGlass(name) }
        try replace(c)
        overrides.removeAll { $0.matches(catalog: catalog, name: name) }
        persistSettings()
    }

    /// A glass name not yet in `catalog`: `base`, `base_2`, `base_3`, …
    public func freeGlassName(_ base: String, in catalog: String) -> String {
        let taken = Set(userCatalog(catalog)?.glasses.keys.map { $0.uppercased() } ?? [])
        let stem = base.replacingOccurrences(of: " ", with: "_")
        if !taken.contains(stem.uppercased()) { return stem }
        return (2...).lazy.map { "\(stem)_\($0)" }.first { !taken.contains($0.uppercased()) }!
    }

    private func replace(_ c: UserCatalog) throws {
        try write(c)
        if let i = userCatalogs.firstIndex(where: { $0.url == c.url }) { userCatalogs[i] = c }
        changed()
    }

    private func write(_ c: UserCatalog) throws {
        try FileManager.default.createDirectory(at: userGlassesDirectory, withIntermediateDirectories: true)
        try MKLightJSON.encode(c.json, pretty: true).write(to: c.url, options: .atomic)
    }

    // MARK: Persistence

    private struct Settings: Codable {
        var version = 1
        var catalogs: [Entry] = []
        var overrides: [GlassOverride] = []
    }

    private func changed() {
        persistSettings()
        revision += 1
    }

    private func persistSettings() {
        let s = Settings(catalogs: entries, overrides: overrides)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try MKLightJSON.encode(s, pretty: true).write(to: settingsURL, options: .atomic)
        } catch {
            problems.append("The glass library could not be saved (\(error.localizedDescription)).")
        }
    }

    /// Reads the settings and the user catalogs from disk.
    public func load() {
        problems = []
        if let data = try? Data(contentsOf: settingsURL) {
            if let s = try? MKLightJSON.decode(Settings.self, from: data) {
                entries = s.catalogs
                overrides = s.overrides
            } else {
                problems.append("\(settingsURL.lastPathComponent) could not be read.")
            }
        }
        let files = (try? FileManager.default.contentsOfDirectory(at: userGlassesDirectory, includingPropertiesForKeys: nil)) ?? []
        userCatalogs = files.filter { $0.pathExtension.lowercased() == "json" }.compactMap { url in
            guard let data = try? Data(contentsOf: url), let v = try? JSONValue.parse(data) else {
                problems.append("\(url.lastPathComponent) is not a glass catalog.")
                return nil
            }
            let name = v["name"]?.string.flatMap { $0.isEmpty ? nil : $0 } ?? url.deletingPathExtension().lastPathComponent
            return UserCatalog(name: name, url: url, glasses: v["glasses"]?.object ?? [:])
        }
        .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        revision += 1
    }
}
