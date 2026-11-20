import Foundation
import MKLightEngine
import MKLightModel
import Observation

/// The design a Glass Catalogs window shows and acts on: the frontmost
/// design window's session, or none.
@MainActor
struct GlassBrowserTarget {
    let id: ObjectIdentifier
    /// The design's name, for the sidebar.
    var title: () -> String
    let workspace: Workspace
    /// The selected surface whose following medium a glass can be assigned
    /// to: its index, or `nil` (no selection, or object / image).
    var glassSurface: () -> Int?
    /// Sets the medium after surface `index` to a catalog glass (undoable).
    var assign: (_ index: Int, _ catalog: String, _ name: String) -> Void
    /// Records a catalog file in the design's project and loads it.
    var addCatalog: (URL) async throws -> CatalogInfo
    /// Removes a catalog from the design's project (undoable).
    var removeCatalog: (String) -> Void
    /// The design's undo stack, so Edit ▸ Undo in the window undoes its
    /// changes to the design.
    var undo: () -> DocumentUndo? = { nil }
}

/// The Glass Catalogs window: catalogs in a sidebar (all, built-in, library,
/// user glasses, the design's own), a searchable, sortable glass table and
/// the selected glass's details, with the actions on them.
///
/// It reads everything from an engine session: the frontmost design's (so
/// it shows exactly the glasses that design resolves against, statuses
/// included), or one of its own with the library when no design is open.
@MainActor
@Observable
final class GlassBrowserModel {
    enum Scope: Hashable {
        case all
        case catalog(String)
    }

    /// Which glasses the table shows besides the search.
    enum StatusFilter: String, CaseIterable, Identifiable {
        case all, available, preferred, substitution
        var id: String { rawValue }
        var title: String {
            switch self {
            case .all: "All Glasses"
            case .available: "Hide Obsolete"
            case .preferred: "Preferred Only"
            case .substitution: "Usable in Glass Substitution"
            }
        }
    }

    static let shared = GlassBrowserModel(library: .shared)

    let library: GlassLibrary
    /// Creates the window's own session when no design is open.
    @ObservationIgnored var engine: (any EngineClient)?
    private(set) var target: GlassBrowserTarget?
    private var ownWorkspace: Workspace?

    private(set) var catalogs: [CatalogInfo] = []
    private(set) var rows: [GlassRow] = []
    var scope: Scope = .all {
        didSet { if scope != oldValue { Task { await loadRows() } } }
    }
    var search = ""
    var filter = StatusFilter.all
    var sortOrder: [KeyPathComparator<GlassRow>] = [KeyPathComparator(\.name)]
    var selection: GlassRow.ID? {
        didSet { if selection != oldValue { Task { await loadDetails() } } }
    }
    private(set) var details: GlassDetails?
    /// A short note about the last action (shown in the window's footer).
    var message: String? {
        didSet { if message != removalMessage { removal = nil } }
    }
    /// The library entry the last Remove from Library took out, while its
    /// note is up (the footer offers Undo).
    private(set) var removal: (entry: GlassLibrary.Entry, index: Int)?
    private var removalMessage: String?
    /// The user glass whose deletion waits for the user's confirmation.
    var pendingDeletion: GlassRow?
    /// What the last reload changed, while its sheet is up.
    var reloadReport: CatalogChanges?
    /// The user glass being edited, while its sheet is up.
    var editor: GlassEditorModel?
    private var rowsRequest = 0

    init(library: GlassLibrary) {
        self.library = library
    }

    // MARK: Design

    /// The workspace whose session the window reads.
    var workspace: Workspace? {
        if let target { return target.workspace }
        if ownWorkspace == nil, let engine {
            let ws = Workspace(engine: engine)
            ws.glassLibrary = library
            ownWorkspace = ws
        }
        return ownWorkspace
    }

    /// A design window became key: the window follows it.
    func attach(_ t: GlassBrowserTarget) {
        guard target?.id != t.id else { return }
        target = t
        Task { await refresh() }
    }

    /// A design window closed.
    func detach(_ id: ObjectIdentifier) {
        guard target?.id == id else { return }
        target = nil
        Task { await refresh() }
    }

    var designTitle: String? { target?.title() }

    // MARK: Catalogs

    var builtinCatalogs: [CatalogInfo] { catalogs.filter(\.isBuiltin) }
    /// Library catalogs other than the user's own.
    var libraryCatalogs: [CatalogInfo] { catalogs.filter { $0.library && !library.isUserCatalog($0.name) } }
    var userCatalogs: [CatalogInfo] { catalogs.filter { $0.library && library.isUserCatalog($0.name) } }
    /// Catalogs only the design's project brings.
    var designCatalogs: [CatalogInfo] { catalogs.filter { $0.inProject && !$0.library } }

    func catalog(_ name: String) -> CatalogInfo? { catalogs.first { $0.name == name } }

    var scopeCatalog: String? {
        if case .catalog(let n) = scope { return n }
        return nil
    }

    /// Reads the catalogs and the glasses again (after the library or the
    /// design's catalogs changed).
    func refresh() async {
        guard let ws = workspace else { return }
        await ws.syncLibrary()
        catalogs = ws.catalogs
        if catalogs.isEmpty { catalogs = await ws.documentEngine.catalogs() }
        if case .catalog(let n) = scope, catalog(n) == nil { scope = .all }
        await loadRows()
    }

    func loadRows() async {
        guard let ws = workspace else { return }
        rowsRequest += 1
        let request = rowsRequest
        let rows = (try? await ws.glassRows(in: scopeCatalog)) ?? []
        guard request == rowsRequest else { return }
        self.rows = rows
        if let s = selection, !rows.contains(where: { $0.id == s }) { selection = nil }
        await loadDetails()
    }

    /// The rows the table shows: filtered by the search (name or catalog,
    /// case-insensitive) and the status filter, in the sort order.
    var visibleRows: [GlassRow] {
        let q = search.trimmingCharacters(in: .whitespaces)
        return rows.filter { r in
            let matches = q.isEmpty || r.name.localizedCaseInsensitiveContains(q) || (scope == .all && r.catalog.localizedCaseInsensitiveContains(q))
            guard matches else { return false }
            switch filter {
            case .all: return true
            case .available: return r.status != .obsolete
            case .preferred: return r.status == .preferred
            case .substitution: return !r.excludeSubstitution && r.status != .obsolete && r.status != .special
            }
        }
        .sorted(using: sortOrder)
    }

    var selectedRow: GlassRow? { rows.first { $0.id == selection } }

    private func loadDetails() async {
        guard let row = selectedRow, let ws = workspace else {
            details = nil
            return
        }
        let d = try? await ws.glassDetails(catalog: row.catalog, name: row.name)
        if selection == row.id { details = d }
    }

    // MARK: Using a glass

    /// What "Use for Surface" would do: the surface index, or why not.
    var assignment: (index: Int?, why: String?) {
        guard let t = target else { return (nil, "Open a design to use a glass in it.") }
        guard let i = t.glassSurface() else { return (nil, "Select a surface in the lens data to use a glass after it.") }
        return (i, nil)
    }

    func useSelected() {
        guard let row = selectedRow, let t = target, let i = assignment.index else { return }
        t.assign(i, row.catalog, row.name)
        message = "Surface \(i) is followed by \(row.name) (\(row.catalog))."
    }

    // MARK: Status

    func setStatus(_ status: GlassStatus?, for row: GlassRow) {
        library.setStatus(status, catalog: row.catalog, name: row.name)
    }

    func setExcluded(_ excluded: Bool?, for row: GlassRow) {
        library.setExcluded(excluded, catalog: row.catalog, name: row.name)
    }

    func resetStatus(of row: GlassRow) {
        library.clearOverride(catalog: row.catalog, name: row.name)
    }

    // MARK: Catalog actions

    func addToLibrary(_ urls: [URL]) {
        let added = urls.filter { library.add($0) }
        message = added.isEmpty ? "Already in the library." : "Added \(added.map(\.lastPathComponent).joined(separator: ", ")) to the library."
    }

    func removeFromLibrary(_ info: CatalogInfo) {
        guard let path = info.path, let e = library.entry(containing: path) else {
            message = "\(info.name) is not a registered library file."
            return
        }
        let index = library.entries.firstIndex(of: e) ?? library.entries.count
        library.remove(e)
        let note = e.isFolder ? "Removed the folder \(e.title) from the library." : "Removed \(info.name) from the library."
        removalMessage = note
        message = note
        removal = (e, index)
    }

    /// Puts the entry the last Remove from Library took out back.
    func undoRemoval() {
        guard let r = removal else { return }
        library.restore(r.entry, at: r.index)
        removalMessage = nil
        message = "Restored \(r.entry.title) to the library."
    }

    func addToDesign(_ url: URL) async {
        guard let t = target else { return }
        do {
            let info = try await t.addCatalog(url)
            message = "Added \(info.name) (\(glassCountText(info.glassCount))) to \(t.title())." + warningsNote(info.warnings)
        } catch {
            message = error.localizedDescription
        }
    }

    func removeFromDesign(_ info: CatalogInfo) {
        guard let t = target else { return }
        t.removeCatalog(info.name)
        message = "Removed \(info.name) from \(t.title()). Undo in the design window to restore it."
    }

    func reload(_ info: CatalogInfo) async {
        guard let ws = workspace else { return }
        do {
            let changes = try await ws.reloadCatalog(named: info.name)
            reloadReport = changes
            message = "\(info.name) reloaded: \(changes.summary.lowercased())." + warningsNote(changes.warnings)
            await refresh()
        } catch {
            message = "\(info.name) could not be reloaded: \(error.localizedDescription)"
        }
    }

    // MARK: User glasses

    /// The user catalog new glasses go to: the selected one, else the first.
    var defaultUserCatalog: String? {
        if let n = scopeCatalog, library.isUserCatalog(n) { return n }
        return library.userCatalogs.first?.name
    }

    func newUserCatalog() -> String? {
        var name = "User Glasses"
        var k = 2
        while library.isUserCatalog(name) {
            name = "User Glasses \(k)"
            k += 1
        }
        do {
            try library.createUserCatalog(named: name)
            scope = .catalog(name)
            return name
        } catch {
            message = error.localizedDescription
            return nil
        }
    }

    /// Opens the editor for a new glass (a user catalog is made if there is none).
    func newGlass() {
        guard let cat = defaultUserCatalog ?? newUserCatalog() else { return }
        editor = GlassEditorModel(newIn: cat, library: library)
    }

    /// Opens the editor on a copy of the selected glass in a user catalog.
    func duplicateSelected() {
        guard let d = details, let cat = defaultUserCatalog ?? newUserCatalog() else { return }
        editor = GlassEditorModel(duplicating: d, into: cat, library: library)
    }

    /// Opens the editor on a copy of `row` in a user catalog.
    func duplicate(_ row: GlassRow) async {
        guard let ws = workspace, let d = try? await ws.glassDetails(catalog: row.catalog, name: row.name),
            let cat = defaultUserCatalog ?? newUserCatalog()
        else { return }
        editor = GlassEditorModel(duplicating: d, into: cat, library: library)
    }

    func editSelected() {
        guard let d = details, library.isUserCatalog(d.row.catalog) else { return }
        editor = GlassEditorModel(editing: d, library: library)
    }

    /// Deletes a user glass (after the window confirmed it).
    func delete(_ row: GlassRow) {
        guard library.isUserCatalog(row.catalog) else { return }
        do {
            try library.deleteGlass(row.name, from: row.catalog)
            if selection == row.id { selection = nil }
            message = "Deleted \(row.name)."
        } catch {
            message = error.localizedDescription
        }
    }

    /// Saves the editor's glass through the engine. Returns whether it saved.
    func saveEditor() -> Bool {
        guard let e = editor, let engine = workspace?.documentEngine else { return false }
        guard e.save(engine: engine) else { return false }
        message = "Saved \(e.name) in \(e.catalog)."
        // Show the saved glass, whatever was searched for.
        search = ""
        filter = .all
        if scope != .all { scope = .catalog(e.catalog) }
        selection = e.catalog + ":" + e.name.trimmingCharacters(in: .whitespaces)
        editor = nil
        return true
    }

    /// A user catalog as AGF text.
    func agfExport(of catalog: String) -> AGFExport? {
        guard let c = library.userCatalog(catalog), let engine = workspace?.documentEngine ?? self.engine else { return nil }
        do {
            return try engine.agf(of: c.json)
        } catch {
            message = error.localizedDescription
            return nil
        }
    }
}

/// The editor of one user glass: a model glass (n_d, V_d, ΔP_g,F) or an AGF
/// dispersion formula with its coefficients, plus range, physical data and
/// status. The engine makes the glass (`EngineClient.glass(from:)`) and
/// computes its n_d and V_d.
@MainActor
@Observable
final class GlassEditorModel {
    enum Kind: String, CaseIterable, Identifiable {
        case model, formula
        var id: String { rawValue }
        var title: String { self == .model ? "Model Glass" : "Dispersion Formula" }
    }

    let library: GlassLibrary
    var catalog: String
    var name: String
    /// The glass's name before editing (`nil` for a new glass).
    let originalName: String?
    var kind: Kind
    var nd = 1.5168
    var vd = 64.17
    var dpgf = 0.0
    var formula = 2 {
        didSet { coefficients = Self.resized(coefficients, to: Self.coefficientNames(formula).count) }
    }
    var coefficients: [Double] = Array(repeating: 0, count: 6)
    var minUm: Double?
    var maxUm: Double?
    var density: Double?
    var tce: Double?
    var relativeCost: Double?
    var status = GlassStatus.standard
    var excludeSubstitution = false
    var comment = ""
    /// The glass edited or duplicated: its other data (thermal, transmittance) are kept.
    let base: JSONValue?
    /// The engine's answer for the current values: the glass, or why not.
    private(set) var preview: JSONValue?
    private(set) var problem: String?

    var title: String { originalName == nil ? "New Glass" : "Edit \(originalName!)" }

    init(newIn catalog: String, library: GlassLibrary) {
        self.library = library
        self.catalog = catalog
        name = library.freeGlassName("NEWGLASS", in: catalog)
        originalName = nil
        kind = .model
        base = nil
    }

    /// A copy of a catalog glass in a user catalog.
    init(duplicating d: GlassDetails, into catalog: String, library: GlassLibrary) {
        self.library = library
        self.catalog = catalog
        name = library.freeGlassName(d.row.name, in: catalog)
        originalName = nil
        kind = .formula
        base = Self.withoutOverrides(d)
        load(d)
    }

    /// An existing user glass.
    init(editing d: GlassDetails, library: GlassLibrary) {
        self.library = library
        catalog = d.row.catalog
        name = d.row.name
        originalName = d.row.name
        kind = .formula
        base = d.glass
        load(d)
    }

    /// The glass as the catalog has it (the user's status overrides are
    /// settings, not the glass's data).
    private static func withoutOverrides(_ d: GlassDetails) -> JSONValue { d.glass }

    private func load(_ d: GlassDetails) {
        let c = Self.clean
        if let m = d.model {
            kind = .model
            nd = c(m.nd)
            vd = c(m.vd)
            dpgf = c(m.dpgf)
        } else if let f = d.formula {
            kind = .formula
            formula = f
            coefficients = Self.resized(d.coefficients, to: Self.coefficientNames(f).count)
            nd = c(d.row.nd ?? nd)
            vd = c(d.row.vd ?? vd)
            dpgf = c(d.row.dpgf ?? 0)
        } else {
            // Tabulated or other data without a formula: a model glass at its n_d, V_d.
            kind = .model
            nd = c(d.row.nd ?? nd)
            vd = c(d.row.vd ?? vd)
            dpgf = c(d.row.dpgf ?? 0)
        }
        minUm = d.row.minUm
        maxUm = d.row.unrestrictedRange ? nil : d.row.maxUm
        // A model glass's default range stays the default (not a set range).
        if d.model != nil, let lo = minUm, let hi = maxUm,
            abs(lo - Self.modelRange.lowerBound) < 1e-9, abs(hi - Self.modelRange.upperBound) < 1e-9
        {
            minUm = nil
            maxUm = nil
        }
        density = d.row.density
        tce = d.row.tce
        relativeCost = d.row.relativeCost
        status = d.catalogStatus
        excludeSubstitution = d.catalogExcludeSubstitution
        comment = d.comment
    }

    /// The range a model glass has when none is set (µm).
    static let modelRange = 0.35...1.1

    /// A number read back from the engine, without the rounding noise of
    /// its computation (ΔP_g,F of 2·10⁻¹⁶ is 0; n_d keeps 10 decimals).
    static func clean(_ v: Double) -> Double {
        guard v.isFinite else { return v }
        if abs(v) < 1e-12 { return 0 }
        return (v * 1e10).rounded() / 1e10
    }

    /// An engine refusal as one sentence, without the error kind repeated
    /// ("Invalid model: invalid model: …").
    static func describe(_ error: any Error) -> String {
        func strip(_ m: String) -> String {
            var t = m.trimmingCharacters(in: .whitespaces)
            for p in ["invalid model:", "invalid glass:"] where t.lowercased().hasPrefix(p) {
                t = String(t.dropFirst(p.count)).trimmingCharacters(in: .whitespaces)
            }
            return t.prefix(1).uppercased() + t.dropFirst()
        }
        switch error as? EngineError {
        case .engine(_, let m)?: return strip(m)
        case .invalidModel(let issues)?: return issues.map(strip).joined(separator: "; ")
        default: return error.localizedDescription
        }
    }

    /// Whether saving renames an existing glass (designs that name it lose it).
    var renames: Bool {
        guard let o = originalName else { return false }
        return o.caseInsensitiveCompare(name.trimmingCharacters(in: .whitespaces)) != .orderedSame
    }

    /// The names of formula `f`'s `CD` coefficients, in order.
    static func coefficientNames(_ f: Int) -> [String] {
        func sellmeier(_ n: Int) -> [String] { (1...n).flatMap { ["K\($0)", "L\($0)"] } }
        switch f {
        case 1: return (0...5).map { "A\($0)" }
        case 2: return sellmeier(3)
        case 3: return ["A", "B", "C", "D", "E", "F"]
        case 4: return ["A", "B1", "λ1", "B2", "λ2"]
        case 5: return ["n0", "A", "B"]
        case 6: return sellmeier(4)
        case 7, 8: return ["A", "B", "C", "D"]
        case 9: return ["A", "B", "C", "D", "E"]
        case 10, 12: return (0...7).map { "a\($0)" }
        case 11: return sellmeier(5)
        case 13: return (0...8).map { "a\($0)" }
        default: return []
        }
    }

    static let formulaNames: [(Int, String)] = [
        (1, "Schott"), (2, "Sellmeier 1"), (3, "Herzberger"), (4, "Sellmeier 2"), (5, "Conrady"), (6, "Sellmeier 3"),
        (7, "Handbook of Optics 1"), (8, "Handbook of Optics 2"), (9, "Sellmeier 4"), (10, "Extended"), (11, "Sellmeier 5"),
        (12, "Extended 2"), (13, "Extended 3"),
    ]

    static func resized(_ a: [Double], to n: Int) -> [Double] {
        Array((a + Array(repeating: 0, count: max(0, n - a.count))).prefix(n))
    }

    /// The engine's `GlassSpec` for the current values.
    var spec: JSONValue {
        var o: [String: JSONValue] = [
            "name": .string(name.trimmingCharacters(in: .whitespaces)),
            "status": status.json,
            "exclude_substitution": .bool(excludeSubstitution),
            "comment": .string(comment),
        ]
        switch kind {
        case .model: o["model"] = .object(["nd": .number(nd), "vd": .number(vd), "dpgf": .number(dpgf)])
        case .formula:
            o["formula"] = .integer(Int64(formula))
            o["coefficients"] = .array(coefficients.map(JSONValue.number))
        }
        if let minUm { o["min_um"] = .number(minUm) }
        if let maxUm { o["max_um"] = .number(maxUm) }
        if let density { o["density"] = .number(density) }
        if let tce { o["tce"] = .number(tce) }
        if let relativeCost { o["relative_cost"] = .number(relativeCost) }
        if let base { o["base"] = base }
        return .object(o)
    }

    /// Asks the engine for the glass; sets `preview` or `problem`.
    @discardableResult
    func check(engine: any EngineClient) -> JSONValue? {
        if let why = GlassLibrary.nameProblem(name, glass: true) {
            preview = nil
            problem = why
            return nil
        }
        do {
            let g = try engine.glass(from: spec)
            preview = g
            problem = nil
            return g
        } catch {
            preview = nil
            problem = Self.describe(error)
            return nil
        }
    }

    /// n_d and V_d of the previewed glass.
    var previewNumbers: (nd: Double, vd: Double)? {
        guard let p = preview, let nd = p["nd"]?.doubleValue, let vd = p["vd"]?.doubleValue else { return nil }
        return (nd, vd)
    }

    /// Makes the glass and stores it in the user catalog. Returns whether it saved.
    func save(engine: any EngineClient) -> Bool {
        guard let g = check(engine: engine) else { return false }
        do {
            try library.putGlass(g, in: catalog, replacing: originalName)
            return true
        } catch {
            problem = error.localizedDescription
            return false
        }
    }
}

/// "1 glass", "13 glasses".
func glassCountText(_ n: Int) -> String { n == 1 ? "1 glass" : "\(n) glasses" }

/// " The reader reported 3 warnings (first: …)." after a catalog load, or "".
func warningsNote(_ warnings: [CatalogWarning]) -> String {
    guard let first = warnings.first else { return "" }
    let count = warnings.count == 1 ? "1 warning" : "\(warnings.count) warnings"
    return " Reading it reported \(count) (first: \(first.text))."
}
