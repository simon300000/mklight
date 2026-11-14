import Charts
import MKLightEngine
import MKLightModel
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    /// AGF glass catalog text.
    static var agfCatalog: UTType { UTType(filenameExtension: "agf", conformingTo: .plainText) ?? .plainText }
}

/// Sort keys of the optional columns: glasses without a value sort last.
extension GlassRow {
    var sortPgf: Double { dpgf ?? .infinity }
    var sortDensity: Double { density ?? .infinity }
    var sortCost: Double { relativeCost ?? .infinity }
    var sortTce: Double { tce ?? .infinity }
    var sortLambda80: Double { lambda80Um ?? .infinity }
    var sortStatus: Int { [1: 0, 0: 1, 4: 2, 3: 3, 2: 4][status.code] ?? 5 }
}

/// The Glass Catalogs window (Lens ▸ Glass Catalogs…, the System inspector):
/// catalogs, a glass table and the selected glass's details.
struct GlassBrowserView: View {
    static let windowID = "glass-catalogs"
    @Bindable var model: GlassBrowserModel
    @State private var columns = NavigationSplitViewVisibility.all
    @State private var importing: ImportTarget?
    @State private var exporting: AGFFile?
    @State private var renaming: String?
    @State private var newName = ""
    @State private var deletingCatalog: String?

    enum ImportTarget { case library, design }

    var body: some View {
        NavigationSplitView(columnVisibility: $columns) {
            GlassCatalogSidebar(model: model, importing: $importing, export: export, renaming: $renaming, deleting: $deletingCatalog)
                .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 300)
        } content: {
            GlassTable(model: model)
                .navigationSplitViewColumnWidth(min: 420, ideal: 640)
        } detail: {
            GlassDetailView(model: model)
                .navigationSplitViewColumnWidth(min: 270, ideal: 300, max: 420)
        }
        .navigationTitle(scopeTitle)
        .navigationSubtitle(model.designTitle.map { "Glasses of \($0)" } ?? "Library")
        .searchable(text: $model.search, placement: .toolbar, prompt: "Glass name")
        .toolbar { toolbar }
        .overlay(alignment: .bottom) { footer }
        .fileImporter(
            isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
            allowedContentTypes: importing == .library ? [.folder, .agfCatalog, .json, .plainText, .data] : [.agfCatalog, .json, .plainText, .data],
            allowsMultipleSelection: importing == .library
        ) { result in
            let target = importing
            importing = nil
            guard case .success(let urls) = result else { return }
            switch target {
            case .library: model.addToLibrary(urls)
            case .design: if let u = urls.first { Task { await model.addToDesign(u) } }
            case nil: break
            }
        }
        .fileExporter(
            isPresented: Binding(get: { exporting != nil }, set: { if !$0 { exporting = nil } }), document: exporting,
            contentType: .agfCatalog, defaultFilename: exporting?.name
        ) { result in
            if case .failure(let e) = result { model.message = e.localizedDescription }
        }
        .sheet(item: Binding(get: { model.editor.map(EditorBox.init) }, set: { if $0 == nil { model.editor = nil } })) { box in
            GlassEditorSheet(editor: box.editor, engine: model.workspace?.documentEngine, save: { model.saveEditor() }, cancel: { model.editor = nil })
        }
        .sheet(item: Binding(get: { model.reloadReport.map(ReportBox.init) }, set: { if $0 == nil { model.reloadReport = nil } })) { box in
            CatalogChangesSheet(changes: box.changes) { model.reloadReport = nil }
        }
        .alert("Rename Catalog", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $newName)
            Button("Rename") {
                if let old = renaming {
                    do {
                        try model.library.renameUserCatalog(old, to: newName)
                        model.scope = .catalog(newName.trimmingCharacters(in: .whitespaces))
                    } catch { model.message = error.localizedDescription }
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Designs that name the catalog in a glass reference keep the old name.")
        }
        .onChange(of: renaming) { _, r in if let r { newName = r } }
        .confirmationDialog(
            "Move “\(deletingCatalog ?? "")” to the Trash?", isPresented: Binding(get: { deletingCatalog != nil }, set: { if !$0 { deletingCatalog = nil } })
        ) {
            Button("Move to Trash", role: .destructive) {
                if let n = deletingCatalog {
                    do {
                        try model.library.deleteUserCatalog(n)
                        model.scope = .all
                        model.message = "Moved \(n) to the Trash."
                    } catch { model.message = error.localizedDescription }
                }
            }
        } message: {
            Text("Designs that use its glasses will not find them.")
        }
        .onAppear(perform: applyLaunchOptions)
        // Edit ▸ Undo undoes the design's changes made here.
        .focusedSceneValue(\.documentUndo, model.target?.undo())
        .task(id: RefreshKey(library: model.library.revision, catalogs: model.workspace?.catalogRevision ?? 0, target: model.designTitle)) {
            await model.refresh()
        }
    }

    /// Debug builds: `-MKLightGlassScope <catalog>` and `-MKLightGlassSearch
    /// <text>` (screenshots, UI checks).
    private func applyLaunchOptions() {
        #if DEBUG
        let d = UserDefaults.standard
        if let c = d.string(forKey: "MKLightGlassScope") { model.scope = .catalog(c) }
        if let q = d.string(forKey: "MKLightGlassSearch") { model.search = q }
        #endif
    }

    private struct RefreshKey: Hashable {
        var library: Int
        var catalogs: Int
        var target: String?
    }

    private struct EditorBox: Identifiable {
        let editor: GlassEditorModel
        var id: ObjectIdentifier { ObjectIdentifier(editor) }
    }

    private struct ReportBox: Identifiable {
        let changes: CatalogChanges
        var id: String { changes.catalog }
    }

    private var scopeTitle: String {
        switch model.scope {
        case .all: "All Glasses"
        case .catalog(let n): n == "BUILTIN" ? "Built-in" : n
        }
    }

    private func export(_ catalog: String) {
        guard let e = model.agfExport(of: catalog) else { return }
        exporting = AGFFile(name: catalog + ".agf", text: e.text)
        if !e.warnings.isEmpty { model.message = "Not exported: " + e.warnings.joined(separator: "; ") }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Menu {
                Picker("Show", selection: $model.filter) {
                    ForEach(GlassBrowserModel.StatusFilter.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.inline)
            } label: {
                Label("Filter", systemImage: model.filter == .all ? "line.3.horizontal.decrease.circle" : "line.3.horizontal.decrease.circle.fill")
            }
            .help("Show only some glasses")
        }
        ToolbarItem(placement: .primaryAction) {
            Menu {
                Button("New Glass…") { model.newGlass() }
                Button("New User Catalog") { _ = model.newUserCatalog() }
                Divider()
                Button("Add Catalog or Folder to Library…") { importing = .library }
                Button("Add Catalog to Design…") { importing = .design }
                    .disabled(model.target == nil)
            } label: {
                Label("Add", systemImage: "plus")
            }
            .help("Add a glass or a catalog")
        }
    }

    @ViewBuilder private var footer: some View {
        if let m = model.message {
            Label(m, systemImage: "info.circle")
                .font(.callout)
                .lineLimit(2)
                .padding(.horizontal, Space.l)
                .padding(.vertical, Space.m)
                .glassEffect(.regular, in: .capsule)
                .padding(.bottom, Space.l)
                .onTapGesture { model.message = nil }
                .task(id: m) {
                    try? await Task.sleep(for: .seconds(6))
                    if model.message == m { model.message = nil }
                }
        }
    }
}

/// AGF text for the file exporter.
struct AGFFile: FileDocument {
    static var readableContentTypes: [UTType] { [.agfCatalog] }
    var name: String
    var text: String

    init(name: String, text: String) {
        self.name = name
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        name = configuration.file.filename ?? "Catalog.agf"
        text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

// MARK: - Sidebar

private struct GlassCatalogSidebar: View {
    let model: GlassBrowserModel
    @Binding var importing: GlassBrowserView.ImportTarget?
    let export: (String) -> Void
    @Binding var renaming: String?
    @Binding var deleting: String?

    var body: some View {
        List(selection: Binding(get: { model.scope }, set: { model.scope = $0 ?? .all })) {
            Label("All Glasses", systemImage: "square.grid.2x2").tag(GlassBrowserModel.Scope.all)
            ForEach(model.builtinCatalogs) { c in row(c, title: "Built-in Glasses", symbol: "shippingbox") }
            Section {
                ForEach(model.libraryCatalogs) { c in
                    row(c, symbol: "books.vertical")
                        .contextMenu {
                            reload(c)
                            Button("Remove from Library") { model.removeFromLibrary(c) }
                            showInFinder(c)
                        }
                }
                if model.libraryCatalogs.isEmpty {
                    Button("Add Catalog…") { importing = .library }
                        .buttonStyle(.link)
                        .help("Catalogs and folders in the library load into every design")
                }
            } header: {
                Text("Library")
            }
            Section("User Glasses") {
                ForEach(model.userCatalogs) { c in
                    row(c, symbol: "person.crop.square")
                        .contextMenu {
                            Button("New Glass…") {
                                model.scope = .catalog(c.name)
                                model.newGlass()
                            }
                            Button("Rename…") { renaming = c.name }
                            Button("Export as AGF…") { export(c.name) }
                            showInFinder(c)
                            Divider()
                            Button("Move to Trash…", role: .destructive) { deleting = c.name }
                        }
                }
                if model.userCatalogs.isEmpty {
                    Button("New User Catalog") { _ = model.newUserCatalog() }
                        .buttonStyle(.link)
                }
            }
            if let title = model.designTitle {
                Section {
                    ForEach(model.designCatalogs) { c in
                        row(c, symbol: "doc.text")
                            .contextMenu {
                                reload(c)
                                Button("Remove from Design") { model.removeFromDesign(c) }
                                showInFinder(c)
                            }
                    }
                    Button("Add Catalog to Design…") { importing = .design }
                        .buttonStyle(.link)
                } header: {
                    Text(title)
                }
            }
            ForEach(model.library.problems + (model.workspace?.libraryWarnings ?? []), id: \.self) { w in
                WarningNote(w)
            }
        }
        .listStyle(.sidebar)
    }

    private func row(_ c: CatalogInfo, title: String? = nil, symbol: String) -> some View {
        Label {
            HStack {
                Text(title ?? c.name).lineLimit(1)
                Spacer()
                Text("\(c.glassCount)").foregroundStyle(.secondary).monospacedDigit()
            }
        } icon: {
            Image(systemName: symbol)
        }
        .help(c.path ?? (c.isBuiltin ? "The glasses built into MKLight" : c.name))
        .tag(GlassBrowserModel.Scope.catalog(c.name))
    }

    private func reload(_ c: CatalogInfo) -> some View {
        Button("Reload from Disk") { Task { await model.reload(c) } }
            .disabled(c.path == nil)
    }

    @ViewBuilder private func showInFinder(_ c: CatalogInfo) -> some View {
        #if os(macOS)
        if let p = c.path {
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: p)]) }
        }
        #endif
    }
}

// MARK: - Table

struct GlassTable: View {
    @Bindable var model: GlassBrowserModel

    var body: some View {
        let rows = model.visibleRows
        Table(rows, selection: $model.selection, sortOrder: $model.sortOrder) {
            Group {
                TableColumn("Glass", value: \.name) { (r: GlassRow) in GlassNameCell(row: r) }
                    .width(min: 90, ideal: 120)
                TableColumn("Catalog", value: \.catalog) { (r: GlassRow) in Text(r.catalog).foregroundStyle(.secondary) }
                    .width(min: 60, ideal: 100)
                TableColumn("nd", value: \.nd) { (r: GlassRow) in NumberCell(text: r.ndText) }
                    .width(min: 60, ideal: 70)
                TableColumn("Vd", value: \.vd) { (r: GlassRow) in NumberCell(text: r.vdText) }
                    .width(min: 46, ideal: 54)
                TableColumn("ΔPg,F", value: \.sortPgf) { (r: GlassRow) in NumberCell(text: r.dpgfText) }
                    .width(min: 54, ideal: 62)
                TableColumn("Status", value: \.sortStatus) { (r: GlassRow) in StatusBadge(status: r.status, overridden: r.overridden) }
                    .width(min: 80, ideal: 104)
            }
            Group {
                TableColumn("Density", value: \.sortDensity) { (r: GlassRow) in NumberCell(text: r.densityText) }
                    .width(min: 50, ideal: 58)
                TableColumn("Cost", value: \.sortCost) { (r: GlassRow) in NumberCell(text: r.costText) }
                    .width(min: 40, ideal: 46)
                TableColumn("TCE", value: \.sortTce) { (r: GlassRow) in NumberCell(text: r.tceText) }
                    .width(min: 40, ideal: 46)
                TableColumn("λ80 nm", value: \.sortLambda80) { (r: GlassRow) in NumberCell(text: r.lambda80Text) }
                    .width(min: 48, ideal: 56)
                TableColumn("Range µm", value: \.maxUm) { (r: GlassRow) in Text(r.rangeText).monospacedDigit().foregroundStyle(.secondary) }
                    .width(min: 70, ideal: 86)
            }
        }
        .contextMenu(forSelectionType: GlassRow.ID.self) { ids in
            if let id = ids.first, let row = model.rows.first(where: { $0.id == id }) {
                GlassRowMenu(model: model, row: row)
            }
        } primaryAction: { _ in
            model.useSelected()
        }
        .overlay {
            if rows.isEmpty {
                if model.search.isEmpty && model.filter == .all {
                    ContentUnavailableView("No Glasses", systemImage: "circle.dotted", description: Text(emptyText))
                } else {
                    ContentUnavailableView.search(text: model.search)
                }
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            if let name = model.scopeCatalog, let c = model.catalog(name) {
                CatalogBar(model: model, catalog: c)
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            HStack {
                Text("\(rows.count) of \(model.rows.count) glasses")
                Spacer()
                Text("Double-click a glass to use it for the selected surface")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, Space.l)
            .padding(.vertical, Space.s)
            .background(.bar)
        }
    }

    private var emptyText: String {
        if let n = model.scopeCatalog, model.library.isUserCatalog(n) { return "Add a glass with the + button." }
        return "This catalog has no glasses."
    }

    static func fixed(_ v: Double, _ digits: Int) -> String {
        v.isFinite ? String(format: "%.\(digits)f", v) : "—"
    }

    static func signed(_ v: Double, _ digits: Int) -> String {
        (v >= 0 ? "+" : "−") + String(format: "%.\(digits)f", abs(v))
    }

    static func range(_ r: GlassRow) -> String {
        r.unrestrictedRange ? "any" : String(format: "%.2f–%.2f", r.minUm, r.maxUm)
    }
}

/// The selected catalog's origin and file, with its actions.
private struct CatalogBar: View {
    let model: GlassBrowserModel
    let catalog: CatalogInfo

    var body: some View {
        let user = model.library.isUserCatalog(catalog.name)
        HStack(spacing: Space.m) {
            VStack(alignment: .leading, spacing: 1) {
                Text(origin(user: user)).font(.callout.weight(.medium))
                if let p = catalog.path {
                    Text(p).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
            }
            Spacer(minLength: Space.m)
            if user {
                Button("New Glass…", systemImage: "plus") { model.newGlass() }
            } else if catalog.path != nil {
                Button("Reload", systemImage: "arrow.clockwise") { Task { await model.reload(catalog) } }
                    .help("Read the catalog file again and list what changed")
            }
            if catalog.inProject {
                Button("Remove from Design", systemImage: "minus.circle") { model.removeFromDesign(catalog) }
                    .help("Remove the catalog from this design (undo in the design window)")
            } else if catalog.library && !user {
                Button("Remove from Library", systemImage: "minus.circle") { model.removeFromLibrary(catalog) }
                    .help("Stop loading this catalog into every design (the file stays)")
            }
        }
        .labelStyle(.titleAndIcon)
        .buttonStyle(.glass)
        .controlSize(.small)
        .padding(.horizontal, Space.l)
        .padding(.vertical, Space.s)
        .background(.bar)
    }

    private func origin(user: Bool) -> String {
        if catalog.isBuiltin { return "Built into MKLight" }
        if user { return "Your glasses · saved in the app's library" }
        switch (catalog.library, catalog.inProject) {
        case (true, true): return "Library · recorded in this design"
        case (true, false): return "Library · loaded into every design"
        default: return "This design only"
        }
    }
}

/// A glass name, with a mark when glass substitution may not use it.
private struct GlassNameCell: View {
    let row: GlassRow
    var body: some View {
        HStack(spacing: Space.xs) {
            Text(row.name).fontWeight(row.status == .preferred ? .medium : .regular)
            if row.excludeSubstitution {
                Image(systemName: "nosign").foregroundStyle(.secondary).help("Excluded from glass substitution")
            }
        }
    }
}

/// A right-aligned number.
private struct NumberCell: View {
    let text: String
    var body: some View {
        Text(text).monospacedDigit().frame(maxWidth: .infinity, alignment: .trailing)
    }
}

extension GlassRow {
    var ndText: String { GlassTable.fixed(nd, 5) }
    var vdText: String { GlassTable.fixed(vd, 2) }
    var dpgfText: String { dpgf.map { GlassTable.signed($0, 4) } ?? "—" }
    var densityText: String { density.map { GlassTable.fixed($0, 2) } ?? "—" }
    var costText: String { relativeCost.map { GlassTable.fixed($0, 1) } ?? "—" }
    var tceText: String { tce.map { GlassTable.fixed($0, 1) } ?? "—" }
    var lambda80Text: String { lambda80Um.map { GlassTable.fixed($0 * 1000, 0) } ?? "—" }
    var rangeText: String { GlassTable.range(self) }
}

/// The context menu of a glass (table rows).
private struct GlassRowMenu: View {
    let model: GlassBrowserModel
    let row: GlassRow

    var body: some View {
        Button(useTitle) {
            model.selection = row.id
            model.useSelected()
        }
        .disabled(model.assignment.index == nil)
        Button("Copy Name") { copy(row.name) }
        Divider()
        Menu("Status") {
            ForEach(GlassStatus.choices) { s in
                Button(s.title) { model.setStatus(s, for: row) }
            }
            Divider()
            Button("Catalog's Status") { model.setStatus(nil, for: row) }
        }
        Button(row.excludeSubstitution ? "Allow in Glass Substitution" : "Exclude from Glass Substitution") {
            model.setExcluded(!row.excludeSubstitution, for: row)
        }
        if row.overridden {
            Button("Reset to Catalog Settings") { model.resetStatus(of: row) }
        }
        Divider()
        Button(model.library.isUserCatalog(row.catalog) ? "Duplicate…" : "Duplicate to User Glasses…") {
            Task { await model.duplicate(row) }
        }
    }

    private var useTitle: String {
        model.assignment.index.map { "Use for Surface \($0)" } ?? "Use for Selected Surface"
    }
}

func copy(_ text: String) {
    #if os(macOS)
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
    #endif
}

/// A glass status as a small coloured label; overridden statuses are marked.
struct StatusBadge: View {
    let status: GlassStatus
    var overridden = false
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        HStack(spacing: Space.xs) {
            Circle()
                .fill(Palette.color(GlassMapStatus(json: status.json).rgb(dark: colorScheme == .dark)))
                .frame(width: 7, height: 7)
            Text(status.title)
            if overridden {
                Image(systemName: "person.fill").font(.caption2).foregroundStyle(.secondary)
                    .help("Your setting (the catalog says otherwise)")
            }
        }
    }
}

// MARK: - Details

private struct GlassDetailView: View {
    @Bindable var model: GlassBrowserModel

    var body: some View {
        if let d = model.details, let row = model.selectedRow, d.row.id == row.id {
            ScrollView {
                VStack(alignment: .leading, spacing: Space.xl) {
                    header(d)
                    actions(d)
                    availability(d)
                    numbers(d)
                    if d.indexCurve.count > 1 { indexChart(d) }
                    dispersion(d)
                    physical(d)
                    if !d.comment.isEmpty {
                        Text(d.comment).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }
                .padding(Space.xl)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            ContentUnavailableView("No Glass Selected", systemImage: "circle.dashed", description: Text("Select a glass to see its data and use it."))
        }
    }

    private func header(_ d: GlassDetails) -> some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            Text(d.row.name).font(.title2.weight(.semibold)).textSelection(.enabled)
            HStack(spacing: Space.m) {
                Text(d.row.catalog).foregroundStyle(.secondary)
                StatusBadge(status: d.row.status, overridden: d.row.overridden).font(.callout)
            }
        }
    }

    private func actions(_ d: GlassDetails) -> some View {
        let user = model.library.isUserCatalog(d.row.catalog)
        let assignment = model.assignment
        return VStack(alignment: .leading, spacing: Space.m) {
            Button {
                model.useSelected()
            } label: {
                Label(assignment.index.map { "Use for Surface \($0)" } ?? "Use for Surface", systemImage: "arrow.down.left.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.glassProminent)
            .controlSize(.large)
            .disabled(assignment.index == nil)
            .help(assignment.why ?? "Make the selected surface's following medium this glass")
            if let why = assignment.why {
                Text(why).font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("Copy Name") { copy(d.row.name) }
                if user {
                    Button("Edit…") { model.editSelected() }
                    Button("Delete", role: .destructive) { model.deleteSelected() }
                } else {
                    Button("Duplicate…") { model.duplicateSelected() }
                        .help("Copy this glass into a user catalog to edit it")
                }
            }
            .buttonStyle(.glass)
        }
    }

    private func availability(_ d: GlassDetails) -> some View {
        let row = d.row
        return GroupBox {
            VStack(alignment: .leading, spacing: Space.m) {
                Picker("Status", selection: Binding(
                    get: { model.library.override(catalog: row.catalog, name: row.name)?.status },
                    set: { model.setStatus($0, for: row) })
                ) {
                    Text("Catalog (\(d.catalogStatus.title))").tag(GlassStatus?.none)
                    Divider()
                    ForEach(GlassStatus.choices) { s in Text(s.title).tag(GlassStatus?.some(s)) }
                }
                Toggle("Use in glass substitution", isOn: Binding(
                    get: { !row.excludeSubstitution },
                    set: { model.setExcluded(!$0 == d.catalogExcludeSubstitution ? nil : !$0, for: row) }))
                Text("Your settings apply to every design and are not saved in projects. Glass substitution skips obsolete, special and excluded glasses by default.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } label: {
            Text("Availability")
        }
    }

    private func numbers(_ d: GlassDetails) -> some View {
        GroupBox("Optical Data") {
            Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: Space.xs) {
                gridRow("nd", Self.fmt(d.row.nd, 6))
                gridRow("Vd", Self.fmt(d.row.vd, 3))
                if let p = d.row.pgf { gridRow("Pg,F", Self.fmt(p, 4)) }
                if let p = d.row.dpgf { gridRow("ΔPg,F", GlassTable.signedText(p)) }
                if let p = d.catalogDpgf { gridRow("ΔPg,F (catalog)", GlassTable.signedText(p)) }
                gridRow("Range", d.row.unrestrictedRange ? "unrestricted" : String(format: "%.3f – %.3f µm", d.row.minUm, d.row.maxUm))
                ForEach(d.lines, id: \.line) { l in
                    gridRow("n\(l.line) (\(String(format: "%.1f", l.um * 1000)) nm)", Self.fmt(l.n, 6))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func indexChart(_ d: GlassDetails) -> some View {
        GroupBox("Index") {
            Chart {
                ForEach(Array(d.indexCurve.enumerated()), id: \.offset) { _, p in
                    LineMark(x: .value("λ (µm)", p.um), y: .value("n", p.n))
                }
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .chartXAxisLabel("λ (µm)")
            .frame(height: 140)
            .accessibilityLabel("Refractive index against wavelength")
        }
    }

    private func dispersion(_ d: GlassDetails) -> some View {
        GroupBox("Dispersion") {
            VStack(alignment: .leading, spacing: Space.s) {
                if let m = d.model {
                    Text("Model glass: nd \(Self.fmt(m.nd, 5)), Vd \(Self.fmt(m.vd, 2)), ΔPg,F \(GlassTable.signedText(m.dpgf))")
                } else {
                    Text(d.formula.map { "Formula \($0): \(d.formulaName)" } ?? d.formulaName)
                }
                if d.model == nil, !d.coefficients.isEmpty, let f = d.formula {
                    let names = GlassEditorModel.coefficientNames(f)
                    Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: 2) {
                        ForEach(Array(d.coefficients.enumerated()), id: \.offset) { i, c in
                            if c != 0 || i < names.count {
                                gridRow(i < names.count ? names[i] : "c\(i)", String(format: "%.10g", c))
                            }
                        }
                    }
                    .font(.callout)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func physical(_ d: GlassDetails) -> some View {
        GroupBox("Physical and Catalog Data") {
            Grid(alignment: .leading, horizontalSpacing: Space.l, verticalSpacing: Space.xs) {
                if let v = d.row.density { gridRow("Density", String(format: "%.3f g/cm³", v)) }
                if let v = d.row.tce { gridRow("TCE −30…70 °C", String(format: "%.2f ·10⁻⁶/K", v)) }
                if let v = d.tce100To300 { gridRow("TCE 100…300 °C", String(format: "%.2f ·10⁻⁶/K", v)) }
                if let v = d.row.relativeCost { gridRow("Relative cost", String(format: "%.2f", v)) }
                if let v = d.row.climateResistance { gridRow("Climate (CR)", Self.fmt(v, 0)) }
                if let v = d.row.stainResistance { gridRow("Stain (FR)", Self.fmt(v, 0)) }
                if let v = d.acidResistance { gridRow("Acid (SR)", Self.fmt(v, 1)) }
                if let v = d.alkaliResistance { gridRow("Alkali (AR)", Self.fmt(v, 1)) }
                if let v = d.phosphateResistance { gridRow("Phosphate (PR)", Self.fmt(v, 1)) }
                if let lo = d.row.transmissionMinUm, let hi = d.row.transmissionMaxUm {
                    gridRow("Transmission data", String(format: "%.3f – %.3f µm", lo, hi))
                }
                if let v = d.row.lambda80Um { gridRow("λ for 80 % (10 mm)", String(format: "%.0f nm", v * 1000)) }
                gridRow("Thermal dn/dT data", d.hasThermalData ? "yes" : "none")
                if let m = d.milCode { gridRow("Glass code", m) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func gridRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).monospacedDigit().textSelection(.enabled)
        }
    }

    static func fmt(_ v: Double, _ digits: Int) -> String { GlassTable.fixedText(v, digits) }
}

extension GlassTable {
    static func fixedText(_ v: Double, _ digits: Int) -> String { fixed(v, digits) }
    static func signedText(_ v: Double) -> String { signed(v, 4) }
}

// MARK: - Reload report

/// What reloading a catalog changed.
struct CatalogChangesSheet: View {
    let changes: CatalogChanges
    let done: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Space.l) {
            Text("\(changes.catalog) Reloaded").font(.title3.weight(.semibold))
            Text(changes.isEmpty ? "The file has the same glasses as before (\(changes.unchanged))." : "\(changes.summary); \(changes.unchanged) unchanged.")
                .foregroundStyle(.secondary)
            if !changes.isEmpty {
                List {
                    if !changes.added.isEmpty {
                        Section("Added") { ForEach(changes.added, id: \.self) { Text($0) } }
                    }
                    if !changes.removed.isEmpty {
                        Section("Removed") { ForEach(changes.removed, id: \.self) { Text($0) } }
                    }
                    if !changes.changed.isEmpty {
                        Section("Changed") {
                            ForEach(changes.changed, id: \.name) { c in
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(c.name)
                                    Text(Self.describe(c)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                                }
                            }
                        }
                    }
                }
                .frame(minHeight: 180, idealHeight: 260)
            }
            HStack {
                Spacer()
                Button("Done", action: done).keyboardShortcut(.defaultAction)
            }
        }
        .padding(Space.xl)
        .frame(width: 440)
    }

    static func describe(_ c: CatalogChanges.Change) -> String {
        var parts: [String] = []
        if c.ndBefore != c.ndAfter { parts.append(String(format: "nd %.5f → %.5f", c.ndBefore, c.ndAfter)) }
        if c.vdBefore != c.vdAfter { parts.append(String(format: "Vd %.3f → %.3f", c.vdBefore, c.vdAfter)) }
        let other = c.fields.filter { !["nd", "vd"].contains($0) }
        if !other.isEmpty { parts.append(other.joined(separator: ", ")) }
        return parts.joined(separator: "; ")
    }
}
