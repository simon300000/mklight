using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// The design a Glass Catalogs window shows and acts on (the frontmost design window's
/// session). Delegates, so the browser stays free of WinUI and window types.
/// </summary>
/// <param name="Key">Identifies the design (its session).</param>
/// <param name="Title">The design's name, for the sidebar.</param>
/// <param name="Engine">The design's engine session.</param>
/// <param name="GlassSurface">The selected surface whose following medium a glass can be assigned to, or null.</param>
/// <param name="Assign">Sets the medium after surface <c>index</c> to a catalog glass (undoable).</param>
/// <param name="AddCatalog">Loads a catalog file and records it in the design's project (undoable).</param>
/// <param name="RemoveCatalog">Removes a catalog from the design's project (undoable).</param>
/// <param name="CatalogsDidChange">The session's catalogs changed: glass names again, a new snapshot.</param>
public sealed record GlassBrowserTarget(
    object Key,
    Func<string> Title,
    IEngineClient Engine,
    Func<int?> GlassSurface,
    Action<int, string, string> Assign,
    Func<string, Task<CatalogInfo>> AddCatalog,
    Action<string> RemoveCatalog,
    Func<Task> CatalogsDidChange);

/// <summary>Which glasses the table shows besides the search.</summary>
public enum GlassFilter { All, Available, Preferred, Substitution }

/// <summary>The glass table's columns (each sortable).</summary>
public enum GlassColumn { Name, Catalog, Nd, Vd, Dpgf, Status, Density, Cost, Tce, Lambda80, Range }

/// <summary>
/// The Glass Catalogs window's model (the Mac app's GlassBrowserModel): catalogs in a sidebar
/// (all, built-in, library, user glasses, the design's own), a searchable, sortable glass
/// table and the selected glass's details, with the actions on them. It reads everything from
/// an engine session: the frontmost design's, or one of its own with the library when no
/// design is open. Free of WinUI types; runs on the UI thread.
/// </summary>
public sealed class GlassBrowserModel
{
    public GlassBrowserModel(GlassLibrary library, Func<IEngineClient>? ownEngine = null)
    {
        Library = library;
        _ownEngineFactory = ownEngine;
    }

    public GlassLibrary Library { get; }
    readonly Func<IEngineClient>? _ownEngineFactory;
    IEngineClient? _ownEngine;

    /// <summary>Raised when catalogs, rows, the scope, search, filter or sort changed (the sidebar and table redraw).</summary>
    public event Action? Changed;
    /// <summary>Raised when the selection or its details changed.</summary>
    public event Action? DetailsChanged;
    /// <summary>A short note about the last action (the window shows it in its footer).</summary>
    public event Action<string>? Message;

    public GlassBrowserTarget? Target { get; private set; }
    public string? DesignTitle => Target?.Title();

    /// <summary>The session the window reads: the design's, or its own.</summary>
    public IEngineClient? Engine => Target?.Engine ?? (_ownEngine ??= _ownEngineFactory?.Invoke());

    /// <summary>The session of its own, once made (the window disposes it when it closes).</summary>
    public IEngineClient? OwnEngine => _ownEngine;

    /// <summary>A design window became the frontmost: the window follows it.</summary>
    public Task AttachAsync(GlassBrowserTarget target)
    {
        if (Target?.Key == target.Key) return Task.CompletedTask;
        Target = target;
        return RefreshAsync();
    }

    /// <summary>A design closed.</summary>
    public Task DetachAsync(object key)
    {
        if (Target?.Key != key) return Task.CompletedTask;
        Target = null;
        return RefreshAsync();
    }

    // ---- Catalogs ----

    public IReadOnlyList<CatalogInfo> Catalogs { get; private set; } = [];
    public IReadOnlyList<GlassRow> Rows { get; private set; } = [];
    /// <summary>Problems of the library and its sync into the session.</summary>
    public IReadOnlyList<string> Warnings { get; private set; } = [];

    public IEnumerable<CatalogInfo> BuiltinCatalogs => Catalogs.Where(c => c.IsBuiltin);
    /// <summary>Library catalogs other than the user's own.</summary>
    public IEnumerable<CatalogInfo> LibraryCatalogs => Catalogs.Where(c => c.Library && !Library.IsUserCatalog(c.Name));
    public IEnumerable<CatalogInfo> UserCatalogs => Catalogs.Where(c => c.Library && Library.IsUserCatalog(c.Name));
    /// <summary>Catalogs only the design's project brings.</summary>
    public IEnumerable<CatalogInfo> DesignCatalogs => Catalogs.Where(c => c.InProject && !c.Library);

    public CatalogInfo? Catalog(string name) => Catalogs.FirstOrDefault(c => c.Name == name);

    string? _scope;
    /// <summary>The catalog the table shows, or null for all glasses.</summary>
    public string? Scope
    {
        get => _scope;
        set
        {
            if (_scope == value) return;
            _scope = value;
            Changed?.Invoke();
            _ = LoadRowsAsync();
        }
    }

    public static string ScopeTitle(string? scope) => scope switch { null => "All Glasses", "BUILTIN" => "Built-in Glasses", _ => scope };

    /// <summary>Reads the catalogs and the glasses again (after the library or the design's catalogs changed).</summary>
    public async Task RefreshAsync()
    {
        if (Engine is not { } engine) return;
        try
        {
            if (await GlassLibrarySync.SyncAsync(engine, Library) && Target is { } t) await t.CatalogsDidChange();
        }
        catch (EngineException e) { Say(e.Message); }
        try { Catalogs = await engine.CatalogsAsync(); }
        catch (EngineException e) { Catalogs = []; Say(e.Message); }
        Warnings = [.. Library.Problems, .. GlassLibrarySync.Warnings(engine)];
        if (_scope is { } s && Catalog(s) is null) _scope = null;
        Changed?.Invoke();
        await LoadRowsAsync();
    }

    int _rowsRequest;

    async Task LoadRowsAsync()
    {
        if (Engine is not { } engine) return;
        var request = ++_rowsRequest;
        IReadOnlyList<GlassRow> rows;
        try { rows = await engine.GlassRowsAsync(_scope); }
        catch (EngineException) { rows = []; }
        if (request != _rowsRequest) return;
        Rows = rows;
        _visible = null;
        if (_selection is { } sel && !rows.Any(r => r.Id == sel)) _selection = null;
        Changed?.Invoke();
        await LoadDetailsAsync();
    }

    // ---- Search, filter, sort ----

    string _search = "";
    public string Search
    {
        get => _search;
        set { if (_search == value) return; _search = value; _visible = null; Changed?.Invoke(); }
    }

    GlassFilter _filter;
    public GlassFilter Filter
    {
        get => _filter;
        set { if (_filter == value) return; _filter = value; _visible = null; Changed?.Invoke(); }
    }

    public static string FilterTitle(GlassFilter f) => f switch
    {
        GlassFilter.Available => "Hide Obsolete",
        GlassFilter.Preferred => "Preferred Only",
        GlassFilter.Substitution => "Usable in Glass Substitution",
        _ => "All Glasses",
    };

    public GlassColumn SortColumn { get; private set; } = GlassColumn.Name;
    public bool SortAscending { get; private set; } = true;

    /// <summary>Sorts by a column; the sorted column again reverses the order.</summary>
    public void SortBy(GlassColumn column)
    {
        if (SortColumn == column) SortAscending = !SortAscending;
        else (SortColumn, SortAscending) = (column, true);
        _visible = null;
        Changed?.Invoke();
    }

    IReadOnlyList<GlassRow>? _visible;

    /// <summary>
    /// The rows the table shows: filtered by the search (name, and catalog under All Glasses;
    /// case-insensitive) and the status filter, in the sort order.
    /// </summary>
    public IReadOnlyList<GlassRow> VisibleRows => _visible ??= Visible(Rows, _search, _filter, _scope is null, SortColumn, SortAscending);

    public static IReadOnlyList<GlassRow> Visible(IEnumerable<GlassRow> rows, string search, GlassFilter filter, bool allCatalogs, GlassColumn column, bool ascending)
    {
        var q = search.Trim();
        var list = rows.Where(r =>
            (q.Length == 0 || r.Name.Contains(q, StringComparison.OrdinalIgnoreCase) || (allCatalogs && r.Catalog.Contains(q, StringComparison.OrdinalIgnoreCase)))
            && filter switch
            {
                GlassFilter.Available => r.Status != GlassStatus.Obsolete,
                GlassFilter.Preferred => r.Status == GlassStatus.Preferred,
                GlassFilter.Substitution => !r.ExcludeSubstitution && r.Status != GlassStatus.Obsolete && r.Status != GlassStatus.Special,
                _ => true,
            }).ToList();
        Comparison<GlassRow> by = column switch
        {
            GlassColumn.Name => (a, b) => GlassLibrary.NameOrder.Compare(a.Name, b.Name),
            GlassColumn.Catalog => (a, b) => GlassLibrary.NameOrder.Compare(a.Catalog, b.Catalog),
            _ => (a, b) => SortKey(a, column).CompareTo(SortKey(b, column)),
        };
        // Ties keep the name order.
        list.Sort((a, b) =>
        {
            var c = by(a, b);
            if (!ascending) c = -c;
            return c != 0 ? c : GlassLibrary.NameOrder.Compare(a.Name, b.Name) is var n && n != 0 ? n : GlassLibrary.NameOrder.Compare(a.Catalog, b.Catalog);
        });
        return list;
    }

    /// <summary>A numeric column's sort key: glasses without a value sort last.</summary>
    public static double SortKey(GlassRow r, GlassColumn column) => column switch
    {
        GlassColumn.Nd => r.Nd ?? double.PositiveInfinity,
        GlassColumn.Vd => r.Vd ?? double.PositiveInfinity,
        GlassColumn.Dpgf => r.Dpgf ?? double.PositiveInfinity,
        GlassColumn.Status => StatusRank(r.Status),
        GlassColumn.Density => r.Density ?? double.PositiveInfinity,
        GlassColumn.Cost => r.RelativeCost ?? double.PositiveInfinity,
        GlassColumn.Tce => r.Tce ?? double.PositiveInfinity,
        GlassColumn.Lambda80 => r.Lambda80Um ?? double.PositiveInfinity,
        GlassColumn.Range => r.MaxUm,
        _ => 0,
    };

    /// <summary>Preferred, standard, melt, special, obsolete, then other codes.</summary>
    public static int StatusRank(GlassStatus s) => s.Code switch { 1 => 0, 0 => 1, 4 => 2, 3 => 3, 2 => 4, _ => 5 };

    // ---- Selection and details ----

    string? _selection;
    /// <summary>The selected glass's id (<c>catalog:name</c>).</summary>
    public string? Selection
    {
        get => _selection;
        set
        {
            if (_selection == value) return;
            _selection = value;
            DetailsChanged?.Invoke();
            _ = LoadDetailsAsync();
        }
    }

    public GlassRow? SelectedRow => Rows.FirstOrDefault(r => r.Id == _selection);
    public GlassDetails? Details { get; private set; }

    async Task LoadDetailsAsync()
    {
        if (SelectedRow is not { } row || Engine is not { } engine)
        {
            if (Details is null) return;
            Details = null;
            DetailsChanged?.Invoke();
            return;
        }
        GlassDetails? d;
        try { d = await engine.GlassDetailsAsync(row.Catalog, row.Name); }
        catch (EngineException) { d = null; }
        if (_selection != row.Id) return;
        Details = d;
        DetailsChanged?.Invoke();
    }

    // ---- Using a glass ----

    /// <summary>What "Use for Surface" would do: the surface index, or why not.</summary>
    public (int? Index, string? Why) Assignment =>
        Target is not { } t ? (null, "Open a design to use a glass in it.")
        : t.GlassSurface() is int i ? (i, null)
        : (null, "Select a surface in the lens data to use a glass after it.");

    public void UseSelected()
    {
        if (SelectedRow is not { } row || Target is not { } t || Assignment.Index is not int i) return;
        t.Assign(i, row.Catalog, row.Name);
        Say($"Surface {i} is followed by {row.Name} ({row.Catalog}).");
    }

    // ---- Status ----

    public void SetStatus(GlassStatus? status, GlassRow row) => Library.SetStatus(status, row.Catalog, row.Name);
    public void SetExcluded(bool? excluded, GlassRow row) => Library.SetExcluded(excluded, row.Catalog, row.Name);
    public void ResetStatus(GlassRow row) => Library.ClearOverride(row.Catalog, row.Name);

    // ---- Catalog actions ----

    public void AddToLibrary(IEnumerable<string> paths)
    {
        var added = paths.Where(Library.Add).Select(Path.GetFileName).ToList();
        Say(added.Count == 0 ? "Already in the library." : $"Added {string.Join(", ", added)} to the library.");
    }

    public void RemoveFromLibrary(CatalogInfo info)
    {
        if (info.Path is not { } path || Library.EntryContaining(path) is not { } e)
        {
            Say($"{info.Name} is not a registered library file.");
            return;
        }
        Library.Remove(e);
        Say(e.IsFolder ? $"Removed the folder {e.Title} from the library." : $"Removed {info.Name} from the library.");
    }

    public async Task AddToDesignAsync(string path)
    {
        if (Target is not { } t) return;
        try
        {
            var info = await t.AddCatalog(path);
            Scope = info.Name;
            Say($"Added {info.Name} ({info.GlassCount} glasses) to {t.Title()}.");
        }
        catch (EngineException e) { Say($"{Path.GetFileName(path)} could not be loaded: {e.Message}"); }
    }

    public void RemoveFromDesign(CatalogInfo info)
    {
        if (Target is not { } t) return;
        t.RemoveCatalog(info.Name);
        Say($"Removed {info.Name} from {t.Title()}. Undo in the design window to restore it.");
    }

    /// <summary>Reads a catalog again from its file; returns what changed (null on failure).</summary>
    public async Task<CatalogChanges?> ReloadAsync(CatalogInfo info)
    {
        if (Engine is not { } engine) return null;
        try
        {
            var changes = await engine.ReloadCatalogAsync(info.Name);
            Say($"{info.Name} reloaded: {changes.Summary.ToLowerInvariant()}.");
            if (Target is { } t) await t.CatalogsDidChange();
            // A library catalog is then read again in every open design.
            if (info.Library) Library.Touch();
            else await RefreshAsync();
            return changes;
        }
        catch (EngineException e)
        {
            Say($"{info.Name} could not be reloaded: {e.Message}");
            return null;
        }
    }

    // ---- User glasses ----

    /// <summary>The user catalog new glasses go to: the selected one, else the first.</summary>
    public string? DefaultUserCatalog =>
        _scope is { } n && Library.IsUserCatalog(n) ? Library.UserCatalogNamed(n)!.Name : Library.UserCatalogs.FirstOrDefault()?.Name;

    public string? NewUserCatalog()
    {
        try
        {
            var c = Library.CreateUserCatalog(Library.FreeCatalogName());
            _scope = c.Name;
            return c.Name;
        }
        catch (Exception e) when (e is GlassLibrary.LibraryException or IOException or UnauthorizedAccessException)
        {
            Say(e.Message);
            return null;
        }
    }

    /// <summary>An editor for a new glass (a user catalog is made if there is none).</summary>
    public GlassEditorModel? NewGlass() =>
        (DefaultUserCatalog ?? NewUserCatalog()) is { } cat ? GlassEditorModel.New(cat, Library) : null;

    /// <summary>An editor on a copy of a glass in a user catalog.</summary>
    public async Task<GlassEditorModel?> DuplicateAsync(GlassRow row)
    {
        if (Engine is not { } engine) return null;
        GlassDetails d;
        try { d = row.Id == Details?.Row.Id ? Details : await engine.GlassDetailsAsync(row.Catalog, row.Name); }
        catch (EngineException e) { Say(e.Message); return null; }
        return (DefaultUserCatalog ?? NewUserCatalog()) is { } cat ? GlassEditorModel.Duplicate(d, cat, Library) : null;
    }

    public GlassEditorModel? EditSelected() =>
        Details is { } d && Library.IsUserCatalog(d.Row.Catalog) ? GlassEditorModel.Edit(d, Library) : null;

    public void DeleteSelected()
    {
        if (SelectedRow is not { } row || !Library.IsUserCatalog(row.Catalog)) return;
        try
        {
            Library.DeleteGlass(row.Name, row.Catalog);
            Selection = null;
            Say($"Deleted {row.Name}.");
        }
        catch (Exception e) when (e is GlassLibrary.LibraryException or IOException or UnauthorizedAccessException) { Say(e.Message); }
    }

    /// <summary>Saves the editor's glass through the engine. Returns whether it saved.</summary>
    public bool SaveEditor(GlassEditorModel editor)
    {
        if (Engine is not { } engine || !editor.Save(engine)) return false;
        var name = editor.Name.Trim();
        Say($"Saved {name} in {editor.Catalog}.");
        // Show the saved glass, whatever was searched for.
        _search = "";
        _filter = GlassFilter.All;
        if (_scope is not null) _scope = editor.Catalog;
        _selection = editor.Catalog + ":" + name;
        _visible = null;
        Changed?.Invoke();
        return true;
    }

    /// <summary>A user catalog as AGF text (null on failure).</summary>
    public AgfExport? AgfExport(string catalog)
    {
        if (Library.UserCatalogNamed(catalog) is not { } c || Engine is not { } engine) return null;
        try { return engine.Agf(c.Json); }
        catch (EngineException e) { Say(e.Message); return null; }
    }

    public void Say(string message) => Message?.Invoke(message);
}
