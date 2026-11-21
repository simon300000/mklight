using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// The app-wide glass library (the Mac app's GlassLibrary): the catalog files and folders the
/// user registers once, the user's own glass catalogs, and the availability overrides
/// (preferred, standard, obsolete, excluded from substitution).
/// <para>
/// It lives in <c>%LOCALAPPDATA%\MKLight</c> (<c>Glass Library.json</c> and the
/// <c>User Glasses</c> folder), never in projects. Every design's engine session loads
/// <see cref="CatalogFiles"/> and applies <see cref="Overrides"/> (<see cref="GlassLibrarySync"/>);
/// a design records in its project only the library catalogs it uses (docs/APP.md, "Glass catalogs").
/// </para>
/// <para>
/// User catalogs are the engine's <c>GlassCatalog</c> JSON, one file per catalog. Their glasses
/// come from the engine (<see cref="IEngineClient.GlassFromSpec"/>), which computes n_d and V_d;
/// this type only stores them. Vendor catalog files are never written. Free of WinUI types.
/// </para>
/// </summary>
public sealed class GlassLibrary
{
    /// <summary>A registered catalog file, or a folder whose <c>.agf</c> and <c>.json</c> files are all loaded.</summary>
    public sealed record Entry(string Path, bool IsFolder)
    {
        public string Title => System.IO.Path.GetFileName(Path.TrimEnd('\\', '/'));
    }

    /// <summary>One of the user's own catalogs (<c>User Glasses\&lt;name&gt;.json</c>).</summary>
    public sealed record UserCatalog(string Name, string FilePath, JsonObject Glasses)
    {
        /// <summary>The catalog in the engine's <c>GlassCatalog</c> JSON form.</summary>
        public JsonObject Json => new() { ["name"] = Name, ["glasses"] = Glasses.DeepClone() };
    }

    public sealed class LibraryException(string message) : Exception(message);

    /// <summary>Names sort as people read them ("Glass 2" before "Glass 10"), ignoring case.</summary>
    public static readonly StringComparer NameOrder =
        StringComparer.Create(CultureInfo.InvariantCulture, CompareOptions.IgnoreCase | CompareOptions.NumericOrdering);

    public GlassLibrary(string directory)
    {
        Directory = directory;
        Load();
    }

    public string Directory { get; }
    public string SettingsPath => Path.Combine(Directory, "Glass Library.json");
    public string UserGlassesDirectory => Path.Combine(Directory, "User Glasses");

    readonly List<Entry> _entries = [];
    readonly List<GlassOverride> _overrides = [];
    readonly List<UserCatalog> _userCatalogs = [];
    readonly List<string> _problems = [];

    public IReadOnlyList<Entry> Entries => _entries;
    public IReadOnlyList<GlassOverride> Overrides => _overrides;
    public IReadOnlyList<UserCatalog> UserCatalogs => _userCatalogs;
    /// <summary>Files that could not be read or written.</summary>
    public IReadOnlyList<string> Problems => _problems;

    /// <summary>Counts every change, so open designs load the library again.</summary>
    public int Revision { get; private set; }

    /// <summary>Raised after every change (and <see cref="Touch"/>): open designs sync again.</summary>
    public event Action? Changed;

    /// <summary>Moves a file to the Recycle Bin (the app sets it; tests and the fallback delete the file).</summary>
    public Action<string>? Trash { get; set; }

    // ---- Files ----

    /// <summary>
    /// Every catalog file the designs load, in order: registered files, the <c>.agf</c> and
    /// <c>.json</c> files of registered folders (by name), then the user catalogs.
    /// </summary>
    public IReadOnlyList<string> CatalogFiles
    {
        get
        {
            var output = new List<string>();
            foreach (var e in _entries)
            {
                if (!e.IsFolder) { output.Add(e.Path); continue; }
                try
                {
                    output.AddRange(System.IO.Directory.EnumerateFiles(e.Path).Where(IsCatalogFile)
                        .OrderBy(p => Path.GetFileName(p), NameOrder));
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
            }
            output.AddRange(_userCatalogs.Select(c => c.FilePath));
            return output;
        }
    }

    /// <summary>Whether the app reads a file as a glass catalog (<c>.agf</c> or <c>.json</c>).</summary>
    public static bool IsCatalogFile(string path) =>
        Path.GetExtension(path).ToLowerInvariant() is ".agf" or ".json";

    public bool IsUserCatalog(string name) => UserCatalogNamed(name) is not null;

    public UserCatalog? UserCatalogNamed(string name) =>
        _userCatalogs.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase));

    /// <summary>The entry a catalog loaded from <paramref name="path"/> belongs to (the file itself or its folder).</summary>
    public Entry? EntryContaining(string path)
    {
        var p = Normalize(path);
        return _entries.FirstOrDefault(e => e.IsFolder
            ? string.Equals(Path.GetDirectoryName(p), Normalize(e.Path), StringComparison.OrdinalIgnoreCase)
            : string.Equals(p, Normalize(e.Path), StringComparison.OrdinalIgnoreCase));
    }

    static string Normalize(string path) => Path.GetFullPath(PlainPath(path)).TrimEnd('\\', '/');

    /// <summary>A path without the extended-length prefix the engine may report (<c>\\?\C:\…</c>, <c>\\?\UNC\server\…</c>).</summary>
    public static string PlainPath(string path) =>
        path.StartsWith(@"\\?\UNC\", StringComparison.OrdinalIgnoreCase) ? @"\\" + path[8..]
        : path.StartsWith(@"\\?\", StringComparison.Ordinal) ? path[4..]
        : path;

    // ---- Library entries ----

    /// <summary>Registers a catalog file or a folder of catalogs. Returns whether it was new.</summary>
    public bool Add(string path)
    {
        var full = Normalize(path);
        if (_entries.Any(e => string.Equals(Normalize(e.Path), full, StringComparison.OrdinalIgnoreCase))) return false;
        _entries.Add(new Entry(full, System.IO.Directory.Exists(full)));
        DidChange();
        return true;
    }

    public void Remove(Entry entry)
    {
        if (_entries.RemoveAll(e => e.Path == entry.Path) > 0) DidChange();
    }

    /// <summary>Tells the open designs to load the library again (a catalog file changed on disk and was reloaded).</summary>
    public void Touch()
    {
        Revision++;
        Changed?.Invoke();
    }

    // ---- Availability overrides ----

    public GlassOverride? OverrideFor(string catalog, string name) => _overrides.FirstOrDefault(o => o.Matches(catalog, name));

    /// <summary>Sets a glass's status; <c>null</c> returns it to the catalog's.</summary>
    public void SetStatus(GlassStatus? status, string catalog, string name) =>
        UpdateOverride(catalog, name, o => o with { Status = status });

    /// <summary>Excludes a glass from glass substitution or allows it; <c>null</c> returns it to the catalog's setting.</summary>
    public void SetExcluded(bool? excluded, string catalog, string name) =>
        UpdateOverride(catalog, name, o => o with { ExcludeSubstitution = excluded });

    /// <summary>Returns a glass to the catalog's status and exclusion.</summary>
    public void ClearOverride(string catalog, string name) =>
        UpdateOverride(catalog, name, o => new GlassOverride(o.Catalog, o.Name));

    void UpdateOverride(string catalog, string name, Func<GlassOverride, GlassOverride> change)
    {
        var i = _overrides.FindIndex(o => o.Matches(catalog, name));
        var before = i >= 0 ? _overrides[i] : new GlassOverride(catalog, name);
        var after = change(before);
        if (after == before) return;
        if (i >= 0) _overrides.RemoveAt(i);
        if (!after.IsEmpty) _overrides.Insert(i >= 0 ? i : _overrides.Count, after);
        DidChange();
    }

    // ---- User catalogs ----

    /// <summary>Why <paramref name="name"/> cannot name a new catalog or glass, if it cannot.</summary>
    public static string? NameProblem(string name, bool glass = false)
    {
        var n = name.Trim();
        if (n.Length == 0) return Loc.S("Glass_Name_Empty");
        if (glass && n.Any(char.IsWhiteSpace)) return Loc.S("Glass_Name_Spaces");
        if (n.IndexOfAny(['/', ':', '\\']) >= 0) return Loc.S("Glass_Name_Separators");
        if (!glass && n.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) return Loc.S("Glass_Name_FileCharacter");
        if (!glass && string.Equals(n, "BUILTIN", StringComparison.OrdinalIgnoreCase)) return Loc.S("Glass_Name_Builtin");
        return null;
    }

    /// <summary>A new, empty user catalog.</summary>
    public UserCatalog CreateUserCatalog(string rawName)
    {
        var name = rawName.Trim();
        if (NameProblem(name) is { } why) throw new LibraryException(why);
        if (UserCatalogNamed(name) is not null) throw Duplicate(name);
        var path = Path.Combine(UserGlassesDirectory, name + ".json");
        CheckFileFree(path, null);
        var c = new UserCatalog(name, path, []);
        Write(c);
        _userCatalogs.Add(c);
        SortUserCatalogs();
        DidChange();
        return c;
    }

    /// <summary>A catalog name not yet taken: "User Glasses", "User Glasses 2", …</summary>
    public string FreeCatalogName(string stem = "User Glasses")
    {
        var name = stem;
        for (var k = 2; IsUserCatalog(name); k++) name = $"{stem} {k}";
        return name;
    }

    /// <summary>Renames a user catalog (its file too). Designs that name the catalog in a glass reference keep the old name.</summary>
    public void RenameUserCatalog(string oldName, string rawName)
    {
        var name = rawName.Trim();
        var c = UserCatalogNamed(oldName) ?? throw new LibraryException(Loc.F("Glass_Library_NoUserCatalog", oldName));
        if (name == c.Name) return;
        if (NameProblem(name) is { } why) throw new LibraryException(why);
        if (UserCatalogNamed(name) is { } other && other.FilePath != c.FilePath) throw Duplicate(name);
        var path = Path.Combine(UserGlassesDirectory, name + ".json");
        CheckFileFree(path, c);
        var renamed = c with { Name = name, FilePath = path };
        // A rename that only changes case keeps the file and rewrites it under the new name.
        Write(renamed);
        if (!string.Equals(renamed.FilePath, c.FilePath, StringComparison.OrdinalIgnoreCase))
        {
            try { File.Delete(c.FilePath); } catch (IOException) { }
        }
        _userCatalogs.Remove(c);
        _userCatalogs.Add(renamed);
        // The glasses' availability settings follow the catalog to its new name.
        for (var i = 0; i < _overrides.Count; i++)
            if (string.Equals(_overrides[i].Catalog, c.Name, StringComparison.OrdinalIgnoreCase))
                _overrides[i] = _overrides[i] with { Catalog = renamed.Name };
        SortUserCatalogs();
        DidChange();
    }

    /// <summary>Refuses a file that belongs to a catalog other than <paramref name="owner"/> (its JSON name differs from the file name).</summary>
    void CheckFileFree(string path, UserCatalog? owner)
    {
        if (_userCatalogs.FirstOrDefault(u => u != owner && string.Equals(u.FilePath, path, StringComparison.OrdinalIgnoreCase)) is { } other)
            throw new LibraryException(Loc.F("Glass_Library_FileTaken", Path.GetFileName(path), other.Name));
    }

    /// <summary>Moves a user catalog's file to the Recycle Bin (deletes it when that is not possible).</summary>
    public void DeleteUserCatalog(string name)
    {
        var c = UserCatalogNamed(name) ?? throw new LibraryException(Loc.F("Glass_Library_NoUserCatalog", name));
        try { (Trash ?? File.Delete)(c.FilePath); }
        catch (Exception) when (Trash is not null) { File.Delete(c.FilePath); }
        _userCatalogs.Remove(c);
        _overrides.RemoveAll(o => string.Equals(o.Catalog, c.Name, StringComparison.OrdinalIgnoreCase));
        DidChange();
    }

    /// <summary>
    /// Adds or replaces a glass (engine JSON with a <c>name</c>) in a user catalog.
    /// <paramref name="replacing"/> is the glass's previous name when it was renamed in the editor.
    /// </summary>
    public void PutGlass(JsonObject glass, string catalog, string? replacing = null)
    {
        var name = glass["name"].StringOrNull() ?? throw new LibraryException(Loc.S("Glass_Library_NoName"));
        if (NameProblem(name, glass: true) is { } why) throw new LibraryException(why);
        var c = UserCatalogNamed(catalog) ?? throw new LibraryException(Loc.F("Glass_Library_NoUserCatalog", catalog));
        var glasses = (JsonObject)c.Glasses.DeepClone();
        var clash = glasses.Select(kv => kv.Key).FirstOrDefault(k => string.Equals(k, name, StringComparison.OrdinalIgnoreCase));
        if (clash is not null && !string.Equals(clash, replacing ?? "", StringComparison.OrdinalIgnoreCase)) throw Duplicate(name);
        if (replacing is not null) glasses.Remove(replacing);
        if (clash is not null) glasses.Remove(clash);
        glasses[name] = glass.DeepClone();
        Replace(c, c with { Glasses = glasses });
        // A renamed glass takes its availability settings along.
        if (replacing is not null && !string.Equals(replacing, name, StringComparison.OrdinalIgnoreCase))
        {
            var i = _overrides.FindIndex(o => o.Matches(c.Name, replacing));
            if (i >= 0) _overrides[i] = _overrides[i] with { Name = name };
        }
        DidChange();
    }

    public void DeleteGlass(string name, string catalog)
    {
        var c = UserCatalogNamed(catalog) ?? throw new LibraryException(Loc.F("Glass_Library_NoUserCatalog", catalog));
        var glasses = (JsonObject)c.Glasses.DeepClone();
        var key = glasses.Select(kv => kv.Key).FirstOrDefault(k => string.Equals(k, name, StringComparison.OrdinalIgnoreCase))
            ?? throw new LibraryException(Loc.F("Glass_Library_NoGlass", name));
        glasses.Remove(key);
        Replace(c, c with { Glasses = glasses });
        _overrides.RemoveAll(o => o.Matches(c.Name, name));
        DidChange();
    }

    /// <summary>A glass name not yet in <paramref name="catalog"/>: <c>base</c>, <c>base_2</c>, <c>base_3</c>, …</summary>
    public string FreeGlassName(string stem, string catalog)
    {
        var taken = (UserCatalogNamed(catalog)?.Glasses.Select(kv => kv.Key) ?? []).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var name = string.Join('_', stem.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        if (name.Length == 0) name = "GLASS";
        if (!taken.Contains(name)) return name;
        for (var k = 2; ; k++)
            if (!taken.Contains($"{name}_{k}")) return $"{name}_{k}";
    }

    void Replace(UserCatalog before, UserCatalog after)
    {
        Write(after);
        _userCatalogs[_userCatalogs.IndexOf(before)] = after;
    }

    void Write(UserCatalog c)
    {
        System.IO.Directory.CreateDirectory(UserGlassesDirectory);
        WriteAtomically(c.FilePath, c.Json.ToJsonString(Pretty));
    }

    void SortUserCatalogs() => _userCatalogs.Sort((a, b) => NameOrder.Compare(a.Name, b.Name));

    static LibraryException Duplicate(string name) => new(Loc.F("Glass_Library_Duplicate", name));

    // ---- Persistence ----

    static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    void DidChange()
    {
        PersistSettings();
        Revision++;
        Changed?.Invoke();
    }

    void PersistSettings()
    {
        var o = new JsonObject
        {
            ["version"] = 1,
            ["catalogs"] = new JsonArray(_entries.Select(e => (JsonNode)new JsonObject { ["path"] = e.Path, ["folder"] = e.IsFolder }).ToArray()),
            ["overrides"] = new JsonArray(_overrides.Select(x => (JsonNode)x.ToJson()).ToArray()),
        };
        try
        {
            System.IO.Directory.CreateDirectory(Directory);
            WriteAtomically(SettingsPath, o.ToJsonString(Pretty));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            _problems.Add(Loc.F("Glass_Library_SaveFailed", e.Message));
        }
    }

    static void WriteAtomically(string path, string text)
    {
        var temp = path + ".tmp";
        File.WriteAllText(temp, text + "\n");
        File.Move(temp, path, overwrite: true);
    }

    /// <summary>Reads the settings and the user catalogs from disk.</summary>
    public void Load()
    {
        _problems.Clear();
        _entries.Clear();
        _overrides.Clear();
        _userCatalogs.Clear();
        if (File.Exists(SettingsPath))
        {
            try
            {
                var o = JsonNode.Parse(File.ReadAllText(SettingsPath));
                foreach (var e in o.At("catalogs").ArrayOrNull() ?? [])
                    if (e.At("path").StringOrNull() is { Length: > 0 } p) _entries.Add(new Entry(p, e.At("folder").BoolOrNull() ?? false));
                foreach (var x in o.At("overrides").ArrayOrNull() ?? [])
                    if (GlassOverride.FromJson(x) is { Catalog.Length: > 0, Name.Length: > 0, IsEmpty: false } ov) _overrides.Add(ov);
            }
            catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
            {
                _problems.Add(Loc.F("Glass_Library_ReadFailed", Path.GetFileName(SettingsPath)));
            }
        }
        if (System.IO.Directory.Exists(UserGlassesDirectory))
        {
            foreach (var file in System.IO.Directory.EnumerateFiles(UserGlassesDirectory, "*.json"))
            {
                try
                {
                    var v = JsonNode.Parse(File.ReadAllText(file)) as JsonObject ?? throw new JsonException();
                    var name = v["name"].StringOrNull() is { Length: > 0 } n ? n : Path.GetFileNameWithoutExtension(file);
                    _userCatalogs.Add(new UserCatalog(name, file, v["glasses"]?.DeepClone() as JsonObject ?? []));
                }
                catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
                {
                    _problems.Add(Loc.F("Glass_Library_NotCatalog", Path.GetFileName(file)));
                }
            }
            SortUserCatalogs();
        }
        Revision++;
    }
}
