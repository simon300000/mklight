using MKLight.App.Support;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Windows;

/// <summary>
/// Where designs autosave and which autosaves a launch offers to recover. A saved design autosaves
/// next to its file (<c>&lt;file&gt;.autosave</c>); an untitled one to
/// <c>&lt;support&gt;\Autosave\&lt;guid&gt;.mklight</c>. An autosave is an ordinary project file. Free of
/// WinUI types.
/// </summary>
public static class Autosaves
{
    /// <summary>How often a design with unsaved changes is written to its autosave.</summary>
    public static readonly TimeSpan Interval = TimeSpan.FromMinutes(2);

    public const string Suffix = ".autosave";

    /// <summary>The folder of untitled designs' autosaves.</summary>
    public static string Folder(string supportDirectory) => Path.Combine(supportDirectory, "Autosave");

    /// <summary>The autosave of a design: next to its file, or in <paramref name="supportDirectory"/> by <paramref name="id"/>.</summary>
    public static string PathFor(string? documentPath, Guid id, string supportDirectory) =>
        documentPath is not null ? documentPath + Suffix : Path.Combine(Folder(supportDirectory), id.ToString("N") + ".mklight");

    /// <summary>An autosave found at launch: <see cref="Original"/> is the design's file, null for an untitled design.</summary>
    public sealed record Entry(string Path, string? Original, DateTime Modified)
    {
        public string Title => Original is { } o ? System.IO.Path.GetFileName(o) : Loc.S("Shell_Autosave_UntitledDesign");
    }

    /// <summary>
    /// Autosaves to offer: every untitled one in the support folder, and the <c>.autosave</c> beside each of
    /// <paramref name="files"/> (the recent designs), newest first.
    /// </summary>
    public static IReadOnlyList<Entry> Find(string supportDirectory, IEnumerable<string> files)
    {
        var found = new List<Entry>();
        try
        {
            var folder = Folder(supportDirectory);
            if (Directory.Exists(folder))
                found.AddRange(Directory.EnumerateFiles(folder, "*.mklight").Where(p => !IsHeld(p)).Select(p => new Entry(p, null, File.GetLastWriteTime(p))));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
        foreach (var f in files.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var p = f + Suffix;
            if (File.Exists(p) && !IsHeld(p)) found.Add(new Entry(p, f, File.GetLastWriteTime(p)));
        }
        return found.OrderByDescending(e => e.Modified).ToList();
    }

    /// <summary>
    /// Writes <paramref name="project"/> to <paramref name="path"/> through the engine (ABI 1.7: every section and
    /// catalog reference, missing ones included; a complete file or none). <paramref name="documentPath"/>: the
    /// design's file, whose folder the catalogs' relative paths are written for (none for an untitled design).
    /// </summary>
    public static void Write(FFIEngine engine, ProjectFile project, string path, string? documentPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var doc = project.WithRelativeCatalogPaths(documentPath is null ? null : Path.GetDirectoryName(Path.GetFullPath(documentPath)));
        engine.AutosaveProject(doc.Encoded(), path);
    }

    /// <summary>An autosave read back through the engine: the project and the catalogs it could not load.</summary>
    public static (ProjectFile Project, IReadOnlyList<string> Warnings) Read(FFIEngine engine, string path)
    {
        var (json, warnings) = engine.OpenProject(path);
        return (ProjectFile.Decode(json), warnings);
    }

    // ---- Ownership: an autosave a running instance still writes is never offered to another launch ----

    static readonly Dictionary<string, FileStream> Held = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Marks <paramref name="path"/> as this process's: an exclusive, delete-on-close lock file beside it, released
    /// when the window lets it go or the process ends (even when it crashes).
    /// </summary>
    public static void Hold(string path)
    {
        lock (Held)
        {
            if (Held.ContainsKey(path)) return;
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                Held[path] = new FileStream(path + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
        }
    }

    public static void Release(string path)
    {
        lock (Held)
        {
            if (Held.Remove(path, out var lockFile)) lockFile.Dispose();
        }
    }

    /// <summary>Whether a running instance (this one included) holds <paramref name="path"/>.</summary>
    public static bool IsHeld(string path)
    {
        lock (Held) if (Held.ContainsKey(path)) return true;
        var lockPath = path + ".lock";
        if (!File.Exists(lockPath)) return false;
        try
        {
            using (new FileStream(lockPath, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { }
            File.Delete(lockPath); // left by an instance that is gone
            return false;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return true; }
    }

    public static void Delete(string path)
    {
        Release(path);
        try { if (File.Exists(path)) File.Delete(path); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }

    /// <summary>
    /// The project a recovered window starts from, before the recovered state is applied as one undoable
    /// step (so the design shows as changed): the saved file for an autosave beside a file (undo goes back
    /// to it), the autosaved system alone for an untitled design.
    /// </summary>
    public static ProjectFile RecoveryBase(Entry entry, ProjectFile recovered)
    {
        if (entry.Original is { } o && File.Exists(o))
        {
            try { return ProjectFile.Read(o); }
            catch (Exception) { /* an unreadable file: start from the autosave's system */ }
        }
        return new ProjectFile(recovered.System);
    }
}

/// <summary>Which files open as designs, and the checks of File ▸ Rename…. Free of WinUI types.</summary>
public static class DocumentFiles
{
    /// <summary>Projects (<c>.mklight</c>, <c>.json</c>) and lens prescription files (<c>.zmx</c>, <c>.seq</c>): what a drop opens.</summary>
    public static bool CanOpen(string path) =>
        Path.GetExtension(path).ToLowerInvariant() is ".mklight" or ".json" or ".zmx" or ".seq";

    /// <summary>The path a design at <paramref name="current"/> moves to when renamed <paramref name="name"/> (its extension kept).</summary>
    public static string RenamedPath(string current, string name) =>
        Path.Combine(Path.GetDirectoryName(current) ?? "", name.Trim() + Path.GetExtension(current));

    /// <summary>Why <paramref name="name"/> cannot be the new name of the design at <paramref name="current"/>, or null.</summary>
    public static string? RenameProblem(string current, string name)
    {
        var n = name.Trim();
        if (n.Length == 0) return Loc.S("Shell_Rename_EnterName");
        if (n.IndexOfAny(['<', '>', ':', '"', '/', '\\', '|', '?', '*']) >= 0 || n.Any(char.IsControl))
            return Loc.S("Shell_Rename_BadCharacters");
        if (n.EndsWith('.')) return Loc.S("Shell_Rename_EndsWithPeriod");
        var target = RenamedPath(current, n);
        // A change of case only renames the same file.
        if (!string.Equals(target, current, StringComparison.OrdinalIgnoreCase) && (File.Exists(target) || Directory.Exists(target)))
            return Loc.F("Shell_Rename_Exists", Path.GetFileName(target));
        return null;
    }
}
