using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>A user glass catalog the project depends on (engine <c>CatalogRef</c>). Vendor catalogs are referenced, never embedded.</summary>
[JsonConverter(typeof(ModelConverter<CatalogReference>))]
public sealed record CatalogReference(string Name, string? Path, string? RelativePath = null, string Format = "agf") : IJsonModel<CatalogReference>
{
    /// <summary>
    /// File locations to try, in order (as the engine does): the relative path from the project's
    /// directory, then the absolute path.
    /// </summary>
    public IReadOnlyList<string> Candidates(string? projectDirectory)
    {
        var output = new List<string>();
        if (projectDirectory is not null && RelativePath is { } rel) output.Add(global::System.IO.Path.Combine(projectDirectory, rel));
        if (Path is { } path) output.Add(path);
        return output;
    }

    public static CatalogReference FromJson(JsonNode? node)
    {
        var o = J.Obj(node);
        return new CatalogReference(
            J.Req(o, "name", J.S), J.OptRef(o, "path", J.S), J.OptRef(o, "relative_path", J.S), J.Def(o, "format", J.S, "agf"));
    }

    public JsonNode ToJson() => new JsonObject { ["name"] = Name, ["path"] = Path, ["relative_path"] = RelativePath, ["format"] = Format };
}

/// <summary>Why a document is not a readable project.</summary>
public enum ProjectFileError { UnsupportedVersion, NotAProject }

public sealed class ProjectFileException : Exception
{
    public ProjectFileError Error { get; }

    /// <summary>The schema version found, for <see cref="ProjectFileError.UnsupportedVersion"/>.</summary>
    public int Version { get; }

    private ProjectFileException(ProjectFileError error, int version, string message) : base(message)
    {
        Error = error;
        Version = version;
    }

    public static ProjectFileException UnsupportedVersion(int version) => new(
        ProjectFileError.UnsupportedVersion, version,
        version > ProjectFile.CurrentVersion
            ? $"This project was written by a newer MKLight (schema version {version}); please update MKLight."
            : $"This project uses schema version {version}, which this MKLight does not read (it reads version {ProjectFile.CurrentVersion}).");

    public static ProjectFileException NotAProject(string why) => new(ProjectFileError.NotAProject, 0, $"Not an MKLight project: {why}");
}

/// <summary>
/// An on-disk <c>.mklight</c> project in the engine's format (docs/FFI.md, "Project file"):
/// <c>{"format", "schema_version", "metadata", "system", "catalogs", "views", "merit", "variables", "tolerances", …}</c>.
/// <para>
/// The app edits only <see cref="System"/> (and the catalog list); every other section, including keys a
/// newer application added, is kept verbatim in <see cref="Sections"/> so a round trip loses nothing. The
/// system itself is decoded strictly, as the engine does: an unknown model field is an error, never
/// dropped. Like the engine, only <see cref="CurrentVersion"/> is read: a document without a
/// <c>schema_version</c> or at another version is refused.
/// </para>
/// </summary>
public sealed record ProjectFile
{
    public const string FormatName = "mklight-project";

    /// <summary>The engine's <c>PROJECT_SCHEMA_VERSION</c> this app reads and writes.</summary>
    public const int CurrentVersion = 1;

    public string Format { get; init; } = FormatName;
    public int SchemaVersion { get; init; } = CurrentVersion;
    public OpticalSystem System { get; init; }

    /// <summary>Every other top-level key, verbatim.</summary>
    public ValueMap<JsonTree> Sections { get; init; }

    /// <summary>A new project around <paramref name="system"/>, with the engine's default sections.</summary>
    public ProjectFile(OpticalSystem system)
    {
        System = system;
        Sections = ValueMap<JsonTree>.From(
        [
            KeyValuePair.Create("metadata", new JsonTree(new JsonObject())),
            KeyValuePair.Create("catalogs", new JsonTree(new JsonArray())),
            KeyValuePair.Create("views", new JsonTree(new JsonArray())),
            KeyValuePair.Create("merit", JsonTree.Null),
            KeyValuePair.Create("variables", JsonTree.Null),
            KeyValuePair.Create("tolerances", JsonTree.Null),
        ]);
    }

    /// <summary>User glass catalogs referenced by the project (unreadable entries read as none).</summary>
    public ValueList<CatalogReference> Catalogs
    {
        get
        {
            try
            {
                return Sections.TryGetValue("catalogs", out var v) ? J.Arr(v.Node, CatalogReference.FromJson) : ValueList<CatalogReference>.Empty;
            }
            catch (JsonException)
            {
                return ValueList<CatalogReference>.Empty;
            }
        }
    }

    public ProjectFile WithCatalogs(IEnumerable<CatalogReference> catalogs) =>
        this with { Sections = Sections.Set("catalogs", new JsonTree(J.Arr(catalogs, c => c.ToJson()))) };

    /// <summary>Adds or replaces (by name) a catalog reference.</summary>
    public ProjectFile AddCatalog(CatalogReference reference) =>
        WithCatalogs(Catalogs.Where(c => c.Name != reference.Name).Append(reference));

    /// <summary>
    /// The catalog references with <c>relative_path</c> for a project in <paramref name="directory"/>, as the
    /// engine's save computes it: the path relative to the folder (with <c>/</c>) when the catalog lies under it,
    /// none when it is a usable file elsewhere, and the previous one kept while the absolute path is unusable
    /// (the only location left). So a project folder that moves keeps finding its catalogs.
    /// </summary>
    public ProjectFile WithRelativeCatalogPaths(string? directory)
    {
        if (string.IsNullOrEmpty(directory)) return this;
        var dir = global::System.IO.Path.GetFullPath(directory);
        var changed = false;
        var refs = Catalogs.Select(c =>
        {
            if (c.Path is not { Length: > 0 } abs || !global::System.IO.Path.IsPathFullyQualified(abs)) return c;
            var full = global::System.IO.Path.GetFullPath(abs);
            var rel = global::System.IO.Path.GetRelativePath(dir, full);
            string? relative = rel != full && !rel.StartsWith("..", StringComparison.Ordinal) && !global::System.IO.Path.IsPathRooted(rel)
                ? rel.Replace('\\', '/') : null;
            if (relative is null && !File.Exists(full)) return c;
            if (relative == c.RelativePath) return c;
            changed = true;
            return c with { RelativePath = relative };
        }).ToList();
        return changed ? WithCatalogs(refs) : this;
    }

    /// <summary>Stamps <c>metadata</c> the way the engine does on save.</summary>
    public ProjectFile StampMetadata(string appName, string appVersion, string? engineVersion, DateTimeOffset? now = null)
    {
        var meta = Sections.TryGetValue("metadata", out var m) && m.Node is JsonObject existing
            ? (JsonObject)existing.DeepClone()
            : new JsonObject();
        var stamp = (now ?? DateTimeOffset.UtcNow).UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture);
        if (!(meta["created"] is JsonValue c && c.TryGetValue<string>(out var created) && created.Length > 0)) meta["created"] = stamp;
        meta["modified"] = stamp;
        meta["app_name"] = appName;
        meta["app_version"] = appVersion;
        if (engineVersion is not null) meta["engine_version"] = engineVersion;
        if (!meta.ContainsKey("author")) meta["author"] = "";
        return this with { Sections = Sections.Set("metadata", new JsonTree(meta)) };
    }

    public static ProjectFile Decode(string json) => FromJson(JsonNode.Parse(json));

    public static ProjectFile Decode(ReadOnlySpan<byte> utf8) => FromJson(JsonNode.Parse(utf8));

    public static ProjectFile FromJson(JsonNode? node)
    {
        if (node is not JsonObject doc) throw ProjectFileException.NotAProject("the document is not a JSON object");
        doc.TryGetPropertyValue("schema_version", out var version);
        switch (version)
        {
            case null: throw ProjectFileException.NotAProject("it has no schema_version");
            case JsonValue v when v.GetValueKind() == JsonValueKind.Number:
                var text = v.ToJsonString();
                if (long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var l) && l >= 0)
                {
                    if (l != CurrentVersion) throw ProjectFileException.UnsupportedVersion((int)Math.Min(l, int.MaxValue));
                    break;
                }
                if (ulong.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out _)) throw ProjectFileException.UnsupportedVersion(int.MaxValue);
                throw ProjectFileException.NotAProject("schema_version must be a non-negative integer");
            default: throw ProjectFileException.NotAProject("schema_version must be a non-negative integer");
        }
        if (!doc.TryGetPropertyValue("system", out var systemNode)) throw ProjectFileException.NotAProject("it has no 'system'");
        doc.TryGetPropertyValue("format", out var formatNode);
        string format;
        switch (formatNode)
        {
            case null: format = FormatName; break;
            case JsonValue f when f.GetValueKind() == JsonValueKind.String: format = f.GetValue<string>(); break;
            default: throw ProjectFileException.NotAProject("'format' must be a string");
        }
        if (format != FormatName) throw ProjectFileException.NotAProject($"format '{format}'");
        var file = new ProjectFile(OpticalSystem.FromJson(systemNode));
        var sections = file.Sections;
        foreach (var (k, v) in doc)
        {
            if (k is "format" or "schema_version" or "system") continue;
            sections = sections.Set(k, new JsonTree(v));
        }
        return file with { Sections = sections };
    }

    public JsonNode ToJson()
    {
        var o = new JsonObject();
        foreach (var (k, v) in Sections) o[k] = v.Node?.DeepClone();
        o["format"] = Format;
        o["schema_version"] = SchemaVersion;
        o["system"] = System.ToJson();
        return o;
    }

    /// <summary>The project as pretty-printed JSON text, keys sorted.</summary>
    public string Encoded() => ModelJson.ToJsonString(ToJson(), pretty: true);

    public static ProjectFile Read(string path) => Decode(File.ReadAllBytes(path));

    /// <summary>Writes the project, replacing <paramref name="path"/> only once the new file is complete.</summary>
    public void Write(string path)
    {
        var temp = path + ".tmp";
        File.WriteAllText(temp, Encoded(), new UTF8Encoding(false));
        File.Move(temp, path, overwrite: true);
    }
}
