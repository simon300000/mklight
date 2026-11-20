using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>
/// One analysis the engine offers (<c>mk_engine_list_analyses</c>): its name,
/// availability and settings schema. Unknown keys of the engine's JSON are
/// ignored, so newer engines stay readable.
/// </summary>
public sealed class AnalysisDescriptor
{
    public string Name { get; }
    public string Title { get; }
    /// <summary>"geometric", "wave", "model", "optimization", …</summary>
    public string Category { get; }
    public string Description { get; }
    public bool Available { get; }
    public string? UnavailableReason { get; }
    /// <summary>
    /// JSON-Schema subset (docs/FFI.md): <c>type</c>, <c>properties</c>, <c>required</c>,
    /// <c>default</c>, <c>minimum</c>, <c>maximum</c>, <c>enum</c>, <c>items</c>, <c>title</c>,
    /// <c>description</c>, <c>unit</c>, <c>const</c>, <c>oneOf</c>.
    /// </summary>
    public JsonObject SettingsSchema { get; }
    /// <summary>Complete default settings; <c>null</c> when some setting is required.</summary>
    public JsonNode? DefaultSettings { get; }
    /// <summary>A running job of the analysis can be stopped early, keeping its best result so far (<see cref="AnalysisJob.Stop"/>).</summary>
    public bool Stoppable { get; }
    /// <summary>The analysis has a GPU implementation (<c>gpu_path</c>).</summary>
    public bool GpuPath { get; }
    /// <summary>The schema's properties, in a stable order (required first, then by name).</summary>
    public IReadOnlyList<SettingsField> SettingsFields { get; }

    public AnalysisDescriptor(
        string name, string title, string category, string description = "", bool available = true,
        string? unavailableReason = null, JsonObject? settingsSchema = null, JsonNode? defaultSettings = null,
        bool stoppable = false, bool gpuPath = false)
    {
        Name = name;
        Title = title;
        Category = category;
        Description = description;
        Available = available;
        UnavailableReason = unavailableReason;
        SettingsSchema = settingsSchema ?? [];
        DefaultSettings = defaultSettings;
        Stoppable = stoppable;
        GpuPath = gpuPath;
        SettingsFields = FieldsOf(SettingsSchema);
    }

    /// <summary>Reads one element of the engine's analysis list; missing keys take defaults.</summary>
    public static AnalysisDescriptor FromJson(JsonNode? json)
    {
        if (json.At("name").StringOrNull() is not { } name)
        {
            throw EngineException.Engine(EngineErrorCode.Parse, "unexpected engine JSON for an analysis: no name");
        }
        return new AnalysisDescriptor(
            name,
            json.At("title").StringOrNull() ?? name,
            json.At("category").StringOrNull() ?? "",
            json.At("description").StringOrNull() ?? "",
            json.At("available").BoolOrNull() ?? true,
            json.At("unavailable_reason").StringOrNull(),
            Json.Clone(json.At("settings_schema")) as JsonObject,
            Json.Clone(json.At("default_settings")),
            json.At("stoppable").BoolOrNull() ?? false,
            json.At("gpu_path").BoolOrNull() ?? false);
    }

    /// <summary>The engine's list (<c>mk_engine_list_analyses</c>).</summary>
    public static IReadOnlyList<AnalysisDescriptor> ListFromJson(string text) =>
        (Json.Parse(text, "analysis list").ArrayOrNull() ?? []).Select(FromJson).ToArray();

    public JsonObject ToJson() => new()
    {
        ["name"] = Name,
        ["title"] = Title,
        ["category"] = Category,
        ["description"] = Description,
        ["available"] = Available,
        ["unavailable_reason"] = UnavailableReason,
        ["settings_schema"] = SettingsSchema.DeepClone(),
        ["default_settings"] = Json.Clone(DefaultSettings),
        ["stoppable"] = Stoppable,
        ["gpu_path"] = GpuPath,
    };

    static IReadOnlyList<SettingsField> FieldsOf(JsonObject schema)
    {
        if (schema["properties"] is not JsonObject props) return [];
        var required = (schema["required"].ArrayOrNull() ?? []).Select(r => r.StringOrNull()).OfType<string>().ToHashSet();
        return SettingsField.Fields(props, required);
    }

    /// <summary>
    /// <paramref name="settings"/> reduced to the keys the schema declares, with values
    /// of an accepted type. The engine rejects unknown keys, so keys this schema does not
    /// declare are dropped and the engine's defaults apply instead.
    /// </summary>
    public Dictionary<string, JsonNode?> AcceptedSettings(IReadOnlyDictionary<string, JsonNode?> settings) =>
        settings.Where(kv => SettingsFields.FirstOrDefault(f => f.Name == kv.Key)?.Accepts(kv.Value) ?? false)
            .ToDictionary(kv => kv.Key, kv => Json.Clone(kv.Value));

    public override string ToString() => Name;
}

/// <summary>One property of an analysis settings schema.</summary>
public sealed class SettingsField
{
    public string Name { get; }
    /// <summary>JSON types accepted (<c>number</c>, <c>integer</c>, <c>string</c>, <c>boolean</c>, <c>array</c>, <c>object</c>, <c>null</c>); empty means any.</summary>
    public IReadOnlyList<string> Types { get; }
    public string Title { get; }
    public string Description { get; }
    public string Unit { get; }
    /// <summary>Whether the schema has a <c>default</c> (which may be JSON null).</summary>
    public bool HasDefault { get; }
    public JsonNode? DefaultValue { get; }
    public double? Minimum { get; }
    public double? Maximum { get; }
    public IReadOnlyList<JsonNode?> EnumValues { get; }
    public bool Required { get; }
    /// <summary>The schema of an array's elements (<c>items</c>), if given.</summary>
    public JsonNode? Items { get; }
    /// <summary>Whether the schema has a fixed value (<c>const</c>): the tag of a variant.</summary>
    public bool HasConst { get; }
    public JsonNode? ConstValue { get; }
    /// <summary>The alternatives of a tagged union (<c>oneOf</c> with <c>const</c> tags).</summary>
    public IReadOnlyList<SettingsVariant> Variants { get; }
    /// <summary>
    /// The fields of an object setting: the schema's <c>properties</c>, or, without any,
    /// inferred from the keys of the default object (each with the JSON type of its
    /// default). Empty for other types.
    /// </summary>
    public IReadOnlyList<SettingsField> Properties { get; }

    public SettingsField(string name, JsonObject schema, bool required)
    {
        Name = name;
        Required = required;
        Types = schema["type"] switch
        {
            JsonArray a => a.Select(t => t.StringOrNull()).OfType<string>().ToArray(),
            var t when t.StringOrNull() is { } s => [s],
            _ => [],
        };
        Title = schema["title"].StringOrNull() ?? name;
        Description = schema["description"].StringOrNull() ?? "";
        Unit = schema["unit"].StringOrNull() ?? "";
        HasDefault = schema.TryGetPropertyValue("default", out var d);
        DefaultValue = Json.Clone(d);
        Minimum = schema["minimum"].DoubleValue();
        Maximum = schema["maximum"].DoubleValue();
        EnumValues = schema["enum"] is JsonArray e ? e.Select(Json.Clone).ToArray() : [];
        Items = Json.Clone(schema["items"]);
        HasConst = schema.TryGetPropertyValue("const", out var c);
        ConstValue = Json.Clone(c);
        Variants = (schema["oneOf"].ArrayOrNull() ?? []).Select(alt =>
        {
            if (alt.At("properties") is not JsonObject props) return null;
            var fields = Fields(props, new HashSet<string>());
            return fields.Any(f => f.HasConst) ? new SettingsVariant(alt.At("title").StringOrNull() ?? "", fields) : null;
        }).OfType<SettingsVariant>().ToArray();
        if (schema["properties"] is JsonObject properties)
        {
            var req = (schema["required"].ArrayOrNull() ?? []).Select(r => r.StringOrNull()).OfType<string>().ToHashSet();
            Properties = Fields(properties, req);
        }
        else if (Variants.Count == 0 && DefaultValue is JsonObject defaults)
        {
            // Only the default object says what the keys are: each keeps the
            // type of its default (a null default has no type).
            Properties = defaults
                .Select(kv => new SettingsField(kv.Key, new JsonObject { ["type"] = Json.TypeName(kv.Value), ["default"] = Json.Clone(kv.Value) }, false))
                .OrderBy(f => f.Name, StringComparer.Ordinal)
                .ToArray();
        }
        else
        {
            Properties = [];
        }
    }

    /// <summary>Properties in a stable order: required first, then by name.</summary>
    internal static IReadOnlyList<SettingsField> Fields(JsonObject props, IReadOnlySet<string> required) =>
        props.Select(kv => new SettingsField(kv.Key, kv.Value as JsonObject ?? [], required.Contains(kv.Key)))
            .OrderBy(f => f.Required ? 0 : 1)
            .ThenBy(f => f.Name, StringComparer.Ordinal)
            .ToArray();

    /// <summary>Whether <c>null</c> is an accepted value.</summary>
    public bool Nullable => Types.Contains("null");

    /// <summary>The accepted types other than <c>null</c>.</summary>
    public IReadOnlySet<string> BaseTypes => Types.Where(t => t != "null").ToHashSet();

    /// <summary>The variant of a tagged-union value, by its tag.</summary>
    public SettingsVariant? VariantOf(JsonNode? value) =>
        value is null ? null : Variants.FirstOrDefault(v => v.Tag is { } tag && Json.DeepEquals(value.At(tag.Key), tag.Value));

    /// <summary>Whether <paramref name="value"/> has one of the accepted types (and, for enums, is a listed value).</summary>
    public bool Accepts(JsonNode? value)
    {
        if (EnumValues.Count > 0 && !EnumValues.Any(e => Json.DeepEquals(e, value))) return false;
        if (Types.Count == 0) return true;
        var kind = Json.TypeName(value);
        return Types.Any(t => (t, kind) switch
        {
            ("integer", "integer") => true,
            ("integer", "number") => value.DoubleValue() is { } x && Math.Round(x) == x,
            ("number", "integer" or "number") => true,
            var (a, b) => a == b,
        });
    }

    public override string ToString() => Name;
}

/// <summary>
/// One alternative of a tagged union (<c>oneOf</c>): its title and the properties of
/// its object, one of which is the tag with a <c>const</c>.
/// </summary>
public sealed class SettingsVariant(string title, IReadOnlyList<SettingsField> properties)
{
    public string Title { get; } = title;
    public IReadOnlyList<SettingsField> Properties { get; } = properties;

    /// <summary>The tag property's name and value (<c>"type": "grid"</c>).</summary>
    public (string Key, JsonNode? Value)? Tag =>
        Properties.FirstOrDefault(f => f.HasConst) is { } f ? (f.Name, f.ConstValue) : null;

    /// <summary>The variant's object with every property at its default.</summary>
    public JsonObject DefaultObject()
    {
        var o = new JsonObject();
        foreach (var f in Properties)
        {
            if (f.HasConst) o[f.Name] = Json.Clone(f.ConstValue);
            else if (f.HasDefault) o[f.Name] = Json.Clone(f.DefaultValue);
        }
        return o;
    }

    public override string ToString() => Title;
}

/// <summary>
/// Which compute backend a job should use. A preference: the engine falls back to the
/// CPU reference (and says so in <see cref="RunInfo.Warnings"/>) when a backend is
/// unavailable or cannot meet the error budget for a design (CONVENTIONS §7).
/// </summary>
public enum BackendPreference
{
    /// <summary>The engine chooses.</summary>
    Auto,
    /// <summary>The CPU f64 reference tracer.</summary>
    CpuReference,
    /// <summary>This platform's GPU backend.</summary>
    Gpu,
}

public static class BackendPreferenceExtensions
{
    public static string Title(this BackendPreference b) => b switch
    {
        BackendPreference.CpuReference => "CPU reference",
        BackendPreference.Gpu => "GPU",
        _ => "Automatic",
    };

    /// <summary>The persisted name: <c>auto</c>, <c>cpu_reference</c>, <c>gpu</c>.</summary>
    public static string WireName(this BackendPreference b) => b switch
    {
        BackendPreference.CpuReference => "cpu_reference",
        BackendPreference.Gpu => "gpu",
        _ => "auto",
    };
}

/// <summary>
/// How a job is executed: what a front end chooses of a job's identity besides the
/// snapshot and the analysis settings. Part of the request, so it takes part in the
/// staleness check: another backend preference makes the old result stale.
/// </summary>
public readonly record struct ExecutionSettings(BackendPreference Backend = BackendPreference.Auto);

/// <summary>
/// A request for any engine analysis by name (<c>{"analysis", "settings", "view_key",
/// "backend", "allow_backend_fallback", "fresh"}</c> on the C ABI). <see cref="Settings"/>
/// may be partial: the engine fills in defaults and records the complete settings in
/// <see cref="RunInfo.Settings"/>. Immutable; the settings are deep-copied.
/// </summary>
public sealed class EngineAnalysisRequest : IEquatable<EngineAnalysisRequest>
{
    public string Analysis { get; }
    public IReadOnlyDictionary<string, JsonNode?> Settings { get; }
    /// <summary>The backend preference.</summary>
    public ExecutionSettings Execution { get; }
    /// <summary>
    /// Recompute instead of answering from the engine's stored results and intermediates
    /// (<c>"fresh": true</c>). It says how to run, not what to compute, so it takes no part
    /// in equality: a fresh request equals the plain one.
    /// </summary>
    public bool Fresh { get; }
    /// <summary>
    /// The engine's view key (<c>"view_key"</c>): a newer job with the same key in the same
    /// session cancels the older one. How to run, not what to compute: no part in equality.
    /// </summary>
    public string? ViewKey { get; }

    public EngineAnalysisRequest(
        string analysis, IEnumerable<KeyValuePair<string, JsonNode?>>? settings = null,
        ExecutionSettings execution = default, bool fresh = false, string? viewKey = null)
    {
        Analysis = analysis;
        Settings = (settings ?? []).ToDictionary(kv => kv.Key, kv => Json.Clone(kv.Value));
        Execution = execution;
        Fresh = fresh;
        ViewKey = viewKey;
    }

    /// <summary>This request, recomputed by the engine (<see cref="Fresh"/>).</summary>
    public EngineAnalysisRequest Freshened() => new(Analysis, Settings, Execution, fresh: true, ViewKey);

    /// <summary>This request with one setting replaced.</summary>
    public EngineAnalysisRequest With(string key, JsonNode? value) =>
        new(Analysis, Settings.Where(kv => kv.Key != key).Append(new(key, value)), Execution, Fresh, ViewKey);

    /// <summary>This request with another execution.</summary>
    public EngineAnalysisRequest With(ExecutionSettings execution) => new(Analysis, Settings, execution, Fresh, ViewKey);

    /// <summary>This request under a view key (<see cref="ViewKey"/>).</summary>
    public EngineAnalysisRequest WithViewKey(string? viewKey) => new(Analysis, Settings, Execution, Fresh, viewKey);

    /// <summary>The settings as a JSON object (a copy).</summary>
    public JsonObject SettingsObject() => Json.ObjectOf(Settings);

    /// <summary>Whether <paramref name="info"/> (normalised by the engine) records these settings: each requested key has the requested value.</summary>
    public bool Matches(RunInfo info)
    {
        if (info.Analysis != Analysis) return false;
        var recorded = info.Settings as JsonObject;
        return Settings.All(kv => recorded is not null && recorded.TryGetPropertyValue(kv.Key, out var v) && Json.DeepEquals(v, kv.Value));
    }

    public bool Equals(EngineAnalysisRequest? other) =>
        other is not null && Analysis == other.Analysis && Execution == other.Execution
        && Settings.Count == other.Settings.Count
        && Settings.All(kv => other.Settings.TryGetValue(kv.Key, out var v) && Json.DeepEquals(kv.Value, v));

    public override bool Equals(object? obj) => Equals(obj as EngineAnalysisRequest);

    public override int GetHashCode()
    {
        var h = new HashCode();
        h.Add(Analysis);
        h.Add(Execution);
        foreach (var key in Settings.Keys.Order(StringComparer.Ordinal)) h.Add(key);
        return h.ToHashCode();
    }

    public static bool operator ==(EngineAnalysisRequest? a, EngineAnalysisRequest? b) => a is null ? b is null : a.Equals(b);
    public static bool operator !=(EngineAnalysisRequest? a, EngineAnalysisRequest? b) => !(a == b);

    public override string ToString() => $"{Analysis} {SettingsObject().ToJsonString()}";
}
