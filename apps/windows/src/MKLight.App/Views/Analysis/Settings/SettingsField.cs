using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// One setting of an analysis' settings schema (docs/FFI.md: a JSON-Schema
/// subset: <c>type</c>, <c>default</c>, bounds, <c>enum</c>, <c>items</c>,
/// <c>properties</c>, <c>oneOf</c> with <c>const</c> tags, <c>title</c>,
/// <c>description</c>, <c>unit</c>).
/// </summary>
public sealed class SettingsField
{
    public string Name { get; }
    /// <summary>Accepted JSON types; empty means any.</summary>
    public IReadOnlyList<string> Types { get; }
    public string Title { get; }
    public string Description { get; }
    public string Unit { get; }
    /// <summary>The schema's default (a C# null is JSON null or none: see <see cref="HasDefault"/>).</summary>
    public JsonNode? Default { get; }
    public bool HasDefault { get; }
    public double? Minimum { get; }
    public double? Maximum { get; }
    /// <summary>The bound itself is not allowed (<c>exclusiveMinimum</c>, <c>exclusiveMaximum</c>).</summary>
    public bool MinimumExclusive { get; }
    public bool MaximumExclusive { get; }
    public IReadOnlyList<JsonNode?> Enum { get; }
    public bool Required { get; }
    public JsonNode? Items { get; }
    /// <summary>A fixed value (<c>const</c>): the tag of a variant.</summary>
    public JsonNode? Const { get; }
    public bool HasConst { get; }
    public IReadOnlyList<Variant> Variants { get; }
    /// <summary>The fields of an object setting: the schema's <c>properties</c>, or the keys of its default object.</summary>
    public IReadOnlyList<SettingsField> Properties { get; }

    /// <summary>One alternative of a tagged union: its title and properties, one of which is the tag.</summary>
    public sealed record Variant(string Title, IReadOnlyList<SettingsField> Properties)
    {
        public (string Key, JsonNode? Value)? Tag =>
            Properties.FirstOrDefault(p => p.HasConst) is SettingsField t ? (t.Name, t.Const) : null;

        /// <summary>The variant's object with every property at its default.</summary>
        public JsonObject DefaultObject()
        {
            var o = new JsonObject();
            foreach (var p in Properties)
            {
                if (p.HasConst) o[p.Name] = p.Const?.DeepClone();
                else if (p.HasDefault) o[p.Name] = p.Default?.DeepClone();
            }
            return o;
        }
    }

    public SettingsField(string name, JsonObject schema, bool required = false)
    {
        Name = name;
        Required = required;
        Types = schema["type"] switch
        {
            JsonArray a => a.Select(Json.Str).Where(s => s is not null).Select(s => s!).ToList(),
            JsonNode t when Json.Str(t) is string s => [s],
            _ => [],
        };
        Title = Json.Str(schema["title"]) ?? name;
        Description = Json.Str(schema["description"]) ?? "";
        Unit = Json.Str(schema["unit"]) ?? "";
        HasDefault = schema.TryGetPropertyValue("default", out var d);
        Default = d;
        Minimum = Json.Number(schema["minimum"]);
        Maximum = Json.Number(schema["maximum"]);
        if (Minimum is null && Json.Number(schema["exclusiveMinimum"]) is double xmin) { Minimum = xmin; MinimumExclusive = true; }
        if (Maximum is null && Json.Number(schema["exclusiveMaximum"]) is double xmax) { Maximum = xmax; MaximumExclusive = true; }
        Enum = schema["enum"] is JsonArray e ? e.ToList() : new List<JsonNode?>();
        Items = schema["items"];
        HasConst = schema.TryGetPropertyValue("const", out var c);
        Const = c;
        Variants = (Json.Arr(schema["oneOf"]) ?? []).Select(alt =>
        {
            if (Json.Obj(Json.At(alt, "properties")) is not JsonObject props) return null;
            var fields = Fields(props, new HashSet<string>());
            return fields.Any(f => f.HasConst) ? new Variant(Json.Str(Json.At(alt, "title")) ?? "", fields) : null;
        }).Where(v => v is not null).Select(v => v!).ToList();
        if (schema["properties"] is JsonObject properties)
        {
            var req = (Json.Arr(schema["required"]) ?? []).Select(Json.Str).Where(s => s is not null).Select(s => s!).ToHashSet();
            Properties = Fields(properties, req);
        }
        else if (Variants.Count == 0 && Default is JsonObject dob)
        {
            // Only the default object says what the keys are: each keeps the
            // type of its default (a null default has no type).
            Properties = dob.OrderBy(p => p.Key, StringComparer.Ordinal)
                .Select(p => new SettingsField(p.Key, new JsonObject { ["type"] = TypeName(p.Value), ["default"] = p.Value?.DeepClone() }))
                .ToList();
        }
        else
        {
            Properties = [];
        }
    }

    /// <summary>Properties in a stable order: required first, then by name.</summary>
    public static List<SettingsField> Fields(JsonObject props, ISet<string> required) =>
        props.Where(p => p.Value is JsonObject).Select(p => new SettingsField(p.Key, (JsonObject)p.Value!, required.Contains(p.Key)))
            .OrderBy(f => f.Required ? 0 : 1).ThenBy(f => f.Name, StringComparer.Ordinal).ToList();

    /// <summary>The top-level settings of an analysis descriptor (its <c>settings_schema.properties</c>).</summary>
    public static List<SettingsField> Of(JsonObject descriptor)
    {
        var schema = Json.Obj(descriptor["settings_schema"]);
        if (Json.Obj(schema?["properties"]) is not JsonObject props) return [];
        var req = (Json.Arr(schema!["required"]) ?? []).Select(Json.Str).Where(s => s is not null).Select(s => s!).ToHashSet();
        return Fields(props, req);
    }

    public static string TypeName(JsonNode? v) => v switch
    {
        null => "null",
        JsonArray => "array",
        JsonObject => "object",
        JsonValue x => x.GetValueKind() switch
        {
            JsonValueKind.True or JsonValueKind.False => "boolean",
            JsonValueKind.String => "string",
            JsonValueKind.Number => Json.Int(x) is not null && !x.ToJsonString().Contains('.') && !x.ToJsonString().Contains('e', StringComparison.OrdinalIgnoreCase) ? "integer" : "number",
            _ => "null",
        },
        _ => "null",
    };

    /// <summary>Whether <c>null</c> is accepted.</summary>
    public bool Nullable => Types.Contains("null");
    /// <summary>The accepted types other than <c>null</c>.</summary>
    public IReadOnlySet<string> BaseTypes => Types.Where(t => t != "null").ToHashSet();
    /// <summary>A whole number only (integer, not number).</summary>
    public bool IsInteger => Types.Contains("integer") && !Types.Contains("number");
    public string? ItemType => Json.Str(Json.At(Items, "type"));

    /// <summary>The variant of a tagged-union value, by its tag.</summary>
    public Variant? VariantOf(JsonNode? value) =>
        value is null ? null : Variants.FirstOrDefault(v => v.Tag is { } t && Json.Same(Json.At(value, t.Key), t.Value));
}
