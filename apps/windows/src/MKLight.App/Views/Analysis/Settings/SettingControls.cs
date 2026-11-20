using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

public enum ControlKind
{
    /// <summary>Not shown; the setting keeps its value.</summary>
    Hidden,
    Toggle,
    Number,
    Text,
    /// <summary>One of the schema's <c>enum</c> values.</summary>
    Picker,
    /// <summary><c>{"type": "all" | "index" | "list"}</c> of the system's fields.</summary>
    FieldSelection,
    /// <summary><c>{"type": "all" | "primary" | "index" | "list"}</c> of the system's wavelengths.</summary>
    WavelengthSelection,
    FieldIndex,
    WavelengthIndex,
    SurfaceIndex,
    /// <summary><c>[x, y]</c> of two numbers (with an automatic state when nullable).</summary>
    Pair,
    /// <summary><c>[i, …]</c> of field, wavelength or surface indices (empty: all).</summary>
    IndexList,
    /// <summary><c>[x, …]</c>, comma separated.</summary>
    NumberList,
    /// <summary>An object with known properties (an on/off switch when nullable).</summary>
    Group,
    /// <summary>A tagged union: a picker of the variants, then the chosen variant's rows.</summary>
    Variants,
    /// <summary>The wave analyses' pupil sampling: its grid and rays.</summary>
    Pupil,
    /// <summary>The spot diagram's through-focus planes.</summary>
    ThroughFocus,
    /// <summary>The run-local spectrum (its own control, last in the form).</summary>
    Spectrum,
    /// <summary>Rows of the same scalar columns, edited as a small table in a popup.</summary>
    Table,
    /// <summary>The chief-ray angle's sensor limit, <c>[[radius, angle], …]</c>.</summary>
    SensorLimit,
    /// <summary>The image simulation's source: a built-in pattern.</summary>
    ImageSource,
    /// <summary>The image simulation's optics sampling: PSF window and spacing, pupil, PSF grid.</summary>
    ImageOptics,
}

public enum IndexDomain { Fields, Wavelengths, Surfaces }

/// <summary>A setting's control and what it needs to know.</summary>
/// <param name="Columns">The columns of a <see cref="ControlKind.Table"/>; with <paramref name="RowsAreArrays"/> they are named "0", "1", ….</param>
public readonly record struct SettingControl(ControlKind Kind, bool Integer = false, bool Nullable = false, IndexDomain Domain = IndexDomain.Fields,
    IReadOnlyList<SettingsField>? Columns = null, bool RowsAreArrays = false)
{
    public static readonly SettingControl Hidden = new(ControlKind.Hidden);
}

/// <summary>
/// The control a setting gets, chosen from its schema alone (the Mac app's
/// SettingControls), so every analysis the engine publishes is editable
/// without typed JSON. A few settings keep a dedicated control (the
/// spectrum, the pupil sampling, the sensor limit, the image simulation's
/// source and optics); settings with no sensible control (explicit ghost
/// path lists, operand lists) are hidden and keep their value.
/// </summary>
public static class SettingControls
{
    /// <param name="pattern">The analysis' pupil pattern setting, which decides whether a <c>seed</c> matters.</param>
    public static SettingControl For(SettingsField f, string analysis, bool hasConfigurations = false, string? pattern = null, bool nested = false)
    {
        var types = f.BaseTypes;
        var isSelection = types.Contains("object") && Json.Str(Json.At(f.Default, "type")) is not null && f.Variants.Count == 0;
        var integer = types.Contains("integer");
        if (!nested)
        {
            if (f.Name == "fields" && isSelection) return new(ControlKind.FieldSelection);
            if (f.Name is "wavelengths" or "wavelength" && isSelection) return new(ControlKind.WavelengthSelection);
            if (f.Name == "field" && integer) return new(ControlKind.FieldIndex, Nullable: f.Nullable);
            if (f.Name is "wavelength" or "long_wavelength" or "short_wavelength" && integer) return new(ControlKind.WavelengthIndex, Nullable: f.Nullable);
            if (f.Name == "surface" && integer) return new(ControlKind.SurfaceIndex, Nullable: f.Nullable);
            if (f.Name == "spectrum" && types.Contains("object")) return new(ControlKind.Spectrum);
            if (f.Name == "pupil" && types.Contains("object") && Json.Has(f.Default, "grid")) return new(ControlKind.Pupil);
            if (f.Name == "through_focus" && analysis == "spot_diagram" && Only(types, "object")) return new(ControlKind.ThroughFocus);
            if (f.Name == "sensor_limit" && Only(types, "array")) return new(ControlKind.SensorLimit);
            if (f.Name == "source" && Only(types, "object") && Json.Str(Json.At(f.Default, "kind")) == "pattern") return new(ControlKind.ImageSource);
            if (f.Name == "optics" && Only(types, "object") && Json.Has(f.Default, "psf_size")) return new(ControlKind.ImageOptics);
            switch (f.Name)
            {
                case "configuration": return hasConfigurations ? new(ControlKind.Number, Integer: true) : SettingControl.Hidden;
                case "delay_ms" or "chunk": return SettingControl.Hidden;
                case "seed" when pattern is not null && pattern is not ("random" or "dithered"):
                    return SettingControl.Hidden;
            }
        }
        if (f.Enum.Count > 0) return new(ControlKind.Picker, Nullable: f.Nullable);
        if (f.Variants.Count > 0) return new(ControlKind.Variants, Nullable: f.Nullable);
        if (Only(types, "boolean")) return new(ControlKind.Toggle);
        if (types.Count > 0 && types.All(t => t is "number" or "integer")) return new(ControlKind.Number, Integer: f.IsInteger, Nullable: f.Nullable);
        if (Only(types, "string")) return new(ControlKind.Text, Nullable: f.Nullable);
        if (Only(types, "array")) return ArrayControl(f);
        if (Only(types, "object")) return f.Properties.Count == 0 ? SettingControl.Hidden : new(ControlKind.Group, Nullable: f.Nullable);
        return SettingControl.Hidden;
    }

    /// <summary>The control of an array setting, from its items and default.</summary>
    static SettingControl ArrayControl(SettingsField f)
    {
        var d = f.Default as JsonArray;
        var itemType = f.ItemType;
        // Two numbers: a pair ([x, y]), with an automatic state when null is allowed.
        if (d is { Count: 2 } && d.All(v => Json.Number(v) is not null))
        {
            var integer = itemType == "integer" || d.All(v => SettingsField.TypeName(v) == "integer");
            return new(ControlKind.Pair, Integer: integer, Nullable: f.Nullable);
        }
        // [min, max] or [x, y] ranges that are automatic by default.
        if ((d is null || d.Count == 0) && f.Nullable && (itemType == "number" || f.Items is null))
            return new(ControlKind.Pair, Nullable: true);
        if (itemType == "integer")
            return IndexDomainOf(f.Name) is IndexDomain dom ? new(ControlKind.IndexList, Domain: dom) : new(ControlKind.NumberList, Integer: true);
        if (itemType == "number") return new(ControlKind.NumberList, Nullable: f.Nullable);
        // Rows of scalars: objects with scalar properties, or fixed-length numeric arrays.
        if (Json.Obj(f.Items) is JsonObject items && new SettingsField(f.Name, items).Properties is { Count: > 0 } props
            && props.All(p => p.BaseTypes.Count > 0 && p.BaseTypes.All(t => t is "number" or "integer" or "boolean" or "string")))
            return Table(props, false);
        if (d is { Count: > 0 })
        {
            if (d[0] is JsonObject o && o.All(p => Json.Number(p.Value) is not null || Json.Bool(p.Value) is not null || Json.Str(p.Value) is not null))
                return Table(o.OrderBy(p => p.Key, StringComparer.Ordinal)
                    .Select(p => new SettingsField(p.Key, new JsonObject { ["type"] = SettingsField.TypeName(p.Value), ["default"] = p.Value?.DeepClone() })).ToList(), false);
            if (d[0] is JsonArray row && row.Count > 0 && row.All(v => Json.Number(v) is not null))
                return Table(row.Select((v, k) => new SettingsField(k.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    new JsonObject { ["type"] = "number", ["default"] = v?.DeepClone() })).ToList(), true);
            return SettingControl.Hidden;
        }
        // Empty by default with items undocumented: the description names the row keys in the engine's [{"hx", "hy"}] form.
        if (TableSetting.RowKeys(f.Description) is { } keys)
            return Table(keys.Select(k => new SettingsField(k, new JsonObject { ["type"] = "number", ["default"] = 0.0 })).ToList(), false);
        return SettingControl.Hidden;
    }

    static SettingControl Table(IReadOnlyList<SettingsField> columns, bool rowsAreArrays) =>
        new(ControlKind.Table, Columns: columns, RowsAreArrays: rowsAreArrays);

    static bool Only(IReadOnlySet<string> types, string type) => types.Count == 1 && types.Contains(type);

    public static IndexDomain? IndexDomainOf(string name) => name switch
    {
        "fields" => IndexDomain.Fields,
        "wavelengths" => IndexDomain.Wavelengths,
        "surfaces" or "exclude_surfaces" => IndexDomain.Surfaces,
        _ => null,
    };

    /// <summary>
    /// The value a nullable setting starts from when switched on: its
    /// non-null default, else the first variant, an object of the properties'
    /// defaults, or an empty pair.
    /// </summary>
    public static JsonNode EnabledValue(SettingsField f)
    {
        if (f.Default is not null) return f.Default.DeepClone();
        if (f.Variants.Count > 0) return f.Variants[0].DefaultObject();
        if (f.Properties.Count > 0)
        {
            var o = new JsonObject();
            foreach (var p in f.Properties.Where(p => p.HasDefault)) o[p.Name] = p.Default?.DeepClone();
            return o;
        }
        if (Only(f.BaseTypes, "array")) return new JsonArray(0, 0);
        if (Only(f.BaseTypes, "boolean")) return true;
        if (f.BaseTypes.Contains("string")) return "";
        return 0;
    }

    /// <summary>Switches a tagged union to <paramref name="v"/>, keeping the values of the properties both have.</summary>
    public static JsonObject ChooseVariant(SettingsField.Variant v, JsonNode? current)
    {
        var o = v.DefaultObject();
        if (current is JsonObject cur)
            foreach (var (k, x) in cur)
                if (o.ContainsKey(k) && v.Properties.FirstOrDefault(p => p.Name == k) is { HasConst: false })
                    o[k] = x?.DeepClone();
        return o;
    }
}
