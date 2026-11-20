using System.Text.Json.Nodes;

namespace MKLight.App.ViewModels;

/// <summary>
/// The lens grid's column widths the user chose (dragged a header divider, or fitted a column to its
/// content): those columns keep their width, the others share the rest (<see cref="LensGridLayout"/>).
/// <para>
/// Saved as window state in the project's <c>views</c> section, which is an array of entries
/// (<see cref="WorkspaceLayout"/>): under <c>ui.lens_columns</c> of the <c>layout</c> entry, an object of
/// stable column keys to widths in device-independent pixels (<c>{"comment": 180, "radius": 132}</c>). The
/// panel persistence keeps that entry's other <c>ui</c> keys when it writes the arrangement.
/// </para>
/// </summary>
public sealed class LensColumnWidths
{
    /// <summary>The key of the widths in the layout entry's <c>ui</c>.</summary>
    public const string UiKey = "lens_columns";

    readonly Dictionary<LensColumn, double> _fixed = [];

    /// <summary>Widths the user chose, by column (the others follow the available width).</summary>
    public IReadOnlyDictionary<LensColumn, double> Fixed => _fixed;

    public bool IsEmpty => _fixed.Count == 0;

    /// <summary>The columns with the chosen widths held (clamped to each column's limits).</summary>
    public LensGridColumn[] Apply(IReadOnlyList<LensGridColumn> columns) =>
        [.. columns.Select(c => _fixed.TryGetValue(c.Column, out var w) ? c.Fixed(w) : c)];

    /// <summary>
    /// Holds <paramref name="limits"/>' column at <paramref name="width"/> (clamped to its limits, whole pixels).
    /// Returns whether the width changed.
    /// </summary>
    public bool Set(LensGridColumn limits, double width)
    {
        if (!double.IsFinite(width)) return false;
        var w = Math.Round(limits.Clamp(width));
        if (_fixed.TryGetValue(limits.Column, out var old) && old == w) return false;
        _fixed[limits.Column] = w;
        return true;
    }

    /// <summary>The width that shows <paramref name="content"/> (the widest cell, padding included) in full, within the limits.</summary>
    public static double FitWidth(LensGridColumn limits, double content) =>
        Math.Round(limits.Clamp(Math.Ceiling(Math.Max(content, 0))));

    /// <summary>Fits the column to <paramref name="content"/> (see <see cref="FitWidth"/>). Returns whether the width changed.</summary>
    public bool Fit(LensGridColumn limits, double content) => Set(limits, FitWidth(limits, content));

    /// <summary>Lets every column follow the available width again. Returns whether anything changed.</summary>
    public bool Reset()
    {
        if (_fixed.Count == 0) return false;
        _fixed.Clear();
        return true;
    }

    public bool SameAs(LensColumnWidths other) =>
        _fixed.Count == other._fixed.Count && _fixed.All(kv => other._fixed.TryGetValue(kv.Key, out var w) && w == kv.Value);

    // MARK: Keys

    /// <summary>A column's stable key (saved widths, automation ids): never its display title.</summary>
    public static string Key(LensColumn c) => c switch
    {
        LensColumn.Surface => "surface",
        LensColumn.Type => "type",
        LensColumn.Comment => "comment",
        LensColumn.Radius => "radius",
        LensColumn.RadiusX => "radius_x",
        LensColumn.Thickness => "thickness",
        LensColumn.Material => "material",
        LensColumn.SemiDiameter => "semi_diameter",
        LensColumn.Conic => "conic",
        _ => "conic_x",
    };

    public static LensColumn? FromKey(string key) =>
        Enum.GetValues<LensColumn>().Cast<LensColumn?>().FirstOrDefault(c => Key(c!.Value) == key);

    // MARK: JSON

    /// <summary>The widths as <c>{key: width}</c>, in column order.</summary>
    public JsonObject ToJson()
    {
        var o = new JsonObject();
        foreach (var (c, w) in _fixed.OrderBy(kv => kv.Key)) o[Key(c)] = w;
        return o;
    }

    /// <summary>Reads <c>{key: width}</c>; unknown keys and widths that are not positive finite numbers are left out.</summary>
    public static LensColumnWidths FromJson(JsonNode? node)
    {
        var widths = new LensColumnWidths();
        if (node is not JsonObject o) return widths;
        foreach (var (key, value) in o)
        {
            if (FromKey(key) is not { } c || Number(value) is not { } w || !double.IsFinite(w) || w <= 0) continue;
            widths._fixed[c] = w;
        }
        return widths;
    }

    /// <summary>The widths saved in a project's <c>views</c> (empty when there are none).</summary>
    public static LensColumnWidths Read(JsonNode? views) => FromJson(LayoutUi(views)?[UiKey]);

    /// <summary>
    /// <paramref name="views"/> with these widths in the layout entry (every other entry and key kept; no widths
    /// removes the key), or null when the views have no layout entry yet: one written alone would read as a saved
    /// layout without panes, so the caller waits until the panel persistence has written it.
    /// </summary>
    public JsonArray? WriteTo(JsonNode? views)
    {
        if (views is not JsonArray array || LayoutIndex(array) is not { } i) return null;
        var copy = (JsonArray)array.DeepClone();
        var entry = (JsonObject)copy[i]!;
        if (entry["ui"] is not JsonObject ui)
        {
            ui = [];
            entry["ui"] = ui;
        }
        if (IsEmpty) ui.Remove(UiKey);
        else ui[UiKey] = ToJson();
        return copy;
    }

    static int? LayoutIndex(JsonArray views)
    {
        for (var i = 0; i < views.Count; i++)
            if (views[i] is JsonObject o && o["key"] is JsonValue v && v.TryGetValue<string>(out var k) && k == "layout") return i;
        return null;
    }

    static JsonObject? LayoutUi(JsonNode? views) =>
        views is JsonArray a && LayoutIndex(a) is { } i ? a[i]!["ui"] as JsonObject : null;

    static double? Number(JsonNode? n)
    {
        if (n is not JsonValue v) return null;
        if (v.TryGetValue<double>(out var d)) return d;
        if (v.TryGetValue<int>(out var k)) return k;
        return v.TryGetValue<long>(out var l) ? l : null;
    }
}
