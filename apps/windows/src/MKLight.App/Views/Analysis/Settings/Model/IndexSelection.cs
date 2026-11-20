using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>What a field, wavelength or surface selection holds.</summary>
public enum SelectionKind
{
    /// <summary>Every one: an empty list, or <c>{"type": "all"}</c>.</summary>
    All,
    /// <summary>The system's primary wavelength, <c>{"type": "primary"}</c>; it follows the design.</summary>
    Primary,
    /// <summary>The checked indices (one or more).</summary>
    Some,
}

/// <summary>How a selection is written in the settings.</summary>
public enum SelectionEncoding
{
    /// <summary><c>[i, …]</c>, empty: all (<c>system_transmission.wavelengths</c>, <c>ghost_irradiance.fields</c>).</summary>
    List,
    /// <summary><c>{"type": "all" | "primary" | "index" | "list"}</c> (most analyses' <c>fields</c> and <c>wavelengths</c>).</summary>
    Tagged,
}

/// <summary>
/// A setting choosing some of the system's fields, wavelengths or surfaces,
/// edited as a menu of check marks: "All" (and "Primary" for a wavelength
/// selection) exclusive, each option toggled on its own; none checked is all,
/// as in the settings themselves. Pure, so the menu's semantics are unit-tested.
/// </summary>
public sealed record IndexSelection(SelectionKind Kind, IReadOnlyList<int> Indices)
{
    public static readonly IndexSelection All = new(SelectionKind.All, []);
    public static readonly IndexSelection Primary = new(SelectionKind.Primary, []);

    /// <summary>The checked indices, sorted and unique; none is all.</summary>
    public static IndexSelection Of(IEnumerable<int> indices)
    {
        var s = indices.Where(i => i >= 0).Distinct().Order().ToList();
        return s.Count == 0 ? All : new(SelectionKind.Some, s);
    }

    // MARK: Reading and writing

    /// <summary>A stored value (null: unset, the schema default).</summary>
    public static IndexSelection Read(JsonNode? v, SelectionEncoding encoding) => encoding switch
    {
        SelectionEncoding.List => Of((Json.Arr(v) ?? []).Select(Json.Int).OfType<int>()),
        _ => Json.Str(Json.At(v, "type")) switch
        {
            "primary" => Primary,
            "index" => Json.Int(Json.At(v, "index")) is int i ? Of([i]) : All,
            "list" => Of((Json.Arr(Json.At(v, "indices")) ?? []).Select(Json.Int).OfType<int>()),
            _ => All,
        },
    };

    /// <summary>The value to store: a sorted list, or the tagged object (one index is <c>"index"</c>, several <c>"list"</c>).</summary>
    public JsonNode Write(SelectionEncoding encoding)
    {
        if (encoding == SelectionEncoding.List)
            return new JsonArray(Kind == SelectionKind.Some ? Indices.Select(i => (JsonNode)i).ToArray() : []);
        return Kind switch
        {
            SelectionKind.Primary => new JsonObject { ["type"] = "primary" },
            SelectionKind.Some when Indices.Count == 1 => new JsonObject { ["type"] = "index", ["index"] = Indices[0] },
            SelectionKind.Some => new JsonObject { ["type"] = "list", ["indices"] = new JsonArray(Indices.Select(i => (JsonNode)i).ToArray()) },
            _ => new JsonObject { ["type"] = "all" },
        };
    }

    // MARK: The menu

    /// <summary>Whether option <paramref name="index"/> has a check mark (none under "All" or "Primary").</summary>
    public bool IsChecked(int index) => Kind == SelectionKind.Some && Indices.Contains(index);

    /// <summary>
    /// The selection after option <paramref name="index"/> is toggled: from
    /// "All" or "Primary" it is that option alone; unchecking the last one is "All".
    /// </summary>
    public IndexSelection Toggle(int index) => Kind != SelectionKind.Some
        ? Of([index])
        : Indices.Contains(index) ? Of(Indices.Where(i => i != index)) : Of([.. Indices, index]);

    /// <summary>
    /// The button's title: "All", "Primary", one option's short title ("2",
    /// "0.5876 µm"), a few short ones ("1, 3"), else "3 of 5". Every option
    /// checked reads "All", as each of them is analysed.
    /// </summary>
    public string Title(IReadOnlyList<(int Index, string Title)> options)
    {
        switch (Kind)
        {
            case SelectionKind.All: return "All";
            case SelectionKind.Primary: return "Primary";
        }
        var known = options.Select(o => o.Index).ToHashSet();
        if (options.Count > 0 && Indices.Count == options.Count && Indices.All(known.Contains)) return "All";
        var titles = Indices.Select(i => ShortTitle(i, options)).ToList();
        if (titles.Count == 1) return titles[0];
        var joined = string.Join(", ", titles);
        if (titles.Count <= 3 && joined.Length <= 12) return joined;
        return options.Count > 0
            ? FormattableString.Invariant($"{Indices.Count} of {options.Count}")
            : FormattableString.Invariant($"{Indices.Count} selected");
    }

    /// <summary>The accessible value: "All 5", "Primary", "1, 3 of 5".</summary>
    public string AccessibleValue(IReadOnlyList<(int Index, string Title)> options) => Kind switch
    {
        SelectionKind.All => options.Count > 0 ? FormattableString.Invariant($"All {options.Count}") : "All",
        SelectionKind.Primary => "Primary",
        _ => string.Join(", ", Indices.Select(i => ShortTitle(i, options)))
            + (options.Count > 0 ? FormattableString.Invariant($" of {options.Count}") : ""),
    };

    /// <summary>The short title of an option ("2" of "2: 14°", "0.5876 µm" of "0.5876 µm (primary)"); its number when unknown.</summary>
    public static string ShortTitle(int index, IReadOnlyList<(int Index, string Title)> options)
    {
        foreach (var (k, t) in options)
            if (k == index) return t.Split(':')[0].Replace(" (primary)", "", StringComparison.Ordinal);
        return (index + 1).ToString(CultureInfo.InvariantCulture);
    }

    // MARK: Text (the design's fields or wavelengths are unknown)

    /// <summary>"1, 3": the checked options' numbers (1 based, as the menus number them); empty for all.</summary>
    public string Text(int offset = 1) => Kind == SelectionKind.Some
        ? string.Join(", ", Indices.Select(i => (i + offset).ToString(CultureInfo.InvariantCulture)))
        : Kind == SelectionKind.Primary ? "primary" : "";

    /// <summary>
    /// Typed numbers ("1, 3", "1 3"; 1 based); empty or "all" is all, "primary"
    /// the primary wavelength where allowed. Null when the text is not a selection.
    /// </summary>
    public static IndexSelection? Parse(string text, bool allowsPrimary, int offset = 1)
    {
        var t = text.Trim();
        if (t.Length == 0 || t.Equals("all", StringComparison.OrdinalIgnoreCase)) return All;
        if (allowsPrimary && t.Equals("primary", StringComparison.OrdinalIgnoreCase)) return Primary;
        var indices = new List<int>();
        foreach (var part in t.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries))
        {
            if (!int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) || n - offset < 0) return null;
            indices.Add(n - offset);
        }
        return Of(indices);
    }
}
