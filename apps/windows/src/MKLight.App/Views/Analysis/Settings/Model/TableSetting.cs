using System.Globalization;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// Rows of the same scalar columns, <c>[{"hx", "hy"}, …]</c> or
/// <c>[[r, θ], …]</c>, edited as a small table: what the rows of a value
/// are, and the value of edited rows. A cell is null while empty; a row with
/// an empty number is not part of the setting until it is filled.
/// </summary>
public static partial class TableSetting
{
    /// <summary>The rows of a value, one cell per column.</summary>
    public static List<JsonNode?[]> Rows(JsonNode? value, IReadOnlyList<SettingsField> columns, bool rowsAreArrays) =>
        (Json.Arr(value) ?? []).Select(r => columns.Select((c, k) =>
            (rowsAreArrays ? (Json.Arr(r) is { } a && k < a.Count ? a[k] : null) : Json.At(r, c.Name))?.DeepClone()).ToArray()).ToList();

    /// <summary>The value of the rows, dropping rows with a number missing.</summary>
    public static JsonArray Value(IEnumerable<JsonNode?[]> rows, IReadOnlyList<SettingsField> columns, bool rowsAreArrays)
    {
        var result = new JsonArray();
        foreach (var r in rows)
        {
            var cells = new List<JsonNode?>();
            var complete = true;
            for (var k = 0; k < columns.Count; k++)
            {
                var v = k < r.Length ? r[k] : null;
                var types = columns[k].BaseTypes;
                if (types.SetEquals(["string"])) cells.Add(v?.DeepClone() ?? "");
                else if (types.SetEquals(["boolean"])) cells.Add(v?.DeepClone() ?? false);
                else if (Json.Number(v) is not null) cells.Add(v!.DeepClone());
                else { complete = false; break; }
            }
            if (!complete) continue;
            if (rowsAreArrays) result.Add(new JsonArray([.. cells]));
            else
            {
                var o = new JsonObject();
                for (var k = 0; k < columns.Count; k++) o[columns[k].Name] = cells[k];
                result.Add(o);
            }
        }
        return result;
    }

    /// <summary>A cell's text ("" when empty).</summary>
    public static string Text(JsonNode? v, bool exact = false) =>
        v is null ? "" : Json.Number(v) is double d ? exact ? NumberText.Exact(d) : Fmt.Short(d) : Json.Str(v) ?? Json.Display(v);

    /// <summary>
    /// A typed cell of <paramref name="column"/>: text, yes/no, or a finite
    /// number (whole for an integer column); empty is an empty cell. Null when the text is not a value.
    /// </summary>
    public static Edit? Parse(string typed, SettingsField column)
    {
        var t = typed.Trim();
        var types = column.BaseTypes;
        if (types.SetEquals(["string"])) return new Edit(t);
        if (types.SetEquals(["boolean"])) return new Edit(t.ToLowerInvariant() is "true" or "yes" or "1" or "on");
        if (t.Length == 0) return new Edit(null);
        if (NumberText.Parse(t) is not double d || !double.IsFinite(d)) return null;
        return types.SetEquals(["integer"]) ? new Edit((long)Math.Truncate(d)) : new Edit(d);
    }

    /// <summary>
    /// What an empty list means, from the description's "empty = …" clause
    /// ("The system fields"), else "Automatic" or "None".
    /// </summary>
    public static string EmptyTitle(SettingsField f)
    {
        var m = EmptyClause().Match(f.Description);
        if (m.Success && m.Groups[1].Value.Trim() is { Length: > 0 } t) return char.ToUpperInvariant(t[0]) + t[1..];
        return Loc.S(f.Nullable ? "Analysis_Settings_Automatic" : "Analysis_Settings_None");
    }

    /// <summary>"3 rows", "1 row", or what an empty list means.</summary>
    public static string ButtonTitle(SettingsField f, JsonNode? value) => (Json.Arr(value)?.Count ?? 0) switch
    {
        0 => EmptyTitle(f),
        1 => Loc.S("Analysis_Settings_Rows_One"),
        var n => Loc.F("Analysis_Settings_Rows_Many", n),
    };

    /// <summary>The row keys a description states as <c>[{"hx", "hy"}]</c>.</summary>
    public static IReadOnlyList<string>? RowKeys(string description)
    {
        var m = RowKeysPattern().Match(description);
        if (!m.Success) return null;
        var keys = Regex.Split(m.Value, "[^A-Za-z0-9_]+").Where(k => k.Length > 0).ToList();
        return keys.Count == 0 ? null : keys;
    }

    [GeneratedRegex(@"empty = ([^;(.]+)")]
    private static partial Regex EmptyClause();

    [GeneratedRegex(@"\[\{(""[a-z_0-9]+""(,\s*)?)+\}\]")]
    private static partial Regex RowKeysPattern();
}

/// <summary>
/// The chief-ray sensor limit, <c>[[image radius mm, max incidence °], …]</c>:
/// linear between points, held beyond the ends; empty is no limit.
/// </summary>
public static class SensorLimit
{
    public static List<(double Radius, double Angle)> Rows(JsonNode? value) =>
        (Json.Arr(value) ?? []).Select(r => (Json.Number(Json.Arr(r)?.ElementAtOrDefault(0)), Json.Number(Json.Arr(r)?.ElementAtOrDefault(1))))
            .Where(p => p.Item1 is not null && p.Item2 is not null).Select(p => (p.Item1!.Value, p.Item2!.Value)).ToList();

    /// <summary>The rows sorted by radius, one per radius, as the setting (null: no limit).</summary>
    public static JsonArray? Value(IEnumerable<(double Radius, double Angle)> rows)
    {
        var unique = new List<(double Radius, double Angle)>();
        foreach (var r in rows.OrderBy(r => r.Radius))
            if (unique.Count == 0 || unique[^1].Radius != r.Radius) unique.Add(r);
        return unique.Count == 0 ? null : new JsonArray(unique.Select(r => (JsonNode)new JsonArray(r.Radius, r.Angle)).ToArray());
    }

    /// <summary>"Sensor limit: None", "Sensor limit: 30°", "Sensor limit: 3 points".</summary>
    public static string Title(JsonNode? value)
    {
        var rows = Rows(value);
        return Loc.F("Analysis_Settings_SensorLimitTitle", rows.Count switch
        {
            0 => Loc.S("Analysis_Settings_None"),
            1 => rows[0].Angle.ToString("0.0", CultureInfo.InvariantCulture) + "°",
            var n => Loc.F("Analysis_Settings_Points_Many", n),
        });
    }

    /// <summary>The row a new point starts as: one mm past the last radius, its angle (30° on an empty table).</summary>
    public static (double Radius, double Angle) Next(IReadOnlyList<(double Radius, double Angle)> rows) =>
        rows.Count == 0 ? (0, 30) : (rows[^1].Radius + 1, rows[^1].Angle);
}
