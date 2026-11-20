using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// Any result read from its JSON and buffers alone (the generic view): scalars
/// as a list, arrays of records as tables, number arrays as curves, buffers
/// as curves or maps. New engine analyses show up as data without code changes.
/// </summary>
public static class GenericData
{
    public const int MaxTableRows = 200;

    public static string Title(string key)
    {
        var words = key.Replace('_', ' ').Replace('.', ' ').Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return string.Join(' ', words.Select(w => char.ToUpperInvariant(w[0]) + w[1..]));
    }

    public static bool IsScalar(JsonNode? v) => v is not JsonArray and not JsonObject;

    /// <summary>Scalars at the top level and one object level down ("a.b"), sorted by key.</summary>
    public static List<(string Key, string Value)> Scalars(JsonNode? json)
    {
        var list = new List<(string, string)>();
        if (json is not JsonObject o) return json is null ? list : [("value", Json.Display(json))];
        foreach (var (k, v) in o.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (v is JsonObject inner)
            {
                foreach (var (k2, v2) in inner.OrderBy(p => p.Key, StringComparer.Ordinal))
                    if (IsScalar(v2)) list.Add(($"{k}.{k2}", Json.Display(v2)));
            }
            else if (IsScalar(v))
            {
                list.Add((k, Json.Display(v)));
            }
        }
        return list;
    }

    /// <summary>A table of records: column keys and display rows (at most <see cref="MaxTableRows"/>).</summary>
    public sealed record Table(string Name, IReadOnlyList<string> Columns, IReadOnlyList<string[]> Rows, int TotalRows);

    /// <summary>Top-level arrays whose elements are all objects, as tables of their scalar columns.</summary>
    public static List<Table> RecordTables(JsonNode? json)
    {
        var list = new List<Table>();
        if (json is not JsonObject o) return list;
        foreach (var (k, v) in o.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (v is not JsonArray { Count: > 0 } a || !a.All(e => e is JsonObject)) continue;
            var cols = a.Take(50).Cast<JsonObject>().SelectMany(r => r.Where(p => IsScalar(p.Value)).Select(p => p.Key))
                .Distinct().Order(StringComparer.Ordinal).ToList();
            if (cols.Count == 0) continue;
            var rows = a.Take(MaxTableRows).Cast<JsonObject>()
                .Select(r => cols.Select(c => r.TryGetPropertyValue(c, out var x) ? Json.Display(x) : "").ToArray()).ToList();
            list.Add(new Table(k, cols, rows, a.Count));
        }
        return list;
    }

    /// <summary>Top-level number arrays of more than four values.</summary>
    public static List<(string Name, double[] Values)> Curves(JsonNode? json)
    {
        var list = new List<(string, double[])>();
        if (json is not JsonObject o) return list;
        foreach (var (k, v) in o.OrderBy(p => p.Key, StringComparer.Ordinal))
            if (v is JsonArray { Count: > 4 } a && a.All(e => Json.Number(e) is not null || e is null))
                list.Add((k, Json.NfArray(a)!));
        return list;
    }

    public enum BufferKind
    {
        /// <summary>Shape and range only.</summary>
        Summary,
        /// <summary>1-D: a curve against the index.</summary>
        Curve,
        /// <summary>[N, 2]: points (x, y).</summary>
        Points,
        /// <summary>2-D [rows, columns] with more than three columns: a map.</summary>
        Map,
    }

    public static BufferKind KindOf(IResultArray b)
    {
        var s = b.Shape;
        if (s.Count == 1 && b.Count > 1 && b.Count <= 20000) return BufferKind.Curve;
        if (s.Count == 2 && s[1] == 2 && b.Count > 0 && b.Count <= 400000) return BufferKind.Points;
        if (s.Count == 2 && s[1] > 3 && s[0] > 1 && s[0] <= 8192 && s[1] <= 8192 && b.Count <= 4_000_000) return BufferKind.Map;
        return BufferKind.Summary;
    }

    /// <summary>"points · f64 [1143, 2]" style description: the name and shape.</summary>
    public static string Describe(IResultArray b) => $"{b.Name}  [{string.Join(", ", b.Shape)}]";

    /// <summary>The finite range and the count of values without one.</summary>
    public static (double Min, double Max, int Missing) Range(IResultArray b)
    {
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        var missing = 0;
        for (var i = 0; i < b.Count; i++)
        {
            var v = b[i];
            if (!double.IsFinite(v)) { missing++; continue; }
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        return (lo, hi, missing);
    }
}

/// <summary>
/// One rule for every map of a quantity (docs/APP.md "Heat-map scales"):
/// signed data keeps a diverging scale centred on 0 and symmetric about it;
/// other data takes the perceptual ramp over its own range; a flat map gets a
/// narrow band around its value.
/// </summary>
public readonly record struct HeatScale(bool Diverging, double Min, double Max)
{
    public static HeatScale For(double lo, double hi)
    {
        if (!double.IsFinite(lo) || !double.IsFinite(hi)) return new(false, 0, 1);
        if (lo < 0 && hi > 0)
        {
            var m = Math.Max(-lo, hi);
            return new(true, -m, m);
        }
        if (hi - lo <= 1e-12 * Math.Max(1, Math.Abs(hi)))
        {
            var d = Math.Max(Math.Abs(hi) * 0.01, 1e-9);
            return new(false, lo - d, hi + d);
        }
        return new(false, lo, hi);
    }

    /// <summary>The value's position on the ramp: −1…1 for a diverging scale, 0…1 otherwise; NaN stays NaN.</summary>
    public double Position(double v)
    {
        if (!double.IsFinite(v)) return double.NaN;
        return Diverging ? Math.Clamp(v / Max, -1, 1) : Math.Clamp((v - Min) / (Max - Min), 0, 1);
    }
}
