using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// Reading the engine's result JSON (docs/RESULTS.md §3): numbers may be
/// JSON numbers, <c>null</c> (undefined) or the strings "inf", "-inf" and
/// "nan" (the <c>nf</c> type). Every reader is lenient: a value of the wrong
/// shape reads as missing, never as an exception.
/// </summary>
public static class Json
{
    /// <summary>An <c>nf</c>: a number, or NaN for null, a missing key or anything else.</summary>
    public static double Nf(JsonNode? n) => Number(n) ?? double.NaN;

    /// <summary>A number (including "inf"/"-inf"/"nan"); null when the node is not one.</summary>
    public static double? Number(JsonNode? n)
    {
        if (n is not JsonValue v) return null;
        switch (v.GetValueKind())
        {
            case JsonValueKind.Number:
                if (v.TryGetValue<double>(out var d)) return d;
                if (v.TryGetValue<long>(out var l)) return l;
                if (v.TryGetValue<int>(out var i)) return i;
                if (v.TryGetValue<ulong>(out var u)) return u;
                if (v.TryGetValue<float>(out var f)) return f;
                if (v.TryGetValue<decimal>(out var m)) return (double)m;
                return double.TryParse(v.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var p) ? p : null;
            case JsonValueKind.String:
                return v.GetValue<string>().Trim().ToLowerInvariant() switch
                {
                    "inf" or "+inf" or "infinity" => double.PositiveInfinity,
                    "-inf" or "-infinity" => double.NegativeInfinity,
                    "nan" => double.NaN,
                    _ => null,
                };
            default:
                return null;
        }
    }

    /// <summary>An array of <c>nf</c>; null when the node is not an array.</summary>
    public static double[]? NfArray(JsonNode? n) => n is JsonArray a ? a.Select(Nf).ToArray() : null;

    /// <summary>A whole number; null otherwise.</summary>
    public static int? Int(JsonNode? n)
    {
        var d = Number(n);
        return d is double x && double.IsFinite(x) && Math.Abs(x) <= int.MaxValue && Math.Round(x) == x ? (int)x : null;
    }

    public static int[]? IntArray(JsonNode? n) =>
        n is JsonArray a ? a.Select(Int).Where(i => i is not null).Select(i => i!.Value).ToArray() : null;

    public static string? Str(JsonNode? n) => n is JsonValue v && v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : null;

    public static bool? Bool(JsonNode? n) => n is JsonValue v ? v.GetValueKind() switch
    {
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        _ => null,
    } : null;

    public static JsonArray? Arr(JsonNode? n) => n as JsonArray;
    public static JsonObject? Obj(JsonNode? n) => n as JsonObject;

    /// <summary>The child at a key path (<c>"psf", "size"</c>); null when any step is missing.</summary>
    public static JsonNode? At(JsonNode? n, params string[] path)
    {
        foreach (var k in path)
        {
            if (n is not JsonObject o || !o.TryGetPropertyValue(k, out n)) return null;
        }
        return n;
    }

    /// <summary>Whether an object has the key (a JSON null counts).</summary>
    public static bool Has(JsonNode? n, string key) => n is JsonObject o && o.ContainsKey(key);

    /// <summary>The buffer a JSON string value names (the wave analyses refer to their buffers by name).</summary>
    public static IResultArray? BufferRef(ResultData data, JsonNode? n) => Str(n) is string name ? data.Buffer(name) : null;

    /// <summary>Numbers given inline (an array) or by a buffer name.</summary>
    public static double[]? Numbers(ResultData data, JsonNode? n) =>
        NfArray(n) ?? (BufferRef(data, n) is IResultArray b ? Doubles(b) : null);

    public static double[] Doubles(IResultArray b)
    {
        var v = new double[b.Count];
        for (var i = 0; i < v.Length; i++) v[i] = b[i];
        return v;
    }

    /// <summary>A JSON value's display text: numbers in the one number format, strings as they are.</summary>
    public static string Display(JsonNode? n)
    {
        switch (n)
        {
            case null: return "—";
            case JsonArray a: return $"[{a.Count}]";
            case JsonObject o: return $"{{{o.Count}}}";
        }
        if (Bool(n) is bool b) return b ? "true" : "false";
        if (n is JsonValue v && v.GetValueKind() == JsonValueKind.Number) return Fmt.Short(Nf(n));
        var s = Str(n) ?? n.ToJsonString();
        return Number(n) is double x ? Fmt.Short(x) : s;
    }

    /// <summary>
    /// Structural equality of two JSON values, numbers compared as numbers
    /// (3 equals 3.0), object keys in any order. A C# null is JSON null.
    /// </summary>
    public static bool Same(JsonNode? a, JsonNode? b)
    {
        if (a is null || b is null) return a is null && b is null;
        switch (a)
        {
            case JsonObject oa:
                if (b is not JsonObject ob || oa.Count != ob.Count) return false;
                foreach (var (k, va) in oa)
                    if (!ob.TryGetPropertyValue(k, out var vb) || !Same(va, vb)) return false;
                return true;
            case JsonArray aa:
                if (b is not JsonArray ab || aa.Count != ab.Count) return false;
                for (var i = 0; i < aa.Count; i++)
                    if (!Same(aa[i], ab[i])) return false;
                return true;
        }
        if (b is not JsonValue) return false;
        var ka = ((JsonValue)a).GetValueKind();
        var kb = ((JsonValue)b).GetValueKind();
        if (ka == JsonValueKind.Number && kb == JsonValueKind.Number) return Nf(a).Equals(Nf(b));
        if (ka != kb) return false;
        return ka switch
        {
            JsonValueKind.String => Str(a) == Str(b),
            JsonValueKind.True or JsonValueKind.False or JsonValueKind.Null => true,
            _ => a.ToJsonString() == b.ToJsonString(),
        };
    }
}
