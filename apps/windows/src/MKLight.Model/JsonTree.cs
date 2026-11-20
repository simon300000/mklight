using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace MKLight.Model;

/// <summary>
/// An arbitrary JSON value with semantic equality (the counterpart of the Swift
/// mirror's <c>JSONValue</c>): object key order is irrelevant and numbers compare
/// by exact value, so <c>10</c> equals <c>10.0</c>. The wrapped
/// <see cref="JsonNode"/> is treated as immutable. Integers keep their full
/// 64-bit precision (Philox seeds are u64, CONVENTIONS §6).
/// </summary>
public readonly struct JsonTree : IEquatable<JsonTree>
{
    public JsonNode? Node { get; }

    public JsonTree(JsonNode? node) => Node = node;

    public static JsonTree Null => default;

    public static JsonTree Parse(string json) => new(JsonNode.Parse(json));

    public static JsonTree Parse(ReadOnlySpan<byte> utf8) => new(JsonNode.Parse(utf8));

    /// <summary>The member <paramref name="key"/> when this is an object that has it.</summary>
    public JsonTree? this[string key] =>
        Node is JsonObject o && o.TryGetPropertyValue(key, out var v) ? new JsonTree(v) : null;

    public bool IsNull => Node is null;

    public string? AsString => Node is JsonValue v && v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : null;

    public bool Equals(JsonTree other) => DeepEquals(Node, other.Node);

    public override bool Equals(object? obj) => obj is JsonTree t && Equals(t);

    public override int GetHashCode() => Hash(Node);

    public static bool operator ==(JsonTree a, JsonTree b) => a.Equals(b);
    public static bool operator !=(JsonTree a, JsonTree b) => !a.Equals(b);

    public override string ToString() => ModelJson.ToJsonString(Node);

    /// <summary>Paths at which two trees differ (empty when equal). For test diagnostics.</summary>
    public IReadOnlyList<string> Differences(JsonTree other)
    {
        var output = new List<string>();
        Diff(Node, other.Node, "$", output);
        return output;
    }

    private static void Diff(JsonNode? a, JsonNode? b, string path, List<string> output)
    {
        switch (a, b)
        {
            case (JsonObject x, JsonObject y):
                foreach (var k in x.Select(p => p.Key).Union(y.Select(p => p.Key)).Order(StringComparer.Ordinal))
                {
                    var hx = x.TryGetPropertyValue(k, out var vx);
                    var hy = y.TryGetPropertyValue(k, out var vy);
                    if (hx && hy) Diff(vx, vy, $"{path}.{k}", output);
                    else output.Add($"{path}.{k}: missing on {(hx ? "right" : "left")}");
                }
                break;
            case (JsonArray x, JsonArray y):
                if (x.Count != y.Count) output.Add($"{path}: array length {x.Count} vs {y.Count}");
                else for (var i = 0; i < x.Count; i++) Diff(x[i], y[i], $"{path}[{i}]", output);
                break;
            default:
                if (!DeepEquals(a, b)) output.Add($"{path}: {ModelJson.ToJsonString(a)} vs {ModelJson.ToJsonString(b)}");
                break;
        }
    }

    // Exact numeric identity: an integer when the value is integral, else the double.
    private readonly record struct NumKey(int Kind, long Signed, ulong Unsigned, double Real);

    private static NumKey? Number(JsonNode? n)
    {
        if (n is not JsonValue v || v.GetValueKind() != JsonValueKind.Number) return null;
        var text = v.ToJsonString();
        if (long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var l)) return new(0, l, 0, 0);
        if (ulong.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var u)) return new(1, 0, u, 0);
        var d = double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (Math.Floor(d) == d)
        {
            if (d >= -9.2233720368547758E+18 && d < 9.2233720368547758E+18) return new(0, (long)d, 0, 0);
            if (d >= 0 && d < 1.8446744073709552E+19) return new(1, 0, (ulong)d, 0);
        }
        return new(2, 0, 0, d);
    }

    public static bool DeepEquals(JsonNode? a, JsonNode? b)
    {
        if (Number(a) is { } x && Number(b) is { } y) return x == y;
        switch (a, b)
        {
            case (null, null): return true;
            case (JsonObject o, JsonObject p):
                return o.Count == p.Count && o.All(kv => p.TryGetPropertyValue(kv.Key, out var w) && DeepEquals(kv.Value, w));
            case (JsonArray o, JsonArray p):
                return o.Count == p.Count && o.Zip(p).All(t => DeepEquals(t.First, t.Second));
            case (JsonValue o, JsonValue p) when o.GetValueKind() == p.GetValueKind():
                return o.GetValueKind() switch
                {
                    JsonValueKind.String => o.GetValue<string>() == p.GetValue<string>(),
                    JsonValueKind.True or JsonValueKind.False => o.GetValue<bool>() == p.GetValue<bool>(),
                    _ => false,
                };
            default: return false;
        }
    }

    private static int Hash(JsonNode? n)
    {
        if (Number(n) is { } k)
            return k.Kind == 2 ? k.Real.GetHashCode() : (k.Kind == 0 ? (double)k.Signed : (double)k.Unsigned).GetHashCode();
        switch (n)
        {
            case null: return 0;
            case JsonObject o:
                var h = 5;
                foreach (var (key, v) in o) h ^= HashCode.Combine(key, Hash(v));
                return h;
            case JsonArray a:
                var ha = new HashCode();
                foreach (var e in a) ha.Add(Hash(e));
                return ha.ToHashCode();
            case JsonValue v when v.GetValueKind() == JsonValueKind.String: return v.GetValue<string>().GetHashCode();
            case JsonValue v: return v.GetValueKind().GetHashCode();
            default: return 1;
        }
    }
}
