using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace MKLight.Engine;

/// <summary>
/// Tolerant accessors for engine JSON (docs/RESULTS.md). Readers of analysis
/// results use them so that a missing key or a different type gives
/// <c>null</c> (and the view falls back to its generic form) instead of an
/// exception.
/// <para>
/// Numbers follow CONVENTIONS §11: <c>null</c> is NaN (undefined) and the
/// strings <c>"inf"</c>, <c>"-inf"</c>, <c>"nan"</c> are non-finite values
/// (<see cref="Nf(JsonNode?)"/>). <see cref="JsonNode"/> represents a JSON
/// <c>null</c> as a C# <c>null</c>, so <c>Nf</c> of a null node is NaN; use
/// <see cref="NfAt"/> to tell a missing key (<c>null</c>) from a JSON null (NaN).
/// </para>
/// </summary>
public static class JsonRead
{
    public static JsonArray? ArrayOrNull(this JsonNode? node) => node as JsonArray;
    public static JsonObject? ObjectOrNull(this JsonNode? node) => node as JsonObject;

    public static string? StringOrNull(this JsonNode? node) =>
        node is JsonValue v && v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : null;

    public static bool? BoolOrNull(this JsonNode? node) =>
        node is JsonValue v && v.GetValueKind() is JsonValueKind.True or JsonValueKind.False ? v.GetValue<bool>() : null;

    /// <summary>Whether the node is a JSON number.</summary>
    public static bool IsNumber(this JsonNode? node) => node is JsonValue v && v.GetValueKind() == JsonValueKind.Number;

    /// <summary>
    /// A number as a <see cref="double"/>, or an extended-number string
    /// (<c>"inf"</c>, <c>"-inf"</c>, <c>"infinity"</c>, a numeric string); <c>null</c> otherwise.
    /// </summary>
    public static double? DoubleValue(this JsonNode? node)
    {
        if (node is not JsonValue v) return null;
        return v.GetValueKind() switch
        {
            JsonValueKind.Number => Json.TryGetDouble(v, out var d) ? d : null,
            JsonValueKind.String => ParseExtendedDouble(v.GetValue<string>()),
            _ => null,
        };
    }

    /// <summary>
    /// An <c>nf</c> number: a number, <c>null</c> (NaN) or one of the strings
    /// <c>inf</c>, <c>-inf</c>, <c>+inf</c>, <c>nan</c>. <c>null</c> for anything else.
    /// </summary>
    public static double? Nf(this JsonNode? node)
    {
        if (node is null) return double.NaN;
        if (node.StringOrNull() is { } s)
        {
            return s.ToLowerInvariant() switch
            {
                "inf" or "+inf" => double.PositiveInfinity,
                "-inf" => double.NegativeInfinity,
                "nan" => double.NaN,
                _ => null,
            };
        }
        return node.IsNumber() ? node.DoubleValue() : null;
    }

    /// <summary>The <c>nf</c> number under <paramref name="key"/>: <c>null</c> when the key is missing, NaN for a JSON null.</summary>
    public static double? NfAt(this JsonNode? node, string key) =>
        node is JsonObject o && o.TryGetPropertyValue(key, out var v) ? v.Nf() : null;

    /// <summary>An integer value (integral numbers a <see cref="long"/> holds only).</summary>
    public static long? IntOrNull(this JsonNode? node)
    {
        if (node is not JsonValue v || v.GetValueKind() != JsonValueKind.Number) return null;
        if (Json.TryGetInt64(v, out var l)) return l;
        if (!Json.TryGetDouble(v, out var d)) return null;
        return d == Math.Floor(d) && d >= -9.2233720368547758E18 && d < 9.2233720368547758E18 ? (long)d : null;
    }

    /// <summary>An unsigned 64-bit integer (seeds), exact above 2^53.</summary>
    public static ulong? UInt64OrNull(this JsonNode? node) =>
        node is JsonValue v && v.GetValueKind() == JsonValueKind.Number && Json.TryGetUInt64(v, out var u) ? u : null;

    /// <summary>An array of <c>nf</c> numbers (<c>null</c> when any element is not one).</summary>
    public static double[]? NfArray(this JsonNode? node)
    {
        if (node is not JsonArray a) return null;
        var output = new double[a.Count];
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].Nf() is not { } d) return null;
            output[i] = d;
        }
        return output;
    }

    /// <summary>An array of integers (<c>null</c> when any element is not one).</summary>
    public static long[]? IntArray(this JsonNode? node)
    {
        if (node is not JsonArray a) return null;
        var output = new long[a.Count];
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].IntOrNull() is not { } n) return null;
            output[i] = n;
        }
        return output;
    }

    /// <summary>Nested lookup: <c>json.At("fields", 0, "rms_radius")</c>. Keys are strings (object keys) or integers (array indices).</summary>
    public static JsonNode? At(this JsonNode? node, params object[] path)
    {
        var current = node;
        foreach (var key in path)
        {
            current = (key, current) switch
            {
                (string s, JsonObject o) => o.TryGetPropertyValue(s, out var v) ? v : null,
                (int i, JsonArray a) => i >= 0 && i < a.Count ? a[i] : null,
                _ => null,
            };
            if (current is null) return null;
        }
        return current;
    }

    /// <summary>The engine's extended-number text: <c>inf</c>, <c>-inf</c>, <c>infinity</c> or a decimal number (not NaN).</summary>
    public static double? ParseExtendedDouble(string text)
    {
        switch (text.Trim().ToLowerInvariant())
        {
            case "inf" or "+inf" or "infinity" or "+infinity": return double.PositiveInfinity;
            case "-inf" or "-infinity": return double.NegativeInfinity;
            case var other:
                return double.TryParse(other, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) && !double.IsNaN(d) ? d : null;
        }
    }

    /// <summary>
    /// A quantity that may legitimately be non-finite, as engine JSON carries
    /// it: a number, <c>null</c> (NaN), <c>"inf"</c>, <c>"-inf"</c> or <c>"nan"</c>.
    /// Anything else throws <see cref="FormatException"/>.
    /// </summary>
    public static double EngineDouble(JsonNode? node)
    {
        if (node is null) return double.NaN;
        if (node.IsNumber()) return node.DoubleValue()!.Value;
        if (node.StringOrNull() is { } s)
        {
            if (s.Trim().Equals("nan", StringComparison.OrdinalIgnoreCase)) return double.NaN;
            if (ParseExtendedDouble(s) is { } d) return d;
            throw new FormatException($"'{s}' is not a number, ±inf or nan");
        }
        throw new FormatException($"{node.ToJsonString()} is not a number, ±inf or nan");
    }

    /// <summary>Writes a value as <see cref="EngineDouble"/> reads it: ±∞ as strings, NaN as null.</summary>
    public static JsonNode? EngineDoubleNode(double value) =>
        double.IsNaN(value) ? null
        : double.IsInfinity(value) ? JsonValue.Create(value > 0 ? "inf" : "-inf")
        : JsonValue.Create(value);
}

/// <summary>JSON plumbing shared by the engine types.</summary>
public static class Json
{
    /// <summary>Parses engine JSON; text that is not JSON is an <see cref="EngineException"/> with code <c>Parse</c>.</summary>
    public static JsonNode? Parse(string text, string what = "engine JSON")
    {
        try
        {
            return JsonNode.Parse(text);
        }
        catch (JsonException e)
        {
            throw EngineException.Engine(EngineErrorCode.Parse, $"unexpected {what}: {e.Message}");
        }
    }

    /// <summary>
    /// Builds every lazily parsed object and array of a parsed tree now, so the tree can be
    /// read from several threads at once (a parsed <see cref="JsonNode"/> fills its children on
    /// first access, which is not thread-safe). Returns <paramref name="node"/>.
    /// </summary>
    public static JsonNode? Materialize(JsonNode? node)
    {
        switch (node)
        {
            case JsonObject o:
                foreach (var (_, v) in o) Materialize(v);
                break;
            case JsonArray a:
                foreach (var v in a) Materialize(v);
                break;
        }
        return node;
    }

    /// <summary>A deep copy (a node can only have one parent).</summary>
    public static JsonNode? Clone(JsonNode? node) => node?.DeepClone();

    /// <summary>
    /// The JSON type name of a value: <c>null</c>, <c>boolean</c>, <c>integer</c>
    /// (an integer literal or a C# integer), <c>number</c>, <c>string</c>, <c>array</c>, <c>object</c>.
    /// </summary>
    public static string TypeName(JsonNode? node) => node switch
    {
        null => "null",
        JsonArray => "array",
        JsonObject => "object",
        JsonValue v => v.GetValueKind() switch
        {
            JsonValueKind.True or JsonValueKind.False => "boolean",
            JsonValueKind.String => "string",
            JsonValueKind.Number => IsIntegerLiteral(v) ? "integer" : "number",
            _ => "null",
        },
        _ => "null",
    };

    static bool IsIntegerLiteral(JsonValue v) => v.GetValue<object>() switch
    {
        JsonElement e => e.GetRawText().IndexOfAny(['.', 'e', 'E']) < 0,
        double or float or decimal or Half => false,
        _ => true,
    };

    internal static bool TryGetDouble(JsonValue v, out double d)
    {
        switch (v.GetValue<object>())
        {
            case JsonElement e when e.ValueKind == JsonValueKind.Number: return e.TryGetDouble(out d);
            case double x: d = x; return true;
            case float x: d = x; return true;
            case decimal x: d = (double)x; return true;
            case Half x: d = (double)x; return true;
            case long x: d = x; return true;
            case int x: d = x; return true;
            case ulong x: d = x; return true;
            case uint x: d = x; return true;
            case short x: d = x; return true;
            case ushort x: d = x; return true;
            case byte x: d = x; return true;
            case sbyte x: d = x; return true;
            default: d = 0; return false;
        }
    }

    internal static bool TryGetInt64(JsonValue v, out long l)
    {
        switch (v.GetValue<object>())
        {
            case JsonElement e when e.ValueKind == JsonValueKind.Number: return e.TryGetInt64(out l);
            case long x: l = x; return true;
            case int x: l = x; return true;
            case uint x: l = x; return true;
            case short x: l = x; return true;
            case ushort x: l = x; return true;
            case byte x: l = x; return true;
            case sbyte x: l = x; return true;
            case ulong x when x <= long.MaxValue: l = (long)x; return true;
            default: l = 0; return false;
        }
    }

    internal static bool TryGetUInt64(JsonValue v, out ulong u)
    {
        if (v.GetValue<object>() is JsonElement e)
        {
            u = 0;
            return e.ValueKind == JsonValueKind.Number && e.TryGetUInt64(out u);
        }
        if (v.GetValue<object>() is ulong x) { u = x; return true; }
        if (TryGetInt64(v, out var l) && l >= 0) { u = (ulong)l; return true; }
        u = 0;
        return false;
    }

    /// <summary>An integer literal or C# integer, exactly (for seeds beyond 2^53).</summary>
    static bool TryGetExactInteger(JsonValue v, out decimal m)
    {
        if (v.GetValue<object>() is JsonElement e)
        {
            m = 0;
            return e.ValueKind == JsonValueKind.Number && e.GetRawText().IndexOfAny(['.', 'e', 'E']) < 0
                && decimal.TryParse(e.GetRawText(), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out m);
        }
        if (TryGetUInt64(v, out var u)) { m = u; return true; }
        if (TryGetInt64(v, out var l)) { m = l; return true; }
        m = 0;
        return false;
    }
    /// <summary>
    /// Structural equality of two JSON values. Numbers compare by value: two integers
    /// exactly (two seeds beyond 2^53 that round to the same double differ), anything
    /// else as doubles (<c>6</c> equals <c>6.0</c>). Object key order does not matter.
    /// </summary>
    public static bool DeepEquals(JsonNode? a, JsonNode? b)
    {
        switch (a, b)
        {
            case (null, null): return true;
            case (null, _) or (_, null): return false;
            case (JsonObject x, JsonObject y):
                if (x.Count != y.Count) return false;
                foreach (var (key, value) in x)
                {
                    if (!y.TryGetPropertyValue(key, out var other) || !DeepEquals(value, other)) return false;
                }
                return true;
            case (JsonArray x, JsonArray y):
                if (x.Count != y.Count) return false;
                for (int i = 0; i < x.Count; i++)
                {
                    if (!DeepEquals(x[i], y[i])) return false;
                }
                return true;
            case (JsonValue x, JsonValue y):
                var kx = x.GetValueKind();
                if (kx != y.GetValueKind()) return false;
                return kx switch
                {
                    JsonValueKind.String => x.GetValue<string>() == y.GetValue<string>(),
                    JsonValueKind.Number => TryGetExactInteger(x, out var mx) && TryGetExactInteger(y, out var my)
                        ? mx == my
                        : TryGetDouble(x, out var dx) && TryGetDouble(y, out var dy) && dx == dy,
                    _ => true,
                };
            default: return false;
        }
    }

    /// <summary>A hash consistent with <see cref="DeepEquals"/> (coarse: type and size only).</summary>
    public static int ShallowHash(JsonNode? node) => node switch
    {
        null => 0,
        JsonObject o => HashCode.Combine(1, o.Count),
        JsonArray a => HashCode.Combine(2, a.Count),
        JsonValue v => v.GetValueKind() == JsonValueKind.String ? HashCode.Combine(3, v.GetValue<string>()) : (int)v.GetValueKind(),
        _ => 0,
    };

    /// <summary>A JSON object of the dictionary's values, deep-copied.</summary>
    public static JsonObject ObjectOf(IEnumerable<KeyValuePair<string, JsonNode?>> pairs)
    {
        var o = new JsonObject();
        foreach (var (k, v) in pairs) o[k] = Clone(v);
        return o;
    }
}
