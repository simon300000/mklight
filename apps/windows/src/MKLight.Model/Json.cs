using System.Collections.Concurrent;
using System.Globalization;
using System.Reflection;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace MKLight.Model;

/// <summary>
/// A model type with the engine's JSON form. The form is built and read by hand
/// (a <see cref="JsonNode"/> tree) so that it matches the engine's serde
/// representation exactly: tagged enums, explicit nulls, strict keys.
/// </summary>
public interface IJsonModel<TSelf> where TSelf : IJsonModel<TSelf>
{
    static abstract TSelf FromJson(JsonNode? node);
    JsonNode ToJson();
}

/// <summary>Lets <see cref="JsonSerializer"/> read and write an <see cref="IJsonModel{T}"/>.</summary>
public sealed class ModelConverter<T> : JsonConverter<T> where T : IJsonModel<T>
{
    public override T Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        T.FromJson(JsonNode.Parse(ref reader));

    public override void Write(Utf8JsonWriter writer, T value, JsonSerializerOptions options) =>
        ModelJson.WriteSorted(writer, value.ToJson());
}

/// <summary>Names a member of an enum in the JSON form (<c>"decenter_x"</c>).</summary>
[AttributeUsage(AttributeTargets.Field)]
public sealed class WireAttribute(string name) : Attribute
{
    public string Name { get; } = name;
}

/// <summary>JSON names of enum members, from <see cref="WireAttribute"/>.</summary>
public static class Wire
{
    private static class Map<T> where T : struct, Enum
    {
        public static readonly Dictionary<T, string> ToName = [];
        public static readonly Dictionary<string, T> FromName = [];

        static Map()
        {
            foreach (var f in typeof(T).GetFields(BindingFlags.Public | BindingFlags.Static))
            {
                var v = (T)f.GetValue(null)!;
                var name = f.GetCustomAttribute<WireAttribute>()?.Name ?? f.Name;
                ToName[v] = name;
                FromName[name] = v;
            }
        }
    }

    public static string Name<T>(T value) where T : struct, Enum => Map<T>.ToName[value];

    public static T? Parse<T>(string name) where T : struct, Enum =>
        Map<T>.FromName.TryGetValue(name, out var v) ? v : null;

    internal static T Read<T>(JsonNode? node, string what) where T : struct, Enum
    {
        var s = J.S(node);
        return Parse<T>(s) ?? throw J.Fail($"unknown {what} '{s}'");
    }

    internal static JsonNode Write<T>(T value) where T : struct, Enum => JsonValue.Create(Name(value));
}

/// <summary>Reading helpers for the hand-written JSON forms.</summary>
internal static class J
{
    public static JsonException Fail(string message) => new(message);

    public static JsonObject Obj(JsonNode? n) =>
        n as JsonObject ?? throw Fail($"expected an object, found {Describe(n)}");

    private static string Describe(JsonNode? n) => n is null ? "null" : n.GetValueKind().ToString().ToLowerInvariant();

    /// <summary>Rejects every key outside <paramref name="allowed"/>, like serde's deny_unknown_fields.</summary>
    public static void AllowOnly(JsonObject o, params string[] allowed)
    {
        foreach (var k in o.Select(p => p.Key).Order(StringComparer.Ordinal))
        {
            if (!allowed.Contains(k))
                throw Fail($"unknown field '{k}', expected one of {string.Join(", ", allowed)}");
        }
    }

    /// <summary>A required key (a missing key is an error; an explicit null reaches the reader).</summary>
    public static T Req<T>(JsonObject o, string key, Func<JsonNode?, T> read)
    {
        if (!o.TryGetPropertyValue(key, out var n)) throw Fail($"missing key '{key}'");
        return read(n);
    }

    /// <summary>A field with <c>#[serde(default)]</c>: missing gives the fallback, an explicit null is an error.</summary>
    public static T Def<T>(JsonObject o, string key, Func<JsonNode?, T> read, T fallback) =>
        o.TryGetPropertyValue(key, out var n) ? read(n) : fallback;

    public static T? OptRef<T>(JsonObject o, string key, Func<JsonNode?, T> read) where T : class =>
        o.TryGetPropertyValue(key, out var n) && n is not null ? read(n) : null;

    public static T? OptVal<T>(JsonObject o, string key, Func<JsonNode?, T> read) where T : struct =>
        o.TryGetPropertyValue(key, out var n) && n is not null ? read(n) : null;

    /// <summary>The required <c>type</c>-like tag of a tagged enum.</summary>
    public static string Tag(JsonObject o, string key) => Req(o, key, S);

    // MARK: primitives

    public static string S(JsonNode? n) =>
        n is JsonValue v && v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : throw Fail($"expected a string, found {Describe(n)}");

    public static bool B(JsonNode? n) =>
        n is JsonValue v && v.GetValueKind() is JsonValueKind.True or JsonValueKind.False
            ? v.GetValue<bool>()
            : throw Fail($"expected a boolean, found {Describe(n)}");

    public static double D(JsonNode? n)
    {
        if (n is JsonValue v && v.GetValueKind() == JsonValueKind.Number)
        {
            if (v.TryGetValue<double>(out var d)) return d;
            return double.Parse(v.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture);
        }
        throw Fail($"expected a number, found {Describe(n)}");
    }

    /// <summary>An integer of the 64-bit signed range.</summary>
    public static long L(JsonNode? n)
    {
        if (n is JsonValue v && v.GetValueKind() == JsonValueKind.Number)
        {
            if (v.TryGetValue<long>(out var l)) return l;
            if (v.TryGetValue<int>(out var i)) return i;
            var text = v.ToJsonString();
            if (long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out l)) return l;
        }
        throw Fail($"expected an integer, found {Describe(n)}");
    }

    public static int I32(JsonNode? n)
    {
        var l = L(n);
        return l is >= int.MinValue and <= int.MaxValue ? (int)l : throw Fail($"{l} is out of range");
    }

    /// <summary>A <c>usize</c>: a negative value is an error, as in serde.</summary>
    public static int Index(JsonNode? n)
    {
        var l = L(n);
        return l is >= 0 and <= int.MaxValue ? (int)l : throw Fail($"index {l} is out of range");
    }

    public static uint U32(JsonNode? n)
    {
        var l = L(n);
        return l is >= 0 and <= uint.MaxValue ? (uint)l : throw Fail($"{l} is out of range");
    }

    public static ulong U64(JsonNode? n)
    {
        if (n is JsonValue v && v.GetValueKind() == JsonValueKind.Number)
        {
            if (v.TryGetValue<ulong>(out var u)) return u;
            if (ulong.TryParse(v.ToJsonString(), NumberStyles.None, CultureInfo.InvariantCulture, out u)) return u;
        }
        throw Fail($"expected an unsigned integer, found {Describe(n)}");
    }

    /// <summary>A length that may be ±∞, written as <c>"inf"</c> / <c>"-inf"</c> (CONVENTIONS §2).</summary>
    public static double Ext(JsonNode? n)
    {
        if (n is JsonValue v && v.GetValueKind() == JsonValueKind.Number) return D(n);
        var s = S(n);
        return ExtendedDouble.Parse(s) ?? throw Fail($"'{s}' is not a number or ±inf");
    }

    public static JsonNode ExtNode(double v) =>
        double.IsInfinity(v) ? JsonValue.Create(v > 0 ? "inf" : "-inf") : JsonValue.Create(v);

    // MARK: containers

    public static ValueList<T> Arr<T>(JsonNode? n, Func<JsonNode?, T> read)
    {
        if (n is not JsonArray a) throw Fail($"expected an array, found {Describe(n)}");
        var items = new T[a.Count];
        for (var i = 0; i < items.Length; i++) items[i] = read(a[i]);
        return new ValueList<T>(items);
    }

    public static ValueList<double> Doubles(JsonNode? n) => Arr(n, D);

    public static ValueMap<T> Map<T>(JsonNode? n, Func<JsonNode?, T> read)
    {
        var o = Obj(n);
        return ValueMap<T>.From(o.Select(p => KeyValuePair.Create(p.Key, read(p.Value))));
    }

    /// <summary>A two-element array <c>[a, b]</c> (the engine writes Rust tuples this way).</summary>
    public static (JsonNode?, JsonNode?) Pair(JsonNode? n)
    {
        if (n is not JsonArray { Count: 2 } a) throw Fail("expected a 2-element array");
        return (a[0], a[1]);
    }

    // MARK: writing

    public static JsonNode Num(double v) => JsonValue.Create(v);

    public static JsonNode Arr<T>(IEnumerable<T> items, Func<T, JsonNode?> write) => new JsonArray(items.Select(write).ToArray());

    public static JsonNode Doubles(IEnumerable<double> items) => Arr(items, Num);

    public static JsonNode PairNode(JsonNode? a, JsonNode? b) => new JsonArray(a, b);

    public static JsonNode Map<T>(ValueMap<T> map, Func<T, JsonNode?> write)
    {
        var o = new JsonObject();
        foreach (var (k, v) in map) o.Add(k, write(v));
        return o;
    }
}

/// <summary>Parsing of ±∞ length strings, matching the engine's <c>serde_util::f64_inf</c>. NaN is rejected.</summary>
public static class ExtendedDouble
{
    public static double? Parse(string text)
    {
        var s = text.Trim().ToLowerInvariant();
        switch (s)
        {
            case "inf" or "+inf" or "infinity" or "+infinity": return double.PositiveInfinity;
            case "-inf" or "-infinity": return double.NegativeInfinity;
        }
        if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) && !double.IsNaN(v)) return v;
        return null;
    }
}

/// <summary>Shared JSON entry points for model and engine documents.</summary>
public static class ModelJson
{
    private static readonly JsonWriterOptions Compact = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    private static readonly JsonWriterOptions Pretty = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping, Indented = true };

    public static JsonNode? Parse(string json) => JsonNode.Parse(json);

    public static JsonNode? Parse(ReadOnlySpan<byte> utf8) => JsonNode.Parse(utf8);

    public static T Decode<T>(string json) where T : IJsonModel<T> => T.FromJson(Parse(json));

    public static T Decode<T>(ReadOnlySpan<byte> utf8) where T : IJsonModel<T> => T.FromJson(Parse(utf8));

    public static ValueList<T> DecodeList<T>(string json) where T : IJsonModel<T> => J.Arr(Parse(json), T.FromJson);

    /// <summary>Deterministic output: sorted keys, so equal values give equal bytes.</summary>
    public static string Encode<T>(T value, bool pretty = false) where T : IJsonModel<T> => ToJsonString(value.ToJson(), pretty);

    public static string EncodeList<T>(IEnumerable<T> values, bool pretty = false) where T : IJsonModel<T> =>
        ToJsonString(J.Arr(values, v => v.ToJson()), pretty);

    public static byte[] EncodeUtf8<T>(T value, bool pretty = false) where T : IJsonModel<T> =>
        System.Text.Encoding.UTF8.GetBytes(Encode(value, pretty));

    /// <summary>Writes a tree with sorted keys. A non-finite number throws <see cref="ArgumentException"/>.</summary>
    public static string ToJsonString(JsonNode? node, bool pretty = false)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, pretty ? Pretty : Compact)) WriteSorted(w, node);
        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }

    internal static void WriteSorted(Utf8JsonWriter w, JsonNode? node)
    {
        switch (node)
        {
            case null: w.WriteNullValue(); break;
            case JsonObject o:
                w.WriteStartObject();
                foreach (var p in o.OrderBy(p => p.Key, StringComparer.Ordinal))
                {
                    w.WritePropertyName(p.Key);
                    WriteSorted(w, p.Value);
                }
                w.WriteEndObject();
                break;
            case JsonArray a:
                w.WriteStartArray();
                foreach (var e in a) WriteSorted(w, e);
                w.WriteEndArray();
                break;
            default: node.WriteTo(w); break;
        }
    }
}
