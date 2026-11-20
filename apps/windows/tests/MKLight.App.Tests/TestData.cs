using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis;

namespace MKLight.App.Tests;

/// <summary>A result buffer held in memory.</summary>
sealed class FakeArray(string name, int[] shape, double[] values) : IResultArray
{
    public string Name => name;
    public IReadOnlyList<int> Shape => shape;
    public int Count => values.Length;
    public double this[int index] => values[index];
}

static class TestData
{
    public static ResultData Result(string analysis, string json, params IResultArray[] buffers) => new()
    {
        Analysis = analysis,
        Json = JsonNode.Parse(json),
        Buffers = buffers,
    };

    public static ResultData Result(string analysis, string json, IReadOnlyList<double> um, IReadOnlyList<string> fields, params IResultArray[] buffers) => new()
    {
        Analysis = analysis,
        Json = JsonNode.Parse(json),
        Buffers = buffers,
        WavelengthsUm = um,
        FieldLabels = fields,
    };

    public static FakeArray Buffer(string name, int[] shape, params double[] values) => new(name, shape, values);

    public static FakeArray Vector(string name, params double[] values) => new(name, [values.Length], values);

    public static JsonObject Obj(string json) => (JsonObject)JsonNode.Parse(json)!;
}
