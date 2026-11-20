using MKLight.App.Support;
using Fmt = MKLight.App.Views.Analysis.Results.Readers.Fmt;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Analysis;

/// <summary>An <see cref="IResultArray"/> over the engine's zero-copy buffer (no copy; converts per element).</summary>
public sealed class EngineResultArray(ResultBuffer buffer) : IResultArray
{
    public string Name => buffer.Name;
    public IReadOnlyList<int> Shape => buffer.Shape;
    public int Count => buffer.Count;
    public double this[int index] => buffer[index];
}

/// <summary>Builds the views' input from an engine output and the design it was computed for.</summary>
public static class ResultHosting
{
    public static ResultData Data(string analysis, AnalysisOutput output, OpticalSystem system) => new()
    {
        Analysis = analysis,
        Json = output.Json,
        Buffers = output.Buffers.Select(b => (IResultArray)new EngineResultArray(b)).ToList(),
        WavelengthsUm = system.Wavelengths.Wavelengths.Select(w => w.Um).ToList(),
        FieldLabels = FieldLabels(system),
    };

    /// <summary>A field as specified: <c>0°</c>, <c>14°</c>, <c>(3, 4) mm</c>.</summary>
    public static IReadOnlyList<string> FieldLabels(OpticalSystem system)
    {
        var unit = system.Fields.Kind == FieldKind.ObjectAngle ? "°" : " mm";
        return system.Fields.Fields.Select(f =>
            f.X == 0 ? Fmt.Short(f.Y) + unit : $"({Fmt.Short(f.X)}, {Fmt.Short(f.Y)}){unit}").ToList();
    }
}
