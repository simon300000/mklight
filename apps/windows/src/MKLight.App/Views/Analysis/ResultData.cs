using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis;

/// <summary>
/// A typed result buffer as the result views read it: name, row-major
/// shape and element access as double (the engine's buffer is not copied;
/// the adapter over MKLight.Engine's ResultBuffer converts per element).
/// </summary>
public interface IResultArray
{
    string Name { get; }
    IReadOnlyList<int> Shape { get; }
    int Count { get; }
    double this[int index] { get; }
}

/// <summary>Everything a result view needs from a finished run (docs/RESULTS.md).</summary>
public sealed class ResultData
{
    public required string Analysis { get; init; }
    /// <summary>The envelope's "result" object.</summary>
    public required JsonNode? Json { get; init; }
    public IReadOnlyList<IResultArray> Buffers { get; init; } = [];
    /// <summary>The system's wavelengths in µm and field count, for colours and labels (may be empty).</summary>
    public IReadOnlyList<double> WavelengthsUm { get; init; } = [];
    public IReadOnlyList<string> FieldLabels { get; init; } = [];
    /// <summary>The settings the engine recorded for the run (<c>run_info.settings</c>): the values automatic settings took, the CRA sensor limit, the image simulation's source.</summary>
    public JsonNode? Settings { get; init; }
    public IResultArray? Buffer(string name) => Buffers.FirstOrDefault(b => b.Name == name);
}

/// <summary>A view's summary for the pane footer.</summary>
public sealed record ResultSummary(string KeyNumbers);
