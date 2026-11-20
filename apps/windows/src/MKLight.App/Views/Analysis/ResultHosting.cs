using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
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

    /// <summary>
    /// The result view factory (Results/ResultViews.Create once the dedicated
    /// views are in); until then every result shows as a short data summary.
    /// </summary>
    public static Func<ResultData, ElementTheme, (UIElement View, ResultSummary Summary)> CreateView { get; set; } = Fallback;

    /// <summary>The settings form factory (Settings/SettingsForm.Create once it is in).</summary>
    public static Func<JsonObject, JsonObject, SettingsContextData, Action<JsonObject>, UIElement>? CreateSettingsForm { get; set; }

    /// <summary>The one-line summary of settings that differ from the defaults.</summary>
    public static Func<JsonObject, JsonObject, string>? DescribeSettings { get; set; }

    static (UIElement, ResultSummary) Fallback(ResultData data, ElementTheme theme)
    {
        var keys = data.Json is JsonObject o ? string.Join(", ", o.Select(kv => kv.Key).Take(8)) : "";
        var text = new TextBlock
        {
            Text = $"Result fields: {keys}\nBuffers: {string.Join(", ", data.Buffers.Select(b => $"{b.Name} [{string.Join("×", b.Shape)}]"))}",
            Style = (Style)Application.Current.Resources["SecondaryText"], TextWrapping = TextWrapping.Wrap, Margin = new Thickness(12),
        };
        return (text, new ResultSummary(""));
    }
}

/// <summary>What a settings form needs to know about the design (labels for fields, wavelengths and surfaces).</summary>
public sealed record SettingsContextData(IReadOnlyList<string> Fields, IReadOnlyList<double> WavelengthsUm, IReadOnlyList<string> Surfaces);
