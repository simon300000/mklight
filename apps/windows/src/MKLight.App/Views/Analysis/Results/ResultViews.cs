using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// The view of a result: the dedicated view of its analysis when there is
/// one and it can read the result, else the generic view. Never fails: an
/// unreadable result is shown as data.
/// </summary>
public static class ResultViews
{
    static readonly string[] Own =
    [
        "spot_diagram", "first_order", "seidel", "psf", "wavefront_map", "pupil_aberration", "relative_illumination_map", "chief_ray_angle",
        "glass_map", "glass_info", "image_simulation", "grid_distortion", "footprint", "ghost_paths", "ghost_irradiance", "stray_light",
        "polarization_pupil_map", "system_transmission", "validate",
    ];

    /// <summary>The analyses with a dedicated view.</summary>
    public static IReadOnlyList<string> Dedicated { get; } = Own.Concat(LineReaders.Analyses).ToArray();

    /// <param name="theme">
    /// The pane's theme. Views draw in their own <c>ActualTheme</c> (inherited
    /// from the pane) and redraw when it changes, so this is informational.
    /// </param>
    /// <param name="spotScale">The spot diagram's box scale to start with (the pane may keep it with its layout).</param>
    public static (UIElement View, ResultSummary Summary) Create(ResultData data, ElementTheme theme, SpotScale spotScale = SpotScale.Grid)
    {
        string keys;
        try { keys = KeyNumbers.Compared(data); }
        catch (Exception) { keys = ""; }
        UIElement? view = null;
        try
        {
            view = Dedicated_(data, spotScale);
        }
        catch (Exception)
        {
            view = null; // a result of an unexpected shape is shown as data
        }
        view ??= new GenericResultView(data);
        return (view, new ResultSummary(keys));
    }

    static UIElement? Dedicated_(ResultData d, SpotScale spotScale) => d.Analysis switch
    {
        "spot_diagram" => SpotGridData.Read(d) is { } grid ? new SpotDiagramView(grid, spotScale, Baseline(d, SpotGridData.Read)) : null,
        "first_order" => FirstOrderData.Read(d.Json) is { } fo ? new FirstOrderView(fo) : null,
        "seidel" => SeidelData.Read(d.Json) is { } s ? new SeidelView(s) : null,
        "psf" => PsfData.Read(d) is { } psf ? new PsfView(psf) : null,
        "wavefront_map" => WavefrontData.Read(d) is { } wf ? new WavefrontView(wf) : null,
        "pupil_aberration" or "relative_illumination_map" or "chief_ray_angle" => FieldMapData.Read(d) is { } fm ? FieldMapViews.Create(d.Analysis, fm) : null,
        "glass_map" => GlassMapData.Read(d) is { Glasses.Count: > 0 } gm ? new GlassMapView(gm) : null,
        "glass_info" => GlassInfoData.Read(d) is { } gi ? new GlassDataView(gi) : null,
        "image_simulation" => ImageSimulationData.Read(d) is { } im ? new ImageSimulationView(d, im) : null,
        "grid_distortion" => GridDistortionData.Read(d) is { } gd ? new GridDistortionView(gd) : null,
        "footprint" => FootprintData.Read(d) is { } fp ? new FootprintView(fp, d.WavelengthsUm) : null,
        "ghost_paths" => GhostPathsData.Read(d) is { } gp ? new GhostPathsView(gp) : null,
        "ghost_irradiance" => SensorMapsData.GhostIrradiance(d) is { } gi2 ? new SensorMapView(gi2) : null,
        "stray_light" => SensorMapsData.StrayLight(d) is { } sl ? new SensorMapView(sl) : null,
        "polarization_pupil_map" => PolarizationData.Read(d) is { } pol ? new PolarizationView(pol) : null,
        "system_transmission" => TransmissionData.Read(d) is { } tr ? new TransmissionView(tr, d.WavelengthsUm) : null,
        "validate" => ValidationData.Read(d.Json) is { } v ? new ValidationView(v) : null,
        _ => LineReaders.Read(d) is ChartModel model && model.Panels.Any(p => p.Series.Count > 0) ? new LineChartView(model, Baseline(d, LineReaders.Read)) : null,
    };

    /// <summary>The design's result read the same way while a candidate is previewed (null otherwise, or when it cannot be read).</summary>
    static T? Baseline<T>(ResultData d, Func<ResultData, T?> read) where T : class
    {
        if (d.Baseline is not { } b) return null;
        try { return read(b); }
        catch (Exception) { return null; }
    }

    /// <summary>
    /// The settings the result was run with, from the pane that shows <paramref name="view"/>
    /// (null when it is not in a pane yet). Views that need them (the CRA sensor limit, the
    /// image simulation's source) ask once loaded.
    /// </summary>
    internal static JsonNode? RunSettings(FrameworkElement view)
    {
        DependencyObject? e = view;
        while (e is not null)
        {
            if (e is AnalysisPaneView pane) return pane.Pane.Output?.RunInfo.Settings;
            e = VisualTreeHelper.GetParent(e);
        }
        return null;
    }
}
