using Microsoft.UI.Xaml;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// The view of a result: the dedicated view of its analysis when there is
/// one and it can read the result, else the generic view. Never fails: an
/// unreadable result is shown as data.
/// </summary>
public static class ResultViews
{
    /// <summary>The analyses with a dedicated view.</summary>
    public static IReadOnlyList<string> Dedicated { get; } = new[] { "spot_diagram" }.Concat(LineReaders.Analyses).ToArray();

    /// <param name="theme">
    /// The pane's theme. Views draw in their own <c>ActualTheme</c> (inherited
    /// from the pane) and redraw when it changes, so this is informational.
    /// </param>
    /// <param name="spotScale">The spot diagram's box scale to start with (the pane may keep it with its layout).</param>
    public static (UIElement View, ResultSummary Summary) Create(ResultData data, ElementTheme theme, SpotScale spotScale = SpotScale.Grid)
    {
        string keys;
        try { keys = KeyNumbers.Text(data); }
        catch (Exception) { keys = ""; }
        UIElement? view = null;
        try
        {
            if (data.Analysis == "spot_diagram")
            {
                if (SpotGridData.Read(data) is SpotGridData grid) view = new SpotDiagramView(grid, spotScale);
            }
            else if (LineReaders.Read(data) is ChartModel model && model.Panels.Any(p => p.Series.Count > 0))
            {
                view = new LineChartView(model);
            }
        }
        catch (Exception)
        {
            view = null; // a result of an unexpected shape is shown as data
        }
        view ??= new GenericResultView(data);
        return (view, new ResultSummary(keys));
    }
}
