using MKLight.App.Support;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// A spoken summary of a line chart for screen readers (the plot's help text): per panel its
/// title, the number of curves, the ranges of both axes and the curves' values half way along
/// the x axis (e.g. "MTF: 6 curves, frequency 0 to 100, modulation 0.08 to 1; at 50: 0.62, 0.58, …").
/// </summary>
public static class ChartSummary
{
    const int MaxValues = 6;

    public static string Describe(ChartModel model) => string.Join(". ", model.Panels.Select(Describe).Where(s => s.Length > 0));

    public static string Describe(ChartPanel panel)
    {
        var series = panel.Series.Where(s => s.X.Length > 0 && s.Y.Length > 0).ToList();
        var title = panel.Title ?? panel.YTitle;
        if (series.Count == 0) return title.Length == 0 ? "" : Loc.F("Results_Summary_NoData", title);
        var xs = series.SelectMany(s => s.X).Where(double.IsFinite).ToList();
        var ys = series.SelectMany(s => s.Y).Where(double.IsFinite).ToList();
        var parts = new List<string> { series.Count == 1 ? Loc.S("Results_Summary_Curves_One") : Loc.F("Results_Summary_Curves_Many", series.Count) };
        if (xs.Count > 0) parts.Add(Range(panel.XTitle, xs.Min(), xs.Max()));
        if (ys.Count > 0) parts.Add(Range(panel.YTitle, ys.Min(), ys.Max()));
        var list = string.Join(", ", parts);
        var text = title.Length == 0 ? list : Loc.F("Results_Summary_Titled", title, list);
        if (xs.Count > 0)
        {
            var mid = (xs.Min() + xs.Max()) / 2;
            var at = series.Take(MaxValues).Select(s => At(s, mid)).Where(double.IsFinite).Select(Fmt.G3).ToList();
            if (at.Count > 0) text = Loc.F("Results_Summary_At", text, Fmt.G3(mid), string.Join(", ", at) + (series.Count > MaxValues ? ", …" : ""));
        }
        return text;
    }

    static string Range(string axis, double lo, double hi)
    {
        var name = axis.Length == 0 ? "" : axis.ToLowerInvariant() + " ";
        return lo == hi ? name + Fmt.G3(lo) : name + Loc.F("Results_Summary_Range", Fmt.G3(lo), Fmt.G3(hi));
    }

    /// <summary>The curve's value at <paramref name="x"/>, interpolated linearly between its points (x sorted either way); NaN outside.</summary>
    public static double At(ChartSeries s, double x)
    {
        var n = Math.Min(s.X.Length, s.Y.Length);
        for (var i = 0; i + 1 < n; i++)
        {
            double x0 = s.X[i], x1 = s.X[i + 1];
            if (!double.IsFinite(x0) || !double.IsFinite(x1) || x < Math.Min(x0, x1) || x > Math.Max(x0, x1)) continue;
            if (x0 == x1) return s.Y[i];
            return s.Y[i] + (s.Y[i + 1] - s.Y[i]) * (x - x0) / (x1 - x0);
        }
        return n == 1 && s.X[0] == x ? s.Y[0] : double.NaN;
    }
}
