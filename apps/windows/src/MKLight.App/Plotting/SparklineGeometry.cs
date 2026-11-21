using System.Globalization;
using System.Numerics;
using MKLight.App.Support;

namespace MKLight.App.Plotting;

/// <summary>
/// The geometry of a sparkline (no Win2D, so it can be tested): the values (non-finite ones break
/// the line) mapped into a rectangle, the y range padded a little and never empty, an optional
/// log scale for quantities spanning decades (a merit function), and the spoken summary.
/// </summary>
public static class SparklineGeometry
{
    /// <summary>The y range drawn: the finite values' (positive ones on a log scale), padded by <paramref name="pad"/> of the span; a flat line sits in the middle.</summary>
    public static (double Min, double Max)? Range(IReadOnlyList<double> values, bool log = false, double pad = 0.08)
    {
        var v = values.Where(x => double.IsFinite(x) && (!log || x > 0)).Select(x => log ? Math.Log10(x) : x).ToList();
        if (v.Count == 0) return null;
        double lo = v.Min(), hi = v.Max();
        if (hi - lo < 1e-12 * Math.Max(1, Math.Abs(hi))) { var d = Math.Max(Math.Abs(hi) * 0.05, 1e-9); return (lo - d, hi + d); }
        var p = (hi - lo) * pad;
        return (lo - p, hi + p);
    }

    /// <summary>
    /// The runs of points to draw in (x, y, width, height), one run per stretch of finite values;
    /// x spreads the values' indices across the width (a single value sits in the middle), y grows downwards.
    /// </summary>
    public static List<List<Vector2>> Runs(IReadOnlyList<double> values, float x, float y, float width, float height, bool log = false)
    {
        var runs = new List<List<Vector2>>();
        if (Range(values, log) is not var (lo, hi)) return runs;
        List<Vector2>? run = null;
        for (var i = 0; i < values.Count; i++)
        {
            var v = values[i];
            if (!double.IsFinite(v) || (log && v <= 0)) { run = null; continue; }
            var t = values.Count == 1 ? 0.5 : i / (double)(values.Count - 1);
            var u = ((log ? Math.Log10(v) : v) - lo) / (hi - lo);
            if (run is null) runs.Add(run = []);
            run.Add(new Vector2(x + (float)(t * width), y + (float)((1 - u) * height)));
        }
        return runs;
    }

    /// <summary>"Merit, 24 values from 3.2 to 0.81, lowest 0.79" for screen readers.</summary>
    public static string Describe(string name, IReadOnlyList<double> values)
    {
        var v = values.Where(double.IsFinite).ToList();
        if (v.Count == 0) return Loc.F("Results_Sparkline_NoValues", name);
        static string G(double x) => x.ToString("G3", CultureInfo.InvariantCulture).Replace("-", "−");
        var s = v.Count == 1 ? Loc.F("Results_Sparkline_OneValue", name, G(v[0])) : Loc.F("Results_Sparkline_Values", name, v.Count, G(v[0]), G(v[^1]));
        return v.Count > 2 ? Loc.F("Results_Sparkline_Extremes", s, G(v.Min()), G(v.Max())) : s;
    }
}
