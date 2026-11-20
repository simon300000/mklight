using System.Globalization;
using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.Graphics.Canvas.Text;
using Microsoft.UI.Xaml;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Plotting;

/// <summary>
/// Colours and type of a plot, resolved from the theme: plots are content,
/// so the data is the most prominent thing; grids and labels stay light
/// (docs/APP.md design brief, principle 13).
/// </summary>
public sealed record PlotStyle(bool Dark, Color Grid, Color Axis, Color Label, Color Title, Color Surface)
{
    public const float LabelSize = 10.5f;

    public static PlotStyle For(ElementTheme theme)
    {
        var dark = theme == ElementTheme.Dark;
        return dark
            ? new PlotStyle(true, Color.FromArgb(0x17, 255, 255, 255), Color.FromArgb(0x66, 255, 255, 255),
                Color.FromArgb(0xA6, 255, 255, 255), Color.FromArgb(0xD9, 255, 255, 255), Color.FromArgb(255, 0x1a, 0x1a, 0x19))
            : new PlotStyle(false, Color.FromArgb(0x14, 0, 0, 0), Color.FromArgb(0x59, 0, 0, 0),
                Color.FromArgb(0x99, 0, 0, 0), Color.FromArgb(0xC4, 0, 0, 0), Color.FromArgb(255, 0xfc, 0xfc, 0xfb));
    }
}

public enum TextAnchor { Start, Middle, End }

/// <summary>Text drawing with Segoe UI Variable and tabular figures (every number in the app uses them).</summary>
public static class PlotText
{
    static readonly Dictionary<(float, bool), CanvasTextFormat> Formats = [];
    static readonly CanvasTypography Tabular = MakeTabular();

    static CanvasTypography MakeTabular()
    {
        var t = new CanvasTypography();
        t.AddFeature(CanvasTypographyFeatureName.TabularFigures, 1);
        return t;
    }

    static CanvasTextFormat Format(float size, bool bold)
    {
        if (!Formats.TryGetValue((size, bold), out var f))
        {
            f = new CanvasTextFormat
            {
                FontFamily = "Segoe UI Variable Text",
                FontSize = size,
                FontWeight = bold ? Microsoft.UI.Text.FontWeights.SemiBold : Microsoft.UI.Text.FontWeights.Normal,
                WordWrapping = CanvasWordWrapping.NoWrap,
            };
            Formats[(size, bold)] = f;
        }
        return f;
    }

    public static CanvasTextLayout Layout(ICanvasResourceCreator rc, string text, float size, bool bold = false)
    {
        var layout = new CanvasTextLayout(rc, text, Format(size, bold), 0, 0);
        layout.SetTypography(0, text.Length, Tabular);
        return layout;
    }

    public static Size Measure(ICanvasResourceCreator rc, string text, float size = PlotStyle.LabelSize, bool bold = false)
    {
        using var l = Layout(rc, text, size, bold);
        var b = l.LayoutBounds;
        return new Size(b.Width, b.Height);
    }

    /// <summary>Draw text anchored horizontally at x, vertically centred on y.</summary>
    public static void Draw(CanvasDrawingSession ds, string text, double x, double y, Color color,
        float size = PlotStyle.LabelSize, TextAnchor anchor = TextAnchor.Middle, bool bold = false, double rotation = 0)
    {
        if (string.IsNullOrEmpty(text)) return;
        using var l = Layout(ds, text, size, bold);
        var b = l.LayoutBounds;
        var dx = anchor switch { TextAnchor.Start => 0, TextAnchor.Middle => -b.Width / 2, _ => -b.Width };
        if (rotation != 0)
        {
            var old = ds.Transform;
            ds.Transform = Matrix3x2.CreateRotation((float)rotation) * Matrix3x2.CreateTranslation((float)x, (float)y) * old;
            ds.DrawTextLayout(l, (float)dx, (float)(-b.Height / 2), color);
            ds.Transform = old;
        }
        else
        {
            ds.DrawTextLayout(l, (float)(x + dx), (float)(y - b.Height / 2), color);
        }
    }
}

/// <summary>"Nice" axis ticks (1, 2, 2.5, 5 × 10ⁿ) and their labels in the app's one number format.</summary>
public static class Ticks
{
    public static double Step(double span, int target)
    {
        if (!(span > 0) || !double.IsFinite(span)) return 1;
        var raw = span / Math.Max(target, 1);
        var mag = Math.Pow(10, Math.Floor(Math.Log10(raw)));
        var n = raw / mag;
        var nice = n < 1.5 ? 1 : n < 2.25 ? 2 : n < 3.5 ? 2.5 : n < 7.5 ? 5 : 10;
        return nice * mag;
    }

    public static List<double> Values(double min, double max, int target)
    {
        var list = new List<double>();
        if (!double.IsFinite(min) || !double.IsFinite(max) || max <= min) return list;
        var step = Step(max - min, target);
        var first = Math.Ceiling(min / step - 1e-9) * step;
        for (var v = first; v <= max + step * 1e-9 && list.Count < 50; v += step)
            list.Add(Math.Abs(v) < step * 1e-9 ? 0 : v);
        return list;
    }

    /// <summary>A tick label: decimals from the step, decimal point, no grouping, never "−0".</summary>
    public static string Label(double v, double step)
    {
        var decimals = step >= 1 || step <= 0 ? 0 : Math.Min(6, (int)Math.Ceiling(-Math.Log10(step) - 1e-9));
        if (step > 0 && Math.Abs(step * Math.Pow(10, decimals) - Math.Round(step * Math.Pow(10, decimals))) > 1e-6) decimals++;
        var s = v.ToString("F" + decimals, CultureInfo.InvariantCulture);
        if (s.StartsWith('-') && s.Trim('-', '0', '.').Length == 0) s = s[1..];
        return s.Replace('-', '−');
    }

    /// <summary>A padded data range; a flat range opens around its value.</summary>
    public static (double Min, double Max) Range(IEnumerable<double> values, double pad = 0.05, bool includeZero = false)
    {
        double lo = double.PositiveInfinity, hi = double.NegativeInfinity;
        foreach (var v in values)
        {
            if (!double.IsFinite(v)) continue;
            lo = Math.Min(lo, v); hi = Math.Max(hi, v);
        }
        if (includeZero) { lo = Math.Min(lo, 0); hi = Math.Max(hi, 0); }
        if (!double.IsFinite(lo)) return (0, 1);
        if (hi - lo < 1e-12 * Math.Max(1, Math.Abs(hi))) { var d = Math.Max(Math.Abs(hi) * 0.1, 1e-6); return (lo - d, hi + d); }
        var p = (hi - lo) * pad;
        return (includeZero && lo == 0 ? 0 : lo - p, hi + p);
    }
}

/// <summary>A series of a line plot.</summary>
public sealed record PlotSeries(IReadOnlyList<Vector2> Points, Color Color, bool Dashed = false, float Width = 1.6f, string? Label = null, bool Markers = false);

/// <summary>
/// One cartesian plot: frame, light grid, few ticks, axis titles and line
/// series. Used by the dedicated line-plot views (ray fan, MTF, field
/// curvature, …) and the generic result view.
/// </summary>
public sealed class LinePlot
{
    public double XMin, XMax = 1, YMin, YMax = 1;
    public string? XTitle, YTitle, Title;
    public List<PlotSeries> Series { get; } = [];
    public bool ZeroLine = true;
    /// <summary>Approximate pixels per tick label (the Mac app's ~70 pt per label).</summary>
    public double XTickSpacing = 70, YTickSpacing = 40;

    public void FitData(bool includeZeroY = false, double pad = 0.05)
    {
        var pts = Series.SelectMany(s => s.Points).ToList();
        (XMin, XMax) = Ticks.Range(pts.Select(p => (double)p.X), 0);
        (YMin, YMax) = Ticks.Range(pts.Select(p => (double)p.Y), pad, includeZeroY);
    }

    /// <summary>Draw into <paramref name="bounds"/>; returns the data rectangle.</summary>
    public Rect Draw(CanvasDrawingSession ds, Rect bounds, PlotStyle style)
    {
        var yTicks = Ticks.Values(YMin, YMax, Math.Max(2, (int)(bounds.Height / YTickSpacing)));
        var yStep = yTicks.Count > 1 ? yTicks[1] - yTicks[0] : 1;
        var yLabelW = yTicks.Select(v => PlotText.Measure(ds, Ticks.Label(v, yStep)).Width).DefaultIfEmpty(0).Max();
        var left = bounds.X + yLabelW + 8 + (YTitle is null ? 0 : 16);
        var top = bounds.Y + (Title is null ? 6 : 22);
        var bottom = bounds.Bottom - 18 - (XTitle is null ? 0 : 16);
        var right = bounds.Right - 8;
        if (right - left < 20 || bottom - top < 20) return Rect.Empty;
        var plot = new Rect(left, top, right - left, bottom - top);

        float MapX(double x) => (float)(plot.X + (x - XMin) / (XMax - XMin) * plot.Width);
        float MapY(double y) => (float)(plot.Bottom - (y - YMin) / (YMax - YMin) * plot.Height);

        var xTicks = Ticks.Values(XMin, XMax, Math.Max(2, (int)(plot.Width / XTickSpacing)));
        var xStep = xTicks.Count > 1 ? xTicks[1] - xTicks[0] : 1;
        foreach (var v in xTicks)
        {
            var x = MapX(v);
            ds.DrawLine(x, (float)plot.Y, x, (float)plot.Bottom, style.Grid, 1);
            // End labels stay inside the bounds (centred unless that would cut them off).
            var label = Ticks.Label(v, xStep);
            var half = PlotText.Measure(ds, label).Width / 2;
            PlotText.Draw(ds, label, Math.Clamp(x, bounds.X + half, Math.Max(bounds.X + half, bounds.Right - half)), plot.Bottom + 9, style.Label);
        }
        foreach (var v in yTicks)
        {
            var y = MapY(v);
            ds.DrawLine((float)plot.X, y, (float)plot.Right, y, style.Grid, 1);
            PlotText.Draw(ds, Ticks.Label(v, yStep), plot.X - 5, y, style.Label, anchor: TextAnchor.End);
        }
        if (ZeroLine && YMin < 0 && YMax > 0)
            ds.DrawLine((float)plot.X, MapY(0), (float)plot.Right, MapY(0), style.Axis, 1);
        if (ZeroLine && XMin < 0 && XMax > 0)
            ds.DrawLine(MapX(0), (float)plot.Y, MapX(0), (float)plot.Bottom, style.Axis, 1);
        ds.DrawLine((float)plot.X, (float)plot.Bottom, (float)plot.Right, (float)plot.Bottom, style.Axis, 1);
        ds.DrawLine((float)plot.X, (float)plot.Y, (float)plot.X, (float)plot.Bottom, style.Axis, 1);

        if (Title is not null) PlotText.Draw(ds, Title, plot.X + plot.Width / 2, bounds.Y + 9, style.Title, 11, bold: true);
        if (XTitle is not null) PlotText.Draw(ds, XTitle, plot.X + plot.Width / 2, bounds.Bottom - 7, style.Label, 11);
        if (YTitle is not null) PlotText.Draw(ds, YTitle, bounds.X + 7, plot.Y + plot.Height / 2, style.Label, 11, rotation: -Math.PI / 2);

        using (ds.CreateLayer(1, plot))
        {
            foreach (var s in Series) DrawSeries(ds, s, MapX, MapY);
        }
        return plot;
    }

    static readonly CanvasStrokeStyle DashStyle = new() { DashStyle = CanvasDashStyle.Dash, LineJoin = CanvasLineJoin.Round, DashCap = CanvasCapStyle.Round };
    static readonly CanvasStrokeStyle SolidStyle = new() { LineJoin = CanvasLineJoin.Round, StartCap = CanvasCapStyle.Round, EndCap = CanvasCapStyle.Round };

    static void DrawSeries(CanvasDrawingSession ds, PlotSeries s, Func<double, float> mx, Func<double, float> my)
    {
        if (s.Points.Count == 0) return;
        using var pb = new CanvasPathBuilder(ds);
        var open = false;
        foreach (var p in s.Points)
        {
            if (!float.IsFinite(p.X) || !float.IsFinite(p.Y))
            {
                if (open) { pb.EndFigure(CanvasFigureLoop.Open); open = false; }
                continue;
            }
            var v = new Vector2(mx(p.X), my(p.Y));
            if (!open) { pb.BeginFigure(v); open = true; } else pb.AddLine(v);
        }
        if (open) pb.EndFigure(CanvasFigureLoop.Open);
        using var geo = CanvasGeometry.CreatePath(pb);
        ds.DrawGeometry(geo, s.Color, s.Width, s.Dashed ? DashStyle : SolidStyle);
        if (s.Markers)
            foreach (var p in s.Points.Where(p => float.IsFinite(p.X) && float.IsFinite(p.Y)))
                ds.FillCircle(mx(p.X), my(p.Y), 2.2f, s.Color);
    }
}
