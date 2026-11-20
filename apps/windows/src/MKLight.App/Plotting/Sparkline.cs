using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Views.Analysis.Results;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Plotting;

/// <summary>
/// A sparkline in the plots' style: a thin line without axes (a merit history, a value over
/// trials), an optional dashed reference level, and the last value marked. <see cref="Draw"/>
/// draws into any Win2D session; <see cref="SparklineView"/> is the control.
/// </summary>
public static class Sparkline
{
    static readonly CanvasStrokeStyle Dash = new() { CustomDashStyle = [2, 2] };
    static readonly CanvasStrokeStyle Round = new() { LineJoin = CanvasLineJoin.Round, StartCap = CanvasCapStyle.Round, EndCap = CanvasCapStyle.Round };

    /// <param name="color">The line; null takes the plot's title ink (neutral).</param>
    /// <param name="reference">A level drawn dashed across (a target, the start), when inside the range.</param>
    public static void Draw(CanvasDrawingSession ds, Rect bounds, IReadOnlyList<double> values, PlotStyle style,
        Color? color = null, double? reference = null, bool log = false, bool markLast = true, float width = 1.4f)
    {
        const float inset = 2.5f; // room for the end marker
        float x = (float)bounds.X + inset, y = (float)bounds.Y + inset;
        float w = (float)bounds.Width - 2 * inset, h = (float)bounds.Height - 2 * inset;
        if (w <= 2 || h <= 2) return;
        var ink = color ?? style.Title;
        if (reference is double r && SparklineGeometry.Range(values, log) is var (lo, hi) && (!log || r > 0))
        {
            var rv = log ? Math.Log10(r) : r;
            if (rv >= lo && rv <= hi)
            {
                var ry = y + (float)((1 - (rv - lo) / (hi - lo)) * h);
                ds.DrawLine(x, ry, x + w, ry, style.Axis, 1, Dash);
            }
        }
        var runs = SparklineGeometry.Runs(values, x, y, w, h, log);
        foreach (var run in runs)
        {
            if (run.Count == 1) { ds.FillCircle(run[0], width, ink); continue; }
            using var pb = new CanvasPathBuilder(ds);
            pb.BeginFigure(run[0]);
            foreach (var p in run.Skip(1)) pb.AddLine(p);
            pb.EndFigure(CanvasFigureLoop.Open);
            using var g = CanvasGeometry.CreatePath(pb);
            ds.DrawGeometry(g, ink, width, Round);
        }
        if (markLast && runs.LastOrDefault()?.LastOrDefault() is { } last && runs.Count > 0)
        {
            ds.FillCircle(last, 2.6f, style.Surface);
            ds.FillCircle(last, 2f, ink);
        }
    }
}

/// <summary>A sparkline control: give it values (and a name for screen readers); it redraws with its theme.</summary>
public sealed partial class SparklineView : UserControl
{
    readonly PlotCanvas _canvas;
    IReadOnlyList<double> _values = [];
    string _name;

    public SparklineView(string name, double height = 22)
    {
        _name = name;
        IsTabStop = false;
        Height = height;
        _canvas = new PlotCanvas((ds, size, style) => Sparkline.Draw(ds, new Rect(0, 0, size.Width, size.Height), _values, style, Color, Reference, Log));
        Content = _canvas;
        Describe();
    }

    /// <summary>The line's colour; null for the neutral ink.</summary>
    public Color? Color { get; set; }
    public double? Reference { get; set; }
    public bool Log { get; set; }

    public IReadOnlyList<double> Values
    {
        get => _values;
        set { _values = value; Describe(); _canvas.Invalidate(); }
    }

    public string Title
    {
        get => _name;
        set { _name = value; Describe(); }
    }

    void Describe() => AutomationProperties.SetName(this, SparklineGeometry.Describe(_name, _values));
}
