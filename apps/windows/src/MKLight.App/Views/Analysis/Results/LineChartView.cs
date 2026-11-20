using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// Every line-plot result (fans, MTF, field curvature, …): the legend over
/// groups of panels stacked vertically, each panel a <see cref="LinePlot"/>.
/// Groups share the height; below <see cref="ChartModel.MinGroupHeight"/>
/// the plots scroll.
/// </summary>
public sealed partial class LineChartView : UserControl
{
    const double GroupLabel = 18, PanelGap = 12, GroupGap = 8, Pad = 8;
    readonly ChartModel _model;
    readonly ScrollViewer _scroll;
    readonly PlotCanvas _canvas;

    public LineChartView(ChartModel model)
    {
        _model = model;
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw);
        _scroll = new ScrollViewer
        {
            Content = _canvas,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
        };
        var root = new Grid { Padding = new Thickness(Pad, 4, Pad, Pad), RowSpacing = 6 };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        var legend = new Legend(model.Legend) { Margin = new Thickness(2, 0, 0, 0) };
        root.Children.Add(legend);
        Grid.SetRow(_scroll, 1);
        root.Children.Add(_scroll);
        Content = root;
        _scroll.SizeChanged += (_, e) => Fit(e.NewSize);
        AutomationProperties();
    }

    void AutomationProperties()
    {
        var titles = _model.Panels.Select(p => p.Title ?? p.YTitle).Distinct();
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(_canvas, "Plot: " + string.Join(", ", titles));
    }

    /// <summary>Groups share the viewport's height, never less than the model's minimum each.</summary>
    void Fit(Size viewport)
    {
        var n = Math.Max(_model.Groups.Count, 1);
        var each = Math.Max(_model.MinGroupHeight, (viewport.Height - GroupGap * (n - 1)) / n);
        _canvas.Height = Math.Max(viewport.Height, each * n + GroupGap * (n - 1));
        _canvas.Invalidate();
    }

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var n = Math.Max(_model.Groups.Count, 1);
        var gh = (size.Height - GroupGap * (n - 1)) / n;
        for (var k = 0; k < _model.Groups.Count; k++)
        {
            var g = _model.Groups[k];
            var top = k * (gh + GroupGap);
            if (g.Label is not null)
            {
                PlotText.Draw(ds, g.Label, 2, top + GroupLabel / 2, style.Title, 11, TextAnchor.Start, bold: true);
                top += GroupLabel;
            }
            var h = gh - (g.Label is null ? 0 : GroupLabel);
            var m = Math.Max(g.Panels.Count, 1);
            var w = (size.Width - PanelGap * (m - 1)) / m;
            for (var p = 0; p < g.Panels.Count; p++)
                DrawPanel(ds, g.Panels[p], new Rect(p * (w + PanelGap), top, Math.Max(w, 0), Math.Max(h, 0)), style);
        }
    }

    static readonly CanvasStrokeStyle RuleDash = new() { CustomDashStyle = [3, 3] };

    /// <summary>One panel: the plot, then its rules (Nyquist, best focus, ±Airy) over the data rectangle.</summary>
    public static void DrawPanel(CanvasDrawingSession ds, ChartPanel panel, Rect bounds, PlotStyle style)
    {
        if (bounds.Width < 40 || bounds.Height < 40) return;
        var plot = new LinePlot { Title = panel.Title, XTitle = panel.XTitle, YTitle = panel.YTitle, ZeroLine = panel.ZeroLine };
        foreach (var s in panel.Series)
        {
            var pts = new Vector2[Math.Min(s.X.Length, s.Y.Length)];
            for (var i = 0; i < pts.Length; i++) pts[i] = new Vector2((float)s.X[i], (float)s.Y[i]);
            plot.Series.Add(new PlotSeries(pts, ChartColors.Resolve(s.Color, style.Dark), s.Dashed, s.Color.Role == ColorRole.Neutral ? 1.3f : 1.6f));
        }
        plot.FitData();
        if (panel.XDomain is (double x0, double x1) && x1 > x0) (plot.XMin, plot.XMax) = (x0, x1);
        if (panel.YDomain is (double y0, double y1) && y1 > y0) (plot.YMin, plot.YMax) = (y0, y1);
        var r = plot.Draw(ds, bounds, style);
        if (r.IsEmpty) return;
        foreach (var rule in panel.Rules)
        {
            if (!double.IsFinite(rule.X) || rule.X < plot.XMin || rule.X > plot.XMax) continue;
            var x = (float)(r.X + (rule.X - plot.XMin) / (plot.XMax - plot.XMin) * r.Width);
            if (rule.Dashed) ds.DrawLine(x, (float)r.Y, x, (float)r.Bottom, style.Axis, 1, RuleDash);
            else ds.DrawLine(x, (float)r.Y, x, (float)r.Bottom, style.Axis, 1);
            if (rule.Label is not null)
            {
                var right = x + 60 < r.Right;
                PlotText.Draw(ds, rule.Label, right ? x + 4 : x - 4, r.Y + 8, style.Label, 10, right ? TextAnchor.Start : TextAnchor.End);
            }
        }
    }
}
