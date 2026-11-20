using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// The signed Cartesian field maps (<c>pupil_aberration</c>, <c>relative_illumination_map</c>,
/// <c>chief_ray_angle</c>), all read by <see cref="FieldMapData"/>: a grid run as a heat map
/// over the normalised field, interpolated between the samples (dots), masked cells hatched;
/// a sweep as a curve. A column picker chooses the quantity; its definition is the help.
/// </summary>
static class FieldMapViews
{
    public static UIElement Create(string analysis, FieldMapData d) => analysis switch
    {
        "relative_illumination_map" => new RelativeIlluminationMapView(d),
        "chief_ray_angle" when d.ColumnIndex("incidence") is not null => new ChiefRayAngleView(d),
        _ => new PupilAberrationView(d),
    };

    /// <summary>The column menu (its definition as tooltip) and the definition as a caption.</summary>
    public static FrameworkElement Header(FieldMapData d, IReadOnlyList<int> columns, int selected, Action<int> changed)
    {
        var p = new FlowPanel { Spacing = 10, LineSpacing = 2 };
        var c = new ComboBox
        {
            ItemsSource = columns.Select(k => d.Columns[k].TitleWithUnit).ToList(), SelectedIndex = Math.Max(columns.ToList().IndexOf(selected), 0),
            FontSize = 12, MinWidth = 0, MinHeight = 26, Padding = new Thickness(8, 2, 6, 3),
        };
        AutomationProperties.SetName(c, "Quantity");
        var def = Kit.Caption(d.Columns[selected].Definition, wrap: true);
        def.VerticalAlignment = VerticalAlignment.Center;
        def.MaxWidth = 420;
        ToolTipService.SetToolTip(c, d.Columns[selected].Definition);
        c.SelectionChanged += (_, _) =>
        {
            if (c.SelectedIndex < 0) return;
            var k = columns[c.SelectedIndex];
            def.Text = d.Columns[k].Definition;
            ToolTipService.SetToolTip(c, d.Columns[k].Definition);
            changed(k);
        };
        p.Children.Add(c);
        p.Children.Add(def);
        return p;
    }

    /// <summary>A column: the map (grid runs) or the curve (sweeps); a column without values says so.</summary>
    public static UIElement Content(FieldMapData d, int column, IReadOnlyList<double>? contours = null, HeatRangeMode mode = HeatRangeMode.Data)
    {
        var col = d.Columns[column];
        if (d.Range(column) is not var (lo, hi))
            return SeidelView.Message("No values", $"{col.Title} is undefined at every sample: {col.Definition}.");
        if (d.Layout == FieldMapData.Layouts.Map)
        {
            var grid = d.Grid(column);
            var signed = d.IsSigned(column);
            var plot = new MapPlot
            {
                Image = new MapImage(grid.Upsampled(192), cells: false),
                Samples = grid,
                Scale = MapScale.ForQuantity(lo, hi, signed, mode, col.IsFraction),
                Unit = col.UnitLabel,
                ShowNodes = true,
                Format = col.Format,
                Contours = (contours ?? []).Select(l => (l, Readers.Contours.MapSegments(grid, l))).ToList(),
                HoverText = (f, p) => MapPlot.NodeHover(f, p, grid, col.Format),
            };
            var canvas = new MapCanvas(() => plot, $"{col.Title} map over the normalized field, {d.Nx} by {d.Ny} samples, from {col.Format(lo)} to {col.Format(hi)}");
            return canvas;
        }
        var sweep = d.Sweep(column);
        var signedSweep = d.IsSigned(column);
        var chart = new PlotCanvas((ds, size, style) =>
        {
            var f = new Frame { XTitle = d.Layout == FieldMapData.Layouts.SweepX ? "Hx" : "Hy", YTitle = col.TitleWithUnit, ZeroLine = signedSweep, YTickSpacing = 40 };
            (f.XMin, f.XMax) = FieldMapData.Span(sweep.Select(p => p.H));
            (f.YMin, f.YMax) = Ticks.Range(sweep.Select(p => p.V), 0.06, signedSweep);
            if (f.Draw(ds, f.Inside(ds, new Rect(0, 0, size.Width, size.Height)), style).IsEmpty) return;
            Curves.Line(ds, f, sweep.Select(p => (p.H, p.V)).ToArray(), Palette.Field(0, style.Dark), 1.8f);
            Curves.Dots(ds, f, sweep.Select(p => (p.H, p.V)), Palette.Field(0, style.Dark), 2.2f);
        });
        AutomationProperties.SetName(chart, $"{col.Title} along {(d.Layout == FieldMapData.Layouts.SweepX ? "Hx" : "Hy")}, from {col.Format(lo)} to {col.Format(hi)}");
        return chart;
    }

    /// <summary>A view of a header row, optional control rows and the content filling the rest.</summary>
    public static Grid Shell(params UIElement?[] rows)
    {
        var g = new Grid { Padding = new Thickness(10, 6, 10, 8), RowSpacing = 6 };
        var shown = rows.OfType<UIElement>().ToList();
        for (var i = 0; i < shown.Count; i++)
        {
            g.RowDefinitions.Add(new RowDefinition { Height = i == shown.Count - 1 ? new GridLength(1, GridUnitType.Star) : GridLength.Auto });
            Grid.SetRow((FrameworkElement)shown[i], i);
            g.Children.Add(shown[i]);
        }
        return g;
    }
}

/// <summary>Line and marker drawing on a <see cref="Frame"/> (NaN breaks a line).</summary>
static class Curves
{
    public static void Line(CanvasDrawingSession ds, Frame f, (double X, double Y)[] pts, Color c, float width = 1.6f, CanvasStrokeStyle? stroke = null)
    {
        using var layer = ds.CreateLayer(1, MapParts.Inflate(f.Plot, 1, 1));
        using var pb = new CanvasPathBuilder(ds);
        var open = false;
        foreach (var (x, y) in pts)
        {
            if (!double.IsFinite(x) || !double.IsFinite(y)) { if (open) { pb.EndFigure(CanvasFigureLoop.Open); open = false; } continue; }
            var v = new Vector2(f.X(x), f.Y(y));
            if (!open) { pb.BeginFigure(v); open = true; } else pb.AddLine(v);
        }
        if (open) pb.EndFigure(CanvasFigureLoop.Open);
        using var g = CanvasGeometry.CreatePath(pb);
        ds.DrawGeometry(g, c, width, stroke ?? Round);
    }

    public static void Dots(CanvasDrawingSession ds, Frame f, IEnumerable<(double X, double Y)> pts, Color c, float r = 2.2f)
    {
        foreach (var (x, y) in pts)
            if (double.IsFinite(x) && double.IsFinite(y)) ds.FillCircle(f.X(x), f.Y(y), r, c);
    }

    public static readonly CanvasStrokeStyle Round = new() { LineJoin = CanvasLineJoin.Round, StartCap = CanvasCapStyle.Round, EndCap = CanvasCapStyle.Round };
    public static readonly CanvasStrokeStyle Dashed = new() { CustomDashStyle = [4, 3], LineJoin = CanvasLineJoin.Round };
}

/// <summary><c>pupil_aberration</c>: a chosen column as a map over the field (diverging for signed quantities), or a curve for a sweep.</summary>
public sealed partial class PupilAberrationView : UserControl
{
    readonly ContentControl _content = new() { IsTabStop = false, HorizontalContentAlignment = HorizontalAlignment.Stretch, VerticalContentAlignment = VerticalAlignment.Stretch };

    public PupilAberrationView(FieldMapData d)
    {
        IsTabStop = false;
        var all = Enumerable.Range(0, d.Columns.Count).ToList();
        var first = d.FirstVaryingColumn();
        _content.Content = FieldMapViews.Content(d, first);
        Content = FieldMapViews.Shell(FieldMapViews.Header(d, all, first, k => _content.Content = FieldMapViews.Content(d, k)), _content);
    }
}

/// <summary>
/// <c>relative_illumination_map</c>: the map with iso-illumination lines, a colour range over
/// the data or 0 to 1, and the illumination against image radius beside it (below in narrow panes).
/// </summary>
public sealed partial class RelativeIlluminationMapView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    readonly FieldMapData _d;
    readonly ContentControl _content = new() { IsTabStop = false, HorizontalContentAlignment = HorizontalAlignment.Stretch, VerticalContentAlignment = VerticalAlignment.Stretch };
    readonly Choice _range;
    int _column;
    HeatRangeMode _mode = HeatRangeMode.Data;

    public RelativeIlluminationMapView(FieldMapData d)
    {
        _d = d;
        IsTabStop = false;
        _column = d.ColumnIndex("relative_illumination") ?? 0;
        _range = new Choice("Color scale", ["Data range", "0 to 1"], menuBelow: 0);
        ToolTipService.SetToolTip(_range, "Colors over the map's own range (small differences show), or over 0 to 1 (to compare designs)");
        _range.Changed += k => { _mode = k == 1 ? HeatRangeMode.Unit : HeatRangeMode.Data; Build(); };
        _opts["unit"] = () => _range.Select(1);
        var header = FieldMapViews.Header(d, Enumerable.Range(0, d.Columns.Count).ToList(), _column, k => { _column = k; Build(); });
        Content = FieldMapViews.Shell(header, _range, _content);
        Build();
    }

    void Build()
    {
        var col = _d.Columns[_column];
        var isMap = _d.Layout == FieldMapData.Layouts.Map;
        _range.Visibility = isMap && col.IsFraction ? Visibility.Visible : Visibility.Collapsed;
        if (!isMap || _d.Range(_column) is null) { _content.Content = FieldMapViews.Content(_d, _column); return; }
        var levels = col.Name == "relative_illumination" && _d.Range(_column) is var (lo, hi) ? Readers.Contours.IlluminationLevels(lo, hi) : [];
        var body = new SideBySide { Breakpoint = 600, SideWidth = 0, StackedShare = 0.6, FirstHeight = w => w - 30 };
        body.Children.Add(FieldMapViews.Content(_d, _column, levels, _mode));
        body.Children.Add(Radial());
        _content.Content = body;
    }

    UIElement Radial()
    {
        var col = _d.Columns[_column];
        var pts = _d.Radial(_column).Select(p => (p.R, p.V)).ToArray();
        var line = _d.IsRadial(_column, 1e-2);
        var unit = col.IsFraction && _mode == HeatRangeMode.Unit;
        var g = new Grid { RowSpacing = 2 };
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        g.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        g.Children.Add(Kit.Text($"{col.Title} vs image radius", "PanelTitleText", 11));
        var chart = new PlotCanvas((ds, size, style) =>
        {
            var f = new Frame { XTitle = "Image radius (mm)", YTickSpacing = 34 };
            (f.XMin, f.XMax) = Ticks.Range(pts.Select(p => p.Item1).Append(0), 0.03);
            f.XMin = Math.Max(f.XMin, 0) == 0 ? 0 : f.XMin;
            (f.YMin, f.YMax) = unit ? (0, 1.02) : Ticks.Range(pts.Select(p => p.Item2), 0.08);
            if (f.Draw(ds, f.Inside(ds, new Rect(0, 0, size.Width, size.Height), top: 8), style).IsEmpty) return;
            var c = Palette.Field(0, style.Dark);
            if (line) Curves.Line(ds, f, pts, c, 1.6f);
            Curves.Dots(ds, f, pts, c, line ? 2f : 2.6f);
        });
        AutomationProperties.SetName(chart, $"{col.Title} versus image radius, {pts.Length} samples");
        Grid.SetRow(chart, 1);
        g.Children.Add(chart);
        return g;
    }
}

/// <summary>
/// <c>chief_ray_angle</c>: the incidence against image height with the sensor limit and the
/// margin between them (red where violated); a map over the field for asymmetric systems.
/// </summary>
public sealed partial class ChiefRayAngleView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    readonly FieldMapData _d;
    readonly int _incidence;
    readonly ContentControl _content = new() { IsTabStop = false, HorizontalContentAlignment = HorizontalAlignment.Stretch, VerticalContentAlignment = VerticalAlignment.Stretch };
    readonly ContentControl _controls = new() { IsTabStop = false, HorizontalContentAlignment = HorizontalAlignment.Stretch };
    (double R, double Deg)[] _limit;
    bool _map;
    int _column;

    public ChiefRayAngleView(FieldMapData d)
    {
        _d = d;
        IsTabStop = false;
        _incidence = d.ColumnIndex("incidence")!.Value;
        _column = _incidence;
        _limit = d.LimitTable(null);
        _map = d.Layout == FieldMapData.Layouts.Map && !d.IsRadial(_incidence, 1e-3);
        Choice? display = null;
        if (d.Layout == FieldMapData.Layouts.Map)
        {
            display = new Choice("Display", ["Curve", "Map"], _map ? 1 : 0, menuBelow: 0);
            display.Changed += k => { _map = k == 1; Build(); };
            _opts["map"] = () => display.Select(1);
            _opts["curve"] = () => display.Select(0);
        }
        var top = new FlowPanel { Spacing = 12, LineSpacing = 4 };
        if (display is not null) top.Children.Add(display);
        top.Children.Add(_controls);
        Content = FieldMapViews.Shell(top, _content);
        Build();
        // The run's own sensor-limit table (the limit column only samples it).
        Loaded += (_, _) =>
        {
            if (ResultViews.RunSettings(this) is { } s)
            {
                _limit = d.LimitTable(s);
                Build();
            }
        };
    }

    void Build()
    {
        if (_map && _d.Layout == FieldMapData.Layouts.Map)
        {
            var cols = Enumerable.Range(0, _d.Columns.Count).Where(c => _d.Columns[c].Name != "limit" && _d.Range(c) is not null).ToList();
            if (!cols.Contains(_column)) _column = _incidence;
            _controls.Content = FieldMapViews.Header(_d, cols, _column, k => { _column = k; _content.Content = FieldMapViews.Content(_d, _column); });
            _content.Content = FieldMapViews.Content(_d, _column);
            return;
        }
        _controls.Content = Legend();
        var chart = new PlotCanvas(DrawCurve);
        AutomationProperties.SetName(chart, "Chief ray angle versus image height" + (_limit.Length == 0 ? "" : ", with the sensor limit") + ". " + Summary());
        _content.Content = chart;
    }

    FrameworkElement Legend()
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var p = new List<LegendEntry> { new("Chief ray incidence", SeriesColor.Field(0)) };
        if (_limit.Length > 0) p.Add(new LegendEntry("Sensor limit", SeriesColor.Neutral, LegendMark.Dashed));
        var row = new FlowPanel { Spacing = 12, LineSpacing = 3 };
        row.Children.Add(new Legend(p));
        if (_limit.Length > 0)
        {
            row.Children.Add(Kit.LegendItem(Kit.Swatch(Palette.Field(5, dark), opacity: 0.3), "Margin"));
            row.Children.Add(Kit.LegendItem(Kit.Swatch(Palette.Field(7, dark), opacity: 0.5), "Violation"));
        }
        return row;
    }

    double[] Heights => _d.Layout switch
    {
        FieldMapData.Layouts.SweepX => _d.Image.Select(p => p.X).ToArray(),
        FieldMapData.Layouts.SweepY => _d.Image.Select(p => p.Y).ToArray(),
        _ => _d.ImageRadius,
    };

    List<(double H, double Cra, double Margin)> Samples()
    {
        var h = Heights;
        var m = _d.ColumnIndex("margin");
        return Enumerable.Range(0, _d.Count)
            .Select(k => (h[k], _d.Value(k, _incidence), m is int mc ? _d.Value(k, mc) : double.NaN))
            .Where(s => double.IsFinite(s.Item1) && double.IsFinite(s.Item2))
            .OrderBy(s => s.Item1).ToList();
    }

    /// <summary>The curve: a sweep in order, a symmetric map's unique radii; null when the incidence is not a function of radius.</summary>
    (double X, double Y)[]? CurvePoints(List<(double H, double Cra, double Margin)> s)
    {
        if (_d.Layout != FieldMapData.Layouts.Map) return s.Select(p => (p.H, p.Cra)).ToArray();
        if (!_d.IsRadial(_incidence)) return null;
        var o = new List<(double, double)>();
        foreach (var p in s) if (o.Count == 0 || Math.Abs(o[^1].Item1 - p.H) > 1e-9) o.Add((p.H, p.Cra));
        return o.ToArray();
    }

    string Summary()
    {
        var s = Samples();
        if (s.Count == 0) return "No values";
        var worst = s.MaxBy(p => p.Cra);
        var t = $"Largest {Fmt.Fixed(worst.Cra, 2)} degrees at {Fmt.G3(worst.H)} millimeters";
        if (_limit.Length > 0)
        {
            var bad = s.Count(p => p.Margin < 0);
            t += bad == 0 ? ", within the sensor limit" : Kit.Inv($", {bad} samples exceed the sensor limit");
        }
        return t;
    }

    void DrawCurve(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var s = Samples();
        if (s.Count == 0) return;
        var line = CurvePoints(s);
        var map = _d.Layout == FieldMapData.Layouts.Map;
        var (x0, x1) = FieldMapData.Span(s.Select(p => p.H).Concat(map ? [0.0] : []));
        // The limit table over the plotted heights (both signs for sweeps), held at its ends.
        var xs = new SortedSet<double> { x0, x1 };
        if (x0 <= 0 && x1 >= 0) xs.Add(0);
        foreach (var (r, _) in _limit) foreach (var x in new[] { r, -r }) if (x >= x0 && x <= x1) xs.Add(x);
        var limit = _limit.Length == 0 ? [] : xs.Select(x => (x, FieldMapData.Limit(_limit, Math.Abs(x)))).ToArray();
        var ymax = s.Select(p => p.Cra).Concat(limit.Select(p => p.Item2)).Where(double.IsFinite).DefaultIfEmpty(1).Max();
        var f = new Frame
        {
            XTitle = map ? "Image radius (mm)" : _d.Layout == FieldMapData.Layouts.SweepX ? "Image height x (mm)" : "Image height y (mm)",
            YTitle = "Chief ray angle (°)", XMin = x0, XMax = x1, YMin = 0, YMax = Math.Max(ymax * 1.08, 1e-3), YTickSpacing = 56,
        };
        if (f.Draw(ds, f.Inside(ds, new Rect(0, 0, size.Width, size.Height)), style).IsEmpty) return;
        var margin = Palette.Field(5, style.Dark);
        var violation = Palette.Field(7, style.Dark);
        if (line is not null)
            foreach (var b in FieldMapData.Bands(line, _limit))
            {
                using var pb = new CanvasPathBuilder(ds);
                pb.BeginFigure(f.X(b.Points[0].X), f.Y(b.Points[0].Hi));
                foreach (var p in b.Points.Skip(1)) pb.AddLine(f.X(p.X), f.Y(p.Hi));
                foreach (var p in Enumerable.Reverse(b.Points)) pb.AddLine(f.X(p.X), f.Y(p.Lo));
                pb.EndFigure(CanvasFigureLoop.Closed);
                using var g = CanvasGeometry.CreatePath(pb);
                ds.FillGeometry(g, b.Violated ? violation.WithAlpha(0.45) : margin.WithAlpha(0.18));
            }
        if (limit.Length > 0) Curves.Line(ds, f, limit, style.Label, 1.4f, Curves.Dashed);
        var c = Palette.Field(0, style.Dark);
        if (line is not null) Curves.Line(ds, f, line, c, 1.8f);
        foreach (var p in s)
        {
            if (p.Margin < 0) Glyphs.Triangle(ds, f.X(p.H), f.Y(p.Cra), 5, violation);
            else ds.FillCircle(f.X(p.H), f.Y(p.Cra), line is null ? 2.6f : 2.1f, c);
        }
    }
}
