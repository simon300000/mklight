using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Model;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Inspector;

/// <summary>A Win2D canvas that redraws on size and theme change and frees its device when unloaded.</summary>
sealed partial class InspectorCanvas : UserControl
{
    readonly Action<CanvasDrawingSession, Size, PlotStyle> _draw;
    CanvasControl? _canvas;

    public InspectorCanvas(Action<CanvasDrawingSession, Size, PlotStyle> draw)
    {
        _draw = draw;
        IsTabStop = false;
        Attach();
        Loaded += (_, _) => { if (_canvas is null) Attach(); };
        Unloaded += (_, _) =>
        {
            _canvas?.RemoveFromVisualTree();
            _canvas = null;
            Content = null;
        };
        ActualThemeChanged += (_, _) => Invalidate();
        SizeChanged += (_, _) => Invalidate();
    }

    void Attach()
    {
        var c = new CanvasControl { ClearColor = Microsoft.UI.Colors.Transparent };
        c.Draw += (s, e) =>
        {
            var size = new Size(s.ActualWidth, s.ActualHeight);
            if (size.Width > 0 && size.Height > 0) _draw(e.DrawingSession, size, PlotStyle.For(ActualTheme));
        };
        _canvas = c;
        Content = c;
    }

    public void Invalidate() => _canvas?.Invalidate();
}

/// <summary>
/// The inspector's picture of a surface shape (the Mac app's <c>ShapePreview</c>): sag profiles (x and y for
/// anamorphic and freeform shapes), a false-colour departure map for freeforms, and for a diffractive surface
/// its phase profile, paraxial power and smallest zone period. Display geometry from <see cref="ShapeSag"/>.
/// </summary>
sealed class ShapePreview : StackPanel
{
    SurfaceShape? _shape;
    double _h = 10;
    double _primaryUm = 0.5875618;
    ShapePreviewKind? _kind;
    readonly List<InspectorCanvas> _canvases = [];
    readonly TextBlock _pv = new();
    readonly StackPanel _figures = new() { Spacing = 8 };
    readonly TextBlock _empty = new();

    public ShapePreview()
    {
        Spacing = 6;
        Padding = new Thickness(12, 8, 12, 10);
    }

    public void Update(SurfaceShape shape, double? semiDiameter, double primaryUm)
    {
        var kind = SurfaceTypes.Descriptor(shape).Preview;
        var h = SurfaceTypes.PreviewRadius(shape, semiDiameter);
        var baseKind = shape is DiffractiveShape d ? d.Base.Kind : shape.Kind;
        var layout = (kind, kind == ShapePreviewKind.Diffractive && baseKind != ShapeKind.Plane);
        if (_kind != kind || _layout != layout) Build(kind, layout.Item2);
        _layout = layout;
        if (Equals(shape, _shape) && h == _h && primaryUm == _primaryUm) return;
        _shape = shape;
        _h = h;
        _primaryUm = primaryUm;
        if (kind == ShapePreviewKind.Freeform) UpdateMapCaption();
        if (kind == ShapePreviewKind.Diffractive) UpdateFigures();
        foreach (var c in _canvases) c.Invalidate();
    }

    (ShapePreviewKind, bool) _layout;

    void Build(ShapePreviewKind kind, bool curvedBase)
    {
        _kind = kind;
        _shape = null;
        Children.Clear();
        _canvases.Clear();
        switch (kind)
        {
            case ShapePreviewKind.Radial:
                AddProfiles([CylinderAxis.Y], null);
                break;
            case ShapePreviewKind.Anamorphic:
                AddProfiles([CylinderAxis.X, CylinderAxis.Y], Loc.S("Lens_Preview_Sag"));
                break;
            case ShapePreviewKind.Freeform:
                AddProfiles([CylinderAxis.X, CylinderAxis.Y], Loc.S("Lens_Preview_Sag"));
                var header = Header(Loc.S("Lens_Preview_Departure"));
                _pv.Style = InspectorStyles.Get("InspectorCaption");
                _pv.HorizontalAlignment = HorizontalAlignment.Right;
                Grid.SetColumn(_pv, 1);
                header.Children.Add(_pv);
                Children.Add(header);
                AddCanvas(DrawMap, 132);
                break;
            case ShapePreviewKind.Diffractive:
                Children.Add(Header(Loc.S("Lens_Preview_Phase")));
                AddCanvas(DrawPhase, 116);
                Children.Add(_figures);
                if (curvedBase) AddProfiles([CylinderAxis.Y], Loc.S("Lens_Preview_BaseSag"));
                break;
        }
    }

    Grid Header(string title)
    {
        var g = new Grid { Margin = new Thickness(0, 2, 0, 0) };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.Children.Add(new TextBlock { Text = title, FontSize = 11, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, Style = InspectorStyles.Get("InspectorLabel") });
        return g;
    }

    void AddProfiles(CylinderAxis[] axes, string? title)
    {
        if (title is not null)
        {
            var header = Header(title);
            if (axes.Length > 1)
            {
                var legend = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 10, HorizontalAlignment = HorizontalAlignment.Right };
                legend.Children.Add(new LegendSwatch(1, dashed: true, Loc.S("Lens_Preview_XSection")));
                legend.Children.Add(new LegendSwatch(0, dashed: false, Loc.S("Lens_Preview_YSection")));
                Grid.SetColumn(legend, 1);
                header.Children.Add(legend);
            }
            Children.Add(header);
        }
        AddCanvas((ds, size, style) => DrawProfiles(ds, size, style, axes), 116);
    }

    void AddCanvas(Action<CanvasDrawingSession, Size, PlotStyle> draw, double height)
    {
        var c = new InspectorCanvas(draw) { Height = height };
        _canvases.Add(c);
        Children.Add(c);
    }

    void DrawProfiles(CanvasDrawingSession ds, Size size, PlotStyle style, CylinderAxis[] axes)
    {
        if (_shape is null) return;
        var shape = _kind == ShapePreviewKind.Diffractive && _shape is DiffractiveShape d ? d.Base : _shape;
        var plot = new LinePlot { XTitle = Loc.S("Lens_Preview_PositionAxis"), YTitle = Loc.S("Lens_Preview_SagAxis"), XTickSpacing = 56, YTickSpacing = 28 };
        foreach (var axis in axes)
        {
            if (SurfaceTypes.Profile(shape, _h, axis, 96) is not { } pts) continue;
            var color = Palette.Field(axis == CylinderAxis.X ? 1 : 0, style.Dark);
            plot.Series.Add(new PlotSeries(pts.Select(p => new Vector2((float)p.T, (float)p.Z)).ToList(), color, axis == CylinderAxis.X));
        }
        if (plot.Series.Count == 0)
        {
            PlotText.Draw(ds, Loc.F("Lens_Preview_NoSag", InspectorFormat.Short(_h)), size.Width / 2, size.Height / 2, style.Label, 11);
            return;
        }
        plot.FitData(includeZeroY: true, pad: 0.08);
        plot.XMin = -_h;
        plot.XMax = _h;
        plot.Draw(ds, new Rect(0, 0, size.Width, size.Height), style);
    }

    void DrawPhase(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        if (_shape is not DiffractiveShape d) return;
        var pts = Enumerable.Range(0, 97).Select(k =>
        {
            var r = _h * k / 96;
            return new Vector2((float)r, (float)(ShapeSag.Phase(d.Phase, r) / (2 * Math.PI)));
        }).ToList();
        var plot = new LinePlot { XTitle = Loc.S("Lens_Preview_RadiusAxis"), YTitle = "Φ (waves)", XTickSpacing = 56, YTickSpacing = 28 };
        plot.Series.Add(new PlotSeries(pts, Palette.Slot(6, style.Dark)));
        plot.FitData(includeZeroY: true, pad: 0.08);
        plot.XMin = 0;
        plot.XMax = _h;
        plot.Draw(ds, new Rect(0, 0, size.Width, size.Height), style);
    }

    const int MapCells = 64;
    double[]? _cells;
    double _peak;

    void UpdateMapCaption()
    {
        _cells = SurfaceTypes.DepartureMap(_shape!, _h, MapCells);
        var finite = _cells.Where(double.IsFinite).ToList();
        var (lo, hi) = finite.Count > 0 ? (finite.Min(), finite.Max()) : (0, 0);
        _peak = Math.Max(Math.Abs(lo), Math.Abs(hi));
        _pv.Text = "PV " + InspectorFormat.Length(hi - lo);
    }

    CanvasBitmap? _bitmap;
    (CanvasDevice, bool, double[])? _bitmapKey;

    void DrawMap(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        if (_cells is null) return;
        var peak = Math.Max(_peak, 1e-15);
        if (_bitmap is null || _bitmapKey is not { } k || k.Item1 != ds.Device || k.Item2 != style.Dark || !ReferenceEquals(k.Item3, _cells))
        {
            _bitmap?.Dispose();
            var colors = _cells.Select(v => double.IsFinite(v) ? Palette.Diverging(v / peak, style.Dark) : Color.FromArgb(0, 0, 0, 0)).ToArray();
            _bitmap = CanvasBitmap.CreateFromColors(ds, colors, MapCells, MapCells);
            _bitmapKey = (ds.Device, style.Dark, _cells);
        }
        var side = Math.Min(size.Height - 4, size.Width - 110);
        var map = new Rect((size.Width - 70 - side) / 2, 2, side, side);
        var center = new Vector2((float)(map.X + side / 2), (float)(map.Y + side / 2));
        using (var clip = CanvasGeometry.CreateCircle(ds, center, (float)side / 2))
        using (ds.CreateLayer(1, clip))
            ds.DrawImage(_bitmap, map, new Rect(0, 0, MapCells, MapCells), 1, CanvasImageInterpolation.Linear);
        ds.DrawCircle(center, (float)side / 2, style.Axis, 0.75f);
        PlotText.Draw(ds, "y ↑", map.X + 2, map.Y + 6, style.Label, anchor: TextAnchor.Start);
        PlotText.Draw(ds, "x →", map.Right - 2, map.Bottom - 6, style.Label, anchor: TextAnchor.End);

        // Colour bar, in µm below 0.1 mm as the PV reads.
        var micro = peak < 0.1;
        var p = peak * (micro ? 1000 : 1);
        var bar = new Rect(map.Right + 22, map.Y + 8, 10, side - 16);
        const int steps = 48;
        for (var i = 0; i < steps; i++)
        {
            var t = (double)i / (steps - 1) * 2 - 1;
            var y0 = bar.Bottom - (i + 1) * bar.Height / steps;
            ds.FillRectangle(new Rect(bar.X, y0, bar.Width, bar.Height / steps + 0.5), Palette.Diverging(t, style.Dark));
        }
        ds.DrawRectangle(bar, style.Grid, 1);
        var end = NumberDisplay.Significant(p, 3);
        PlotText.Draw(ds, "+" + end, bar.Right + 4, bar.Y + 4, style.Label, anchor: TextAnchor.Start);
        PlotText.Draw(ds, "0", bar.Right + 4, bar.Y + bar.Height / 2, style.Label, anchor: TextAnchor.Start);
        PlotText.Draw(ds, "−" + end, bar.Right + 4, bar.Bottom - 4, style.Label, anchor: TextAnchor.Start);
        PlotText.Draw(ds, micro ? "µm" : "mm", bar.X + bar.Width / 2, bar.Y - 8, style.Label);
    }

    /// <summary>The DOE's paraxial power (from A₁), its focal length, the smallest zone period and the zones.</summary>
    void UpdateFigures()
    {
        _figures.Children.Clear();
        if (_shape is not DiffractiveShape d) return;
        var phase = d.Phase;
        var trace = ShapeSag.PhasePower(phase, phase.TraceOrder, _primaryUm);
        var design = ShapeSag.PhasePower(phase, phase.DesignOrder, phase.DesignWavelengthUm);
        var period = ShapeSag.MinimumPeriod(phase, _h);
        var zones = Math.Abs(ShapeSag.Phase(phase, _h)) / (2 * Math.PI);
        var sameAsDesign = phase.TraceOrder == phase.DesignOrder && Math.Abs(_primaryUm - phase.DesignWavelengthUm) < 1e-9;
        string Order(int m, double um) => Loc.F("Lens_Preview_Order", m, InspectorFormat.Short(um));
        Figure(Loc.S("Lens_Preview_ParaxialPower"), Order(phase.TraceOrder, _primaryUm), trace == 0 ? "0" : InspectorFormat.Short(trace) + " mm⁻¹",
            trace == 0 ? null : "f " + InspectorFormat.Short(1 / trace) + " mm");
        if (!sameAsDesign)
            Figure(Loc.S("Lens_Preview_AtDesign"), Order(phase.DesignOrder, phase.DesignWavelengthUm), design == 0 ? "0" : InspectorFormat.Short(design) + " mm⁻¹",
                design == 0 ? null : "f " + InspectorFormat.Short(1 / design) + " mm");
        Figure(Loc.S("Lens_Preview_SmallestPeriod"), period is { } p ? Loc.F("Lens_Preview_AtRadius", InspectorFormat.Short(p.Radius)) : "",
            period is { } q ? InspectorFormat.Length(q.Period) : "—", null);
        Figure(Loc.S("Lens_Preview_Zones"), Loc.F("Lens_Preview_Over", InspectorFormat.Short(_h)), InspectorFormat.Fixed(zones, 1), null);
        ToolTipService.SetToolTip(_figures, Loc.S("Lens_Preview_PowerHelp"));
    }

    void Figure(string title, string detail, string value, string? extra)
    {
        var g = new Grid { ColumnSpacing = 8 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var left = new StackPanel();
        left.Children.Add(new TextBlock { Text = title, Style = InspectorStyles.Get("InspectorLabel") });
        if (detail.Length > 0) left.Children.Add(new TextBlock { Text = detail, Style = InspectorStyles.Get("InspectorCaption") });
        var right = new StackPanel { HorizontalAlignment = HorizontalAlignment.Right };
        right.Children.Add(new TextBlock { Text = value, Style = InspectorStyles.Get("InspectorLabel"), FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Right });
        if (extra is not null) right.Children.Add(new TextBlock { Text = extra, Style = InspectorStyles.Get("InspectorCaption"), HorizontalAlignment = HorizontalAlignment.Right });
        Grid.SetColumn(right, 1);
        g.Children.Add(left);
        g.Children.Add(right);
        _figures.Children.Add(g);
    }

    /// <summary>A legend entry: a short solid or dashed line in a field colour, and its label.</summary>
    sealed partial class LegendSwatch : StackPanel
    {
        public LegendSwatch(int slot, bool dashed, string label)
        {
            Orientation = Orientation.Horizontal;
            Spacing = 4;
            var line = new InspectorCanvas((ds, size, style) =>
            {
                var c = Palette.Field(slot, style.Dark);
                var y = (float)size.Height / 2;
                using var stroke = new CanvasStrokeStyle { DashStyle = dashed ? CanvasDashStyle.Dash : CanvasDashStyle.Solid };
                ds.DrawLine(0, y, (float)size.Width, y, c, 1.6f, stroke);
            }) { Width = 16, Height = 10, VerticalAlignment = VerticalAlignment.Center };
            Children.Add(line);
            Children.Add(new TextBlock { Text = label, Style = InspectorStyles.Get("InspectorCaption") });
        }
    }
}
