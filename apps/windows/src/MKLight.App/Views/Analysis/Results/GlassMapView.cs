using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// <c>glass_map</c>: catalogue glasses in the n_d – V_d or P_g,F – V_d plane, V_d decreasing
/// to the right as glass maps are drawn; status colours (obsolete hollow, so status does not
/// rely on colour alone); the design's glasses ringed in the accent and labelled with their
/// surfaces; hover for a glass's data, click to inspect it.
/// </summary>
public sealed partial class GlassMapView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    readonly GlassMapData _d;
    readonly PlotCanvas _canvas;
    readonly ContentControl _side = new() { IsTabStop = false, HorizontalContentAlignment = HorizontalAlignment.Stretch };
    readonly FlowPanel _legend = new() { Spacing = 12, LineSpacing = 3 };
    bool _pgf;
    string? _catalog;
    string? _selected;
    Point? _hover;
    Frame? _frame;

    public GlassMapView(GlassMapData d)
    {
        _d = d;
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw);
        var plane = new Choice("Plane", ["n_d", "P_g,F"], menuBelow: 0);
        ToolTipService.SetToolTip(plane, "Refractive index nd or partial dispersion Pg,F against Abbe number Vd");
        plane.Changed += k => { _pgf = k == 1; _canvas.Invalidate(); };
        _opts["pgf"] = () => plane.Select(1);
        var top = new FlowPanel { Spacing = 12, LineSpacing = 4 };
        top.Children.Add(plane);
        var catalogs = d.Catalogs.ToList();
        if (catalogs.Count > 1)
        {
            var c = new ComboBox { ItemsSource = catalogs.Prepend("All catalogs").ToList(), SelectedIndex = 0, FontSize = 12, MinHeight = 26, MinWidth = 0, Padding = new Thickness(8, 2, 6, 3) };
            AutomationProperties.SetName(c, "Catalog");
            c.SelectionChanged += (_, _) => { _catalog = c.SelectedIndex <= 0 ? null : catalogs[c.SelectedIndex - 1]; _canvas.Invalidate(); };
            top.Children.Add(c);
        }
        top.Children.Add(_legend);
        var body = new SideBySide { Breakpoint = 620, SideWidth = 200, StackedShare = 0.68 };
        body.Children.Add(_canvas);
        body.Children.Add(Kit.Page(_side));
        Content = FieldMapViews.Shell(top, body);
        AutomationProperties.SetName(_canvas, $"Glass map, {d.Glasses.Count(g => g.Kind == "catalog")} glasses; in the design: " +
            string.Join("; ", d.Glasses.Where(g => g.InDesign).Select(g => $"{g.Name} {g.SurfaceLabel}")));
        _canvas.PointerMoved += (_, e) => { _hover = e.GetCurrentPoint(_canvas).Position; _canvas.Invalidate(); };
        _canvas.PointerExited += (_, _) => { _hover = null; _canvas.Invalidate(); };
        _canvas.PointerPressed += (_, e) =>
        {
            _selected = Nearest(e.GetCurrentPoint(_canvas).Position)?.Id;
            Side();
            _canvas.Invalidate();
        };
        ActualThemeChanged += (_, _) => { BuildLegend(); Side(); };
        BuildLegend();
        Side();
    }

    IEnumerable<GlassPoint> Shown => _d.Glasses.Where(g => (_catalog is null || g.Catalog == _catalog || g.InDesign) && double.IsFinite(g.Vd) && double.IsFinite(Y(g)));
    double Y(GlassPoint g) => _pgf ? g.Pgf : g.Nd;

    static Color StatusColor(GlassStatus s, bool dark) => s.Slot() < 0 ? (dark ? Palette.Rgb(0x9a9a95) : Palette.Rgb(0x8a8a85)) : Palette.Slot(s.Slot(), dark);

    void BuildLegend()
    {
        _legend.Children.Clear();
        var dark = ActualTheme == ElementTheme.Dark;
        var statuses = _d.Statuses;
        if (statuses.Count == 0) statuses = [GlassStatus.Unknown];
        foreach (var s in statuses)
            _legend.Children.Add(Kit.LegendItem(Kit.Dot(StatusColor(s, dark), 7, s.Hollow()), statuses.Count == 1 && s == GlassStatus.Unknown ? "Catalog glass" : s.Label()));
        var ring = new Grid { Width = 13, Height = 13, VerticalAlignment = VerticalAlignment.Center };
        ring.Children.Add(new Ellipse { Width = 8, Height = 8, Fill = new SolidColorBrush(StatusColor(GlassStatus.Standard, dark)) });
        ring.Children.Add(new Ellipse { Width = 13, Height = 13, Stroke = new SolidColorBrush(Kit.Accent(this)), StrokeThickness = 2 });
        _legend.Children.Add(Kit.LegendItem(ring, "In the design"));
        if (_d.Glasses.Any(g => g.Kind == "offset"))
            _legend.Children.Add(Kit.LegendItem(new Line { X1 = 0, X2 = 14, Y1 = 5, Y2 = 5, Width = 14, Height = 10, Stroke = new SolidColorBrush(Kit.Accent(this)), StrokeThickness = 1.2, StrokeDashArray = Dash(2, 2), VerticalAlignment = VerticalAlignment.Center }, "Melt offset"));
    }

    void Side()
    {
        var p = new StackPanel { Spacing = 12, Padding = new Thickness(0, 2, 4, 0) };
        if (_selected is not null && _d.Glasses.FirstOrDefault(g => g.Id == _selected) is { } g)
        {
            var card = Kit.Card(GlassCard(g, compact: false));
            Kit.ThemeCard(card, ActualTheme);
            p.Children.Add(card);
        }
        else p.Children.Add(Kit.Caption("Click a glass to inspect it.", wrap: true));
        var design = _d.Glasses.Where(x => x.InDesign).ToList();
        if (design.Count > 0)
        {
            var list = new StackPanel { Spacing = 3 };
            list.Children.Add(Kit.Text("In the design", "PanelTitleText", 11));
            foreach (var x in design)
            {
                var row = new Grid { ColumnSpacing = 8 };
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                row.Children.Add(Kit.Text(x.DisplayName, "BodyText", 12));
                var s = Kit.Text(x.SurfaceLabel, "SecondaryText", 12);
                Grid.SetColumn(s, 1);
                row.Children.Add(s);
                var b = new Button { Content = row, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch, Padding = new Thickness(6, 2, 6, 3), Background = new SolidColorBrush(Color.FromArgb(0, 0, 0, 0)), BorderThickness = new Thickness(0) };
                var id = x.Id;
                b.Click += (_, _) => { _selected = id; Side(); _canvas.Invalidate(); };
                AutomationProperties.SetName(b, $"{x.Name}, surfaces {x.SurfaceLabel}");
                list.Children.Add(b);
            }
            p.Children.Add(list);
        }
        _side.Content = p;
    }

    /// <summary>A glass's data: name, catalogue · status, n_d, V_d, P_g,F (full: ΔP_g,F, surfaces, substitution).</summary>
    static FrameworkElement GlassCard(GlassPoint g, bool compact)
    {
        var p = new StackPanel { Spacing = compact ? 1 : 4 };
        p.Children.Add(Kit.Text(g.DisplayName, "PanelTitleText", compact ? 12 : 14));
        var sub = string.Join(" · ", new[] { g.Catalog ?? (g.Kind == "model" ? "Model glass" : ""), g.Kind == "offset" ? "melt offset" : "", g.Status == GlassStatus.Unknown && g.OtherStatus is null ? "" : g.StatusLabel }.Where(s => s.Length > 0));
        if (sub.Length > 0) p.Children.Add(Kit.Caption(sub));
        var rows = new List<(string, string)> { ("n_d", Fmt.Fixed(g.Nd, 5)), ("V_d", Fmt.Fixed(g.Vd, 2)), ("P_g,F", double.IsFinite(g.Pgf) ? Fmt.Fixed(g.Pgf, 4) : "—") };
        if (!compact)
        {
            rows.Add(("ΔP_g,F", double.IsFinite(g.Pgf) ? Fmt.Signed(g.DeltaPgf, 4).Trim() : "—"));
            if (g.Surfaces.Count > 0) rows.Add(("Surfaces", g.SurfaceLabel));
            if (g.ExcludeSubstitution == true) rows.Add(("Substitution", "Excluded"));
        }
        p.Children.Add(Kit.KeyValues(rows, compact ? 11 : 12, subscripts: true));
        return p;
    }

    GlassPoint? Nearest(Point p)
    {
        if (_frame is not { } f || !MapParts.Contains(MapParts.Inflate(f.Plot, 8, 8), p)) return null;
        GlassPoint? best = null;
        var bd = double.PositiveInfinity;
        foreach (var g in Shown)
        {
            var dx = f.X(g.Vd) - p.X;
            var dy = f.Y(Y(g)) - p.Y;
            var d = dx * dx + dy * dy - (g.InDesign ? 9 : 0); // design glasses win ties: they are drawn larger
            if (d < bd) { bd = d; best = g; }
        }
        return bd <= 100 ? best : null;
    }

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var shown = Shown.ToList();
        var vd = shown.Select(g => g.Vd).DefaultIfEmpty(20).ToList();
        var ys = shown.Select(Y).ToList();
        double vlo = vd.Min(), vhi = vd.Max();
        var vpad = Math.Max((vhi - vlo) * 0.05, 2);
        double ylo = ys.DefaultIfEmpty(_pgf ? 0.5 : 1.4).Min(), yhi = ys.DefaultIfEmpty(_pgf ? 0.65 : 2).Max();
        var ypad = Math.Max((yhi - ylo) * 0.06, _pgf ? 0.005 : 0.02);
        var f = new Frame
        {
            XMin = vlo - vpad, XMax = vhi + vpad, YMin = ylo - ypad, YMax = yhi + ypad, ReverseX = true, XTickSpacing = 50, YTickSpacing = 45,
            XTitle = "Abbe number V_d", YTitle = _pgf ? "Partial dispersion P_g,F" : "Refractive index n_d",
        };
        _frame = f;
        var r = f.Draw(ds, f.Inside(ds, new Rect(0, 0, size.Width, size.Height)), style);
        if (r.IsEmpty) return;
        var accent = Kit.Accent(this);
        using (ds.CreateLayer(1, MapParts.Inflate(r, 4, 4)))
        {
            if (_pgf)
                ds.DrawLine(f.X(f.XMin), f.Y(GlassPoint.NormalLine(f.XMin)), f.X(f.XMax), f.Y(GlassPoint.NormalLine(f.XMax)), style.Label.WithAlpha(0.7), 1, Curves.Dashed);
            foreach (var g in shown.Where(g => !g.InDesign))
            {
                var c = StatusColor(g.Status, style.Dark);
                if (g.Status.Hollow()) ds.DrawCircle(f.X(g.Vd), f.Y(Y(g)), 2.6f, c, 1);
                else ds.FillCircle(f.X(g.Vd), f.Y(Y(g)), 2.6f, c.WithAlpha(0.85));
            }
            var marked = shown.Where(g => g.InDesign).ToList();
            foreach (var g in marked.Where(g => g.Kind == "offset"))
            {
                if (_d.Glasses.FirstOrDefault(n => n.Id == g.NominalId) is not { } nominal) continue;
                ds.DrawLine(f.X(nominal.Vd), f.Y(Y(nominal)), f.X(g.Vd), f.Y(Y(g)), accent.WithAlpha(0.8), 1.2f, new CanvasStrokeStyle { CustomDashStyle = [2, 2] });
                ds.DrawCircle(f.X(nominal.Vd), f.Y(Y(nominal)), 3.5f, accent.WithAlpha(0.8), 1);
            }
            var placed = new List<Rect>();
            foreach (var g in marked)
            {
                float x = f.X(g.Vd), y = f.Y(Y(g));
                ds.FillCircle(x, y, 5, StatusColor(g.Status, style.Dark));
                ds.DrawCircle(x, y, 7, accent, 2);
                placed.Add(new Rect(x - 8, y - 8, 16, 16));
            }
            foreach (var g in marked)
            {
                float x = f.X(g.Vd), y = f.Y(Y(g));
                var label = g.Name + "  " + g.SurfaceLabel;
                var sz = PlotText.Measure(ds, label, 10, true);
                Rect[] boxes =
                [
                    new(x + 10, y - sz.Height / 2, sz.Width + 6, sz.Height), new(x - 16 - sz.Width, y - sz.Height / 2, sz.Width + 6, sz.Height),
                    new(x - sz.Width / 2 - 3, y - 11 - sz.Height, sz.Width + 6, sz.Height), new(x - sz.Width / 2 - 3, y + 11, sz.Width + 6, sz.Height),
                ];
                var box = boxes.FirstOrDefault(b => Inside(r, b) && !placed.Any(p => MapParts.Intersects(p, b)), boxes.FirstOrDefault(b => Inside(r, b), boxes[0]));
                placed.Add(box);
                ds.FillRoundedRectangle(box, 3, 3, style.Dark ? Color.FromArgb(0x8C, 0, 0, 0) : Color.FromArgb(0xCC, 255, 255, 255));
                PlotText.Draw(ds, label, box.X + 3, box.Y + box.Height / 2, style.Title, 10, TextAnchor.Start, bold: true);
            }
            if (_selected is not null && shown.FirstOrDefault(g => g.Id == _selected) is { } s)
                ds.DrawCircle(f.X(s.Vd), f.Y(Y(s)), s.InDesign ? 10 : 7, style.Title, 1.5f, new CanvasStrokeStyle { CustomDashStyle = [3, 2] });
        }
        if (_hover is { } h && Nearest(h) is { } hg)
        {
            var lines = new List<(string, bool)> { (hg.DisplayName, true) };
            var sub = string.Join(" · ", new[] { hg.Catalog ?? "", hg.Status == GlassStatus.Unknown && hg.OtherStatus is null ? "" : hg.StatusLabel }.Where(x => x.Length > 0));
            if (sub.Length > 0) lines.Add((sub, false));
            lines.Add(($"n_d {Fmt.Fixed(hg.Nd, 5)}   V_d {Fmt.Fixed(hg.Vd, 2)}", false));
            lines.Add(($"P_g,F {(double.IsFinite(hg.Pgf) ? Fmt.Fixed(hg.Pgf, 4) : "—")}", false));
            if (hg.Surfaces.Count > 0) lines.Add((hg.SurfaceLabel, false));
            MapParts.Card(ds, f.X(hg.Vd), f.Y(Y(hg)), new Rect(0, 0, size.Width, size.Height), lines, style);
        }
    }

    static DoubleCollection Dash(double on, double off)
    {
        var d = new DoubleCollection();
        d.Add(on);
        d.Add(off);
        return d;
    }

    static bool Inside(Rect outer, Rect r) => r.X >= outer.X && r.Right <= outer.Right && r.Y >= outer.Y && r.Bottom <= outer.Bottom;
}
