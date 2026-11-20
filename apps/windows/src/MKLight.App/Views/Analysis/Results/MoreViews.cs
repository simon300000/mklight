using Microsoft.Graphics.Canvas;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>Square plots in image (or surface) millimetres: equal scales on both axes.</summary>
static class SquarePlot
{
    public static Frame? Frame(CanvasDrawingSession ds, Size size, double half, string xTitle, string yTitle, PlotStyle style)
    {
        var f = new Frame { XMin = -half, XMax = half, YMin = -half, YMax = half, XTitle = xTitle, YTitle = yTitle, XTickSpacing = 60, YTickSpacing = 50, MinTicks = 3 };
        var inside = f.Inside(ds, new Rect(0, 0, size.Width, size.Height));
        var side = Math.Min(inside.Width, inside.Height);
        if (side < 30) return null;
        var plot = new Rect(inside.X + (inside.Width - side) / 2, inside.Y, side, side);
        return f.Draw(ds, plot, style).IsEmpty ? null : f;
    }

    public static double Half(IEnumerable<double> coords, double atLeast = 0)
    {
        var m = coords.Where(double.IsFinite).Select(Math.Abs).DefaultIfEmpty(1).Max();
        return Math.Max(Math.Max(m, atLeast), 1e-9) * 1.06;
    }
}

/// <summary><c>grid_distortion</c>: the real chief-ray grid over the dashed paraxial grid.</summary>
public sealed partial class GridDistortionView : UserControl
{
    public GridDistortionView(GridDistortionData d)
    {
        IsTabStop = false;
        var legend = new FlowPanel { Spacing = 14, LineSpacing = 3 };
        legend.Children.Add(new Legend([new LegendEntry("Paraxial grid", SeriesColor.Neutral, LegendMark.Dashed), new LegendEntry("Real chief rays", SeriesColor.Field(0))]));
        legend.Children.Add(Kit.Caption("Max " + Fmt.Fixed(d.MaxDistortion, 3) + " %"));
        var canvas = new PlotCanvas((ds, size, style) =>
        {
            var half = SquarePlot.Half(d.Paraxial.Concat(d.Real).SelectMany(p => new[] { p.X, p.Y }));
            if (SquarePlot.Frame(ds, size, half, "Image x (mm)", "Image y (mm)", style) is not { } f) return;
            foreach (var line in d.Lines(d.Paraxial)) Curves.Line(ds, f, line, style.Label.WithAlpha(0.55), 1, Curves.Dashed);
            foreach (var line in d.Lines(d.Real)) Curves.Line(ds, f, line, Palette.Field(0, style.Dark), 1.3f);
        });
        AutomationProperties.SetName(canvas, $"Grid distortion, maximum {Fmt.Fixed(d.MaxDistortion, 3)} percent");
        Content = FieldMapViews.Shell(legend, canvas);
    }
}

/// <summary><c>footprint</c>: the hit points on the surface by wavelength, the clipped points in red, the clear aperture dashed.</summary>
public sealed partial class FootprintView : UserControl
{
    public FootprintView(FootprintData d, IReadOnlyList<double> um)
    {
        IsTabStop = false;
        double Um(int w) => w >= 0 && w < um.Count ? um[w] : 0.55;
        var entries = d.Wavelengths.Select(w => new LegendEntry(Fmt.Wavelength(Um(w)), SeriesColor.Wavelength(Um(w)), LegendMark.Dot)).ToList();
        if (double.IsFinite(d.SemiDiameter)) entries.Add(new LegendEntry("Clear aperture " + Fmt.Value(d.SemiDiameter, "mm"), SeriesColor.Neutral, LegendMark.DashedCircle));
        var legend = new FlowPanel { Spacing = 14, LineSpacing = 3 };
        legend.Children.Add(Kit.Text(Kit.Inv($"Surface {d.Surface}"), "PanelTitleText", 12));
        legend.Children.Add(new Legend(entries));
        if (d.Clipped.Length > 0) legend.Children.Add(Kit.LegendItem(Kit.Dot(Palette.Slot(7, false)), Kit.Inv($"{d.Clipped.Length} clipped")));
        var canvas = new PlotCanvas((ds, size, style) =>
        {
            var half = SquarePlot.Half(d.Points.SelectMany(p => new[] { p.X, p.Y }).Concat(d.Clipped.SelectMany(p => new[] { p.X, p.Y })), double.IsFinite(d.SemiDiameter) ? d.SemiDiameter : 0);
            if (SquarePlot.Frame(ds, size, half, "x (mm)", "y (mm)", style) is not { } f) return;
            if (double.IsFinite(d.SemiDiameter))
            {
                var r = f.X(d.SemiDiameter) - f.X(0);
                ds.DrawCircle(f.X(0), f.Y(0), r, style.Label, 1, Curves.Dashed);
            }
            var dot = d.Points.Length > 4000 ? 1f : 1.4f;
            foreach (var p in d.Points) if (double.IsFinite(p.X)) ds.FillCircle(f.X(p.X), f.Y(p.Y), dot, Palette.Wavelength(Um(p.Wavelength), style.Dark).WithAlpha(0.8));
            var red = Palette.Slot(7, style.Dark);
            foreach (var p in d.Clipped) if (double.IsFinite(p.X)) ds.FillCircle(f.X(p.X), f.Y(p.Y), dot, red);
        });
        AutomationProperties.SetName(canvas, $"Footprint on surface {d.Surface}, {d.Points.Length} rays pass, {d.Clipped.Length} clipped");
        Content = FieldMapViews.Shell(legend, canvas);
    }
}

/// <summary><c>ghost_paths</c>: the ranked ghosts as a table, near-focus ghosts flagged.</summary>
public sealed partial class GhostPathsView : UserControl
{
    public GhostPathsView(GhostPathsData d)
    {
        IsTabStop = false;
        if (d.Ghosts.Count == 0)
        {
            Content = SeidelView.Message("No ghost paths", "The enumeration found no ghost reaching the sensor.");
            return;
        }
        var g = new Grid { ColumnSpacing = 16, RowSpacing = 3 };
        string[] heads = ["Ghost", "Rel. irradiance", "Focus (mm)", "Blur ⌀ (mm)", "ΠR"];
        string[] tips = ["Surfaces reflecting, in order", "Estimated ghost irradiance relative to the signal", "Signed distance of the ghost focus from the sensor", "Ghost blur diameter on the sensor", "Product of the reflectances"];
        for (var c = 0; c < heads.Length; c++)
        {
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var h = Kit.Text(heads[c], "ColumnHeaderText", 11);
            h.HorizontalAlignment = c == 0 ? HorizontalAlignment.Left : HorizontalAlignment.Right;
            ToolTipService.SetToolTip(h, tips[c]);
            Grid.SetColumn(h, c);
            g.Children.Add(h);
        }
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        for (var r = 0; r < d.Ghosts.Count; r++)
        {
            var x = d.Ghosts[r];
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var name = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
            name.Children.Add(Kit.Text(x.Label, "NumberText", 12));
            if (x.NearFocus)
            {
                var w = Kit.Text("⚠", "BodyText", 11);
                w.Foreground = ThemeLookup.Brush(this, "WarningBrush");
                ToolTipService.SetToolTip(w, "Focused near the sensor");
                AutomationProperties.SetName(w, "near focus");
                name.Children.Add(w);
            }
            Grid.SetRow(name, r + 1);
            g.Children.Add(name);
            string[] cells = [GhostPathsData.Sci(x.Irradiance), Fmt.Fixed(x.Focus, 2), Fmt.Fixed(x.Spot, 2), GhostPathsData.Sci(x.Reflectance)];
            for (var c = 0; c < cells.Length; c++)
            {
                var t = Kit.Text(cells[c], "NumberText", 12);
                t.HorizontalAlignment = HorizontalAlignment.Right;
                t.IsTextSelectionEnabled = true;
                Grid.SetRow(t, r + 1); Grid.SetColumn(t, c + 1);
                g.Children.Add(t);
            }
        }
        var stack = new StackPanel { Spacing = 10, Padding = new Thickness(12, 8, 12, 12) };
        stack.Children.Add(new ScrollViewer { Content = g, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled, HorizontalScrollMode = ScrollMode.Auto, VerticalScrollMode = ScrollMode.Disabled });
        if (d.Truncated) stack.Children.Add(Kit.Caption("Higher-order paths were dropped by the path budget.", wrap: true));
        Content = Kit.Page(stack);
    }
}

/// <summary>
/// Sensor maps (<c>ghost_irradiance</c>, <c>stray_light</c>): a map picker, linear or log
/// magnitude scale (log by default: ghosts are faint), the colour bar, and the summary beside.
/// </summary>
public sealed partial class SensorMapView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    readonly SensorMapsData _d;
    readonly MapCanvas _canvas;
    readonly TextBlock _caption;
    MapPlot? _plot;
    int _item;
    bool _log = true;

    public SensorMapView(SensorMapsData d)
    {
        _d = d;
        IsTabStop = false;
        _canvas = new MapCanvas(() => _plot, "Sensor map");
        _caption = Kit.Caption("");
        var picker = new ItemPicker("Map", d.Maps.Select(m => m.Title).ToList(), k => { _item = k; Rebuild(); });
        var scale = new Choice("Color scale", ["Linear", "Log"], 1, menuBelow: 0);
        scale.Changed += k => { _log = k == 1; Rebuild(); };
        _opts["linear"] = () => scale.Select(0);
        var top = new FlowPanel { Spacing = 12, LineSpacing = 4 };
        top.Children.Add(picker);
        top.Children.Add(scale);
        var body = new SideBySide { Breakpoint = 560, SideWidth = 210, StackedShare = 0.66, FirstHeight = w => w - 30 };
        body.Children.Add(_canvas);
        body.Children.Add(Kit.Page(Kit.KeyValues(d.Rows, 12)));
        Content = FieldMapViews.Shell(top, _caption, body);
        Rebuild();
    }

    void Rebuild()
    {
        var (title, grid) = _d.Maps[Math.Clamp(_item, 0, _d.Maps.Count - 1)];
        var (lo, hi) = grid.Range() ?? (0, 1);
        _plot = new MapPlot
        {
            Image = new MapImage(grid, cells: true),
            Scale = MapScale.Magnitude(lo, hi, _log),
            Unit = _d.Unit,
            XTitle = "x (mm)",
            YTitle = "y (mm)",
            HoverText = (f, p) =>
            {
                if (!MapParts.Contains(f.Plot, p)) return null;
                var i = (int)Math.Round((f.DataX(p.X) - grid.X0) / (grid.X1 - grid.X0) * (grid.Nx - 1));
                var j = (int)Math.Round((f.DataY(p.Y) - grid.Y0) / (grid.Y1 - grid.Y0) * (grid.Ny - 1));
                if (i < 0 || j < 0 || i >= grid.Nx || j >= grid.Ny) return null;
                return [(Fmt.Value(grid.Values[j * grid.Nx + i], _d.Unit), true), ($"x {Fmt.Fixed(f.DataX(p.X), 2)}  y {Fmt.Fixed(f.DataY(p.Y), 2)} mm".Replace('-', '−'), false)];
            },
        };
        var (x0, x1, y0, y1) = _plot.Image.Extent;
        _caption.Text = $"Sensor {Fmt.Fixed(x1 - x0, 2)} × {Fmt.Fixed(y1 - y0, 2)} mm · peak {Fmt.Short(hi)} {_d.Unit}";
        _canvas.SetName($"{title} map over the sensor" + (_log ? ", logarithmic scale" : ""));
        _canvas.Invalidate();
    }
}

/// <summary><c>polarization_pupil_map</c>: intensity, diattenuation or retardance over the pupil, with the pupil statistics.</summary>
public sealed partial class PolarizationView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    readonly PolarizationData _d;
    readonly MapCanvas _canvas;
    MapPlot? _plot;

    public PolarizationView(PolarizationData d)
    {
        _d = d;
        IsTabStop = false;
        _canvas = new MapCanvas(() => _plot, "Polarization pupil map");
        var choice = new Choice("Quantity", PolarizationData.Quantities);
        choice.Changed += k => Rebuild(PolarizationData.Quantities[k]);
        _opts["diattenuation"] = () => choice.Select(1);
        var body = new SideBySide { Breakpoint = 560, SideWidth = 200, StackedShare = 0.7, FirstHeight = w => w - 30 };
        body.Children.Add(_canvas);
        body.Children.Add(Kit.Page(Kit.KeyValues(d.Stats, 12)));
        Content = FieldMapViews.Shell(choice, body);
        Rebuild(PolarizationData.Quantities[0]);
    }

    void Rebuild(string q)
    {
        if (_d.Map(q) is not { } grid) { _plot = null; _canvas.Invalidate(); return; }
        var (lo, hi) = grid.Range() ?? (0, 1);
        var unit = q == "Retardance" ? "rad" : "";
        _plot = new MapPlot
        {
            Image = new MapImage(grid, cells: true),
            Scale = MapScale.ForQuantity(lo, hi, signed: false),
            Unit = unit,
            XTitle = "Px",
            YTitle = "Py",
            HatchMissing = false,
            HoverText = (f, p) =>
            {
                if (!MapParts.Contains(f.Plot, p)) return null;
                var i = (int)Math.Round((f.DataX(p.X) + 1) / 2 * (grid.Nx - 1));
                var j = (int)Math.Round((f.DataY(p.Y) + 1) / 2 * (grid.Ny - 1));
                if (i < 0 || j < 0 || i >= grid.Nx || j >= grid.Ny || !double.IsFinite(grid.Values[j * grid.Nx + i])) return null;
                return [(Fmt.Value(grid.Values[j * grid.Nx + i], unit), true), ($"Px {Fmt.Fixed(f.DataX(p.X), 2)}  Py {Fmt.Fixed(f.DataY(p.Y), 2)}".Replace('-', '−'), false)];
            },
        };
        _canvas.SetName($"{q} over the pupil, from {Fmt.Short(lo)} to {Fmt.Short(hi)}");
        _canvas.Invalidate();
    }
}

/// <summary><c>system_transmission</c>: transmission against wavelength per field (bars per field for one wavelength), with each field's polychromatic value.</summary>
public sealed partial class TransmissionView : UserControl
{
    public TransmissionView(TransmissionData d, IReadOnlyList<double> um)
    {
        IsTabStop = false;
        var fields = d.Fields;
        string Label(int f) => (f < d.FieldLabels.Count ? d.FieldLabels[f] : Kit.Inv($"Field {f + 1}")) + (f < d.Polychromatic.Count ? " · " + Fmt.Percent(d.Polychromatic[f]) : "");
        var legend = new Legend(fields.Select(f => new LegendEntry(Label(f), SeriesColor.Field(f), d.Spectral ? LegendMark.Line : LegendMark.Dot)).ToList());
        var canvas = new PlotCanvas((ds, size, style) =>
        {
            if (d.Spectral)
            {
                var f = new Frame { XTitle = "Wavelength (µm)", YTitle = "Transmission", YMin = 0, YMax = 1.02, YTickSpacing = 50 };
                (f.XMin, f.XMax) = Ticks.Range(d.Points.Select(p => p.Um), 0.04);
                if (f.Draw(ds, f.Inside(ds, new Rect(0, 0, size.Width, size.Height)), style).IsEmpty) return;
                foreach (var fi in fields)
                {
                    var pts = d.Points.Where(p => p.Field == fi).OrderBy(p => p.Um).Select(p => (p.Um, p.T)).ToArray();
                    Curves.Line(ds, f, pts, Palette.Field(fi, style.Dark), 1.6f);
                    Curves.Dots(ds, f, pts, Palette.Field(fi, style.Dark), 2.4f);
                }
                return;
            }
            var b = new Frame { XTitle = "Field", YTitle = "Transmission", YMin = 0, YMax = 1, XTicks = [], XMin = -0.5, XMax = fields.Count - 0.5 };
            var r = b.Draw(ds, b.Inside(ds, new Rect(0, 0, size.Width, size.Height)), style);
            if (r.IsEmpty) return;
            var w = r.Width / Math.Max(fields.Count, 1) * 0.6;
            for (var k = 0; k < fields.Count; k++)
            {
                var t = d.Points.First(p => p.Field == fields[k]).T;
                var x = b.X(k);
                PlotText.Draw(ds, (fields[k] + 1).ToString(System.Globalization.CultureInfo.InvariantCulture), x, r.Bottom + 10, style.Label);
                if (double.IsFinite(t)) ds.FillRoundedRectangle(new Rect(x - w / 2, b.Y(t), w, Math.Max(r.Bottom - b.Y(t), 0.8)), 3, 3, Palette.Field(fields[k], style.Dark));
            }
        });
        AutomationProperties.SetName(canvas, "System transmission per field" + (d.Spectral ? " against wavelength" : ""));
        Content = FieldMapViews.Shell(legend, canvas);
    }
}
