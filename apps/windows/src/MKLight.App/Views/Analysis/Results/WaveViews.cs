using Microsoft.Graphics.Canvas;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Model;
using Windows.Foundation;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>A canvas showing one <see cref="MapPlot"/>, with the hover card of the point under the pointer.</summary>
sealed partial class MapCanvas : UserControl
{
    readonly PlotCanvas _canvas;

    public MapCanvas(Func<MapPlot?> plot, string name)
    {
        IsTabStop = false;
        _canvas = new PlotCanvas((ds, size, style) => plot()?.Draw(ds, new Rect(0, 0, size.Width, size.Height), style));
        Content = _canvas;
        AutomationProperties.SetName(_canvas, name);
        _canvas.PointerMoved += (_, e) => { if (plot() is { } p) { p.Hover = e.GetCurrentPoint(_canvas).Position; _canvas.Invalidate(); } };
        _canvas.PointerExited += (_, _) => { if (plot() is { } p) { p.Hover = null; _canvas.Invalidate(); } };
    }

    public void SetName(string name) => AutomationProperties.SetName(_canvas, name);
    public void SetHelp(string text) => AutomationProperties.SetHelpText(_canvas, text);
    public void Invalidate() => _canvas.Invalidate();
}

/// <summary>
/// <c>psf</c>: the PSF as an image on the one-hue magnitude ramp (zero recedes into the plot),
/// linear or logarithmic over five decades, axes in µm about the grid centre, its colour bar,
/// and the Strehl ratio and peak; a menu of the fields (and wavelengths). Until the scale is
/// chosen, a PSF with a low Strehl ratio is shown on the log scale (with a note saying so), so
/// its structure shows at a glance; the core window follows the energy.
/// </summary>
public sealed partial class PsfView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    readonly PsfData _d;
    readonly MapCanvas _map;
    readonly TextBlock _keys;
    MapPlot? _plot;
    int _item;
    bool? _log;
    bool _full;
    readonly Choice _scale;

    public PsfView(PsfData d, bool? log = null)
    {
        _d = d;
        _log = log;
        IsTabStop = false;
        _keys = Kit.Caption("", wrap: true);
        _map = new MapCanvas(() => _plot, Loc.S("Results_Psf_MapName"));
        var picker = new ItemPicker(Loc.S("Results_Axis_Field"), d.Items.Select(i => i.Title).ToList(), k => { _item = k; Rebuild(); });
        var scale = _scale = new Choice(Loc.S("Results_Map_ColorScale"), [Loc.S("Results_Map_Linear"), Loc.S("Results_Map_Log")], 0, menuBelow: 0);
        ToolTipService.SetToolTip(scale, Loc.F("Results_Psf_ScaleToolTip", Fmt.G3(PsfData.LogBelowStrehl)));
        scale.Changed += k => { if (!_syncing) { _log = k == 1; Rebuild(); } };
        var zoom = new Choice(Loc.S("Results_Psf_Extent"), [Loc.S("Results_Psf_ExtentCore"), Loc.S("Results_Psf_ExtentFull")], 0, menuBelow: 0);
        ToolTipService.SetToolTip(zoom, Loc.S("Results_Psf_ExtentToolTip"));
        zoom.Changed += k => { _full = k == 1; Rebuild(); };
        _opts["log"] = () => scale.Select(1);
        _opts["full"] = () => zoom.Select(1);
        var top = new FlowPanel { Spacing = 12, LineSpacing = 4 };
        top.Children.Add(picker);
        top.Children.Add(scale);
        top.Children.Add(zoom);
        var root = new Grid { Padding = new Thickness(10, 6, 10, 8), RowSpacing = 6 };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.Children.Add(top);
        Grid.SetRow(_keys, 1);
        root.Children.Add(_keys);
        Grid.SetRow(_map, 2);
        root.Children.Add(_map);
        Content = root;
        Rebuild();
    }

    bool _syncing;

    void Rebuild()
    {
        var it = _d.Items[Math.Clamp(_item, 0, _d.Items.Count - 1)];
        var auto = _log is null && PsfData.PrefersLog(it);
        var log = _log ?? auto;
        _syncing = true;
        _scale.Select(log ? 1 : 0);
        _syncing = false;
        var u = _d.Units;
        var micro = u.Micro;
        var unitEnergy = it.Normalization == "unit_energy";
        var lo = it.Grid.Range()?.Lo ?? 0;
        var shown = _full ? it.Grid : PsfData.Focus(it.Grid);
        _plot = new MapPlot
        {
            Image = new MapImage(shown, cells: true),
            Scale = MapScale.Magnitude(lo, it.Peak, log),
            Unit = unitEnergy ? u.Density.Replace("⁻¹", "⁻²") : "",
            XTitle = $"x ({micro})",
            YTitle = $"y ({micro})",
            Format = v => Fmt.Short(v),
            HoverText = (f, p) =>
            {
                if (!MapParts.Contains(f.Plot, p)) return null;
                var g = shown;
                var i = (int)Math.Round((f.DataX(p.X) - g.X0) / (g.X1 - g.X0) * (g.Nx - 1));
                var j = (int)Math.Round((f.DataY(p.Y) - g.Y0) / (g.Y1 - g.Y0) * (g.Ny - 1));
                if (i < 0 || j < 0 || i >= g.Nx || j >= g.Ny) return null;
                var v = g.Values[j * g.Nx + i];
                var value = it.Peak > 0 ? Loc.F("Results_Psf_OfPeak", Fmt.Short(v), Fmt.G3(v / it.Peak)) : Fmt.Short(v);
                return [(value, true), ($"x {Fmt.Fixed(f.DataX(p.X), 2)}  y {Fmt.Fixed(f.DataY(p.Y), 2)} {micro}".Replace('-', '−'), false)];
            },
        };
        var width = it.Size * it.SpacingMm;
        _keys.Text = string.Join("   ", new[]
        {
            Loc.F(auto ? "Results_Psf_StrehlAuto" : "Results_Psf_Strehl", Fmt.Fixed(it.Strehl, 3)),
            unitEnergy ? Loc.F("Results_Psf_Peak", Fmt.Short(it.Peak)) : null,
            Loc.F("Results_Psf_Samples", it.Size, u.Length(it.SpacingMm)),
            Loc.F("Results_Psf_Window", u.Length(width)),
        }.OfType<string>());
        _map.SetName(Loc.F("Results_Psf_Name", it.Title));
        _map.SetHelp(Loc.F("Results_Psf_Help", Fmt.Fixed(it.Strehl, 3), Loc.S(log ? "Results_Psf_HelpLogarithmic" : "Results_Psf_HelpLinear"),
            Loc.S(_full ? "Results_Psf_HelpWholeWindow" : "Results_Psf_HelpCore"), u.Length((shown.X1 - shown.X0) / 1000), Fmt.Short(it.Peak)));
        _map.Invalidate();
    }
}

/// <summary>
/// <c>wavefront_map</c>: the OPD over the pupil on the diverging scale centred on zero, the
/// clear-aperture rim, RMS / P-V / Strehl, and the Zernike coefficients beside the map (below
/// it in narrow panes).
/// </summary>
public sealed partial class WavefrontView : UserControl
{
    readonly WavefrontData _d;
    readonly MapCanvas _map;
    readonly TextBlock _keys;
    readonly ContentControl _zernike = new() { IsTabStop = false, HorizontalContentAlignment = HorizontalAlignment.Stretch, VerticalContentAlignment = VerticalAlignment.Stretch };
    MapPlot? _plot;
    int _item;

    public WavefrontView(WavefrontData d)
    {
        _d = d;
        IsTabStop = false;
        _keys = Kit.Caption("");
        _map = new MapCanvas(() => _plot, Loc.S("Results_Wavefront_MapName"));
        var picker = new ItemPicker(Loc.S("Results_Axis_Field"), d.Maps.Select(m => m.Title).ToList(), k => { _item = k; Rebuild(); });
        var body = new SideBySide { Breakpoint = 520, SideWidth = 190, StackedShare = 0.62, FirstHeight = w => w - 30 };
        body.Children.Add(_map);
        body.Children.Add(_zernike);
        var root = new Grid { Padding = new Thickness(10, 6, 10, 8), RowSpacing = 6 };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.Children.Add(picker);
        Grid.SetRow(_keys, 1);
        root.Children.Add(_keys);
        Grid.SetRow(body, 2);
        root.Children.Add(body);
        Content = root;
        Rebuild();
    }

    void Rebuild()
    {
        var m = _d.Maps[Math.Clamp(_item, 0, _d.Maps.Count - 1)];
        var (lo, hi) = m.Grid.Range() ?? (-1, 1);
        _plot = new MapPlot
        {
            Image = new MapImage(m.Grid, cells: true),
            Scale = MapScale.ForQuantity(lo, hi, signed: true),
            Unit = "waves",
            XTitle = "Px",
            YTitle = "Py",
            HatchMissing = false,
            Overlay = (ds, f, style) =>
            {
                foreach (var loop in m.Rim.Where(l => l.Length > 2))
                {
                    using var pb = new Microsoft.Graphics.Canvas.Geometry.CanvasPathBuilder(ds);
                    pb.BeginFigure(f.X(loop[0].U), f.Y(loop[0].V));
                    foreach (var q in loop.Skip(1)) pb.AddLine(f.X(q.U), f.Y(q.V));
                    pb.EndFigure(Microsoft.Graphics.Canvas.Geometry.CanvasFigureLoop.Closed);
                    using var g = Microsoft.Graphics.Canvas.Geometry.CanvasGeometry.CreatePath(pb);
                    ds.DrawGeometry(g, style.Axis, 0.8f);
                }
            },
            HoverText = (f, p) =>
            {
                if (!MapParts.Contains(f.Plot, p)) return null;
                var g = m.Grid;
                var i = (int)Math.Round((f.DataX(p.X) - g.X0) / (g.X1 - g.X0) * (g.Nx - 1));
                var j = (int)Math.Round((f.DataY(p.Y) - g.Y0) / (g.Y1 - g.Y0) * (g.Ny - 1));
                if (i < 0 || j < 0 || i >= g.Nx || j >= g.Ny) return null;
                var v = g.Values[j * g.Nx + i];
                return [(double.IsFinite(v) ? Fmt.Signed(v, 4).Trim() + " waves" : Loc.S("Results_Wavefront_OutsidePupil"), true),
                    ($"Px {Fmt.Fixed(f.DataX(p.X), 2)}  Py {Fmt.Fixed(f.DataY(p.Y), 2)}".Replace('-', '−'), false)];
            },
        };
        _keys.Text = Loc.F("Results_Wavefront_Keys", Fmt.Fixed(m.Rms, 4), Fmt.Fixed(m.Pv, 4), Fmt.Fixed(m.Strehl, 3));
        _map.SetName(Loc.F("Results_Wavefront_Name", m.Title, Fmt.Fixed(m.Rms, 3), Fmt.Fixed(m.Pv, 3)));
        _zernike.Content = m.Zernike.Length > 0 ? Zernike(m.Zernike, m.Ordering) : null;
        _map.Invalidate();
    }

    static FrameworkElement Zernike(double[] c, string ordering)
    {
        var g = new Grid { ColumnSpacing = 10, RowSpacing = 2 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var names = WavefrontData.FringeNames;
        for (var k = 0; k < c.Length; k++)
        {
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var z = Kit.Text(Kit.Inv($"Z{k + 1}"), "SecondaryNumberText", 11);
            var name = Kit.Text(ordering == "fringe" && k < names.Length ? names[k] : "", "SecondaryText", 11);
            var big = Math.Abs(c[k]) >= 0.05;
            var v = Kit.Text(Fmt.Signed(c[k], 4), big ? "NumberText" : "SecondaryNumberText", 11);
            v.HorizontalAlignment = HorizontalAlignment.Right;
            v.IsTextSelectionEnabled = true;
            Grid.SetRow(z, k); Grid.SetRow(name, k); Grid.SetRow(v, k);
            Grid.SetColumn(name, 1); Grid.SetColumn(v, 2);
            g.Children.Add(z); g.Children.Add(name); g.Children.Add(v);
        }
        var p = new Grid { RowSpacing = 4 };
        p.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        p.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        p.Children.Add(Kit.Text(Loc.F("Results_Zernike_Header", EngineText.T(GenericData.Title(ordering))), "PanelTitleText", 11));
        var s = Kit.Page(g);
        s.Padding = new Thickness(0, 0, 12, 0);
        Grid.SetRow(s, 1);
        p.Children.Add(s);
        return p;
    }
}
