using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// <c>first_order</c>: the keyed report as a clean table (label, value, unit), with the x / y
/// first order in a card when the design is not rotationally symmetric, and the DOE powers
/// when it has diffractive surfaces.
/// </summary>
public sealed partial class FirstOrderView : UserControl
{
    readonly List<Border> _cards = [];

    public FirstOrderView(FirstOrderData d)
    {
        IsTabStop = false;
        var stack = new StackPanel { Spacing = 14, Padding = new Thickness(14, 10, 14, 14) };
        if (d.IsAnamorphic && d.Primary is { } p) stack.Children.Add(Card("Anamorphic", Anamorphic(d, p)));
        else if (d.Primary is { Doe.Count: > 0 } q) stack.Children.Add(Card("Diffractive", Doe(q)));

        // Labels take the room left (trimmed in narrow panes); values and units keep theirs.
        var g = new Grid { ColumnSpacing = 6, RowSpacing = 5, HorizontalAlignment = HorizontalAlignment.Left, MaxWidth = 480 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto, MinWidth = 22 });
        for (var i = 0; i < d.Entries.Count; i++)
        {
            var e = d.Entries[i];
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var label = Kit.Text(e.Label, "SecondaryText", 12.5);
            label.Margin = new Thickness(0, 0, 14, 0);
            ToolTipService.SetToolTip(label, e.Label);
            var value = Kit.Text(Fmt.Short(e.Value), "NumberText", 12.5);
            value.HorizontalAlignment = HorizontalAlignment.Right;
            value.IsTextSelectionEnabled = true;
            var unit = Kit.Text(e.Unit, "SecondaryText", 12.5);
            Grid.SetRow(label, i); Grid.SetRow(value, i); Grid.SetRow(unit, i);
            Grid.SetColumn(value, 1); Grid.SetColumn(unit, 2);
            g.Children.Add(label); g.Children.Add(value); g.Children.Add(unit);
            AutomationProperties.SetName(value, $"{e.Label}: {e.Formatted}");
        }
        stack.Children.Add(g);
        if (double.IsFinite(d.WavelengthUm)) stack.Children.Add(Kit.Caption("At λ = " + Fmt.Wavelength(d.WavelengthUm)));
        Content = Kit.Page(stack);
        ActualThemeChanged += (_, _) => Theme();
        Loaded += (_, _) => Theme();
    }

    void Theme()
    {
        foreach (var c in _cards) Kit.ThemeCard(c, ActualTheme);
    }

    Border Card(string title, UIElement body)
    {
        var p = new StackPanel { Spacing = 8 };
        p.Children.Add(Kit.Heading(title));
        p.Children.Add(body);
        var b = Kit.Card(p);
        _cards.Add(b);
        return b;
    }

    static UIElement Anamorphic(FirstOrderData d, FirstOrderData.XyRow p)
    {
        var s = new StackPanel { Spacing = 6 };
        var g = new Grid { ColumnSpacing = 18, RowSpacing = 4, HorizontalAlignment = HorizontalAlignment.Left };
        for (var c = 0; c < 3; c++) g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        void Row(int r, string label, string x, string? y, string? tip = null)
        {
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var l = Kit.Text(label, "SecondaryText");
            Grid.SetRow(l, r);
            g.Children.Add(l);
            if (tip is not null) ToolTipService.SetToolTip(l, tip);
            var vx = Kit.Text(x, r == 0 ? "CaptionText" : "NumberText");
            vx.HorizontalAlignment = HorizontalAlignment.Right;
            Grid.SetRow(vx, r); Grid.SetColumn(vx, 1);
            if (y is null) Grid.SetColumnSpan(vx, 2);
            g.Children.Add(vx);
            if (y is null) return;
            var vy = Kit.Text(y, r == 0 ? "CaptionText" : "NumberText");
            vy.HorizontalAlignment = HorizontalAlignment.Right;
            Grid.SetRow(vy, r); Grid.SetColumn(vy, 2);
            g.Children.Add(vy);
        }
        Row(0, "", "x (mm)", "y (mm)");
        Row(1, "EFL", Fmt.Short(p.EflX), Fmt.Short(p.EflY));
        Row(2, "BFL", Fmt.Short(p.BflX), Fmt.Short(p.BflY));
        Row(3, "Squeeze", FirstOrderData.SqueezeText(p.Squeeze), null, "Squeeze: " + (d.SqueezeDefinition ?? "signed EFL y / EFL x"));
        if (d.Symmetry is not null) Row(4, "Symmetry", d.SymmetryTitle, null);
        s.Children.Add(g);
        s.Children.Add(Kit.Caption("At λ = " + Fmt.Wavelength(p.WavelengthUm) + " (primary)"));
        if (p.Doe.Count > 0) s.Children.Add(Doe(p));
        foreach (var w in d.Warnings)
        {
            var t = Kit.Caption("⚠ " + w, wrap: true);
            t.Foreground = new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 0xE0, 0x78, 0x00));
            s.Children.Add(t);
        }
        var others = d.Others.ToList();
        if (others.Count > 0)
        {
            var t = new Grid { ColumnSpacing = 12, RowSpacing = 3 };
            string[] heads = ["λ µm", "EFL x", "EFL y", "BFL x", "BFL y", "Squeeze"];
            for (var c = 0; c < heads.Length; c++)
            {
                t.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                var h = Kit.Caption(heads[c]);
                h.HorizontalAlignment = HorizontalAlignment.Right;
                Grid.SetColumn(h, c);
                t.Children.Add(h);
            }
            t.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            for (var r = 0; r < others.Count; r++)
            {
                t.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
                var o = others[r];
                string[] cells = [Fmt.Fixed(o.WavelengthUm, 4), Fmt.Short(o.EflX), Fmt.Short(o.EflY), Fmt.Short(o.BflX), Fmt.Short(o.BflY), FirstOrderData.SqueezeText(o.Squeeze)];
                for (var c = 0; c < cells.Length; c++)
                {
                    var v = Kit.Text(cells[c], "NumberText", 12);
                    v.HorizontalAlignment = HorizontalAlignment.Right;
                    Grid.SetRow(v, r + 1); Grid.SetColumn(v, c);
                    t.Children.Add(v);
                }
            }
            s.Children.Add(new Expander
            {
                Header = Kit.Inv($"Other wavelengths ({others.Count})"), Content = new ScrollViewer { Content = t, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled },
                HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Left, Padding = new Thickness(8),
            });
        }
        return s;
    }

    static UIElement Doe(FirstOrderData.XyRow p)
    {
        var g = Kit.KeyValues(p.Doe.Select(d => (Kit.Inv($"DOE power, surface {d.Surface}"), Fmt.Value(d.Power, "1/mm"))), 13);
        ToolTipService.SetToolTip(g, "Phase power of the trace order (base refractive power excluded)");
        return g;
    }
}

/// <summary>
/// <c>glass_info</c>: the glass's catalogue data, its dispersion curve with the system
/// wavelengths marked, its indices at those wavelengths and its properties.
/// </summary>
public sealed partial class GlassDataView : UserControl
{
    readonly Border _header;

    public GlassDataView(GlassInfoData d)
    {
        IsTabStop = false;
        var stack = new StackPanel { Spacing = 18, Padding = new Thickness(14, 10, 14, 14) };

        // Header: name, catalogue, offset badge and the four key values.
        var head = new StackPanel { Spacing = 10 };
        var title = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        var name = Kit.Text(d.Name, "BodyText", 18);
        name.FontWeight = FontWeights.SemiBold;
        name.IsTextSelectionEnabled = true;
        title.Children.Add(name);
        if (d.Catalog.Length > 0) { var c = Kit.Text(d.Catalog, "SecondaryText"); c.VerticalAlignment = VerticalAlignment.Bottom; c.Margin = new Thickness(0, 0, 0, 2); title.Children.Add(c); }
        if (d.HasOffset)
        {
            var badge = new Border { CornerRadius = new CornerRadius(9), Padding = new Thickness(8, 1, 8, 2), VerticalAlignment = VerticalAlignment.Center, Child = Kit.Text("Offset", "CaptionText") };
            badge.Background = ThemeLookup.Brush(this, "AccentFillColorSecondaryBrush");
            ToolTipService.SetToolTip(badge, $"Δn_d {Fmt.Signed(d.Dnd, 5).Trim()}, ΔV_d {Fmt.Signed(d.Dvd, 2).Trim()} from the catalog values".Replace("_", ""));
            title.Children.Add(badge);
        }
        head.Children.Add(title);
        var stats = new FlowPanel { Spacing = 26, LineSpacing = 10 };
        stats.Children.Add(Stat("n_d", Fmt.Fixed(d.Nd, 6), "Refractive index at the d line (587.56 nm)"));
        stats.Children.Add(Stat("V_d", Fmt.Fixed(d.Vd, 2), "Abbe number (nd − 1) / (nF − nC)"));
        stats.Children.Add(Stat("P_g,F", Fmt.Fixed(d.Pgf, 4), "Relative partial dispersion (ng − nF) / (nF − nC)"));
        stats.Children.Add(Stat("n_e", Fmt.Fixed(d.Ne, 6), "Refractive index at the e line (546.07 nm)"));
        head.Children.Add(stats);
        if (d.HasOffset) head.Children.Add(Kit.Sub($"Catalog n_d {Fmt.Fixed(d.CatalogNd, 6)}, V_d {Fmt.Fixed(d.CatalogVd, 2)}", 11));
        _header = Kit.Card(head);
        stack.Children.Add(_header);

        // Dispersion.
        var range = double.IsFinite(d.MinUm) && double.IsFinite(d.MaxUm) ? $"valid {Fmt.Fixed(d.MinUm, 3)}–{Fmt.Fixed(d.MaxUm, 3)} µm" : "";
        if (d.Curve.Count > 1)
        {
            var chart = new PlotCanvas((ds, size, style) => DrawDispersion(ds, size, style, d)) { Height = 220 };
            AutomationProperties.SetName(chart, "Refractive index versus wavelength");
            var body = new StackPanel { Spacing = 6 };
            body.Children.Add(chart);
            if (d.Indices.Count > 0)
                body.Children.Add(new Legend(d.Indices.Select(i => new LegendEntry(Fmt.Wavelength(i.Um), SeriesColor.Wavelength(i.Um), LegendMark.Dot)).ToList()));
            stack.Children.Add(Section("Dispersion", range, body));
        }
        else stack.Children.Add(Section("Dispersion", range, Kit.Caption("Set Curve Samples above zero to plot the index over the glass's range.", wrap: true)));

        // Indices.
        if (d.Indices.Count > 0)
        {
            var env = string.Join(", ", new[]
            {
                double.IsFinite(d.TemperatureC) ? Fmt.Fixed(d.TemperatureC, 1) + " °C" : null,
                double.IsFinite(d.PressureAtm) ? Fmt.Fixed(d.PressureAtm, 3) + " atm" : null,
            }.OfType<string>());
            stack.Children.Add(Section("Indices", env, Indices(d)));
        }

        // Properties.
        var rows = new List<(string, string)>();
        if (d.Formula is not null) rows.Add(("Dispersion formula", d.Formula));
        if (range.Length > 0) rows.Add(("Valid range", range[6..]));
        rows.Add(("Thermo-optic data (dn/dT)", d.HasThermalData ? "Yes" : "None"));
        rows.Add(("Thermal expansion", d.Cte is double cte ? Fmt.Fixed(cte * 1e6, 2) + " × 10⁻⁶/K" : "—"));
        if (d.Density is double den) rows.Add(("Density", Fmt.Fixed(den, 3) + " g/cm³"));
        if (d.RelativeCost is double cost) rows.Add(("Relative cost", Fmt.Fixed(cost, 2)));
        var props = new StackPanel { Spacing = 6 };
        props.Children.Add(Kit.KeyValues(rows, 13));
        if (d.Comment is not null) props.Children.Add(Kit.Caption(d.Comment, wrap: true));
        stack.Children.Add(Section("Properties", "", props));

        Content = Kit.Page(stack);
        ActualThemeChanged += (_, _) => Kit.ThemeCard(_header, ActualTheme);
        Loaded += (_, _) => Kit.ThemeCard(_header, ActualTheme);
    }

    static FrameworkElement Stat(string label, string value, string help)
    {
        var p = new StackPanel { Spacing = 1 };
        p.Children.Add(Kit.Sub(label, 11));
        var v = Kit.Text(value, "NumberText", 18);
        v.IsTextSelectionEnabled = true;
        p.Children.Add(v);
        ToolTipService.SetToolTip(p, help);
        AutomationProperties.SetName(p, $"{Kit.Spoken(label)}: {value}");
        return p;
    }

    static FrameworkElement Section(string title, string subtitle, UIElement body)
    {
        var p = new StackPanel { Spacing = 8 };
        var h = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        h.Children.Add(Kit.Heading(title));
        if (subtitle.Length > 0) { var s = Kit.Caption(subtitle); s.VerticalAlignment = VerticalAlignment.Bottom; s.Margin = new Thickness(0, 0, 0, 1); h.Children.Add(s); }
        AutomationProperties.SetHeadingLevel(h, Microsoft.UI.Xaml.Automation.Peers.AutomationHeadingLevel.Level2);
        p.Children.Add(h);
        p.Children.Add(body);
        return p;
    }

    static UIElement Indices(GlassInfoData d)
    {
        var g = new Grid { ColumnSpacing = 18, RowSpacing = 4, HorizontalAlignment = HorizontalAlignment.Left };
        string[] heads = ["λ (µm)", "n", "n cat.", "α (1/mm)"];
        string[] tips = ["", "Index relative to air at the analysis temperature and pressure", "Index at the catalog's reference conditions", "Internal absorption coefficient"];
        for (var c = 0; c < heads.Length; c++)
        {
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var h = Kit.Text(heads[c], "ColumnHeaderText", 11);
            h.HorizontalAlignment = c == 0 ? HorizontalAlignment.Left : HorizontalAlignment.Right;
            if (tips[c].Length > 0) ToolTipService.SetToolTip(h, tips[c]);
            Grid.SetColumn(h, c);
            g.Children.Add(h);
        }
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        for (var r = 0; r < d.Indices.Count; r++)
        {
            var i = d.Indices[r];
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var first = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
            first.Children.Add(new WavelengthDot(i.Um));
            first.Children.Add(Kit.Text(Fmt.Fixed(i.Um, 4), "NumberText", 13));
            if (!i.InRange)
            {
                var warn = Kit.Text("⚠", "BodyText", 12);
                warn.Foreground = new SolidColorBrush(global::Windows.UI.Color.FromArgb(255, 0xE0, 0x78, 0x00));
                ToolTipService.SetToolTip(warn, "Outside the glass's data range: the index is extrapolated");
                first.Children.Add(warn);
            }
            Grid.SetRow(first, r + 1);
            g.Children.Add(first);
            string[] cells = [Fmt.Fixed(i.N, 6), Fmt.Fixed(i.NCatalog, 6), double.IsFinite(i.AbsorptionPerMm) ? Fmt.Short(i.AbsorptionPerMm) : "—"];
            for (var c = 0; c < cells.Length; c++)
            {
                var v = Kit.Text(cells[c], c == 0 ? "NumberText" : "SecondaryNumberText", 13);
                v.HorizontalAlignment = HorizontalAlignment.Right;
                v.IsTextSelectionEnabled = true;
                Grid.SetRow(v, r + 1); Grid.SetColumn(v, c + 1);
                g.Children.Add(v);
            }
        }
        return new ScrollViewer { Content = g, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled, HorizontalScrollMode = ScrollMode.Auto, VerticalScrollMode = ScrollMode.Disabled };
    }

    /// <summary>n against λ over the glass's range, the system wavelengths as coloured points on it (triangles outside the data range).</summary>
    static void DrawDispersion(Microsoft.Graphics.Canvas.CanvasDrawingSession ds, global::Windows.Foundation.Size size, Plotting.PlotStyle style, GlassInfoData d)
    {
        var pts = d.Curve.Where(p => double.IsFinite(p.Um) && double.IsFinite(p.N)).ToList();
        var marks = d.Indices.Where(i => double.IsFinite(i.Um) && double.IsFinite(i.N)).ToList();
        var xs = pts.Select(p => p.Um).Concat(marks.Select(m => m.Um)).ToList();
        var ys = pts.Select(p => p.N).Concat(marks.Select(m => m.N)).ToList();
        var f = new Frame { XTitle = "Wavelength (µm)", YTitle = "n", YTickSpacing = 45 };
        (f.XMin, f.XMax) = (xs.Min(), xs.Max());
        if (!(f.XMax > f.XMin)) (f.XMin, f.XMax) = (0.3, 1);
        (f.YMin, f.YMax) = Plotting.Ticks.Range(ys);
        var r = f.Draw(ds, f.Inside(ds, new global::Windows.Foundation.Rect(0, 0, size.Width, size.Height)), style);
        if (r.IsEmpty) return;
        using (ds.CreateLayer(1, r))
        {
            var dash = new Microsoft.Graphics.Canvas.Geometry.CanvasStrokeStyle { CustomDashStyle = [2, 3] };
            foreach (var m in marks)
                ds.DrawLine(f.X(m.Um), (float)r.Y, f.X(m.Um), (float)r.Bottom, Palette.Wavelength(m.Um, style.Dark).WithAlpha(0.4), 1, dash);
            var ink = style.Title.WithAlpha(0.75);
            for (var k = 1; k < pts.Count; k++) ds.DrawLine(f.X(pts[k - 1].Um), f.Y(pts[k - 1].N), f.X(pts[k].Um), f.Y(pts[k].N), ink, 1.6f);
        }
        foreach (var m in marks)
        {
            var c = Palette.Wavelength(m.Um, style.Dark);
            if (m.InRange) ds.FillCircle(f.X(m.Um), f.Y(m.N), 4, c);
            else Glyphs.Triangle(ds, f.X(m.Um), f.Y(m.N), 5, c);
        }
    }
}

/// <summary>A wavelength's colour dot (its spectral slot), theme-aware.</summary>
public sealed partial class WavelengthDot : UserControl
{
    public WavelengthDot(double um)
    {
        IsTabStop = false;
        var e = new Microsoft.UI.Xaml.Shapes.Ellipse { Width = 8, Height = 8, VerticalAlignment = VerticalAlignment.Center };
        Content = e;
        void Paint() => e.Fill = new SolidColorBrush(Palette.Wavelength(um, ActualTheme == ElementTheme.Dark));
        Loaded += (_, _) => Paint();
        ActualThemeChanged += (_, _) => Paint();
        VerticalAlignment = VerticalAlignment.Center;
    }
}

static class Glyphs
{
    public static void Triangle(Microsoft.Graphics.Canvas.CanvasDrawingSession ds, float x, float y, float r, global::Windows.UI.Color c)
    {
        using var g = Microsoft.Graphics.Canvas.Geometry.CanvasGeometry.CreatePolygon(ds, [new(x, y - r), new(x + r * 0.9f, y + r * 0.6f), new(x - r * 0.9f, y + r * 0.6f)]);
        ds.FillGeometry(g, c);
    }
}
