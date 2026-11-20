using System.Globalization;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;

namespace MKLight.App.Views.Glass;

/// <summary>
/// The detail column of the Glass Catalogs window (<c>mk_session_glass_details</c>): Use for
/// Surface n (the one primary action), Copy Name, Edit / Delete or Duplicate, the status and
/// substitution settings, the optical data, an index curve, the dispersion formula and its
/// coefficients, physical and catalog data, and the catalog's load notes about the glass.
/// </summary>
public sealed class GlassDetailView : UserControl
{
    readonly GlassBrowserModel _model;
    readonly ScrollViewer _scroll = new() { HorizontalScrollMode = ScrollMode.Disabled, Padding = new Thickness(0, 0, 4, 0) };
    bool _building;

    public GlassDetailView(GlassBrowserModel model)
    {
        _model = model;
        Content = _scroll;
        model.DetailsChanged += Build;
        model.Changed += Build;
        ActualThemeChanged += (_, _) => Build();
        Build();
    }

    /// <summary>The window's actions on the shown glass.</summary>
    public Action? Copy, Edit, Delete, Duplicate;

    static Style S(string key) => (Style)Application.Current.Resources[key];

    /// <summary>Builds again (the design's selected surface changed: Use for Surface n).</summary>
    public void Refresh()
    {
        _shown = null;
        Build();
    }

    string? _shown;
    int _shownRevision;

    void Build()
    {
        var d = _model.Details;
        var row = _model.SelectedRow;
        if (d is null || row is null || d.Row.Id != row.Id)
        {
            _shown = null;
            _scroll.Content = Placeholder();
            return;
        }
        // Rebuilding on every table change would reset the scroll position: only when the glass or its data changed.
        var key = d.Row.Id + "|" + d.Row.Status.Code + "|" + d.Row.ExcludeSubstitution + "|" + _model.Assignment.Index + "|" + (_model.Target?.Key.GetHashCode() ?? 0) + "|" + ActualTheme;
        if (key == _shown && _shownRevision == _model.Library.Revision && _scroll.Content is StackPanel) return;
        _shown = key;
        _shownRevision = _model.Library.Revision;
        _building = true;
        var stack = new StackPanel { Spacing = 14, Padding = new Thickness(4, 4, 8, 20) };
        stack.Children.Add(Header(d));
        stack.Children.Add(Actions(d));
        stack.Children.Add(Section("Availability", Availability(d)));
        if (Index(d) is { } index) stack.Children.Add(Section("Index", index));
        stack.Children.Add(Section("Optical Data", Optical(d)));
        stack.Children.Add(Section("Dispersion", Dispersion(d)));
        stack.Children.Add(Section("Physical and Catalog Data", Physical(d)));
        if (Notes(d) is { } notes) stack.Children.Add(Section("Catalog Notes", notes));
        if (d.Comment.Length > 0)
            stack.Children.Add(new TextBlock { Text = d.Comment, Style = S("SecondaryText"), TextWrapping = TextWrapping.Wrap, IsTextSelectionEnabled = true });
        _scroll.Content = stack;
        _building = false;
    }

    UIElement Placeholder()
    {
        var p = new StackPanel { Spacing = 6, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(16, 160, 16, 0) };
        p.Children.Add(new FontIcon { Glyph = "\uEA3A", FontSize = 28, Foreground = ThemeLookup.Brush(this, "TextFillColorTertiaryBrush") });
        p.Children.Add(new TextBlock { Text = "No Glass Selected", Style = S("PanelTitleText"), FontSize = 15, HorizontalAlignment = HorizontalAlignment.Center });
        p.Children.Add(new TextBlock { Text = "Select a glass to see its data and use it.", Style = S("SecondaryText"), HorizontalAlignment = HorizontalAlignment.Center, TextWrapping = TextWrapping.Wrap, TextAlignment = TextAlignment.Center });
        return p;
    }

    UIElement Header(GlassDetails d)
    {
        var p = new StackPanel { Spacing = 4 };
        p.Children.Add(new TextBlock { Text = d.Row.Name, FontFamily = (FontFamily)Application.Current.Resources["UIDisplayFont"], FontSize = 22, FontWeight = FontWeights.SemiBold, IsTextSelectionEnabled = true });
        var line = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 10 };
        line.Children.Add(new TextBlock { Text = GlassBrowserModel.ScopeTitle(d.Row.Catalog), Style = S("SecondaryText") });
        line.Children.Add(StatusBadge(d.Row));
        p.Children.Add(line);
        return p;
    }

    UIElement StatusBadge(GlassRow row)
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var color = GlassRowItem.StatusColor(row.Status, dark);
        var obsolete = row.Status == GlassStatus.Obsolete;
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
        p.Children.Add(new Ellipse
        {
            Width = 7, Height = 7, VerticalAlignment = VerticalAlignment.Center, StrokeThickness = 1.2,
            Fill = new SolidColorBrush(obsolete ? Microsoft.UI.Colors.Transparent : color), Stroke = new SolidColorBrush(color),
        });
        p.Children.Add(new TextBlock { Text = row.Status.Title, Style = S("BodyText") });
        if (row.Overridden)
        {
            var mark = new FontIcon { Glyph = "\uE77B", FontSize = 11, Foreground = ThemeLookup.Brush(this, "TextFillColorSecondaryBrush") };
            ToolTipService.SetToolTip(mark, "Your setting (the catalog says otherwise)");
            p.Children.Add(mark);
        }
        return p;
    }

    UIElement Actions(GlassDetails d)
    {
        var user = _model.Library.IsUserCatalog(d.Row.Catalog);
        var (index, why) = _model.Assignment;
        var p = new StackPanel { Spacing = 8 };
        var use = new Button
        {
            Style = S("AccentButtonStyle"), HorizontalAlignment = HorizontalAlignment.Stretch, Height = 34, CornerRadius = new CornerRadius(17),
            IsEnabled = index is not null, Content = IconText("\uE8B0", index is int i ? $"Use for Surface {i}" : "Use for Surface"),
        };
        ToolTipService.SetToolTip(use, why ?? "Make the selected surface's following medium this glass");
        use.Click += (_, _) => _model.UseSelected();
        p.Children.Add(use);
        if (why is not null) p.Children.Add(new TextBlock { Text = why, Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap });
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        buttons.Children.Add(Capsule("Copy Name", "\uE8C8", Copy));
        if (user)
        {
            buttons.Children.Add(Capsule("Edit…", "\uE70F", Edit));
            buttons.Children.Add(Capsule("Delete", "\uE74D", Delete));
        }
        else buttons.Children.Add(Capsule("Duplicate…", "\uE8F4", Duplicate, "Copy this glass into a user catalog to edit it"));
        p.Children.Add(buttons);
        return p;
    }

    static StackPanel IconText(string glyph, string text)
    {
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
        p.Children.Add(new FontIcon { Glyph = glyph, FontSize = 13 });
        p.Children.Add(new TextBlock { Text = text });
        return p;
    }

    static Button Capsule(string text, string glyph, Action? action, string? tip = null)
    {
        var b = new Button { Style = S("CapsuleButton"), Content = IconText(glyph, text), Height = 30 };
        if (tip is not null) ToolTipService.SetToolTip(b, tip);
        b.Click += (_, _) => action?.Invoke();
        return b;
    }

    UIElement Availability(GlassDetails d)
    {
        var row = d.Row;
        var g = new Grid { ColumnSpacing = 12, RowSpacing = 6 };
        g.ColumnDefinitions.Add(new ColumnDefinition());
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        for (var i = 0; i < 3; i++) g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

        var status = new ComboBox { MinWidth = 150, HorizontalAlignment = HorizontalAlignment.Right };
        status.Items.Add(new ComboBoxItem { Content = $"Catalog ({d.CatalogStatus.Title})", Tag = -1 });
        foreach (var s in GlassStatus.Choices) status.Items.Add(new ComboBoxItem { Content = s.Title, Tag = s.Code });
        var own = _model.Library.OverrideFor(row.Catalog, row.Name)?.Status;
        status.SelectedIndex = own is { } o ? 1 + GlassStatus.Choices.ToList().IndexOf(o) : 0;
        status.SelectionChanged += (_, _) =>
        {
            if (_building || status.SelectedItem is not ComboBoxItem { Tag: int code }) return;
            _model.SetStatus(code < 0 ? null : new GlassStatus(code), row);
        };
        AddRow(g, 0, "Status", status);

        var use = new ToggleSwitch { IsOn = !row.ExcludeSubstitution, OnContent = "", OffContent = "", HorizontalAlignment = HorizontalAlignment.Right };
        // Only the switch itself: no room for On / Off text.
        use.Resources["ToggleSwitchThemeMinWidth"] = 0.0;
        use.Resources["ToggleSwitchPreContentMargin"] = 0.0;
        use.Resources["ToggleSwitchPostContentMargin"] = 0.0;
        use.Toggled += (_, _) =>
        {
            if (_building) return;
            var excluded = !use.IsOn;
            _model.SetExcluded(excluded == d.CatalogExcludeSubstitution ? null : excluded, row);
        };
        AddRow(g, 1, "Use in glass substitution", use);

        var note = new TextBlock
        {
            Text = "Your settings apply to every design and are not saved in projects. Glass substitution skips obsolete, special and excluded glasses by default.",
            Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 2, 0, 0),
        };
        Grid.SetRow(note, 2);
        Grid.SetColumnSpan(note, 2);
        g.Children.Add(note);
        return g;
    }

    static void AddRow(Grid g, int row, string label, FrameworkElement control)
    {
        var l = new TextBlock { Text = label, Style = S("BodyText"), TextWrapping = TextWrapping.Wrap };
        Grid.SetRow(l, row);
        Grid.SetRow(control, row);
        Grid.SetColumn(control, 1);
        g.Children.Add(l);
        g.Children.Add(control);
    }

    /// <summary>The index curve over the glass's own wavelength range (an IR glass's too).</summary>
    static UIElement? Index(GlassDetails d) => d.IndexCurve.Count > 1 ? new IndexCurveView { Details = d } : null;

    UIElement Optical(GlassDetails d)
    {
        var g = new Rows();
        var r = d.Row;
        // An IR glass: "—", why in the tooltip and a note, and what the catalog lists (vendors list n_d 1, V_d 0).
        var tip = GlassFormat.IrTip(r);
        g.Add("nd", GlassFormat.Fixed(r.Nd, 6), tip);
        g.Add("Vd", GlassFormat.Fixed(r.Vd, 3), tip);
        if (tip is not null && GlassFormat.CatalogListed(r) is { } listed) g.Add("Catalog lists", listed, tip);
        if (r.Pgf is { } pgf) g.Add("Pg,F", GlassFormat.Fixed(pgf, 4));
        if (r.Dpgf is { } dp) g.Add("ΔPg,F", GlassFormat.Signed(dp));
        // An IR glass's file lists ΔPg,F 0 as a placeholder (its g, F, C lines are outside the range).
        if (d.CatalogDpgf is { } cdp && !(cdp == 0 && r.Pgf is null)) g.Add("ΔPg,F (catalog)", GlassFormat.Signed(cdp));
        g.Add("Range", r.UnrestrictedRange ? "unrestricted" : $"{GlassFormat.Fixed(r.MinUm, 3)} – {GlassFormat.Fixed(r.MaxUm, 3)} µm");
        foreach (var l in d.Lines) g.Add($"n{l.Line}  ({GlassFormat.Fixed(l.Um * 1000, 1)} nm)", GlassFormat.Fixed(l.N, 6));
        if (tip is not null) g.Note(r.Nd is null
            ? "No nd or Vd: the d line (587.6 nm) is outside this glass's range."
            : "No Vd: the F or C line is outside this glass's range.");
        return g.Grid;
    }

    UIElement Dispersion(GlassDetails d)
    {
        var p = new StackPanel { Spacing = 8 };
        string title = d.AbbeModel is { } m
            ? $"Model glass: nd {GlassFormat.Fixed(m.Nd, 5)}, Vd {GlassFormat.Fixed(m.Vd, 2)}, ΔPg,F {GlassFormat.Signed(m.Dpgf)}"
            : d.Formula is int f ? $"Formula {f}: {d.FormulaName}" : d.FormulaName;
        p.Children.Add(new TextBlock { Text = title, Style = S("BodyText"), TextWrapping = TextWrapping.Wrap });
        if (d.AbbeModel is null && d.Coefficients.Count > 0 && d.Formula is int formula)
        {
            var names = GlassEditorModel.CoefficientNames(formula);
            var g = new Rows();
            for (var i = 0; i < d.Coefficients.Count; i++)
                if (d.Coefficients[i] != 0 || i < names.Count) g.Add(i < names.Count ? names[i] : $"c{i}", GlassFormat.Significant(d.Coefficients[i]));
            p.Children.Add(g.Grid);
        }
        return p;
    }

    UIElement Physical(GlassDetails d)
    {
        var g = new Rows();
        var r = d.Row;
        if (r.Density is { } density) g.Add("Density", GlassFormat.Fixed(density, 3) + " g/cm³");
        if (r.Tce is { } tce) g.Add("TCE −30…70 °C", GlassFormat.Fixed(tce, 2) + " ·10⁻⁶/K");
        if (d.Tce100To300 is { } tce2) g.Add("TCE 100…300 °C", GlassFormat.Fixed(tce2, 2) + " ·10⁻⁶/K");
        if (r.RelativeCost is { } cost) g.Add("Relative cost", GlassFormat.Fixed(cost, 2));
        if (r.ClimateResistance is { } cr) g.Add("Climate (CR)", GlassFormat.Fixed(cr, 0));
        if (r.StainResistance is { } fr) g.Add("Stain (FR)", GlassFormat.Fixed(fr, 0));
        if (d.AcidResistance is { } sr) g.Add("Acid (SR)", GlassFormat.Fixed(sr, 1));
        if (d.AlkaliResistance is { } ar) g.Add("Alkali (AR)", GlassFormat.Fixed(ar, 1));
        if (d.PhosphateResistance is { } pr) g.Add("Phosphate (PR)", GlassFormat.Fixed(pr, 1));
        if (r.TransmissionMinUm is { } lo && r.TransmissionMaxUm is { } hi) g.Add("Transmission data", $"{GlassFormat.Fixed(lo, 3)} – {GlassFormat.Fixed(hi, 3)} µm");
        if (r.Lambda80Um is { } l80) g.Add("λ for 80 % (10 mm)", GlassFormat.Fixed(l80 * 1000, 0) + " nm");
        g.Add("Thermal dn/dT data", d.HasThermalData ? "yes" : "none");
        if (d.MeltFrequency is int mf) g.Add("Melt frequency", mf.ToString(CultureInfo.InvariantCulture));
        if (d.MilCode is { Length: > 0 } mil) g.Add("Glass code", mil);
        return g.Grid;
    }

    /// <summary>What the engine noticed about this glass when it read the catalog file.</summary>
    UIElement? Notes(GlassDetails d)
    {
        var warnings = _model.Catalog(d.Row.Catalog)?.Warnings.Where(w => string.Equals(w.Glass, d.Row.Name, StringComparison.OrdinalIgnoreCase)).ToList();
        if (warnings is not { Count: > 0 }) return null;
        var p = new StackPanel { Spacing = 6 };
        foreach (var w in warnings)
        {
            var line = new Grid { ColumnSpacing = 8 };
            line.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            line.ColumnDefinitions.Add(new ColumnDefinition());
            line.Children.Add(new FontIcon { Glyph = "\uE7BA", FontSize = 12, Foreground = ThemeLookup.Brush(this, "WarningBrush"), VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 3, 0, 0) });
            var text = new TextBlock { Text = (w.Line is int n ? $"Line {n}: " : "") + w.Message, Style = S("BodyText"), TextWrapping = TextWrapping.Wrap };
            Grid.SetColumn(text, 1);
            line.Children.Add(text);
            p.Children.Add(line);
        }
        return p;
    }

    /// <summary>A titled card (the Windows settings card: subtle fill, hairline stroke).</summary>
    UIElement Section(string title, UIElement content)
    {
        var p = new StackPanel { Spacing = 6 };
        p.Children.Add(new TextBlock { Text = title, Style = S("ColumnHeaderText"), Margin = new Thickness(2, 0, 0, 0) });
        p.Children.Add(new Border
        {
            Background = ThemeLookup.Brush(this, "CardBackgroundFillColorDefaultBrush"),
            BorderBrush = ThemeLookup.Brush(this, "CardStrokeColorDefaultBrush"),
            BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(8), Padding = new Thickness(12, 10, 12, 10),
            Child = content,
        });
        return p;
    }

    /// <summary>Label – value rows: labels secondary, values in tabular figures, selectable.</summary>
    sealed class Rows
    {
        public Grid Grid { get; } = new() { ColumnSpacing = 12, RowSpacing = 4 };

        public Rows()
        {
            Grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            Grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        }

        public void Add(string label, string value, string? tip = null)
        {
            var i = Grid.RowDefinitions.Count;
            Grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var l = new TextBlock { Text = label, Style = S("SecondaryText") };
            var v = new TextBlock { Text = value, Style = S("NumberText"), IsTextSelectionEnabled = true, TextWrapping = TextWrapping.Wrap };
            if (tip is not null) ToolTipService.SetToolTip(v, tip);
            Grid.SetRow(l, i);
            Grid.SetRow(v, i);
            Grid.SetColumn(v, 1);
            Grid.Children.Add(l);
            Grid.Children.Add(v);
        }

        /// <summary>A caption across both columns.</summary>
        public void Note(string text)
        {
            var i = Grid.RowDefinitions.Count;
            Grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var t = new TextBlock { Text = text, Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 4, 0, 0) };
            Grid.SetRow(t, i);
            Grid.SetColumnSpan(t, 2);
            Grid.Children.Add(t);
        }
    }
}
