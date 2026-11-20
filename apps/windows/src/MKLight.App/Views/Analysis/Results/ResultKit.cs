using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Support;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>Small building blocks of the dedicated result views: text, pickers, tables, subscripts.</summary>
static class Kit
{
    public static Style Style(string key) => (Style)Application.Current.Resources[key];

    public static TextBlock Text(string s, string style = "BodyText", double? size = null)
    {
        var t = new TextBlock { Text = s, Style = Style(style) };
        if (size is double z) t.FontSize = z;
        return t.Tabular();
    }

    public static TextBlock Caption(string s, bool wrap = false)
    {
        var t = Text(s, "CaptionText");
        if (wrap) { t.TextWrapping = TextWrapping.Wrap; t.TextTrimming = TextTrimming.None; }
        return t;
    }

    public static TextBlock Heading(string s) => Text(s, "PanelTitleText", 12);

    /// <summary>
    /// A label with <c>_</c> subscripts set as real subscripts (<c>S_III</c>, <c>n_d</c>,
    /// <c>P_g,F</c>): the part of a word after <c>_</c> smaller and lowered. Unicode has no
    /// subscript capitals, so the Seidel terms need this.
    /// </summary>
    public static FrameworkElement Sub(string label, double size = 12, string style = "SecondaryText", global::Windows.UI.Text.FontWeight? weight = null)
    {
        var panel = new StackPanel { Orientation = Orientation.Horizontal, VerticalAlignment = VerticalAlignment.Center };
        foreach (var (text, sub) in Pieces(label))
        {
            var t = new TextBlock { Text = text, Style = Style(style), FontSize = sub ? size * 0.72 : size, VerticalAlignment = VerticalAlignment.Top, TextTrimming = TextTrimming.None };
            if (weight is { } w) t.FontWeight = w;
            if (sub) t.Margin = new Thickness(0.5, size * 0.55, 0, 0);
            panel.Children.Add(t.Tabular());
        }
        AutomationProperties.SetName(panel, Spoken(label));
        return panel;
    }

    /// <summary>Plain and subscript pieces: a subscript runs from <c>_</c> to the end of its word.</summary>
    public static List<(string Text, bool Sub)> Pieces(string label)
    {
        var o = new List<(string, bool)>();
        void Add(string s, bool sub)
        {
            if (s.Length == 0) return;
            if (o.Count > 0 && o[^1].Item2 == sub) o[^1] = (o[^1].Item1 + s, sub);
            else o.Add((s, sub));
        }
        var words = label.Split(' ');
        for (var k = 0; k < words.Length; k++)
        {
            if (k > 0) Add(" ", false);
            var w = words[k];
            var u = w.IndexOf('_');
            if (u > 0) { Add(w[..u], false); Add(w[(u + 1)..], true); }
            else Add(w, false);
        }
        return o;
    }

    public static string Spoken(string label) => label.Replace('_', ' ');

    /// <summary>A small coloured dot (legend swatch); hollow draws a ring.</summary>
    public static FrameworkElement Dot(Color c, double d = 7, bool hollow = false) => hollow
        ? new Ellipse { Width = d, Height = d, Stroke = new SolidColorBrush(c), StrokeThickness = 1.2, VerticalAlignment = VerticalAlignment.Center }
        : new Ellipse { Width = d, Height = d, Fill = new SolidColorBrush(c), VerticalAlignment = VerticalAlignment.Center };

    public static FrameworkElement Swatch(Color c, double w = 12, double h = 8, double opacity = 1) =>
        new Rectangle { Width = w, Height = h, RadiusX = 2, RadiusY = 2, Fill = new SolidColorBrush(c) { Opacity = opacity }, VerticalAlignment = VerticalAlignment.Center };

    public static FrameworkElement LegendItem(FrameworkElement swatch, FrameworkElement label)
    {
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
        p.Children.Add(swatch);
        p.Children.Add(label);
        return p;
    }

    public static FrameworkElement LegendItem(FrameworkElement swatch, string label) => LegendItem(swatch, Text(label, "SecondaryText", 11));

    /// <summary>A two-column key / value table, values right-aligned with tabular figures.</summary>
    public static Grid KeyValues(IEnumerable<(string Key, string Value)> rows, double size = 12, bool subscripts = false)
    {
        // Keys take the room left and trim in narrow panes; values are never cut.
        var g = new Grid { ColumnSpacing = 12, RowSpacing = 3, HorizontalAlignment = HorizontalAlignment.Left, MaxWidth = 380 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var r = 0;
        foreach (var (key, value) in rows)
        {
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            FrameworkElement k = subscripts ? Sub(key, size) : Text(key, "SecondaryText", size);
            ToolTipService.SetToolTip(k, Spoken(key));
            var v = Text(value, "NumberText", size);
            v.HorizontalAlignment = HorizontalAlignment.Right;
            v.IsTextSelectionEnabled = true;
            Grid.SetRow(k, r);
            Grid.SetRow(v, r);
            Grid.SetColumn(v, 1);
            g.Children.Add(k);
            g.Children.Add(v);
            r++;
        }
        return g;
    }

    /// <summary>A quiet card: a light fill behind grouped content (the Mac's quinary fill).</summary>
    public static Border Card(UIElement child) => new()
    {
        Child = child,
        Padding = new Thickness(12, 10, 12, 10),
        CornerRadius = new CornerRadius(8),
        Background = new SolidColorBrush(Color.FromArgb(0, 0, 0, 0)),
        Tag = "card",
    };

    /// <summary>Applies the theme's card fill (call on load and theme change).</summary>
    public static void ThemeCard(Border b, ElementTheme theme) =>
        b.Background = new SolidColorBrush(theme == ElementTheme.Dark ? Color.FromArgb(0x0F, 255, 255, 255) : Color.FromArgb(0x0A, 0, 0, 0));

    /// <summary>A scrolling page of sections (vertical only).</summary>
    public static ScrollViewer Page(UIElement content) => new()
    {
        Content = content,
        HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
        VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
    };

    public static Color Accent(FrameworkElement e) =>
        ThemeLookup.Brush(e, "AccentFillColorDefaultBrush") is SolidColorBrush b ? b.Color : Color.FromArgb(255, 0x00, 0x67, 0xC0);

    public static string Inv(FormattableString s) => FormattableString.Invariant(s);
}

/// <summary>
/// A choice among a few options: a small segmented control while it fits, a menu
/// (ComboBox) when the pane is narrower (below about 240 px, or whatever the
/// segments need), so narrow panes never overflow.
/// </summary>
public sealed partial class Choice : UserControl
{
    readonly string[] _options;
    readonly StackPanel _segments = new() { Orientation = Orientation.Horizontal };
    readonly Border _segmentBox;
    readonly ComboBox _menu;
    readonly List<Button> _buttons = [];
    int _selected;
    bool _asMenu;

    public Choice(string name, IReadOnlyList<string> options, int selected = 0, double menuBelow = 240)
    {
        _options = options.ToArray();
        _selected = Math.Clamp(selected, 0, Math.Max(_options.Length - 1, 0));
        MenuBelow = menuBelow;
        IsTabStop = false;
        HorizontalAlignment = HorizontalAlignment.Left;
        for (var i = 0; i < _options.Length; i++)
        {
            var k = i;
            var b = new Button
            {
                Content = _options[i].Contains('_') ? Kit.Sub(_options[i], 12, "BodyText") : _options[i], FontSize = 12, Padding = new Thickness(10, 2, 10, 3), MinHeight = 22, MinWidth = 0,
                CornerRadius = new CornerRadius(11), BorderThickness = new Thickness(0),
            };
            b.Click += (_, _) => Select(k);
            AutomationProperties.SetName(b, Kit.Spoken(_options[i]));
            _buttons.Add(b);
            _segments.Children.Add(b);
        }
        _segmentBox = new Border { Child = _segments, CornerRadius = new CornerRadius(13), BorderThickness = new Thickness(1), Padding = new Thickness(2) };
        _menu = new ComboBox { ItemsSource = _options.Select(o => o.Replace("_", "")).ToList(), SelectedIndex = _selected, FontSize = 12, MinWidth = 0, Padding = new Thickness(8, 2, 6, 3), MinHeight = 26 };
        _menu.SelectionChanged += (_, _) => { if (_menu.SelectedIndex >= 0) Select(_menu.SelectedIndex); };
        AutomationProperties.SetName(_menu, name);
        AutomationProperties.SetName(this, name);
        Content = _segmentBox;
        ActualThemeChanged += (_, _) => Restyle();
        Loaded += (_, _) => Restyle();
    }

    /// <summary>The width below which the choice is a menu.</summary>
    public double MenuBelow { get; set; }

    public int Selected => _selected;
    public event Action<int>? Changed;

    public void Select(int k)
    {
        if (k == _selected || k < 0 || k >= _options.Length) return;
        _selected = k;
        if (_menu.SelectedIndex != k) _menu.SelectedIndex = k;
        Restyle();
        Changed?.Invoke(k);
    }

    protected override Size MeasureOverride(Size available)
    {
        _segmentBox.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
        var menu = available.Width < Math.Max(MenuBelow, 0) && double.IsFinite(available.Width) || _segmentBox.DesiredSize.Width > available.Width;
        if (menu != _asMenu)
        {
            _asMenu = menu;
            Content = menu ? _menu : _segmentBox;
        }
        return base.MeasureOverride(available);
    }

    void Restyle()
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var ink = dark ? Color.FromArgb(0xE6, 255, 255, 255) : Color.FromArgb(0xE4, 0, 0, 0);
        var quiet = dark ? Color.FromArgb(0x99, 255, 255, 255) : Color.FromArgb(0x8F, 0, 0, 0);
        var fill = dark ? Color.FromArgb(0x24, 255, 255, 255) : Color.FromArgb(0x10, 0, 0, 0);
        _segmentBox.BorderBrush = new SolidColorBrush(dark ? Color.FromArgb(0x1A, 255, 255, 255) : Color.FromArgb(0x17, 0, 0, 0));
        for (var i = 0; i < _buttons.Count; i++)
        {
            var on = i == _selected;
            _buttons[i].Background = new SolidColorBrush(on ? fill : Color.FromArgb(0, 0, 0, 0));
            _buttons[i].Foreground = new SolidColorBrush(on ? ink : quiet);
            AutomationProperties.SetItemStatus(_buttons[i], on ? "Selected" : "");
        }
    }
}

/// <summary>A menu of a result's items (fields, wavelengths, maps); one item is a caption, none hides it.</summary>
public sealed partial class ItemPicker : UserControl
{
    public ItemPicker(string name, IReadOnlyList<string> titles, Action<int> changed, int selected = 0)
    {
        IsTabStop = false;
        HorizontalAlignment = HorizontalAlignment.Left;
        if (titles.Count > 1)
        {
            var c = new ComboBox { ItemsSource = titles.ToList(), SelectedIndex = selected, FontSize = 12, MinWidth = 0, MinHeight = 26, Padding = new Thickness(8, 2, 6, 3), MaxWidth = 420 };
            c.SelectionChanged += (_, _) => { if (c.SelectedIndex >= 0) changed(c.SelectedIndex); };
            AutomationProperties.SetName(c, name);
            Content = c;
        }
        else if (titles.Count == 1)
        {
            Content = Kit.Text(titles[0], "PanelTitleText", 12);
        }
    }
}

/// <summary>Arranges children in one row when wide, else stacked (map beside its table or chart).</summary>
public sealed partial class SideBySide : Panel
{
    public double Breakpoint { get; set; } = 560;
    public double Gap { get; set; } = 12;
    /// <summary>Width of the second child when side by side (0: half).</summary>
    public double SideWidth { get; set; }
    /// <summary>Height share of the first child when stacked (0…1; 0: its desired height).</summary>
    public double StackedShare { get; set; }
    /// <summary>The height the first child wants at a width when stacked (a map: about square); limits the share.</summary>
    public Func<double, double>? FirstHeight { get; set; }

    double StackedFirst(double w, double h)
    {
        var share = StackedShare > 0 ? h * StackedShare : double.NaN;
        if (FirstHeight is { } f) share = double.IsNaN(share) ? f(w) : Math.Min(share, f(w));
        return share;
    }

    bool Wide(double w) => w >= Breakpoint;

    protected override Size MeasureOverride(Size available)
    {
        var w = double.IsFinite(available.Width) ? available.Width : 800;
        var h = double.IsFinite(available.Height) ? available.Height : 600;
        if (Children.Count < 2)
        {
            foreach (var c in Children) c.Measure(new Size(w, h));
            return new Size(w, Children.Count > 0 ? Children[0].DesiredSize.Height : 0);
        }
        if (Wide(w))
        {
            var side = SideWidth > 0 ? SideWidth : (w - Gap) / 2;
            Children[0].Measure(new Size(Math.Max(w - side - Gap, 0), h));
            Children[1].Measure(new Size(side, h));
            return new Size(w, double.IsFinite(available.Height) ? h : Math.Max(Children[0].DesiredSize.Height, Children[1].DesiredSize.Height));
        }
        var fixedFirst = StackedFirst(w, h);
        Children[0].Measure(new Size(w, double.IsNaN(fixedFirst) ? h : fixedFirst));
        var h0 = double.IsNaN(fixedFirst) ? Children[0].DesiredSize.Height : fixedFirst;
        Children[1].Measure(new Size(w, Math.Max(h - h0 - Gap, 0)));
        return new Size(w, double.IsFinite(available.Height) ? h : h0 + Gap + Children[1].DesiredSize.Height);
    }

    protected override Size ArrangeOverride(Size final)
    {
        if (Children.Count < 2)
        {
            foreach (var c in Children) c.Arrange(new Rect(0, 0, final.Width, final.Height));
            return final;
        }
        if (Wide(final.Width))
        {
            var side = SideWidth > 0 ? SideWidth : (final.Width - Gap) / 2;
            Children[0].Arrange(new Rect(0, 0, Math.Max(final.Width - side - Gap, 0), final.Height));
            Children[1].Arrange(new Rect(final.Width - side, 0, side, final.Height));
            return final;
        }
        var sf = StackedFirst(final.Width, final.Height);
        var h0 = double.IsNaN(sf) ? Children[0].DesiredSize.Height : sf;
        Children[0].Arrange(new Rect(0, 0, final.Width, h0));
        Children[1].Arrange(new Rect(0, h0 + Gap, final.Width, Math.Max(final.Height - h0 - Gap, 0)));
        return final;
    }
}
