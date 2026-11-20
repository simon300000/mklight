using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;
using Windows.System;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// Frame of a job page (the Mac app's <c>JobPanelFrame</c>): a header (glyph, title, subtitle), the scrolling body,
/// and the actions in a glass bar at the bottom whose buttons change with the job's state (setup → running →
/// review → done). The single primary action is the accent capsule.
/// </summary>
public sealed partial class JobPanelFrame : UserControl
{
    readonly TextBlock _subtitle;
    readonly ScrollViewer _scroller;
    readonly StackPanel _actions = new() { Orientation = Orientation.Horizontal, Spacing = 8, HorizontalAlignment = HorizontalAlignment.Stretch };
    readonly Grid _actionGrid = new() { ColumnSpacing = 8 };

    public JobPanelFrame(string title, string glyph)
    {
        var root = new Grid();
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

        var header = new Grid { ColumnSpacing = 10, Padding = new Thickness(4, 2, 4, 10) };
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        var icon = new FontIcon { Glyph = glyph, FontSize = 20, Width = 28, VerticalAlignment = VerticalAlignment.Center };
        JobUI.Bind(icon, "AccentTextFillColorPrimaryBrush", b => icon.Foreground = b);
        header.Children.Add(icon);
        var titles = new StackPanel { Spacing = 1, VerticalAlignment = VerticalAlignment.Center };
        titles.Children.Add(new TextBlock { Text = title, Style = JobUI.Style("BodyText"), FontSize = 16, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, FontFamily = (FontFamily)Application.Current.Resources["UIDisplayFont"] });
        _subtitle = new TextBlock { Style = JobUI.Style("CaptionText") };
        titles.Children.Add(_subtitle);
        Grid.SetColumn(titles, 1);
        header.Children.Add(titles);
        root.Children.Add(header);

        _scroller = new ScrollViewer { HorizontalScrollMode = ScrollMode.Disabled, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, Padding = new Thickness(0, 0, 10, 0) };
        Grid.SetRow(_scroller, 1);
        root.Children.Add(_scroller);

        var bar = new Border
        {
            Style = JobUI.Style<Border>("GlassChip"), CornerRadius = new CornerRadius(20), Padding = new Thickness(5), Margin = new Thickness(0, 10, 0, 0),
            Child = _actionGrid,
        };
        Grid.SetRow(bar, 2);
        root.Children.Add(bar);
        Content = root;
    }

    public string Subtitle { set => _subtitle.Text = value; }

    /// <summary>The page body (scrolls).</summary>
    public UIElement? Body
    {
        get => _scroller.Content as UIElement;
        set => _scroller.Content = value;
    }

    public ScrollViewer Scroller => _scroller;

    /// <summary>
    /// The action bar: <paramref name="leading"/> fills the free width (a progress capsule), then the buttons; the
    /// last button stretches when there is no leading element (the primary action fills the bar, as on the Mac).
    /// </summary>
    public void SetActions(UIElement? leading, params Button[] buttons)
    {
        _actionGrid.Children.Clear();
        _actionGrid.ColumnDefinitions.Clear();
        var col = 0;
        if (leading is not null)
        {
            _actionGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            _actionGrid.Children.Add(leading);
            col++;
        }
        for (var i = 0; i < buttons.Length; i++)
        {
            var stretch = leading is null && i == buttons.Length - 1;
            _actionGrid.ColumnDefinitions.Add(new ColumnDefinition { Width = stretch ? new GridLength(1, GridUnitType.Star) : GridLength.Auto });
            if (stretch) buttons[i].HorizontalAlignment = HorizontalAlignment.Stretch;
            Grid.SetColumn(buttons[i], col++);
            _actionGrid.Children.Add(buttons[i]);
        }
    }
}

/// <summary>Builders for the job pages' controls (Fluent translations of the Mac app's job views).</summary>
public static class JobUI
{
    public static Style Style(string key) => (Style)Application.Current.Resources[key];
    public static Style Style<T>(string key) => (Style)Application.Current.Resources[key];

    public static TextBlock Text(string text, string style = "BodyText", double? size = null, bool wrap = false)
    {
        var t = new TextBlock { Text = text, Style = Style(style) };
        if (size is { } s) t.FontSize = s;
        if (wrap) { t.TextWrapping = TextWrapping.Wrap; t.TextTrimming = TextTrimming.None; }
        return t;
    }

    /// <summary>Secondary explanatory text, wrapped (the Mac's <c>.hint()</c>).</summary>
    public static TextBlock Hint(string text, double size = 12) => Text(text, "SecondaryText", size, wrap: true);

    /// <summary>A group with a title above a rounded card (the Mac's GroupBox).</summary>
    public static StackPanel Group(string title, UIElement content, UIElement? trailing = null)
    {
        var p = new StackPanel { Spacing = 6 };
        var head = new Grid();
        head.Children.Add(Text(title, "PanelTitleText"));
        if (trailing is FrameworkElement fe)
        {
            fe.HorizontalAlignment = HorizontalAlignment.Right;
            head.Children.Add(fe);
        }
        p.Children.Add(head);
        p.Children.Add(Card(content));
        return p;
    }

    public static Border Card(UIElement content, double padding = 10)
    {
        var b = new Border { Child = content, Padding = new Thickness(padding), CornerRadius = new CornerRadius(8), BorderThickness = new Thickness(1) };
        Bind(b, "CardBackgroundFillColorDefaultBrush", x => b.Background = x);
        Bind(b, "CardStrokeColorDefaultBrush", x => b.BorderBrush = x);
        return b;
    }

    /// <summary>Sets a theme brush for the element's own theme now and whenever it changes (the window's theme, not the app's).</summary>
    public static void Bind(FrameworkElement e, string key, Action<Brush> set)
    {
        set((Brush)ThemeLookup.Resource(e.ActualTheme, key));
        e.Loaded += (_, _) => set(ThemeLookup.Brush(e, key));
        e.ActualThemeChanged += (_, _) => set(ThemeLookup.Brush(e, key));
    }

    /// <summary>A label and its control on one row, the control right-aligned (the Mac's LabeledContent).</summary>
    public static Grid Labeled(string label, UIElement control, string? help = null)
    {
        var g = new Grid { ColumnSpacing = 8, MinHeight = 30 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var t = Text(label, "BodyText");
        g.Children.Add(t);
        Grid.SetColumn((FrameworkElement)control, 1);
        g.Children.Add(control);
        if (help is not null) ToolTipService.SetToolTip(t, help);
        return g;
    }

    /// <summary>A large number with its label (iteration, merit before and after).</summary>
    public static StackPanel BigNumber(string label, string value, string? help = null)
    {
        var p = new StackPanel { Spacing = 1 };
        p.Children.Add(Text(label, "CaptionText"));
        p.Children.Add(new TextBlock { Text = value, Style = Style("NumberText"), FontSize = 20, FontWeight = Microsoft.UI.Text.FontWeights.Medium, FontFamily = (FontFamily)Application.Current.Resources["UIDisplayFont"] });
        if (help is not null) ToolTipService.SetToolTip(p, help);
        AutomationProperties.SetName(p, $"{label} {value}");
        return p;
    }

    public static StackPanel Row(double spacing, params UIElement[] children)
    {
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = spacing };
        foreach (var c in children) p.Children.Add(c);
        return p;
    }

    public static StackPanel Column(double spacing, params UIElement[] children)
    {
        var p = new StackPanel { Spacing = spacing };
        foreach (var c in children) p.Children.Add(c);
        return p;
    }

    /// <summary>The flow's one primary action: the accent capsule.</summary>
    public static Button Primary(string text, string? glyph, Action click, string? help = null, bool enabled = true)
    {
        var b = new Button { Style = Style("AccentCapsuleButton"), Content = Label(text, glyph), IsEnabled = enabled, MinHeight = 32, CornerRadius = new CornerRadius(16) };
        b.Click += (_, _) => click();
        b.KeyboardAccelerators.Add(new KeyboardAccelerator { Key = VirtualKey.Enter, Modifiers = VirtualKeyModifiers.Control });
        if (help is not null) ToolTipService.SetToolTip(b, help);
        AutomationProperties.SetName(b, text);
        return b;
    }

    public static Button Secondary(string text, string? glyph, Action click, string? help = null, bool enabled = true, bool destructive = false)
    {
        var b = new Button { Style = Style("CapsuleButton"), Content = Label(text, glyph), IsEnabled = enabled, MinHeight = 32, CornerRadius = new CornerRadius(16) };
        if (destructive) Bind(b, "ErrorBrush", x => b.Foreground = x);
        b.Click += (_, _) => click();
        if (help is not null) ToolTipService.SetToolTip(b, help);
        AutomationProperties.SetName(b, text);
        return b;
    }

    static object Label(string text, string? glyph)
    {
        if (glyph is null) return text;
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 7, HorizontalAlignment = HorizontalAlignment.Center };
        p.Children.Add(new FontIcon { Glyph = glyph, FontSize = 12 });
        p.Children.Add(new TextBlock { Text = text });
        return p;
    }

    /// <summary>A small icon button (pin, remove, add, move).</summary>
    public static Button Icon(string glyph, string help, Action click, bool enabled = true, double size = 26)
    {
        var b = new Button { Style = Style("CircleButton"), Content = glyph, Width = size, Height = size, FontSize = 12, IsEnabled = enabled };
        b.Click += (_, _) => click();
        ToolTipService.SetToolTip(b, help);
        AutomationProperties.SetName(b, help);
        return b;
    }

    /// <summary>The live progress capsule between the actions while a job runs.</summary>
    public static Border Progress(JobProgress p, string detail = "")
    {
        var ring = new ProgressRing { Width = 16, Height = 16, IsIndeterminate = p.Fraction <= 0, Value = p.Fraction * 100, Minimum = 0, Maximum = 100 };
        var texts = new StackPanel { VerticalAlignment = VerticalAlignment.Center };
        texts.Children.Add(Text(detail.Length > 0 ? detail : p.Stage.Length > 0 ? p.Stage : "Running", "NumberText"));
        if (p.Fraction >= 0.01) texts.Children.Add(Text(MeritFormat.Percent(p.Fraction, 0), "CaptionText"));
        var row = Row(8, ring, texts);
        row.VerticalAlignment = VerticalAlignment.Center;
        var capsule = new Border { Child = row, Padding = new Thickness(12, 2, 12, 2), CornerRadius = new CornerRadius(16), MinHeight = 32 };
        Bind(capsule, "SubtleFillColorSecondaryBrush", b => capsule.Background = b);
        AutomationProperties.SetName(capsule, ((TextBlock)texts.Children[0]).Text);
        return capsule;
    }

    /// <summary>A job's failure, the same page for every job.</summary>
    public static StackPanel Failure(string title, string reason)
    {
        var p = new StackPanel { Spacing = 8, Margin = new Thickness(0, 40, 0, 0), HorizontalAlignment = HorizontalAlignment.Center, MaxWidth = 280 };
        var icon = new FontIcon { Glyph = "", FontSize = 28, HorizontalAlignment = HorizontalAlignment.Center };
        Bind(icon, "WarningBrush", b => icon.Foreground = b);
        p.Children.Add(icon);
        var t = Text(title, "PanelTitleText", 15);
        t.HorizontalAlignment = HorizontalAlignment.Center;
        p.Children.Add(t);
        var r = Hint(reason);
        r.TextAlignment = TextAlignment.Center;
        p.Children.Add(r);
        return p;
    }

    /// <summary>An empty state: glyph, title and a description.</summary>
    public static StackPanel Empty(string glyph, string title, string description)
    {
        var p = new StackPanel { Spacing = 8, Margin = new Thickness(0, 40, 0, 0), HorizontalAlignment = HorizontalAlignment.Center, MaxWidth = 280 };
        var icon = new FontIcon { Glyph = glyph, FontSize = 28, HorizontalAlignment = HorizontalAlignment.Center };
        Bind(icon, "TextFillColorTertiaryBrush", b => icon.Foreground = b);
        p.Children.Add(icon);
        var t = Text(title, "PanelTitleText", 15);
        t.HorizontalAlignment = HorizontalAlignment.Center;
        p.Children.Add(t);
        var r = Hint(description);
        r.TextAlignment = TextAlignment.Center;
        p.Children.Add(r);
        return p;
    }

    /// <summary>Shown at the top of a result once it is applied to the design.</summary>
    public static StackPanel AppliedNotice()
    {
        var icon = new FontIcon { Glyph = "", FontSize = 14 };
        icon.Foreground = new SolidColorBrush(Palette.Slot(5, false));
        icon.Loaded += (_, _) => icon.Foreground = new SolidColorBrush(Palette.Slot(5, icon.ActualTheme == ElementTheme.Dark));
        return Row(6, icon, Text("Applied to the design. Undo with Ctrl+Z.", "SecondaryText"));
    }

    /// <summary>A warning line (orange glyph and text).</summary>
    public static StackPanel Warning(string text)
    {
        var icon = new FontIcon { Glyph = "", FontSize = 12, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) };
        var t = Hint(text);
        Bind(icon, "WarningBrush", b => { icon.Foreground = b; t.Foreground = b; });
        var p = Row(6, icon, t);
        t.MaxWidth = 250;
        return p;
    }

    /// <summary>A segmented choice of a few options (the Mac's segmented picker).</summary>
    public static Border Segments(IReadOnlyList<string> titles, int selected, Action<int> changed) => Segmented.Make(titles, selected, changed);

    /// <summary>A ComboBox over titles.</summary>
    public static ComboBox Choice(IReadOnlyList<string> titles, int selected, Action<int> changed, double minWidth = 140)
    {
        var c = new ComboBox { MinWidth = minWidth, FontSize = 13 };
        foreach (var t in titles) c.Items.Add(t);
        c.SelectedIndex = selected;
        c.SelectionChanged += (_, _) => { if (c.SelectedIndex >= 0 && c.SelectedIndex != selected) changed(c.SelectedIndex); };
        return c;
    }

    public static CheckBox Check(string text, bool on, Action<bool> changed, string? help = null)
    {
        var c = new CheckBox { Content = text, IsChecked = on, MinWidth = 0, FontSize = 13 };
        c.Click += (_, _) => changed(c.IsChecked == true);
        if (help is not null) ToolTipService.SetToolTip(c, help);
        return c;
    }

    /// <summary>A − value + stepper for small integers.</summary>
    public static StackPanel Stepper(string label, int value, int min, int max, Action<int> changed)
    {
        var minus = Icon("", $"Fewer {label.ToLowerInvariant()}", () => changed(Math.Max(min, value - 1)), value > min, 24);
        var plus = Icon("", $"More {label.ToLowerInvariant()}", () => changed(Math.Min(max, value + 1)), value < max, 24);
        var t = Text($"{label} {value.ToString(CultureInfo.InvariantCulture)}", "NumberText");
        t.MinWidth = 62;
        return Row(2, t, minus, plus);
    }
}

/// <summary>A small segmented control: toggle buttons in a capsule, one checked.</summary>
public static class Segmented
{
    public static Border Make(IReadOnlyList<string> titles, int selected, Action<int> changed)
    {
        var border = new Border();
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 2 };
        for (var i = 0; i < titles.Count; i++)
        {
            var k = i;
            var b = new Microsoft.UI.Xaml.Controls.Primitives.ToggleButton
            {
                Content = titles[i], IsChecked = i == selected, CornerRadius = new CornerRadius(12), MinHeight = 24, Height = 26,
                Padding = new Thickness(11, 0, 11, 1), BorderThickness = new Thickness(0), FontSize = 12.5,
            };
            if (i != selected) b.Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent);
            b.Click += (_, _) => { if (k != selected) changed(k); else b.IsChecked = true; };
            p.Children.Add(b);
        }
        border.Child = p;
        border.Padding = new Thickness(2);
        border.CornerRadius = new CornerRadius(14);
        JobUI.Bind(border, "SubtleFillColorSecondaryBrush", b => border.Background = b);
        border.HorizontalAlignment = HorizontalAlignment.Left;
        return border;
    }
}

/// <summary>
/// A number field with the app's one text-field rule: compact text at rest, the exact value with the focus; only
/// edited text commits (Enter, or when the focus leaves); text that is not a value reverts; Esc reverts. Empty text
/// is "none" (null) when the field is optional. <see cref="Key"/> lets a rebuilt page give the focus back.
/// </summary>
public sealed partial class NumberField : TextBox
{
    readonly Func<double?> _value;
    readonly Func<double, string> _format;
    readonly Action<double?> _commit;
    readonly bool _optional;
    readonly bool _infinite;
    string _shown = "";

    public NumberField(string key, Func<double?> value, Action<double?> commit, double width = 76, string placeholder = "",
        Func<double, string>? format = null, bool optional = true, bool allowsInfinite = false, string? name = null)
    {
        Key = key;
        _value = value;
        _commit = commit;
        _format = format ?? MeritFormat.Short;
        _optional = optional;
        _infinite = allowsInfinite;
        Width = width;
        MinWidth = 0;
        MinHeight = 28;
        FontSize = 12.5;
        Padding = new Thickness(7, 4, 7, 4);
        PlaceholderText = placeholder;
        TextAlignment = TextAlignment.Right;
        VerticalAlignment = VerticalAlignment.Center;
        Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(this, FontNumeralAlignment.Tabular);
        if (name is not null) AutomationProperties.SetName(this, name);
        Show(false);
        GotFocus += (_, _) => { Show(true); SelectAll(); };
        LostFocus += (_, _) => { Commit(); Show(false); };
        KeyDown += OnKey;
    }

    public string Key { get; }

    void OnKey(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter) { Commit(); Show(true); SelectAll(); e.Handled = true; }
        else if (e.Key == VirtualKey.Escape && Text != _shown) { Show(true); e.Handled = true; }
    }

    void Show(bool exact)
    {
        var v = _value();
        _shown = v is { } d ? (exact ? BoundsFormat.Exact(d) : _format(d)) : "";
        Text = _shown;
    }

    void Commit()
    {
        if (Text == _shown) return;
        var t = Text.Trim();
        if (t.Length == 0)
        {
            if (_optional) _commit(null);
            return;
        }
        var v = LensValueParser.Number(t);
        if (v is { } d && (double.IsFinite(d) || _infinite)) _commit(d);
    }
}

/// <summary>
/// The merit sparkline: a log-scale line of the best merit per iteration (the Mac's <c>Sparkline</c>), in the accent
/// colour, with its first and last values at the right edge.
/// </summary>
public sealed partial class Sparkline : Grid
{
    readonly Canvas _canvas = new();
    readonly TextBlock _top = JobUI.Text("", "CaptionText");
    readonly TextBlock _bottom = JobUI.Text("", "CaptionText");
    IReadOnlyList<double> _values = [];

    public Sparkline(double height)
    {
        Height = height;
        ColumnSpacing = 6;
        ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        Children.Add(_canvas);
        var labels = new Grid();
        _top.VerticalAlignment = VerticalAlignment.Top;
        _bottom.VerticalAlignment = VerticalAlignment.Bottom;
        labels.Children.Add(_top);
        labels.Children.Add(_bottom);
        SetColumn(labels, 1);
        Children.Add(labels);
        _canvas.SizeChanged += (_, _) => Draw();
        ActualThemeChanged += (_, _) => Draw();
        AutomationProperties.SetName(this, "Merit by iteration");
    }

    public IReadOnlyList<double> Values
    {
        get => _values;
        set { _values = value; Draw(); }
    }

    void Draw()
    {
        _canvas.Children.Clear();
        var v = _values.Where(x => double.IsFinite(x) && x > 0).ToList();
        _top.Text = v.Count > 0 ? MeritFormat.Merit(v.Max()) : "";
        _bottom.Text = v.Count > 1 ? MeritFormat.Merit(v.Min()) : "";
        double w = _canvas.ActualWidth, h = _canvas.ActualHeight;
        if (w <= 0 || h <= 0) return;
        var grid = ThemeLookup.Brush(this, "DividerBrush");
        foreach (var y in new[] { 0.5, h - 0.5 })
            _canvas.Children.Add(new Line { X1 = 0, X2 = w, Y1 = y, Y2 = y, Stroke = grid, StrokeThickness = 1, StrokeDashArray = [2, 3] });
        if (v.Count == 0) return;
        double lo = Math.Log10(v.Min()), hi = Math.Log10(v.Max());
        if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
        var line = new Polyline { Stroke = ThemeLookup.Brush(this, "AccentFillColorDefaultBrush"), StrokeThickness = 2, StrokeLineJoin = PenLineJoin.Round };
        for (var i = 0; i < v.Count; i++)
        {
            var x = v.Count == 1 ? w / 2 : i * (w - 2) / (v.Count - 1) + 1;
            var y = 2 + (hi - Math.Log10(v[i])) / (hi - lo) * (h - 4);
            line.Points.Add(new global::Windows.Foundation.Point(x, y));
        }
        _canvas.Children.Add(line);
    }
}
