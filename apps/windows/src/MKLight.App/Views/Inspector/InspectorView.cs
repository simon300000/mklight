using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Support;
using MKLight.App.ViewModels;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// The inspector column: a row of page tabs (glass, on Mica) over the selected page, as the Mac app's inspector
/// tab bar: Surface and System as words, the design jobs as glyphs with tooltips, a dot on a running job's tab.
/// Pages come from <see cref="InspectorPages"/>; each is built once per window and kept while hidden, so a
/// running job keeps its panel.
/// </summary>
public sealed partial class InspectorView : UserControl
{
    readonly DesignSession _session;
    readonly StackPanel _tabs = new() { Orientation = Orientation.Horizontal, Spacing = 2 };
    readonly Grid _host = new();
    readonly Dictionary<string, UIElement> _built = [];
    readonly Dictionary<string, (Button Button, Ellipse Dot)> _buttons = [];

    public InspectorView(DesignSession session)
    {
        _session = session;
        var root = new Grid { Padding = new Thickness(0, 4, 0, 0), RowSpacing = 4 };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        var chip = new Border { Style = (Style)Application.Current.Resources["GlassChip"], HorizontalAlignment = HorizontalAlignment.Center, Child = _tabs };
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(chip, "Inspector pages");
        root.Children.Add(chip);
        Grid.SetRow(_host, 1);
        root.Children.Add(_host);
        Content = root;
        int? group = null;
        foreach (var page in InspectorPages.All)
        {
            // A divider between the design's pages and the design jobs.
            var g = page.Order / 10;
            if (group is { } previous && previous != g)
                _tabs.Children.Add(new Border { Width = 1, Height = 16, Margin = new Thickness(3, 0, 3, 0), VerticalAlignment = VerticalAlignment.Center, Style = null, Background = null, Tag = "divider" });
            group = g;
            var glyph = page.Glyph.Length == 1;
            var b = new Button
            {
                Content = glyph ? new FontIcon { Glyph = page.Glyph, FontSize = 15 } : new TextBlock { Text = page.Title, FontSize = 13 },
                CornerRadius = new CornerRadius(14), Height = 28, MinWidth = 34, Padding = new Thickness(glyph ? 0 : 11, 0, glyph ? 0 : 11, 1),
                BorderThickness = new Thickness(0), Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent),
            };
            ToolTipService.SetToolTip(b, Help(page));
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(b, page.Title);
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(b, $"Inspector.Tab.{page.Id}");
            var id = page.Id;
            b.Click += (_, _) => Show(id);
            var dot = new Ellipse { Width = 6, Height = 6, HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 3, 4, 0), IsHitTestVisible = false, Visibility = Visibility.Collapsed };
            var cell = new Grid();
            cell.Children.Add(b);
            cell.Children.Add(dot);
            _buttons[id] = (b, dot);
            _tabs.Children.Add(cell);
        }
        session.InspectorPageRequested += Show;
        session.RunningJobsChanged += Paint;
        ActualThemeChanged += (_, _) => Paint();
        if (InspectorPages.All.FirstOrDefault() is { } first) Show(first.Id);
    }

    /// <summary>The Mac app's tab help.</summary>
    static string Help(InspectorPageInfo page) => page.Id switch
    {
        "surface" => "The selected surface",
        "system" => "Aperture, fields, wavelengths and environment",
        "optimize" => "Optimize the design",
        "glass_substitution" => "Replace glasses with catalog glasses, re-optimizing each combination",
        "tolerance" => "Tolerance analysis: sensitivities and Monte Carlo",
        "solves" => "Preview and apply the design's solves",
        _ => page.Title,
    };

    public string? Page { get; private set; }

    /// <summary>The built view of a page, if it was shown.</summary>
    public UIElement? PageView(string id) => _built.TryGetValue(id, out var v) ? v : null;

    public void Show(string id)
    {
        if (InspectorPages.All.FirstOrDefault(p => p.Id == id) is not { } page) return;
        Page = id;
        if (!_built.TryGetValue(id, out var view))
        {
            view = page.Create(_session);
            _built[id] = view;
            _host.Children.Add(view);
        }
        foreach (var (key, v) in _built) v.Visibility = key == id ? Visibility.Visible : Visibility.Collapsed;
        Paint();
    }

    /// <summary>The selected tab in the accent (a tinted capsule), a dot on running jobs' tabs.</summary>
    void Paint()
    {
        var accent = InspectorColors.Accent(this);
        var accentText = ThemeLookup.Brush(this, "AccentTextFillColorPrimaryBrush");
        var secondary = ThemeLookup.Brush(this, "TextFillColorSecondaryBrush");
        foreach (var (id, (button, dot)) in _buttons)
        {
            var selected = id == Page;
            button.Background = new SolidColorBrush(selected ? accent.WithAlpha(0.16) : Microsoft.UI.Colors.Transparent);
            button.Foreground = selected ? accentText : secondary;            Microsoft.UI.Xaml.Automation.AutomationProperties.SetItemStatus(button, selected ? "Selected" : _session.RunningJobs.ContainsKey(id) ? "Running" : "");
            if (button.Content is TextBlock t) t.FontWeight = selected ? Microsoft.UI.Text.FontWeights.SemiBold : Microsoft.UI.Text.FontWeights.Normal;
            dot.Fill = accentText;
            dot.Visibility = _session.RunningJobs.ContainsKey(id) ? Visibility.Visible : Visibility.Collapsed;
        }
        foreach (var d in _tabs.Children.OfType<Border>().Where(b => (string?)b.Tag == "divider"))
            d.Background = ThemeLookup.Brush(this, "DividerStrokeColorDefaultBrush");
    }
}
