using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using MKLight.App.ViewModels;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// The inspector column: a row of page tabs (glass, on Mica) over the
/// selected page. Pages come from <see cref="InspectorPages"/>; each is built
/// once per window and kept while hidden, so a running job keeps its panel.
/// </summary>
public sealed class InspectorView : UserControl
{
    readonly DesignSession _session;
    readonly StackPanel _tabs = new() { Orientation = Orientation.Horizontal };
    readonly Grid _host = new();
    readonly Dictionary<string, UIElement> _built = [];
    readonly Dictionary<string, ToggleButton> _buttons = [];

    public InspectorView(DesignSession session)
    {
        _session = session;
        var root = new Grid { Padding = new Thickness(12, 4, 12, 12), RowSpacing = 12 };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.Children.Add(new Border { Style = (Style)Application.Current.Resources["GlassChip"], HorizontalAlignment = HorizontalAlignment.Center, Child = _tabs });
        Grid.SetRow(_host, 1);
        root.Children.Add(_host);
        Content = root;
        foreach (var page in InspectorPages.All)
        {
            var b = new ToggleButton
            {
                Content = page.Glyph.Length == 1 ? new FontIcon { Glyph = page.Glyph, FontSize = 14 } : page.Title,
                CornerRadius = new CornerRadius(13), Height = 26, MinWidth = 34, Padding = new Thickness(10, 0, 10, 0), BorderThickness = new Thickness(0),
            };
            ToolTipService.SetToolTip(b, page.Title);
            var id = page.Id;
            b.Click += (_, _) => Show(id);
            _buttons[id] = b;
            _tabs.Children.Add(b);
        }
        session.InspectorPageRequested += Show;
        if (InspectorPages.All.FirstOrDefault() is { } first) Show(first.Id);
    }

    public string? Page { get; private set; }

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
        foreach (var (key, b) in _buttons) b.IsChecked = key == id;
    }
}
