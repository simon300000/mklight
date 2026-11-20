using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;

namespace MKLight.App.Views.Glass;

/// <summary>
/// The glass table of the Glass Catalogs window: sortable columns (name, catalog, n_d, V_d,
/// ΔP_g,F, status, density, relative cost, TCE, λ80, range), the marks for user statuses and
/// exclusions, a count footer; double-click uses a glass, right-click opens its menu.
/// </summary>
public sealed partial class GlassTableView : UserControl
{
    static readonly (GlassColumn Column, string Title, bool Number, string Tip)[] Columns =
    [
        (GlassColumn.Name, "Glass", false, "Glass name"),
        (GlassColumn.Catalog, "Catalog", false, "The catalog the glass comes from"),
        (GlassColumn.Nd, "nd", true, "Refractive index at the d line (587.6 nm)"),
        (GlassColumn.Vd, "Vd", true, "Abbe number"),
        (GlassColumn.Dpgf, "ΔPg,F", true, "Anomalous partial dispersion"),
        (GlassColumn.Status, "Status", false, "Availability (your setting or the catalog's)"),
        (GlassColumn.Density, "Density", true, "g/cm³"),
        (GlassColumn.Cost, "Cost", true, "Relative cost"),
        (GlassColumn.Tce, "TCE", true, "Thermal expansion, 10⁻⁶/K (−30…+70 °C)"),
        (GlassColumn.Lambda80, "λ80 nm", true, "Wavelength of 80 % internal transmittance through 10 mm"),
        (GlassColumn.Range, "Range µm", false, "The wavelength range the dispersion formula is valid for"),
    ];

    GlassBrowserModel? _model;
    readonly List<(GlassColumn Column, FontIcon Arrow)> _arrows = [];
    bool _updating;

    public GlassTableView()
    {
        InitializeComponent();
        for (var i = 0; i < Columns.Length; i++)
        {
            var (column, title, number, tip) = Columns[i];
            var arrow = new FontIcon { Glyph = "\uE70E", FontSize = 9, Opacity = 0, Margin = new Thickness(4, 1, 0, 0), VerticalAlignment = VerticalAlignment.Center };
            var label = new StackPanel { Orientation = Orientation.Horizontal };
            label.Children.Add(new TextBlock { Text = title, Style = (Style)Application.Current.Resources["ColumnHeaderText"] });
            label.Children.Add(arrow);
            var b = new Button
            {
                Style = (Style)Resources["HeaderButton"], Content = label,
                HorizontalContentAlignment = number ? HorizontalAlignment.Right : HorizontalAlignment.Left,
            };
            if (column == GlassColumn.Range) b.Padding = new Thickness(6, 0, 10, 0);
            ToolTipService.SetToolTip(b, tip);
            b.Click += (_, _) => _model?.SortBy(column);
            Grid.SetColumn(b, i);
            Header.Children.Add(b);
            _arrows.Add((column, arrow));
        }
        ActualThemeChanged += (_, _) => Update();
    }

    public GlassBrowserModel? Model
    {
        get => _model;
        set
        {
            if (_model is { } old) { old.Changed -= Update; old.DetailsChanged -= SyncSelection; }
            _model = value;
            if (value is null) return;
            value.Changed += Update;
            value.DetailsChanged += SyncSelection;
            Update();
        }
    }

    /// <summary>Asks the window for a glass's context menu at a point of this view.</summary>
    public event Action<GlassRow, UIElement, global::Windows.Foundation.Point>? ContextMenuRequested;

    List<GlassRowItem> _items = [];

    void Update()
    {
        if (_model is not { } m) return;
        var dark = ActualTheme == ElementTheme.Dark;
        var ink = ThemeLookup.Brush(this, "TextFillColorPrimaryBrush");
        var dim = ThemeLookup.Brush(this, "TextFillColorTertiaryBrush");
        var rows = m.VisibleRows;
        _items = rows.Select(r => new GlassRowItem(r, dark, ink, dim)).ToList();
        _updating = true;
        List.ItemsSource = _items;
        List.SelectedItem = _items.FirstOrDefault(i => i.Id == m.Selection);
        _updating = false;
        if (List.SelectedItem is { } sel) List.ScrollIntoView(sel);
        foreach (var (column, arrow) in _arrows)
        {
            arrow.Opacity = column == m.SortColumn ? 1 : 0;
            arrow.Glyph = m.SortAscending ? "\uE70E" : "\uE70D";
        }
        CountText.Text = string.Create(CultureInfo.InvariantCulture, $"{rows.Count} of {m.Rows.Count} glasses");
        var empty = rows.Count == 0;
        Empty.Visibility = empty ? Visibility.Visible : Visibility.Collapsed;
        if (empty)
        {
            var searching = m.Search.Trim().Length > 0 || m.Filter != GlassFilter.All;
            EmptyIcon.Glyph = searching ? "\uE721" : "\uE7B8";
            EmptyTitle.Text = searching ? "No Results" : "No Glasses";
            EmptyText.Text = searching
                ? (m.Search.Trim().Length > 0 ? $"No glass matches “{m.Search.Trim()}”" : "No glass passes the filter") + (m.Filter != GlassFilter.All ? $" ({GlassBrowserModel.FilterTitle(m.Filter)})." : ".")
                : m.Scope is { } n && m.Library.IsUserCatalog(n) ? "Add a glass with New Glass." : "This catalog has no glasses.";
        }
    }

    void SyncSelection()
    {
        if (_model is null) return;
        var item = _items.FirstOrDefault(i => i.Id == _model.Selection);
        if (ReferenceEquals(List.SelectedItem, item)) return;
        _updating = true;
        List.SelectedItem = item;
        _updating = false;
    }

    void OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_updating || _model is null) return;
        _model.Selection = (List.SelectedItem as GlassRowItem)?.Id;
    }

    void OnDoubleTapped(object sender, DoubleTappedRoutedEventArgs e)
    {
        if ((e.OriginalSource as FrameworkElement)?.DataContext is GlassRowItem) _model?.UseSelected();
    }

    void OnRightTapped(object sender, RightTappedRoutedEventArgs e)
    {
        if ((e.OriginalSource as FrameworkElement)?.DataContext is not GlassRowItem item || _model is null) return;
        _model.Selection = item.Id;
        ContextMenuRequested?.Invoke(item.Row, List, e.GetPosition(List));
        e.Handled = true;
    }

    /// <summary>Puts the keyboard focus on the table (after a search, Down moves into it).</summary>
    public void FocusList() => List.Focus(FocusState.Programmatic);
}
