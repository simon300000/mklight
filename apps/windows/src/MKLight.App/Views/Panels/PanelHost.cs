using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Layout;

namespace MKLight.App.Views.Panels;

/// <summary>
/// The design window's content area: the lens data, the layout and the
/// analysis panes. Round 1 is a fixed split (lens data across the top, the
/// layout and the panes side by side below). The panel arrangement (tabs,
/// splits, drag and drop, separate windows, saved in the project's
/// <c>views</c>) replaces this class's insides; its public surface is what
/// the window uses.
/// </summary>
public sealed class PanelHost : UserControl
{
    readonly DesignSession _session;
    readonly Grid _root = new();
    readonly Grid _panelsRow = new();
    readonly List<AnalysisPaneView> _paneViews = [];

    public PanelHost(DesignSession session)
    {
        _session = session;
        LensData = new LensDataPanel { Host = session };
        Layout = new LayoutPanel(session);
        _root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(0.36, GridUnitType.Star), MinHeight = 150 });
        _root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1) });
        _root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(0.64, GridUnitType.Star) });
        _root.Children.Add(LensData);
        var divider = new Border { Background = ThemeLookup.Brush(this, "DividerBrush") };
        ActualThemeChanged += (_, _) => divider.Background = ThemeLookup.Brush(this, "DividerBrush");
        Grid.SetRow(divider, 1);
        _root.Children.Add(divider);
        Grid.SetRow(_panelsRow, 2);
        _root.Children.Add(_panelsRow);
        Content = _root;
        session.Workspace.PanesChanged += Rebuild;
        Rebuild();
    }

    public LensDataPanel LensData { get; }
    public LayoutPanel Layout { get; }
    public IReadOnlyList<AnalysisPaneView> PaneViews => _paneViews;

    /// <summary>Raised when the user focuses a pane (the sidebar follows).</summary>
    public event Action? FocusChanged;

    /// <summary>Show and focus a panel kind: <c>lens_data</c> or <c>layout</c> (analysis panes are shown through the workspace).</summary>
    public void ShowPanel(string kind)
    {
        if (kind == "lens_data") LensData.Focus(FocusState.Programmatic);
        else if (kind == "layout") Layout.Fit();
    }

    void Rebuild()
    {
        var ws = _session.Workspace;
        _panelsRow.Children.Clear();
        _panelsRow.ColumnDefinitions.Clear();
        _paneViews.RemoveAll(v => !ws.Panes.Contains(v.Pane));

        // Layout on the left (a little wider), then the panes side by side, hairline dividers between.
        _panelsRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(ws.Panes.Count == 0 ? 1 : 1.25, GridUnitType.Star), MinWidth = 220 });
        _panelsRow.Children.Add(Layout);
        var col = 1;
        foreach (var pane in ws.Panes)
        {
            var view = _paneViews.FirstOrDefault(v => v.Pane == pane);
            if (view is null)
            {
                view = new AnalysisPaneView(_session, pane);
                view.CloseRequested += v => ws.RemovePane(v.Pane);
                view.Focused += v => { ws.FocusedPaneId = v.Pane.Id; UpdateFocus(); FocusChanged?.Invoke(); };
                _paneViews.Add(view);
            }
            _panelsRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1) });
            var divider = new Border { Background = ThemeLookup.Brush(this, "DividerBrush") };
            Grid.SetColumn(divider, col++);
            _panelsRow.Children.Add(divider);
            _panelsRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star), MinWidth = 200 });
            Grid.SetColumn(view, col++);
            _panelsRow.Children.Add(view);
        }
        UpdateFocus();
    }

    public void UpdateFocus()
    {
        foreach (var v in _paneViews) v.SetFocused(_session.Workspace.Panes.Count > 1 && v.Pane.Id == _session.Workspace.FocusedPane?.Id);
    }
}
