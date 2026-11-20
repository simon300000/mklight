using System.Text;
using Microsoft.UI;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using Windows.Foundation;

namespace MKLight.App.Views.Panels;

/// <summary>
/// A window's panels: its tree of splits (hairline dividers that drag;
/// double-click for equal shares) and tab groups. While a panel is dragged,
/// the place it would land is outlined in the accent colour. The tree is
/// rebuilt when its shape changes; a tab switch or a divider's drag updates
/// it in place.
/// </summary>
internal sealed class PanelTreeView : UserControl
{
    readonly PanelHost _host;
    readonly Grid _root = new();
    readonly Canvas _edgeOverlay = new() { IsHitTestVisible = false };
    readonly List<PanelGroupView> _groups = [];
    readonly List<PanelSplitView> _splits = [];
    string? _shape;

    public PanelTreeView(PanelHost host, PanelWindowModel model)
    {
        _host = host;
        Model = model;
        Content = _root;
        host.Drag.TargetChanged += ShowTarget;
    }

    /// <summary>The tree is discarded (its window closed): it stops listening to the host's drags.</summary>
    public void Detach() => _host.Drag.TargetChanged -= ShowTarget;

    public PanelHost Host => _host;
    public PanelWindowModel Model { get; }
    /// <summary>The window the tree is in (main or separate).</summary>
    public Microsoft.UI.Xaml.Window? HostWindow { get; set; }
    public IReadOnlyList<PanelGroupView> Groups => _groups;

    /// <summary>Whether the tree's shape (nodes, panels per group, theme) differs from what is on screen.</summary>
    public bool NeedsRebuild => Shape() != _shape;

    string Shape()
    {
        var sb = new StringBuilder(ActualTheme.ToString());
        void Walk(PanelNode? n)
        {
            switch (n)
            {
                case PanelGroup g: sb.Append("g").Append(g.Id).Append('[').AppendJoin(',', g.Panels).Append(']'); break;
                case PanelSplit s:
                    sb.Append("s").Append(s.Id).Append(s.Axis).Append('(');
                    foreach (var c in s.Children) Walk(c);
                    sb.Append(')');
                    break;
                default: sb.Append('-'); break;
            }
        }
        Walk(Model.Root);
        return sb.ToString();
    }

    public void Refresh()
    {
        var shape = Shape();
        if (shape == _shape)
        {
            foreach (var g in _groups) g.Refresh();
            foreach (var s in _splits) s.ApplyFractions();
            return;
        }
        _shape = shape;
        _root.Children.Clear();
        _groups.Clear();
        _splits.Clear();
        _root.Children.Add(Model.Root is { } root ? Build(root) : new EmptyPanelsView(_host, Model));
        _root.Children.Add(_edgeOverlay);
        ShowTarget();
    }

    UIElement Build(PanelNode node)
    {
        if (node is PanelGroup g)
        {
            var gv = new PanelGroupView(this, g);
            _groups.Add(gv);
            gv.Refresh();
            return gv;
        }
        var s = (PanelSplit)node;
        var sv = new PanelSplitView(this, s, s.Children.Select(Build).ToList());
        _splits.Add(sv);
        return sv;
    }

    public void UpdateTitles()
    {
        foreach (var g in _groups) g.RefreshTabs();
    }

    /// <summary>The content area in window coordinates (effective pixels).</summary>
    public Rect Bounds => BoundsOf(this);

    internal static Rect BoundsOf(FrameworkElement e) =>
        e.XamlRoot is null ? Rect.Empty : e.TransformToVisual(null).TransformBounds(new Rect(0, 0, e.ActualWidth, e.ActualHeight));

    // ---- Drop highlights ----

    void ShowTarget()
    {
        foreach (var g in _groups) g.ShowTarget(_host.Drag.Target);
        _edgeOverlay.Children.Clear();
        if (_host.Drag.Target is not PanelDropTarget.WindowEdge { } e || e.Window != Model.Id) return;
        double w = ActualWidth, h = ActualHeight;
        var r = e.Edge switch
        {
            PanelEdge.Leading => new Rect(0, 0, w * 0.22, h),
            PanelEdge.Trailing => new Rect(w * 0.78, 0, w * 0.22, h),
            PanelEdge.Top => new Rect(0, 0, w, h * 0.22),
            _ => new Rect(0, h * 0.78, w, h * 0.22),
        };
        _edgeOverlay.Children.Add(DropZone(this, r, e.Edge.Axis() == SplitAxis.Horizontal ? "Full-Height Column" : "Full-Width Row", EdgeGlyph(e.Edge)));
    }

    internal static string EdgeGlyph(PanelEdge e) => e switch
    {
        PanelEdge.Leading => "\uE90C",
        PanelEdge.Trailing => "\uE90D",
        _ => "\uE90E",
    };

    /// <summary>Where a dragged panel would land: an accent outline over a soft accent fill, its label on a glass chip.</summary>
    internal static FrameworkElement DropZone(FrameworkElement theme, Rect r, string label, string glyph)
    {
        var accent = (global::Windows.UI.Color)ThemeLookup.Resource(theme.ActualTheme, "SystemAccentColor");
        var inset = new Rect(r.X + 4, r.Y + 4, Math.Max(r.Width - 8, 0), Math.Max(r.Height - 8, 0));
        var zone = new Grid
        {
            Width = inset.Width, Height = inset.Height,
            CornerRadius = new CornerRadius(8), BorderThickness = new Thickness(2),
            BorderBrush = ThemeLookup.Brush(theme, "AccentFillColorDefaultBrush"),
            Background = new SolidColorBrush(accent) { Opacity = theme.ActualTheme == ElementTheme.Dark ? 0.18 : 0.12 },
        };
        if (inset.Width > 150 && inset.Height > 48) zone.Children.Add(Chip(theme, label, glyph));
        Canvas.SetLeft(zone, inset.X);
        Canvas.SetTop(zone, inset.Y);
        return zone;
    }

    /// <summary>A label on a glass capsule (drop zones, the drag's ghost).</summary>
    internal static Border Chip(FrameworkElement theme, string label, string glyph) => new()
    {
        HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center,
        Background = ThemeLookup.Brush(theme, "AcrylicInAppFillColorDefaultBrush"),
        BorderBrush = ThemeLookup.Brush(theme, "CapsuleStrokeBrush"), BorderThickness = new Thickness(1),
        CornerRadius = new CornerRadius(16), Padding = new Thickness(14, 6, 16, 7),
        Child = new StackPanel
        {
            Orientation = Orientation.Horizontal, Spacing = 8,
            Children =
            {
                new FontIcon { Glyph = glyph, FontSize = 14, Foreground = ThemeLookup.Brush(theme, "AccentTextFillColorPrimaryBrush") },
                new TextBlock { Text = label, FontSize = 13, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, VerticalAlignment = VerticalAlignment.Center },
            },
        },
    };
}

/// <summary>
/// Panels sharing a place: a compact tab strip when there are several, then
/// the panel in front.
/// </summary>
internal sealed class PanelGroupView : Grid
{
    const double StripHeight = 34;
    readonly PanelTreeView _tree;
    readonly Border _content = new();
    readonly Grid _strip = new() { Height = StripHeight };
    readonly StackPanel _tabs = new() { Orientation = Orientation.Horizontal, Spacing = 2, Padding = new Thickness(6, 4, 6, 4) };
    readonly Border _rule = new() { Height = 1 };
    readonly Canvas _overlay = new() { IsHitTestVisible = false };
    readonly List<PanelTab> _tabViews = [];
    PanelId? _shown;

    public PanelGroupView(PanelTreeView tree, PanelGroup group)
    {
        _tree = tree;
        Group = group;
        Background = ThemeLookup.Brush(tree, "ContentSurfaceBrush");
        RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        var scroller = new ScrollViewer
        {
            HorizontalScrollMode = ScrollMode.Auto, HorizontalScrollBarVisibility = ScrollBarVisibility.Hidden,
            VerticalScrollMode = ScrollMode.Disabled, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
            Content = _tabs,
        };
        _strip.Children.Add(scroller);
        Children.Add(_strip);
        _rule.Background = ThemeLookup.Brush(tree, "DividerBrush");
        SetRow(_rule, 1);
        Children.Add(_rule);
        SetRow(_content, 2);
        Children.Add(_content);
        SetRowSpan(_overlay, 3);
        Children.Add(_overlay);
        // A click into the panel focuses it (the panel commands act on it).
        _content.AddHandler(PointerPressedEvent, new PointerEventHandler((_, _) => { if (Group.Current is { } p) tree.Host.Focus(p); }), true);
        SizeChanged += (_, _) => ShowTarget(tree.Host.Drag.Target);
    }

    public PanelGroup Group { get; }
    public PanelTreeView Tree => _tree;
    public IReadOnlyList<PanelTab> Tabs => _tabViews;

    /// <summary>The tabs and the panel in front.</summary>
    public void Refresh()
    {
        RefreshTabs();
        var current = Group.Current;
        var view = current is { } c ? _tree.Host.ViewOf(c) : null;
        if (_shown is { } old && old != current) Release(old);
        if (view is not null && !ReferenceEquals(_content.Child, view)) Show(view);
        _shown = current;
        if (current is { } p) _tree.Host.Hosted(p, this);
        if (view is not null) view.Opacity = _tree.Host.Drag.Dragging == current ? 0.45 : 1;
        SetHeaderTitleVisible(view, Group.Panels.Count <= 1);
    }

    /// <summary>
    /// Puts a view in. A view that just left another place goes in once it has unloaded there: WinUI may
    /// otherwise raise its Unloaded after the new Loaded, and views (Win2D canvases, panes) that let go of
    /// their resources and subscriptions on Unloaded would stay blank.
    /// </summary>
    void Show(FrameworkElement view)
    {
        Detach(view);
        if (!_tree.Host.IsLive(view)) { _content.Child = view; return; }
        var timer = DispatcherQueue.CreateTimer();
        void Attach()
        {
            view.Unloaded -= OnUnloaded;
            timer.Stop();
            if (view.Parent is null && _shown is { } s && ReferenceEquals(_tree.Host.ViewOf(s), view)) _content.Child = view;
        }
        void OnUnloaded(object sender, RoutedEventArgs e) => Attach();
        view.Unloaded += OnUnloaded;
        // Unloaded may not come (the view never reached a live tree): at most a second.
        timer.Interval = TimeSpan.FromSeconds(1);
        timer.Tick += (_, _) => Attach();
        timer.Start();
    }

    /// <summary>Takes a panel's view out of this group (it moves elsewhere or is hidden).</summary>
    public void Release(PanelId panel)
    {
        if (_tree.Host.ViewOf(panel) is { } v && ReferenceEquals(_content.Child, v)) _content.Child = null;
        if (_shown == panel) _shown = null;
    }

    static void Detach(FrameworkElement view)
    {
        switch (view.Parent)
        {
            case Border b: b.Child = null; break;
            case Panel p: p.Children.Remove(view); break;
            case ContentControl cc: cc.Content = null; break;
        }
    }

    /// <summary>As one of several tabs, an analysis pane's header leaves its title to the tab (the analysis menu's chevron stays).</summary>
    static void SetHeaderTitleVisible(FrameworkElement? view, bool visible)
    {
        if (view is not Analysis.AnalysisPaneView pane) return;
        var v = visible ? Visibility.Visible : Visibility.Collapsed;
        if (pane.FindName("TitleText") is UIElement text) text.Visibility = v;
        if (pane.FindName("TitleGlyph") is UIElement glyph) glyph.Visibility = v;
    }

    public void RefreshTabs()
    {
        var tabbed = Group.Panels.Count > 1;
        _strip.Visibility = _rule.Visibility = tabbed ? Visibility.Visible : Visibility.Collapsed;
        if (!tabbed) { _tabs.Children.Clear(); _tabViews.Clear(); return; }
        if (!_tabViews.Select(t => t.Panel).SequenceEqual(Group.Panels))
        {
            _tabs.Children.Clear();
            _tabViews.Clear();
            foreach (var p in Group.Panels)
            {
                var tab = new PanelTab(_tree.Host, this, p);
                _tabViews.Add(tab);
                _tabs.Children.Add(tab);
            }
        }
        foreach (var t in _tabViews) t.Update(Group.Current == t.Panel);
    }

    // ---- Drop zones ----

    /// <summary>The header (the top 34 effective pixels) and the middle add a tab; the outer 30 % towards an edge splits the group on that side.</summary>
    public PanelDropTarget Zone(Point p, Rect r)
    {
        var fromTop = p.Y - r.Y;
        if (fromTop < StripHeight) return new PanelDropTarget.Tab(Group.Id, TabIndexAt(p));
        var nx = (p.X - r.X) / Math.Max(r.Width, 1);
        var ny = fromTop / Math.Max(r.Height, 1);
        (PanelEdge Edge, double D)[] edges = [(PanelEdge.Leading, nx), (PanelEdge.Trailing, 1 - nx), (PanelEdge.Top, ny), (PanelEdge.Bottom, 1 - ny)];
        var nearest = edges.MinBy(e => e.D);
        return nearest.D < 0.3 ? new PanelDropTarget.Split(Group.Id, nearest.Edge) : new PanelDropTarget.Tab(Group.Id, null);
    }

    /// <summary>Over a tab strip: the position between the tabs under the pointer.</summary>
    int? TabIndexAt(Point p)
    {
        if (Group.Panels.Count <= 1) return null;
        var index = 0;
        for (var k = 0; k < _tabViews.Count; k++)
        {
            var r = PanelTreeView.BoundsOf(_tabViews[k]);
            if (!r.IsEmpty && p.X > r.X + r.Width / 2) index = k + 1;
        }
        return index;
    }

    public void ShowTarget(PanelDropTarget? target)
    {
        _overlay.Children.Clear();
        double w = ActualWidth, h = ActualHeight;
        switch (target)
        {
            case PanelDropTarget.Tab t when t.Group == Group.Id:
                _overlay.Children.Add(PanelTreeView.DropZone(this, new Rect(0, 0, w, h), "Add as Tab", "\uE7C4"));
                break;
            case PanelDropTarget.Split s when s.Group == Group.Id:
                var r = s.Edge switch
                {
                    PanelEdge.Leading => new Rect(0, 0, w / 2, h),
                    PanelEdge.Trailing => new Rect(w / 2, 0, w / 2, h),
                    PanelEdge.Top => new Rect(0, 0, w, h / 2),
                    _ => new Rect(0, h / 2, w, h / 2),
                };
                _overlay.Children.Add(PanelTreeView.DropZone(this, r, "Split", PanelTreeView.EdgeGlyph(s.Edge)));
                break;
        }
        if (Group.Current is { } c && _tree.Host.ViewOf(c) is { } v) v.Opacity = _tree.Host.Drag.Dragging == c ? 0.45 : 1;
        foreach (var t in _tabViews) t.Opacity = _tree.Host.Drag.Dragging == t.Panel ? 0.4 : 1;
    }
}

/// <summary>
/// One tab of a group: the panel's glyph and title, the front one on a
/// quiet capsule with its glyph in the accent. Click to switch, drag to move
/// (to another place, another window or out into a new one); the close
/// button shows on hover and on the front tab.
/// </summary>
internal sealed class PanelTab : Grid
{
    readonly PanelHost _host;
    readonly FontIcon _glyph = new() { FontSize = 12 };
    readonly TextBlock _title = new() { FontSize = 13, VerticalAlignment = VerticalAlignment.Center, TextTrimming = TextTrimming.CharacterEllipsis, MaxWidth = 220 };
    readonly Button _close;
    bool _selected, _hover;

    public PanelTab(PanelHost host, PanelGroupView group, PanelId panel)
    {
        _host = host;
        Panel = panel;
        Height = 26;
        CornerRadius = new CornerRadius(13);
        Padding = new Thickness(10, 0, 3, 0);
        BorderThickness = new Thickness(1);
        _close = new Button
        {
            Width = 18, Height = 18, Padding = new Thickness(0), CornerRadius = new CornerRadius(9), MinWidth = 0, MinHeight = 0,
            Background = new SolidColorBrush(Colors.Transparent), BorderThickness = new Thickness(0),
            Content = new FontIcon { Glyph = "\uE711", FontSize = 8 },
            IsTabStop = false,
        };
        _close.Click += (_, _) => host.Close(panel);
        var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6, VerticalAlignment = VerticalAlignment.Center };
        row.Children.Add(_glyph);
        row.Children.Add(_title);
        row.Children.Add(_close);
        Children.Add(row);
        PointerEntered += (_, _) => { _hover = true; Paint(); };
        PointerExited += (_, _) => { _hover = false; Paint(); };
        PanelHandles.Attach(host, this, panel, click: () => { host.Arrangement.Select(panel); host.Focus(panel); }, exclude: _close);
        AutomationProperties.SetName(this, host.Title(panel));
    }

    public PanelId Panel { get; }

    public void Update(bool selected)
    {
        _selected = selected;
        var title = _host.Title(Panel);
        _title.Text = title;
        _glyph.Glyph = _host.Glyph(Panel);
        ToolTipService.SetToolTip(this, title);
        ToolTipService.SetToolTip(_close, Panel.PaneId is null ? $"Hide {title}" : $"Close {title}");
        AutomationProperties.SetName(_close, Panel.PaneId is null ? $"Hide {title}" : $"Close {title}");
        Paint();
    }

    void Paint()
    {
        _title.FontWeight = _selected ? Microsoft.UI.Text.FontWeights.SemiBold : Microsoft.UI.Text.FontWeights.Normal;
        _title.Foreground = ThemeLookup.Brush(this, _selected ? "TextFillColorPrimaryBrush" : "TextFillColorSecondaryBrush");
        _glyph.Foreground = ThemeLookup.Brush(this, _selected ? "AccentTextFillColorPrimaryBrush" : "TextFillColorSecondaryBrush");
        Background = _selected || _hover ? ThemeLookup.Brush(this, _selected ? "SubtleFillColorSecondaryBrush" : "SubtleFillColorTertiaryBrush") : new SolidColorBrush(Colors.Transparent);
        BorderBrush = _selected ? ThemeLookup.Brush(this, "CapsuleStrokeBrush") : new SolidColorBrush(Colors.Transparent);
        _close.Foreground = ThemeLookup.Brush(this, "TextFillColorSecondaryBrush");
        // Hidden, it takes no clicks: a click near a background tab's end selects the tab.
        _close.Opacity = _selected || _hover ? 1 : 0;
        _close.IsHitTestVisible = _selected || _hover;
    }
}

/// <summary>
/// Children side by side or stacked, hairline dividers between them that
/// drag (double-click: equal shares). Each child keeps at least 200 (side by
/// side) or 120 (stacked) effective pixels when the space allows.
/// </summary>
internal sealed class PanelSplitView : Grid
{
    readonly PanelTreeView _tree;
    readonly PanelSplit _split;
    readonly List<object> _defs = [];
    readonly List<Border> _rules = [];
    bool Horizontal => _split.Axis == SplitAxis.Horizontal;
    double MinSize => Horizontal ? 200 : 120;

    public PanelSplitView(PanelTreeView tree, PanelSplit split, IReadOnlyList<UIElement> children)
    {
        _tree = tree;
        _split = split;
        for (var k = 0; k < children.Count; k++)
        {
            if (k > 0)
            {
                AddDefinition(new GridLength(1));
                var rule = new Border { Background = ThemeLookup.Brush(tree, "DividerBrush") };
                Place(rule, 2 * k - 1);
                _rules.Add(rule);
                Children.Add(rule);
            }
            _defs.Add(AddDefinition(new GridLength(1, GridUnitType.Star)));
            Place(children[k], 2 * k);
            Children.Add(children[k]);
        }
        // The handles go last: above the panels on either side.
        for (var k = 0; k + 1 < children.Count; k++)
        {
            var handle = new DividerHandle(this, k);
            Place(handle, 2 * k + 1);
            Children.Add(handle);
        }
        ApplyFractions();
        SizeChanged += (_, _) => ApplyMinimums();
    }

    public PanelSplit Split => _split;

    object AddDefinition(GridLength length)
    {
        if (Horizontal) { var c = new ColumnDefinition { Width = length }; ColumnDefinitions.Add(c); return c; }
        var r = new RowDefinition { Height = length };
        RowDefinitions.Add(r);
        return r;
    }

    void Place(UIElement e, int index)
    {
        if (Horizontal) SetColumn((FrameworkElement)e, index); else SetRow((FrameworkElement)e, index);
    }

    public void ApplyFractions()
    {
        for (var k = 0; k < _defs.Count && k < _split.Fractions.Count; k++)
        {
            var len = new GridLength(Math.Max(_split.Fractions[k], 1e-4), GridUnitType.Star);
            if (_defs[k] is ColumnDefinition c) c.Width = len; else ((RowDefinition)_defs[k]).Height = len;
        }
        ApplyMinimums();
    }

    /// <summary>The space for the children (without the dividers).</summary>
    public double Available => Math.Max((Horizontal ? ActualWidth : ActualHeight) - (_defs.Count - 1), 1);

    void ApplyMinimums()
    {
        var min = Available > MinSize * _defs.Count ? MinSize : 0;
        foreach (var d in _defs)
            if (d is ColumnDefinition c) c.MinWidth = min; else ((RowDefinition)d).MinHeight = min;
    }

    /// <summary>A hit area 7 effective pixels wide over the hairline after child <c>k</c>.</summary>
    sealed class DividerHandle : Grid
    {
        readonly PanelSplitView _owner;
        readonly int _k;
        double[]? _start;
        double _origin;

        public DividerHandle(PanelSplitView owner, int k)
        {
            _owner = owner;
            _k = k;
            Background = new SolidColorBrush(Colors.Transparent);
            if (owner.Horizontal) { Width = 7; Margin = new Thickness(-3, 0, -3, 0); }
            else { Height = 7; Margin = new Thickness(0, -3, 0, -3); }
            ProtectedCursor = InputSystemCursor.Create(owner.Horizontal ? InputSystemCursorShape.SizeWestEast : InputSystemCursorShape.SizeNorthSouth);
            AutomationProperties.SetName(this, owner.Horizontal ? "Vertical divider" : "Horizontal divider");
            PointerPressed += OnPressed;
            PointerMoved += OnMoved;
            PointerReleased += OnReleased;
            PointerCaptureLost += (_, _) => _start = null;
            DoubleTapped += (_, e) =>
            {
                owner._split.Equalize();
                owner.ApplyFractions();
                owner._tree.Host.SaveSoon();
                e.Handled = true;
            };
        }

        double Position(PointerRoutedEventArgs e)
        {
            var p = e.GetCurrentPoint(_owner).Position;
            return _owner.Horizontal ? p.X : p.Y;
        }

        void OnPressed(object sender, PointerRoutedEventArgs e)
        {
            if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed) return;
            _start = [.. _owner._split.Fractions];
            _origin = Position(e);
            CapturePointer(e.Pointer);
            e.Handled = true;
        }

        void OnMoved(object sender, PointerRoutedEventArgs e)
        {
            if (_start is not { } start || _k + 1 >= start.Length) return;
            var avail = _owner.Available;
            var delta = (Position(e) - _origin) / avail;
            var pair = start[_k] + start[_k + 1];
            var minF = Math.Min(_owner.MinSize / avail, pair / 2);
            var a = Math.Clamp(start[_k] + delta, minF, pair - minF);
            _owner._split.Resize(_k, a, pair - a);
            _owner.ApplyFractions();
            e.Handled = true;
        }

        void OnReleased(object sender, PointerRoutedEventArgs e)
        {
            if (_start is null) return;
            _start = null;
            ReleasePointerCapture(e.Pointer);
            _owner._tree.Host.SaveSoon();
            e.Handled = true;
        }
    }
}

/// <summary>A window with every panel closed or moved away.</summary>
internal sealed class EmptyPanelsView : Grid
{
    public EmptyPanelsView(PanelHost host, PanelWindowModel model)
    {
        Background = ThemeLookup.Brush(host, "ContentSurfaceBrush");
        var stack = new StackPanel { HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center, Spacing = 8, MaxWidth = 420 };
        stack.Children.Add(new FontIcon { Glyph = "\uE737", FontSize = 28, Foreground = ThemeLookup.Brush(host, "TextFillColorTertiaryBrush") });
        stack.Children.Add(new TextBlock { Text = "No Panels", FontSize = 15, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Center });
        stack.Children.Add(new TextBlock
        {
            Text = "Show the lens data or the layout, or add an analysis. Panels moved to other windows can come back here.",
            TextWrapping = TextWrapping.Wrap, TextAlignment = TextAlignment.Center, FontSize = 13,
            Foreground = ThemeLookup.Brush(host, "TextFillColorSecondaryBrush"),
        });
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(0, 6, 0, 0) };
        Button Capsule(string text, Action click)
        {
            var b = new Button { Content = text, CornerRadius = new CornerRadius(14), MinHeight = 28, Padding = new Thickness(12, 3, 12, 4) };
            b.Click += (_, _) => click();
            return b;
        }
        if (!host.Arrangement.Contains(PanelId.LensData)) buttons.Children.Add(Capsule("Show Lens Data", () => host.ShowPanel("lens_data")));
        if (!host.Arrangement.Contains(PanelId.Layout)) buttons.Children.Add(Capsule("Show Layout", () => host.ShowPanel("layout")));
        var add = new DropDownButton { Content = "Add Analysis", CornerRadius = new CornerRadius(14), MinHeight = 28 };
        var menu = new MenuFlyout();
        PanelMenus.FillAnalyses(menu, host.Session.Workspace, a =>
        {
            if (model == host.Arrangement.Main) host.Session.Workspace.AddPane(a, after: host.Session.Workspace.FocusedPane);
            else host.AddPane(a, model.Id);
        });
        add.Flyout = menu;
        buttons.Children.Add(add);
        stack.Children.Add(buttons);
        if (model == host.Arrangement.Main && host.Arrangement.Detached.Count > 0)
        {
            var all = Capsule("Move All Panels Here", () => { foreach (var w in host.Arrangement.Detached.ToList()) host.Dock(w); });
            all.HorizontalAlignment = HorizontalAlignment.Center;
            stack.Children.Add(all);
        }
        Children.Add(stack);
    }
}
