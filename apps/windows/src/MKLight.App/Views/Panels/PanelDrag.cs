using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using Windows.Foundation;
using Windows.System;
using PointInt32 = Windows.Graphics.PointInt32;
using RectInt32 = Windows.Graphics.RectInt32;

namespace MKLight.App.Views.Panels;

/// <summary>
/// Moving panels by dragging their title or tab, within a window, between a
/// design's windows, and out into a window of its own (like a browser tab).
/// The drag captures the pointer in the window it started in, so its moves
/// arrive while the pointer is anywhere on screen; each position is judged
/// against the design's windows, front to back: over a group, a tab (its
/// header or middle) or a split (its outer 30 %); along a window's edge, a
/// whole row or column; outside the design's windows, a new window,
/// previewed by an outline where it will open. Escape cancels.
/// </summary>
internal sealed partial class PanelDragController(PanelHost host)
{
    UIElement? _capture;
    Pointer? _pointer;
    Popup? _ghost;
    PanelTreeView? _source;
    Size _sourceSize = new(560, 420);
    PointerEventHandler? _moved, _released;
    PointerEventHandler? _lost;
    KeyEventHandler? _key;

    /// <summary>The panel being dragged.</summary>
    public PanelId? Dragging { get; private set; }
    /// <summary>Where it would land now.</summary>
    public PanelDropTarget? Target { get; private set; }
    /// <summary>A new window's frame (screen pixels) while the target is a new window.</summary>
    public RectInt32? NewWindowFrame { get; private set; }

    public event Action? TargetChanged;

    /// <summary>The handle moved past the threshold with the button down: the drag starts.</summary>
    public void Begin(PanelId panel, PointerRoutedEventArgs e, UIElement handle)
    {
        if (Dragging is not null || handle.XamlRoot?.Content is not UIElement root || host.TreeOf(handle.XamlRoot) is not { } tree) return;
        _source = tree;
        Dragging = panel;
        if (host.GroupViewOf(panel) is { } gv) _sourceSize = new Size(gv.ActualWidth, gv.ActualHeight);
        _capture = root;
        _pointer = e.Pointer;
        _moved = (_, a) => Update(Screen(tree, a.GetCurrentPoint(null).Position));
        _released = (_, a) => { Update(Screen(tree, a.GetCurrentPoint(null).Position)); Finish(drop: true); };
        _lost = (_, a) => { if (ReferenceEquals(a.OriginalSource, root)) Finish(drop: false); };
        _key = (_, a) => { if (a.Key == VirtualKey.Escape) { a.Handled = true; Finish(drop: false); } };
        root.AddHandler(UIElement.PointerMovedEvent, _moved, true);
        root.AddHandler(UIElement.PointerReleasedEvent, _released, true);
        root.AddHandler(UIElement.PointerCaptureLostEvent, _lost, true);
        root.AddHandler(UIElement.KeyDownEvent, _key, true);
        // Taking the capture from the handle also cancels the handle's own click (a title's menu).
        root.CapturePointer(e.Pointer);
        ShowGhost(tree, panel);
        Update(Screen(tree, e.GetCurrentPoint(null).Position));
    }

    /// <summary>Abandons a drag (the window closing).</summary>
    public void Cancel() => Finish(drop: false);

    /// <summary>Launch actions: the drag state at a point on screen, as if the pointer were there.</summary>
    public void Simulate(PanelId panel, PointInt32 screen)
    {
        Dragging = panel;
        if (host.GroupViewOf(panel) is { } gv) _sourceSize = new Size(gv.ActualWidth, gv.ActualHeight);
        Update(screen);
    }

    /// <summary>Launch actions: drop where the simulated drag points.</summary>
    public void DropSimulated() => Finish(drop: true);

    void Update(PointInt32 p)
    {
        if (Dragging is not { } panel) return;
        var t = HitTest(p, panel);
        NewWindowFrame = t is PanelDropTarget.NewWindow ? FrameAt(p) : null;
        if (t != Target)
        {
            Target = t;
            TargetChanged?.Invoke();
        }
        MoveGhost(p, t);
    }

    void Finish(bool drop)
    {
        if (Dragging is not { } panel) return;
        var target = drop ? Target : null;
        var frame = NewWindowFrame;
        Dragging = null;
        Target = null;
        NewWindowFrame = null;
        if (_capture is { } root)
        {
            root.RemoveHandler(UIElement.PointerMovedEvent, _moved);
            root.RemoveHandler(UIElement.PointerReleasedEvent, _released);
            root.RemoveHandler(UIElement.PointerCaptureLostEvent, _lost);
            root.RemoveHandler(UIElement.KeyDownEvent, _key);
            if (_pointer is not null) root.ReleasePointerCapture(_pointer);
        }
        _capture = null;
        _pointer = null;
        if (_ghost is not null) _ghost.IsOpen = false;
        _ghost = null;
        _source = null;
        TargetChanged?.Invoke();
        if (target is not null) host.Move(panel, target, frame);
    }

    // ---- Hit testing ----

    PanelDropTarget? HitTest(PointInt32 p, PanelId panel)
    {
        // The app's frontmost window under the pointer: drops are judged against MKLight's own windows.
        var ours = host.Trees.Where(t => t.HostWindow is not null).ToDictionary(t => WinRT.Interop.WindowNative.GetWindowHandle(t.HostWindow));
        var all = App.Windows.Select(WinRT.Interop.WindowNative.GetWindowHandle).Concat(ours.Keys).ToHashSet();
        var front = Native.FrontmostAt(p, all);
        // Not over one of the design's windows: a window of its own.
        if (front == 0 || !ours.TryGetValue(front, out var tree)) return Allowed(panel, new PanelDropTarget.NewWindow());
        var scale = tree.XamlRoot?.RasterizationScale ?? 1;
        var origin = Native.ClientOrigin(front);
        var pt = new Point((p.X - origin.X) / scale, (p.Y - origin.Y) / scale);
        var rect = tree.Bounds;
        if (rect.IsEmpty || !rect.Contains(pt)) return null;

        // Along the window's edges, when it has more than one group: a whole row or column.
        const double band = 16;
        PanelEdge? edge = pt.X - rect.X < band ? PanelEdge.Leading : rect.Right - pt.X < band ? PanelEdge.Trailing
            : pt.Y - rect.Y < band ? PanelEdge.Top : rect.Bottom - pt.Y < band ? PanelEdge.Bottom : null;
        if (edge is { } e && tree.Model.Groups.Count() > 1 && Allowed(panel, new PanelDropTarget.WindowEdge(tree.Model.Id, e)) is { } edgeTarget)
            return edgeTarget;
        foreach (var g in tree.Groups)
        {
            var r = PanelTreeView.BoundsOf(g);
            if (!r.IsEmpty && r.Contains(pt)) return Allowed(panel, g.Zone(pt, r));
        }
        return null;
    }

    PanelDropTarget? Allowed(PanelId panel, PanelDropTarget t) => host.Arrangement.CanMove(panel, t) ? t : null;

    /// <summary>A pointer position in a tree's window (effective pixels) on screen (pixels).</summary>
    internal static PointInt32 Screen(PanelTreeView tree, Point p)
    {
        var scale = tree.XamlRoot?.RasterizationScale ?? 1;
        var origin = tree.HostWindow is { } w ? Native.ClientOrigin(WinRT.Interop.WindowNative.GetWindowHandle(w)) : default;
        return new PointInt32(origin.X + (int)Math.Round(p.X * scale), origin.Y + (int)Math.Round(p.Y * scale));
    }

    // ---- New windows ----

    double Scale => _source?.XamlRoot?.RasterizationScale ?? host.XamlRoot?.RasterizationScale ?? 1;

    /// <summary>A new window under the pointer, the size of the group the panel left, its title bar at the pointer.</summary>
    RectInt32 FrameAt(PointInt32 p)
    {
        var s = Scale;
        var w = Math.Clamp(_sourceSize.Width, 480, 1100) * s;
        var h = Math.Clamp(_sourceSize.Height + 48, 320, 860) * s;
        return OnScreen(new RectInt32(p.X - (int)Math.Min(160 * s, w / 2), p.Y - (int)(20 * s), (int)w, (int)h), p);
    }

    /// <summary>Move to New Window: over the panel's place now, a little offset.</summary>
    public RectInt32 DefaultFrame(PanelId panel)
    {
        if (host.GroupViewOf(panel) is { } gv && gv.Tree.XamlRoot is { } root)
        {
            var s = root.RasterizationScale;
            var r = PanelTreeView.BoundsOf(gv);
            var origin = Screen(gv.Tree, new Point(r.X + 40, r.Y + 40));
            var w = Math.Clamp(r.Width, 480, 1100) * s;
            var h = Math.Clamp(r.Height + 48, 320, 860) * s;
            return OnScreen(new RectInt32(origin.X, origin.Y, (int)w, (int)h), origin);
        }
        var main = host.Window?.AppWindow;
        var c = main is null ? new PointInt32(400, 300) : new PointInt32(main.Position.X + main.Size.Width / 2, main.Position.Y + main.Size.Height / 2);
        var sc = Scale;
        return OnScreen(new RectInt32(c.X - (int)(300 * sc), c.Y - (int)(220 * sc), (int)(600 * sc), (int)(440 * sc)), c);
    }

    static RectInt32 OnScreen(RectInt32 f, PointInt32 near)
    {
        var area = Microsoft.UI.Windowing.DisplayArea.GetFromPoint(near, Microsoft.UI.Windowing.DisplayAreaFallback.Nearest).WorkArea;
        var w = Math.Min(f.Width, area.Width);
        var h = Math.Min(f.Height, area.Height);
        var x = Math.Clamp(f.X, area.X, area.X + area.Width - w);
        var y = Math.Clamp(f.Y, area.Y, area.Y + area.Height - h);
        return new RectInt32(x, y, w, h);
    }

    // ---- The ghost ----

    /// <summary>What follows the pointer: the panel's title on a glass capsule, or, outside the design's windows, the outline of the window it would open in.</summary>
    void ShowGhost(PanelTreeView tree, PanelId panel)
    {
        _ghost = new Popup
        {
            XamlRoot = tree.XamlRoot,
            ShouldConstrainToRootBounds = false,
            IsHitTestVisible = false,
            Child = new Grid { IsHitTestVisible = false, RequestedTheme = tree.ActualTheme },
        };
        _ghostTitle = host.Title(panel);
        _ghostGlyph = host.Glyph(panel);
        _ghost.IsOpen = true;
    }

    string _ghostTitle = "", _ghostGlyph = "";
    bool? _ghostAsWindow;

    void MoveGhost(PointInt32 p, PanelDropTarget? t)
    {
        if (_ghost is not { Child: Grid grid } || _source is not { } tree) return;
        var s = Scale;
        var origin = Screen(tree, new Point(0, 0));
        var asWindow = NewWindowFrame is not null;
        if (_ghostAsWindow != asWindow)
        {
            _ghostAsWindow = asWindow;
            grid.Children.Clear();
            var chip = PanelTreeView.Chip(tree, _ghostTitle, _ghostGlyph);
            chip.HorizontalAlignment = HorizontalAlignment.Left;
            chip.VerticalAlignment = VerticalAlignment.Top;
            if (asWindow)
            {
                var accent = (global::Windows.UI.Color)ThemeLookup.Resource(tree.ActualTheme, "SystemAccentColor");
                grid.Children.Add(new Border
                {
                    CornerRadius = new CornerRadius(8), BorderThickness = new Thickness(2),
                    BorderBrush = ThemeLookup.Brush(tree, "AccentFillColorDefaultBrush"),
                    Background = new SolidColorBrush(accent) { Opacity = 0.08 },
                });
                chip.Margin = new Thickness(12);
                grid.Children.Add(chip);
                grid.Children.Add(PanelTreeView.Chip(tree, "New Window", "\uE8A7"));
            }
            else grid.Children.Add(chip);
        }
        if (NewWindowFrame is { } f)
        {
            grid.Width = f.Width / s;
            grid.Height = f.Height / s;
            _ghost.HorizontalOffset = (f.X - origin.X) / s;
            _ghost.VerticalOffset = (f.Y - origin.Y) / s;
        }
        else
        {
            grid.Width = double.NaN;
            grid.Height = double.NaN;
            _ghost.HorizontalOffset = (p.X - origin.X) / s + 12;
            _ghost.VerticalOffset = (p.Y - origin.Y) / s + 10;
        }
        grid.Opacity = t is null ? 0.55 : 1;
    }

    // ---- Windows on screen ----

    static partial class Native
    {
        [StructLayout(LayoutKind.Sequential)]
        struct RECT { public int Left, Top, Right, Bottom; }

        [StructLayout(LayoutKind.Sequential)]
        struct POINT { public int X, Y; }

        /// <summary>The first window of <paramref name="candidates"/> under <paramref name="p"/>, in z-order (0: none).</summary>
        public static nint FrontmostAt(PointInt32 p, HashSet<nint> candidates)
        {
            for (var h = GetTopWindow(0); h != 0; h = GetWindow(h, 2 /* GW_HWNDNEXT */))
            {
                if (!candidates.Contains(h) || !IsWindowVisible(h) || IsIconic(h)) continue;
                if (GetWindowRect(h, out var r) && p.X >= r.Left && p.X < r.Right && p.Y >= r.Top && p.Y < r.Bottom) return h;
            }
            return 0;
        }

        public static PointInt32 ClientOrigin(nint hwnd)
        {
            var pt = new POINT();
            ClientToScreen(hwnd, ref pt);
            return new PointInt32(pt.X, pt.Y);
        }

        [LibraryImport("user32.dll")]
        private static partial nint GetTopWindow(nint hwnd);

        [LibraryImport("user32.dll")]
        private static partial nint GetWindow(nint hwnd, uint cmd);

        [LibraryImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static partial bool IsWindowVisible(nint hwnd);

        [LibraryImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static partial bool IsIconic(nint hwnd);

        [LibraryImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static partial bool GetWindowRect(nint hwnd, out RECT rect);

        [LibraryImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static partial bool ClientToScreen(nint hwnd, ref POINT point);
    }
}

/// <summary>
/// Makes an element a handle that moves a panel: press and drag it to
/// another place, another window or out into a window of its own; a click
/// (no drag) runs <c>click</c>; its context menu moves, hides or closes the
/// panel.
/// </summary>
internal static class PanelHandles
{
    const double Threshold = 6;

    public static void Attach(PanelHost host, UIElement handle, PanelId panel, Action? click = null, UIElement? exclude = null)
    {
        (uint Id, Point Start)? press = null;
        handle.AddHandler(UIElement.PointerPressedEvent, new PointerEventHandler((_, e) =>
        {
            var pt = e.GetCurrentPoint(null);
            if (!pt.Properties.IsLeftButtonPressed || (exclude is not null && IsWithin(e.OriginalSource as DependencyObject, exclude))) { press = null; return; }
            press = (e.Pointer.PointerId, pt.Position);
        }), true);
        handle.AddHandler(UIElement.PointerMovedEvent, new PointerEventHandler((_, e) =>
        {
            if (press is not { } p || p.Id != e.Pointer.PointerId) return;
            var pt = e.GetCurrentPoint(null);
            if (!pt.Properties.IsLeftButtonPressed) { press = null; return; }
            if (Math.Abs(pt.Position.X - p.Start.X) < Threshold && Math.Abs(pt.Position.Y - p.Start.Y) < Threshold) return;
            press = null;
            host.Drag.Begin(panel, e, handle);
        }), true);
        handle.AddHandler(UIElement.PointerReleasedEvent, new PointerEventHandler((_, e) =>
        {
            var wasClick = press is { } p && p.Id == e.Pointer.PointerId;
            press = null;
            if (wasClick) click?.Invoke();
        }), true);
        handle.AddHandler(UIElement.PointerCanceledEvent, new PointerEventHandler((_, _) => press = null), true);
        if (handle is FrameworkElement fe) fe.ContextFlyout = PanelMenus.ForPanel(host, panel);
    }

    /// <summary>A panel view's title becomes its handle: an analysis pane's title button, the lens data's title, the layout's title chip.</summary>
    public static void AttachToPanel(PanelHost host, FrameworkElement view, PanelId panel)
    {
        void Hook()
        {
            var handles = new List<UIElement>();
            UIElement? header = null;
            if (view is Analysis.AnalysisPaneView pane)
            {
                if (pane.FindName("TitleButton") is UIElement title) handles.Add(header = title);
            }
            else if (FindText(view, panel == PanelId.LensData ? "Lens Data" : "Layout") is { } text)
            {
                header = text;
                // The title and its glyph (the layout's whole chip).
                var parent = text.Parent as Panel;
                if (panel == PanelId.Layout && parent?.Parent is Border chip) handles.Add(chip);
                else
                {
                    handles.Add(text);
                    if (parent?.Children.OfType<FontIcon>().FirstOrDefault() is { } glyph) handles.Add(glyph);
                }
            }
            // A stable automation id (UI tests), unless the view set one.
            if (header is not null && string.IsNullOrEmpty(AutomationProperties.GetAutomationId(header)))
                AutomationProperties.SetAutomationId(header, $"Panel.{panel.Key}.Header");
            foreach (var h in handles)
            {
                Attach(host, h, panel);
                ToolTipService.SetToolTip(h, ToolTipService.GetToolTip(h) is string tip
                    ? tip + "\nDrag to move it, or out of the window to open it in a window of its own."
                    : "Drag to move, or out of the window to open in a window of its own.");
            }
        }
        Hook();
    }

    static bool IsWithin(DependencyObject? e, UIElement ancestor)
    {
        for (; e is not null; e = VisualTreeHelper.GetParent(e))
            if (ReferenceEquals(e, ancestor)) return true;
        return false;
    }

    /// <summary>The first text block reading <paramref name="text"/> in an element's content (logical children).</summary>
    static TextBlock? FindText(object? e, string text) => e switch
    {
        TextBlock t when t.Text == text => t,
        UserControl u => FindText(u.Content, text),
        ContentControl c => FindText(c.Content, text),
        Border b => FindText(b.Child, text),
        Panel p => p.Children.Select(c => FindText(c, text)).FirstOrDefault(t => t is not null),
        _ => null,
    };
}

/// <summary>The panel menus: a panel's context menu, and the analyses for Add Analysis.</summary>
internal static class PanelMenus
{
    /// <summary>Moving and closing a panel (its title's and tab's context menu), built when it opens.</summary>
    public static MenuFlyout ForPanel(PanelHost host, PanelId panel)
    {
        var menu = new MenuFlyout();
        menu.Opening += (_, _) =>
        {
            menu.Items.Clear();
            AddPanelItems(menu.Items, host, panel);
        };
        return menu;
    }

    /// <summary>Move to Main Window, Move to New Window, then Hide or Close Pane (see <see cref="PanelHost.AddPanelMenuItems"/>).</summary>
    public static void AddPanelItems(IList<MenuFlyoutItemBase> items, PanelHost host, PanelId panel)
    {
        var a = host.Arrangement;
        var start = items.Count;
        if (a.IsDetached(panel)) items.Add(Item("Move to Main Window", "\uE73F", () => host.MoveToMainWindow(panel)));
        if (!a.IsAloneInWindow(panel)) items.Add(Item("Move to New Window", "\uE8A7", () => host.MoveToNewWindow(panel)));
        if (items.Count > start) items.Add(new MenuFlyoutSeparator());
        items.Add(panel.PaneId is null
            ? Item($"Hide {host.Title(panel)}", "\uED1A", () => host.Close(panel))
            : Item("Close Pane", "\uE711", () => host.Close(panel)));
    }
    public static MenuFlyoutItem Item(string text, string glyph, Action click)
    {
        var item = new MenuFlyoutItem { Text = text, Icon = new FontIcon { Glyph = glyph } };
        item.Click += (_, _) => click();
        return item;
    }

    /// <summary>Every analysis by category (unavailable ones dimmed), as in the sidebar.</summary>
    public static void FillAnalyses(MenuFlyout menu, Workspace ws, Action<string> add)
    {
        void Fill()
        {
            menu.Items.Clear();
            foreach (var group in ws.PaneAnalyses.GroupBy(d => d.Category).OrderBy(g => AnalysisCatalog.CategoryRank(g.Key)).ThenBy(g => g.Key))
            {
                var sub = new MenuFlyoutSubItem { Text = AnalysisCatalog.CategoryTitle(group.Key) };
                foreach (var d in group.OrderBy(d => AnalysisCatalog.Title(d.Name, d.Title), StringComparer.Ordinal))
                {
                    var name = d.Name;
                    var item = new MenuFlyoutItem { Text = AnalysisCatalog.Title(d.Name, d.Title), IsEnabled = d.Available, Icon = new FontIcon { Glyph = AnalysisCatalog.Glyph(d.Name) } };
                    item.Click += (_, _) => add(name);
                    sub.Items.Add(item);
                }
                menu.Items.Add(sub);
            }
        }
        menu.Opening += (_, _) => Fill();
        Fill();
    }
}
