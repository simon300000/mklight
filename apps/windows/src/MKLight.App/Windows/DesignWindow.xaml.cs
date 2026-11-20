using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using Fmt = MKLight.App.Views.Analysis.Results.Readers.Fmt;
using MKLight.App.ViewModels;
using MKLight.App.Views;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Layout;
using MKLight.App.Views.Panels;
using MKLight.App.Views.Inspector;
using MKLight.Engine;
using MKLight.Model;
using Windows.System;

namespace MKLight.App.Windows;

/// <summary>
/// A design window: custom title bar with the status capsule, the sidebar
/// (Design entries and every engine analysis by category, searchable), the
/// lens data across the top, the layout and the analysis panes below, and
/// the inspector column. Round 1 has a fixed split (tabs, dragging and
/// separate windows come later).
/// </summary>
public sealed partial class DesignWindow : Window
{
    readonly DesignSession _session;
    readonly PanelHost _panels;
    readonly InspectorView _inspector;
    bool _buildingSidebar;

    public DesignWindow(DesignSession session)
    {
        _session = session;
        InitializeComponent();
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        AppWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "MKLight.ico"));
        ThemeSwitch.Attach(this, Root, SyncThemeMenu);

        _panels = new PanelHost(session);
        ContentCard.Child = _panels;
        _panels.FocusChanged += UpdateSidebarState;
        _inspector = new InspectorView(session);
        Inspector.Child = _inspector;
        session.Toast += ShowToast;
        session.Document.Changed += UpdateTitle;
        session.Workspace.StateChanged += UpdateStatus;
        session.Workspace.StateChanged += UpdateSidebarState;
        session.Workspace.PanesChanged += () => { UpdateSidebarState(); UpdateAddMenu(); };
        session.Workspace.PreviewChanged += UpdateStatus;
        session.RunningJobsChanged += UpdateStatus;
        Status.IssueInvoked += i => { if (i.Surface is int s) session.SelectedSurface = s; };

        Status.SizeChanged += (_, _) => UpdatePassthrough();
        Menus.SizeChanged += (_, _) => UpdatePassthrough();
        Root.SizeChanged += (_, _) => UpdatePassthrough();
        BuildAccelerators();
        BuildMenus();
        UpdateTitle();
        UpdateStatus();
        // The capsule's issue glyphs are brushes picked in code.
        Root.ActualThemeChanged += (_, _) => UpdateStatus();
        UpdateAddMenu();
        _panels.Arrangement.Changed += () => { UpdateSidebarState(); UpdateMenus(); };
        InitDocuments();
        Closed += (_, _) => session.Dispose();
        AppWindow.Closing += OnClosing;
        AppHooks.RaiseWindowCreated(this);
    }

    bool _closeConfirmed;

    /// <summary>Unsaved changes: Save, Don't Save or Cancel (the window's one modal question).</summary>
    async void OnClosing(Microsoft.UI.Windowing.AppWindow sender, Microsoft.UI.Windowing.AppWindowClosingEventArgs args)
    {
        if (_closeConfirmed || !_session.Document.IsDirty || App.Options.IsScripted) return;
        args.Cancel = true;
        await CloseAsync();
    }

    /// <summary>Closes the window (File ▸ Close), asking about unsaved changes first; false when the user cancels.</summary>
    public async Task<bool> CloseAsync()
    {
        if (_closeConfirmed || !_session.Document.IsDirty || App.Options.IsScripted)
        {
            _closeConfirmed = true;
            Close();
            return true;
        }
        var dialog = new ContentDialog
        {
            Title = $"Save the changes to “{_session.Document.DisplayName}”?",
            Content = "Your changes are lost if you don't save them.",
            PrimaryButtonText = "Save", SecondaryButtonText = "Don't Save", CloseButtonText = "Cancel",
            DefaultButton = ContentDialogButton.Primary, XamlRoot = Root.XamlRoot, RequestedTheme = Root.ActualTheme,
        };
        var answer = await dialog.ShowAsync();
        if (answer == ContentDialogResult.None) return false;
        if (answer == ContentDialogResult.Primary && !await SaveAsync(false)) return false;
        _closeConfirmed = true;
        Close();
        return true;
    }

    public DesignSession Session => _session;
    /// <summary>The content area (lens data, layout, panes).</summary>
    public PanelHost Panels => _panels;
    /// <summary>The inspector column.</summary>
    public InspectorView InspectorPanel => _inspector;
    /// <summary>The window's root element (its theme, keyboard accelerators, XamlRoot).</summary>
    public FrameworkElement RootElement => Root;

    /// <summary>
    /// The capsule sits over the title bar's drag area: let the pointer through to it (and to the menu bar).
    /// It stays centred on the window unless that would cover the menu bar or the trailing buttons; the
    /// design's name after the menu bar shows only where it fits before the capsule.
    /// </summary>
    void UpdatePassthrough()
    {
        if (Root.XamlRoot is null || Status.ActualWidth == 0) return;
        // The trailing buttons end before the caption buttons.
        var inset = Math.Round(AppWindow.TitleBar.RightInset / Root.XamlRoot.RasterizationScale) + 8;
        if (Math.Abs(RightButtons.Margin.Right - inset) > 0.5)
        {
            RightButtons.Margin = new Thickness(0, 0, inset, 0);
            DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, UpdatePassthrough);
            return;
        }
        var menuEnd = Menus.TransformToVisual(Root).TransformPoint(new global::Windows.Foundation.Point(Menus.ActualWidth, 0)).X;
        var buttonsStart = RightButtons.TransformToVisual(Root).TransformPoint(default).X;
        // A narrow window: Undo and Redo leave the title bar (the Edit menu has them) so the capsule fits.
        var room = buttonsStart - menuEnd - 24 - Status.ActualWidth;
        var compact = UndoButton.Visibility == Visibility.Collapsed;
        if (compact ? room >= 80 : room < 0)
        {
            UndoButton.Visibility = RedoButton.Visibility = UndoDivider.Visibility = compact ? Visibility.Visible : Visibility.Collapsed;
            DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, UpdatePassthrough);
            return;
        }
        var centredLeft = (Root.ActualWidth - Status.ActualWidth) / 2;
        var left = Math.Max(centredLeft, menuEnd + 16);
        if (left + Status.ActualWidth > buttonsStart - 8) left = Math.Max(menuEnd + 8, buttonsStart - 8 - Status.ActualWidth);
        var shift = Math.Round(left - centredLeft);
        DocTitle.Visibility = left - menuEnd - 10 >= 80 ? Visibility.Visible : Visibility.Collapsed;
        DocTitle.MaxWidth = Math.Max(0, left - menuEnd - 26);
        if (Math.Abs(Status.Margin.Left - shift * 2) > 1)
        {
            // A left margin moves the centred capsule by half of it; the regions follow once it has moved.
            Status.Margin = new Thickness(shift * 2, 0, 0, 0);
            DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, UpdatePassthrough);
            return;
        }
        var scale = Root.XamlRoot.RasterizationScale;
        global::Windows.Graphics.RectInt32 Rect(FrameworkElement e, double padLeft = 0)
        {
            var t = e.TransformToVisual(Root).TransformPoint(new global::Windows.Foundation.Point(-padLeft, 0));
            return new((int)(t.X * scale), (int)(t.Y * scale), (int)((e.ActualWidth + padLeft) * scale), (int)(e.ActualHeight * scale));
        }
        // Setting the regions replaces the title bar's own, so its buttons and pane toggle are listed too.
        Microsoft.UI.Input.InputNonClientPointerSource.GetForWindowId(AppWindow.Id)
            .SetRegionRects(Microsoft.UI.Input.NonClientRegionKind.Passthrough,
                [Rect(Status), Rect(RightButtons), Rect(Menus), new global::Windows.Graphics.RectInt32(0, 0, (int)(52 * scale), (int)(AppTitleBar.ActualHeight * scale))]);
    }

    // ---- Title, status ----

    void UpdateTitle()
    {
        var doc = _session.Document;
        var name = doc.DisplayName + (doc.IsDirty ? " •" : "");
        Title = $"{doc.DisplayName} — MKLight";
        DocTitle.Text = name;
        ToolTipService.SetToolTip(DocTitle, doc.Path);
        UndoButton.IsEnabled = doc.CanUndo;
        RedoButton.IsEnabled = doc.CanRedo;
        ToolTipService.SetToolTip(UndoButton, doc.UndoActionName is { } u ? $"Undo {u} (Ctrl+Z)" : "Undo (Ctrl+Z)");
        ToolTipService.SetToolTip(RedoButton, doc.RedoActionName is { } r ? $"Redo {r} (Ctrl+Y)" : "Redo (Ctrl+Y)");
    }

    void ShowToast(string message)
    {
        ToastText.Text = message;
        Toast.Opacity = 1;
        var timer = DispatcherQueue.CreateTimer();
        timer.Interval = TimeSpan.FromSeconds(3);
        timer.IsRepeating = false;
        timer.Tick += (_, _) => Toast.Opacity = 0;
        timer.Start();
    }

    // ---- Sidebar ----

    string _query = "";
    readonly Dictionary<string, NavigationViewItem> _sidebarItems = [];

    public void BuildSidebar()
    {
        _buildingSidebar = true;
        Nav.MenuItems.Clear();
        _sidebarItems.Clear();
        var ws = _session.Workspace;
        var q = _query.Trim().ToLowerInvariant();
        if (q.Length == 0)
        {
            Nav.MenuItems.Add(new NavigationViewItemHeader { Content = "Design" });
            Nav.MenuItems.Add(SidebarItem(":lens", "Lens Data", AnalysisCatalog.LensDataGlyph, "Show the lens data"));
            Nav.MenuItems.Add(SidebarItem(":layout", "Layout", AnalysisCatalog.LayoutGlyph, "Show the layout"));
        }
        if (ws.Issues.Count > 0 && q.Length == 0)
        {
            var issues = SidebarItem(":issues", "Issues", ws.Issues.Any(i => i.Severity == "error") ? AnalysisCatalog.ErrorGlyph : AnalysisCatalog.WarningGlyph, "Show the design's validation issues");
            issues.InfoBadge = new InfoBadge { Value = ws.Issues.Count };
            Nav.MenuItems.Add(issues);
        }
        var list = ws.PaneAnalyses.Where(d => q.Length == 0 || AnalysisCatalog.Title(d.Name, d.Title).ToLowerInvariant().Contains(q)
            || d.Name.Contains(q) || d.Description.ToLowerInvariant().Contains(q));
        foreach (var group in list.GroupBy(d => d.Category).OrderBy(g => AnalysisCatalog.CategoryRank(g.Key)).ThenBy(g => g.Key))
        {
            Nav.MenuItems.Add(new NavigationViewItemHeader { Content = AnalysisCatalog.CategoryTitle(group.Key) });
            foreach (var d in group.OrderBy(d => AnalysisCatalog.Title(d.Name, d.Title), StringComparer.Ordinal))
            {
                var item = SidebarItem(d.Name, AnalysisCatalog.Title(d.Name, d.Title), AnalysisCatalog.Glyph(d.Name),
                    d.Available ? d.Description : d.UnavailableReason ?? "Not available in this engine build");
                if (!d.Available) item.Opacity = 0.45;
                Nav.MenuItems.Add(item);
            }
        }
        if (Nav.MenuItems.Count == 0)
            Nav.MenuItems.Add(new NavigationViewItemHeader { Content = $"No analyses match “{_query}”" });
        UpdateSidebarState();
        _buildingSidebar = false;
    }

    NavigationViewItem SidebarItem(string tag, string title, string glyph, string tip)
    {
        var item = new NavigationViewItem { Content = title, Tag = tag, Icon = new FontIcon { Glyph = glyph } };
        ToolTipService.SetToolTip(item, tip);
        var menu = new MenuFlyout();
        if (tag is ":lens" or ":layout")
        {
            var panel = tag == ":lens" ? PanelId.LensData : PanelId.Layout;
            var show = new MenuFlyoutItem { Text = "Show " + title };
            show.Click += (_, _) => _panels.ShowPanel(panel.Key);
            var hide = new MenuFlyoutItem { Text = "Hide " + title };
            hide.Click += (_, _) => _panels.Close(panel);
            menu.Items.Add(show);
            menu.Items.Add(hide);
            menu.Opening += (_, _) =>
            {
                var shown = _panels.Arrangement.Contains(panel);
                show.Visibility = shown ? Visibility.Collapsed : Visibility.Visible;
                hide.Visibility = shown ? Visibility.Visible : Visibility.Collapsed;
            };
            item.ContextFlyout = menu;
        }
        else if (!tag.StartsWith(':'))
        {
            var open = new MenuFlyoutItem { Text = "Open in New Pane" };
            open.Click += (_, _) => _session.Workspace.AddPane(tag, after: _session.Workspace.FocusedPane);
            menu.Items.Add(open);
            item.ContextFlyout = menu;
        }
        _sidebarItems[tag] = item;
        return item;
    }

    /// <summary>Open analyses carry a dot (a spinner while running); the focused pane's analysis is selected.</summary>
    void UpdateSidebarState()
    {
        if (_sidebarItems.Count == 0 && _session.Workspace.Descriptors.Count > 0) { BuildSidebar(); return; }
        var ws = _session.Workspace;
        if (ws.Issues.Count > 0 != _sidebarItems.ContainsKey(":issues") && _query.Length == 0) { BuildSidebar(); return; }
        foreach (var (tag, item) in _sidebarItems)
        {
            if (tag is ":lens" or ":layout")
            {
                // A hidden panel's row is dimmed (a click shows it again).
                var title = tag == ":lens" ? "lens data" : "layout";
                var shown = _panels.Arrangement.Contains(tag == ":lens" ? PanelId.LensData : PanelId.Layout);
                item.Opacity = shown ? 1 : 0.5;
                ToolTipService.SetToolTip(item, shown ? $"Show the {title}" : $"The {title} is hidden: click to show it");
                continue;
            }
            if (tag.StartsWith(':')) continue;
            var pane = ws.Panes.FirstOrDefault(p => p.Analysis == tag);
            item.InfoBadge = pane is null ? null : new InfoBadge { Width = 6, Height = 6, MinWidth = 6, Opacity = pane.IsRunning ? 1 : 0.55, Background = ThemeLookup.Brush(Root, pane.IsRunning ? "AccentFillColorDefaultBrush" : "TextFillColorTertiaryBrush") };
        }
        _buildingSidebar = true;
        var focused = ws.FocusedPane?.Analysis;
        Nav.SelectedItem = focused is not null && _sidebarItems.TryGetValue(focused, out var sel) ? sel : null;
        _buildingSidebar = false;
    }

    void OnSidebarSelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (_buildingSidebar || args.SelectedItem is not NavigationViewItem { Tag: string tag }) return;
        switch (tag)
        {
            case ":lens": _panels.ShowPanel("lens_data"); break;
            case ":layout": _panels.ShowPanel("layout"); break;
            case ":issues": Status.ShowIssues(); break;
            default:
                if (_session.Workspace.Descriptor(tag) is { Available: false } d) { ShowToast(d.UnavailableReason ?? "Not available."); break; }
                var shift = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(global::Windows.UI.Core.CoreVirtualKeyStates.Down);
                if (_session.Workspace.Panes.Count == 0 || shift) _session.Workspace.AddPane(tag, after: _session.Workspace.FocusedPane);
                else _session.Workspace.Open(tag);
                break;
        }
        UpdateSidebarState();
    }

    void OnSidebarSearchChanged(AutoSuggestBox sender, AutoSuggestBoxTextChangedEventArgs args)
    {
        _query = sender.Text ?? "";
        BuildSidebar();
    }

    // ---- Panels ----

    void UpdateAddMenu()
    {
        AddAnalysisMenu.Items.Clear();
        foreach (var group in _session.Workspace.PaneAnalyses.GroupBy(d => d.Category).OrderBy(g => AnalysisCatalog.CategoryRank(g.Key)).ThenBy(g => g.Key))
        {
            var sub = new MenuFlyoutSubItem { Text = AnalysisCatalog.CategoryTitle(group.Key) };
            foreach (var d in group.OrderBy(d => AnalysisCatalog.Title(d.Name, d.Title)))
            {
                var item = new MenuFlyoutItem { Text = AnalysisCatalog.Title(d.Name, d.Title), IsEnabled = d.Available, Icon = new FontIcon { Glyph = AnalysisCatalog.Glyph(d.Name) } };
                var name = d.Name;
                item.Click += (_, _) => _session.Workspace.AddPane(name, after: _session.Workspace.FocusedPane);
                sub.Items.Add(item);
            }
            AddAnalysisMenu.Items.Add(sub);
        }
    }

    // ---- Commands ----

    void BuildAccelerators()
    {
        void Add(VirtualKey key, VirtualKeyModifiers mods, Action action)
        {
            var a = new KeyboardAccelerator { Key = key, Modifiers = mods };
            a.Invoked += (_, e) => { action(); e.Handled = true; };
            Root.KeyboardAccelerators.Add(a);
        }
        Root.KeyboardAcceleratorPlacementMode = KeyboardAcceleratorPlacementMode.Hidden;
        Add(VirtualKey.Z, VirtualKeyModifiers.Control, Undo);
        Add(VirtualKey.Y, VirtualKeyModifiers.Control, Redo);
        Add(VirtualKey.Z, VirtualKeyModifiers.Control | VirtualKeyModifiers.Shift, Redo);
        Add(VirtualKey.S, VirtualKeyModifiers.Control, () => _ = SaveAsync(false));
        Add(VirtualKey.S, VirtualKeyModifiers.Control | VirtualKeyModifiers.Shift, () => _ = SaveAsync(true));
        Add(VirtualKey.O, VirtualKeyModifiers.Control, () => _ = App.OpenWithPickerAsync(this));
        Add(VirtualKey.N, VirtualKeyModifiers.Control, App.ShowWelcome);
        Add(VirtualKey.W, VirtualKeyModifiers.Control, () => _ = CloseAsync());
        Add(VirtualKey.F5, VirtualKeyModifiers.None, () => _session.Workspace.RerunAll());
        Add(VirtualKey.Escape, VirtualKeyModifiers.Control, () => _session.Workspace.CancelAll());
        Add(VirtualKey.P, VirtualKeyModifiers.Control | VirtualKeyModifiers.Menu, ToggleInspector);
        Add(VirtualKey.Number0, VirtualKeyModifiers.Control, () => _panels.Layout.Fit());
    }

    static string Shortcut(VirtualKey key, VirtualKeyModifiers m) =>
        (m.HasFlag(VirtualKeyModifiers.Control) ? "Ctrl+" : "") + (m.HasFlag(VirtualKeyModifiers.Menu) ? "Alt+" : "")
        + (m.HasFlag(VirtualKeyModifiers.Shift) ? "Shift+" : "") + key.ToString();

    void Undo() { if (!IsTextEditing()) _session.Document.Undo(); }
    void Redo() { if (!IsTextEditing()) _session.Document.Redo(); }

    /// <summary>While a text box has the focus, Ctrl+Z belongs to it.</summary>
    bool IsTextEditing() => FocusManager.GetFocusedElement(Root.XamlRoot) is TextBox or AutoSuggestBox or NumberBox;

    void OnUndo(object sender, RoutedEventArgs e) => _session.Document.Undo();
    void OnRedo(object sender, RoutedEventArgs e) => _session.Document.Redo();

    public async Task<bool> SaveAsync(bool saveAs, string? path = null)
    {
        var doc = _session.Document;
        _panels.SaveNow(); // the panel arrangement's pending write (views) goes into this save
        path ??= saveAs || doc.Path is null ? await App.PickSavePathAsync(this, doc.Path is null ? doc.System.Name : Path.GetFileNameWithoutExtension(doc.Path)) : doc.Path;
        if (path is null) return false;
        try
        {
            IReadOnlyList<CatalogReference> inUse = [];
            try
            {
                var json = await _session.Workspace.Engine.LibraryCatalogsInUseJsonAsync();
                inUse = (JsonNode.Parse(json) as JsonArray ?? []).Select(CatalogReference.FromJson).ToList();
            }
            catch (Exception e) when (e is EngineException or System.Text.Json.JsonException) { }
            doc.Save(path, _session.Workspace.Engine.Info.Version, inUse);
            RecentFiles.Add(path);
            UpdateTitle();
            ShowToast($"Saved {Path.GetFileName(path)}");
            return true;
        }
        catch (Exception ex)
        {
            ShowToast($"Could not save: {ex.Message}");
            return false;
        }
    }

    void OnInspectorToggle(object sender, RoutedEventArgs e) => SetInspector(InspectorToggle.IsChecked == true);
    void ToggleInspector() { InspectorToggle.IsChecked = !(InspectorToggle.IsChecked == true); SetInspector(InspectorToggle.IsChecked == true); }

    bool IsInspectorVisible => InspectorToggle.IsChecked == true;

    void SetInspector(bool visible)
    {
        InspectorToggle.IsChecked = visible;
        Inspector.Visibility = visible ? Visibility.Visible : Visibility.Collapsed;
        InspectorColumn.Width = visible ? new GridLength(320) : new GridLength(0);
    }

    /// <summary>The title bar's pane button: a hidden sidebar comes back, a shown one switches between full and compact.</summary>
    void OnPaneToggleRequested(TitleBar sender, object args)
    {
        if (!Nav.IsPaneVisible) { Nav.IsPaneVisible = true; Nav.IsPaneOpen = true; }
        else Nav.IsPaneOpen = !Nav.IsPaneOpen;
    }

    // ---- Seams for the document commands and launch options ----

    /// <summary>The sidebar column is shown (View ▸ Sidebar, <c>--sidebar hidden</c>).</summary>
    public bool IsSidebarVisible
    {
        get => Nav.IsPaneVisible;
        set => Nav.IsPaneVisible = value;
    }

    /// <summary>Opens the issues list of the status capsule (Design ▸ Show Issues, <c>--show-issues</c>).</summary>
    public void ShowIssues() => Status.ShowIssues();
}
