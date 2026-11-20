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
using MKLight.Engine;
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
    readonly LayoutPanel _layout = new();
    readonly List<AnalysisPaneView> _paneViews = [];
    bool _buildingSidebar;

    public DesignWindow(DesignSession session)
    {
        _session = session;
        InitializeComponent();
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        AppWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "MKLight.ico"));
        Root.RequestedTheme = AppSettings.Theme;
        SyncThemeMenu();

        LensData.Host = session;
        _layout.SurfaceClicked += i => session.SelectedSurface = i;
        session.SelectionChanged += () => _layout.SetSelectedSurface(session.SelectedSurface);
        session.Toast += ShowToast;
        session.Document.Changed += UpdateTitle;
        session.Workspace.StateChanged += UpdateStatus;
        session.Workspace.StateChanged += UpdateSidebarState;
        session.Workspace.PanesChanged += RebuildPanes;
        session.Workspace.Layout.Changed += UpdateLayout;
        Status.IssueInvoked += i => { if (i.Surface is int s) session.SelectedSurface = s; };

        Status.SizeChanged += (_, _) => UpdatePassthrough();
        Root.SizeChanged += (_, _) => UpdatePassthrough();
        BuildAccelerators();
        UpdateTitle();
        UpdateStatus();
        RebuildPanes();
        UpdateLayout();
        Closed += (_, _) => session.Workspace.CancelAll();
    }

    public DesignSession Session => _session;

    public static void ApplyTheme(ElementTheme theme)
    {
        foreach (var w in App.Windows.OfType<DesignWindow>()) { w.Root.RequestedTheme = theme; w.SyncThemeMenu(); }
    }

    void SyncThemeMenu()
    {
        ThemeSystem.IsChecked = AppSettings.Theme == ElementTheme.Default;
        ThemeLight.IsChecked = AppSettings.Theme == ElementTheme.Light;
        ThemeDark.IsChecked = AppSettings.Theme == ElementTheme.Dark;
        // The caption buttons follow the content's theme.
        var dark = Root.ActualTheme == ElementTheme.Dark;
        AppWindow.TitleBar.ButtonForegroundColor = dark ? Microsoft.UI.Colors.White : Microsoft.UI.Colors.Black;
        AppWindow.TitleBar.ButtonHoverForegroundColor = AppWindow.TitleBar.ButtonForegroundColor;
    }

    /// <summary>The capsule sits over the title bar's drag area: let the pointer through to it.</summary>
    void UpdatePassthrough()
    {
        if (Root.XamlRoot is null || Status.ActualWidth == 0) return;
        var scale = Root.XamlRoot.RasterizationScale;
        global::Windows.Graphics.RectInt32 Rect(FrameworkElement e, double padLeft = 0)
        {
            var t = e.TransformToVisual(Root).TransformPoint(new global::Windows.Foundation.Point(-padLeft, 0));
            return new((int)(t.X * scale), (int)(t.Y * scale), (int)((e.ActualWidth + padLeft) * scale), (int)(e.ActualHeight * scale));
        }
        // Setting the regions replaces the title bar's own, so its buttons and pane toggle are listed too.
        Microsoft.UI.Input.InputNonClientPointerSource.GetForWindowId(AppWindow.Id)
            .SetRegionRects(Microsoft.UI.Input.NonClientRegionKind.Passthrough,
                [Rect(Status), Rect(RightButtons), new global::Windows.Graphics.RectInt32(0, 0, (int)(52 * scale), (int)(AppTitleBar.ActualHeight * scale))]);
    }

    // ---- Title, status ----

    void UpdateTitle()
    {
        var doc = _session.Document;
        var name = doc.DisplayName + (doc.IsDirty ? " •" : "");
        Title = $"{doc.DisplayName} — MKLight";
        AppTitleBar.Title = name;
        AppTitleBar.Subtitle = doc.System.Name;
        UndoButton.IsEnabled = doc.CanUndo;
        RedoButton.IsEnabled = doc.CanRedo;
        ToolTipService.SetToolTip(UndoButton, doc.UndoActionName is { } u ? $"Undo {u} (Ctrl+Z)" : "Undo (Ctrl+Z)");
        ToolTipService.SetToolTip(RedoButton, doc.RedoActionName is { } r ? $"Redo {r} (Ctrl+Y)" : "Redo (Ctrl+Y)");
    }

    void UpdateStatus()
    {
        var ws = _session.Workspace;
        var running = ws.Panes.Where(p => p.IsRunning).ToList();
        Status.SetActivity(ws.IsBusy ? (running.Count == 1 && running[0].Progress.Fraction > 0
            ? $"{AnalysisCatalog.ShortTitle(running[0].Analysis)} {running[0].Progress.Fraction * 100:0}%".Replace(',', '.')
            : "") : null);
        var json = ws.Summary.Output?.Json;
        var entries = (json?["entries"] as JsonArray ?? []).OfType<JsonObject>()
            .Select(e => (Key: e["key"].StringOrNull() ?? "", Value: e["value"].Nf(), Unit: e["unit"].StringOrNull() ?? "", Label: e["label"].StringOrNull() ?? ""))
            .Where(e => e.Value is not null).ToList();
        (string, string)? Entry(string label, params string[] keys) =>
            keys.Select(k => entries.FirstOrDefault(e => e.Key == k)).FirstOrDefault(e => e.Key is { Length: > 0 }) is { Key.Length: > 0 } e
                ? (label, Fmt.Value(e.Value!.Value, e.Unit)) : null;
        var numbers = new[] { Entry("EFL", "efl"), Entry("F/#", "working_f_number", "image_f_number"), Entry("TT", "total_track") }
            .Where(n => n is not null).Select(n => n!.Value).ToList();
        var hasErrors = ws.ModelErrors.Count > 0 || ws.Issues.Any(i => i.Severity == "error");
        Status.SetNumbers(hasErrors && numbers.Count == 0 ? [] : numbers,
            hasErrors ? "Design has errors" : string.IsNullOrEmpty(_session.Document.System.Name) ? "Untitled" : _session.Document.System.Name);
        Status.SetIssues(ws.Issues);

        var rows = new List<(string, string?)> { ("First-Order Data", null) };
        rows.AddRange(entries.Select(e => (e.Label, (string?)Fmt.Value(e.Value!.Value, e.Unit))));
        rows.Add(("Engine", null));
        rows.Add(("Engine", ws.Engine.Info.Name + " " + ws.Engine.Info.Version));
        rows.Add(("Snapshot", ws.Snapshot?.Hash.ToString() is { Length: > 16 } h ? h[..16] + "…" : "—"));
        rows.Add(("Compute backend", ws.Backend switch { BackendPreference.Auto => "Automatic", var b => b.ToString() }));
        if (App.Engine is FFIEngine ffi) rows.Add(("GPU kernels", GpuText(ffi.GetGpuPreparation())));
        Status.SetDetails(rows);
    }

    static string GpuText(GpuPreparation p) => p switch
    {
        GpuPreparation.Preparing x => $"Preparing ({x.Done} of {x.Total})",
        GpuPreparation.Ready x => x.FromCache > 0 ? $"Ready ({x.Kernels}, loaded from cache)" : $"Ready ({x.Kernels}, prepared in {x.Seconds:0.0} s)".Replace(',', '.'),
        GpuPreparation.Unavailable x => "No GPU: " + x.Reason,
        GpuPreparation.Failed x => $"Compiled on first use ({x.Message})",
        _ => "Not prepared",
    };

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
        if (!tag.StartsWith(':'))
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
            case ":lens": LensData.Focus(FocusState.Programmatic); break;
            case ":layout": _layout.Fit(); break;
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

    void RebuildPanes()
    {
        var ws = _session.Workspace;
        PanelsRow.Children.Clear();
        PanelsRow.ColumnDefinitions.Clear();
        foreach (var v in _paneViews.Where(v => !ws.Panes.Contains(v.Pane))) v.CloseRequested -= ClosePane;
        _paneViews.RemoveAll(v => !ws.Panes.Contains(v.Pane));

        // Layout on the left (a little wider), then the panes side by side, hairline dividers between.
        PanelsRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(ws.Panes.Count == 0 ? 1 : 1.25, GridUnitType.Star), MinWidth = 220 });
        PanelsRow.Children.Add(_layout);
        Grid.SetColumn(_layout, 0);
        var col = 1;
        foreach (var pane in ws.Panes)
        {
            var view = _paneViews.FirstOrDefault(v => v.Pane == pane);
            if (view is null)
            {
                view = new AnalysisPaneView(_session, pane);
                view.CloseRequested += ClosePane;
                view.Focused += v => { ws.FocusedPaneId = v.Pane.Id; UpdateFocus(); UpdateSidebarState(); };
                _paneViews.Add(view);
            }
            PanelsRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1) });
            var divider = new Border { Background = ThemeLookup.Brush(Root, "DividerBrush") };
            Grid.SetColumn(divider, col++);
            PanelsRow.Children.Add(divider);
            PanelsRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star), MinWidth = 200 });
            Grid.SetColumn(view, col++);
            PanelsRow.Children.Add(view);
        }
        UpdateFocus();
        UpdateSidebarState();
        UpdateAddMenu();
    }

    void UpdateFocus()
    {
        foreach (var v in _paneViews) v.SetFocused(_session.Workspace.Panes.Count > 1 && v.Pane.Id == _session.Workspace.FocusedPane?.Id);
    }

    void ClosePane(AnalysisPaneView view) => _session.Workspace.RemovePane(view.Pane);

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

    void UpdateLayout()
    {
        var layout = _session.Workspace.Layout;
        _layout.SetScene(_session.Section, layout.Phase == PanePhase.Failed ? layout.Message : _session.Workspace.ModelErrors.FirstOrDefault());
        _layout.SetStale(layout.IsStale, layout.IsRunning);
        _layout.SetSelectedSurface(_session.SelectedSurface);
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
        Add(VirtualKey.F5, VirtualKeyModifiers.None, () => _session.Workspace.RerunAll());
        Add(VirtualKey.Escape, VirtualKeyModifiers.Control, () => _session.Workspace.CancelAll());
        Add(VirtualKey.P, VirtualKeyModifiers.Control | VirtualKeyModifiers.Menu, ToggleInspector);
        Add(VirtualKey.Number0, VirtualKeyModifiers.Control, _layout.Fit);
    }

    void Undo() { if (!IsTextEditing()) _session.Document.Undo(); }
    void Redo() { if (!IsTextEditing()) _session.Document.Redo(); }

    /// <summary>While a text box has the focus, Ctrl+Z belongs to it.</summary>
    bool IsTextEditing() => FocusManager.GetFocusedElement(Root.XamlRoot) is TextBox or AutoSuggestBox or NumberBox;

    void OnUndo(object sender, RoutedEventArgs e) => _session.Document.Undo();
    void OnRedo(object sender, RoutedEventArgs e) => _session.Document.Redo();
    void OnNew(object sender, RoutedEventArgs e) => App.ShowWelcome();
    void OnOpen(object sender, RoutedEventArgs e) => _ = App.OpenWithPickerAsync(this);
    void OnSave(object sender, RoutedEventArgs e) => _ = SaveAsync(false);
    void OnSaveAs(object sender, RoutedEventArgs e) => _ = SaveAsync(true);

    public async Task<bool> SaveAsync(bool saveAs, string? path = null)
    {
        var doc = _session.Document;
        path ??= saveAs || doc.Path is null ? await App.PickSavePathAsync(this, doc.Path is null ? doc.System.Name : Path.GetFileNameWithoutExtension(doc.Path)) : doc.Path;
        if (path is null) return false;
        try
        {
            doc.Save(path, _session.Workspace.Engine.Info.Version);
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

    void OnTheme(object sender, RoutedEventArgs e)
    {
        AppSettings.Theme = (sender as FrameworkElement)?.Tag switch { "light" => ElementTheme.Light, "dark" => ElementTheme.Dark, _ => ElementTheme.Default };
        ApplyTheme(AppSettings.Theme);
        // Theme resources WinUI resolves for the whole app (styles set in code, e.g. the lens table's
        // cells) follow the app's theme, which is fixed at launch.
        var app = Application.Current.RequestedTheme == ApplicationTheme.Dark ? ElementTheme.Dark : ElementTheme.Light;
        if (AppSettings.Theme != ElementTheme.Default && AppSettings.Theme != app)
            ShowToast("The appearance is fully applied the next time MKLight starts.");
    }

    void OnInspectorToggle(object sender, RoutedEventArgs e) => SetInspector(InspectorToggle.IsChecked == true);
    void ToggleInspector() { InspectorToggle.IsChecked = !(InspectorToggle.IsChecked == true); SetInspector(InspectorToggle.IsChecked == true); }

    void SetInspector(bool visible)
    {
        Inspector.Visibility = visible ? Visibility.Visible : Visibility.Collapsed;
        InspectorColumn.Width = visible ? new GridLength(280) : new GridLength(0);
    }

    void OnInspectorTab(object sender, RoutedEventArgs e)
    {
        var system = ReferenceEquals(sender, InspectorSystemTab);
        InspectorSystemTab.IsChecked = system;
        InspectorSurfaceTab.IsChecked = !system;
        InspectorSystemTab.Background = system ? null : new SolidColorBrush(Microsoft.UI.Colors.Transparent);
        InspectorSurfaceTab.Background = system ? new SolidColorBrush(Microsoft.UI.Colors.Transparent) : null;
        InspectorTitle.Text = system ? "System" : "Surface";
    }

    void OnPaneToggleRequested(TitleBar sender, object args) => Nav.IsPaneOpen = !Nav.IsPaneOpen;
}
