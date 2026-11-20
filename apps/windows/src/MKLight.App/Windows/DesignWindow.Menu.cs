using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Automation.Provider;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Layout;
using MKLight.App.Views.Viewport3D;
using MKLight.Engine;
using MKLight.Model;
using Windows.System;

namespace MKLight.App.Windows;

/// <summary>
/// The design window's menu bar (File, Edit, View, Lens, Design, Analysis, Help) in the title bar, as
/// Windows 11 apps have it. Shortcut texts are shown; the keys themselves are handled by the window's
/// accelerators (<see cref="BuildAccelerators"/>), the lens table and the text boxes, never twice.
/// Items are enabled by context, refreshed when the pointer or the focus reaches the menu bar and on
/// document, selection and arrangement changes. Commands features register (<see cref="AppHooks.Commands"/>)
/// go into their menu above its final group.
/// </summary>
public sealed partial class DesignWindow
{
    static readonly string[] MenuTitles = ["File", "Edit", "View", "Lens", "Design", "Analysis", "Help"];

    readonly Dictionary<string, MenuBarItem> _menus = new(StringComparer.OrdinalIgnoreCase);
    /// <summary>The separator above a menu's final group (hooked commands go above it); none: at the end.</summary>
    readonly Dictionary<string, MenuFlyoutSeparator> _finalGroup = new(StringComparer.OrdinalIgnoreCase);
    /// <summary>Enabled state and labels, recomputed by <see cref="UpdateMenus"/>.</summary>
    readonly List<Action> _menuState = [];

    MenuFlyoutSubItem _recentMenu = null!, _newPaneMenu = null!;
    RadioMenuFlyoutItem[] _themeItems = [];
    /// <summary>The text box the Edit menu's Cut, Copy, Paste and Select All act on (the last one focused).</summary>
    TextBox? _editText;

    void BuildMenus()
    {
        foreach (var title in MenuTitles)
        {
            var m = new MenuBarItem { Title = title, AccessKey = title[..1] };
            m.AccessKeyDisplayRequested += (_, _) => RefreshMenus();
            _menus[title] = m;
            Menus.Items.Add(m);
        }
        BuildFileMenu();
        BuildEditMenu();
        BuildViewMenu();
        BuildLensMenu();
        BuildDesignMenu();
        BuildAnalysisMenu();
        BuildHelpMenu();
        AddCommands();

        // Opening a menu: the pointer or the keyboard reaches the bar first.
        Menus.PointerEntered += (_, _) => RefreshMenus();
        Menus.GotFocus += (_, _) => RefreshMenus();
        Root.GotFocus += (_, e) =>
        {
            if (e.OriginalSource is TextBox tb) _editText = tb;
            else if (e.OriginalSource is not (MenuBarItem or MenuBar)) _editText = null;
        };
        _session.Document.Changed += UpdateMenus;
        _session.SelectionChanged += UpdateMenus;
        _session.Workspace.PanesChanged += UpdateMenus;
        _session.RunningJobsChanged += UpdateMenus;
        _panels.FocusChanged += UpdateMenus;
        RefreshMenus();
    }

    /// <summary>Lists that change outside the window (recent files, analyses), then the enabled state.</summary>
    void RefreshMenus()
    {
        BuildRecentMenu();
        BuildNewPaneMenu();
        UpdateMenus();
    }

    void UpdateMenus()
    {
        foreach (var update in _menuState) update();
    }

    // ---- Building blocks ----

    MenuFlyoutItem Item(string menu, string text, Action invoke, string? shortcut = null, Func<bool>? enabled = null, string? glyph = null)
    {
        var item = new MenuFlyoutItem { Text = text };
        Fill(item, invoke, shortcut, enabled, glyph);
        _menus[menu].Items.Add(item);
        return item;
    }

    void Fill(MenuFlyoutItem item, Action invoke, string? shortcut, Func<bool>? enabled, string? glyph)
    {
        item.Click += (_, _) => invoke();
        if (shortcut is not null) item.KeyboardAcceleratorTextOverride = shortcut;
        if (glyph is not null) item.Icon = new FontIcon { Glyph = glyph };
        if (enabled is not null) _menuState.Add(() => item.IsEnabled = enabled());
    }

    ToggleMenuFlyoutItem Toggle(string menu, string text, Action invoke, Func<bool> isOn, string? shortcut = null)
    {
        var item = new ToggleMenuFlyoutItem { Text = text };
        item.Click += (_, _) => invoke();
        if (shortcut is not null) item.KeyboardAcceleratorTextOverride = shortcut;
        _menuState.Add(() => item.IsChecked = isOn());
        _menus[menu].Items.Add(item);
        return item;
    }

    void Separator(string menu) => _menus[menu].Items.Add(new MenuFlyoutSeparator());

    /// <summary>Marks where the menu's final group starts (registered commands go above it).</summary>
    void FinalGroup(string menu)
    {
        var s = new MenuFlyoutSeparator();
        _finalGroup[menu] = s;
        _menus[menu].Items.Add(s);
    }

    MenuFlyoutSubItem Sub(string text, string? glyph = null) =>
        new() { Text = text, Icon = glyph is null ? null : new FontIcon { Glyph = glyph } };

    RadioMenuFlyoutItem[] Radios<T>(IList<MenuFlyoutItemBase> into, string group, (string Text, T Value)[] choices, Func<T> current, Action<T> set, Func<bool>? enabled = null)
    {
        var items = choices.Select(c =>
        {
            var r = new RadioMenuFlyoutItem { Text = c.Text, GroupName = group };
            r.Click += (_, _) => set(c.Value);
            into.Add(r);
            return r;
        }).ToArray();
        _menuState.Add(() =>
        {
            var now = current();
            for (var i = 0; i < items.Length; i++)
            {
                items[i].IsChecked = EqualityComparer<T>.Default.Equals(choices[i].Value, now);
                if (enabled is not null) items[i].IsEnabled = enabled();
            }
        });
        return items;
    }

    /// <summary>The built-in samples as the welcome window groups them: sample designs, then surface types.</summary>
    static IEnumerable<IReadOnlyList<SampleEntry>> SampleGroups()
    {
        var all = SampleSystems.Bundled.ToList();
        var extended = all.Where(s => !s.System.Surfaces.All(f => f.Shape.IsLegacy)).ToList();
        yield return all.Except(extended).ToList();
        if (extended.Count > 0) yield return extended;
    }

    static void AddSamples(MenuFlyoutSubItem sub, Action<string> pick)
    {
        foreach (var group in SampleGroups())
        {
            if (sub.Items.Count > 0) sub.Items.Add(new MenuFlyoutSeparator());
            foreach (var e in group)
            {
                var item = new MenuFlyoutItem { Text = e.System.Name };
                var id = e.Id;
                item.Click += (_, _) => pick(id);
                sub.Items.Add(item);
            }
        }
    }

    // ---- File ----

    void BuildFileMenu()
    {
        const string m = "File";
        var samples = Sub("New from Sample", "\uE8A5");
        AddSamples(samples, id => App.OpenDocument(DesignDocument.FromSample(id)));
        _menus[m].Items.Add(samples);
        Item(m, "Show Welcome Window", App.ShowWelcome, "Ctrl+N");
        Item(m, "Open…", () => _ = App.OpenWithPickerAsync(this), "Ctrl+O", glyph: "\uE838");
        _recentMenu = Sub("Open Recent", "\uE81C");
        _menus[m].Items.Add(_recentMenu);
        Separator(m);
        Item(m, "Close", () => _ = CloseAsync(), "Ctrl+W");
        Item(m, "Save", () => _ = SaveAsync(false), "Ctrl+S", glyph: "\uE74E");
        Item(m, "Save As…", () => _ = SaveAsync(true), "Ctrl+Shift+S");
        Item(m, "Duplicate", () => _ = DuplicateAsync());
        Item(m, "Rename…", () => _ = RenameAsync(), enabled: () => CanRename);
        Item(m, "Revert to Saved", () => _ = RevertAsync(), enabled: () => CanRevert);
        FinalGroup(m);
        Item(m, "Exit", () => _ = ExitAsync());
    }

    void BuildRecentMenu()
    {
        _recentMenu.Items.Clear();
        foreach (var path in RecentFiles.List)
        {
            var item = new MenuFlyoutItem { Text = Path.GetFileName(path) };
            ToolTipService.SetToolTip(item, path);
            item.Click += (_, _) => _ = App.OpenFileAsync(path);
            _recentMenu.Items.Add(item);
        }
        _recentMenu.IsEnabled = _recentMenu.Items.Count > 0;
    }

    /// <summary>Closes every window, asking about each design's unsaved changes; stops at a Cancel.</summary>
    static async Task ExitAsync()
    {
        foreach (var w in App.Windows.OfType<DesignWindow>().ToList())
            if (!await w.CloseAsync()) return;
        foreach (var w in App.Windows.ToList()) w.Close();
    }

    // ---- Edit ----

    void BuildEditMenu()
    {
        const string m = "Edit";
        var doc = _session.Document;
        var undo = Item(m, "Undo", MenuUndo, "Ctrl+Z", () => doc.CanUndo || EditText?.CanUndo == true, "\uE7A7");
        var redo = Item(m, "Redo", MenuRedo, "Ctrl+Y", () => doc.CanRedo || EditText?.CanRedo == true, "\uE7A6");
        _menuState.Add(() =>
        {
            var typing = EditText is not null;
            undo.Text = !typing && doc.UndoActionName is { } u ? "Undo " + u : "Undo";
            redo.Text = !typing && doc.RedoActionName is { } r ? "Redo " + r : "Redo";
        });
        Separator(m);
        // The focused text box's own commands (its keys reach it directly).
        Item(m, "Cut", () => OnText(t => t.CutSelectionToClipboard()), "Ctrl+X", () => EditText is { SelectionLength: > 0, IsReadOnly: false }, "\uE8C6");
        Item(m, "Copy", () => OnText(t => t.CopySelectionToClipboard()), "Ctrl+C", () => EditText is { SelectionLength: > 0 }, "\uE8C8");
        Item(m, "Paste", () => OnText(t => t.PasteFromClipboard()), "Ctrl+V", () => EditText is { IsReadOnly: false }, "\uE77F");
        Item(m, "Select All", () => OnText(t => t.SelectAll()), "Ctrl+A", () => EditText is not null);
    }

    /// <summary>The text box last focused, while it is still on screen.</summary>
    TextBox? EditText => _editText is { IsLoaded: true, XamlRoot: not null } t ? t : null;

    void OnText(Action<TextBox> action)
    {
        if (EditText is not { } t) return;
        t.Focus(FocusState.Programmatic);
        action(t);
    }

    /// <summary>Text being edited undoes its typing first, as Ctrl+Z in it does.</summary>
    void MenuUndo()
    {
        if (EditText is { CanUndo: true }) OnText(x => x.Undo());
        else _session.Document.Undo();
    }

    void MenuRedo()
    {
        if (EditText is { CanRedo: true }) OnText(x => x.Redo());
        else _session.Document.Redo();
    }

    // ---- View ----

    void BuildViewMenu()
    {
        const string m = "View";
        var arr = _panels.Arrangement;
        Toggle(m, "Sidebar", () => IsSidebarVisible = !IsSidebarVisible, () => IsSidebarVisible);
        Toggle(m, "Inspector", ToggleInspector, () => IsInspectorVisible, "Ctrl+Alt+P");
        Separator(m);
        var lens = Item(m, "Hide Lens Data", () => TogglePanel(PanelId.LensData), glyph: AnalysisCatalog.LensDataGlyph);
        var layout = Item(m, "Hide Layout", () => TogglePanel(PanelId.Layout), glyph: AnalysisCatalog.LayoutGlyph);
        _menuState.Add(() =>
        {
            lens.Text = arr.Contains(PanelId.LensData) ? "Hide Lens Data" : "Show Lens Data";
            layout.Text = arr.Contains(PanelId.Layout) ? "Hide Layout" : "Show Layout";
        });
        var panel = Sub("Panel");
        _menus[m].Items.Add(panel);
        void PanelItem(string text, Action<PanelId> act, Func<PanelId, bool> can, Func<PanelId, string>? label = null)
        {
            var item = new MenuFlyoutItem { Text = text };
            item.Click += (_, _) => { if (_panels.FocusedPanel is { } p) act(p); };
            _menuState.Add(() =>
            {
                var p = _panels.FocusedPanel;
                item.IsEnabled = p is { } f && arr.Contains(f) && can(f);
                if (label is not null) item.Text = p is { } f2 ? label(f2) : text;
            });
            panel.Items.Add(item);
        }
        PanelItem("Move to New Window", p => _panels.MoveToNewWindow(p), p => !arr.IsAloneInWindow(p));
        PanelItem("Move to Main Window", p => _panels.MoveToMainWindow(p), arr.IsDetached);
        PanelItem("Close Pane", p => _panels.Close(p), _ => true,
            p => p == PanelId.LensData ? "Hide Lens Data" : p == PanelId.Layout ? "Hide Layout" : "Close Pane");
        Separator(m);
        var layoutShown = () => arr.Contains(PanelId.Layout);
        Radios(_menus[m].Items, "viewport", [("Section View", ViewportMode.SectionYZ), ("X–Z Section", ViewportMode.SectionXZ), ("3D View", ViewportMode.Orbit3D)],
            () => _panels.Layout.Mode, v => _panels.Layout.Mode = v, layoutShown);
        Item(m, "Fit Layout", () => _panels.Layout.Fit(), "Ctrl+0", layoutShown);
        var colors = Sub("Ray Colors");
        Radios(colors.Items, "coloring", [("By Field", RayColoring.Field), ("By Wavelength", RayColoring.Wavelength)],
            () => _panels.Layout.Coloring, c => _panels.Layout.Coloring = c, layoutShown);
        _menus[m].Items.Add(colors);
        FinalGroup(m);
        var appearance = Sub("Appearance", "\uE793");
        _themeItems = Radios(appearance.Items, "theme", [("System", ElementTheme.Default), ("Light", ElementTheme.Light), ("Dark", ElementTheme.Dark)],
            () => AppSettings.Theme, ThemeSwitch.Set);
        _menus[m].Items.Add(appearance);
    }

    /// <summary>The theme items follow the app's theme (every window's menu).</summary>
    void SyncThemeMenu()
    {
        var theme = AppSettings.Theme;
        if (_themeItems.Length == 3)
        {
            _themeItems[0].IsChecked = theme == ElementTheme.Default;
            _themeItems[1].IsChecked = theme == ElementTheme.Light;
            _themeItems[2].IsChecked = theme == ElementTheme.Dark;
        }
    }

    void TogglePanel(PanelId panel)
    {
        if (_panels.Arrangement.Contains(panel)) _panels.Close(panel);
        else _panels.ShowPanel(panel.Key);
    }

    // ---- Lens ----

    void BuildLensMenu()
    {
        const string m = "Lens";
        var lens = _panels.LensData;
        bool Shown() => _panels.Arrangement.Contains(PanelId.LensData);
        int? Sel() => _session.SelectedSurface is int i && i < _session.Document.System.Surfaces.Count ? i : null;
        // The lens table handles these keys itself: the menu shows them.
        Item(m, "Insert Surface Before", lens.InsertBefore, "Ctrl+Shift+I", Shown);
        Item(m, "Insert Surface After", lens.InsertAfter, "Ctrl+Alt+I", Shown);
        Item(m, "Delete Surface", lens.DeleteSelected, "Ctrl+Del", () => Shown() && Sel() is int i && LensCommands.CanDelete(_session.Document.System, i));
        Separator(m);
        Item(m, "Make Aperture Stop", lens.MakeSelectedStop, "Ctrl+Shift+T", () => Shown() && Sel() is int i && LensCommands.CanMakeStop(_session.Document.System, i));
        Item(m, "Toggle Variable", () => lens.ToggleVariable(), "Ctrl+Shift+V", () => Shown() && Sel() is > 0);
        Separator(m);
        var replace = Sub("Replace with Sample");
        AddSamples(replace, ReplaceWithSample);
        _menus[m].Items.Add(replace);
    }

    // ---- Design ----

    void BuildDesignMenu()
    {
        const string m = "Design";
        void Job(string text, string page, string glyph) => Item(m, text, () => { SetInspector(true); _session.ShowInspectorPage(page); }, glyph: glyph);
        Job("Optimize…", "optimize", "\uE9D9");
        Job("Glass Substitution…", "glass_substitution", "\uEB42");
        Job("Tolerance…", "tolerance", "\uE9E9");
        Job("Apply Solves…", "solves", "\uE943");
        Separator(m);
        Item(m, "Show Issues", ShowIssues, enabled: () => _session.Workspace.Issues.Count > 0);
    }

    // ---- Analysis ----

    void BuildAnalysisMenu()
    {
        const string m = "Analysis";
        var ws = _session.Workspace;
        _newPaneMenu = Sub("New Analysis Pane", "\uECCD");
        _menus[m].Items.Add(_newPaneMenu);
        Separator(m);
        Item(m, "Run Analyses", ws.RerunAll, "F5", () => ws.Panes.Count > 0 || arrHasLayout(), "\uE768");
        Item(m, "Cancel Running", ws.CancelAll, "Ctrl+Esc", () => ws.IsBusy, "\uE711");
        Separator(m);
        var compute = Sub("Compute On");
        Radios(compute.Items, "backend", [("Automatic", BackendPreference.Auto), ("CPU Reference (f64)", BackendPreference.CpuReference), ("GPU", BackendPreference.Gpu)],
            () => ws.Backend, b => ChooseBackend(new ExecutionSettings(b, b == BackendPreference.Gpu ? ws.GpuName : null))); // saved with the design like the picker's choice
        _menus[m].Items.Add(compute);
        bool arrHasLayout() => _panels.Arrangement.Contains(PanelId.Layout);
    }

    int _newPaneCount = -1;

    /// <summary>The Add Analysis drop-down's categories.</summary>
    void BuildNewPaneMenu()
    {
        var ws = _session.Workspace;
        var list = ws.PaneAnalyses.ToList();
        if (list.Count == _newPaneCount) return;
        _newPaneCount = list.Count;
        _newPaneMenu.Items.Clear();
        foreach (var group in list.GroupBy(d => d.Category).OrderBy(g => AnalysisCatalog.CategoryRank(g.Key)).ThenBy(g => g.Key))
        {
            var sub = new MenuFlyoutSubItem { Text = AnalysisCatalog.CategoryTitle(group.Key) };
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(sub, "NewPane." + group.Key);
            foreach (var d in group.OrderBy(d => AnalysisCatalog.Title(d.Name, d.Title)))
            {
                var item = new MenuFlyoutItem { Text = AnalysisCatalog.Title(d.Name, d.Title), IsEnabled = d.Available, Icon = new FontIcon { Glyph = AnalysisCatalog.Glyph(d.Name) } };
                Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(item, "NewPane." + d.Name);
                var name = d.Name;
                item.Click += (_, _) => ws.AddPane(name, after: ws.FocusedPane);
                sub.Items.Add(item);
            }
            _newPaneMenu.Items.Add(sub);
        }
        _newPaneMenu.IsEnabled = _newPaneMenu.Items.Count > 0;
    }

    // ---- Help ----

    void BuildHelpMenu()
    {
        const string m = "Help";
        Item(m, "Keyboard Shortcuts", () => _ = ShowShortcutsAsync(), glyph: "\uE765");
        FinalGroup(m);
        Item(m, "About MKLight", () => _ = ShowAboutAsync(), glyph: "\uE946");
    }

    ContentDialog Dialog(string title, UIElement content) => new()
    {
        Title = title, Content = content, CloseButtonText = "Close", DefaultButton = ContentDialogButton.Close,
        XamlRoot = Root.XamlRoot, RequestedTheme = Root.ActualTheme,
    };

    async Task ShowAboutAsync()
    {
        var info = _session.Workspace.Engine.Info;
        var text = new StackPanel { Spacing = 4 };
        text.Children.Add(new TextBlock { Text = "Optical design, computed by the MKLight engine.", TextWrapping = TextWrapping.Wrap });
        text.Children.Add(new TextBlock { Text = $"Version {App.AppVersion}", Margin = new Thickness(0, 8, 0, 0) });
        text.Children.Add(new TextBlock { Text = $"Engine: {info.Name} {info.Version}", Foreground = ThemeLookup.Brush(Root, "TextFillColorSecondaryBrush") });
        await Dialog("MKLight", text).ShowAsync();
    }

    /// <summary>The window's keys, as the menus show them (and the lens table's).</summary>
    static readonly (string Keys, string Command)[] Shortcuts =
    [
        ("Ctrl+N", "Show Welcome Window"), ("Ctrl+O", "Open…"), ("Ctrl+S", "Save"), ("Ctrl+Shift+S", "Save As…"), ("Ctrl+W", "Close"),
        ("Ctrl+Z", "Undo"), ("Ctrl+Y / Ctrl+Shift+Z", "Redo"),
        ("Ctrl+Alt+P", "Show or hide the inspector"), ("Ctrl+0", "Fit the layout"),
        ("Ctrl+Shift+I", "Insert surface before"), ("Ctrl+Alt+I", "Insert surface after"), ("Ctrl+Del", "Delete surface"),
        ("Ctrl+Shift+T", "Make aperture stop"), ("Ctrl+Shift+V", "Toggle variable"),
        ("F5", "Run analyses"), ("Ctrl+Esc", "Cancel running analyses"),
    ];

    async Task ShowShortcutsAsync()
    {
        var grid = new Grid { ColumnSpacing = 24, RowSpacing = 6 };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var rows = Shortcuts.Concat(AppHooks.Commands.Where(c => c.Key is not null).Select(c => (Shortcut(c.Key!.Value, c.Modifiers), c.Text.TrimEnd('…'))));
        foreach (var (keys, command) in rows)
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var k = new TextBlock { Text = keys, Foreground = ThemeLookup.Brush(Root, "TextFillColorSecondaryBrush") };
            var c = new TextBlock { Text = command };
            Grid.SetRow(k, grid.RowDefinitions.Count - 1);
            Grid.SetRow(c, grid.RowDefinitions.Count - 1);
            Grid.SetColumn(c, 1);
            grid.Children.Add(k);
            grid.Children.Add(c);
        }
        await Dialog("Keyboard Shortcuts", new ScrollViewer { Content = grid, MaxHeight = 480 }).ShowAsync();
    }

    // ---- Registered commands ----

    /// <summary>Commands features registered (<see cref="AppHooks.Commands"/>): in their menu above its final group, with their keys.</summary>
    void AddCommands()
    {
        foreach (var group in AppHooks.Commands.GroupBy(c => c.Menu is { } name && _menus.ContainsKey(name) ? _menus[name].Title : "File"))
        {
            var items = _menus[group.Key].Items;
            int at;
            if (_finalGroup.TryGetValue(group.Key, out var final)) at = items.IndexOf(final);
            else
            {
                items.Add(new MenuFlyoutSeparator());
                at = items.Count;
            }
            foreach (var c in group.OrderBy(c => c.Order))
            {
                var item = new MenuFlyoutItem { Text = c.Text, Icon = new FontIcon { Glyph = c.Glyph } };
                item.Click += (_, _) => c.Invoke(this);
                if (c.Key is { } key)
                {
                    item.KeyboardAcceleratorTextOverride = Shortcut(key, c.Modifiers);
                    var a = new KeyboardAccelerator { Key = key, Modifiers = c.Modifiers };
                    a.Invoked += (_, e) => { c.Invoke(this); e.Handled = true; };
                    Root.KeyboardAccelerators.Add(a);
                }
                items.Insert(at++, item);
            }
            if (final is not null) items.Insert(at, new MenuFlyoutSeparator());
        }
    }


    // ---- Launch action: a menu open, for screenshots ----

    /// <summary>
    /// <c>--do menu=File</c> opens that menu; <c>--do "menu=File/New from Sample"</c> also opens that submenu.
    /// </summary>
    internal static class MenuLaunchActions
    {
        [ModuleInitializer]
        internal static void Register() => AppHooks.LaunchActions["menu"] = (w, arg) => w.ShowMenuForCaptureAsync(arg ?? "File");
    }

    /// <summary>
    /// Opens the menu (and submenus) through the automation peers. Menus are popups in windows of their own,
    /// which a capture of the design window leaves out, so each open menu is also rendered into an overlay of
    /// the window at its place (on the menu's fallback colour, as acrylic does not render to a bitmap).
    /// </summary>
    async Task ShowMenuForCaptureAsync(string path)
    {
        var parts = path.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0 || !_menus.TryGetValue(parts[0], out var menu)) return;
        RefreshMenus();
        new MenuBarItemAutomationPeer(menu).Expand();
        await Task.Delay(400);
        FocusFirst(OpenPopups().Select(p => p.Child).OfType<MenuFlyoutPresenter>().FirstOrDefault());
        foreach (var name in parts.Skip(1))
        {
            var sub = OpenPopups().SelectMany(p => p.Child is { } c ? Descendants<MenuFlyoutSubItem>(c) : [])
                .LastOrDefault(s => string.Equals(s.Text, name, StringComparison.OrdinalIgnoreCase));
            if (sub is null || FrameworkElementAutomationPeer.CreatePeerForElement(sub)?.GetPattern(PatternInterface.ExpandCollapse) is not IExpandCollapseProvider p) break;
            p.Expand();
            await Task.Delay(400);
            // As a pointer opens it: the submenu's first item without the keyboard focus ring.
            if (sub.Items.FirstOrDefault(i => i is MenuFlyoutItem) is Control first) first.Focus(FocusState.Pointer);
            await Task.Delay(200);
        }
        await Task.Delay(300);
        await OverlayOpenPopupsAsync();
    }

    /// <summary>
    /// Renders every open menu into an overlay of the window at its place, so a capture of the
    /// window shows them (on the popup's fallback colour, as acrylic does not render to a bitmap).
    /// </summary>
    internal async Task OverlayOpenPopupsAsync()
    {
        var overlay = new Canvas { IsHitTestVisible = false };
        Grid.SetRowSpan(overlay, 2);
        var fill = ThemeLookup.Brush(Root, "AcrylicBackgroundFillColorDefaultBrush") is Microsoft.UI.Xaml.Media.AcrylicBrush a
            ? new SolidColorBrush(a.FallbackColor) : ThemeLookup.Brush(Root, "SolidBackgroundFillColorQuarternaryBrush");
        foreach (var popup in OpenPopups().Reverse())
        {
            if (popup.Child is not MenuFlyoutPresenter child || child.ActualWidth == 0) continue;
            var bitmap = new Microsoft.UI.Xaml.Media.Imaging.RenderTargetBitmap();
            await bitmap.RenderAsync(child);
            var at = child.TransformToVisual(Root).TransformPoint(default);
            var frame = new Border
            {
                Width = child.ActualWidth, Height = child.ActualHeight, CornerRadius = new CornerRadius(8), Background = fill,
                BorderBrush = ThemeLookup.Brush(Root, "SurfaceStrokeColorFlyoutBrush"), BorderThickness = new Thickness(1),
                Child = new Image { Source = bitmap, Stretch = Stretch.Fill },
            };
            Canvas.SetLeft(frame, at.X);
            Canvas.SetTop(frame, at.Y);
            overlay.Children.Add(frame);
        }
        Root.Children.Add(overlay);
    }

    /// <summary>As a pointer opens a menu: its first item has the focus without the keyboard focus ring.</summary>
    static void FocusFirst(MenuFlyoutPresenter? menu) =>
        (menu is null ? null : Descendants<MenuFlyoutItemBase>(menu).FirstOrDefault(i => i.IsEnabled && i is not MenuFlyoutSeparator))?.Focus(FocusState.Pointer);

    IEnumerable<Microsoft.UI.Xaml.Controls.Primitives.Popup> OpenPopups() => VisualTreeHelper.GetOpenPopupsForXamlRoot(Root.XamlRoot);

    static IEnumerable<T> Descendants<T>(DependencyObject root) where T : DependencyObject
    {
        for (var i = 0; i < VisualTreeHelper.GetChildrenCount(root); i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is T t) yield return t;
            foreach (var d in Descendants<T>(child)) yield return d;
        }
    }
}
