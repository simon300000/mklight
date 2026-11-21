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
    /// <summary>The menus by English name (their lookup key) and their titles in the UI language.</summary>
    static (string Name, string Title)[] MenuTitles =>
    [
        ("File", Loc.S("Shell_Menu_File")), ("Edit", Loc.S("Shell_Menu_Edit")), ("View", Loc.S("Shell_Menu_View")),
        ("Lens", Loc.S("Shell_Menu_Lens")), ("Design", Loc.S("Shell_Menu_Design")), ("Analysis", Loc.S("Shell_Menu_Analysis")),
        ("Help", Loc.S("Shell_Menu_Help")),
    ];

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
        foreach (var (name, title) in MenuTitles)
        {
            // Looked up by the English name (launch actions, registered commands); the title is translated.
            var m = new MenuBarItem { Title = title, AccessKey = name[..1] };
            m.AccessKeyDisplayRequested += (_, _) => RefreshMenus();
            _menus[name] = m;
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

    /// <summary>A submenu; its English text is kept in <c>Tag</c>, so <c>--do "menu=View/Ray Colors"</c> finds it in any language.</summary>
    static MenuFlyoutSubItem Sub(string key, string english, string? glyph = null) =>
        new() { Text = Loc.S(key), Tag = english, Icon = glyph is null ? null : new FontIcon { Glyph = glyph } };

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
                var item = new MenuFlyoutItem { Text = EngineText.T(e.System.Name) };
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
        var samples = Sub("Shell_File_NewFromSample", "New from Sample", "\uE8A5");
        AddSamples(samples, id => App.OpenDocument(DesignDocument.FromSample(id)));
        _menus[m].Items.Add(samples);
        Item(m, Loc.S("Shell_File_ShowWelcome"), App.ShowWelcome, "Ctrl+N");
        Item(m, Loc.S("Shell_File_Open"), () => _ = App.OpenWithPickerAsync(this), "Ctrl+O", glyph: "\uE838");
        _recentMenu = Sub("Shell_File_OpenRecent", "Open Recent", "\uE81C");
        _menus[m].Items.Add(_recentMenu);
        Separator(m);
        Item(m, Loc.S("Shell_File_Close"), () => _ = CloseAsync(), "Ctrl+W");
        Item(m, Loc.S("Shell_File_Save"), () => _ = SaveAsync(false), "Ctrl+S", glyph: "\uE74E");
        Item(m, Loc.S("Shell_File_SaveAs"), () => _ = SaveAsync(true), "Ctrl+Shift+S");
        Item(m, Loc.S("Shell_File_Duplicate"), () => _ = DuplicateAsync());
        Item(m, Loc.S("Shell_File_Rename"), () => _ = RenameAsync(), enabled: () => CanRename);
        Item(m, Loc.S("Shell_File_Revert"), () => _ = RevertAsync(), enabled: () => CanRevert);
        FinalGroup(m);
        Item(m, Loc.S("Shell_File_Exit"), () => _ = ExitAsync());
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
        var undo = Item(m, Loc.S("Shell_Edit_Undo"), MenuUndo, "Ctrl+Z", () => doc.CanUndo || EditText?.CanUndo == true, "\uE7A7");
        var redo = Item(m, Loc.S("Shell_Edit_Redo"), MenuRedo, "Ctrl+Y", () => doc.CanRedo || EditText?.CanRedo == true, "\uE7A6");
        _menuState.Add(() =>
        {
            var typing = EditText is not null;
            undo.Text = !typing && doc.UndoActionName is { } u ? Loc.F("Shell_Edit_UndoAction", u) : Loc.S("Shell_Edit_Undo");
            redo.Text = !typing && doc.RedoActionName is { } r ? Loc.F("Shell_Edit_RedoAction", r) : Loc.S("Shell_Edit_Redo");
        });
        Separator(m);
        // The focused text box's own commands (its keys reach it directly).
        Item(m, Loc.S("Shell_Edit_Cut"), () => OnText(t => t.CutSelectionToClipboard()), "Ctrl+X", () => EditText is { SelectionLength: > 0, IsReadOnly: false }, "\uE8C6");
        Item(m, Loc.S("Shell_Edit_Copy"), () => OnText(t => t.CopySelectionToClipboard()), "Ctrl+C", () => EditText is { SelectionLength: > 0 }, "\uE8C8");
        Item(m, Loc.S("Shell_Edit_Paste"), () => OnText(t => t.PasteFromClipboard()), "Ctrl+V", () => EditText is { IsReadOnly: false }, "\uE77F");
        Item(m, Loc.S("Shell_Edit_SelectAll"), () => OnText(t => t.SelectAll()), "Ctrl+A", () => EditText is not null);
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
        Toggle(m, Loc.S("Shell_View_Sidebar"), () => IsSidebarVisible = !IsSidebarVisible, () => IsSidebarVisible);
        Toggle(m, Loc.S("Shell_View_Inspector"), ToggleInspector, () => IsInspectorVisible, "Ctrl+Alt+P");
        Separator(m);
        var lens = Item(m, Loc.S("Shell_View_HideLensData"), () => TogglePanel(PanelId.LensData), glyph: AnalysisCatalog.LensDataGlyph);
        var layout = Item(m, Loc.S("Shell_View_HideLayout"), () => TogglePanel(PanelId.Layout), glyph: AnalysisCatalog.LayoutGlyph);
        _menuState.Add(() =>
        {
            lens.Text = Loc.S(arr.Contains(PanelId.LensData) ? "Shell_View_HideLensData" : "Shell_View_ShowLensData");
            layout.Text = Loc.S(arr.Contains(PanelId.Layout) ? "Shell_View_HideLayout" : "Shell_View_ShowLayout");
        });
        var panel = Sub("Shell_View_Panel", "Panel");
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
        PanelItem(Loc.S("Shell_View_MoveToNewWindow"), p => _panels.MoveToNewWindow(p), p => !arr.IsAloneInWindow(p));
        PanelItem(Loc.S("Shell_View_MoveToMainWindow"), p => _panels.MoveToMainWindow(p), arr.IsDetached);
        PanelItem(Loc.S("Shell_View_ClosePane"), p => _panels.Close(p), _ => true,
            p => Loc.S(p == PanelId.LensData ? "Shell_View_HideLensData" : p == PanelId.Layout ? "Shell_View_HideLayout" : "Shell_View_ClosePane"));
        Separator(m);
        var layoutShown = () => arr.Contains(PanelId.Layout);
        Radios(_menus[m].Items, "viewport", [(Loc.S("Shell_View_SectionView"), ViewportMode.SectionYZ), (Loc.S("Shell_View_SectionXZ"), ViewportMode.SectionXZ), (Loc.S("Shell_View_View3D"), ViewportMode.Orbit3D)],
            () => _panels.Layout.Mode, v => _panels.Layout.Mode = v, layoutShown);
        Item(m, Loc.S("Shell_View_FitLayout"), () => _panels.Layout.Fit(), "Ctrl+0", layoutShown);
        var colors = Sub("Shell_View_RayColors", "Ray Colors");
        Radios(colors.Items, "coloring", [(Loc.S("Shell_View_ByField"), RayColoring.Field), (Loc.S("Shell_View_ByWavelength"), RayColoring.Wavelength)],
            () => _panels.Layout.Coloring, c => _panels.Layout.Coloring = c, layoutShown);
        _menus[m].Items.Add(colors);
        FinalGroup(m);
        var appearance = Sub("Shell_View_Appearance", "Appearance", "\uE793");
        _themeItems = Radios(appearance.Items, "theme", [(Loc.S("Shell_View_ThemeSystem"), ElementTheme.Default), (Loc.S("Shell_View_ThemeLight"), ElementTheme.Light), (Loc.S("Shell_View_ThemeDark"), ElementTheme.Dark)],
            () => AppSettings.Theme, ThemeSwitch.Set);
        _menus[m].Items.Add(appearance);
        // The UI language: saved, shown from the next launch (the app is unpackaged; docs/WINDOWS.md "Languages").
        var language = Sub("Shell_Language_Title", "Language", "");
        var system = Languages.Resolve(null, UiLanguage.SystemLanguages());
        Radios(language.Items, "language",
            [(Loc.F("Shell_Language_System", system == Languages.SimplifiedChinese ? Loc.S("Shell_Language_Chinese") : Loc.S("Shell_Language_English")), Languages.System),
             (Loc.S("Shell_Language_English"), Languages.English), (Loc.S("Shell_Language_Chinese"), Languages.SimplifiedChinese)],
            () => AppSettings.Language, v =>
            {
                if (v == AppSettings.Language) return;
                AppSettings.Language = v;
                _session.ShowToast(Loc.S("Shell_Language_Note"));
            });
        _menus[m].Items.Add(language);
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
        Item(m, Loc.S("Shell_Lens_InsertBefore"), lens.InsertBefore, "Ctrl+Shift+I", Shown);
        Item(m, Loc.S("Shell_Lens_InsertAfter"), lens.InsertAfter, "Ctrl+Alt+I", Shown);
        Item(m, Loc.S("Shell_Lens_DeleteSurface"), lens.DeleteSelected, "Ctrl+Del", () => Shown() && Sel() is int i && LensCommands.CanDelete(_session.Document.System, i));
        Separator(m);
        Item(m, Loc.S("Shell_Lens_MakeStop"), lens.MakeSelectedStop, "Ctrl+Shift+T", () => Shown() && Sel() is int i && LensCommands.CanMakeStop(_session.Document.System, i));
        Item(m, Loc.S("Shell_Lens_ToggleVariable"), () => lens.ToggleVariable(), "Ctrl+Shift+V", () => Shown() && Sel() is > 0);
        Separator(m);
        var replace = Sub("Shell_Lens_ReplaceWithSample", "Replace with Sample");
        AddSamples(replace, ReplaceWithSample);
        _menus[m].Items.Add(replace);
    }

    // ---- Design ----

    void BuildDesignMenu()
    {
        const string m = "Design";
        void Job(string text, string page, string glyph) => Item(m, text, () => { SetInspector(true); _session.ShowInspectorPage(page); }, glyph: glyph);
        Job(Loc.S("Shell_Design_Optimize"), "optimize", "\uE9D9");
        Job(Loc.S("Shell_Design_GlassSubstitution"), "glass_substitution", "\uEB42");
        Job(Loc.S("Shell_Design_Tolerance"), "tolerance", "\uE9E9");
        Job(Loc.S("Shell_Design_ApplySolves"), "solves", "\uE943");
        Separator(m);
        Item(m, Loc.S("Shell_Design_ShowIssues"), ShowIssues, enabled: () => _session.Workspace.Issues.Count > 0);
    }

    // ---- Analysis ----

    void BuildAnalysisMenu()
    {
        const string m = "Analysis";
        var ws = _session.Workspace;
        _newPaneMenu = Sub("Shell_Analysis_NewPane", "New Analysis Pane", "\uECCD");
        _menus[m].Items.Add(_newPaneMenu);
        Separator(m);
        Item(m, Loc.S("Shell_Analysis_Run"), ws.RerunAll, "F5", () => ws.Panes.Count > 0 || arrHasLayout(), "\uE768");
        Item(m, Loc.S("Shell_Analysis_CancelRunning"), ws.CancelAll, "Ctrl+Esc", () => ws.IsBusy, "\uE711");
        Separator(m);
        var compute = Sub("Shell_Analysis_ComputeOn", "Compute On");
        Radios(compute.Items, "backend", [(Loc.S("Shell_Analysis_BackendAuto"), BackendPreference.Auto), (Loc.S("Shell_Analysis_BackendCpu"), BackendPreference.CpuReference), ("GPU", BackendPreference.Gpu)],
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
            var sub = new MenuFlyoutSubItem { Text = AnalysisCatalog.CategoryTitle(group.Key), Tag = AnalysisCatalog.EnglishCategoryTitle(group.Key) };
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
        Item(m, Loc.S("Shell_Help_Shortcuts"), () => _ = ShowShortcutsAsync(), glyph: "\uE765");
        FinalGroup(m);
        Item(m, Loc.S("Shell_Help_About"), () => _ = ShowAboutAsync(), glyph: "\uE946");
    }

    ContentDialog Dialog(string title, UIElement content) => new()
    {
        Title = title, Content = content, CloseButtonText = Loc.S("Common_Close"), DefaultButton = ContentDialogButton.Close,
        XamlRoot = Root.XamlRoot, RequestedTheme = Root.ActualTheme,
    };

    async Task ShowAboutAsync()
    {
        var info = _session.Workspace.Engine.Info;
        var text = new StackPanel { Spacing = 4 };
        text.Children.Add(new TextBlock { Text = Loc.S("Shell_About_Tagline"), TextWrapping = TextWrapping.Wrap });
        text.Children.Add(new TextBlock { Text = Loc.F("Shell_About_Version", App.AppVersion), Margin = new Thickness(0, 8, 0, 0) });
        text.Children.Add(new TextBlock { Text = Loc.F("Shell_About_Engine", info.Name, info.Version), Foreground = ThemeLookup.Brush(Root, "TextFillColorSecondaryBrush") });
        await Dialog("MKLight", text).ShowAsync();
    }

    /// <summary>The window's keys, as the menus show them (and the lens table's).</summary>
    static (string Keys, string Command)[] Shortcuts =>
    [
        ("Ctrl+N", Loc.S("Shell_File_ShowWelcome")), ("Ctrl+O", Loc.S("Shell_File_Open")), ("Ctrl+S", Loc.S("Shell_File_Save")),
        ("Ctrl+Shift+S", Loc.S("Shell_File_SaveAs")), ("Ctrl+W", Loc.S("Shell_File_Close")),
        ("Ctrl+Z", Loc.S("Shell_Edit_Undo")), ("Ctrl+Y / Ctrl+Shift+Z", Loc.S("Shell_Edit_Redo")),
        ("Ctrl+Alt+P", Loc.S("Shell_Shortcut_Inspector")), ("Ctrl+0", Loc.S("Shell_Shortcut_FitLayout")),
        ("Ctrl+Shift+I", Loc.S("Shell_Shortcut_InsertBefore")), ("Ctrl+Alt+I", Loc.S("Shell_Shortcut_InsertAfter")),
        ("Ctrl+Del", Loc.S("Shell_Shortcut_DeleteSurface")),
        ("Ctrl+Shift+T", Loc.S("Shell_Shortcut_MakeStop")), ("Ctrl+Shift+V", Loc.S("Shell_Shortcut_ToggleVariable")),
        ("F5", Loc.S("Shell_Shortcut_RunAnalyses")), ("Ctrl+Esc", Loc.S("Shell_Shortcut_CancelAnalyses")),
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
        await Dialog(Loc.S("Shell_Help_Shortcuts"), new ScrollViewer { Content = grid, MaxHeight = 480 }).ShowAsync();
    }

    // ---- Registered commands ----

    /// <summary>Commands features registered (<see cref="AppHooks.Commands"/>): in their menu above its final group, with their keys.</summary>
    void AddCommands()
    {
        // Grouped by the menus' English names (the dictionary's keys; the titles are translated).
        foreach (var group in AppHooks.Commands.GroupBy(c => c.Menu is { } name && _menus.ContainsKey(name) ? name : "File", StringComparer.OrdinalIgnoreCase))
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
                .LastOrDefault(s => string.Equals(s.Text, name, StringComparison.OrdinalIgnoreCase)
                    || s.Tag is string english && string.Equals(english, name, StringComparison.OrdinalIgnoreCase));
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
