using System.Diagnostics;
using System.Globalization;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.Windows.Storage.Pickers;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Glass;
using MKLight.Engine;
using MKLight.Model;
using Windows.ApplicationModel.DataTransfer;
using Windows.System;

namespace MKLight.App.Windows;

/// <summary>
/// The Glass Catalogs window (File ▸ Glass Catalogs…, Ctrl+Alt+G): catalogs in a sidebar (All
/// Glasses, Built-in, Library, User Glasses, the design's own, with counts), a bar naming the
/// selected catalog's origin and file with its actions, a searchable, sortable, filterable glass
/// table and the selected glass's details. It follows the frontmost design window and reads that
/// design's session (with no design open, a session of its own with the library). One per app.
/// </summary>
public sealed partial class GlassCatalogsWindow : Window
{
    // ---- The app-wide library ----

    static GlassLibrary? _library;

    /// <summary>
    /// The app-wide glass library, in <c>%LOCALAPPDATA%\MKLight</c>, or in the folder the
    /// <c>MKLIGHT_SUPPORT_DIRECTORY</c> environment variable names (checks with a scratch library).
    /// </summary>
    public static GlassLibrary Library
    {
        get
        {
            if (_library is not null) return _library;
            var dir = Environment.GetEnvironmentVariable("MKLIGHT_SUPPORT_DIRECTORY") is { Length: > 0 } d ? d : AppSettings.Directory;
            _library = new GlassLibrary(dir) { Trash = MoveToRecycleBin };
            _library.Changed += () => _ = LibraryChangedAsync();
            return _library;
        }
    }

    static void MoveToRecycleBin(string path) =>
        Microsoft.VisualBasic.FileIO.FileSystem.DeleteFile(path, Microsoft.VisualBasic.FileIO.UIOption.OnlyErrorDialogs,
            Microsoft.VisualBasic.FileIO.RecycleOption.SendToRecycleBin);

    /// <summary>The library changed: every open design loads it again, then this window reads its catalogs again.</summary>
    static async Task LibraryChangedAsync()
    {
        foreach (var w in App.Windows.OfType<DesignWindow>().ToList())
        {
            try
            {
                if (await GlassLibrarySync.SyncAsync(w.Session.Workspace.Engine, Library)) await w.Session.Workspace.CatalogsDidChangeAsync();
            }
            catch (EngineException e) { w.Session.ShowToast(e.Message); }
        }
        if (Instance is { } g) await g._model.RefreshAsync();
    }

    // ---- The window ----

    public static GlassCatalogsWindow? Instance { get; private set; }

    /// <summary>Shows the window (making it if needed), following <paramref name="design"/>.</summary>
    public static GlassCatalogsWindow Show(DesignWindow? design)
    {
        var w = Instance;
        if (w is null)
        {
            w = Instance = App.Track(new GlassCatalogsWindow());
            App.Size(w, 1480, 900);
        }
        if (design is not null) w.Follow(design);
        else _ = w._model.RefreshAsync();
        w.Activate();
        return w;
    }

    readonly GlassBrowserModel _model;
    readonly GlassDetailView _detail;
    DesignWindow? _design;
    bool _buildingSidebar;

    GlassCatalogsWindow()
    {
        InitializeComponent();
        Title = "Glass Catalogs — MKLight";
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        AppWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "MKLight.ico"));
        ThemeSwitch.Attach(this, Root);

        _model = new GlassBrowserModel(Library, () => App.Engine!.MakeDocumentEngine());
        _model.Changed += UpdateChrome;
        // The sidebar's links and the filter's accent are brushes picked in code.
        Root.ActualThemeChanged += (_, _) => UpdateChrome();
        _model.Message += ShowToast;
        Table.Model = _model;
        Table.ContextMenuRequested += (row, target, at) => GlassMenu(row).ShowAt(target, at);
        _detail = new GlassDetailView(_model)
        {
            Copy = () => { if (_model.SelectedRow is { } r) CopyName(r); },
            Edit = () => _ = EditSelectedAsync(),
            Delete = () => _ = DeleteSelectedAsync(),
            Duplicate = () => { if (_model.SelectedRow is { } r) _ = DuplicateAsync(r); },
        };
        DetailHost.Child = _detail;
        BuildFilterMenu();
        BuildAccelerators();
        UpdateChrome();
        Closed += (_, _) =>
        {
            Unfollow();
            if (ReferenceEquals(Instance, this)) Instance = null;
            if (_model.OwnEngine is IDisposable own) own.Dispose();
        };
    }

    // ---- Following the frontmost design ----

    /// <summary>A design window became the frontmost: the window reads its session.</summary>
    public void Follow(DesignWindow design)
    {
        if (ReferenceEquals(_design, design)) return;
        Unfollow();
        _design = design;
        var session = design.Session;
        session.SelectionChanged += OnDesignSelectionChanged;
        session.Document.Changed += OnDesignSelectionChanged;
        _ = _model.AttachAsync(Target(design));
    }

    /// <summary>A design window closed: follow another one, or none.</summary>
    public void Forget(DesignWindow design)
    {
        if (!ReferenceEquals(_design, design)) return;
        Unfollow();
        if (App.Windows.OfType<DesignWindow>().FirstOrDefault(w => !ReferenceEquals(w, design)) is { } other) Follow(other);
        else _ = _model.DetachAsync(design.Session);
    }

    void Unfollow()
    {
        if (_design is null) return;
        _design.Session.SelectionChanged -= OnDesignSelectionChanged;
        _design.Session.Document.Changed -= OnDesignSelectionChanged;
        _design = null;
    }

    void OnDesignSelectionChanged() => _detail.Refresh();

    static GlassBrowserTarget Target(DesignWindow design)
    {
        var session = design.Session;
        var doc = session.Document;
        return new GlassBrowserTarget(
            session,
            () => string.IsNullOrEmpty(doc.System.Name) ? doc.DisplayName : doc.System.Name,
            session.Workspace.Engine,
            () => session.SelectedSurface is int i && i > 0 && i < doc.System.ImageIndex ? i : null,
            (i, catalog, name) => doc.Edit("Change Material", s => i < s.Surfaces.Count
                ? s.WithSurface(i, f => f with { Material = LensValueParser.Replace(f.Material, new MaterialRef.CatalogGlass { Catalog = catalog, Name = name }) })
                : null),
            path => AddCatalogAsync(session, path),
            name => doc.EditProject("Remove Glass Catalog", p => p.WithCatalogs(p.Catalogs.Where(c => c.Name != name))),
            () => session.Workspace.CatalogsDidChangeAsync());
    }

    /// <summary>Loads a catalog file into the design's session and records it in the project (undoable).</summary>
    static async Task<CatalogInfo> AddCatalogAsync(DesignSession session, string path)
    {
        var info = await session.Workspace.Engine.LoadCatalogAsync(path);
        var format = Path.GetExtension(path).Equals(".json", StringComparison.OrdinalIgnoreCase) ? "json" : "agf";
        // The project section's change makes the session follow (DesignSession); a reference it already had needs a resnapshot only.
        if (!session.Document.EditProject("Add Glass Catalog", p => p.AddCatalog(new CatalogReference(info.Name, path, null, format))))
            await session.Workspace.CatalogsDidChangeAsync();
        return info;
    }

    /// <summary>File ▸ Add Glass Catalog to Design…: a file picker owned by <paramref name="owner"/>.</summary>
    public static async Task AddCatalogToDesignAsync(DesignWindow owner)
    {
        if (await PickCatalogFilesAsync(owner, multiple: false) is not [var path]) return;
        try
        {
            var info = await AddCatalogAsync(owner.Session, path);
            owner.Session.ShowToast($"Added {info.Name} ({info.GlassCount} glasses) to the design.");
        }
        catch (EngineException e) { owner.Session.ShowToast($"{Path.GetFileName(path)} could not be loaded: {e.Message}"); }
    }

    // ---- Chrome: title, sidebar, catalog bar ----

    void UpdateChrome()
    {
        AppTitleBar.Subtitle = _model.DesignTitle is { } t ? $"Glasses of {t}" : "Library";
        if (!string.Equals(SearchBox.Text, _model.Search, StringComparison.Ordinal)) SearchBox.Text = _model.Search;
        FilterIcon.Glyph = _model.Filter == GlassFilter.All ? "\uE71C" : "\uE16E";
        if (_model.Filter == GlassFilter.All) FilterIcon.ClearValue(IconElement.ForegroundProperty);
        else FilterIcon.Foreground = ThemeLookup.Brush(Root, "AccentTextFillColorPrimaryBrush");
        foreach (var item in FilterMenu.Items.OfType<RadioMenuFlyoutItem>()) item.IsChecked = (GlassFilter)item.Tag == _model.Filter;
        BuildSidebar();
        UpdateCatalogBar();
    }

    string SidebarKey => string.Join("|", _model.Catalogs.Select(c => $"{c.Name}:{c.GlassCount}:{c.Library}:{c.InProject}"))
        + "#" + _model.DesignTitle + "#" + string.Join("|", _model.Warnings) + "#" + Root.ActualTheme;
    string? _sidebarBuilt;

    void BuildSidebar()
    {
        var key = SidebarKey;
        if (key == _sidebarBuilt) { SelectScopeItem(); return; }
        _sidebarBuilt = key;
        _buildingSidebar = true;
        Nav.MenuItems.Clear();
        var total = _model.Catalogs.Sum(c => c.GlassCount);
        Nav.MenuItems.Add(Item("", "All Glasses", "\uF0E2", total, "Every glass the design can use"));
        foreach (var c in _model.BuiltinCatalogs)
            Nav.MenuItems.Add(Item(c.Name, "Built-in Glasses", "\uE7B8", c.GlassCount, "The glasses built into MKLight", null));

        Nav.MenuItems.Add(new NavigationViewItemHeader { Content = "Library" });
        foreach (var c in _model.LibraryCatalogs)
            Nav.MenuItems.Add(Item(c.Name, c.Name, "\uE8F1", c.GlassCount, Shown(c), LibraryMenu(c), c.Warnings.Count));
        if (!_model.LibraryCatalogs.Any())
            Nav.MenuItems.Add(LinkItem("Add Catalog…", "Catalogs and folders in the library load into every design", () => _ = AddLibraryFilesAsync()));

        Nav.MenuItems.Add(new NavigationViewItemHeader { Content = "User Glasses" });
        foreach (var c in _model.UserCatalogs)
            Nav.MenuItems.Add(Item(c.Name, c.Name, "\uE77B", c.GlassCount, Shown(c), UserMenu(c)));
        if (!_model.UserCatalogs.Any())
            Nav.MenuItems.Add(LinkItem("New User Catalog", "A catalog of your own glasses, kept with the library", () => { _model.NewUserCatalog(); }));

        if (_model.DesignTitle is { } title)
        {
            Nav.MenuItems.Add(new NavigationViewItemHeader { Content = title });
            foreach (var c in _model.DesignCatalogs)
                Nav.MenuItems.Add(Item(c.Name, c.Name, "\uE8A5", c.GlassCount, Shown(c), DesignMenu(c), c.Warnings.Count));
            Nav.MenuItems.Add(LinkItem("Add Catalog to Design…", "Record a catalog file in this design", () => _ = AddToDesignAsync()));
        }

        WarningsPanel.Children.Clear();
        foreach (var w in _model.Warnings)
        {
            var line = new Grid { ColumnSpacing = 6 };
            line.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            line.ColumnDefinitions.Add(new ColumnDefinition());
            line.Children.Add(new FontIcon { Glyph = "\uE7BA", FontSize = 12, Foreground = ThemeLookup.Brush(Root, "WarningBrush"), VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) });
            var text = new TextBlock { Text = w, Style = (Style)Application.Current.Resources["CaptionText"], TextWrapping = TextWrapping.Wrap };
            Grid.SetColumn(text, 1);
            line.Children.Add(text);
            WarningsPanel.Children.Add(line);
        }
        _buildingSidebar = false;
        SelectScopeItem();
    }

    /// <summary>A catalog's file as people read it (its name when it has none).</summary>
    static string Shown(CatalogInfo c) => c.Path is { } p ? GlassLibrary.PlainPath(p) : c.Name;

    void SelectScopeItem()
    {
        _buildingSidebar = true;
        var tag = _model.Scope ?? "";
        Nav.SelectedItem = Nav.MenuItems.OfType<NavigationViewItem>().FirstOrDefault(i => i.Tag as string == tag);
        _buildingSidebar = false;
    }

    NavigationViewItem Item(string tag, string title, string glyph, int count, string tip, MenuFlyout? menu = null, int warnings = 0)
    {
        var content = new Grid { ColumnSpacing = 8 };
        content.ColumnDefinitions.Add(new ColumnDefinition());
        content.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        content.Children.Add(new TextBlock { Text = title, TextTrimming = TextTrimming.CharacterEllipsis, VerticalAlignment = VerticalAlignment.Center });
        var right = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6, VerticalAlignment = VerticalAlignment.Center };
        if (warnings > 0)
        {
            var mark = new FontIcon { Glyph = "\uE7BA", FontSize = 11, Foreground = ThemeLookup.Brush(Root, "WarningBrush") };
            ToolTipService.SetToolTip(mark, $"{warnings} note{(warnings == 1 ? "" : "s")} while reading the file");
            right.Children.Add(mark);
        }
        right.Children.Add(new TextBlock { Text = count.ToString(CultureInfo.InvariantCulture), Style = (Style)Application.Current.Resources["SecondaryNumberText"], FontSize = 12 });
        Grid.SetColumn(right, 1);
        content.Children.Add(right);
        var item = new NavigationViewItem
        {
            Content = content, Tag = tag, Icon = new FontIcon { Glyph = glyph },
            HorizontalContentAlignment = HorizontalAlignment.Stretch,
        };
        ToolTipService.SetToolTip(item, tip);
        if (menu is not null) item.ContextFlyout = menu;
        return item;
    }

    NavigationViewItem LinkItem(string title, string tip, Action action)
    {
        var item = new NavigationViewItem
        {
            Content = new TextBlock { Text = title, Foreground = ThemeLookup.Brush(Root, "AccentTextFillColorPrimaryBrush") },
            Icon = new FontIcon { Glyph = "\uE710", FontSize = 12, Foreground = ThemeLookup.Brush(Root, "AccentTextFillColorPrimaryBrush") },
            SelectsOnInvoked = false, Tag = null,
        };
        ToolTipService.SetToolTip(item, tip);
        item.Tapped += (_, _) => action();
        return item;
    }

    void OnSidebarSelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (_buildingSidebar || args.SelectedItem is not NavigationViewItem { Tag: string tag }) return;
        _model.Scope = tag.Length == 0 ? null : tag;
    }

    void UpdateCatalogBar()
    {
        if (_model.Scope is not { } name || _model.Catalog(name) is not { } c)
        {
            CatalogBar.Visibility = Visibility.Collapsed;
            return;
        }
        var user = Library.IsUserCatalog(c.Name);
        CatalogBar.Visibility = Visibility.Visible;
        CatalogIcon.Glyph = c.IsBuiltin ? "\uE7B8" : user ? "\uE77B" : c.Library ? "\uE8F1" : "\uE8A5";
        CatalogTitle.Text = GlassBrowserModel.ScopeTitle(c.Name);
        CatalogOrigin.Text = GlassFormat.Origin(c, user);
        CatalogPath.Text = c.Path is { } file ? GlassLibrary.PlainPath(file) : (c.IsBuiltin ? "Published Sellmeier data of a few common glasses" : "");
        CatalogPath.Visibility = CatalogPath.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        ToolTipService.SetToolTip(CatalogPath, c.Path is null ? null : CatalogPath.Text);
        CatalogActions.Children.Clear();
        if (c.Warnings.Count > 0)
            CatalogActions.Children.Add(BarButton($"{c.Warnings.Count} note{(c.Warnings.Count == 1 ? "" : "s")}", "\uE7BA", "What the engine noticed while reading the file",
                () => _ = GlassDialogs.ShowListAsync(Root, $"{c.Name}: Notes", "Read from " + (Shown(c)) + ":",
                    c.Warnings.Select(w => (w.Line is int n ? $"Line {n}: " : "") + w.Message))));
        if (user)
            CatalogActions.Children.Add(BarButton("New Glass…", "\uE710", "Add a glass to this catalog", () => _ = NewGlassAsync()));
        else if (c.Path is not null)
            CatalogActions.Children.Add(BarButton("Reload", "\uE72C", "Read the catalog file again and list what changed", () => _ = ReloadAsync(c)));
        if (c.InProject)
            CatalogActions.Children.Add(BarButton("Remove from Design", "\uE738", "Remove the catalog from this design (undo with Ctrl+Z)", () => _model.RemoveFromDesign(c)));
        else if (c.Library && !user)
            CatalogActions.Children.Add(BarButton("Remove from Library", "\uE738", "Stop loading this catalog into every design (the file stays)", () => _model.RemoveFromLibrary(c)));
    }

    static Button BarButton(string text, string glyph, string tip, Action action)
    {
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 7 };
        p.Children.Add(new FontIcon { Glyph = glyph, FontSize = 12 });
        p.Children.Add(new TextBlock { Text = text });
        var b = new Button { Style = (Style)Application.Current.Resources["CapsuleButton"], Content = p, Height = 30 };
        ToolTipService.SetToolTip(b, tip);
        b.Click += (_, _) => action();
        return b;
    }

    // ---- Menus ----

    static MenuFlyoutItem MenuItem(string text, Action action, string? glyph = null, bool enabled = true)
    {
        var i = new MenuFlyoutItem { Text = text, IsEnabled = enabled };
        if (glyph is not null) i.Icon = new FontIcon { Glyph = glyph };
        i.Click += (_, _) => action();
        return i;
    }

    MenuFlyout LibraryMenu(CatalogInfo c)
    {
        var m = new MenuFlyout();
        m.Items.Add(MenuItem("Reload from Disk", () => _ = ReloadAsync(c), "\uE72C", c.Path is not null));
        m.Items.Add(MenuItem("Remove from Library", () => _model.RemoveFromLibrary(c), "\uE738"));
        if (c.Path is { } p) m.Items.Add(MenuItem("Show in File Explorer", () => Reveal(p), "\uEC50"));
        return m;
    }

    MenuFlyout UserMenu(CatalogInfo c)
    {
        var m = new MenuFlyout();
        m.Items.Add(MenuItem("New Glass…", () => { _model.Scope = c.Name; _ = NewGlassAsync(); }, "\uE710"));
        m.Items.Add(MenuItem("Rename…", () => _ = RenameAsync(c.Name), "\uE8AC"));
        m.Items.Add(MenuItem("Export as AGF…", () => _ = ExportAsync(c.Name), "\uEDE1"));
        if (c.Path is { } p) m.Items.Add(MenuItem("Show in File Explorer", () => Reveal(p), "\uEC50"));
        m.Items.Add(new MenuFlyoutSeparator());
        m.Items.Add(MenuItem("Move to Recycle Bin…", () => _ = DeleteCatalogAsync(c.Name), "\uE74D"));
        return m;
    }

    MenuFlyout DesignMenu(CatalogInfo c)
    {
        var m = new MenuFlyout();
        m.Items.Add(MenuItem("Reload from Disk", () => _ = ReloadAsync(c), "\uE72C", c.Path is not null));
        m.Items.Add(MenuItem("Remove from Design", () => _model.RemoveFromDesign(c), "\uE738"));
        if (c.Path is { } p) m.Items.Add(MenuItem("Show in File Explorer", () => Reveal(p), "\uEC50"));
        return m;
    }

    /// <summary>A glass's context menu (table rows).</summary>
    MenuFlyout GlassMenu(GlassRow row)
    {
        var m = new MenuFlyout();
        var (index, _) = _model.Assignment;
        m.Items.Add(MenuItem(index is int i ? $"Use for Surface {i}" : "Use for Selected Surface", () => { _model.Selection = row.Id; _model.UseSelected(); }, "\uE8B0", index is not null));
        m.Items.Add(MenuItem("Copy Name", () => CopyName(row), "\uE8C8"));
        m.Items.Add(new MenuFlyoutSeparator());
        var status = new MenuFlyoutSubItem { Text = "Status" };
        var own = Library.OverrideFor(row.Catalog, row.Name)?.Status;
        foreach (var s in GlassStatus.Choices)
        {
            var item = new RadioMenuFlyoutItem { Text = s.Title, GroupName = "status", IsChecked = own == s };
            item.Click += (_, _) => _model.SetStatus(s, row);
            status.Items.Add(item);
        }
        status.Items.Add(new MenuFlyoutSeparator());
        var catalog = new RadioMenuFlyoutItem { Text = "Catalog's Status", GroupName = "status", IsChecked = own is null };
        catalog.Click += (_, _) => _model.SetStatus(null, row);
        status.Items.Add(catalog);
        m.Items.Add(status);
        m.Items.Add(MenuItem(row.ExcludeSubstitution ? "Allow in Glass Substitution" : "Exclude from Glass Substitution",
            () => _model.SetExcluded(!row.ExcludeSubstitution, row), "\uE733"));
        if (row.Overridden) m.Items.Add(MenuItem("Reset to Catalog Settings", () => _model.ResetStatus(row), "\uE777"));
        m.Items.Add(new MenuFlyoutSeparator());
        m.Items.Add(MenuItem(Library.IsUserCatalog(row.Catalog) ? "Duplicate…" : "Duplicate to User Glasses…", () => _ = DuplicateAsync(row), "\uE8F4"));
        return m;
    }

    void BuildFilterMenu()
    {
        foreach (var f in Enum.GetValues<GlassFilter>())
        {
            var item = new RadioMenuFlyoutItem { Text = GlassBrowserModel.FilterTitle(f), GroupName = "filter", Tag = f };
            item.Click += (_, _) => _model.Filter = f;
            FilterMenu.Items.Add(item);
            if (f == GlassFilter.All) FilterMenu.Items.Add(new MenuFlyoutSeparator());
        }
    }

    void OnAddMenuOpening(object sender, object e) => AddToDesignItem.IsEnabled = _model.Target is not null;

    // ---- Actions ----

    void OnSearchChanged(AutoSuggestBox sender, AutoSuggestBoxTextChangedEventArgs args) => _model.Search = sender.Text ?? "";

    void OnSearchKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Down) { Table.FocusList(); e.Handled = true; }
    }

    void OnNewGlass(object sender, RoutedEventArgs e) => _ = NewGlassAsync();
    void OnNewUserCatalog(object sender, RoutedEventArgs e) => _model.NewUserCatalog();
    void OnAddLibraryFiles(object sender, RoutedEventArgs e) => _ = AddLibraryFilesAsync();
    void OnAddLibraryFolder(object sender, RoutedEventArgs e) => _ = AddLibraryFolderAsync();
    void OnAddToDesign(object sender, RoutedEventArgs e) => _ = AddToDesignAsync();

    public async Task NewGlassAsync()
    {
        if (_model.NewGlass() is { } editor) await GlassEditorDialog.ShowAsync(editor, _model, Root);
    }

    public async Task DuplicateAsync(GlassRow row)
    {
        if (await _model.DuplicateAsync(row) is { } editor) await GlassEditorDialog.ShowAsync(editor, _model, Root);
    }

    public async Task EditSelectedAsync()
    {
        if (_model.EditSelected() is { } editor) await GlassEditorDialog.ShowAsync(editor, _model, Root);
    }

    async Task DeleteSelectedAsync()
    {
        if (_model.SelectedRow is not { } row) return;
        if (await GlassDialogs.ConfirmAsync(Root, $"Delete {row.Name}?", $"The glass is removed from {row.Catalog}. Designs that use it will not find it.", "Delete"))
            _model.DeleteSelected();
    }

    public async Task ReloadAsync(CatalogInfo c)
    {
        if (await _model.ReloadAsync(c) is { } changes) await GlassDialogs.ShowChangesAsync(Root, changes);
    }

    async Task RenameAsync(string name)
    {
        if (await GlassDialogs.AskNameAsync(Root, "Rename Catalog", name, "Designs that name the catalog in a glass reference keep the old name.") is not { } newName) return;
        try
        {
            Library.RenameUserCatalog(name, newName);
            _model.Scope = newName;
        }
        catch (Exception e) when (e is GlassLibrary.LibraryException or IOException or UnauthorizedAccessException) { ShowToast(e.Message); }
    }

    async Task DeleteCatalogAsync(string name)
    {
        if (!await GlassDialogs.ConfirmAsync(Root, $"Move “{name}” to the Recycle Bin?", "Designs that use its glasses will not find them.", "Move to Recycle Bin")) return;
        try
        {
            Library.DeleteUserCatalog(name);
            _model.Scope = null;
            ShowToast($"Moved {name} to the Recycle Bin.");
        }
        catch (Exception e) when (e is GlassLibrary.LibraryException or IOException or UnauthorizedAccessException) { ShowToast(e.Message); }
    }

    async Task ExportAsync(string catalog)
    {
        if (_model.AgfExport(catalog) is not { } export) return;
        var picker = new FileSavePicker(AppWindow.Id) { SuggestedFileName = catalog, DefaultFileExtension = ".agf", SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        picker.FileTypeChoices.Add("AGF glass catalog", [".agf"]);
        if ((await picker.PickSaveFileAsync())?.Path is not { } path) return;
        try
        {
            await File.WriteAllTextAsync(path, export.Text);
            ShowToast($"Exported {catalog} to {Path.GetFileName(path)}.");
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { ShowToast($"Could not export: {e.Message}"); return; }
        if (export.Warnings.Count > 0)
            await GlassDialogs.ShowListAsync(Root, "Not Exported Exactly", "AGF cannot express everything in this catalog:", export.Warnings);
    }

    async Task AddLibraryFilesAsync()
    {
        if (await PickCatalogFilesAsync(this, multiple: true) is { Count: > 0 } paths) _model.AddToLibrary(paths);
    }

    async Task AddLibraryFolderAsync()
    {
        var picker = new FolderPicker(AppWindow.Id) { SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        if ((await picker.PickSingleFolderAsync())?.Path is { } path) _model.AddToLibrary([path]);
    }

    async Task AddToDesignAsync()
    {
        if (_model.Target is null) return;
        if (await PickCatalogFilesAsync(this, multiple: false) is [var path]) await _model.AddToDesignAsync(path);
    }

    static async Task<IReadOnlyList<string>> PickCatalogFilesAsync(Window owner, bool multiple)
    {
        var picker = new FileOpenPicker(owner.AppWindow.Id) { SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        picker.FileTypeFilter.Add(".agf");
        picker.FileTypeFilter.Add(".json");
        if (multiple) return (await picker.PickMultipleFilesAsync()).Select(r => r.Path).ToList();
        return (await picker.PickSingleFileAsync())?.Path is { } p ? [p] : [];
    }

    static void CopyName(GlassRow row)
    {
        var data = new DataPackage();
        data.SetText(row.Name);
        Clipboard.SetContent(data);
    }

    static void Reveal(string path)
    {
        try { Process.Start(new ProcessStartInfo("explorer.exe", $"/select,\"{GlassLibrary.PlainPath(path)}\"") { UseShellExecute = false }); }
        catch (Exception) { }
    }

    // ---- Undo, toast ----

    void BuildAccelerators()
    {
        void Add(VirtualKey key, VirtualKeyModifiers mods, Action action)
        {
            var a = new KeyboardAccelerator { Key = key, Modifiers = mods };
            a.Invoked += (_, e) => { action(); e.Handled = true; };
            Root.KeyboardAccelerators.Add(a);
        }
        Root.KeyboardAcceleratorPlacementMode = KeyboardAcceleratorPlacementMode.Hidden;
        // Ctrl+Z here undoes the design's changes made from this window (Use for Surface, Remove from Design).
        Add(VirtualKey.Z, VirtualKeyModifiers.Control, () => { if (!IsTextEditing()) _design?.Session.Document.Undo(); });
        Add(VirtualKey.Y, VirtualKeyModifiers.Control, () => { if (!IsTextEditing()) _design?.Session.Document.Redo(); });
        Add(VirtualKey.F, VirtualKeyModifiers.Control, () => SearchBox.Focus(FocusState.Keyboard));
        // Return on a table row uses the glass (as a double-click does).
        var enter = new KeyboardAccelerator { Key = VirtualKey.Enter };
        enter.Invoked += (_, e) =>
        {
            if (FocusManager.GetFocusedElement(Root.XamlRoot) is not ListViewItem) return;
            _model.UseSelected();
            e.Handled = true;
        };
        Root.KeyboardAccelerators.Add(enter);
    }

    bool IsTextEditing() => FocusManager.GetFocusedElement(Root.XamlRoot) is TextBox or AutoSuggestBox;

    void ShowToast(string message)
    {
        ToastText.Text = message;
        Toast.Opacity = 1;
        var timer = DispatcherQueue.CreateTimer();
        timer.Interval = TimeSpan.FromSeconds(5);
        timer.IsRepeating = false;
        timer.Tick += (_, _) => { if (ToastText.Text == message) Toast.Opacity = 0; };
        timer.Start();
    }

    // ---- Launch actions (checks and screenshots) ----

    internal GlassBrowserModel Model => _model;
    internal FrameworkElement RootElement => Root;
}
