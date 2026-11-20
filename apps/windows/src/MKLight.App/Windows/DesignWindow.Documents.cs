using Microsoft.UI.Dispatching;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.Windows.Storage.Pickers;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;
using Windows.ApplicationModel.DataTransfer;
using Windows.Graphics;
using Windows.Storage;

namespace MKLight.App.Windows;

/// <summary>
/// The design window's document commands: Revert to Saved, Duplicate, Rename, Replace with Sample,
/// autosave, dropped files and the import report. The menu bar calls these.
/// <para>
/// A design keeps no file handle: Revert opens the saved file in a new window at the same place and
/// closes this one (<see cref="ReplacedBy"/>); Rename moves the file and saves the design to its new
/// path. Autosave writes the document's whole project (system and every section) every
/// <see cref="Autosaves.Interval"/> while it has unsaved changes, and keeps it when the window closes
/// with unsaved changes it was not asked about; a save or a close that answered the question removes it.
/// </para>
/// </summary>
public sealed partial class DesignWindow
{
    readonly Guid _autosaveId = Guid.NewGuid();
    /// <summary>A recovered untitled design keeps writing to the autosave it came from.</summary>
    string? _untitledAutosave;
    readonly HashSet<string> _autosavesWritten = new(StringComparer.OrdinalIgnoreCase);
    DateTime? _changedSince;
    DispatcherQueueTimer? _autosaveTimer;
    /// <summary>The lens file this design was imported from, while it has not been saved as a project.</summary>
    string? _importSource;

    /// <summary>Called once from the constructor (drag and drop, the autosave timer).</summary>
    void InitDocuments()
    {
        FileDrop.Attach(Root);
        _session.Document.Changed += NoteChange;
        _session.Document.SectionChanged += _ => NoteChange();
        _autosaveTimer = DispatcherQueue.CreateTimer();
        _autosaveTimer.Interval = TimeSpan.FromSeconds(5);
        _autosaveTimer.Tick += (_, _) => AutosaveTick();
        _autosaveTimer.Start();
        // Subscribed before the constructor's own Closed handler, so the engine session is still there.
        Closed += (_, _) => OnDocumentClosed();
    }

    /// <summary>Revert to Saved is possible: a saved design with changes.</summary>
    public bool CanRevert => _session.Document.Path is not null && _session.Document.IsDirty;

    /// <summary>Rename is possible: the design has a file.</summary>
    public bool CanRename => _session.Document.Path is not null;

    /// <summary>The window that took this one's place (Revert to Saved), for scripts that go on with it.</summary>
    public DesignWindow? ReplacedBy { get; private set; }

    // ---- Revert, Duplicate, Rename ----

    /// <summary>File ▸ Revert to Saved: after confirming, the saved file opens in this window's place.</summary>
    public async Task RevertAsync()
    {
        if (!CanRevert || _session.Document.Path is not { } path) return;
        if (!App.Options.IsScripted)
        {
            var dialog = new ContentDialog
            {
                Title = $"Revert “{_session.Document.DisplayName}” to the saved version?",
                Content = "Your changes since the last save are lost.",
                PrimaryButtonText = "Revert", CloseButtonText = "Cancel", DefaultButton = ContentDialogButton.Close,
                XamlRoot = Root.XamlRoot, RequestedTheme = Root.ActualTheme,
            };
            if (await ShowDialogAsync(dialog) != ContentDialogResult.Primary) return;
        }
        DesignDocument saved;
        try { saved = DesignDocument.Open(path); }
        catch (Exception e) { _session.ShowToast($"Could not read {Path.GetFileName(path)}: {e.Message}"); return; }
        var window = OpenBeside(saved, offset: 0);
        ReplacedBy = window;
        DeleteAutosaves();
        _closeConfirmed = true; // the changes were given up: no save question
        Close();
    }

    /// <summary>File ▸ Duplicate: a new untitled design with this one's project (the system and every section).</summary>
    public Task DuplicateAsync()
    {
        _panels.SaveNow(); // the panel arrangement goes with the copy
        var duplicate = new DesignDocument(_session.Document.Project);
        duplicate.MarkChanged(); // never saved: closing it asks, and it autosaves
        var copy = OpenBeside(duplicate, offset: 28);
        AppHooks.ScreenshotTarget = copy;
        return Task.CompletedTask;
    }

    /// <summary>File ▸ Rename…: a new name for the design's file; the file moves and the design follows.</summary>
    public async Task RenameAsync()
    {
        if (_session.Document.Path is not { } current) return;
        var dirty = _session.Document.IsDirty;
        var box = new TextBox { Text = Path.GetFileNameWithoutExtension(current), Header = "Name", MinWidth = 340 };
        var problem = new TextBlock { Style = (Style)Application.Current.Resources["CaptionText"], TextWrapping = TextWrapping.Wrap, Visibility = Visibility.Collapsed };
        ThemeLookup.Follow(problem, () => problem.Foreground = ThemeLookup.Brush(problem, "ErrorBrush"));
        var content = new StackPanel { Spacing = 8, Children = { box, problem } };
        if (dirty)
            content.Children.Add(new TextBlock
            {
                Text = "The design has unsaved changes: they are saved with the new name.",
                Style = (Style)Application.Current.Resources["SecondaryText"], TextWrapping = TextWrapping.Wrap,
            });
        var dialog = new ContentDialog
        {
            Title = "Rename", Content = content,
            PrimaryButtonText = dirty ? "Save and Rename" : "Rename", CloseButtonText = "Cancel", DefaultButton = ContentDialogButton.Primary,
            XamlRoot = Root.XamlRoot, RequestedTheme = Root.ActualTheme,
        };
        void Check()
        {
            var p = DocumentFiles.RenameProblem(current, box.Text);
            problem.Text = p ?? "";
            problem.Visibility = p is null || box.Text.Trim().Length == 0 ? Visibility.Collapsed : Visibility.Visible;
            dialog.IsPrimaryButtonEnabled = p is null;
        }
        box.TextChanged += (_, _) => Check();
        box.Loaded += (_, _) => box.SelectAll();
        Check();
        if (await ShowDialogAsync(dialog) != ContentDialogResult.Primary) return;
        await RenameToAsync(box.Text);
    }

    /// <summary>Moves the design's file to <paramref name="name"/> in its folder and saves the design there.</summary>
    public async Task<bool> RenameToAsync(string name)
    {
        if (_session.Document.Path is not { } current) return false;
        if (DocumentFiles.RenameProblem(current, name) is { } problem) { _session.ShowToast(problem); return false; }
        var target = DocumentFiles.RenamedPath(current, name);
        if (string.Equals(target, current, StringComparison.Ordinal)) return true;
        try { File.Move(current, target); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { _session.ShowToast($"Could not rename: {e.Message}"); return false; }
        // The design has no path of its own to change: saving to the new path moves it there (DesignDocument.Save).
        if (!await SaveAsync(false, target))
        {
            try { File.Move(target, current); } catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
            return false;
        }
        Autosaves.Delete(current + Autosaves.Suffix);
        _session.ShowToast($"Renamed to {Path.GetFileName(target)}");
        return true;
    }

    /// <summary>Opens <paramref name="document"/> in a new window placed like this one (shifted by <paramref name="offset"/> px).</summary>
    DesignWindow OpenBeside(DesignDocument document, int offset)
    {
        var maximized = AppWindow.Presenter is OverlappedPresenter { State: OverlappedPresenterState.Maximized };
        var (position, size) = (AppWindow.Position, AppWindow.Size);
        var window = App.OpenDocument(document);
        if (maximized && window.AppWindow.Presenter is OverlappedPresenter p) p.Maximize();
        else
        {
            var shift = (int)(offset * (Root.XamlRoot?.RasterizationScale ?? 1));
            // A launch option's size (which may exceed the display) is already the new window's.
            if (App.Options.Size is null) window.AppWindow.MoveAndResize(new RectInt32(position.X + shift, position.Y + shift, size.Width, size.Height));
            else window.AppWindow.Move(new PointInt32(position.X + shift, position.Y + shift));
        }
        window.IsSidebarVisible = IsSidebarVisible;
        return window;
    }

    // ---- Replace with Sample ----

    /// <summary>Lens ▸ Replace with Sample: the sample's system replaces the design's (undoable).</summary>
    public void ReplaceWithSample(string sampleId)
    {
        if (SampleSystems.Get(sampleId) is not { } system) { _session.ShowToast($"No sample named “{sampleId}”."); return; }
        _session.Document.Replace(system, "Replace with Sample");
    }

    // ---- Autosave ----

    string SupportDirectory => AppSettings.SupportDirectory;

    /// <summary>Where this design autosaves now.</summary>
    public string AutosavePath => _session.Document.Path is null && _untitledAutosave is { } u
        ? u : Autosaves.PathFor(_session.Document.Path, _autosaveId, SupportDirectory);

    /// <summary>A recovered design goes on with the autosave it came from (and removes it once saved).</summary>
    internal void AdoptAutosave(string path)
    {
        if (_session.Document.Path is null) _untitledAutosave = path;
        _autosavesWritten.Add(path);
        Autosaves.Hold(path);
    }

    void NoteChange() => _changedSince ??= DateTime.UtcNow;

    void AutosaveTick()
    {
        var doc = _session.Document;
        // Autosaves that no longer apply: all once the design is saved, an old one after a rename or a first save.
        foreach (var p in _autosavesWritten.Where(p => !doc.IsDirty || !string.Equals(p, AutosavePath, StringComparison.OrdinalIgnoreCase)).ToList())
        {
            Autosaves.Delete(p);
            _autosavesWritten.Remove(p);
        }
        if (!doc.IsDirty) { _changedSince = null; return; }
        if (_changedSince is { } since && DateTime.UtcNow - since >= Autosaves.Interval) AutosaveNow();
    }

    /// <summary>Writes the design to its autosave now; returns the path (null when it could not).</summary>
    public string? AutosaveNow()
    {
        var path = AutosavePath;
        try
        {
            _panels.SaveNow();
            Autosaves.Hold(path); // before the file appears, so no other launch offers it
            if (App.Engine is not FFIEngine engine) throw new InvalidOperationException("no engine to autosave with");
            Autosaves.Write(engine, _session.Document.Project, path, _session.Document.Path);
            _autosavesWritten.Add(path);
            _changedSince = null;
            return path;
        }
        catch (Exception e)
        {
            App.Log($"autosave {path}: {e.Message}");
            return null;
        }
    }

    void DeleteAutosaves()
    {
        foreach (var p in _autosavesWritten) Autosaves.Delete(p);
        _autosavesWritten.Clear();
    }

    /// <summary>
    /// Changes nobody was asked about (the window closed without the save question) stay recoverable in
    /// the autosave; a clean close, or one that answered the question, removes it.
    /// </summary>
    void OnDocumentClosed()
    {
        _autosaveTimer?.Stop();
        if (_session.Document.IsDirty && !_closeConfirmed)
        {
            if (AutosaveNow() is { } kept) Autosaves.Release(kept); // left for the next launch to offer
        }
        else DeleteAutosaves();
    }

    // ---- Import report ----

    /// <summary>
    /// What an import left out: the report dialog, or a short note when everything mapped.
    /// <paramref name="source"/> is the lens file, which Load Glass Catalog… imports again.
    /// </summary>
    public void PresentImportReport(ImportReport report, string? source)
    {
        _importSource = source;
        if (report.Warnings.Count == 0) { _session.ShowToast($"Imported {report.FileName}: {report.Summary}."); return; }
        _ = ShowImportReportAsync(report);
    }

    async Task ShowImportReportAsync(ImportReport report)
    {
        if (Root.XamlRoot is null)
        {
            var loaded = new TaskCompletionSource();
            Root.Loaded += (_, _) => loaded.TrySetResult();
            await loaded.Task;
        }
        var canLoad = _importSource is not null && _session.Document.Path is null;
        if (_reportDialog is { } old) { old.Hide(); await Task.Delay(250); } // an import again replaces the report it came from
        var dialog = _reportDialog = ImportReportDialog.Create(Root, report, canLoad);
        var answer = await ShowDialogAsync(dialog);
        if (_reportDialog == dialog) _reportDialog = null;
        if (answer == ContentDialogResult.Secondary) await LoadCatalogAndImportAgainAsync();
    }

    ContentDialog? _reportDialog;

    /// <summary>Load Glass Catalog…: a catalog file goes into the design (undoable), then the lens file is imported again with it.</summary>
    async Task LoadCatalogAndImportAgainAsync()
    {
        var picker = new FileOpenPicker(AppWindow.Id) { SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        picker.FileTypeFilter.Add(".agf");
        picker.FileTypeFilter.Add(".json");
        if ((await picker.PickSingleFileAsync())?.Path is not { } path) return;
        await LoadCatalogAndImportAgainAsync(path);
    }

    internal async Task LoadCatalogAndImportAgainAsync(string catalogPath)
    {
        var ws = _session.Workspace;
        var doc = _session.Document;
        try
        {
            var info = await ws.Engine.LoadCatalogAsync(catalogPath);
            var format = Path.GetExtension(catalogPath).Equals(".json", StringComparison.OrdinalIgnoreCase) ? "json" : "agf";
            if (!doc.EditProject("Add Glass Catalog", p => p.AddCatalog(new CatalogReference(info.Name, catalogPath, null, format))))
                await ws.CatalogsDidChangeAsync();
            _session.ShowToast($"Loaded glass catalog {info.Name} ({info.GlassCount} glasses).");
        }
        catch (EngineException e) { _session.ShowToast($"Glass catalog: {e.Message}"); return; }
        if (_importSource is not { } source || doc.Path is not null) return;
        try
        {
            var imported = await ws.Engine.ImportPrescriptionAsync(source);
            var system = App.ImportedSystem(imported, source);
            doc.Edit("Import Lens File Again", _ => system);
            ws.Resnapshot(); // the import set the session's system itself
            PresentImportReport(ImportReport.From(imported, Path.GetFileName(source), system.Surfaces.Count), source);
        }
        catch (Exception e) when (e is EngineException or IOException) { _session.ShowToast($"Import: {e.Message}"); }
    }

    /// <summary>One dialog at a time per window: a second waits for none and is skipped.</summary>
    static async Task<ContentDialogResult> ShowDialogAsync(ContentDialog dialog)
    {
        try { return await dialog.ShowAsync(); }
        catch (Exception e) { App.Log($"dialog: {e.Message}"); return ContentDialogResult.None; }
    }
}

/// <summary>Projects and lens files dropped on a window open as designs (each in its own window).</summary>
static class FileDrop
{
    public static void Attach(Panel target)
    {
        target.AllowDrop = true;
        // Only a filled element takes part in hit testing: a transparent fill lets drops land on bare chrome.
        target.Background ??= new Microsoft.UI.Xaml.Media.SolidColorBrush(Microsoft.UI.Colors.Transparent);
        target.DragOver += async (_, e) =>
        {
            if (!e.DataView.Contains(StandardDataFormats.StorageItems)) return;
            var deferral = e.GetDeferral();
            try
            {
                var items = await e.DataView.GetStorageItemsAsync();
                if (!items.Any(i => i is StorageFile && DocumentFiles.CanOpen(i.Path))) return;
                e.AcceptedOperation = DataPackageOperation.Copy;
                e.DragUIOverride.Caption = "Open";
                e.DragUIOverride.IsCaptionVisible = true;
            }
            catch (Exception) { /* not a file list after all */ }
            finally { deferral.Complete(); }
        };
        target.Drop += async (_, e) =>
        {
            if (!e.DataView.Contains(StandardDataFormats.StorageItems)) return;
            List<string> paths;
            var deferral = e.GetDeferral();
            try { paths = (await e.DataView.GetStorageItemsAsync()).OfType<StorageFile>().Select(f => f.Path).Where(DocumentFiles.CanOpen).ToList(); }
            catch (Exception) { return; }
            finally { deferral.Complete(); }
            foreach (var path in paths) await App.OpenFileAsync(path);
        };
    }
}
