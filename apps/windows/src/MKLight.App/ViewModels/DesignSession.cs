using System.Text.Json.Nodes;
using MKLight.App.Views;
using MKLight.App.Views.Layout;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// One open design (the Mac app's DesignSession): the document, its engine
/// workspace and the window state (selection, toasts). The document's
/// changes feed the workspace; the workspace's results feed the views.
/// </summary>
public sealed class DesignSession : ILensDataHost, IDisposable
{
    readonly IEngineClient _appEngine;

    public DesignSession(DesignDocument document, IEngineClient engine)
    {
        Document = document;
        _appEngine = engine;
        Workspace = new Workspace(engine.MakeDocumentEngine());
        Document.SystemChanged += () => Workspace.Update(Document.System);
        Document.SectionChanged += key => { if (key == "catalogs") _ = SyncProjectCatalogsAsync(); };
        Workspace.StateChanged += () => EngineDataChanged?.Invoke();
        Workspace.Layout.Changed += OnLayoutChanged;
    }

    /// <summary>
    /// Ends the design's engine session when its last window closes. Only a session of its own is
    /// disposed: an engine without per-document sessions returns itself from MakeDocumentEngine.
    /// </summary>
    public void Dispose()
    {
        Workspace.CancelAll();
        if (!ReferenceEquals(Workspace.Engine, _appEngine) && Workspace.Engine is IDisposable owned) owned.Dispose();
    }

    public DesignDocument Document { get; }
    public Workspace Workspace { get; }

    public async Task StartAsync(IEnumerable<string>? analyses = null, JsonObject? extraSettings = null)
    {
        foreach (var hook in Support.AppHooks.SessionStarting)
        {
            try { await hook(this); } catch (Exception e) { ShowToast(e.Message); }
        }
        // The project's glass catalogs are in the session before its first snapshot.
        await Workspace.LoadProjectCatalogsAsync(Document.Project.Catalogs, ProjectDirectory);
        Workspace.Update(Document.System, immediately: true);
        await Workspace.LoadEngineMetadataAsync();
        // A project's saved panes and panel arrangement (views); the defaults otherwise.
        if (analyses is null && Views.Panels.PanelPersistence.RestorePanes(this)) return;
        foreach (var a in analyses ?? DefaultPanes)
        {
            var settings = Workspace.StartingSettings(a);
            if (extraSettings is not null)
                foreach (var kv in extraSettings) settings[kv.Key] = kv.Value?.DeepClone();
            Workspace.AddPane(a, settings);
        }
        Workspace.FocusedPaneId = Workspace.Panes.FirstOrDefault()?.Id;
    }

    string? ProjectDirectory => Document.Path is { } p ? System.IO.Path.GetDirectoryName(p) : null;

    /// <summary>A catalog added to or removed from the project (or undone): the session follows.</summary>
    async Task SyncProjectCatalogsAsync()
    {
        var wanted = Document.Project.Catalogs.Select(c => c.Name).ToHashSet();
        try
        {
            var loaded = await Workspace.Engine.CatalogsAsync();
            foreach (var c in loaded.Where(c => c.InProject && !c.Library && !wanted.Contains(c.Name)))
                await Workspace.Engine.RemoveCatalogAsync(c.Name);
            var missing = Document.Project.Catalogs.Where(c => !loaded.Any(l => l.Name == c.Name && l.InProject)).ToList();
            if (missing.Count > 0) await Workspace.LoadProjectCatalogsAsync(missing, ProjectDirectory);
            else await Workspace.CatalogsDidChangeAsync();
        }
        catch (EngineException e) { ShowToast(e.Message); }
    }

    // ---- Design jobs in the status capsule ----

    readonly Dictionary<string, (string Text, Action? Open)> _jobs = [];

    /// <summary>Running design jobs by id: the status capsule shows each ("Optimizing · 479"); a click opens it.</summary>
    public IReadOnlyDictionary<string, (string Text, Action? Open)> RunningJobs => _jobs;
    public event Action? RunningJobsChanged;

    /// <summary>Show (text) or clear (null) a running job's line in the status capsule.</summary>
    public void SetJobActivity(string id, string? text, Action? open = null)
    {
        if (text is null) _jobs.Remove(id); else _jobs[id] = (text, open);
        RunningJobsChanged?.Invoke();
    }

    // ---- Inspector ----

    /// <summary>Asks the window to show an inspector page (<c>surface</c>, <c>system</c>, <c>optimize</c>, …).</summary>
    public event Action<string>? InspectorPageRequested;
    public void ShowInspectorPage(string page) => InspectorPageRequested?.Invoke(page);

    /// <summary>The panes a new window opens with.</summary>
    public static readonly string[] DefaultPanes = ["spot_diagram", "mtf"];

    // ---- Selection ----

    int? _selected;
    public int? SelectedSurface
    {
        get => _selected;
        set { if (_selected == value) return; _selected = value; SelectionChanged?.Invoke(); }
    }
    public event Action? SelectionChanged;

    // ---- Engine data for the lens table ----

    public IReadOnlyDictionary<int, IReadOnlyList<IssueItem>> IssuesBySurface => Workspace.IssuesBySurface;
    public IReadOnlyDictionary<int, double> AutoSemiDiameters { get; private set; } = new Dictionary<int, double>();
    public IReadOnlyList<string> MaterialNames => Workspace.MaterialNames;
    public event Action? EngineDataChanged;

    /// <summary>The engine's layout scene (3-D polylines, elements, mesh), read once per layout output.</summary>
    public LayoutScene? LayoutScene { get; private set; }
    /// <summary>Raised after <see cref="LayoutScene"/> and <see cref="Section"/> changed.</summary>
    public event Action? LayoutSceneChanged;

    /// <summary>The layout's section, read once per layout output.</summary>
    public SectionScene? Section { get; private set; }
    public string? LayoutMessage { get; private set; }
    AnalysisOutput? _layoutRead;

    void OnLayoutChanged()
    {
        var output = Workspace.Layout.Output;
        if (ReferenceEquals(output, _layoutRead)) return;
        // The scene is copied out of the buffers, so the previous output can go (unless a preview keeps it).
        if (_layoutRead is { } old && !Workspace.IsRetained(old)) old.Dispose();
        _layoutRead = output;
        var scene = Engine.LayoutScene.FromOutput(output);
        LayoutScene = scene;
        Section = scene is null ? null : EngineLayout.Section(scene);
        LayoutSceneChanged?.Invoke();
        AutoSemiDiameters = scene?.Surfaces.Where(s => double.IsFinite(s.SemiDiameter)).GroupBy(s => s.Index)
            .ToDictionary(g => g.Key, g => g.First().SemiDiameter) ?? new Dictionary<int, double>();
        EngineDataChanged?.Invoke();
    }

    // ---- Toasts ----

    public event Action<string>? Toast;
    public void ShowToast(string message) => Toast?.Invoke(message);
}
