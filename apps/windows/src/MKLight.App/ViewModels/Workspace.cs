using System.Text.Json.Nodes;
using MKLight.App.Views;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// Engine snapshots and analysis panes of one open design (the Mac app's
/// Workspace). Each design value is debounced (150 ms) and frozen into an
/// engine snapshot; panes run asynchronously against the snapshot they were
/// submitted with; a newer request cancels the older job of the same pane.
/// Results are stale as soon as the snapshot or the request differs; panes
/// with automatic update re-run, the others wait for Update. Runs on the UI
/// thread (async continuations return to it).
/// </summary>
public sealed class Workspace
{
    public Workspace(IEngineClient engine)
    {
        Engine = engine;
        foreach (var p in Internal) p.Workspace = this;
    }

    public IEngineClient Engine { get; }
    public IReadOnlyList<AnalysisDescriptor> Descriptors { get; private set; } = [];
    public IReadOnlyList<string> MaterialNames { get; private set; } = [];
    public ModelSnapshot? Snapshot { get; private set; }
    public IReadOnlyList<string> ModelErrors { get; private set; } = [];
    public IReadOnlyList<IssueItem> Issues => _refusal.Count > 0 ? _refusal : _validation;
    IReadOnlyList<IssueItem> _refusal = [];
    IReadOnlyList<IssueItem> _validation = [];
    /// <summary>Validation issues with their surface index, for the lens table's badges.</summary>
    public IReadOnlyDictionary<int, IReadOnlyList<IssueItem>> IssuesBySurface { get; private set; } = new Dictionary<int, IReadOnlyList<IssueItem>>();

    public BackendPreference Backend { get; set; } = BackendPreference.Auto;

    public List<AnalysisPaneModel> Panes { get; } = [];
    public AnalysisPaneModel Layout { get; } = new("layout", id: "layout");
    public AnalysisPaneModel Summary { get; } = new("first_order", id: "summary");
    readonly AnalysisPaneModel _validationPane = new("validate", id: "validation");
    IEnumerable<AnalysisPaneModel> Internal => [_validationPane, Summary, Layout];
    IEnumerable<AnalysisPaneModel> AllPanes => Internal.Concat(Panes);

    public string? FocusedPaneId { get; set; }

    /// <summary>Raised when panes are added or removed.</summary>
    public event Action? PanesChanged;
    /// <summary>Raised when the snapshot, issues, descriptors or any pane's state change (the status capsule, sidebar dots).</summary>
    public event Action? StateChanged;

    public TimeSpan Debounce { get; set; } = TimeSpan.FromMilliseconds(150);
    OpticalSystem? _latest;
    CancellationTokenSource? _pending;

    public bool IsBusy => AllPanes.Any(p => p.IsRunning);

    public AnalysisDescriptor? Descriptor(string name) => Descriptors.FirstOrDefault(d => d.Name == name);

    /// <summary>The analyses offered in the sidebar (not the design jobs, diagnostics or raw data).</summary>
    public IEnumerable<AnalysisDescriptor> PaneAnalyses => Descriptors.Where(d =>
        d.Category is not ("optimization" or "diagnostic") && d.Name is not ("layout" or "ray_grid" or "convert_spec"));

    public async Task LoadEngineMetadataAsync()
    {
        Descriptors = await Engine.AnalysisDescriptorsAsync();
        try { MaterialNames = await Engine.MaterialNamesAsync(); } catch (EngineException) { MaterialNames = []; }
        foreach (var p in Panes.Where(p => p.Analysis == "glass_info" && p.Settings["name"] is null))
            p.Settings = StartingSettings(p.Analysis);
        StateChanged?.Invoke();
        foreach (var p in AllPanes.Where(p => p.Output is null)) RefreshPane(p);
    }

    // ---- Panes ----

    public AnalysisPaneModel AddPane(string analysis, JsonObject? settings = null, AnalysisPaneModel? after = null)
    {
        var pane = new AnalysisPaneModel(analysis, settings ?? StartingSettings(analysis), AutoUpdateDefault(analysis)) { Workspace = this };
        var i = after is null ? -1 : Panes.IndexOf(after);
        if (i >= 0) Panes.Insert(i + 1, pane); else Panes.Add(pane);
        FocusedPaneId = pane.Id;
        PanesChanged?.Invoke();
        RefreshPane(pane);
        return pane;
    }

    public void RemovePane(AnalysisPaneModel pane)
    {
        pane.Job?.Cancel();
        Panes.Remove(pane);
        if (FocusedPaneId == pane.Id) FocusedPaneId = Panes.FirstOrDefault()?.Id;
        PanesChanged?.Invoke();
    }

    public AnalysisPaneModel? FocusedPane => Panes.FirstOrDefault(p => p.Id == FocusedPaneId) ?? Panes.FirstOrDefault();

    /// <summary>Show an analysis: focus its pane if one is open, else replace the focused pane's analysis (or add one).</summary>
    public void Open(string analysis, bool inNewPane = false)
    {
        var open = Panes.FirstOrDefault(p => p.Analysis == analysis);
        if (!inNewPane && open is not null) { FocusedPaneId = open.Id; StateChanged?.Invoke(); return; }
        if (!inNewPane && FocusedPane is { } p)
        {
            Show(p, analysis, StartingSettings(analysis));
            p.AutoUpdate = AutoUpdateDefault(analysis);
            return;
        }
        AddPane(analysis, after: FocusedPane);
    }

    internal void Show(AnalysisPaneModel pane, string analysis, JsonObject? settings)
    {
        pane.Job?.Cancel();
        pane.Job = null;
        pane.Analysis = analysis;
        pane.Settings = settings ?? [];
        pane.Output = null;
        pane.Phase = PanePhase.Idle;
        pane.RaiseChanged();
        PanesChanged?.Invoke();
        RefreshPane(pane, force: true);
    }

    internal void SetSettings(AnalysisPaneModel pane, JsonObject settings)
    {
        if (JsonNode.DeepEquals(pane.Settings, settings)) return;
        pane.Settings = settings;
        pane.RaiseChanged();
        RefreshPane(pane, force: true);
    }

    /// <summary>Settings a new pane starts with (the spot diagram's three focus planes, a glass for Glass Data).</summary>
    public JsonObject StartingSettings(string analysis) => analysis switch
    {
        "spot_diagram" => new JsonObject { ["through_focus"] = new JsonObject { ["min"] = null, ["max"] = null, ["steps"] = 3 } },
        // A visible patch: 10 % of the field on axis at 256 px with the engine's automatic PSF window and pupil.
        "image_simulation" => new JsonObject
        {
            ["size"] = new JsonArray(256, 256), ["optics"] = new JsonObject { ["field_extent"] = 0.1 },
            ["source"] = new JsonObject { ["kind"] = "pattern", ["pattern"] = "siemens_star" },
        },
        "glass_info" => FirstGlass() is { } g ? new JsonObject { ["name"] = g, ["curve_samples"] = 200 } : new JsonObject { ["curve_samples"] = 200 },
        _ => [],
    };

    string? FirstGlass() =>
        _latest?.Surfaces.Select(s => s.Material).OfType<MaterialRef.CatalogGlass>().Select(c => c.Name).FirstOrDefault() ?? MaterialNames.FirstOrDefault();

    /// <summary>Long analyses start manual.</summary>
    public static bool AutoUpdateDefault(string analysis) =>
        analysis is not ("stray_light" or "ghost_irradiance" or "through_focus_mtf" or "mtf_vs_field" or "image_simulation");

    // ---- Design changes ----

    /// <summary>A new design value: debounced, then frozen into a snapshot and the panes refreshed. A design change ends a preview.</summary>
    public void Update(OpticalSystem system, bool immediately = false)
    {
        _design = system;
        if (PreviewSystem is not null)
        {
            PreviewSystem = null;
            PreviewTitle = null;
            _baseline.Clear();
            PreviewChanged?.Invoke();
        }
        Schedule(system, immediately ? TimeSpan.Zero : Debounce);
    }

    void Schedule(OpticalSystem system, TimeSpan delay)
    {
        _latest = system;
        _pending?.Cancel();
        var cts = _pending = new CancellationTokenSource();
        _ = OpenSnapshotLater(system, delay, cts.Token);
    }

    /// <summary>Take the snapshot again now (after a glass catalog change: the library hash differs).</summary>
    public void Resnapshot()
    {
        if (_design is not null) Schedule(PreviewSystem ?? _design, TimeSpan.Zero);
    }

    // ---- Preview of a candidate design (an optimisation or glass substitution result) ----

    OpticalSystem? _design;
    readonly Dictionary<string, AnalysisOutput> _baseline = [];

    /// <summary>The design itself (not a previewed candidate).</summary>
    public OpticalSystem? DesignSystem => _design;
    /// <summary>The snapshot of the design itself; design jobs always run on it, never on a preview.</summary>
    public ModelSnapshot? DesignSnapshot { get; private set; }
    /// <summary>A candidate the layout and every pane show instead of the design, not applied.</summary>
    public OpticalSystem? PreviewSystem { get; private set; }
    /// <summary>What is previewed, e.g. "Optimize: iteration 42" (shown on every pane and the layout).</summary>
    public string? PreviewTitle { get; private set; }
    public bool IsPreviewing => PreviewSystem is not null;
    /// <summary>The outputs of the design taken just before the preview started, by pane id (for before/after views).</summary>
    public IReadOnlyDictionary<string, AnalysisOutput> Baseline => _baseline;
    /// <summary>Raised when a preview starts, changes or ends.</summary>
    public event Action? PreviewChanged;

    /// <summary>Show <paramref name="candidate"/> in the layout and every pane (null ends the preview).</summary>
    public void Preview(OpticalSystem? candidate, string? title = null)
    {
        if (Equals(candidate, PreviewSystem) && title == PreviewTitle) return;
        if (candidate is not null && PreviewSystem is null)
        {
            _baseline.Clear();
            foreach (var p in AllPanes) if (p.Output is { } o && !IsStale(p)) _baseline[p.Id] = o;
        }
        if (candidate is null) _baseline.Clear();
        PreviewSystem = candidate;
        PreviewTitle = candidate is null ? null : title;
        PreviewChanged?.Invoke();
        StateChanged?.Invoke();
        if ((candidate ?? _design) is { } target) Schedule(target, TimeSpan.Zero);
    }

    /// <summary>Whether an output must be kept (a baseline of the preview).</summary>
    public bool IsRetained(AnalysisOutput output) => _baseline.ContainsValue(output);

    /// <summary>
    /// Submit a design job (optimize, glass substitution, tolerance, solves, merit evaluation) on the
    /// design's own snapshot; null while there is none (the design has errors or is not yet frozen).
    /// </summary>
    public AnalysisJob? SubmitDesignJob(EngineAnalysisRequest request) =>
        DesignSnapshot is { } snap ? Engine.Submit(request, snap) : null;

    // ---- Glass catalogs ----

    /// <summary>Problems loading the project's catalogs (a file that is not found, …).</summary>
    public IReadOnlyList<string> CatalogWarnings { get; private set; } = [];

    /// <summary>Raised after catalogs were loaded, removed or changed (the glass library hash differs).</summary>
    public event Action? CatalogsChanged;

    /// <summary>Load the project's user catalogs into the session (before the first snapshot).</summary>
    public async Task LoadProjectCatalogsAsync(IEnumerable<CatalogReference> catalogs, string? projectDirectory)
    {
        var warnings = new List<string>();
        foreach (var c in catalogs)
        {
            var loaded = false;
            string? error = null;
            foreach (var path in c.Candidates(projectDirectory))
            {
                if (!File.Exists(path)) continue;
                try { await Engine.LoadCatalogAsync(path); loaded = true; break; }
                catch (EngineException e) { error = e.Message; }
            }
            if (!loaded)
                warnings.Add(error is null ? $"Glass catalog “{c.Name}” was not found ({c.Path ?? c.RelativePath ?? "no path"})." : $"{c.Name}: {error}");
        }
        CatalogWarnings = warnings;
        await CatalogsDidChangeAsync();
    }

    /// <summary>After any catalog change: glass names again, a new snapshot (its library hash), listeners.</summary>
    public async Task CatalogsDidChangeAsync()
    {
        try { MaterialNames = await Engine.MaterialNamesAsync(); } catch (EngineException) { }
        CatalogsChanged?.Invoke();
        StateChanged?.Invoke();
        Resnapshot();
    }

    async Task OpenSnapshotLater(OpticalSystem system, TimeSpan delay, CancellationToken ct)
    {
        try
        {
            if (delay > TimeSpan.Zero) await Task.Delay(delay, ct);
        }
        catch (TaskCanceledException) { return; }
        if (ct.IsCancellationRequested) return;
        try
        {
            var snap = await Engine.OpenSnapshotAsync(ModelJson.Encode(system));
            if (ReferenceEquals(system, _design)) DesignSnapshot = snap;
            if (!ReferenceEquals(system, _latest)) return;
            Snapshot = snap;
            ModelErrors = [];
            _refusal = [];
            UpdateIssuesBySurface();
            StateChanged?.Invoke();
            Refresh(forced: false);
        }
        catch (EngineException e)
        {
            if (!ReferenceEquals(system, _latest)) return;
            var messages = e.Issues.Count > 0 ? e.Issues : [e.Message];
            // The model mirror's structural problems name their surface.
            var structural = system.StructuralIssues().Select(i => new IssueItem("error", i.Message, i.Surface is int s ? $"Surface {s}" : null, i.Surface));
            ModelErrors = messages;
            _refusal = [.. messages.Select(m => new IssueItem("error", m, null)), .. structural];
            UpdateIssuesBySurface();
            StateChanged?.Invoke();
            foreach (var p in AllPanes) p.RaiseChanged();
        }
    }

    // ---- Running ----

    EngineAnalysisRequest Request(AnalysisPaneModel pane)
    {
        var settings = pane.Settings.ToDictionary(kv => kv.Key, kv => kv.Value?.DeepClone());
        if (Descriptor(pane.Analysis) is { SettingsFields.Count: > 0 } d) settings = d.AcceptedSettings(settings);
        return new EngineAnalysisRequest(pane.Analysis, settings, new ExecutionSettings(Backend));
    }

    public bool IsStale(AnalysisPaneModel pane)
    {
        if (pane.Output is null) return false;
        if (ModelErrors.Count > 0) return true;
        return !pane.Output.IsCurrent(Snapshot, Request(pane));
    }

    void Refresh(bool forced)
    {
        foreach (var p in AllPanes) RefreshPane(p, forced);
    }

    void RefreshPane(AnalysisPaneModel pane, bool force = false)
    {
        if (Snapshot is null || ModelErrors.Count > 0) return;
        if (!pane.AutoUpdate && pane.Output is not null && !force)
        {
            pane.RaiseChanged(); // now stale
            return;
        }
        Run(pane, force: false);
    }

    public void RerunAll() { foreach (var p in AllPanes) Run(p, force: true, fresh: true); }
    public void CancelAll() { foreach (var p in AllPanes) p.Job?.Cancel(); }

    public void Run(AnalysisPaneModel pane, bool force, bool fresh = false)
    {
        if (Snapshot is not { } snapshot || ModelErrors.Count > 0) return;
        var request = Request(pane);
        if (!force)
        {
            if (pane.Output is { } o && o.IsCurrent(snapshot, request)) return;
            if (pane.IsRunning && Equals(pane.LastRequest, request) && pane.Job?.SnapshotHash == snapshot.Hash) return;
        }
        if (Descriptor(pane.Analysis) is { Available: false } d)
        {
            pane.Job?.Cancel();
            pane.Job = null; // the cancelled job's outcome must not replace this state
            SetPhase(pane, PanePhase.Unavailable, d.UnavailableReason ?? "Not available in this engine build.");
            return;
        }
        if (Descriptors.Count > 0 && Descriptor(pane.Analysis) is null)
        {
            SetPhase(pane, PanePhase.Unavailable, $"This engine has no analysis named “{pane.Analysis}”.");
            return;
        }
        pane.Job?.Cancel();
        var job = Engine.Submit(fresh ? request.Freshened() : request, snapshot);
        pane.Job = job;
        pane.LastRequest = request;
        pane.Progress = new JobProgress(0, "Queued");
        SetPhase(pane, PanePhase.Running, null);
        _ = Observe(job, pane);
    }

    void SetPhase(AnalysisPaneModel pane, PanePhase phase, string? message)
    {
        pane.Phase = phase;
        pane.Message = message;
        pane.RaiseChanged();
        StateChanged?.Invoke();
    }

    async Task Observe(AnalysisJob job, AnalysisPaneModel pane)
    {
        _ = FollowProgress(job, pane);
        AnalysisOutput? output = null;
        EngineException? error = null;
        try { output = await job.ResultAsync(); }
        catch (EngineException e) { error = e; }
        catch (Exception e) { error = EngineException.Failed(e.Message); }
        if (!ReferenceEquals(pane.Job, job)) { output?.Dispose(); return; }
        pane.Job = null;
        if (output is not null)
        {
            // The previous output's buffers are freed when its last reference goes (views may still draw it).
            pane.Output = output;
            if (ReferenceEquals(pane, _validationPane)) SetValidation(output.Json);
            SetPhase(pane, PanePhase.Finished, null);
        }
        else if (error!.Kind == EngineErrorKind.Cancelled) SetPhase(pane, PanePhase.Cancelled, null);
        else if (error.Kind == EngineErrorKind.Unsupported) SetPhase(pane, PanePhase.Unavailable, error.Message);
        else SetPhase(pane, PanePhase.Failed, error.Message);
    }

    async Task FollowProgress(AnalysisJob job, AnalysisPaneModel pane)
    {
        try
        {
            await foreach (var p in job.ProgressAsync())
            {
                if (!ReferenceEquals(pane.Job, job) || !pane.IsRunning) continue;
                pane.Progress = p;
                pane.RaiseChanged();
            }
        }
        catch (Exception) { /* progress is advisory */ }
    }

    void SetValidation(JsonNode? json)
    {
        var list = new List<IssueItem>();
        foreach (var i in json?["issues"] as JsonArray ?? [])
        {
            var surface = i?["surface"].IntOrNull() is long s ? (int)s : (int?)null;
            list.Add(new IssueItem(i?["severity"].StringOrNull() ?? "warning", i?["message"].StringOrNull() ?? "",
                surface is int k ? $"Surface {k}" : null, surface));
        }
        _validation = list;
        UpdateIssuesBySurface();
    }

    void UpdateIssuesBySurface() =>
        IssuesBySurface = Issues.Where(i => i.Surface is not null).GroupBy(i => i.Surface!.Value)
            .ToDictionary(g => g.Key, g => (IReadOnlyList<IssueItem>)g.ToList());
}
