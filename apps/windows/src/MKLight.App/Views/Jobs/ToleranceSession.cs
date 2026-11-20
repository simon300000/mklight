using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Tolerance job (the Mac app's <c>ToleranceSession</c>): the design's tolerance set (the project's
/// <c>tolerances</c> section, every edit one undoable step), the run with its timing, Stop (keep what completed) and
/// Cancel, the result, and previews of a build (a trial or a sensitivity case) rebuilt by the engine
/// (<c>preview_case</c> with the run's own settings and system), shown read-only in the window.
/// </summary>
public sealed class ToleranceSession
{
    readonly DesignSession _s;
    readonly OptimizeSession _optimize;
    AnalysisJob? _job;
    AnalysisJob? _previewJob;
    IReadOnlyDictionary<SurfaceID, double> _abbe = new Dictionary<SurfaceID, double>();

    public ToleranceSession(DesignSession session, OptimizeSession optimize)
    {
        _s = session;
        _optimize = optimize;
        Settings = Load();
        session.Document.Changed += () =>
        {
            var next = Load();
            if (!next.SameAs(Settings)) Settings = ToleranceSettings.KeepingIds(next, Settings);
            Changed?.Invoke();
        };
        session.Document.SystemChanged += () => _ = LoadAbbeNumbersAsync();
        session.EngineDataChanged += () => Changed?.Invoke();
        _ = LoadAbbeNumbersAsync();
    }

    public DesignSession Session => _s;
    public event Action? Changed;

    public ToleranceSettings Settings { get; private set; }
    public JobPhase Phase { get; private set; } = JobPhase.Setup;
    public JobProgress Progress { get; private set; }
    public string? Failure { get; private set; }
    public ToleranceResult? Result { get; private set; }
    /// <summary>The rows of the last run, in the engine's operand order.</summary>
    public IReadOnlyList<ToleranceRow> ResultRows { get; private set; } = [];
    public OpticalSystem? RunSystem { get; private set; }
    Dictionary<string, JsonNode?>? _runSettings;
    public IReadOnlyDictionary<string, JsonNode?>? RunSettings => _runSettings;
    public DateTime? StartedAt { get; private set; }
    DateTime? _monteCarloStartedAt;
    double? _lastSecondsPerTrial;
    public TolerancePreviewCase? Previewed { get; private set; }
    /// <summary>The histogram bar whose trials the results list (<c>null</c>: the worst trials).</summary>
    public int? SelectedBin { get; private set; }
    /// <summary>The keys of the tolerance groups collapsed in the table (<c>L1</c>, <c>Air after L1</c>).</summary>
    public HashSet<string> CollapsedGroups { get; } = [];
    public TolerancePreview? Preview { get; private set; }
    public string? PreviewError { get; private set; }

    public bool IsRunning => Phase == JobPhase.Running;
    public bool CanStop => IsRunning && (_job?.CanStop ?? false);
    public MeritSettings Merit => MeritSettings.FromSection(_s.Document.Section("merit"));

    ToleranceSettings Load()
    {
        var section = _s.Document.Section("tolerances");
        return section.IsNull ? ToleranceSettings.Defaults(_s.Document.System) : ToleranceSettings.FromSection(section.Node, _s.Document.System);
    }

    /// <summary>The design with its clear semi-diameters (the layout's) and the V_d of its catalogue glasses.</summary>
    public ToleranceContext Context()
    {
        var system = _s.Document.System;
        var sd = new Dictionary<SurfaceID, double>();
        foreach (var (i, d) in _s.AutoSemiDiameters) if (i < system.Surfaces.Count) sd[system.Surfaces[i].Id] = d;
        return Settings.Context(new ToleranceContext(system) { SemiDiameters = sd, AbbeNumbers = _abbe });
    }

    async Task LoadAbbeNumbersAsync()
    {
        try
        {
            var rows = await _s.Workspace.Engine.GlassRowsAsync(null);
            var system = _s.Document.System;
            var map = new Dictionary<SurfaceID, double>();
            foreach (var sf in system.Surfaces)
            {
                if (sf.Material is not MaterialRef.CatalogGlass g) continue;
                var row = rows.FirstOrDefault(r => r.Name == g.Name && (g.Catalog is null || r.Catalog == g.Catalog));
                if (row?.Vd is { } vd) map[sf.Id] = vd + g.Offset.DvD;
            }
            _abbe = map;
            Changed?.Invoke();
        }
        catch (EngineException) { }
    }

    public void NotifyChanged() => Changed?.Invoke();

    /// <summary>Applies an edit to the tolerance set as one undoable step (saved in the <c>tolerances</c> section).</summary>
    public void Edit(string actionName, Action<ToleranceSettings> change)
    {
        var next = Settings.Clone();
        change(next);
        if (next.SameAs(Settings)) { Settings = next; Changed?.Invoke(); return; }
        Settings = next;
        _s.Document.EditSection("tolerances", new JsonTree(next.Section(Context(), Merit)), actionName);
        Changed?.Invoke();
    }

    /// <summary>Why the run cannot start, if it cannot.</summary>
    public string? RunProblem()
    {
        if (_s.Workspace.DesignSnapshot is null || _s.Workspace.ModelErrors.Count > 0) return "Fix the design's errors first.";
        var blocking = Settings.Issues(Context()).Count;
        if (blocking > 0) return blocking == 1 ? "Fix the tolerance marked in red first." : $"Fix the {blocking} tolerances marked in red first.";
        if (Settings.Rows.Count == 0) return "Add a tolerance first.";
        if (Settings.Mode != ToleranceMode.Sensitivity && Settings.Trials < 2) return "Run at least 2 Monte Carlo trials.";
        return null;
    }

    public void Start(bool fresh = false)
    {
        var merit = Merit;
        var (settings, ids) = Settings.JobSettings(Context(), merit);
        var system = _s.Document.System;
        EndPreview();
        var job = _s.Workspace.SubmitDesignJob(new EngineAnalysisRequest("tolerance", settings, new ExecutionSettings(_s.Workspace.Backend), fresh));
        if (job is null)
        {
            Phase = JobPhase.Failed;
            Failure = "Fix the design's errors first.";
            Changed?.Invoke();
            return;
        }
        _job = job;
        ResultRows = ids.Select(id => Settings.Rows.First(r => r.Id == id)).Select(r => r.Clone()).ToList();
        RunSystem = system;
        _runSettings = settings;
        _display = Settings.Criterion.Display(merit);
        Result = null;
        Failure = null;
        StartedAt = DateTime.UtcNow;
        _monteCarloStartedAt = null;
        Phase = JobPhase.Running;
        Progress = new JobProgress(0, "Queued");
        UpdateActivity();
        Changed?.Invoke();
        _ = Follow(job);
        _ = Observe(job, system);
    }

    CriterionDisplay _display;

    async Task Follow(AnalysisJob job)
    {
        try
        {
            await foreach (var p in job.ProgressAsync())
            {
                if (!ReferenceEquals(_job, job) || Phase != JobPhase.Running) continue;
                if (_monteCarloStartedAt is null && p.Stage.Contains("monte carlo", StringComparison.OrdinalIgnoreCase)) _monteCarloStartedAt = DateTime.UtcNow;
                Progress = p;
                UpdateActivity();
                Changed?.Invoke();
            }
        }
        catch (Exception) { /* progress is advisory */ }
    }

    async Task Observe(AnalysisJob job, OpticalSystem system)
    {
        try
        {
            using var output = await job.ResultAsync();
            if (!ReferenceEquals(_job, job)) return;
            Result = new ToleranceResult(output.Json, system, _display);
            SelectedBin = null;
            if (Result.SecondsPerTrial is > 0 and { } t) _lastSecondsPerTrial = t;
            Phase = JobPhase.Finished;
        }
        catch (EngineException e) when (e.Kind == EngineErrorKind.Cancelled)
        {
            if (!ReferenceEquals(_job, job)) return;
            Phase = JobPhase.Setup;
        }
        catch (EngineException e)
        {
            if (!ReferenceEquals(_job, job)) return;
            Phase = JobPhase.Failed;
            Failure = e.Message;
        }
        _job = null;
        UpdateActivity();
        Changed?.Invoke();
    }

    void UpdateActivity()
    {
        if (Phase == JobPhase.Running)
            _s.SetJobActivity("tolerance", "Tolerancing · " + MeritFormat.Percent(Progress.Fraction, 0), () => _s.ShowInspectorPage("tolerance"));
        else _s.SetJobActivity("tolerance", null);
    }

    /// <summary>Trials done, time per trial and time left, from the progress.</summary>
    public ToleranceTiming Timing()
    {
        var now = DateTime.UtcNow;
        return ToleranceTiming.Estimate(StartedAt is { } s ? (now - s).TotalSeconds : 0, _monteCarloStartedAt is { } m ? (now - m).TotalSeconds : null,
            Progress.Fraction, Settings.Mode, Settings.Trials, ResultRows.Count, _lastSecondsPerTrial);
    }

    public void Stop() => _job?.Stop();

    public void Cancel()
    {
        if (_job is null) return;
        _job.Cancel();
        _job = null;
        Phase = JobPhase.Setup;
        UpdateActivity();
        Changed?.Invoke();
    }

    /// <summary>Back to the editor (the result is kept for Show the last results).</summary>
    public void EditSettings()
    {
        EndPreview();
        Phase = JobPhase.Setup;
        Changed?.Invoke();
    }

    public void ShowResult()
    {
        if (Result is null) return;
        Phase = JobPhase.Finished;
        Changed?.Invoke();
    }

    // ---- Table and histogram state ----

    /// <summary>Lists the trials of histogram bar <paramref name="bin"/> (<c>null</c>: the worst trials).</summary>
    public void SelectBin(int? bin)
    {
        if (SelectedBin == bin) return;
        SelectedBin = bin;
        Changed?.Invoke();
    }

    /// <summary>Collapses or expands a group of the tolerances table.</summary>
    public void ToggleGroup(string key)
    {
        if (!CollapsedGroups.Remove(key)) CollapsedGroups.Add(key);
        Changed?.Invoke();
    }

    // ---- Export ----

    OpticalSystem ReportSystem => RunSystem ?? _s.Document.System;

    /// <summary>The results as text for the clipboard.</summary>
    public string? SummaryText() => Result is { } r ? ToleranceReport.Summary(r, ResultRows, ReportSystem, _s.Document.System.Name) : null;

    /// <summary>The sensitivity table as CSV, if the run had one.</summary>
    public string? SensitivityCsv() => Result is { HasSensitivity: true } r ? ToleranceReport.SensitivityCsv(r, ResultRows, ReportSystem) : null;

    /// <summary>The Monte-Carlo trials as CSV (with the recorded compensators and perturbations), if the run had them.</summary>
    public string? TrialsCsv() => Result is { HasMonteCarlo: true } r ? ToleranceReport.TrialsCsv(r, ResultRows, ReportSystem) : null;

    // ---- Previews ----

    /// <summary>Rebuilds a build of the last run in the engine and previews it in the window (null ends the preview).</summary>
    public async Task PreviewAsync(TolerancePreviewCase? c)
    {
        _previewJob?.Cancel();
        _previewJob = null;
        if (c is null || _runSettings is null || RunSystem is null) { EndPreview(); return; }
        Previewed = c;
        Preview = null;
        PreviewError = null;
        // A previewed trial shows its bar's trials.
        if (c is TolerancePreviewCase.Trial t && Result is { } res)
        {
            var bins = ToleranceHistogram.Bins(res);
            if (ToleranceHistogram.BinOfTrial(res, bins, t.K) is { } b && SelectedBin != b) SelectedBin = b;
        }
        Changed?.Invoke();
        var settings = new Dictionary<string, JsonNode?>(_runSettings.Select(kv => KeyValuePair.Create(kv.Key, kv.Value?.DeepClone())))
        {
            ["preview_case"] = c.ToJson(), ["record_trials"] = 0,
        };
        try
        {
            var engine = _s.Workspace.Engine;
            var snapshot = await engine.OpenSnapshotAsync(ModelJson.Encode(RunSystem));
            if (!Equals(Previewed, c)) return;
            var job = engine.Submit(new EngineAnalysisRequest("tolerance", settings, new ExecutionSettings(_s.Workspace.Backend), viewKey: "tolerance-preview"), snapshot);
            _previewJob = job;
            using var output = await job.ResultAsync();
            if (!Equals(Previewed, c)) return;
            if (TolerancePreview.FromJson(output.Json?["preview"], c) is not { } p)
            {
                PreviewError = "The engine could not rebuild this build: no ray reached the image.";
                _s.Workspace.Preview(null);
            }
            else
            {
                Preview = p;
                _s.Workspace.Preview(p.System, c.Title(Result, RunSystem, ResultRows));
            }
        }
        catch (EngineException e) when (e.Kind == EngineErrorKind.Cancelled) { return; }
        catch (EngineException e)
        {
            if (!Equals(Previewed, c)) return;
            PreviewError = e.Message;
        }
        Changed?.Invoke();
    }

    public void EndPreview()
    {
        _previewJob?.Cancel();
        _previewJob = null;
        var was = Previewed is not null;
        Previewed = null;
        Preview = null;
        PreviewError = null;
        if (was || _s.Workspace.PreviewTitle?.StartsWith("Tolerance", StringComparison.Ordinal) == true) _s.Workspace.Preview(null);
        Changed?.Invoke();
    }
}
