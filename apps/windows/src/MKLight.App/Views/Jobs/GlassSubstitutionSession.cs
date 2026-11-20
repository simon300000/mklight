using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Glass Substitution job (the Mac app's <c>GlassSubstitutionSession</c>): which lenses' glasses may change and
/// from which candidates, the search, live progress, the ranked combinations (the selected one previewed in the
/// window) and Apply as one undoable step. The merit and continuous variables are the Optimize page's.
/// </summary>
public sealed class GlassSubstitutionSession
{
    readonly DesignSession _s;
    readonly OptimizeSession _optimize;
    AnalysisJob? _job;

    public GlassSubstitutionSession(DesignSession session, OptimizeSession optimize)
    {
        _s = session;
        _optimize = optimize;
        Settings.Configure(session.Document.System);
        Settings.UseDesignVariables = DesignVariables.Of(session.Document).Count > 0;
        session.Document.SystemChanged += () => { Settings.Configure(session.Document.System); Changed?.Invoke(); };
    }

    public DesignSession Session => _s;
    public OptimizeSession Optimize => _optimize;
    public GlassSubstitutionSettings Settings { get; } = new();
    public event Action? Changed;

    public JobPhase Phase { get; private set; } = JobPhase.Setup;
    public JobProgress Progress { get; private set; }
    public GlassProgressInfo? Live { get; private set; }
    public string? Failure { get; private set; }
    public List<double> Trace { get; } = [];
    public GlassSubstitutionResult? Result { get; private set; }
    public int? SelectedRank { get; private set; }
    public bool Applied { get; private set; }
    /// <summary>The design's glasses when the run started (with their melt offsets), for the comparison.</summary>
    public IReadOnlyDictionary<SurfaceID, string> StartGlasses { get; private set; } = new Dictionary<SurfaceID, string>();
    /// <summary>The design the run started from (the before layout, and the glasses on the map a candidate replaces).</summary>
    public OpticalSystem? StartSystem { get; private set; }
    /// <summary>Whether the candidate's glass map shows the P_g,F plane (else n_d).</summary>
    public bool MapPgf { get; set; }

    public bool IsRunning => Phase == JobPhase.Running;
    public bool CanStop => IsRunning && (_job?.CanStop ?? false);
    public GlassCombination? Selected => Result?.Ranked.FirstOrDefault(c => c.Rank == SelectedRank);

    public IReadOnlyList<GlassSlot> Slots => GlassSlot.All(_s.Document.System);

    public bool CannotRun =>
        _s.Workspace.DesignSnapshot is null || _s.Workspace.ModelErrors.Count > 0 || (Settings.Selected.Count == 0 && !Slots.Any(x => x.IsModel));

    public void NotifyChanged() => Changed?.Invoke();

    public void Start(bool fresh = false)
    {
        var system = _s.Document.System;
        var settings = Settings.Settings(system, MeritSettings.FromSection(_s.Document.Section("merit")), DesignVariables.Of(_s.Document));
        StartGlasses = GlassSlot.All(system).ToDictionary(x => x.Id, x => x.Label);
        StartSystem = system;
        EndPreview();
        var job = _s.Workspace.SubmitDesignJob(new EngineAnalysisRequest("glass_substitution", settings, new ExecutionSettings(_s.Workspace.Backend), fresh));
        if (job is null)
        {
            Phase = JobPhase.Failed;
            Failure = "Fix the design's errors first.";
            Changed?.Invoke();
            return;
        }
        _job = job;
        Live = null;
        Trace.Clear();
        Result = null;
        SelectedRank = null;
        Applied = false;
        Failure = null;
        Phase = JobPhase.Running;
        Progress = new JobProgress(0, "Preparing candidates");
        UpdateActivity();
        Changed?.Invoke();
        _ = Follow(job);
        _ = Observe(job);
    }

    async Task Follow(AnalysisJob job)
    {
        try
        {
            await foreach (var p in job.ProgressAsync())
            {
                if (!ReferenceEquals(_job, job) || Phase != JobPhase.Running) continue;
                Progress = p;
                if (GlassProgressInfo.Parse(p.Stage) is { } info)
                {
                    Live = info;
                    if (double.IsFinite(info.BestMerit) && (Trace.Count == 0 || Trace[^1] != info.BestMerit)) Trace.Add(info.BestMerit);
                }
                UpdateActivity();
                Changed?.Invoke();
            }
        }
        catch (Exception) { /* progress is advisory */ }
    }

    async Task Observe(AnalysisJob job)
    {
        try
        {
            using var output = await job.ResultAsync();
            if (!ReferenceEquals(_job, job)) return;
            Result = GlassSubstitutionResult.FromJson(output.Json);
            if (Result is null)
            {
                Phase = JobPhase.Failed;
                Failure = "The engine returned no glass combinations.";
            }
            else
            {
                if (Result.History.Count > 0) { Trace.Clear(); Trace.AddRange(Result.History); }
                Phase = JobPhase.Finished;
            }
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
        if (Result?.Ranked.FirstOrDefault() is { } best) Select(best.Rank);
        Changed?.Invoke();
    }

    void UpdateActivity()
    {
        if (Phase == JobPhase.Running)
            _s.SetJobActivity("glass_substitution", Live is { } l ? $"Substituting · {l.Combinations}" : "Substituting", () => _s.ShowInspectorPage("glass_substitution"));
        else _s.SetJobActivity("glass_substitution", null);
    }

    /// <summary>Previews a combination in the window (layout and every analysis).</summary>
    public void Select(int? rank)
    {
        SelectedRank = rank;
        if (Selected is { System: { } system } c && !Applied)
        {
            var title = $"Glass candidate {c.Rank} · merit {MeritFormat.Merit(c.Merit)}";
            if (Result is { StartingMerit: > 0 } r && double.IsFinite(c.Merit))
                title += $" ({(c.Merit <= r.StartingMerit ? "−" : "+")}{MeritFormat.Percent(Math.Abs(1 - c.Merit / r.StartingMerit), 0)})";
            _s.Workspace.Preview(system, title);
        }
        else EndPreview();
        Changed?.Invoke();
    }

    void EndPreview()
    {
        if (_s.Workspace.IsPreviewing && _s.Workspace.PreviewTitle?.StartsWith("Glass candidate", StringComparison.Ordinal) == true) _s.Workspace.Preview(null);
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

    /// <summary>Applies the previewed combination's design as one undoable step.</summary>
    public void Apply()
    {
        if (Selected?.System is not { } system || Applied) return;
        Applied = true;
        DesignVariables.ReplaceSystem(_s.Document, system, "Apply Glass Substitution");
        EndPreview();
        Changed?.Invoke();
    }

    public void Reset()
    {
        EndPreview();
        Phase = JobPhase.Setup;
        Result = null;
        Live = null;
        Trace.Clear();
        SelectedRank = null;
        Applied = false;
        Failure = null;
        Changed?.Invoke();
    }
}
