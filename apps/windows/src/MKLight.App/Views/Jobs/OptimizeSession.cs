using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>The design jobs of one design (shared by its windows; the pages and the launch actions find them here).</summary>
public sealed class DesignJobs
{
    static readonly ConditionalWeakTable<DesignSession, DesignJobs> Table = [];

    DesignJobs(DesignSession s)
    {
        Optimize = new OptimizeSession(s);
        Solves = new SolvesSession(s);
        Glass = new GlassSubstitutionSession(s, Optimize);
        Tolerance = new ToleranceSession(s, Optimize);
    }

    public static DesignJobs Of(DesignSession s) => Table.GetValue(s, x => new DesignJobs(x));

    public OptimizeSession Optimize { get; }
    public SolvesSession Solves { get; }
    public GlassSubstitutionSession Glass { get; }
    public ToleranceSession Tolerance { get; }

    /// <summary>Ends every running job (the design's window closed).</summary>
    public void CancelAll()
    {
        Optimize.Cancel();
        Glass.Cancel();
        Tolerance.Cancel();
    }
}

/// <summary>
/// The Optimize job (the Mac app's <c>OptimizeSession</c>): the merit function's working copy with its live
/// evaluation, the algorithm settings, a running job with its merit trace, Stop (keep the best design so far) and
/// Cancel, and the result previewed in the window and applied as one undoable step. Runs on the UI thread.
/// </summary>
public sealed class OptimizeSession
{
    public static readonly (string Id, string Title)[] Algorithms =
    [
        ("lm", "Damped least squares (LM)"), ("dogleg", "Trust region (dogleg)"), ("de", "Differential evolution"),
        ("cma_es", "CMA-ES"), ("nelder_mead", "Nelder–Mead"), ("hammer", "Basin hopping"),
    ];

    readonly DesignSession _s;
    AnalysisJob? _job;
    AnalysisJob? _evaluationJob;
    CancellationTokenSource? _evaluationDelay;
    string? _evaluatedKey;
    string? _pendingKey;

    public OptimizeSession(DesignSession session)
    {
        _s = session;
        Merit = MeritSettings.FromSection(session.Document.Section("merit"));
        UseDesignVariables = Variables.Count > 0;
        session.Document.Changed += () =>
        {
            Load();
            ScheduleEvaluation();
            Changed?.Invoke();
        };
        session.Workspace.StateChanged += () => { if (_s.Workspace.DesignSnapshot?.Hash.Hex != _lastSnapshot) ScheduleEvaluation(); };
    }

    string? _lastSnapshot;

    public DesignSession Session => _s;
    public DesignDocument Document => _s.Document;
    public Workspace Workspace => _s.Workspace;

    /// <summary>Raised on any change of state, settings, trace, result or evaluation.</summary>
    public event Action? Changed;
    /// <summary>Raised when only the evaluation (or its pending state) changed: the readout and operand values.</summary>
    public event Action? EvaluationChanged;

    public string Algorithm { get; set; } = "lm";
    /// <summary>null: the algorithm's default.</summary>
    public int? MaxIterations { get; set; }
    public int Seed { get; set; }
    /// <summary>Use the design's variables (else the engine's automatic set: every curvature and the back focus).</summary>
    public bool UseDesignVariables { get; set; }

    /// <summary>The merit being edited: a working copy of the document's <c>merit</c> section whose operands keep their identity.</summary>
    public MeritSettings Merit { get; private set; }
    public MeritEvaluation? Evaluation { get; private set; }
    public bool Evaluating { get; private set; }
    public string? EvaluationError { get; private set; }

    public JobPhase Phase { get; private set; } = JobPhase.Setup;
    public JobProgress Progress { get; private set; }
    public string? Failure { get; private set; }
    public List<MeritPoint> Trace { get; } = [];
    public OptimizeResult? Result { get; private set; }
    public bool Applied { get; private set; }

    public bool IsRunning => Phase == JobPhase.Running;
    public bool CanStop => IsRunning && (_job?.CanStop ?? false);
    public bool Stochastic => Algorithm is "de" or "cma_es" or "hammer";

    public IReadOnlyList<DesignVariable> Variables => DesignVariables.Of(Document);

    public bool Blocked => Workspace.DesignSnapshot is null || Workspace.ModelErrors.Count > 0;

    // ---- Settings ----

    public string? Problem => OptimizeVariables.Problem(Variables, UseDesignVariables, Document.System);

    public Dictionary<string, JsonNode?> Settings(MeritSettings? merit = null, bool evaluation = false)
    {
        var s = (merit ?? Merit).JobSettings();
        s["algorithm"] = Algorithm;
        if (!evaluation)
        {
            s["seed"] = Seed;
            if (MaxIterations is { } n) s["max_iterations"] = n;
        }
        if (OptimizeVariables.Settings(Variables, UseDesignVariables, Workspace.DesignSystem ?? Document.System) is { } v) s["variables"] = v;
        return s;
    }

    public void SetAlgorithm(string id)
    {
        Algorithm = id;
        ScheduleEvaluation();
        Changed?.Invoke();
    }

    public void SetUseDesignVariables(bool on)
    {
        UseDesignVariables = on;
        ScheduleEvaluation();
        Changed?.Invoke();
    }

    public void SetIterations(int? n) { MaxIterations = n is { } k ? Math.Max(1, k) : null; Changed?.Invoke(); }
    public void SetSeed(int n) { Seed = Math.Max(0, n); Changed?.Invoke(); }

    // ---- Merit function ----

    /// <summary>Takes the document's merit when it changed outside the editor (open, undo, redo).</summary>
    void Load()
    {
        var m = MeritSettings.FromSection(Document.Section("merit"));
        if (m.SameAs(Merit)) return;
        // Keep the identity of operands that are still there (undo of an edit keeps the selection).
        for (var i = 0; i < Math.Min(m.Operands.Count, Merit.Operands.Count); i++)
            if (JsonNode.DeepEquals(m.Operands[i].Kind, Merit.Operands[i].Kind)) m.Operands[i].Id = Merit.Operands[i].Id;
        Merit = m;
    }

    /// <summary>Applies an edit to the merit function as one undoable step.</summary>
    public void EditMerit(string actionName, Action<MeritSettings> change)
    {
        Load();
        var next = Merit.Clone();
        change(next);
        Merit = next;
        Document.EditSection("merit", next.Section, actionName);
        Changed?.Invoke();
    }

    /// <summary>
    /// Evaluates the merit at the design 200 ms after the last change (a newer change restarts the pause and cancels
    /// a pending request; going back to what is shown drops it), unless the same design and merit were evaluated.
    /// </summary>
    public void ScheduleEvaluation(TimeSpan? delay = null)
    {
        if (IsRunning || Workspace.DesignSnapshot is not { } snap) return;
        _lastSnapshot = snap.Hash.Hex;
        var settings = Settings(evaluation: true);
        var key = snap.Hash.Hex + (snap.LibraryHash ?? "") + new JsonObject(settings.Select(kv => KeyValuePair.Create(kv.Key, kv.Value?.DeepClone()))).ToJsonString();
        if (key == _pendingKey) return;
        _evaluationDelay?.Cancel();
        if (key == _evaluatedKey)
        {
            _evaluationJob?.Cancel();
            _evaluationJob = null;
            _pendingKey = null;
            Evaluating = false;
            EvaluationChanged?.Invoke();
            return;
        }
        _pendingKey = key;
        Evaluating = true;
        EvaluationChanged?.Invoke();
        var cts = _evaluationDelay = new CancellationTokenSource();
        _ = EvaluateLater(settings, key, delay ?? TimeSpan.FromMilliseconds(200), cts.Token);
    }

    async Task EvaluateLater(Dictionary<string, JsonNode?> settings, string key, TimeSpan delay, CancellationToken ct)
    {
        try { await Task.Delay(delay, ct); }
        catch (TaskCanceledException) { return; }
        _evaluationJob?.Cancel();
        var job = Workspace.SubmitDesignJob(new EngineAnalysisRequest("merit_evaluation", settings, new ExecutionSettings(Workspace.Backend), viewKey: "merit-evaluation"));
        if (job is null) { _pendingKey = null; Evaluating = false; EvaluationChanged?.Invoke(); return; }
        _evaluationJob = job;
        try
        {
            using var output = await job.ResultAsync();
            if (!ReferenceEquals(_evaluationJob, job)) return;
            Evaluation = MeritEvaluation.FromJson(output.Json);
            EvaluationError = Evaluation is null ? "The engine returned no merit." : null;
        }
        catch (EngineException e) when (e.Kind == EngineErrorKind.Cancelled) { return; }
        catch (EngineException e)
        {
            if (!ReferenceEquals(_evaluationJob, job)) return;
            EvaluationError = e.Message;
        }
        _evaluatedKey = key;
        _pendingKey = null;
        Evaluating = false;
        EvaluationChanged?.Invoke();
    }

    /// <summary>
    /// Replaces the default merit by the operands the engine generates for it (<c>merit_spec</c>), one undoable step.
    /// The EFL hold stays with the "Hold the focal length" switch, which adds it again.
    /// </summary>
    public async Task ConvertToCustomAsync()
    {
        Load();
        var defaults = Merit.Clone();
        defaults.Mode = MeritMode.Default;
        var job = Workspace.SubmitDesignJob(new EngineAnalysisRequest("merit_evaluation", Settings(defaults, evaluation: true), new ExecutionSettings(Workspace.Backend)));
        if (job is null) return;
        MeritEvaluation? e;
        Evaluating = true;
        EvaluationChanged?.Invoke();
        try
        {
            using var output = await job.ResultAsync();
            e = MeritEvaluation.FromJson(output.Json);
        }
        catch (EngineException x)
        {
            EvaluationError = x.Message;
            Evaluating = false;
            EvaluationChanged?.Invoke();
            return;
        }
        Evaluating = false;
        var operands = (e?.SpecOperands ?? []).Where(o => !MeritSettings.IsEflHold(o) || !defaults.HoldEfl).ToList();
        if (e is null || operands.Count == 0)
        {
            EvaluationError = "The engine generated no operands for this design.";
            EvaluationChanged?.Invoke();
            return;
        }
        EditMerit("Convert Merit to Custom", m =>
        {
            m.Operands = operands;
            m.SpecOptions = e.SpecOptions;
            m.Mode = MeritMode.Custom;
        });
        ScheduleEvaluation(TimeSpan.Zero);
    }

    // ---- Running ----

    /// <summary>Starts the run on the design's snapshot (<paramref name="fresh"/>: recompute, Try Again).</summary>
    public void Start(bool fresh = false)
    {
        _evaluationDelay?.Cancel();
        _evaluationJob?.Cancel();
        _pendingKey = null;
        Evaluating = false;
        var job = Workspace.SubmitDesignJob(new EngineAnalysisRequest("optimize", Settings(), new ExecutionSettings(Workspace.Backend), fresh));
        if (job is null)
        {
            Phase = JobPhase.Failed;
            Failure = "Fix the design's errors first.";
            Changed?.Invoke();
            return;
        }
        _job = job;
        Trace.Clear();
        Result = null;
        Applied = false;
        Failure = null;
        Phase = JobPhase.Running;
        Progress = new JobProgress(0, "Queued");
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
                if (MeritPoint.Parse(p.Stage) is { } m && (Trace.Count == 0 || Trace[^1].Iteration != m.Iteration)) Trace.Add(m);
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
            Result = OptimizeResult.FromJson(output.Json);
            if (Result is { History.Count: > 0 } r) { Trace.Clear(); Trace.AddRange(r.History); }
            Phase = Result is null ? JobPhase.Failed : JobPhase.Finished;
            Failure = Result is null ? "The engine returned no optimized design." : null;
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
        SyncPreview();
        Changed?.Invoke();
    }

    void UpdateActivity()
    {
        if (Phase == JobPhase.Running)
        {
            var text = Trace.Count > 0 ? $"Optimizing · {Trace[^1].Iteration}" : "Optimizing";
            _s.SetJobActivity("optimize", text, () => _s.ShowInspectorPage("optimize"));
        }
        else _s.SetJobActivity("optimize", null);
    }

    /// <summary>A finished, unapplied, improved result is previewed in the window until it is applied or discarded.</summary>
    void SyncPreview()
    {
        if (Phase == JobPhase.Finished && !Applied && Result is { Improved: true, System: { } system } r)
            Workspace.Preview(system, r.PreviewTitle());
        else if (Workspace.IsPreviewing && Workspace.PreviewTitle?.StartsWith("Optimize", StringComparison.Ordinal) == true)
            Workspace.Preview(null);
    }

    /// <summary>Ends the run early, keeping the best design so far.</summary>
    public void Stop() => _job?.Stop();

    /// <summary>Ends the run and discards it.</summary>
    public void Cancel()
    {
        if (_job is null) return;
        _job.Cancel();
        _job = null;
        Phase = JobPhase.Setup;
        UpdateActivity();
        Changed?.Invoke();
    }

    /// <summary>Applies the optimised design as one undoable step.</summary>
    public void Apply()
    {
        if (Result?.System is not { } system || Applied) return;
        Applied = true;
        DesignVariables.ReplaceSystem(Document, system, "Apply Optimization");
        SyncPreview();
        Changed?.Invoke();
    }

    /// <summary>Back to the settings (Discard, Done, Back).</summary>
    public void Reset()
    {
        Phase = JobPhase.Setup;
        Result = null;
        Trace.Clear();
        Applied = false;
        Failure = null;
        SyncPreview();
        ScheduleEvaluation();
        Changed?.Invoke();
    }

    public void NotifyChanged() => Changed?.Invoke();

    /// <summary>Shows or hides the result in the window.</summary>
    public void SetPreview(bool on)
    {
        if (on && Result is { System: { } s } r) Workspace.Preview(s, r.PreviewTitle());
        else Workspace.Preview(null);
        Changed?.Invoke();
    }
}
