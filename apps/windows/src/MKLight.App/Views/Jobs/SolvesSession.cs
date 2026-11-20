using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// Apply Solves (the Mac app's <c>SolvesSession</c>): a dry run (the <c>solves</c> analysis) shown as a report, then
/// one undoable commit of the solved design (<see cref="IEngineClient.ApplySolvesAsync"/>).
/// </summary>
public sealed class SolvesSession(DesignSession session)
{
    AnalysisJob? _job;

    public DesignSession Session => session;
    public JobPhase Phase { get; private set; } = JobPhase.Setup;
    public string? Failure { get; private set; }
    public IReadOnlyList<SolveOutcomeRow> Outcomes { get; private set; } = [];
    public IReadOnlyList<string> Warnings { get; private set; } = [];
    public bool Committed { get; private set; }
    public int Changes => Outcomes.Count(o => o.Changes);

    public event Action? Changed;

    /// <summary>The values the solves would set (nothing changes yet). <paramref name="fresh"/>: recompute (Try Again).</summary>
    public async Task DryRunAsync(bool fresh = false)
    {
        Committed = false;
        _job?.Cancel();
        var job = session.Workspace.SubmitDesignJob(new EngineAnalysisRequest("solves", [], default, fresh, viewKey: "solves-dry-run"));
        if (job is null)
        {
            Phase = JobPhase.Failed;
            Failure = session.Workspace.ModelErrors.FirstOrDefault() ?? "The design is not ready yet.";
            Changed?.Invoke();
            return;
        }
        _job = job;
        Phase = JobPhase.Running;
        Changed?.Invoke();
        try
        {
            using var output = await job.ResultAsync();
            if (!ReferenceEquals(_job, job)) return;
            Read(output.Json, session.Workspace.DesignSystem ?? session.Document.System);
            Phase = JobPhase.Finished;
        }
        catch (EngineException e) when (e.Kind == EngineErrorKind.Cancelled) { return; }
        catch (EngineException e)
        {
            if (!ReferenceEquals(_job, job)) return;
            Phase = JobPhase.Failed;
            Failure = e.Message;
        }
        Changed?.Invoke();
    }

    void Read(JsonNode? report, OpticalSystem system)
    {
        Outcomes = SolveOutcomeRow.Read(report, system);
        Warnings = SolveOutcomeRow.Warnings(report);
    }

    /// <summary>Solves in the engine and replaces the design as one undoable step.</summary>
    public async Task CommitAsync()
    {
        if (session.Workspace.DesignSnapshot is not { } snap) return;
        try
        {
            var applied = await session.Workspace.Engine.ApplySolvesAsync(new Dictionary<string, JsonNode?>(), snap);
            var system = OpticalSystem.FromJson(JsonNode.Parse(applied.Snapshot.SystemJson));
            Read(applied.Report, session.Document.System);
            DesignVariables.ReplaceSystem(session.Document, system, "Apply Solves");
            Committed = true;
            Phase = JobPhase.Finished;
        }
        catch (EngineException e)
        {
            Phase = JobPhase.Failed;
            Failure = e.Message;
        }
        Changed?.Invoke();
    }
}
