using System.Text.Json.Nodes;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

public enum PanePhase { Idle, Running, Finished, Failed, Unavailable, Cancelled }

/// <summary>
/// One analysis pane (the Mac app's AnalysisPane): the analysis by name, its
/// settings, automatic update, and the latest output with its phase.
/// The workspace runs it; staleness is the workspace's verdict.
/// </summary>
public sealed class AnalysisPaneModel
{
    public AnalysisPaneModel(string analysis, JsonObject? settings = null, bool autoUpdate = true, string? id = null)
    {
        Id = id ?? "pane-" + Guid.NewGuid().ToString("N")[..8];
        Analysis = analysis;
        Settings = settings ?? [];
        AutoUpdate = autoUpdate;
    }

    public string Id { get; }
    public string Analysis { get; internal set; }
    /// <summary>The pane's settings (only what differs from the engine's defaults, plus the app's starting settings).</summary>
    public JsonObject Settings { get; internal set; }
    public bool AutoUpdate { get; set; }
    public bool ShowsSettings { get; set; }
    /// <summary>The spot diagram's box scale: common to the grid or per field.</summary>
    public Views.Analysis.Results.Readers.SpotScale SpotScale { get; set; }

    public AnalysisOutput? Output { get; internal set; }
    public PanePhase Phase { get; internal set; }
    public JobProgress Progress { get; internal set; }
    public string? Message { get; internal set; }

    internal AnalysisJob? Job;
    internal EngineAnalysisRequest? LastRequest;

    public bool IsRunning => Phase == PanePhase.Running;

    /// <summary>Raised on the UI thread whenever the pane's state or output changes.</summary>
    public event Action? Changed;
    internal void RaiseChanged() => Changed?.Invoke();

    internal Workspace? Workspace;
    public bool IsStale => Workspace?.IsStale(this) ?? false;

    public void Show(string analysis, JsonObject? settings = null) => Workspace?.Show(this, analysis, settings);
    public void SetSettings(JsonObject settings) => Workspace?.SetSettings(this, settings);
    public void Rerun(bool fresh = false) => Workspace?.Run(this, force: true, fresh: fresh);
    public void Cancel() => Job?.Cancel();
}
