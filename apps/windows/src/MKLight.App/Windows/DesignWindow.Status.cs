using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Windows;

/// <summary>
/// The design window's status capsule: GPU kernel preparation, activity and running jobs, key
/// numbers, issues, the details flyout (first order, the x / y data of an anamorphic design,
/// engine, snapshot, glass library, GPU kernels, compute backend) and the preview badge.
/// </summary>
public sealed partial class DesignWindow
{
    void UpdateStatus()
    {
        var ws = _session.Workspace;
        var running = ws.Panes.Where(p => p.IsRunning).ToList();
        var monitor = GpuPreparationMonitor.Shared;
        Status.SetGpu(GpuPreparationMonitor.BadgeText(monitor.State, monitor.Shown), monitor.Shown == GpuPreparationMonitor.Display.Preparing);
        Status.SetActivityAction(_session.RunningJobs.Values.Select(j => j.Open).FirstOrDefault(o => o is not null));
        Status.SetActivity(_session.RunningJobs.Count > 0 ? string.Join("  ·  ", _session.RunningJobs.Values.Select(j => j.Text)) : ws.IsBusy ? (running.Count == 1 && running[0].Progress.Fraction > 0
            ? AnalysisCatalog.ShortTitle(running[0].Analysis) + " " + (running[0].Progress.Fraction * 100).ToString("0", CultureInfo.InvariantCulture) + "%"
            : "") : null);

        var json = ws.Summary.Output?.Json;
        var firstOrder = FirstOrderData.Read(json);
        var anamorphic = firstOrder is not null && FirstOrderXY.IsAnamorphic(json, _session.Document.System);
        var numbers = StatusCapsuleModel.KeyNumbers(firstOrder, anamorphic);
        var hasErrors = ws.ModelErrors.Count > 0 || ws.Issues.Any(i => i.Severity == "error");
        Status.SetNumbers(hasErrors && numbers.Count == 0 ? [] : numbers,
            hasErrors ? "Design has errors" : string.IsNullOrEmpty(_session.Document.System.Name) ? "Untitled" : _session.Document.System.Name,
            stale: ws.Summary.Output is not null && ws.Summary.IsStale);
        Status.SetIssues(ws.Issues);

        var gpu = monitor.State is GpuPreparation.Idle && App.Engine is FFIEngine ffi ? ffi.GetGpuPreparation() : monitor.State;
        Status.SetDetails([
            .. StatusCapsuleModel.FirstOrder(firstOrder, anamorphic),
            .. StatusCapsuleModel.Engine(ws.Engine.Info.Name + " " + ws.Engine.Info.Version, ws.Snapshot, gpu),
        ]);

        var current = CurrentExecution(ws);
        var options = ComputeBackendOptions.From(BackendList(ws), current);
        Status.SetBackends(options, ComputeBackendOptions.IndexOf(options, current), i => ChooseBackend(options[i].Execution));
        Status.SetPreview(ws.PreviewTitle, () => ws.Preview(null));
    }

    /// <summary>What the design's analyses request: the workspace's preference and, for a GPU, the session's chosen one.</summary>
    static ExecutionSettings CurrentExecution(Workspace ws) => ws.Execution;

    /// <summary>Simulated backend list (<c>--do simulate-backends</c>); null: the engine's.</summary>
    static IReadOnlyList<EngineBackend>? _simulatedBackends;

    static IReadOnlyList<EngineBackend> BackendList(Workspace ws)
    {
        if (_simulatedBackends is { } simulated) return simulated;
        try { return (ws.Engine as FFIEngine ?? App.Engine as FFIEngine)?.BackendList() ?? []; }
        catch (EngineException) { return []; }
    }

    /// <summary>A choice of the picker: every analysis follows, and the design keeps it in its views.</summary>
    void ChooseBackend(ExecutionSettings execution)
    {
        var ws = _session.Workspace;
        if (execution == CurrentExecution(ws)) return;
        Apply(ws, execution);
        if (StatusCapsuleModel.WithBackend(_session.Document.Section("views").Node, execution) is { } views)
            _session.Document.SetSection("views", new JsonTree(views));
        ws.RerunAll();
        UpdateStatus();
    }

    static void Apply(Workspace ws, ExecutionSettings execution)
    {
        ws.Backend = execution.Backend;
        ws.GpuName = execution.GpuName;
        if (ws.Engine is FFIEngine ffi) ffi.PreferredGpuName = execution.GpuName;
    }

    [ModuleInitializer]
    internal static void RegisterStatus()
    {
        // A design's compute backend, from its views, before its first analysis.
        AppHooks.SessionStarting.Add(session =>
        {
            Apply(session.Workspace, StatusCapsuleModel.ReadBackend(session.Document.Section("views").Node));
            return Task.CompletedTask;
        });
        AppHooks.WindowCreated += w =>
        {
            var monitor = GpuPreparationMonitor.Shared;
            void Changed() => w.UpdateStatus();
            monitor.Changed += Changed;
            w.Closed += (_, _) => monitor.Changed -= Changed;
            if (App.Engine is FFIEngine ffi) _ = monitor.StartAsync(ffi.GetGpuPreparation);
        };

        // Screenshots and checks of states this machine's engine may not reach.
        // --do status-details: opens the capsule's details flyout.
        AppHooks.LaunchActions["status-details"] = (w, _) =>
        {
            w.Status.ShowDetails(TimeSpan.FromSeconds(20));
            return Task.CompletedTask;
        };
        // --do compute-backend=<auto|cpu_reference|gpu|gpu:name>: the picker's choice, as if made in it.
        AppHooks.LaunchActions["compute-backend"] = (w, arg) =>
        {
            w.ChooseBackend(ExecutionSettings.FromWireName(arg));
            return Task.CompletedTask;
        };
        // --do simulate-gpu-preparation[=percent|ready|cache-warning]: the capsule as while the GPU's kernels are prepared.
        AppHooks.LaunchActions["simulate-gpu-preparation"] = (w, arg) =>
        {
            const int total = 205;
            var (state, shown) = arg switch
            {
                "ready" => ((GpuPreparation)new GpuPreparation.Ready(SimulatedDevice, total, 0, 31.4), GpuPreparationMonitor.Display.Finished),
                "cache-warning" => (new GpuPreparation.Ready(SimulatedDevice, total, 0, 31.4, "the cache directory is read-only"), GpuPreparationMonitor.Display.Hidden),
                _ => (new GpuPreparation.Preparing(SimulatedDevice, (int)Math.Round(total * (double.TryParse(arg, NumberStyles.Float, CultureInfo.InvariantCulture, out var p) ? p : 42) / 100), total),
                    GpuPreparationMonitor.Display.Preparing),
            };
            GpuPreparationMonitor.Shared.Simulate(state, shown);
            return Task.CompletedTask;
        };
        // --do simulate-backends[=file.json]: the picker as with a GPU backend (the engine's backend list JSON).
        AppHooks.LaunchActions["simulate-backends"] = (w, arg) =>
        {
            _simulatedBackends = EngineBackend.ListFromJson(arg is { Length: > 0 } path ? File.ReadAllText(path) : SimulatedBackendsJson);
            w.UpdateStatus();
            return Task.CompletedTask;
        };
        // --do simulate-gpu-results: every pane's result as computed on gpu:cuda (the second as a GPU failure completed on the CPU).
        AppHooks.LaunchActions["simulate-gpu-results"] = (w, _) =>
        {
            var k = 0;
            foreach (var pane in w.Session.Workspace.Panes)
            {
                if (pane.Output is not { } o) continue;
                pane.Output = SimulatedGpuOutput(o, failed: k++ == 1);
                pane.RaiseChanged();
            }
            w.UpdateStatus();
            return Task.CompletedTask;
        };
        // --do run-details[=pane index]: opens a pane's run details (its backend badge's flyout).
        AppHooks.LaunchActions["run-details"] = (w, arg) =>
        {
            var views = w.Panels.PaneViews;
            var i = int.TryParse(arg, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) ? n : 0;
            if (i >= 0 && i < views.Count && views[i].FindName("BackendBadge") is Button { Flyout: { } flyout } badge)
            {
                // Shown again if another window takes the focus while the screenshot waits.
                var until = DateTime.UtcNow.AddSeconds(20);
                flyout.Closed += (_, _) => { if (DateTime.UtcNow < until) w.DispatcherQueue.TryEnqueue(() => flyout.ShowAt(badge)); };
                flyout.ShowAt(badge);
            }
            return Task.CompletedTask;
        };
    }

    const string SimulatedDevice = "NVIDIA GeForce RTX 4090";

    /// <summary>A backend list shaped as <c>mk_engine_list_backends</c> documents it, with a CUDA backend.</summary>
    const string SimulatedBackendsJson = """
        [{"backend": {"kind": "cpu_reference"}, "id": "cpu_reference", "title": "CPU reference (f64)", "available": true, "reason": null,
          "precision": "f64_reference", "capabilities": {"fp64": true, "reference": true, "analyses": [], "device_name": null, "unified_memory": null}},
         {"backend": {"kind": "gpu", "name": "cuda"}, "id": "gpu:cuda", "title": "CUDA", "available": true, "reason": null,
          "precision": "f32_gpu", "capabilities": {"fp64": false, "reference": false, "analyses": ["spot_diagram", "ray_fan", "geometric_mtf"],
          "device_name": "RTX 4090", "unified_memory": false}}]
        """;

    /// <summary>
    /// <paramref name="o"/> relabelled as docs/RESULTS.md §5 reports a GPU run: an automatic request the cost
    /// model sent to CUDA, or (<paramref name="failed"/>) a CUDA request that failed and completed on the CPU.
    /// The result itself and its buffers are the real ones.
    /// </summary>
    static AnalysisOutput SimulatedGpuOutput(AnalysisOutput o, bool failed)
    {
        var run = o.RunInfo.ToJson();
        var backend = failed
            ? JsonNode.Parse("""
                {"requested": "gpu:cuda", "used": "cpu_reference", "precision": "f64_reference", "device": null,
                 "fallback_reason": "native spot_diagram failed after 12 ms: CUDA_ERROR_OUT_OF_MEMORY",
                 "execution": [], "cost_decision": {"decision": "explicit_backend", "reason": "cost rules bypassed"}}
                """)
            : JsonNode.Parse("""
                {"requested": "auto", "used": "gpu:cuda", "precision": "f32_gpu", "device": "NVIDIA GeForce RTX 4090",
                 "fallback_reason": null, "execution": [],
                 "cost_decision": {"decision": "qualified_cuda", "rule": "cost model", "version": 5, "before_device": false,
                                   "predicted_ms": {"cpu": 38.2, "cuda": 6.1}, "advantage": 6.26, "required": {"ratio": 1.5, "gain_ms": 5}}}
                """);
        run["backend"] = backend!["used"]!.GetValue<string>();
        run["precision"] = backend["precision"]!.GetValue<string>();
        if (failed)
            run["failed_attempts"] = JsonNode.Parse("""[{"backend": "gpu:cuda", "outcome": "failed", "reason": "CUDA_ERROR_OUT_OF_MEMORY: out of memory", "elapsed_ms": 12.3}]""");
        return new AnalysisOutput(RunInfo.FromJson(run), o.Json, o.Buffers, o.LibraryHash, o.EngineRequest, BackendReport.FromJson(backend));
    }
}
