using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml.Media;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Support;

/// <summary>
/// In-app timing of the open panes (launch actions):
/// <c>--do bench-out=&lt;file&gt;</c> chooses the JSON-lines file (default <c>bench-&lt;time&gt;.jsonl</c> beside the log);
/// <c>--do bench-panes=N[:backend]</c> runs each open pane, one at a time, once untimed and then N times, each
/// after a unique tiny edit of the design (the first curved surface's curvature changed by one to two parts
/// per billion, so nothing is reused from the engine's analysis graph), on <c>backend</c>
/// (<c>cpu_reference</c>, <c>auto</c>, <c>gpu</c>, <c>cuda</c> or <c>gpu:&lt;name&gt;</c>; default: the window's own).
/// Other panes do not rerun meanwhile; the design and the backend are put back afterwards. The document
/// is never edited.
/// Each timed run writes one line: <c>pane</c>, <c>analysis</c>, <c>run</c>, <c>backend_requested</c>,
/// <c>backend_used</c>, <c>engine_ms</c> (run_info.duration_ms), <c>device</c> (from the backend's execution
/// report, when a GPU ran: per execution its backend, <c>wall_ms</c>, <c>device_ms</c> and per stage
/// <c>[name, device_ms, wall_ms]</c>), <c>app_view_ms</c> (submit to the pane's view
/// rebuilt) and <c>app_frame_ms</c> (submit to the next frame after that), or <c>error</c>.
/// </summary>
static class BenchPanes
{
    static string? _out;

    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.LaunchActions["bench-out"] = (_, arg) =>
        {
            _out = string.IsNullOrWhiteSpace(arg) ? null : Path.GetFullPath(arg);
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["bench-panes"] = (w, arg) => RunAsync(w.Session.Workspace, arg);
    }

    static async Task RunAsync(Workspace ws, string? arg)
    {
        var (runs, backend) = BenchPlan.Parse(arg);
        if (ws.DesignSystem is not { } original) { App.Log("bench-panes: no design"); return; }
        var path = _out ?? Path.Combine(AppSettings.Directory, $"bench-{DateTime.Now:yyyyMMdd-HHmmss}.jsonl");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var (savedBackend, savedGpu) = (ws.Backend, ws.GpuName);
        if (backend is { } b) (ws.Backend, ws.GpuName) = (b.Backend, b.GpuName);
        ws.HoldRefresh = true;
        var serial = 0;
        try
        {
            await using var file = new StreamWriter(path, append: true);
            foreach (var pane in ws.Panes.ToList())
                for (var run = 0; run <= runs; run++) // run 0: the untimed warm-up
                {
                    var line = await TimeOneAsync(ws, pane, BenchPlan.Edited(original, ++serial));
                    if (run == 0) continue;
                    line["run"] = run;
                    await file.WriteLineAsync(line.ToJsonString());
                    await file.FlushAsync();
                }
            App.Log($"bench-panes: {runs} runs of {ws.Panes.Count} panes written to {path}");
        }
        finally
        {
            (ws.Backend, ws.GpuName) = (savedBackend, savedGpu);
            ws.HoldRefresh = false;
            ws.Update(original, immediately: true);
        }
    }

    /// <summary>The timings of the backend's execution report (an array of executions): the rest stays out of the file.</summary>
    static JsonArray Timings(JsonNode? execution)
    {
        static double? Ms(JsonNode? seconds) => seconds is JsonValue v && v.TryGetValue<double>(out var x) ? Math.Round(x * 1000, 4) : null;
        var list = new JsonArray();
        foreach (var e in execution as JsonArray ?? [])
        {
            if (e?["settings"]?["resident_execution"]?["plan_stats"] is not JsonObject stats) continue;
            var stages = new JsonArray();
            foreach (var st in stats["stages"] as JsonArray ?? [])
                stages.Add(new JsonArray((JsonNode?)st?["name"]?.DeepClone(), Ms(st?["device_seconds"]), Ms(st?["wall_seconds"])));
            list.Add(new JsonObject
            {
                ["backend"] = e["backend"]?.DeepClone(), ["wall_ms"] = Ms(stats["wall_seconds"]), ["device_ms"] = Ms(stats["device_seconds"]),
                ["stages"] = stages,
            });
        }
        return list;
    }

    static async Task<JsonObject> TimeOneAsync(Workspace ws, AnalysisPaneModel pane, OpticalSystem system)
    {
        var line = new JsonObject { ["pane"] = pane.Id, ["analysis"] = pane.Analysis, ["backend_requested"] = ws.Execution.WireName };
        // The snapshot of the edited design first (not timed): the request then runs on it.
        var before = ws.Snapshot?.Hash;
        ws.Update(system, immediately: true);
        var wait = Stopwatch.StartNew();
        while ((ws.Snapshot?.Hash == before || ws.ModelErrors.Count > 0) && wait.Elapsed < TimeSpan.FromSeconds(30)) await Task.Delay(2);
        if (ws.Snapshot?.Hash == before) { line["error"] = "no new snapshot"; return line; }

        var done = new TaskCompletionSource<double>(TaskCreationOptions.RunContinuationsAsynchronously);
        var clock = new Stopwatch();
        // Subscribed after the pane's view, so this runs once the view has rebuilt for the new output.
        void Changed()
        {
            if (clock.IsRunning && pane.Phase is not PanePhase.Running) done.TrySetResult(clock.Elapsed.TotalMilliseconds);
        }
        pane.Changed += Changed;
        try
        {
            clock.Start();
            ws.Run(pane, force: true);
            var viewMs = await done.Task.WaitAsync(TimeSpan.FromMinutes(10));
            var frame = new TaskCompletionSource<double>(TaskCreationOptions.RunContinuationsAsynchronously);
            void Rendering(object? s, object e) => frame.TrySetResult(clock.Elapsed.TotalMilliseconds);
            CompositionTarget.Rendering += Rendering;
            double frameMs;
            try { frameMs = await frame.Task.WaitAsync(TimeSpan.FromSeconds(5)); }
            finally { CompositionTarget.Rendering -= Rendering; }
            if (pane.Phase != PanePhase.Finished || pane.Output is not { } o) { line["error"] = pane.Message ?? pane.Phase.ToString(); return line; }
            line["backend_used"] = o.Backend?.Used ?? o.RunInfo.Backend;
            line["engine_ms"] = o.RunInfo.DurationMs;
            if (Timings(o.Backend?.Execution) is { Count: > 0 } timings) line["device"] = timings;
            line["app_view_ms"] = Math.Round(viewMs, 3);
            line["app_frame_ms"] = Math.Round(frameMs, 3);
        }
        catch (TimeoutException) { line["error"] = "timed out"; }
        finally { pane.Changed -= Changed; }
        return line;
    }
}
