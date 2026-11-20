using System.Runtime.CompilerServices;
using MKLight.App.Support;
using MKLight.App.Windows;
using MKLight.Model;

namespace MKLight.App.Views.Analysis;

/// <summary>
/// Launch actions for the analysis pane's chrome (UI testing and screenshots):
/// <list type="bullet">
/// <item><c>--do pane-preview[=title]</c>: previews a candidate (the design with surface 3 0.1 mm thicker) in every pane.</item>
/// <item><c>--do pane-more[=analysis]</c>: opens the More menu of the first pane (of that analysis).</item>
/// <item><c>--do pane-state=kind[:analysis]</c>: shows a state in the panes (of that analysis), whatever the
/// pane's own: <c>running</c>, <c>unavailable</c>, <c>errors</c>, <c>failed</c>, <c>notready</c>, <c>cancelled</c>,
/// <c>onrequest</c>, <c>waiting</c>, <c>updating</c> (the pulse over a result), <c>rerun-failed</c>, or <c>clear</c>.</item>
/// <item><c>--do pane-motion=off|on|system</c>: motion on, off (Reduce Motion), or as Windows has it; logs whether each pane pulses.</item>
/// </list>
/// </summary>
static class AnalysisPaneLaunchActions
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.LaunchActions["pane-preview"] = (w, arg) =>
        {
            var sys = w.Session.Document.System;
            var i = Math.Min(3, sys.Surfaces.Count - 2);
            var candidate = sys with { Surfaces = sys.Surfaces.Select((s, k) => k == i ? s with { Thickness = s.Thickness + 0.1 } : s).ToValueList() };
            w.Session.Workspace.Preview(candidate, string.IsNullOrEmpty(arg) ? "Candidate 3" : arg);
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["pane-more"] = async (w, arg) =>
        {
            if (Panes(w, arg).FirstOrDefault() is { } pane) await pane.ShowMoreMenuForCaptureAsync();
            await Task.Delay(400);
        };
        AppHooks.LaunchActions["pane-state"] = async (w, arg) =>
        {
            var parts = (arg ?? "").Split(':', 2);
            foreach (var pane in Panes(w, parts.Length > 1 ? parts[1] : null))
            {
                pane.ForcedFacts = Forced(parts[0], pane);
                pane.Refresh();
            }
            await Task.Delay(300);
        };
        AppHooks.LaunchActions["pane-motion"] = async (w, arg) =>
        {
            Motion.Override(arg switch { "off" => false, "on" => true, _ => null });
            await Task.Delay(300);
            // For scripted checks: whether each pane's updating pulse runs now.
            foreach (var pane in w.Panels.PaneViews) App.Log($"pane {pane.Pane.Analysis}: motion {Motion.Enabled}, pulsing {pane.IsPulsing}");
        };
    }

    static IEnumerable<AnalysisPaneView> Panes(DesignWindow w, string? analysis) =>
        w.Panels.PaneViews.Where(p => string.IsNullOrEmpty(analysis) || p.Pane.Analysis == analysis);

    static PaneFacts? Forced(string kind, AnalysisPaneView view)
    {
        var p = new PaneFacts { AutoUpdate = view.Pane.AutoUpdate };
        return kind switch
        {
            "running" => p with { Running = true, Stage = "Tracing rays" },
            "unavailable" => p with { Unavailable = true, Message = "This analysis needs a GPU, and none was found." },
            "errors" => p with { Failed = true, DesignHasErrors = true, Message = "Surface 4: the semi-diameter is negative." },
            "failed" => p with { Failed = true, Message = "The chief ray missed surface 6 at field 3." },
            "notready" => p with { ModelError = "Surface 4: the semi-diameter is negative." },
            "cancelled" => p with { Cancelled = true },
            "onrequest" => p with { AutoUpdate = false },
            "waiting" => p with { AutoUpdate = true },
            "updating" => p with { HasOutput = true, Running = true, Stage = "Tracing rays" },
            "rerun-failed" => p with { HasOutput = true, Failed = true, Message = "The chief ray missed surface 6 at field 3." },
            _ => null,
        };
    }
}
