using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using Fmt = MKLight.App.Views.Analysis.Results.Readers.Fmt;
using MKLight.App.ViewModels;
using MKLight.App.Views;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Layout;
using MKLight.App.Views.Panels;
using MKLight.App.Views.Inspector;
using MKLight.Engine;
using MKLight.Model;
using Windows.System;

namespace MKLight.App.Windows;

/// <summary>
/// The design window's status capsule: activity and running jobs, key numbers, issues, the details flyout
/// (first order, engine, snapshot, compute backend, GPU kernels) and the preview badge.
/// </summary>
public sealed partial class DesignWindow
{
    void UpdateStatus()
    {
        var ws = _session.Workspace;
        var running = ws.Panes.Where(p => p.IsRunning).ToList();
        Status.SetActivityAction(_session.RunningJobs.Values.Select(j => j.Open).FirstOrDefault(o => o is not null));
        Status.SetActivity(_session.RunningJobs.Count > 0 ? string.Join("  ·  ", _session.RunningJobs.Values.Select(j => j.Text)) : ws.IsBusy ? (running.Count == 1 && running[0].Progress.Fraction > 0
            ? AnalysisCatalog.ShortTitle(running[0].Analysis) + " " + (running[0].Progress.Fraction * 100).ToString("0", CultureInfo.InvariantCulture) + "%"
            : "") : null);
        var json = ws.Summary.Output?.Json;
        var entries = (json?["entries"] as JsonArray ?? []).OfType<JsonObject>()
            .Select(e => (Key: e["key"].StringOrNull() ?? "", Value: e["value"].Nf(), Unit: e["unit"].StringOrNull() ?? "", Label: e["label"].StringOrNull() ?? ""))
            .Where(e => e.Value is not null).ToList();
        (string, string)? Entry(string label, params string[] keys) =>
            keys.Select(k => entries.FirstOrDefault(e => e.Key == k)).FirstOrDefault(e => e.Key is { Length: > 0 }) is { Key.Length: > 0 } e
                ? (label, Fmt.Value(e.Value!.Value, e.Unit)) : null;
        var numbers = new[] { Entry("EFL", "efl"), Entry("F/#", "working_f_number", "image_f_number"), Entry("TT", "total_track") }
            .Where(n => n is not null).Select(n => n!.Value).ToList();
        var hasErrors = ws.ModelErrors.Count > 0 || ws.Issues.Any(i => i.Severity == "error");
        Status.SetNumbers(hasErrors && numbers.Count == 0 ? [] : numbers,
            hasErrors ? "Design has errors" : string.IsNullOrEmpty(_session.Document.System.Name) ? "Untitled" : _session.Document.System.Name);
        Status.SetIssues(ws.Issues);

        var rows = new List<(string, string?)> { ("First-Order Data", null) };
        rows.AddRange(entries.Select(e => (e.Label, (string?)Fmt.Value(e.Value!.Value, e.Unit))));
        rows.Add(("Engine", null));
        rows.Add(("Engine", ws.Engine.Info.Name + " " + ws.Engine.Info.Version));
        rows.Add(("Snapshot", ws.Snapshot?.Hash.ToString() is { Length: > 16 } h ? h[..16] + "…" : "—"));
        BackendPreference[] backends = [BackendPreference.Auto, BackendPreference.CpuReference, BackendPreference.Gpu];
        Status.SetBackends(backends.Select(b => b.Title()).ToList(), Array.IndexOf(backends, ws.Backend), i =>
        {
            ws.Backend = backends[i];
            ws.RerunAll();
        });
        Status.SetPreview(ws.PreviewTitle, () => ws.Preview(null));
        if (App.Engine is FFIEngine ffi) rows.Add(("GPU kernels", GpuText(ffi.GetGpuPreparation())));
        Status.SetDetails(rows);
    }

    static string GpuText(GpuPreparation p) => p switch
    {
        GpuPreparation.Preparing x => $"Preparing ({x.Done} of {x.Total})",
        GpuPreparation.Ready x => x.FromCache > 0 ? $"Ready ({x.Kernels}, loaded from cache)" : $"Ready ({x.Kernels}, prepared in {x.Seconds.ToString("0.0", CultureInfo.InvariantCulture)} s)",
        GpuPreparation.Unavailable x => "No GPU: " + x.Reason,
        GpuPreparation.Failed x => $"Compiled on first use ({x.Message})",
        _ => "Not prepared",
    };
}
