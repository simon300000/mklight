using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels.Jobs;
using MKLight.App.Windows;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// Launch actions for the Tolerance page's screenshots and checks (after <c>--do job=tolerance</c> or
/// <c>--do tolerance-run[=trials]</c>):
/// <list type="bullet">
/// <item><c>--do tolerance-bin[=k[:t]]</c>: lists the trials of histogram bar k (zero-based; without k the tallest bar),
/// previews its t-th trial (worst first) when given, and scrolls the Monte Carlo section to the top.</item>
/// <item><c>--do tolerance-pivot[=last|s&lt;i&gt;|clash,…]</c>: back to the tolerance set with every group but the first
/// element's collapsed and its range and pivot line in view; <c>last</c> or <c>s&lt;i&gt;</c> moves its pivot,
/// <c>clash</c> moves its last element row alone (the rows starting there are flagged).</item>
/// <item><c>--do tolerance-export=&lt;file.csv&gt;</c>: writes the sensitivity table there, the trials to
/// <c>&lt;file&gt;-trials.csv</c> and the summary to <c>&lt;file&gt;-summary.txt</c>.</item>
/// </list>
/// </summary>
static class ToleranceLaunchActions
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.LaunchActions["tolerance-bin"] = async (w, arg) =>
        {
            var s = Session(w);
            if (s.Result is not { } r) return;
            var bins = ToleranceHistogram.Bins(r);
            if (bins.Count == 0) return;
            var parts = (arg ?? "").Split(':');
            var k = int.TryParse(parts[0], out var i) ? Math.Clamp(i, 0, bins.Count - 1) : bins.Select((b, j) => (b.Count, j)).Max().j;
            s.ShowResult();
            s.SelectBin(k);
            if (parts.Length > 1 && int.TryParse(parts[1], out var t) && ToleranceHistogram.Trials(r, bins, k) is { Count: > 0 } trials)
                await s.PreviewAsync(new TolerancePreviewCase.Trial(trials[Math.Clamp(t, 0, trials.Count - 1)]));
            await Task.Delay(600);
            // The Monte Carlo section at the top of the page.
            if (Descendants<StackPanel>(w.RootElement).FirstOrDefault(p => p.Tag is "tolerance-monte-carlo") is { } mc
                && Ancestor<ScrollViewer>(mc) is { } scroller && scroller.Content is UIElement content)
            {
                var y = mc.TransformToVisual(content).TransformPoint(default).Y;
                scroller.ChangeView(null, Math.Max(0, y - 4), null, true);
            }
            await Task.Delay(400);
        };
        AppHooks.LaunchActions["tolerance-pivot"] = async (w, arg) =>
        {
            var s = Session(w);
            s.EditSettings();
            var system = w.Session.Document.System;
            if (s.Settings.Rows.FirstOrDefault(r => r.Kind.IsElement() && r.MovingSurfaces(system).Count > 1) is not { } row) return;
            var moving = row.MovingSurfaces(system);
            int f = moving[0], l = moving[^1];
            foreach (var option in (arg ?? "").ToLowerInvariant().Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                switch (option)
                {
                    case "last":
                        s.Edit("Change Element Pivot", x => TolerancePivotChoice.Last.Apply(x, row.Id, system, f, l));
                        break;
                    case ['s', .. var n] when int.TryParse(n, out var p):
                        s.Edit("Change Element Pivot", x => new TolerancePivotChoice(TolerancePivotPlace.Surface, p).Apply(x, row.Id, system, f, l));
                        break;
                    case "clash":
                        // The last row starting on the same surface moved alone: the group no longer agrees, both rows are flagged.
                        if (s.Settings.Rows.LastOrDefault(r => r.Kind.IsElement() && r.Surface == row.Surface && r.Id != row.Id) is { } other)
                            s.Edit("Change Element", x => { if (x.Rows.FirstOrDefault(r => r.Id == other.Id) is { } o) o.Last = system.Surfaces[f].Id; });
                        break;
                }
            }
            var groups = ToleranceTableGroup.Of(s.Settings.Rows, system);
            var keep = groups.FirstOrDefault(g => g.Rows.Any(r => r.Id == row.Id))?.Key;
            s.CollapsedGroups.Clear();
            foreach (var g in groups) if (g.Key != keep) s.CollapsedGroups.Add(g.Key);
            s.NotifyChanged();
            await Task.Delay(600);
            if (Descendants<StackPanel>(w.RootElement).FirstOrDefault(p => p.Tag is "tolerance-element-line") is { } line)
                line.StartBringIntoView(new BringIntoViewOptions { VerticalAlignmentRatio = 0.35, AnimationDesired = false });
            await Task.Delay(400);
        };
        AppHooks.LaunchActions["tolerance-export"] = async (w, arg) =>
        {
            var s = Session(w);
            if (arg is null || s.Result is null) return;
            var path = Path.GetFullPath(arg);
            var stem = Path.Combine(Path.GetDirectoryName(path) ?? "", Path.GetFileNameWithoutExtension(path));
            if (s.SensitivityCsv() is { } sens) await File.WriteAllTextAsync(path, sens);
            if (s.TrialsCsv() is { } trials) await File.WriteAllTextAsync(stem + "-trials.csv", trials);
            if (s.SummaryText() is { } summary) await File.WriteAllTextAsync(stem + "-summary.txt", summary + "\n");
        };
    }

    static ToleranceSession Session(DesignWindow w)
    {
        w.Session.ShowInspectorPage("tolerance");
        return DesignJobs.Of(w.Session).Tolerance;
    }

    static T? Ancestor<T>(DependencyObject d) where T : DependencyObject
    {
        for (var p = VisualTreeHelper.GetParent(d); p is not null; p = VisualTreeHelper.GetParent(p))
            if (p is T t) return t;
        return null;
    }

    static IEnumerable<T> Descendants<T>(DependencyObject root) where T : DependencyObject
    {
        var n = VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is T t) yield return t;
            foreach (var d in Descendants<T>(child)) yield return d;
        }
    }
}
