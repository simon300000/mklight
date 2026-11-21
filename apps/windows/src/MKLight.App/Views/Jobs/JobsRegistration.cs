using System.Runtime.CompilerServices;
using MKLight.App.Support;
using MKLight.App.Views.Inspector;
using MKLight.App.Windows;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// Registers the design-job pages of the inspector, their launch actions (for screenshots and checks) and ends a
/// design's jobs when its window closes.
/// <list type="bullet">
/// <item><c>--do job=optimize</c> (or <c>glass_substitution</c>, <c>tolerance</c>, <c>solves</c>) opens a page.</item>
/// <item><c>--do optimize-run[=lm|de|…]</c> runs Optimize, waits for the result and previews it;
/// <c>--do optimize-apply</c> applies it; <c>--do optimize-running=s</c> starts it and returns after s seconds
/// (a screenshot of the running state); <c>--do merit-custom</c> converts the merit to custom.</item>
/// <item><c>--do solves-run</c>, <c>--do glass-run</c>, <c>--do tolerance-run</c> run those jobs and wait.</item>
/// </list>
/// </summary>
static class JobsRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        InspectorPages.Add(new("optimize", "Optimize", "\uE9D9", 20, s => new OptimizePanel(s)));
        InspectorPages.Add(new("glass_substitution", "Glass Substitution", "\uEB42", 21, s => new GlassSubstitutionPanel(s)));
        InspectorPages.Add(new("tolerance", "Tolerance", "\uE9E9", 22, s => new TolerancePanel(s)));
        InspectorPages.Add(new("solves", "Apply Solves", "\uE943", 23, s => new SolvesPanel(s)));

        AppHooks.WindowCreated += w => w.Closed += (_, _) => DesignJobs.Of(w.Session).CancelAll();

        AppHooks.LaunchActions["job"] = (w, arg) =>
        {
            w.Session.ShowInspectorPage(arg ?? "optimize");
            return Task.CompletedTask;
        };
        // `variables=c1,c3,t11`: make those values variables (c curvature, t thickness, by surface index).
        AppHooks.LaunchActions["variables"] = (w, arg) =>
        {
            var doc = w.Session.Document;
            foreach (var item in (arg ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                if (item.Length < 2 || !int.TryParse(item[1..], out var i) || i <= 0 || i >= doc.System.Surfaces.Count) continue;
                var id = doc.System.Surfaces[i].Id;
                ViewModels.Jobs.DesignVariables.Set(doc, item[0] == 't' ? new Model.ParamPath.Thickness(id) : new Model.ParamPath.Curvature(id), true);
            }
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["optimize-run"] = async (w, arg) =>
        {
            var s = Open(w, "optimize").Optimize;
            if (arg is { Length: > 0 }) s.SetAlgorithm(arg);
            await WaitSnapshot(w);
            s.Start();
            await WaitUntil(() => !s.IsRunning, 120);
        };
        AppHooks.LaunchActions["optimize-running"] = async (w, arg) =>
        {
            var s = Open(w, "optimize").Optimize;
            await WaitSnapshot(w);
            s.SetAlgorithm("de");
            s.SetIterations(100_000);
            s.Start();
            await Task.Delay(TimeSpan.FromSeconds(double.TryParse(arg, System.Globalization.CultureInfo.InvariantCulture, out var sec) ? sec : 3));
        };
        AppHooks.LaunchActions["optimize-apply"] = (w, _) =>
        {
            Open(w, "optimize").Optimize.Apply();
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["merit-custom"] = async (w, _) =>
        {
            var s = Open(w, "optimize").Optimize;
            await WaitSnapshot(w);
            await s.ConvertToCustomAsync();
            await WaitUntil(() => !s.Evaluating, 30);
        };
        // `glass-run[=N]`: run the substitution and wait; N previews the combination of rank N.
        AppHooks.LaunchActions["glass-run"] = async (w, arg) =>
        {
            var s = Open(w, "glass_substitution").Glass;
            await WaitSnapshot(w);
            s.Settings.MaxCombinations = 24;
            s.Settings.LocalIterations = 20;
            s.Start();
            await WaitUntil(() => !s.IsRunning, 240);
            if (int.TryParse(arg, out var rank)) s.Select(rank);
        };
        AppHooks.LaunchActions["glass-apply"] = (w, _) =>
        {
            Open(w, "glass_substitution").Glass.Apply();
            return Task.CompletedTask;
        };
        // `tolerance-run[=trials]`: run the default set (both analyses) and wait for the results.
        AppHooks.LaunchActions["tolerance-run"] = async (w, arg) =>
        {
            var s = Open(w, "tolerance").Tolerance;
            await WaitSnapshot(w);
            await Task.Delay(1500); // the layout's semi-diameters and the glasses' Abbe numbers
            if (int.TryParse(arg, out var trials)) s.Edit("Change Trials", x => x.Trials = trials);
            s.Start();
            await WaitUntil(() => !s.IsRunning, 300);
        };
        // `tolerance-preview=k`: preview Monte-Carlo trial k of the last run (`tolerance-preview=s0` the first sensitivity).
        AppHooks.LaunchActions["tolerance-preview"] = async (w, arg) =>
        {
            var s = Open(w, "tolerance").Tolerance;
            if (s.Result is not { } r) return;
            ViewModels.Jobs.TolerancePreviewCase c = arg is ['s', ..] && int.TryParse(arg[1..], out var o)
                ? new ViewModels.Jobs.TolerancePreviewCase.Sensitivity(r.Sensitivities.OrderByDescending(r.Share).ElementAt(o).Id, true)
                : new ViewModels.Jobs.TolerancePreviewCase.Trial(int.TryParse(arg, out var k) ? k : r.TrialsWorstFirst.FirstOrDefault());
            await s.PreviewAsync(c);
        };
        // `add-solve`: a marginal-ray-height solve on the back focus (screenshots of Apply Solves).
        AppHooks.LaunchActions["add-solve"] = (w, _) =>
        {
            var doc = w.Session.Document;
            var target = new Model.ParamPath.Thickness(doc.System.Surfaces[doc.System.ImageIndex - 1].Id);
            doc.Edit(Loc.S("Jobs_Registration_AddSolve"), s => s with { Solves = s.Solves.Add(new Model.Solve(target, new Model.SolveKind.MarginalRayHeight(0))) });
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["solves-run"] = async (w, _) =>
        {
            var s = Open(w, "solves").Solves;
            await WaitSnapshot(w);
            await s.DryRunAsync();
        };
    }

    static DesignJobs Open(DesignWindow w, string page)
    {
        w.Session.ShowInspectorPage(page);
        return DesignJobs.Of(w.Session);
    }

    static Task WaitSnapshot(DesignWindow w) => WaitUntil(() => w.Session.Workspace.DesignSnapshot is not null, 30);

    internal static async Task WaitUntil(Func<bool> done, double seconds)
    {
        var end = DateTime.UtcNow.AddSeconds(seconds);
        while (!done() && DateTime.UtcNow < end) await Task.Delay(100);
    }
}
