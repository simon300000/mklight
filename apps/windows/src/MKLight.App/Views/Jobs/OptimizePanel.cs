using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Optimize page of the inspector (the Mac app's <c>OptimizePanel</c>): the merit function, the variables and
/// their bounds and the algorithm; while running the iteration, the search merit and a log-scale sparkline; the
/// result previewed in the window with the design's and the candidate's spot numbers and layouts side by side, every variable
/// before → after, and Apply to Design as one undoable step.
/// </summary>
public sealed partial class OptimizePanel : UserControl
{
    readonly OptimizeSession _s;
    readonly JobPanelFrame _frame = new("Optimize", "\uE9D9");
    readonly MeritEditor _merit;
    bool _queued;
    JobPhase? _builtPhase;
    StackPanel? _comparison;

    public OptimizePanel(DesignSession session)
    {
        _s = DesignJobs.Of(session).Optimize;
        _merit = new MeritEditor(_s);
        Content = _frame;
        _s.Changed += Queue;
        _s.EvaluationChanged += () => { if (_s.Phase == JobPhase.Setup) _merit.UpdateEvaluation(); };
        session.Workspace.StateChanged += () => { if (_comparison is not null) FillComparison(_comparison); };
        session.Workspace.PreviewChanged += Queue;
        Loaded += (_, _) => { Rebuild(); _s.ScheduleEvaluation(); };
        KeyDown += OnKey;
    }

    void OnKey(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == global::Windows.System.VirtualKey.Escape && _s.IsRunning) { _s.Cancel(); e.Handled = true; }
    }

    /// <summary>Rebuilds once per UI turn, however many changes arrived.</summary>
    void Queue()
    {
        if (_queued) return;
        _queued = true;
        DispatcherQueue.TryEnqueue(() => { _queued = false; Rebuild(); });
    }

    void Rebuild()
    {
        var focused = XamlRoot is null ? null : FocusManager.GetFocusedElement(XamlRoot) as NumberField;
        var key = focused?.Key;
        var offset = _builtPhase == _s.Phase ? _frame.Scroller.VerticalOffset : 0;
        _builtPhase = _s.Phase;
        _frame.Subtitle = _s.Document.System.Name.Length > 0 ? _s.Document.System.Name : "Untitled design";
        _comparison = null;
        _frame.Body = _s.Phase switch
        {
            JobPhase.Running => Running(),
            JobPhase.Finished => Result(),
            JobPhase.Failed => JobUI.Failure("Optimization Failed", _s.Failure ?? ""),
            _ => Setup(),
        };
        Actions();
        if (offset > 0) DispatcherQueue.TryEnqueue(() => _frame.Scroller.ChangeView(null, offset, null, disableAnimation: true));
        if (key is not null) DispatcherQueue.TryEnqueue(() => Find(_frame.Body as DependencyObject, key)?.Focus(FocusState.Programmatic));
    }

    static NumberField? Find(DependencyObject? root, string key)
    {
        if (root is null) return null;
        if (root is NumberField f && f.Key == key) return f;
        var n = Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
            if (Find(Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChild(root, i), key) is { } hit) return hit;
        return null;
    }

    // ---- Setup ----

    UIElement Setup()
    {
        var p = new StackPanel { Spacing = 16, Padding = new Thickness(0, 0, 0, 8) };
        p.Children.Add(_merit.Build());

        var vars = _s.Variables;
        var v = new StackPanel { Spacing = 8 };
        var design = new RadioButton { Content = $"Design variables ({vars.Count})", IsChecked = _s.UseDesignVariables, GroupName = "vary", IsEnabled = vars.Count > 0, MinWidth = 0 };
        var auto = new RadioButton { Content = "Every curvature and the back focus", IsChecked = !_s.UseDesignVariables, GroupName = "vary", MinWidth = 0 };
        design.Checked += (_, _) => { if (!_s.UseDesignVariables) _s.SetUseDesignVariables(true); };
        auto.Checked += (_, _) => { if (_s.UseDesignVariables) _s.SetUseDesignVariables(false); };
        v.Children.Add(JobUI.Column(0, design, auto));
        if (_s.UseDesignVariables && vars.Count > 0) v.Children.Add(VariablesBoundsList.Build(_s.Document, Rebuild));
        else if (vars.Count == 0)
            v.Children.Add(JobUI.Hint("Mark values with V in the lens data (right-click a value, or Ctrl+Shift+V) to choose them; each gets sensible bounds you can edit here."));
        if (_s.Problem is { } why) v.Children.Add(JobUI.Warning(why));
        p.Children.Add(JobUI.Group("Variables", v));

        var a = new StackPanel { Spacing = 6 };
        var algorithms = OptimizeSession.Algorithms;
        a.Children.Add(JobUI.Labeled("Method", JobUI.Choice(algorithms.Select(x => x.Title).ToList(), Array.FindIndex(algorithms, x => x.Id == _s.Algorithm),
            k => _s.SetAlgorithm(algorithms[k].Id), 150)));
        a.Children.Add(JobUI.Labeled("Iterations", new NumberField("iterations", () => _s.MaxIterations, x => _s.SetIterations(x is { } d ? (int)Math.Round(d) : null),
            90, "default", MeritFormat.Short, name: "Iterations")));
        if (_s.Stochastic)
            a.Children.Add(JobUI.Labeled("Seed", new NumberField("seed", () => _s.Seed, x => _s.SetSeed((int)Math.Round(x ?? 0)), 90, "0", MeritFormat.Short, optional: false, name: "Seed")));
        p.Children.Add(JobUI.Group("Algorithm", a));
        if (_s.Workspace.ModelErrors.Count > 0) p.Children.Add(JobUI.Warning("Fix the design's errors first."));
        return p;
    }

    // ---- Running ----

    UIElement Running()
    {
        var p = new StackPanel { Spacing = 14 };
        var last = _s.Trace.Count > 0 ? _s.Trace[^1] : (MeritPoint?)null;
        var population = _s.Algorithm is "de" or "cma_es" or "nelder_mead";
        p.Children.Add(JobUI.Row(28,
            JobUI.BigNumber("Iteration", last is { } l ? MeritFormat.Integer(l.Iteration) : "—"),
            JobUI.BigNumber("Search merit", last is { } m ? MeritFormat.Merit(m.Merit) : "—",
                population ? "The search's merit, including the penalty that holds the focal length; the result reports the merit itself."
                    : "The best merit so far, as the search evaluates it.")));
        if (_s.Trace.Count > 0 && _s.Trace[0].Merit > 0)
            p.Children.Add(JobUI.BigNumber("Improvement", MeritFormat.Percent(1 - _s.Trace[^1].Merit / _s.Trace[0].Merit)));
        p.Children.Add(new Sparkline(120) { Values = _s.Trace.Select(t => t.Merit).ToList() });
        p.Children.Add(JobUI.Hint("Stop keeps the best design so far. Cancel discards the run. You can close this panel; the job continues."));
        return p;
    }

    // ---- Result ----

    UIElement Result()
    {
        var p = new StackPanel { Spacing = 14 };
        if (_s.Result is not { } r) return p;
        var system = _s.Document.System;
        if (_s.Applied) p.Children.Add(JobUI.AppliedNotice());
        p.Children.Add(JobUI.Row(28, JobUI.BigNumber("Merit before", MeritFormat.Merit(r.StartMerit)), JobUI.BigNumber("After", MeritFormat.Merit(r.FinalMerit))));
        p.Children.Add(JobUI.BigNumber("Improvement", MeritFormat.Percent(r.Improvement)));
        if (!_s.Applied && r.System is not null && r.Improved)
        {
            var preview = JobUI.Check("Preview in the window", _s.Workspace.IsPreviewing, _s.SetPreview,
                "Show the optimized design in the layout and analyses before applying it");
            p.Children.Add(preview);
            if (_s.Workspace.IsPreviewing)
            {
                _comparison = new StackPanel { Spacing = 4 };
                FillComparison(_comparison);
                p.Children.Add(JobUI.Card(_comparison, 8));
            }
        }
        if (r.System is not null && r.Improved)
            p.Children.Add(new LayoutComparison(_s.Workspace, _s.StartSystem, r.System, afterTitle: _s.Applied ? "Applied" : "Candidate"));
        p.Children.Add(JobUI.Hint(r.StatusText()));
        if (_s.Trace.Count > 1) p.Children.Add(new Sparkline(64) { Values = _s.Trace.Select(t => t.Merit).ToList() });
        p.Children.Add(JobUI.Text("Variables", "PanelTitleText"));
        var list = new StackPanel { Spacing = 8 };
        foreach (var v in r.Variables)
        {
            var head = JobUI.Row(6, JobUI.Text(v.Param is { } path ? path.Label(r.System ?? system) : v.Label));
            if (v.AtBound)
            {
                var b = JobUI.Text("at bound", "CaptionText");
                JobUI.Bind(b, "WarningBrush", x => b.Foreground = x);
                head.Children.Add(b);
            }
            var start = OptimizeResult.Displayed(v.Start, v.Param);
            var end = OptimizeResult.Displayed(v.Value, v.Param);
            var unit = v.Param is { } pp ? (pp.IsCurvature ? "mm" : pp.Unit(system)) : "";
            var values = new Grid();
            var arrow = new FontIcon { Glyph = "\uE72A", FontSize = 10, VerticalAlignment = VerticalAlignment.Center };
            var after = JobUI.Text(MeritFormat.Short(end), "NumberText");
            after.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
            values.Children.Add(JobUI.Row(6, JobUI.Text(MeritFormat.Short(start), "NumberText"), arrow, after));
            var delta = JobUI.Text(MeritFormat.Signed(end - start) + (unit.Length > 0 ? " " + unit : ""), "SecondaryNumberText");
            delta.HorizontalAlignment = HorizontalAlignment.Right;
            values.Children.Add(delta);
            var row = JobUI.Column(1, head, values);
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(row, $"{(v.Param is { } q ? q.Label(system) : v.Label)}: {MeritFormat.Short(start)} to {MeritFormat.Short(end)}");
            list.Children.Add(row);
        }
        p.Children.Add(list);
        if (r.EvaluatedOn.Length > 0)
            p.Children.Add(JobUI.Hint(r.EvaluatedOn.StartsWith("cpu", StringComparison.Ordinal)
                ? "Evaluated on the CPU f64 reference." : $"Searched on {r.EvaluatedOn}; merits reported on the CPU f64 reference.", 11));
        return p;
    }

    /// <summary>The design's and the candidate's spot numbers and merit, side by side.</summary>
    void FillComparison(StackPanel card)
    {
        var r = _s.Result;
        PreviewComparison.Fill(card, _s.Workspace, _s.Workspace.DesignSystem ?? _s.Document.System, "Candidate",
            r is null ? [] : [("Merit", MeritFormat.Merit(r.StartMerit), MeritFormat.Merit(r.FinalMerit))]);
    }

    // ---- Actions ----

    void Actions()
    {
        var cannotRun = _s.Blocked || _s.Problem is not null;
        switch (_s.Phase)
        {
            case JobPhase.Setup:
                _frame.SetActions(null, JobUI.Primary("Optimize", "\uE768", () => _s.Start(), cannotRun ? _s.Problem ?? "Fix the design's errors first." : "Start optimizing (Ctrl+Enter)", !cannotRun));
                break;
            case JobPhase.Failed:
                _frame.SetActions(null,
                    JobUI.Secondary("Back", null, _s.Reset, "Back to the job's settings"),
                    JobUI.Primary("Try Again", "\uE72C", () => _s.Start(fresh: true), "Run the job again with the same settings", !cannotRun));
                break;
            case JobPhase.Running:
                _frame.SetActions(JobUI.Progress(_s.Progress, _s.Trace.Count > 0 ? $"Iteration {_s.Trace[^1].Iteration}" : ""),
                    JobUI.Secondary("Cancel", null, _s.Cancel, "End the run and discard it (Esc)", destructive: true),
                    JobUI.Secondary("Stop", "\uE71A", _s.Stop, "End now and keep the best design so far", _s.CanStop));
                break;
            default:
                if (_s.Applied) _frame.SetActions(null, JobUI.Primary("Done", null, _s.Reset, "Back to the merit function and variables"));
                else
                    _frame.SetActions(null,
                        JobUI.Secondary("Discard", null, _s.Reset, "Drop this result; the design stays as it is"),
                        JobUI.Primary("Apply to Design", null, _s.Apply, "Apply the result as one step you can undo (Ctrl+Enter)",
                            _s.Result?.System is not null && _s.Result.Improved));
                break;
        }
    }
}
