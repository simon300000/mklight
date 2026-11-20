using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Apply Solves page (the Mac app's <c>SolvesPanel</c>): a dry run of the design's solves (target, kind,
/// now → solved, residual) that changes nothing, then Apply to Design as one undoable commit.
/// </summary>
public sealed partial class SolvesPanel : UserControl
{
    readonly SolvesSession _s;
    readonly DesignSession _session;
    readonly JobPanelFrame _frame = new("Apply Solves", "");
    bool _queued;
    bool _dirty = true;

    public SolvesPanel(DesignSession session)
    {
        _session = session;
        _s = DesignJobs.Of(session).Solves;
        Content = _frame;
        _s.Changed += Queue;
        // A changed design needs a new dry run (unless the change was the commit itself).
        session.Workspace.StateChanged += () =>
        {
            if (_s.Committed || _s.Phase == JobPhase.Running || session.Workspace.DesignSnapshot is null) return;
            if (_lastSnapshot != session.Workspace.DesignSnapshot.Hash.Hex) { _dirty = true; if (IsLoaded && Visibility == Visibility.Visible) Run(); }
        };
        Loaded += (_, _) => { if (_s.Committed || _dirty) Run(); Rebuild(); };
        RegisterPropertyChangedCallback(VisibilityProperty, (_, _) => { if (Visibility == Visibility.Visible && (_dirty || _s.Committed)) Run(); });
    }

    string? _lastSnapshot;

    void Run()
    {
        if (_session.Document.System.Solves.Count == 0) { Rebuild(); return; }
        _dirty = false;
        _lastSnapshot = _session.Workspace.DesignSnapshot?.Hash.Hex;
        if (_lastSnapshot is null) return;
        _ = _s.DryRunAsync();
    }

    void Queue()
    {
        if (_queued) return;
        _queued = true;
        DispatcherQueue.TryEnqueue(() => { _queued = false; Rebuild(); });
    }

    void Rebuild()
    {
        var sys = _session.Document.System;
        var n = sys.Solves.Count;
        _frame.Subtitle = $"{n} solve{(n == 1 ? "" : "s")} in {(sys.Name.Length > 0 ? sys.Name : "the design")}";
        if (n == 0 && !_s.Committed)
        {
            _frame.Body = JobUI.Empty("", "No Solves",
                "Add solves in the surface inspector: marginal or chief ray height on a thickness, angle or F-number on a curvature, pickups.");
            _frame.SetActions(null, JobUI.Primary("Apply to Design", null, () => { }, null, false));
            return;
        }
        switch (_s.Phase)
        {
            case JobPhase.Failed:
                _frame.Body = JobUI.Failure("Solving Failed", _s.Failure ?? "");
                _frame.SetActions(null, JobUI.Primary("Try Again", "", () => _ = _s.DryRunAsync(fresh: true), "Run the job again with the same settings"));
                return;
            case JobPhase.Setup or JobPhase.Running:
                var busy = JobUI.Column(10, new ProgressRing { IsActive = true, Width = 24, Height = 24 }, JobUI.Text("Solving…", "SecondaryText"));
                busy.HorizontalAlignment = HorizontalAlignment.Center;
                busy.Margin = new Thickness(0, 40, 0, 0);
                ((FrameworkElement)busy.Children[1]).HorizontalAlignment = HorizontalAlignment.Center;
                _frame.Body = busy;
                _frame.SetActions(null, JobUI.Primary("Apply to Design", null, () => { }, null, false));
                return;
        }
        _frame.Body = Report();
        if (_s.Committed)
            _frame.SetActions(null, JobUI.Primary("Done", null, () => { _dirty = true; Run(); }, "Back to the design's solves"));
        else
            _frame.SetActions(null, JobUI.Primary("Apply to Design", null, () => _ = _s.CommitAsync(),
                _s.Changes == 0 ? "The solves change nothing: the design already holds their values" : "Apply the result as one step you can undo (Ctrl+Enter)", _s.Changes > 0));
    }

    UIElement Report()
    {
        var p = new StackPanel { Spacing = 10 };
        if (_s.Committed) p.Children.Add(JobUI.AppliedNotice());
        p.Children.Add(JobUI.Hint(_s.Committed ? "The design now holds these solved values." : "Dry run: the values the solves would set. Nothing has changed yet."));
        var list = new StackPanel { Spacing = 0 };
        for (var k = 0; k < _s.Outcomes.Count; k++)
        {
            var o = _s.Outcomes[k];
            if (k > 0)
            {
                var rule = new Border { Height = 1, Margin = new Thickness(0, 6, 0, 6) };
                JobUI.Bind(rule, "DividerBrush", b => rule.Background = b);
                list.Children.Add(rule);
            }
            var head = new Grid();
            head.Children.Add(JobUI.Text(o.Target));
            var kind = JobUI.Text(o.Kind, "CaptionText");
            kind.HorizontalAlignment = HorizontalAlignment.Right;
            head.Children.Add(kind);
            var values = new Grid();
            var arrow = new FontIcon { Glyph = "", FontSize = 10, VerticalAlignment = VerticalAlignment.Center };
            var after = JobUI.Text(MeritFormat.Short(o.Value), "NumberText");
            if (o.Changes) after.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
            values.Children.Add(JobUI.Row(6, JobUI.Text(o.Previous is { } pv ? MeritFormat.Short(pv) : "—", "NumberText"), arrow, after));
            var residual = JobUI.Text("residual " + MeritFormat.Short(o.Residual), "CaptionText");
            residual.HorizontalAlignment = HorizontalAlignment.Right;
            values.Children.Add(residual);
            var row = JobUI.Column(2, head, values);
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(row,
                $"{o.Target}, {o.Kind}: {(o.Previous is { } q ? MeritFormat.Short(q) : "none")} to {MeritFormat.Short(o.Value)}, residual {MeritFormat.Short(o.Residual)}");
            list.Children.Add(row);
        }
        p.Children.Add(JobUI.Card(list));
        foreach (var w in _s.Warnings) p.Children.Add(JobUI.Warning(w));
        return p;
    }
}
