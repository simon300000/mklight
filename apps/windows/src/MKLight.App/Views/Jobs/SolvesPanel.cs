using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Apply Solves page (the Mac app's <c>SolvesPanel</c>): a dry run of the design's solves (target, kind,
/// now → solved, residual) that changes nothing, then Apply to Design as one undoable commit.
/// </summary>
public sealed partial class SolvesPanel : UserControl
{
    readonly SolvesSession _s;
    readonly DesignSession _session;
    readonly JobPanelFrame _frame = new(Loc.S("Jobs_Solves_Title"), "\uE943");
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
        _frame.Subtitle = Loc.F(n == 1 ? "Jobs_Solves_Subtitle_One" : "Jobs_Solves_Subtitle_Many", n, sys.Name.Length > 0 ? sys.Name : Loc.S("Jobs_Solves_TheDesign"));
        if (n == 0 && !_s.Committed)
        {
            _frame.Body = JobUI.Empty("\uE943", Loc.S("Jobs_Solves_EmptyTitle"), Loc.S("Jobs_Solves_EmptyDescription"));
            _frame.SetActions(null, JobUI.Primary(Loc.S("Jobs_Controls_ApplyToDesign"), null, () => { }, null, false));
            return;
        }
        switch (_s.Phase)
        {
            case JobPhase.Failed:
                _frame.Body = JobUI.Failure(Loc.S("Jobs_Solves_FailedTitle"), EngineText.T(_s.Failure ?? ""));
                _frame.SetActions(null, JobUI.Primary(Loc.S("Jobs_Controls_TryAgain"), "\uE72C", () => _ = _s.DryRunAsync(fresh: true), Loc.S("Jobs_Controls_TryAgainHelp")));
                return;
            case JobPhase.Setup or JobPhase.Running:
                var busy = JobUI.Column(10, new ProgressRing { IsActive = true, Width = 24, Height = 24 }, JobUI.Text(Loc.S("Jobs_Solves_Solving"), "SecondaryText"));
                busy.HorizontalAlignment = HorizontalAlignment.Center;
                busy.Margin = new Thickness(0, 40, 0, 0);
                ((FrameworkElement)busy.Children[1]).HorizontalAlignment = HorizontalAlignment.Center;
                _frame.Body = busy;
                _frame.SetActions(null, JobUI.Primary(Loc.S("Jobs_Controls_ApplyToDesign"), null, () => { }, null, false));
                return;
        }
        _frame.Body = Report();
        if (_s.Committed)
            _frame.SetActions(null, JobUI.Primary(Loc.S("Jobs_Controls_Done"), null, () => { _dirty = true; Run(); }, Loc.S("Jobs_Solves_DoneHelp")));
        else
            _frame.SetActions(null, JobUI.Primary(Loc.S("Jobs_Controls_ApplyToDesign"), null, () => _ = _s.CommitAsync(),
                Loc.S(_s.Changes == 0 ? "Jobs_Solves_ApplyHelpNoChange" : "Jobs_Solves_ApplyHelp"), _s.Changes > 0));
    }

    UIElement Report()
    {
        var p = new StackPanel { Spacing = 10 };
        if (_s.Committed) p.Children.Add(JobUI.AppliedNotice());
        p.Children.Add(JobUI.Hint(Loc.S(_s.Committed ? "Jobs_Solves_CommittedHint" : "Jobs_Solves_DryRunHint")));
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
            var arrow = new FontIcon { Glyph = "\uE72A", FontSize = 10, VerticalAlignment = VerticalAlignment.Center };
            var after = JobUI.Text(MeritFormat.Short(o.Value), "NumberText");
            if (o.Changes) after.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
            values.Children.Add(JobUI.Row(6, JobUI.Text(o.Previous is { } pv ? MeritFormat.Short(pv) : "—", "NumberText"), arrow, after));
            var residual = JobUI.Text(Loc.F("Jobs_Solves_Residual", MeritFormat.Short(o.Residual)), "CaptionText");
            residual.HorizontalAlignment = HorizontalAlignment.Right;
            values.Children.Add(residual);
            var row = JobUI.Column(2, head, values);
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(row,
                Loc.F("Jobs_Solves_RowName", o.Target, o.Kind, o.Previous is { } q ? MeritFormat.Short(q) : Loc.S("Jobs_Solves_None"), MeritFormat.Short(o.Value), MeritFormat.Short(o.Residual)));
            list.Children.Add(row);
        }
        p.Children.Add(JobUI.Card(list));
        foreach (var w in _s.Warnings) p.Children.Add(JobUI.Warning(EngineText.T(w)));
        return p;
    }
}
