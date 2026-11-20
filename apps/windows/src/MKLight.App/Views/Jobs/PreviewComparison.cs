using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.ViewModels;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Model;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The design's and the previewed candidate's numbers side by side, so it is plain which design every number belongs
/// to: the spot diagram's key numbers (the design's from <see cref="Workspace.Baseline"/>, taken just before the
/// preview; the candidate's from the pane now) and the job's own rows (the merit).
/// </summary>
public static class PreviewComparison
{
    /// <summary>Fills <paramref name="card"/>; <paramref name="extra"/> adds the job's rows (label, design, candidate).</summary>
    public static void Fill(StackPanel card, Workspace ws, OpticalSystem design, string candidateTitle, IEnumerable<(string Label, string Design, string Candidate)> extra)
    {
        card.Children.Clear();
        var pane = ws.Panes.FirstOrDefault(p => p.Analysis == "spot_diagram");
        var head = new Grid { ColumnSpacing = 8 };
        Columns(head);
        head.Children.Add(JobUI.Text(pane is null ? "Design and candidate" : "Spot diagram", "PanelTitleText", 12));
        Cell(head, JobUI.Text("Design", "CaptionText"), 1);
        var cand = JobUI.Text(candidateTitle, "CaptionText");
        JobUI.Bind(cand, "AccentTextFillColorPrimaryBrush", b => cand.Foreground = b);
        Cell(head, cand, 2);
        card.Children.Add(head);
        if (pane is not null)
        {
            IReadOnlyList<(string Label, string Value)> Keys(Engine.AnalysisOutput? o, OpticalSystem s) =>
                o is null ? [] : KeyNumbers.Of(ResultHosting.Data("spot_diagram", o, s));
            var before = Keys(ws.Baseline.GetValueOrDefault(pane.Id), design);
            var candidate = !ws.IsPreviewing || pane.IsRunning || pane.IsStale ? [] : Keys(pane.Output, ws.PreviewSystem ?? design);
            foreach (var label in before.Select(k => k.Label).Concat(candidate.Select(k => k.Label)).Distinct())
                Row(card, label, before.FirstOrDefault(k => k.Label == label).Value ?? "—",
                    candidate.FirstOrDefault(k => k.Label == label).Value ?? (pane.IsRunning ? "…" : "—"));
            if (before.Count == 0 && candidate.Count == 0)
                card.Children.Add(JobUI.Hint(pane.IsRunning ? "Computing the candidate's spot diagram…" : "The spot diagram has no result yet.", 11.5));
        }
        else card.Children.Add(JobUI.Hint("Open a Spot Diagram to compare its numbers here.", 11.5));
        foreach (var (label, d, c) in extra) Row(card, label, d, c);
    }

    static void Row(StackPanel card, string label, string design, string candidate)
    {
        var row = new Grid { ColumnSpacing = 8 };
        Columns(row);
        row.Children.Add(JobUI.Text(label, "SecondaryText"));
        Cell(row, JobUI.Text(design, "NumberText"), 1);
        var c = JobUI.Text(candidate, "NumberText");
        c.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
        Cell(row, c, 2);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(row, $"{label}: design {design}, candidate {candidate}");
        card.Children.Add(row);
    }

    static void Columns(Grid g)
    {
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(72) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(76) });
    }

    static void Cell(Grid g, FrameworkElement e, int column)
    {
        e.HorizontalAlignment = HorizontalAlignment.Right;
        Grid.SetColumn(e, column);
        g.Children.Add(e);
    }
}
