using MKLight.App.Views.Analysis;

namespace MKLight.Tests.Panels;

public class PaneChromeTests
{
    static PanePlaceholderKind Kind(PaneFacts f) => AnalysisPaneChrome.Placeholder(f).Kind;

    [Fact]
    public void PlaceholdersFollowTheMacOrder()
    {
        Assert.Equal(PanePlaceholderKind.None, Kind(new() { HasOutput = true }));
        Assert.Equal(PanePlaceholderKind.Running, Kind(new() { Running = true }));
        Assert.Equal(PanePlaceholderKind.Unavailable, Kind(new() { Unavailable = true }));
        Assert.Equal(PanePlaceholderKind.Unavailable, Kind(new() { Unavailable = true, HasOutput = true }));
        Assert.Equal(PanePlaceholderKind.Failed, Kind(new() { Failed = true, Message = "x" }));
        Assert.Equal(PanePlaceholderKind.DesignHasErrors, Kind(new() { Failed = true, DesignHasErrors = true }));
        Assert.Equal(PanePlaceholderKind.NotReady, Kind(new() { ModelError = "bad", Cancelled = true }));
        Assert.Equal(PanePlaceholderKind.Cancelled, Kind(new() { Cancelled = true }));
        Assert.Equal(PanePlaceholderKind.Waiting, Kind(new() { AutoUpdate = true }));
        Assert.Equal(PanePlaceholderKind.OnRequest, Kind(new() { AutoUpdate = false }));
    }

    [Fact]
    public void PlaceholderWordingMatchesTheMac()
    {
        var p = AnalysisPaneChrome.Placeholder(new() { Unavailable = true });
        Assert.Equal(("Not Available", "This engine build does not provide this analysis."), (p.Title, p.Text));
        Assert.Null(p.Action);
        p = AnalysisPaneChrome.Placeholder(new() { Failed = true, Message = "Ray missed." });
        Assert.Equal(("Analysis Failed", "Ray missed.", "Try Again"), (p.Title, p.Text, p.Action));
        p = AnalysisPaneChrome.Placeholder(new() { Failed = true, DesignHasErrors = true, Message = "m" });
        Assert.Equal(("Design Has Errors", "Show Issues"), (p.Title, p.Action));
        p = AnalysisPaneChrome.Placeholder(new() { ModelError = "Surface 4 is bad" });
        Assert.Equal(("Design Not Ready", "Surface 4 is bad", "Show Issues"), (p.Title, p.Text, p.Action));
        p = AnalysisPaneChrome.Placeholder(new() { Cancelled = true });
        Assert.Equal(("Cancelled", "Run"), (p.Title, p.Action));
        p = AnalysisPaneChrome.Placeholder(new() { Running = true });
        Assert.Equal(("Computing…", "Cancel"), (p.Text, p.Action));
        Assert.Equal("Tracing", AnalysisPaneChrome.Placeholder(new() { Running = true, Stage = "Tracing" }).Text);
    }

    [Fact]
    public void FailedRerunKeepsTheResult()
    {
        Assert.True(AnalysisPaneChrome.RerunFailed(new() { HasOutput = true, Failed = true }));
        Assert.False(AnalysisPaneChrome.RerunFailed(new() { HasOutput = false, Failed = true }));
        Assert.False(AnalysisPaneChrome.RerunFailed(new() { HasOutput = true }));
        // A failed rerun over a result is not a placeholder.
        Assert.Equal(PanePlaceholderKind.None, Kind(new() { HasOutput = true, Failed = true }));
    }

    [Fact]
    public void PulseOnlyWhileUpdatingAResultWithMotion()
    {
        var updating = new PaneFacts { HasOutput = true, Running = true };
        Assert.True(AnalysisPaneChrome.Pulses(updating, motion: true));
        Assert.False(AnalysisPaneChrome.Pulses(updating, motion: false));
        Assert.False(AnalysisPaneChrome.Pulses(new() { Running = true }, motion: true));
        Assert.False(AnalysisPaneChrome.Pulses(new() { HasOutput = true }, motion: true));
    }

    [Fact]
    public void HeaderFoldsClusterFirstThenShortensTheTitle()
    {
        // Title 150 (short 40), cluster 112 (compact 56), chrome 68.
        (bool, bool) Fold(double w) => AnalysisPaneChrome.Fold(w, 150, 40, 112, 56, 68);
        Assert.Equal((false, false), Fold(400));
        Assert.Equal((true, false), Fold(300));   // 330 > 300 ≥ 274
        Assert.Equal((true, true), Fold(250));    // the title no longer fits beside Run and More
        Assert.Equal((false, false), Fold(0));    // not measured yet
        // Below the fold width the cluster folds even when everything would fit.
        Assert.Equal((true, false), AnalysisPaneChrome.Fold(230, 60, 30, 112, 56, 50));
        // No shorter title: keep the full one (it truncates).
        Assert.Equal((true, false), AnalysisPaneChrome.Fold(150, 150, 150, 112, 56, 68));
    }

    [Fact]
    public void AutomationIdsAndNames()
    {
        Assert.Equal("Pane.spot_diagram.Header", AnalysisPaneChrome.AutomationId("spot_diagram", "Header"));
        Assert.Equal("Previewing Candidate 3 · not applied", AnalysisPaneChrome.PreviewLabel("Candidate 3"));
        Assert.Equal("MTF pane", AnalysisPaneChrome.AccessibleName("MTF", null));
        Assert.Equal("MTF pane, Previewing Trial 2 · not applied", AnalysisPaneChrome.AccessibleName("MTF", "Trial 2"));
    }
}
