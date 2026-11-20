using FlaUI.Core.AutomationElements;
using FlaUI.Core.Definitions;

namespace MKLight.UITests;

/// <summary>Analysis panes: the sidebar opens them, Add Analysis adds them, Run updates them, the settings form changes them.</summary>
public sealed class PaneTests
{
    [Fact]
    public void SidebarSelectionOpensAnAnalysis()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            Assert.Null(ui.PaneWithTitle("Ray Fan"));
            ui.SidebarItem("Ray Fan").Patterns.SelectionItem.Pattern.Select();
            Ui.Wait(() => ui.PaneWithTitle("Ray Fan"), "a Ray Fan pane");
        });
    }

    [Fact]
    public void AddAnalysisMenuAddsAPane()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var panes = ui.PaneCount();
            var geometric = ui.OpenMenu(() => ui.AddAnalysisButton.Patterns.ExpandCollapse.Pattern.Expand(), "Geometric");
            ui.OpenMenu(() => geometric.Patterns.ExpandCollapse.Pattern.Expand(), "Seidel Coefficients").Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => ui.PaneCount() == panes + 1, "a new pane");
            Assert.Contains("Seidel", Ui.NameOf(ui.PaneControl(ui.PaneIds().Except(["mtf", "spot_diagram"]).Single(), "Title")));
        });
    }

    [Fact]
    public void RunUpdatesAStalePane()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            const string stale = "This result is from an earlier design.";
            // Manual updates for MTF, then change the design: the pane goes stale.
            var auto = ui.PaneControl("mtf", "AutoUpdate");
            Assert.Equal(ToggleState.On, auto.Patterns.Toggle.Pattern.ToggleState.Value);
            auto.Patterns.Toggle.Pattern.Toggle();
            Ui.WaitUntil(() => auto.Patterns.Toggle.Pattern.ToggleState.Value == ToggleState.Off, "automatic updates off");

            ui.LensCell(3, "radius").Patterns.Value.Pattern.SetValue("21");
            Ui.WaitUntil(() => ui.ShowsText(stale), "the MTF pane to go stale");

            ui.PaneControl("mtf", "Run").Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => !ui.ShowsText(stale), "Run to bring the MTF pane up to date", TimeSpan.FromSeconds(60));
        });
    }

    [Fact]
    public void SettingsFormChangesTheMtfMaxFrequency()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var state = () => Ui.NameOf(ui.PaneControl("mtf", "State"));
            Assert.DoesNotContain("80", state());
            ui.PaneControl("mtf", "Settings").Patterns.Toggle.Pattern.Toggle();
            var field = ui.SettingsField("Max Frequency");
            field.Focus();
            field.Patterns.Value.Pattern.SetValue("80");
            ui.FocusAway(); // the field commits as it loses the focus
            Ui.WaitUntil(() => state().Contains("80", StringComparison.Ordinal), "the MTF pane's state line to show 80 cycles/mm", TimeSpan.FromSeconds(60));
        });
    }

    [Fact]
    public void PostedClickOnTheSidebarOpensAnAnalysis()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            app.Mouse.Click(ui.SidebarItem("Ray Fan"));
            WindowMouse.ExpectEffect(() => ui.PaneWithTitle("Ray Fan") is not null, "click on a sidebar entry", nameof(SidebarSelectionOpensAnAnalysis));
        });
    }
}
