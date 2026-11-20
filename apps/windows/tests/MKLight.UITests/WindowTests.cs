using System.Drawing;
using FlaUI.Core.AutomationElements;

namespace MKLight.UITests;

/// <summary>Menus, the panel arrangement, the layout's 3-D view and the inspector.</summary>
public sealed class WindowTests
{
    [Fact]
    public void LensMenuCommandOpensTheGlassCatalogs()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            ui.OpenMenu(() => Ui.Open(ui.MenuBarItem("Lens")),"Glass Catalogs…").Patterns.Invoke.Pattern.Invoke();
            Ui.Wait(() => app.TopLevelWindows().FirstOrDefault(w => Ui.NameOf(w).Contains("Glass", StringComparison.Ordinal)), "the Glass Catalogs window");
        });
    }

    [Fact]
    public void PanelMenuMovesAPaneToANewWindow()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var windows = app.TopLevelWindows().Length;
            ui.OpenMenu(() => Ui.ShowContextMenu(ui.PaneControl("mtf", "Title")), "Move to New Window").Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => app.TopLevelWindows().Length > windows, "a new window for the MTF pane");
            Ui.WaitUntil(() => ui.PaneOrNull("mtf") is null, "MTF to leave the main window");
        });
    }

    [Fact]
    public void ThreeDButtonSwitchesTheLayout()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            Ui.WaitUntil(() => Ui.NameOf(ui.LayoutCanvas).Contains("meridional section", StringComparison.Ordinal), "the section view");
            Assert.True(ui.LayoutMode("section").Patterns.SelectionItem.Pattern.IsSelected.Value);
            ui.LayoutMode("3d").Patterns.SelectionItem.Pattern.Select();
            Ui.WaitUntil(() => Ui.NameOf(ui.LayoutCanvas).Contains("3D view", StringComparison.Ordinal), "the layout canvas to name the 3D view");
            Ui.WaitUntil(() => ui.LayoutMode("3d").Patterns.SelectionItem.Pattern.IsSelected.Value, "the 3D mode to be selected");
        });
    }

    [Fact]
    public void InspectorPagesSwitch()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            ui.SelectLensRow(3);
            Ui.WaitUntil(() => ui.InspectorShowsSurface(3), "the surface page for surface 3");
            ui.FocusAway();

            ui.InspectorTab("system").Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => !ui.InspectorShowsSurface(3), "the System page");
            var system = ui.InspectorPageTexts();

            ui.InspectorTab("optimize").Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => !ui.InspectorPageTexts().SequenceEqual(system), "the Optimize page");

            ui.InspectorTab("surface").Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => ui.InspectorShowsSurface(3), "the Surface page again");
        });
    }

    // ---- Posted mouse messages (skipped while WinUI ignores them) ----

    [Fact]
    public void PostedDragOfAPaneHeaderOntoTheLensDataSplitsIt()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var lens = Ui.Bounds(ui.LensData);
            var title = Ui.Bounds(ui.PaneControl("mtf", "Title"));
            Assert.True(title.Y > lens.Bottom, "MTF starts below the lens data");

            // Onto the lens data's right edge (its outer 30 %: a split to the right).
            var from = new Point(title.X + title.Width / 2, title.Y + title.Height / 2);
            var to = new Point(lens.Right - lens.Width / 12, lens.Y + lens.Height / 2);
            app.Mouse.Drag(from, to, steps: 16);

            // The arrangement, read from the automation tree: MTF now sits beside the lens data, which gave up width.
            WindowMouse.ExpectEffect(() => Ui.Bounds(ui.PaneControl("mtf", "Title")).Y < Ui.Bounds(ui.LensData).Bottom,
                "drag of the MTF pane's title", nameof(PanelMenuMovesAPaneToANewWindow));
            var after = Ui.Bounds(ui.LensData);
            Assert.True(after.Width < lens.Width * 0.8, $"the lens data is narrower ({lens.Width} → {after.Width})");
            Assert.True(Ui.Bounds(ui.PaneControl("mtf", "Title")).X >= after.Right - 16, "MTF is right of the lens data");
        });
    }

    [Fact]
    public void PostedDragOrbitsThe3DLayout()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            ui.LayoutMode("3d").Patterns.SelectionItem.Pattern.Select();
            Thread.Sleep(1500); // the first 3-D frames
            var region = CanvasRegion(ui);
            using var before = WindowImage.CaptureRegion(app.Hwnd, region);
            var canvas = Ui.Bounds(ui.LayoutCanvas);
            var c = new Point(canvas.X + canvas.Width / 2, canvas.Y + canvas.Height / 2);
            app.Mouse.Drag(c, new Point(c.X + canvas.Width / 4, c.Y + canvas.Height / 10), steps: 10);
            WindowMouse.ExpectEffect(() =>
            {
                using var after = WindowImage.CaptureRegion(app.Hwnd, region);
                return WindowImage.Difference(before, after) > 0.01;
            }, "drag on the 3-D layout", "no UI Automation equivalent (the camera has no automation surface)");
        });
    }

    [Fact]
    public void PostedWheelZoomsTheSection()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var scale = ui.ScaleBarName();
            var canvas = Ui.Bounds(ui.LayoutCanvas);
            var c = new Point(canvas.X + canvas.Width / 2, canvas.Y + canvas.Height / 2);
            for (var i = 0; i < 6; i++) app.Mouse.Wheel(c, 120);
            WindowMouse.ExpectEffect(() => ui.ScaleBarName() != scale, "mouse wheel on the layout", "no UI Automation equivalent (zoom has no automation surface)");
        });
    }

    /// <summary>The middle of the layout canvas, clear of its overlaid controls.</summary>
    static Rectangle CanvasRegion(Ui ui)
    {
        var canvas = Ui.Bounds(ui.LayoutCanvas);
        return Rectangle.Inflate(canvas, -canvas.Width / 10, -canvas.Height / 6);
    }
}
