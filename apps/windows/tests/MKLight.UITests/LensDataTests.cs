using FlaUI.Core.AutomationElements;

namespace MKLight.UITests;

/// <summary>The lens data: selecting rows, editing values, the row menu's surface commands, undo and redo.</summary>
public sealed class LensDataTests
{
    [Fact]
    public void RowContextMenuSelectsTheRow()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            Assert.True(ui.ShowsText("No Surface Selected"));
            ui.OpenMenu(() => Ui.ShowContextMenu(ui.LensRow(3)), "Make Aperture Stop"); // the row's menu is open …
            Ui.WaitUntil(() => ui.InspectorShowsSurface(3), "the inspector to show surface 3"); // … and the row is selected
        });
    }

    [Fact]
    public void InspectorRadiusEditUpdatesTheTableAndTheEfl()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var efl = ui.Efl();
            Assert.Equal("19.3010", ui.LensCellText(3, "Radius"));
            Ui.ShowContextMenu(ui.LensRow(3)); // selects surface 3
            Ui.WaitUntil(() => ui.InspectorShowsSurface(3), "the surface page for surface 3");

            var radius = ui.InspectorField("Radius");
            Assert.StartsWith("19.301", radius.Patterns.Value.Pattern.Value.Value);
            radius.Focus();
            radius.Patterns.Value.Pattern.SetValue("21");
            ui.FocusAway(); // the field commits as it loses the focus

            Ui.WaitUntil(() => ui.LensCellText(3, "Radius") == "21.0000", "the new radius in the lens data");
            Ui.WaitUntil(() => ui.Efl() != efl, $"the EFL to change from {efl}");

            ui.Undo.Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => ui.LensCellText(3, "Radius") == "19.3010" && ui.Efl() == efl, "undo to restore the radius and the EFL");
        });
    }

    [Fact]
    public void RowMenuMakesTheStopAndUndoRedoFollow()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            Assert.EndsWith("aperture stop", ui.LensRowName(6));
            Assert.False(ui.Undo.IsEnabled);

            ui.OpenMenu(() => Ui.ShowContextMenu(ui.LensRow(3)), "Make Aperture Stop").Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => ui.LensRowName(3).EndsWith("aperture stop", StringComparison.Ordinal), "surface 3 to become the stop");
            Assert.Equal("Surface 6", ui.LensRowName(6));

            Ui.WaitUntil(() => ui.Undo.IsEnabled, "Undo to be enabled");
            ui.Undo.Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => ui.LensRowName(6).EndsWith("aperture stop", StringComparison.Ordinal), "undo to put the stop back on 6");
            Assert.Equal("Surface 3", ui.LensRowName(3));

            Ui.WaitUntil(() => ui.Redo.IsEnabled, "Redo to be enabled");
            ui.Redo.Patterns.Invoke.Pattern.Invoke();
            Ui.WaitUntil(() => ui.LensRowName(3).EndsWith("aperture stop", StringComparison.Ordinal), "redo to move the stop to 3 again");
        });
    }

    [Fact]
    public void RowMenuInsertsASurface()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var before = ui.LensCellText(4, "Comment");
            ui.OpenMenu(() => Ui.ShowContextMenu(ui.LensRow(4)), "Insert Surface Before").Patterns.Invoke.Pattern.Invoke();
            // The new surface takes number 4; the old one moves down, and there are 12.
            Ui.WaitUntil(() => ui.LensCellText(5, "Comment") == before, "the surfaces below to move down");
            Assert.Equal("Surface 12", Ui.NameOf(ui.LensRow(12)));
        });
    }

    // ---- Posted mouse messages (skipped while WinUI ignores them) ----

    [Fact]
    public void PostedClickSelectsARow()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            app.Mouse.Click(ui.LensCell(3, "Comment"));
            WindowMouse.ExpectEffect(() => ui.InspectorShowsSurface(3), "click on a lens row", nameof(RowContextMenuSelectsTheRow));
        });
    }

    [Fact]
    public void PostedDoubleClickEditsACell()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            var efl = ui.Efl();
            app.Mouse.DoubleClick(ui.LensCell(3, "Radius"));
            WindowMouse.ExpectEffect(() => ui.LensEditorOrNull() is not null, "double click on a lens cell", nameof(InspectorRadiusEditUpdatesTheTableAndTheEfl));
            var editor = ui.LensEditor;
            Assert.StartsWith("19.301", editor.Patterns.Value.Pattern.Value.Value);
            editor.Patterns.Value.Pattern.SetValue("21");
            ui.FocusAway();
            Ui.WaitUntil(() => ui.LensCellText(3, "Radius") == "21.0000", "the new radius in the lens data");
            Ui.WaitUntil(() => ui.Efl() != efl, "the EFL to change");
        });
    }

    [Fact]
    public void PostedRightClickOpensTheRowMenu()
    {
        using var app = AppSession.Start();
        app.Run(() =>
        {
            var ui = app.Ui;
            app.Mouse.RightClick(ui.LensCell(3, "Comment"));
            WindowMouse.ExpectEffect(() => ui.IsMenuOpen("Make Aperture Stop"), "right click on a lens row", nameof(RowMenuMakesTheStopAndUndoRedoFollow));
        });
    }
}
