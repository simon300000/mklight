using System.Drawing;
using FlaUI.Core.AutomationElements;
using FlaUI.Core.Conditions;
using FlaUI.Core.Definitions;

namespace MKLight.UITests;

/// <summary>
/// Every element lookup the tests make, in one place. Where the app has no AutomationId yet the
/// lookup goes by name, position or x:Name; each such spot names the id it would rather use
/// (README.md lists them), so ids can be swapped in here alone.
/// </summary>
sealed class Ui(AppSession app)
{
    public static readonly TimeSpan Timeout = TimeSpan.FromSeconds(30);

    ConditionFactory Cf => app.Automation.ConditionFactory;
    Window W => app.Window;

    // ---- Waiting ----

    public static T Wait<T>(Func<T?> probe, string what, TimeSpan? timeout = null) where T : class
    {
        var end = DateTime.UtcNow + (timeout ?? Timeout);
        while (true)
        {
            T? v = null;
            try { v = probe(); }
            catch (Exception e) when (e is System.Runtime.InteropServices.COMException or InvalidOperationException
                or FlaUI.Core.Exceptions.PropertyNotSupportedException or FlaUI.Core.Exceptions.ElementNotAvailableException) { }
            if (v is not null) return v;
            if (DateTime.UtcNow > end) throw new TimeoutException($"Timed out waiting for {what}.");
            Thread.Sleep(100);
        }
    }

    public static void WaitUntil(Func<bool> probe, string what, TimeSpan? timeout = null) =>
        Wait(() => probe() ? "" : null, what, timeout);

    /// <summary>UI Automation's ShowContextMenu (IUIAutomationElement3): WinUI raises ContextRequested / opens the ContextFlyout, as a right click would.</summary>
    public static void ShowContextMenu(AutomationElement e) =>
        ((FlaUI.UIA3.UIA3FrameworkAutomationElement)e.FrameworkAutomationElement).NativeElement3.ShowContextMenu();

    public static string NameOf(AutomationElement e) => e.Properties.Name.ValueOrDefault ?? "";
    public static string IdOf(AutomationElement e) => e.Properties.AutomationId.ValueOrDefault ?? "";
    public static Rectangle Bounds(AutomationElement e) => e.Properties.BoundingRectangle.ValueOrDefault;
    static bool Shown(AutomationElement e) => Bounds(e) is { Width: > 0, Height: > 0 };

    AutomationElement? ById(string id, AutomationElement? root = null) => (root ?? W).FindFirstDescendant(Cf.ByAutomationId(id));
    AutomationElement? ByName(string name, ControlType? type = null, AutomationElement? root = null) =>
        (root ?? W).FindFirstDescendant(type is { } t ? Cf.ByName(name).And(Cf.ByControlType(t)) : Cf.ByName(name));

    // ---- Title bar and status capsule ----

    public AutomationElement Undo => Wait(() => ById("UndoButton"), "the Undo button");
    public AutomationElement Redo => Wait(() => ById("RedoButton"), "the Redo button");
    public AutomationElement FileMenuButton => Wait(() => ById("FileButton"), "the File button");
    public AutomationElement AddAnalysisButton => Wait(() => ById("AddAnalysisButton"), "the Add Analysis button");

    /// <summary>The EFL in the status capsule ("49.997 mm"). Wanted id: <c>Status.EFL</c> (today: the text after the "EFL" label in NumbersButton).</summary>
    public string Efl()
    {
        var numbers = Wait(() => ById("NumbersButton"), "the status capsule's numbers");
        var texts = numbers.FindAllChildren(Cf.ByControlType(ControlType.Text));
        for (var i = 0; i + 1 < texts.Length; i++)
            if (NameOf(texts[i]) == "EFL") return NameOf(texts[i + 1]);
        throw new InvalidOperationException("No EFL in the status capsule.");
    }

    // ---- Sidebar ----

    /// <summary>A sidebar entry ("Lens Data", "MTF"). Wanted id: <c>Sidebar.{analysis}</c>.</summary>
    public AutomationElement SidebarItem(string title) => Wait(() => ByName(title, ControlType.ListItem, ById("Nav")), $"the sidebar entry {title}");

    // ---- Lens data ----

    /// <summary>Wanted id: <c>Panel.lens_data</c>.</summary>
    public AutomationElement LensData => Wait(() => ByName("Lens data", ControlType.Group), "the lens data");

    public void WaitForLensRows() => Wait(() => LensRowOrNull(1), "the lens data's rows");

    AutomationElement? LensRowOrNull(int index)
    {
        var data = ByName("Lens data", ControlType.Group);
        return data?.FindAllDescendants(Cf.ByControlType(ControlType.Group))
            .FirstOrDefault(e => NameOf(e) is var n && (n == $"Surface {index}" || n.StartsWith($"Surface {index},", StringComparison.Ordinal)));
    }

    /// <summary>The row of surface <paramref name="index"/> ("Surface 3", "Surface 6, aperture stop"). Wanted id: <c>LensRow.{index}</c>.</summary>
    public AutomationElement LensRow(int index) => Wait(() => LensRowOrNull(index), $"lens row {index}");

    public string LensRowName(int index) => NameOf(LensRow(index));

    /// <summary>A column title of the lens grid ("Radius").</summary>
    AutomationElement ColumnHeader(string title) =>
        Wait(() => ById("HorizontalScroller", LensData)?.FindFirstChild(Cf.ByName(title).And(Cf.ByControlType(ControlType.Text))), $"the {title} column");

    /// <summary>The middle of a cell (its row × its column title). Wanted id: <c>LensCell.{row}.{column}</c> on the cell's slot.</summary>
    public Point LensCell(int row, string column)
    {
        var r = Bounds(LensRow(row));
        var c = Bounds(ColumnHeader(column));
        return new Point(c.X + c.Width / 2, r.Y + r.Height / 2);
    }

    /// <summary>The text shown in a cell. Wanted: the cell's own element (<c>LensCell.{row}.{column}</c>) with a Value or Name.</summary>
    public string LensCellText(int row, string column)
    {
        var c = Bounds(ColumnHeader(column));
        var texts = LensRow(row).FindAllChildren(Cf.ByControlType(ControlType.Text));
        return texts.Where(t => Bounds(t) is { Width: > 0 } b && b.X < c.Right && b.Right > c.X).Select(NameOf).FirstOrDefault() ?? "";
    }

    /// <summary>The in-place editor while a cell edits. Wanted id: <c>LensEditor</c>.</summary>
    public AutomationElement? LensEditorOrNull() => LensData.FindFirstDescendant(Cf.ByControlType(ControlType.Edit));

    public AutomationElement LensEditor => Wait(LensEditorOrNull, "the lens cell editor");

    // ---- Menus (flyouts open in the app's own popup windows) ----

    /// <summary>A menu item of an open menu or flyout, in the window or one of the app's popup windows.</summary>
    public AutomationElement MenuItem(string text) => Wait(() => MenuItemOrNull(text), $"the menu item {text}");

    AutomationElement? MenuItemOrNull(string text)
    {
        var cond = Cf.ByName(text).And(Cf.ByControlType(ControlType.MenuItem));
        if (W.FindFirstDescendant(cond) is { } inWindow && Shown(inWindow)) return inWindow;
        // Windowed popups are top-level windows of the app's process.
        var pid = W.Properties.ProcessId.Value;
        foreach (var top in app.Automation.GetDesktop().FindAllChildren(Cf.ByProcessId(pid)))
            if (top.FindFirstDescendant(cond) is { } item && Shown(item)) return item;
        return null;
    }

    public bool IsMenuOpen(string text) => MenuItemOrNull(text) is not null;

    /// <summary>
    /// Opens a menu with <paramref name="open"/> and returns its item <paramref name="text"/>, opening it
    /// again when it does not show: a flyout light-dismisses when another window activates (other apps
    /// starting on a shared desktop do).
    /// </summary>
    public AutomationElement OpenMenu(Action open, string text)
    {
        for (var attempt = 1; ; attempt++)
        {
            open();
            try { return Wait(() => MenuItemOrNull(text), $"the menu item {text}", TimeSpan.FromSeconds(attempt < 3 ? 5 : 15)); }
            catch (TimeoutException) when (attempt < 3) { }
        }
    }

    // ---- Analysis panes ----

    /// <summary>A pane's title button (it opens the pane's menu and is the drag handle). Wanted id: <c>Panel.{analysis}.Header</c>.</summary>
    public AutomationElement PaneTitle(string title) =>
        Wait(() => W.FindAllDescendants(Cf.ByAutomationId("TitleButton"))
            .FirstOrDefault(b => b.FindFirstChild(Cf.ByAutomationId("TitleText")) is { } t && NameOf(t) == title), $"the {title} pane");

    public AutomationElement? PaneTitleOrNull(string title) =>
        W.FindAllDescendants(Cf.ByAutomationId("TitleButton"))
            .FirstOrDefault(b => b.FindFirstChild(Cf.ByAutomationId("TitleText")) is { } t && NameOf(t) == title);

    /// <summary>
    /// A control in a pane's header row by x:Name ("RunButton", "SettingsButton", "StateText"): the
    /// nearest one right of the title on the same line (panes have no container element in the
    /// automation tree yet). Wanted ids: <c>Pane.{analysis}.Run</c>, <c>Pane.{analysis}.Settings</c>, <c>Pane.{analysis}.State</c>.
    /// </summary>
    public AutomationElement PaneControl(string title, string xName) => Wait(() =>
    {
        var t = Bounds(PaneTitle(title));
        return W.FindAllDescendants(Cf.ByAutomationId(xName))
            .Where(b => Bounds(b) is var r && r.Width > 0 && Math.Abs(r.Y - t.Y) < t.Height * 3 && r.X >= t.X)
            .OrderBy(b => Bounds(b).X - t.X).FirstOrDefault();
    }, $"{xName} of the {title} pane");

    /// <summary>A field of an open settings form, by its label ("Max Frequency"). Wanted id: <c>Settings.{analysis}.{field}</c>.</summary>
    public AutomationElement SettingsField(string label) => Wait(() =>
    {
        var e = W.FindAllDescendants(Cf.ByName(label)).FirstOrDefault(e => e.ControlType != ControlType.Text && Shown(e));
        // A number with a unit is a group (field + unit): the field is its Edit.
        return e is null || e.Patterns.Value.IsSupported ? e : e.FindFirstDescendant(Cf.ByControlType(ControlType.Edit));
    }, $"the settings field {label}");

    /// <summary>A text field of the inspector's current page, by its label ("Radius", "Comment"). Wanted id: <c>Inspector.{page}.{field}</c>.</summary>
    public AutomationElement InspectorField(string label) => Wait(() =>
    {
        var tabs = ByName("Inspector pages", ControlType.Group);
        var t = tabs is null ? Rectangle.Empty : Bounds(tabs);
        return W.FindAllDescendants(Cf.ByName(label).And(Cf.ByControlType(ControlType.Edit)))
            .FirstOrDefault(e => Bounds(e) is var r && r.Width > 0 && r.X >= t.X - t.Width / 4);
    }, $"the inspector field {label}");

    /// <summary>The texts shown on the inspector's current page (below its page buttons). Wanted id: <c>Inspector.Page</c> on the page host, with the page's id.</summary>
    public List<string> InspectorPageTexts()
    {
        var t = Bounds(Wait(() => ByName("Inspector pages", ControlType.Group), "the inspector"));
        return W.FindAllDescendants(Cf.ByControlType(ControlType.Text))
            .Where(e => Bounds(e) is var r && r.Width > 0 && r.X >= t.X - t.Width / 4 && r.Y > t.Bottom)
            .Select(NameOf).ToList();
    }

    /// <summary>Whether the inspector shows the page of surface <paramref name="index"/> (its "Surface 3" heading).</summary>
    public bool InspectorShowsSurface(int index)
    {
        var tabs = ByName("Inspector pages", ControlType.Group);
        if (tabs is null) return false;
        var t = Bounds(tabs);
        return W.FindAllDescendants(Cf.ByName($"Surface {index}").And(Cf.ByControlType(ControlType.Text)))
            .Any(e => Bounds(e) is var r && r.Width > 0 && r.X >= t.X - t.Width / 4);
    }

    /// <summary>Moves the keyboard focus (UIA SetFocus) to a neutral control, so the field being left commits.</summary>
    public void FocusAway() => Wait(() => ById("FileButton"), "the File button").Focus();

    // ---- Layout ----

    /// <summary>The layout's canvas (its name describes the view). Wanted id: <c>Layout.Canvas</c> (x:Name "Surface" today).</summary>
    public AutomationElement LayoutCanvas => Wait(() => W.FindAllDescendants(Cf.ByAutomationId("Surface"))
        .FirstOrDefault(e => NameOf(e).StartsWith("Layout", StringComparison.Ordinal)), "the layout canvas");

    /// <summary>The scale bar's name ("Scale bar 20 mm"): it follows the zoom.</summary>
    public string ScaleBarName() => NameOf(Wait(() => ById("ScaleBar"), "the layout's scale bar"));

    /// <summary>"Section", "3D view". Wanted id: <c>Layout.Mode.{mode}</c>.</summary>
    public AutomationElement LayoutModeButton(string name) => Wait(() => ByName(name, ControlType.Button), $"the layout's {name} button");

    // ---- Panels ----

    /// <summary>The lens data's title text (its drag handle). Wanted id: <c>Panel.lens_data.Header</c>.</summary>
    public AutomationElement LensDataHeader => Wait(() => ByName("Lens Data", ControlType.Text, LensData), "the lens data's title");

    // ---- Inspector ----

    /// <summary>An inspector page button ("Surface", "System", "Optimize"). Wanted id: <c>Inspector.Tab.{id}</c>.</summary>
    public AutomationElement InspectorTab(string title) =>
        Wait(() => ByName(title, ControlType.Button, ByName("Inspector pages", ControlType.Group)), $"the inspector's {title} page button");

    /// <summary>Whether text reading <paramref name="text"/> shows anywhere in the window.</summary>
    public bool ShowsText(string text) => W.FindAllDescendants(Cf.ByName(text)).Any(Shown);

    /// <summary>Whether text starting with <paramref name="prefix"/> shows anywhere in the window.</summary>
    public bool ShowsTextStarting(string prefix) =>
        W.FindAllDescendants(Cf.ByControlType(ControlType.Text)).Any(e => NameOf(e).StartsWith(prefix, StringComparison.Ordinal) && Shown(e));
}
