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
            catch (Exception e) when (e is System.Runtime.InteropServices.COMException or InvalidOperationException or TimeoutException
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
    /// <summary>An entry of the menu bar ("File", "Lens"): a MenuBarItem with no id of its own, found by its name under <c>Menus</c>.</summary>
    public AutomationElement MenuBarItem(string name) => Wait(() => ByName(name, ControlType.MenuItem, ById("Menus")), $"the {name} menu");

    /// <summary>Opens a menu bar item or a drop-down button: ExpandCollapse when it has it, else Invoke.</summary>
    public static void Open(AutomationElement e)
    {
        if (e.Patterns.ExpandCollapse.IsSupported) e.Patterns.ExpandCollapse.Pattern.Expand();
        else e.Patterns.Invoke.Pattern.Invoke();
    }
    public AutomationElement AddAnalysisButton => Wait(() => ById("AddAnalysisButton"), "the Add Analysis button");

    /// <summary>The EFL in the status capsule ("49.997 mm"), <c>Status.EFL</c>; the others are <c>Status.FNumber</c>, <c>Status.TT</c> and the button <c>Status.Issues</c>.</summary>
    public string Efl() => NameOf(Wait(() => ById("Status.EFL"), "the EFL in the status capsule"));

    // ---- Sidebar ----

    /// <summary>A sidebar entry, <c>Sidebar.{key}</c>: an analysis id ("mtf") or lens_data, layout, issues.</summary>
    public AutomationElement SidebarItem(string key) => Wait(() => ById($"Sidebar.{key}"), $"the sidebar entry {key}");

    // ---- Lens data ----

    /// <summary>The lens data panel's container, <c>Panel.lens_data</c> (<c>Panel.lens_data.Header</c> is its title).</summary>
    public AutomationElement LensData => Wait(() => ById("Panel.lens_data"), "the lens data");

    public void WaitForLensRows() => Wait(() => ById("LensRow.1"), "the lens data's rows");

    /// <summary>Row <c>LensRow.{index}</c>: 0 is the object, then surface by surface ("Surface 6, aperture stop").</summary>
    public AutomationElement LensRow(int index) => Wait(() => ById($"LensRow.{index}"), $"lens row {index}");

    public string LensRowName(int index) => NameOf(LensRow(index));

    /// <summary>Selects a row as a click does: its SelectionItem, else its Invoke.</summary>
    public void SelectLensRow(int index)
    {
        var row = LensRow(index);
        if (row.Patterns.SelectionItem.IsSupported) row.Patterns.SelectionItem.Pattern.Select();
        else if (row.Patterns.Invoke.IsSupported) row.Patterns.Invoke.Pattern.Invoke();
        else throw new InvalidOperationException($"LensRow.{index} supports neither SelectionItem nor Invoke.");
    }

    /// <summary>
    /// A cell, <c>LensCell.{row}.{key}</c> (key: surface, type, comment, radius, thickness, material,
    /// semi_diameter, conic …): Value reads and commits like typing, Invoke starts editing.
    /// </summary>
    public AutomationElement LensCell(int row, string key) => Wait(() => ById($"LensCell.{row}.{key}"), $"lens cell {row}.{key}");

    /// <summary>The text shown in a cell (its Value).</summary>
    public string LensCellText(int row, string key) => LensCell(row, key).Patterns.Value.Pattern.Value.ValueOrDefault ?? "";

    /// <summary>The in-place text editor while a cell edits, <c>LensEditor</c>.</summary>
    public AutomationElement? LensEditorOrNull() => ById("LensEditor", LensData) is { } e && Shown(e) ? e : null;

    public AutomationElement LensEditor => Wait(LensEditorOrNull, "the lens cell editor");

    // ---- Menus (flyouts open in the app's own popup windows) ----

    /// <summary>A menu item of an open menu or flyout, in the window or one of the app's popup windows.</summary>
    public AutomationElement MenuItem(string text) => Wait(() => MenuItemOrNull(text), $"the menu item {text}");

    /// <summary>A menu item by its AutomationId (<c>AddAnalysis.{category}</c>, <c>AddAnalysis.{analysis}</c>).</summary>
    public AutomationElement MenuItemById(string id) => Wait(() => MenuItemOrNull(id, byId: true), $"the menu item {id}");

    AutomationElement? MenuItemOrNull(string text, bool byId = false)
    {
        var cond = (byId ? Cf.ByAutomationId(text) : Cf.ByName(text)).And(Cf.ByControlType(ControlType.MenuItem));
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
    public AutomationElement OpenMenu(Action open, string text, bool byId = false)
    {
        for (var attempt = 1; ; attempt++)
        {
            open();
            try { return Wait(() => MenuItemOrNull(text, byId), $"the menu item {text}", TimeSpan.FromSeconds(attempt < 3 ? 5 : 15)); }
            catch (TimeoutException) when (attempt < 3) { }
        }
    }

    // ---- Analysis panes ----

    /// <summary>Pane <c>Pane.{analysis}</c> (analysis id: "mtf", "spot_diagram" …).</summary>
    public AutomationElement Pane(string analysis) => Wait(() => ById($"Pane.{analysis}"), $"the {analysis} pane");

    public AutomationElement? PaneOrNull(string analysis) => ById($"Pane.{analysis}");

    /// <summary>The analyses of the panes in the main window (their <c>Pane.{analysis}</c> ids).</summary>
    public List<string> PaneIds() => W.FindAllDescendants(Cf.ByClassName("AnalysisPane"))
        .Select(IdOf).Where(i => i.StartsWith("Pane.", StringComparison.Ordinal)).Select(i => i["Pane.".Length..]).ToList();

    public int PaneCount() => PaneIds().Count;

    /// <summary>A pane part, <c>Pane.{analysis}.{part}</c>: Header, Title (the drag handle), Run, AutoUpdate, Settings, State, More, Backend …</summary>
    public AutomationElement PaneControl(string analysis, string part) => Wait(() => ById($"Pane.{analysis}.{part}"), $"{part} of the {analysis} pane");

    /// <summary>The analysis id of a pane titled <paramref name="title"/> ("Ray Fan" gives "ray_fan"), or null when no pane shows it.</summary>
    public string? PaneWithTitle(string title) =>
        W.FindAllDescendants(Cf.ByName(title).And(Cf.ByControlType(ControlType.Button)))
            .Select(IdOf).FirstOrDefault(id => id.StartsWith("Pane.", StringComparison.Ordinal) && id.EndsWith(".Title", StringComparison.Ordinal)) is { } i
            ? i["Pane.".Length..^".Title".Length] : null;

    /// <summary>A field of an open settings form, <c>Settings.{analysis}.{field}</c> (field: the setting's key path, "max_frequency").</summary>
    public AutomationElement SettingsField(string analysis, string field) => Wait(() =>
    {
        var e = ById($"Settings.{analysis}.{field}");
        // A number with a unit is a group (field + unit): the field is its Edit.
        return e is null || e.Patterns.Value.IsSupported ? e : e.FindFirstDescendant(Cf.ByControlType(ControlType.Edit));
    }, $"the settings field {analysis}.{field}");

    /// <summary>The text field of inspector row <c>Inspector.Row.{field}</c> ("radius", "comment"): the row itself when it takes a value, else its Edit.</summary>
    public AutomationElement InspectorField(string field) => Wait(() =>
    {
        var row = ById($"Inspector.Row.{field}");
        return row is null || row.ControlType == ControlType.Edit ? row : row.FindFirstDescendant(Cf.ByControlType(ControlType.Edit));
    }, $"the inspector field {field}");

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
    public void FocusAway() => Wait(() => ById("AddAnalysisButton"), "the Add Analysis button").Focus();

    // ---- Layout ----

    /// <summary>The layout's canvas, <c>Layout.Canvas</c> (its name describes the view).</summary>
    public AutomationElement LayoutCanvas => Wait(() => ById("Layout.Canvas"), "the layout canvas");

    /// <summary>The scale bar's name ("Scale bar 20 mm"): it follows the zoom.</summary>
    public string ScaleBarName() => NameOf(Wait(() => ById("ScaleBar"), "the layout's scale bar"));

    /// <summary>A view selector item, <c>Layout.Mode.{mode}</c> ("section", "xz", "3d"); it supports SelectionItem.</summary>
    public AutomationElement LayoutMode(string mode) => Wait(() => ById($"Layout.Mode.{mode}"), $"the layout's {mode} mode");

    // ---- Panels ----

    /// <summary>A panel's title, <c>Panel.{key}.Header</c> (its drag handle).</summary>
    public AutomationElement PanelHeader(string key) => Wait(() => ById($"Panel.{key}.Header"), $"the {key} panel's title");

    // ---- Inspector ----

    /// <summary>An inspector page button, <c>Inspector.Tab.{id}</c> (surface, system, optimize, glass_substitution, tolerance, solves).</summary>
    public AutomationElement InspectorTab(string id) => Wait(() => ById($"Inspector.Tab.{id}"), $"the inspector's {id} page button");

    /// <summary>Whether text reading <paramref name="text"/> shows anywhere in the window.</summary>
    public bool ShowsText(string text) => W.FindAllDescendants(Cf.ByName(text)).Any(Shown);

    /// <summary>Whether text starting with <paramref name="prefix"/> shows anywhere in the window.</summary>
    public bool ShowsTextStarting(string prefix) =>
        W.FindAllDescendants(Cf.ByControlType(ControlType.Text)).Any(e => NameOf(e).StartsWith(prefix, StringComparison.Ordinal) && Shown(e));
}
