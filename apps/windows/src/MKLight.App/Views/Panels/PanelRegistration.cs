using System.Globalization;
using System.Runtime.CompilerServices;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Windows;
using Windows.Foundation;
using Windows.System;

namespace MKLight.App.Views.Panels;

/// <summary>
/// The panels' File menu command (Reset Panel Layout), the design window
/// hook (separate windows open and close with it) and launch actions for
/// self-checks:
/// <list type="bullet">
/// <item><c>panels-demo</c>: two columns, lens data over the layout and a tab group over a pane.</item>
/// <item><c>panels-detach=&lt;panel&gt;</c>: the panel in a separate window, which the screenshot captures.</item>
/// <item><c>panels-drag=&lt;panel&gt;@fx,fy[@window]</c>: a drag of the panel held at a point of a window's content (fractions; window 0 is the main window, 1 the first separate window), showing its drop zone; <c>panels-drop</c> drops it.</item>
/// <item><c>panels-hide=&lt;panel&gt;</c>, <c>panels-show=&lt;panel&gt;</c>, <c>panels-reset</c>.</item>
/// <item><c>panels-capture=&lt;window&gt;</c>: the screenshot captures that window (0 the main window).</item>
/// </list>
/// A panel is <c>lens_data</c>, <c>layout</c>, a pane id or an analysis name (its first pane).
/// </summary>
static class PanelRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.WindowCreated += w => w.Panels.Attach(w);
        AppHooks.Commands.Add(new("Reset Panel Layout", "\uE72C", w => w.Panels.ResetLayout(), Order: 60, Menu: "View"));

        AppHooks.LaunchActions["panels-demo"] = (w, _) => { Demo(w.Panels); return Task.CompletedTask; };
        AppHooks.LaunchActions["panels-detach"] = (w, arg) =>
        {
            if (Resolve(w.Panels, arg) is not { } panel) return Task.CompletedTask;
            w.Panels.MoveToNewWindow(panel);
            if (w.Panels.Arrangement.Location(panel) is { } l) AppHooks.ScreenshotTarget = w.Panels.WindowOf(l.Window);
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["panels-drag"] = (w, arg) =>
        {
            var parts = (arg ?? "").Split('@');
            if (parts.Length < 2 || Resolve(w.Panels, parts[0]) is not { } panel) return Task.CompletedTask;
            var xy = parts[1].Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
            // A third part picks the window: 0 the main window, 1 the first separate window, …
            var index = parts.Length > 2 ? int.Parse(parts[2], CultureInfo.InvariantCulture) : 0;
            if (w.Panels.Trees.ElementAtOrDefault(index) is not { } tree) return Task.CompletedTask;
            var r = tree.Bounds;
            w.Panels.Drag.Simulate(panel, PanelDragController.Screen(tree, new Point(r.X + r.Width * xy[0], r.Y + r.Height * xy[1])));
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["panels-drop"] = (w, _) => { w.Panels.Drag.DropSimulated(); return Task.CompletedTask; };
        AppHooks.LaunchActions["panels-hide"] = (w, arg) => { if (Resolve(w.Panels, arg) is { } p) w.Panels.Close(p); return Task.CompletedTask; };
        AppHooks.LaunchActions["panels-show"] = (w, arg) => { if (arg is not null) w.Panels.ShowPanel(Resolve(w.Panels, arg)?.Key ?? arg); return Task.CompletedTask; };
        AppHooks.LaunchActions["panels-capture"] = (w, arg) =>
        {
            var index = int.TryParse(arg, CultureInfo.InvariantCulture, out var i) ? i : 0;
            AppHooks.ScreenshotTarget = w.Panels.Trees.ElementAtOrDefault(index)?.HostWindow ?? w;
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["panels-reset"] = (w, _) => { w.Panels.ResetLayout(); return Task.CompletedTask; };
    }

    static PanelId? Resolve(PanelHost host, string? arg)
    {
        if (arg is null) return null;
        if (arg is "lens_data" or "layout") return new PanelId(arg);
        var ws = host.Session.Workspace;
        return (ws.Panes.FirstOrDefault(p => p.Id == arg) ?? ws.Panes.FirstOrDefault(p => p.Analysis == arg)) is { } pane ? PanelId.Analysis(pane.Id) : null;
    }

    /// <summary>Two columns: the lens data over the layout; the panes as tabs over a field curvature pane.</summary>
    static void Demo(PanelHost host)
    {
        var ws = host.Session.Workspace;
        foreach (var a in new[] { "ray_fan", "field_curvature" })
            if (ws.Descriptor(a) is not null && ws.Panes.All(p => p.Analysis != a)) ws.AddPane(a);
        host.Reconcile();
        var panes = ws.Panes.Select(p => PanelId.Analysis(p.Id)).ToList();
        if (panes.Count < 2) return;
        var arr = host.Arrangement;
        var first = arr.Location(panes[0])!.Value.Group.Id;
        var bottom = panes.Count > 2 ? panes[^1] : (PanelId?)null;
        foreach (var p in panes.Skip(1).Where(p => p != bottom)) arr.Move(p, new PanelDropTarget.Tab(first, null));
        if (bottom is { } b) arr.Move(b, new PanelDropTarget.Split(first, PanelEdge.Bottom));
        arr.Move(PanelId.LensData, new PanelDropTarget.Split(arr.Location(PanelId.Layout)!.Value.Group.Id, PanelEdge.Top));
        if (arr.Main.Root is PanelSplit { Axis: SplitAxis.Horizontal } row && row.Children.Count == 2)
        {
            row.Resize(0, 0.5, 0.5);
            if (row.Children[0] is PanelSplit left) left.Resize(0, 0.45, 0.55);
            if (row.Children[1] is PanelSplit right) right.Resize(0, 0.6, 0.4);
        }
        arr.Select(panes[1]);
        host.Focus(panes[1]);
    }
}
