using System.Text.Json.Nodes;

namespace MKLight.App.ViewModels;

// ---- Panels ----

/// <summary>
/// Something a window shows in a panel: the lens data, the layout, or one
/// analysis pane (by its key). Each is open in at most one place across a
/// design's windows. The key is what the project saves.
/// </summary>
public readonly record struct PanelId(string Key)
{
    public static PanelId LensData => new("lens_data");
    public static PanelId Layout => new("layout");
    public static PanelId Analysis(string paneId) => new(paneId);

    /// <summary>The analysis pane's id, null for the lens data and the layout.</summary>
    public string? PaneId => Key is "lens_data" or "layout" ? null : Key;

    public override string ToString() => Key;
}

/// <summary>How a split lays out its children: side by side (horizontal) or stacked (vertical).</summary>
public enum SplitAxis { Horizontal, Vertical }

/// <summary>A side of a panel group or of a window's content.</summary>
public enum PanelEdge { Leading, Trailing, Top, Bottom }

public static class PanelEdges
{
    public static SplitAxis Axis(this PanelEdge e) => e is PanelEdge.Leading or PanelEdge.Trailing ? SplitAxis.Horizontal : SplitAxis.Vertical;
    /// <summary>The new panel comes before the existing content (leading or top).</summary>
    public static bool IsLeadingSide(this PanelEdge e) => e is PanelEdge.Leading or PanelEdge.Top;
}

/// <summary>Where a dragged panel lands.</summary>
public abstract record PanelDropTarget
{
    /// <summary>Into a tab group, at a tab position (null: after the last tab).</summary>
    public sealed record Tab(Guid Group, int? Index) : PanelDropTarget;
    /// <summary>Beside a tab group, splitting its space.</summary>
    public sealed record Split(Guid Group, PanelEdge Edge) : PanelDropTarget;
    /// <summary>Along a whole edge of a window's content.</summary>
    public sealed record WindowEdge(Guid Window, PanelEdge Edge) : PanelDropTarget;
    /// <summary>Into a window of its own.</summary>
    public sealed record NewWindow : PanelDropTarget;
}

/// <summary>A separate window's frame on screen: effective pixels, the origin at the top left.</summary>
public readonly record struct PanelFrame(double X, double Y, double Width, double Height);

// ---- Tree ----

/// <summary>A node of a window's panel tree: a tab group or a split.</summary>
public abstract class PanelNode
{
    public Guid Id { get; } = Guid.NewGuid();
    /// <summary>The tab groups, in reading order.</summary>
    public abstract IEnumerable<PanelGroup> Groups { get; }
    public IEnumerable<PanelId> Panels => Groups.SelectMany(g => g.Panels);
}

/// <summary>Panels shown in one place, one at a time; a tab strip switches between them when there are several.</summary>
public sealed class PanelGroup : PanelNode
{
    internal readonly List<PanelId> Items;

    public PanelGroup(IEnumerable<PanelId> panels, PanelId? selected = null)
    {
        Items = [.. panels];
        Selected = selected;
    }

    public new IReadOnlyList<PanelId> Panels => Items;
    public PanelId? Selected { get; set; }

    /// <summary>The panel on screen.</summary>
    public PanelId? Current => Selected is { } s && Items.Contains(s) ? s : Items.Count > 0 ? Items[0] : null;

    public override IEnumerable<PanelGroup> Groups { get { yield return this; } }
}

/// <summary>Children side by side (horizontal) or stacked (vertical), each with a share of the space.</summary>
public sealed class PanelSplit : PanelNode
{
    List<PanelNode> _children;
    double[] _fractions;

    public PanelSplit(SplitAxis axis, IEnumerable<PanelNode> children, IEnumerable<double>? fractions = null)
    {
        Axis = axis;
        _children = [.. children];
        var f = fractions?.ToArray() ?? [];
        _fractions = Normalized(f.Length == _children.Count ? f : [.. Enumerable.Repeat(1.0, _children.Count)]);
    }

    public SplitAxis Axis { get; }
    public IReadOnlyList<PanelNode> Children => _children;
    /// <summary>The children's shares of the space (they sum to 1).</summary>
    public IReadOnlyList<double> Fractions => _fractions;

    public override IEnumerable<PanelGroup> Groups => _children.SelectMany(c => c.Groups);

    /// <summary>Moves the divider after child <paramref name="k"/>: the pair's new shares (the divider's drag).</summary>
    public void Resize(int k, double first, double second)
    {
        if (k < 0 || k + 1 >= _fractions.Length) return;
        _fractions[k] = first;
        _fractions[k + 1] = second;
    }

    /// <summary>Equal shares for every child.</summary>
    public void Equalize()
    {
        for (var i = 0; i < _fractions.Length; i++) _fractions[i] = 1.0 / Math.Max(_children.Count, 1);
    }

    internal void Set(IEnumerable<PanelNode> children, IEnumerable<double> fractions)
    {
        _children = [.. children];
        _fractions = Normalized([.. fractions]);
    }

    internal static double[] Normalized(double[] f)
    {
        var clean = f.Select(v => double.IsFinite(v) && v > 0 ? v : 0).ToArray();
        var sum = clean.Sum();
        if (sum <= 0) return [.. Enumerable.Repeat(1.0 / Math.Max(f.Length, 1), f.Length)];
        return [.. clean.Select(v => v / sum)];
    }
}

/// <summary>The panels of one window: the design's main window, or a window panels were moved out to.</summary>
public sealed class PanelWindowModel(PanelNode? root, PanelFrame? frame = null)
{
    public Guid Id { get; } = Guid.NewGuid();
    public PanelNode? Root { get; internal set; } = root;
    /// <summary>A separate window's frame on screen, saved with the project.</summary>
    public PanelFrame? Frame { get; set; } = frame;

    public IEnumerable<PanelId> Panels => Root?.Panels ?? [];
    public IEnumerable<PanelGroup> Groups => Root?.Groups ?? [];
}

// ---- Arrangement ----

/// <summary>
/// Where a design's panels are (the Mac app's PanelArrangement): a tree of
/// splits and tab groups for the main window and for each separate window.
/// A panel is in one place at a time; closing one takes it out of the tree
/// (an analysis pane is then removed from the workspace, the lens data and
/// the layout can be shown again). Platform-neutral: the views draw the
/// trees, the window code opens a window for each detached model.
/// </summary>
public sealed class PanelArrangement
{
    readonly List<PanelWindowModel> _detached;

    public PanelArrangement(PanelNode? main, IEnumerable<PanelWindowModel>? detached = null)
    {
        Main = new PanelWindowModel(main);
        _detached = [.. detached ?? []];
        Normalize();
    }

    public PanelWindowModel Main { get; }
    public IReadOnlyList<PanelWindowModel> Detached => _detached;
    public IEnumerable<PanelWindowModel> Windows => _detached.Prepend(Main);

    /// <summary>Raised after the tree changed (not for a divider's drag, which the view saves itself).</summary>
    public event Action? Changed;
    void Notify() => Changed?.Invoke();

    public PanelWindowModel? Window(Guid id) => Windows.FirstOrDefault(w => w.Id == id);
    public IEnumerable<PanelId> AllPanels => Windows.SelectMany(w => w.Panels);
    public bool Contains(PanelId panel) => Location(panel) is not null;

    public (PanelWindowModel Window, PanelGroup Group)? Location(PanelId panel)
    {
        foreach (var w in Windows)
            if (w.Groups.FirstOrDefault(g => g.Items.Contains(panel)) is { } g) return (w, g);
        return null;
    }

    public (PanelWindowModel Window, PanelGroup Group)? Group(Guid id)
    {
        foreach (var w in Windows)
            if (w.Groups.FirstOrDefault(g => g.Id == id) is { } g) return (w, g);
        return null;
    }

    /// <summary>Whether the panel is in a separate window.</summary>
    public bool IsDetached(PanelId panel) => Location(panel) is { } l && l.Window != Main;

    /// <summary>Whether the panel is alone in a separate window (moving it to a new window would change nothing).</summary>
    public bool IsAloneInWindow(PanelId panel) =>
        Location(panel) is { } l && l.Window != Main && l.Window.Panels.SequenceEqual([panel]);

    // ---- Showing and closing ----

    /// <summary>Brings a panel to the front of its tab group.</summary>
    public void Select(PanelId panel)
    {
        if (Location(panel) is not { } l || l.Group.Current == panel) return;
        l.Group.Selected = panel;
        Notify();
    }

    /// <summary>Takes a panel out of the tree; a window left empty goes away.</summary>
    public void Remove(PanelId panel)
    {
        if (Location(panel) is not { } l) return;
        Take(panel, l.Group);
        Normalize();
        Notify();
    }

    /// <summary>Removes a separate window, returning the panels it showed.</summary>
    public IReadOnlyList<PanelId> Discard(PanelWindowModel window)
    {
        if (window == Main) return [];
        var panels = window.Panels.ToList();
        if (_detached.Remove(window)) Notify();
        return panels;
    }

    /// <summary>
    /// Moves a separate window's panels into the main window, keeping their arrangement: as more columns
    /// of the area under the lens data (a lone panel goes where its kind belongs, as <see cref="ReturnToMain"/>).
    /// </summary>
    public void Dock(PanelWindowModel window)
    {
        if (window == Main) return;
        if (window.Root is not { } root) { Discard(window); return; }
        if (window.Panels.Count() == 1) { ReturnToMain(window.Panels.First()); return; }
        _detached.Remove(window);
        window.Root = null;
        if (LowerArea is { } area) AddColumn(root, area, Main);
        else Attach(root, Main, Main.Root is { } r && r.Panels.SequenceEqual([PanelId.LensData]) ? PanelEdge.Bottom : PanelEdge.Trailing, 0.6);
        Normalize();
        Notify();
    }

    /// <summary>
    /// Moves a panel from a separate window back to where its kind belongs in the main window: an analysis
    /// beside the other analyses, the lens data across the top, the layout at the left.
    /// </summary>
    public void ReturnToMain(PanelId panel)
    {
        if (Location(panel) is not { } l || l.Window == Main) return;
        Take(panel, l.Group);
        Normalize();
        if (panel.PaneId is not null)
        {
            var neighbour = Main.Panels.LastOrDefault(p => p.PaneId is not null);
            PlaceCore(panel, neighbour.Key is null ? null : neighbour);
        }
        else ShowCore(panel);
        Normalize();
        if (Location(panel) is { } now) now.Group.Selected = panel;
        Notify();
    }

    /// <summary>Replaces the whole arrangement with <paramref name="other"/>'s main window (Reset Panel Layout).</summary>
    public void Reset(PanelArrangement other)
    {
        _detached.Clear();
        Main.Root = other.Main.Root;
        Normalize();
        Notify();
    }

    /// <summary>Replaces the arrangement with <paramref name="other"/>'s windows (a saved arrangement read back).</summary>
    public void Replace(PanelArrangement other)
    {
        _detached.Clear();
        _detached.AddRange(other._detached);
        Main.Root = other.Main.Root;
        Normalize();
        Notify();
    }

    /// <summary>
    /// Shows the lens data or the layout where they belong by default: the lens data across the top of the
    /// main window, the layout at the left of the area under it.
    /// </summary>
    public void Show(PanelId panel)
    {
        if (Contains(panel)) return;
        ShowCore(panel);
        Normalize();
        Notify();
    }

    void ShowCore(PanelId panel)
    {
        var node = new PanelGroup([panel]);
        if (panel == PanelId.LensData) Attach(node, Main, PanelEdge.Top, 0.36);
        else if (panel == PanelId.Layout)
        {
            if (LowerArea is { } area) Insert(node, area, Main, PanelEdge.Leading, 0.42);
            else Attach(node, Main, PanelEdge.Bottom, 0.64);
        }
        else PlaceCore(panel, null);
    }

    /// <summary>
    /// Places a new analysis pane: as another column beside <paramref name="hint"/> (the pane it was added
    /// after), else beside the layout, else in the area under the lens data, else along the main window's
    /// trailing edge.
    /// </summary>
    public void Place(PanelId panel, PanelId? hint)
    {
        if (Contains(panel)) return;
        PlaceCore(panel, hint);
        Normalize();
        Notify();
    }

    void PlaceCore(PanelId panel, PanelId? hint)
    {
        if (Contains(panel)) return;
        var node = new PanelGroup([panel]);
        if (hint is { } h && Location(h) is { } at) AddColumn(node, at.Group, at.Window);
        else if (Location(PanelId.Layout) is { } lay && lay.Window == Main) AddColumn(node, lay.Group, Main);
        else if (LowerArea is { } area) AddColumn(node, area, Main);
        else if (Main.Root is { } r && r.Panels.SequenceEqual([PanelId.LensData])) Attach(node, Main, PanelEdge.Bottom, 0.64);
        else Attach(node, Main, PanelEdge.Trailing, 0.4);
    }

    /// <summary>Places a new analysis pane in a given window, along its trailing edge.</summary>
    public void Place(PanelId panel, Guid windowId)
    {
        if (Contains(panel) || Window(windowId) is not { } w) return;
        Attach(new PanelGroup([panel]), w, PanelEdge.Trailing, 1.0 / (w.Groups.Count() + 1));
        Normalize();
        Notify();
    }

    // ---- Moving ----

    /// <summary>
    /// Whether dropping <paramref name="panel"/> on <paramref name="target"/> changes anything. Dropping the
    /// only panel of a group on that group, or the only panel of a window on that window's edge, does not.
    /// </summary>
    public bool CanMove(PanelId panel, PanelDropTarget target)
    {
        var source = Location(panel);
        switch (target)
        {
            case PanelDropTarget.Tab t:
                return Group(t.Group) is { } a && (a.Group != source?.Group || a.Group.Items.Count > 1);
            case PanelDropTarget.Split s:
                return Group(s.Group) is { } b && (b.Group != source?.Group || b.Group.Items.Count > 1);
            case PanelDropTarget.WindowEdge e:
                return Window(e.Window) is { } w && !w.Panels.SequenceEqual([panel]);
            case PanelDropTarget.NewWindow:
                return source is not { } src || src.Window == Main || !src.Window.Panels.SequenceEqual([panel]);
            default:
                return false;
        }
    }

    /// <summary>
    /// Moves a panel. A <see cref="PanelDropTarget.NewWindow"/> target opens a separate window with
    /// <paramref name="frame"/>; it is returned. Windows left empty go away.
    /// </summary>
    public PanelWindowModel? Move(PanelId panel, PanelDropTarget target, PanelFrame? frame = null)
    {
        if (!CanMove(panel, target)) return null;
        PanelWindowModel? opened = null;
        switch (target)
        {
            case PanelDropTarget.Tab t:
            {
                if (Group(t.Group) is not { } loc) return null;
                var g = loc.Group;
                var i = g.Items.IndexOf(panel);
                if (i >= 0)
                {
                    // Reorder within the group.
                    g.Items.RemoveAt(i);
                    var j = Math.Clamp((t.Index ?? g.Items.Count) - ((t.Index ?? 0) > i ? 1 : 0), 0, g.Items.Count);
                    g.Items.Insert(j, panel);
                }
                else
                {
                    if (Location(panel) is { } from) Take(panel, from.Group);
                    g.Items.Insert(Math.Clamp(t.Index ?? g.Items.Count, 0, g.Items.Count), panel);
                }
                g.Selected = panel;
                break;
            }
            case PanelDropTarget.Split s:
            {
                if (Location(panel) is { } from) Take(panel, from.Group);
                Normalize();
                if (Group(s.Group) is not { } loc) return null;
                Insert(new PanelGroup([panel]), loc.Group, loc.Window, s.Edge, 0.5);
                break;
            }
            case PanelDropTarget.WindowEdge e:
            {
                if (Location(panel) is { } from) Take(panel, from.Group);
                Normalize();
                if (Window(e.Window) is not { } w) return null;
                Attach(new PanelGroup([panel]), w, e.Edge, 0.3);
                break;
            }
            case PanelDropTarget.NewWindow:
            {
                if (Location(panel) is { } from) Take(panel, from.Group);
                opened = new PanelWindowModel(new PanelGroup([panel]), frame);
                _detached.Add(opened);
                break;
            }
        }
        Normalize();
        Notify();
        return opened;
    }

    // ---- Tree editing ----

    static void Take(PanelId panel, PanelGroup g)
    {
        var i = g.Items.IndexOf(panel);
        if (i < 0) return;
        g.Items.RemoveAt(i);
        if (g.Selected == panel) g.Selected = g.Items.Count == 0 ? null : g.Items[Math.Min(i, g.Items.Count - 1)];
    }

    /// <summary>
    /// The area under the lens data in the main window (the last child of a vertical root), or the whole
    /// content when there is no lens data on top; null when the main window shows only the lens data.
    /// </summary>
    PanelNode? LowerArea
    {
        get
        {
            if (Main.Root is not { } root) return null;
            if (root is PanelSplit { Axis: SplitAxis.Vertical } s && s.Children.Count > 0 && s.Children[0].Panels.Contains(PanelId.LensData))
                return s.Children[^1];
            if (root.Panels.Contains(PanelId.LensData)) return null;
            return root;
        }
    }

    /// <summary>Adds <paramref name="node"/> as a column after <paramref name="anchor"/>: in a row, an equal share of it; otherwise splitting the anchor's space.</summary>
    void AddColumn(PanelNode node, PanelNode anchor, PanelWindowModel w)
    {
        if (Parent(anchor.Id, w.Root) is { Split.Axis: SplitAxis.Horizontal } found)
        {
            var (parent, i) = found;
            var n = (double)parent.Children.Count;
            var kids = parent.Children.ToList();
            var f = parent.Fractions.Select(v => v * n / (n + 1)).ToList();
            kids.Insert(i + 1, node);
            f.Insert(i + 1, 1 / (n + 1));
            parent.Set(kids, f);
        }
        else if (anchor is PanelSplit { Axis: SplitAxis.Horizontal } s)
        {
            var n = (double)s.Children.Count;
            s.Set(s.Children.Append(node), s.Fractions.Select(v => v * n / (n + 1)).Append(1 / (n + 1)));
        }
        else Insert(node, anchor, w, PanelEdge.Trailing, 0.5);
    }

    /// <summary>Puts <paramref name="node"/> beside <paramref name="anchor"/>, taking <paramref name="share"/> of the anchor's space.</summary>
    void Insert(PanelNode node, PanelNode anchor, PanelWindowModel w, PanelEdge edge, double share)
    {
        if (Parent(anchor.Id, w.Root) is { } found && found.Split.Axis == edge.Axis())
        {
            var (parent, i) = found;
            var kids = parent.Children.ToList();
            var f = parent.Fractions.ToList();
            var taken = f[i] * share;
            f[i] -= taken;
            var at = edge.IsLeadingSide() ? i : i + 1;
            kids.Insert(at, node);
            f.Insert(at, taken);
            parent.Set(kids, f);
            return;
        }
        if (anchor is PanelSplit s && s.Axis == edge.Axis())
        {
            var keep = s.Fractions.Select(v => v * (1 - share));
            if (edge.IsLeadingSide()) s.Set(s.Children.Prepend(node), keep.Prepend(share));
            else s.Set(s.Children.Append(node), keep.Append(share));
            return;
        }
        var split = new PanelSplit(edge.Axis(),
            edge.IsLeadingSide() ? [node, anchor] : [anchor, node],
            edge.IsLeadingSide() ? [share, 1 - share] : [1 - share, share]);
        ReplaceNode(anchor.Id, split, w);
    }

    /// <summary>Puts <paramref name="node"/> along a whole edge of a window's content.</summary>
    void Attach(PanelNode node, PanelWindowModel w, PanelEdge edge, double share)
    {
        if (w.Root is not { } root) { w.Root = node; return; }
        Insert(node, root, w, edge, share);
    }

    void ReplaceNode(Guid id, PanelNode node, PanelWindowModel w)
    {
        if (w.Root?.Id == id) w.Root = node;
        else if (Parent(id, w.Root) is { } found)
        {
            var (parent, i) = found;
            var kids = parent.Children.ToList();
            kids[i] = node;
            parent.Set(kids, parent.Fractions);
        }
    }

    static (PanelSplit Split, int Index)? Parent(Guid id, PanelNode? node)
    {
        if (node is not PanelSplit s) return null;
        for (var i = 0; i < s.Children.Count; i++)
            if (s.Children[i].Id == id) return (s, i);
        foreach (var c in s.Children)
            if (Parent(id, c) is { } found) return found;
        return null;
    }

    /// <summary>Drops empty groups, collapses splits of one child, merges a split into a parent of the same axis, and removes empty separate windows.</summary>
    internal void Normalize()
    {
        foreach (var w in Windows)
        {
            var n = w.Root is null ? null : Normalized(w.Root);
            if (n?.Id != w.Root?.Id) w.Root = n;
        }
        _detached.RemoveAll(w => w.Root is null);
    }

    static PanelNode? Normalized(PanelNode node)
    {
        switch (node)
        {
            case PanelGroup g:
                return g.Items.Count == 0 ? null : g;
            case PanelSplit s:
                var kids = new List<PanelNode>();
                var f = new List<double>();
                for (var i = 0; i < s.Children.Count; i++)
                {
                    var share = s.Fractions[i];
                    if (Normalized(s.Children[i]) is not { } n) continue;
                    if (n is PanelSplit inner && inner.Axis == s.Axis)
                    {
                        kids.AddRange(inner.Children);
                        f.AddRange(inner.Fractions.Select(v => v * share));
                    }
                    else
                    {
                        kids.Add(n);
                        f.Add(share);
                    }
                }
                if (kids.Count == 0) return null;
                if (kids.Count == 1) return kids[0];
                s.Set(kids, f);
                return s;
            default:
                return null;
        }
    }

    // ---- Defaults and saving ----

    /// <summary>The default window: the lens data across the top (36 %); under it the layout (42 %) and the analysis panes side by side.</summary>
    public static PanelArrangement Standard(IReadOnlyList<string> panes, bool showLensData = true, bool showLayout = true)
    {
        const double editorFraction = 0.36, viewportFraction = 0.42;
        var lower = new List<PanelNode>();
        var f = new List<double>();
        if (showLayout)
        {
            lower.Add(new PanelGroup([PanelId.Layout]));
            f.Add(panes.Count == 0 ? 1 : viewportFraction);
        }
        foreach (var key in panes)
        {
            lower.Add(new PanelGroup([PanelId.Analysis(key)]));
            f.Add((showLayout ? 1 - viewportFraction : 1) / panes.Count);
        }
        PanelNode? below = lower.Count == 0 ? null : lower.Count == 1 ? lower[0] : new PanelSplit(SplitAxis.Horizontal, lower, f);
        PanelNode? top = showLensData ? new PanelGroup([PanelId.LensData]) : null;
        var root = (top, below) switch
        {
            ({ } t, { } b) => new PanelSplit(SplitAxis.Vertical, [t, b], [editorFraction, 1 - editorFraction]),
            ({ } t, null) => t,
            _ => below,
        };
        return new PanelArrangement(root);
    }

    /// <summary>
    /// The arrangement as saved in the project: <c>{"main": node, "windows": [{"frame": [x, y, w, h], "root": node}]}</c>,
    /// a node being <c>{"tabs": [keys], "selected": key}</c> or <c>{"axis": "h"|"v", "children": [nodes], "fractions": [shares]}</c>.
    /// </summary>
    public JsonObject ToJson()
    {
        var o = new JsonObject();
        if (Main.Root is { } r) o["main"] = Encode(r);
        var windows = new JsonArray();
        foreach (var w in _detached)
        {
            if (w.Root is not { } root) continue;
            var wo = new JsonObject();
            if (w.Frame is { } f)
                wo["frame"] = new JsonArray([.. new[] { f.X, f.Y, f.Width, f.Height }.Select(v => (JsonNode?)JsonValue.Create(Math.Round(v * 10) / 10 + 0.0))]);
            wo["root"] = Encode(root);
            windows.Add(wo);
        }
        o["windows"] = windows;
        return o;
    }

    /// <summary>Reads a saved arrangement, keeping only the panels in <paramref name="known"/> (each once); null when <paramref name="json"/> is not one.</summary>
    public static PanelArrangement? Decode(JsonNode? json, ISet<PanelId> known)
    {
        if (json is not JsonObject o) return null;
        var seen = new HashSet<PanelId>();
        var main = o["main"] is { } m ? DecodeNode(m, known, seen) : null;
        var windows = new List<PanelWindowModel>();
        foreach (var w in o["windows"] as JsonArray ?? [])
        {
            if (w?["root"] is not { } rootJson || DecodeNode(rootJson, known, seen) is not { } root) continue;
            var f = (w["frame"] as JsonArray ?? []).Select(Number).OfType<double>().ToList();
            PanelFrame? frame = f.Count == 4 && f.All(double.IsFinite) && f[2] > 100 && f[3] > 100 ? new PanelFrame(f[0], f[1], f[2], f[3]) : null;
            windows.Add(new PanelWindowModel(root, frame));
        }
        return new PanelArrangement(main, windows);
    }

    static JsonObject Encode(PanelNode node)
    {
        switch (node)
        {
            case PanelGroup g:
                var o = new JsonObject { ["tabs"] = new JsonArray([.. g.Items.Select(p => (JsonNode?)JsonValue.Create(p.Key))]) };
                if (g.Current is { } front && g.Items.Count > 1) o["selected"] = front.Key;
                return o;
            case PanelSplit s:
                return new JsonObject
                {
                    ["axis"] = s.Axis == SplitAxis.Horizontal ? "h" : "v",
                    ["children"] = new JsonArray([.. s.Children.Select(c => (JsonNode?)Encode(c))]),
                    ["fractions"] = new JsonArray([.. s.Fractions.Select(v => (JsonNode?)JsonValue.Create(Math.Round(v * 10000) / 10000 + 0.0))]),
                };
            default:
                throw new InvalidOperationException();
        }
    }

    static PanelNode? DecodeNode(JsonNode node, ISet<PanelId> known, HashSet<PanelId> seen)
    {
        if (node["tabs"] is JsonArray tabs)
        {
            var panels = new List<PanelId>();
            foreach (var t in tabs)
            {
                if (t is not JsonValue v || !v.TryGetValue<string>(out var k)) continue;
                var p = new PanelId(k);
                if (known.Contains(p) && seen.Add(p)) panels.Add(p);
            }
            if (panels.Count == 0) return null;
            PanelId? selected = node["selected"] is JsonValue sv && sv.TryGetValue<string>(out var sk) ? new PanelId(sk) : null;
            return new PanelGroup(panels, selected);
        }
        var axis = (node["axis"] as JsonValue)?.TryGetValue<string>(out var a) == true ? a : null;
        if (axis is not ("h" or "v") || node["children"] is not JsonArray children) return null;
        var shares = (node["fractions"] as JsonArray ?? []).Select(Number).ToList();
        var kids = new List<PanelNode>();
        var f = new List<double>();
        for (var i = 0; i < children.Count; i++)
        {
            if (children[i] is not { } c || DecodeNode(c, known, seen) is not { } n) continue;
            kids.Add(n);
            f.Add(i < shares.Count && shares[i] is { } share ? share : 1);
        }
        if (kids.Count == 0) return null;
        return kids.Count == 1 ? kids[0] : new PanelSplit(axis == "h" ? SplitAxis.Horizontal : SplitAxis.Vertical, kids, f);
    }

    static double? Number(JsonNode? n) =>
        n is JsonValue v && v.GetValueKind() == System.Text.Json.JsonValueKind.Number
            && double.TryParse(v.ToJsonString(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var d) ? d : null;
}
