using System.Text.Json.Nodes;
using MKLight.App.ViewModels;

namespace MKLight.Tests.Panels;

/// <summary>
/// The panel tree: moving panels into tabs, splits and separate windows,
/// showing and hiding, and saving it with the project.
/// </summary>
public class PanelArrangementTests
{
    static readonly PanelId Lens = PanelId.LensData, Layout = PanelId.Layout;
    static PanelId P(string k) => PanelId.Analysis(k);

    /// <summary>The tree as text: <c>v(...)</c>/<c>h(...)</c> for splits, <c>[a,b]</c> for tab groups (<c>*</c> marks the front tab of a group of several).</summary>
    static string Describe(PanelNode? node) => node switch
    {
        null => "-",
        PanelGroup g => "[" + string.Join(",", g.Panels.Select(p => (g.Panels.Count > 1 && g.Current == p ? "*" : "") + Short(p))) + "]",
        PanelSplit s => (s.Axis == SplitAxis.Horizontal ? "h(" : "v(") + string.Join("|", s.Children.Select(Describe)) + ")",
        _ => "?",
    };

    static string Short(PanelId p) => p == Lens ? "lens" : p.Key;
    static string Tree(PanelArrangement a) => Describe(a.Main.Root);
    static PanelGroup Group(PanelArrangement a, PanelId p) => a.Location(p)!.Value.Group;

    [Fact]
    public void StandardArrangement()
    {
        var a = PanelArrangement.Standard(["p1", "p2"]);
        Assert.Equal("v([lens]|h([layout]|[p1]|[p2]))", Tree(a));
        var root = Assert.IsType<PanelSplit>(a.Main.Root);
        var lower = Assert.IsType<PanelSplit>(root.Children[1]);
        Assert.Equal(0.36, root.Fractions[0], 9);
        Assert.Equal(0.42, lower.Fractions[0], 9);
        Assert.Equal(0.29, lower.Fractions[1], 9);
        Assert.Equal("[lens]", Tree(PanelArrangement.Standard([], showLayout: false)));
        Assert.Equal("h([layout]|[p1])", Tree(PanelArrangement.Standard(["p1"], showLensData: false)));
    }

    [Fact]
    public void DropOnAGroupMakesTabs()
    {
        var a = PanelArrangement.Standard(["p1", "p2"]);
        var target = Group(a, P("p1"));
        a.Move(P("p2"), new PanelDropTarget.Tab(target.Id, null));
        Assert.Equal("v([lens]|h([layout]|[p1,*p2]))", Tree(a));
        // Reorder within the group, then take a tab out beside the layout.
        a.Move(P("p2"), new PanelDropTarget.Tab(target.Id, 0));
        Assert.Equal("v([lens]|h([layout]|[*p2,p1]))", Tree(a));
        a.Move(P("p1"), new PanelDropTarget.Split(Group(a, Layout).Id, PanelEdge.Bottom));
        Assert.Equal("v([lens]|h(v([layout]|[p1])|[p2]))", Tree(a));
    }

    [Fact]
    public void SplitsMergeIntoARowOfTheSameAxis()
    {
        var a = PanelArrangement.Standard(["p1", "p2"]);
        a.Move(P("p2"), new PanelDropTarget.Split(Group(a, Layout).Id, PanelEdge.Leading));
        Assert.Equal("v([lens]|h([p2]|[layout]|[p1]))", Tree(a));
        a.Move(Lens, new PanelDropTarget.WindowEdge(a.Main.Id, PanelEdge.Trailing));
        Assert.Equal("h([p2]|[layout]|[p1]|[lens])", Tree(a));
        var shares = Assert.IsType<PanelSplit>(a.Main.Root).Fractions;
        Assert.Equal(1, shares.Sum(), 9);
        Assert.Equal(0.3, shares[^1], 9);
    }

    [Fact]
    public void NothingHappensWhereNothingWouldChange()
    {
        var a = PanelArrangement.Standard(["p1"]);
        var g = Group(a, P("p1"));
        Assert.False(a.CanMove(P("p1"), new PanelDropTarget.Tab(g.Id, null)));
        Assert.False(a.CanMove(P("p1"), new PanelDropTarget.Split(g.Id, PanelEdge.Leading)));
        Assert.True(a.CanMove(P("p1"), new PanelDropTarget.NewWindow()));
        var w = a.Move(P("p1"), new PanelDropTarget.NewWindow(), new PanelFrame(10, 20, 500, 400))!;
        // The only panel of a separate window: its window is the panel.
        Assert.False(a.CanMove(P("p1"), new PanelDropTarget.NewWindow()));
        Assert.False(a.CanMove(P("p1"), new PanelDropTarget.WindowEdge(w.Id, PanelEdge.Leading)));
        Assert.True(a.CanMove(P("p1"), new PanelDropTarget.WindowEdge(a.Main.Id, PanelEdge.Trailing)));
        Assert.Null(a.Move(P("p1"), new PanelDropTarget.NewWindow()));
    }

    [Fact]
    public void SeparateWindowsComeAndGo()
    {
        var a = PanelArrangement.Standard(["p1", "p2"]);
        var w = a.Move(P("p1"), new PanelDropTarget.NewWindow(), new PanelFrame(10, 20, 500, 400))!;
        Assert.Single(a.Detached);
        Assert.Equal([P("p1")], w.Panels);
        Assert.Equal(500, w.Frame!.Value.Width);
        Assert.True(a.IsDetached(P("p1")) && a.IsAloneInWindow(P("p1")));
        Assert.Equal("v([lens]|h([layout]|[p2]))", Tree(a));
        // A second panel into the separate window, beside the first.
        a.Move(P("p2"), new PanelDropTarget.Split(Group(a, P("p1")).Id, PanelEdge.Trailing));
        Assert.Equal("h([p1]|[p2])", Describe(w.Root));
        Assert.False(a.IsAloneInWindow(P("p1")));
        // Docking puts the window's arrangement back under the lens data.
        a.Dock(w);
        Assert.Empty(a.Detached);
        Assert.Equal("v([lens]|h([layout]|[p1]|[p2]))", Tree(a));
        // A lone panel goes back where its kind belongs.
        a.Move(P("p1"), new PanelDropTarget.NewWindow());
        a.Move(Lens, new PanelDropTarget.NewWindow());
        a.ReturnToMain(P("p1"));
        Assert.Equal("h([layout]|[p2]|[p1])", Tree(a));
        a.Dock(a.Detached[0]);
        Assert.Equal("v([lens]|h([layout]|[p2]|[p1]))", Tree(a));
        Assert.Empty(a.Detached);
        // Moving a separate window's last panel away closes the window.
        var w2 = a.Move(Lens, new PanelDropTarget.NewWindow())!;
        Assert.Equal([w2.Id], a.Detached.Select(d => d.Id));
        a.Move(Lens, new PanelDropTarget.WindowEdge(a.Main.Id, PanelEdge.Top));
        Assert.Empty(a.Detached);
        Assert.Empty(a.Discard(a.Main));
        // Closing a separate window hands back its panels.
        var w3 = a.Move(P("p2"), new PanelDropTarget.NewWindow())!;
        Assert.Equal([P("p2")], a.Discard(w3));
        Assert.False(a.Contains(P("p2")));
    }

    [Fact]
    public void ShowingAndHiding()
    {
        var a = PanelArrangement.Standard(["p1"]);
        a.Remove(Layout);
        Assert.Equal("v([lens]|[p1])", Tree(a));
        a.Show(Layout);
        Assert.Equal("v([lens]|h([layout]|[p1]))", Tree(a));
        a.Remove(Lens);
        Assert.Equal("h([layout]|[p1])", Tree(a));
        a.Show(Lens);
        Assert.Equal("v([lens]|h([layout]|[p1]))", Tree(a));
        a.Remove(Layout);
        a.Remove(P("p1"));
        a.Remove(Lens);
        Assert.Null(a.Main.Root);
        a.Show(Layout);
        a.Show(Lens);
        Assert.Equal("v([lens]|[layout])", Tree(a));
    }

    [Fact]
    public void NewPanesGoBesideTheirNeighbour()
    {
        var a = PanelArrangement.Standard(["p1", "p2"]);
        a.Place(P("p3"), P("p1"));
        Assert.Equal("v([lens]|h([layout]|[p1]|[p3]|[p2]))", Tree(a));
        var a2 = PanelArrangement.Standard([]);
        a2.Place(P("p1"), null);
        Assert.Equal("v([lens]|h([layout]|[p1]))", Tree(a2));
        var a3 = PanelArrangement.Standard([], showLayout: false);
        a3.Place(P("p1"), null);
        Assert.Equal("v([lens]|[p1])", Tree(a3));
        var w = a.Move(P("p2"), new PanelDropTarget.NewWindow())!;
        a.Place(P("p4"), w.Id);
        Assert.Equal("h([p2]|[p4])", Describe(w.Root));
    }

    [Fact]
    public void ResetBringsEveryPanelBack()
    {
        var a = PanelArrangement.Standard(["p1", "p2"]);
        a.Move(P("p1"), new PanelDropTarget.NewWindow());
        a.Remove(Lens);
        a.Reset(PanelArrangement.Standard(["p1", "p2"]));
        Assert.Empty(a.Detached);
        Assert.Equal("v([lens]|h([layout]|[p1]|[p2]))", Tree(a));
    }

    [Fact]
    public void DividersResizeAndEqualize()
    {
        var a = PanelArrangement.Standard(["p1", "p2"]);
        var row = (PanelSplit)((PanelSplit)a.Main.Root!).Children[1];
        row.Resize(0, 0.2, 0.51);
        Assert.Equal([0.2, 0.51, 0.29], row.Fractions.Select(f => Math.Round(f, 9)));
        row.Equalize();
        Assert.All(row.Fractions, f => Assert.Equal(1.0 / 3, f, 9));
    }

    [Fact]
    public void ChangesAreAnnounced()
    {
        var a = PanelArrangement.Standard(["p1"]);
        var n = 0;
        a.Changed += () => n++;
        a.Remove(Layout);
        a.Show(Layout);
        a.Select(Layout); // already in front: no change
        a.Move(P("p1"), new PanelDropTarget.NewWindow());
        Assert.Equal(3, n);
    }

    [Fact]
    public void SavedWithTheProject()
    {
        var a = PanelArrangement.Standard(["p1", "p2", "p3"]);
        a.Move(P("p2"), new PanelDropTarget.Tab(Group(a, P("p1")).Id, null));
        a.Move(P("p3"), new PanelDropTarget.NewWindow(), new PanelFrame(100, 120, 640, 480));
        var known = new HashSet<PanelId> { Lens, Layout, P("p1"), P("p2"), P("p3") };
        var json = a.ToJson();
        Assert.Equal("h", json["main"]!["children"]![1]!["axis"]!.GetValue<string>());
        Assert.Equal("p2", json["main"]!["children"]![1]!["children"]![1]!["selected"]!.GetValue<string>());
        var back = PanelArrangement.Decode(JsonNode.Parse(json.ToJsonString()), known)!;
        Assert.Equal(Tree(a), Tree(back));
        Assert.Single(back.Detached);
        Assert.Equal(new PanelFrame(100, 120, 640, 480), back.Detached[0].Frame);
        Assert.Equal("[p3]", Describe(back.Detached[0].Root));
        // Unknown panes (closed since) and duplicates are dropped; the tree tidies up.
        var fewer = PanelArrangement.Decode(json, new HashSet<PanelId> { Lens, P("p1") })!;
        Assert.Equal("v([lens]|[p1])", Tree(fewer));
        Assert.Empty(fewer.Detached);
        var dup = PanelArrangement.Decode(JsonNode.Parse("""{"main": {"axis": "h", "children": [{"tabs": ["p1"]}, {"tabs": ["p1", "layout"]}], "fractions": [1, 3]}}"""), known)!;
        Assert.Equal("h([p1]|[layout])", Tree(dup));
        Assert.Equal(0.75, ((PanelSplit)dup.Main.Root!).Fractions[1], 9);
        Assert.Null(PanelArrangement.Decode(new JsonArray(), known));
        // A frame too small to be a window is dropped (the window cascades instead).
        var tiny = PanelArrangement.Decode(JsonNode.Parse("""{"windows": [{"frame": [0, 0, 50, 50], "root": {"tabs": ["p3"]}}]}"""), known)!;
        Assert.Null(tiny.Detached[0].Frame);
    }

    /// <summary>Views without an arrangement open in the standard one; a saved arrangement is kept.</summary>
    [Fact]
    public void WorkspaceLayoutCarriesTheArrangement()
    {
        var plain = JsonNode.Parse("""
            [{"key": "layout", "ui": {"mode": "3D", "stretch": 5}, "settings": {"rays": 9}},
             {"key": "p1", "analysis": "spot_diagram", "settings": {"x": 1}},
             {"key": "p2", "analysis": "ray_fan", "ui": {"auto_update": false, "settings_visible": true}},
             {"key": "p2", "analysis": "mtf"},
             {"key": "p3"}]
            """);
        var layout = WorkspaceLayout.Read(plain);
        Assert.True(layout.IsSaved);
        Assert.Equal(["p1", "p2"], layout.Panes.Select(p => p.Key));
        Assert.False(layout.Panes[1].AutoUpdate);
        Assert.True(layout.Panes[1].ShowsSettings);
        Assert.True(layout.Panes[0].AutoUpdate);
        var standard = layout.MakeArrangement();
        Assert.Equal("v([lens]|h([layout]|[p1]|[p2]))", Tree(standard));
        var row = (PanelSplit)((PanelSplit)standard.Main.Root!).Children[1];
        Assert.Equal(row.Fractions[1], row.Fractions[2], 9);

        standard.Move(P("p2"), new PanelDropTarget.NewWindow(), new PanelFrame(0, 0, 600, 400));
        layout.Arrangement = standard.ToJson();
        var views = layout.ToViews();
        // The layout entry keeps what the layout's own code saved.
        Assert.Equal("3D", views[0]!["ui"]!["mode"]!.GetValue<string>());
        Assert.Equal(9, views[0]!["settings"]!["rays"]!.GetValue<int>());
        var reread = WorkspaceLayout.Read(JsonNode.Parse(views.ToJsonString()));
        Assert.True(JsonNode.DeepEquals(reread.Arrangement, standard.ToJson()));
        Assert.True(JsonNode.DeepEquals(reread.ToViews(), views));
        var again = reread.MakeArrangement();
        Assert.Equal("v([lens]|h([layout]|[p1]))", Tree(again));
        Assert.Single(again.Detached);

        Assert.False(WorkspaceLayout.Read(new JsonArray()).IsSaved);
        Assert.Equal("v([lens]|[layout])", Tree(WorkspaceLayout.Read(null).MakeArrangement()));
    }
}
