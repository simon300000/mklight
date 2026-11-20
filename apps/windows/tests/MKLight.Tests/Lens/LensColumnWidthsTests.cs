using System.Text.Json.Nodes;
using MKLight.App.ViewModels;

namespace MKLight.Tests.Lens;

public class LensColumnWidthsTests
{
    static LensGridColumn Limits(LensColumn c) => LensGridLayout.Columns(false).First(x => x.Column == c);

    static JsonArray Views(JsonObject? layoutUi = null) =>
    [
        new JsonObject { ["key"] = "layout", ["analysis"] = "layout", ["settings"] = new JsonObject(), ["ui"] = layoutUi ?? new JsonObject { ["mode"] = "3D" } },
        new JsonObject { ["key"] = "pane-1", ["analysis"] = "spot_diagram", ["settings"] = new JsonObject(), ["ui"] = new JsonObject { ["auto_update"] = true } },
    ];

    [Fact]
    public void KeysAreStableAndRoundTrip()
    {
        foreach (var c in Enum.GetValues<LensColumn>()) Assert.Equal(c, LensColumnWidths.FromKey(LensColumnWidths.Key(c)));
        Assert.Equal("semi_diameter", LensColumnWidths.Key(LensColumn.SemiDiameter));
        Assert.Equal("radius_x", LensColumnWidths.Key(LensColumn.RadiusX));
        // Display titles are not keys.
        Assert.Null(LensColumnWidths.FromKey("Semi-Diameter"));
        Assert.Null(LensColumnWidths.FromKey("Radius"));
    }

    [Fact]
    public void SetClampsToTheColumnLimitsInWholePixels()
    {
        var w = new LensColumnWidths();
        var radius = Limits(LensColumn.Radius); // 84 … 220
        Assert.True(w.Set(radius, 150.4));
        Assert.Equal(150, w.Fixed[LensColumn.Radius]);
        Assert.False(w.Set(radius, 150.2)); // the same whole width
        Assert.True(w.Set(radius, 10));
        Assert.Equal(84, w.Fixed[LensColumn.Radius]);
        Assert.True(w.Set(radius, 5000));
        Assert.Equal(220, w.Fixed[LensColumn.Radius]);
        Assert.False(w.Set(radius, double.NaN));
        // The comment column can grow without limit.
        Assert.True(w.Set(Limits(LensColumn.Comment), 900));
        Assert.Equal(900, w.Fixed[LensColumn.Comment]);
    }

    [Fact]
    public void HeldColumnsKeepTheirWidthAndTheOthersShareTheRest()
    {
        var columns = LensGridLayout.Columns(false);
        var w = new LensColumnWidths();
        w.Set(Limits(LensColumn.Comment), 200);
        w.Set(Limits(LensColumn.Radius), 150);
        var applied = w.Apply(columns);
        const double total = 1100;
        var widths = LensGridLayout.Widths(applied, total);
        var k = Array.IndexOf(LensColumns.Visible, LensColumn.Comment);
        Assert.Equal(200, widths[k], 6);
        Assert.Equal(150, widths[Array.IndexOf(LensColumns.Visible, LensColumn.Radius)], 6);
        Assert.Equal(total, widths.Sum(), 6);
        // Below the minimum the grid scrolls: the held widths count in full.
        Assert.Equal(LensGridLayout.MinimumWidth(columns) - 72 - 84 + 200 + 150, LensGridLayout.MinimumWidth(applied), 6);
    }

    [Fact]
    public void FitTakesTheWidestContentWithinTheLimits()
    {
        var material = Limits(LensColumn.Material); // 84 … 240
        Assert.Equal(131, LensColumnWidths.FitWidth(material, 130.2));
        Assert.Equal(84, LensColumnWidths.FitWidth(material, 20));
        Assert.Equal(240, LensColumnWidths.FitWidth(material, 400));
        var w = new LensColumnWidths();
        Assert.True(w.Fit(material, 130.2));
        Assert.False(w.Fit(material, 130.9));
        Assert.Equal(131, w.Fixed[LensColumn.Material]);
    }

    [Fact]
    public void ResetLetsEveryColumnFollowTheWidthAgain()
    {
        var w = new LensColumnWidths();
        Assert.False(w.Reset());
        w.Set(Limits(LensColumn.Conic), 100);
        Assert.True(w.Reset());
        Assert.True(w.IsEmpty);
        Assert.Equal(LensGridLayout.Columns(false), w.Apply(LensGridLayout.Columns(false)));
    }

    [Fact]
    public void JsonRoundTripsAndSkipsWhatItCannotRead()
    {
        var w = new LensColumnWidths();
        w.Set(Limits(LensColumn.Radius), 140);
        w.Set(Limits(LensColumn.Comment), 210);
        var json = w.ToJson();
        Assert.Equal("""{"comment":210,"radius":140}""", json.ToJsonString());
        Assert.True(LensColumnWidths.FromJson(JsonNode.Parse(json.ToJsonString())).SameAs(w));

        var odd = LensColumnWidths.FromJson(JsonNode.Parse("""{"radius": -3, "Radius": 120, "nope": 50, "conic": "wide", "thickness": 0, "material": 130}"""));
        Assert.Equal(new[] { LensColumn.Material }, odd.Fixed.Keys);
        Assert.True(LensColumnWidths.FromJson(JsonNode.Parse("[1, 2]")).IsEmpty);
        Assert.True(LensColumnWidths.FromJson(null).IsEmpty);
    }

    [Fact]
    public void WidthsGoIntoTheLayoutEntryKeepingEverythingElse()
    {
        var views = Views();
        var before = views.ToJsonString();
        var w = new LensColumnWidths();
        w.Set(Limits(LensColumn.Radius), 140);
        var written = w.WriteTo(views)!;
        Assert.Equal(before, views.ToJsonString()); // the input is not changed
        Assert.Equal(2, written.Count);
        var ui = (JsonObject)written[0]!["ui"]!;
        Assert.Equal("3D", (string?)ui["mode"]);
        Assert.Equal(140, (double)ui[LensColumnWidths.UiKey]!["radius"]!);
        Assert.Equal(views[1]!.ToJsonString(), written[1]!.ToJsonString());
        Assert.True(LensColumnWidths.Read(written).SameAs(w));

        // No widths: the key goes, the rest stays.
        var cleared = new LensColumnWidths().WriteTo(written)!;
        Assert.Null(cleared[0]!["ui"]![LensColumnWidths.UiKey]);
        Assert.Equal("3D", (string?)cleared[0]!["ui"]!["mode"]);
    }

    [Fact]
    public void NothingIsWrittenBeforeTheViewsHaveALayoutEntry()
    {
        var w = new LensColumnWidths();
        w.Set(Limits(LensColumn.Radius), 140);
        Assert.Null(w.WriteTo(null));
        Assert.Null(w.WriteTo(new JsonArray()));
        Assert.Null(w.WriteTo(new JsonArray { new JsonObject { ["key"] = "pane-1", ["analysis"] = "spot_diagram" } }));
        Assert.True(LensColumnWidths.Read(new JsonArray()).IsEmpty);
    }

    [Fact]
    public void TheTypeMenuOffersBothCylinderPresetsAndChecksACylindersOwn()
    {
        var anamorphic = SurfaceTypeCatalog.All.Where(e => e.Group == SurfaceTypeGroup.Anamorphic).Select(e => e.Title).ToArray();
        Assert.Equal(new[] { "Toroid", "Biconic", "Cylinder (power in x)", "Cylinder (power in y)" }, anamorphic);
        foreach (var axis in Enum.GetValues<MKLight.Model.CylinderAxis>())
        {
            var shape = MKLight.Model.SurfaceShape.Cylinder(0.02, axis);
            var entry = SurfaceTypeCatalog.EntryFor(SurfaceTypeChoice.For(shape));
            Assert.Equal(axis == MKLight.Model.CylinderAxis.X ? "Cylinder (power in x)" : "Cylinder (power in y)", entry.Title);
        }
        // Titles are unique: the type cell's automation value names one entry.
        Assert.Equal(SurfaceTypeCatalog.All.Length, SurfaceTypeCatalog.All.Select(e => e.Title.ToUpperInvariant()).Distinct().Count());
    }

    [Fact]
    public void ThePanelPersistenceKeepsTheWidthsWhenItRewritesTheViews()
    {
        // PanelPersistence.Write reads the views, replaces panes and arrangement, and writes them back.
        var w = new LensColumnWidths();
        w.Set(Limits(LensColumn.Comment), 230);
        var views = w.WriteTo(Views())!;
        var layout = WorkspaceLayout.Read(views);
        layout.Panes.Clear();
        layout.Arrangement = new JsonObject { ["windows"] = new JsonArray() };
        new WorkspaceLayout.ViewState("2D", "wavelength", null).WriteTo(layout.LayoutUi);
        var rewritten = layout.ToViews();
        Assert.True(LensColumnWidths.Read(rewritten).SameAs(w));
    }
}
