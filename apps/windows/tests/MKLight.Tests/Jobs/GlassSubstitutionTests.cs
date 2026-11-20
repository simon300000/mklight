using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;

namespace MKLight.Tests.Jobs;

public class GlassSubstitutionTests
{
    static OpticalSystem Cooke() => SampleSystems.Get("cooke_triplet")!;

    [Fact]
    public void SlotsAreTheGlassGapsAndAllAreSelectedFirst()
    {
        var s = Cooke();
        var slots = GlassSlot.All(s);
        Assert.Equal([1, 3, 5], slots.Select(x => x.Index));
        Assert.Equal(["N-SK16", "F2", "N-SK16"], slots.Select(x => x.Material));
        var settings = new GlassSubstitutionSettings();
        settings.Configure(s);
        Assert.Equal(3, settings.Selected.Count);
    }

    [Fact]
    public void SettingsCarryCandidatesMeritAndFreeVariables()
    {
        var s = Cooke();
        var settings = new GlassSubstitutionSettings();
        settings.Configure(s);
        var slots = GlassSlot.All(s);
        settings.Selected.Remove(slots[2].Id);
        settings.Overrides[slots[1].Id] = new GlassCandidates { Rule = CandidateRule.Window, WindowNd = 0.02 };
        var vars = new[]
        {
            new DesignVariable(new ParamPath.Curvature(s.Surfaces[1].Id)),
            new DesignVariable(new ParamPath.MaterialDnD(slots[0].Id)), // the replaced glass's offset: left out
            new DesignVariable(new ParamPath.Thickness(s.Surfaces[2].Id), 1, 1), // pinned: left out
        };
        var j = settings.Settings(s, new MeritSettings(), vars);
        Assert.Equal("tabu", j["algorithm"]!.GetValue<string>());
        Assert.True(j.ContainsKey("default_merit"));
        var v = Assert.IsType<JsonArray>(j["variables"]);
        Assert.Single(v);
        var mv = Assert.IsType<JsonArray>(j["material_variables"]);
        Assert.Equal(2, mv.Count);
        Assert.Equal(10, mv[0]!["candidates"]!["nearest"]!.GetValue<int>());
        Assert.Null(mv[1]!["candidates"]!["nearest"]);
        Assert.Equal(0.02, mv[1]!["candidates"]!["window"]!["nd"]!.GetValue<double>());
    }

    [Fact]
    public void CandidatesSummaryAndNames()
    {
        Assert.Equal("10 nearest · all catalogs", new GlassCandidates().Summary);
        Assert.Equal(["N-BK7", "SCHOTT:F2"], GlassCandidates.Names("N-BK7, SCHOTT:F2;"));
        var statuses = new GlassCandidates().ToJson()["statuses"]!.AsArray().Select(n => n!.GetValue<string>());
        Assert.Equal(["preferred", "standard", "melt"], statuses);
    }

    [Fact]
    public void ProgressAndResultParse()
    {
        var p = GlassProgressInfo.Parse("glass_substitution: combinations 12; best merit 3.5e-4; glasses [{\"name\":\"N-LAK9\"},{\"name\":\"SF5\"}]")!;
        Assert.Equal(12, p.Combinations);
        Assert.Equal(3.5e-4, p.BestMerit);
        Assert.Equal(["N-LAK9", "SF5"], p.Glasses);
        Assert.Null(GlassProgressInfo.Parse("optimize 3: merit 1"));

        var r = GlassSubstitutionResult.FromJson(JsonNode.Parse("""
            {"status": "finished", "starting_merit": 4e-4, "statistics": {"combinations": 24, "failed": 1},
             "history": [{"best_merit": 3e-4}, {"best_merit": 3.2e-4}, {"best_merit": 2e-4}],
             "results": [{"rank": 1, "merit": 2e-4, "glasses": [{"surface": 1, "catalog": "SCHOTT", "name": "N-LAK9", "nd": 1.691, "vd": 54.7}],
                          "variables": [{"param": {"param": "curvature", "surface": 1}, "start": 0.04, "value": 0.041}]}]}
            """))!;
        Assert.Equal([3e-4, 3e-4, 2e-4], r.History);
        Assert.Equal("1 variable re-optimized", r.Ranked[0].ChangeSummary);
        Assert.Equal("24 combinations re-optimized, 1 failed. Select a combination to preview it in the window.", r.StatusText(false));
    }

    [Fact]
    public void OffsetNoteNamesTheMeltOffset()
    {
        var s = Cooke();
        s = s.WithSurface(1, sf => sf with { Material = new MaterialRef.CatalogGlass { Name = "N-SK16", Offset = new MaterialOffset(0.001, -0.5) } });
        var settings = new GlassSubstitutionSettings();
        settings.Configure(s);
        var note = settings.OffsetNote(s);
        Assert.NotNull(note);
        Assert.StartsWith("S1 N-SK16 +0.001/−0.5 carries a melt offset", note);
    }

    [Fact]
    public void ApplyIsOneUndoStep()
    {
        var doc = new DesignDocument(new ProjectFile(Cooke()));
        var changed = doc.System.WithSurface(3, sf => sf with { Material = MaterialRef.Glass("SF5") });
        Assert.True(DesignVariables.ReplaceSystem(doc, changed, "Apply Glass Substitution"));
        Assert.Equal("SF5", doc.System.Surfaces[3].Material.DisplayName);
        doc.Undo();
        Assert.Equal("F2", doc.System.Surfaces[3].Material.DisplayName);
        Assert.False(doc.CanUndo);
    }
}
