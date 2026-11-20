using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;

namespace MKLight.Tests.Jobs;

public class MeritTests
{
    [Fact]
    public void DefaultMeritJobSettingsHoldEflAndTheGenerator()
    {
        var m = new MeritSettings();
        var s = m.JobSettings();
        Assert.True(s["hold_efl"]!.GetValue<bool>());
        var dm = Assert.IsType<JsonObject>(s["default_merit"]);
        Assert.Equal("rms_spot", dm["criterion"]!["type"]!.GetValue<string>());
        Assert.Equal("polychromatic", dm["criterion"]!["scope"]!.GetValue<string>());
        Assert.Equal(1.0, dm["thickness"]!["glass_min_centre"]!.GetValue<double>());
        Assert.False(s.ContainsKey("merit"));
    }

    [Fact]
    public void SectionRoundTripsDefaultAndCustom()
    {
        var m = new MeritSettings
        {
            Mode = MeritMode.Custom, HoldEfl = false,
            Defaults = new DefaultMeritSettings { Criterion = MeritCriterion.Mtf, MtfFrequencies = [20, 40], MtfTarget = 0.6, WorstField = true, Thickness = false, Fields = [0, 2] },
        };
        m.Add(OperandCatalog.Descriptor("efl")!);
        m.Add(OperandCatalog.Descriptor("rms_spot")!);
        m.SpecOptions["on_failure"] = new JsonObject { ["mode"] = "penalty", ["value"] = 1000.0 };
        var back = MeritSettings.FromSection(JsonTree.Parse(m.Section.ToString()));
        Assert.True(back.SameAs(m));
        Assert.Equal(MeritMode.Custom, back.Mode);
        Assert.Equal(2, back.Operands.Count);
        Assert.Equal(new DefaultMeritSettings { Criterion = MeritCriterion.Mtf, MtfFrequencies = [20, 40], MtfTarget = 0.6, WorstField = true, Thickness = false, Fields = [0, 2] }, back.Defaults);
        var spec = Assert.IsType<JsonObject>(back.JobSettings()["merit"]);
        Assert.Equal(1, spec["schema_version"]!.GetValue<int>());
        Assert.Equal("penalty", spec["on_failure"]!["mode"]!.GetValue<string>());
    }

    [Fact]
    public void SingleFieldOperandsStartAtTheFirstField()
    {
        var op = new MeritOperand(OperandCatalog.Descriptor("ray_aberration")!.NewKind());
        Assert.Equal("index", op.Field["field"]!.GetValue<string>());
        Assert.Equal("y", op.Kind["axis"]!.GetValue<string>());
        Assert.Empty(op.LocalIssues(3, 3));
        op.Field = new JsonObject { ["field"] = "all" };
        Assert.Contains("Needs one field, not all fields.", op.LocalIssues(3, 3));
    }

    [Fact]
    public void CombineReferencesFollowInsertDeleteAndMove()
    {
        var m = new MeritSettings();
        m.Add(OperandCatalog.Descriptor("efl")!);            // 0
        m.Add(OperandCatalog.Descriptor("bfl")!);            // 1
        m.Add(OperandCatalog.Descriptor("combine")!);        // 2
        m.Operands[2].Kind["args"] = new JsonArray(0, 1);
        int[] Args() => m.Operands.First(o => o.Type == "combine").Kind["args"]!.AsArray().Select(n => n!.GetValue<int>()).ToArray();

        m.Add(OperandCatalog.Descriptor("magnification")!, after: -1); // inserted at 0
        Assert.Equal([1, 2], Args());
        m.Move(0, 2); // magnification after bfl
        Assert.Equal([0, 1], Args());
        m.Delete(new HashSet<Guid> { m.Operands[0].Id });
        Assert.Equal([0], Args());
        var copies = m.Duplicate(new HashSet<Guid> { m.Operands[0].Id });
        Assert.Single(copies);
        Assert.Equal([0], Args());
        Assert.Equal(m.Operands[0].Type, m.Operands[1].Type);
        Assert.NotEqual(m.Operands[0].Id, m.Operands[1].Id);
    }

    [Fact]
    public void EvaluationReadsRowsAndSpec()
    {
        var json = JsonNode.Parse("""
        {"merit": 0.25, "violation": 0, "status": {"status": "degraded", "failures": 2},
         "backend": {"reported_on": "cpu-f64-reference"},
         "merit_spec": {"name": "default", "on_failure": {"mode": "penalty"}, "operands": [
            {"kind": {"type": "efl"}, "target": 50, "weight": 0, "role": "constraint", "label": "hold EFL"},
            {"kind": {"type": "rms_spot"}, "target": 0, "weight": 1}]},
         "operands": [{"index": 0, "kind": "efl", "value": 50, "contribution_fraction": null},
                      {"index": 1, "kind": "rms_spot", "value": null, "error": "ray failed"}]}
        """);
        var e = MeritEvaluation.FromJson(json)!;
        Assert.Equal(0.25, e.Merit);
        Assert.Equal(MeritStatus.Degraded, e.Status);
        Assert.Equal(2, e.SpecOperands.Count);
        Assert.True(MeritSettings.IsEflHold(e.SpecOperands[0]));
        Assert.True(e.SpecOptions.ContainsKey("on_failure"));
        Assert.False(e.SpecOptions.ContainsKey("operands"));
        Assert.Equal("ray failed", e.RowAt(1, "rms_spot")!.Error);
        Assert.Null(e.RowAt(1, "efl"));
        Assert.Contains("2 residuals carry penalty values", e.Help());
    }

    [Fact]
    public void ProgressStagesParse()
    {
        Assert.Equal(new MeritPoint(12, 3.4e-4), MeritPoint.Parse("lm 12: merit 3.4e-4"));
        Assert.Null(MeritPoint.Parse("preparing the merit evaluator"));
    }

    [Fact]
    public void OptimizeResultNamesThePreview()
    {
        var r = OptimizeResult.FromJson(JsonNode.Parse("""
            {"status": "finished", "improved": true, "start": {"merit": 0.02}, "final": {"merit": 0.0116, "feasible": true},
             "iterations": 7, "evaluations": 51, "variables": [], "history": [{"iteration": 1, "best_merit": 0.02}]}
            """))!;
        Assert.Equal("Optimize result · merit 0.0116 (−42 %)", r.PreviewTitle());
        Assert.Equal("7 iterations, 51 evaluations.", r.StatusText());
        Assert.Single(r.History);
    }

    [Fact]
    public void MeritEditsAreUndoableSections()
    {
        var doc = DesignDocument.FromSample("double_gauss");
        var m = MeritSettings.FromSection(doc.Section("merit"));
        m.Defaults = m.Defaults with { Rings = 4 };
        Assert.True(doc.EditSection("merit", m.Section, "Change Pupil Rings"));
        Assert.Equal(4, MeritSettings.FromSection(doc.Section("merit")).Defaults.Rings);
        doc.Undo();
        Assert.Equal(3, MeritSettings.FromSection(doc.Section("merit")).Defaults.Rings);
    }
}
