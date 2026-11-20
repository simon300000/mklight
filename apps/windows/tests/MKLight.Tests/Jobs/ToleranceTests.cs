// The engine test awaits the job without the test's token on purpose (see JobTests).
#pragma warning disable xUnit1051
using System.Text.Json.Nodes;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;
using MKLight.Tests.Engine;

namespace MKLight.Tests.Jobs;

public class ToleranceTests
{
    static OpticalSystem DoubleGauss => SampleSystems.Get("double_gauss")!;

    static SurfaceID Id(OpticalSystem s, int index) => s.Surfaces[index].Id;

    /// <summary>The Double Gauss with a 10 mm clear semi-diameter everywhere and V_d known for every catalogue glass.</summary>
    static ToleranceContext Context(OpticalSystem s)
    {
        var semi = s.Surfaces.ToDictionary(x => x.Id, _ => 10.0);
        var abbe = new Dictionary<SurfaceID, double>();
        foreach (var x in s.Surfaces) if (x.Material is MaterialRef.CatalogGlass) abbe[x.Id] = x.Material.DisplayName == "F2" ? 36.4 : 60.3;
        return new ToleranceContext(s) { SemiDiameters = semi, AbbeNumbers = abbe };
    }

    // ---- Generator ----

    [Fact]
    public void GeneratorRowsOfTheDoubleGauss()
    {
        var s = DoubleGauss;
        var rows = new ToleranceGenerator().Rows(s);
        var counts = rows.GroupBy(r => r.Kind).ToDictionary(g => g.Key, g => g.Count());
        // Ten lens surfaces (S1–5, S7–11), the stop's gap included, S11 → image is the back focus (compensator).
        Assert.Equal(10, counts[ToleranceKind.Radius]);
        Assert.Equal(10, counts[ToleranceKind.IrregularityAstigmatism]);
        Assert.Equal(10, counts[ToleranceKind.IrregularitySpherical]);
        Assert.Equal(10, counts[ToleranceKind.Thickness]);
        Assert.Equal(10, counts[ToleranceKind.DecenterX]);
        Assert.Equal(10, counts[ToleranceKind.TiltY]);
        Assert.Equal(6, counts[ToleranceKind.Index]);
        Assert.Equal(6, counts[ToleranceKind.Abbe]);
        // Four elements (L1 … L4), each a rigid group.
        Assert.All(new[] { ToleranceKind.ElementDecenterX, ToleranceKind.ElementDecenterY, ToleranceKind.ElementTiltX, ToleranceKind.ElementTiltY }, k => Assert.Equal(4, counts[k]));
        Assert.Equal(108, rows.Count);
        Assert.DoesNotContain(rows, r => r.Kind == ToleranceKind.Thickness && r.Surface == Id(s, 11));
        var l2 = rows.Single(r => r.Kind == ToleranceKind.ElementTiltX && r.Surface == Id(s, 3));
        Assert.Equal(Id(s, 5), l2.Last);
        Assert.Equal("Element tilt x L2", l2.Label(s));
        Assert.Equal("S3–5", l2.SurfaceLabel(s));
        Assert.Equal(new[] { "L1", "L2", "L3", "L4" }, LensElements.Groups(s).Select(g => g.Title));
        Assert.Equal("Air after L1", LensElements.GroupTitle(rows.Single(r => r.Kind == ToleranceKind.Thickness && r.Surface == Id(s, 2)), s, LensElements.Groups(s)));

        var defaults = ToleranceSettings.Defaults(s);
        Assert.Equal(108, defaults.Rows.Count);
        var bf = Assert.Single(defaults.Compensators);
        Assert.Equal(new ParamPath.Thickness(Id(s, 11)), bf.Param);
        Assert.Equal(0.5, bf.Range);
        Assert.Equal("Back focus", bf.Title(s));
    }

    [Fact]
    public void GeneratorLeavesZeroKindsOutAndAppliesPresets()
    {
        var s = DoubleGauss;
        var g = new ToleranceGenerator(ToleranceGeneratorPreset.Commercial) { TiltArcmin = 0, ElementTiltArcmin = 0 };
        var rows = g.Rows(s);
        Assert.DoesNotContain(rows, r => r.Kind is ToleranceKind.TiltX or ToleranceKind.TiltY or ToleranceKind.ElementTiltX or ToleranceKind.ElementTiltY);
        Assert.Equal(5, rows.First(r => r.Kind == ToleranceKind.Radius).Minus);
        Assert.Equal(ToleranceGeneratorPreset.Commercial, g.Preset);
        Assert.Equal(0.1, new ToleranceGenerator(ToleranceGeneratorPreset.Commercial).ElementDecentreMM);
        var back = ToleranceGenerator.FromJson(g.ToJson());
        Assert.True(JsonNode.DeepEquals(g.ToJson(), back.ToJson()));
    }

    [Fact]
    public void CompensatorCandidatesListEachParameterOnce()
    {
        var s = DoubleGauss;
        var groups = ToleranceCompensator.Candidates(s, [new ParamPath.Curvature(Id(s, 1)), new ParamPath.Thickness(Id(s, 11))]);
        Assert.Equal("Back focus", groups[0].Group);
        Assert.Equal(["Back focus", "Air spaces", "Design variables", "Surfaces"], groups.Select(g => g.Group));
        var all = groups.SelectMany(g => g.Params).ToList();
        Assert.Equal(all.Count, all.Distinct().Count());
        // The back focus is not repeated among the variables or the surface thicknesses.
        Assert.Single(groups[2].Params);
        Assert.Contains(groups[1].Params, p => p is ParamPath.Thickness t && t.Surface == Id(s, 2));
    }

    // ---- Fringes and unit conversion ----

    [Fact]
    public void FringesConvertToCurvatureByTheRimDeparture()
    {
        // Flat: Nλ/h².
        Assert.Equal(0.6328e-3 / 100, ToleranceRow.CurvaturePerFringe(0, 10, 0.6328)!.Value, 15);
        // Δc = λ(1 + cos θ)/(2h²), sin θ = c·h.
        var c = 0.05;
        var cosTheta = Math.Sqrt(1 - c * c * 100);
        Assert.Equal(0.5876e-3 * (1 + cosTheta) / 200, ToleranceRow.CurvaturePerFringe(c, 10, 0.5876)!.Value, 15);
        // A clear aperture wider than its sphere (or none) has no conversion.
        Assert.Null(ToleranceRow.CurvaturePerFringe(0.2, 10, 0.6328));
        Assert.Null(ToleranceRow.CurvaturePerFringe(0.05, 0, 0.6328));
        Assert.Null(ToleranceRow.CurvaturePerFringe(0.05, 10, 0));
    }

    [Fact]
    public void RadiusFringesBecomeEngineCurvatureLimitsAndConvertBack()
    {
        var s = DoubleGauss;
        var ctx = Context(s);
        var id = Id(s, 1);
        var curv = s.ValueAt(new ParamPath.Curvature(id))!.Value;
        var k = ToleranceRow.CurvaturePerFringe(curv, 10, 0.6328)!.Value;
        var row = new ToleranceRow(ToleranceKind.Radius, id, 3, 2);
        var l = row.EngineLimits(ctx);
        Assert.True(l.Ok);
        Assert.Equal(3 * k, l.Minus, 15);
        Assert.Equal(2 * k, l.Plus, 15);
        // The test wavelength scales the limits.
        Assert.Equal(2 * 3 * k, row.EngineLimits(ctx with { TestWavelengthUm = 2 * 0.6328 }).Minus, 14);

        // ΔR: the same engine limits, written as radii.
        var mm = row.ConvertedTo(ToleranceUnit.RadiusMM, ctx)!;
        Assert.Equal(ToleranceUnit.RadiusMM, mm.Unit);
        var lm = mm.EngineLimits(ctx);
        Assert.Equal(l.Minus, lm.Minus, 10);
        Assert.Equal(l.Plus, lm.Plus, 10);
        var r = 1 / curv;
        Assert.Equal(r - 1 / (curv + l.Plus), mm.Minus, 6); // a larger curvature is a smaller radius
        Assert.Equal(1 / (curv - l.Minus) - r, mm.Plus, 6);
        // % of R and back to fringes.
        var pct = row.ConvertedTo(ToleranceUnit.Percent, ctx)!;
        Assert.Equal(mm.Minus / Math.Abs(r) * 100, pct.Minus, 6);
        var back = pct.ConvertedTo(ToleranceUnit.Fringes, ctx)!;
        Assert.Equal(3, back.Minus, 6);
        Assert.Equal(2, back.Plus, 6);
        Assert.Equal("±3 fr", new ToleranceRow(ToleranceKind.Radius, id, 3).LimitsText);
        Assert.Equal("−3 / +2 fr", row.LimitsText);

        // A flat surface has no ΔR, but fringes.
        var flat = new ToleranceRow(ToleranceKind.Radius, Id(s, 4), 1, unit: ToleranceUnit.RadiusMM);
        Assert.Equal("Surface 4 is flat: give its power in fringes.", flat.EngineLimits(ctx).Issue!.Text);
        Assert.Null(new ToleranceRow(ToleranceKind.Radius, Id(s, 4), 1).ConvertedTo(ToleranceUnit.RadiusMM, ctx));
        Assert.True(new ToleranceRow(ToleranceKind.Radius, Id(s, 4), 1).EngineLimits(ctx).Ok);
    }

    [Fact]
    public void TiltAndAbbeRowsConvertUnits()
    {
        var s = DoubleGauss;
        var ctx = Context(s);
        var tilt = new ToleranceRow(ToleranceKind.TiltX, Id(s, 2), 1.5, 0.6);
        Assert.Equal(ToleranceUnit.Arcmin, tilt.Unit);
        Assert.Equal(1.5 / 60, tilt.EngineLimits(ctx).Minus, 15);
        var deg = tilt.ConvertedTo(ToleranceUnit.Degrees, ctx)!;
        Assert.Equal(0.025, deg.Minus, 12);
        Assert.Equal(0.01, deg.Plus, 12);
        Assert.Equal(1.5 / 60, deg.EngineLimits(ctx).Minus, 15);
        Assert.Equal(1.5, deg.ConvertedTo(ToleranceUnit.Arcmin, ctx)!.Minus, 12);
        Assert.Equal("±1′", new ToleranceRow(ToleranceKind.TiltX, Id(s, 2), 1).LimitsText);
        Assert.Equal("+0.6′", tilt.LimitText(true));

        var abbe = new ToleranceRow(ToleranceKind.Abbe, Id(s, 1), 0.5);
        Assert.Equal(ToleranceUnit.Percent, abbe.Unit);
        Assert.Equal(60.3 * 0.5 / 100, abbe.EngineLimits(ctx).Plus, 12);
        var dv = abbe.ConvertedTo(ToleranceUnit.Abbe, ctx)!;
        Assert.Equal(0.3015, dv.Minus, 9);
        Assert.Equal(0.5, dv.ConvertedTo(ToleranceUnit.Percent, ctx)!.Minus, 9);
        // V_d of a catalogue glass is not known without the engine's glass data.
        Assert.Null(abbe.ConvertedTo(ToleranceUnit.Abbe, new ToleranceContext(s)));
        Assert.Contains("not known yet", abbe.EngineLimits(new ToleranceContext(s)).Issue!.Text);
        // A unit the kind does not have changes nothing.
        Assert.Equal(0.5, abbe.ConvertedTo(ToleranceUnit.Mm, ctx)!.Minus);
    }

    // ---- Operands, saved form ----

    [Fact]
    public void OperandsCarryParameterPathLimitsDistributionAndLabel()
    {
        var s = DoubleGauss;
        var ctx = Context(s);
        var row = new ToleranceRow(ToleranceKind.DecenterY, Id(s, 3), 0.02, 0.05, distribution: ToleranceDistribution.Normal(2));
        var o = row.Operand(ctx)!;
        Assert.Equal("decenter_y", o["param"]!["param"]!.GetValue<string>());
        Assert.Equal(3UL, o["param"]!["surface"]!.GetValue<ulong>());
        Assert.Equal(0.02, o["minus"]!.GetValue<double>());
        Assert.Equal(0.05, o["plus"]!.GetValue<double>());
        Assert.Equal("truncated_normal", o["distribution"]!["type"]!.GetValue<string>());
        Assert.Equal(2, o["distribution"]!["sigmas"]!.GetValue<double>());
        Assert.Equal("Surface decenter y S3", o["label"]!.GetValue<string>());

        var cat = new ToleranceRow(ToleranceKind.Index, Id(s, 1), 0.0005).Operand(ctx)!;
        Assert.Equal("material_dn_d", cat["param"]!["param"]!.GetValue<string>());
        var element = new ToleranceRow(ToleranceKind.ElementTiltY, Id(s, 3), 1, last: Id(s, 5)).Operand(ctx)!;
        Assert.Equal("element_placement", element["param"]!["param"]!.GetValue<string>());
        Assert.Equal("tilt_y", element["param"]!["component"]!.GetValue<string>());
        Assert.Equal(5UL, element["param"]!["last"]!.GetValue<ulong>());
        Assert.Equal(1.0 / 60, element["minus"]!.GetValue<double>(), 15);
        var model = s.WithSurface(1, x => x with { Material = new MaterialRef.ModelGlass(1.6, 50, 0) });
        Assert.Equal("model_nd", new ToleranceRow(ToleranceKind.Index, Id(s, 1), 0.001).Path(model).ToJson()["param"]!.GetValue<string>());
        Assert.Equal(50 * 0.01, new ToleranceRow(ToleranceKind.Abbe, Id(s, 1), 1).EngineLimits(new ToleranceContext(model)).Minus, 12);
    }

    [Fact]
    public void RowJsonRoundTripsAndOperandsBecomeRows()
    {
        var s = DoubleGauss;
        var rows = new[]
        {
            new ToleranceRow(ToleranceKind.Radius, Id(s, 1), 3, 2, ToleranceUnit.Fringes, distribution: ToleranceDistribution.Parabolic),
            new ToleranceRow(ToleranceKind.ElementDecenterX, Id(s, 3), 0.03, last: Id(s, 5), pivot: Id(s, 4)),
            new ToleranceRow(ToleranceKind.Other, Id(s, 2), 0.1, param: new ParamPath.Conic(Id(s, 2))),
            new ToleranceRow(ToleranceKind.Abbe, Id(s, 1), 0.1, unit: ToleranceUnit.Abbe),
        };
        foreach (var r in rows) Assert.True(r.SameAs(ToleranceRow.FromJson(r.ToJson())!));
        Assert.Null(ToleranceRow.FromJson(new JsonObject { ["kind"] = "other", ["surface"] = 1, ["minus"] = 1.0 }));
        Assert.Null(ToleranceRow.FromJson(new JsonObject { ["kind"] = "bogus", ["surface"] = 1 }));
        // An unsupported unit falls back to the kind's default.
        Assert.Equal(ToleranceUnit.Arcmin, ToleranceRow.FromJson(new JsonObject { ["kind"] = "tilt_x", ["surface"] = 1, ["unit"] = "fringes" })!.Unit);

        // An operand written elsewhere: a curvature in 1/mm becomes ΔR, a tilt in degrees arc minutes.
        var curv = s.ValueAt(new ParamPath.Curvature(Id(s, 1)))!.Value;
        var fromOp = ToleranceRow.FromOperand(new JsonObject
        {
            ["param"] = new ParamPath.Curvature(Id(s, 1)).ToJson(), ["minus"] = 0.001, ["plus"] = 0.002,
        }, s)!;
        Assert.Equal(ToleranceKind.Radius, fromOp.Kind);
        Assert.Equal(ToleranceUnit.RadiusMM, fromOp.Unit);
        // A larger curvature is a smaller radius: the − limit of ΔR belongs to the + limit of c.
        Assert.Equal(1 / curv - 1 / (curv + 0.002), fromOp.Minus, 12);
        Assert.Equal(1 / (curv - 0.001) - 1 / curv, fromOp.Plus, 12);
        var tilt = ToleranceRow.FromOperand(new JsonObject { ["param"] = new ParamPath.TiltX(Id(s, 2)).ToJson(), ["minus"] = 0.5, ["plus"] = 0.5 }, s)!;
        Assert.Equal((ToleranceKind.TiltX, 30.0), (tilt.Kind, tilt.Minus));
        var other = ToleranceRow.FromOperand(new JsonObject { ["param"] = new ParamPath.Conic(Id(s, 2)).ToJson(), ["minus"] = 1.0, ["plus"] = 1.0 }, s)!;
        Assert.Equal(ToleranceKind.Other, other.Kind);
        Assert.Equal("Conic 2", other.Label(s));
        var fr = ToleranceRow.FromOperand(new JsonObject { ["param"] = new ParamPath.Curvature(Id(s, 1)).ToJson(), ["minus"] = 2.0, ["plus"] = 2.0, ["unit"] = "fringes" }, s)!;
        Assert.Equal(ToleranceUnit.Fringes, fr.Unit);
    }

    [Fact]
    public void SectionRoundTripsSettings()
    {
        var s = DoubleGauss;
        var ctx = Context(s);
        var t = ToleranceSettings.Defaults(s);
        t.Mode = ToleranceMode.MonteCarlo;
        t.Trials = 50;
        t.Seed = 12345;
        t.TestWavelengthUm = 0.5876;
        t.Percentiles = [50, 95, 99.7];
        t.Criterion.Kind = ToleranceCriterionKind.Wavefront;
        t.Criterion.Threshold = 0.07;
        t.Generator.Apply(ToleranceGeneratorPreset.HighPrecision);
        t.Generator.Distribution = ToleranceDistribution.Normal(3);
        t.Rows[0].Distribution = ToleranceDistribution.EndPoint;
        t.Rows.Add(new ToleranceRow(ToleranceKind.Other, Id(s, 2), 0.01, param: new ParamPath.Conic(Id(s, 2))));
        t.Compensators.Add(new ToleranceCompensator(new ParamPath.Thickness(Id(s, 6))));

        var section = t.Section(ctx);
        Assert.Equal(1, section["schema_version"]!.GetValue<int>());
        Assert.Equal(109, section["operands"]!.AsArray().Count);
        Assert.Equal(50, section["monte_carlo"]!["trials"]!.GetValue<int>());
        Assert.Equal(0.07 * 0.07, section["monte_carlo"]!["threshold"]!.GetValue<double>(), 15);
        Assert.Equal([50.0, 95.0, 99.7], section["monte_carlo"]!["percentiles"]!.AsArray().Select(n => n!.GetValue<double>()));
        Assert.Equal(50, section["monte_carlo"]!["record_trials"]!.GetValue<int>());
        Assert.Equal("high_precision", section["app"]!["generator"]!["preset"]!.GetValue<string>());
        Assert.Equal(0.5876, section["app"]!["test_wavelength_um"]!.GetValue<double>());
        Assert.Equal(109, section["app"]!["rows"]!.AsArray().Count);
        Assert.Null(section["criterion"]);

        var back = ToleranceSettings.FromSection(JsonNode.Parse(section.ToJsonString()), s);
        Assert.True(back.SameAs(t));
        Assert.Equal(109, back.Rows.Count);
        Assert.Equal(12345, back.Seed);
        Assert.Equal(ToleranceMode.MonteCarlo, back.Mode);
        Assert.Equal(ToleranceCriterionKind.Wavefront, back.Criterion.Kind);
        Assert.Equal(0.07, back.Criterion.Threshold);
        Assert.Equal(0.5876, back.TestWavelengthUm);
        Assert.Equal([50.0, 95.0, 99.7], back.Percentiles);
        // Levels outside [0, 100] and repeats are dropped when read; with none left the default stays.
        var messy = JsonNode.Parse(section.ToJsonString())!;
        messy["monte_carlo"]!["percentiles"] = new JsonArray(97.7, 50, 50, 120, 90);
        Assert.Equal([50.0, 90.0, 97.7], ToleranceSettings.FromSection(messy, s).Percentiles);
        messy["monte_carlo"]!["percentiles"] = new JsonArray(120, -1);
        Assert.Equal(ToleranceSettings.DefaultPercentiles, ToleranceSettings.FromSection(messy, s).Percentiles);
        Assert.Equal(2, back.Compensators.Count);
        Assert.Null(back.Compensators[1].Range);
        // Edits show in the comparison; the rows' session ids do not.
        var copy = t.Clone();
        foreach (var r in copy.Rows) r.Id = Guid.NewGuid();
        Assert.True(copy.SameAs(t));
        copy.Rows[3].Minus *= 2;
        Assert.False(copy.SameAs(t));
        // A spec written elsewhere (no `app`): its operands become rows.
        var spec = new JsonObject { ["operands"] = new JsonArray(t.Rows.Take(3).Select(r => (JsonNode)r.Operand(ctx)!).ToArray()), ["compensators"] = new JsonArray() };
        Assert.Equal(3, ToleranceSettings.FromSection(spec, s).Rows.Count);
        Assert.True(ToleranceSettings.FromSection(null, s).IsEmpty);
    }

    // ---- Job settings and issues ----

    [Fact]
    public void JobSettingsShape()
    {
        var s = DoubleGauss;
        var ctx = Context(s);
        var t = ToleranceSettings.Defaults(s);
        t.Trials = 20;
        t.Seed = 3;
        t.RecordTrials = 500;
        var (settings, ids) = t.JobSettings(ctx, new MeritSettings());
        Assert.Equal(108, ids.Count);
        Assert.Equal(t.Rows.Select(r => r.Id), ids);
        var ops = settings["tolerances"]!.AsArray();
        Assert.Equal(108, ops.Count);
        Assert.Equal("curvature", ops[0]!["param"]!["param"]!.GetValue<string>());
        Assert.Equal("Radius S1", ops[0]!["label"]!.GetValue<string>());
        Assert.Equal("uniform", ops[0]!["distribution"]!["type"]!.GetValue<string>());
        var comp = Assert.Single(settings["compensators"]!.AsArray());
        Assert.Equal("thickness", comp!["param"]!["param"]!.GetValue<string>());
        Assert.Equal(0.5, comp["range"]!.GetValue<double>());
        Assert.Equal("both", settings["mode"]!.GetValue<string>());
        Assert.Equal(20, settings["trials"]!.GetValue<int>());
        Assert.Equal(3L, settings["seed"]!.GetValue<long>());
        Assert.Equal(20, settings["record_trials"]!.GetValue<int>()); // at most every trial
        Assert.Equal([50.0, 90.0, 97.7], settings["percentiles"]!.AsArray().Select(n => n!.GetValue<double>()));
        Assert.Equal(0.6328, settings["irregularity_wavelength_um"]!.GetValue<double>());
        var dm = settings["default_merit"]!;
        Assert.Equal("rms_spot", dm["criterion"]!["type"]!.GetValue<string>());
        Assert.Null(dm["thickness"]);
        Assert.False(settings.ContainsKey("threshold"));
        Assert.False(settings.ContainsKey("criterion"));

        // Sensitivity runs record nothing; a yield threshold goes in as a merit (µm → mm²).
        t.Mode = ToleranceMode.Sensitivity;
        t.Criterion.Threshold = 20;
        var sens = t.JobSettings(ctx, new MeritSettings()).Settings;
        Assert.Equal(0, sens["record_trials"]!.GetValue<int>());
        Assert.Equal(4e-4, sens["threshold"]!.GetValue<double>(), 15);

        // MTF judges the shortfall from 1; the optimiser's merit takes the Optimize page's settings.
        t.Criterion.Kind = ToleranceCriterionKind.Mtf;
        var mtf = t.JobSettings(ctx, new MeritSettings()).Settings["default_merit"]!["criterion"]!;
        Assert.Equal("mtf", mtf["type"]!.GetValue<string>());
        Assert.Equal(1, mtf["target"]!.GetValue<double>());
        Assert.Equal(20, mtf["frequencies"]![0]!.GetValue<double>());
        t.Criterion.Kind = ToleranceCriterionKind.DesignMerit;
        t.Criterion.Threshold = null;
        var merit = new MeritSettings { Defaults = new DefaultMeritSettings { Criterion = MeritCriterion.Wavefront } };
        var dmerit = t.JobSettings(ctx, merit).Settings["default_merit"]!;
        Assert.Equal("wavefront", dmerit["criterion"]!["type"]!.GetValue<string>());
        Assert.Null(dmerit["thickness"]);
        var custom = new MeritSettings { Mode = MeritMode.Custom };
        custom.Add(OperandCatalog.Descriptor("efl")!);
        var cj = t.JobSettings(ctx, custom).Settings;
        Assert.Equal(1, cj["criterion"]!["schema_version"]!.GetValue<int>());
        Assert.False(cj.ContainsKey("default_merit"));
        Assert.Equal(CriterionDisplay.Merit, t.Criterion.Display(custom));
        Assert.Equal(CriterionDisplay.Wavefront, t.Criterion.Display(merit));
        Assert.Equal(CriterionDisplay.Spot, t.Criterion.Display(new MeritSettings()));
        // An MTF merit reads as an MTF only with the shortfall from 1 as its target.
        Assert.Equal(CriterionDisplay.Merit, t.Criterion.Display(new MeritSettings { Defaults = new DefaultMeritSettings { Criterion = MeritCriterion.Mtf } }));
        Assert.Equal(CriterionDisplay.Mtf, t.Criterion.Display(new MeritSettings { Defaults = new DefaultMeritSettings { Criterion = MeritCriterion.Mtf, MtfTarget = 1 } }));
    }

    [Fact]
    public void IssuesNameTheRowsThatCannotRun()
    {
        var s = DoubleGauss;
        var ctx = Context(s);
        var t = new ToleranceSettings();
        var good = new ToleranceRow(ToleranceKind.Thickness, Id(s, 1), 0.05);
        var zero = new ToleranceRow(ToleranceKind.Thickness, Id(s, 2), 0, 0);
        var negative = new ToleranceRow(ToleranceKind.DecenterX, Id(s, 2), -1);
        var image = new ToleranceRow(ToleranceKind.Thickness, Id(s, 12), 0.1);
        var noAperture = new ToleranceRow(ToleranceKind.Radius, Id(s, 1), 3);
        var tooWide = new ToleranceRow(ToleranceKind.Radius, Id(s, 5), 3);
        var air = new ToleranceRow(ToleranceKind.Index, Id(s, 2), 0.001);
        var gone = new ToleranceRow(ToleranceKind.Thickness, new SurfaceID(999), 0.1);
        var badParam = new ToleranceRow(ToleranceKind.Other, Id(s, 2), 1, param: new ParamPath.Asphere(Id(s, 2), 0));
        var elementA = new ToleranceRow(ToleranceKind.ElementTiltX, Id(s, 3), 1, last: Id(s, 5));
        var elementB = new ToleranceRow(ToleranceKind.ElementTiltY, Id(s, 3), 1, last: Id(s, 4));
        var badRange = new ToleranceRow(ToleranceKind.ElementDecenterX, Id(s, 5), 1, last: Id(s, 3));
        t.Rows = [good, zero, negative, image, noAperture, tooWide, air, gone, badParam, elementA, elementB, badRange];

        // The clear apertures are not known for S1; S5 (c·h = 0.87 at 10 mm) is wider than its sphere at 12 mm.
        var semi = s.Surfaces.Skip(2).ToDictionary(x => x.Id, _ => 10.0);
        semi[Id(s, 5)] = 12;
        var issues = t.Issues(new ToleranceContext(s) { SemiDiameters = semi });
        Assert.DoesNotContain(good.Id, issues.Keys);
        Assert.Equal("Both limits are zero.", issues[zero.Id].Text);
        Assert.Equal("Limits must be zero or positive numbers.", issues[negative.Id].Text);
        Assert.Equal("The image surface has no thickness.", issues[image.Id].Text);
        Assert.StartsWith("Fringes need the clear aperture of surface 1", issues[noAperture.Id].Text);
        Assert.Equal("The clear aperture of surface 5 is wider than its sphere.", issues[tooWide.Id].Text);
        Assert.Equal("Surface 2 is followed by air, not a glass with n_d and V_d.", issues[air.Id].Text);
        Assert.Equal("The surface no longer exists.", issues[gone.Id].Text);
        Assert.Equal("The parameter no longer exists.", issues[badParam.Id].Text);
        Assert.Equal("An element moves surfaces between the object and the image, first to last.", issues[badRange.Id].Text);
        Assert.StartsWith("Element tolerances starting at S3 move one group", issues[elementA.Id].Text);
        Assert.Equal(issues[elementA.Id], issues[elementB.Id]);
        Assert.Equal(11, issues.Count);

        // Rows with an issue stay out of the job.
        var (settings, ids) = t.JobSettings(new ToleranceContext(s) { SemiDiameters = semi }, new MeritSettings());
        Assert.Equal([good.Id], ids);
        Assert.Single(settings["tolerances"]!.AsArray());
    }

    [Fact]
    public void CatalogueOffsetsStayWithinTheEngineBounds()
    {
        var s = DoubleGauss;
        var edited = s.WithSurface(1, x => x with { Material = new MaterialRef.CatalogGlass { Name = "N-SK16", Offset = new MaterialOffset(0.049, 0) } });
        var ctx = new ToleranceContext(edited);
        var big = new ToleranceRow(ToleranceKind.Index, Id(s, 1), 0.002).EngineLimits(ctx);
        Assert.False(big.Ok);
        Assert.StartsWith("With the melt offset (0.049), Δn_d would pass ±0.05", big.Issue!.Text);
        Assert.True(new ToleranceRow(ToleranceKind.Index, Id(s, 1), 0.0005).EngineLimits(ctx).Ok);
    }

    // ---- Editing ----

    [Fact]
    public void EditingKeepsElementGroupsAndIds()
    {
        var s = DoubleGauss;
        var t = new ToleranceSettings();
        var a = new ToleranceRow(ToleranceKind.ElementTiltX, Id(s, 3), 1, last: Id(s, 5), pivot: Id(s, 4));
        var b = new ToleranceRow(ToleranceKind.Thickness, Id(s, 1), 0.1);
        t.Add([a, b]);
        // A new element row on the same first surface joins the existing range and pivot.
        var c = new ToleranceRow(ToleranceKind.ElementDecenterX, Id(s, 3), 0.03);
        t.Add([c], after: a.Id);
        Assert.Equal(new[] { a.Id, c.Id, b.Id }, t.Rows.Select(r => r.Id));
        Assert.Equal((Id(s, 5), Id(s, 4)), (c.Last!.Value, c.Pivot!.Value));
        // Moving one moves the group; a pivot on the first surface is stored as none.
        t.SetElement(c.Id, Id(s, 3), Id(s, 4), pivot: Id(s, 3));
        Assert.Equal(Id(s, 4), a.Last);
        Assert.Null(a.Pivot);
        Assert.Equal(a.Last, c.Last);
        t.SetElement(b.Id, Id(s, 1), Id(s, 2), null); // not an element row: unchanged
        Assert.Null(b.Last);

        var ids = t.Duplicate(new HashSet<Guid> { a.Id, b.Id });
        Assert.Equal(2, ids.Count);
        Assert.Equal(5, t.Rows.Count);
        Assert.Equal(t.Rows[0].Kind, t.Rows[1].Kind);
        Assert.NotEqual(t.Rows[0].Id, t.Rows[1].Id);
        t.Delete(new HashSet<Guid>(ids));
        Assert.Equal(3, t.Rows.Count);

        t.Generate(s);
        Assert.Equal(108, t.Rows.Count);

        // A changed row keeps its id (same kind and surface), unchanged rows theirs, duplicates their own.
        var old = new ToleranceSettings { Rows = [b.Clone(), b.Clone(), a.Clone()], Compensators = [ToleranceCompensator.BackFocus(s)!] };
        old.Rows[1].Id = Guid.NewGuid();
        var next = old.Clone();
        foreach (var r in next.Rows) r.Id = Guid.NewGuid();
        foreach (var r in next.Compensators) r.Id = Guid.NewGuid();
        next.Rows[1].Minus = 0.3;
        var kept = ToleranceSettings.KeepingIds(next, old);
        Assert.Equal(old.Rows.Select(r => r.Id), kept.Rows.Select(r => r.Id));
        Assert.Equal(old.Compensators[0].Id, kept.Compensators[0].Id);
        Assert.Equal(0.3, kept.Rows[1].Minus);
        Assert.NotEqual(old.Rows[0].Id, old.Rows[1].Id);
    }

    // ---- Result ----

    static JsonNode ResultJson() => JsonNode.Parse("""
        {
          "status": "stopped",
          "compensators": [ { "param": { "param": "thickness", "surface": 11 }, "range": 0.5 } ],
          "sensitivity": {
            "nominal": { "criterion": 1e-4, "compensators": [ 28.9 ] },
            "operands": [
              { "index": 0, "label": "Radius S1", "param": { "param": "curvature", "surface": 1 }, "minus": 0.001, "plus": 0.001,
                "criterion_minus": 1.2e-4, "criterion_plus": 1.1e-4, "change_minus": 2e-5, "change_plus": 1e-5, "worst": 2e-5,
                "expected_change": 9e-6, "ok": true, "compensators_minus": [ 28.95 ], "compensators_plus": [ 28.85 ] },
              { "index": 1, "label": "", "param": { "param": "thickness", "surface": 2 }, "minus": 0.05, "plus": 0.05,
                "criterion_minus": 1.5e-4, "criterion_plus": 1.4e-4, "change_minus": 5e-5, "change_plus": 4e-5, "worst": 5e-5,
                "expected_change": 2.1e-5, "ok": true, "compensators_minus": [ 29.3 ], "compensators_plus": [ 28.4 ] },
              { "index": 2, "label": "Tilt x S3", "minus": 0.01, "plus": 0.01, "criterion_minus": null, "criterion_plus": "inf",
                "change_minus": null, "change_plus": 0, "worst": null, "ok": false }
            ],
            "ranking": [ 1, 0, 2 ], "rss_change": 5.4e-5, "predicted_criterion": 1.3e-4, "cancelled": false
          },
          "monte_carlo": {
            "trials": 4, "failures": 1, "seed": 7, "mean": 1.4e-4, "std_dev": 2e-5, "min": 1.1e-4, "max": 1.9e-4,
            "percentiles": [ { "percentile": 50, "value": 1.3e-4 }, { "percentile": 97.7, "value": 1.9e-4 }, { "percentile": "bad" } ],
            "criteria": [ 1.1e-4, null, 1.9e-4, 1.3e-4 ],
            "records": [ { "trial": 0, "criterion": 1.1e-4, "ok": true, "compensators": [ 28.92 ], "perturbations": [ 0.0005, -0.02, 0 ] },
                         { "trial": 2, "criterion": 1.9e-4, "ok": true, "compensators": [ 29.1 ], "perturbations": [ -0.001, 0.04, 0.005 ] } ],
            "seconds_per_trial": 0.25, "yield_fraction": 0.75, "threshold": 2e-4,
            "compensator_mean": [ 28.95 ], "compensator_std": [ 0.1 ]
          }
        }
        """)!;

    [Fact]
    public void ResultParsesFromJson()
    {
        var s = DoubleGauss;
        var r = new ToleranceResult(ResultJson(), s, CriterionDisplay.Spot);
        Assert.Equal("stopped", r.Status);
        Assert.True(r.Stopped);
        Assert.True(r.HasSensitivity && r.HasMonteCarlo && r.HasNominal);
        var comp = Assert.Single(r.Compensators);
        Assert.Equal(new ParamPath.Thickness(Id(s, 11)), comp.Param);
        Assert.Equal(0.5, comp.Range);
        Assert.Equal(1e-4, r.Nominal);
        Assert.Equal([28.9], r.NominalCompensators);
        // The engine's ranking, the label from the parameter when the engine gave none.
        Assert.Equal([1, 0, 2], r.Sensitivities.Select(x => x.Id));
        Assert.Equal("Thickness 2", r.Sensitivities[0].Label);
        Assert.Equal(new ParamPath.Thickness(Id(s, 2)), r.Sensitivities[0].Param);
        Assert.True(double.IsNaN(r.Sensitivities[2].CriterionMinus));
        Assert.True(double.IsPositiveInfinity(r.Sensitivities[2].CriterionPlus));
        Assert.True(double.IsNaN(r.Sensitivities[2].ExpectedChange));
        Assert.False(r.Sensitivities[2].Ok);
        Assert.True(r.Sensitivities[0].WorsePlus is false);
        Assert.Equal(1.3e-4, r.Predicted);
        Assert.Equal(5.4e-5, r.RssChange);
        Assert.Equal((4, 1, 7), (r.Trials, r.Failures, (int)r.Seed));
        Assert.Equal([new ToleranceResult.Percentile(50, 1.3e-4), new ToleranceResult.Percentile(97.7, 1.9e-4)], r.Percentiles);
        Assert.Equal(4, r.TrialCriteria.Count);
        Assert.Equal(3, r.Criteria.Count);
        Assert.Equal(0.25, r.SecondsPerTrial);
        Assert.Equal(0.75, r.Yield);
        Assert.Equal(2e-4, r.Threshold);
        Assert.Equal([28.95], r.CompensatorMean);
        Assert.Equal(2, r.Records.Count);
        Assert.Equal(0.04, r.Record(2)!.Perturbations[1]);
        Assert.Null(r.Record(1));

        // Shares of the predicted change (expected changes, 0 for the missing one), the worst offenders, the distribution.
        var share = r.Sensitivities.ToDictionary(x => x.Id, r.Share);
        Assert.Equal(2.1e-5 / 3e-5, share[1], 12);
        Assert.Equal(9e-6 / 3e-5, share[0], 12);
        Assert.Equal(0, share[2]);
        Assert.Equal<int[]>([1], [.. r.WorstOffenders]);
        Assert.Equal(2.0 / 3, r.Cumulative(1.3e-4), 12);
        Assert.Equal<int[]>([2, 3, 0], [.. r.TrialsWorstFirst]);
        Assert.Equal<int[]>([3, 0], [.. r.TrialsShownIn(10, 12)]); // 10 … 12 µm
        var range = r.CompensatorRange(0)!.Value;
        Assert.Equal(-0.5, range.Low, 9); // 28.4 − 28.9 is the lowest, 29.3 − 28.9 the highest
        Assert.Equal(0.4, range.High, 9);
        Assert.Null(r.CompensatorRange(1));

        // Without expected changes the shares are those of the squared worst changes.
        var old = ResultJson();
        foreach (var o in old["sensitivity"]!["operands"]!.AsArray()) o!.AsObject().Remove("expected_change");
        var r2 = new ToleranceResult(old, null);
        Assert.Equal(25.0 / 29, r2.Share(r2.Sensitivities[0]), 12);
        Assert.Equal("Radius S1", r2.Sensitivities[1].Label);
        Assert.Equal("", new ToleranceResult(new JsonObject(), null).Status);
        Assert.False(new ToleranceResult(new JsonObject(), null).HasSensitivity);
    }

    [Fact]
    public void CriterionValuesReadAsTheDesignerSeesThem()
    {
        Assert.Equal(10.0, CriterionDisplay.Spot.Value(1e-4), 12);
        Assert.Equal(1e-4, CriterionDisplay.Spot.Criterion(10), 15);
        Assert.Equal(0.75, CriterionDisplay.Mtf.Value(0.0625), 12);
        Assert.Equal(0.0625, CriterionDisplay.Mtf.Criterion(0.75), 12);
        Assert.True(double.IsNaN(CriterionDisplay.Wavefront.Value(double.NaN)));
        Assert.Equal("3.16 µm", CriterionDisplay.Spot.FormatWithUnit(3.1623));
        Assert.Equal("12.6", CriterionDisplay.Spot.Format(12.6));
        Assert.Equal("0.075 waves", CriterionDisplay.Wavefront.FormatWithUnit(0.0752));
        Assert.Equal("—", CriterionDisplay.Mtf.Format(double.NaN));
        Assert.Equal("+0.012", CriterionDisplay.Mtf.FormatChange(0.012));
        Assert.Equal("−0.012", CriterionDisplay.Mtf.FormatChange(-0.012));
        Assert.Equal(" 0.000", CriterionDisplay.Mtf.FormatChange(-0.0001));
        Assert.Equal("0.000", CriterionDisplay.Mtf.Format(-0.0001));
        Assert.True(CriterionDisplay.Mtf.HigherIsBetter());
        Assert.Equal("Normal (limits at ±2 σ)", ToleranceDistribution.Normal(2).Title);
    }

    [Fact]
    public void ReportsSummariseTheResult()
    {
        var s = DoubleGauss;
        var rows = new List<ToleranceRow>
        {
            new(ToleranceKind.Radius, Id(s, 1), 3),
            new(ToleranceKind.Thickness, Id(s, 2), 0.05),
            new(ToleranceKind.TiltX, Id(s, 3), 1),
        };
        var r = new ToleranceResult(ResultJson(), s, CriterionDisplay.Spot);
        var text = ToleranceReport.Summary(r, rows, s, "Double Gauss");
        var lines = text.Split('\n');
        Assert.Equal("Tolerance analysis: Double Gauss", lines[0]);
        Assert.Equal("Criterion: RMS spot (µm)", lines[1]);
        Assert.Contains("Stopped early: statistics of the completed evaluations.", lines);
        Assert.Contains("Nominal: 10.0", lines);
        Assert.Contains("Predicted mean (nominal + expected changes): 11.4", lines);
        Assert.Contains("Monte Carlo: 4 trials, seed 7, 1 failed", lines);
        Assert.Contains("  0.25 s per trial", lines);
        Assert.Contains("  50 % of builds at most: 11.4", lines);
        Assert.Contains("  Yield: 75.0 %", lines);
        Assert.Contains("  Thickness S2–3 ±0.05 mm: 70 % of the predicted change", lines);
        Assert.Contains("Compensator Back focus: moved −0.5 to +0.4 mm of ±0.5; Monte Carlo σ 0.1", lines);

        var csv = ToleranceReport.SensitivityCsv(r, rows, s).Split('\n');
        Assert.Equal("rank,tolerance,minus,plus,unit,parameter,criterion_minus,criterion_plus,rms_spot_um_minus,rms_spot_um_plus,change_minus,change_plus,share", csv[0]);
        Assert.StartsWith("1,Thickness S2–3,0.05,0.05,mm,Thickness 2,0.00015,0.00014,12.24744871,11.83215957,", csv[1]);
        Assert.Equal("", csv[^1]);
        Assert.Equal(5, csv.Length);

        var trials = ToleranceReport.TrialsCsv(r, rows, s).Split('\n');
        Assert.Equal("trial,criterion,rms_spot_um,compensator_1,Radius S1,Thickness S2–3,Surface tilt x S3", trials[0]);
        Assert.Equal("1,0.00011,10.48808848,28.92,0.0005,-0.02,0", trials[1]);
        Assert.Equal("2,,,,,,", trials[2]);
        Assert.Equal("3,0.00019,13.78404875,29.1,-0.001,0.04,0.005", trials[3]);
        Assert.Equal("4,0.00013,11.40175425,,,,", trials[4]);
        Assert.Equal("trial,criterion,rms_spot_um", ToleranceReport.TrialsCsv(new ToleranceResult(new JsonObject { ["monte_carlo"] = new JsonObject { ["trials"] = 1, ["criteria"] = new JsonArray(JsonValue.Create(1e-4)) } }, null, CriterionDisplay.Spot)).Split('\n')[0]);
    }

    // ---- Preview and timing ----

    [Fact]
    public void PreviewCasesAndBuildChanges()
    {
        var s = DoubleGauss;
        Assert.Equal("""{"type":"trial","trial":16}""", new TolerancePreviewCase.Trial(16).ToJson().ToJsonString());
        Assert.Equal("""{"type":"sensitivity","operand":2,"positive":true}""", new TolerancePreviewCase.Sensitivity(2, true).ToJson().ToJsonString());

        var rows = new List<ToleranceRow>
        {
            new(ToleranceKind.TiltX, Id(s, 2), 1, 2),
            new(ToleranceKind.Thickness, Id(s, 2), 0.05),
            new(ToleranceKind.Radius, Id(s, 1), 3),
        };
        var ctx = Context(s);
        var (settings, ids) = new ToleranceSettings { Rows = rows }.JobSettings(ctx, new MeritSettings());
        var limits = ToleranceBuildChange.RunLimits(settings);
        Assert.Equal(3, ids.Count);
        Assert.Equal(1.0 / 60, limits[0].Minus, 15);
        var k = limits[2].Plus / 3;
        var changes = ToleranceBuildChange.Changes([0.5 * 2.0 / 60, -0.04, 0, 0.0], limits, rows, s);
        Assert.Equal(2, changes.Count);
        Assert.Equal(1, changes[0].Id); // −0.04 of 0.05 is the larger share
        Assert.Equal(-0.8, changes[0].Fraction, 12);
        Assert.Equal(("−0.04 mm", "±0.05 mm", "Thickness S2–3"), (changes[0].Text, changes[0].Limits, changes[0].Label));
        Assert.Equal(0, changes[1].Id);
        Assert.Equal(0.5, changes[1].Fraction, 12);
        Assert.Equal(("+1′", "−1 / +2′"), (changes[1].Text, changes[1].Limits));
        var radius = ToleranceBuildChange.Changes([0, 0, -2 * k], limits, rows, null);
        Assert.Equal("−2 fr", radius[0].Text);
        Assert.Equal("Radius", radius[0].Label);

        var result = new ToleranceResult(ResultJson(), s);
        Assert.Equal("Tolerance: trial 17, compensated", new TolerancePreviewCase.Trial(16).Title(result, s, rows));
        Assert.Equal("Tolerance: Surface tilt x S2 at +2′, compensated", new TolerancePreviewCase.Sensitivity(0, true).Title(result, s, rows));
        Assert.Equal("Tolerance: Radius S1 at −3 fr", new TolerancePreviewCase.Sensitivity(0, false).Title(null, s, [.. rows.Skip(2)]));
        Assert.Equal("Tolerance: operand 6", new TolerancePreviewCase.Sensitivity(5, false).Title(null, null, rows));

        var preview = TolerancePreview.FromJson(new JsonObject
        {
            ["system"] = s.ToJson(), ["criterion"] = 1e-4, ["compensators"] = new JsonArray(JsonValue.Create(28.9)),
            ["perturbations"] = new JsonArray(JsonValue.Create(0.1), JsonValue.Create(0.0)),
        }, new TolerancePreviewCase.Trial(0))!;
        Assert.Equal(s, preview.System);
        Assert.Equal([28.9], preview.Compensators);
        Assert.Equal([0.1, 0.0], preview.Perturbations);
        Assert.Null(TolerancePreview.FromJson(new JsonObject(), new TolerancePreviewCase.Trial(0)));
        Assert.Null(TolerancePreview.FromJson(null, new TolerancePreviewCase.Trial(0)));
    }

    [Fact]
    public void TimingEstimatesTheRestOfTheRun()
    {
        // Both: the sensitivity takes a fifth. At 10 % (half of it) after 10 s with 4 operands (9 evaluations).
        var t = ToleranceTiming.Estimate(10, null, 0.1, ToleranceMode.Both, 100, 4, null);
        Assert.Null(t.TrialsDone);
        Assert.Equal(10 / (0.5 * 9), t.SecondsPerTrial!.Value, 12);
        Assert.Equal(10 + 100 * t.SecondsPerTrial!.Value, t.SecondsLeft!.Value, 12);
        // The last run's time per trial is better than the sensitivity's.
        Assert.Equal(10 + 100 * 0.3, ToleranceTiming.Estimate(10, null, 0.1, ToleranceMode.Both, 100, 4, 0.3).SecondsLeft!.Value, 12);
        // Too early to tell.
        Assert.Equal(default, ToleranceTiming.Estimate(0.2, null, 0.005, ToleranceMode.Both, 100, 4, null));
        var early = ToleranceTiming.Estimate(0.2, null, 0.005, ToleranceMode.Both, 100, 4, 0.5);
        Assert.Equal(0.5, early.SecondsPerTrial!.Value);
        Assert.Equal(9 * 0.5 + 100 * 0.5, early.SecondsLeft!.Value, 12);
        // Sensitivity only.
        var sens = ToleranceTiming.Estimate(4, null, 0.5, ToleranceMode.Sensitivity, 0, 4, null);
        Assert.Equal(4.0, sens.SecondsLeft!.Value, 12);
        Assert.Null(sens.SecondsPerTrial);
        // Monte Carlo: trials done by the progress, the rate by the measured time.
        var mc = ToleranceTiming.Estimate(30, 20, 0.5, ToleranceMode.MonteCarlo, 100, 4, 9);
        Assert.Equal(50, mc.TrialsDone);
        Assert.Equal(0.4, mc.SecondsPerTrial!.Value, 12);
        Assert.Equal(20.0, mc.SecondsLeft!.Value, 12);
        // In both: 60 % overall is half of the Monte Carlo; with fewer than 2 trials done the last run's rate is used.
        var both = ToleranceTiming.Estimate(30, 5, 0.6, ToleranceMode.Both, 100, 4, 0.1);
        Assert.Equal(50, both.TrialsDone);
        Assert.Equal(0.1, both.SecondsPerTrial!.Value, 12);
        Assert.Equal(5.0, both.SecondsLeft!.Value, 12);
        var barely = ToleranceTiming.Estimate(30, 5, 0.2, ToleranceMode.Both, 100, 4, 0.1);
        Assert.Equal((0, 0.1), (barely.TrialsDone!.Value, barely.SecondsPerTrial!.Value));
        Assert.Equal(0.2, ToleranceTiming.SensitivityShare(ToleranceMode.Both));
    }

}

/// <summary>The job settings against the real engine (skipped without <c>mklight_ffi.dll</c>).</summary>
[Collection("FFI engine")]
public class ToleranceEngineTests
{
    [Fact]
    public async Task EngineRunsTheJobSettingsOfTheDoubleGauss()
    {
        EngineTestSupport.RequireEngine();
        var engine = EngineTestSupport.SharedEngine();
        var s = SampleSystems.Get("double_gauss")!;
        var t = ToleranceSettings.Defaults(s);
        // A sample of every kind keeps the run short.
        t.Rows = [.. t.Rows.Where(r => r.Surface.Value is 1 or 3).Where(r => r.Kind != ToleranceKind.Abbe || r.Surface.Value == 1)];
        t.Mode = ToleranceMode.Both;
        t.Trials = 3;
        t.RecordTrials = 3;
        t.Seed = 11;
        t.Criterion.Threshold = 20;
        var semi = s.Surfaces.ToDictionary(x => x.Id, _ => 10.0);
        var abbe = s.Surfaces.Where(x => x.Material is MaterialRef.CatalogGlass).ToDictionary(x => x.Id, _ => 50.0);
        var (settings, ids) = t.JobSettings(new ToleranceContext(s) { SemiDiameters = semi, AbbeNumbers = abbe }, new MeritSettings());
        Assert.Equal(t.Rows.Count, ids.Count);
        var snapshot = await engine.OpenSnapshotAsync(ModelJson.Encode(s));
        using var output = await engine.Submit(new EngineAnalysisRequest("tolerance", settings), snapshot).ResultAsync();
        var result = new ToleranceResult(output.Json, s, t.Criterion.Display(new MeritSettings()));
        Assert.Equal(CriterionDisplay.Spot, result.Display);
        Assert.True(result.HasSensitivity);
        Assert.True(result.HasNominal);
        Assert.Equal(ids.Count, result.Sensitivities.Count);
        Assert.Equal(Enumerable.Range(0, ids.Count).Order(), result.Sensitivities.Select(x => x.Id).Order());
        Assert.Equal(3, result.Trials);
        Assert.Equal(3, result.TrialCriteria.Count);
        Assert.Single(result.Compensators);
        Assert.Equal(3, result.Records.Count);
        Assert.Equal(ids.Count, result.Records[0].Perturbations.Count);
        Assert.NotNull(result.Yield);
        Assert.Equal(20, CriterionDisplay.Spot.Value(result.Threshold!.Value), 9);
        Assert.Equal(t.Percentiles.Count, result.Percentiles.Count);
        Assert.True(result.Nominal > 0 && result.Nominal < 1, "the nominal spot of the Double Gauss is well below a millimetre");
        Assert.All(result.Sensitivities, x => Assert.False(string.IsNullOrEmpty(x.Label)));
        // The report text of the real result.
        Assert.Contains("Nominal:", ToleranceReport.Summary(result, [.. ids.Select(id => t.Rows.First(r => r.Id == id))], s, "Double Gauss"));
    }
}
