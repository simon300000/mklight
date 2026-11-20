using System.Text.Json.Nodes;
using MKLight.App.ViewModels;
using MKLight.Model;

namespace MKLight.Tests.Inspector;

public class InspectorVariablesTests
{
    static readonly OpticalSystem Singlet = SampleSystems.Singlet();
    static SurfaceID Id(int i) => Singlet.Surfaces[i].Id;

    [Fact]
    public void MakingAVariableAddsItWithDefaultBoundsAndRemovingTakesItOut()
    {
        var path = new ParamPath.Thickness(Id(1));
        var on = InspectorVariables.Set(JsonTree.Null, Singlet, path, true);
        Assert.NotNull(on);
        Assert.True(InspectorVariables.IsVariable(on!.Value, Singlet, path));
        var entry = (JsonObject)((JsonArray)on.Value.Node!)[0]!;
        Assert.Equal(path, ParamPath.FromJson(entry["param"]));
        Assert.Equal(1.0, entry["lower"]!.GetValue<double>()); // glass keeps a minimum centre thickness
        Assert.False(entry.ContainsKey("upper"));
        Assert.Null(InspectorVariables.Set(on.Value, Singlet, path, true)); // already one
        var off = InspectorVariables.Set(on.Value, Singlet, path, false);
        Assert.Empty((JsonArray)off!.Value.Node!);
    }

    [Fact]
    public void OtherEntriesStayAsTheyAre()
    {
        var other = new JsonArray(new JsonObject { ["param"] = new ParamPath.Curvature(Id(1)).ToJson(), ["lower"] = -0.1, ["upper"] = 0.1 }, new JsonObject { ["unknown"] = 1 });
        var on = InspectorVariables.Set(new JsonTree(other), Singlet, new ParamPath.Conic(Id(1)), true)!.Value;
        var list = (JsonArray)on.Node!;
        Assert.Equal(3, list.Count);
        Assert.True(JsonNode.DeepEquals(other[0], list[0]));
        Assert.True(JsonNode.DeepEquals(other[1], list[1]));
    }

    [Fact]
    public void ASolvedParameterCannotBecomeOne()
    {
        var path = new ParamPath.Thickness(Id(1));
        var solved = Singlet with { Solves = [new Solve(path, new SolveKind.MarginalRayHeight(0))] };
        Assert.NotNull(InspectorVariables.Refusal(solved, path));
        Assert.Null(InspectorVariables.Set(JsonTree.Null, solved, path, true));
    }

    [Fact]
    public void ToroidCurvatureYIsTheCurvatureVariable()
    {
        var toroid = Singlet.SetShapeKind(ShapeKind.Toroid, 1);
        var on = InspectorVariables.Set(JsonTree.Null, toroid, new ParamPath.Curvature(Id(1)), true)!.Value;
        Assert.True(InspectorVariables.IsVariable(on, toroid, new ParamPath.ShapeParam(Id(1), ShapeParameter.CurvatureY)));
    }

    [Fact]
    public void DefaultBoundsFollowTheParameter()
    {
        var last = Singlet.ImageIndex - 1;
        Assert.Equal((0.0, (double?)null), InspectorVariables.DefaultBounds(new ParamPath.Thickness(Id(last)), Singlet));
        Assert.Equal((null, null), InspectorVariables.DefaultBounds(new ParamPath.Thickness(Id(0)), Singlet));
        Assert.Equal((1.4, 2.1), InspectorVariables.DefaultBounds(new ParamPath.ModelNd(Id(1)), Singlet));
        Assert.Equal((-MaterialOffset.MaxDnD, MaterialOffset.MaxDnD), InspectorVariables.DefaultBounds(new ParamPath.MaterialDnD(Id(1)), Singlet));
        Assert.Equal((1e-3, (double?)null), InspectorVariables.DefaultBounds(new ParamPath.ShapeParam(Id(1), ShapeParameter.NormRadius), Singlet));
        Assert.Equal((null, null), InspectorVariables.DefaultBounds(new ParamPath.Curvature(Id(1)), Singlet));
    }
}

public class InspectorEditsTests
{
    static readonly OpticalSystem Singlet = SampleSystems.Singlet();

    [Fact]
    public void TitlesNameObjectImageAndSurfaces()
    {
        Assert.Equal("Object", InspectorEdits.SurfaceTitle(Singlet, 0));
        Assert.Equal("Surface 1", InspectorEdits.SurfaceTitle(Singlet, 1));
        Assert.Equal("Image", InspectorEdits.SurfaceTitle(Singlet, Singlet.ImageIndex));
    }

    [Fact]
    public void AnIdentityPlacementIsRemovedExceptOnACoordinateBreak()
    {
        var s = Singlet.Surfaces[1];
        var tilted = InspectorEdits.WithPlacement(s, p => p with { TiltX = 2 });
        Assert.Equal(2, tilted.Placement!.TiltX);
        Assert.Null(InspectorEdits.WithPlacement(tilted, p => p with { TiltX = 0 }).Placement);
        Assert.NotNull(InspectorEdits.WithPlacement(s, p => p with { RestoreAfter = true }).Placement);
        var cb = s with { Shape = new CoordinateBreakShape() };
        Assert.NotNull(InspectorEdits.WithPlacement(cb, p => p with { TiltX = 0 }).Placement);
    }

    [Fact]
    public void ClippingStartsAtTheSemiDiameter()
    {
        Assert.Equal(new ApertureShape.Annular(2, 6), InspectorEdits.MakeClip(ClipShape.Annular, 6, 9));
        Assert.Equal(new ApertureShape.Circular(9), InspectorEdits.MakeClip(ClipShape.Circular, null, 9));
        Assert.Equal(new ApertureShape.Rectangular(5, 5), InspectorEdits.MakeClip(ClipShape.Rectangular, null, null));
        Assert.Null(InspectorEdits.MakeClip(ClipShape.None, 1, 1));
        Assert.Equal(ClipShape.Elliptical, InspectorEdits.Clip(new ApertureShape.Elliptical(1, 2)));
        Assert.Equal(new SemiDiameter.Fixed(12.5), InspectorEdits.SemiDiameterMode(Singlet.Surfaces[1], false, 12.5));
        Assert.IsType<SemiDiameter.Auto>(InspectorEdits.SemiDiameterMode(Singlet.Surfaces[1], true, 12.5));
    }

    [Fact]
    public void MeltOffsetsStayWithinTheEnginesBounds()
    {
        Assert.True(InspectorEdits.OffsetDnD("0.002").Ok);
        Assert.False(InspectorEdits.OffsetDnD("0.06").Ok);
        Assert.False(InspectorEdits.OffsetDvD("−11").Ok);
        var glass = Singlet.Surfaces[1];
        var offset = InspectorEdits.WithOffset(glass, o => o with { DnD = 0.001 })!;
        Assert.Equal(0.001, ((MaterialRef.CatalogGlass)offset.Material).Offset.DnD);
        Assert.Null(InspectorEdits.WithOffset(Singlet.Surfaces[2], o => o)); // air
    }

    [Fact]
    public void ADiffractiveBaseKeepsItsCurvatureAndConic()
    {
        var doe = new DiffractiveShape(new ConicShape(0.01, -1), new DiffractivePhase { NormRadius = 5, DesignWavelengthUm = 0.55 });
        var even = (DiffractiveShape)InspectorEdits.WithDiffractiveBase(doe, ShapeKind.EvenAsphere)!;
        Assert.Equal(new EvenAsphereShape(0.01, -1, []), even.Base);
        Assert.Equal(new PlaneShape(), ((DiffractiveShape)InspectorEdits.WithDiffractiveBase(doe, ShapeKind.Plane)!).Base);
        Assert.Equal(doe.Phase, even.Phase);
    }

    [Fact]
    public void EfficiencyIsFullAtTheDesignWavelength()
    {
        var phase = new DiffractivePhase { NormRadius = 5, DesignWavelengthUm = 0.55 };
        Assert.Equal(1, InspectorEdits.Efficiency(new DiffractionEfficiency.Kinoform(), phase, 0.55), 12);
        Assert.True(InspectorEdits.Efficiency(new DiffractionEfficiency.Kinoform(), phase, 0.45) < 1);
        Assert.Equal(Math.Pow(Math.Sin(Math.PI / 2) / (Math.PI / 2), 2), InspectorEdits.Efficiency(new DiffractionEfficiency.Multilevel(2), phase, 0.55), 12);
        Assert.Equal(1, InspectorEdits.Efficiency(new DiffractionEfficiency.Ideal(), phase, 0.45));
    }

    [Fact]
    public void SolvesAreAddedReplacingOneOnTheSameTarget()
    {
        var s = Singlet.Surfaces[1];
        var presets = InspectorEdits.SolvePresets(s);
        Assert.Equal(4, presets.Count);
        var target = new ParamPath.Thickness(s.Id);
        var once = InspectorEdits.AddSolve(Singlet, target, new SolveKind.MarginalRayHeight(0));
        var twice = InspectorEdits.AddSolve(once, target, new SolveKind.ChiefRayHeight(0));
        Assert.Single(twice.Solves);
        Assert.IsType<SolveKind.ChiefRayHeight>(twice.Solves[0].Kind);
        Assert.Equal(2, InspectorEdits.SolvePresets(s with { Shape = new ParaxialShape(50) }).Count);
    }

    [Fact]
    public void FieldsAndWavelengthsAddAfterTheLast()
    {
        var withField = InspectorEdits.AddField(Singlet);
        var last = Singlet.Fields.Fields[^1];
        Assert.Equal(last.Y * 1.25 + (last.Y == 0 ? 5 : 0), withField.Fields.Fields[^1].Y);
        var withWavelength = InspectorEdits.AddWavelength(Singlet);
        Assert.Equal(Singlet.Wavelengths.Wavelengths[^1].Um + 0.05, withWavelength.Wavelengths.Wavelengths[^1].Um, 12);
        var photopic = InspectorEdits.ApplyPreset(Singlet, 5);
        Assert.Equal(5, photopic.Wavelengths.Wavelengths.Count);
        Assert.Equal(0.555, photopic.Wavelengths.PrimaryUm);
        Assert.Equal(7, InspectorEdits.WithField(withField, 0, f => f with { Weight = 7 }).Fields.Fields[0].Weight);
        Assert.Equal(withField, InspectorEdits.WithField(withField, 99, f => f with { Weight = 7 }));
    }

    [Fact]
    public void ConvertSpecAsksForOneKind()
    {
        var s = InspectorEdits.ConvertSpecSettings(FieldKind.ParaxialImageHeight, null);
        Assert.Equal("paraxial_image_height", s["field_kind"]!.GetValue<string>());
        Assert.Null(s["aperture"]);
        Assert.Equal("object_space_na", InspectorEdits.ConvertSpecSettings(null, ApertureKind.ObjectSpaceNA)["aperture"]!.GetValue<string>());
    }

    [Fact]
    public void NumbersUseTheOneFormat()
    {
        Assert.Equal("19.301", InspectorFormat.Short(19.301));
        Assert.Equal("−2.5", InspectorFormat.Short(-2.5));
        Assert.Equal("0.00", InspectorFormat.Fixed(-0.00001, 2));
        Assert.Equal("−1.25", InspectorFormat.Fixed(-1.25, 2));
        Assert.Equal("12.34 µm", InspectorFormat.Length(0.01234));
        Assert.Equal("0.5117 mm", InspectorFormat.Length(0.51171));
        Assert.Equal("100.0 %", InspectorFormat.Percent(1));
        Assert.Equal("∞", InspectorFormat.Exact(double.PositiveInfinity));
        Assert.Equal("0.1", InspectorFormat.Exact(0.1));
        Assert.Equal(0.1, LensValueParser.Number(InspectorFormat.Exact(0.1)));
        Assert.Equal(1e-7, LensValueParser.Number(InspectorFormat.Exact(1e-7)));
    }

    [Fact]
    public void FirstOrderXYReadsThePrimaryRow()
    {
        var json = JsonNode.Parse("""
            {"paraxial_xy":{"symmetry":"meridional","primary_wavelength":1,"squeeze_definition":"EFL y / EFL x",
             "wavelengths":[{"wavelength":0,"efl_x":10,"efl_y":20,"bfl_x":1,"bfl_y":2},
                            {"wavelength":1,"efl_x":50,"efl_y":100,"bfl_x":5,"bfl_y":6,"squeeze":2}]}}
            """);
        var p = FirstOrderXY.Read(json)!;
        Assert.Equal(50, p.EflX);
        Assert.Equal(100, p.EflY);
        Assert.Equal(2, p.Squeeze);
        Assert.Equal("meridional", p.Symmetry);
        Assert.True(FirstOrderXY.IsAnamorphic(json, Singlet));
        Assert.False(FirstOrderXY.IsAnamorphic(JsonNode.Parse("""{"symmetry":"rotational"}"""), Singlet.SetShapeKind(ShapeKind.Toroid, 1)));
        Assert.True(FirstOrderXY.IsAnamorphic(null, Singlet.SetShapeKind(ShapeKind.Toroid, 1)));
        Assert.Null(FirstOrderXY.Read(JsonNode.Parse("{}")));
        Assert.Equal("2.0000×", FirstOrderXY.SqueezeText(2));
    }
}
