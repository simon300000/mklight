using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.Model;

namespace MKLight.Tests.Model;

public class FixtureRoundTripTests
{
    [Theory]
    [InlineData("empty")]
    [InlineData("cooke_triplet")]
    [InlineData("kitchen_sink")]
    [InlineData("extended_shapes")]
    public void SystemRoundTrip(string name)
    {
        var s = Fixtures.RoundTrip<OpticalSystem>(name);
        Assert.True(s.StructuralIssues().Count == 0, $"engine-valid fixture must be structurally valid: {string.Join("; ", s.StructuralIssues())}");
    }

    [Fact]
    public void VariantsRoundTrip()
    {
        var systems = Fixtures.RoundTripList<OpticalSystem>("variants");
        Assert.Equal(
            [ApertureKind.ImageSpaceFNumber, ApertureKind.ParaxialWorkingFNumber, ApertureKind.FloatByStopSize, ApertureKind.EntrancePupilDiameter, ApertureKind.EntrancePupilDiameter],
            systems.Select(s => s.Aperture.Kind));
        Assert.Equal(
            [FieldKind.ParaxialImageHeight, FieldKind.RealImageHeight, FieldKind.ObjectAngle, FieldKind.ObjectAngle, FieldKind.ObjectAngle],
            systems.Select(s => s.Fields.Kind));
        Assert.Equal(new RayAiming.Real(1e-10), systems[1].RayAiming);
        Assert.Equal(double.NegativeInfinity, systems[3].Surfaces[0].Thickness);
        Assert.Equal(double.PositiveInfinity, systems[3].Surfaces[2].Thickness);
        // A sag overlay keeps its base's kind; the element placement round-trips.
        var moved = systems[4];
        var id = moved.Surfaces[1].Id;
        Assert.True(moved.Surfaces[1].Shape.TypeName == "sag_overlay" && moved.Surfaces[1].Shape.Kind == ShapeKind.Conic);
        Assert.Equal(
            new ElementPlacement { Last = id, Motion = new Placement { DecenterY = 0.02, TiltX = 0.1 } }, moved.Surfaces[1].ElementPlacement);
        Assert.Equal<double?>(-0.25, moved.ValueAt(new ParamPath.Irregularity(id, IrregularityMode.Spherical)));
        Assert.Equal<double?>(0.02, moved.ValueAt(new ParamPath.Curvature(id)));
        var withFringes = moved.SetValue(new ParamPath.Irregularity(id, IrregularityMode.Astigmatism), 1);
        var withTilt = moved.SetValue(new ParamPath.ElementPlacement(id, id, null, PlacementComponent.TiltY), 0.3);
        var otherPivot = moved.SetValue(new ParamPath.ElementPlacement(id, id, id, PlacementComponent.TiltY), 0.3);
        Assert.NotNull(withFringes);
        Assert.Equal<double?>(1, withFringes.ValueAt(new ParamPath.Irregularity(id, IrregularityMode.Astigmatism)));
        Assert.NotNull(withTilt);
        Assert.Equal(0.3, withTilt.Surfaces[1].ElementPlacement?.Motion.TiltY);
        Assert.Null(otherPivot); // one range and pivot per first surface
    }

    [Fact]
    public void ParamPathsRoundTrip()
    {
        var paths = Fixtures.RoundTripList<ParamPath>("param_paths");
        Assert.Equal(27 + Enum.GetValues<CoefficientSet>().Length + Enum.GetValues<ShapeParameter>().Length, paths.Count);
        Assert.Contains(new ParamPath.Irregularity(Sid.Of(7), IrregularityMode.Spherical), paths);
        Assert.Contains(new ParamPath.ElementPlacement(Sid.Of(7), Sid.Of(9), null, PlacementComponent.DecenterX), paths);
        Assert.Equal(paths.Count, paths.ToHashSet().Count); // every variant decodes to a distinct case
        Assert.Equal(new ParamPath.Asphere(Sid.Of(7), 2), paths[3]);
        Assert.Contains(new ParamPath.ApertureValue(), paths);
        Assert.Contains(new ParamPath.PressureAtm(), paths);
        foreach (var set in Enum.GetValues<CoefficientSet>())
            Assert.Contains(new ParamPath.ShapeCoefficient(Sid.Of(7), set, 2), paths);
        foreach (var p in Enum.GetValues<ShapeParameter>())
            Assert.Contains(new ParamPath.ShapeParam(Sid.Of(7), p), paths);
    }

    [Fact]
    public void EmptyMatchesNewEmpty() => Assert.Equal(OpticalSystem.NewEmpty(), Fixtures.Load("empty"));

    [Fact]
    public void CookeMatchesSample() => Assert.Equal(SampleSystems.CookeTriplet(), Fixtures.Load("cooke_triplet"));

    [Fact]
    public void KitchenSinkCoversEveryVariant()
    {
        var s = Fixtures.Load("kitchen_sink");
        // The extended shapes are in extended_shapes.json.
        Assert.True(s.Surfaces.Select(x => x.Shape.Kind).ToHashSet().SetEquals(Enum.GetValues<ShapeKind>().Where(k => k.IsLegacy())));
        Assert.True(s.Surfaces.Select(x => x.Interaction).ToHashSet().SetEquals(Enum.GetValues<Interaction>()));
        var materials = s.Surfaces.Select(x => x.Material).ToList();
        Assert.Contains(new MaterialRef.Air(), materials);
        Assert.Contains(new MaterialRef.Vacuum(), materials);
        Assert.Contains(new MaterialRef.CatalogGlass { Catalog = "BUILTIN", Name = "N-BK7", Offset = new MaterialOffset(0.0012, -0.8) }, materials);
        Assert.Contains(new MaterialRef.ModelGlass(1.6, 40, 0.002), materials);
        Assert.Contains(new MaterialRef.ConstantIndex(1.5), materials);
        Assert.Equal(4, s.Surfaces.Count(x => x.Aperture is not null));
        Assert.Equal(new SemiDiameter.Fixed(12.5), s.Surfaces[1].SemiDiameter);
        Assert.Equal(new ScatterModel.Gaussian(0.02, 0.1), s.Surfaces[1].Edge.EdgeScatter);
        Assert.Equal(TransformOrder.TiltThenDecenter, s.Surfaces[5].Placement?.Order);
        Assert.True(s.Surfaces[6].Placement?.RestoreAfter);
        Assert.Equal(
            new CoatingDef.ThinFilm(0.55, [new CoatingLayer(new MaterialRef.ConstantIndex(1.38), 0.1)], new ComplexIndex(0.12, 3.4)),
            s.Coatings["Silver"]);
        var measured = Assert.IsType<CoatingDef.Tabulated>(s.Coatings["Measured"]);
        Assert.Equal([0.4, 0.55, 0.7], measured.Reflectance.Select(x => x.Um));
        Assert.Equal(2, s.Mechanics.Count);
        Assert.Equal(3, s.Sensor?.SpectralResponse.Count);
        Assert.Equal(0.5 * 4000 * 0.00376, s.Sensor?.HalfWidth);
        Assert.Equal(new ParamOverride(new ParamPath.Thickness(Sid.Of(0)), 80.0), s.Configurations[0].Overrides[0]);
        Assert.Equal(new EnvironmentConditions(35.5, 0.75), s.Environment);
        Assert.Equal(new ApertureSpec.ObjectSpaceNA(0.05), s.Aperture);
    }

    [Fact]
    public void MissingDefaultedFieldsDecode()
    {
        // Fields the engine marks `#[serde(default)]` may be absent.
        const string json = """
            {"aperture":{"type":"float_by_stop_size"},"fields":{"kind":"object_angle","fields":[{"x":0,"y":0}]},
             "wavelengths":{"wavelengths":[{"um":0.55}],"primary":0},
             "surfaces":[{"id":0,"shape":{"type":"plane"},"thickness":"Infinity","material":{"type":"air"}},
                         {"id":1,"shape":{"type":"conic","curvature":0.01},"thickness":5,"material":{"type":"model","nd":1.5,"vd":60}},
                         {"id":2,"shape":{"type":"plane"},"thickness":0,"material":{"type":"air"}}],
             "stop":1,"ray_aiming":{"mode":"real"}}
            """;
        var s = ModelJson.Decode<OpticalSystem>(json);
        Assert.True(s.Name == "" && s.Coatings.Count == 0 && s.Sensor is null);
        Assert.Equal(double.PositiveInfinity, s.Surfaces[0].Thickness);
        Assert.Equal(new ConicShape(0.01, 0), s.Surfaces[1].Shape);
        Assert.Equal(new MaterialRef.ModelGlass(1.5, 60, 0), s.Surfaces[1].Material);
        Assert.Equal(1, s.Fields.Fields[0].Weight);
        Assert.Equal(1, s.Wavelengths.Wavelengths[0].Weight);
        Assert.Equal(new RayAiming.Real(RayAiming.DefaultTolerance), s.RayAiming);
        Assert.Equal(new EnvironmentConditions(), s.Environment);
    }

    /// <summary>
    /// <c>snapshot_hashes.json</c> holds the engine's BLAKE3 hashes of the system fixtures. Only an engine
    /// binding can reproduce them, so the engine tests compare against it; here we check that it covers
    /// exactly the valid system fixtures.
    /// </summary>
    [Fact]
    public void SnapshotHashFixtureCoversSystemFixtures()
    {
        var hashes = Fixtures.Tree("snapshot_hashes").AsObject().ToDictionary(p => p.Key, p => p.Value!.GetValue<string>());
        Assert.True(hashes.Keys.ToHashSet().SetEquals(["empty.json", "cooke_triplet.json", "kitchen_sink.json", "solved_triplet.json", "extended_shapes.json"]));
        foreach (var (file, hex) in hashes)
        {
            Assert.True(hex.Length == 64 && hex.All(c => c is (>= '0' and <= '9') or (>= 'a' and <= 'f')), file);
            Fixtures.Load(file[..^5]);
        }
        Assert.Equal(hashes.Count, hashes.Values.ToHashSet().Count);
    }

    [Fact]
    public void UnknownTagIsRejected() =>
        Assert.Throws<JsonException>(() => ModelJson.Decode<SurfaceShape>("""{"type":"freeform","curvature":0}"""));
}
