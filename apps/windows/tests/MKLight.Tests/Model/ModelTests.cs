using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.Model;

namespace MKLight.Tests.Model;

public class DisplayTests
{
    [Fact]
    public void RadiusCurvatureConversion()
    {
        Assert.Equal(0.02, RadiusDisplay.CurvatureFromRadius(50));
        Assert.Equal(0, RadiusDisplay.CurvatureFromRadius(double.PositiveInfinity));
        Assert.Equal(0, RadiusDisplay.CurvatureFromRadius(double.NegativeInfinity));
        Assert.Equal(0, RadiusDisplay.CurvatureFromRadius(0));
        Assert.Equal(double.PositiveInfinity, RadiusDisplay.RadiusFromCurvature(0));
        Assert.Equal(-25, RadiusDisplay.RadiusFromCurvature(-0.04));
    }

    [Fact]
    public void SettingRadiusPromotesPlane()
    {
        var s = new Surface { Id = Sid.Of(1), Thickness = 1 }.WithRadius(40);
        Assert.Equal(new ConicShape(1 / 40.0, 0), s.Shape);
        s = s.WithRadius(double.PositiveInfinity);
        Assert.Equal(new ConicShape(0, 0), s.Shape);
        Assert.Equal(double.PositiveInfinity, s.Radius);
        var p = new Surface { Id = Sid.Of(2), Shape = new ParaxialShape(10) }.WithRadius(5);
        Assert.Equal(new ParaxialShape(10), p.Shape); // paraxial surfaces have no curvature
    }

    [Fact]
    public void ExtendedDoubleParsing()
    {
        Assert.Equal(double.PositiveInfinity, ExtendedDouble.Parse("inf"));
        Assert.Equal(double.PositiveInfinity, ExtendedDouble.Parse(" +Infinity "));
        Assert.Equal(double.NegativeInfinity, ExtendedDouble.Parse("-inf"));
        Assert.Equal(0.0025, ExtendedDouble.Parse("2.5e-3"));
        Assert.Null(ExtendedDouble.Parse("abc"));
        Assert.Null(ExtendedDouble.Parse("NaN"));
    }

    [Fact]
    public void NumberFormatting()
    {
        Assert.Equal("Infinity", NumberDisplay.Format(double.PositiveInfinity));
        Assert.Equal("12.3457", NumberDisplay.Format(12.3456789));
        Assert.Equal("1.5000e-05", NumberDisplay.Format(1.5e-5));
        Assert.Equal("0.1", NumberDisplay.Exact(0.1));
        Assert.Equal("inf", NumberDisplay.Exact(double.PositiveInfinity));
    }

    [Fact]
    public void FormattingNeverShowsNegativeZero()
    {
        Assert.Equal("0.0000", NumberDisplay.Format(-0.0));
        Assert.Equal("0.00", NumberDisplay.Format(-0.004, 2));
        Assert.Equal("-0.01", NumberDisplay.Format(-0.006, 2));
        Assert.Equal("0", NumberDisplay.Significant(-0.0));
    }

    [Fact]
    public void SignificantDigits()
    {
        Assert.Equal("1.5", NumberDisplay.Significant(1.5));
        Assert.Equal("1.7552", NumberDisplay.Significant(1.7552));
        Assert.Equal("0.0001", NumberDisplay.Significant(0.0001));
        Assert.Equal("1e-05", NumberDisplay.Significant(0.00001));
        Assert.Equal("12345.678", NumberDisplay.Significant(12345.678));
        Assert.Equal("1e+10", NumberDisplay.Significant(1e10));
        Assert.Equal("∞", NumberDisplay.Significant(double.PositiveInfinity));
        Assert.Equal("0.3333333333", NumberDisplay.Significant(1.0 / 3));
    }

    [Fact]
    public void FormattingIsCultureInvariant()
    {
        var saved = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            Assert.Equal("12345.6789", NumberDisplay.Format(12345.6789));
            Assert.Equal("1.5", NumberDisplay.Significant(1.5));
            Assert.Equal("1.51680/64.17", new MaterialRef.ModelGlass(1.5168, 64.17, 0).DisplayName);
            Assert.Equal(1.5, NumberDisplay.Parse("1,5"));
            var json = ModelJson.Encode(new EnvironmentConditions(20.5, 1));
            Assert.Contains("20.5", json);
        }
        finally
        {
            CultureInfo.CurrentCulture = saved;
        }
    }

    [Fact]
    public void ParsingAcceptsDecimalCommaAndInfinity()
    {
        Assert.Equal(1.5, NumberDisplay.Parse("1,5"));
        Assert.Equal(1.5, NumberDisplay.Parse(" 1.5 "));
        Assert.Equal(-2.5, NumberDisplay.Parse("−2,5"));
        Assert.Equal(double.PositiveInfinity, NumberDisplay.Parse("∞"));
        Assert.Equal(double.PositiveInfinity, NumberDisplay.Parse("inf"));
        Assert.Equal(double.NegativeInfinity, NumberDisplay.Parse("-∞"));
        Assert.Null(NumberDisplay.Parse("abc"));
        Assert.Null(NumberDisplay.Parse(""));
        Assert.Null(NumberDisplay.Parse("nan"));
        Assert.Null(NumberDisplay.Parse("1,234.5")); // no digit grouping
    }

    [Fact]
    public void MaterialDisplayNames()
    {
        Assert.Equal("AIR", new MaterialRef.Air().DisplayName);
        Assert.Equal("N-BK7", MaterialRef.Glass("N-BK7").DisplayName);
        Assert.Equal("1.51680/64.17", new MaterialRef.ModelGlass(1.5168, 64.17, 0).DisplayName);
        Assert.Equal("n=1.5", new MaterialRef.ConstantIndex(1.5).DisplayName);
    }
}

public class EditingTests
{
    [Fact]
    public void InsertBeforeStopShiftsStop()
    {
        var (s, id) = SampleSystems.CookeTriplet().InsertSurface(2);
        Assert.Equal(10, s.Surfaces.Count);
        Assert.Equal(id, s.Surfaces[2].Id);
        Assert.Equal(6, s.Stop);
        Assert.Equal("STOP", s.Surfaces[s.Stop].Label);
        Assert.Equal(Sid.Of(9), id); // fresh ids never collide
        Assert.Empty(s.StructuralIssues());
    }

    [Fact]
    public void InsertAfterStopKeepsStop()
    {
        var (s, _) = SampleSystems.CookeTriplet().InsertSurface(7);
        Assert.Equal(5, s.Stop);
    }

    [Fact]
    public void InsertClampsToValidRange()
    {
        var (s, _) = OpticalSystem.NewEmpty().InsertSurface(0);
        Assert.Equal(double.PositiveInfinity, s.Surfaces[0].Thickness); // object stays first
        (s, _) = s.InsertSurface(99);
        Assert.Equal(Sid.Of(2), s.Surfaces[^1].Id); // image stays last
    }

    [Fact]
    public void DeleteKeepsStopConsistent()
    {
        var s = SampleSystems.CookeTriplet().DeleteSurface(1)!;
        Assert.True(s.Stop == 4 && s.Surfaces[4].Label == "STOP");
        s = s.DeleteSurface(4)!; // deleting the stop moves the stop to its neighbour
        Assert.Equal(4, s.Stop);
        Assert.Null(s.DeleteSurface(0)); // object and image are permanent
        Assert.Null(s.DeleteSurface(s.ImageIndex));
        Assert.Empty(s.StructuralIssues());
    }

    [Fact]
    public void CannotDeleteLastInnerSurface()
    {
        var s = OpticalSystem.NewEmpty();
        Assert.Null(s.DeleteSurface(1));
        Assert.Equal(3, s.Surfaces.Count);
    }

    [Fact]
    public void DeleteDropsDanglingOverrides()
    {
        var s = SampleSystems.CookeTriplet() with
        {
            Configurations =
            [
                new Configuration
                {
                    Name = "c",
                    Overrides = [new ParamOverride(new ParamPath.Thickness(Sid.Of(3)), 1.0), new ParamOverride(new ParamPath.TemperatureC(), 30.0)],
                },
            ],
        };
        s = s.DeleteSurface(3)!;
        Assert.Equal([new ParamOverride(new ParamPath.TemperatureC(), 30.0)], s.Configurations[0].Overrides);
    }

    [Fact]
    public void ShapeKindChangeKeepsCurvature()
    {
        var s = SampleSystems.CookeTriplet();
        var c = s.Surfaces[1].Shape.Curvature;
        s = s.SetShapeKind(ShapeKind.EvenAsphere, 1);
        Assert.Equal(new EvenAsphereShape(c, 0, []), s.Surfaces[1].Shape);
        s = s.SetShapeKind(ShapeKind.CoordinateBreak, 1);
        Assert.Equal(new Placement(), s.Surfaces[1].Placement);
    }

    [Fact]
    public void SettingStop()
    {
        var s = SampleSystems.CookeTriplet();
        Assert.Equal(3, s.SetStop(3).Stop);
        Assert.Equal(5, s.SetStop(0).Stop);
        Assert.Equal(5, s.SetStop(s.ImageIndex).Stop);
    }
}

/// <summary>One model and the engine's verdict on it (<c>validation_cases.json</c>).</summary>
internal sealed record EngineVerdict(string Name, OpticalSystem System, string? Error)
{
    public static IReadOnlyList<EngineVerdict> All() =>
        Fixtures.Tree("validation_cases").AsArray().Select(n => new EngineVerdict(
            n!["name"]!.GetValue<string>(), OpticalSystem.FromJson(n["system"]), n["error"]?.GetValue<string>())).ToList();
}

public class ValidationTests
{
    [Fact]
    public void DetectsEngineValidationErrors()
    {
        var s = OpticalSystem.NewEmpty() with
        {
            Stop = 2,
            Fields = new FieldSet { Kind = FieldKind.ObjectHeight },
        };
        s = s.WithSurface(1, x => x with { Thickness = double.PositiveInfinity, Coating = "missing" })
            .WithSurface(2, x => x with { Interaction = Interaction.Reflect });
        var messages = s.StructuralIssues().Select(i => i.Message).ToList();
        Assert.Contains(messages, m => m.Contains("stop index 2"));
        Assert.Contains(messages, m => m.Contains("infinite thickness"));
        Assert.Contains(messages, m => m.Contains("unknown coating"));
        Assert.Contains(messages, m => m.Contains("finite object distance"));
        Assert.Contains(messages, m => m.Contains("cannot be a mirror"));
    }

    [Fact]
    public void DetectsDanglingConfigurationOverride()
    {
        var s = OpticalSystem.NewEmpty() with
        {
            Configurations = [new Configuration { Overrides = [new ParamOverride(new ParamPath.FieldY(3), 1)] }],
        };
        Assert.Single(s.StructuralIssues());
    }

    /// <summary>
    /// Engine verdicts on parameter paths that do or do not fit the surface's shape, material or semi-diameter
    /// mode (the cases with a configuration override). The mirror must agree case by case.
    /// </summary>
    [Fact]
    public void AgreesWithEngineOnParameterPaths()
    {
        var cases = EngineVerdict.All().Where(c => c.System.Configurations.Count > 0).ToList();
        Assert.True(cases.Count >= 70 && cases.Any(c => c.Error is null) && cases.Any(c => c.Error is not null));
        foreach (var c in cases)
        {
            var errors = c.System.StructuralIssues().Where(i => i.Severity == IssueSeverity.Error).ToList();
            var over = c.System.Configurations[0].Overrides[0];
            var path = over.Path;
            if (c.Error is { } engine)
            {
                Assert.True(errors.Count == 1, $"{c.Name}: engine rejects ({engine}), found {string.Join("; ", errors)}");
                var why = path.UnresolvedReason(c.System) ?? path.ValueRejection(over.Value);
                Assert.NotNull(why);
                Assert.True(engine.Trim('\'').EndsWith(why, StringComparison.Ordinal), $"{c.Name}: {engine} vs {why}");
                // Reading and writing agree with the engine's get/set (whose semi-diameter setter fixes an automatic one).
                if (path is not ParamPath.SemiDiameter)
                    Assert.True(c.System.SetValue(path, over.Value) is null, $"{c.Name}: set what the engine refuses");
            }
            else
            {
                Assert.True(errors.Count == 0, $"{c.Name}: engine accepts, found {string.Join("; ", errors)}");
                Assert.True(c.System.ValueAt(path) is not null, c.Name);
                var edited = c.System.SetValue(path, over.Value);
                Assert.True(edited is not null, $"{c.Name}: refuses what the engine sets");
                Assert.Equal<double?>(over.Value, edited.ValueAt(path));
            }
        }
    }

    /// <summary>
    /// Engine verdicts on the structural rules (the cases without a configuration, each with at most one
    /// problem). The mirror must accept what the engine accepts and report the engine's one problem as an
    /// error, with the engine's code and path where its message carries them.
    /// </summary>
    [Fact]
    public void AgreesWithEngineOnStructuralRules()
    {
        var cases = EngineVerdict.All().Where(c => c.System.Configurations.Count == 0).ToList();
        Assert.True(cases.Count(c => c.Error is null) >= 15 && cases.Count(c => c.Error is not null) >= 60);
        var located = 0;
        foreach (var c in cases)
        {
            var issues = c.System.StructuralIssues();
            Assert.True(issues.All(i => i.Severity == IssueSeverity.Error), c.Name);
            if (c.Error is not { } engine)
            {
                Assert.True(issues.Count == 0, $"{c.Name}: engine accepts, found {string.Join("; ", issues)}");
                continue;
            }
            Assert.True(issues.Count == 1, $"{c.Name}: engine rejects ({engine}), found {string.Join("; ", issues)}");
            // "invalid model: [code] path: message".
            var message = issues[0].Message;
            var start = engine.IndexOf('[');
            if (!message.StartsWith('[') || start < 0) continue;
            var end = engine.IndexOf(": ", start, StringComparison.Ordinal);
            if (end < 0) continue;
            var prefix = engine[start..(end + 2)];
            Assert.True(message.StartsWith(prefix, StringComparison.Ordinal), $"{c.Name}: {engine} vs {message}");
            located++;
        }
        Assert.True(located >= 50);
    }

    private static List<string> Messages(Func<OpticalSystem, OpticalSystem> edit)
    {
        var issues = edit(SampleSystems.CookeTriplet()).StructuralIssues();
        Assert.All(issues, i => Assert.Equal(IssueSeverity.Error, i.Severity));
        return issues.Select(i => i.Message).ToList();
    }

    private static OpticalSystem SetShape(OpticalSystem s, int i, SurfaceShape shape) => s.WithSurface(i, x => x with { Shape = shape });

    [Fact]
    public void EvenAsphereHoldsTwentyCoefficients()
    {
        static SurfaceShape Asphere(int terms) => new EvenAsphereShape(0.01, 0, Enumerable.Repeat(1e-30, terms).ToValueList());
        Assert.Empty(Messages(s => SetShape(s, 2, Asphere(20))));
        Assert.Equal(
            ["[coefficient_count] surfaces[2].shape.coefficients: at most 20 coefficients"], Messages(s => SetShape(s, 2, Asphere(21))));
        // The parameter path and its setter stop at the same term.
        var sys = SetShape(SampleSystems.CookeTriplet(), 2, Asphere(1));
        var id = sys.Surfaces[2].Id;
        var last = sys.SetValue(new ParamPath.Asphere(id, 19), 1e-40);
        Assert.NotNull(last);
        Assert.Equal(20, last.Surfaces[2].Shape.AsphereCoefficients.Count);
        Assert.Empty(last.StructuralIssues());
        var beyond = new ParamPath.Asphere(id, 20);
        Assert.Equal("[coefficient_count] index exceeds shape limit", beyond.UnresolvedReason(last));
        Assert.Null(last.ValueAt(beyond));
        Assert.Null(last.SetValue(beyond, 1));
    }

    [Fact]
    public void WavelengthTablesIncreaseStrictly()
    {
        static List<string> Coating(params (double, double)[] table) =>
            Messages(s => s with { Coatings = s.Coatings.Set("T", new CoatingDef.Tabulated(table.Select(t => new SpectralSample(t.Item1, t.Item2)).ToValueList())) });
        Assert.Empty(Coating((0.4, 0.01), (0.55, 0.03), (0.7, 0.02)));
        Assert.Empty(Coating((0.55, 0.03)));
        Assert.Equal(["[invalid_coating] coatings[\"T\"].reflectance: a tabulated coating needs at least one sample"], Coating());
        Assert.Equal(
            ["[invalid_coating] coatings[\"T\"].reflectance[2][0]: wavelengths must increase (0.55 follows 0.7)"],
            Coating((0.4, 0.01), (0.7, 0.02), (0.55, 0.03)));
        Assert.Equal(
            ["[invalid_coating] coatings[\"T\"].reflectance[1][0]: wavelengths must increase (0.4 follows 0.4)"],
            Coating((0.4, 0.01), (0.4, 0.02), (0.55, 0.03)));
        static List<string> Sensor(params (double, double)[] response) => Messages(s => s with
        {
            Sensor = new Sensor
            {
                PixelsX = 10, PixelsY = 10, PitchX = 0.004, PitchY = 0.004,
                SpectralResponse = response.Select(t => new SpectralSample(t.Item1, t.Item2)).ToValueList(),
            },
        });
        Assert.Empty(Sensor());
        Assert.Empty(Sensor((0.45, 0.6), (0.55, 1), (0.65, 0.7)));
        Assert.Equal(
            ["[invalid_sensor] sensor.spectral_response[1][0]: wavelengths must increase (0.45 follows 0.55)"],
            Sensor((0.55, 1), (0.45, 0.6), (0.65, 0.7)));
        Assert.Equal(
            ["[invalid_sensor] sensor.spectral_response[2][0]: wavelengths must increase (0.55 follows 0.55)"],
            Sensor((0.45, 0.6), (0.55, 1), (0.55, 0.7)));
    }

    [Theory]
    [InlineData(1.0, "1.0")]
    [InlineData(0.5, "0.5")]
    [InlineData(0.0, "0.0")]
    [InlineData(-1.5, "-1.5")]
    public void ModelGlassIndexIsAboveOne(double nd, string shown)
    {
        Assert.Equal(
            [$"[invalid_material] surfaces[1].material.nd: must be > 1 (got {shown})"],
            Messages(s => s.WithSurface(1, x => x with { Material = new MaterialRef.ModelGlass(nd, 50, 0) })));
    }

    [Fact]
    public void ModelGlassJustAboveOneAndCoatingLayers()
    {
        Assert.Empty(Messages(s => s.WithSurface(1, x => x with { Material = new MaterialRef.ModelGlass(1.0001, 50, 0) })));
        // Also the material of a coating layer.
        var layer = new CoatingLayer(new MaterialRef.ModelGlass(1, 50, 0), QuarterWaves: 1);
        Assert.Equal(
            ["[invalid_material] coatings[\"AR\"].layers[0].material.nd: must be > 1 (got 1.0)"],
            Messages(s => s with { Coatings = s.Coatings.Set("AR", new CoatingDef.ThinFilm(0.55, [layer])) }));
    }

    [Fact]
    public void DetectsValuesJsonCannotCarry()
    {
        var s = SampleSystems.CookeTriplet();
        Assert.Empty(s.StructuralIssues());
        var inf = s.WithSurface(1, x => x with { Shape = x.Shape.WithCurvature(double.PositiveInfinity)! });
        var issues = inf.StructuralIssues();
        Assert.True(issues.Count == 1 && issues[0].Surface == 1 && issues[0].Message.Contains("curvature"), string.Join("; ", issues));
        var nan = s with { Fields = s.Fields with { Fields = s.Fields.Fields.With(1, f => f with { Y = double.NaN }) } };
        Assert.Contains(nan.StructuralIssues(), i => i.Message.Contains("fields"));
        var negInf = s.WithSurface(0, x => x with { Thickness = double.NegativeInfinity });
        Assert.Empty(negInf.StructuralIssues()); // ±∞ object distance has a JSON form
    }

    /// <summary>The engine rejects a part attached to a surface that does not exist.</summary>
    [Fact]
    public void DetectsMechanicsOnMissingSurfaces()
    {
        var s = SampleSystems.CookeTriplet() with
        {
            Mechanics =
            [
                new MechanicalPart.AnnularBaffle(Sid.Of(42), 0, 1, 2, new ScatterModel.Lambertian(0.1)),
                new MechanicalPart.Barrel(Sid.Of(1), Sid.Of(43), 9, new ScatterModel.Lambertian(0.1)),
            ],
        };
        var issues = s.StructuralIssues();
        Assert.Equal([IssueSeverity.Error, IssueSeverity.Error], issues.Select(i => i.Severity));
        Assert.Equal(
            ["[missing_surface] mechanics[0].surface: no surface with id 42", "[missing_surface] mechanics[1].to: no surface with id 43"],
            issues.Select(i => i.Message));
    }
}

public class ReferenceMaintenanceTests
{
    private static readonly MechanicalPart Baffle = new MechanicalPart.AnnularBaffle(Sid.Of(3), 0, 1, 2, new ScatterModel.Lambertian(0.1));
    private static readonly MechanicalPart Barrel = new MechanicalPart.Barrel(Sid.Of(1), Sid.Of(7), 9, new ScatterModel.Lambertian(0.1));

    [Fact]
    public void DeleteRemovesMechanicsOnTheSurface()
    {
        var s = (SampleSystems.CookeTriplet() with { Mechanics = [Baffle, Barrel] }).DeleteSurface(3)!;
        Assert.Equal([Barrel], s.Mechanics);
        Assert.Empty(s.StructuralIssues());
    }

    [Fact]
    public void RemovingFieldsAndWavelengthsRenumbersOverrides()
    {
        var s = SampleSystems.CookeTriplet() with
        {
            Configurations =
            [
                new Configuration
                {
                    Overrides =
                    [
                        new ParamOverride(new ParamPath.FieldY(2), 18), new ParamOverride(new ParamPath.FieldX(0), 1),
                        new ParamOverride(new ParamPath.FieldWeight(1), 2), new ParamOverride(new ParamPath.WavelengthUm(2), 0.6),
                        new ParamOverride(new ParamPath.WavelengthWeight(0), 3),
                    ],
                },
            ],
        };
        var field2 = s.Fields.Fields[2];
        s = s.RemoveField(0)!;
        Assert.Equal(field2, s.Fields.Fields[1]);
        Assert.Equal<ParamPath[]>(
            [new ParamPath.FieldY(1), new ParamPath.FieldWeight(0), new ParamPath.WavelengthUm(2), new ParamPath.WavelengthWeight(0)],
            s.Configurations[0].Overrides.Select(o => o.Path).ToArray());
        var primary = s.Wavelengths.Wavelengths[s.Wavelengths.Primary];
        s = s.RemoveWavelength(0)!;
        Assert.Equal(primary, s.Wavelengths.Wavelengths[s.Wavelengths.Primary]);
        Assert.Equal<ParamPath[]>(
            [new ParamPath.FieldY(1), new ParamPath.FieldWeight(0), new ParamPath.WavelengthUm(1)],
            s.Configurations[0].Overrides.Select(o => o.Path).ToArray());
        Assert.Empty(s.StructuralIssues());
        // Removing the primary wavelength moves the primary to a neighbour.
        s = s with { Wavelengths = s.Wavelengths with { Primary = s.Wavelengths.Wavelengths.Count - 1 } };
        s = s.RemoveWavelength(s.Wavelengths.Primary)!;
        Assert.True(s.Wavelengths.Wavelengths.IsValidIndex(s.Wavelengths.Primary));
        Assert.Null(s.RemoveWavelength(0)); // the last wavelength stays
    }

    [Fact]
    public void ShapeChangeDropsOverridesItInvalidates()
    {
        var id = Sid.Of(1);
        var s = SampleSystems.CookeTriplet() with
        {
            Configurations = [new Configuration { Overrides = [new ParamOverride(new ParamPath.Conic(id), -1), new ParamOverride(new ParamPath.Thickness(id), 3)] }],
        };
        s = s.SetShapeKind(ShapeKind.Plane, 1);
        Assert.Equal([new ParamOverride(new ParamPath.Thickness(id), 3)], s.Configurations[0].Overrides);
        Assert.Empty(s.StructuralIssues());
    }

    [Fact]
    public void EditsDropOnlyReferencesTheyInvalidate()
    {
        var before = SampleSystems.CookeTriplet().WithSurface(1, x => x with { SemiDiameter = new SemiDiameter.Fixed(8) });
        var broken = new ParamOverride(new ParamPath.ModelNd(Sid.Of(1)), 1.6); // already invalid: catalog glass
        before = before with { Configurations = [new Configuration { Overrides = [new ParamOverride(new ParamPath.SemiDiameter(Sid.Of(1)), 9), broken] }] };
        var after = before.WithSurface(1, x => x with { SemiDiameter = new SemiDiameter.Auto() }).DropReferencesInvalidated(before);
        Assert.Equal([broken], after.Configurations[0].Overrides); // pre-existing problems stay visible
        Assert.Single(after.StructuralIssues(), i => i.Severity == IssueSeverity.Error);
    }
}

public class ProjectFileTests
{
    [Fact]
    public void EnvelopeRoundTrip()
    {
        var p = new ProjectFile(SampleSystems.CookeTriplet());
        var back = ProjectFile.Decode(p.Encoded());
        Assert.Equal(p, back);
        var tree = JsonTree.Parse(p.Encoded());
        Assert.Equal("mklight-project", tree["format"]?.AsString);
        Assert.NotNull(tree["system"]?["surfaces"]);
    }

    [Fact]
    public void RejectsOtherSchemaVersions()
    {
        var newer = Assert.Throws<ProjectFileException>(() => ProjectFile.Decode("""{"format":"mklight-project","schema_version":99,"system":{}}"""));
        Assert.Equal((ProjectFileError.UnsupportedVersion, 99), (newer.Error, newer.Version));
        Assert.Contains("newer MKLight", newer.Message);
        var older = Assert.Throws<ProjectFileException>(() => ProjectFile.Decode("""{"format":"mklight-project","schema_version":0,"system":{}}"""));
        Assert.Equal((ProjectFileError.UnsupportedVersion, 0), (older.Error, older.Version));
        var foreign = Assert.Throws<ProjectFileException>(() => ProjectFile.Decode("""{"format":"something-else","schema_version":1,"system":{}}"""));
        Assert.Equal(ProjectFileError.NotAProject, foreign.Error);
        // A bare optical system has no schema version.
        var bare = Assert.Throws<ProjectFileException>(() => ProjectFile.Decode(Fixtures.Text("kitchen_sink")));
        Assert.Equal("Not an MKLight project: it has no schema_version", bare.Message);
        var array = Assert.Throws<ProjectFileException>(() => ProjectFile.Decode("[]"));
        Assert.Equal(ProjectFileError.NotAProject, array.Error);
    }

    /// <summary>Sections the app does not edit (views, merit, metadata, unknown keys) survive a round trip unchanged.</summary>
    [Fact]
    public void PreservesUnknownSections()
    {
        var tree = JsonNode.Parse(new ProjectFile(SampleSystems.Singlet()).Encoded())!.AsObject();
        tree["views"] = new JsonArray(new JsonObject
        {
            ["key"] = "w1", ["analysis"] = "system_summary", ["settings"] = new JsonObject(), ["ui"] = new JsonObject { ["zoom"] = 1.5 },
        });
        tree["future_key"] = new JsonObject { ["x"] = 3 };
        tree["catalogs"] = new JsonArray(new JsonObject { ["name"] = "MYCAT", ["path"] = "/tmp/my.agf", ["relative_path"] = null, ["format"] = "agf" });
        var p = ProjectFile.Decode(tree.ToJsonString());
        Assert.Equal([new CatalogReference("MYCAT", "/tmp/my.agf")], p.Catalogs);
        var back = JsonTree.Parse(p.Encoded());
        Assert.True(back.Differences(new JsonTree(tree)).Count == 0, string.Join("; ", back.Differences(new JsonTree(tree))));
        p = p.AddCatalog(new CatalogReference("MYCAT", "/other/my.agf"))
            .AddCatalog(new CatalogReference("SECOND", "/x/second.json", Format: "json"));
        Assert.Equal(["MYCAT", "SECOND"], p.Catalogs.Select(c => c.Name));
        Assert.Equal("/other/my.agf", p.Catalogs[0].Path);
        p = p.StampMetadata("MKLight", "1.0", "0.1.0");
        Assert.Equal("MKLight", p.Sections["metadata"]["app_name"]?.AsString);
        Assert.NotNull(p.Sections["metadata"]["created"]);
        Assert.Equal("0.1.0", p.Sections["metadata"]["engine_version"]?.AsString);
        Assert.Equal("", p.Sections["metadata"]["author"]?.AsString);
        // A second stamp keeps the creation time.
        var created = p.Sections["metadata"]["created"]!.Value;
        var again = p.StampMetadata("MKLight", "1.1", null, DateTimeOffset.UtcNow.AddDays(1));
        Assert.Equal(created, again.Sections["metadata"]["created"]!.Value);
    }

    [Fact]
    public void CatalogCandidatesTryRelativeThenAbsolute()
    {
        var c = new CatalogReference("X", "/abs/x.agf", "x.agf");
        Assert.Equal([Path.Combine("proj", "x.agf"), "/abs/x.agf"], c.Candidates("proj"));
        Assert.Equal(["/abs/x.agf"], c.Candidates(null));
    }

    [Fact]
    public void WritesAndReadsFiles()
    {
        var path = Path.Combine(Path.GetTempPath(), $"mklight-{Guid.NewGuid():N}.mklight");
        try
        {
            var p = new ProjectFile(SampleSystems.CookeTriplet());
            p.Write(path);
            p.Write(path); // replaces
            Assert.Equal(p, ProjectFile.Read(path));
            Assert.False(File.Exists(path + ".tmp"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void EncodingIsDeterministic() =>
        Assert.Equal(ModelJson.Encode(SampleSystems.CookeTriplet()), ModelJson.Encode(SampleSystems.CookeTriplet()));

    [Fact]
    public void SerializerIntegration()
    {
        var s = SampleSystems.CookeTriplet();
        var text = JsonSerializer.Serialize(s);
        Assert.Equal(s, JsonSerializer.Deserialize<OpticalSystem>(text));
        Assert.Equal(JsonTree.Parse(ModelJson.Encode(s)), JsonTree.Parse(text));
    }
}

public class JsonTreeTests
{
    [Fact]
    public void NumbersCompareByValue()
    {
        var a = JsonTree.Parse("""{"a": 10.0, "b": [1, 2.5]}""");
        var b = JsonTree.Parse("""{"b": [1.0, 2.5], "a": 10}""");
        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.Empty(a.Differences(b));
    }

    [Fact]
    public void ReportsDifferences()
    {
        var a = JsonTree.Parse("""{"a": 1, "b": null}""");
        var b = JsonTree.Parse("""{"a": 2}""");
        Assert.Equal(["$.a: 1 vs 2", "$.b: missing on right"], a.Differences(b));
    }

    /// <summary>Philox seeds are full u64 values (CONVENTIONS §6); a double would round them above 2^53.</summary>
    [Fact]
    public void IntegersStayExact()
    {
        const ulong big = 12_345_678_901_234_567_891;
        var v = JsonTree.Parse($"{{\"seed\": {big}}}");
        Assert.Equal(JsonTree.Parse(big.ToString()), v["seed"]!.Value);
        Assert.Equal($"{{\"seed\":{big}}}", ModelJson.ToJsonString(v.Node));
        Assert.NotEqual(JsonTree.Parse((1UL << 60).ToString()), JsonTree.Parse(((1UL << 60) + 1).ToString()));
        Assert.NotEqual(JsonTree.Parse("9007199254740993"), new JsonTree(JsonValue.Create(9_007_199_254_740_992.0)));
        // Mixed representations of the same value are equal and hash alike.
        Assert.Equal(new JsonTree(JsonValue.Create(6)), new JsonTree(JsonValue.Create(6.0)));
        Assert.Equal(new JsonTree(JsonValue.Create(6)).GetHashCode(), new JsonTree(JsonValue.Create(6.0)).GetHashCode());
        Assert.Equal(JsonTree.Parse((1UL << 63).ToString()), new JsonTree(JsonValue.Create(9223372036854775808.0)));
        Assert.Equal(2, new[] { JsonTree.Parse("-2"), new JsonTree(JsonValue.Create(-2.0)), JsonTree.Parse("2.5") }.ToHashSet().Count);
    }
}

public class StrictDecodingTests
{
    /// <summary>serde rejects these; the mirror must too, or a file the app accepts would be refused by the engine.</summary>
    [Fact]
    public void RejectsWhatSerdeRejects()
    {
        Assert.Throws<JsonException>(() => ModelJson.Decode<ParamPath>("""{"param":"field_x","index":-1}"""));
        Assert.Throws<JsonException>(() => ModelJson.Decode<Configuration>("""{"name":null,"overrides":[]}"""));
        Assert.Throws<JsonException>(() => ModelJson.Decode<WavelengthSet>("""{"wavelengths":[{"um":0.55}],"primary":-1}"""));
        var json = SampleSystems.Singlet().ToJson();
        json["surfaces"]![1]!["thickness"] = "nan";
        Assert.Throws<JsonException>(() => OpticalSystem.FromJson(json));
        Assert.Null(ExtendedDouble.Parse("NaN"));
        Assert.Equal(double.NegativeInfinity, ExtendedDouble.Parse("-inf"));
    }

    [Fact]
    public void RejectsMalformedValues()
    {
        Assert.Throws<JsonException>(() => ModelJson.Decode<SurfaceShape>("[]"));
        Assert.Throws<JsonException>(() => ModelJson.Decode<SurfaceShape>("""{"type":"conic","curvature":"x"}"""));
        Assert.Throws<JsonException>(() => ModelJson.Decode<SurfaceShape>("""{"type":"conic","curvature":null}"""));
        Assert.Throws<JsonException>(() => ModelJson.Decode<SurfaceShape>("""{"type":"conic"}"""));
        Assert.Throws<JsonException>(() => ModelJson.Decode<DiffractionEfficiency>("""{"type":"multilevel","levels":0}"""));
        Assert.Throws<JsonException>(() => ModelJson.Decode<DiffractionEfficiency>("""{"type":"multilevel","levels":65536}"""));
        Assert.Equal(new DiffractionEfficiency.Multilevel(65535), ModelJson.Decode<DiffractionEfficiency>("""{"type":"multilevel","levels":65535}"""));
        Assert.Throws<JsonException>(() => ModelJson.Decode<CoatingDef>("""{"type":"ideal","reflectance":0,"transmittance":1,"x":1}"""));
        Assert.Throws<JsonException>(() => SpectralSample.FromJson(JsonNode.Parse("[1]")));
    }

    /// <summary>
    /// The engine's model types deny unknown fields. The mirror must refuse them too: it re-encodes every
    /// snapshot and save, so an unknown key it accepted would vanish silently. Every object of the
    /// engine-written fixtures (every model variant) gets an extra key in turn.
    /// </summary>
    [Theory]
    [InlineData("kitchen_sink", "system")]
    [InlineData("extended_shapes", "system")]
    [InlineData("variants", "systems")]
    [InlineData("solved_triplet", "system")]
    [InlineData("param_paths", "paths")]
    [InlineData("solve_kinds", "solves")]
    public void RejectsUnknownKeysEverywhere(string fixture, string kind)
    {
        void Decode(JsonNode json)
        {
            switch (kind)
            {
                case "system": OpticalSystem.FromJson(json); break;
                case "systems": J(json, OpticalSystem.FromJson); break;
                case "paths": J(json, ParamPath.FromJson); break;
                default: J(json, Solve.FromJson); break;
            }
        }
        var original = Fixtures.Tree(fixture);
        Decode(original);
        var paths = ObjectPaths(original, []);
        Assert.True(paths.Count > 3);
        foreach (var path in paths)
        {
            var tampered = original.DeepClone();
            var target = path.Aggregate(tampered, (n, step) => int.TryParse(step, out var i) && n is JsonArray ? n[i]! : n[step]!);
            target.AsObject()["future_field"] = 1;
            var ex = Assert.Throws<JsonException>(() => Decode(tampered));
            Assert.True(ex.Message.Contains("unknown field 'future_field'"), $"{fixture} $.{string.Join(".", path)}: {ex.Message}");
        }
    }

    private static void J<T>(JsonNode json, Func<JsonNode?, T> read)
    {
        foreach (var n in json.AsArray()) read(n);
    }

    /// <summary>Paths (object keys and array indices) of every JSON object, except name-keyed maps (<c>coatings</c>), whose keys are data.</summary>
    private static List<List<string>> ObjectPaths(JsonNode json, List<string> path)
    {
        var output = new List<List<string>>();
        switch (json)
        {
            case JsonObject o:
                if (path.LastOrDefault() != "coatings") output.Add(path);
                foreach (var (k, v) in o.OrderBy(p => p.Key, StringComparer.Ordinal))
                    if (v is not null) output.AddRange(ObjectPaths(v, [.. path, k]));
                break;
            case JsonArray a:
                for (var i = 0; i < a.Count; i++)
                    if (a[i] is { } e) output.AddRange(ObjectPaths(e, [.. path, i.ToString()]));
                break;
        }
        return output;
    }
}

/// <summary>Applying a <c>convert_spec</c> result to the field kind and aperture type.</summary>
public class SpecConversionTests
{
    [Fact]
    public void FieldValuesReplaceTheNumbersAndKeepWeights()
    {
        var s = SampleSystems.CookeTriplet();
        s = s with { Fields = s.Fields with { Fields = s.Fields.Fields.With(1, f => f with { Weight = 2 }) } };
        var json = JsonNode.Parse("""{"fields": {"kind": "paraxial_image_height", "values": [[0, 0], [0, 3.5], [0, 7]]}, "fields_error": null, "aperture": null, "aperture_error": null, "stop_semi_diameter": 4.2}""");
        var c = SpecConversion.FromJson(json);
        Assert.NotNull(c);
        Assert.True(c.StopSemiDiameter == 4.2 && c.Aperture is null);
        var converted = s.SetFieldKind(FieldKind.ParaxialImageHeight, c);
        Assert.Equal([0, 3.5, 7], converted.Fields.Fields.Select(f => f.Y));
        Assert.Equal(2, converted.Fields.Fields[1].Weight);
        // A result for another field count is not applied.
        var other = s with { Fields = s.Fields with { Fields = s.Fields.Fields.RemoveAt(2) } };
        var other2 = other.SetFieldKind(FieldKind.ParaxialImageHeight, c);
        Assert.Equal(FieldKind.ParaxialImageHeight, other2.Fields.Kind);
        Assert.Equal(other.Fields.Fields, other2.Fields.Fields);
    }

    [Fact]
    public void NonConversionResultsAreRefused()
    {
        Assert.Null(SpecConversion.FromJson(JsonNode.Parse("[]")));
        Assert.Null(SpecConversion.FromJson(JsonNode.Parse("""{"fields":{"values":[[1,2,3]]}}"""))!.FieldValues); // not (x, y) pairs
        Assert.NotNull(SpecConversion.FromJson(JsonNode.Parse("{}")));
    }

    [Fact]
    public void FloatByStopSizeFixesOnlyAnAutomaticStop()
    {
        var s = SampleSystems.CookeTriplet().SetApertureKind(ApertureKind.FloatByStopSize, new SpecConversion { StopSemiDiameter = 4.2 });
        Assert.Equal(new ApertureSpec.FloatByStopSize(), s.Aperture);
        Assert.Equal(new SemiDiameter.Fixed(4.2), s.Surfaces[s.Stop].SemiDiameter);
        var f = SampleSystems.CookeTriplet();
        f = f.WithSurface(f.Stop, x => x with { SemiDiameter = new SemiDiameter.Fixed(3) })
            .SetApertureKind(ApertureKind.FloatByStopSize, new SpecConversion { StopSemiDiameter = 4.2 });
        Assert.Equal(new SemiDiameter.Fixed(3), f.Surfaces[f.Stop].SemiDiameter); // a fixed stop is the user's
        var e = SampleSystems.CookeTriplet().SetApertureKind(ApertureKind.ImageSpaceFNumber, new SpecConversion { Aperture = new ApertureSpec.ImageSpaceFNumber(4.5) });
        Assert.Equal(new ApertureSpec.ImageSpaceFNumber(4.5), e.Aperture);
        // Without a converted value the old number stays.
        var d = SampleSystems.CookeTriplet().SetApertureKind(ApertureKind.ObjectSpaceNA);
        Assert.Equal(new ApertureSpec.ObjectSpaceNA(10), d.Aperture);
    }
}

public class CoatingSubstrateTests
{
    [Fact]
    public void SubstrateSideRoundTripsAndDefaultsToHigherIndex()
    {
        var old = """{"type":"thin_film","reference_wavelength_um":0.55,"layers":[]}""";
        var def = JsonSerializer.Deserialize<CoatingDef>(old);
        Assert.Equal(new CoatingDef.ThinFilm(0.55, [], null), def);
        var written = JsonSerializer.Serialize(def);
        Assert.DoesNotContain("\"substrate\"", written);
        var set = """{"type":"thin_film","reference_wavelength_um":0.55,"layers":[],"substrate":"before"}""";
        var side = JsonSerializer.Deserialize<CoatingDef>(set);
        Assert.Equal(new CoatingDef.ThinFilm(0.55, [], null, CoatingSubstrate.Before), side);
        Assert.Equal(side, JsonSerializer.Deserialize<CoatingDef>(JsonSerializer.Serialize(side)));
    }
}

public class DiffractionEfficiencyTests
{
    [Fact]
    public void IdealPutsEverythingIntoTheTraceOrder()
    {
        var ideal = new DiffractionEfficiency.Ideal();
        Assert.Equal(1, ideal.Efficiency(1, 1, 1));
        Assert.Equal(0, ideal.Efficiency(2, 1, 1));
    }

    [Fact]
    public void KinoformAtItsDesignWavelengthIsFullyEfficient()
    {
        var k = new DiffractionEfficiency.Kinoform();
        Assert.Equal(1, k.Efficiency(1, 1, 1), 12);
        Assert.Equal(0, k.Efficiency(2, 1, 1), 12);
        Assert.True(k.Efficiency(1, 1, 0.9) < 1);
    }
}
