using System.Text.Json.Nodes;
using MKLight.Model;

namespace MKLight.Tests.Model;

/// <summary>Solves and solve reports against JSON written by the Rust side.</summary>
public class SolveFixtureTests
{
    [Fact]
    public void EverySolveKindRoundTrips()
    {
        var solves = Fixtures.RoundTripList<Solve>("solve_kinds");
        Assert.True(solves.Select(s => s.Kind.Type).ToHashSet().SetEquals(Enum.GetValues<SolveType>()));
        Assert.Equal(new SolveKind.MarginalRayAngle(-0.1), solves[1].Kind);
        Assert.Equal(new SolveKind.Pickup(new ParamPath.Conic(Sid.Of(2)), 2, -1), solves[4].Kind);
        Assert.Equal(new SolveKind.EdgeThickness(2, 8), solves[7].Kind);
        Assert.Equal(new SolveKind.EdgeThickness(1.5, null), solves[8].Kind);
    }

    [Fact]
    public void PickupDefaultsDecode()
    {
        var json = """{"target":{"param":"conic","surface":1},"kind":{"type":"pickup","source":{"param":"conic","surface":2}}}""";
        var s = ModelJson.Decode<Solve>(json);
        Assert.Equal(new SolveKind.Pickup(new ParamPath.Conic(Sid.Of(2)), 1, 0), s.Kind);
    }

    [Fact]
    public void SolvedSystemRoundTripsAndValidates()
    {
        var s = Fixtures.RoundTrip<OpticalSystem>("solved_triplet");
        Assert.Equal(6, s.Solves.Count);
        Assert.Equal(["infinity", "2 m"], s.Configurations.Select(c => c.Name));
        Assert.True(s.StructuralIssues().Count == 0, string.Join("; ", s.StructuralIssues()));
        Assert.Contains(new ParamPath.Thickness(Sid.Of(7)), s.SolvedParams);
        Assert.Equal(new SolveKind.ChiefRayAngle(0.25), s.SolveFor(new ParamPath.Curvature(Sid.Of(3)))?.Kind);
    }

    /// <summary>Without solves the key is omitted, so such systems hash as before.</summary>
    [Fact]
    public void EmptySolvesAreOmitted()
    {
        var tree = JsonTree.Parse(ModelJson.Encode(SampleSystems.CookeTriplet()));
        Assert.Null(tree["solves"]);
    }

    private sealed record ReportEntry(OpticalSystem System, SolveReport Report) : IJsonModel<ReportEntry>
    {
        public static ReportEntry FromJson(JsonNode? node)
        {
            var o = node!.AsObject();
            return new ReportEntry(OpticalSystem.FromJson(o["system"]), SolveReport.FromJson(o["report"]));
        }

        public JsonNode ToJson() =>
            new JsonObject { ["system"] = System.ToJson(), ["report"] = Report.ToJson() };
    }

    [Fact]
    public void SolveReportsRoundTrip()
    {
        var entries = Fixtures.RoundTripList<ReportEntry>("solve_reports");
        Assert.Equal([null, 0, 1], entries.Select(e => e.Report.Configuration));
        var given = entries[0];
        Assert.Equal(2, given.System.Configurations.Count); // solving as given keeps the configurations
        Assert.Equal(2, given.Report.Outcomes.Count);
        Assert.All(given.Report.Outcomes, o => Assert.True(Math.Abs(o.Residual) < 1e-6));
        Assert.NotNull(given.Report.RunInfo["engine_version"]);
        // Per-configuration results are flattened, and the override of a solved thickness is reported.
        Assert.True(entries[1].System.Configurations.Count == 0 && entries[2].System.Configurations.Count == 0);
        Assert.Contains("solve wins", entries[2].Report.Warnings[0]);
        Assert.Equal(JsonTree.Parse("1"), entries[2].Report.RunInfo["settings"]!.Value["configuration"]!.Value);
        Assert.True(entries[1].Report.SolvedParams.ToHashSet().SetEquals(
            [new ParamPath.Thickness(Sid.Of(2)), new ParamPath.Curvature(Sid.Of(2))]));
    }

    [Fact]
    public void SlopeIsLabelledAsParaxialAngle()
    {
        Assert.Equal("paraxial angle u", SolveType.MarginalRayAngle.ValueLabel()?.Label);
        Assert.Equal("", SolveType.ChiefRayAngle.ValueLabel()?.Unit);
        var k = new SolveKind.ChiefRayAngle(0.1).WithMainValue(0.2);
        Assert.Equal(new SolveKind.ChiefRayAngle(0.2), k); // the slope passes through unconverted
    }
}

public class SolveEditingTests
{
    private static OpticalSystem Solved() => SampleSystems.CookeTriplet() with
    {
        Solves =
        [
            new Solve(new ParamPath.Thickness(Sid.Of(7)), new SolveKind.MarginalRayHeight(0)),
            new Solve(new ParamPath.Curvature(Sid.Of(6)), new SolveKind.Pickup(new ParamPath.Curvature(Sid.Of(1)), -1, 0)),
            new Solve(new ParamPath.FieldY(2), new SolveKind.Pickup(new ParamPath.FieldY(1), 1.5, 0)),
        ],
    };

    [Fact]
    public void DeletingASurfaceDropsItsSolves()
    {
        var s = Solved().DeleteSurface(1)!;
        Assert.Equal<ParamPath[]>([new ParamPath.Thickness(Sid.Of(7)), new ParamPath.FieldY(2)], s.Solves.Select(x => x.Target).ToArray());
    }

    [Fact]
    public void RemovingAFieldRenumbersSolves()
    {
        var s = Solved().RemoveField(0)!;
        Assert.Equal(new ParamPath.FieldY(1), s.Solves[2].Target);
        Assert.Equal(new SolveKind.Pickup(new ParamPath.FieldY(0), 1.5, 0), s.Solves[2].Kind);
        s = s.RemoveField(0)!;
        Assert.Equal(2, s.Solves.Count); // a pickup of a removed field goes
    }

    [Fact]
    public void InvalidatedSolveIsDropped()
    {
        var before = Solved();
        var s = before.WithSurface(6, x => x with { Shape = new ParaxialShape(40) }).DropReferencesInvalidated(before);
        Assert.Equal(2, s.Solves.Count);
        Assert.Empty(s.StructuralIssues());
    }

    [Fact]
    public void StructuralChecks()
    {
        var s = Solved();
        s = s with
        {
            Solves = s.Solves.AddRange(
            [
                new Solve(new ParamPath.Thickness(Sid.Of(7)), new SolveKind.ChiefRayHeight(1)),
                new Solve(new ParamPath.Conic(Sid.Of(5)), new SolveKind.FNumber(-1)),
                new Solve(new ParamPath.Thickness(Sid.Of(3)), new SolveKind.EdgeThickness(double.PositiveInfinity, null)),
            ]),
        };
        var messages = s.StructuralIssues().Select(i => i.Message).ToList();
        Assert.Contains(messages, m => m.Contains("already has a solve"));
        Assert.Contains(messages, m => m.Contains("surface has no conic"));
        Assert.Contains(messages, m => m.Contains("F/# must be > 0"));
        Assert.Contains(messages, m => m.Contains("finite"));
        // An automatic semi-diameter may be a target (the solve fixes it).
        var t = SampleSystems.CookeTriplet() with
        {
            Solves = [new Solve(new ParamPath.SemiDiameter(Sid.Of(1)), new SolveKind.Pickup(new ParamPath.Thickness(Sid.Of(1)), 3, 0))],
        };
        Assert.Empty(t.StructuralIssues());
    }
}
