using MKLight.App.Support;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.Tests.Bench;

public class BenchPlanTests
{
    [Fact]
    public void ParsesRunsAndBackend()
    {
        Assert.Equal((10, "gpu:cuda"), (BenchPlan.Parse("10:cuda").Runs, BenchPlan.Parse("10:cuda").Backend!.Value.WireName));
        Assert.Equal("gpu:cuda", BenchPlan.Parse("3:gpu:cuda").Backend!.Value.WireName);
        Assert.Equal("cpu_reference", BenchPlan.Parse("3:cpu_reference").Backend!.Value.WireName);
        Assert.Equal("auto", BenchPlan.Parse("3:auto").Backend!.Value.WireName);
        Assert.Equal((7, (ExecutionSettings?)null), BenchPlan.Parse("7"));
        Assert.Equal(5, BenchPlan.Parse(null).Runs);
    }

    [Fact]
    public void EachEditIsUniqueAndTiny()
    {
        var system = SampleSystems.Get("double_gauss")!;
        var r = system.Surfaces[1].Radius;
        var radii = Enumerable.Range(1, 50).Select(k => BenchPlan.Edited(system, k).Surfaces[1].Radius).ToList();
        Assert.Equal(50, radii.Distinct().Count());
        Assert.All(radii, x => Assert.InRange(Math.Abs(x / r - 1), 0.9e-9, 3e-9));
        // Only that surface changes.
        Assert.Equal(system.Surfaces[2], BenchPlan.Edited(system, 1).Surfaces[2]);
    }
}
