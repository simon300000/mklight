using MKLight.App.Plotting;

namespace MKLight.App.Tests.Results;

public class SparklineGeometryTests
{
    [Fact]
    public void RunsSpanTheRectangleAndBreakAtGaps()
    {
        var runs = SparklineGeometry.Runs([3, 2, double.NaN, 1, 0], 0, 0, 100, 10);
        Assert.Equal(2, runs.Count);
        Assert.Equal(0, runs[0][0].X);
        Assert.Equal(100, runs[1][^1].X);
        Assert.True(runs[0][0].Y < runs[1][^1].Y); // the highest value is drawn highest
        Assert.All(runs.SelectMany(r => r), p => Assert.InRange(p.Y, 0, 10));
    }

    [Fact]
    public void RangeIsNeverEmptyAndLogSkipsNonPositiveValues()
    {
        var (lo, hi) = SparklineGeometry.Range([5, 5, 5])!.Value;
        Assert.True(lo < 5 && hi > 5);
        var (llo, lhi) = SparklineGeometry.Range([0, 1, 100], log: true)!.Value;
        Assert.True(llo < 0 && lhi > 2);
        Assert.Null(SparklineGeometry.Range([double.NaN]));
        Assert.Single(SparklineGeometry.Runs([0, 1, 100], 0, 0, 10, 10, log: true)); // 0 dropped, the rest one run
    }

    [Fact]
    public void DescribeSaysFirstLastAndExtremes()
    {
        Assert.Equal("Merit, 4 values from 3.2 to 0.81, lowest 0.79, highest 3.2", SparklineGeometry.Describe("Merit", [3.2, 1, 0.79, 0.81]));
        Assert.Equal("Merit, no values", SparklineGeometry.Describe("Merit", []));
    }
}
