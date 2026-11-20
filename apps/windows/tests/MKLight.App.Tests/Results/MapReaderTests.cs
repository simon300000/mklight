using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests.Results;

public class MapReaderTests
{
    [Fact]
    public void SignedDataIsDivergingAndSymmetricAboutZero()
    {
        var s = MapScale.ForQuantity(-0.2, 0.5, signed: true);
        Assert.Equal(MapScaleKind.Diverging, s.Kind);
        Assert.Equal((-0.5, 0.5), (s.Min, s.Max));
        Assert.Equal(0.5, s.Position(0));
        Assert.Equal(1, s.Position(0.5));
        Assert.Equal(0.3, s.Position(-0.2), 12);
        Assert.True(double.IsNaN(s.Position(double.NaN)));
    }

    [Fact]
    public void OtherDataTakesThePerceptualRampOverItsRangeOrZeroToOneForFractions()
    {
        var data = MapScale.ForQuantity(0.85, 1, signed: false);
        Assert.Equal((MapScaleKind.Perceptual, 0.85, 1.0), (data.Kind, data.Min, data.Max));
        var unit = MapScale.ForQuantity(0.85, 1, signed: false, HeatRangeMode.Unit, isFraction: true);
        Assert.Equal((0.0, 1.0), (unit.Min, unit.Max));
        // Not a fraction: the unit mode does not apply.
        Assert.Equal(0.85, MapScale.ForQuantity(0.85, 1, false, HeatRangeMode.Unit, isFraction: false).Min);
    }

    [Fact]
    public void AFlatMapGetsANarrowBandAroundItsValue()
    {
        var s = MapScale.ForQuantity(2, 2, signed: false);
        Assert.True(s.Min < 2 && s.Max > 2);
        Assert.Equal(0.004, s.Max - s.Min, 9);
        Assert.Equal(0.5, s.Position(2), 9);
    }

    [Fact]
    public void LogMagnitudeSpansFiveDecadesBelowThePeak()
    {
        var s = MapScale.Magnitude(1e-9, 0.02, log: true);
        Assert.Equal(2e-7, s.Min, 15);
        Assert.Equal(0, s.Position(1e-12));
        Assert.Equal(1, s.Position(0.02), 12);
        Assert.Equal(0.6, s.Position(2e-4), 9);
        Assert.Equal(2e-4, s.ValueAt(0.6), 12);
        var ticks = s.Ticks(6);
        Assert.Equal([-6.0, -5, -4, -3, -2], ticks.Select(v => Math.Round(Math.Log10(v), 9)));
        var linear = MapScale.Magnitude(0, 0.02, log: false);
        Assert.Equal((0.0, 0.02), (linear.Min, linear.Max));
    }

    [Fact]
    public void BarLabelsUseTheOneNumberFormat()
    {
        Assert.Equal("0.852", MapScale.Label(0.852));
        Assert.Equal("1", MapScale.Label(1));
        Assert.Equal("−0.515", MapScale.Label(-0.515));
        Assert.Equal("10⁻³", MapScale.Label(1e-3));
        Assert.Equal("2.85×10⁻⁶", MapScale.Label(2.85e-6));
    }

    [Fact]
    public void IlluminationLevelsAreRoundValuesStrictlyInsideTheRange()
    {
        Assert.Equal([0.88, 0.9, 0.92, 0.94, 0.96, 0.98], Contours.IlluminationLevels(0.87, 1));
        Assert.Empty(Contours.IlluminationLevels(1, 1));
    }

    [Fact]
    public void MarchingSquaresCrossesTheCellEdgesLinearly()
    {
        // Values rise with x: the 0.5 line is the vertical x = 0.
        var g = new MapGrid(2, 2, [0, 1, 0, 1]) { X0 = -1, X1 = 1, Y0 = -1, Y1 = 1 };
        var s = Assert.Single(Contours.Segments(g, 0.5));
        Assert.Equal(0, s.Ax, 12);
        Assert.Equal(0, s.Bx, 12);
        Assert.Equal(2, Math.Abs(s.Ay - s.By), 12);
        // A cell with a missing corner is skipped.
        Assert.Empty(Contours.Segments(new MapGrid(2, 2, [0, 1, double.NaN, 1]), 0.5));
    }

    [Fact]
    public void UpsamplingAUnitDiskMapLeavesOnlyTheOutsideEmpty()
    {
        // 3 × 3 over [−1, 1]²: the corners are outside the field (no data).
        var n = double.NaN;
        var g = new MapGrid(3, 3, [n, 1, n, 1, 1, 1, n, 1, n]) { UnitDiskNoData = true };
        var u = g.Upsampled(21);
        Assert.Equal(21, u.Nx);
        Assert.Equal(21, u.Ny);
        Assert.True(double.IsNaN(u.Values[0]));                // a corner: outside
        Assert.Equal(1, u.Values[10 * 21 + 10], 12);           // the centre
        Assert.Equal(1, u.Values[3 * 21 + 3], 12);             // inside, next to a missing corner: from the finite ones
    }

    [Fact]
    public void NodesAndNearestAreInPlotUnits()
    {
        var g = new MapGrid(3, 2, [1, 2, 3, 4, 5, 6]) { X0 = 0, X1 = 2, Y0 = 0, Y1 = 1 };
        Assert.Equal((1.0, 1.0), g.Node(4));
        Assert.Equal(5, g.Nearest(2.2, 0.9));
        Assert.Equal((1.0, 6.0), g.Range());
    }

    [Fact]
    public void BuffersAndSlicesBecomeGrids()
    {
        var b = TestData.Buffer("m", [2, 2, 3], 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        var g = MapGrid.From(b, 1, -3, 3, -2, 2)!;
        Assert.Equal((3, 2), (g.Nx, g.Ny));
        Assert.Equal([7.0, 8, 9, 10, 11, 12], g.Values);
        Assert.Equal((-3.0, 3.0), (g.X0, g.X1));
        Assert.Null(MapGrid.From(b, 2));
    }

    [Fact]
    public void PsfFocusCropsToTheVisibleCoreAboutThePeak()
    {
        // 41 × 41 cells of 1 µm, a 3 × 3 spot in the middle.
        var v = new double[41 * 41];
        for (var j = 19; j <= 21; j++) for (var i = 19; i <= 21; i++) v[j * 41 + i] = 1;
        v[20 * 41 + 20] = 2;
        v[0] = 1e-4; // far below 0.1 % of the peak: outside the core
        var g = new MapGrid(41, 41, v) { X0 = -20, X1 = 20, Y0 = -20, Y1 = 20 };
        var f = PsfData.Focus(g);
        Assert.Equal(17, f.Nx); // ±8 cells (the minimum)
        Assert.Equal((-8.0, 8.0), (f.X0, f.X1));
        Assert.Equal(2, f.Values[8 * 17 + 8]);
        // A spread PSF is not cropped below its energy.
        var spread = new MapGrid(41, 41, Enumerable.Repeat(1.0, 41 * 41).ToArray()) { X0 = -20, X1 = 20, Y0 = -20, Y1 = 20 };
        Assert.Same(spread, PsfData.Focus(spread));
    }
}
