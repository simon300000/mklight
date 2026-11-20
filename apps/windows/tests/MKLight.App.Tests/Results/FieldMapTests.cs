using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests.Results;

public class FieldMapTests
{
    const string RiColumns = """
        { "columns": [
            {"name": "relative_illumination", "unit": "1", "definition": "Projected solid angle divided by on-axis value"},
            {"name": "vignetting_fraction", "unit": "1", "definition": "Clipped fraction of stop area"} ],
          "field_coordinates": "field_coordinates", "image_mm": "image_mm", "values": "values", "shape": [3, 3] }
        """;

    /// <summary>A 3 × 3 map over [−1, 1]²: corners outside the field have no data (null everywhere).</summary>
    static ResultData RiMap()
    {
        var n = double.NaN;
        double[] f = [-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1];
        double[] img = [n, n, 0, -10, n, n, -10, 0, 0, 0, 10, 0, n, n, 0, 10, n, n];
        double[] v = [n, n, 0.9, 0.1, n, n, 0.9, 0.1, 1, 0, 0.9, 0.1, n, n, 0.9, 0.1, n, n];
        return TestData.Result("relative_illumination_map", RiColumns,
            TestData.Buffer("field_coordinates", [3, 3, 2], f), TestData.Buffer("image_mm", [3, 3, 2], img), TestData.Buffer("values", [3, 3, 2], v));
    }

    [Fact]
    public void ReadsColumnsSamplesAndImagePoints()
    {
        var d = FieldMapData.Read(RiMap())!;
        Assert.Equal(FieldMapData.Layouts.Map, d.Layout);
        Assert.Equal(["relative_illumination", "vignetting_fraction"], d.Columns.Select(c => c.Name));
        Assert.Equal("Relative illumination", d.Columns[0].Title);
        Assert.Equal("", d.Columns[0].UnitLabel);
        Assert.True(d.Columns[0].IsFraction);
        Assert.Equal((0.0, 0.0), d.Field[4]);
        Assert.Equal(1, d.Value(4, 0));
        Assert.Equal((0.9, 1.0), d.Range(0));
        Assert.True(d.OutsideFieldIsNoData);
        var g = d.Grid(0);
        Assert.True(g.UnitDiskNoData);
        Assert.Equal((-1.0, 1.0, -1.0, 1.0), (g.X0, g.X1, g.Y0, g.Y1));
        Assert.True(d.IsRadial(0));
        Assert.False(d.IsSigned(0));
        // By radius: the axis, then the four edge midpoints.
        Assert.Equal([(0.0, 1.0), (10.0, 0.9)], d.Radial(0).Take(2));
    }

    [Fact]
    public void KeyNumbersOfTheMaps()
    {
        Assert.Equal("Min RI 0.900  ·  Max vignetting 10.0 %", KeyNumbers.Text(RiMap()));
    }

    [Fact]
    public void SignedColumnsAndSweeps()
    {
        var json = """
            { "columns": [ {"name": "stop_error_x", "unit": "mm", "definition": "d"}, {"name": "stop_error_y", "unit": "mm", "definition": "d"},
                           {"name": "variation_y", "unit": "1", "definition": "d"} ],
              "field_coordinates": "field_coordinates", "image_mm": "image_mm", "values": "values", "shape": [1, 3] }
            """;
        // A y sweep: nx = 1, ny = 3.
        var r = TestData.Result("pupil_aberration", json,
            TestData.Buffer("field_coordinates", [3, 1, 2], 0, 0, 0, 0.5, 0, 1),
            TestData.Buffer("image_mm", [3, 1, 2], 0, 0, 0, 5, 0, 10),
            TestData.Buffer("values", [3, 1, 3], 0, 0, 0, 0, 0.002, -0.01, 0, 0.008, -0.03));
        var d = FieldMapData.Read(r)!;
        Assert.Equal(FieldMapData.Layouts.SweepY, d.Layout);
        Assert.Equal(1, d.FirstVaryingColumn());
        Assert.True(d.IsSigned(1)); // an error, zero on axis
        Assert.True(d.IsSigned(2)); // negative
        Assert.Equal([(0.0, 0.0), (0.5, 0.002), (1.0, 0.008)], d.Sweep(1));
        Assert.Equal("8.00 µm", d.Columns[1].Format(0.008));
        Assert.Equal("Max stop error 8.00 µm  ·  Max variation -3.00 %", KeyNumbers.Text(r));
    }

    [Fact]
    public void SensorLimitInterpolatesAndHoldsItsEnds()
    {
        var t = FieldMapData.SensorLimit(JsonNode.Parse("[[0, 10], [10, 20], [20, 20]]"));
        Assert.Equal(10, FieldMapData.Limit(t, -1));
        Assert.Equal(15, FieldMapData.Limit(t, 5));
        Assert.Equal(20, FieldMapData.Limit(t, 30));
        Assert.True(double.IsNaN(FieldMapData.Limit([], 1)));
    }

    [Fact]
    public void BandsSplitWhereTheCurveCrossesTheLimit()
    {
        // CRA rises 0 → 30° over 0…20 mm; the limit is a flat 20°: margin, then violation from 13.33 mm.
        (double, double)[] curve = [(0, 0), (10, 15), (20, 30)];
        var bands = FieldMapData.Bands(curve, [(0, 20)]);
        Assert.Equal(2, bands.Count);
        Assert.False(bands[0].Violated);
        Assert.True(bands[1].Violated);
        Assert.Equal(40.0 / 3, bands[0].Points[^1].X, 9);
        Assert.Equal(40.0 / 3, bands[1].Points[0].X, 9);
        Assert.Equal((20.0, 20.0, 30.0), (bands[1].Points[^1].X, bands[1].Points[^1].Lo, bands[1].Points[^1].Hi));
        Assert.Empty(FieldMapData.Bands(curve, []));
    }

    [Fact]
    public void WithoutTheRunSettingsTheLimitColumnGivesTheTable()
    {
        var json = """
            { "columns": [ {"name": "incidence", "unit": "deg", "definition": "i"}, {"name": "limit", "unit": "deg", "definition": "l"},
                           {"name": "margin", "unit": "deg", "definition": "m"} ],
              "field_coordinates": "field_coordinates", "image_mm": "image_mm", "values": "values", "shape": [1, 3] }
            """;
        var r = TestData.Result("chief_ray_angle", json,
            TestData.Buffer("field_coordinates", [3, 1, 2], 0, 0, 0, 0.5, 0, 1),
            TestData.Buffer("image_mm", [3, 1, 2], 0, 0, 0, 5, 0, 10),
            TestData.Buffer("values", [3, 1, 3], 0, 10, 10, 12, 15, 3, 25, 20, -5));
        var d = FieldMapData.Read(r)!;
        Assert.Equal([(0.0, 10.0), (5.0, 15.0), (10.0, 20.0)], d.LimitTable(null));
        // The run's own table wins; an empty one means no limit.
        Assert.Equal([(0.0, 30.0)], d.LimitTable(JsonNode.Parse("""{"sensor_limit": [[0, 30]]}""")));
        Assert.Empty(d.LimitTable(JsonNode.Parse("""{"sensor_limit": []}""")));
        Assert.Equal("Max CRA 25.00°  ·  at 10 mm  ·  Min margin -5.00°", KeyNumbers.Text(r));
    }

    [Fact]
    public void AResultOfAnotherShapeIsNotAFieldMap()
    {
        Assert.Null(FieldMapData.Read(TestData.Result("pupil_aberration", RiColumns, TestData.Buffer("values", [3, 3, 2], new double[18]))));
        Assert.Null(FieldMapData.Read(TestData.Result("pupil_aberration", RiColumns,
            TestData.Buffer("field_coordinates", [3, 3, 2], new double[18]), TestData.Buffer("values", [3, 3, 5], new double[45]))));
    }
}
