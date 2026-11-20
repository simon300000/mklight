using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests;

public class SpotGridTests
{
    const string Json = """
    {
      "airy_radius": 0.002, "defocus": 0.0, "wavelengths": [0, 1], "wavelengths_um": [0.4861, 0.5876, 0.6563],
      "fields": [
        { "field": 0, "field_value": [0, 0], "chief": [0, 0], "centroid": [0, 0], "reference": [0, 0],
          "rms_radius": 0.01, "geo_radius": 0.02, "point_range": [0, 2],
          "through_focus": [
            {"defocus": -0.1, "rms_radius_chief": 0.05, "rms_radius_centroid": 0.04},
            {"defocus": 0.0, "rms_radius_chief": 0.01, "rms_radius_centroid": 0.009},
            {"defocus": 0.1, "rms_radius_chief": 0.06, "rms_radius_centroid": 0.05}] },
        { "field": 1, "field_value": [0, 14], "chief": [0, 5], "centroid": [0, 5.001], "reference": [0, 5],
          "rms_radius": 0.02, "geo_radius": 0.03, "point_range": [2, 2],
          "through_focus": [
            {"defocus": -0.1, "rms_radius_chief": 0.07, "rms_radius_centroid": 0.06},
            {"defocus": 0.0, "rms_radius_chief": 0.02, "rms_radius_centroid": 0.019},
            {"defocus": 0.1, "rms_radius_chief": "inf", "rms_radius_centroid": null}] }
      ]
    }
    """;

    static ResultData Data(bool slopes = true)
    {
        var buffers = new List<IResultArray>
        {
            TestData.Buffer("points", [4, 2], 0.01, 0, 0, 0.01, 0.02, 0, 0, -0.02),
            TestData.Vector("point_wavelength", 0, 1, 0, 1),
            TestData.Vector("point_field", 0, 0, 1, 1),
        };
        if (slopes) buffers.Add(TestData.Buffer("point_slope", [4, 2], 0.1, 0, 0, 0.1, -0.1, 0, 0, 0.2));
        return TestData.Result("spot_diagram", Json, [0.4861, 0.5876, 0.6563], ["0°", "14°"], [.. buffers]);
    }

    [Fact]
    public void PlacesThePointsOnEveryPlaneFromTheirSlopes()
    {
        var g = SpotGridData.Read(Data())!;
        Assert.Equal([-0.1, 0.0, 0.1], g.Planes);
        Assert.Equal(1, g.BaseColumn);
        Assert.Equal(2, g.Rows.Count);
        // Field 1 at −0.1: (0.01, 0) + (−0.1)(0.1, 0) = (0, 0); (0, 0.01) + (−0.1)(0, 0.1) = (0, 0).
        var cell = g.Rows[0].Cells[0];
        Assert.Equal(2, cell.Groups.Count);
        Assert.Equal(0, cell.Groups[0].Wavelength);
        Assert.Equal(0.0, cell.Groups[0].Xy[0], 12);
        Assert.Equal(0.0, cell.Groups[1].Xy[1], 12);
        // Field 2 at +0.1: (0.02, 0) + 0.1(−0.1, 0) = (0.01, 0); (0, −0.02) + 0.1(0, 0.2) = (0, 0).
        var far = g.Rows[1].Cells[2];
        Assert.Equal(0.01, far.Groups[0].Xy[0], 12);
        Assert.Equal(0.0, far.Groups[1].Xy[1], 12);
        Assert.Equal(0.01, far.Extent, 12);
        // The base plane keeps the points as they are.
        Assert.Equal(0.02, g.Rows[1].Cells[1].Groups[0].Xy[0], 12);
    }

    [Fact]
    public void TakesEachCellsRmsFromThePlanesStatistics()
    {
        var g = SpotGridData.Read(Data())!;
        Assert.Equal("chief", g.Reference);
        Assert.Equal(0.05, g.Rows[0].Cells[0].Rms);
        Assert.Equal(0.01, g.Rows[0].Cells[1].Rms);
        Assert.Equal(double.PositiveInfinity, g.Rows[1].Cells[2].Rms);
        // GEO on the base plane is the engine's.
        Assert.Equal(0.02, g.Rows[0].Cells[1].Geo);
        Assert.Equal("Field 2", g.Rows[1].Title);
        Assert.Equal("14°", g.Rows[1].Value);
        Assert.Equal(0.002, g.Airy);
        // One unit for the whole grid: µm, even for a radius above 0.1 mm (Fmt.Length alone would say mm).
        Assert.True(g.Micrometres);
        Assert.Equal("50.00 µm", g.Length(0.05));
        Assert.Equal("200.70 µm", g.Length(0.2007));
        Assert.Equal("0.2007 mm", Fmt.Length(0.2007, micrometres: false));
    }

    [Fact]
    public void ReadsTheCentroidReferenceFromTheFields()
    {
        var json = Json.Replace("\"reference\": [0, 5]", "\"reference\": [0, 5.001]");
        var d = TestData.Result("spot_diagram", json, Data().Buffers.ToArray());
        var g = SpotGridData.Read(d)!;
        Assert.Equal("centroid", g.Reference);
        Assert.Equal(0.04, g.Rows[0].Cells[0].Rms);
        Assert.Contains("about the centroid", g.ScaleNote(SpotScale.Grid));
    }

    [Fact]
    public void ANarrowPaneFoldsToTheBasePlane()
    {
        var g = SpotGridData.Read(Data())!;
        var b = g.BaseOnly();
        Assert.Same(b, g.BaseOnly());
        Assert.Equal([0.0], b.Planes);
        Assert.Equal(0, b.BaseColumn);
        Assert.All(b.Rows, r => Assert.Equal(0, Assert.Single(r.Cells).Column));
        Assert.Equal(0.01, b.Rows[0].Cells[0].Rms);
        // Three planes fold below about 260 px; one plane never does.
        Assert.True(SpotGridLayout.Folds(3, 250));
        Assert.False(SpotGridLayout.Folds(3, 300));
        Assert.False(SpotGridLayout.Folds(1, 100));
    }

    [Fact]
    public void WithoutSlopesShowsTheBasePlaneOnly()
    {
        var g = SpotGridData.Read(Data(slopes: false))!;
        Assert.Equal([0.0], g.Planes);
        Assert.Equal(0, g.BaseColumn);
        Assert.Equal(0.01, g.Rows[0].Cells[0].Rms);
        Assert.Equal(1, g.SampledPlanes);
    }

    [Fact]
    public void RejectsMalformedResults()
    {
        Assert.Null(SpotGridData.Read(TestData.Result("spot_diagram", """{"fields": []}""")));
        Assert.Null(SpotGridData.Read(TestData.Result("spot_diagram", """{"other": 1}""")));
        Assert.Null(SpotGridData.Read(TestData.Result("spot_diagram", Json, TestData.Vector("points", 1, 2, 3))));
    }

    [Fact]
    public void BoxScaleIsCommonOrPerField()
    {
        var extents = new List<IReadOnlyList<double>> { new[] { 0.01, 0.02 }, new[] { 0.04, double.NaN } };
        Assert.Equal([0.046, 0.046], SpotGridData.HalfWidths(extents, 0.001, SpotScale.Grid).Select(v => Math.Round(v, 9)));
        Assert.Equal([0.023, 0.046], SpotGridData.HalfWidths(extents, 0.001, SpotScale.Field).Select(v => Math.Round(v, 9)));
        // The Airy disk is always inside the box.
        Assert.Equal(0.115, SpotGridData.HalfWidths(extents, 0.1, SpotScale.Grid)[0], 9);
    }

    [Fact]
    public void ThinsPlanesKeepingTheEndsAndTheBase()
    {
        Assert.Equal([0, 1, 2], SpotGridData.Thin(3, 9, 1));
        var t = SpotGridData.Thin(15, 9, 7);
        Assert.Equal(9, t.Length);
        Assert.Contains(0, t);
        Assert.Contains(14, t);
        Assert.Contains(7, t);
        var u = SpotGridData.Thin(20, 9, 10);
        Assert.Equal(9, u.Length);
        Assert.Contains(10, u);
    }

    [Fact]
    public void LabelsPlanesAndSizesMarkers()
    {
        Assert.Equal("0", SpotGridData.PlaneLabel(0));
        Assert.Equal("+0.08 mm", SpotGridData.PlaneLabel(0.08));
        Assert.Equal("−0.0125 µm", SpotGridData.PlaneLabel(-0.0000125));
        Assert.Equal("+0.5 µm", SpotGridData.PlaneLabel(0.0005));
        var (r, o) = SpotGridData.MarkerStyle(100, 100);
        Assert.Equal(1.25, r);
        Assert.Equal(0.85, o, 9);
        var (r2, o2) = SpotGridData.MarkerStyle(5000, 60);
        Assert.Equal(0.5, r2);
        Assert.Equal(0.5, o2, 9);
    }

    [Fact]
    public void CellsFillTheWidthAndRowsScroll()
    {
        var m = SpotGridLayout.Compute(3, 2, 600, 400);
        Assert.Equal(84, m.RowHeader);
        Assert.False(m.ScrollsHorizontally);
        Assert.True(m.Cell >= SpotGridLayout.MinCell && m.Cell <= SpotGridLayout.MaxCell);
        Assert.True(SpotGridLayout.Compute(9, 1, 300, 400).ScrollsHorizontally);
        // Many fields keep comfortable cells and scroll rather than shrink.
        Assert.True(SpotGridLayout.Compute(3, 12, 900, 400).Cell >= SpotGridLayout.ComfortableCell);
    }

    [Fact]
    public void FooterShowsTheLargestRmsAndTheAiryRadius()
    {
        Assert.Equal("Max RMS 20.00 µm  ·  Airy 2.00 µm", KeyNumbers.Text(Data()));
    }
}
