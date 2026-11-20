using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests.Results;

public class ImageSimulationSettingsTests
{
    [Fact]
    public void FieldOfViewChangesOnlyTheExtent()
    {
        var s = TestData.Obj("""{"size": [256, 256], "optics": {"field_extent": 0.1, "psf_size": 129}, "source": {"kind": "pattern", "pattern": "grid"}}""");
        Assert.Equal(0.1, ImageSimulationSettings.FieldExtent(s));
        var t = ImageSimulationSettings.WithFieldExtent(s, 0.02);
        Assert.Equal(0.02, ImageSimulationSettings.FieldExtent(t));
        Assert.Equal(129, Json.Int(t["optics"]!["psf_size"]));
        Assert.Equal(0.1, ImageSimulationSettings.FieldExtent(s)); // input unchanged
        Assert.Equal(ImageSimulationSettings.FullExtent, ImageSimulationSettings.FieldExtent(TestData.Obj("{}")));
    }

    [Fact]
    public void ExtentTitlesAndSensorWidths()
    {
        Assert.Equal(["full field", "50 %", "25 %", "10 %", "5 %", "2 %"], ImageSimulationSettings.Extents.Select(ImageSimulationSettings.ExtentTitle));
        // A run of 256 px at 9.49 µm over a 10 % patch: 12.15 mm half-width per unit of extent.
        var k = ImageSimulationSettings.MmPerExtent(0.00949, 256, 0.1)!.Value;
        Assert.Equal(12.1472, k, 4);
        Assert.Equal("5 % · 1.2147 mm wide, pixel 4.75 µm", ImageSimulationSettings.ExtentLabel(0.05, k, 256));
        Assert.Equal("5 %", ImageSimulationSettings.ExtentLabel(0.05, null, 256));
        Assert.Null(ImageSimulationSettings.MmPerExtent(double.NaN, 256, 0.1));
    }

    [Fact]
    public void CentreIsRoundedAndOtherOpticsKept()
    {
        var s = TestData.Obj("""{"optics": {"field_extent": 0.1}}""");
        Assert.Equal((0.0, 0.0), ImageSimulationSettings.FieldCenter(s));
        var t = ImageSimulationSettings.WithFieldCenter(s, (-0.000001, 0.714285714));
        Assert.Equal("[0,0.7143]", t["optics"]!["field_center"]!.ToJsonString());
        Assert.Equal(0.1, ImageSimulationSettings.FieldExtent(t));
        Assert.Equal((0.0, 0.7143), ImageSimulationSettings.FieldCenter(t));
    }

    [Fact]
    public void ResolutionSetsBothSides()
    {
        var s = ImageSimulationSettings.WithSize(TestData.Obj("""{"size": [256, 256], "seed": 3}"""), 512);
        Assert.Equal("[512,512]", s["size"]!.ToJsonString());
        Assert.Equal(512, ImageSimulationSettings.Size(s));
        Assert.Equal(3, Json.Int(s["seed"]));
        Assert.Equal(64, ImageSimulationSettings.Size(TestData.Obj("{}")));
        Assert.Equal([128, 256, 512, 1024], ImageSimulationSettings.Sizes);
    }

    [Fact]
    public void FieldPointsAreNormalisedListedOnceAndSnappedTo()
    {
        var points = ImageSimulationSettings.FieldPoints([(0, 0), (0, 10), (0, 14), (0, 10)]);
        Assert.Equal(["On axis", "Field 2 (0.00, +0.71)", "Field 3 (0.00, +1.00)"], points.Select(p => p.Title));
        Assert.Equal(10 / 14.0, points[1].Y, 12);
        Assert.Equal("on axis", ImageSimulationSettings.CenterTitle((0, 0), points));
        Assert.Equal("Field 3 (0.00, +1.00)", ImageSimulationSettings.CenterTitle((0, 1), points));
        Assert.Equal("(+0.30, −0.45)", ImageSimulationSettings.CenterTitle((0.3, -0.45), points));
        Assert.Equal((0.0, 1.0), ImageSimulationSettings.Snap((0.02, 0.97), points, 0.06));
        Assert.Equal((0.0, 0.0), ImageSimulationSettings.Snap((0.03, -0.02), points, 0.06));
        Assert.Equal((0.5, 0.5), ImageSimulationSettings.Snap((0.5, 0.5), points, 0.06));
        Assert.Single(ImageSimulationSettings.FieldPoints(null));
    }

    [Fact]
    public void PatchOutsideTheField()
    {
        Assert.False(ImageSimulationSettings.OutsideField((0, 0), ImageSimulationSettings.FullExtent));
        Assert.True(ImageSimulationSettings.OutsideField((0, 1), 0.1));
        Assert.False(ImageSimulationSettings.OutsideField((0, 0.7), 0.1));
    }
}
