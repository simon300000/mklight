using System.Text.Json.Nodes;
using MKLight.App.Views.Viewport3D;

namespace MKLight.Tests.Viewport;

public class LayoutModesTests
{
    [Fact]
    public void TheEngineSymmetryDecidesAnamorphic()
    {
        Assert.True(LayoutModes.IsAnamorphic(JsonNode.Parse("""{"paraxial_xy":{"symmetry":"meridional"}}"""), null));
        Assert.False(LayoutModes.IsAnamorphic(JsonNode.Parse("""{"symmetry":"rotational"}"""), null));
        Assert.False(LayoutModes.IsAnamorphic(null, null));
    }

    [Fact]
    public void SelectorOffersBothSectionsOnlyWithoutRotationalSymmetry()
    {
        Assert.Equal([ViewportMode.SectionYZ, ViewportMode.Orbit3D], LayoutModes.Available(false, ViewportMode.Orbit3D));
        Assert.Equal(3, LayoutModes.Available(true, ViewportMode.SectionYZ).Length);
        Assert.Equal(3, LayoutModes.Available(false, ViewportMode.SectionXZ).Length); // while X–Z is shown
        Assert.Equal("Section", LayoutModes.Title(ViewportMode.SectionYZ, 2));
        Assert.Equal("Y–Z", LayoutModes.Title(ViewportMode.SectionYZ, 3));
        Assert.Equal("xz", LayoutModes.SectionSetting(ViewportMode.SectionXZ));
        Assert.Null(LayoutModes.SectionSetting(ViewportMode.Orbit3D));
        Assert.False(LayoutModes.SceneMatches(ViewportMode.SectionXZ, "yz"));
        Assert.True(LayoutModes.SceneMatches(ViewportMode.Orbit3D, "xz"));
    }

    [Theory]
    [InlineData(20, "20 mm")]
    [InlineData(0.5, "0.5 mm")]
    [InlineData(68.93561, "68.94 mm")]
    [InlineData(-0.0, "0 mm")]
    public void LengthLabels(double mm, string text) => Assert.Equal(text, LayoutModes.LengthLabel(mm));
}
