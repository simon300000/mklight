using MKLight.App.Views.Viewport3D;
using MKLight.Engine;
using Kind = MKLight.Engine.LayoutScene.PolylineKind;

namespace MKLight.Tests.Viewport;

public class LayoutTouchAndSummaryTests
{
    const double W = 800, H = 600;

    [Theory]
    [InlineData(ViewportMode.SectionYZ, 1)]
    [InlineData(ViewportMode.SectionYZ, 5)]
    [InlineData(ViewportMode.SectionXZ, 1)]
    [InlineData(ViewportMode.Orbit3D, 1)]
    public void PinchKeepsThePointUnderTheFingers(ViewportMode mode, double stretch)
    {
        var camera = new ViewportCamera { Center = new(0, 1, 20), PointsPerMM = 6, Stretch = stretch, IsFitted = true };
        // The pinch centre starts at (500, 200) and moves by (30, −40) while the fingers spread 1.5×.
        double x0 = 500, y0 = 200, dx = 30, dy = -40, scale = 1.5;
        var world = Unproject(camera, mode, x0, y0);
        var after = ViewportMath.Pinch(camera, mode, scale, x0 + dx, y0 + dy, dx, dy, W, H);
        Assert.Equal(camera.PointsPerMM * scale, after.PointsPerMM, 9);
        var (sx, sy) = ViewportMath.ToScreen(after, mode, world, W, H);
        Assert.Equal(x0 + dx, sx, 6);
        Assert.Equal(y0 + dy, sy, 6);
    }

    [Fact]
    public void PinchIgnoresAMeaninglessScale()
    {
        var camera = new ViewportCamera { PointsPerMM = 3 };
        Assert.Equal(3, ViewportMath.Pinch(camera, ViewportMode.SectionYZ, double.NaN, 10, 10, 0, 0, W, H).PointsPerMM);
        Assert.Equal(3, ViewportMath.Pinch(camera, ViewportMode.SectionYZ, 0, 10, 10, 0, 0, W, H).PointsPerMM);
    }

    /// <summary>The world point (on the view plane through the camera centre) drawn at a screen point.</summary>
    static V3 Unproject(ViewportCamera c, ViewportMode mode, double x, double y)
    {
        var r = ViewportMath.Rotation(c, mode);
        var v = new V3((x - W / 2) / c.PointsPerMM, (H / 2 - y) / (c.PointsPerMM * c.VerticalScale(mode)), 0);
        return c.Center + r.Transpose() * v;
    }

    static LayoutScene Scene(int fields) => new()
    {
        Polylines =
        [
            new(Kind.Ray, [new(0, 1, 0), new(0, 0, 50)], true, Field: 0),
            new(Kind.Ray, [new(1, 0, 0), new(0, 0, 50)], false, Field: 1),
            new(Kind.SurfaceProfile, [new(0, -5, 0), new(0, 5, 0)], true, Surface: 1),
        ],
        Elements = [],
        Surfaces = [],
        Bounds = new Bounds3(new(-5, -5, -10), new(5, 5, 50)),
        FieldCount = fields,
    };

    [Fact]
    public void SummaryNamesTheViewSurfacesAndFields()
    {
        var (name, help) = LayoutModes.Summary(Scene(3), ViewportMode.SectionYZ, surfaceCount: 8, stretch: 5, selectedSurface: 2);
        Assert.Equal("Layout, meridional section, 6 surfaces, 3 fields", name);
        Assert.Equal("1 ray, 60 millimeters long, vertical stretch 5 times, surface 2 selected", help);

        (name, help) = LayoutModes.Summary(Scene(1), ViewportMode.Orbit3D, surfaceCount: 3, stretch: 5, stale: true, preview: "trial 9");
        Assert.Equal("Layout, 3D view, 1 surface, 1 field", name);
        Assert.Equal("2 rays, 60 millimeters long, previewing trial 9, not applied, out of date", help);

        Assert.Equal(("Layout, not available", "No Layout. The ray trace failed."),
            LayoutModes.Summary(null, ViewportMode.SectionYZ, 8, state: "No Layout. The ray trace failed."));
    }

    [Fact]
    public void PlaceholdersFollowTheMacApp()
    {
        Assert.Null(LayoutModes.Placeholder(true, null, "", null, false, null));
        // A viewport that cannot draw wins, scene or not, and offers to try again.
        var unavailable = LayoutModes.Placeholder(true, "No device", "", null, false, null)!;
        Assert.Equal("Viewport Unavailable", unavailable.Title);
        Assert.True(unavailable.CanRetry);
        Assert.Equal("Layout Not Available", LayoutModes.Placeholder(false, null, "unavailable", "why", false, null)!.Title);
        Assert.Equal("Design Has Errors", LayoutModes.Placeholder(false, null, "failed", "why", true, null)!.Title);
        Assert.Equal("No Layout", LayoutModes.Placeholder(false, null, "failed", "why", false, null)!.Title);
        Assert.Equal("Design Not Ready", LayoutModes.Placeholder(false, null, "", null, false, "No surfaces")!.Title);
        Assert.True(LayoutModes.Placeholder(false, null, "", null, false, null)!.Busy);
    }
}
