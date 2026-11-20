using System.Numerics;
using MKLight.App.Views.Viewport3D;

namespace MKLight.Tests.Viewport;

public class ViewportMathTests
{
    static readonly ViewportMode[] Modes = [ViewportMode.SectionYZ, ViewportMode.SectionXZ, ViewportMode.Orbit3D];

    [Fact]
    public void RotationsAreOrthonormal()
    {
        var cam = new ViewportCamera { Yaw = 0.7, Pitch = -0.4 };
        foreach (var mode in Modes)
        {
            var r = ViewportMath.Rotation(cam, mode);
            var i = r * r.Transpose();
            Assert.Equal(1, i.R0.X, 12); Assert.Equal(1, i.R1.Y, 12); Assert.Equal(1, i.R2.Z, 12);
            Assert.Equal(0, i.R0.Y, 12); Assert.Equal(0, i.R0.Z, 12); Assert.Equal(0, i.R1.Z, 12);
        }
    }

    [Fact]
    public void SectionsPutTheAxisRightAndTheirCoordinateUp()
    {
        var cam = new ViewportCamera();
        var yz = ViewportMath.Rotation(cam, ViewportMode.SectionYZ);
        Assert.Equal(new V3(1, 0, 0), yz * new V3(0, 0, 1));
        Assert.Equal(new V3(0, 1, 0), yz * new V3(0, 1, 0));
        var xz = ViewportMath.Rotation(cam, ViewportMode.SectionXZ);
        Assert.Equal(new V3(1, 0, 0), xz * new V3(0, 0, 1));
        Assert.Equal(new V3(0, 1, 0), xz * new V3(1, 0, 0));
        // At zero orbit angles the 3-D view is the y–z section.
        var flat = ViewportMath.Rotation(new ViewportCamera { Yaw = 0, Pitch = 0 }, ViewportMode.Orbit3D);
        Assert.Equal(yz, flat);
    }

    /// <summary>The GPU matrix and the CPU projection agree, in both frames (scene-centred and camera-relative).</summary>
    [Theory]
    [InlineData(ViewportMode.SectionYZ)]
    [InlineData(ViewportMode.SectionXZ)]
    [InlineData(ViewportMode.Orbit3D)]
    public void ViewProjectionMatchesScreenProjection(ViewportMode mode)
    {
        var cam = new ViewportCamera { Center = new V3(1, -2, 30), PointsPerMM = 7, Yaw = -0.55, Pitch = 0.35, Stretch = 5 };
        double w = 640, h = 480;
        var origin = new V3(0.5, 0.25, 25);
        var frames = new[]
        {
            new GeometryFrame(origin, null, null),
            new GeometryFrame(cam.Center, ViewportMath.Rotation(cam, mode), (100, 100)),
        };
        var world = new V3(3, 4, 40);
        var (sx, sy) = ViewportMath.ToScreen(cam, mode, world, w, h);
        foreach (var frame in frames)
        {
            var m = ViewportMath.ViewProjection(cam, mode, frame, w, h, depthRadius: 100);
            var q = frame.Rotation is { } r ? r * (world - frame.Origin) : world - frame.Origin;
            var clip = ViewportMath.Apply(m, new Vector3((float)q.X, (float)q.Y, (float)q.Z));
            var px = (clip.X / clip.W * 0.5 + 0.5) * w;
            var py = (0.5 - clip.Y / clip.W * 0.5) * h;
            Assert.Equal(sx, px, 2);
            Assert.Equal(sy, py, 2);
            Assert.InRange(clip.Z, 0, 1);
        }
    }

    [Fact]
    public void CameraCentreIsTheViewCentre()
    {
        var cam = new ViewportCamera { Center = new V3(1, 2, 3), PointsPerMM = 3 };
        foreach (var mode in Modes)
        {
            var (x, y) = ViewportMath.ToScreen(cam, mode, cam.Center, 300, 200);
            Assert.Equal(150, x, 9);
            Assert.Equal(100, y, 9);
        }
    }

    [Fact]
    public void FitFramesEveryPointInsideTheUsableBand()
    {
        var pts = new List<V3> { new(-5, -10, 0), new(5, 10, 0), new(0, 0, 80), new(0, 8, 60) };
        foreach (var mode in Modes)
        {
            var cam = ViewportMath.Fit(new ViewportCamera(), mode, pts, 800, 600, top: 50, bottom: 30);
            Assert.True(cam.IsFitted);
            foreach (var p in pts)
            {
                var (x, y) = ViewportMath.ToScreen(cam, mode, p, 800, 600);
                Assert.InRange(x, 0, 800);
                Assert.InRange(y, 50 - 1e-6, 570 + 1e-6);
            }
        }
    }

    [Fact]
    public void ZoomKeepsThePointUnderTheCursor()
    {
        var cam = new ViewportCamera { Center = new V3(0, 1, 20), PointsPerMM = 4, Stretch = 10 };
        foreach (var mode in Modes)
        {
            double w = 600, h = 400, cx = 120, cy = -50; // from the centre, y up
            var before = Unproject(cam, mode, w, h, cx, cy);
            var zoomed = ViewportMath.Zoom(cam, mode, 2.5, cx, cy);
            Assert.Equal(10, zoomed.PointsPerMM, 9);
            var (x, y) = ViewportMath.ToScreen(zoomed, mode, before, w, h);
            Assert.Equal(w / 2 + cx, x, 6);
            Assert.Equal(h / 2 - cy, y, 6);
        }
    }

    [Fact]
    public void PanMovesTheSceneWithThePointer()
    {
        var cam = new ViewportCamera { PointsPerMM = 4, Stretch = 2 };
        foreach (var mode in Modes)
        {
            var p = new V3(1, 2, 3);
            var (x0, y0) = ViewportMath.ToScreen(cam, mode, p, 400, 400);
            var (x1, y1) = ViewportMath.ToScreen(ViewportMath.Pan(cam, mode, 30, 20), mode, p, 400, 400);
            Assert.Equal(x0 + 30, x1, 9);
            Assert.Equal(y0 - 20, y1, 9); // y up: a pan up moves the point up the screen
        }
    }

    [Fact]
    public void OrbitClampsThePitch()
    {
        var c = ViewportMath.Orbit(new ViewportCamera { Pitch = 0 }, 100, -1000);
        Assert.Equal(1.55, c.Pitch);
        Assert.Equal(-0.55 + 1, c.Yaw, 12);
    }

    [Fact]
    public void StretchIsOnlyForSections()
    {
        var c = new ViewportCamera { Stretch = 20 };
        Assert.Equal(20, c.VerticalScale(ViewportMode.SectionXZ));
        Assert.Equal(1, c.VerticalScale(ViewportMode.Orbit3D));
    }

    [Theory]
    [InlineData(100, 600, 1)]  // a sixth of the view: readable
    [InlineData(70, 600, 5)]   // under an eighth: the smallest step that reaches a third (200 / 70 → ×5)
    [InlineData(1, 600, 50)]   // at most ×50
    [InlineData(0, 600, 1)]
    public void AutomaticStretchFollowsTheFirstFitRule(double height, double view, double expected) =>
        Assert.Equal(expected, ViewportCamera.AutomaticStretch(height, view));

    [Theory]
    [InlineData(0.9, 1)]
    [InlineData(2.2, 2)]
    [InlineData(4, 5)]
    [InlineData(8, 10)]
    [InlineData(0.034, 0.02)]
    public void NiceLengths(double target, double expected) => Assert.Equal(expected, ViewportMath.NiceLength(target), 12);

    static V3 Unproject(ViewportCamera cam, ViewportMode mode, double w, double h, double cx, double cy)
    {
        var r = ViewportMath.Rotation(cam, mode);
        var view = new V3(cx / cam.PointsPerMM, cy / (cam.PointsPerMM * cam.VerticalScale(mode)), 0);
        return cam.Center + r.Transpose() * view;
    }
}
