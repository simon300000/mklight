using System.Numerics;
using System.Text.Json.Nodes;
using MKLight.App.Views.Layout;
using MKLight.App.Views.Viewport3D;
using MKLight.Engine;
using Kind = MKLight.Engine.LayoutScene.PolylineKind;

namespace MKLight.Tests.Viewport;

public class SceneGeometryTests
{
    static readonly Vector3 Accent = new(0, 0, 1);

    static ViewportStyle Style(ViewportMode mode, int? selected = null, FootprintOverlay? footprints = null, float scale = 1) => new(
        mode, Dark: false, FieldRgb: i => new Vector3(i, 0, 0), WavelengthRgb: um => new Vector3(0, (float)um, 0),
        DoeRgb: new Vector3(0.5f), Accent: Accent, SelectedSurface: selected, Scale: scale, Footprints: footprints);

    /// <summary>A two-surface lens with a ray, a rim (3-D only), the axis and one element.</summary>
    static LayoutScene Scene(IReadOnlyList<LayoutScene.MeshTriangle>? mesh = null)
    {
        Point3[] front = [new(0, -5, 0), new(0, 0, -0.5), new(0, 5, 0)];
        Point3[] back = [new(0, -5, 4), new(0, 0, 4.5), new(0, 5, 4)];
        var lines = new List<LayoutScene.Polyline>
        {
            new(Kind.Ray, [new(0, 2, -10), new(0, 2, 0), new(0, 0, 50)], true, Field: 1, Wavelength: 0),
            new(Kind.SurfaceProfile, front, true, Surface: 1),
            new(Kind.SurfaceProfile, back, true, Surface: 2),
            new(Kind.SurfaceRim, [new(5, 0, 0), new(0, 5, 0), new(-5, 0, 0), new(0, -5, 0), new(5, 0, 0)], false, Surface: 1),
            new(Kind.Axis, [new(0, 0, -10), new(0, 0, 50)], true),
            new(Kind.ImagePlane, [new(0, -1, 50), new(0, 1, 50)], true, Surface: 3),
        };
        return new LayoutScene
        {
            Polylines = lines,
            Elements = [new LayoutScene.Element(1, 2, front, back)],
            Surfaces = [],
            Bounds = Bounds3.Enclosing(lines.SelectMany(l => l.Points)),
            WavelengthsUm = [0.55],
            Mesh = mesh ?? [],
        };
    }

    [Fact]
    public void SectionShowsInSectionLinesAndGlassFills()
    {
        var (segs, fills) = SceneGeometry.Build(Scene(), Style(ViewportMode.SectionYZ));
        // ray 2 + profiles 2 + 2 + axis 1 + image 1; the rim is 3-D only.
        Assert.Equal(8, segs.Count);
        // Two quads (four triangles) of glass between the profiles.
        Assert.Equal(12, fills.Count);
    }

    [Fact]
    public void ThreeDShowsRimsAndNoGlassFills()
    {
        var (segs, fills) = SceneGeometry.Build(Scene(), Style(ViewportMode.Orbit3D));
        Assert.Equal(12, segs.Count);
        Assert.Empty(fills);
    }

    [Fact]
    public void RaysFirstSelectedSurfaceLastInTheAccentAndWider()
    {
        var (segs, _) = SceneGeometry.Build(Scene(), Style(ViewportMode.Orbit3D, selected: 1, scale: 2));
        Assert.Equal(new Vector3(1, 0, 0), new Vector3(segs[0].Color.X, segs[0].Color.Y, segs[0].Color.Z)); // field 1
        Assert.Equal(1.1f * 2, segs[0].Params.X, 5);
        var last = segs[^1];
        Assert.Equal(Accent, new Vector3(last.Color.X, last.Color.Y, last.Color.Z));
        // The selected profile (2 segments, 3.6 px) and rim (4 segments, 1.8 px) close the list.
        Assert.Equal(6, segs.Count(s => s.Color.Z == 1 && s.Color.X == 0));
        Assert.Contains(segs, s => s.Params.X == 3.6f * 2);
        // The axis is dashed.
        Assert.Contains(segs, s => s.Params.Y == 10 * 2);
    }

    [Fact]
    public void PositionsAreRelativeToTheSceneCentre()
    {
        var scene = Scene();
        var (segs, _) = SceneGeometry.Build(scene, Style(ViewportMode.SectionYZ));
        var o = SceneGeometry.Origin(scene);
        var ray = segs[0];
        Assert.Equal((float)(0 - o.X), ray.P0.X, 5);
        Assert.Equal((float)(2 - o.Y), ray.P0.Y, 5);
        Assert.Equal((float)(-10 - o.Z), ray.P0.Z, 5);
        Assert.Equal(1, ray.P0.W);
    }

    [Fact]
    public void MeshIsFlatShadedIn3DOnly()
    {
        var mesh = new[]
        {
            new LayoutScene.MeshTriangle(new(0, 0, 0), new(1, 0, 0), new(0, 1, 0), 1),
            new LayoutScene.MeshTriangle(new(0, 0, 0), new(1, 0, 0), new(2, 0, 0), 2), // degenerate: skipped
        };
        var (_, fills3d) = SceneGeometry.Build(Scene(mesh), Style(ViewportMode.Orbit3D, selected: 1));
        Assert.Equal(3, fills3d.Count);
        Assert.Equal(new Vector4(Accent, 0.32f), fills3d[0].Color);
        var (_, fills2d) = SceneGeometry.Build(Scene(mesh), Style(ViewportMode.SectionYZ));
        Assert.Equal(12, fills2d.Count); // glass only
    }

    [Fact]
    public void FootprintsAreOutlinedAndFilledInTheFieldColour()
    {
        var fp = new FootprintOverlay { Outlines = [new FootprintOverlay.Outline(1, 2, 0, [new(1, 0, 0), new(0, 1, 0), new(-1, 0, 0), new(0, -1, 0)])] };
        var (segs, fills) = SceneGeometry.Build(Scene(), Style(ViewportMode.Orbit3D, footprints: fp));
        Assert.Equal(6, fills.Count); // a quad as a fan of two triangles
        Assert.All(fills, f => Assert.Equal(new Vector4(2, 0, 0, 0.16f), f.Color));
        Assert.Equal(12 + 4, segs.Count); // the closed outline
        Assert.Equal(0.9f * 0.35f, segs[0].Color.W, 5); // rays step back
    }

    [Fact]
    public void SegmentClipping()
    {
        var h = (1.0, 1.0);
        Assert.Null(SceneGeometry.Clip(new V3(2, 2, 0), new V3(3, 3, 0), h));
        var (a, b) = SceneGeometry.Clip(new V3(-2, 0, 0), new V3(2, 0, 5), h)!.Value;
        Assert.Equal(new V3(-1, 0, 1.25), a);
        Assert.Equal(new V3(1, 0, 3.75), b);
        var inside = SceneGeometry.Clip(new V3(0.1, 0.2, 0), new V3(-0.3, 0.4, 0), h)!.Value;
        Assert.Equal(new V3(0.1, 0.2, 0), inside.A);
    }

    [Fact]
    public void PolygonClipping()
    {
        var tri = new List<V3> { new(-2, -0.5, 0), new(2, -0.5, 0), new(0, 3, 0) };
        var poly = SceneGeometry.ClipPolygon(tri, (1, 1));
        Assert.True(poly.Count >= 4);
        Assert.All(poly, p => { Assert.InRange(p.X, -1 - 1e-12, 1 + 1e-12); Assert.InRange(p.Y, -1 - 1e-12, 1 + 1e-12); });
        Assert.Empty(SceneGeometry.ClipPolygon([new(5, 5, 0), new(6, 5, 0), new(5, 6, 0)], (1, 1)));
    }

    [Fact]
    public void DeepZoomMovesTheFrameToTheCamera()
    {
        var scene = Scene();
        var near = SceneGeometry.Frame(scene, new ViewportCamera { PointsPerMM = 10 }, ViewportMode.Orbit3D, 800, 600);
        Assert.Null(near.Rotation);
        Assert.Null(near.ClipHalfExtent);
        var cam = new ViewportCamera { Center = new V3(0, 0, 40), PointsPerMM = 1e6 };
        var deep = SceneGeometry.Frame(scene, cam, ViewportMode.Orbit3D, 800, 600);
        Assert.Equal(cam.Center, deep.Origin);
        Assert.NotNull(deep.Rotation);
        Assert.Equal(800 / 1e6, deep.ClipHalfExtent!.Value.X, 15);
        // Everything uploaded is within the clip window, so float keeps sub-pixel precision.
        var (segs, _) = SceneGeometry.Build(scene, Style(ViewportMode.Orbit3D), deep);
        Assert.NotEmpty(segs);
        Assert.All(segs, s => Assert.InRange(Math.Abs(s.P0.X), 0, 800 / 1e6 + 1e-9));
    }

    [Fact]
    public void SlimScenesFitTheirLens()
    {
        var scene = Scene();
        Assert.False(SceneGeometry.IsSlim(scene, ViewportMode.SectionYZ));
        Point3[] longRay = [new(0, 1, 0), new(0, 0, 2000)];
        var slim = new LayoutScene
        {
            Polylines = [.. scene.Polylines, new LayoutScene.Polyline(Kind.Ray, longRay, true)],
            Elements = scene.Elements, Surfaces = [], Bounds = scene.Bounds,
        };
        Assert.True(SceneGeometry.IsSlim(slim, ViewportMode.SectionYZ));
        var lens = SceneGeometry.FitPoints(slim, ViewportMode.SectionYZ, lensOnly: true);
        Assert.True(lens.Max(p => p.Z) < 100);
        // The x–z section pads upward along x.
        var xz = SceneGeometry.FitPoints(scene, ViewportMode.SectionXZ, lensOnly: true);
        Assert.Contains(xz, p => p.X != 0);
    }

    [Fact]
    public void FootprintOverlayReadsGlobalOutlines()
    {
        double[] pts = [0, 0, 1, 1, 0, 1, 0, 1, 1, double.NaN, 0, 0, 1, 1, 1, 2, 2, 2];
        var json = new JsonObject
        {
            ["overlays"] = new JsonArray(
                new JsonObject { ["surface"] = 3, ["field"] = 1, ["wavelength"] = 0, ["outline"] = new JsonArray(0, 3) },
                new JsonObject { ["surface"] = 4, ["field"] = 0, ["outline"] = new JsonArray(2, 3) },   // has a NaN: skipped
                new JsonObject { ["surface"] = 5, ["field"] = 0, ["outline"] = new JsonArray(4, 3) }),  // out of range: skipped
        };
        var output = new AnalysisOutput(new RunInfo
        {
            EngineVersion = "1", SnapshotHash = new SnapshotHash("aa"), Analysis = "footprint", Settings = new JsonObject(),
            Backend = "cpu_reference", Precision = Precision.F64Reference,
        }, json, [ResultBuffer.FromDoubles("global_outlines", pts, [6, 3])]);
        var overlay = FootprintOverlay.FromOutput(output);
        Assert.NotNull(overlay);
        var o = Assert.Single(overlay.Outlines);
        Assert.Equal((3, 1), (o.Surface, o.Field));
        Assert.Equal(new Point3(0, 1, 1), o.Points[2]);
        Assert.Null(FootprintOverlay.FromOutput(new AnalysisOutput(output.RunInfo, new JsonObject())));
    }
}
