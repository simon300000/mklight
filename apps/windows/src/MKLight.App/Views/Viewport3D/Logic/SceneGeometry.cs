using System.Numerics;
using System.Runtime.InteropServices;
using MKLight.App.Views.Layout;
using MKLight.Engine;
using Kind = MKLight.Engine.LayoutScene.PolylineKind;

namespace MKLight.App.Views.Viewport3D;

/// <summary>GPU instance data for one polyline segment (matches <c>Segment</c> in the shader: four float4, 64 bytes).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct SegmentInstance
{
    public Vector4 P0;
    public Vector4 P1;
    /// <summary>Straight (non-premultiplied) RGBA.</summary>
    public Vector4 Color;
    /// <summary>x: width (pixels), y: dash period (pixels, 0 = solid).</summary>
    public Vector4 Params;
}

/// <summary>Matches <c>FillVertex</c> in the shader.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct FillVertex
{
    public Vector4 Position;
    public Vector4 Color;
}

/// <summary>
/// How the scene is drawn: colours, widths and what is visible. Colours are sRGB in 0…1;
/// the palette lookups are passed in so this stays free of UI types.
/// </summary>
public sealed record ViewportStyle(
    ViewportMode Mode,
    bool Dark,
    Func<int, Vector3> FieldRgb,
    Func<double, Vector3> WavelengthRgb,
    Vector3 DoeRgb,
    Vector3 Accent,
    RayColoring Coloring = RayColoring.Field,
    int? SelectedSurface = null,
    bool Stale = false,
    /// <summary>Device pixels per point.</summary>
    float Scale = 1,
    /// <summary>Beam footprints drawn on the surfaces in 3-D.</summary>
    FootprintOverlay? Footprints = null);

/// <summary>Converts an engine <see cref="LayoutScene"/> into GPU buffers in a <see cref="GeometryFrame"/> (the Mac app's SceneGeometry).</summary>
public static class SceneGeometry
{
    public static V3 Origin(LayoutScene scene) => V3.Of(scene.Bounds.Center);

    /// <summary>Largest projected float error (points) accepted before switching to a camera-relative, clipped frame.</summary>
    public const double MaxFloatErrorPoints = 0.02;

    /// <summary>
    /// The frame to build <paramref name="scene"/> in for this camera: scene-centred (camera moves
    /// only change the GPU matrix) while the float error is negligible, else camera-relative,
    /// view-aligned and clipped (rebuilt whenever the camera changes).
    /// </summary>
    public static GeometryFrame Frame(LayoutScene scene, in ViewportCamera camera, ViewportMode mode, double width, double height)
    {
        var o = Origin(scene);
        var reach = SceneReach(scene, o) + (camera.Center - o).Length;
        if (reach * camera.PointsPerMM * Math.Pow(2, -24) <= MaxFloatErrorPoints) return new GeometryFrame(o, null, null);
        var half = (Math.Max(width, 1) / camera.PointsPerMM, Math.Max(height, 1) / camera.VerticalScale(mode) / camera.PointsPerMM);
        return new GeometryFrame(camera.Center, ViewportMath.Rotation(camera, mode), half);
    }

    /// <summary>Distance from <paramref name="o"/> to the farthest corner of the scene bounds.</summary>
    public static double SceneReach(LayoutScene scene, V3 o)
    {
        var b = scene.Bounds;
        var lo = V3.Of(b.Min) - o;
        var hi = V3.Of(b.Max) - o;
        return V3.Max(lo.Abs(), hi.Abs()).Length;
    }

    /// <summary>Clips segment a–b to |x| ≤ h.X, |y| ≤ h.Y (Liang–Barsky); depth is not clipped.</summary>
    public static (V3 A, V3 B)? Clip(V3 a, V3 b, (double X, double Y) h)
    {
        var d = b - a;
        double t0 = 0, t1 = 1;
        Span<(double P, double Q)> planes = [(-d.X, a.X + h.X), (d.X, h.X - a.X), (-d.Y, a.Y + h.Y), (d.Y, h.Y - a.Y)];
        foreach (var (p, q) in planes)
        {
            if (p == 0)
            {
                if (q < 0) return null;
                continue;
            }
            var r = q / p;
            if (p < 0)
            {
                if (r > t1) return null;
                t0 = Math.Max(t0, r);
            }
            else
            {
                if (r < t0) return null;
                t1 = Math.Min(t1, r);
            }
        }
        return (t0 == 0 ? a : a + t0 * d, t1 == 1 ? b : a + t1 * d);
    }

    /// <summary>Clips a convex polygon to the same window (Sutherland–Hodgman).</summary>
    public static List<V3> ClipPolygon(IReadOnlyList<V3> polygon, (double X, double Y) h)
    {
        var poly = polygon.ToList();
        Func<V3, double>[] planes = [p => h.X - p.X, p => h.X + p.X, p => h.Y - p.Y, p => h.Y + p.Y];
        foreach (var inside in planes)
        {
            if (poly.Count == 0) break;
            var output = new List<V3>(poly.Count + 2);
            var prev = poly[^1];
            var dPrev = inside(prev);
            foreach (var cur in poly)
            {
                var dCur = inside(cur);
                if ((dCur >= 0) != (dPrev >= 0)) output.Add(prev + dPrev / (dPrev - dCur) * (cur - prev));
                if (dCur >= 0) output.Add(cur);
                prev = cur;
                dPrev = dCur;
            }
            poly = output;
        }
        return poly;
    }

    /// <summary>Scene-centred build (world axes, no clipping).</summary>
    public static (List<SegmentInstance> Segments, List<FillVertex> Fills) Build(LayoutScene scene, ViewportStyle style) =>
        Build(scene, style, new GeometryFrame(Origin(scene), null, null));

    public static (List<SegmentInstance> Segments, List<FillVertex> Fills) Build(LayoutScene scene, ViewportStyle style, GeometryFrame frame)
    {
        var ink = style.Dark ? new Vector3(0.88f, 0.89f, 0.91f) : new Vector3(0.13f, 0.14f, 0.16f);
        var faint = style.Dark ? new Vector3(0.55f, 0.56f, 0.60f) : new Vector3(0.55f, 0.56f, 0.58f);
        var accent = style.Accent;
        var s = style.Scale;
        var orbit = style.Mode == ViewportMode.Orbit3D;

        var segs = new List<SegmentInstance>(scene.Polylines.Sum(l => Math.Max(l.Points.Count - 1, 0)));
        void Add(IReadOnlyList<Point3> points, Vector3 rgb, float alpha, float width, float dash = 0)
        {
            if (points.Count < 2) return;
            var color = new Vector4(rgb, alpha);
            var prm = new Vector4(width * s, dash * s, 0, 0);
            var a = frame.Apply(points[0]);
            for (int k = 1; k < points.Count; k++)
            {
                var b = frame.Apply(points[k]);
                var (p0, p1) = (a, b);
                a = b;
                if (!double.IsFinite(p0.X + p0.Y + p0.Z + p1.X + p1.Y + p1.Z)) continue;
                if (frame.ClipHalfExtent is { } h)
                {
                    if (Clip(p0, p1, h) is not { } c) continue;
                    (p0, p1) = c;
                }
                segs.Add(new SegmentInstance { P0 = p0.ToFloat4(), P1 = p1.ToFloat4(), Color = color, Params = prm });
            }
        }

        // Rays first so surfaces draw on top of them.
        foreach (var line in scene.Polylines)
        {
            if (line.Kind != Kind.Ray || !(orbit || line.InSection)) continue;
            var rgb = style.Coloring == RayColoring.Field
                ? style.FieldRgb(line.Field ?? 0)
                : style.WavelengthRgb(line.Wavelength is int w && w >= 0 && w < scene.WavelengthsUm.Count ? scene.WavelengthsUm[w] : 0.55);
            var alpha = line.Vignetted ? 0.35f : 0.9f;
            if (style.Stale) alpha *= 0.35f;
            // Footprints are the subject: the rays step back.
            if (orbit && style.Footprints is not null) alpha *= 0.35f;
            Add(line.Points, rgb, alpha, 1.1f);
        }
        // The selected surface is drawn last, so the element outline (which retraces the front and
        // rear profiles) cannot paint over it.
        bool IsSelected(LayoutScene.Polyline l) => l.Surface is not null && l.Surface == style.SelectedSurface && l.Kind != Kind.ElementEdge;
        var outlines = scene.Polylines.Where(l => l.Kind != Kind.Ray && (orbit || l.InSection)).ToList();
        foreach (var line in outlines.Where(l => !IsSelected(l)).Concat(outlines.Where(IsSelected)))
        {
            var sel = IsSelected(line);
            switch (line.Kind)
            {
                case Kind.Axis: Add(line.Points, faint, 0.8f, 0.8f, 10); break;
                // Edges are tagged with the element's first surface; they are never highlighted.
                case Kind.ElementEdge: Add(line.Points, ink, 1, 1.4f); break;
                case Kind.SurfaceProfile:
                    Add(line.Points, sel ? accent : ink, 1, sel ? 3.6f : 1.4f);
                    // A diffractive surface: a fine violet hatch along its profile.
                    if (line.Doe) Add(line.Points, style.DoeRgb, 0.85f, sel ? 4.2f : 3.4f, 3);
                    break;
                case Kind.SurfaceRim: Add(line.Points, sel ? accent : faint, 0.8f, sel ? 1.8f : 1); break;
                case Kind.Stop: Add(line.Points, sel ? accent : ink, 1, sel ? 2.8f : 2.2f); break;
                case Kind.ImagePlane: Add(line.Points, sel ? accent : ink, 1, sel ? 2.4f : 1.8f); break;
            }
        }

        var fills = new List<FillVertex>();
        void Fill(V3[] tri, Vector4 color)
        {
            IReadOnlyList<V3> poly = frame.ClipHalfExtent is { } h ? ClipPolygon(tri, h) : tri;
            for (int k = 1; k + 1 < poly.Count; k++)
            {
                fills.Add(new FillVertex { Position = poly[0].ToFloat4(), Color = color });
                fills.Add(new FillVertex { Position = poly[k].ToFloat4(), Color = color });
                fills.Add(new FillVertex { Position = poly[k + 1].ToFloat4(), Color = color });
            }
        }

        // Beam footprints (3-D): each surface's convex beam outline, filled translucently and
        // outlined, in the field's colour.
        if (orbit && style.Footprints is { } fp)
        {
            foreach (var o in fp.Outlines)
            {
                if (o.Points.Count < 3) continue;
                var rgb = style.FieldRgb(o.Field);
                Add([.. o.Points, o.Points[0]], rgb, style.Stale ? 0.35f : 0.9f, 1.2f);
                var pts = o.Points.Select(frame.Apply).ToArray();
                for (int j = 1; j + 1 < pts.Length; j++) Fill([pts[0], pts[j], pts[j + 1]], new Vector4(rgb, 0.16f));
            }
        }
        // Surface mesh (3-D): the engine's triangulated surfaces, flat shaded so freeform and
        // anamorphic shapes read as solids.
        if (orbit && scene.Mesh.Count > 0)
        {
            var baseRgb = style.Dark ? new V3(0.42, 0.62, 0.95) : new V3(0.40, 0.62, 0.92);
            var light = new V3(-0.35, 0.75, 0.55);
            light /= light.Length;
            foreach (var t in scene.Mesh)
            {
                V3 a = V3.Of(t.A), b = V3.Of(t.B), c = V3.Of(t.C);
                var n = V3.Cross(b - a, c - a);
                var len = n.Length;
                if (!(len > 0)) continue;
                var shade = 0.55 + 0.45 * Math.Abs(V3.Dot(n / len, light));
                var selected = t.Surface == style.SelectedSurface;
                var rgb = selected ? accent : new Vector3((float)(baseRgb.X * shade), (float)(baseRgb.Y * shade), (float)(baseRgb.Z * shade));
                Fill([frame.Apply(t.A), frame.Apply(t.B), frame.Apply(t.C)], new Vector4(rgb, selected ? 0.32f : 0.2f));
            }
        }
        // Glass fill between matched front/back profiles (section views only).
        if (!orbit)
        {
            var glass = style.Dark ? new Vector4(0.35f, 0.60f, 0.95f, 0.22f) : new Vector4(0.45f, 0.68f, 0.95f, 0.25f);
            foreach (var e in scene.Elements)
            {
                if (e.Front.Count != e.Back.Count || e.Front.Count < 2) continue;
                for (int k = 0; k + 1 < e.Front.Count; k++)
                {
                    V3 a = frame.Apply(e.Front[k]), b = frame.Apply(e.Back[k]);
                    V3 c = frame.Apply(e.Front[k + 1]), d = frame.Apply(e.Back[k + 1]);
                    Fill([a, b, c], glass);
                    Fill([c, b, d], glass);
                }
            }
        }
        return (segs, fills);
    }

    /// <summary>
    /// Whether the whole scene is much longer than it is high and the lens a small part of it (a
    /// slow lens far from its image): then the lens deserves the first fit.
    /// </summary>
    public static bool IsSlim(LayoutScene scene, ViewportMode mode)
    {
        static (double Z, double H) Extent(List<V3> p)
        {
            if (p.Count == 0) return (0, 0);
            var h = Math.Max(p.Max(q => Math.Abs(q.Y)), p.Max(q => Math.Abs(q.X)));
            return (p.Max(q => q.Z) - p.Min(q => q.Z), 2 * h);
        }
        var a = Extent(FitPoints(scene, mode));
        var l = Extent(FitPoints(scene, mode, lensOnly: true));
        return a.H > 0 && a.Z > 12 * a.H && l.Z * 5 < a.Z;
    }

    /// <summary>Points used to fit the camera (only what the mode shows); the lens alone with room for the rays around it.</summary>
    public static List<V3> FitPoints(LayoutScene scene, ViewportMode mode, bool lensOnly = false)
    {
        static bool Lens(LayoutScene.Polyline l) => l.Kind is not (Kind.Ray or Kind.Axis or Kind.ImagePlane);
        var pts = scene.Polylines
            .Where(l => (mode == ViewportMode.Orbit3D || l.InSection) && (!lensOnly || Lens(l)))
            .SelectMany(l => l.Points).Select(V3.Of)
            .Where(p => double.IsFinite(p.X) && double.IsFinite(p.Y) && double.IsFinite(p.Z)).ToList();
        if (pts.Count == 0 && lensOnly) return FitPoints(scene, mode);
        if (lensOnly && pts.Count > 0)
        {
            double z0 = pts.Min(p => p.Z), z1 = pts.Max(p => p.Z);
            var h = pts.Max(p => Math.Max(Math.Abs(p.X), Math.Abs(p.Y)));
            var m = 0.6 * Math.Max(z1 - z0, 2 * h);
            // Above and below along the axis the view draws upwards: x in the x–z section.
            var up = mode == ViewportMode.SectionXZ ? new V3(1.3 * h, 0, 0) : new V3(0, 1.3 * h, 0);
            pts.AddRange([new V3(0, 0, z0 - m), new V3(0, 0, z1 + m), new V3(0, 0, z0) + up, new V3(0, 0, z0) - up]);
        }
        return pts;
    }
}
