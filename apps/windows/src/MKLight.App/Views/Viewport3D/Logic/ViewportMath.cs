using System.Numerics;
using MKLight.Engine;

namespace MKLight.App.Views.Viewport3D;

/// <summary>How the layout is viewed: the meridional (y–z) section, the x–z section, or orbiting in 3-D.</summary>
public enum ViewportMode { SectionYZ, SectionXZ, Orbit3D }

/// <summary>A double-precision 3-vector (positions are transformed in double before rounding to float).</summary>
public readonly record struct V3(double X, double Y, double Z)
{
    public static V3 Of(Point3 p) => new(p.X, p.Y, p.Z);
    public static V3 operator +(V3 a, V3 b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static V3 operator -(V3 a, V3 b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static V3 operator -(V3 a) => new(-a.X, -a.Y, -a.Z);
    public static V3 operator *(double s, V3 a) => new(s * a.X, s * a.Y, s * a.Z);
    public static V3 operator /(V3 a, double s) => new(a.X / s, a.Y / s, a.Z / s);
    public double Length => Math.Sqrt(X * X + Y * Y + Z * Z);
    public static double Dot(V3 a, V3 b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    public static V3 Cross(V3 a, V3 b) => new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    public static V3 Min(V3 a, V3 b) => new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Min(a.Z, b.Z));
    public static V3 Max(V3 a, V3 b) => new(Math.Max(a.X, b.X), Math.Max(a.Y, b.Y), Math.Max(a.Z, b.Z));
    public V3 Abs() => new(Math.Abs(X), Math.Abs(Y), Math.Abs(Z));
    public Vector4 ToFloat4() => new((float)X, (float)Y, (float)Z, 1);
}

/// <summary>A double-precision 3×3 matrix, stored by rows (<c>M * v</c> takes column vectors).</summary>
public readonly record struct M3(V3 R0, V3 R1, V3 R2)
{
    public static M3 FromColumns(V3 c0, V3 c1, V3 c2) => new(new(c0.X, c1.X, c2.X), new(c0.Y, c1.Y, c2.Y), new(c0.Z, c1.Z, c2.Z));
    public static V3 operator *(M3 m, V3 v) => new(V3.Dot(m.R0, v), V3.Dot(m.R1, v), V3.Dot(m.R2, v));
    public static M3 operator *(M3 a, M3 b)
    {
        var t = b.Transpose();
        return new(new(V3.Dot(a.R0, t.R0), V3.Dot(a.R0, t.R1), V3.Dot(a.R0, t.R2)),
                   new(V3.Dot(a.R1, t.R0), V3.Dot(a.R1, t.R1), V3.Dot(a.R1, t.R2)),
                   new(V3.Dot(a.R2, t.R0), V3.Dot(a.R2, t.R1), V3.Dot(a.R2, t.R2)));
    }
    public M3 Transpose() => FromColumns(R0, R1, R2);
}

/// <summary>
/// The layout camera (the Mac app's <c>ViewportCamera</c>): orthographic, a world point at the
/// centre of the view, a zoom in points (DIPs) per millimetre, orbit angles for the 3-D view and
/// the vertical stretch of the section views.
/// </summary>
public record struct ViewportCamera()
{
    /// <summary>World point (mm) at the centre of the view.</summary>
    public V3 Center = new(0, 0, 0);
    /// <summary>Zoom in points per millimetre.</summary>
    public double PointsPerMM = 4;
    /// <summary>Orbit angles (radians) of the 3-D view.</summary>
    public double Yaw = -0.55;
    public double Pitch = 0.35;
    /// <summary>False until the camera has been fitted to a scene.</summary>
    public bool IsFitted = false;
    /// <summary>Vertical exaggeration of the section views (1 = true proportions); the 3-D view always keeps true proportions.</summary>
    public double Stretch = 1;
    /// <summary>The stretch was chosen (in the menu), so a first fit keeps it instead of choosing one.</summary>
    public bool StretchChosen = false;

    /// <summary>The vertical scale factor in <paramref name="mode"/>.</summary>
    public readonly double VerticalScale(ViewportMode mode) => mode == ViewportMode.Orbit3D ? 1 : Math.Max(Stretch, 1e-3);

    /// <summary>The stretches offered in the layout's controls.</summary>
    public static readonly double[] Stretches = [1, 2, 5, 10, 20, 50];

    /// <summary>
    /// The stretch that makes a section of <paramref name="height"/> points (at true proportions)
    /// readable in a view <paramref name="viewHeight"/> points tall: 1 unless it fills less than an
    /// eighth of the view, else the smallest offered stretch that brings it to a third (at most ×50).
    /// </summary>
    public static double AutomaticStretch(double height, double viewHeight)
    {
        if (!(height > 0) || !(viewHeight > 0) || height >= viewHeight / 8) return 1;
        var want = viewHeight / 3 / height;
        foreach (var s in Stretches) if (s >= want) return s;
        return Stretches[^1];
    }
}

/// <summary>
/// The coordinate frame vertex positions are uploaded in. Positions are transformed in double
/// precision and only then rounded to float. At deep zoom far from the scene centre the frame
/// moves to the camera ("relative to eye"): positions are rotated into view axes on the CPU and
/// clipped to a window a little larger than the view, so every uploaded coordinate stays within
/// about one viewport of the origin and the float error a tiny fraction of a pixel.
/// </summary>
public readonly record struct GeometryFrame(V3 Origin, M3? Rotation, (double X, double Y)? ClipHalfExtent)
{
    public V3 Apply(Point3 p)
    {
        var q = V3.Of(p) - Origin;
        return Rotation is { } r ? r * q : q;
    }
}

/// <summary>
/// View transforms of the layout viewport. Orthographic only: optical layouts are read with true
/// proportions. World → view: world +z (the optical axis) to the right and world +y up, so the
/// 2-D view is the meridional Y–Z section; the X–Z section puts world x up. The 3-D view orbits
/// the Y–Z basis by yaw (about view up) and pitch (about view right).
/// </summary>
public static class ViewportMath
{
    /// <summary>Rotation from world to view axes for <paramref name="mode"/>.</summary>
    public static M3 Rotation(in ViewportCamera camera, ViewportMode mode)
    {
        // Columns are the images of world x, y, z: x → −view z, y → view y, z → view x.
        var basis = M3.FromColumns(new(0, 0, -1), new(0, 1, 0), new(1, 0, 0));
        // The x–z section: world z to the right, world x up, world y towards the viewer.
        if (mode == ViewportMode.SectionXZ) return M3.FromColumns(new(0, 1, 0), new(0, 0, 1), new(1, 0, 0));
        if (mode != ViewportMode.Orbit3D) return basis;
        double cy = Math.Cos(camera.Yaw), sy = Math.Sin(camera.Yaw), cp = Math.Cos(camera.Pitch), sp = Math.Sin(camera.Pitch);
        var yaw = M3.FromColumns(new(cy, 0, -sy), new(0, 1, 0), new(sy, 0, cy));
        var pitch = M3.FromColumns(new(1, 0, 0), new(0, cp, sp), new(0, -sp, cp));
        return pitch * yaw * basis;
    }

    /// <summary>
    /// Matrix taking positions in <paramref name="frame"/> to clip space for a viewport of
    /// <paramref name="width"/> × <paramref name="height"/> points (row-major: clip = M · [p, 1];
    /// M11…M14 is the first row). Built in double precision; depth maps [−D, D] to [1, 0].
    /// </summary>
    public static Matrix4x4 ViewProjection(in ViewportCamera camera, ViewportMode mode, GeometryFrame frame, double width, double height, double depthRadius)
    {
        var rc = Rotation(camera, mode);
        var r = frame.Rotation is { } fr ? rc * fr.Transpose() : rc; // frame axes → view axes
        var t = rc * (frame.Origin - camera.Center);                    // translation in view space
        var sx = 2 * camera.PointsPerMM / Math.Max(width, 1);
        var sy = 2 * camera.PointsPerMM * camera.VerticalScale(mode) / Math.Max(height, 1);
        var sz = 0.5 / Math.Max(depthRadius, 1e-9);
        return new Matrix4x4(
            (float)(sx * r.R0.X), (float)(sx * r.R0.Y), (float)(sx * r.R0.Z), (float)(sx * t.X),
            (float)(sy * r.R1.X), (float)(sy * r.R1.Y), (float)(sy * r.R1.Z), (float)(sy * t.Y),
            (float)(-sz * r.R2.X), (float)(-sz * r.R2.Y), (float)(-sz * r.R2.Z), (float)(0.5 - sz * t.Z),
            0, 0, 0, 1);
    }

    /// <summary>Applies a row-major <see cref="ViewProjection"/> matrix to a point (clip = M · [p, 1]).</summary>
    public static Vector4 Apply(in Matrix4x4 m, Vector3 p) => new(
        m.M11 * p.X + m.M12 * p.Y + m.M13 * p.Z + m.M14,
        m.M21 * p.X + m.M22 * p.Y + m.M23 * p.Z + m.M24,
        m.M31 * p.X + m.M32 * p.Y + m.M33 * p.Z + m.M34,
        m.M41 * p.X + m.M42 * p.Y + m.M43 * p.Z + m.M44);

    /// <summary>
    /// A world point in view points from the top-left corner (y down), computed in double.
    /// </summary>
    public static (double X, double Y) ToScreen(in ViewportCamera camera, ViewportMode mode, V3 world, double width, double height)
    {
        var v = Rotation(camera, mode) * (world - camera.Center);
        return (width / 2 + v.X * camera.PointsPerMM, height / 2 - v.Y * camera.PointsPerMM * camera.VerticalScale(mode));
    }

    /// <summary>
    /// A camera that frames <paramref name="points"/> in a viewport of the given size (points) with
    /// a margin, keeping the orbit angles. <paramref name="top"/> and <paramref name="bottom"/> are
    /// bands the floating controls and the scale bar cover, kept clear.
    /// </summary>
    public static ViewportCamera Fit(ViewportCamera camera, ViewportMode mode, IReadOnlyList<V3> points, double width, double height,
        double top = 0, double bottom = 0)
    {
        if (points.Count == 0 || width <= 1 || height <= 1) return camera;
        var r = Rotation(camera, mode);
        var lo = new V3(double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity);
        var hi = -lo;
        foreach (var p in points)
        {
            var v = r * p;
            if (!double.IsFinite(v.X) || !double.IsFinite(v.Y) || !double.IsFinite(v.Z)) continue;
            lo = V3.Min(lo, v);
            hi = V3.Max(hi, v);
        }
        if (!double.IsFinite(lo.X)) return camera;
        var c = camera;
        var vs = camera.VerticalScale(mode);
        var usable = Math.Max(height - top - bottom, height * 0.4);
        var w = Math.Max(hi.X - lo.X, 1e-6);
        var h = Math.Max(hi.Y - lo.Y, 1e-6) * vs;
        c.PointsPerMM = Math.Min(width / (w * 1.12), usable / (h * 1.12));
        // The centre of the usable band sits (top − bottom) / 2 below the view's centre.
        var mid = (lo + hi) / 2 + new V3(0, (top - bottom) / 2 / (c.PointsPerMM * vs), 0);
        c.Center = r.Transpose() * mid;
        c.IsFitted = true;
        return c;
    }

    /// <summary>Width and height (mm, true proportions) of <paramref name="points"/> in the view.</summary>
    public static (double Width, double Height) Extent(in ViewportCamera camera, ViewportMode mode, IReadOnlyList<V3> points)
    {
        if (points.Count == 0) return (0, 0);
        var r = Rotation(camera, mode);
        var lo = new V3(double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity);
        var hi = -lo;
        foreach (var p in points)
        {
            var v = r * p;
            lo = V3.Min(lo, v);
            hi = V3.Max(hi, v);
        }
        return (hi.X - lo.X, hi.Y - lo.Y);
    }

    /// <summary>Pans by a screen delta in points (y up).</summary>
    public static ViewportCamera Pan(ViewportCamera camera, ViewportMode mode, double dx, double dy)
    {
        var r = Rotation(camera, mode);
        var c = camera;
        c.Center -= r.Transpose() * new V3(dx / c.PointsPerMM, dy / (c.PointsPerMM * c.VerticalScale(mode)), 0);
        return c;
    }

    /// <summary>Zooms by <paramref name="factor"/> keeping the point under the cursor fixed (points from the viewport centre, y up).</summary>
    public static ViewportCamera Zoom(ViewportCamera camera, ViewportMode mode, double factor, double cursorX, double cursorY)
    {
        if (!double.IsFinite(factor) || factor <= 0) return camera;
        var c = camera;
        var newZoom = Math.Clamp(c.PointsPerMM * factor, 1e-4, 1e6);
        var o = new V3(cursorX, cursorY / c.VerticalScale(mode), 0);
        var shift = o / c.PointsPerMM - o / newZoom;
        c.Center += Rotation(camera, mode).Transpose() * shift;
        c.PointsPerMM = newZoom;
        return c;
    }

    /// <summary>
    /// One step of a two-finger touch gesture: zooms by <paramref name="scale"/> about where the
    /// pinch centre was (<paramref name="x"/>, <paramref name="y"/>: its position now, in points
    /// from the top-left corner, y down, less this step's <paramref name="dx"/>, <paramref name="dy"/>),
    /// then pans by the centre's move, so the scene under the fingers stays under them.
    /// </summary>
    public static ViewportCamera Pinch(ViewportCamera camera, ViewportMode mode, double scale, double x, double y, double dx, double dy,
        double width, double height)
    {
        var before = (X: x - dx - width / 2, Y: height / 2 - (y - dy));
        var c = Zoom(camera, mode, double.IsFinite(scale) && scale > 0 ? scale : 1, before.X, before.Y);
        return Pan(c, mode, dx, -dy);
    }

    /// <summary>Orbits by a screen delta in points (y up).</summary>
    public static ViewportCamera Orbit(ViewportCamera camera, double dx, double dy)
    {
        var c = camera;
        c.Yaw += dx * 0.01;
        c.Pitch = Math.Clamp(c.Pitch - dy * 0.01, -1.55, 1.55);
        return c;
    }

    /// <summary>A "nice" length (1-2-5 series) close to <paramref name="target"/> mm.</summary>
    public static double NiceLength(double target)
    {
        if (!(target > 0) || !double.IsFinite(target)) return 1;
        var e = Math.Pow(10, Math.Floor(Math.Log10(target)));
        var m = target / e;
        return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * e;
    }
}
