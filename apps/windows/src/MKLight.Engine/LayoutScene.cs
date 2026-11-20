namespace MKLight.Engine;

/// <summary>A point in global coordinates (mm). Equality treats NaN as equal to NaN, so scenes compare.</summary>
public readonly record struct Point3(double X, double Y, double Z);

public readonly record struct Bounds3(Point3 Min, Point3 Max)
{
    public static readonly Bounds3 Empty = new(default, default);

    public Point3 Center => new((Min.X + Max.X) / 2, (Min.Y + Max.Y) / 2, (Min.Z + Max.Z) / 2);

    /// <summary>Smallest box containing <paramref name="points"/> (<see cref="Empty"/> when there are none).</summary>
    public static Bounds3 Enclosing(IEnumerable<Point3> points)
    {
        double lx = double.PositiveInfinity, ly = lx, lz = lx, hx = double.NegativeInfinity, hy = hx, hz = hx;
        foreach (var p in points)
        {
            lx = Math.Min(lx, p.X); ly = Math.Min(ly, p.Y); lz = Math.Min(lz, p.Z);
            hx = Math.Max(hx, p.X); hy = Math.Max(hy, p.Y); hz = Math.Max(hz, p.Z);
        }
        return double.IsFinite(lx) ? new Bounds3(new(lx, ly, lz), new(hx, hy, hz)) : Empty;
    }
}

/// <summary>
/// Geometry for the layout viewport, entirely computed by the engine: the UI only
/// projects and draws these polylines. <see cref="FromOutput"/> reads the engine's
/// <c>layout</c> result.
/// </summary>
public sealed class LayoutScene
{
    public enum PolylineKind
    {
        /// <summary>A surface's profile curve.</summary>
        SurfaceProfile,
        /// <summary>The clear-aperture rim of a surface (3-D only).</summary>
        SurfaceRim,
        /// <summary>Edge/flange lines closing a lens element.</summary>
        ElementEdge,
        Stop,
        ImagePlane,
        Ray,
        Axis,
    }

    /// <param name="InSection">Lies in the section plane: drawn in the 2-D section view.</param>
    /// <param name="Vignetted">For rays: whether the ray was clipped before the image.</param>
    /// <param name="Doe">Belongs to a diffractive (DOE) surface (layout flag bit 2).</param>
    public sealed record Polyline(
        PolylineKind Kind, IReadOnlyList<Point3> Points, bool InSection, int? Surface = null, int? Field = null,
        int? Wavelength = null, bool Vignetted = false, bool Doe = false);

    /// <summary>A lens element's section as two profiles sampled at matching heights (Front[i] and Back[i] share a height), for filling.</summary>
    public sealed record Element(int FirstSurface, int LastSurface, IReadOnlyList<Point3> Front, IReadOnlyList<Point3> Back);

    /// <param name="Id">The surface's id (the model's SurfaceID).</param>
    /// <param name="Vertex">Global vertex position.</param>
    /// <param name="SemiDiameter">Clear semi-diameter used for drawing (fixed, or computed from rays).</param>
    public sealed record SurfaceInfo(int Index, ulong Id, Point3 Vertex, double SemiDiameter, bool SemiDiameterIsAuto, bool IsDiffractive = false);

    /// <summary>One triangle of the surface mesh, in global coordinates.</summary>
    public readonly record struct MeshTriangle(Point3 A, Point3 B, Point3 C, int Surface);

    public required IReadOnlyList<Polyline> Polylines { get; init; }
    public required IReadOnlyList<Element> Elements { get; init; }
    public required IReadOnlyList<SurfaceInfo> Surfaces { get; init; }
    public required Bounds3 Bounds { get; init; }
    public IReadOnlyList<double> WavelengthsUm { get; init; } = [];
    public int FieldCount { get; init; } = 1;
    /// <summary>The 2-D section the in-section polylines lie in: <c>yz</c> or <c>xz</c>.</summary>
    public string Section { get; init; } = "yz";
    /// <summary>The optional triangulated surfaces (<c>layout</c> with <c>mesh_rings</c>).</summary>
    public IReadOnlyList<MeshTriangle> Mesh { get; init; } = [];

    static readonly Dictionary<string, PolylineKind> KindsByName = new()
    {
        ["surface_profile"] = PolylineKind.SurfaceProfile, ["surface_rim"] = PolylineKind.SurfaceRim,
        ["element_edge"] = PolylineKind.ElementEdge, ["stop"] = PolylineKind.Stop, ["image_plane"] = PolylineKind.ImagePlane,
        ["ray"] = PolylineKind.Ray, ["axis"] = PolylineKind.Axis, ["element_outline"] = PolylineKind.ElementEdge,
    };

    /// <summary>
    /// Reads the engine's <c>layout</c> result (docs/RESULTS.md, docs/ANALYSIS.md §2.5:
    /// polylines in typed buffers, global frame). Glass fills are built from each element's
    /// front and back profiles in the section. <c>null</c> when the output is not a layout
    /// this reader understands.
    /// </summary>
    public static LayoutScene? FromOutput(AnalysisOutput? output)
    {
        if (output is null) return null;
        var json = output.Json;
        var precise = output.Buffer("vertices_f64");
        var verts = precise ?? output.Buffer("vertices");
        if (output.Buffer("offsets") is not { } offsets || output.Buffer("kinds") is not { } kinds
            || verts is null || verts.Shape.Count == 0 || verts.Shape[^1] != 3) return null;
        var origin = json.At("origin").NfArray() ?? [0, 0, 0];
        double[] shift = precise is null && origin.Length == 3 ? origin : [0, 0, 0];
        var surf = output.Buffer("surfaces");
        var fld = output.Buffer("fields");
        var wav = output.Buffer("wavelengths");
        var flags = output.Buffer("flags");
        int n = kinds.Count;
        if (offsets.Count != n + 1 || verts.Count != verts.Rows * 3 || (flags is not null && flags.Count != n)) return null;
        var names = (json.At("kind_names").ArrayOrNull() ?? []).Select(k => k.StringOrNull()).OfType<string>().ToArray();
        var v = verts.ToDoubleArray();
        var polylines = new List<Polyline>(n);
        for (int i = 0; i < n; i++)
        {
            if (Exact(offsets[i]) is not { } a || Exact(offsets[i + 1]) is not { } b || a < 0 || b > verts.Rows || a > b) continue;
            var pts = new Point3[b - a];
            for (int k = a; k < b; k++) pts[k - a] = new Point3(v[k * 3] + shift[0], v[k * 3 + 1] + shift[1], v[k * 3 + 2] + shift[2]);
            int f = flags is null ? 1 : Exact(flags[i]) ?? 0;
            int? Index(ResultBuffer? buf) => buf is not null && i < buf.Count && Exact(buf[i]) is { } x && x >= 0 ? x : null;
            var name = Exact(kinds[i]) is { } kind && kind >= 0 && kind < names.Length ? names[kind] : null;
            polylines.Add(new Polyline(
                name is not null && KindsByName.TryGetValue(name, out var pk) ? pk : PolylineKind.ElementEdge, pts,
                (f & 1) != 0, Index(surf), Index(fld), Index(wav), (f & 2) != 0, (f & 4) != 0));
        }
        var section = json.At("section").StringOrNull() == "xz" ? "xz" : "yz";
        // The section's "up" coordinate: y for the y–z section, x for x–z.
        Func<Point3, double> up = section == "xz" ? p => p.X : p => p.Y;
        MarkNearSection(polylines, section);
        var surfaces = (json.At("surfaces").ArrayOrNull() ?? []).Select(s =>
        {
            if (s.At("index").IntOrNull() is not { } index || s.At("id").IntOrNull() is not { } id || id < 0
                || s.At("vertex").NfArray() is not { Length: 3 } vx) return null;
            return new SurfaceInfo(
                (int)index, (ulong)id, new Point3(vx[0], vx[1], vx[2]), s.NfAt("semi_diameter") ?? double.NaN,
                s.At("semi_diameter_is_auto").BoolOrNull() ?? false, s.At("is_diffractive").BoolOrNull() ?? false);
        }).OfType<SurfaceInfo>().ToArray();
        // Glass fills: the in-section profiles of each element's first and last surface,
        // ordered by height so their samples pair up.
        Point3[]? Profile(long surface) => polylines
            .FirstOrDefault(p => p.Kind == PolylineKind.SurfaceProfile && p.InSection && p.Surface == surface)?.Points
            .OrderBy(up).ToArray();
        var elements = (json.At("elements").ArrayOrNull() ?? []).Select(e =>
            e.At("first_surface").IntOrNull() is { } first && e.At("last_surface").IntOrNull() is { } last
            && Profile(first) is { } front && Profile(last) is { } back
                ? new Element((int)first, (int)last, front, Resample(back, front, up)) : null).OfType<Element>().ToArray();
        var lo = json.At("bounds", "min").NfArray() ?? [];
        var hi = json.At("bounds", "max").NfArray() ?? [];
        Bounds3 bounds;
        if (lo.Length == 3 && hi.Length == 3 && lo.Concat(hi).All(double.IsFinite))
        {
            bounds = new Bounds3(new(lo[0], lo[1], lo[2]), new(hi[0], hi[1], hi[2]));
        }
        else
        {
            var all = polylines.SelectMany(p => p.Points).ToArray();
            if (all.Length == 0) return null;
            bounds = new Bounds3(
                new(all.Min(p => p.X), all.Min(p => p.Y), all.Min(p => p.Z)), new(all.Max(p => p.X), all.Max(p => p.Y), all.Max(p => p.Z)));
        }
        return new LayoutScene
        {
            Polylines = polylines,
            Elements = elements,
            Surfaces = surfaces,
            Bounds = bounds,
            WavelengthsUm = json.At("wavelengths_um").NfArray() ?? [],
            FieldCount = (int)(json.At("field_count").IntOrNull() ?? 1),
            Section = section,
            Mesh = MeshOf(output, origin),
        };
    }

    static int? Exact(double d) => d == Math.Floor(d) && d >= int.MinValue && d <= int.MaxValue ? (int)d : null;

    /// <summary>
    /// A slightly decentred or tilted system (a toleranced build) has no polyline exactly
    /// in the section plane, so the engine flags none and the section view would be empty.
    /// Then profiles, stops, edges and rays that stay within 2 % of the scene's height of
    /// the plane count as in the section (rims and skew rays stay out).
    /// </summary>
    public static void MarkNearSection(List<Polyline> polylines, string section)
    {
        static bool Lens(PolylineKind k) => k is PolylineKind.SurfaceProfile or PolylineKind.Stop or PolylineKind.ElementEdge or PolylineKind.Ray;
        if (polylines.Any(p => p.InSection && Lens(p.Kind))) return;
        Func<Point3, double> off = section == "xz" ? p => Math.Abs(p.Y) : p => Math.Abs(p.X);
        Func<Point3, double> up = section == "xz" ? p => Math.Abs(p.X) : p => Math.Abs(p.Y);
        var height = polylines.SelectMany(p => p.Points).Select(up).Where(double.IsFinite).DefaultIfEmpty(0).Max();
        if (!(height > 0)) return;
        var tol = 0.02 * height;
        for (int i = 0; i < polylines.Count; i++)
        {
            var p = polylines[i];
            if (!p.InSection && (Lens(p.Kind) || p.Kind == PolylineKind.Axis) && p.Points.All(q => off(q) <= tol))
            {
                polylines[i] = p with { InSection = true };
            }
        }
    }

    /// <summary>The optional surface mesh: <c>surface_mesh</c> f32 <c>[T, 3, 3]</c> relative to <c>origin</c>, <c>mesh_surfaces</c> u32 <c>[T]</c>.</summary>
    static MeshTriangle[] MeshOf(AnalysisOutput output, double[] origin)
    {
        if (output.Buffer("surface_mesh") is not { Shape: [var t, 3, 3] } m) return [];
        var ids = output.Buffer("mesh_surfaces");
        double[] o = origin.Length == 3 ? origin : [0, 0, 0];
        if (t <= 0 || m.Count < t * 9) return [];
        var v = m.ToDoubleArray();
        var triangles = new MeshTriangle[t];
        for (int k = 0; k < t; k++)
        {
            Point3 P(int c)
            {
                int b = k * 9 + c * 3;
                return new Point3(v[b] + o[0], v[b + 1] + o[1], v[b + 2] + o[2]);
            }
            int surface = ids is not null && k < ids.Count && Exact(ids[k]) is { } s ? s : -1;
            triangles[k] = new MeshTriangle(P(0), P(1), P(2), surface);
        }
        return triangles;
    }

    /// <summary><paramref name="points"/> resampled to the heights of <paramref name="like"/> (linear), so a fill can pair them; heights outside its range clamp to the ends.</summary>
    public static Point3[] Resample(IReadOnlyList<Point3> points, IReadOnlyList<Point3> like, Func<Point3, double>? up = null)
    {
        up ??= p => p.Y;
        if (points.Count < 2) return like.Select(_ => points.Count > 0 ? points[0] : default).ToArray();
        double lo = up(points[0]), hi = up(points[^1]);
        return like.Select(q =>
        {
            var y = Math.Min(Math.Max(up(q), lo), hi);
            int k = 0;
            while (k < points.Count - 2 && up(points[k + 1]) < y) k++;
            Point3 a = points[k], b = points[k + 1];
            var t = up(b) == up(a) ? 0 : (y - up(a)) / (up(b) - up(a));
            return new Point3(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t, a.Z + (b.Z - a.Z) * t);
        }).ToArray();
    }

    /// <summary>Automatic semi-diameters the engine drew, by surface id.</summary>
    public static IReadOnlyDictionary<ulong, double> AutoSemiDiameters(LayoutScene? scene)
    {
        var map = new Dictionary<ulong, double>();
        foreach (var s in scene?.Surfaces ?? [])
        {
            if (double.IsFinite(s.SemiDiameter)) map.TryAdd(s.Id, s.SemiDiameter);
        }
        return map;
    }
}
