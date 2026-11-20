using MKLight.Engine;

namespace MKLight.App.Views.Viewport3D;

/// <summary>
/// Beam footprints on every surface for the 3-D view, read from the engine's <c>footprint</c>
/// overlays (docs/RESULTS.md): one convex outline per (surface, field, wavelength), as global
/// points (<c>global_outlines</c>).
/// </summary>
public sealed class FootprintOverlay
{
    public sealed record Outline(int Surface, int Field, int Wavelength, IReadOnlyList<Point3> Points);

    public required IReadOnlyList<Outline> Outlines { get; init; }

    /// <summary>The footprint analysis's settings for the overlay (hexapolar, 6 rings, the primary wavelength).</summary>
    public static Dictionary<string, System.Text.Json.Nodes.JsonNode?> Settings() => new()
    {
        ["pattern"] = "hexapolar",
        ["rings"] = 6,
        ["wavelengths"] = new System.Text.Json.Nodes.JsonObject { ["type"] = "primary" },
    };

    /// <summary><c>null</c> when the output has no overlays.</summary>
    public static FootprintOverlay? FromOutput(AnalysisOutput? output)
    {
        if (output?.Json.At("overlays").ArrayOrNull() is not { Count: > 0 } overlays
            || output.Buffer("global_outlines") is not { } buf || buf.Shape.Count == 0 || buf.Shape[^1] != 3) return null;
        var v = buf.ToDoubleArray();
        var rows = buf.Rows;
        var list = new List<Outline>();
        foreach (var o in overlays)
        {
            if (o.At("outline").IntArray() is not { Length: 2 } range || range[1] < 3) continue;
            long start = range[0], n = range[1];
            if (start < 0 || start + n > rows) continue;
            var pts = new Point3[n];
            var finite = true;
            for (long k = 0; k < n; k++)
            {
                var b = (start + k) * 3;
                pts[k] = new Point3(v[b], v[b + 1], v[b + 2]);
                finite &= double.IsFinite(pts[k].X) && double.IsFinite(pts[k].Y) && double.IsFinite(pts[k].Z);
            }
            if (!finite) continue;
            list.Add(new Outline((int)(o.At("surface").IntOrNull() ?? 0), (int)(o.At("field").IntOrNull() ?? 0),
                (int)(o.At("wavelength").IntOrNull() ?? 0), pts));
        }
        return list.Count == 0 ? null : new FootprintOverlay { Outlines = list };
    }
}
