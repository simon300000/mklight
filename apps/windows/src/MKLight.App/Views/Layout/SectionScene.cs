using System.Numerics;

namespace MKLight.App.Views.Layout;

public enum SectionKind { SurfaceProfile, SurfaceRim, ElementEdge, Stop, ImagePlane, Ray, Axis }

/// <summary>A polyline of the 2-D section, in mm: X along the axis (z), Y up (y, or x in the x–z section).</summary>
public sealed record SectionLine(SectionKind Kind, Vector2[] Points, int? Surface, int? Field, int? Wavelength, bool Vignetted, bool Doe);

/// <summary>A lens element's glass fill: front and back profiles sampled at matching heights.</summary>
public sealed record SectionElement(int FirstSurface, int LastSurface, Vector2[] Front, Vector2[] Back);

/// <summary>
/// The layout's 2-D section view: the engine's in-section polylines projected
/// onto (z, up). Built from the engine's layout result by
/// <see cref="EngineLayout"/>; the app only projects and draws.
/// </summary>
public sealed class SectionScene
{
    public required IReadOnlyList<SectionLine> Lines { get; init; }
    public required IReadOnlyList<SectionElement> Elements { get; init; }
    public required IReadOnlyList<double> WavelengthsUm { get; init; }
    public int FieldCount { get; init; } = 1;
    /// <summary>"yz" or "xz".</summary>
    public string Section { get; init; } = "yz";

    /// <summary>Bounds of everything drawn (mm).</summary>
    public (Vector2 Min, Vector2 Max) Bounds(bool lensOnly = false)
    {
        var min = new Vector2(float.PositiveInfinity);
        var max = new Vector2(float.NegativeInfinity);
        foreach (var l in Lines)
        {
            if (lensOnly && l.Kind is SectionKind.Ray or SectionKind.Axis) continue;
            foreach (var p in l.Points)
            {
                if (!float.IsFinite(p.X) || !float.IsFinite(p.Y)) continue;
                min = Vector2.Min(min, p); max = Vector2.Max(max, p);
            }
        }
        if (!float.IsFinite(min.X)) return (new Vector2(-1), new Vector2(1));
        return (min, max);
    }
}
