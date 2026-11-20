using System.Numerics;
using MKLight.Engine;

namespace MKLight.App.Views.Layout;

/// <summary>
/// Projects the engine's layout scene (read by <see cref="LayoutScene.FromOutput"/>)
/// onto its 2-D section: the in-section polylines and the elements' glass
/// fills, as (z, up) with up = y in the y–z section and x in the x–z section.
/// </summary>
public static class EngineLayout
{
    public static SectionScene Section(LayoutScene scene)
    {
        Func<Point3, float> up = scene.Section == "xz" ? p => (float)p.X : p => (float)p.Y;
        Vector2 P(Point3 p) => new((float)p.Z, up(p));
        var lines = scene.Polylines
            .Where(l => l.InSection && l.Kind != LayoutScene.PolylineKind.SurfaceRim)
            .Select(l => new SectionLine(Kind(l.Kind), l.Points.Select(P).ToArray(), l.Surface, l.Field, l.Wavelength, l.Vignetted, l.Doe))
            .ToList();
        var elements = scene.Elements
            .Select(e => new SectionElement(e.FirstSurface, e.LastSurface, e.Front.Select(P).ToArray(), e.Back.Select(P).ToArray()))
            .ToList();
        return new SectionScene
        {
            Lines = lines, Elements = elements, WavelengthsUm = scene.WavelengthsUm,
            FieldCount = scene.FieldCount, Section = scene.Section,
        };
    }

    static SectionKind Kind(LayoutScene.PolylineKind k) => k switch
    {
        LayoutScene.PolylineKind.SurfaceProfile => SectionKind.SurfaceProfile,
        LayoutScene.PolylineKind.SurfaceRim => SectionKind.SurfaceRim,
        LayoutScene.PolylineKind.Stop => SectionKind.Stop,
        LayoutScene.PolylineKind.ImagePlane => SectionKind.ImagePlane,
        LayoutScene.PolylineKind.Ray => SectionKind.Ray,
        LayoutScene.PolylineKind.Axis => SectionKind.Axis,
        _ => SectionKind.ElementEdge,
    };
}
