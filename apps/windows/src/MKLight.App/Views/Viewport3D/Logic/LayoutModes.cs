using System.Text.Json.Nodes;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Viewport3D;

/// <summary>Which views the layout offers, and how they are named.</summary>
public static class LayoutModes
{
    /// <summary>
    /// Whether the design lacks rotational symmetry, so both sections matter: the engine's
    /// symmetry class (<c>paraxial_xy.symmetry</c> or <c>symmetry</c> of the first-order result)
    /// when it reports one, else the surface kinds (toroid, biconic, XY polynomial, Zernike sag).
    /// </summary>
    public static bool IsAnamorphic(JsonNode? firstOrder, OpticalSystem? system)
    {
        var symmetry = firstOrder.At("paraxial_xy", "symmetry").StringOrNull() ?? firstOrder.At("symmetry").StringOrNull();
        if (symmetry is "rotational" or "meridional" or "none") return symmetry != "rotational";
        return system?.Surfaces.Any(s => s.Shape.Kind is ShapeKind.Toroid or ShapeKind.Biconic or ShapeKind.XyPolynomial or ShapeKind.ZernikeSag) ?? false;
    }

    /// <summary>The modes of the selector: Y–Z, X–Z and 3D without rotational symmetry (or while X–Z is shown), else Section and 3D.</summary>
    public static ViewportMode[] Available(bool anamorphic, ViewportMode current) =>
        anamorphic || current == ViewportMode.SectionXZ
            ? [ViewportMode.SectionYZ, ViewportMode.SectionXZ, ViewportMode.Orbit3D]
            : [ViewportMode.SectionYZ, ViewportMode.Orbit3D];

    public static string Title(ViewportMode mode, int count) => mode switch
    {
        ViewportMode.SectionYZ => count == 3 ? "Y–Z" : "Section",
        ViewportMode.SectionXZ => "X–Z",
        _ => "3D",
    };

    /// <summary>The layout's <c>section</c> setting for a mode: <c>"xz"</c> for the X–Z section, else none (y–z).</summary>
    public static string? SectionSetting(ViewportMode mode) => mode == ViewportMode.SectionXZ ? "xz" : null;

    /// <summary>Whether a scene drawn for <paramref name="section"/> (<c>yz</c> or <c>xz</c>) is the one <paramref name="mode"/> shows.</summary>
    public static bool SceneMatches(ViewportMode mode, string section) => mode switch
    {
        ViewportMode.SectionXZ => section == "xz",
        ViewportMode.SectionYZ => section != "xz",
        _ => true,
    };

    /// <summary>The layout in words: the view, the surfaces, the rays drawn and the scene's length.</summary>
    public static string AccessibilityText(LayoutScene? scene, ViewportMode mode, int surfaceCount)
    {
        if (scene is null) return "Layout, not available";
        var rays = scene.Polylines.Count(l => l.Kind == LayoutScene.PolylineKind.Ray);
        var length = scene.Bounds.Max.Z - scene.Bounds.Min.Z;
        var view = mode switch { ViewportMode.Orbit3D => "3D view", ViewportMode.SectionXZ => "x–z section", _ => "meridional section" };
        return $"Layout, {view}, {Math.Max(surfaceCount - 2, 0)} surfaces, {rays} rays, {Format(length)} millimeters long";
    }

    /// <summary>A length in mm: up to 4 significant digits, decimal point, no grouping, never −0.</summary>
    public static string Format(double mm)
    {
        if (!double.IsFinite(mm)) return "—";
        if (mm == 0) return "0";
        var digits = Math.Max(0, 3 - (int)Math.Floor(Math.Log10(Math.Abs(mm))));
        var r = Math.Round(mm, Math.Min(digits, 12));
        if (r == 0) r = 0; // no −0
        return r.ToString("0.############", System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>A scale-bar length: "20 mm", "0.5 mm".</summary>
    public static string LengthLabel(double mm) => Format(mm) + " mm";
}
