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

    /// <summary>
    /// The canvas' automation name and help text: the name says what is shown ("Layout, meridional
    /// section, 6 surfaces, 3 fields"), the help text the rest (rays, length, stretch, the selected
    /// surface, a stale or previewed scene). Without a scene, the reason (<paramref name="state"/>).
    /// </summary>
    public static (string Name, string HelpText) Summary(LayoutScene? scene, ViewportMode mode, int surfaceCount,
        double stretch = 1, int? selectedSurface = null, bool stale = false, string? preview = null, string? state = null)
    {
        if (scene is null) return ("Layout, not available", state ?? "");
        var view = mode switch { ViewportMode.Orbit3D => "3D view", ViewportMode.SectionXZ => "x–z section", _ => "meridional section" };
        var surfaces = Math.Max(surfaceCount - 2, 0);
        var fields = scene.FieldCount;
        var name = $"Layout, {view}, {surfaces} surface{(surfaces == 1 ? "" : "s")}, {fields} field{(fields == 1 ? "" : "s")}";
        var rays = scene.Polylines.Count(l => l.Kind == LayoutScene.PolylineKind.Ray && (mode == ViewportMode.Orbit3D || l.InSection));
        var parts = new List<string> { $"{rays} ray{(rays == 1 ? "" : "s")}", $"{Format(scene.Bounds.Max.Z - scene.Bounds.Min.Z)} millimeters long" };
        if (mode != ViewportMode.Orbit3D && stretch != 1) parts.Add($"vertical stretch {Format(stretch)} times");
        if (selectedSurface is int s) parts.Add($"surface {s} selected");
        if (preview is not null) parts.Add($"previewing {preview}, not applied");
        if (stale) parts.Add("out of date");
        return (name, string.Join(", ", parts));
    }

    /// <summary>What the layout shows instead of a scene: a titled message, or (<see cref="Busy"/>) a progress ring.</summary>
    public sealed record PlaceholderState(string Title, string Glyph, string Text, bool Busy = false, bool CanRetry = false);

    /// <summary>
    /// The layout's empty states, as the Mac app's: the viewport cannot draw ("Viewport Unavailable"),
    /// the engine has no layout ("Layout Not Available"), the design has errors, the layout failed, the
    /// design cannot be modelled yet ("Design Not Ready"), else a progress ring while the first scene
    /// comes. Null while there is a scene to draw.
    /// </summary>
    /// <param name="phase">The layout pane's phase: <c>unavailable</c>, <c>failed</c> or anything else.</param>
    public static PlaceholderState? Placeholder(bool hasScene, string? viewportError, string phase, string? message,
        bool designHasErrors, string? modelError)
    {
        if (viewportError is not null) return new("Viewport Unavailable", "\uE7BA", viewportError, CanRetry: true);
        if (hasScene) return null;
        return phase switch
        {
            "unavailable" => new("Layout Not Available", "\uE733", message ?? ""),
            "failed" when designHasErrors => new("Design Has Errors", "\uEA39", message ?? ""),
            "failed" => new("No Layout", "\uE7BA", message ?? ""),
            _ when modelError is not null => new("Design Not Ready", "\uEA39", modelError),
            _ => new("", "", "", Busy: true),
        };
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
