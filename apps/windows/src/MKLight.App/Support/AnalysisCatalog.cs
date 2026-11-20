using System.Globalization;

namespace MKLight.App.Support;

/// <summary>
/// Presentation of engine analyses: short titles, Segoe Fluent glyphs and
/// category names (the Mac app's AnalysisCatalog). Analyses the app does not
/// know keep the engine's title and a generic glyph, so new analyses appear
/// without code changes.
/// </summary>
public static class AnalysisCatalog
{
    static readonly Dictionary<string, string> Titles = new()
    {
        ["first_order"] = "First-Order Data", ["system_summary"] = "System Data", ["validate"] = "Validation",
        ["spot_diagram"] = "Spot Diagram", ["ray_fan"] = "Ray Fan", ["opd_fan"] = "OPD Fan",
        ["field_curvature_distortion"] = "Field Curvature & Distortion", ["grid_distortion"] = "Grid Distortion",
        ["seidel"] = "Seidel Coefficients", ["longitudinal_aberration"] = "Longitudinal Aberration",
        ["chromatic_focal_shift"] = "Chromatic Focal Shift", ["lateral_colour"] = "Lateral Color",
        ["relative_illumination"] = "Relative Illumination", ["footprint"] = "Footprint", ["encircled_energy"] = "Encircled Energy",
        ["geometric_mtf"] = "Geometric MTF", ["best_focus"] = "Best Focus", ["ray_grid"] = "Ray Grid", ["layout"] = "Layout",
        ["wavefront_map"] = "Wavefront Map", ["psf"] = "PSF", ["mtf"] = "MTF", ["mtf_vs_field"] = "MTF vs Field",
        ["through_focus_mtf"] = "Through-Focus MTF", ["diffraction_encircled_energy"] = "Diffraction Encircled Energy",
        ["line_spread"] = "Line Spread", ["ghost_paths"] = "Ghost Paths", ["ghost_irradiance"] = "Ghost Irradiance",
        ["stray_light"] = "Stray Light", ["polarization_pupil_map"] = "Polarization Pupil Map",
        ["system_transmission"] = "System Transmission", ["glass_info"] = "Glass Data", ["solves"] = "Solves",
        ["optimize"] = "Optimize", ["tolerance"] = "Tolerance", ["rng_stream"] = "Random Stream",
        ["pupil_aberration"] = "Pupil Aberration", ["relative_illumination_map"] = "Relative Illumination Map",
        ["chief_ray_angle"] = "Chief Ray Angle", ["glass_map"] = "Glass Map", ["image_simulation"] = "Image Simulation",
        ["glass_substitution"] = "Glass Substitution",
    };

    /// <summary>Abbreviations for narrow panes (the full title stays in the tooltip).</summary>
    static readonly Dictionary<string, string> ShortTitles = new()
    {
        ["first_order"] = "First Order", ["system_summary"] = "System", ["spot_diagram"] = "Spot",
        ["field_curvature_distortion"] = "Field Curv. & Dist.", ["grid_distortion"] = "Grid Dist.", ["seidel"] = "Seidel",
        ["longitudinal_aberration"] = "Long. Aberration", ["chromatic_focal_shift"] = "Focal Shift",
        ["relative_illumination"] = "Rel. Illumination", ["encircled_energy"] = "Encircled En.", ["geometric_mtf"] = "Geom. MTF",
        ["wavefront_map"] = "Wavefront", ["through_focus_mtf"] = "Thru-Focus MTF", ["diffraction_encircled_energy"] = "Diffr. Enc. En.",
        ["ghost_irradiance"] = "Ghost Irrad.", ["polarization_pupil_map"] = "Pol. Pupil Map", ["system_transmission"] = "Transmission",
        ["pupil_aberration"] = "Pupil Aberr.", ["relative_illumination_map"] = "RI Map", ["chief_ray_angle"] = "CRA",
        ["image_simulation"] = "Image Sim.", ["glass_substitution"] = "Glass Subst.",
    };

    /// <summary>Segoe Fluent Icons glyphs, chosen to echo the Mac app's SF Symbols.</summary>
    static readonly Dictionary<string, string> Glyphs = new()
    {
        ["first_order"] = "", ["system_summary"] = "", ["validate"] = "",
        ["spot_diagram"] = "", ["ray_fan"] = "", ["opd_fan"] = "",
        ["field_curvature_distortion"] = "", ["grid_distortion"] = "",
        ["seidel"] = "", ["longitudinal_aberration"] = "", ["chromatic_focal_shift"] = "",
        ["lateral_colour"] = "", ["relative_illumination"] = "", ["footprint"] = "",
        ["encircled_energy"] = "", ["geometric_mtf"] = "", ["best_focus"] = "",
        ["ray_grid"] = "", ["wavefront_map"] = "", ["psf"] = "", ["mtf"] = "",
        ["mtf_vs_field"] = "", ["through_focus_mtf"] = "",
        ["diffraction_encircled_energy"] = "", ["line_spread"] = "", ["ghost_paths"] = "",
        ["ghost_irradiance"] = "", ["stray_light"] = "", ["polarization_pupil_map"] = "",
        ["system_transmission"] = "", ["glass_info"] = "", ["solves"] = "",
        ["pupil_aberration"] = "", ["relative_illumination_map"] = "",
        ["chief_ray_angle"] = "", ["glass_map"] = "", ["image_simulation"] = "",
    };

    public const string LensDataGlyph = "";
    public const string LayoutGlyph = "";
    public const string WarningGlyph = "";
    public const string ErrorGlyph = "";

    static readonly Dictionary<string, string> CategoryTitles = new()
    {
        ["model"] = "System", ["materials"] = "Materials", ["geometric"] = "Geometric", ["wave"] = "Wave Optics",
        ["radiometry"] = "Radiometry", ["polarization"] = "Polarization", ["ghost"] = "Ghosts", ["stray"] = "Stray Light",
        ["optimization"] = "Design", ["diagnostic"] = "Diagnostics",
    };

    static readonly string[] CategoryOrder = ["geometric", "wave", "model", "radiometry", "polarization", "ghost", "stray", "materials"];

    public static string Title(string name, string? engineTitle = null) =>
        Titles.TryGetValue(name, out var t) ? t : engineTitle is { Length: > 0 } ? TitleCase(engineTitle) : TitleCase(name.Replace('_', ' '));

    public static string ShortTitle(string name) => ShortTitles.TryGetValue(name, out var t) ? t : Title(name);
    public static string Glyph(string name) => Glyphs.TryGetValue(name, out var g) ? g : "";
    public static string CategoryTitle(string c) => CategoryTitles.TryGetValue(c, out var t) ? t : TitleCase(c);

    /// <summary>Category order: the known categories first, then the rest alphabetically.</summary>
    public static int CategoryRank(string c)
    {
        var i = Array.IndexOf(CategoryOrder, c);
        return i >= 0 ? i : CategoryOrder.Length;
    }

    static string TitleCase(string s)
    {
        string[] small = ["a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs"];
        var words = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        for (var k = 0; k < words.Length; k++)
        {
            var w = words[k];
            words[k] = k > 0 && small.Contains(w.ToLowerInvariant())
                ? w.ToLowerInvariant()
                : char.ToUpper(w[0], CultureInfo.InvariantCulture) + w[1..];
        }
        return string.Join(' ', words);
    }
}
