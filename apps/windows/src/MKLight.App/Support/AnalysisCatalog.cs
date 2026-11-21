using System.Globalization;
using MKLight.Model;

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
        ["first_order"] = "\uE8FD", ["system_summary"] = "\uE8A9", ["validate"] = "\uE930",
        ["spot_diagram"] = "\uECCB", ["ray_fan"] = "\uEAFC", ["opd_fan"] = "\uE95E",
        ["field_curvature_distortion"] = "\uE7AD", ["grid_distortion"] = "\uE809",
        ["seidel"] = "\uE9D2", ["longitudinal_aberration"] = "\uE8AB", ["chromatic_focal_shift"] = "\uE790",
        ["lateral_colour"] = "\uE771", ["relative_illumination"] = "\uE706", ["footprint"] = "\uE81D",
        ["encircled_energy"] = "\uEA3A", ["geometric_mtf"] = "\uE9D2", ["best_focus"] = "\uE8A3",
        ["ray_grid"] = "\uE80A", ["wavefront_map"] = "\uE909", ["psf"] = "\uE734", ["mtf"] = "\uE9D2",
        ["mtf_vs_field"] = "\uE9D2", ["through_focus_mtf"] = "\uEAFC",
        ["diffraction_encircled_energy"] = "\uEA3A", ["line_spread"] = "\uE9D9", ["ghost_paths"] = "\uE7AD",
        ["ghost_irradiance"] = "\uE7B3", ["stray_light"] = "\uE82F", ["polarization_pupil_map"] = "\uE9E9",
        ["system_transmission"] = "\uE72A", ["glass_info"] = "\uEB42", ["solves"] = "\uE943",
        ["pupil_aberration"] = "\uECCA", ["relative_illumination_map"] = "\uE793",
        ["chief_ray_angle"] = "\uE741", ["glass_map"] = "\uE9D2", ["image_simulation"] = "\uE91B",
    };

    public const string LensDataGlyph = "\uE80A";
    public const string LayoutGlyph = "\uE81E";
    public const string WarningGlyph = "\uE7BA";
    public const string ErrorGlyph = "\uEA39";

    static readonly Dictionary<string, string> CategoryTitles = new()
    {
        ["model"] = "System", ["materials"] = "Materials", ["geometric"] = "Geometric", ["wave"] = "Wave Optics",
        ["radiometry"] = "Radiometry", ["polarization"] = "Polarization", ["ghost"] = "Ghosts", ["stray"] = "Stray Light",
        ["optimization"] = "Design", ["diagnostic"] = "Diagnostics",
    };

    static readonly string[] CategoryOrder = ["geometric", "wave", "model", "radiometry", "polarization", "ghost", "stray", "materials"];

    /// <summary>
    /// The analysis' title as shown (engine text, through <see cref="EngineText.T"/>): the app's title, else the
    /// engine's. Without a translation of the app's title, the engine title's translation stands in for it.
    /// </summary>
    public static string Title(string name, string? engineTitle = null)
    {
        if (Titles.TryGetValue(name, out var t))
        {
            var shown = EngineText.T(t);
            return shown == t && engineTitle is { Length: > 0 } && EngineText.T(engineTitle) is var e && e != engineTitle ? e : shown;
        }
        if (engineTitle is { Length: > 0 })
            return EngineText.T(engineTitle) is var translated && translated != engineTitle ? translated : EngineText.T(TitleCase(engineTitle));
        return EngineText.T(TitleCase(name.Replace('_', ' ')));
    }

    /// <summary>The abbreviated title; without a translation of the abbreviation, the translated full title.</summary>
    public static string ShortTitle(string name)
    {
        if (!ShortTitles.TryGetValue(name, out var t)) return Title(name);
        var shown = EngineText.T(t);
        return shown == t && Titles.TryGetValue(name, out var english) && Title(name) is var full && full != english ? full : shown;
    }

    public static string Glyph(string name) => Glyphs.TryGetValue(name, out var g) ? g : "\uE9D2";
    /// <summary>The category's title: the app's own (resources), else the English title through the engine text catalog.</summary>
    public static string CategoryTitle(string c) => Loc.Find("Analysis_Category_" + c) ?? EngineText.T(EnglishCategoryTitle(c));

    /// <summary>The category's title in English (menu tags for <c>--do menu=…</c>).</summary>
    public static string EnglishCategoryTitle(string c) => CategoryTitles.TryGetValue(c, out var t) ? t : TitleCase(c);

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
