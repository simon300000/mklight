using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>What makes a setting automatic.</summary>
public enum AutoSentinel
{
    /// <summary><c>0</c> (grids, PSF sizes).</summary>
    Zero,
    /// <summary><c>null</c> (limits, ranges, spacings).</summary>
    Null,
    /// <summary>An empty list (fixed MTF frequencies).</summary>
    Empty,
}

/// <summary>
/// Settings the engine chooses from the system when they are 0, null or
/// empty (docs/WAVE.md "Automatic sampling and ranges"). The form shows them
/// as "Auto (n)" with the value the last run used, never as a bare 0.
/// </summary>
public static class AutoSettings
{
    /// <summary>The automatic settings of an analysis, by key path (<c>pupil.grid</c>).</summary>
    public static AutoSentinel? Sentinel(string key, string analysis) => key switch
    {
        "pupil.grid" or "optics.pupil.grid" or "optics.psf_size" => AutoSentinel.Zero,
        "grid_size" when analysis is "psf" or "diffraction_encircled_energy" or "line_spread" => AutoSentinel.Zero,
        "max_frequency" when analysis == "mtf" => AutoSentinel.Null,
        "focus_range_mm" when analysis == "through_focus_mtf" => AutoSentinel.Null,
        "spacing_mm" when analysis == "psf" => AutoSentinel.Null,
        "optics.psf_spacing_mm" => AutoSentinel.Null,
        "through_focus.min" or "through_focus.max" when analysis == "spot_diagram" => AutoSentinel.Null,
        "frequencies" when analysis is "mtf_vs_field" or "through_focus_mtf" => AutoSentinel.Empty,
        _ => null,
    };

    /// <summary>Whether <paramref name="value"/> asks the engine to choose (a C# null is JSON null or unset).</summary>
    public static bool IsAuto(JsonNode? value, AutoSentinel sentinel) => sentinel switch
    {
        AutoSentinel.Zero => value is null || Json.Number(value) == 0,
        AutoSentinel.Null => value is null,
        _ => value is null || value is JsonArray { Count: 0 },
    };

    /// <summary>"Auto (282)", or "Auto" before a run has said what it used.</summary>
    public static string Display(string? used) => string.IsNullOrEmpty(used) ? Loc.S("Analysis_Settings_Auto") : Loc.F("Analysis_Settings_AutoUsed", used);

    /// <summary>The value the engine used for the automatic setting <paramref name="key"/>, read from the last result.</summary>
    public static string? Used(string key, string analysis, ResultData? result)
    {
        if (result is null || result.Analysis != analysis) return null;
        var j = result.Json;
        switch (analysis, key)
        {
            case ("psf", "grid_size"):
                return Range((Json.Arr(Json.At(j, "psfs")) ?? []).Select(p => Json.Number(Json.At(p, "psf", "size"))));
            case ("psf", "spacing_mm"):
                var s = (Json.Arr(Json.At(j, "psfs")) ?? []).Select(p => Json.Number(Json.At(p, "psf", "spacing_mm"))).OfType<double>().ToList();
                return s.Count == 0 ? null : Micrometres(s.Min(), s.Max());
            case ("line_spread", "grid_size"):
                return Range((Json.Arr(Json.At(j, "fields")) ?? []).Select(f => Json.BufferRef(result, Json.At(f, "positions_mm"))?.Count is int n ? (double?)n : null));
            case ("diffraction_encircled_energy", "grid_size"):
                return Range((Json.Arr(Json.At(j, "fields")) ?? []).Select(f => Json.Number(Json.At(f, "grid_size"))));
            case ("mtf", "max_frequency"):
                return Json.Numbers(result, Json.At(j, "frequencies")) is { Length: > 0 } f ? Fmt.G3(f[^1]) : null;
            case ("mtf_vs_field", "frequencies") or ("through_focus_mtf", "frequencies"):
                return Json.Numbers(result, Json.At(j, "frequencies")) is { Length: > 0 } fs ? string.Join(", ", fs.Select(Fmt.G3)) : null;
            case ("through_focus_mtf", "focus_range_mm"):
                var focus = Json.Numbers(result, Json.At(j, "focus_mm"));
                var half = focus?.Where(double.IsFinite).Select(Math.Abs).DefaultIfEmpty(double.NaN).Max() ?? double.NaN;
                return double.IsFinite(half) ? "±" + Fmt.G3(half) : null;
            case ("spot_diagram", "through_focus"):
                // The planes the run sampled (fields[].through_focus[].defocus).
                var planes = (Json.Arr(Json.At(Json.Arr(Json.At(j, "fields"))?.FirstOrDefault(), "through_focus")) ?? [])
                    .Select(e => Json.Number(Json.At(e, "defocus"))).OfType<double>().Where(double.IsFinite).ToList();
                if (planes.Count == 0) return null;
                double lo = planes.Min(), hi = planes.Max();
                return Math.Abs(lo + hi) <= 1e-9 * Math.Max(1, Math.Abs(hi)) ? "±" + Fmt.G3(hi) : Loc.F("Analysis_Settings_Range", Fmt.G3(lo), Fmt.G3(hi));
            case (_, "pupil.grid"):
                // Wavefront maps report their grids; other analyses report their pupil sampling per field.
                var grids = (Json.Arr(Json.At(j, "maps")) ?? []).Select(m => Json.Number(Json.At(m, "grid"))).ToList();
                if (grids.Count == 0)
                    grids = (Json.Arr(Json.At(j, "fields")) ?? Json.Arr(Json.At(j, "psfs")) ?? [])
                        .SelectMany(f => Json.Arr(Json.At(f, "pupil_sampling")) ?? []).Select(p => Json.Number(Json.At(p, "grid"))).ToList();
                if (grids.Count == 0) // mtf_vs_field: one list per field sample
                    grids = (Json.Arr(Json.At(j, "pupil_sampling")) ?? []).SelectMany(l => Json.Arr(l) ?? [])
                        .Select(p => Json.Number(Json.At(p, "grid"))).ToList();
                return Range(grids);
            case ("image_simulation", "optics.psf_size"):
                return Range([Json.Number(Json.At(j, "psf_size"))]);
            case ("image_simulation", "optics.psf_spacing_mm"):
                return Json.Number(Json.At(j, "psf_spacing_mm")) is double sp ? Micrometres(sp, sp) : null;
            default:
                return null;
        }
    }

    /// <summary>"282", or "254–371" when the values differ.</summary>
    static string? Range(IEnumerable<double?> values)
    {
        var v = values.OfType<double>().Where(double.IsFinite).ToList();
        if (v.Count == 0) return null;
        long lo = (long)v.Min(), hi = (long)v.Max();
        return lo == hi ? FormattableString.Invariant($"{lo}") : FormattableString.Invariant($"{lo}–{hi}");
    }

    static string Micrometres(double lo, double hi)
    {
        var a = Fmt.G3(lo * 1000);
        return hi > lo * (1 + 1e-6) ? $"{a}–{Fmt.G3(hi * 1000)} µm" : a + " µm";
    }
}
