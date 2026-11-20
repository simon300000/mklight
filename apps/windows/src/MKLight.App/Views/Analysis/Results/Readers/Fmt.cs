using System.Globalization;
using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// The app's one number format for result views (docs/APP.md "One number
/// format"): a decimal point, no digit grouping, whatever the system region,
/// and never "−0".
/// </summary>
public static class Fmt
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;

    /// <summary>Up to 6 significant digits, scientific for very small or large values.</summary>
    public static string Short(double v)
    {
        if (double.IsNaN(v)) return "—";
        if (double.IsInfinity(v)) return v > 0 ? "∞" : "−∞";
        if (v == 0) return "0";
        var a = Math.Abs(v);
        if (a < 1e-3 || a >= 1e6) return v.ToString("0.000e+00", C);
        return v.ToString("G6", C);
    }

    /// <summary>A value with its unit ("12.5 mm", "14°").</summary>
    public static string Value(double v, string unit)
    {
        var s = Short(v);
        if (string.IsNullOrEmpty(unit) || !double.IsFinite(v)) return s;
        return unit == "°" ? s + "°" : s + " " + unit;
    }

    /// <summary>Fixed decimals, never "-0.00".</summary>
    public static string Fixed(double v, int decimals)
    {
        if (!double.IsFinite(v)) return Short(v);
        return NoNegativeZero(v.ToString("F" + decimals, C));
    }

    /// <summary>A length in mm, shown in µm below 0.1 mm: "25.92 µm", "0.1234 mm".</summary>
    public static string Length(double mm)
    {
        if (!double.IsFinite(mm)) return Short(mm);
        return Math.Abs(mm) < 0.1 ? Fixed(mm * 1000, 2) + " µm" : Fixed(mm, 4) + " mm";
    }

    /// <summary>A signed value with fixed decimals ("+0.0123"); zero gets a figure space for the sign.</summary>
    public static string Signed(double v, int decimals = 4)
    {
        if (!double.IsFinite(v)) return "—";
        var s = v.ToString("+0." + new string('0', decimals) + ";-0." + new string('0', decimals), C);
        if (s[1..].All(c => c is '0' or '.')) return " " + s[1..];
        return s;
    }

    public static string Percent(double fraction, int digits = 1) =>
        double.IsFinite(fraction) ? Fixed(fraction * 100, digits) + " %" : "—";

    /// <summary>"%.3g": three significant digits (axis-free values such as frequencies).</summary>
    public static string G3(double v) => double.IsFinite(v) ? NoNegativeZero(v.ToString("G3", C)) : Short(v);

    /// <summary>"%.4g": four significant digits (fractions of field maps).</summary>
    public static string G4(double v) => double.IsFinite(v) ? NoNegativeZero(v.ToString("G4", C)) : Short(v);

    static string NoNegativeZero(string s) => s.StartsWith('-') && s[1..].All(c => c is '0' or '.') ? s[1..] : s;

    /// <summary>
    /// A field as specified, for labels: "0°", "14°", "(3, 4) mm"; the
    /// system's label for a system field when the view has one.
    /// </summary>
    public static string Field(double[]? v, string unit = "")
    {
        if (v is null || v.Length != 2) return "";
        var u = unit == "°" ? "°" : unit.Length > 0 ? " " + unit : "";
        if (v[0] == 0) return Short(v[1]) + u;
        return $"({Short(v[0])}, {Short(v[1])})" + u;
    }

    /// <summary>"Field 2 · 14°": the system's label for field <paramref name="index"/>, else its value.</summary>
    public static string FieldTitle(ResultData data, int index, JsonNode? fieldValue)
    {
        var value = index >= 0 && index < data.FieldLabels.Count ? data.FieldLabels[index] : Field(Json.NfArray(fieldValue));
        return string.IsNullOrEmpty(value) ? $"Field {index + 1}" : $"Field {index + 1} · {value}";
    }

    /// <summary>A wavelength: "0.5876 µm".</summary>
    public static string Wavelength(double um) => double.IsFinite(um) ? um.ToString("0.0000", C) + " µm" : "—";
}

/// <summary>
/// Units of a wave result's image-space quantities (the result's
/// <c>units</c>): lengths on the image surface, or, for an afocal image
/// space, angles about the chief ray (fields named *_mm then hold mrad).
/// </summary>
public readonly record struct ImageUnits(bool Afocal)
{
    public static ImageUnits Of(JsonNode? result) => new(Json.Bool(Json.At(result, "units", "afocal")) ?? false);

    public string Frequency => Afocal ? "cycles/mrad" : "cycles/mm";
    /// <summary>The unit of a *_mm value × 1000.</summary>
    public string Micro => Afocal ? "µrad" : "µm";
    public string Density => Afocal ? "mrad⁻¹" : "mm⁻¹";
    public string FocusAxis => Afocal ? "Vergence (mD)" : "Focus shift (mm)";

    public string Length(double v)
    {
        if (!Afocal) return Fmt.Length(v);
        if (!double.IsFinite(v)) return Fmt.Short(v);
        return Math.Abs(v) < 0.1 ? Fmt.Fixed(v * 1000, 2) + " µrad" : Fmt.Fixed(v, 4) + " mrad";
    }
}
