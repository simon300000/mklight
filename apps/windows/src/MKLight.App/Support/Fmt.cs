using System.Globalization;

namespace MKLight.App.Support;

/// <summary>
/// The app's one number format for results (the Mac app's Fmt): a decimal
/// point, no digit grouping, whatever the system region; six significant
/// digits, scientific outside 1e-3…1e6; ∞ for infinities; never "−0".
/// </summary>
public static class Fmt
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;

    public static string Short(double v)
    {
        if (double.IsNaN(v)) return "—";
        if (double.IsInfinity(v)) return v > 0 ? "∞" : "−∞";
        if (v == 0) return "0";
        var a = Math.Abs(v);
        if (a < 1e-3 || a >= 1e6) return v.ToString("0.000e+00", C);
        return v.ToString("G6", C);
    }

    public static string Value(double v, string unit)
    {
        var s = Short(v);
        if (string.IsNullOrEmpty(unit) || !double.IsFinite(v)) return s;
        return unit == "°" ? s + "°" : s + " " + unit;
    }

    /// <summary>A length in mm, shown in µm below 0.1 mm.</summary>
    public static string Length(double mm)
    {
        if (!double.IsFinite(mm)) return Short(mm);
        var s = Math.Abs(mm) < 0.1 ? (mm * 1000).ToString("F2", C) + " µm" : mm.ToString("F4", C) + " mm";
        return s.StartsWith('-') && s.Skip(1).All(ch => "0. µm".Contains(ch)) ? s[1..] : s;
    }

    public static string Int(long v) => v.ToString(C);
}
