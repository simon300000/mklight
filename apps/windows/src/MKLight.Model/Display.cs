using System.Globalization;

namespace MKLight.Model;

/// <summary>Radius ↔ curvature conversion for display. The model stores curvature <c>c = 1/R</c>; a plane is <c>c = 0</c>, shown as an infinite radius.</summary>
public static class RadiusDisplay
{
    /// <summary>Radius for a curvature (<c>+∞</c> for <c>c = 0</c>).</summary>
    public static double RadiusFromCurvature(double c) => c == 0 ? double.PositiveInfinity : 1 / c;

    /// <summary>Curvature for a radius; <c>±∞</c> and <c>0</c> both mean a plane.</summary>
    public static double CurvatureFromRadius(double r) => double.IsInfinity(r) || r == 0 ? 0 : 1 / r;
}

public abstract partial record SurfaceShape
{
    /// <summary>Radius of curvature (mm), <c>∞</c> for planes and shapes without curvature.</summary>
    public double Radius => RadiusDisplay.RadiusFromCurvature(Curvature);
}

public sealed partial record Surface
{
    /// <summary>Radius of curvature for display.</summary>
    public double Radius => Shape.Radius;

    /// <summary>
    /// The surface with its radius set, which sets the curvature (a plane becomes a sphere).
    /// A shape without curvature (paraxial lens, coordinate break) stays as it is.
    /// </summary>
    public Surface WithRadius(double radius) =>
        Shape.WithCurvature(RadiusDisplay.CurvatureFromRadius(radius)) is { } s ? this with { Shape = s } : this;
}

/// <summary>
/// Number formatting for tables and inspectors.
/// <para>
/// <b>One rule app-wide:</b> numbers are shown with a decimal point and no digit grouping,
/// whatever the system region, as in lens prescriptions, glass catalogues and the engine's
/// files (<c>1.5</c>, <c>12345.678</c>). Everything here is culture-invariant; typed numbers also
/// accept a decimal comma (<see cref="Parse"/>).
/// </para>
/// </summary>
public static class NumberDisplay
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    /// <summary>
    /// Up to <paramref name="digits"/> significant digits, trailing zeros dropped, without
    /// exponent for ordinary magnitudes: <c>1.5</c>, <c>1.7552</c>, <c>0.0001</c> (C's <c>%g</c>).
    /// </summary>
    public static string Significant(double v, int digits = 10)
    {
        if (double.IsNaN(v)) return "NaN";
        if (double.IsInfinity(v)) return v > 0 ? "∞" : "−∞";
        if (v == 0) return "0";
        var p = Math.Max(digits, 1);
        var e = v.ToString("E" + (p - 1), Inv);
        var at = e.IndexOf('E');
        var exp = int.Parse(e[(at + 1)..], Inv);
        if (exp < -4 || exp >= p)
        {
            var mantissa = e[..at];
            if (mantissa.Contains('.')) mantissa = mantissa.TrimEnd('0').TrimEnd('.');
            return mantissa + "e" + (exp < 0 ? "-" : "+") + Math.Abs(exp).ToString("00", Inv);
        }
        var f = v.ToString("F" + Math.Max(0, p - 1 - exp), Inv);
        return f.Contains('.') ? f.TrimEnd('0').TrimEnd('.') : f;
    }

    /// <summary>
    /// Fixed decimals for ordinary magnitudes, scientific outside them; <c>Infinity</c> for
    /// infinities. A value that rounds to zero never shows a minus sign (<c>-0.0000</c>).
    /// </summary>
    public static string Format(double v, int decimals = 4)
    {
        if (double.IsNaN(v)) return "NaN";
        if (double.IsInfinity(v)) return v > 0 ? "Infinity" : "-Infinity";
        var a = Math.Abs(v);
        decimals = Math.Max(decimals, 0);
        string s;
        if (a != 0 && (a < 1e-3 || a >= 1e7))
        {
            var d = Math.Max(decimals, 1);
            s = v.ToString("0." + new string('0', d) + "e+00", Inv);
        }
        else
        {
            s = v.ToString("F" + decimals, Inv);
        }
        return s.StartsWith('-') && s.All(c => c is '-' or '0' or '.') ? s[1..] : s;
    }

    /// <summary>Shortest representation that parses back to the same <see cref="double"/> (<c>inf</c> / <c>-inf</c> for infinities).</summary>
    public static string Exact(double v) => double.IsInfinity(v) ? (v > 0 ? "inf" : "-inf") : Fmt.D(v);

    /// <summary>
    /// Parses a typed number: a decimal point or a decimal comma, <c>inf</c> / <c>∞</c> (also signed,
    /// also with the typographic minus). Digit grouping, NaN and empty text are refused (<c>null</c>).
    /// </summary>
    public static double? Parse(string? text)
    {
        if (text is null) return null;
        var s = text.Trim().Replace('−', '-').Replace("∞", "inf");
        if (s.Length == 0) return null;
        if (s.Contains(',') && !s.Contains('.')) s = s.Replace(',', '.');
        return ExtendedDouble.Parse(s);
    }
}

/// <summary>Formats numbers the way Swift's string interpolation does, for messages the engine and the Swift mirror share.</summary>
internal static class Fmt
{
    public static string D(double v)
    {
        if (double.IsNaN(v)) return "nan";
        if (double.IsInfinity(v)) return v > 0 ? "inf" : "-inf";
        if (v == 0) return double.IsNegative(v) ? "-0.0" : "0.0";
        var r = v.ToString("R", CultureInfo.InvariantCulture);
        if (r.Contains('E')) return r.Replace('E', 'e');
        return r.Contains('.') ? r : r + ".0";
    }
}
